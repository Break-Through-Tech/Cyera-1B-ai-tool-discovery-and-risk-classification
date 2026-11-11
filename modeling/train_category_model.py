"""Train a local description -> AI-tool-category classifier (no API calls).

Install: python -m pip install scikit-learn numpy joblib
Train: python train_category_model.py train --input dataset_b_final_fixed.csv --out-dir artifacts/category_tfidf
Predict: python train_category_model.py predict --model artifacts/category_tfidf/category_model.joblib --text "AI assistant that completes and debugs code"

Evaluation uses grouped cross-validation, fitting TF-IDF only on training folds.
The saved final pipeline is then fitted on all eligible AI rows. CV estimates
are not an independent final test after repeated model-selection experiments.
Only load joblib files you created or trust; they can execute Python code.
"""
import argparse
import csv
import hashlib
import json
import platform
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

CATEGORIES = [
    'Coding / Developer Tools', 'General AI Assistants', 'Content Generation',
    'Image / Video Generation', 'Meeting / Transcription Tools', 'Data Analytics',
    'Customer Support', 'Sales / Marketing', 'Productivity', 'Research / Search',
    'AI Infrastructure / Model Platforms', 'Other',
]


def normalized(text):
    return re.sub(r'\W+', ' ', text.casefold()).strip()


def hostname(value):
    try:
        return (urlsplit(value if '://' in value else 'https://' + value).hostname or '').lower().removeprefix('www.')
    except ValueError:
        return ''


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_training(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'tool_id','tool_name','domain','description','is_ai_tool','category'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Missing columns: ' + str(sorted(required-set(reader.fieldnames or []))))
        raw = list(reader)
    selected, excluded, seen = [], [], set()
    for r in raw:
        if None in r or any(v is None for v in r.values()):
            raise ValueError('Malformed CSV row')
        tid = r['tool_id'].strip()
        if not tid or tid in seen:
            raise ValueError('Missing/duplicate tool_id: ' + tid)
        seen.add(tid)
        flag = r['is_ai_tool'].strip().lower()
        if flag in ('false','0'):
            reason = 'non_ai_not_used_for_category_training'
        elif flag not in ('true','1'):
            reason = 'unresolved_or_invalid_ai_label'
        elif not r['description'].strip():
            reason = 'missing_description'
        elif r['category'].strip() not in CATEGORIES:
            reason = 'missing_or_unknown_category'
        else:
            reason = ''
        if reason:
            excluded.append({'tool_id':tid, 'reason':reason})
        else:
            r['description'] = ' '.join(r['description'].split())
            r['category'] = r['category'].strip()
            selected.append(r)
    if set(r['category'] for r in selected) != set(CATEGORIES):
        raise ValueError('Need eligible examples for all 12 categories; inspect missing labels/descriptions.')
    # Union related rows, including transitive connections. No group crosses folds.
    parent = list(range(len(selected)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    keys = {}
    desc_labels = {}
    for i,r in enumerate(selected):
        desc = normalized(r['description'])
        if desc in desc_labels and desc_labels[desc] != r['category']:
            raise ValueError('Identical normalized descriptions have conflicting categories: ' + r['tool_id'])
        desc_labels[desc] = r['category']
        for kind,value in [('name',normalized(r['tool_name'])),('host',hostname(r['domain'])),('description',desc)]:
            if not value:
                continue
            key = (kind,value)
            if key in keys:
                parent[find(i)] = find(keys[key])
            keys[key] = i
    return selected, excluded, np.array([find(i) for i in range(len(selected))])


def make_model():
    # One- and two-word phrases. No category, vendor, risk, or approval features.
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=20000,
                                sublinear_tf=True, strip_accents='unicode')),
        ('classifier', LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=3000)),
    ])


def train(args):
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError('Use a new/empty output directory to protect previous results.')
    rows, excluded, groups = load_training(args.input)
    X = np.array([r['description'] for r in rows])
    y = np.array([r['category'] for r in rows])
    labels = sorted(CATEGORIES)
    group_counts = {c:len(set(groups[y==c])) for c in labels}
    folds = min(args.folds, min(group_counts.values()))
    if folds < 2:
        raise ValueError('Need at least two independent groups in every category.')
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(X,y,groups))
    for tr,va in splits:
        if set(y[tr]) != set(labels):
            raise ValueError('A training fold lacks a category. Try --folds 3 or add examples.')
        assert not set(groups[tr]) & set(groups[va])
    pred = np.empty(len(y),dtype=object)
    base = np.empty(len(y),dtype=object)
    scores = np.zeros(len(y))
    fold_ids = np.zeros(len(y),dtype=int)
    fold_metrics=[]
    for fold,(tr,va) in enumerate(splits,1):
        model = make_model().fit(X[tr],y[tr])
        pred[va] = model.predict(X[va])
        scores[va] = model.predict_proba(X[va]).max(axis=1)
        dummy = DummyClassifier(strategy='most_frequent').fit(np.zeros((len(tr),1)),y[tr])
        base[va] = dummy.predict(np.zeros((len(va),1)))
        fold_ids[va] = fold
        metric = {'fold':fold,'training_rows':len(tr),'validation_rows':len(va),
                  'accuracy':accuracy_score(y[va],pred[va]),
                  'macro_f1':f1_score(y[va],pred[va],labels=labels,average='macro',zero_division=0)}
        fold_metrics.append(metric)
        print(f"Fold {fold}/{folds}: accuracy={metric['accuracy']:.3f}, macro-F1={metric['macro_f1']:.3f}")
    report = classification_report(y,pred,labels=labels,output_dict=True,zero_division=0)
    metrics = {'evaluation':'Out-of-fold predictions; fixed hyperparameters; not a separate final holdout.',
               'training_rows':len(rows),'excluded_rows':len(excluded),'category_counts':dict(Counter(y)),
               'groups_per_category':group_counts,'folds':fold_metrics,
               'accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro'),
               'weighted_f1':f1_score(y,pred,average='weighted'),
               'majority_baseline_accuracy':accuracy_score(y,base),
               'majority_baseline_macro_f1':f1_score(y,base,average='macro',zero_division=0),
               'classification_report':report,'seed':args.seed,
               'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
               'versions':{'python':platform.python_version(),'scikit_learn':sklearn.__version__,'numpy':np.__version__},
               'limitations':['Small categories have unstable estimates.','Scores are not calibrated confidence guarantees.',
                              'Exact identity/description grouping does not detect every near duplicate.',
                              'Existing category labels are treated as targets, not independently verified.']}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    text_report = classification_report(y,pred,labels=labels,zero_division=0)
    (args.out_dir/'classification_report.txt').write_text(text_report,encoding='utf-8')
    predictions=[{'tool_id':r['tool_id'],'tool_name':r['tool_name'],'description':r['description'],
                  'group':int(groups[i]),'fold':int(fold_ids[i]),'actual_category':y[i],
                  'predicted_category':pred[i],'max_probability_uncalibrated':float(scores[i]),
                  'correct':bool(y[i]==pred[i])} for i,r in enumerate(rows)]
    write_csv(args.out_dir/'cv_predictions.csv',predictions,list(predictions[0]))
    write_csv(args.out_dir/'excluded_rows.csv',excluded,['tool_id','reason'])
    cm=confusion_matrix(y,pred,labels=labels)
    write_csv(args.out_dir/'confusion_matrix.csv',
              [dict(actual_category=c,**{label:int(cm[i,j]) for j,label in enumerate(labels)}) for i,c in enumerate(labels)],
              ['actual_category']+labels)
    # Refit for deployment only AFTER producing out-of-fold evaluation.
    final = make_model().fit(X,y)
    joblib.dump(final,args.out_dir/'category_model.joblib')
    vocab=final.named_steps['tfidf'].get_feature_names_out()
    classifier=final.named_steps['classifier']
    features=[{'category':c,'feature':vocab[j],'weight':float(classifier.coef_[i,j])}
              for i,c in enumerate(classifier.classes_) for j in np.argsort(classifier.coef_[i])[-15:][::-1]]
    write_csv(args.out_dir/'top_features.csv',features,['category','feature','weight'])
    print('\n'+text_report)
    print(f"Out-of-fold macro-F1: {metrics['macro_f1']:.3f}; majority baseline: {metrics['majority_baseline_macro_f1']:.3f}")
    print(f"Saved final model fitted on {len(rows)} AI examples to {args.out_dir/'category_model.joblib'}")


def predict(args):
    if not args.text.strip():
        raise ValueError('Provide a nonempty product description.')
    model=joblib.load(args.model)
    probs=model.predict_proba([args.text])[0]
    classes=model.named_steps['classifier'].classes_
    order=np.argsort(probs)[::-1][:3]
    no_features=model.named_steps['tfidf'].transform([args.text]).nnz==0
    print(json.dumps({'prediction':str(classes[order[0]]) if not no_features else None,
                      'needs_review':bool(no_features or probs[order[0]]<args.review_threshold),
                      'review_threshold_note':'Heuristic only; tune using validation data.',
                      'top_categories':[{'category':str(classes[i]),'uncalibrated_probability':float(probs[i])} for i in order],
                      'no_known_vocabulary':no_features},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--input',type=Path,required=True)
    t.add_argument('--out-dir',type=Path,required=True);t.add_argument('--seed',type=int,default=42)
    t.add_argument('--folds',type=int,default=5)
    q=sub.add_parser('predict');q.add_argument('--model',type=Path,required=True)
    q.add_argument('--text',required=True);q.add_argument('--review-threshold',type=float,default=0.5)
    a=p.parse_args()
    if a.command=='train' and a.folds<2:p.error('--folds must be at least 2')
    if a.command=='predict' and not 0<=a.review_threshold<=1:p.error('Threshold must be between 0 and 1')
    try:
        train(a) if a.command=='train' else predict(a)
    except (ValueError,FileNotFoundError) as exc:
        p.exit(1,f'Error: {exc}\n')


if __name__=='__main__':main()
