"""Train TF-IDF baselines; select binary model using validation macro F1 only.

Outputs are development results against weak labels, not real-world accuracy.
"""
import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'preprocessing_scripts'))
from prepare_classification import load, save, validate


def estimator():
    return Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=50000,sublinear_tf=True)),
                     ('classifier',LogisticRegression(C=1.0,max_iter=2000,random_state=42))])


def balanced_source_weights(rows):
    # Balance effective class mass after downweighting provisional memberships.
    raw=np.array([.25 if r['label_source']=='provisional_catalog_membership' else 1.0 for r in rows])
    y=np.array([int(r['is_ai_tool']) for r in rows])
    for label in (0,1):
        mask=y==label
        if not mask.any():raise ValueError('Both labels are required')
        raw[mask]*=len(rows)/(2*raw[mask].sum())
    return raw


def metrics(y,pred,labels,names):
    return dict(accuracy=float(accuracy_score(y,pred)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        macro_f1=float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)),
        confusion_matrix=confusion_matrix(y,pred,labels=labels).tolist(),
        confusion_matrix_label_order=names,
        per_class=classification_report(y,pred,labels=labels,target_names=names,output_dict=True,zero_division=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/processed/classification_full')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/baseline')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()) and not a.overwrite:p.error('Output exists; use --overwrite or a new --out')
    splits={s:load(a.data/f'{s}.csv') for s in ('train','validation','test')}
    # Preparation's check expects numeric binary labels.
    validate({s:[dict(r,is_ai_tool=int(r['is_ai_tool'])) for r in rows] for s,rows in splits.items()})
    for s in ('validation','test'):
        if any(r['label_source']=='provisional_catalog_membership' for r in splits[s]):
            raise ValueError('Evaluation labels cannot be unreviewed catalog membership')
    a.out.mkdir(parents=True,exist_ok=True)
    datasets=dict(reviewed_only=[r for r in splits['train'] if r['label_source']!='provisional_catalog_membership'],
                  full_catalog=splits['train'])
    fitted={}; reports={}; names=['non_ai','ai'];labels=[0,1]
    x={s:[r['text_for_model'] for r in rows] for s,rows in splits.items()}
    y={s:[int(r['is_ai_tool']) for r in rows] for s,rows in splits.items()}
    for name,rows in datasets.items():
        model=estimator()
        model.fit([r['text_for_model'] for r in rows],[int(r['is_ai_tool']) for r in rows],
                  classifier__sample_weight=balanced_source_weights(rows))
        fitted[name]=model
        reports[name]=dict(training_rows=len(rows),training_labels=dict(Counter(r['is_ai_tool'] for r in rows)),
            validation=metrics(y['validation'],model.predict(x['validation']),labels,names))
    # In a tie prefer fewer provisional labels, fixed before viewing test results.
    selected=max(('reviewed_only','full_catalog'),key=lambda n:reports[n]['validation']['macro_f1'])
    dummy=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(y['train']),1)),y['train'])
    reports['majority_class']=dict(validation=metrics(y['validation'],dummy.predict(np.zeros((len(y['validation']),1))),labels,names))
    # Test results are produced after selecting; never used to select/refit a model.
    for name,model in fitted.items():
        reports[name]['test']=metrics(y['test'],model.predict(x['test']),labels,names)
        joblib.dump(model,a.out/f'binary_{name}.joblib')
        predicted=model.predict(x['test']);scores=model.predict_proba(x['test'])[:,list(model.classes_).index(1)]
        save(a.out/f'test_predictions_{name}.csv',
             [dict(application_id=r['application_id'],tool_name=r['tool_name'],
                   expected_is_ai_tool=truth,predicted_is_ai_tool=int(pred),model_ai_score=float(score))
              for r,truth,pred,score in zip(splits['test'],y['test'],predicted,scores)],
             ['application_id','tool_name','expected_is_ai_tool','predicted_is_ai_tool','model_ai_score'])
    reports['majority_class']['test']=metrics(y['test'],dummy.predict(np.zeros((len(y['test']),1))),labels,names)
    # Begin business-purpose categorization using only available reviewed categories.
    cr=[r for r in splits['train'] if r['is_ai_tool']=='1' and r['category']]
    categories=sorted({r['category'] for r in cr})
    category_model=estimator()
    counts=Counter(r['category'] for r in cr)
    weights=np.array([len(cr)/(len(counts)*counts[r['category']]) for r in cr])
    category_model.fit([r['text_for_model'] for r in cr],[r['category'] for r in cr],classifier__sample_weight=weights)
    category_reports={}
    for split in ('validation','test'):
        rows=[r for r in splits[split] if r['is_ai_tool']=='1' and r['category']]
        cy=[r['category'] for r in rows]
        cp=category_model.predict([r['text_for_model'] for r in rows])
        category_reports[split]=dict(rows=len(rows),missing_categories=sorted(set(categories)-set(cy)),
            metrics=metrics(cy,cp,categories,categories))
    joblib.dump(category_model,a.out/'category.joblib')
    results=dict(selected_binary_model=selected,selection_rule='highest validation macro F1; ties prefer reviewed_only',
        binary=reports,category=category_reports,category_labels=categories,
        threshold=.5,feature='text_for_model only',
        weak_label_development_results=True,independently_verified_accuracy=False,
        limitations=['Only four non-AI examples in each evaluation split.',
                    'Model-reviewed positive labels and team negative labels have not been independently audited.',
                    'Model scores are uncalibrated and are not risk scores.',
                    'Category evaluation is conditional on an AI label, not end-to-end detection accuracy.',
                    'Source writing style may confound classification.'])
    (a.out/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    model_config=dict(selected_binary_model=selected,categories=categories,threshold=.5,
        versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,joblib=joblib.__version__),
        inputs={f'{s}.csv':hashlib.sha256((a.data/f'{s}.csv').read_bytes()).hexdigest() for s in splits},
        training=dict(vectorizer='word TF-IDF, unigrams+bigrams, sublinear TF, max_features=50000',
                      classifier='LogisticRegression(C=1.0, max_iter=2000, random_state=42)',
                      binary_sample_weight='source weights: 1 reviewed/seed, 0.25 provisional; then balance effective class mass',
                      category_sample_weight='inverse category frequency',refit_on_evaluation=False))
    (a.out/'model_config.json').write_text(json.dumps(model_config,indent=2)+'\n')
    print(json.dumps(dict(selected=selected,binary={name:{s:round(r[s]['macro_f1'],4) for s in ('validation','test')} for name,r in reports.items()},
                         category_test_accuracy=category_reports['test']['metrics']['accuracy']),indent=2))


if __name__=='__main__':main()
