"""Fit/select v2 classifiers on train/validation; evaluate in a separate command."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from train_baseline import ROOT, estimator, metrics
from embedding_utils import ENCODER_ID, ENCODER_REVISION, encode_rows, load_encoder

DATA = ROOT/'data/processed/classification_v2'
OUT = ROOT/'artifacts/classification_v2'
CACHE = ROOT/'artifacts/embedding_cache'
FAMILIES = ('tfidf','embeddings')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(data, split, manifest):
    path = Path(data)/f'{split}.csv'
    if sha(path) != manifest['sha256'][path.name]:
        raise ValueError('Frozen dataset hash mismatch: '+split)
    with path.open(newline='') as f:rows = list(csv.DictReader(f))
    return rows


def text(rows):
    # Fix predictor inputs before evaluation; provenance and targets never enter features.
    return [' '.join(r['description'].split()) for r in rows]


def choose(scores):
    # Fixed tie breaker prefers the simpler TF-IDF model.
    return max(FAMILIES,key=lambda f:scores[f])


def check_training(train, validation, protected):
    if {r['is_ai_tool'] for r in train} != {'0','1'}:
        raise ValueError('Both training classes are required')
    if {r['family_id'] for r in train} & set(protected['families']):
        raise ValueError('Held-out family in training')
    hashes = {hashlib.sha256(' '.join(r['description'].lower().split()).encode()).hexdigest() for r in train}
    if hashes & set(protected['description_sha256']):
        raise ValueError('Held-out description in training')
    for key in ('label_id','family_id'):
        if {r[key] for r in train} & {r[key] for r in validation}:
            raise ValueError('Training/validation overlap: '+key)
    if any(r['label_method']!='assistant_source_review' for r in train+validation):
        raise ValueError('Unexpected label method')
    categories = {r['category'] for r in train if r['is_ai_tool']=='1'}
    if any(r['category'] not in categories for r in validation if r['is_ai_tool']=='1'):
        raise ValueError('Unseen validation category')


def vectors(encoder, rows, cache):
    return encode_rows(encoder,[{'text_for_model':s} for s in text(rows)],Path(cache)/'v2_vectors')


def train(data, out, cache):
    data,out = Path(data),Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output exists; use a new versioned directory')
    manifest = json.loads((data/'manifest.json').read_text())
    if manifest['dataset_version']!='2.0':raise ValueError('Expected v2 dataset')
    train_rows = load_split(data,'train',manifest)
    val = load_split(data,'validation',manifest)
    protected_path = data/'protected_evaluation_groups.json'
    if sha(protected_path)!=manifest['sha256'][protected_path.name]:raise ValueError('Protected group hash mismatch')
    check_training(train_rows,val,json.loads(protected_path.read_text()))
    encoder = load_encoder(Path(cache)/'encoder')
    xt,xv = vectors(encoder,train_rows,cache),vectors(encoder,val,cache)
    y = np.array([int(r['is_ai_tool']) for r in train_rows])
    vy = [int(r['is_ai_tool']) for r in val]
    cmask = y==1; vmask = np.array(vy)==1
    cy = [r['category'] for r in train_rows if r['is_ai_tool']=='1']
    cvy = [r['category'] for r in val if r['is_ai_tool']=='1']
    categories = sorted(set(cy))
    validation = {}; fitted = {}
    for family in FAMILIES:
        binary = estimator() if family=='tfidf' else LogisticRegression(C=1,max_iter=2000,random_state=42)
        category = estimator() if family=='tfidf' else LogisticRegression(C=1,max_iter=2000,random_state=42)
        bx,bv = (text(train_rows),text(val)) if family=='tfidf' else (xt,xv)
        cx = [s for s,keep in zip(text(train_rows),cmask) if keep] if family=='tfidf' else xt[cmask]
        cv = [s for s,keep in zip(text(val),vmask) if keep] if family=='tfidf' else xv[vmask]
        weight_key = 'classifier__sample_weight' if family=='tfidf' else 'sample_weight'
        binary.fit(bx,y,**{weight_key:compute_sample_weight('balanced',y)})
        category.fit(cx,cy,**{weight_key:compute_sample_weight('balanced',cy)})
        validation[family] = {'binary':metrics(vy,binary.predict(bv),[0,1],['non_ai','ai']),
                              'category':metrics(cvy,category.predict(cv),categories,categories)}
        fitted[family] = (binary,category)
    # Nothing below reads test labels or changes any fitted parameters.
    config = dict(dataset_version='2.0',feature='description only; normalized whitespace',threshold=.5,
        default_binary_family=choose({f:validation[f]['binary']['macro_f1'] for f in FAMILIES}),
        default_category_family=choose({f:validation[f]['category']['macro_f1'] for f in FAMILIES}),
        selection_rule='Validation macro F1, independently for binary and category; ties prefer TF-IDF',
        encoder_id=ENCODER_ID,encoder_revision=ENCODER_REVISION,encoder_frozen=True,
        category_labels=categories,training_rows=len(train_rows),category_training_rows=len(cy),
        validation_rows=len(val),refit_on_evaluation=False,catalog_augmentation=False,
        dataset_manifest_sha256=sha(data/'manifest.json'),inputs=manifest['sha256'],
        training_code_sha256=sha(__file__),versions={'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'joblib':joblib.__version__},
        label_scope='Primary purpose of the described application or explicitly named AI feature; not AI-feature availability or risk',
        human_verified_labels=False,probability_calibrated=False)
    out.mkdir(parents=True,exist_ok=True)
    for family,(binary,category) in fitted.items():
        joblib.dump(binary,out/f'{family}_binary.joblib')
        joblib.dump(category,out/f'{family}_category.joblib')
    config['model_sha256'] = {p.name:sha(p) for p in sorted(out.glob('*.joblib'))}
    (out/'model_config.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'validation_metrics.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps({'selected_binary':config['default_binary_family'],'selected_category':config['default_category_family'],'validation':{f:{t:validation[f][t]['macro_f1'] for t in ('binary','category')} for f in FAMILIES}},indent=2))


def verified_config(out, data=None):
    out = Path(out)
    config = json.loads((out/'model_config.json').read_text())
    for name,expected in config['model_sha256'].items():
        if sha(out/name)!=expected:raise ValueError('Model hash mismatch: '+name)
    if data is not None and sha(Path(data)/'manifest.json')!=config['dataset_manifest_sha256']:
        raise ValueError('Training/evaluation dataset mismatch')
    if config['encoder_id']!=ENCODER_ID or config['encoder_revision']!=ENCODER_REVISION:
        raise ValueError('Encoder revision mismatch')
    return config


def evaluate(data, out, cache):
    data,out = Path(data),Path(out)
    if (out/'test_metrics.json').exists():raise ValueError('Test evaluation already exists; do not repeatedly tune on this holdout')
    config = verified_config(out,data)
    # Persisted selection was made by train() before this command opens test.csv.
    selection_hash = sha(out/'model_config.json')
    manifest = json.loads((data/'manifest.json').read_text())
    rows = load_split(data,'test',manifest)
    encoder = load_encoder(Path(cache)/'encoder')
    x = vectors(encoder,rows,cache)
    y = np.array([int(r['is_ai_tool']) for r in rows]); mask = y==1
    cy = [r['category'] for r in rows if r['is_ai_tool']=='1']; categories=config['category_labels']
    reports = {}; predictions = {}
    for family in FAMILIES:
        binary = joblib.load(out/f'{family}_binary.joblib')
        category = joblib.load(out/f'{family}_category.joblib')
        bx = text(rows) if family=='tfidf' else x
        bp = binary.predict(bx); cp = category.predict(bx)
        scores = binary.predict_proba(bx)[:,list(binary.classes_).index(1)]
        predictions[family] = (bp,cp,scores)
        reports[family] = {'binary':metrics(y,bp,[0,1],['non_ai','ai']),
                           'category_conditional_on_true_ai':metrics(cy,cp[mask],categories,categories)}
    bf,cf = config['default_binary_family'],config['default_category_family']
    bp,_,scores = predictions[bf];cp=predictions[cf][1]
    end_correct = [(pred==truth and (truth==0 or category==r['category'])) for r,truth,pred,category in zip(rows,y,bp,cp)]
    rows_out = []
    for i,r in enumerate(rows):
        item = dict(label_id=r['label_id'],tool_name=r['tool_name'],expected_is_ai_tool=int(y[i]),expected_category=r['category'],selected_is_ai_tool=int(bp[i]),selected_category=str(cp[i]) if bp[i]==1 else '',selected_ai_score=float(scores[i]))
        for f in FAMILIES:
            item[f+'_is_ai_tool']=int(predictions[f][0][i]);item[f+'_category']=str(predictions[f][1][i])
        rows_out.append(item)
    report = dict(selection_config_sha256=selection_hash,selected_binary_family=bf,selected_category_family=cf,
        test_rows=len(rows),test_class_counts={'non_ai':int(sum(y==0)),'ai':int(sum(y==1))},models=reports,
        selected_end_to_end_accuracy=float(np.mean(end_correct)),
        majority_class={'binary':metrics(y,np.zeros(len(y),dtype=int),[0,1],['non_ai','ai'])},
        scope='Frozen v2 holdout, assistant-reviewed labels and assistant-authored descriptions; no v1/catalog training augmentation',
        test_used_for_selection=False,pretrained_encoder_may_have_seen_products=True,independently_human_verified=False,
        caveats=['36 examples per binary class; three positives per category.', 'Category scores are conditional on true AI labels; end-to-end requires both detection and category.', 'Scores are uncalibrated; no governance-risk or AI-feature prediction is trained.', 'Author style and public product knowledge may inflate results.'])
    with (out/'test_predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows_out[0]));writer.writeheader();writer.writerows(rows_out)
    (out/'test_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'binary_family':bf,'category_family':cf,'test_binary_macro_f1':reports[bf]['binary']['macro_f1'],'test_category_accuracy':reports[cf]['category_conditional_on_true_ai']['accuracy'],'end_to_end_accuracy':report['selected_end_to_end_accuracy']},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['train','evaluate'])
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--cache',type=Path,default=CACHE)
    a=parser.parse_args()
    (train if a.stage=='train' else evaluate)(a.data,a.out,a.cache)


if __name__=='__main__':main()
