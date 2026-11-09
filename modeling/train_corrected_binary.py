"""Train/select TF-IDF logistic regression, then separately evaluate the frozen holdout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import sklearn
from sklearn.metrics import brier_score_loss,log_loss
from sklearn.utils.class_weight import compute_sample_weight
from train_baseline import ROOT,estimator,metrics

DATA=ROOT/'data/processed/classification_corrected_full'
OUT=ROOT/'artifacts/corrected_full_binary'
CANDIDATES=(.1,1.,10.)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_split(data,split,manifest):
    p=Path(data)/(split+'.csv')
    if sha(p)!=manifest['outputs_sha256'][p.name]:raise ValueError('Frozen split hash mismatch: '+split)
    with p.open(newline='') as f:return list(csv.DictReader(f))
def inputs(rows):return [r['description'] for r in rows]
def targets(rows):return np.array([int(r['is_ai_tool']) for r in rows])
def select(reports):return max(reports,key=lambda c:reports[c]['macro_f1'])
def report(y,pred,scores):
    result=metrics(y,pred,[0,1],['non_ai','ai'])
    result['brier_score']=float(brier_score_loss(y,scores));result['log_loss']=float(log_loss(y,np.column_stack([1-scores,scores]),labels=[0,1]))
    return result

def train(data,out):
    data,out=Path(data),Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Output exists; choose a new experiment directory')
    manifest=json.loads((data/'manifest.json').read_text())
    rows=load_split(data,'train',manifest);val=load_split(data,'validation',manifest)
    if {r['group_id'] for r in rows}&{r['group_id'] for r in val}:raise ValueError('Training/validation group overlap')
    y=targets(rows);vy=targets(val);weights=compute_sample_weight('balanced',y)
    models={};reports={}
    for c in CANDIDATES:
        m=estimator();m.set_params(classifier__C=c)
        m.fit(inputs(rows),y,classifier__sample_weight=weights)
        scores=m.predict_proba(inputs(val))[:,list(m.classes_).index(1)]
        reports[c]=report(vy,(scores>=.5).astype(int),scores);models[c]=m
    # CANDIDATES ordered small-to-large: exact ties prefer smaller C.
    chosen=select(reports);model=models[chosen]
    out.mkdir(parents=True,exist_ok=True);joblib.dump(model,out/'binary.joblib')
    majority=int(np.bincount(y).argmax())
    config=dict(dataset_version=manifest['version'],dataset_manifest_sha256=sha(data/'manifest.json'),
        input_sha256=manifest['input_sha256'],split_sha256=manifest['outputs_sha256'],
        training_rows=len(rows),training_class_counts={str(c):int(sum(y==c)) for c in (0,1)},
        feature='description only; no product names, vendor, source labels, categories or governance fields',
        model='word TF-IDF unigrams/bigrams + LogisticRegression',class_weight='balanced, calculated from training labels only',
        candidates=list(CANDIDATES),selected_C=chosen,threshold=.5,
        selection_rule='Highest validation macro F1; exact ties prefer smaller C. Fixed threshold 0.5.',
        majority_class=majority,refit_on_evaluation=False,model_sha256=sha(out/'binary.joblib'),
        code_sha256=sha(__file__),versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,joblib=joblib.__version__),
        human_verified_accuracy=False,probabilities_calibrated=False)
    (out/'model_config.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'validation_metrics.json').write_text(json.dumps({'candidates':reports,'selected_C':chosen},indent=2)+'\n')
    print(json.dumps({'selected_C':chosen,'validation':{c:round(r['macro_f1'],6) for c,r in reports.items()}},indent=2))

def verified(out,data=None):
    out=Path(out);config=json.loads((out/'model_config.json').read_text())
    if sha(out/'binary.joblib')!=config['model_sha256']:raise ValueError('Model bytes do not match selection config')
    if data is not None and sha(Path(data)/'manifest.json')!=config['dataset_manifest_sha256']:raise ValueError('Dataset version differs from training')
    return config

def evaluate(data,out):
    data,out=Path(data),Path(out)
    if (out/'test_metrics.json').exists():raise ValueError('Holdout already evaluated; do not tune against it')
    config=verified(out,data);selection_hash=sha(out/'model_config.json')
    manifest=json.loads((data/'manifest.json').read_text());rows=load_split(data,'test',manifest)
    model=joblib.load(out/'binary.joblib');y=targets(rows)
    scores=model.predict_proba(inputs(rows))[:,list(model.classes_).index(1)];pred=(scores>=config['threshold']).astype(int)
    selected=report(y,pred,scores)
    majority=metrics(y,np.full(len(y),config['majority_class']),[0,1],['non_ai','ai'])
    results=dict(selected_C=config['selected_C'],selection_config_sha256=selection_hash,
        test_rows=len(rows),test_class_counts={str(c):int(sum(y==c)) for c in (0,1)},binary=selected,majority_class=majority,
        test_used_for_selection=False,independently_verified_accuracy=False,
        caveats=['Labels are mostly inherited source assignments with limited assistant source review.',
                 'Description length/style differs by source; high scores can reflect dataset-origin artifacts.',
                 'Grouped splits reduce exact overlaps; incomplete alias/vendor coverage remains possible.',
                 'Test products may have appeared in earlier experiments and research; not a newly collected independent benchmark.',
                 'Uncalibrated model scores are not governance-risk probabilities.'])
    with (out/'test_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['record_id','tool_name','expected_is_ai_tool','predicted_is_ai_tool','model_ai_score'],lineterminator='\n');w.writeheader()
        for r,t,p,s in zip(rows,y,pred,scores):w.writerow(dict(record_id=r['record_id'],tool_name=r['tool_name'],expected_is_ai_tool=int(t),predicted_is_ai_tool=int(p),model_ai_score=float(s)))
    (out/'test_metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'test_accuracy':selected['accuracy'],'test_macro_f1':selected['macro_f1'],'confusion_matrix':selected['confusion_matrix'],'majority_accuracy':majority['accuracy'],'majority_macro_f1':majority['macro_f1']},indent=2))

def predict(description,out):
    value=' '.join(description.split())
    if not value:raise ValueError('A concrete description is required')
    config=verified(out);model=joblib.load(Path(out)/'binary.joblib')
    if not model.named_steps['tfidf'].transform([value]).nnz:return dict(is_ai_tool=None,model_ai_score=None,needs_review=True,reason='No known vocabulary')
    score=float(model.predict_proba([value])[0,list(model.classes_).index(1)])
    return dict(is_ai_tool=int(score>=config['threshold']),model_ai_score=score,needs_review=True,
                caveat='Uncalibrated primary-purpose classifier; does not predict optional AI features or governance risk.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['train','evaluate','predict'])
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--description',default='')
    a=p.parse_args()
    if a.stage=='train':train(a.data,a.out)
    elif a.stage=='evaluate':evaluate(a.data,a.out)
    else:print(json.dumps(predict(a.description,a.out),indent=2))
if __name__=='__main__':main()
