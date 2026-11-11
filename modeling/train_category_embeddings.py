"""Compare frozen local sentence embeddings to the existing TF-IDF experiment.
Keep train_category_model.py beside this script. No API key needed.
First use downloads the pretrained encoder; encoding and classifier training are local.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
try:
    from .train_category_model import load_training, write_csv, CATEGORIES
except ImportError:
    from train_category_model import load_training, write_csv, CATEGORIES


def load_folds(path, rows, groups):
    with path.open(encoding='utf-8-sig',newline='') as f:
        prior=list(csv.DictReader(f))
    by={r['tool_id']:r for r in prior}
    if len(by)!=len(prior) or set(by)!={r['tool_id'] for r in rows}:
        raise ValueError('Baseline prediction IDs do not match this dataset.')
    for r in rows:
        old=by[r['tool_id']]
        if old['description']!=r['description'] or old['actual_category']!=r['category']:
            raise ValueError('Descriptions/labels changed since baseline. Rerun baseline first.')
    folds=np.array([int(by[r['tool_id']]['fold']) for r in rows])
    if len(set(folds))<2:raise ValueError('Need at least two validation folds.')
    for g in set(groups):
        if len(set(folds[groups==g]))!=1:raise ValueError('Baseline folds split related products.')
    return folds, np.array([by[r['tool_id']]['predicted_category'] for r in rows])


def evaluate(embeddings,y,folds):
    labels=sorted(CATEGORIES)
    pred=np.empty(len(y),dtype=object);prob=np.zeros(len(y));results=[]
    for f in sorted(set(folds)):
        tr=np.flatnonzero(folds!=f);va=np.flatnonzero(folds==f)
        if set(y[tr])!=set(labels):raise ValueError('Training fold lacks a category.')
        classifier=LogisticRegression(C=1.0,class_weight='balanced',solver='lbfgs',max_iter=3000)
        classifier.fit(embeddings[tr],y[tr]);pred[va]=classifier.predict(embeddings[va])
        prob[va]=classifier.predict_proba(embeddings[va]).max(axis=1)
        result={'fold':int(f),'accuracy':accuracy_score(y[va],pred[va]),
                'macro_f1':f1_score(y[va],pred[va],labels=labels,average='macro',zero_division=0)}
        results.append(result)
        print(f"Fold {f}: accuracy={result['accuracy']:.3f}, macro-F1={result['macro_f1']:.3f}")
    return pred,prob,results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--baseline-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--encoder',default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--device',default='cpu',help='cpu by default; cuda if you have compatible GPU support')
    a=p.parse_args()
    if a.out_dir.exists() and any(a.out_dir.iterdir()):p.error('Choose an empty/new output directory.')
    rows,excluded,groups=load_training(a.input)
    baseline=json.loads((a.baseline_dir/'metrics.json').read_text(encoding='utf-8'))
    if baseline['input_sha256']!=hashlib.sha256(a.input.read_bytes()).hexdigest():
        p.error('Dataset differs from baseline input; use the same file or rerun baseline.')
    folds,baseline_pred=load_folds(a.baseline_dir/'cv_predictions.csv',rows,groups)
    y=np.array([r['category'] for r in rows]);texts=[r['description'] for r in rows]
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    print('Loading pretrained encoder (first run requires an internet download)...')
    encoder=SentenceTransformer(a.encoder,device=a.device,trust_remote_code=False)
    # This frozen encoder is not fitted/tuned on this dataset; precomputing all
    # embeddings is safe. Only the classifier learns from each training fold.
    embeddings=encoder.encode(texts,batch_size=32,normalize_embeddings=True,
                              convert_to_numpy=True,show_progress_bar=True)
    if embeddings.ndim!=2 or len(embeddings)!=len(rows) or not np.isfinite(embeddings).all():
        raise ValueError('Invalid encoder output.')
    pred,prob,fold_metrics=evaluate(embeddings,y,folds)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    comparison=[]
    for name,values in [('tfidf',baseline_pred),('embeddings',pred)]:
        comparison.append({'model':name,'accuracy':accuracy_score(y,values),
                           'macro_f1':f1_score(y,values,average='macro',zero_division=0),
                           'weighted_f1':f1_score(y,values,average='weighted',zero_division=0)})
    report=classification_report(y,pred,labels=sorted(CATEGORIES),output_dict=True,zero_division=0)
    metrics={'encoder':a.encoder,'encoder_frozen':True,'normalization':True,'classifier_C':1.0,
             'class_weight':'balanced','training_rows':len(rows),'folds':fold_metrics,
             'comparison':comparison,'classification_report':report,'input_sha256':baseline['input_sha256'],
             'versions':{'sklearn':sklearn.__version__,'sentence_transformers':sentence_transformers.__version__},
             'note':'Same held-out folds as baseline. CV is for model comparison, not an untouched final test. Probabilities are uncalibrated.'}
    (a.out_dir/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    write_csv(a.out_dir/'comparison.csv',comparison,['model','accuracy','macro_f1','weighted_f1'])
    details=[{'tool_id':r['tool_id'],'tool_name':r['tool_name'],'description':r['description'],
              'fold':int(folds[i]),'actual_category':y[i],'tfidf_prediction':baseline_pred[i],
              'predicted_category':pred[i],'max_probability_uncalibrated':float(prob[i]),
              'correct':bool(pred[i]==y[i])} for i,r in enumerate(rows)]
    write_csv(a.out_dir/'cv_predictions.csv',details,list(details[0]))
    labels=sorted(CATEGORIES);cm=confusion_matrix(y,pred,labels=labels)
    write_csv(a.out_dir/'confusion_matrix.csv',
              [dict(actual_category=c,**{label:int(cm[i,j]) for j,label in enumerate(labels)}) for i,c in enumerate(labels)],['actual_category']+labels)
    text=classification_report(y,pred,labels=labels,zero_division=0)
    (a.out_dir/'classification_report.txt').write_text(text,encoding='utf-8')
    final=LogisticRegression(C=1.0,class_weight='balanced',solver='lbfgs',max_iter=3000).fit(embeddings,y)
    joblib.dump(final,a.out_dir/'classifier.joblib')
    encoder.save(str(a.out_dir/'encoder'))
    print('\n'+text+'\nComparison:\n'+json.dumps(comparison,indent=2))
    print('Saved local encoder and final classifier. No training CSV modified.')


if __name__=='__main__':main()
