"""Tune C and class_weight with nested grouped cross-validation.

Legacy 12-category reproduction. Use python -m modeling.train_categories for the
active original-Kaggle 14-category comparison.

Keep train_category_model.py and train_category_embeddings.py beside this file.
Uses the encoder saved by the previous embedding run. No data or old outputs changed.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
try:
    from .train_category_model import load_training, write_csv, CATEGORIES
    from .train_category_embeddings import load_folds
except ImportError:
    from train_category_model import load_training, write_csv, CATEGORIES
    from train_category_embeddings import load_folds


def macro_scorer(estimator, X, y):
    return f1_score(y,estimator.predict(X),labels=sorted(CATEGORIES),average='macro',zero_division=0)


def search(X,y,groups,seed,inner_folds,c_values,jobs):
    labels=sorted(CATEGORIES)
    n=min(inner_folds,min(len(set(groups[y==c])) for c in labels))
    if n<2:raise ValueError('Not enough independent groups for inner cross-validation.')
    splits=list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(X,y,groups))
    for tr,va in splits:
        if set(y[tr])!=set(labels):raise ValueError('Inner training fold lacks a category; try --inner-folds 2.')
        assert not set(groups[tr]) & set(groups[va])
    grid=GridSearchCV(LogisticRegression(solver='lbfgs',max_iter=5000),
        {'C':c_values,'class_weight':[None,'balanced']},
        scoring=macro_scorer,
        cv=splits,refit=True,n_jobs=jobs,error_score='raise',return_train_score=False)
    grid.fit(X,y)
    table=[{'C':p['C'],'class_weight':p['class_weight'] or 'none',
            'mean_inner_macro_f1':float(grid.cv_results_['mean_test_score'][i]),
            'std_inner_macro_f1':float(grid.cv_results_['std_test_score'][i]),
            'rank':int(grid.cv_results_['rank_test_score'][i])}
           for i,p in enumerate(grid.cv_results_['params'])]
    return grid,table,n


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--baseline-dir',type=Path,default=Path('artifacts/category_tfidf'))
    p.add_argument('--embedding-dir',type=Path,default=Path('artifacts/category_embeddings'))
    p.add_argument('--out-dir',type=Path,default=Path('artifacts/category_embeddings_tuned'))
    p.add_argument('--inner-folds',type=int,default=3)
    p.add_argument('--c-values',type=float,nargs='+',default=[0.01,0.1,1.0,10.0,100.0])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--jobs',type=int,default=1,help='Use 1 for low memory usage; -1 for all CPU cores.')
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.inner_folds<2 or any(not np.isfinite(c) or c<=0 for c in a.c_values):p.error('Invalid folds or C values.')
    if a.out_dir.exists() and any(a.out_dir.iterdir()):p.error('Choose a new/empty output directory.')
    rows,excluded,groups=load_training(a.input)
    digest=hashlib.sha256(a.input.read_bytes()).hexdigest()
    old=json.loads((a.embedding_dir/'metrics.json').read_text(encoding='utf-8'))
    base=json.loads((a.baseline_dir/'metrics.json').read_text(encoding='utf-8'))
    if old['input_sha256']!=digest or base['input_sha256']!=digest:p.error('Input changed since previous runs. Use the same dataset.')
    folds,tfidf_pred=load_folds(a.baseline_dir/'cv_predictions.csv',rows,groups)
    embedding_folds,embedding_pred=load_folds(a.embedding_dir/'cv_predictions.csv',rows,groups)
    if not np.array_equal(folds,embedding_folds):p.error('Previous models used different folds.')
    if not old.get('encoder_frozen') or old.get('normalization') is not True:p.error('Expected frozen, normalized baseline embeddings.')
    encoder_path=a.embedding_dir/'encoder'
    if not encoder_path.is_dir():p.error('Saved encoder missing; rerun the original embedding script first.')
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    print('Loading your saved encoder locally...',flush=True)
    encoder=SentenceTransformer(str(encoder_path.resolve()),device=a.device,local_files_only=True,trust_remote_code=False)
    X=encoder.encode([r['description'] for r in rows],normalize_embeddings=True,
                     convert_to_numpy=True,batch_size=32,show_progress_bar=True)
    if X.ndim!=2 or len(X)!=len(rows) or not np.isfinite(X).all():raise ValueError('Invalid embeddings.')
    y=np.array([r['category'] for r in rows]);labels=sorted(CATEGORIES)
    predictions=np.empty(len(y),dtype=object);probabilities=np.zeros(len(y))
    all_searches=[];outer=[]
    for fold in sorted(set(folds)):
        tr=np.flatnonzero(folds!=fold);va=np.flatnonzero(folds==fold)
        assert not set(groups[tr]) & set(groups[va])
        print(f'Outer fold {fold}: tuning {len(a.c_values)*2} configurations on training rows only...',flush=True)
        grid,table,n=search(X[tr],y[tr],groups[tr],a.seed+int(fold),a.inner_folds,a.c_values,a.jobs)
        predictions[va]=grid.predict(X[va]);probabilities[va]=grid.predict_proba(X[va]).max(axis=1)
        row={'fold':int(fold),'best_params':grid.best_params_,'inner_folds':n,
             'best_inner_macro_f1':float(grid.best_score_),
             'outer_accuracy':accuracy_score(y[va],predictions[va]),
             'outer_macro_f1':f1_score(y[va],predictions[va],labels=labels,average='macro',zero_division=0)}
        outer.append(row);all_searches.extend(dict(stage=f'outer_{fold}',**t) for t in table)
        print(json.dumps(row),flush=True)
    comparison=[]
    for name,pred in [('tfidf',tfidf_pred),('embeddings_default',embedding_pred),('embeddings_tuned_nested_cv',predictions)]:
        comparison.append({'model':name,'accuracy':accuracy_score(y,pred),
            'macro_f1':f1_score(y,pred,labels=labels,average='macro',zero_division=0),
            'weighted_f1':f1_score(y,pred,average='weighted',zero_division=0)})
    print('Selecting final deployment settings using CV on all eligible rows...',flush=True)
    final,table,n=search(X,y,groups,a.seed,a.inner_folds,a.c_values,a.jobs)
    all_searches.extend(dict(stage='final_refit_selection',**t) for t in table)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump(final.best_estimator_,a.out_dir/'classifier.joblib')
    encoder.save(str(a.out_dir/'encoder'))
    report=classification_report(y,predictions,labels=labels,zero_division=0)
    (a.out_dir/'classification_report.txt').write_text(report,encoding='utf-8')
    write_csv(a.out_dir/'comparison.csv',comparison,['model','accuracy','macro_f1','weighted_f1'])
    write_csv(a.out_dir/'tuning_results.csv',all_searches,list(all_searches[0]))
    details=[{'tool_id':r['tool_id'],'tool_name':r['tool_name'],'description':r['description'],
              'fold':int(folds[i]),'actual_category':y[i],'predicted_category':predictions[i],
              'previous_embedding_prediction':embedding_pred[i],
              'max_probability_uncalibrated':float(probabilities[i]),'correct':bool(y[i]==predictions[i])}
             for i,r in enumerate(rows)]
    write_csv(a.out_dir/'cv_predictions.csv',details,list(details[0]))
    cm=confusion_matrix(y,predictions,labels=labels)
    write_csv(a.out_dir/'confusion_matrix.csv',
        [dict(actual_category=c,**{label:int(cm[i,j]) for j,label in enumerate(labels)}) for i,c in enumerate(labels)],['actual_category']+labels)
    metrics={'comparison':comparison,'outer_folds':outer,'final_best_params':final.best_params_,
        'final_selection_inner_macro_f1':float(final.best_score_),'final_selection_inner_folds':n,
        'classification_report':classification_report(y,predictions,labels=labels,output_dict=True,zero_division=0),
        'input_sha256':digest,'training_rows':len(rows),'excluded_rows':len(excluded),
        'encoder':old['encoder'],'encoder_frozen':True,'normalization':True,
        'search_grid':{'C':a.c_values,'class_weight':[None,'balanced']},'seed':a.seed,
        'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'sentence_transformers':sentence_transformers.__version__},
        'evaluation_note':'Nested outer-fold predictions evaluate the tuning procedure. Final selection score is not a test score. Prior experiments inspected these outer folds; use fresh labeled data for final confirmation.',
        'probability_note':'Uncalibrated scores, not guarantees. No automatic acceptance threshold selected.'}
    (a.out_dir/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    assert hashlib.sha256(a.input.read_bytes()).hexdigest()==digest
    print('\n'+report+'\nComparison:\n'+json.dumps(comparison,indent=2))
    print('Final settings:',final.best_params_)
    print('Saved:',a.out_dir)


if __name__=='__main__':main()
