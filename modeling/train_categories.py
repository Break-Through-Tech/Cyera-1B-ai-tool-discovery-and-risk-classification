"""Compare category classifiers on the current 14-category taxonomy, locally."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
from sklearn.model_selection import GridSearchCV,StratifiedGroupKFold

from .train_category_model import load_training,make_model,write_csv

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/final/dataset_b_final_kaggle_categories.csv'
OUT=ROOT/'artifacts/categories_kaggle14_2026-11-11'
ENCODER=ROOT/'artifacts/category_embeddings/encoder'
CATEGORIES=['Ai And Machine Learning','Business And Finance','Communication And Support',
 'Creative And Entertainment','Data And Analytics','Education And Learning','Industry Specific',
 'Marketing And Sales','Media And Content Creation','Operations And Management','Personal And Lifestyle',
 'Security And Privacy','Specialized Technologies','Technology And Development']
C_VALUES=[.01,.1,1.,10.,100.]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tree_hash(path):
    paths=sorted(p for p in Path(path).rglob('*') if p.is_file())
    return hashlib.sha256('\n'.join(str(p.relative_to(path))+':'+sha(p) for p in paths).encode()).hexdigest()
def classifier(c=1.,weight='balanced'):
    return LogisticRegression(C=c,class_weight=weight,max_iter=5000,solver='lbfgs')
def splits(y,groups,folds=4,seed=42,minimum_training_groups=1):
    n=min(folds,min(len(set(groups[y==c])) for c in CATEGORIES))
    if n<2:raise ValueError('Insufficient independent groups per category')
    # Structural seed search only: never inspect model scores. Rare categories
    # can otherwise vanish from training under grouped stratification.
    for candidate in range(seed,seed+100):
        result=list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=candidate).split(np.zeros(len(y)),y,groups))
        valid=all(set(y[tr])==set(CATEGORIES) and all(len(set(groups[tr][y[tr]==c]))>=minimum_training_groups for c in CATEGORIES) for tr,va in result)
        if valid:
            for tr,va in result:
                if set(groups[tr])&set(groups[va]):raise ValueError('Related products cross folds')
            return result
    raise ValueError('Cannot construct category-complete grouped folds')

def macro_score(estimator,x,y):
    return f1_score(y,estimator.predict(x),labels=CATEGORIES,average='macro',zero_division=0)
def tune(x,y,groups,seed):
    grid=GridSearchCV(classifier(),{'C':C_VALUES,'class_weight':[None,'balanced']},
                      cv=splits(y,groups,2,seed),scoring=macro_score,refit=True,n_jobs=1,error_score='raise')
    grid.fit(x,y)
    return grid

def metrics(y,pred):
    return dict(accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,labels=CATEGORIES,average='macro',zero_division=0)),
        weighted_f1=float(f1_score(y,pred,average='weighted',zero_division=0)))


def train(data=DATA,out=OUT,encoder_path=ENCODER):
    data,out,encoder_path=Path(data),Path(out),Path(encoder_path)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty experiment directory')
    rows,excluded,groups=load_training(data,CATEGORIES)
    texts=np.array([r['description'] for r in rows]);y=np.array([r['category'] for r in rows])
    # Encoder is already checked in; no downloads or weight updates.
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    encoder=SentenceTransformer(str(encoder_path.resolve()),local_files_only=True,trust_remote_code=False,device='cpu')
    vectors=encoder.encode(list(texts),normalize_embeddings=True,convert_to_numpy=True,batch_size=32,show_progress_bar=True)
    if not np.isfinite(vectors).all():raise ValueError('Invalid embedding output')
    folds=splits(y,groups,minimum_training_groups=2)
    families=['tfidf','embeddings_default','embeddings_tuned']
    predictions={name:np.empty(len(rows),dtype=object) for name in families}
    scores={name:np.zeros(len(rows)) for name in families};fold_ids=np.zeros(len(rows),dtype=int)
    baseline=np.empty(len(rows),dtype=object);searches=[]
    for number,(tr,va) in enumerate(folds,1):
        fold_ids[va]=number
        baseline[va]=Counter(y[tr]).most_common(1)[0][0]
        models={'tfidf':make_model().fit(texts[tr],y[tr]),
                'embeddings_default':classifier().fit(vectors[tr],y[tr])}
        tuned=tune(vectors[tr],y[tr],groups[tr],42+number);models['embeddings_tuned']=tuned.best_estimator_
        searches.append(dict(outer_fold=number,selected=tuned.best_params_,inner_macro_f1=float(tuned.best_score_)))
        for name,model in models.items():
            values=texts[va] if name=='tfidf' else vectors[va]
            predictions[name][va]=model.predict(values);scores[name][va]=model.predict_proba(values).max(axis=1)
        print(json.dumps({'fold':number,'results':{name:metrics(y[va],predictions[name][va]) for name in families}}),flush=True)
    reports={name:metrics(y,pred) for name,pred in predictions.items()}
    # Exploratory selection: report selection bias, not an independent final test.
    chosen=max(families,key=lambda name:reports[name]['macro_f1'])
    final_tuning=tune(vectors,y,groups,42)
    final_models={'tfidf':make_model().fit(texts,y),'embeddings_default':classifier().fit(vectors,y),
                  'embeddings_tuned':final_tuning.best_estimator_}
    out.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name,model in final_models.items():
        file=out/(name+'.joblib');joblib.dump(model,file);hashes[file.name]=sha(file)
        report=classification_report(y,predictions[name],labels=CATEGORIES,output_dict=True,zero_division=0)
        (out/(name+'_report.json')).write_text(json.dumps(report,indent=2)+'\n')
        cm=confusion_matrix(y,predictions[name],labels=CATEGORIES)
        write_csv(out/(name+'_confusion.csv'),[dict(actual=c,**{label:int(cm[i,j]) for j,label in enumerate(CATEGORIES)}) for i,c in enumerate(CATEGORIES)],['actual',*CATEGORIES])
    comparison=[dict(model=name,**report) for name,report in reports.items()]
    write_csv(out/'comparison.csv',comparison,['model','accuracy','macro_f1','weighted_f1'])
    prediction_rows=[dict(tool_id=r['tool_id'],tool_name=r['tool_name'],group=int(groups[i]),fold=int(fold_ids[i]),actual_category=y[i],
                         **{name+'_prediction':predictions[name][i] for name in families}) for i,r in enumerate(rows)]
    write_csv(out/'cv_predictions.csv',prediction_rows,list(prediction_rows[0]))
    write_csv(out/'excluded_rows.csv',excluded,['tool_id','reason'])
    config=dict(taxonomy='original_kaggle_14',categories=CATEGORIES,input_sha256=sha(data),rows=len(rows),excluded_rows=len(excluded),
        category_counts=dict(Counter(y)),outer_folds=len(folds),inner_folds=2,seed=42,
        fold_policy="First of up to 100 sequential seeds meeting training-category/group coverage, using labels/groups only; no model-score selection.",comparison=reports,majority_baseline=metrics(y,baseline),
        selected_family=chosen,selection_rule='Exploratory highest outer grouped-CV macro F1; family selection reuses these estimates.',
        outer_tuning=searches,final_tuned_params=final_tuning.best_params_,model_sha256=hashes,
        encoder_path=str(encoder_path.relative_to(ROOT)),encoder_sha256=tree_hash(encoder_path),encoder_frozen=True,
        probabilities_calibrated=False,independent_final_test=False,
        versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,sentence_transformers=sentence_transformers.__version__),
        limitations=['Only 306 labeled AI records; smallest category has four examples.',
            'Existing Kaggle categories can be noisy and are not independently verified.',
            'Nested tuning protects outer folds from parameter selection, but family comparison is exploratory.',
            'Many products were seen in earlier experiments; this is not a fresh independent benchmark.',
            'Final deployment models are refitted on all eligible records after cross-validation.',
            'Scores are uncalibrated; suggestions always request review.'])
    (out/'model_config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps({'selected_family':chosen,'comparison':reports,'baseline':config['majority_baseline']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--encoder',type=Path,default=ENCODER)
    a=p.parse_args();train(a.data,a.out,a.encoder)

if __name__=='__main__':main()
