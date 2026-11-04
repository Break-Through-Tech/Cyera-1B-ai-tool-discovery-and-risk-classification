"""Predict with the validation-selected family and show similar training tools."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from embedding_utils import ENCODER_ID, ENCODER_REVISION, load_encoder, nearest
from train_baseline import ROOT


class EmbeddingPredictor:
    def __init__(self,models,baseline,cache):
        self.models,self.baseline=Path(models),Path(baseline)
        self.config=json.loads((self.models/'model_config.json').read_text())
        if self.config['encoder_id']!=ENCODER_ID or self.config['encoder_revision']!=ENCODER_REVISION:
            raise ValueError('Saved classifier and configured encoder do not match')
        if hashlib.sha256((self.baseline/'model_config.json').read_bytes()).hexdigest()!=self.config['baseline_config_sha256']:
            raise ValueError('Baseline changed since embedding comparison; reproduce comparison before predicting')
        self.encoder=load_encoder(Path(cache)/'encoder')
        self.records=json.loads((self.models/'reference_records.json').read_text())
        with np.load(self.models/'reference_embeddings.npz',allow_pickle=False) as f:
            self.reference=f['embeddings'].copy()
        if self.reference.shape!=(len(self.records),self.config['embedding_dimension']):
            raise ValueError('Similarity reference metadata and embeddings do not align')
        self.binary_models={
            'embeddings':joblib.load(self.models/f"binary_{self.config['selected_embedding_variant']}.joblib"),
            'tfidf':joblib.load(self.baseline/f"binary_{self.config['baseline_binary_variant']}.joblib")}
        self.category_models={'embeddings':joblib.load(self.models/'category.joblib'),
                              'tfidf':joblib.load(self.baseline/'category.joblib')}

    def predict(self,name,description,family='auto',k=3):
        if not name.strip() or not description.strip():
            raise ValueError('A name and concrete product description are required')
        if family not in ('auto','embeddings','tfidf'):
            raise ValueError('Unknown classifier family')
        text=f"{' '.join(name.split())}. {' '.join(description.split())}"
        vector=self.encoder.encode([text],convert_to_numpy=True,normalize_embeddings=True)
        binary_family=self.config['default_binary_family'] if family=='auto' else family
        model=self.binary_models[binary_family]
        label,score=None,None
        if binary_family=='embeddings' or model.named_steps['tfidf'].transform([text]).nnz:
            score=float(model.predict_proba(vector if binary_family=='embeddings' else [text])[0,list(model.classes_).index(1)])
            label=int(score>=self.config['threshold'])
        category,category_score=None,None
        category_family=self.config['default_category_family'] if family=='auto' else family
        if label==1:
            cm=self.category_models[category_family]
            scores=cm.predict_proba(vector if category_family=='embeddings' else [text])[0]
            i=int(scores.argmax());category=str(cm.classes_[i]);category_score=float(scores[i])
        ids,scores=nearest(vector,self.reference,k)
        matches=[]
        for idx,similarity in zip(ids[0],scores[0]):
            r=self.records[int(idx)]
            matches.append(dict(application_id=r['application_id'],tool_name=r['tool_name'],
                description=r['description'],cosine_similarity=float(similarity),
                supplied_is_ai_tool=int(r['is_ai_tool']),supplied_category=r['category'] or None,
                label_source=r['label_source']))
        return dict(tool_name=name,is_ai_tool=label,model_ai_score=score,binary_family=binary_family,
            category=category,category_model_score=category_score,category_family=category_family,
            similar_training_tools=matches,label_scope=self.config['label_scope'],needs_review=True,
            caveat='Experimental weak-label models; cosine similarity is relatedness, not proof of AI functionality or governance risk')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,default=ROOT/'artifacts/embeddings')
    p.add_argument('--baseline',type=Path,default=ROOT/'artifacts/baseline')
    p.add_argument('--cache',type=Path,default=ROOT/'artifacts/embedding_cache')
    p.add_argument('--name',required=True)
    p.add_argument('--description',required=True)
    p.add_argument('--family',choices=['auto','embeddings','tfidf'],default='auto')
    p.add_argument('--top-k',type=int,default=3)
    a=p.parse_args()
    model=EmbeddingPredictor(a.models,a.baseline,a.cache)
    print(json.dumps(model.predict(a.name,a.description,a.family,a.top_k),indent=2))


if __name__=='__main__':main()
