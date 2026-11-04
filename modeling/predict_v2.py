"""Predict primary-purpose AI status and category using validation-selected v2 models."""
import argparse
import json
from pathlib import Path

import joblib
from embedding_utils import load_encoder
from train_v2 import OUT, CACHE, verified_config


class V2Predictor:
    def __init__(self, models=OUT, cache=CACHE):
        self.models,self.cache=Path(models),Path(cache)
        self.config=verified_config(self.models)
        self.binary_family=self.config['default_binary_family']
        self.category_family=self.config['default_category_family']
        self.binary=joblib.load(self.models/f'{self.binary_family}_binary.joblib')
        self.category=joblib.load(self.models/f'{self.category_family}_category.joblib')
        self.encoder=None
        if 'embeddings' in (self.binary_family,self.category_family):
            self.encoder=load_encoder(self.cache/'encoder')

    def predict(self,description,name=''):
        description=' '.join(description.split())
        if not description:raise ValueError('A concrete product description is required')
        vector=None
        if self.encoder is not None:
            vector=self.encoder.encode([description],convert_to_numpy=True,normalize_embeddings=True)
        bx=[description] if self.binary_family=='tfidf' else vector
        label=score=category=category_score=None
        if self.binary_family!='tfidf' or self.binary.named_steps['tfidf'].transform(bx).nnz:
            score=float(self.binary.predict_proba(bx)[0,list(self.binary.classes_).index(1)])
            label=int(score>=self.config['threshold'])
        if label==1:
            cx=[description] if self.category_family=='tfidf' else vector
            if self.category_family!='tfidf' or self.category.named_steps['tfidf'].transform(cx).nnz:
                probabilities=self.category.predict_proba(cx)[0]
                i=int(probabilities.argmax());category=str(self.category.classes_[i]);category_score=float(probabilities[i])
        return dict(tool_name=name,is_ai_tool=label,model_ai_score=score,category=category,category_model_score=category_score,
            binary_family=self.binary_family,category_family=self.category_family,dataset_version='2.0',
            needs_review=True,label_scope=self.config['label_scope'],
            caveat='Experimental assistant-reviewed labels. Scores are uncalibrated; AI features and governance risk are not predicted.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--description',required=True)
    p.add_argument('--name',default='')
    p.add_argument('--models',type=Path,default=OUT)
    p.add_argument('--cache',type=Path,default=CACHE)
    a=p.parse_args()
    print(json.dumps(V2Predictor(a.models,a.cache).predict(a.description,a.name),indent=2))


if __name__=='__main__':main()
