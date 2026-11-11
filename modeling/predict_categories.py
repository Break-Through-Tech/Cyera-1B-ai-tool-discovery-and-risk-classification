"""Load verified 14-category models and return provisional purpose suggestions."""
import json
from pathlib import Path
import joblib
from .train_categories import CATEGORIES,ROOT,OUT,sha,tree_hash


class CategoryPredictor:
    def __init__(self,path=OUT):
        self.path=Path(path);self.config=json.loads((self.path/'model_config.json').read_text())
        if self.config['taxonomy']!='original_kaggle_14' or self.config['categories']!=CATEGORIES:
            raise ValueError('Category taxonomy differs from the active 14-category scheme')
        family=self.config['selected_family'];self.family=family
        file=self.path/(family+'.joblib')
        if sha(file)!=self.config['model_sha256'][file.name]:raise ValueError('Category model hash mismatch')
        self.model=joblib.load(file)
        if set(self.model.classes_)!=set(CATEGORIES):raise ValueError('Saved classifier categories mismatch')
        self.encoder=None
        if family!='tfidf':
            source=ROOT/self.config['encoder_path']
            if tree_hash(source)!=self.config['encoder_sha256']:raise ValueError('Category encoder hash mismatch')
            from sentence_transformers import SentenceTransformer
            self.encoder=SentenceTransformer(str(source.resolve()),local_files_only=True,trust_remote_code=False,device='cpu')

    def predict(self,description):
        text=' '.join(description.split())
        if not text:return dict(category=None,needs_review=True,reason='Missing product description')
        if self.encoder is None:
            if not self.model.named_steps['tfidf'].transform([text]).nnz:
                return dict(category=None,needs_review=True,reason='No recognized category vocabulary')
            values=[text]
        else:values=self.encoder.encode([text],normalize_embeddings=True,convert_to_numpy=True)
        scores=self.model.predict_proba(values)[0];order=scores.argsort()[::-1][:3]
        return dict(category=str(self.model.classes_[order[0]]),model_score=float(scores[order[0]]),
            family=self.family,taxonomy='original_kaggle_14',needs_review=True,
            top_categories=[dict(category=str(self.model.classes_[i]),model_score=float(scores[i])) for i in order],
            caveat='Experimental purpose suggestion, not independently verified; scores are uncalibrated.')
