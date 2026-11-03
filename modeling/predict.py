"""Predict AI/non-AI and suggest a business category for a described application."""
import argparse
import json
from pathlib import Path
import joblib

ROOT=Path(__file__).resolve().parents[1]


def predict(models,name,description,model_name=None):
    if not name.strip() or not description.strip():
        raise ValueError('Provide both a name and a concrete product description')
    config=json.loads((models/'model_config.json').read_text())
    choice=model_name or config['selected_binary_model']
    if choice not in ('reviewed_only','full_catalog'):raise ValueError('Unknown model')
    binary=joblib.load(models/f"binary_{choice}.joblib")
    text=f"{' '.join(name.split())}. {' '.join(description.split())}"
    vector=binary.named_steps['tfidf'].transform([text])
    if vector.nnz==0:
        return dict(tool_name=name,is_ai_tool=None,category=None,needs_review=True,reason='No known vocabulary; insufficient evidence')
    score=float(binary.predict_proba([text])[0,list(binary.classes_).index(1)])
    label=int(score>=config['threshold'])
    category=None;category_score=None
    if label:
        categorizer=joblib.load(models/'category.joblib')
        scores=categorizer.predict_proba([text])[0]
        i=int(scores.argmax());category=str(categorizer.classes_[i]);category_score=float(scores[i])
    return dict(tool_name=name,is_ai_tool=label,model_ai_score=score,category=category,
                category_model_score=category_score,needs_review=True,
                reason='Experimental model trained with weak labels and 30 negative examples; scores are uncalibrated')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,default=ROOT/'artifacts/baseline')
    p.add_argument('--binary',choices=['reviewed_only','full_catalog'],help='Override validation-selected default')
    p.add_argument('--name',required=True)
    p.add_argument('--description',required=True)
    a=p.parse_args()
    print(json.dumps(predict(a.models,a.name,a.description,a.binary),indent=2))


if __name__=='__main__':main()
