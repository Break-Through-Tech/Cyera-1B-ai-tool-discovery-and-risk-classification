"""Frozen sentence embeddings + classifiers, cosine neighbors and baseline comparison.

Training/evaluation reuse issue #6's splits and source-review exclusions.
Unseen means held out from this project's classifier, not encoder pretraining.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from train_baseline import ROOT, apply_source_review, balanced_source_weights, load, metrics, save, validate
from embedding_utils import ENCODER_ID, ENCODER_REVISION, encode_rows, load_encoder, nearest


def classifier():
    return LogisticRegression(C=1.0,max_iter=2000,random_state=42)


def choose_default(embedding_score, baseline_score):
    return 'embeddings' if embedding_score > baseline_score else 'tfidf'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/processed/classification_full')
    p.add_argument('--source-review',type=Path,default=ROOT/'data/reviews/source_review_2026-11-04.json')
    p.add_argument('--baseline',type=Path,default=ROOT/'artifacts/baseline')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/embeddings')
    p.add_argument('--cache',type=Path,default=ROOT/'artifacts/embedding_cache')
    p.add_argument('--overwrite',action='store_true')
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()) and not a.overwrite:
        p.error('Output directory exists; use --overwrite or another --out')
    raw = {s:load(a.data/f'{s}.csv') for s in ('train','validation','test')}
    validate({s:[dict(r,is_ai_tool=int(r['is_ai_tool'])) for r in rows] for s,rows in raw.items()})
    baseline_config = json.loads((a.baseline/'model_config.json').read_text())
    hashes = {f'{s}.csv':hashlib.sha256((a.data/f'{s}.csv').read_bytes()).hexdigest() for s in raw}
    review_hash = hashlib.sha256(a.source_review.read_bytes()).hexdigest()
    if hashes != baseline_config['inputs'] or review_hash != baseline_config['source_review']['sha256']:
        raise ValueError('Baseline comparison requires the identical dataset and source-review version')
    splits,excluded = apply_source_review(raw,json.loads(a.source_review.read_text()))
    for s in ('validation','test'):
        if any(r['label_source']=='provisional_catalog_membership' for r in splits[s]):
            raise ValueError('Provisional membership is not an evaluation label')
    a.out.mkdir(parents=True,exist_ok=True)
    encoder = load_encoder(a.cache/'encoder')
    features = {s:encode_rows(encoder,rows,a.cache/'vectors') for s,rows in splits.items()}
    y = {s:np.array([int(r['is_ai_tool']) for r in rows]) for s,rows in splits.items()}
    anchor_mask = np.array([r['label_source']!='provisional_catalog_membership' for r in splits['train']])
    masks = {'reviewed_only':anchor_mask,'full_catalog':np.ones(len(splits['train']),dtype=bool)}
    reports,models = {},{}
    for name,mask in masks.items():
        rows = [r for r,keep in zip(splits['train'],mask) if keep]
        model = classifier().fit(features['train'][mask],y['train'][mask],sample_weight=balanced_source_weights(rows))
        models[name] = model
        reports[name] = dict(training_rows=len(rows),
            validation=metrics(y['validation'],model.predict(features['validation']),[0,1],['non_ai','ai']))
        joblib.dump(model,a.out/f'binary_{name}.joblib')
    selected = max(('reviewed_only','full_catalog'),key=lambda n:reports[n]['validation']['macro_f1'])
    baseline_report = json.loads((a.baseline/'metrics.json').read_text())
    base_selected = baseline_config['selected_binary_model']
    default_family = choose_default(reports[selected]['validation']['macro_f1'],
                                   baseline_report['binary'][base_selected]['validation']['macro_f1'])
    # Select before examining test predictions.
    for name,model in models.items():
        preds = model.predict(features['test']);scores = model.predict_proba(features['test'])[:,list(model.classes_).index(1)]
        reports[name]['test'] = metrics(y['test'],preds,[0,1],['non_ai','ai'])
        save(a.out/f'test_predictions_{name}.csv',
             [dict(application_id=r['application_id'],tool_name=r['tool_name'],expected_is_ai_tool=int(truth),
                   predicted_is_ai_tool=int(pred),model_ai_score=float(score))
              for r,truth,pred,score in zip(splits['test'],y['test'],preds,scores)],
             ['application_id','tool_name','expected_is_ai_tool','predicted_is_ai_tool','model_ai_score'])
    category_mask = np.array([r['is_ai_tool']=='1' and bool(r['category']) and
                              r['label_source']=='gemini_text_review' for r in splits['train']])
    category_rows = [r for r,keep in zip(splits['train'],category_mask) if keep]
    cy = [r['category'] for r in category_rows];categories = sorted(set(cy));counts = Counter(cy)
    category_model = classifier().fit(features['train'][category_mask],cy,
        sample_weight=np.array([len(cy)/(len(categories)*counts[label]) for label in cy]))
    joblib.dump(category_model,a.out/'category.joblib')
    category_report = {}
    for s in ('validation','test'):
        mask = np.array([r['is_ai_tool']=='1' and bool(r['category']) for r in splits[s]])
        labels = [r['category'] for r,keep in zip(splits[s],mask) if keep]
        pred = category_model.predict(features[s][mask])
        category_report[s] = dict(rows=len(labels),metrics=metrics(labels,pred,categories,categories))
    category_family = choose_default(category_report['validation']['metrics']['macro_f1'],
                                    baseline_report['category']['validation']['metrics']['macro_f1'])
    refs = [r for r,keep in zip(splits['train'],anchor_mask) if keep]
    reference_vectors = features['train'][anchor_mask]
    heldout_groups = {r['group_id'] for s in ('validation','test') for r in splits[s]}
    if heldout_groups & {r['group_id'] for r in refs}:
        raise ValueError('Held-out application leaked into similarity references')
    np.savez_compressed(a.out/'reference_embeddings.npz',embeddings=reference_vectors)
    (a.out/'reference_records.json').write_text(json.dumps(refs,indent=2)+'\n')
    neighbor_report = {}
    neighbor_rows = []
    for s in ('validation','test'):
        ids,scores = nearest(features[s],reference_vectors,k=3)
        preds = [int(refs[row[0]]['is_ai_tool']) for row in ids]
        neighbor_report[s] = metrics(y[s],preds,[0,1],['non_ai','ai'])
        for query,indices,similarities in zip(splits[s],ids,scores):
            for rank,(idx,sim) in enumerate(zip(indices,similarities),1):
                ref = refs[int(idx)]
                neighbor_rows.append(dict(split=s,query_id=query['application_id'],query_name=query['tool_name'],
                    expected_is_ai_tool=query['is_ai_tool'],rank=rank,reference_id=ref['application_id'],
                    reference_name=ref['tool_name'],reference_is_ai_tool=ref['is_ai_tool'],
                    reference_category=ref['category'],cosine_similarity=float(sim)))
    save(a.out/'heldout_neighbors.csv',neighbor_rows,list(neighbor_rows[0]))
    config = dict(encoder_id=ENCODER_ID,encoder_revision=ENCODER_REVISION,embedding_dimension=int(reference_vectors.shape[1]),
        encoder_frozen=True,normalized=True,device='cpu',threshold=.5,
        selected_embedding_variant=selected,default_binary_family=default_family,default_category_family=category_family,
        baseline_binary_variant=base_selected,category_labels=categories,
        reference_rows=len(refs),reference_policy='model-reviewed/seed training records only; excludes held-out groups',
        inputs=hashes,source_review_sha256=review_hash,excluded_training_ids=[r['source_record_id'] for r in excluded],
        label_scope=baseline_config['label_scope'],
        versions=dict(sentence_transformers=__import__('sentence_transformers').__version__,
                      transformers=__import__('transformers').__version__,torch=__import__('torch').__version__,
                      numpy=np.__version__,sklearn=__import__('sklearn').__version__),
        baseline_config_sha256=hashlib.sha256((a.baseline/'model_config.json').read_bytes()).hexdigest())
    result = dict(binary=reports,category=category_report,nearest_neighbor_binary=neighbor_report,
        comparison=dict(tfidf_selected=base_selected,embedding_selected=selected,
            tfidf_binary_test=baseline_report['binary'][base_selected]['test'],
            tfidf_category_test=baseline_report['category']['test'],
            default_binary_family=default_family,default_category_family=category_family),
        evaluation_scope='Same previously inspected 50-row weak-label holdout; unseen only relative to this project classifier training',
        pretrained_encoder_may_have_seen_products=True,independent_fresh_evaluation=False,
        similarity_relevance_human_verified=False,
        limitations=['Only four negative examples per evaluation split.',
                    'Semantic similarity does not establish AI functionality, approval or governance risk.',
                    'Pretrained encoder is frozen; project test data never train either classifier.',
                    'Known source-review exclusions are applied identically to baseline and embeddings.'])
    for name,data in [('model_config.json',config),('metrics.json',result)]:
        (a.out/name).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(selected_embedding_variant=selected,default_binary_family=default_family,
        default_category_family=category_family,binary_test_macro_f1={n:r['test']['macro_f1'] for n,r in reports.items()},
        category_test_accuracy=category_report['test']['metrics']['accuracy']),indent=2),flush=True)


if __name__=='__main__':main()
