"""Train Kaggle-14 category classifiers from prepare_category_data.py outputs.

Python 3.11/3.12 recommended. See README.md. Training never runs during inference.
--validate-only checks the input contract without third-party packages.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

VERSION = 'kaggle14-training-v1'
TEXT_VERSION = 'purpose-description-functionality-v1'
EMBEDDING_VERSION = 'token-chunks-mean-normalized-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def load_prepared(directory):
    manifest = read_json(directory/'manifest.json')
    taxonomy = read_json(directory/'taxonomy.json')
    if manifest.get('text_contract_version') != TEXT_VERSION or manifest.get('feature_column') != 'model_text':
        raise ValueError('Unsupported preparation text contract; do not silently reuse old models.')
    for filename in ['training_rows.csv', 'taxonomy.json']:
        if manifest.get('outputs', {}).get(filename) != sha(directory/filename):
            raise ValueError(f'Preparation hash mismatch: {filename}; rerun preparation instead of editing outputs.')
    labels = taxonomy.get('categories', [])
    if taxonomy.get('version') != 'original_kaggle_14' or len(labels) != 14 or len(set(labels)) != 14:
        raise ValueError('Expected original Kaggle 14-category taxonomy')
    if taxonomy.get('sha256') != hashlib.sha256(json.dumps(labels).encode()).hexdigest():
        raise ValueError('Taxonomy hash mismatch')
    with (directory/'training_rows.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        needed = {'source_id','group_id','category','model_text','description_only_text','text_origin','text_sha256','text_contract_version','eligible'}
        if not needed <= set(reader.fieldnames or []):
            raise ValueError('Missing prepared columns')
        rows = list(reader)
    if not rows or len({r['source_id'] for r in rows}) != len(rows):
        raise ValueError('Empty dataset or duplicate source IDs')
    text_groups = {}
    for r in rows:
        if None in r or any(v is None for v in r.values()):
            raise ValueError('Malformed prepared CSV')
        if r['category'] not in labels or not r['group_id'] or not r['model_text'].strip():
            raise ValueError('Invalid category, group or text')
        if r['eligible'].lower() != 'true' or r['text_contract_version'] != TEXT_VERSION:
            raise ValueError('Ineligible or incompatible row found')
        if hashlib.sha256(r['model_text'].encode()).hexdigest() != r['text_sha256']:
            raise ValueError('Model text hash mismatch')
        k = ' '.join(r['model_text'].casefold().split())
        if k in text_groups and text_groups[k] != r['group_id']:
            raise ValueError('Identical text crosses preparation groups')
        text_groups[k] = r['group_id']
    if set(r['category'] for r in rows) != set(labels):
        raise ValueError('At least one category has no eligible examples')
    return rows, labels, manifest


def snapshot_hash(directory):
    files = sorted(p for p in directory.rglob('*') if p.is_file())
    if not files:
        raise ValueError('Empty encoder snapshot')
    return fingerprint({p.relative_to(directory).as_posix(): sha(p) for p in files})


def encode_texts(encoder, texts, batch_size=32):
    """Use all tokens, not just the first MiniLM window. Reuse at inference."""
    import numpy as np
    width = int(encoder.max_seq_length) - encoder.tokenizer.num_special_tokens_to_add(pair=False)
    if width < 8:
        raise ValueError('Invalid encoder context length')
    result, chunk_counts = [], []
    # Bound memory and preserve input order; batch_size controls encoder inference.
    for offset in range(0, len(texts), 128):
        chunks, spans = [], []
        for text in texts[offset:offset+128]:
            tokens = encoder.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            if not tokens:
                raise ValueError('Cannot embed empty text')
            start = len(chunks)
            for j in range(0, len(tokens), width):
                chunks.append(encoder.tokenizer.decode(tokens[j:j+width], skip_special_tokens=True))
            spans.append((start, len(chunks)))
            chunk_counts.append(len(chunks)-start)
        vectors = encoder.encode(chunks, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        for start, stop in spans:
            v = vectors[start:stop].mean(axis=0)
            norm = np.linalg.norm(v)
            if norm <= 0 or not np.isfinite(v).all():
                raise ValueError('Invalid embedding')
            result.append(v/norm)
        print(f'Embedded {min(offset+128,len(texts))}/{len(texts)} products', flush=True)
    return np.asarray(result, dtype=np.float32), chunk_counts


def embeddings(texts, args):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    cache = args.embedding_cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    encoder_dir = cache/'encoder'
    encoder_meta = cache/'encoder_source.json'
    source = {'model_id': args.encoder, 'requested_revision': args.encoder_revision}
    if encoder_dir.exists():
        if not encoder_meta.exists() or read_json(encoder_meta)['source'] != source:
            raise ValueError('Encoder cache source differs or is incomplete. Choose another --embedding-cache.')
        expected = read_json(encoder_meta)['snapshot_sha256']
        if snapshot_hash(encoder_dir) != expected:
            raise ValueError('Encoder cache was modified')
        encoder = SentenceTransformer(str(encoder_dir), local_files_only=True, trust_remote_code=False, device=args.device)
    else:
        encoder = SentenceTransformer(args.encoder, revision=args.encoder_revision,
                                      trust_remote_code=False, device=args.device)
        encoder.save(str(encoder_dir))
        expected = snapshot_hash(encoder_dir)
        write_json(encoder_meta, {'source': source, 'snapshot_sha256': expected})
    config = {'source': source, 'snapshot_sha256': expected, 'encoding': EMBEDDING_VERSION,
              'max_seq_length': encoder.max_seq_length, 'texts_sha256': fingerprint(texts),
              'sentence_transformers': importlib.metadata.version('sentence-transformers')}
    key = fingerprint(config)
    matrix_path, meta_path = cache/(key+'.npy'), cache/(key+'.json')
    if matrix_path.exists() and meta_path.exists():
        meta = read_json(meta_path)
        if meta['config'] != config or meta['matrix_sha256'] != sha(matrix_path):
            raise ValueError('Embedding cache hash mismatch')
        matrix = np.load(matrix_path, allow_pickle=False)
        chunks = meta['chunk_counts']
        print('Reusing cached embeddings.', flush=True)
    else:
        matrix, chunks = encode_texts(encoder, texts, args.batch_size)
        np.save(matrix_path, matrix, allow_pickle=False)
        write_json(meta_path, {'config': config, 'matrix_sha256': sha(matrix_path), 'chunk_counts': chunks})
    if matrix.ndim != 2 or matrix.shape[0] != len(texts) or not np.isfinite(matrix).all():
        raise ValueError('Invalid cached embedding matrix')
    return matrix, dict(config, encoder_path=str(encoder_dir), products_chunked=sum(n>1 for n in chunks))


def metrics(y, prediction, labels, groups):
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
    counts = Counter(groups)
    weights = np.array([1/counts[g] for g in groups])
    return {'rows': len(y), 'groups': len(counts),
            'accuracy': float(accuracy_score(y, prediction)),
            'macro_f1': float(f1_score(y, prediction, labels=labels, average='macro', zero_division=0)),
            'group_weighted_macro_f1': float(f1_score(y, prediction, labels=labels, average='macro', sample_weight=weights, zero_division=0)),
            'per_category': classification_report(y, prediction, labels=labels, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, prediction, labels=labels).tolist()}


def make_estimator(kind, c, class_weight, seed, max_iter):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    classifier = LogisticRegression(C=c, class_weight=class_weight, solver='lbfgs', max_iter=max_iter, random_state=seed)
    if kind == 'tfidf':
        return Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=1,
                         max_features=60000, sublinear_tf=True)), ('classifier', classifier)])
    return classifier


def check_split(y, groups, train, test, labels):
    if set(groups[train]) & set(groups[test]):
        raise ValueError('Group leakage detected')
    for name, idx in [('training',train), ('validation/test',test)]:
        if set(y[idx]) != set(labels):
            raise ValueError(f'{name} partition lacks a category; reduce --cv-folds/--outer-folds or inspect groups.')


def run(args):
    rows, labels, prep_manifest = load_prepared(args.prepared)
    summary = {'rows':len(rows), 'groups':len({r['group_id'] for r in rows}),
               'categories':dict(Counter(r['category'] for r in rows)),
               'text_origins':dict(Counter(r['text_origin'] for r in rows))}
    print(json.dumps(summary,indent=2),flush=True)
    if args.validate_only:
        print('Input validation passed. No split, training, downloads, or files written.')
        return
    import numpy as np
    import joblib
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('error', category=ConvergenceWarning)
    config = dict(version=VERSION, script_sha256=sha(Path(__file__)),
                  data_sha256=sha(args.prepared/'training_rows.csv'),
                  preparation_manifest_sha256=sha(args.prepared/'manifest.json'),
                  taxonomy_sha256=sha(args.prepared/'taxonomy.json'),
                  models=args.models, seed=args.seed, cv_folds=args.cv_folds, outer_folds=args.outer_folds,
                  c_values=args.c_values, class_weights=[None,'balanced'], max_iter=args.max_iter,
                  encoder=args.encoder if 'embeddings' in args.models else None,
                  encoder_revision=args.encoder_revision, description_ablation=args.description_ablation,
                  text_contract_version=TEXT_VERSION, embedding_version=EMBEDDING_VERSION,
                  python=platform.python_version(),
                  packages={name:importlib.metadata.version(name) for name in
                            ['numpy','scipy','scikit-learn','joblib'] +
                            (['sentence-transformers','torch','transformers'] if 'embeddings' in args.models else [])})
    run_id = fingerprint(config)
    if args.out.exists():
        finished = args.out/'manifest.json'
        if finished.exists():
            old = read_json(finished)
            if old.get('run_id') == run_id and all((args.out/name).is_file() and sha(args.out/name)==h for name,h in old['artifacts'].items()):
                print('Identical completed run exists; no retraining. See',args.out/'metrics.json')
                return
        raise ValueError('Output exists but is incomplete, different, or modified. Choose a new --out.')
    args.out.mkdir(parents=True)
    write_json(args.out/'run_config.json', config)
    y = np.asarray([r['category'] for r in rows])
    groups = np.asarray([r['group_id'] for r in rows])
    texts = np.asarray([r['model_text'] for r in rows], dtype=object)
    desc = np.asarray([r['description_only_text'] for r in rows], dtype=object)
    outer = StratifiedGroupKFold(n_splits=args.outer_folds,shuffle=True,random_state=args.seed)
    dev, test = next(outer.split(texts,y,groups))
    check_split(y,groups,dev,test,labels)
    cv = StratifiedGroupKFold(n_splits=args.cv_folds,shuffle=True,random_state=args.seed+1)
    folds = [(dev[a],dev[b]) for a,b in cv.split(texts[dev],y[dev],groups[dev])]
    for tr,va in folds:
        check_split(y,groups,tr,va,labels)
    fold_map = {int(i):j for j,(_,va) in enumerate(folds) for i in va}
    dev_set = set(dev.tolist())
    write_csv(args.out/'split_manifest.csv', [dict(source_id=r['source_id'],group_id=r['group_id'],category=r['category'],
              partition='development' if i in dev_set else 'test',cv_validation_fold=fold_map.get(i,'')) for i,r in enumerate(rows)],
              ['source_id','group_id','category','partition','cv_validation_fold'])
    print(f'Development: {len(dev)}; held-out test: {len(test)}. No groups cross the boundary.',flush=True)
    features = {'tfidf':texts}
    embedding_meta = None
    if 'embeddings' in args.models:
        features['embeddings'], embedding_meta = embeddings(texts.tolist(),args)
        write_json(args.out/'embedding_manifest.json',embedding_meta)
    comparisons, chosen, cv_predictions = [], {}, {}
    for kind in args.models:
        best = None
        for c in args.c_values:
            for weight in [None,'balanced']:
                oof = np.empty(len(rows),dtype=object)
                start = time.perf_counter()
                for fold,(tr,va) in enumerate(folds):
                    print(f'{kind}, C={c}, class_weight={weight}, fold {fold+1}/{len(folds)}',flush=True)
                    estimator = make_estimator(kind,c,weight,args.seed,args.max_iter)
                    estimator.fit(features[kind][tr],y[tr])
                    oof[va] = estimator.predict(features[kind][va])
                m = metrics(y[dev],oof[dev],labels,groups[dev])
                row = dict(model=kind,C=c,class_weight=weight or 'none',
                           cv_macro_f1=m['macro_f1'],cv_group_weighted_macro_f1=m['group_weighted_macro_f1'],
                           cv_accuracy=m['accuracy'],seconds=round(time.perf_counter()-start,2))
                comparisons.append(row)
                if best is None or row['cv_macro_f1'] > best['cv_macro_f1']:
                    best = row
                    cv_predictions[kind] = oof[dev].copy()
        chosen[kind] = best
    winner = max(args.models,key=lambda k:chosen[k]['cv_macro_f1'])
    # Lock winner/configurations before any test prediction or metric is computed.
    write_json(args.out/'selection.json',{'winner':winner,'criterion':'development pooled out-of-fold macro-F1',
                                         'selected_configs':chosen,'test_used_for_selection':False})
    write_csv(args.out/'model_comparison.csv',comparisons,list(comparisons[0]))
    write_csv(args.out/'cv_predictions.csv', [dict(source_id=rows[i]['source_id'],group_id=groups[i],actual=y[i],
              **{k:str(cv_predictions[k][j]) for k in args.models}) for j,i in enumerate(dev)],
              ['source_id','group_id','actual']+args.models)
    report = {'winner':winner,'selection':'development_cv_only','labels':labels,'development_rows':len(dev),
              'test_rows':len(test),'models':{},'baseline':{},'description_ablation':None,
              'limitations':[
                  'Kaggle labels are supplied ground truth, not independently validated business-purpose truth.',
                  'Prepared groups cover documented duplicate signals; undetected paraphrases and aliases remain a leakage risk.',
                  'Kaggle text and web-enriched text differ. Kaggle holdout scores do not establish discovery accuracy.',
                  'Scores are uncalibrated; no automatic category-acceptance threshold or API fallback is implemented.',
                  'Saved models were fitted on development rows only. Held-out test products were not used for training.',
                  'A frozen pretrained encoder may have encountered product names in pretraining; held-out means absent from supervised fitting.',
                  'Do not choose a new winner based on these test scores; any further tuning needs a fresh final evaluation.']}
    majority = Counter(y[dev]).most_common(1)[0][0]
    report['baseline'] = dict(label=majority,**metrics(y[test],np.repeat(majority,len(test)),labels,groups[test]))
    test_predictions, per_category, confusion_pairs = [], [], []
    for kind in args.models:
        cfg = chosen[kind]
        estimator = make_estimator(kind,cfg['C'],None if cfg['class_weight']=='none' else cfg['class_weight'],args.seed,args.max_iter)
        estimator.fit(features[kind][dev],y[dev])
        pred = estimator.predict(features[kind][test])
        probs = estimator.predict_proba(features[kind][test])
        m = metrics(y[test],pred,labels,groups[test])
        m['subsets'] = {}
        for subset in ['description_plus_primary_task','description_only','primary_task_only']:
            positions = np.asarray([j for j,i in enumerate(test) if rows[i]['text_origin']==subset],dtype=int)
            m['subsets'][subset] = metrics(y[test][positions],pred[positions],labels,groups[test][positions]) if len(positions) else None
        report['models'][kind] = {'development_cv':cfg,'test':m}
        bundle = {'estimator':estimator,'model_kind':kind,'taxonomy':labels,'text_contract_version':TEXT_VERSION,
                  'feature_column':'model_text','training_partition':'development_only','run_id':run_id,
                  'embedding':embedding_meta if kind=='embeddings' else None,'config':config}
        joblib.dump(bundle,args.out/(kind+'.joblib'))
        for j,i in enumerate(test):
            order = np.argsort(probs[j])[::-1][:3]
            test_predictions.append(dict(model=kind,source_id=rows[i]['source_id'],group_id=groups[i],
                tool_name=rows[i].get('tool_name',''),actual=y[i],predicted=pred[j],correct=bool(pred[j]==y[i]),
                text_origin=rows[i]['text_origin'],top_score_uncalibrated=float(probs[j,order[0]]),
                top_three=json.dumps([{'category':str(estimator.classes_[k]),'score':float(probs[j,k])} for k in order]),
                model_text=rows[i]['model_text']))
        for label in labels:
            per_category.append(dict(model=kind,category=label,**m['per_category'][label]))
        write_csv(args.out/(kind+'_confusion_matrix.csv'),
                  [dict(actual=label,**dict(zip(labels,m['confusion_matrix'][i]))) for i,label in enumerate(labels)], ['actual']+labels)
        for i,a in enumerate(labels):
            for j,b in enumerate(labels):
                if i!=j and m['confusion_matrix'][i][j]:
                    confusion_pairs.append(dict(model=kind,actual=a,predicted=b,count=m['confusion_matrix'][i][j]))
    if args.description_ablation:
        # Diagnostic only, not a third candidate or another test-set selection.
        cfg = chosen.get('tfidf')
        if cfg is None:
            raise ValueError('--description-ablation requires tfidf in --models')
        predictions, truth, gs, combined_pred = [],[],[],[]
        combined_by_idx = dict(zip(dev.tolist(),cv_predictions['tfidf']))
        for tr,va in folds:
            tr = np.asarray([i for i in tr if desc[i].strip()],dtype=int)
            va = np.asarray([i for i in va if desc[i].strip()],dtype=int)
            check_split(y,groups,tr,va,labels)
            estimator = make_estimator('tfidf',cfg['C'],None if cfg['class_weight']=='none' else cfg['class_weight'],args.seed,args.max_iter)
            estimator.fit(desc[tr],y[tr])
            predictions.extend(estimator.predict(desc[va])); truth.extend(y[va]); gs.extend(groups[va])
            combined_pred.extend(combined_by_idx[int(i)] for i in va)
        report['description_ablation'] = {'scope':'development OOF, same description-bearing validation rows; fixed combined-selected TF-IDF hyperparameters',
            'description_only':metrics(truth,predictions,labels,gs),
            'combined_on_same_rows':metrics(truth,combined_pred,labels,gs),
            'note':'Task-only rows excluded from this diagnostic. Descriptive comparison, not independent model selection.'}
    write_csv(args.out/'test_predictions.csv',test_predictions,list(test_predictions[0]))
    write_csv(args.out/'test_errors.csv',[r for r in test_predictions if not r['correct']],list(test_predictions[0]))
    write_csv(args.out/'per_category.csv',per_category,['model','category','precision','recall','f1-score','support'])
    write_csv(args.out/'confusion_pairs.csv',sorted(confusion_pairs,key=lambda r:-r['count']),['model','actual','predicted','count'])
    write_json(args.out/'metrics.json',report)
    shutil.copyfile(args.out/(winner+'.joblib'),args.out/'winner.joblib')
    shutil.copyfile(args.prepared/'taxonomy.json',args.out/'taxonomy.json')
    write_json(args.out/'manifest.json',{'run_id':run_id,'config':config,'winner':winner,
        'artifacts':{p.name:sha(p) for p in sorted(args.out.iterdir()) if p.is_file()}})
    print('\nWinner selected by development CV:',winner)
    for kind in args.models:
        print(kind,'held-out macro-F1:',round(report['models'][kind]['test']['macro_f1'],4))
    print('Saved models and reports:',args.out.resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--models',nargs='+',choices=['tfidf','embeddings'],default=['tfidf','embeddings'])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cv-folds',type=int,default=3)
    p.add_argument('--outer-folds',type=int,default=5,help='First grouped fold is held out; default about 20%%')
    p.add_argument('--c-values',type=float,nargs='+',default=[0.5,2.0])
    p.add_argument('--max-iter',type=int,default=3000)
    p.add_argument('--encoder',default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--encoder-revision',default=None)
    p.add_argument('--embedding-cache',type=Path,default=Path('data/cache/category_embeddings'))
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cpu')
    p.add_argument('--description-ablation',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    args = p.parse_args()
    if args.cv_folds<2 or args.outer_folds<2 or args.max_iter<1 or args.batch_size<1 or any(c<=0 for c in args.c_values):
        p.error('Fold counts must be >=2; other numeric parameters must be positive.')
    if len(set(args.models)) != len(args.models):
        p.error('Duplicate model names')
    if args.description_ablation and 'tfidf' not in args.models:
        p.error('--description-ablation requires --models tfidf (optionally embeddings too)')
    try:
        run(args)
    except (OSError,ValueError,ImportError,importlib.metadata.PackageNotFoundError) as exc:
        p.exit(1,f'Error: {exc}\nSee README.md for dependencies and a fresh output directory.\n')


if __name__ == '__main__':
    main()
