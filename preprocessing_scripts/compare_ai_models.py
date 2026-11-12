"""Compare five description-only binary classifiers locally.

Install: python -m pip install numpy scikit-learn joblib sentence-transformers
Run from project root; outputs default to models/ai_detection.
No Gemini/JEV calls. First encoder use downloads pretrained weights.
Cached runs skip training. --force-retrain repeats training/evaluation but reuses
embeddings; --recompute-embeddings regenerates embeddings and forces retraining.
Only load your own trusted joblib files. Run one process per output directory.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

VERSION = 'binary-comparison-1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def json_write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def csv_write(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def normalized(text):
    return re.sub(r'\W+', ' ', text.casefold()).strip()


def load_data(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'tool_id', 'tool_name', 'domain', 'description', 'is_ai_tool'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Missing columns: ' + str(required - set(reader.fieldnames or [])))
        raw = list(reader)
    rows, excluded, seen = [], [], set()
    for r in raw:
        if None in r or any(v is None for v in r.values()):
            raise ValueError('Malformed CSV record')
        tid = r['tool_id'].strip()
        if not tid or tid in seen:
            raise ValueError('Missing or duplicate tool_id: ' + tid)
        seen.add(tid)
        label = r['is_ai_tool'].strip().lower()
        if label not in {'true', 'false', '1', '0'}:
            raise ValueError('Resolve missing/invalid AI label before training: ' + tid)
        text = ' '.join(r['description'].split())
        if not normalized(text):
            excluded.append({'tool_id': tid, 'reason': 'missing_description'}); continue
        rows.append({'tool_id': tid, 'tool_name': r['tool_name'], 'domain': r['domain'],
                     'description': text, 'label': int(label in {'true', '1'})})
    rows.sort(key=lambda r: r['tool_id'])
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    keys, description_labels, identity_labels = {}, {}, {}
    for i, r in enumerate(rows):
        name, desc = normalized(r['tool_name']), normalized(r['description'])
        try:
            host = (urlsplit(r['domain'] if '://' in r['domain'] else 'https://' + r['domain']).hostname or '').lower().removeprefix('www.')
        except ValueError:
            raise ValueError('Invalid domain: ' + r['tool_id'])
        if not name or not host:
            raise ValueError('Name/domain required for grouping: ' + r['tool_id'])
        for index, key in [(description_labels, desc), (identity_labels, (name, r['domain'].strip().lower()))]:
            if key in index and index[key] != r['label']:
                raise ValueError('Conflicting labels for identical description or identity: ' + r['tool_id'])
            index[key] = r['label']
        for kind, value in [('name', name), ('host', host), ('description', desc)]:
            key = (kind, value)
            if key in keys: parent[find(i)] = find(keys[key])
            keys[key] = i
    if {r['label'] for r in rows} != {0, 1}:
        raise ValueError('Need both AI and non-AI descriptions')
    return rows, excluded, np.array([find(i) for i in range(len(rows))])


def splits_for(y, groups, requested, seed):
    n = min(requested, *(len(set(groups[y == label])) for label in [0, 1]))
    if n < 2: raise ValueError('Not enough independent groups for evaluation')
    splits = list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(y, y, groups))
    for tr, va in splits:
        if set(y[tr]) != {0, 1} or set(y[va]) != {0, 1}:
            raise ValueError('A split lacks a class; add examples or change split settings before evaluating')
        assert not set(groups[tr]) & set(groups[va])
    return splits


def models(seed):
    def tfidf(classifier):
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                            max_features=20000, strip_accents='unicode'), classifier)
    return {
        'tfidf_logistic': tfidf(LogisticRegression(C=1, class_weight='balanced', max_iter=3000, random_state=seed)),
        'tfidf_svm': tfidf(LinearSVC(C=1, class_weight='balanced', random_state=seed, max_iter=10000)),
        'embeddings_logistic': LogisticRegression(C=1, class_weight='balanced', max_iter=3000, random_state=seed),
        'embeddings_knn': KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine'),
        'embeddings_forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                   class_weight='balanced', random_state=seed, n_jobs=1),
    }


def scores(y, pred):
    return {'accuracy': float(accuracy_score(y, pred)),
            'ai_precision': float(precision_score(y, pred, zero_division=0)),
            'ai_recall': float(recall_score(y, pred, zero_division=0)),
            'non_ai_recall': float(recall_score(y, pred, pos_label=0, zero_division=0)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'confusion_matrix_labels_0_1': confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def embeddings_for(texts, root, encoder_name, revision, device, recompute):
    encoder_dir = root / 'encoder'
    array_path = root / 'embeddings.npy'
    if array_path.exists() and encoder_dir.exists() and not recompute:
        x = np.load(array_path, allow_pickle=False)
        if x.ndim == 2 and len(x) == len(texts) and np.isfinite(x).all():
            print('Reusing cached embeddings.'); return x
        raise ValueError('Invalid embedding cache; use --recompute-embeddings')
    from sentence_transformers import SentenceTransformer
    if encoder_dir.exists():
        encoder = SentenceTransformer(str(encoder_dir.resolve()), device=device, local_files_only=True)
    else:
        encoder = SentenceTransformer(encoder_name, revision=revision, device=device, trust_remote_code=False)
        encoder.save(str(encoder_dir))
    x = encoder.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True,
                       batch_size=32, show_progress_bar=True)
    if x.ndim != 2 or len(x) != len(texts) or not np.isfinite(x).all():
        raise ValueError('Invalid encoder output')
    tmp = root / 'embeddings.tmp.npy'; np.save(tmp, x); tmp.replace(array_path)
    return x


def display(report, run):
    print('\nCross-validation on development data (test data excluded):')
    for r in report['comparison']:
        s = r['cv_scores']
        print(f"{r['model']:24s} macro-F1={s['macro_f1']:.3f} AI precision={s['ai_precision']:.3f} AI recall={s['ai_recall']:.3f}")
    print('\nSelected by CV:', report['winner'])
    print('Winner held-out test:', json.dumps(report['test_scores'], indent=2))
    print('Files:', run.resolve())
    print('Saved winner.joblib was fitted only on development rows. Test rows remain excluded.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, default=Path('models/ai_detection'))
    p.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--encoder-revision', help='Optional Hugging Face commit ID for reproducibility')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--force-retrain', action='store_true')
    p.add_argument('--recompute-embeddings', action='store_true')
    p.add_argument('--validate-only', action='store_true', help='Check data/splits; no downloads, training or output files')
    a = p.parse_args()
    if a.folds < 2: p.error('--folds must be at least 2')
    rows, excluded, groups = load_data(a.input)
    texts = np.array([r['description'] for r in rows]); y = np.array([r['label'] for r in rows])
    # Reserve the first of five grouped splits, chosen without model scores.
    dev, test = splits_for(y, groups, 5, a.seed)[0]
    splits = splits_for(y[dev], groups[dev], a.folds, a.seed + 1)
    if min(len(tr) for tr, _ in splits) < 5: raise ValueError('Too few training rows for 5-neighbor KNN')
    print(f'Eligible: {len(rows)}; AI: {sum(y)}; non-AI: {sum(y == 0)}; excluded: {len(excluded)}')
    print(f'Development: {len(dev)}; test: {len(test)}; grouped CV folds: {len(splits)}')
    if a.validate_only: return
    versions = {n: importlib.metadata.version(n) for n in ['numpy', 'scikit-learn', 'joblib', 'sentence-transformers', 'torch']}
    config = {'version': VERSION, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_sha256': hashlib.sha256(a.input.read_bytes()).hexdigest(), 'seed': a.seed,
              'folds': a.folds, 'encoder': a.encoder, 'encoder_revision': a.encoder_revision,
              'device': a.device, 'versions': versions, 'python': platform.python_version()}
    run = a.out_dir / 'runs' / digest(config)[:20]
    report_path = run / 'metrics.json'
    required = ['comparison.csv', 'cv_predictions.csv', 'test_predictions.csv', 'split_manifest.csv',
                'excluded_rows.csv', 'winner.joblib', 'manifest.json'] + [n + '.joblib' for n in models(a.seed)]
    if report_path.exists() and all((run / n).exists() for n in required) and not (a.force_retrain or a.recompute_embeddings):
        print('CACHE HIT: using existing trained models and metrics; no training/downloads.')
        report = json.loads(report_path.read_text()); display(report, run); return
    run.mkdir(parents=True, exist_ok=True)
    # Remove completion marker first. Interrupted reruns cannot look complete.
    report_path.unlink(missing_ok=True)
    emb_key = digest({'texts': list(texts), 'encoder': a.encoder, 'revision': a.encoder_revision,
                      'versions': versions, 'device': a.device})[:20]
    emb_root = a.out_dir / 'embeddings' / emb_key; emb_root.mkdir(parents=True, exist_ok=True)
    x = embeddings_for(texts, emb_root, a.encoder, a.encoder_revision, a.device, a.recompute_embeddings)
    fold_ids = {}
    for fold, (_, va) in enumerate(splits, 1):
        for i in va: fold_ids[int(dev[i])] = fold
    csv_write(run / 'split_manifest.csv', [dict(tool_id=r['tool_id'], group=int(groups[i]),
              partition='development' if i in fold_ids else 'test', fold=fold_ids.get(i, ''))
              for i, r in enumerate(rows)], ['tool_id', 'group', 'partition', 'fold'])
    csv_write(run / 'excluded_rows.csv', excluded, ['tool_id', 'reason'])
    comparison, predictions = [], []
    for name, estimator in models(a.seed).items():
        features = texts if name.startswith('tfidf') else x
        oof = np.empty(len(dev), dtype=int); fit_time = predict_time = 0.0
        for fold, (tr, va) in enumerate(splits, 1):
            model = clone(estimator); start = time.perf_counter()
            model.fit(features[dev[tr]], y[dev[tr]]); fit_time += time.perf_counter() - start
            start = time.perf_counter(); oof[va] = model.predict(features[dev[va]])
            predict_time += time.perf_counter() - start
        score = scores(y[dev], oof)
        print(f"{name}: CV macro-F1={score['macro_f1']:.3f}", flush=True)
        comparison.append({'model': name, 'cv_scores': score, 'cv_fit_seconds': fit_time,
                           'cv_predict_ms_per_description': predict_time * 1000 / len(dev)})
        predictions.extend(dict(model=name, tool_id=rows[i]['tool_id'], fold=fold_ids[int(i)],
                                actual=int(y[i]), predicted=int(oof[j])) for j, i in enumerate(dev))
        estimator.fit(features[dev], y[dev])
        joblib.dump(estimator, run / (name + '.joblib'))
    comparison.sort(key=lambda r: (-r['cv_scores']['macro_f1'], r['model']))
    winner = comparison[0]['model']
    estimator = joblib.load(run / (winner + '.joblib'))
    test_x = texts[test] if winner.startswith('tfidf') else x[test]
    pred = estimator.predict(test_x)
    dummy = DummyClassifier(strategy='most_frequent').fit(np.zeros((len(dev), 1)), y[dev])
    baseline = scores(y[test], dummy.predict(np.zeros((len(test), 1))))
    bundle = {'model_name': winner, 'estimator': estimator,
              'encoder_path': str((emb_root / 'encoder').resolve()) if winner.startswith('embeddings') else None,
              'labels': {0: 'non_ai', 1: 'ai'}, 'input_column': 'description', 'config': config,
              'note': 'Development-only fit; scores uncalibrated; no unknown threshold selected.'}
    joblib.dump(bundle, run / 'winner.joblib')
    csv_write(run / 'cv_predictions.csv', predictions, ['model', 'tool_id', 'fold', 'actual', 'predicted'])
    csv_write(run / 'test_predictions.csv', [dict(tool_id=rows[i]['tool_id'], actual=int(y[i]), predicted=int(pred[j]))
              for j, i in enumerate(test)], ['tool_id', 'actual', 'predicted'])
    flat = [dict(model=r['model'], **{k:v for k,v in r['cv_scores'].items() if k != 'confusion_matrix_labels_0_1'},
                 cv_fit_seconds=r['cv_fit_seconds'], cv_predict_ms_per_description=r['cv_predict_ms_per_description']) for r in comparison]
    csv_write(run / 'comparison.csv', flat, list(flat[0]))
    json_write(run / 'manifest.json', config)
    report = {'comparison': comparison, 'winner': winner, 'test_scores': scores(y[test], pred),
              'majority_test_baseline': baseline, 'eligible_rows': len(rows), 'excluded_rows': len(excluded),
              'development_rows': len(dev), 'test_rows': len(test), 'embedding_cache': str(emb_root.resolve()),
              'notes': ['Only descriptions are model inputs.', 'Winner selected on development CV macro-F1; only winner evaluated on test.',
                        'Fixed model settings; no hyperparameter search.', 'Repeated experiments after inspecting this test require fresh final evaluation.',
                        'Related names, hosts and exact normalized descriptions grouped; near-duplicates may remain.',
                        'Timing excludes shared encoder computation. Probabilities are not calibrated.',
                        'This does not yet connect predictions to enrich.py or choose an abstention threshold.']}
    json_write(report_path, report)  # Completion marker written last.
    display(report, run)


if __name__ == '__main__':
    try: main()
    except (ValueError, OSError, importlib.metadata.PackageNotFoundError) as exc:
        raise SystemExit(str(exc))
