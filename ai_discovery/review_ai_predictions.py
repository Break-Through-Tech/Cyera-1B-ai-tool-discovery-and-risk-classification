"""Inspect held-out mistakes and classify descriptions in the enrichment cache.

Uses compare_ai_models.py outputs. No training, web requests or file writes.
Place in ai_discovery/. Run from project root with --run-dir, --input and --cache.
Only load a winner.joblib you created/trust. Predictions remain provisional;
this script does not verify product identity or update catalog/cache labels.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import joblib
import numpy as np


def read_csv(path, required):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(f'Missing required columns in {path}')
        rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f'Malformed CSV: {path}')
    return rows


def label(value):
    value = str(value).strip().lower()
    if value in {'true', '1'}: return 1
    if value in {'false', '0'}: return 0
    raise ValueError('Invalid binary label: ' + value)


def name_key(text):
    return re.sub(r'\W+', ' ', text.casefold()).strip()


def host_key(endpoint):
    p = urlsplit(endpoint if '://' in endpoint else 'https://' + endpoint)
    return (p.hostname or '').lower().removeprefix('www.')


def section_title(line):
    # Accept the numbered Markdown headings used by enrich.py, also bold/plain headings.
    clean = re.sub(r'^[ \t]*#{1,6}\s*', '', line.strip())
    clean = clean.replace('**', '').replace('__', '')
    clean = re.sub(r'^\d+[.)]\s*', '', clean).strip().rstrip(':').strip()
    return clean.casefold()


SECTION_TITLES = {'product identity', 'product description', 'description',
                  'ai functionality', 'alternate names and domains',
                  'alternate names/domains', 'uncertainty'}


def extract_description(blocks):
    """Extract only an explicitly labeled description. Never fall back to full report.

    Return citation offsets relative to their original report block. Citations
    overlapping the selected section are retained as evidence, not as verification.
    """
    found = []
    for block_index, block in enumerate(blocks):
        text = block.get('text', '')
        if not isinstance(text, str): raise ValueError('Report block text must be a string')
        headers = []
        for match in re.finditer(r'^.*$', text, flags=re.MULTILINE):
            line = match.group()
            title = section_title(line)
            if title in SECTION_TITLES or re.match(r'^\s*#{1,6}\s+', line):
                headers.append((match.start(), match.end(), title))
        for j, (_, end, title) in enumerate(headers):
            if title not in {'product description', 'description'}: continue
            stop = headers[j + 1][0] if j + 1 < len(headers) else len(text)
            raw = text[end:stop]
            # Clean display/citation markers without generating or rewriting facts.
            clean = re.sub(r'\[([^\]]+)\]\(https?://[^)]+\)', r'\1', raw)
            clean = re.sub(r'\[\d+(?:\s*[,–-]\s*\d+)*\]', '', clean)
            clean = ' '.join(clean.replace('**', '').replace('`', '').split())
            citations = []
            for c in block.get('citations', []) or []:
                start, finish = c.get('start_index'), c.get('end_index')
                if isinstance(start, int) and isinstance(finish, int) and start < stop and finish > end:
                    citations.append(c)
            found.append({'description': clean, 'report_block': block_index,
                          'original_section_start': end, 'original_section_end': stop,
                          'description_citations': citations})
    if len(found) != 1:
        return None, 'description_section_missing' if not found else 'multiple_description_sections'
    result = found[0]
    if len(re.findall(r'\b\w+\b', result['description'])) < 5:
        return None, 'description_too_short'
    return result, None


class Predictor:
    def __init__(self, bundle, encoder_override=None):
        self.estimator = bundle['estimator']
        self.kind = bundle['model_name']
        self.encoder_path = encoder_override or bundle.get('encoder_path')
        self.encoder = None

    def predict(self, description):
        coverage = None
        if self.kind.startswith('tfidf'):
            vectorizer = self.estimator.steps[0][1]
            vector = vectorizer.transform([description])
            terms = vectorizer.build_analyzer()(description)
            coverage = sum(t in vectorizer.vocabulary_ for t in terms) / len(terms) if terms else 0.0
            if vector.nnz == 0:
                return {'predicted_is_ai_tool': None, 'status': 'unknown',
                        'reason': 'no_known_vocabulary', 'vocabulary_coverage': coverage}
            features = [description]
        elif self.kind.startswith('embeddings'):
            if not self.encoder_path or not Path(self.encoder_path).is_dir():
                raise ValueError('Saved encoder missing; supply --encoder-dir if you moved it')
            if self.encoder is None:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(str(Path(self.encoder_path).resolve()),
                                                   local_files_only=True, trust_remote_code=False)
            features = self.encoder.encode([description], normalize_embeddings=True,
                                           convert_to_numpy=True, show_progress_bar=False)
        else:
            raise ValueError('Unsupported saved model: ' + self.kind)
        prediction = int(self.estimator.predict(features)[0])
        if prediction not in (0, 1): raise ValueError('Model did not return binary labels')
        result = {'predicted_is_ai_tool': bool(prediction), 'status': 'provisional_prediction',
                  'vocabulary_coverage': coverage}
        classes = list(self.estimator.classes_)
        if hasattr(self.estimator, 'predict_proba'):
            probabilities = self.estimator.predict_proba(features)[0]
            result['ai_probability_uncalibrated'] = float(probabilities[classes.index(1)])
        elif hasattr(self.estimator, 'decision_function'):
            margin = float(np.asarray(self.estimator.decision_function(features)).ravel()[0])
            result['ai_decision_margin'] = margin if classes[1] == 1 else -margin
        return result


def review_test(rows, run):
    by_id = {r['tool_id']: r for r in rows}
    if len(by_id) != len(rows): raise ValueError('Duplicate input tool IDs')
    predictions = read_csv(run / 'test_predictions.csv', ['tool_id', 'actual', 'predicted'])
    splits = read_csv(run / 'split_manifest.csv', ['tool_id', 'partition'])
    partitions = {r['tool_id']: r['partition'] for r in splits}
    if len(partitions) != len(splits): raise ValueError('Duplicate split IDs')
    ids = [r['tool_id'] for r in predictions]
    if len(ids) != len(set(ids)) or set(ids) != {k for k,v in partitions.items() if v == 'test'}:
        raise ValueError('Test predictions and split manifest disagree')
    mistakes = []
    for r in predictions:
        source = by_id.get(r['tool_id'])
        if source is None or label(source['is_ai_tool']) != label(r['actual']):
            raise ValueError('Test labels differ from source dataset')
        actual, pred = label(r['actual']), label(r['predicted'])
        if actual != pred:
            mistakes.append({'tool_id': r['tool_id'], 'tool_name': source['tool_name'],
                             'domain': source['domain'], 'description': source['description'],
                             'actual_is_ai_tool': bool(actual), 'predicted_is_ai_tool': bool(pred),
                             'error_type': 'false_positive' if pred else 'false_negative'})
    return mistakes, partitions, len(predictions)


def overlap(name, endpoint, description, rows, partitions):
    matches = []
    for r in rows:
        signals = []
        if name_key(name) == name_key(r['tool_name']): signals.append('name')
        if host_key(endpoint) and host_key(endpoint) == host_key(r['domain']): signals.append('host')
        if description and name_key(description) == name_key(r['description']): signals.append('description')
        if signals:
            matches.append({'tool_id': r['tool_id'], 'partition': partitions.get(r['tool_id'], 'excluded'),
                            'matching_signals': signals})
    return matches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True, help='Exact CSV used for training')
    p.add_argument('--cache', type=Path, default=Path('data/cache/enrichment_cache.json'))
    p.add_argument('--encoder-dir', type=Path, help='Only needed if an embedding winner encoder was moved')
    a = p.parse_args()
    manifest = json.loads((a.run_dir / 'manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(a.input.read_bytes()).hexdigest() != manifest['input_sha256']:
        p.error('Input CSV differs from training input. Use the original unchanged CSV.')
    rows = read_csv(a.input, ['tool_id', 'tool_name', 'domain', 'description', 'is_ai_tool'])
    mistakes, partitions, count = review_test(rows, a.run_dir)
    print(f'HELD-OUT TEST: {len(mistakes)} mistakes out of {count}')
    for m in mistakes: print(json.dumps(m, indent=2, ensure_ascii=False))
    if not a.cache.exists():
        p.exit(1, f'Enrichment cache not found: {a.cache}. Test review above is complete.\n')
    cache = json.loads(a.cache.read_text(encoding='utf-8'))
    if cache.get('version') != 1 or not isinstance(cache.get('entries'), dict):
        p.error('Unsupported enrichment cache format; expected enrich.py version 1 JSON')
    bundle = joblib.load(a.run_dir / 'winner.joblib')
    if bundle.get('config') != manifest or bundle.get('input_column') != 'description':
        p.error('Saved winner and manifest do not match')
    predictor = Predictor(bundle, a.encoder_dir)
    counts = Counter()
    print('\nCACHED PRODUCT DESCRIPTION PREDICTIONS')
    for key, entry in cache['entries'].items():
        identity = json.loads(key)
        if not isinstance(identity, list) or len(identity) != 2 or not all(isinstance(v,str) for v in identity):
            raise ValueError('Invalid identity cache key')
        name, endpoint = identity
        report = entry.get('report', {})
        result = {'application_name': name, 'endpoint': endpoint, 'model': bundle['model_name'],
                  'retrieved_at': entry.get('retrieved_at'), 'needs_review': True,
                  'identity_verified': False, 'confirmed_is_ai_tool': None}
        if report.get('enrichment_status') != 'cited_report_for_review':
            extracted, reason = None, 'enrichment_not_successful'
        else:
            extracted, reason = extract_description(report.get('report_blocks', []))
        if extracted is None:
            result.update(status='unknown', reason=reason, predicted_is_ai_tool=None)
        else:
            result.update(extracted)
            result.update(predictor.predict(extracted['description']))
            result['dataset_overlap'] = overlap(name, endpoint, extracted['description'], rows, partitions)
            result['evidence_note'] = ('Citations overlap the extracted section; identity still needs review.'
                                      if extracted['description_citations'] else
                                      'No citation offsets overlap the description; inspect original report.')
        counts[result['status']] += 1
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(json.dumps({'cache_entries': len(cache['entries']), 'prediction_counts': dict(counts)}, indent=2))
    print('No files written, API calls, retraining, or catalog updates. Probabilities are uncalibrated.')
    print('Dataset-overlap matches are not independent unseen-tool tests. Inspect identity and description before accepting any label.')


if __name__ == '__main__':
    try: main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f'Error: {exc}')
