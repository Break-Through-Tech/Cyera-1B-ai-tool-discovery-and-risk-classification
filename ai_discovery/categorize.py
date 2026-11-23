"""Component 3: conservative catalog reuse and saved-model purpose categorization.

Run from the repository root: python -m ai_discovery.categorize --help
No training. Inputs and AI decisions are never edited. No network calls unless
--category-api jev is given; then Jev categorizes confirmed AI apps that have no
catalog category, and the local model remains the offline fallback.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import urllib.error
from urllib.parse import urlsplit

from . import review_discovery_categories as helpers

VERSION = 'purpose-categorization-v1'
ROOT = Path(__file__).resolve().parents[1]


def identity(name, endpoint, prep):
    # Query/fragment can distinguish tenants or products: do not merge silently.
    parsed = urlsplit(endpoint if '://' in endpoint else 'https://' + endpoint)
    if parsed.query or parsed.fragment:
        raise ValueError('Query/fragment identity requires review')
    return helpers.identity(name, endpoint, prep)


class Catalog:
    def __init__(self, paths, prep, context=None):
        self.prep = prep
        self.rows = defaultdict(list)
        self.warnings = []
        self.inputs = {str(p): helpers.sha(p) for p in paths}
        urls = {}
        if context:
            self.inputs[str(context)] = helpers.sha(context)
            for r in helpers.csv_rows(context):
                if r['tool_id'] in urls:
                    raise ValueError('Duplicate context tool_id')
                urls[r['tool_id']] = r['website_url_clean']
        for source_index, path in enumerate(paths):
            for i, r in enumerate(helpers.csv_rows(path), 1):
                raw = {'Name', 'Website', 'Category'} <= r.keys()
                if not raw and not {'tool_id', 'tool_name', 'domain', 'category'} <= r.keys():
                    raise ValueError('Catalog needs raw Kaggle or Dataset B columns')
                tid = f'tool_{i:05d}' if raw else r['tool_id']
                name = r['Name'] if raw else r['tool_name']
                url = r['Website'] if raw else r['domain']
                if source_index == 0 and not raw:
                    url = urls.get(tid, url)
                value = 'true' if raw else r.get('is_ai_tool', 'true').strip().lower()
                label = True if value in {'true', '1'} else False if value in {'false', '0'} else None
                try:
                    key = identity(name, url, prep)
                except ValueError as exc:
                    self.warnings.append({'file': str(path), 'row': i, 'reason': str(exc)})
                    continue
                self.rows[key].append({'tool_id': tid, 'file': str(path), 'row': i,
                    'source_sha256': self.inputs[str(path)], 'source_index': source_index,
                    'is_ai_tool': label, 'category': (r['Category'] if raw else r['category']).strip()})
        # A supplied finalized reference overrides the AI-catalog assumption only
        # for this exact name AND endpoint, never a hostname or fuzzy name alone.
        for key, records in self.rows.items():
            finalized = [r for r in records if r['source_index'] == 1]
            self.rows[key] = finalized or records

    def lookup(self, name, endpoint):
        try:
            return self.rows.get(identity(name, endpoint, self.prep), [])
        except ValueError:
            return []


def gate(record):
    if record.get('catalog_detection_conflict'):
        return 'pending_detection_conflict'
    if record.get('classification_status') != 'classified' or record.get('technical_error'):
        return 'pending_ai_detection'
    if record.get('is_ai_tool') is False:
        return 'not_applicable'
    if record.get('is_ai_tool') is not True:
        return 'pending_ai_detection'
    if record.get('identity_supported') is not True:
        return 'pending_identity_review'
    return None


class Engine:
    def __init__(self, args):
        self.args = args
        self.prep = helpers.import_file(args.prep_script, 'purpose_text_contract')
        self.training = helpers.import_file(args.training_script, 'purpose_encoding_contract')
        self.taxonomy = self.prep.TAXONOMY
        self.catalog = Catalog([p for p in [args.catalog, args.known_tools] if p], self.prep, args.catalog_context)
        self.cache = helpers.read_json(args.cache) if args.cache.exists() else {'version': 1, 'entries': {}}
        if self.cache.get('version') != 1 or not isinstance(self.cache.get('entries'), dict):
            raise ValueError('Unsupported category cache')
        self.predictor = None
        self.model_meta = None
        self.model_path = args.model_dir / (args.model + '.joblib')
        self.changed = False
        self.jev = None
        if getattr(args, 'category_api', 'none') == 'jev':
            self.jev = JevCategorizer(args, self.taxonomy)

    def api_settings(self):
        return self.jev.settings() if self.jev else {'provider': None}

    def model_metadata(self):
        if self.model_meta is None:
            m = helpers.read_json(self.args.model_dir / 'manifest.json')
            taxonomy_path = self.args.model_dir / 'taxonomy.json'
            labels = helpers.read_json(taxonomy_path)
            for p in [self.model_path, taxonomy_path]:
                if m['artifacts'].get(p.name) != helpers.sha(p):
                    raise ValueError('Model/taxonomy artifact hash mismatch: ' + p.name)
            if labels.get('categories') != self.taxonomy or labels.get('version') != 'original_kaggle_14':
                raise ValueError('Expected exact Kaggle-14 taxonomy')
            if m['config']['text_contract_version'] != self.prep.TEXT_VERSION:
                raise ValueError('Model and extraction contracts differ')
            self.model_meta = m
        return self.model_meta

    def predict(self, text):
        m = self.model_metadata()
        if self.predictor is None:
            expected = m['config']['packages']['scikit-learn']
            if importlib.metadata.version('scikit-learn') != expected:
                raise ValueError(f'Use scikit-learn {expected}: python -m pip install "scikit-learn=={expected}"')
            import joblib
            bundle = joblib.load(self.model_path)
            if (bundle['run_id'] != m['run_id'] or bundle['taxonomy'] != self.taxonomy
                    or bundle['text_contract_version'] != self.prep.TEXT_VERSION):
                raise ValueError('Model bundle metadata mismatch')
            self.predictor = helpers.Predictor(bundle, self.training, self.args.encoder_dir)
        return self.predictor.predict(text)

    def categorize(self, record, review=None, selection=None):
        evidence_hash = helpers.fingerprint(record.get('reports', []))
        result = {'category': None, 'category_status': gate(record), 'category_source': None,
                  'top_three': [], 'cache_hit': False, 'evidence_sha256': evidence_hash,
                  'taxonomy': 'original_kaggle_14', 'catalog_records': [], 'model_text': ''}
        if result['category_status']:
            if review:
                raise ValueError('Category review cannot override AI/identity gate')
            return result
        try:
            key = identity(record['application_name'], record['endpoint'], self.prep)
        except ValueError:
            result['category_status'] = 'pending_identity_review'
            return result
        records = self.catalog.lookup(record['application_name'], record['endpoint'])
        result['catalog_records'] = records
        if records and {r['is_ai_tool'] for r in records} != {True}:
            result['category_status'] = 'pending_detection_conflict'
            return result
        if review:
            helpers.validate_review(review, record, evidence_hash)
            result.update(category=review['reviewed_category'].strip(), category_status='assigned',
                          category_source='human_review', reviewer=review['reviewer'], reason=review['reason'])
            return result
        categories = {r['category'] for r in records if r['category']}
        if len(categories) > 1 or categories - set(self.taxonomy):
            result['category_status'] = 'pending_catalog_category_review'
            return result
        if categories:
            result.update(category=next(iter(categories)), category_status='assigned', category_source='catalog',
                          identity_resolution='exact_normalized_name_and_endpoint')
            return result
        if selection and selection[1] != evidence_hash:
            raise ValueError('Report selection is stale; evidence changed')
        evidence = helpers.candidate(record, self.prep, selection[0] if selection else None)
        result.update(model_text=evidence['model_text'], report_index=evidence['report_index'],
                      sections=evidence['sections'], category_status=evidence['status'])
        if evidence['status'] != 'ready':
            return result
        if self.jev is not None:
            text = evidence.get('description_only_text') or ''
            api, reason = self.jev.categorize(text) if text.strip() else (None, 'no Product Description section for API input')
            if api is not None:
                result.update(api)
                return result
            result['api_fallback_reason'] = reason
        m = self.model_metadata()
        cache_key = helpers.fingerprint({'version': VERSION, 'identity': key,
            'evidence': evidence_hash, 'text': evidence['model_text'], 'model': helpers.sha(self.model_path),
            'taxonomy': self.taxonomy, 'catalog': self.catalog.inputs,
            'extractor': helpers.sha(self.args.prep_script), 'encoding': helpers.sha(self.args.training_script),
            'runtime': helpers.sha(Path(__file__)), 'predictor': helpers.sha(Path(helpers.__file__)),
            'detection': {k: record.get(k) for k in ['is_ai_tool', 'classification_status', 'identity_supported', 'identity_mode']}})
        prediction = self.cache['entries'].get(cache_key)
        if prediction is None:
            prediction = self.predict(evidence['model_text'])
            self.cache['entries'][cache_key] = prediction
            self.changed = True
        else:
            result['cache_hit'] = True
        if prediction['category'] is not None and prediction['category'] not in self.taxonomy:
            raise ValueError('Prediction outside taxonomy')
        result.update(category=prediction['category'], category_source='model' if prediction['category'] else None,
                      category_status='provisional_prediction' if prediction['category'] else prediction['reason'],
                      top_three=prediction['top_three'], model_run_id=m['run_id'],
                      model_sha256=helpers.sha(self.model_path), prediction_key=cache_key)
        return result

    def save_cache(self):
        if self.changed:
            helpers.atomic_cache(self.args.cache, self.cache)
        if self.jev is not None:
            self.jev.save_cache()


class JevCategorizer:
    """Jev (Typesafe System One) category choice, exactly as evaluated.

    Same question, category definitions (artifacts/jev_category_eval/criteria.json),
    description-only text and request key as category_training/jev_category_eval.py,
    so that evaluation's result (macro-F1 0.59 vs 0.48 for the local model) describes
    this step, and its cached answers are reused without new paid calls.
    """

    ERRORS = (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError)

    def __init__(self, args, taxonomy):
        self.args = args
        self.mod = helpers.import_file(args.jev_script, 'jev_category_contract')
        named = helpers.read_json(args.jev_criteria)
        if set(named) != set(taxonomy):
            raise ValueError('Jev criteria must cover exactly the Kaggle-14 taxonomy')
        self.criteria = {self.mod.slug(c): text for c, text in named.items()}
        self.by_slug = {self.mod.slug(c): c for c in taxonomy}
        if not 0 <= args.api_review_threshold <= 1:
            raise ValueError('--api-review-threshold must be between 0 and 1')
        self.threshold = args.api_review_threshold
        self.cache_path = args.jev_cache
        self.cache = helpers.read_json(self.cache_path) if self.cache_path.exists() else {
            'version': self.mod.VERSION, 'entries': {}}
        self.changed = False
        self.key = None
        self.calls = 0

    def settings(self):
        return {'provider': 'jev', 'model': self.args.jev_model, 'review_threshold': self.threshold,
                'criteria_sha256': helpers.sha(self.args.jev_criteria),
                'question_version': self.mod.VERSION, 'evaluation': 'docs/jev_category_evaluation.md'}

    def request_key(self, text):
        spec = {'v': self.mod.VERSION, 'model': self.args.jev_model, 'instructions': self.mod.INSTRUCTIONS,
                'criteria': self.criteria, 'evidence': text}
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    def api_key(self):
        if self.key is None:
            try:
                from dotenv import load_dotenv
                load_dotenv(ROOT/'.env', override=False)
            except ImportError:
                pass
            self.key = os.environ.get('TYPESAFE_API_KEY') or ''
        return self.key

    def categorize(self, text):
        """Return (categorization fields, None) or (None, fallback reason)."""
        key = self.request_key(text)
        entry = self.cache['entries'].get(key)
        hit = entry is not None
        if not hit:
            if not self.api_key():
                return None, 'TYPESAFE_API_KEY not set; used local model'
            try:
                answer = self.mod.call_jev(self.args.jev_model, text, self.criteria,
                                           self.args.api_timeout, self.api_key())
            except self.ERRORS as exc:
                return None, f'Jev request failed ({type(exc).__name__}); used local model'
            self.calls += 1
            entry = {'source': 'ai_discovery.categorize', 'result': answer,
                     'retrieved_at': datetime.now(timezone.utc).isoformat()}
            self.cache['entries'][key] = entry
            self.changed = True
        answer = entry['result']
        probs = answer['probabilities']
        if set(probs) != set(self.criteria) or answer['choice'] not in self.criteria:
            return None, 'Jev answer outside the taxonomy; used local model'
        p = float(probs[answer['choice']])
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        accepted = p >= self.threshold
        fields = {'category': self.by_slug[answer['choice']] if accepted else None,
                  'category_status': 'provisional_prediction' if accepted else 'needs_category_review',
                  'category_source': 'api', 'api_provider': 'jev',
                  'api_model': answer.get('returned_model') or self.args.jev_model,
                  'api_probability': round(p, 4), 'api_review_threshold': self.threshold,
                  'top_three': [{'category': self.by_slug[k], 'probability': round(float(v), 4)} for k, v in top],
                  'cache_hit': hit, 'api_request_key': key}
        if not accepted:
            fields['suggested_category'] = self.by_slug[answer['choice']]
        return fields, None

    def save_cache(self):
        if self.changed:
            helpers.atomic_cache(self.cache_path, self.cache)


def options(p):
    p.add_argument('--catalog', type=Path, default=ROOT/'data/processed/kaggle_dataset_b/kaggle_dataset_b_filled.csv')
    p.add_argument('--known-tools', type=Path, default=ROOT/'data/processed/final/dataset_b_final_kaggle_categories.csv')
    p.add_argument('--catalog-context', type=Path)
    p.add_argument('--model-dir', type=Path, default=ROOT/'models/categories/kaggle14_run1')
    p.add_argument('--model', choices=['winner', 'tfidf', 'embeddings'], default='winner')
    p.add_argument('--cache', type=Path, default=ROOT/'data/cache/purpose_categories.json')
    p.add_argument('--prep-script', type=Path, default=ROOT/'preprocessing_scripts/prepare_category_data.py')
    p.add_argument('--training-script', type=Path, default=ROOT/'category_training/train_categories.py')
    p.add_argument('--encoder-dir', type=Path)
    p.add_argument('--reviews', type=Path, help='Evidence-bound human category reviews')
    p.add_argument('--report-selections', type=Path)
    p.add_argument('--category-api', choices=['none', 'jev'], default='none',
                   help='Categorize confirmed AI apps without a catalog category with Jev; '
                        'the local model is the fallback. Default: none (fully offline).')
    p.add_argument('--api-review-threshold', type=float, default=0.5,
                   help='Jev probability below this leaves the category empty and marks it for human review')
    p.add_argument('--jev-model', default='jev-latest')
    p.add_argument('--jev-criteria', type=Path, default=ROOT/'artifacts/jev_category_eval/criteria.json')
    p.add_argument('--jev-cache', type=Path, default=ROOT/'data/cache/jev_category_cache.json')
    p.add_argument('--jev-script', type=Path, default=ROOT/'category_training/jev_category_eval.py')
    p.add_argument('--api-timeout', type=float, default=60)
    p.add_argument('--out', type=Path, required=True, help='New output directory')


def preflight(args, extra_inputs=()):
    if args.out.exists():
        raise ValueError('Output directory exists; choose a new --out')
    inputs = [args.catalog, args.known_tools, args.catalog_context, args.prep_script,
              args.training_script, args.reviews, args.report_selections, *extra_inputs]
    caches = [args.cache]
    if getattr(args, 'category_api', 'none') == 'jev':
        inputs += [args.jev_criteria, args.jev_script]
        caches.append(args.jev_cache)
    sources = {p.resolve() for p in inputs if p}
    for cache in (c.resolve() for c in caches):
        if (cache in sources or cache.is_relative_to(args.model_dir.resolve())
                or cache.is_relative_to(args.out.resolve())):
            raise ValueError('Cache must not overwrite a source or model file')


def categorize_records(records, engine, args):
    prep = engine.prep
    keys = []
    for r in records:
        try:
            key = identity(r['application_name'], r['endpoint'], prep)
        except ValueError:
            key = json.dumps(['unresolved_endpoint', prep.key(r['application_name']), r['endpoint']])
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate normalized discovery identities')
    reviews = helpers.load_reviews(args.reviews, prep, engine.taxonomy)
    selections = {}
    for r in helpers.csv_rows(args.report_selections) if args.report_selections else []:
        if not r['report_index'].strip():
            continue
        key = helpers.identity(r['application_name'], r['endpoint'], prep)
        if key in selections:
            raise ValueError('Duplicate report selection')
        selections[key] = (int(r['report_index']), r['evidence_sha256'])
    if (reviews.keys() | selections.keys()) - set(keys):
        raise ValueError('Review/selection identity absent from input')
    return [dict(r, categorization=engine.categorize(r, reviews.get(k), selections.get(k)))
            for r, k in zip(records, keys)]


def save_outputs(args, records, engine, extra=None):
    args.out.mkdir(parents=True, exist_ok=False)
    flat = [dict(application_name=r['application_name'], endpoint=r['endpoint'],
                 is_ai_tool=r.get('is_ai_tool'), classification_status=r.get('classification_status'),
                 **r['categorization']) for r in records]
    api = engine.api_settings()
    sources_note = ('Category sources: human_review, catalog, api (Jev; below the review threshold the '
                    'category is left empty for human review), then model when Jev is unavailable.'
                    if api['provider'] else 'No API categorizer used; category sources are catalog, model or human_review.')
    report = {'version': VERSION, 'summary': dict(Counter(r['category_status'] for r in flat)),
              'results': records, 'catalog_warnings': engine.catalog.warnings, 'category_api': api,
              'limitations': ['Model and Jev scores are not calibrated; predictions remain provisional.',
                  'Exact name/endpoint matching is an identity policy, not independent usage verification.',
                  sources_note], **(extra or {})}
    helpers.write_json(args.out/'categorized_applications.json', report)
    helpers.write_csv(args.out/'categories.csv', flat, ['application_name', 'endpoint', 'is_ai_tool',
        'classification_status', 'category', 'category_status', 'category_source', 'cache_hit', 'evidence_sha256'])
    helpers.write_csv(args.out/'category_review_template.csv', flat,
        ['application_name', 'endpoint', 'evidence_sha256', 'reviewed_category', 'reviewer', 'reason', 'model_text'])
    helpers.write_csv(args.out/'report_selection_template.csv',
        [dict(r, report_index='') for r in flat if r['category_status']=='needs_report_selection'],
        ['application_name', 'endpoint', 'evidence_sha256', 'report_index'])
    helpers.write_json(args.out/'taxonomy.json', {'categories': engine.taxonomy, 'version': 'original_kaggle_14'})
    paths = [args.catalog, args.known_tools, args.catalog_context, args.prep_script, args.training_script,
             args.reviews, args.report_selections, getattr(args, 'discovery', None), getattr(args, 'activity', None)]
    if api['provider']:
        paths += [args.jev_criteria, args.jev_script]
    helpers.write_json(args.out/'manifest.json', {'version': VERSION,
        'inputs': {str(p): helpers.sha(p) for p in paths if p and p.is_file()},
        'script_sha256': helpers.sha(Path(__file__)), 'model': engine.model_meta, 'category_api': api})
    engine.save_cache()
    print(json.dumps(report['summary'], indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    options(p)
    p.add_argument('--discovery', type=Path, default=ROOT/'data/processed/ai_discovery_classification.json')
    a = p.parse_args()
    try:
        preflight(a, [a.discovery])
        engine = Engine(a)
        data = helpers.read_json(a.discovery)
        if not isinstance(data.get('results'), list):
            raise ValueError('Discovery input must be a classify.py report with results')
        records = categorize_records(data['results'], engine, a)
        save_outputs(a, records, engine)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        p.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
