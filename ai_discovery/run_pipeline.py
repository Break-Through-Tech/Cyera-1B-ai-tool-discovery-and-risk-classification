"""Run Components 1-3: activity extraction, AI discovery, purpose categorization.

Offline by default. --online enables existing enrich/classify services for
unresolved products. Governance stays in the separate legacy report workflow.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from . import categorize as cat
from .ingest import extract_applications


def index_discovery(data, prep):
    indexed = {}
    if not isinstance(data.get('results'), list):
        raise ValueError('Discovery report requires a results array')
    for r in data['results']:
        try:
            key = cat.identity(r['application_name'], r['endpoint'], prep)
        except ValueError:
            continue
        if key in indexed:
            raise ValueError('Duplicate discovery identity')
        indexed[key] = r
    return indexed


def merge_records(applications, prior, engine):
    records = []
    for app in applications:
        name, endpoint = app['application_name'], app['observed_endpoint']
        try:
            key = cat.identity(name, endpoint, engine.prep)
        except ValueError:
            key = None
        candidates = engine.catalog.lookup(name, endpoint) if key else []
        labels = {r['is_ai_tool'] for r in candidates}
        old = prior.get(key) if key else None
        if old is not None:
            record = dict(old)
        elif candidates and len(labels) == 1 and None not in labels:
            ai = next(iter(labels))
            record = {'application_name': name, 'endpoint': endpoint, 'is_ai_tool': ai,
                'classification_status': 'classified', 'identity_supported': True,
                'decision_source': 'catalog', 'identity_mode': 'exact_normalized_name_and_endpoint',
                'reports': [], 'label': 'ai' if ai else 'non_ai'}
        else:
            record = {'application_name': name, 'endpoint': endpoint, 'is_ai_tool': None,
                'classification_status': 'unresolved', 'identity_supported': False,
                'decision_source': None, 'reports': [], 'label': 'unknown',
                'reason': 'No resolved exact catalog identity or applicable saved discovery decision.'}
        # Retain activity identity and statistics separately; never join by host only.
        record['observed_identity'] = app
        record['application_id'] = app['application_id']
        record['catalog_detection_conflict'] = bool(old and candidates and
            old.get('classification_status') == 'classified' and
            (len(labels) != 1 or old.get('is_ai_tool') not in labels))
        records.append(record)
    return records


def online_discovery(args, apps, engine, prior, stage):
    from . import enrich
    targets = []
    purpose_only = {}
    for app in apps:
        name, endpoint = app['application_name'], app['observed_endpoint']
        try:
            key = cat.identity(name, endpoint, engine.prep)
        except ValueError:
            continue
        candidates = engine.catalog.lookup(name, endpoint)
        old = prior.get(key)
        labels = {r['is_ai_tool'] for r in candidates}
        categories = {r['category'] for r in candidates if r['category']}
        known_ai = bool(candidates and labels == {True})
        confirmed_ai = old and cat.gate(old) is None
        needs_purpose = (known_ai or confirmed_ai) and not categories and not (old or {}).get('reports')
        if candidates and labels in ({True}, {False}) and not needs_purpose:
            continue
        if old and old.get('classification_status') == 'classified' and not old.get('technical_error') and not needs_purpose:
            continue
        record = dict(app, status='unknown', reason='unresolved_exact_catalog_identity')
        # Existing enrich.py strips ports. Never send a different identity.
        if cat.identity(name, enrich.public_identity(record)['endpoint'], engine.prep) != key:
            continue
        if needs_purpose:
            purpose_only[key] = dict(old) if old else {
                'application_name': name, 'endpoint': endpoint, 'is_ai_tool': True,
                'classification_status': 'classified', 'identity_supported': True,
                'decision_source': 'catalog', 'identity_mode': 'exact_normalized_name_and_endpoint',
                'label': 'ai', 'reports': []}
        targets.append(record)
    if not targets:
        return prior, []
    cache = enrich.JsonCache(args.enrichment_cache)
    client = None
    def api():
        nonlocal client
        if client is None:
            from dotenv import load_dotenv
            load_dotenv(cat.ROOT/'.env', override=False)
            from google import genai
            key = os.environ.get('GEMINI_API_KEY')
            if not key:
                raise ValueError('Set GEMINI_API_KEY for --online')
            client = genai.Client(api_key=key)
        return client
    outcomes = []
    try:
        outcomes = list(enrich.enrich_applications(targets, api, args.api_model, cache, False))
    finally:
        if client is not None:
            client.close()
    # Include only successful CURRENT retrievals. A failed refresh cannot silently
    # fall back to a historical successful report during this run.
    successful = {r['application_id'] for r in outcomes if r.get('enrichment_status') == 'cited_report_for_review'}
    entries = {}
    prior = dict(prior)
    for r in targets:
        ekey = enrich.cache_key(r)
        if r['application_id'] not in successful or ekey not in cache.entries:
            continue
        key = cat.identity(r['application_name'], r['observed_endpoint'], engine.prep)
        if key in purpose_only:
            prior[key] = dict(purpose_only[key], reports=[cache.entries[ekey]['report']])
        else:
            entries[ekey] = cache.entries[ekey]
    if not entries:
        return prior, outcomes
    source = stage/'scoped_enrichment.json'
    decisions = stage/'discovery_decision_cache.json'
    output = stage/'discovery_classification.json'
    cat.helpers.write_json(source, {'version': 1, 'entries': entries})
    # Work on a new snapshot: neither the provided decision cache nor the input
    # discovery report is overwritten. Supply this snapshot on a later run.
    if args.decision_cache and args.decision_cache.exists():
        shutil.copyfile(args.decision_cache, decisions)
    command = [sys.executable, '-m', 'ai_discovery.classify', '--cache', str(source),
        '--decision-cache', str(decisions), '--output', str(output), '--model', args.api_model]
    if args.trust_application_names:
        command.append('--trust-application-names')
    subprocess.run(command, check=True, cwd=cat.ROOT)
    current = index_discovery(cat.helpers.read_json(output), engine.prep)
    # These records correspond only to targets selected above.
    return dict(prior, **current), outcomes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cat.options(p)
    p.add_argument('--activity', type=Path, required=True)
    p.add_argument('--discovery', type=Path, help='Existing classify.py report; omitted means catalog-only detection offline')
    p.add_argument('--online', action='store_true', help='Allow enrichment and AI-detection API calls')
    p.add_argument('--api-model', default='gemini-3.8-flash')
    p.add_argument('--trust-application-names', action='store_true')
    p.add_argument('--enrichment-cache', type=Path, default=cat.ROOT/'data/cache/enrichment_cache.json')
    p.add_argument('--decision-cache', type=Path, default=cat.ROOT/'data/cache/ai_classification_cache.json')
    a = p.parse_args()
    try:
        cat.preflight(a, [a.activity, a.discovery, a.enrichment_cache, a.decision_cache])
        sources = [a.activity, a.discovery, a.catalog, a.known_tools, a.catalog_context,
                   a.reviews, a.report_selections, a.decision_cache, a.prep_script, a.training_script]
        if (a.enrichment_cache.resolve() in {s.resolve() for s in sources if s}
                or a.enrichment_cache.resolve().is_relative_to(a.model_dir.resolve())
                or a.enrichment_cache.resolve().is_relative_to(a.out.resolve())):
            raise ValueError('Enrichment cache must not overwrite a source/model file')
        engine = cat.Engine(a)
        ingestion = extract_applications(a.activity)
        prior = index_discovery(cat.helpers.read_json(a.discovery), engine.prep) if a.discovery else {}
        with tempfile.TemporaryDirectory(prefix='component3_') as temp:
            stage = Path(temp)
            outcomes = []
            if a.online:
                prior, outcomes = online_discovery(a, ingestion['applications'], engine, prior, stage)
            records = merge_records(ingestion['applications'], prior, engine)
            records = cat.categorize_records(records, engine, a)
            for r in records:
                if r['catalog_detection_conflict']:
                    r['categorization'].update(category=None, category_source=None,
                        category_status='pending_detection_conflict', top_three=[])
            report = cat.save_outputs(a, records, engine, {
                'scope': 'All extracted activity applications; Components 1-3. Governance is not recomputed.',
                'ingestion': ingestion, 'online_enabled': a.online, 'enrichment_outcomes': outcomes,
                'detection_counts': dict(Counter(str(r.get('is_ai_tool')) for r in records)),
                'runner_sha256': cat.helpers.sha(Path(__file__))})
            for f in stage.iterdir():
                shutil.copyfile(f, a.out/f.name)
            cat.helpers.write_json(a.out/'pipeline_summary.json', {
                'ingestion': ingestion['summary'], 'detection_counts': report['detection_counts'],
                'category_counts': report['summary'], 'online_enabled': a.online})
    except (OSError, ValueError, KeyError, ImportError, subprocess.CalledProcessError) as exc:
        p.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
