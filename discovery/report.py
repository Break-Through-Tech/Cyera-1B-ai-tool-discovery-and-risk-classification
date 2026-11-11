"""Join exact catalog identities, optional unseen descriptions and usage risk."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from governance import risk

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'data/processed/corrected_labels_2026-11-05/all_corrected.csv'


def catalog_index(rows):
    index = {}
    for row in rows:
        try:
            key = risk.identity(row.get('domain', ''))
        except ValueError:
            continue
        index.setdefault(key, []).append(row)
    return index


def identify(key, index, descriptions, predictor):
    matches = index.get(key, [])
    # Multiple product scopes on a shared domain are never resolved by majority vote.
    signatures = {(r['tool_name'].strip().casefold(), r['is_ai_tool']) for r in matches}
    if len(signatures) == 1 and matches[0]['is_ai_tool'] in ('0', '1'):
        return dict(is_ai_tool=int(matches[0]['is_ai_tool']), method='catalog_label',
                    model_ai_score=None, record_ids=sorted(r['record_id'] for r in matches),
                    needs_review=True, reason='Supplied catalog label; not independent verification.')
    if matches:
        return dict(is_ai_tool=None, method='unresolved_catalog', model_ai_score=None,
                    record_ids=sorted(r['record_id'] for r in matches), needs_review=True,
                    reason='Ambiguous product scope or unresolved label; no automatic override.')
    description = descriptions.get(key, '').strip()
    if description:
        return dict(predictor(description), method='description_model', record_ids=[])
    return dict(is_ai_tool=None, method='unknown', model_ai_score=None, record_ids=[],
                needs_review=True, reason='No exact catalog identity or supplied description. Names alone are not classified.')


def run(company, activity, catalog=CATALOG, descriptions=None, model=None, scenarios=None):
    # Empty governance metadata is intentional: classification labels/descriptions
    # are not vendor security evidence. Explicit fictional overlays remain separate.
    report = risk.run(company, activity, ROOT/'data/processed/dataset_b/dataset_b_master.csv',
                      risk.POLICY, scenarios)
    index = catalog_index(risk.read_csv(catalog))
    supplied = {}
    for row in risk.read_csv(descriptions) if descriptions else []:
        key = risk.identity(row['domain'])
        if key in supplied:
            raise ValueError('Duplicate description identity: ' + key)
        supplied[key] = row['description']
    def predictor(text):
        if model is None:
            return dict(is_ai_tool=None, model_ai_score=None, needs_review=True,
                        reason='Description supplied but no model enabled.')
        from modeling.train_corrected_binary import predict
        return predict(text, model)
    for app in report['applications']:
        app['detection'] = identify(app['application_key'], index, supplied, predictor)
        # Unknown detections stay visible even when the observed risk is low.
        app['needs_review'] = app['needs_review'] or app['detection']['needs_review']
    counts = Counter('ai' if a['detection']['is_ai_tool'] == 1 else
                     'conventional' if a['detection']['is_ai_tool'] == 0 else 'unknown'
                     for a in report['applications'])
    report['detection_counts'] = {k: counts[k] for k in ('ai','conventional','unknown')}
    report['review_queue'] = [a['application_key'] for a in report['applications'] if a['needs_review']]
    report['inputs']['classification_catalog'] = {'file':Path(catalog).name,
        'sha256':hashlib.sha256(Path(catalog).read_bytes()).hexdigest()}
    if descriptions:
        report['inputs']['supplied_descriptions'] = {'file':Path(descriptions).name,
            'sha256':hashlib.sha256(Path(descriptions).read_bytes()).hexdigest()}
    if model:
        from modeling.train_corrected_binary import verified
        config = verified(model)
        report['model_sha256'] = config['model_sha256']
    report['limitations'] = [x for x in report['limitations'] if 'separate upstream task' not in x]
    report['limitations'] += [
        'Catalog labels are mostly source assignments; all detections request review.',
        'Unknown applications need supplied descriptions; no website enrichment is performed.',
        'AI classification and governance priority are separate; unknown tools are retained.',
        'No updated purpose-category classifier or conversational assistant is integrated.',
        'Description model is experimental and its scores are uncalibrated.']
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--company', required=True)
    p.add_argument('--activity', type=Path, required=True)
    p.add_argument('--catalog', type=Path, default=CATALOG)
    p.add_argument('--descriptions', type=Path, help='CSV: domain,description; exact identities only')
    p.add_argument('--model', type=Path, help='Trusted local corrected binary model directory')
    p.add_argument('--scenarios', type=Path, help='Explicitly fictional company policy overlay')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('Output exists; choose a new file')
    report = run(a.company,a.activity,a.catalog,a.descriptions,a.model,a.scenarios)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'company':a.company,'applications':report['application_count'],
                      'detection_counts':report['detection_counts']},indent=2))


if __name__ == '__main__':
    main()
