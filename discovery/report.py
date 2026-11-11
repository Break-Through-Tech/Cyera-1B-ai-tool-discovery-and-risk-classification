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


# Compatibility helpers above remain for the paused upload draft. The reporting
# entry point now uses the shared ingestion/catalog implementation.
from ai_discovery.report import run, main

if __name__ == '__main__':
    main()
