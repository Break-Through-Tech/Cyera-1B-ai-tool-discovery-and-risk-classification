"""Map ALL cleaned catalog rows to Dataset B; no API calls or row removal.

Run from project root (paths are relative to your current working directory):
  python scripts/map_dataset_b.py --input data/processed/kaggle_clean.csv
Dependencies: pandas. Blank CSV cells mean unknown, not false.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SCHEMA = ['tool_id', 'tool_name', 'domain', 'description', 'is_ai_tool',
          'category', 'vendor', 'approved_status', 'enterprise_plan_available',
          'sso_available', 'data_retention', 'training_on_customer_data',
          'admin_controls_available', 'risk_label']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('data/processed/kaggle_clean.csv'))
    p.add_argument('--out-dir', type=Path, default=Path('data/processed/dataset_b'))
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    raw = pd.read_csv(a.input, dtype=str, keep_default_na=False).fillna('')
    required = ['tool_id', 'tool_name', 'domain', 'description', 'category_raw',
                'primary_task', 'keywords', 'domain_valid', 'site_status_group',
                'is_platform_host', 'website_url_clean', 'url_path']
    missing = set(required) - set(raw.columns)
    if missing:
        p.error(f'Missing input columns: {sorted(missing)}')
    if raw.tool_id.str.strip().eq('').any() or raw.tool_id.duplicated().any():
        p.error('tool_id must be nonempty and unique.')
    paths = [a.out_dir / n for n in ['dataset_b_master.csv', 'dataset_b_context.csv', 'mapping_summary.json']]
    if a.input.resolve() in [x.resolve() for x in paths]:
        p.error('Output must not overwrite the input.')
    if not a.overwrite and any(x.exists() for x in paths):
        p.error('Outputs already exist. Use another --out-dir or --overwrite.')
    master = pd.DataFrame('', index=raw.index, columns=SCHEMA)
    for c in ['tool_id', 'tool_name', 'domain', 'description']:
        master[c] = raw[c]
    master['is_ai_tool'] = 'true'
    context = raw.copy()
    context['ai_label_provenance'] = 'provisional_catalog_membership'
    context['human_review_status'] = 'not_reviewed'
    context['governance_provenance'] = 'not_populated'
    a.out_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(paths[0], index=False, encoding='utf-8-sig')
    context.to_csv(paths[1], index=False, encoding='utf-8-sig')
    summary = {'rows_in': len(raw), 'rows_out': len(master), 'schema': SCHEMA,
               'input_sha256': hashlib.sha256(a.input.read_bytes()).hexdigest(),
               'note': 'AI labels provisional; blank enrichment and governance values are unknown.'}
    paths[2].write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Mapped all {len(master):,} rows. Outputs: {a.out_dir.resolve()}')


if __name__ == '__main__':
    main()
