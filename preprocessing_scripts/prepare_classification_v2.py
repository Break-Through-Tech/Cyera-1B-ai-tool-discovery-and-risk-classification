"""Build frozen, source-reviewed v2 splits. Never trains or evaluates a model."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/labels_v2/source_labels.json'
OUTPUT = ROOT / 'data/processed/classification_v2'
SEED = 20261104


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate(rows):
    required = {'label_id', 'tool_name', 'description', 'is_ai_tool', 'category',
                'family_id', 'source_url', 'resolved_source_url', 'source_checked_on',
                'source_text_sha256', 'has_ai_features', 'human_verified'}
    if len({r['label_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate label IDs')
    for row in rows:
        if not required <= row.keys() or not all(row[k] for k in required - {'category', 'is_ai_tool', 'human_verified'}):
            raise ValueError('Missing provenance or model input')
        if row['is_ai_tool'] not in (0, 1) or bool(row['category']) != bool(row['is_ai_tool']):
            raise ValueError('Category must be present only for AI examples')
        if row['has_ai_features'] not in ('yes', 'unknown'):
            raise ValueError('Absence of an AI claim does not establish AI-free status')
        if row['has_ai_features'] == 'yes' and not row.get('ai_feature_evidence_url'):
            raise ValueError('AI-feature assertion needs a source')
        if row['human_verified'] is not False:
            raise ValueError('This release contains assistant reviews, not human reviews')
        if not row['source_url'].startswith('https://') or len(row['source_text_sha256']) != 64:
            raise ValueError('Invalid source evidence')


def groups_for(rows):
    # Union families with identical descriptions to prevent text duplicates leaking.
    parent = {r['family_id']: r['family_id'] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    seen = {}
    for r in rows:
        key = ' '.join(r['description'].lower().split())
        if key in seen:
            a, b = sorted((find(r['family_id']), find(seen[key])))
            parent[b] = a
        seen[key] = r['family_id']
    groups = collections.defaultdict(list)
    for r in rows:
        groups[find(r['family_id'])].append(r)
    return dict(groups)


def select_split(groups, categories, per_category, negatives, seed):
    rng = random.Random(seed)
    # Search only for count/family constraints, never model performance.
    for _ in range(10000):
        keys = sorted(groups)
        rng.shuffle(keys)
        selected = []
        counts = collections.Counter()
        n = 0
        for key in keys:
            g = groups[key]
            c = collections.Counter(r['category'] for r in g if r['is_ai_tool'])
            gn = sum(1-r['is_ai_tool'] for r in g)
            if c and all(counts[k]+v <= per_category for k,v in c.items()) and n+gn <= negatives:
                selected.append(key); counts.update(c); n += gn
        if any(counts[c] != per_category for c in categories):
            continue
        for key in keys:
            if key in selected or any(r['is_ai_tool'] for r in groups[key]):
                continue
            if n+len(groups[key]) <= negatives:
                selected.append(key); n += len(groups[key])
        if n == negatives:
            return selected
    raise ValueError('Cannot satisfy family and balance constraints')


def build(source_bytes):
    rows = json.loads(source_bytes)
    validate(rows)
    groups = groups_for(rows)
    categories = sorted({r['category'] for r in rows if r['is_ai_tool']})
    remaining = dict(groups)
    splits = {}
    for name, per_cat, negatives, seed in [('test', 3, 36, SEED), ('validation', 1, 12, SEED+1)]:
        keys = select_split(remaining, categories, per_cat, negatives, seed)
        splits[name] = [r for key in keys for r in remaining.pop(key)]
    splits['train'] = [r for g in remaining.values() for r in g]
    files = {}
    summary = {}
    for name, records in splits.items():
        records = sorted(records, key=lambda r:r['label_id'])
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
        files[name+'.csv'] = stream.getvalue().encode()
        summary[name] = {'rows': len(records), 'class_counts': dict(sorted(collections.Counter(str(r['is_ai_tool']) for r in records).items())), 'ai_category_counts': dict(sorted(collections.Counter(r['category'] for r in records if r['is_ai_tool']).items()))}
    protected = {'dataset_version': '2.0', 'families': sorted({r['family_id'] for s in ('validation','test') for r in splits[s]}), 'description_sha256': sorted({digest(' '.join(r['description'].lower().split()).encode()) for s in ('validation','test') for r in splits[s]}), 'instruction': 'Exclude all related applications, aliases, vendors and duplicate descriptions from training, including any v1/catalog augmentation. Family IDs require explicit mapping for legacy data; exact string filtering alone is insufficient.'}
    files['protected_evaluation_groups.json'] = (json.dumps(protected, indent=2)+'\n').encode()
    manifest = {'dataset_version':'2.0', 'seed':SEED, 'source_labels_sha256':digest(source_bytes), 'labels':'assistant_source_review; not independently human verified', 'test_policy':'Frozen before v2 model development. Do not tune on test. Existing v1 models may have trained on overlapping products.', 'splits':summary, 'sha256':{name:digest(data) for name,data in sorted(files.items())}}
    files['manifest.json'] = (json.dumps(manifest,indent=2)+'\n').encode()
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true', help='Write the initial release only; refuses changes to an existing manifest')
    args = parser.parse_args()
    files = build(SOURCE.read_bytes())
    manifest = OUTPUT/'manifest.json'
    if manifest.exists() and manifest.read_bytes() != files['manifest.json']:
        raise SystemExit('Frozen release changed. Create a new dataset version instead of overwriting v2.')
    if args.freeze:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        for name,data in files.items(): (OUTPUT/name).write_bytes(data)
    else:
        for name,data in files.items():
            if not (OUTPUT/name).exists() or (OUTPUT/name).read_bytes() != data:
                raise SystemExit('Artifact missing or does not reproduce: '+name)
    print('v2 artifacts reproduce; no model trained or evaluated.')
    print(json.dumps(json.loads(files['manifest.json'])['splits'],indent=2))


if __name__ == '__main__':
    main()
