"""Prepare issue #5's binary AI detection dataset. Python standard library only.

Run from any directory: python3 preprocessing_scripts/prepare_classification.py
Labels are weak supervision inherited from the team's sources, not human gold.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['application_id', 'group_id', 'tool_name', 'domain', 'description',
          'text_for_model', 'is_ai_tool', 'category', 'category_source',
          'label_source', 'label_quality', 'source_record_id', 'source_file', 'split']


def load(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def normalize(text):
    return re.sub(r'\W+', ' ', text.casefold()).strip()


def clean_host(value):
    u = urlsplit(value if '://' in value else 'https://' + value)
    return (u.hostname or '').casefold().removeprefix('www.')


def build(ai, non_ai, seed=42):
    rows, excluded, identities = [], [], {}
    for source, data, label in [('gemini_text_review', ai, 1), ('team_non_ai_seed', non_ai, 0)]:
        for raw in data:
            name = ' '.join(raw.get('tool_name', '').split())
            desc = ' '.join(raw.get('description', '').split())
            host = clean_host(raw.get('domain', ''))
            record_id = raw.get('tool_id', '')
            why = ''
            if raw.get('is_ai_tool', '').lower() not in ('true' if label else 'false', str(label)):
                why = 'unexpected_source_label'
            elif not name or not desc:
                why = 'missing_name_or_description'
            elif not re.fullmatch(r'[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,}', host):
                why = 'invalid_domain'
            if why:
                excluded.append(dict(source_record_id=record_id, tool_name=name, reason=why))
                continue
            # Same name at the same host is one application. Different AI features
            # sharing a host remain distinct applications but share a split group.
            key = (normalize(name), host)
            app_id = 'app_' + hashlib.sha256(repr(key).encode()).hexdigest()[:16]
            row = dict(application_id=app_id, tool_name=name, domain=host,
                       description=desc, text_for_model=f'{name}. {desc}', is_ai_tool=label,
                       category=raw.get('category', '') if label else '',
                       category_source='gemini_text_review' if label else '',
                       label_source=source, label_quality='weak_supervision',
                       source_record_id=record_id,
                       source_file='accepted_dataset_b.csv' if label else 'non_ai_enterprise_tools.csv')
            if key in identities:
                existing = identities[key]
                if existing['is_ai_tool'] != label:
                    existing['_conflict'] = True
                    row['_conflict'] = True
                    rows.append(row)
                else:
                    excluded.append(dict(source_record_id=record_id, tool_name=name, reason='duplicate_application'))
            else:
                identities[key] = row
                rows.append(row)
    conflict_ids = {r['application_id'] for r in rows if r.get('_conflict')}
    clean = []
    for r in rows:
        if r['application_id'] in conflict_ids:
            excluded.append(dict(source_record_id=r['source_record_id'], tool_name=r['tool_name'], reason='conflicting_identity_labels'))
        else:
            clean.append(r)
    rows = sorted(clean, key=lambda r: r['application_id'])
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, r in enumerate(rows):
        for key in [('host', r['domain']), ('name', normalize(r['tool_name'])),
                    ('description', normalize(r['description']))]:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[find(i)].append(r)
    strata = defaultdict(list)
    for g in groups.values():
        # Mixed-label shared-host groups are valid, and must stay together.
        strata[tuple(sorted({r['is_ai_tool'] for r in g}))].append(g)
    splits = {s: [] for s in ('train', 'validation', 'test')}
    rng = random.Random(seed)
    for labels, gs in sorted(strata.items()):
        rng.shuffle(gs)
        if len(labels) == 1 and len(gs) < 3:
            raise ValueError(f'Class {labels[0]} needs at least three independent application groups')
        n = max(1, math.floor(len(gs) * .15 + .5)) if len(gs) >= 3 else 0
        for split, chosen in [('test', gs[:n]), ('validation', gs[n:2*n]), ('train', gs[2*n:])]:
            for g in chosen:
                gid = 'group_' + hashlib.sha256('|'.join(sorted(r['application_id'] for r in g)).encode()).hexdigest()[:16]
                splits[split].extend(dict(r, group_id=gid, split=split) for r in g)
    for s in splits:
        splits[s].sort(key=lambda r: r['application_id'])
    validate(splits)
    return splits, excluded


def validate(splits):
    for split, rows in splits.items():
        if {r['is_ai_tool'] for r in rows} != {0, 1}:
            raise ValueError(f'{split} must contain both AI and non-AI examples')
    for field in ['application_id', 'group_id', 'domain', 'tool_name', 'description']:
        values = [{normalize(r[field]) for r in rows} for rows in splits.values()]
        if any(values[i] & values[j] for i in range(3) for j in range(i)):
            raise ValueError(f'Cross-split leakage: {field}')
    ids = [r['application_id'] for data in splits.values() for r in data]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate application IDs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ai', type=Path, default=ROOT/'data/processed/dataset_b/sample/accepted_dataset_b.csv')
    p.add_argument('--non-ai', type=Path, default=ROOT/'data/raw/non_ai_enterprise_tools.csv')
    p.add_argument('--out', type=Path, default=ROOT/'data/processed/classification')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()) and not a.overwrite:
        p.error('Output directory is not empty; use --overwrite or a new --out')
    outputs = [a.out/n for n in ['classification.csv','train.csv','validation.csv','test.csv','excluded.csv','summary.json','manifest.json']]
    if any(x.resolve() in {a.ai.resolve(), a.non_ai.resolve()} for x in outputs):
        p.error('Output may not overwrite an input')
    ai, neg = load(a.ai), load(a.non_ai)
    splits, excluded = build(ai, neg, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    all_rows = [r for rows in splits.values() for r in rows]
    save(a.out/'classification.csv', all_rows, FIELDS)
    for s, rows in splits.items():
        save(a.out/f'{s}.csv', rows, FIELDS)
    save(a.out/'excluded.csv', excluded, ['source_record_id','tool_name','reason'])
    summary = dict(target='is_ai_tool', label_mapping={'0':'non-AI primary application purpose','1':'AI-focused tool'},
        input_rows={'ai':len(ai),'non_ai':len(neg)}, output_rows=len(all_rows), excluded_rows=len(excluded),
        class_counts=dict(Counter(str(r['is_ai_tool']) for r in all_rows)),
        splits={s:dict(rows=len(rows),class_counts=dict(Counter(str(r['is_ai_tool']) for r in rows)),
            ai_category_counts=dict(Counter(r['category'] for r in rows if r['is_ai_tool']==1))) for s,rows in splits.items()},
        checks={'both_classes_every_split':True,'unique_application_ids':True,'no_cross_split_identity_host_name_description_overlap':True},
        ready_for_baseline_development=True, gold_standard_evaluation=False,
        label_quality='weak_supervision',
        limitations=['AI labels and categories are model-reviewed, not independently verified.',
            'Non-AI labels are inherited from team seed data; ordinary platforms may include AI features.',
            '30 negatives provide a small, imbalanced evaluation sample.',
            'Different source writing styles may confound AI detection.',
            'No real-world accuracy or governance risk is established by these labels.'])
    manifest=dict(seed=a.seed,split_ratios={'train':.7,'validation':.15,'test':.15},
        split_unit='connected application groups; approximate ratios',
        features=['text_for_model'],text_composition='tool_name + description, same fields for both classes',
        excluded_features=['category','label_source','label_quality','source_file','domain','risk_label','approved_status'],
        recovered_non_ai_source='87f760769e39e89ba3bcb1691d8f0eb32c60e4b2:data/non_ai_enterprise_tools.csv',
        inputs={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('ai',a.ai),('non_ai',a.non_ai)]})
    for name,data in [('summary.json',summary),('manifest.json',manifest)]:
        (a.out/name).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({s:len(rows) for s,rows in splits.items()}))


if __name__ == '__main__':
    main()
