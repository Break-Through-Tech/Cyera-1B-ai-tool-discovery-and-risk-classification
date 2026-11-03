"""Full available-data preparation: catalog positives + recovered team negatives.

Unreviewed catalog membership is weak positive supervision, never gold truth.
Only model-accepted positives and team negatives enter validation/test.
"""
import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from prepare_classification import ROOT, clean_host, load, normalize, save, validate

FIELDS = ['application_id','group_id','tool_name','domain','description','text_for_model',
          'is_ai_tool','category','category_source','label_source','label_quality',
          'source_record_id','source_file','split']
AUDIT_FIELDS = ['source_record_id','tool_name','domain','review_status','disposition','reason']


def prepare(catalog, reviews, negatives, seed=42):
    review = {r['tool_id']:r for r in reviews}
    if len(review) != len(reviews):
        raise ValueError('Duplicate tool IDs in reviews')
    if len({r['tool_id'] for r in catalog}) != len(catalog):
        raise ValueError('Duplicate tool IDs in catalog')
    if set(review) - {r['tool_id'] for r in catalog}:
        raise ValueError('Review contains IDs absent from catalog')
    negative_keys = set()
    for n in negatives:
        negative_keys.update([('name',normalize(n['tool_name'])),('host',clean_host(n['domain'])),
                              ('desc',normalize(n['description']))])
    audit, candidates, chosen = {}, [], {}
    # Prefer model-accepted records when a catalog application is duplicated.
    catalog = sorted(catalog, key=lambda r:(review.get(r['tool_id'],{}).get('review_status')!='accepted',r['tool_id']))
    for raw in catalog:
        tid = raw['tool_id']
        rv = review.get(tid,{})
        status = rv.get('review_status','unreviewed')
        name, desc = [' '.join(raw.get(k,'').split()) for k in ('tool_name','description')]
        host = clean_host(raw.get('domain',''))
        a = dict(source_record_id=tid,tool_name=name,domain=host,review_status=status,
                 disposition='excluded',reason='')
        audit[tid] = a
        if status not in ('accepted','unreviewed'):
            a['reason'] = 'review_' + status
        elif status=='accepted' and (rv.get('ai_status')!='ai' or rv.get('description_usable','').lower()!='true'):
            a['reason'] = 'inconsistent_accepted_review'
        elif not name or not desc:
            a['reason'] = 'missing_name_or_description'
        elif raw.get('domain_valid','').lower()!='true' or not re.fullmatch(r'[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,}',host):
            a['reason'] = 'invalid_domain'
        elif raw.get('site_status_group')=='dead':
            a['reason'] = 'flagged_dead_site'
        elif status=='unreviewed' and any(k in negative_keys for k in
                [('name',normalize(name)),('host',host),('desc',normalize(desc))]):
            a['reason'] = 'unreviewed_overlap_with_negative_seed'
        if a['reason']:
            continue
        key = (normalize(name),host)
        if key in chosen:
            a['reason'] = 'duplicate_application'
            continue
        chosen[key] = tid
        is_reviewed = status=='accepted'
        candidates.append(dict(application_id='app_'+hashlib.sha256(repr(key).encode()).hexdigest()[:16],
            tool_name=name,domain=host,description=desc,text_for_model=f'{name}. {desc}',is_ai_tool=1,
            category=rv.get('category','') if is_reviewed else '',
            category_source='gemini_text_review' if is_reviewed else '',
            label_source='gemini_text_review' if is_reviewed else 'provisional_catalog_membership',
            label_quality='weak_supervision',source_record_id=tid,source_file='kaggle_clean.csv'))
    negative_ids=set()
    for raw in negatives:
        if raw['tool_id'] in negative_ids or raw['is_ai_tool'].lower() not in ('false','0'):
            raise ValueError('Invalid or duplicate negative seed label/ID')
        negative_ids.add(raw['tool_id'])
        name,desc=[' '.join(raw.get(k,'').split()) for k in ('tool_name','description')]
        host=clean_host(raw.get('domain',''))
        if not name or not desc or '.' not in host:
            raise ValueError('Negative seed needs name, description and valid hostname')
        key=(normalize(name),host)
        # Contradiction for the exact same application must be resolved, not trained.
        if key in chosen:
            raise ValueError(f'Conflicting accepted AI and negative identity: {name}')
        if any(r['tool_name']==name and r['domain']==host and r['is_ai_tool']==0 for r in candidates):
            raise ValueError('Duplicate negative application identity')
        candidates.append(dict(application_id='app_'+hashlib.sha256(repr(key).encode()).hexdigest()[:16],
            tool_name=name,domain=host,description=desc,text_for_model=f'{name}. {desc}',is_ai_tool=0,
            category='',category_source='',label_source='team_non_ai_seed',label_quality='weak_supervision',
            source_record_id=raw['tool_id'],source_file='non_ai_enterprise_tools.csv'))
    candidates.sort(key=lambda r:r['application_id'])
    parent=list(range(len(candidates)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    seen={}
    for i,r in enumerate(candidates):
        for key in [('host',r['domain']),('name',normalize(r['tool_name'])),('desc',normalize(r['description']))]:
            if key in seen:
                parent[find(i)]=find(seen[key])
            else:
                seen[key]=i
    groups=defaultdict(list)
    for i,r in enumerate(candidates):
        groups[find(i)].append(r)
    strata=defaultdict(list)
    assignments={}
    for gid,g in groups.items():
        anchors=[r for r in g if r['label_source']!='provisional_catalog_membership']
        if anchors:
            strata[tuple(sorted({r['is_ai_tool'] for r in anchors}))].append(gid)
        else:
            assignments[gid]='train'
    rng=random.Random(seed)
    for labels,gids in sorted(strata.items()):
        # Stable ordering independent of input rows.
        gids.sort(key=lambda gid:min(r['application_id'] for r in groups[gid]))
        rng.shuffle(gids)
        if len(gids)<3:
            raise ValueError(f'Not enough independent evaluation groups for {labels}')
        n=max(1,math.floor(len(gids)*.15+.5))
        for split,selected in [('test',gids[:n]),('validation',gids[n:2*n]),('train',gids[2*n:])]:
            assignments.update({gid:split for gid in selected})
    splits={s:[] for s in ('train','validation','test')}
    for gid,g in groups.items():
        split=assignments[gid]
        group_id='group_'+hashlib.sha256('|'.join(sorted(r['application_id'] for r in g)).encode()).hexdigest()[:16]
        for r in g:
            if split!='train' and r['label_source']=='provisional_catalog_membership':
                audit[r['source_record_id']].update(disposition='quarantined',reason='related_to_heldout_application')
                continue
            splits[split].append(dict(r,group_id=group_id,split=split))
            if r['is_ai_tool']==1:
                audit[r['source_record_id']].update(disposition=split,reason='included')
    for rows in splits.values():rows.sort(key=lambda r:r['application_id'])
    validate(splits)
    for split in ('validation','test'):
        if any(r['label_source']=='provisional_catalog_membership' for r in splits[split]):
            raise ValueError('Unreviewed positives in evaluation')
    if any(not a['reason'] for a in audit.values()):
        raise ValueError('Unaccounted catalog row')
    return splits, sorted(audit.values(),key=lambda a:a['source_record_id'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,default=ROOT/'data/processed/kaggle_clean.csv')
    p.add_argument('--reviews',type=Path,default=ROOT/'data/processed/dataset_b/sample/review_results.csv')
    p.add_argument('--non-ai',type=Path,default=ROOT/'data/raw/non_ai_enterprise_tools.csv')
    p.add_argument('--out',type=Path,default=ROOT/'data/processed/classification_full')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    inputs=[a.catalog,a.reviews,a.non_ai]
    if any(a.out.resolve()==p.parent.resolve() for p in inputs):p.error('Output must use a separate directory')
    if a.out.exists() and any(a.out.iterdir()) and not a.overwrite:p.error('Output exists; use --overwrite or a new --out')
    splits,audit=prepare(*(load(p) for p in inputs),seed=a.seed)
    a.out.mkdir(parents=True,exist_ok=True)
    combined=[r for rows in splits.values() for r in rows]
    for name,rows in [('classification',combined),*splits.items()]:save(a.out/f'{name}.csv',rows,FIELDS)
    save(a.out/'catalog_audit.csv',audit,AUDIT_FIELDS)
    summary=dict(catalog_records=len(audit),negative_seed_records=len(load(a.non_ai)),
        prepared_records=len(combined),catalog_dispositions=dict(Counter(r['disposition'] for r in audit)),
        catalog_reasons=dict(Counter(r['reason'] for r in audit)),
        splits={s:dict(rows=len(rows),class_counts=dict(Counter(str(r['is_ai_tool']) for r in rows)),
            label_sources=dict(Counter(r['label_source'] for r in rows))) for s,rows in splits.items()},
        target='is_ai_tool',label_mapping={'0':'conventional primary application purpose','1':'AI-focused tool'},
        governance_risk_target=False,independently_verified_test_labels=False,
        checks={'all_catalog_rows_accounted_for':True,'no_cross_split_identity_overlap':True,
                'both_classes_in_every_split':True,'no_provisional_catalog_labels_in_evaluation':True})
    manifest=dict(seed=a.seed,inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        features=['text_for_model'],text_composition='tool_name + description',
        split_policy='70/15/15 groups of model-reviewed/seed anchors; provisional-only groups train; heldout-related provisional rows quarantined',
        label_policy='Weak supervision only. Review exclusions override catalog membership. Negative-overlap unreviewed catalog rows excluded.',
        unknown_categories='blank; never fabricated')
    for name,data in [('summary.json',summary),('manifest.json',manifest)]:
        (a.out/name).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
