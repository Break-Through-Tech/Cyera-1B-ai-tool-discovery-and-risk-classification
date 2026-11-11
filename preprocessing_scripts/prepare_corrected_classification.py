"""Prepare grouped binary splits from the corrected full source dataset."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/processed/corrected_labels_2026-11-05/all_corrected.csv'
OUT=ROOT/'data/processed/classification_corrected_full'
SEED=20261109
FIELDS=['record_id','group_id','tool_name','domain','description','is_ai_tool','label_method','split']

def norm(s):return re.sub(r'\W+',' ',s.casefold()).strip()
def host(s):
    try:return (urlsplit(s if '://' in s else 'https://'+s).hostname or '').lower().removeprefix('www.')
    except ValueError:return ''
def family(s):
    bits=s.split('.')
    # Conservative shared-host grouping is intentional: no unrelated tenant is split across sets.
    suffix='.'.join(bits[-2:])
    return '.'.join(bits[-3:] if suffix in {'co.uk','org.uk','com.au','co.in','co.jp','com.br','com.cn'} else bits[-2:])
def csv_bytes(rows,fields):
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows);return s.getvalue().encode()
def sha(b):return hashlib.sha256(b).hexdigest()

def prepare(raw,seed=SEED):
    if len({r['record_id'] for r in raw})!=len(raw):raise ValueError('Duplicate input IDs')
    candidates=[];excluded=[]
    def omit(r,reason):excluded.append(dict(record_id=r['record_id'],reason=reason))
    for r in raw:
        why=[]
        if r['is_ai_tool'] not in ('0','1'):why.append('unresolved_label')
        if not norm(r['tool_name']) or not norm(r['description']):why.append('missing_name_or_description')
        h=host(r['domain'])
        if not re.fullmatch(r'[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,}',h):why.append('invalid_domain')
        if why:omit(r,';'.join(why));continue
        candidates.append(dict(r,domain=h,description=' '.join(r['description'].split())))
    # Quarantine contradictory exact-name or exact-description labels before deduplication.
    sets=defaultdict(set)
    for r in candidates:
        for k in [('name',norm(r['tool_name'])),('description',norm(r['description']))]:sets[k].add(r['is_ai_tool'])
    conflict_keys={k for k,v in sets.items() if len(v)>1}
    unique={}
    for r in sorted(candidates,key=lambda r:(r['label_method'] not in ('assistant_official_source_review','official_source_scope_resolution'),-len(r['description']),r['record_id'])):
        if ('name',norm(r['tool_name'])) in conflict_keys or ('description',norm(r['description'])) in conflict_keys:
            omit(r,'remaining_cross_label_name_or_description');continue
        key=(norm(r['tool_name']),r['domain'])
        if key in unique:omit(r,'duplicate_name_and_host');continue
        unique[key]=r
    rows=sorted(unique.values(),key=lambda r:r['record_id'])
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    # Vendor strings are only grouping keys, never predictor features.
    for i,r in enumerate(rows):
        keys=[('host',r['domain']),('host_family',family(r['domain'])),('name',norm(r['tool_name'])),('description',norm(r['description']))]
        if r.get('vendor_status')=='model_grounded_found' and norm(r.get('vendor','')):keys.append(('vendor',norm(r['vendor'])))
        for k in keys:
            if k in seen:parent[find(i)]=find(seen[k])
            else:seen[k]=i
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[find(i)].append(r)
    totals=Counter(r['is_ai_tool'] for r in rows)
    if set(totals)!={'0','1'}:raise ValueError('Both classes required')
    gs=list(groups.values());random.Random(seed).shuffle(gs)
    gs.sort(key=lambda g:-max(sum(r['is_ai_tool']==c for r in g)/totals[c] for c in ('0','1')))
    fractions={'train':.7,'validation':.15,'test':.15};counts={s:Counter() for s in fractions};splits={s:[] for s in fractions}
    for group in gs:
        c=Counter(r['is_ai_tool'] for r in group)
        def cost(s):
            return sum(((counts[s][label]+c[label]-totals[label]*fractions[s])**2-(counts[s][label]-totals[label]*fractions[s])**2)/(totals[label]*fractions[s]) for label in totals)
        split=min(fractions,key=cost)
        gid='g_'+sha('|'.join(sorted(r['record_id'] for r in group)).encode())[:16]
        counts[split].update(c)
        splits[split].extend({**{k:r[k] for k in FIELDS if k not in ('group_id','split')},'group_id':gid,'split':split} for r in group)
    for s in splits:splits[s].sort(key=lambda r:r['record_id'])
    validate(splits)
    return splits,sorted(excluded,key=lambda r:r['record_id'])

def validate(splits):
    if set(splits)!={'train','validation','test'}:raise ValueError('Expected three splits')
    for s,rows in splits.items():
        if {r['is_ai_tool'] for r in rows}!={'0','1'}:raise ValueError(s+' lacks a class')
    for key in ('record_id','group_id','tool_name','domain','description'):
        seen={}
        for s,rows in splits.items():
            for r in rows:
                value=norm(r[key])
                if value in seen and seen[value]!=s:raise ValueError('Cross-split leakage: '+key)
                seen[value]=s
    ids=[r['record_id'] for rows in splits.values() for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate IDs')

def build(source=SOURCE,version="corrected-full-2026-11-09",seed=SEED):
    data=Path(source).read_bytes();raw=list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
    splits,excluded=prepare(raw,seed=seed)
    files={s+'.csv':csv_bytes(rows,FIELDS) for s,rows in splits.items()}
    files['excluded_records.csv']=csv_bytes(excluded,['record_id','reason'])
    if sum(map(len,splits.values()))+len(excluded)!=len(raw):raise ValueError('Input accounting failed')
    manifest=dict(version=version,seed=seed,input_sha256=sha(data),input_rows=len(raw),
        split_counts={s:dict(rows=len(rows),classes=dict(Counter(r['is_ai_tool'] for r in rows)),groups=len({r['group_id'] for r in rows})) for s,rows in splits.items()},
        excluded_rows=len(excluded),exclusions=dict(Counter(r['reason'] for r in excluded)),
        outputs_sha256={name:sha(b) for name,b in files.items()},
        feature='description only; no names, vendor, review, category or governance predictor features',
        grouping='Transitive host, conservative host family, exact normalized name/description, and known normalized vendor; no vendor ownership inference for uncached records',
        limitations=['Labels mostly source-assigned; not independently human verified.', 'Splits are new, but many products were seen in earlier experiments or label review.', 'Source-dependent description style and length can confound prediction.', 'Grouping is conservative, not a complete global ownership/alias database.'])
    files['manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    return files,manifest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--version',default='corrected-full-2026-11-09');p.add_argument('--seed',type=int,default=SEED);a=p.parse_args()
    files,m=build(a.source,a.version,a.seed);out=a.out;existing=out/'manifest.json'
    if existing.exists() and existing.read_bytes()!=files['manifest.json']:raise SystemExit('Frozen dataset changed; create a new version')
    if a.freeze:
        out.mkdir(parents=True,exist_ok=True)
        for name,b in files.items():(out/name).write_bytes(b)
    else:
        for name,b in files.items():
            if not (out/name).exists() or (out/name).read_bytes()!=b:raise SystemExit('Does not reproduce: '+name)
    print(json.dumps(m,indent=2))
if __name__=='__main__':main()
