"""Prepare synthetic catalog governance, reusing finalized reference values.

Default: print summary and sample; write nothing. Pass --output to save a NEW
CSV and provenance JSON. Original CSVs are never edited. Vendor is omitted.
Descriptions, categories, identities and AI labels remain unchanged.
Approval/risk are generic demo values, NOT company-specific decisions or facts.
"""
import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

GOV=['approved_status','enterprise_plan_available','sso_available','data_retention',
     'training_on_customer_data','admin_controls_available']
PROFILES={
 'strong_controls':dict(enterprise_plan_available='true',sso_available='true',admin_controls_available='true'),
 'mixed_controls':dict(enterprise_plan_available='true',sso_available='false',admin_controls_available='true'),
 'limited_controls':dict(enterprise_plan_available='false',sso_available='false',admin_controls_available='false')}
POINTS={'approved_status':{'approved':0,'under_review':1,'unapproved':2},
        'sso_available':{'true':0,'false':1},'admin_controls_available':{'true':0,'false':1},
        'data_retention':{'no_retention':0,'30_days':0,'90_days':1,'365_days':2,'indefinite':3},
        'training_on_customer_data':{'true':3,'false':0}}
VALID={**{k:set(v) for k,v in POINTS.items()},'enterprise_plan_available':{'true','false'},'risk_label':{'low','medium','high'}}


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f);fields=r.fieldnames or [];rows=list(r)
    required={'tool_id','tool_name','domain','description','category','is_ai_tool',*GOV,'risk_label'}
    if not required<=set(fields) or len(fields)!=len(set(fields)):raise ValueError(f'{path}: missing/duplicate columns')
    ids=set()
    for row in rows:
        if None in row or any(v is None for v in row.values()):raise ValueError('Malformed CSV')
        if not row['tool_id'] or row['tool_id'] in ids:raise ValueError('Missing/duplicate tool IDs')
        ids.add(row['tool_id'])
        for k,allowed in VALID.items():
            if row[k] and row[k] not in allowed:raise ValueError(f"{row['tool_id']}: unsupported {k}={row[k]!r}")
    return fields,rows


def identity(row):
    name=re.sub(r'\s+',' ',row['tool_name'].strip().casefold())
    raw=row['domain'].strip()
    try:
        p=urlsplit(raw if '://' in raw else 'https://'+raw)
        endpoint=(p.hostname or '').lower().removeprefix('www.')+p.path.rstrip('/')
    except ValueError:endpoint=''
    if not name or not endpoint:raise ValueError('Missing name/domain for '+row['tool_id'])
    return name,endpoint


def generate(seed,tid):
    # Matches the earlier fill_governance_dataset_b.py algorithm exactly.
    rng=random.Random(int(hashlib.sha256(f'{seed}:{tid}'.encode()).hexdigest(),16))
    values=dict(PROFILES[rng.choices(list(PROFILES),weights=[35,30,25])[0]])
    values['approved_status']=rng.choices(['approved','unapproved','under_review'],weights=[50,30,20])[0]
    values['data_retention']=rng.choice(['no_retention','30_days','90_days','365_days','indefinite'])
    values['training_on_customer_data']=rng.choices(['true','false'],weights=[25,75])[0]
    return values


def risk(row):
    score=sum(weights[row[k]] for k,weights in POINTS.items())
    return 'low' if score<=2 else 'medium' if score<=5 else 'high'


def prepare(rows,references,seed=42):
    index=defaultdict(list)
    for r in references:index[identity(r)].append(r)
    result=[];audit=[];label_conflicts=[];risk_conflicts=[];matched=0;overwrites=0
    for original in rows:
        r={k:v for k,v in original.items() if k!='vendor'}
        matches=index.get(identity(r),[])
        resolved={}
        for field in GOV+['risk_label']:
            values={x[field] for x in matches if x[field]}
            if len(values)>1:raise ValueError(f"Conflicting reference {field} for {r['tool_name']}; refusing to guess")
            resolved[field]=next(iter(values),'')
        if matches:matched+=1
        reference_labels={x['is_ai_tool'].strip().lower() for x in matches}
        if reference_labels and r['is_ai_tool'].strip().lower() not in reference_labels:
            label_conflicts.append({'tool_id':r['tool_id'],'tool_name':r['tool_name'],'catalog_label':r['is_ai_tool'],'reference_labels':sorted(reference_labels)})
        generated=generate(seed,r['tool_id']);provenance={}
        for field in GOV:
            if resolved[field]:
                overwrites+=bool(r[field] and r[field]!=resolved[field])
                r[field]=resolved[field];provenance[field]='finalized_reference'
            elif r[field]:provenance[field]='existing_catalog'
            else:r[field]=generated[field];provenance[field]='synthetic_generated'
        calculated=risk(r)
        if resolved['risk_label']:
            overwrites+=bool(r['risk_label'] and r['risk_label']!=resolved['risk_label'])
            r['risk_label']=resolved['risk_label'];provenance['risk_label']='finalized_reference'
        elif r['risk_label']:provenance['risk_label']='existing_catalog'
        else:r['risk_label']=calculated;provenance['risk_label']='synthetic_rule_calculated'
        if r['risk_label']!=calculated:risk_conflicts.append({'tool_id':r['tool_id'],'stored':r['risk_label'],'rule_result':calculated})
        assert all(r[k]==original[k] for k in ['tool_id','tool_name','domain','description','category','is_ai_tool'])
        result.append(r);audit.append({'tool_id':r['tool_id'],'reference_ids':[x['tool_id'] for x in matches],'fields':provenance})
    summary={'rows':len(rows),'matched_reference_rows':matched,'rows_without_reference':len(rows)-matched,
             'governance_cells_overridden_to_match_reference':overwrites,
             'missing_governance_cells_after':sum(not r[k] for r in result for k in GOV+['risk_label']),
             'empty_descriptions_preserved':sum(not r['description'].strip() for r in result),
             'risk_counts':dict(Counter(r['risk_label'] for r in result)),
             'ai_label_conflicts_preserved':label_conflicts,'stored_risk_rule_disagreements':risk_conflicts}
    return result,summary,audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--preview',type=int,default=3)
    p.add_argument('--output',type=Path,help='Optional: explicitly save CSV and provenance JSON; otherwise terminal only.')
    a=p.parse_args()
    if a.preview<0:p.error('--preview must be nonnegative')
    if a.output:
        audit_path=a.output.with_suffix('.synthetic_provenance.json')
        if a.output.resolve()==audit_path.resolve():p.error('Use a .csv output filename')
        for path in [a.output,audit_path]:
            if path.exists() or path.resolve() in {a.input.resolve(),a.reference.resolve()}:p.error('Refusing to overwrite '+str(path))
    fields,rows=read(a.input);_,reference=read(a.reference)
    output,summary,audit=prepare(rows,reference,a.seed)
    print(json.dumps(summary,indent=2));print('\nPreview:');print(json.dumps(output[:a.preview],indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=[k for k in fields if k!='vendor']);w.writeheader();w.writerows(output)
        provenance={'policy':'Synthetic demonstration governance, not factual vendor policies or company-specific approvals.',
                    'precedence':'Nonblank finalized reference > existing catalog > deterministic generation.',
                    'identity_policy':'Exact normalized product name and endpoint, not row ID alone or broad domain.',
                    'seed':a.seed,'summary':summary,'records':audit,
                    'input_hashes':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.input,a.reference]}}
        audit_path.write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
        print(f'Saved {a.output} and {audit_path}')
    else:print('\nPreview only. No files written. Add --output only when you want to save.')


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,csv.Error) as exc:raise SystemExit(str(exc))
