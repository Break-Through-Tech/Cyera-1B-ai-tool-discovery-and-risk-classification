"""Fill synthetic governance fields without API calls. Python 3.10+, standard library.
Run from project root: python scripts/fill_governance_dataset_b.py
Vendor cells are deliberately blank. Inputs are never modified.
Exactly three CSV exports, each containing only the 14 Dataset B columns.
All governance values are fictional demonstration data, not vendor policy facts.
Complete synthetic governance profiles; risk labels are low, medium, or high.
Risk labels are illustrative rule outputs, not verified real-world assessments.
Use --overwrite to replace this script's prior exports. Default directory is separate
from vendor-enriched outputs so existing research is not overwritten.
"""
import argparse
import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

COLUMNS = ['tool_id','tool_name','domain','description','is_ai_tool','category',
           'vendor','approved_status','enterprise_plan_available','sso_available',
           'data_retention','training_on_customer_data','admin_controls_available','risk_label']
GOV = ['approved_status','enterprise_plan_available','sso_available','data_retention',
       'training_on_customer_data','admin_controls_available']
PROFILES = {
    'strong_controls': dict(enterprise_plan_available='true',sso_available='true',
                           admin_controls_available='true'),
    'mixed_controls': dict(enterprise_plan_available='true',sso_available='false',
                          admin_controls_available='true'),
    'limited_controls': dict(enterprise_plan_available='false',sso_available='false',
                            admin_controls_available='false')
}
# Illustrative policy version 1; AI label and business category are NOT score inputs.
POINTS = {
 'approved_status': {'approved':0,'under_review':1,'unapproved':2},
 'sso_available': {'true':0,'false':1},
 'admin_controls_available': {'true':0,'false':1},
 'data_retention': {'no_retention':0,'30_days':0,'90_days':1,'365_days':2,'indefinite':3},
 'training_on_customer_data': {'true':3,'false':0}
}
def read_csv(path, required):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(f'{path}: missing columns {set(required)-set(reader.fieldnames or [])}')
        rows=[]
        for i,row in enumerate(reader,2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'{path}: malformed CSV at line {i}')
            rows.append({k:v.strip() for k,v in row.items()})
        return rows

def write_csv(path, rows, columns):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def rng_for(seed, tool_id):
    return random.Random(int(hashlib.sha256(f'{seed}:{tool_id}'.encode()).hexdigest(),16))

def generate(seed, tool_id):
    rng=rng_for(seed,tool_id)
    profile=rng.choices(list(PROFILES),weights=[35,30,25])[0]
    values=dict(PROFILES[profile])
    values['approved_status']=rng.choices(['approved','unapproved','under_review'],weights=[50,30,20])[0]
    # Approval varies independently: approval does not guarantee strong controls.
    values['data_retention']=rng.choice(['no_retention','30_days','90_days','365_days','indefinite'])
    values['training_on_customer_data']=rng.choices(['true','false'],weights=[25,75])[0]
    return values

def assess(row):
    score=sum(weights[row[field]] for field,weights in POINTS.items())
    return 'low' if score<=2 else 'medium' if score<=5 else 'high'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ai',type=Path,default=Path('data/processed/dataset_b/sample/accepted_dataset_b.csv'))
    p.add_argument('--non-ai',type=Path,default=Path('data/processed/non_ai/accepted_non_ai_dataset_b.csv'))
    p.add_argument('--out-dir',type=Path,default=Path('data/processed/governance_only'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    rows=[];ids=set()
    for path,expected in [(a.ai,'true'),(a.non_ai,'false')]:
        for source in read_csv(path,COLUMNS):
            r={k:source[k] for k in COLUMNS}
            tid=r['tool_id']
            if not tid or tid in ids: p.error(f'Missing/duplicate tool_id: {tid}')
            ids.add(tid)
            if r['is_ai_tool'].lower()!=expected: p.error(f'{tid}: unexpected AI label')
            if any(r[k] for k in GOV+['risk_label']):
                p.error(f'{tid}: use the original accepted files with blank governance fields')
            if r['vendor'] and r['vendor'].lower()!='unknown':
                p.error(f'{tid}: vendor already populated; use original accepted inputs to avoid losing it')
            r['is_ai_tool']=expected
            r['domain']=r['domain'].lower().removeprefix('www.')
            if not r['domain'] or '/' in r['domain']: p.error(f'{tid}: expected hostname-only domain')
            r['vendor']=''
            rows.append(r)
    if not rows:p.error('No input records')
    paths=[a.out_dir/n for n in ['ai_tools_filled.csv','non_ai_tools_filled.csv','dataset_b_combined.csv']]
    if any(x.resolve() in {a.ai.resolve(),a.non_ai.resolve()} for x in paths):
        p.error('Output cannot overwrite an input')
    if not a.overwrite and any(x.exists() for x in paths):
        p.error('Outputs exist; use --overwrite or another --out-dir')
    for r in rows:
        r.update(generate(a.seed,r['tool_id']))
        r['risk_label']=assess(r)
    groups=[[r for r in rows if r['is_ai_tool']=='true'],
            [r for r in rows if r['is_ai_tool']=='false'],rows]
    a.out_dir.mkdir(parents=True,exist_ok=True)
    for path,group in zip(paths,groups):
        write_csv(path.with_suffix('.tmp'),group,COLUMNS)
    for path in paths:path.with_suffix('.tmp').replace(path)
    for path,group in zip(paths,groups):print(f'{path}: {len(group)} rows')
    print('Risk counts:',dict(Counter(r['risk_label'] for r in rows)))
    print('Vendor cells blank. All governance attributes are synthetic; no API calls.')

if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError) as exc:raise SystemExit(str(exc))
