"""Deterministic governance rules with explicit evidence, missingness and company scope."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
POLICY = Path(__file__).with_name('policy.json')
FIELDS = ('approved_status','sso_available','data_retention','training_on_customer_data',
          'admin_controls_available','enterprise_plan_available')


def identity(value):
    """Exact hostname plus application path; never collapse GitHub/Copilot."""
    p=urlsplit(value.strip() if '://' in value else 'https://'+value.strip())
    if not p.hostname or p.username or p.password or p.port:
        raise ValueError('Expected application hostname/path without credentials or ports')
    return p.hostname.lower().removeprefix('www.')+p.path.rstrip('/').lower()


def boolean(value):
    if value in (True,False):
        # Avoid Python treating numeric zero/one as explicit policy evidence.
        if isinstance(value,bool):return value
    if isinstance(value,str):
        if value.lower().strip() in ('true','yes'):return True
        if value.lower().strip() in ('false','no'):return False
    return None


def retention(value):
    if not isinstance(value,str):return None
    value=value.strip().lower()
    if value in ('no_retention','zero_retention'):return 0
    if value in ('indefinite','indefinitely'):return math.inf
    match=re.fullmatch(r'(\d+)_days',value)
    return int(match[1]) if match else None


def normalize_metadata(raw):
    result={}
    status=str(raw.get('approved_status','')).strip().lower()
    result['approved_status']=status if status in ('approved','unapproved') else None
    result['data_retention']=raw.get('data_retention') if retention(raw.get('data_retention')) is not None else None
    for field in FIELDS:
        if field not in result:result[field]=boolean(raw.get(field))
    return result


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def aggregate(rows):
    groups={}; seen=set()
    for row in rows:
        if not row.get('event_id') or row['event_id'] in seen:raise ValueError('Missing or duplicate event_id')
        seen.add(row['event_id'])
        for key in ('timestamp','employee_id','department','application_name','domain'):
            if not str(row.get(key,'')).strip():raise ValueError('Missing activity field: '+key)
        when=datetime.fromisoformat(row['timestamp'].replace('Z','+00:00'))
        # Existing synthetic CSVs use naive times; interpret them as UTC consistently.
        when=when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)
        key=identity(row['domain'])
        g=groups.setdefault(key,dict(application_key=key,names=set(),users=set(),departments=set(),events=0,uploads=[],downloads=[],timestamps=[]))
        g['names'].add(row['application_name']);g['users'].add(row['employee_id']);g['departments'].add(row['department'])
        g['events']+=1;g['timestamps'].append(when.isoformat())
        for source,target in [('bytes_uploaded','uploads'),('bytes_downloaded','downloads')]:
            value=row.get(source,'')
            if value is None or str(value).strip()=='':continue
            if not re.fullmatch(r'\d+',str(value)):raise ValueError('Expected nonnegative integer: '+source)
            g[target].append(int(value))
    result=[]
    for key,g in sorted(groups.items()):
        result.append(dict(application_key=key,application_names=sorted(g['names']),unique_users=len(g['users']),
            departments=sorted(g['departments']),events=g['events'],first_seen=min(g['timestamps']),last_seen=max(g['timestamps']),
            upload_observations=len(g['uploads']),download_observations=len(g['downloads']),
            total_bytes_uploaded=sum(g['uploads']) if g['uploads'] else None,
            total_bytes_downloaded=sum(g['downloads']) if g['downloads'] else None,
            mean_bytes_uploaded=sum(g['uploads'])/len(g['uploads']) if g['uploads'] else None,
            max_bytes_uploaded=max(g['uploads']) if g['uploads'] else None,
            mean_bytes_downloaded=sum(g['downloads'])/len(g['downloads']) if g['downloads'] else None))
    return result


def validate_policy(p):
    required={'unapproved','no_sso','customer_data_training','long_retention','no_admin_controls','no_enterprise_plan','sensitive_department','large_upload','large_download','broad_adoption'}
    if set(p['points'])!=required:raise ValueError('Policy rule set mismatch')
    numbers=[*p['points'].values(),p['medium_score'],p['high_score'],p['large_mean_upload_bytes'],p['large_single_upload_bytes'],p['large_mean_download_bytes'],p['broad_adoption_users'],p['long_retention_days']]
    if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<0 for v in numbers):raise ValueError('Invalid policy number')
    if not 0<p['medium_score']<p['high_score']<=100:raise ValueError('Invalid risk band thresholds')


def assess(usage,raw_metadata,policy,source='unknown'):
    validate_policy(policy)
    metadata=normalize_metadata(raw_metadata)
    reasons=[]
    def add(rule,evidence,explanation):
        reasons.append(dict(rule=rule,points=policy['points'][rule],evidence=evidence,explanation=explanation))
    if metadata['approved_status']=='unapproved':add('unapproved',{'approved_status':'unapproved'},'Explicitly unapproved in this company scenario.')
    for field,rule,description in [('sso_available','no_sso','SSO unavailable in supplied metadata.'),('admin_controls_available','no_admin_controls','Administrative controls unavailable in supplied metadata.'),('enterprise_plan_available','no_enterprise_plan','Enterprise plan unavailable in supplied metadata.')]:
        if metadata[field] is False:add(rule,{field:False},description)
    if metadata['training_on_customer_data'] is True:add('customer_data_training',{'training_on_customer_data':True},'Supplied policy indicates customer content may be used for model training.')
    days=retention(metadata['data_retention'])
    if days is not None and days>policy['long_retention_days']:add('long_retention',{'data_retention':metadata['data_retention'],'threshold_days':policy['long_retention_days']},'Retention exceeds the configured review threshold.')
    sensitive=sorted(set(d.casefold() for d in usage['departments']) & {d.casefold() for d in policy['sensitive_departments']})
    if sensitive:add('sensitive_department',{'departments':sensitive},'Activity comes from a potentially sensitive business function; content sensitivity is not observed.')
    mean,maximum=usage['mean_bytes_uploaded'],usage['max_bytes_uploaded']
    if (mean is not None and mean>=policy['large_mean_upload_bytes']) or (maximum is not None and maximum>=policy['large_single_upload_bytes']):
        add('large_upload',{'mean_bytes_uploaded':mean,'max_bytes_uploaded':maximum,'mean_threshold':policy['large_mean_upload_bytes'],'single_threshold':policy['large_single_upload_bytes']},'Upload volume exceeds a configured threshold; this does not establish data leakage.')
    download=usage['mean_bytes_downloaded']
    if download is not None and download>=policy['large_mean_download_bytes']:add('large_download',{'mean_bytes_downloaded':download,'threshold':policy['large_mean_download_bytes']},'Download volume exceeds the configured review threshold.')
    if usage['unique_users']>=policy['broad_adoption_users']:add('broad_adoption',{'unique_users':usage['unique_users'],'threshold':policy['broad_adoption_users']},'Broad adoption increases the scope of governance review.')
    score=min(100,sum(r['points'] for r in reasons))
    band='high' if score>=policy['high_score'] else 'medium' if score>=policy['medium_score'] else 'low'
    unknown=[field for field in FIELDS if metadata[field] is None]
    gaps=list(unknown)
    for field,count in [('bytes_uploaded',usage['upload_observations']),('bytes_downloaded',usage['download_observations'])]:
        if count<usage['events']:gaps.append(field)
    return dict(**usage,risk_score=score,uncapped_risk_score=sum(r['points'] for r in reasons),risk_level=band,assessment_status='incomplete' if gaps else 'complete',
        needs_review=bool(gaps or score>=policy['medium_score']),unknown_fields=gaps,metadata=metadata,metadata_source=source,
        policy_version=policy['version'],reasons=reasons,
        caveat='Observed-evidence governance priority, not breach probability. Low with incomplete evidence is not an assurance of safety.')


def metadata_index(rows):
    index={}
    for row in rows:
        if not row.get('domain'):continue
        key=identity(row['domain'])
        # Duplicate catalog domains are ambiguous: never choose a policy arbitrarily.
        index.setdefault(key,[]).append(row)
    return {key:values[0] for key,values in index.items() if len(values)==1}


def run(company,activity,metadata_path,policy_path,scenarios=None):
    policy=json.loads(Path(policy_path).read_text());validate_policy(policy)
    catalog=metadata_index(read_csv(metadata_path))
    overrides={}
    if scenarios:
        fixture=json.loads(Path(scenarios).read_text())
        if fixture.get('synthetic') is not True:raise ValueError('Scenario overlay must explicitly be synthetic')
        if company not in fixture['companies']:raise ValueError('No scenario for requested company')
        overrides=fixture['companies'][company]
        for key,values in overrides.items():
            if identity(key)!=key:raise ValueError('Scenario keys must be canonical application identities')
            if set(values)-set(FIELDS):raise ValueError('Scenario may override governance fields only')
    results=[]
    for usage in aggregate(read_csv(activity)):
        key=usage['application_key'];raw=dict(catalog.get(key,{}));source='dataset_b' if raw else 'unknown'
        # Dataset B approval is not company-scoped. Only explicit company overlay may supply it.
        raw.pop('approved_status',None)
        if key in overrides:raw.update(overrides[key]);source='synthetic_company_scenario'
        result=assess(usage,raw,policy,source)
        result['metadata_match']=key in catalog
        result['company']=company
        results.append(result)
    results.sort(key=lambda r:(-r['risk_score'],r['application_key']))
    paths={'activity':activity,'metadata':metadata_path,'policy':policy_path}
    if scenarios:paths['synthetic_scenarios']=scenarios
    return dict(company=company,synthetic_activity=True,synthetic_scenario_applied=bool(scenarios),
        inputs={name:{'file':Path(path).name,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()} for name,path in paths.items()},
        policy_version=policy['version'],application_count=len(results),applications=results,
        limitations=['Catalog policies are mostly missing; missing values add no points and remain unknown.',
                     'Synthetic scenarios do not assert real vendor policies or enterprise approval.',
                     'All observed applications are assessed; AI identification remains a separate upstream task.',
                     'Thresholds are demonstration policy choices, not learned or validated security cutoffs.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--company',required=True)
    p.add_argument('--activity',type=Path,required=True)
    p.add_argument('--metadata',type=Path,default=ROOT/'data/processed/dataset_b/dataset_b_master.csv')
    p.add_argument('--policy',type=Path,default=POLICY)
    p.add_argument('--scenarios',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Output exists; choose a new file')
    result=run(a.company,a.activity,a.metadata,a.policy,a.scenarios)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'company':a.company,'applications':result['application_count'],'highest_priority':[{k:r[k] for k in ('application_key','risk_score','risk_level','assessment_status')} for r in result['applications'][:5]]},indent=2))


if __name__=='__main__':main()
