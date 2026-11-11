"""Combine Sadit's ingestion/catalog output with optional classification and risk."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path

from .catalog import discover_known, identity_endpoint
from governance import risk

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT/'data/raw/kaggle.csv'
KNOWN = ROOT/'data/processed/final/dataset_b_final_kaggle_categories.csv'


def usage_summary(app, events):
    """Use accepted event membership exactly; missing observations stay missing."""
    uploads=[];downloads=[];times=[];invalid_times=0
    for row in events:
        for field,values in [('bytes_uploaded',uploads),('bytes_downloaded',downloads)]:
            value=row.get(field,'').strip()
            if re.fullmatch(r'\d+',value):values.append(int(value))
        try:
            value=datetime.fromisoformat(row.get('timestamp','').replace('Z','+00:00'))
            if value.tzinfo is None:value=value.replace(tzinfo=timezone.utc)
            times.append(value.astimezone(timezone.utc).isoformat())
        except ValueError:invalid_times+=1
    return dict(application_id=app['application_id'],application_key=app['observed_endpoint'],
        application_names=json.loads(app['name_variants']),unique_users=app['unique_employee_count'],
        departments=json.loads(app['departments']),events=app['event_count'],
        first_seen=min(times) if times else None,last_seen=max(times) if times else None,
        invalid_or_missing_timestamps=invalid_times,
        upload_observations=len(uploads),download_observations=len(downloads),
        total_bytes_uploaded=sum(uploads) if uploads else None,
        total_bytes_downloaded=sum(downloads) if downloads else None,
        mean_bytes_uploaded=sum(uploads)/len(uploads) if uploads else None,
        max_bytes_uploaded=max(uploads) if uploads else None,
        mean_bytes_downloaded=sum(downloads)/len(downloads) if downloads else None)


def run(company,activity,catalog=CATALOG,descriptions=None,model=None,scenarios=None,known_tools=KNOWN,catalog_context=None):
    if not company.strip():raise ValueError('Company is required')
    discovery=discover_known(activity,catalog,known_tools,catalog_context)
    ingestion=discovery['ingestion']
    # source_row is a data-record index, not an event ID: repeats remain separate.
    rows=risk.read_csv(activity);events=defaultdict(list)
    for mapping in ingestion['event_map']:
        events[mapping['application_id']].append(rows[mapping['source_row']-1])
    descriptions_by_endpoint={}
    for row in risk.read_csv(descriptions) if descriptions else []:
        _,authority,path=identity_endpoint(row['domain']);key=authority+path
        if key in descriptions_by_endpoint:raise ValueError('Duplicate description endpoint: '+key)
        descriptions_by_endpoint[key]=row['description'].strip()
    endpoints=Counter(identity_endpoint(a['observed_endpoint'])[1:3] for a in ingestion['applications'])
    overlays={}
    if scenarios:
        fixture=json.loads(Path(scenarios).read_text())
        if fixture.get('synthetic') is not True:raise ValueError('Scenario must explicitly be synthetic')
        overlays=fixture['companies'][company]
        for key,values in overlays.items():
            if set(values)-set(risk.FIELDS):raise ValueError('Scenario may override governance fields only')
    policy=json.loads(risk.POLICY.read_text());risk.validate_policy(policy)
    matches={m['application_id']:m for m in discovery['matches']}
    results=[]
    for app in ingestion['applications']:
        match=matches[app['application_id']]
        usage=usage_summary(app,events[app['application_id']])
        _,authority,path=identity_endpoint(app['observed_endpoint']);endpoint=authority+path
        # A policy/description endpoint must identify one observed product scope.
        unique=endpoints[(authority,path)]==1
        metadata=overlays.get(endpoint,{}) if unique else {}
        result=risk.assess(usage,metadata,policy,'synthetic_company_scenario' if metadata else 'unknown')
        detection=dict(is_ai_tool=None if match['is_ai_tool'] is None else int(match['is_ai_tool']),
            method=match['match_method'] or match['status'],catalog_status=match['status'],
            model_ai_score=None,reason=match['reason'],candidates=match['candidates'],needs_review=True)
        text=descriptions_by_endpoint.get(endpoint,'') if unique else ''
        # Ambiguous matches need identity resolution, not a classifier override.
        if match['status']=='unknown' and text and model:
            from modeling.train_corrected_binary import predict
            detection.update(predict(text,model),method='description_model')
        result.update(company=company,detection=detection,
            category=match['category'],category_source='catalog_assignment' if match['category'] else 'unresolved',
            observed_identity=app,needs_review=True)
        gaps=[]
        if app['events_missing_employee_id']:gaps.append('employee_id')
        if app['events_missing_department']:gaps.append('department')
        if usage['invalid_or_missing_timestamps']:gaps.append('timestamp')
        result['unknown_fields']+=gaps
        if gaps:result['assessment_status']='incomplete'
        results.append(result)
    results.sort(key=lambda r:(-r['risk_score'],r['application_id']))
    labels=Counter('ai' if a['detection']['is_ai_tool']==1 else 'conventional' if a['detection']['is_ai_tool']==0 else 'unknown' for a in results)
    paths={'activity':activity,'ai_catalog':catalog,'known_tools':known_tools,'policy':risk.POLICY}
    if descriptions:paths['descriptions']=descriptions
    if scenarios:paths['scenarios']=scenarios
    if catalog_context:paths['catalog_context']=catalog_context
    ids=[row.get('event_id','') for row in rows]
    report=dict(company=company,synthetic_activity=True,synthetic_scenario_applied=bool(scenarios),
        application_count=len(results),applications=results,detection_counts={k:labels[k] for k in ['ai','conventional','unknown']},
        review_queue=[a['application_id'] for a in results],
        ingestion=ingestion,catalog_warnings=discovery['catalog_warnings'],
        repeated_nonempty_event_ids=sum(n-1 for key,n in Counter(ids).items() if key and n>1),
        inputs={k:{'file':Path(p).name,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for k,p in paths.items()},
        limitations=['Sadit ingestion counts every accepted record, including repeated event IDs; rejected rows are reported separately.',
            'Application identity includes name and endpoint; it is not simply a hostname count.',
            'Catalog categories and labels are supplied assignments, not independently verified truth.',
            'Unknown or ambiguous applications remain visible; models require supplied descriptions.',
            'Governance uses accepted activity and explicit company synthetic overlays only; no generated catalog controls are imported.',
            'Low priority with incomplete evidence is not assurance of safety. No web/API calls occur.',
            'The saved binary model is unchanged; the older 12-category models are not used with the new 14-category catalog.'])
    if model:
        from modeling.train_corrected_binary import verified
        report['model_sha256']=verified(model)['model_sha256']
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--company',required=True);p.add_argument('--activity',type=Path,required=True)
    p.add_argument('--catalog',type=Path,default=CATALOG);p.add_argument('--known-tools',type=Path,default=KNOWN)
    p.add_argument('--catalog-context',type=Path);p.add_argument('--descriptions',type=Path)
    p.add_argument('--model',type=Path);p.add_argument('--scenarios',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Output exists; choose a new file')
    result=run(a.company,a.activity,a.catalog,a.descriptions,a.model,a.scenarios,a.known_tools,a.catalog_context)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'applications':result['application_count'],'detection_counts':result['detection_counts'],
                      'ingestion':result['ingestion']['summary']},indent=2))


if __name__=='__main__':main()
