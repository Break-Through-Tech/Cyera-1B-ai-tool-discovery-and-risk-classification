"""Offline dashboard adapter using the operational ingestion, catalog and risk code."""
import copy
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from . import categorize as cat
from .ingest import extract_applications
from .run_pipeline import index_discovery,merge_records
from governance import risk,evidence

ROOT=Path(__file__).resolve().parents[1]
SAMPLE=ROOT/'data/processed/pipeline_runs/novatech_component4_v2.json'
ACTIVITY=ROOT/'data/raw/novatech_usage.csv'
CATALOG=ROOT/'data/processed/kaggle_dataset_b/kaggle_dataset_b_filled.csv'
KNOWN=ROOT/'data/processed/final/dataset_b_final_kaggle_categories.csv'
SCENARIOS=ROOT/'governance/synthetic_scenarios.json'
REQUIRED=('application_name','domain')


def digest(data):return hashlib.sha256(data).hexdigest()
def load_report(content):
    report=json.loads(content.decode('utf-8-sig'))
    if not isinstance(report.get('results'),list) or not report['results']:raise ValueError('Expected a nonempty Component 4 report')
    ids=[r.get('application_id') for r in report['results']]
    if not all(ids) or len(set(ids))!=len(ids):raise ValueError('Missing or duplicate application IDs')
    companies={r.get('governance',{}).get('company') for r in report['results']}
    if len(companies)!=1 or not next(iter(companies)):raise ValueError('Saved report must have one explicit company')
    for record in report['results']:
        if 'observed_identity' not in record or 'governance' not in record:raise ValueError('Report is missing usage/governance details')
    report=copy.deepcopy(report)
    report['dashboard_context']={'mode':'saved_report','company':next(iter(companies)),
        'source_sha256':digest(content),'online_calls_made':0,
        'classification_note':'Saved catalog/research decisions; no classification was rerun. Jev is not connected.'}
    return report


def analyze(content,company,reference=None,demo_controls=False,metadata=None):
    if not company.strip():raise ValueError('Enter a fictional company name')
    if len(content)>25*1024*1024:raise ValueError('Activity upload exceeds the 25 MB prototype limit')
    reader=csv.DictReader(io.StringIO(content.decode('utf-8-sig')))
    headers=reader.fieldnames or []
    if len(headers)!=len(set(headers)) or not set(REQUIRED)<=set(headers):raise ValueError('Activity CSV requires unique application_name and domain columns')
    rows=list(reader)
    if not rows or len(rows)>100000:raise ValueError('Upload between 1 and 100,000 activity rows')
    prep=cat.helpers.import_file(ROOT/'preprocessing_scripts/prepare_category_data.py','dashboard_purpose_contract')
    catalog=cat.Catalog([CATALOG,KNOWN],prep)
    prior=index_discovery(reference,prep) if reference else {}
    with tempfile.TemporaryDirectory(prefix='cyera_dashboard_') as directory:
        folder=Path(directory);activity=folder/'activity.csv';activity.write_bytes(content)
        ingestion=extract_applications(activity)
        if not ingestion['applications']:raise ValueError('No valid application identities; check the CSV domains and names')
        engine=SimpleNamespace(catalog=catalog,prep=prep)
        records=merge_records(ingestion['applications'],prior,engine)
        reused=0
        for record in records:
            try:key=cat.identity(record['application_name'],record['endpoint'],prep)
            except ValueError:key=None
            old=prior.get(key) if key else None
            record.pop('governance',None)  # Never transfer cached usage/company controls.
            if old and record['classification_status']=='classified':
                reused+=1;record['dashboard_detection_source']='saved_product_decision'
            else:record['dashboard_detection_source']=record.get('decision_source') or 'unresolved'
            if record.get('catalog_detection_conflict'):
                record.update(is_ai_tool=None,classification_status='unresolved',identity_supported=False,label='unknown')
                record['categorization']={'category':None,'category_status':'pending_detection_conflict','category_source':None}
                continue
            gate=cat.gate(record)
            if gate:
                record['categorization']={'category':None,'category_status':gate,'category_source':None}
            elif old and old.get('categorization'):
                record['categorization']=copy.deepcopy(old['categorization'])
                record['categorization']['reused_saved_prediction']=True
            else:
                candidates=catalog.lookup(record['application_name'],record['endpoint'])
                categories={r['category'] for r in candidates if r['category'] in prep.TAXONOMY}
                category=next(iter(categories)) if len(categories)==1 else None
                record['categorization']={'category':category,'category_status':'assigned' if category else 'pending_product_evidence',
                                           'category_source':'catalog' if category else None}
        controls=[]
        if demo_controls:controls=[CATALOG,KNOWN]
        if metadata:
            path=folder/'uploaded_controls.csv';path.write_bytes(metadata);controls.append(path)
        scenario=SCENARIOS if demo_controls and company in json.loads(SCENARIOS.read_text())['companies'] else None
        supplied,warnings=evidence.assemble(records,company,controls,'synthetic',company if demo_controls else None,scenario)
        assessed=risk.assess_records(records,ingestion,activity,json.loads(risk.POLICY.read_text()),company,supplied,'synthetic')
        # Store only daily totals; employee IDs are not exported by this adapter.
        accepted={m['source_row'] for m in ingestion['event_map']};daily={};invalid_times=0
        for number,row in enumerate(rows,1):
            if number not in accepted:continue
            try:
                when=datetime.fromisoformat((row.get('timestamp') or '').replace('Z','+00:00'))
                if when.tzinfo is None:when=when.replace(tzinfo=timezone.utc)
                day=when.astimezone(timezone.utc).date().isoformat();daily[day]=daily.get(day,0)+1
            except ValueError:invalid_times+=1
        ingestion['summary']['input_file']='uploaded_activity.csv'
        return dict(version='dashboard-offline-v1',results=assessed,ingestion=ingestion,
            governance_counts={s:sum(r['governance']['assessment_status']==s for r in assessed) for s in ('complete','incomplete')},
            dashboard_context=dict(mode='fresh_offline_analysis',company=company,activity_sha256=digest(content),
                reference_report_sha256=digest(json.dumps(reference,sort_keys=True).encode()) if reference else None,
                demo_controls=demo_controls,uploaded_controls=bool(metadata),cached_decisions_reused=reused,online_calls_made=0,
                classification_note='Exact catalog matches and saved product decisions reused; no new Jev/Gemini classification. Unknown products remain unresolved.',
                daily_events=dict(sorted(daily.items())),events_missing_or_invalid_timestamp=invalid_times),
            governance_evidence_report={'warnings':warnings},
            limitations=['Fictional usage and explicitly synthetic supplied controls; not verified vendor claims.',
                'Categories can be provisional; cached product decisions are not independent verification.',
                'Distinct user counts belong to each application and must not be summed across applications.',
                'No online model or web calls are performed. Jev integration remains pending.'])


def flattened(report):
    result=[]
    for r in report['results']:
        app=r['observed_identity'];g=r['governance'];category=r.get('categorization',{})
        label=r.get('is_ai_tool') if r.get('classification_status')=='classified' and not r.get('catalog_detection_conflict') else None
        departments=app.get('departments',[])
        if isinstance(departments,str):departments=json.loads(departments)
        approval=g.get('evidence',{}).get('approved_status',{})
        status=approval.get('value') if approval.get('company')==g.get('company') else None
        result.append(dict(application_id=r['application_id'],application=r['application_name'],endpoint=r.get('endpoint',app.get('observed_endpoint')),
            ai_status='AI' if label is True else 'Conventional' if label is False else 'Unknown',
            category=category.get('category') or 'Not assigned',category_source=category.get('category_source') or 'Unknown',
            category_status=category.get('category_status') or 'Unknown',
            users=app.get('unique_employee_count'),events=app.get('event_count'),departments=departments,
            approval=status or 'Unknown',risk_level=g.get('risk_level','unknown'),risk_score=g.get('risk_score'),
            evidence_status=g.get('assessment_status','unknown'),needs_review=bool(g.get('needs_review') or label is None or category.get('category_source')=='model'),
            classification_source=r.get('dashboard_detection_source') or r.get('decision_source') or 'Unknown'))
    return sorted(result,key=lambda r:(r['risk_score'] is None,-(r['risk_score'] or 0),r['application']))


def csv_export(rows):
    if not rows:return b''
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
    for row in rows:
        safe={}
        for key,value in row.items():
            if isinstance(value,list):value=', '.join(value)
            if isinstance(value,str) and value.startswith(('=','+','-','@')):value="'"+value
            safe[key]=value
        writer.writerow(safe)
    return stream.getvalue().encode('utf-8-sig')
