"""Connect saved product evidence, catalog Jev predictions and fresh company activity.

All functions are offline. Product knowledge can be reused; company approvals and
usage are always rebuilt from the selected activity and explicit scenario controls.
"""
from __future__ import annotations
import copy
import csv
import io
import json
from pathlib import Path
from functools import lru_cache
import tempfile

import pandas as pd
from ai_discovery.dashboard_data import ROOT,analyze,csv_export,cat
from .data import CompanyRun,load_config,build_apps,attach_events,_path

CATALOG_RESULTS=ROOT/'data/processed/jev_catalog/all_classified.csv'
SAMPLE_RESULTS=ROOT/'data/processed/pipeline_runs/novatech_jev_classification.json'


@lru_cache(maxsize=1)
def contract():
    return cat.helpers.import_file(ROOT/'preprocessing_scripts/prepare_category_data.py','unified_dashboard_identity')

@lru_cache(maxsize=65536)
def identity(name,endpoint):
    return cat.identity(name,endpoint,contract())


def product_reference():
    records={}
    # Preserve the team's reviewed categories and saved unseen-product research.
    for entry in load_config():
        report=json.loads((_path(entry['run_dir'])/'categorized_applications.json').read_text())
        for original in report['results']:
            try:key=identity(original['application_name'],original['endpoint'])
            except ValueError:continue
            item=copy.deepcopy(original)
            for field in ('governance','observed_identity','application_id'):item.pop(field,None)
            previous=records.get(key)
            if previous and previous.get('categorization',{}).get('category_source')=='human_review':continue
            records[key]=item
    reviewed={k:copy.deepcopy(r["categorization"]) for k,r in records.items() if r.get("is_ai_tool") is True and r.get("categorization",{}).get("category_source")=="human_review"}
    predictions=[]
    if CATALOG_RESULTS.exists():
        with CATALOG_RESULTS.open(encoding='utf-8-sig',newline='') as file:
            for row in csv.DictReader(file):
                # CSV exports escape formula-like names; recover the actual identity for matching.
                name=row['application_name'];endpoint=row['endpoint']
                if name.startswith("'") and name[1:2] in '=+-@':name=name[1:]
                predictions.append({'application_name':name,'endpoint':endpoint,'label':row['jev_label'],
                    'probabilities':{label:float(row['p_'+label]) for label in ('ai','non_ai','unknown')},
                    'returned_model':row['returned_model'],'input_fingerprint':row['input_fingerprint'],
                    'source_label_disagreement':row['disagrees_with_source']=='True'})
    # The enterprise sample has richer product research; keep its explicit evidence provenance.
    if SAMPLE_RESULTS.exists():predictions.extend(json.loads(SAMPLE_RESULTS.read_text())['results'])
    for prediction in predictions:
        try:key=identity(prediction['application_name'],prediction['endpoint'])
        except ValueError:continue
        old=records.get(key,{})
        label=prediction['label'];ai=None if label=='unknown' else label=='ai'
        item=copy.deepcopy(old)
        if old.get('is_ai_tool')!=ai:item.pop('categorization',None)
        item.update(application_name=prediction['application_name'],endpoint=prediction['endpoint'],
            label=label,is_ai_tool=ai,classification_status='unresolved' if ai is None else 'classified',
            identity_supported=ai is not None,identity_verified=False,decision_source='jev',
            probabilities=prediction['probabilities'],returned_model=prediction.get('returned_model'),
            jev_prediction=label,source_label_disagreement=prediction.get('source_label_disagreement',False),
            reason='Saved Jev primary-purpose prediction; unresolved/conflicting evidence requires review.')
        records[key]=item
    for key,category in reviewed.items():
        if records[key].get('is_ai_tool') is True:records[key]['categorization']=category
    return {'results':list(records.values())}


def reference_for_activity(content,reference):
    """Limit the analysis/reference hash to products in the upload, not the entire catalog."""
    if len(content)>25*1024*1024:raise ValueError('Activity upload exceeds the 25 MB prototype limit')
    reader=csv.DictReader(io.StringIO(content.decode('utf-8-sig')))
    wanted=set()
    for row in reader:
        try:wanted.add(identity(row.get('application_name',''),row.get('domain','')))
        except (ValueError,AttributeError):continue
    return {'results':[r for r in reference['results'] if identity(r['application_name'],r['endpoint']) in wanted]}


def from_report(report,content):
    company=report['dashboard_context']['company']
    activity=pd.read_csv(io.BytesIO(content),dtype=str).fillna('')
    for field in ('employee_id','department','timestamp'):
        if field not in activity:activity[field]=''
    return CompanyRun(company,build_apps(report,company),attach_events(report,activity),report,[])


def uploaded_run(content,company,reference,demo_controls=False,metadata=None):
    scoped=reference_for_activity(content,reference)
    report=analyze(content,company,scoped,demo_controls,metadata)
    return from_report(report,content)


@lru_cache(maxsize=1)
def search_encoder(path):
    from ai_discovery.retrieval import load_encoder
    return load_encoder(Path(path))

def report_answer(report,company,question,encoder_path=None):
    """Local report retrieval and constrained evidence answers; never calls Gemini."""
    from ai_discovery.retrieval import build,EvidenceIndex,DEFAULT_ENCODER
    from ai_discovery.ask import answer
    encoder=Path(encoder_path) if encoder_path else DEFAULT_ENCODER
    if not encoder.exists():raise ValueError('Install the local report-search encoder to enable evidence questions.')
    encoder_model=search_encoder(str(encoder.resolve()))
    with tempfile.TemporaryDirectory(prefix='governance_question_') as directory:
        folder=Path(directory);source=folder/'report.json'
        source.write_text(json.dumps(report,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        build([source],folder/'index',encoder,encoder_model)
        return answer(EvidenceIndex(folder/'index',encoder_model),question,company)
