"""Match enterprise applications to known AI/non-AI records. Terminal output only.

Place beside ingest.py in ai_discovery/. Standard library; no API/network calls.
python ai_discovery/catalog.py --activity data/raw/novatech_usage.csv --ai-catalog data/raw/kaggle.csv --known-tools data/processed/final/dataset_b_final_kaggle_categories.csv

Raw Kaggle is preferred here because Website retains product paths that Dataset B's
hostname-only domain may omit. Both input formats are supported. Optional
--catalog-context supplies tool_id, website_url_clean from the preparation script.

Kaggle entries are AI by project assumption. Finalized labels override that
assumption only for the same normalized name + host + product path identity.
An optional full correction export applies checked source-record/product-scope
resolutions to both inputs; unresolved corrected labels remain ambiguous.
Conflicting finalized labels remain ambiguous. Absence never means non-AI.
No fuzzy names, inferred domain aliases, or parent-domain collapsing.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .ingest import extract_applications, clean_name, normalize_endpoint
except ImportError:
    from ingest import extract_applications, clean_name, normalize_endpoint

# A host shared by unrelated products cannot identify one product by itself.
SHARED_HOSTS = {'github.com','gitlab.com','huggingface.co','apps.apple.com',
                'play.google.com','chromewebstore.google.com','chrome.google.com',
                'gumroad.com','discord.com','discord.gg','youtube.com','youtu.be',
                'notion.site','notion.so','bit.ly','t.co','linktr.ee','awin1.com'}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames or [];rows=list(reader)
    if len(fields)!=len(set(fields)) or any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f'Malformed CSV: {path}')
    return fields,rows


def identity_endpoint(value):
    host,path,query,fragment,endpoint=normalize_endpoint(value)
    # Query/fragment are preserved in ingestion, but not treated as evidence for
    # domain/path matching. Shared-host name/path safeguards apply below.
    authority=endpoint.split('/',1)[0].split('?',1)[0].split('#',1)[0]
    return host,authority,path.rstrip('/')


def load_catalog(ai_catalog, known_tools, catalog_context=None, corrections=None):
    """Return an indexed catalog and input warnings; never writes files."""
    corrected={}
    if corrections:
        _,correction_rows=read_csv(corrections)
        for item in correction_rows:
            tid=item["source_record_id"]
            if tid in corrected:raise ValueError("Duplicate source record in correction export: "+tid)
            corrected[tid]=item
    context={}
    if catalog_context:
        fields,rows=read_csv(catalog_context)
        if not {'tool_id','website_url_clean'}<=set(fields):raise ValueError('Context needs tool_id and website_url_clean')
        for r in rows:
            if r['tool_id'] in context:raise ValueError('Duplicate tool ID in context')
            context[r['tool_id']]=r['website_url_clean']
    records=[];warnings=[]
    for source,path in [('ai_catalog',ai_catalog),('finalized',known_tools)]:
        fields,rows=read_csv(path)
        raw={'Name','Website','Category'}<=set(fields)
        if source=='finalized' and not {'tool_id','tool_name','domain','is_ai_tool','category'}<=set(fields):
            raise ValueError('Finalized input must use Dataset B columns')
        if not raw and not {'tool_id','tool_name','domain','category'}<=set(fields):
            raise ValueError('AI catalog must be raw Kaggle or Dataset B')
        for i,r in enumerate(rows,1):
            name=clean_name(r['Name'] if raw else r['tool_name'])
            tid=f'tool_{i:05d}' if raw else r['tool_id']
            url=r['Website'] if raw else (context.get(tid,r['domain']) if source=='ai_catalog' else r['domain'])
            try:
                host,authority,product_path=identity_endpoint(url)
                if not name:raise ValueError('missing name')
            except ValueError as exc:
                warnings.append({'source':source,'tool_id':tid,'reason':str(exc)});continue
            if source=='ai_catalog':label=True
            else:
                value=r['is_ai_tool'].strip().lower()
                if value not in ('true','false','1','0'):
                    warnings.append({'source':source,'tool_id':tid,'reason':'unresolved label','retained':True})
                    label=None
                else:label=value in ('true','1')
            if tid in corrected:
                correction=corrected[tid]
                correction_host=identity_endpoint(correction['domain'])[0]
                if correction_host!=host or name.casefold() not in {clean_name(correction['tool_name']).casefold(),clean_name(correction['original_tool_name']).casefold()}:
                    raise ValueError('Correction/source identity mismatch: '+tid)
                name=clean_name(correction['tool_name'])
                value=correction['is_ai_tool']
                label={'1':True,'0':False,'':None}[value]
            records.append(dict(tool_id=tid,tool_name=name,name_key=name.casefold(),domain=host,
                                authority=authority,path=product_path,is_ai_tool=label,
                                category=r['Category'].strip() if raw else r['category'].strip(),source=source,
                                label_method=corrected[tid].get('label_method','corrected_export') if tid in corrected else source,
                                evidence_url=corrected[tid].get('review_evidence_url','') if tid in corrected else ''))
    # Only an exact identity can override a source assumption, not a whole host.
    identities=defaultdict(list)
    for r in records:identities[(r['name_key'],r['authority'],r['path'])].append(r)
    effective=[]
    for group in identities.values():
        final=[r for r in group if r['source']=='finalized']
        effective.extend(final if final else group)
    by_host=defaultdict(list);by_name=defaultdict(list)
    for r in effective:
        by_host[r['authority']].append(r);by_name[r['name_key']].append(r)
    return {'by_host':dict(by_host),'by_name':dict(by_name),'warnings':warnings,
            'loaded_records':len(records),'effective_records':len(effective)}


def match_applications(applications,catalog):
    """Return one result per input application, including all ambiguous candidates."""
    results=[]
    for app in applications:
        name=app['normalized_name'];host,authority,path=identity_endpoint(app['observed_endpoint'])
        candidates=catalog['by_host'].get(authority,[])
        selected=[];method='';reason='no_matching_identity'
        # First match the most-specific stored path prefix, with segment boundaries.
        scoped=[r for r in candidates if r['path'] and (path==r['path'] or path.startswith(r['path']+'/'))]
        if scoped:
            longest=max(len(r['path']) for r in scoped)
            selected=[r for r in scoped if len(r['path'])==longest]
            exact_names=[r for r in selected if r['name_key']==name]
            if exact_names:selected=exact_names
            method='product_path_and_name' if exact_names else 'product_path'
        else:
            roots=[r for r in candidates if not r['path']]
            same=[r for r in roots if r['name_key']==name]
            if same:
                selected=same;method='name_and_host'
            elif not path and host not in SHARED_HOSTS and len({r['name_key'] for r in roots})==1:
                selected=roots;method='unique_product_host'
            elif candidates:reason='host_present_but_product_not_resolved'
            elif name in catalog['by_name']:reason='name_present_but_domain_differs'
        labels={r['is_ai_tool'] for r in selected}
        identities={(r['name_key'],r['path']) for r in selected}
        result=dict(application_id=app['application_id'],application_name=app['application_name'],
                    observed_endpoint=app['observed_endpoint'],event_count=app['event_count'],
                    status='unknown',is_ai_tool=None,category=None,match_method=method,reason=reason,candidates=[])
        if selected:
            if None in labels:
                result.update(status='ambiguous',reason='unresolved_corrected_label')
            elif len(labels)!=1 or len(identities)!=1:
                result.update(status='ambiguous',reason='conflicting_labels_or_multiple_products')
            else:
                ai=next(iter(labels));cats={r['category'] for r in selected if r['category']}
                result.update(status='known_ai' if ai else 'known_non_ai',is_ai_tool=ai,
                              category=next(iter(cats)) if ai and len(cats)==1 else None,
                              reason='matched_catalog_record' if not ai or len(cats)==1 else 'label_matched_category_unresolved')
            result['candidates']=[{**{k:r[k] for k in ['tool_id','tool_name','domain','path','is_ai_tool','category','source']},
                'label_method':r.get('label_method',r['source']),'evidence_url':r.get('evidence_url','')} for r in selected]
        results.append(result)
    return results


def discover_known(activity_path,ai_catalog,known_tools,catalog_context=None,corrections=None):
    """Import this from the future pipeline; all outputs remain in memory."""
    ingestion=extract_applications(activity_path)
    catalog=load_catalog(ai_catalog,known_tools,catalog_context,corrections)
    matches=match_applications(ingestion['applications'],catalog)
    return {'ingestion':ingestion,'matches':matches,
            'unresolved':[r for r in matches if r['status'] in ('unknown','ambiguous')],
            'catalog_warnings':catalog['warnings'],
            'summary':{'applications':len(matches),'status_counts':dict(Counter(r['status'] for r in matches)),
                       'catalog_records_loaded':catalog['loaded_records'],'catalog_rows_skipped':sum(not w.get('retained',False) for w in catalog['warnings']),
                       'catalog_unresolved_labels_retained':sum(w.get('retained',False) for w in catalog['warnings']),
                       'rejected_activity_events':ingestion['summary']['rejected_events']}}


def print_results(result):
    print(json.dumps(result['summary'],indent=2))
    for i,r in enumerate(result['matches'],1):
        print(f"\n{i}. {r['application_name']} ({r['observed_endpoint']}) -> {r['status']}")
        print(f"   Method: {r['match_method'] or 'none'}; reason: {r['reason']}")
        if r['category']:print('   Category:',r['category'])
        if r['candidates']:
            print('   Catalog:', '; '.join(f"{c['tool_name']} [{c['source']}:{c['tool_id']}]" for c in r['candidates']))
    if result['catalog_warnings']:
        print('\nSkipped catalog rows (first 5):',json.dumps(result['catalog_warnings'][:5]))
    print('\nUnknown and ambiguous applications remain unresolved; no model or web lookup was run.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--activity',type=Path,required=True)
    p.add_argument('--ai-catalog',type=Path,required=True)
    p.add_argument('--known-tools',type=Path,required=True)
    p.add_argument('--catalog-context',type=Path)
    p.add_argument('--corrections',type=Path)
    a=p.parse_args()
    try:print_results(discover_known(a.activity,a.ai_catalog,a.known_tools,a.catalog_context,a.corrections))
    except (ValueError,OSError,csv.Error) as exc:p.exit(1,f'Error: {exc}\n')


if __name__=='__main__':main()
