"""Rebuild vendor enrichment from exact cached records, without API calls."""
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'data/cache/dataset_b_vendor_only'
BASE=ROOT/'data/processed/governance_only/dataset_b_combined.csv'
FINAL=ROOT/'data/processed/final/dataset_b_final.csv'
KEYS=('tool_name','domain','description')

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def key(record):return json.dumps({k:record[k] for k in KEYS},sort_keys=True)

def vendor_index():
    index={};hashes={}
    for path in sorted(CACHE.glob('*.json')):
        x=json.loads(path.read_text());r=x.get('record',{})
        if set(r)!=set(KEYS) or x.get('version')!='vendor-only-1':continue
        expected=hashlib.sha256(json.dumps([x['version'],x.get('model'),r],sort_keys=True).encode()).hexdigest()
        if expected!=path.stem:raise ValueError('Cache fingerprint mismatch: '+path.name)
        status=x.get('status');vendor=x.get('vendor','').strip();source=x.get('source',{})
        if status=='found':
            if not vendor or vendor.upper()=='UNKNOWN' or not source.get('url','').startswith('https://') or not x.get('evidence','').strip():
                raise ValueError('Cached vendor lacks supporting citation')
        elif status=='unknown':vendor='UNKNOWN'
        else:continue
        k=key(r)
        if k in index and index[k]['vendor']!=vendor:raise ValueError('Conflicting cache entries')
        index[k]=dict(vendor=vendor,status=status,evidence_url=source.get('url',''),checked_at=x.get('checked_at',''),cache_file=path.name)
        hashes[path.name]=sha(path)
    return index,hashes

def enrich_full(rows):
    index,_=vendor_index();base={r['tool_id']:r for r in read(BASE)}
    for r in rows:
        r.update(vendor='',vendor_status='not_researched',vendor_evidence_url='')
        original=base.get(r['source_record_id'])
        # Do not transfer provider evidence to a newly scoped product/feature.
        if original and r['product_scope_corrected'] not in (True,'True'):
            if original['tool_name']!=r['original_tool_name'] or original['domain']!=r['domain'] or original['description']!=(r['original_description'] if r['description_corrected'] in (True,'True') else r['description']):raise ValueError('Identity changed during vendor join')
            found=index.get(key(original))
            if found:r.update(vendor=found['vendor'],vendor_status='model_grounded_'+found['status'],vendor_evidence_url=found['evidence_url'])
            else:r['vendor_status']='pending_lookup'
    return rows

def final_files(corrected):
    index,hashes=vendor_index();rows=read(BASE);by={r['source_record_id']:r for r in corrected};provenance={}
    for r in rows:
        original=dict(r);match=index.get(key(original));r['vendor']=match['vendor'] if match else ''
        c=by.get(r['tool_id'])
        if c is None:raise ValueError('Subset ID missing from full corrected export')
        if c['original_tool_name']!=r['tool_name'] or c['domain']!=r['domain']:raise ValueError('Subset identity mismatch')
        r['is_ai_tool']={'1':'true','0':'false','':''}[str(c['is_ai_tool'])]
        if c['product_scope_corrected'] in (True,'True'):
            raise ValueError('Re-research the vendor before changing subset product scope')
        provenance[r['tool_id']]=dict(vendor_status=match['status'] if match else 'pending',vendor_evidence=match or {},
            label_status=c['review_status'],source_is_ai_tool=original['is_ai_tool'],corrected_is_ai_tool=r['is_ai_tool'],
            excluded_from_binary_preparation=not bool(r['is_ai_tool']))
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    content=stream.getvalue().encode('utf-8-sig')
    counts={k:sum(r['is_ai_tool']==k for r in rows) for k in ('true','false','')}
    manifest=dict(governance_provenance='synthetic_demonstration',approval_scope='not_company_scoped',
        risk_label_policy='fill_governance_dataset_b.py illustrative policy; not governance/policy.json',
        vendor_provenance='model-interpreted cached search citations; not independently verified',
        rows=len(rows),label_counts=counts,vendor_found=sum(bool(r['vendor']) and r['vendor']!='UNKNOWN' for r in rows),
        vendor_unknown=sum(r['vendor']=='UNKNOWN' for r in rows),vendor_pending=sum(not r['vendor'] for r in rows),
        base_sha256=sha(BASE),label_review_sha256=sha(ROOT/'data/reviews/conflicts_2026-11-05/recommendations.csv'),cache_sha256=hashes,output_sha256=hashlib.sha256(content).hexdigest(),records=provenance)
    return {FINAL:content,FINAL.with_suffix('.provenance.json'):(json.dumps(manifest,indent=2)+'\n').encode()}
