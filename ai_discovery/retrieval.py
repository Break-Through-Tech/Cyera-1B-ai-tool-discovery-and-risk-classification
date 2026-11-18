"""Local, company-scoped semantic retrieval over Component 4 report snapshots."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

# Small local indexes do not need nested BLAS/OpenMP thread pools. Apply before
# lazy imports of FAISS/PyTorch; preserve an explicitly configured environment.
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_ENCODER=ROOT/'artifacts/category_embeddings/encoder'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def encoder_hash(path):
    files=sorted(p for p in Path(path).rglob('*') if p.is_file())
    if not files:raise ValueError('Local encoder not found; supply --encoder pointing to a local MiniLM bundle')
    return hashlib.sha256('\n'.join(str(p.relative_to(path))+':'+sha(p) for p in files).encode()).hexdigest()
def dump(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def integer(value,label):
    if isinstance(value,bool) or not isinstance(value,int) or value<0:raise ValueError('Invalid '+label)
    return value

def documents(reports):
    """One document per application; one report snapshot per company."""
    docs=[];sources=[];companies=set()
    for path in reports:
        path=Path(path).resolve();report=read(path);fingerprint=sha(path)
        if not isinstance(report.get('results'),list) or not report['results']:
            raise ValueError('Expected a nonempty Component 4 results report: '+path.name)
        company_set={r.get('governance',{}).get('company') for r in report['results']}
        if len(company_set)!=1 or not next(iter(company_set)):
            raise ValueError('Every application must have the same explicit governance company')
        company=next(iter(company_set))
        if company.casefold() in companies:raise ValueError('Use exactly one report snapshot per company')
        companies.add(company.casefold());seen=set()
        sources.append(dict(path=str(path),sha256=fingerprint,company=company))
        for i,row in enumerate(report['results']):
            app_id=row.get('application_id');g=row['governance'];observed=row.get('observed_identity',{})
            if not app_id or app_id in seen:raise ValueError('Missing/duplicate application identity')
            seen.add(app_id)
            name=row.get('application_name');endpoint=row.get('endpoint') or observed.get('observed_endpoint')
            if not isinstance(name,str) or not name.strip() or not isinstance(endpoint,str):raise ValueError('Missing application name/endpoint')
            departments=observed.get('departments',[])
            if isinstance(departments,str):departments=json.loads(departments)
            if not isinstance(departments,list) or any(not isinstance(d,str) for d in departments):raise ValueError('Invalid departments')
            label=row.get('is_ai_tool')
            if label is not None and not isinstance(label,bool):raise ValueError('AI status must be boolean or null')
            score=g.get('risk_score');level=g.get('risk_level')
            if score is None:
                if level!='unknown':raise ValueError('Missing score must have unknown risk level')
            elif isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=100:raise ValueError('Invalid risk score')
            elif level not in ('low','medium','high'):raise ValueError('Invalid risk level')
            controls=dict(g.get('evidence',{}));approval=controls.get('approved_status',{})
            approval_value=approval.get('value') if approval.get('company')==company else None
            if approval.get('value') is not None and approval.get('company')!=company:
                controls['approved_status']={'value':None,'provenance':'unknown','source':'Approval belongs to another or unspecified company'}
            facts=dict(company=company,application_id=app_id,application_name=name,endpoint=endpoint,
                ai_status='ai' if label is True else 'non_ai' if label is False else 'unknown',
                detection_source=row.get('decision_source','unknown'),
                category=row.get('categorization',{}).get('category'),
                category_source=row.get('categorization',{}).get('category_source'),
                category_status=row.get('categorization',{}).get('category_status'),
                departments=sorted(set(departments)),
                unique_users=integer(observed['unique_employee_count'],'user count') if observed.get('unique_employee_count') is not None else None,
                events=integer(observed['event_count'],'event count') if observed.get('event_count') is not None else None,
                user_count_incomplete=bool(observed.get('events_missing_employee_id')) or observed.get('unique_employee_count') is None,
                user_count_scope='Whole application across all departments, not a department-specific count',
                risk_score=score,risk_level=level,approval=approval_value,
                assessment_status=g.get('assessment_status','unknown'),needs_review=g.get('needs_review',True),
                missing_evidence_rules=g.get('missing_evidence_rules',[]),
                contains_synthetic_evidence=g.get('contains_synthetic_evidence',False),
                activity_provenance=g.get('activity_provenance','unknown'),
                assessment_basis=g.get('assessment_basis','unknown'),policy_version=g.get('policy_version'),
                controls=controls,reasons=g.get('reasons',[]),checks=g.get('checks',[]),
                data_quality_warnings=g.get('data_quality_warnings',[]),caveat=g.get('caveat',''))
            # No raw enrichment instructions/HTML, individual users or event logs.
            text=(f"{name} ({endpoint}). Company {company}. AI status {facts['ai_status']}. "
                  f"Purpose {facts['category'] or 'unknown'}. Departments {', '.join(facts['departments'])}. "
                  f"Governance priority {level}, score {score}. Approval {approval_value or 'unknown'}. "
                  +' '.join(str(r.get('explanation','')) for r in facts['reasons'])
                  +' Missing evidence: '+', '.join(facts['missing_evidence_rules']))
            doc_id='ev_'+hashlib.sha256((fingerprint+':'+str(i)).encode()).hexdigest()[:20]
            docs.append(dict(evidence_id=doc_id,facts=facts,text=text,
                source=dict(file=path.name,report_sha256=fingerprint,json_pointer='/results/'+str(i))))
    return docs,sources


def load_encoder(path):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(Path(path).resolve()),device='cpu',local_files_only=True,trust_remote_code=False)

def vectors(encoder,texts):
    import numpy as np
    values=np.asarray(encoder.encode(texts,normalize_embeddings=True,convert_to_numpy=True),dtype='float32')
    if values.ndim!=2 or values.shape[0]!=len(texts) or not np.isfinite(values).all():raise ValueError('Invalid embeddings')
    norms=np.linalg.norm(values,axis=1,keepdims=True)
    if (norms==0).any():raise ValueError('Zero embedding')
    return np.ascontiguousarray(values/norms,dtype='float32')

def build(reports,out,encoder_path=DEFAULT_ENCODER,encoder=None):
    import faiss
    out=Path(out)
    if out.exists():raise ValueError('Index output already exists; use a new directory')
    if any(Path(p).resolve().is_relative_to(out.resolve()) for p in reports):raise ValueError('Report cannot be inside output directory')
    docs,sources=documents(reports)
    fingerprint=encoder_hash(encoder_path);encoder=encoder or load_encoder(encoder_path)
    matrix=vectors(encoder,[d['text'] for d in docs]);index=faiss.IndexFlatIP(matrix.shape[1]);index.add(matrix)
    # Stage the complete bundle before making it visible; never overwrite an index.
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='retrieval_',dir=out.parent) as temp:
        stage=Path(temp);(stage/'documents.json').write_text(dump(docs)+'\n',encoding='utf-8')
        faiss.write_index(index,str(stage/'vectors.faiss'))
        manifest=dict(version=2,documents=len(docs),dimension=int(matrix.shape[1]),metric='cosine_via_normalized_inner_product',
            sources=sources,encoder_path=str(Path(encoder_path).resolve()),encoder_sha256=fingerprint,
            files_sha256={p.name:sha(p) for p in stage.iterdir()},faiss_version=faiss.__version__)
        (stage/'manifest.json').write_text(dump(manifest)+'\n',encoding='utf-8')
        stage.rename(out)
    return manifest


class EvidenceIndex:
    def __init__(self,path,encoder=None):
        import faiss
        self.path=Path(path);self.manifest=read(self.path/'manifest.json')
        if self.manifest.get('version')!=2:raise ValueError('Unsupported retrieval schema; rebuild the index with current code')
        for name in ('documents.json','vectors.faiss'):
            if sha(self.path/name)!=self.manifest['files_sha256'][name]:raise ValueError('Index bundle hash mismatch')
        for source in self.manifest['sources']:
            if sha(source['path'])!=source['sha256']:raise ValueError('Source report changed; rebuild index')
        self.docs=read(self.path/'documents.json');self.index=faiss.read_index(str(self.path/'vectors.faiss'))
        if self.index.ntotal!=len(self.docs) or len(self.docs)!=self.manifest['documents'] or self.index.d!=self.manifest['dimension']:raise ValueError('Index/document dimensions mismatch')
        if len({d['evidence_id'] for d in self.docs})!=len(self.docs):raise ValueError('Duplicate evidence IDs')
        source_keys={(s['sha256'],s['company']) for s in self.manifest['sources']}
        if any((d['source']['report_sha256'],d['facts']['company']) not in source_keys for d in self.docs):raise ValueError('Document/source scope mismatch')
        self.encoder=encoder

    def matching(self,company,department=None,ai_status=None,risk_level=None,approval=None,application=None,control_filters=None):
        if not company or not company.strip():raise ValueError('Company scope is required')
        if ai_status not in (None,'ai','non_ai','unknown') or risk_level not in (None,'low','medium','high','unknown'):raise ValueError('Invalid status filter')
        if approval not in (None,'approved','unapproved','under_review','unknown'):raise ValueError('Invalid approval filter')
        allowed_controls={'sso_available','training_on_customer_data','admin_controls_available','enterprise_plan_available','data_retention'}
        if any(k not in allowed_controls for k in (control_filters or {})):raise ValueError('Unsupported control filter')
        matches=[]
        for i,doc in enumerate(self.docs):
            f=doc['facts']
            if f['company'].casefold()!=company.strip().casefold():continue
            if department and department.casefold() not in {d.casefold() for d in f['departments']}:continue
            if ai_status and f['ai_status']!=ai_status:continue
            if risk_level and f['risk_level']!=risk_level:continue
            if approval and (f['approval'] or 'unknown')!=approval:continue
            rule_for={'sso_available':'no_sso','training_on_customer_data':'customer_data_training','admin_controls_available':'no_admin_controls','enterprise_plan_available':'no_enterprise_plan','data_retention':'long_retention'}
            statuses={c['rule']:c['status'] for c in f.get('checks',[])}
            if any(statuses.get(rule_for[k])=='not_applicable' or dump(f['controls'].get(k,{}).get('value'))!=dump(v) for k,v in (control_filters or {}).items()):continue
            if application and application.casefold() not in (f['application_name'].casefold(),f['endpoint'].casefold(),f['application_id'].casefold()):continue
            matches.append(i)
        return matches

    def search(self,question,company,top_k=5,**filters):
        if not question.strip():raise ValueError('Question is required')
        if not 1<=top_k<=50:raise ValueError('top_k must be between 1 and 50')
        eligible=self.matching(company,**filters)
        if not eligible:return dict(total_matches=0,hits=[],matching_application_ids=[])
        if self.encoder is None:
            path=self.manifest['encoder_path']
            if encoder_hash(path)!=self.manifest['encoder_sha256']:raise ValueError('Encoder changed; rebuild index')
            self.encoder=load_encoder(path)
        query=vectors(self.encoder,[question])
        if query.shape[1]!=self.index.d:raise ValueError('Query/index dimension mismatch')
        # Exact full ranking; metadata filters are applied before selecting top-k.
        scores,ids=self.index.search(query,self.index.ntotal);allowed=set(eligible)
        hits=[dict(self.docs[int(i)],similarity=float(score)) for i,score in zip(ids[0],scores[0]) if int(i) in allowed][:top_k]
        return dict(total_matches=len(eligible),hits=hits,
                    matching_application_ids=[self.docs[i]['facts']['application_id'] for i in eligible])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,action='append',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--encoder',type=Path,default=DEFAULT_ENCODER)
    a=p.parse_args()
    try:
        manifest=build(a.report,a.out,a.encoder)
        print(json.dumps({'documents':manifest['documents'],'companies':[s['company'] for s in manifest['sources']],'out':str(a.out)},indent=2))
    except (ValueError,OSError,KeyError) as error:p.exit(1,'Error: '+str(error)+'\n')

if __name__=='__main__':main()
