"""Evaluate explicit query relevance labels without natural-language filter shortcuts."""
import argparse
import json
from pathlib import Path
from .retrieval import EvidenceIndex,sha,read


def evaluate(index,query_file,top_k=3):
    data=read(query_file);company=data['company']
    if data['report_sha256'] not in {s['sha256'] for s in index.manifest['sources'] if s['company']==company}:raise ValueError('Evaluation labels refer to a different report snapshot')
    if not data['queries']:raise ValueError('No evaluation queries')
    results=[]
    for row in data['queries']:
        relevant=set(row['relevant_application_names'])
        hits=index.search(row['question'],company,top_k)['hits']
        names=[h['facts']['application_name'] for h in hits]
        ranks=[i+1 for i,n in enumerate(names) if n in relevant]
        results.append(dict(id=row['id'],question=row['question'],expected=sorted(relevant),retrieved=names,
            hit_at_1=bool(names and names[0] in relevant),hit_at_k=bool(ranks),reciprocal_rank_at_k=1/min(ranks) if ranks else 0,
            evidence_ids=[h['evidence_id'] for h in hits]))
    n=len(results)
    return dict(company=company,queries=n,top_k=top_k,hit_rate_at_1=sum(r['hit_at_1'] for r in results)/n,
        hit_rate_at_k=sum(r['hit_at_k'] for r in results)/n,
        mean_reciprocal_rank_at_k=sum(r['reciprocal_rank_at_k'] for r in results)/n,
        query_file_sha256=sha(query_file),index_manifest_sha256=sha(index.path/'manifest.json'),
        source_report_sha256=data['report_sha256'],scope=data['evaluation_scope'],results=results,
        natural_language_filter_inference_used=False,live_generation_evaluated=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True)
    p.add_argument('--queries',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Output exists; choose a new file')
    with EvidenceIndex(a.index) as index:
        result=evaluate(index,a.queries)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('queries','hit_rate_at_1','hit_rate_at_k','mean_reciprocal_rank_at_k')},indent=2))

if __name__=='__main__':main()
