"""Check question routing, filters and abstention against explicit QA fixtures."""
import argparse
import json
from pathlib import Path
from .ask import answer
from .retrieval import EvidenceIndex,read,sha


def evaluate(index,case_file):
    specification=read(case_file);results=[]
    for case in specification['cases']:
        result=answer(index,case['question'],case['company'],case.get('top_k',5),case.get('filters'))
        failures=[]
        if result['answer_status']!=case['expected_status']:failures.append('answer_status')
        if 'expected_count' in case and result['total_matches']!=case['expected_count']:failures.append('filtered_count')
        names=[d['facts']['application_name'] for d in result['hits']]
        if 'expected_names' in case and set(names)!=set(case['expected_names']):failures.append('retrieved_names')
        if any(d['facts']['company']!=case['company'] for d in result['hits']):failures.append('company_leak')
        results.append(dict(id=case['id'],question=case['question'],passed=not failures,failures=failures,
            answer_status=result['answer_status'],count=result['total_matches'],names=names,applied_filters=result['applied_filters']))
    return dict(cases=len(results),passed=sum(r['passed'] for r in results),results=results,
        fixture_file_sha256=sha(case_file),index_manifest_sha256=sha(index.path/'manifest.json'),
        scope=specification['scope'],live_generation_evaluated=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():p.error('Output exists; choose a new file')
    with EvidenceIndex(a.index) as index:
        result=evaluate(index,a.cases)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'cases':result['cases'],'passed':result['passed']},indent=2))
    if result['cases']!=result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
