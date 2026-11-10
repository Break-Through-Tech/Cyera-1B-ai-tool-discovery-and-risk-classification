"""Copy project ground-truth Kaggle categories to AI rows in Dataset B.

python map_kaggle_categories.py --kaggle data/raw/kaggle.csv --input data/processed/final/dataset_b_final_fixed.csv --output data/processed/final/dataset_b_final_kaggle_categories.csv

Standard library only. No API calls or model predictions. Non-AI rows and all
non-category cells stay unchanged. Matching requires normalized name AND host;
original tool_NNNNN row IDs are used only after those identity checks pass.
Unmatched/ambiguous AI entries cause a stop with a report, not a mixed taxonomy.
"""
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit


def name_key(s):
    s=unicodedata.normalize('NFKC',s).casefold()
    return re.sub(r'\W+',' ',s).strip()


def host(s):
    s=s.strip()
    if not s:return ''
    try:
        h=(urlsplit(s if '://' in s else 'https://'+s).hostname or '').lower().rstrip('.').removeprefix('www.')
        return h.encode('idna').decode()
    except (ValueError,UnicodeError):return ''


def read(path,required):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames or []
        if not set(required)<=set(fields):raise ValueError(f'{path}: missing required columns')
        rows=list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):raise ValueError(f'{path}: malformed CSV')
    return fields,rows


def write(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kaggle',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report_path=a.output.with_suffix('.category_mapping.csv')
    summary_path=a.output.with_suffix('.category_mapping.json')
    inputs={a.input.resolve(),a.kaggle.resolve()}
    for path in [a.output,report_path,summary_path]:
        if path.resolve() in inputs or path.exists():p.error(f'Output would overwrite a file: {path}')
    _,kaggle=read(a.kaggle,['Name','Website','Category'])
    fields,rows=read(a.input,['tool_id','tool_name','domain','description','is_ai_tool','category'])
    ids=[r['tool_id'] for r in rows]
    if len(ids)!=len(set(ids)) or not all(ids):p.error('Input has duplicate/missing tool IDs')
    index=defaultdict(list)
    for i,r in enumerate(kaggle,1):
        n,h=name_key(r['Name']),host(r['Website'])
        if n and h:index[(n,h)].append((i,r))
    original=[dict(r) for r in rows];report=[]
    for r in rows:
        flag=r['is_ai_tool'].strip().lower()
        if flag not in ('true','false','1','0'):p.error('Unresolved AI label: '+r['tool_id'])
        entry=dict(tool_id=r['tool_id'],tool_name=r['tool_name'],domain=r['domain'],
                   old_category=r['category'],new_category=r['category'],status='',match_method='',kaggle_rows='')
        if flag in ('false','0'):
            entry['status']='non_ai_unchanged';report.append(entry);continue
        key=(name_key(r['tool_name']),host(r['domain']))
        candidates=index.get(key,[])
        idmatch=re.fullmatch(r'tool_(\d+)',r['tool_id'])
        # Even a valid ordinal cannot override conflicting duplicates at the same identity.
        categories={x['Category'].strip() for _,x in candidates}
        if not candidates:entry['status']='unmatched'
        elif '' in categories:entry['status']='source_category_missing'
        elif len(categories)!=1:entry['status']='ambiguous_source_categories'
        else:
            entry.update(status='mapped',new_category=next(iter(categories)),
                         match_method='normalized_name_and_host',kaggle_rows=';'.join(str(i) for i,_ in candidates))
            if idmatch and any(i==int(idmatch.group(1)) for i,_ in candidates):
                entry['match_method']='original_row_id_validated_by_name_and_host'
            r['category']=entry['new_category']
        report.append(entry)
    unresolved=[r for r in report if r['status'] not in ('mapped','non_ai_unchanged')]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    write(report_path,report,['tool_id','tool_name','domain','old_category','new_category','status','match_method','kaggle_rows'])
    summary=dict(input_rows=len(rows),kaggle_rows=len(kaggle),status_counts=dict(Counter(r['status'] for r in report)),
                 changed_categories=sum(x['category']!=y['category'] for x,y in zip(original,rows)),
                 ai_category_counts=dict(Counter(r['category'] for r in rows if r['is_ai_tool'].lower() in ('true','1'))),
                 dataset_written=not unresolved,
                 policy='Original Kaggle categories are project ground truth; only AI-row categories are updated. Non-AI categories preserved and excluded from AI category training.',
                 source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.input,a.kaggle]})
    summary_path.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    if unresolved:
        raise SystemExit(f'{len(unresolved)} AI rows need identity review. Dataset NOT written. See {report_path}')
    for old,new in zip(original,rows):
        assert all(old[k]==new[k] for k in fields if k!='category')
        if old['is_ai_tool'].lower() in ('false','0'):assert old==new
    write(a.output,rows,fields)
    print(json.dumps(summary,indent=2))
    print(f'Wrote {a.output}\nAudit: {report_path}')


if __name__=='__main__':main()
