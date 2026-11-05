"""Label every source record: AI catalog=1, conventional catalog=0."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
SOURCES=[('ai_catalog',ROOT/'data/processed/dataset_b/dataset_b_master.csv',1),
         ('non_ai_catalog',ROOT/'data/processed/non_ai/dataset_b_non_ai_master.csv',0)]
OUT=ROOT/'data/processed/source_membership'
FIELDS=['record_id','source_dataset','source_record_id','tool_name','domain','description','is_ai_tool',
        'original_is_ai_tool','label_method','potential_label_conflict','conflict_signals']


def norm(value):return ' '.join(value.casefold().split())


def domain(value):
    try:
        p=urlsplit(value if '://' in value else 'https://'+value)
        return (p.hostname or '').lower().removeprefix('www.')+p.path.rstrip('/').lower()
    except ValueError:return ''


def label(sources):
    rows=[];seen=set();inputs={}
    for source,path,target in sources:
        if target not in (0,1):raise ValueError('Expected binary source label')
        data=Path(path).read_bytes()
        inputs[source]={'file':Path(path).relative_to(ROOT).as_posix() if Path(path).is_relative_to(ROOT) else Path(path).name,
                        'sha256':hashlib.sha256(data).hexdigest(),'assigned_label':target}
        parsed=list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
        for r in parsed:
            sid=r.get('tool_id','').strip();rid=source+':'+sid
            if not sid or rid in seen:raise ValueError('Missing or duplicate source ID')
            seen.add(rid)
            rows.append(dict(record_id=rid,source_dataset=source,source_record_id=sid,
                tool_name=r.get('tool_name',''),domain=r.get('domain',''),description=r.get('description',''),
                is_ai_tool=target,original_is_ai_tool=r.get('is_ai_tool',''),label_method='source_dataset_membership',
                potential_label_conflict=False,conflict_signals=''))
        inputs[source]['rows']=len(parsed)
    indices={key:defaultdict(list) for key in ('name','domain','description')}
    for i,r in enumerate(rows):
        for key,value in [('name',norm(r['tool_name'])),('domain',domain(r['domain'])),('description',norm(r['description']))]:
            if value:indices[key][value].append(i)
    conflicts=[];signals=defaultdict(set)
    for key,index in indices.items():
        for value,ids in sorted(index.items()):
            if len({rows[i]['is_ai_tool'] for i in ids})>1:
                gid=hashlib.sha256((key+':'+value).encode()).hexdigest()[:16]
                for i in ids:
                    signals[i].add(key)
                    conflicts.append({'conflict_id':gid,'signal':key,'record_id':rows[i]['record_id'],'tool_name':rows[i]['tool_name'],'is_ai_tool':rows[i]['is_ai_tool']})
    for i,keys in signals.items():
        rows[i]['potential_label_conflict']=True;rows[i]['conflict_signals']=';'.join(sorted(keys))
    summary={'label_method':'source_dataset_membership','independently_verified':False,
        'total_rows':len(rows),'class_counts':dict(Counter(str(r['is_ai_tool']) for r in rows)),
        'potential_conflict_rows':len(signals),'potential_conflict_groups':len({r['conflict_id'] for r in conflicts}),
        'inputs':inputs,'policy':'All records retained and labeled by source, including duplicates, missing descriptions and prior rejected/uncertain records. Conflict flags do not change labels. No model or API called.'}
    return rows,conflicts,summary


def csv_bytes(rows,fields):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
    writer.writeheader();writer.writerows(rows);return stream.getvalue().encode('utf-8')


def build():
    rows,conflicts,summary=label(SOURCES)
    files={'all_labeled.csv':csv_bytes(rows,FIELDS)}
    for target,name in [(1,'ai_labeled.csv'),(0,'non_ai_labeled.csv')]:
        files[name]=csv_bytes([r for r in rows if r['is_ai_tool']==target],FIELDS)
    files['potential_conflicts.csv']=csv_bytes(conflicts,['conflict_id','signal','record_id','tool_name','is_ai_tool'])
    summary['outputs_sha256']={name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
    files['summary.json']=(json.dumps(summary,indent=2)+'\n').encode()
    return files,summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    files,summary=build()
    if not a.check:OUT.mkdir(parents=True,exist_ok=True)
    for name,data in files.items():
        if a.check:
            if not (OUT/name).exists() or (OUT/name).read_bytes()!=data:raise SystemExit('Does not reproduce: '+name)
        else:(OUT/name).write_bytes(data)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
