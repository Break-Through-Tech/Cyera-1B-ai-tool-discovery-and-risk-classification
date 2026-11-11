"""Apply the six documented subset scope resolutions to a new full-data release."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/processed/corrected_labels_2026-11-05/all_corrected.csv'
SUBSET=ROOT/'data/processed/final/dataset_b_final_fixed.csv'
AUDIT=SUBSET.with_suffix('.label_fixes.json')
OUT=ROOT/'data/processed/reconciled_labels_2026-11-11'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def encode(rows,fields):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
    writer.writeheader();writer.writerows(rows);return stream.getvalue().encode()


def build():
    audit=json.loads(AUDIT.read_text())
    data=SUBSET.read_bytes()
    crlf=hashlib.sha256(data.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')).hexdigest()
    if audit['output_sha256'] not in (sha(SUBSET),crlf):raise ValueError('Resolved subset differs from its label audit')
    subset={r['tool_id']:r for r in read(SUBSET)}
    changes={r['tool_id']:r for r in audit['changes']}
    if len(changes)!=6:raise ValueError('Expected exactly six documented scope resolutions')
    rows=read(SOURCE);applied=[]
    for tid,change in changes.items():
        ref=subset[tid]
        matches=[r for r in rows if r['source_record_id']==tid and r['tool_name']==ref['tool_name'] and r['domain']==ref['domain']]
        if len(matches)!=1 or matches[0]['is_ai_tool']!='':raise ValueError('Resolution must identify one currently unresolved record: '+tid)
        row=matches[0]
        if change['tool_name']!=row['tool_name'] or change['new_label']!=ref['is_ai_tool']:raise ValueError('Resolution evidence/identity mismatch')
        label={'true':'1','false':'0'}[change['new_label']]
        row.update(is_ai_tool=label,label_method='official_source_scope_resolution',
                   review_status='resolved_product_scope_2026-11-10',review_evidence_url=change['reference_url'])
        row['exclusion_reason']=';'.join(x for x in row['exclusion_reason'].split(';') if x!='unresolved_identity_or_product_scope')
        row['eligible_for_binary_preparation']=str(bool(row['description'].strip() and not row['exclusion_reason']))
        applied.append(dict(record_id=row['record_id'],old_label='',new_label=label,reason=change['reason'],evidence_url=change['reference_url']))
    fields=list(rows[0]);unknown=[r for r in rows if not r['is_ai_tool']]
    files={'all_corrected.csv':encode(rows,fields),'unresolved.csv':encode(unknown,fields)}
    report=dict(version='reconciled-2026-11-11',rows=len(rows),class_counts=dict(Counter(r['is_ai_tool'] or 'unresolved' for r in rows)),
        changes=applied,audit_digest_matches_original_crlf=(crlf==audit['output_sha256']),input_sha256={'source':sha(SOURCE),'resolved_subset':sha(SUBSET),'resolution_audit':sha(AUDIT)},
        output_sha256={name:hashlib.sha256(value).hexdigest() for name,value in files.items()},
        independently_human_verified=False,note='Six documented scope decisions propagated; all other labels and source files preserved.')
    files['summary.json']=(json.dumps(report,indent=2)+'\n').encode()
    return files,report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    files,report=build()
    for name,value in files.items():
        path=OUT/name
        if a.check:
            if not path.exists() or path.read_bytes()!=value:raise ValueError('Does not reproduce: '+name)
        else:
            if path.exists() and path.read_bytes()!=value:raise ValueError('Release already exists with different bytes')
            OUT.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    print(json.dumps(report['class_counts'],indent=2))

if __name__=='__main__':main()
