"""Resolve six blank labels under the dataset's AI-focused-product policy.

Usage:
 python fix_final_ai_labels.py --input data/processed/final/dataset_b_final.csv --output data/processed/final/dataset_b_final_fixed.csv

false means general application scope, NOT absence of AI functionality.
No input overwritten. Only six is_ai_tool cells change. A JSON audit is saved.
Run this AFTER reconcile_vendor_enrichment.py; that script otherwise restores blanks.
Python standard library only.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

POLICY = ('Classify the named product scope represented by the supplied row: '
          'true for a specific AI-focused product/feature; false for a broad conventional '
          'application/suite. False does not mean no AI features exist. These are '
          'dataset-scope decisions, not current domain-wide AI capability judgments.')
FIXES = {
 'tool_09844': ('Freshworks','freshworks.com','false',
   'The row represents the broad business-software suite, not the specific Freddy AI product.',
   'https://www.freshworks.com/assets/resources/AI-in-ITSM.pdf'),
 'tool_14954': ('Semrush outline builder','semrush.com','true',
   'The named Outline Builder is a specific feature that uses AI to generate content outlines. Historical product evidence; not a claim of current availability.',
   'https://www.semrush.com/news/243009-build-seo-optimized-content-faster-with-the-content-outline-builder-from-semrush/'),
 'saas_00036': ('Notion','notion.so','false',
   'The supplied row describes the general docs/wiki/project workspace, not Notion AI.',
   'https://www.notion.com/product/projects'),
 'saas_00054': ('Airtable','airtable.com','false',
   'The supplied row describes the general spreadsheet/database application platform, not Airtable AI or Omni. The current platform offers AI functionality.',
   'https://www.airtable.com/platform/ai'),
 'saas_00447': ('Podium','podium.com','false',
   'The supplied row describes general communications, reviews and payments, not Podium AI Employee. Current Podium offers AI functionality. This is not the similarly named podcast product.',
   'https://www.podium.com/product/payments'),
 'saas_00433': ('Bubble','bubble.io','false',
   'The supplied row describes the general no-code development platform, not the specific AI App Builder. Current Bubble offers AI generation functionality.',
   'https://manual.bubble.io/help-guides/ai'),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    audit_path=a.output.with_suffix('.label_fixes.json')
    if a.input.resolve()==a.output.resolve():p.error('Output must differ from input.')
    if a.output.exists() or audit_path.exists():p.error('Output/audit exists; choose another output filename.')
    data=a.input.read_bytes()
    with a.input.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    if not {'tool_id','tool_name','domain','is_ai_tool'}<=set(fields or []):p.error('Required columns missing.')
    if any(None in r or any(v is None for v in r.values()) for r in rows):p.error('Malformed CSV.')
    ids=[r['tool_id'] for r in rows]
    if len(ids)!=len(set(ids)) or not all(ids):p.error('Missing or duplicate tool IDs.')
    if not set(FIXES)<=set(ids):p.error('Expected six target records are not all present.')
    before=[dict(r) for r in rows];changes=[]
    for r in rows:
        if r['tool_id'] in FIXES:
            name,domain,label,reason,url=FIXES[r['tool_id']]
            if (r['tool_name'],r['domain'])!=(name,domain):p.error('Target identity mismatch: '+r['tool_id'])
            if r['is_ai_tool'].strip():p.error('Target is already labeled; refusing to overwrite: '+r['tool_id'])
            changes.append(dict(tool_id=r['tool_id'],tool_name=name,old_label=r['is_ai_tool'],new_label=label,
                                reason=reason,reference_url=url,decision_type='supplied_product_scope_policy',review_date='2026-11-10'))
            r['is_ai_tool']=label
        elif r['is_ai_tool'] not in ('true','false'):
            p.error('Unexpected unresolved/invalid label outside the six targets: '+r['tool_id'])
    differences=[(i,k) for i,(old,new) in enumerate(zip(before,rows)) for k in fields if old[k]!=new[k]]
    assert len(differences)==6 and all(k=='is_ai_tool' for _,k in differences)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    report=dict(policy=POLICY,changes=changes,rows=len(rows),counts=dict(Counter(r['is_ai_tool'] for r in rows)),
                input_sha256=hashlib.sha256(data).hexdigest(),output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
                scope='Six labels only. Original descriptions/categories and synthetic governance fields preserved. Other labels not re-reviewed.',
                regeneration='Run after reconcile_vendor_enrichment.py; its old review inputs otherwise recreate blank labels.')
    audit_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(a.output),'audit':str(audit_path),'rows':len(rows),'changed_labels':6,'counts':report['counts']},indent=2))


if __name__=='__main__':main()
