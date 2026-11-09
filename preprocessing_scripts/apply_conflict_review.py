"""Apply approved research recommendations to a separate versioned labeling export."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'preprocessing_scripts'))
SOURCE=ROOT/'data/processed/source_membership/all_labeled.csv'
REVIEW=ROOT/'data/reviews/conflicts_2026-11-05'
OUT=ROOT/'data/processed/corrected_labels_2026-11-05'


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def apply(rows,recommendations,descriptions):
    lookup={r['record_id']:r for r in recommendations}
    if len(lookup)!=len(recommendations):raise ValueError('Duplicate review IDs')
    source={r['record_id']:r for r in rows}
    if len(source)!=len(rows) or set(lookup)-set(source):raise ValueError('Duplicate source IDs or unknown review IDs')
    if set(lookup)!={r['record_id'] for r in rows if r['potential_label_conflict']=='True'}:
        raise ValueError('Review must cover exactly the flagged source records')
    out=[]
    for r in rows:
        x=dict(r)
        x.update(source_is_ai_tool=r['is_ai_tool'],original_tool_name=r['tool_name'],
                 original_description='',review_status='not_in_flagged_review',review_evidence_url='',
                 product_scope_corrected=False,description_corrected=False,exclusion_reason='')
        if r['record_id'] in lookup:
            review=lookup[r['record_id']]
            if review['source_label']!=r['is_ai_tool']:raise ValueError('Review/source label mismatch')
            target=review['recommended_label']
            if target not in ('0','1',''):raise ValueError('Invalid recommended label')
            x['is_ai_tool']=target
            x['label_method']='assistant_official_source_review'
            x['review_evidence_url']=review['evidence_url']
            x['review_status']='resolved_recommendation_applied' if target else 'unresolved_set_aside'
            if not target:
                x['exclusion_reason']='unresolved_identity_or_product_scope'
            else:
                if review['scope_change_required']=='True':
                    x['tool_name']=review['proposed_product_scope'];x['product_scope_corrected']=True
                if x['product_scope_corrected'] or not r['description'].strip():
                    if r['source_record_id'] not in descriptions:raise ValueError('Missing reviewed scoped description')
                    x['original_description']=r['description']
                    x['description']=descriptions[r['source_record_id']]
                    x['description_corrected']=True
        if not x['description'].strip():
            x['exclusion_reason']=';'.join(filter(None,[x['exclusion_reason'],'missing_description']))
        # This is only a preparation gate, not a final train/evaluation assignment.
        x['eligible_for_binary_preparation']=x['is_ai_tool'] in ('0','1') and not x['exclusion_reason']
        out.append(x)
    return out


def csv_bytes(rows,fields):
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows);return s.getvalue().encode()


def build():
    review_summary=json.loads((REVIEW/'summary.json').read_text())
    if sha(SOURCE)!=review_summary['input_labeled_csv_sha256']:raise ValueError('Source changed since research')
    if sha(REVIEW/'recommendations.csv')!=review_summary['recommendations_sha256']:raise ValueError('Review changed since research')
    original=read(SOURCE)
    rows=apply(original,read(REVIEW/'recommendations.csv'),json.loads((REVIEW/'scoped_descriptions.json').read_text()))
    from reconcile_vendor_enrichment import enrich_full, final_files
    rows=enrich_full(rows)
    fields=list(rows[0]);unresolved=[r for r in rows if r['review_status']=='unresolved_set_aside']
    files={'all_corrected.csv':csv_bytes(rows,fields),'unresolved.csv':csv_bytes(unresolved,fields)}
    summary=dict(version='2026-11-05',total_rows=len(rows),
        class_counts=dict(Counter(r['is_ai_tool'] or 'unresolved' for r in rows)),
        binary_label_changes=sum(r['is_ai_tool'] in ('0','1') and r['is_ai_tool']!=r['source_is_ai_tool'] for r in rows),
        unresolved_rows=len(unresolved),scope_corrections=sum(r['product_scope_corrected'] for r in rows),
        description_corrections=sum(r['description_corrected'] for r in rows),
        eligible_for_binary_preparation=sum(r['eligible_for_binary_preparation'] for r in rows),
        input_sha256={'source':sha(SOURCE),'recommendations':sha(REVIEW/'recommendations.csv'),'scoped_descriptions':sha(REVIEW/'scoped_descriptions.json')},
        output_sha256={k:hashlib.sha256(v).hexdigest() for k,v in files.items()},
        source_labels_modified=False,models_retrained=False,independently_human_verified=False,
        note='All source rows retained. Unresolved review labels are blank and excluded from binary preparation. Unreviewed records retain source labels. This is not deduplicated or split into training/evaluation data.')
    summary['vendor_found']=sum(r['vendor_status']=='model_grounded_found' for r in rows)
    summary['vendor_unknown']=sum(r['vendor_status']=='model_grounded_unknown' for r in rows)
    files['summary.json']=(json.dumps(summary,indent=2)+'\n').encode()
    # Final subset remains separate from the full classification source.
    files.update(final_files(rows))
    return files,summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    files,summary=build()
    if not a.check:OUT.mkdir(parents=True,exist_ok=True)
    for name,data in files.items():
        path=name if isinstance(name,Path) else OUT/name
        if a.check:
            if not path.exists() or path.read_bytes()!=data:raise SystemExit('Does not reproduce: '+str(name))
        else:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
