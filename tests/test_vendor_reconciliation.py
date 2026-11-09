import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'preprocessing_scripts'))
sys.path.insert(0,str(ROOT))
import reconcile_vendor_enrichment as v
from governance.risk import run, POLICY

class VendorReconciliationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=v.read(v.FINAL)
        cls.manifest=json.loads(v.FINAL.with_suffix('.provenance.json').read_text())
    def test_final_subset_preserves_nonlabel_nonvendor_fields(self):
        base=v.read(v.BASE)
        self.assertEqual([r['tool_id'] for r in self.rows],[r['tool_id'] for r in base])
        for a,b in zip(base,self.rows):
            self.assertEqual(set(a),set(b))
            for k in a:
                if k not in ('vendor','is_ai_tool'):self.assertEqual(a[k],b[k])
        counts={k:sum(r['is_ai_tool']==k for r in self.rows) for k in ('true','false','')}
        self.assertEqual(counts,{'true':305,'false':386,'':6})
        by={r['tool_name']:r for r in self.rows}
        for n in ('Akismet','Auphonic'):self.assertEqual(by[n]['is_ai_tool'],'true')
    def test_all_filled_vendors_have_exact_cached_identity_and_provenance(self):
        index,_=v.vendor_index();base={r['tool_id']:r for r in v.read(v.BASE)}
        for r in self.rows:
            match=index.get(v.key(base[r['tool_id']]))
            self.assertEqual(r['vendor'],match['vendor'] if match else '')
        self.assertEqual(self.manifest['vendor_found'],676)
        self.assertEqual(self.manifest['vendor_unknown'],11)
        self.assertEqual(self.manifest['vendor_pending'],10)
        self.assertEqual(v.sha(v.FINAL),self.manifest['output_sha256'])
    def test_vendor_evidence_not_transferred_to_new_feature_scope(self):
        base=v.read(v.BASE)[0]
        row=dict(source_record_id=base['tool_id'],product_scope_corrected=True)
        self.assertEqual(v.enrich_full([row])[0]['vendor_status'],'not_researched')
    def test_synthetic_policy_requires_explicit_demo_opt_in(self):
        activity=ROOT/'data/raw/summit_financial_usage (1).csv'
        with self.assertRaisesRegex(ValueError,'synthetic policies'):run('Summit Financial',activity,v.FINAL,POLICY)
        report=run('Summit Financial',activity,v.FINAL,POLICY,synthetic_metadata=True)
        self.assertTrue(report['synthetic_scenario_applied'])
        self.assertTrue(any(r['metadata_source']=='synthetic_dataset_b' for r in report['applications']))
        for r in report['applications']:
            # The subset's approval is not scoped to this company.
            self.assertIsNone(r['metadata']['approved_status'])

if __name__=='__main__':unittest.main()
