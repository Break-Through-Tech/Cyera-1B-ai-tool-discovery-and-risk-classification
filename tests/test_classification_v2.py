import collections
import csv
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_v2', ROOT/'preprocessing_scripts/prepare_classification_v2.py')
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


class ClassificationV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads(v2.SOURCE.read_bytes())
        cls.files = v2.build(v2.SOURCE.read_bytes())
        cls.splits = {s:list(csv.DictReader(cls.files[s+'.csv'].decode().splitlines())) for s in ('train','validation','test')}

    def test_release_reproduces_exactly(self):
        for name, data in self.files.items():
            self.assertEqual((v2.OUTPUT/name).read_bytes(), data, name)

    def test_balanced_evaluation_and_category_coverage(self):
        for name, count, per_category in [('test',36,3),('validation',12,1)]:
            rows = self.splits[name]
            self.assertEqual(collections.Counter(r['is_ai_tool'] for r in rows), {'0':count,'1':count})
            categories = collections.Counter(r['category'] for r in rows if r['is_ai_tool']=='1')
            self.assertEqual(len(categories),12)
            self.assertEqual(set(categories.values()), {per_category})

    def test_no_id_family_or_description_leakage(self):
        for key in ('label_id','family_id','description'):
            sets = [set(r[key].strip().lower() for r in self.splits[s]) for s in self.splits]
            for i in range(len(sets)):
                for j in range(i):self.assertFalse(sets[i] & sets[j], key)
        self.assertEqual(sum(map(len,self.splits.values())), len(self.rows))

    def test_parent_product_and_assistant_are_grouped(self):
        memberships = {r['tool_name']:s for s,rows in self.splits.items() for r in rows}
        for names in [('ClickUp','ClickUp Brain'),('GitHub Enterprise','GitHub Copilot','Power BI','Microsoft Copilot'),('HubSpot Marketing Hub','HubSpot Breeze Prospecting Agent')]:
            self.assertEqual(len({memberships[n] for n in names}),1)

    def test_primary_purpose_is_separate_from_ai_features(self):
        rows = {r['tool_name']:r for r in self.rows}
        self.assertEqual(rows['ClickUp']['is_ai_tool'],0)
        self.assertEqual(rows['ClickUp']['has_ai_features'],'yes')
        self.assertEqual(rows['ClickUp Brain']['is_ai_tool'],1)
        self.assertTrue(all(r['human_verified'] is False for r in self.rows))
        self.assertEqual(collections.Counter(r['is_ai_tool'] for r in self.rows), {0:200,1:96})

    def test_protected_groups_cover_all_evaluation_examples(self):
        protected = json.loads(self.files['protected_evaluation_groups.json'])
        expected = {r['family_id'] for s in ('test','validation') for r in self.splits[s]}
        self.assertEqual(set(protected['families']),expected)

    def test_missing_evidence_and_false_ai_free_claim_rejected(self):
        for changes in ({'source_url':''},{'human_verified':True},{'has_ai_features':'no'},{'category':''}):
            row = dict(next(r for r in self.rows if r['is_ai_tool']))
            row.update(changes)
            with self.assertRaises(ValueError):v2.validate([row])

    def test_duplicate_text_unions_different_vendor_families(self):
        rows = [{'family_id':'a','description':'Same description'}, {'family_id':'b','description':' same  description '}]
        self.assertEqual(len(v2.groups_for(rows)),1)


if __name__ == '__main__':unittest.main()
