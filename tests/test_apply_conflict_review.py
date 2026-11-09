import csv
import importlib.util
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('apply_review',ROOT/'preprocessing_scripts/apply_conflict_review.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ApplyConflictReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=m.read(m.SOURCE);cls.reviews=m.read(m.REVIEW/'recommendations.csv')
        cls.descriptions=json.loads((m.REVIEW/'scoped_descriptions.json').read_text())
        cls.rows=m.apply(cls.original,cls.reviews,cls.descriptions)
    def test_all_records_and_source_assignments_preserved(self):
        self.assertEqual(len(self.rows),len(self.original))
        for before,after in zip(self.original,self.rows):
            self.assertEqual(before['record_id'],after['record_id'])
            self.assertEqual(before['is_ai_tool'],after['source_is_ai_tool'])
            self.assertEqual(before['description'],after['original_description'] if after['description_corrected'] else after['description'])
    def test_recommendations_applied_and_unresolved_excluded(self):
        by={r['record_id']:r for r in self.rows}
        for r in self.reviews:
            x=by[r['record_id']];self.assertEqual(x['is_ai_tool'],r['recommended_label'])
            if not x['is_ai_tool']:self.assertFalse(x['eligible_for_binary_preparation'])
            if r['recommended_label'] and r['scope_change_required']=='True':
                self.assertEqual(x['tool_name'],r['proposed_product_scope'])
                self.assertTrue(x['description_corrected'])
        _,summary=m.build()
        self.assertEqual(summary['class_counts'],{'1':16740,'0':516,'unresolved':36})
        self.assertEqual(summary['binary_label_changes'],30)
    def test_incomplete_review_or_missing_scoped_description_rejected(self):
        with self.assertRaises(ValueError):m.apply(self.original,self.reviews[:-1],self.descriptions)
        with self.assertRaises(ValueError):m.apply(self.original,self.reviews,{})
    def test_artifacts_reproduce(self):
        files,_=m.build()
        for name,data in files.items():self.assertEqual((m.OUT/name).read_bytes(),data)
if __name__=='__main__':unittest.main()
