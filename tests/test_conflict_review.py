import csv
import hashlib
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
class ConflictReviewTests(unittest.TestCase):
    def test_review_covers_exactly_flagged_records_and_preserves_labels(self):
        p=ROOT/'data/processed/source_membership/all_labeled.csv'
        with p.open() as f:source={r['record_id']:r for r in csv.DictReader(f) if r['potential_label_conflict']=='True'}
        with (ROOT/'data/reviews/conflicts_2026-11-05/recommendations.csv').open() as f:review=list(csv.DictReader(f))
        self.assertEqual(len(review),201)
        self.assertEqual({r['record_id'] for r in review},set(source))
        self.assertEqual(len({r['record_id'] for r in review}),len(review))
        for r in review:
            self.assertEqual(r['source_label'],source[r['record_id']]['is_ai_tool'])
            self.assertIn(r['recommended_label'],('0','1',''))
            self.assertEqual(r['human_verified'],'False')
            self.assertTrue(r['evidence_url'].startswith('https://'))
            self.assertTrue(r['evidence_summary'])
            if r['recommended_label']=='':self.assertEqual(r['action'],'set_aside_pending_resolution')
        summary=json.loads((ROOT/'data/reviews/conflicts_2026-11-05/summary.json').read_text())
        self.assertEqual(summary['input_labeled_csv_sha256'],hashlib.sha256(p.read_bytes()).hexdigest())
        self.assertFalse(summary['source_labels_modified'])

    def test_non_ai_view_matches_full_review(self):
        base=ROOT/'data/reviews/conflicts_2026-11-05'
        with (base/'recommendations.csv').open() as f:full={r['record_id']:r for r in csv.DictReader(f) if r['source_label']=='0'}
        with (base/'non_ai_recommendations.csv').open() as f:subset=list(csv.DictReader(f))
        self.assertEqual(len(subset),102)
        self.assertEqual({r['record_id'] for r in subset},set(full))
        for r in subset:self.assertTrue(all(value==full[r['record_id']][key] for key,value in r.items()))
if __name__=='__main__':unittest.main()
