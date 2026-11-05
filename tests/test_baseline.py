import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'modeling'))
AVAILABLE=importlib.util.find_spec('sklearn') is not None
if AVAILABLE:
    import joblib
    import numpy as np
    from train_baseline import apply_source_review, balanced_source_weights, estimator, metrics
    from predict import predict


@unittest.skipUnless(AVAILABLE,'Install modeling/requirements.txt for model tests')
class BaselineTests(unittest.TestCase):
    def test_effective_class_mass_balanced_after_source_weights(self):
        rows=[dict(is_ai_tool='1',label_source='provisional_catalog_membership') for _ in range(20)]
        rows+=[dict(is_ai_tool='1',label_source='gemini_text_review'),dict(is_ai_tool='0',label_source='team_non_ai_seed')]
        weights=balanced_source_weights(rows)
        self.assertAlmostEqual(weights[:-1].sum(),weights[-1])
        self.assertAlmostEqual(weights[-2]/weights[0],4.0)

    def test_test_vocabulary_is_not_fitted(self):
        model=estimator().fit(['automates writing using language models','project ticket tracking','image generation ai','payroll accounting records'],[1,0,1,0])
        before=dict(model.named_steps['tfidf'].vocabulary_)
        model.predict(['holdoutvocabularyonly'])
        self.assertEqual(before,model.named_steps['tfidf'].vocabulary_)
        self.assertNotIn('holdoutvocabularyonly',before)

    def test_artifact_selection_uses_validation(self):
        m=json.loads((ROOT/'artifacts/baseline/metrics.json').read_text())
        expected=max(('reviewed_only','full_catalog'),key=lambda n:m['binary'][n]['validation']['macro_f1'])
        self.assertEqual(m['selected_binary_model'],expected)

    def test_saved_model_reproduces_reported_test_metrics(self):
        import csv
        with (ROOT/'data/processed/classification_full/test.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        report=json.loads((ROOT/'artifacts/baseline/metrics.json').read_text())
        for name in ('reviewed_only','full_catalog'):
            model=joblib.load(ROOT/f'artifacts/baseline/binary_{name}.joblib')
            m=metrics([int(r['is_ai_tool']) for r in rows],model.predict([r['text_for_model'] for r in rows]),[0,1],['non_ai','ai'])
            self.assertEqual(m,report['binary'][name]['test'])

    def test_saved_input_hashes_match_current_dataset(self):
        import hashlib
        config=json.loads((ROOT/'artifacts/baseline/model_config.json').read_text())
        for filename, expected in config['inputs'].items():
            actual=hashlib.sha256((ROOT/'data/processed/classification_full'/filename).read_bytes()).hexdigest()
            self.assertEqual(actual,expected)

    def test_source_review_withholds_whole_training_groups(self):
        splits=dict(train=[dict(source_record_id='a',group_id='g1',is_ai_tool='1'),
                           dict(source_record_id='alias',group_id='g1',is_ai_tool='1'),
                           dict(source_record_id='b',group_id='g2',is_ai_tool='1'),
                           dict(source_record_id='n',group_id='g3',is_ai_tool='0')],
                    validation=[dict(source_record_id='v',is_ai_tool='1')],
                    test=[dict(source_record_id='t',is_ai_tool='0')])
        review=dict(schema_version=1,records=[dict(source_record_id='a',training_action='exclude_training')])
        filtered,excluded=apply_source_review(splits,review)
        self.assertEqual({r['source_record_id'] for r in filtered['train']},{'b','n'})
        self.assertEqual({r['source_record_id'] for r in excluded},{'a','alias'})
        self.assertEqual(filtered['test'],splits['test'])
        self.assertEqual(filtered['validation'],splits['validation'])
        self.assertEqual(len(splits['train']),4)
        review['records'][0]['source_record_id']='t'
        with self.assertRaises(ValueError):apply_source_review(splits,review)
        review['records'][0]['source_record_id']='unknown'
        with self.assertRaises(ValueError):apply_source_review(splits,review)

    def test_source_review_file_matches_saved_model(self):
        import hashlib
        config=json.loads((ROOT/'artifacts/baseline/model_config.json').read_text())
        review_path=ROOT/'data/reviews'/config['source_review']['file']
        self.assertEqual(hashlib.sha256(review_path.read_bytes()).hexdigest(),config['source_review']['sha256'])
        self.assertEqual(set(config['source_review']['excluded_training_ids']),
                         {'tool_01528','tool_00986','tool_00072','tool_00289'})
        self.assertIn('not presence or absence of AI',config['label_scope'])

    def test_source_review_coverage_and_honest_provenance(self):
        review=json.loads((ROOT/'data/reviews/source_review_2026-11-04.json').read_text())
        self.assertFalse(review['human_verified'])
        self.assertEqual(len([r for r in review['records'] if r['sample']=='all_negative_seeds']),30)
        self.assertEqual(len([r for r in review['records'] if r['sample']=='one_training_record_per_category']),12)
        self.assertEqual(len(review['initial_binary_test_errors']),3)
        self.assertTrue(all(r['source_urls'] for r in review['records']))

    def test_inference_abstains_for_unknown_vocabulary(self):
        result=predict(ROOT/'artifacts/baseline','zzqjxvv','zzqjxvv')
        self.assertIsNone(result['is_ai_tool'])
        self.assertTrue(result['needs_review'])

    def test_inference_has_separate_binary_category_no_risk(self):
        result=predict(ROOT/'artifacts/baseline','Example File Archive','A cloud file storage service for sharing documents and organizing team folders.')
        self.assertIn(result['is_ai_tool'],[0,1])
        self.assertIn('category',result)
        self.assertNotIn('risk_level',result)
        self.assertNotIn('risk_label',result)
        self.assertTrue(result['needs_review'])
        with self.assertRaises(ValueError):predict(ROOT/'artifacts/baseline','Example','')


if __name__=='__main__':unittest.main()
