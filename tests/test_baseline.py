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
    from train_baseline import balanced_source_weights, estimator, metrics
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
