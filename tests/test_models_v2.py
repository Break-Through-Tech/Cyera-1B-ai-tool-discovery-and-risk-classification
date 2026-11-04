import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'modeling'))
AVAILABLE=importlib.util.find_spec('sklearn') is not None
if AVAILABLE:
    import train_v2 as v2
    from predict_v2 import V2Predictor


@unittest.skipUnless(AVAILABLE,'Install modeling/requirements-embeddings.txt')
class ModelsV2Tests(unittest.TestCase):
    def test_model_features_exclude_names_labels_and_provenance(self):
        r={'description':'  Project   tracking software. ','tool_name':'AI AI AI','category':'secret','is_ai_tool':'1','has_ai_features':'yes'}
        self.assertEqual(v2.text([r]),['Project tracking software.'])
        other=dict(r,tool_name='Conventional',category='different',source_url='https://example.com')
        self.assertEqual(v2.text([r]),v2.text([other]))

    def test_validation_only_selection_and_fixed_tie_break(self):
        self.assertEqual(v2.choose({'tfidf':.8,'embeddings':.8}),'tfidf')
        self.assertEqual(v2.choose({'tfidf':.7,'embeddings':.8}),'embeddings')

    def test_modified_frozen_split_rejected(self):
        manifest=json.loads((v2.DATA/'manifest.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'train.csv').write_text('not the frozen data')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):v2.load_split(d,'train',manifest)

    def test_training_family_and_description_leaks_rejected(self):
        manifest=json.loads((v2.DATA/'manifest.json').read_text())
        train=v2.load_split(v2.DATA,'train',manifest)
        val=v2.load_split(v2.DATA,'validation',manifest)
        protected=json.loads((v2.DATA/'protected_evaluation_groups.json').read_text())
        v2.check_training(train,val,protected)
        for field in ('family_id','description'):
            bad=copy.deepcopy(train);bad[0][field]=val[0][field]
            with self.assertRaisesRegex(ValueError,'Held-out'):v2.check_training(bad,val,protected)

    def test_training_stage_never_loads_test_split(self):
        requested=[]
        def intercept(data,split,manifest):
            requested.append(split)
            if split=='validation':raise RuntimeError('stop before training')
            return []
        with tempfile.TemporaryDirectory() as d,patch.object(v2,'load_split',side_effect=intercept):
            with self.assertRaisesRegex(RuntimeError,'stop before training'):v2.train(v2.DATA,Path(d)/'out',v2.CACHE)
        self.assertEqual(requested,['train','validation'])

    def test_repeated_test_evaluation_refused(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'test_metrics.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'already exists'):v2.evaluate(v2.DATA,d,v2.CACHE)

    def test_saved_model_tampering_rejected_before_loading(self):
        if not (v2.OUT/'model_config.json').exists():self.skipTest('No trained v2 artifacts')
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);config=json.loads((v2.OUT/'model_config.json').read_text())
            (out/'model_config.json').write_text(json.dumps(config))
            for name in config['model_sha256']:(out/name).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Model hash mismatch'):v2.verified_config(out)

    def test_saved_evaluation_uses_preselected_intact_models(self):
        if not (v2.OUT/'test_metrics.json').exists():self.skipTest('No evaluated v2 artifacts')
        config=v2.verified_config(v2.OUT,v2.DATA)
        report=json.loads((v2.OUT/'test_metrics.json').read_text())
        validation=json.loads((v2.OUT/'validation_metrics.json').read_text())
        self.assertEqual(report['selection_config_sha256'],v2.sha(v2.OUT/'model_config.json'))
        for task in ('binary','category'):
            expected=v2.choose({f:validation[f][task]['macro_f1'] for f in v2.FAMILIES})
            self.assertEqual(config['default_'+task+'_family'],expected)
            self.assertEqual(report['selected_'+task+'_family'],expected)
        self.assertFalse(report['test_used_for_selection'])
        self.assertEqual(report['test_class_counts'],{'non_ai':36,'ai':36})

    def test_prediction_rejects_empty_and_abstains_on_unknown_vocabulary(self):
        if not (v2.OUT/'model_config.json').exists():self.skipTest('No trained v2 artifacts')
        model=V2Predictor()
        with self.assertRaises(ValueError):model.predict('  ')
        if model.binary_family=='tfidf':
            result=model.predict('zzzzqqqqyyyxxx')
            self.assertIsNone(result['is_ai_tool'])
            self.assertIsNone(result['category'])
        a=model.predict('Store, synchronize and share files across devices.','Fictional storage')
        b=model.predict('Store, synchronize and share files across devices.','AI AI AI')
        self.assertEqual(a['is_ai_tool'],b['is_ai_tool'])
        self.assertEqual(a['model_ai_score'],b['model_ai_score'])


if __name__=='__main__':unittest.main()
