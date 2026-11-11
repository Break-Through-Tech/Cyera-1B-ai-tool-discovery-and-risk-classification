import csv
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'preprocessing_scripts'));sys.path.insert(0,str(ROOT/'modeling'))
import prepare_corrected_classification as prep
AVAILABLE=importlib.util.find_spec('sklearn') is not None
if AVAILABLE:import train_corrected_binary as trainer

class CorrectedSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with prep.SOURCE.open() as f:cls.source=list(csv.DictReader(f))
        cls.splits,cls.excluded=prep.prepare(cls.source)
    def test_complete_record_accounting_and_reproduction(self):
        ids=[r['record_id'] for rows in self.splits.values() for r in rows]+[r['record_id'] for r in self.excluded]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{r['record_id'] for r in self.source})
        files,_=prep.build()
        for name,b in files.items():self.assertEqual((prep.OUT/name).read_bytes(),b)
    def test_no_family_vendor_or_text_leakage(self):
        by={r['record_id']:r for r in self.source}
        for key in ('family','vendor'):
            seen={}
            for s,rows in self.splits.items():
                for r in rows:
                    raw=by[r['record_id']]
                    value=prep.family(r['domain']) if key=='family' else (prep.norm(raw.get('vendor','')) if raw.get('vendor_status')=='model_grounded_found' else '')
                    if not value:continue
                    self.assertIn(seen.get(value,s),(s,))
                    seen[value]=s
        prep.validate(self.splits)
    def test_conflicting_names_and_duplicates_are_not_silently_trained(self):
        rows=[dict(record_id=f'{c}-{i}',tool_name=f'App {c} {i}',domain=f'app{c}-{i}.example.com',description=f'Function {c} {i}',is_ai_tool=str(c),label_method='source_dataset_membership') for c in (0,1) for i in range(40)]
        # Distinct actual host families, rather than one shared example.com family.
        for r in rows:r['domain']=r['record_id']+'.com'
        # Prefix labels make these numeric-leading domain names valid.
        first=rows[0]
        rows.append(dict(first,record_id='duplicate'))
        rows.append(dict(rows[1],record_id='opposite',is_ai_tool='1'))
        splits,excluded=prep.prepare(rows)
        reasons={r['record_id']:r['reason'] for r in excluded}
        self.assertIn('duplicate_name_and_host',reasons.values())
        self.assertEqual(reasons['opposite'],'remaining_cross_label_name_or_description')
        self.assertEqual(reasons['0-1'],'remaining_cross_label_name_or_description')
    def test_target_and_metadata_never_enter_feature_schema(self):
        self.assertFalse({'vendor','risk_label','approved_status','review_status'} & set(prep.FIELDS))

@unittest.skipUnless(AVAILABLE,'Install modeling/requirements.txt')
class CorrectedModelTests(unittest.TestCase):
    def test_description_is_only_predictor_input(self):
        a={'description':'An application description','tool_name':'AI','is_ai_tool':'1','vendor':'X'}
        b=dict(a,tool_name='non-AI',is_ai_tool='0',vendor='Y',risk_label='high')
        self.assertEqual(trainer.inputs([a]),trainer.inputs([b]))
    def test_training_does_not_open_test_split(self):
        seen=[]
        def intercept(data,split,manifest):
            seen.append(split)
            if split=='validation':raise RuntimeError('stop before fitting')
            return []
        with tempfile.TemporaryDirectory() as d,patch.object(trainer,'load_split',side_effect=intercept):
            with self.assertRaisesRegex(RuntimeError,'stop before fitting'):trainer.train(trainer.DATA,Path(d)/'new')
        self.assertEqual(seen,['train','validation'])
    def test_modified_split_rejected(self):
        manifest=json.loads((trainer.DATA/'manifest.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'test.csv').write_text('modified')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):trainer.load_split(d,'test',manifest)
    def test_prediction_abstention_and_repeated_evaluation_guard(self):
        if not (trainer.OUT/'model_config.json').exists():self.skipTest('Model not trained')
        self.assertIsNone(trainer.predict('zzzzqqqqyyyxxx',trainer.OUT)['is_ai_tool'])
        with self.assertRaises(ValueError):trainer.predict(' ',trainer.OUT)
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'test_metrics.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'already evaluated'):trainer.evaluate(trainer.DATA,d)

if __name__=='__main__':unittest.main()
