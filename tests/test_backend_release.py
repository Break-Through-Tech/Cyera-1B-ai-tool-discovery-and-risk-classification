import csv
import json
from pathlib import Path
import tempfile
import unittest

from ai_discovery.catalog import load_catalog,match_applications
from ai_discovery.report import ROOT,run
from preprocessing_scripts import reconcile_scope_resolutions as release
from preprocessing_scripts.prepare_corrected_classification import build
from modeling.train_categories import CATEGORIES,DATA,OUT,sha,splits
from modeling.train_category_model import load_training


class BackendReleaseTests(unittest.TestCase):
    def test_six_resolutions_reproduce_without_changing_other_records(self):
        files,summary=release.build()
        for name,value in files.items():self.assertEqual((release.OUT/name).read_bytes(),value)
        original={r['record_id']:r for r in release.read(release.SOURCE)}
        updated=release.read(release.OUT/'all_corrected.csv')
        changed={r['record_id'] for r in updated if r!=original[r['record_id']]}
        self.assertEqual(changed,{r['record_id'] for r in summary['changes']})
        self.assertEqual(summary['class_counts'],{'1':16741,'0':521,'unresolved':30})
        for row in updated:
            old=original[row['record_id']]
            for key in ('description','tool_name','domain','vendor','source_is_ai_tool'):
                self.assertEqual(row[key],old[key])

    def test_reconciled_splits_reproduce_and_account_for_every_input(self):
        out=ROOT/'data/processed/classification_reconciled_2026-11-11'
        files,manifest=build(release.OUT/'all_corrected.csv','reconciled-full-2026-11-11',20261111)
        for name,value in files.items():self.assertEqual((out/name).read_bytes(),value)
        self.assertEqual(sum(x['rows'] for x in manifest['split_counts'].values())+manifest['excluded_rows'],17292)
        self.assertEqual(manifest['exclusions']['unresolved_label'],30)

    def test_category_training_taxonomy_and_fold_group_integrity(self):
        import numpy as np
        rows,excluded,groups=load_training(DATA,CATEGORIES)
        self.assertEqual((len(rows),len(excluded)),(306,391))
        y=np.array([r['category'] for r in rows]);covered=[]
        for tr,va in splits(y,groups,minimum_training_groups=2):
            self.assertFalse(set(groups[tr])&set(groups[va]))
            self.assertEqual(set(y[tr]),set(CATEGORIES));covered.extend(va.tolist())
        self.assertEqual(sorted(covered),list(range(306)))
        config=json.loads((OUT/'model_config.json').read_text())
        self.assertEqual(config['input_sha256'],sha(DATA))
        self.assertEqual(config['categories'],CATEGORIES)
        self.assertFalse(config['independent_final_test'])

    def test_explicit_unresolved_final_label_blocks_catalog_assumption(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);raw=folder/'raw.csv';known=folder/'known.csv'
            raw.write_text('Name,Website,Category\nTool,tool.example,Technology And Development\n')
            known.write_text('tool_id,tool_name,domain,is_ai_tool,category\nx,Tool,tool.example,,Technology And Development\n')
            index=load_catalog(raw,known)
            app=dict(application_id='a',application_name='Tool',normalized_name='tool',observed_endpoint='tool.example',event_count=1)
            result=match_applications([app],index)[0]
            self.assertEqual(result['status'],'ambiguous');self.assertIsNone(result['is_ai_tool'])

    def test_full_correction_overrides_raw_and_finalized_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);raw=folder/'raw.csv';known=folder/'known.csv';correction=folder/'correction.csv'
            raw.write_text('Name,Website,Category\nTool,tool.example,Technology And Development\n')
            known.write_text('tool_id,tool_name,domain,is_ai_tool,category\ntool_00001,Tool,tool.example,true,Technology And Development\n')
            correction.write_text('source_record_id,tool_name,original_tool_name,domain,is_ai_tool\ntool_00001,Tool,Tool,tool.example,0\n')
            index=load_catalog(raw,known,corrections=correction)
            app=dict(application_id='a',application_name='Tool',normalized_name='tool',observed_endpoint='tool.example',event_count=1)
            self.assertFalse(match_applications([app],index)[0]['is_ai_tool'])

    def test_category_taxonomy_mismatch_rejected_before_loading(self):
        from modeling.predict_categories import CategoryPredictor
        config=json.loads((OUT/'model_config.json').read_text());config['taxonomy']='legacy_12'
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'model_config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'taxonomy'):CategoryPredictor(tmp)

    def test_unseen_example_exercises_binary_category_and_risk(self):
        result=run('Example Enterprise',ROOT/'data/examples/unseen_activity.csv',
            descriptions=ROOT/'data/examples/unseen_descriptions.csv',
            model=ROOT/'artifacts/reconciled_binary_2026-11-11',category_model=OUT)
        app=next(r for r in result['applications'] if r['application_key']=='new-code-helper.example')
        self.assertEqual(app['detection']['method'],'description_model')
        self.assertEqual(app['category_source'],'experimental_model')
        self.assertIn(app['category'],CATEGORIES)
        self.assertTrue(app['purpose_prediction']['needs_review'])
        self.assertEqual(app['events'],2)
        self.assertTrue(any(r['rule']=='large_upload' for r in app['reasons']))
        # This checks routing, not an assertion that the category prediction is correct.

if __name__=='__main__':unittest.main()
