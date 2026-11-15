"""Standard-library integration guards; run python -m unittest discover -s tests -p test_component3_pipeline.py."""
import argparse
import copy
import csv
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from ai_discovery import categorize as c
from ai_discovery.run_pipeline import index_discovery, merge_records, online_discovery


class StubEngine(c.Engine):
    calls = 0
    def predict(self, text):
        self.calls += 1
        return {'category': 'Technology And Development', 'top_three': [], 'reason': 'test_double'}


class CategorizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = self.root/'catalog.csv'
        self.known = self.root/'known.csv'
        self.write_catalog(self.catalog, [])
        self.write_catalog(self.known, [])
        self.args = argparse.Namespace(catalog=self.catalog, known_tools=self.known, catalog_context=None,
            cache=self.root/'cache.json', model_dir=self.root/'model', model='winner', encoder_dir=None,
            prep_script=c.ROOT/'preprocessing_scripts/prepare_category_data.py',
            training_script=c.ROOT/'category_training/train_categories.py', reviews=None,
            report_selections=None, out=self.root/'out')
        prep = c.helpers.import_file(self.args.prep_script, 'test_prep_contract')
        self.args.model_dir.mkdir()
        (self.args.model_dir/'winner.joblib').write_bytes(b'test sentinel; NEVER deserialize')
        c.helpers.write_json(self.args.model_dir/'taxonomy.json',
            {'categories': prep.TAXONOMY, 'version': 'original_kaggle_14'})
        c.helpers.write_json(self.args.model_dir/'manifest.json', {'run_id': 'test',
            'config': {'text_contract_version': prep.TEXT_VERSION, 'packages': {'scikit-learn': '1.7.2'}},
            'artifacts': {n:c.helpers.sha(self.args.model_dir/n) for n in ['winner.joblib', 'taxonomy.json']}})
        self.record = {'application_name': 'Code Tool', 'endpoint': 'example.com',
            'is_ai_tool': True, 'classification_status': 'classified', 'identity_supported': True,
            'decision_source': 'human_review', 'reports': [{'report_blocks': [{'text':
                '## Product Description\nAn editor for writing and debugging computer programs.\n'
                '## AI Functionality\nGenerates code and tests from natural language instructions.\n'
                '## Uncertainty\nNone.', 'citations': []}]}]}

    def write_catalog(self, path, rows):
        with path.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['tool_id','tool_name','domain','category','is_ai_tool'])
            w.writeheader(); w.writerows(rows)

    def row(self, **changes):
        return dict({'tool_id':'x', 'tool_name':'Code Tool', 'domain':'example.com',
            'category':'Technology And Development', 'is_ai_tool':'true'}, **changes)

    def test_catalog_reuse_without_loading_model(self):
        self.write_catalog(self.catalog, [self.row()])
        e = StubEngine(self.args)
        result = e.categorize(self.record)
        self.assertEqual(result['category_source'], 'catalog')
        self.assertEqual(e.calls, 0)
        self.assertEqual(result['catalog_records'][0]['source_sha256'], c.helpers.sha(self.catalog))

    def test_non_ai_unresolved_and_unsupported_identity_never_predict(self):
        for changes, status in [({'is_ai_tool':False}, 'not_applicable'),
                ({'classification_status':'unresolved'}, 'pending_ai_detection'),
                ({'identity_supported':False}, 'pending_identity_review'),
                ({'technical_error':True}, 'pending_ai_detection')]:
            e = StubEngine(self.args)
            result = e.categorize(dict(self.record, **changes))
            self.assertEqual(result['category_status'], status)
            self.assertIsNone(result['category'])
            self.assertEqual(e.calls, 0)

    def test_shared_host_and_fuzzy_name_are_not_catalog_identity(self):
        self.write_catalog(self.catalog, [self.row(tool_name='Code Tool Pro')])
        e = StubEngine(self.args)
        self.assertEqual(e.categorize(self.record)['category_source'], 'model')
        self.assertEqual(e.calls, 1)

    def test_path_mismatch_not_catalog_identity(self):
        self.write_catalog(self.catalog, [self.row(domain='example.com/product')])
        e = StubEngine(self.args)
        self.assertEqual(e.categorize(self.record)['category_source'], 'model')

    def test_conflicting_or_invalid_catalog_categories_stay_pending(self):
        for rows in [[self.row(), self.row(tool_id='y',category='Media And Content Creation')],
                     [self.row(category='Old 12 Category')]]:
            self.write_catalog(self.catalog, rows)
            e = StubEngine(self.args)
            self.assertEqual(e.categorize(self.record)['category_status'], 'pending_catalog_category_review')
            self.assertEqual(e.calls, 0)

    def test_finalized_non_ai_conflict_preserves_original_detection(self):
        self.write_catalog(self.catalog, [self.row()])
        self.write_catalog(self.known, [self.row(is_ai_tool='false')])
        e = StubEngine(self.args)
        before = copy.deepcopy(self.record)
        self.assertEqual(e.categorize(self.record)['category_status'], 'pending_detection_conflict')
        self.assertEqual(self.record, before)

    def test_cache_reuse_and_evidence_invalidation(self):
        e = StubEngine(self.args)
        first = e.categorize(self.record)
        self.assertEqual(first['category_status'], 'provisional_prediction')
        self.assertFalse(first['cache_hit'])
        self.assertTrue(e.categorize(self.record)['cache_hit'])
        self.assertEqual(e.calls, 1)
        changed = copy.deepcopy(self.record)
        changed['reports'][0]['report_blocks'][0]['text'] += '\nChanged source evidence.'
        self.assertFalse(e.categorize(changed)['cache_hit'])
        self.assertEqual(e.calls, 2)
        e.save_cache()
        self.assertTrue(StubEngine(self.args).categorize(self.record)['cache_hit'])

    def test_model_and_catalog_change_invalidate_cache(self):
        e = StubEngine(self.args); e.categorize(self.record); e.save_cache()
        self.write_catalog(self.catalog, [self.row(tool_name='Different Tool')])
        changed = StubEngine(self.args)
        self.assertFalse(changed.categorize(self.record)['cache_hit'])
        (self.args.model_dir/'winner.joblib').write_bytes(b'changed model')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            StubEngine(self.args).categorize(self.record)

    def test_missing_multiple_and_selected_reports(self):
        e = StubEngine(self.args)
        self.assertEqual(e.categorize(dict(self.record,reports=[]))['category_status'], 'missing_evidence')
        record = dict(self.record,reports=self.record['reports']*2)
        self.assertEqual(e.categorize(record)['category_status'], 'needs_report_selection')
        with self.assertRaisesRegex(ValueError, 'stale'):
            e.categorize(record, selection=(0,'stale'))
        result = e.categorize(record, selection=(1,c.helpers.fingerprint(record['reports'])))
        self.assertEqual(result['category_source'], 'model')

    def test_human_review_bound_to_evidence(self):
        e = StubEngine(self.args)
        review = {'reviewed_category':'Technology And Development','reviewer':'Tester','reason':'Reviewed purpose',
                  'evidence_sha256':c.helpers.fingerprint(self.record['reports'])}
        self.assertEqual(e.categorize(self.record,review)['category_source'], 'human_review')
        with self.assertRaisesRegex(ValueError,'stale'):
            e.categorize(self.record,dict(review,evidence_sha256='stale'))
        self.assertEqual(e.calls,0)

    def test_pipeline_join_keeps_unknown_apps_and_activity(self):
        e = StubEngine(self.args)
        prior = index_discovery({'results':[self.record]},e.prep)
        apps = [{'application_id':'a','application_name':'Code Tool','observed_endpoint':'example.com','event_count':5},
                {'application_id':'b','application_name':'Other Tool','observed_endpoint':'example.com','event_count':2}]
        result = merge_records(apps, prior, e)
        self.assertTrue(result[0]['is_ai_tool'])
        self.assertEqual(result[0]['observed_identity']['event_count'],5)
        self.assertIsNone(result[1]['is_ai_tool'])
        self.assertEqual(len(result),2)

    def test_query_identity_remains_pending_without_merging_rows(self):
        e = StubEngine(self.args)
        records = [dict(self.record,endpoint='example.com?product='+p) for p in ['one','two']]
        result = c.categorize_records(records,e,self.args)
        self.assertEqual([r['categorization']['category_status'] for r in result],
                         ['pending_identity_review','pending_identity_review'])

    def test_input_output_protection(self):
        self.args.cache = self.catalog
        with self.assertRaises(ValueError):
            c.preflight(self.args)
        self.args.cache = self.root/'cache.json'
        self.args.out.mkdir()
        with self.assertRaises(ValueError):
            c.preflight(self.args)

    def test_offline_cli_preserves_activity_and_does_not_need_model(self):
        self.write_catalog(self.catalog, [self.row()])
        activity = self.root/'activity.csv'
        activity.write_text('application_name,domain,employee_id\nCode Tool,example.com,a\n'
                            'Code Tool,example.com,a\nUnknown,unknown.example,b\nBad,invalid,b\n',encoding='utf-8')
        result = subprocess.run([sys.executable,'-m','ai_discovery.run_pipeline',
            '--activity',str(activity),'--catalog',str(self.catalog),'--known-tools',str(self.known),
            '--out',str(self.args.out),'--cache',str(self.args.cache),
            '--model-dir',str(self.root/'nonexistent_model')],cwd=c.ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        report = c.helpers.read_json(self.args.out/'categorized_applications.json')
        self.assertEqual(len(report['results']),2)
        self.assertEqual(len(report['ingestion']['event_map']),3)
        self.assertEqual(len(report['ingestion']['rejected_rows']),1)
        self.assertEqual(report['summary'],{'assigned':1,'pending_ai_detection':1})
        self.assertFalse(report['online_enabled'])
        self.assertFalse(self.args.cache.exists())

    def test_online_targets_are_scoped_and_input_decision_cache_unchanged(self):
        self.write_catalog(self.catalog,[self.row()])
        e = StubEngine(self.args)
        self.args.enrichment_cache = self.root/'enrichment.json'
        self.args.decision_cache = self.root/'old_decisions.json'
        self.args.decision_cache.write_text('{"version":1,"entries":{}}')
        before = self.args.decision_cache.read_bytes()
        self.args.api_model = 'test-model'
        self.args.trust_application_names = False
        apps = [{'application_id':'known','application_name':'Code Tool','observed_endpoint':'example.com'},
                {'application_id':'new','application_name':'New Tool','observed_endpoint':'new.example'}]
        def key(r):
            return json.dumps([r['application_name'].lower(),r['observed_endpoint']],separators=(',',':'))
        class Cache:
            def __init__(self,path): self.entries={}
        def retrieve(records, api, model, cache, refresh):
            self.assertEqual([r['application_id'] for r in records],['new'])
            for r in records:
                cache.entries[key(r)]={'model':model,'report':{'report_blocks':[]}}
                yield {'application_id':r['application_id'],'enrichment_status':'cited_report_for_review'}
        fake = types.ModuleType('ai_discovery.enrich')
        fake.JsonCache=Cache; fake.cache_key=key; fake.enrich_applications=retrieve
        fake.public_identity=lambda r:{'application_name':r['application_name'],'endpoint':r['observed_endpoint']}
        def classify(command,**kwargs):
            output=Path(command[command.index('--output')+1])
            source=Path(command[command.index('--cache')+1])
            self.assertEqual(len(c.helpers.read_json(source)['entries']),1)
            c.helpers.write_json(output,{'results':[dict(self.record,application_name='New Tool',endpoint='new.example')]})
        stage=self.root/'stage';stage.mkdir()
        with patch.dict(sys.modules,{'ai_discovery.enrich':fake}), patch('ai_discovery.run_pipeline.subprocess.run',side_effect=classify):
            prior,outcomes=online_discovery(self.args,apps,e,{},stage)
        self.assertEqual(len(prior),1)
        self.assertEqual(len(outcomes),1)
        self.assertEqual(self.args.decision_cache.read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
