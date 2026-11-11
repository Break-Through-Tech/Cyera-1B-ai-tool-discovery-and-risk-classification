import unittest
import csv
import json
from pathlib import Path
import tempfile
from ai_discovery.report import run, ROOT
from ai_discovery.catalog import match_applications


class DiscoveryTests(unittest.TestCase):
    def test_summit_preserves_unknown_and_risk(self):
        result = run('Summit Financial',ROOT/'data/raw/summit_financial_usage (1).csv')
        apps = {a['application_key']:a for a in result['applications']}
        doc = apps['documind-ai.com']
        self.assertEqual(doc['risk_score'],30)
        self.assertIsNone(doc['detection']['is_ai_tool'])
        self.assertIn(doc['application_id'],result['review_queue'])
        self.assertEqual(sum(result['detection_counts'].values()),22)
        self.assertNotIn('S0001',str(result))
        self.assertEqual(len(result['ingestion']['event_map']),25120)

    def test_company_overlay_is_explicit_and_scoped(self):
        result = run('Summit Financial',ROOT/'data/raw/summit_financial_usage (1).csv',
                     scenarios=ROOT/'governance/synthetic_scenarios.json')
        doc = next(a for a in result['applications'] if a['application_key']=='documind-ai.com')
        self.assertEqual(doc['risk_score'],55)
        self.assertEqual(doc['metadata_source'],'synthetic_company_scenario')


class SharedDiscoveryTests(unittest.TestCase):
    def test_retains_rejections_duplicate_events_and_missingness(self):
        rows=[dict(event_id='same',application_name='Unknown',domain='unknown.example',employee_id='e1',department='Finance',bytes_uploaded='12000000',timestamp='bad'),
              dict(event_id='same',application_name='Unknown',domain='unknown.example',employee_id='',department='',bytes_uploaded='invalid',timestamp=''),
              dict(event_id='bad',application_name='',domain='invalid',employee_id='e2',department='Legal',bytes_uploaded='0',timestamp='')]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'activity.csv'
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            result=run('Example',p)
        app=result['applications'][0]
        self.assertEqual(app['events'],2)
        self.assertEqual(result['repeated_nonempty_event_ids'],1)
        self.assertEqual(result['ingestion']['summary']['rejected_events'],1)
        self.assertEqual(len(result['ingestion']['event_map']),2)
        self.assertEqual(app['upload_observations'],1)
        self.assertEqual(app['max_bytes_uploaded'],12000000)
        self.assertTrue({'bytes_uploaded','employee_id','department','timestamp'}<=set(app['unknown_fields']))
        self.assertIsNone(app['detection']['is_ai_tool'])

    def test_same_endpoint_different_names_does_not_merge_or_apply_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'activity.csv'
            p.write_text('application_name,domain\nProduct A,shared.example\nProduct B,shared.example\n')
            scenario=Path(tmp)/'scenario.json'
            scenario.write_text(json.dumps({'synthetic':True,'companies':{'Example':{'shared.example':{'approved_status':'unapproved'}}}}))
            result=run('Example',p,scenarios=scenario)
        self.assertEqual(result['application_count'],2)
        self.assertTrue(all(a['metadata']['approved_status'] is None for a in result['applications']))

    def test_optional_description_uses_frozen_binary_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'activity.csv';p.write_text('application_name,domain\nNew Product,new-product.example\n')
            d=Path(tmp)/'descriptions.csv';d.write_text('domain,description\nnew-product.example,AI coding assistant generates source code from natural language instructions\n')
            result=run('Example',p,descriptions=d,model=ROOT/'artifacts/corrected_full_binary')
        app=result['applications'][0]
        self.assertEqual(app['detection']['method'],'description_model')
        self.assertEqual(app['detection']['is_ai_tool'],1)
        self.assertIsNone(app['category'])
        self.assertIn('model_sha256',result)


class CatalogIdentityTests(unittest.TestCase):
    def record(self,name,label,path=''):
        return dict(tool_id=name,tool_name=name,name_key=name.casefold(),domain='github.com',
                    authority='github.com',path=path,is_ai_tool=label,category='Coding',source='finalized')

    def app(self,name,endpoint):
        return dict(application_id='app',application_name=name,normalized_name=name.casefold(),
                    observed_endpoint=endpoint,event_count=1)

    def test_shared_host_does_not_inherit_feature_label(self):
        row=self.record('Copilot',True,'/copilot')
        catalog={'by_host':{'github.com':[row]},'by_name':{'copilot':[row]}}
        root=match_applications([self.app('GitHub','github.com')],catalog)[0]
        feature=match_applications([self.app('Copilot','github.com/copilot')],catalog)[0]
        self.assertIsNone(root['is_ai_tool'])
        self.assertTrue(feature['is_ai_tool'])

    def test_conflicting_exact_identity_stays_ambiguous(self):
        rows=[self.record('Tool',True),self.record('Tool',False)]
        result=match_applications([self.app('Tool','github.com')],
                                  {'by_host':{'github.com':rows},'by_name':{'tool':rows}})[0]
        self.assertEqual(result['status'],'ambiguous')
        self.assertIsNone(result['is_ai_tool'])
