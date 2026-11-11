import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from ai_discovery.report import run, ROOT


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

if __name__=='__main__':unittest.main()
