import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ai_discovery import jev
from ai_discovery.dashboard_data import ACTIVITY,flattened


def response(choice='ai',p=.9):
    other=[k for k in jev.QUESTION['criteria'] if k!=choice]
    return {'model':'jev-test','answers':{'ai_detection':{'type':'choice','choice':choice,
        'probabilities':{choice:p,other[0]:(1-p)/2,other[1]:(1-p)/2}}},'usage':{'input_tokens':20}}

class JevTests(unittest.TestCase):
    def test_unique_public_evidence_has_no_usage_or_target_labels(self):
        states=jev.prepare(ACTIVITY)
        self.assertEqual(len(states),21)
        for app,state,sources in states:
            self.assertEqual(set(state),{'observed_application','catalog_descriptions','reports','sources'})
            self.assertEqual(set(state['observed_application']),{'application_name','endpoint'})
            self.assertNotIn('is_ai_tool',json.dumps(state))
            self.assertNotIn('approved_status',json.dumps(state))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'events.csv'
            path.write_text('application_name,domain,employee_id\nChatGPT,chatgpt.com,private_123\nChatGPT,chatgpt.com,private_456\n')
            prepared=jev.prepare(path)
            self.assertEqual(len(prepared),1)
            self.assertNotIn('private_',json.dumps(prepared[0][1]))

    def test_validation_rejects_invalid_probabilities(self):
        for value in [float('nan'),-1,2,True]:
            data=response();data['answers']['ai_detection']['probabilities']['ai']=value
            with self.assertRaises(jev.JevError):jev.validate(data)
        self.assertEqual(jev.validate(response())[0],'ai')

    def test_cache_prevents_repeat_calls_and_uncertainty_stays_unknown(self):
        app={'application_name':'Example','observed_endpoint':'example.com','application_id':'x'}
        state={'observed_application':{'application_name':'Example','endpoint':'example.com'}}
        with tempfile.TemporaryDirectory() as folder,patch.object(jev,'prepare',return_value=[(app,state,[])]):
            root=Path(folder);activity=root/'activity.csv';activity.write_text('unused input')
            calls=[]
            def sender(payload,key):calls.append(payload);return response(p=.6)
            report=jev.run(activity,root/'output.json',root/'cache.json',sender=sender,key='test-only')
            self.assertEqual(report['results'][0]['label'],'unknown')
            again=jev.run(activity,root/'output.json',root/'cache.json',sender=sender)
            self.assertEqual(len(calls),1)
            self.assertEqual(again['online_calls_made'],0)
            self.assertNotIn('test-only',(root/'output.json').read_text())

    def test_api_error_stops_without_cached_success_or_false_label(self):
        app={'application_name':'Example','observed_endpoint':'example.com','application_id':'x'}
        with tempfile.TemporaryDirectory() as folder,patch.object(jev,'prepare',return_value=[(app,{},[])]):
            root=Path(folder);activity=root/'activity.csv';activity.write_text('input')
            def failed(*args):raise jev.JevError('HTTP 401')
            with self.assertRaises(jev.JevError):jev.run(activity,root/'out.json',root/'cache.json',sender=failed,key='test')
            report=json.loads((root/'out.json').read_text())
            self.assertEqual(report['run_status'],'stopped_on_error')
            self.assertEqual(report['results'],[])
            self.assertFalse((root/'cache.json').exists())

    def test_completed_live_artifact_and_dashboard_agree(self):
        report=json.loads(jev.OUTPUT.read_text());analysis=json.loads(jev.ANALYSIS.read_text())
        self.assertEqual(report['run_status'],'complete')
        self.assertEqual(report['completed_applications'],report['expected_unique_applications'])
        self.assertEqual(report['summary'],{'ai':9,'non_ai':12})
        self.assertTrue(all(r['decision_source']=='jev' for r in analysis['results']))
        self.assertEqual(sum(r['ai_status']=='AI' for r in flattened(analysis)),9)
        self.assertTrue(all(r['governance']['contains_synthetic_evidence'] for r in analysis['results']))

if __name__=='__main__':unittest.main()
