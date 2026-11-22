import csv
import io
import json
import unittest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch
from dashboard.workflow import product_reference,uploaded_run,identity,CATALOG_RESULTS
from dashboard.data import build_apps
from ai_discovery.jev import classify_reports

class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference=product_reference()
        cls.lookup={identity(r['application_name'],r['endpoint']):r for r in cls.reference['results']}
        with CATALOG_RESULTS.open(encoding='utf-8-sig',newline='') as f:cls.catalog=list(csv.DictReader(f))

    def upload(self,name,domain):
        text=io.StringIO();writer=csv.writer(text);writer.writerow(['application_name','domain','employee_id','department'])
        writer.writerow([name,domain,'private_employee','Finance'])
        return uploaded_run(text.getvalue().encode(),'Other Company',self.reference)

    def test_reviewed_categories_and_unseen_saved_evidence_survive(self):
        chat=self.lookup[identity('ChatGPT','chatgpt.com')]
        self.assertEqual(chat['decision_source'],'jev')
        self.assertEqual(chat['categorization']['category_source'],'human_review')
        docs=[r for r in self.reference['results'] if r['application_name'].casefold()=='documind ai']
        self.assertEqual(len(docs),1)
        self.assertEqual(docs[0]['categorization']['category_source'],'human_review')
        self.assertNotIn('governance',docs[0])

    def test_catalog_predictions_are_used_for_uploads_and_company_controls_are_not_copied(self):
        row=next(r for r in self.catalog if r['jev_label']=='ai' and r['disagrees_with_source']=='False' and not r['application_name'].startswith("'"))
        with patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('offline')):
            run=self.upload(row['application_name'],row['endpoint'])
        rec=run.raw['results'][0]
        self.assertEqual(rec['decision_source'],'jev');self.assertTrue(rec['is_ai_tool'])
        self.assertEqual(rec['observed_identity']['event_count'],1)
        self.assertIsNone(rec['governance']['evidence']['approved_status']['value'])
        self.assertEqual(rec['governance']['company'],'Other Company')
        self.assertNotIn('private_employee',json.dumps(run.raw))
        self.assertEqual(len(run.events),1)
        self.assertTrue(run.events['timestamp'].isna().all())

    def test_unknown_and_conflicting_predictions_are_not_forced_into_binary_labels(self):
        for kind in ('unknown','conflict'):
            row=next(r for r in self.catalog if (r['jev_label']=='unknown' if kind=='unknown' else r['disagrees_with_source']=='True') and not r['application_name'].startswith("'"))
            run=self.upload(row['application_name'],row['endpoint'])
            self.assertIsNone(run.raw['results'][0]['is_ai_tool'])
            if kind=='conflict':self.assertTrue(run.raw['results'][0]['catalog_detection_conflict'])
        run=self.upload('Different Product','chatgpt.com')
        self.assertIsNone(run.raw['results'][0]['is_ai_tool'])

    def test_unknown_risk_is_not_displayed_as_zero(self):
        frame=build_apps({'results':[{'application_id':'x','application_name':'Unknown','governance':{'risk_score':None,'risk_level':'unknown'}}]},'Company')
        self.assertIsNone(frame.iloc[0]['risk_score'])

    def test_classifier_cli_routes_to_jev_and_reuses_its_cache(self):
        from ai_discovery import classify,jev
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.json';out=root/'result.json';cache=root/'decisions.json'
            source.write_text(json.dumps({'version':1,'entries':{json.dumps(['New Tool','example.org/product']):{'report':{'report_blocks':[]}}}}))
            decision=dict(label='ai',identity_supported=True,reason_code='classified',reason='Jev test evidence',missing_information='',next_action='none')
            meta={'model':'jev-1.14.0','probabilities':{'ai':.9,'non_ai':.05,'unknown':.05}}
            argv=['classify','--provider','jev','--cache',str(source),'--output',str(out),'--decision-cache',str(cache)]
            with patch.object(sys,'argv',argv),patch.object(jev,'classify_reports',return_value=(decision,meta)) as selected,patch.object(classify,'classify',side_effect=AssertionError('Gemini classifier must not run')):
                classify.main();classify.main()
                self.assertEqual(selected.call_count,1)
            result=json.loads(out.read_text())['results'][0]
            self.assertEqual(result['decision_source'],'jev')
            self.assertEqual(result['probabilities']['ai'],.9)
            self.assertTrue(result['cache_hit'])

    def test_jev_cited_evidence_adapter_is_not_a_gemini_classifier(self):
        evidence=[{'report_blocks':[{'text':'AI-powered document analysis.', 'citations':[{'url':'https://example.com/product'}]}]}]
        sent=[]
        def sender(payload,key):
            sent.append(payload)
            return {'model':'jev-1.14.0','answers':{'ai_detection':{'choice':'ai','probabilities':{'ai':.9,'non_ai':.05,'unknown':.05}}},'usage':{'input_tokens':10}}
        result,meta=classify_reports({'application_name':'New Tool','endpoint':'example.com/product'},evidence,key='test',sender=sender)
        self.assertEqual(result['label'],'ai')
        self.assertIn('strict_endpoint',sent[0]['questions']['ai_detection']['instructions'])
        self.assertEqual(meta['probabilities']['ai'],.9)
        missing,_=classify_reports({'application_name':'New Tool','endpoint':'example.com'},[],key='test',sender=sender)
        self.assertEqual(missing['label'],'unknown');self.assertEqual(len(sent),1)

if __name__=='__main__':unittest.main()
