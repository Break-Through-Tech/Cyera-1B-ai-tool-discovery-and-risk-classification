import copy
import hashlib
import json
from pathlib import Path
import tempfile
from datetime import datetime,timezone
from types import SimpleNamespace
import unittest
from governance.research import VERSION, EXTRACTION_VERSION, validate_fields, enrich_missing


class ExtractionRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.blocks=[{'text':'Enterprise supports SSO. Policy varies by plan.',
                      'citations':[{'url':'https://vendor.example/security'}]}]
        self.record={'application_id':'a','is_ai_tool':True,'classification_status':'classified','identity_supported':True,
                     'observed_identity':{'application_name':'Vendor','observed_endpoint':'vendor.example'}}
        self.evidence={'version':1,'applications':{'a':{f:{'value':True,'provenance':'synthetic','source':'fixture'}
            for f in ('admin_controls_available','enterprise_plan_available')}}}
        self.missing=['sso_available','data_retention','training_on_customer_data']
        self.identity={'name':'Vendor','endpoint':'vendor.example'}
        self.key=hashlib.sha256(json.dumps([VERSION,'test',self.identity,self.missing],sort_keys=True).encode()).hexdigest()
        self.payload={'identity_matches':True,'fields':{
            'sso_available':{'status':'known','value':True,'plan_context':'Enterprise','source_blocks':[0]},
            'data_retention':{'status':'unknown','reason':'Configurable; duration not supplied'},
            'training_on_customer_data':{'status':'known','value':False,'plan_context':'Enterprise','source_blocks':[0],
                'policy_scope':'specific_plans','applicable_plans':['Enterprise']}}}
    def check(self,payload=None,plan=None):
        return validate_fields(payload or self.payload,self.blocks,self.missing,plan)
    def test_policy_requires_matching_plan_but_capabilities_do_not(self):
        accepted,diags=self.check()
        self.assertIn('sso_available',accepted)
        self.assertNotIn('training_on_customer_data',accepted)
        self.assertEqual(diags[-1]['status'],'conditional')
    def test_specific_plan_matches_exactly(self):
        plan={'plan':'Enterprise','company':'A','provenance':'synthetic','source':'scenario'}
        self.assertIn('training_on_customer_data',self.check(plan=plan)[0])
        self.assertNotIn('training_on_customer_data',self.check(plan=dict(plan,plan='Enterprise Plus'))[0])
    def test_all_plans_policy_needs_no_company_plan(self):
        self.payload['fields']['training_on_customer_data']['policy_scope']='all_plans'
        self.assertIn('training_on_customer_data',self.check()[0])
    def test_missing_plan_context_is_diagnostic_not_silent(self):
        del self.payload['fields']['sso_available']['plan_context']
        a,d=self.check();self.assertEqual(d[0]['reason'],'missing_plan_context')
    def test_missing_scope_and_malformed_payload_are_rejected(self):
        del self.payload['fields']['training_on_customer_data']['policy_scope']
        self.assertEqual(self.check()[1][-1]['reason'],'missing_policy_scope')
        self.assertEqual(validate_fields([],self.blocks,self.missing)[1][0]['reason'],'invalid_response_schema')
    def test_unknown_field_does_not_trigger_schema_retry(self):
        self.assertEqual(self.check()[1][1]['status'],'unknown')
    def cache(self,payload):
        path=self.root/'old.json'
        path.write_text(json.dumps({'version':1,'entries':{self.key:{'created_at':datetime.now(timezone.utc).isoformat(),
            'identity':self.identity,'model':'test','report_blocks':self.blocks,'extraction':payload}}}))
        return path
    def test_legacy_cache_repaired_without_search_and_input_unchanged(self):
        old=copy.deepcopy(self.payload);del old['fields']['sso_available']['plan_context']
        path=self.cache(old);before=path.read_bytes();calls=[]
        def create(**kw):
            self.assertNotIn('tools',kw);calls.append(kw)
            response={'steps':[{'type':'model_output','content':[{'type':'text','text':json.dumps(self.payload)}]}]}
            return SimpleNamespace(model_dump=lambda **_:response)
        client=SimpleNamespace(interactions=SimpleNamespace(create=create),close=lambda:None)
        e,out,c=enrich_missing([self.record],self.evidence,'test',path,client_factory=lambda:client,
                              repair_only=True,progress=lambda _:None)
        self.assertEqual(len(calls),1);self.assertEqual(path.read_bytes(),before)
        self.assertEqual(c['entries'][self.key]['extraction_version'],EXTRACTION_VERSION)
        self.assertEqual(e['applications']['a']['training_on_customer_data']['provenance'],'unknown')
        self.assertIn('candidate',e['applications']['a']['training_on_customer_data'])
    def test_repair_only_never_searches_missing_cache(self):
        e,out,c=enrich_missing([self.record],self.evidence,'test',repair_only=True,
                              client_factory=lambda:self.fail('No network expected'),progress=lambda _:None)
        self.assertEqual(out[0]['status'],'missing_cached_research')
    def test_schema_repair_is_bounded_and_reports_rejections(self):
        path=self.cache(self.payload);calls=[]
        def create(**kw):
            calls.append(kw);self.assertNotIn('tools',kw)
            response={'steps':[{'type':'model_output','content':[{'type':'text','text':'not JSON'}]}]}
            return SimpleNamespace(model_dump=lambda **_:response)
        client=SimpleNamespace(interactions=SimpleNamespace(create=create),close=lambda:None)
        e,out,c=enrich_missing([self.record],self.evidence,'test',path,client_factory=lambda:client,
                              repair_only=True,progress=lambda _:None)
        self.assertEqual(len(calls),2)
        self.assertEqual(out[0]['status'],'extraction_incomplete')
        self.assertTrue(out[0]['diagnostics'])
    def test_new_cache_rechecks_plan_without_api(self):
        path=self.cache(self.payload)
        document=json.loads(path.read_text());document['entries'][self.key]['extraction_version']=EXTRACTION_VERSION
        path.write_text(json.dumps(document))
        plan={'plan':'Enterprise','company':'A','source':'scenario','provenance':'synthetic'}
        e,out,c=enrich_missing([self.record],self.evidence,'test',path,company_plans={'a':plan},
                              client_factory=lambda:self.fail('No calls for plan reevaluation'),progress=lambda _:None)
        self.assertEqual(out[0]['status'],'cache_hit')
        self.assertEqual(e['applications']['a']['training_on_customer_data']['company_plan'],plan)


if __name__=='__main__':unittest.main()
