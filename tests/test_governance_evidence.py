import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from governance.evidence import assemble
from governance.research import accept_fields, enrich_missing


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.catalog=self.root/'catalog.csv'
        self.records=[dict(application_id='one',observed_identity=dict(application_name='Tool',
            normalized_name='tool',observed_endpoint='example.com/A'),is_ai_tool=False,
            classification_status='classified',identity_supported=True)]
        self.write([dict(tool_name='Tool',domain='example.com/A',sso_available='false',
                         approved_status='approved',risk_label='low')])
    def write(self,rows):
        with self.catalog.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def test_catalog_controls_not_labels_and_approval_opt_in(self):
        e,w=assemble(self.records,'A',[self.catalog],'synthetic')
        self.assertFalse(e['applications']['one']['sso_available']['value'])
        self.assertNotIn('risk_label',e['applications']['one'])
        self.assertNotIn('approved_status',e['applications']['one'])
        e,w=assemble(self.records,'A',[self.catalog],'synthetic','A')
        self.assertEqual(e['applications']['one']['approved_status']['company'],'A')
        with self.assertRaises(ValueError):assemble(self.records,'B',[self.catalog],'synthetic','A')
    def test_name_and_full_endpoint_match(self):
        for name,endpoint in [('other','example.com/A'),('tool','example.com/a'),('tool','example.com'),('tool','example.com/A?x=1')]:
            self.records[0]['observed_identity'].update(normalized_name=name,observed_endpoint=endpoint)
            self.assertEqual(assemble(self.records,'A',[self.catalog],'synthetic')[0]['applications']['one'],{})
    def test_conflicting_catalogs_do_not_choose_arbitrarily(self):
        self.write([dict(tool_name='Tool',domain='example.com/A',sso_available=v) for v in ['true','false']])
        e,w=assemble(self.records,'A',[self.catalog],'synthetic')
        self.assertEqual(e['applications']['one']['sso_available']['provenance'],'unknown')
        self.assertEqual(w[0]['reason'],'conflicting_catalog_values')
    def test_scenarios_then_explicit_evidence_override(self):
        scenario=self.root/'scenario.json'
        scenario.write_text(json.dumps({'synthetic':True,'companies':{'A':{'example.com/A':{'sso_available':True}}}}))
        e,w=assemble(self.records,'A',[self.catalog],'synthetic',scenarios=scenario)
        self.assertTrue(e['applications']['one']['sso_available']['value'])
        override={'version':1,'applications':{'one':{'sso_available':{'value':False,'provenance':'verified','source':'review'}}}}
        e,w=assemble(self.records,'A',[self.catalog],'synthetic',scenarios=scenario,overrides=override)
        self.assertEqual(e['applications']['one']['sso_available'],override['applications']['one']['sso_available'])
    def test_shared_endpoint_scenario_not_applied(self):
        self.records.append(dict(self.records[0],application_id='two'))
        scenario=self.root/'scenario.json'
        scenario.write_text(json.dumps({'synthetic':True,'companies':{'A':{'example.com/A':{'approved_status':'approved'}}}}))
        e,w=assemble(self.records,'A',scenarios=scenario)
        self.assertEqual(e['applications'],{'one':{},'two':{}})
        self.assertTrue(w)
    def test_research_requires_citation_identity_and_plan(self):
        blocks=[{'text':'SSO','citations':[{'url':'https://example.com/security'}]}]
        payload={'identity_matches':True,'fields':{'sso_available':{'status':'known','value':True,'source_blocks':[0],'plan_context':'Enterprise'}}}
        self.assertEqual(accept_fields(payload,blocks,['sso_available'])['sso_available']['provenance'],'researched')
        self.assertEqual(accept_fields(dict(payload,identity_matches=False),blocks,['sso_available']),{})
        self.assertEqual(accept_fields(payload,[{'citations':[]}],['sso_available']),{})
        payload['fields']['sso_available']['source_blocks']=[9]
        self.assertEqual(accept_fields(payload,blocks,['sso_available']),{})
    def test_online_missing_fields_cache_and_non_ai(self):
        calls=[]
        def create(**kwargs):
            calls.append(kwargs)
            if 'tools' in kwargs:
                self.assertNotIn('training_on_customer_data',kwargs['input'])
                value={'steps':[{'type':'model_output','content':[{'type':'text','text':'SSO supported',
                        'annotations':[{'type':'url_citation','url':'https://example.com/security'}]}]}]}
            else:
                value={'steps':[{'type':'model_output','content':[{'type':'text','text':json.dumps({
                    'identity_matches':True,'fields':{'sso_available':{'status':'known','value':True,'source_blocks':[0],'plan_context':'Enterprise'},**{f:{'status':'unknown','reason':'Not documented'} for f in ('data_retention','admin_controls_available','enterprise_plan_available')}}})}]}]}
            return SimpleNamespace(model_dump=lambda **kw:value)
        client=SimpleNamespace(interactions=SimpleNamespace(create=create),close=lambda:None)
        initial={'version':1,'applications':{'one':{}}}
        e,out,cache=enrich_missing(self.records,initial,'test',client_factory=lambda:client)
        self.assertEqual(len(calls),2)
        self.assertTrue(e['applications']['one']['sso_available']['value'])
        self.assertEqual(initial['applications']['one'],{})
        path=self.root/'cache.json';path.write_text(json.dumps(cache))
        e,out,cache=enrich_missing(self.records,initial,'test',path,client_factory=lambda:self.fail('cache miss'))
        self.assertEqual(out[0]['status'],'cache_hit')
    def test_online_failure_preserves_unknown(self):
        def fail():raise RuntimeError('network failure')
        initial={'version':1,'applications':{'one':{}}}
        e,out,cache=enrich_missing(self.records,initial,'test',client_factory=fail)
        self.assertEqual(e,initial)
        self.assertEqual(out[0]['status'],'research_failed')
        self.assertEqual(cache['entries'],{})


    def test_timeout_is_passed_and_progress_reports_failure(self):
        calls=[];logs=[]
        def create(**kwargs):
            calls.append(kwargs);raise TimeoutError('simulated network timeout')
        client=SimpleNamespace(interactions=SimpleNamespace(create=create),close=lambda:None)
        e,out,cache=enrich_missing(self.records,{'version':1,'applications':{'one':{}}},'test',
            client_factory=lambda:client,timeout_seconds=7,progress=logs.append)
        self.assertEqual(calls[0]['timeout'],7)
        self.assertEqual(out[0]['stage'],'search')
        self.assertEqual(out[0]['error_type'],'TimeoutError')
        self.assertTrue(any('starting' in x for x in logs))
        self.assertTrue(any('failed' in x for x in logs))

    def test_interrupted_extraction_resumes_saved_search(self):
        checkpoint=self.root/'checkpoint.json';calls=[]
        search={'steps':[{'type':'model_output','content':[{'type':'text','text':'SSO supported',
                'annotations':[{'type':'url_citation','url':'https://example.com/security'}]}]}]}
        extracted={'steps':[{'type':'model_output','content':[{'type':'text','text':json.dumps({
            'identity_matches':True,'fields':{'sso_available':{'status':'known','value':True,'source_blocks':[0],'plan_context':'Enterprise'},**{f:{'status':'unknown','reason':'Not documented'} for f in ('data_retention','admin_controls_available','enterprise_plan_available')}}})}]}]}
        def first(**kwargs):
            calls.append(kwargs)
            if 'tools' not in kwargs:raise KeyboardInterrupt()
            return SimpleNamespace(model_dump=lambda **kw:search)
        client=SimpleNamespace(interactions=SimpleNamespace(create=first),close=lambda:None)
        evidence={'version':1,'applications':{'one':{}}}
        with self.assertRaises(KeyboardInterrupt):
            enrich_missing(self.records,evidence,'test',client_factory=lambda:client,
                           checkpoint_path=checkpoint,progress=lambda x:None)
        stored=json.loads(checkpoint.read_text())
        self.assertEqual(len(stored['entries']),1)
        self.assertNotIn('extraction',next(iter(stored['entries'].values())))
        def second(**kwargs):
            self.assertNotIn('tools',kwargs)
            return SimpleNamespace(model_dump=lambda **kw:extracted)
        client.interactions.create=second
        e,out,cache=enrich_missing(self.records,evidence,'test',client_factory=lambda:client,
                                  checkpoint_path=checkpoint,progress=lambda x:None)
        self.assertTrue(e['applications']['one']['sso_available']['value'])
        saved=json.loads(checkpoint.read_text())
        self.assertIn('extraction',next(iter(saved['entries'].values())))
        e,out,cache=enrich_missing(self.records,evidence,'test',checkpoint_path=checkpoint,
            client_factory=lambda:self.fail('Completed checkpoint should avoid API calls'),progress=lambda x:None)
        self.assertEqual(out[0]['status'],'cache_hit')

    def test_checkpoint_refuses_unrelated_existing_file(self):
        checkpoint=self.root/'other.json';checkpoint.write_text('{"version":1,"entries":{}}')
        before=checkpoint.read_bytes()
        with self.assertRaisesRegex(ValueError,'non-governance'):
            enrich_missing(self.records,{'version':1,'applications':{'one':{}}},'test',checkpoint_path=checkpoint)
        self.assertEqual(checkpoint.read_bytes(),before)

    def test_limit_defers_other_apps(self):
        import copy
        records=copy.deepcopy(self.records)*2
        records[1]=copy.deepcopy(records[1]);records[1]['application_id']='two'
        records[1]['observed_identity']['application_name']='Other'
        evidence={'version':1,'applications':{'one':{},'two':{}}}
        def fail():raise TimeoutError('offline')
        e,out,cache=enrich_missing(records,evidence,'test',max_apps=1,client_factory=fail,progress=lambda x:None)
        self.assertEqual([r['status'] for r in out],['research_failed','deferred_by_limit'])

    def test_invalid_timeout_and_limit_rejected_before_network(self):
        for args in ({'timeout_seconds':0},{'timeout_seconds':float('nan')},{'max_apps':0}):
            with self.assertRaises(ValueError):
                enrich_missing(self.records,{'version':1,'applications':{'one':{}}},'test',**args)


if __name__=='__main__':unittest.main()
