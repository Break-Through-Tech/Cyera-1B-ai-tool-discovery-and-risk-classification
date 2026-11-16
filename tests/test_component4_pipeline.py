import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from ai_discovery.ingest import extract_applications
from governance.risk import POLICY, assess_records


class Component4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'events.csv'
        self.policy = json.loads(POLICY.read_text())
        self.load([
            {'application_name':'Tool','domain':'example.com','employee_id':'one','department':'Finance',
             'bytes_uploaded':'12000000','bytes_downloaded':'0','event_id':'duplicate'},
            {'application_name':'Tool','domain':'example.com','employee_id':'','department':'',
             'bytes_uploaded':'bad','bytes_downloaded':'','event_id':'duplicate'},
            {'application_name':'Other','domain':'example.com','employee_id':'two','department':'Engineering',
             'bytes_uploaded':'0','bytes_downloaded':'0','event_id':''},
            {'application_name':'Bad','domain':'invalid','employee_id':'','department':'',
             'bytes_uploaded':'1','bytes_downloaded':'0','event_id':''},
        ])

    def load(self, rows):
        with self.path.open('w', newline='') as stream:
            w = csv.DictWriter(stream, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        self.ingestion = extract_applications(self.path)
        self.records = [dict(application_id=a['application_id'], observed_identity=a,
            application_name=a['application_name'], endpoint=a['observed_endpoint'],
            classification_status='classified', identity_supported=True, is_ai_tool=True,
            categorization={'category':None}, risk_level='synthetic_high') for a in self.ingestion['applications']]
        self.key = next(r['application_id'] for r in self.records if r['application_name']=='Tool')

    def evaluate(self, fields=None, **kwargs):
        evidence = {'version':1,'applications':{self.key:fields or {}}}
        result = assess_records(self.records, self.ingestion, self.path, self.policy,
                                evidence=evidence, **kwargs)
        return next(r['governance'] for r in result if r['application_id']==self.key)

    def test_preserves_all_upstream_data_and_accepted_events(self):
        before = copy.deepcopy((self.records,self.ingestion))
        result = assess_records(self.records,self.ingestion,self.path,self.policy)
        self.assertEqual(before,(self.records,self.ingestion))
        self.assertEqual(len(result),2)
        for old,new in zip(self.records,result):
            self.assertEqual(old,{k:v for k,v in new.items() if k!='governance'})
        self.assertEqual(len(self.ingestion['event_map']),3)
        self.assertEqual(len(self.ingestion['rejected_rows']),1)

    def test_unknown_is_not_low_or_catalog_risk(self):
        r = self.evaluate()
        self.assertIsNone(r['risk_score'])
        self.assertEqual(r['risk_level'],'unknown')
        self.assertEqual(r['assessment_status'],'incomplete')
        self.assertEqual(r['reasons'],[])

    def test_synthetic_contributes_to_primary_prototype_score(self):
        r = self.evaluate({'sso_available':{'value':False,'provenance':'synthetic','source':'fixture'}},
                          activity_provenance='synthetic')
        self.assertEqual(r['risk_score'],35)
        self.assertTrue(r['contains_synthetic_evidence'])
        self.assertNotIn('synthetic_demonstration',r)

    def test_mixed_evidence_scores_all_supported_fields(self):
        r = self.evaluate({'sso_available':{'value':False,'provenance':'verified','source':'review'},
                          'admin_controls_available':{'value':False,'provenance':'synthetic','source':'fixture'}})
        self.assertEqual(r['risk_score'],20)

    def test_verified_usage_recovers_single_upload_max_and_missingness(self):
        r = self.evaluate(activity_provenance='verified')
        self.assertEqual(r['risk_score'],25)
        self.assertIn('large_upload',r['missing_evidence_rules'])
        reason = next(x for x in r['reasons'] if x['rule']=='large_upload')
        self.assertEqual(reason['evidence']['max_bytes_uploaded'],12000000)
        self.assertIn('broad_adoption',r['missing_evidence_rules'])

    def test_non_ai_checks_remain_general_and_training_not_applicable(self):
        for record in self.records:record['is_ai_tool']=False
        r = self.evaluate({'training_on_customer_data':{'value':True,'provenance':'verified','source':'review'}},
                          activity_provenance='verified')
        self.assertEqual(r['risk_score'],25)
        self.assertEqual(next(c['status'] for c in r['checks'] if c['rule']=='customer_data_training'),'not_applicable')
        self.assertNotIn('customer_data_training',r['missing_evidence_rules'])

    def test_unresolved_or_conflicted_ai_stays_pending(self):
        for change in ({'is_ai_tool':None},{'catalog_detection_conflict':True},
                       {'identity_supported':False},{'technical_error':True}):
            original = copy.deepcopy(self.records)
            for record in self.records:record.update(change)
            r = self.evaluate({'training_on_customer_data':{'value':True,'provenance':'verified','source':'review'}})
            self.assertEqual(next(c['status'] for c in r['checks'] if c['rule']=='customer_data_training'),'pending_ai_detection')
            self.assertIsNone(r['risk_score'])
            self.records=original

    def test_approval_requires_matching_company(self):
        field={'approved_status':{'value':'unapproved','provenance':'verified','source':'policy','company':'A'}}
        for company in (None,'B'):
            with self.assertRaisesRegex(ValueError,'company'):self.evaluate(field,company=company)
        self.assertEqual(self.evaluate(field,company='A')['risk_score'],25)

    def test_invalid_evidence_and_inherited_labels_rejected(self):
        for fields in ({'risk_level':{'value':'high'}}, {'sso_available':False},
                       {'sso_available':{'value':False,'provenance':'verified'}},
                       {'sso_available':{'value':'maybe','provenance':'verified','source':'review'}}):
            with self.assertRaises(ValueError):self.evaluate(fields)

    def test_stale_input_or_identity_rejected(self):
        self.path.write_text(self.path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'changed'):self.evaluate()

    def test_evidence_does_not_join_shared_hosts(self):
        result=assess_records(self.records,self.ingestion,self.path,self.policy,
            evidence={'version':1,'applications':{self.key:{'sso_available':{
                'value':False,'provenance':'verified','source':'review'}}}})
        other=next(r for r in result if r['application_id']!=self.key)
        self.assertIsNone(other['governance']['risk_score'])
        with self.assertRaisesRegex(ValueError,'identity'):
            assess_records(self.records,self.ingestion,self.path,self.policy,
                evidence={'version':1,'applications':{'example.com':{}}})

    def test_grammarly_warning_does_not_change_category(self):
        app=next(a for a in self.ingestion['applications'] if a['application_id']==self.key)
        app['normalized_name']='grammarly'
        record=next(r for r in self.records if r['application_id']==self.key)
        record['categorization']={'category':'Security And Privacy'}
        self.assertTrue(self.evaluate()['data_quality_warnings'])
        self.assertEqual(record['categorization']['category'],'Security And Privacy')

    def test_endpoint_variants_are_never_collapsed(self):
        endpoints=['example.com/A','example.com/a','example.com:8443/A',
                   'example.com/A?x=1','example.com/A?x=2','example.com/A#part']
        self.load([dict(application_name='Tool',domain=e) for e in endpoints])
        result=assess_records(self.records,self.ingestion,self.path,self.policy)
        self.assertEqual(len(result),len(endpoints))
        self.assertEqual({r['observed_identity']['observed_endpoint'] for r in result},set(endpoints))

    def test_complete_non_ai_does_not_require_training_evidence(self):
        self.load([dict(application_name='Tool',domain='example.com',employee_id='one',
                        department='Engineering',bytes_uploaded='0',bytes_downloaded='0')])
        self.records[0]['is_ai_tool']=False
        fields={k:dict(value=v,provenance='verified',source='review',company='A') for k,v in
                dict(approved_status='approved',sso_available=True,data_retention='30_days',
                     admin_controls_available=True,enterprise_plan_available=True).items()}
        r=self.evaluate(fields,company='A',activity_provenance='verified')
        self.assertEqual((r['risk_score'],r['risk_level'],r['assessment_status']),(0,'low','complete'))
        self.assertFalse(r['needs_review'])

    def test_researched_control_scores_without_claiming_verification(self):
        r=self.evaluate({'sso_available':{'value':False,'provenance':'researched',
                                         'source':'https://example.com/security','plan_context':'Basic'}})
        self.assertEqual(r['risk_score'],10)
        self.assertEqual(r['reasons'][0]['evidence_provenance'],'researched')

    def test_under_review_is_retained_and_not_treated_as_approved(self):
        r=self.evaluate({'approved_status':{'value':'under_review','provenance':'synthetic',
                                            'source':'fixture','company':'A'}},company='A',
                        activity_provenance='synthetic')
        self.assertEqual(r['evidence']['approved_status']['value'],'under_review')
        self.assertIn('unapproved',r['missing_evidence_rules'])
        self.assertNotIn('unapproved',[x['rule'] for x in r['reasons']])

    def test_plan_dependent_researched_policy_cannot_bypass_scoring_gate(self):
        fields={'training_on_customer_data':{'value':False,'provenance':'researched',
            'source':'https://example.com/policy','policy_scope':'specific_plans',
            'plan_context':'Enterprise','applicable_plans':['Enterprise']}}
        r=self.evaluate(fields,company='A',activity_provenance='synthetic')
        self.assertEqual(r['evidence']['training_on_customer_data']['provenance'],'unknown')
        self.assertIn('customer_data_training',r['missing_evidence_rules'])
        fields['training_on_customer_data']['company_plan']={'plan':'Enterprise','company':'B','source':'scenario','provenance':'synthetic'}
        r=self.evaluate(fields,company='A',activity_provenance='synthetic')
        self.assertEqual(r['evidence']['training_on_customer_data']['provenance'],'unknown')

    def test_matched_synthetic_company_plan_and_all_plan_policy(self):
        field={'value':True,'provenance':'researched','source':'https://example.com/policy',
            'policy_scope':'specific_plans','plan_context':'Basic','applicable_plans':['Basic'],
            'company_plan':{'plan':'Basic','company':'A','source':'scenario','provenance':'synthetic'}}
        r=self.evaluate({'training_on_customer_data':field},company='A')
        self.assertEqual(r['risk_score'],20)
        self.assertTrue(r['contains_synthetic_evidence'])
        field.pop('company_plan');field['policy_scope']='all_plans'
        self.assertEqual(self.evaluate({'training_on_customer_data':field})['risk_score'],20)

if __name__ == '__main__':unittest.main()
