import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from governance.risk import POLICY,FIELDS,aggregate,assess,boolean,identity,metadata_index,retention,run

POL=json.loads(POLICY.read_text())
def event(**changes):
    r=dict(event_id='a',timestamp='2026-09-01 08:00:00',employee_id='e1',department='Engineering',application_name='Example',domain='example.com',bytes_uploaded='0',bytes_downloaded='0')
    return dict(r,**changes)
def usage(**changes):
    return dict(aggregate([event()])[0],**changes)
def controls(**changes):
    return dict(approved_status='approved',sso_available=True,data_retention='30_days',training_on_customer_data=False,admin_controls_available=True,enterprise_plan_available=True,**changes)

class GovernanceTests(unittest.TestCase):
    def test_unknown_does_not_mean_unapproved_or_no_sso(self):
        r=assess(usage(),{},POL)
        self.assertEqual(r['risk_score'],0)
        self.assertEqual(r['assessment_status'],'incomplete')
        self.assertEqual(set(r['unknown_fields']),set(FIELDS))
        self.assertTrue(r['needs_review'])
        self.assertFalse(r['reasons'])

    def test_explicit_controls_low_complete(self):
        r=assess(usage(),controls(),POL)
        self.assertEqual((r['risk_level'],r['assessment_status']),('low','complete'))
        self.assertFalse(r['needs_review'])

    def test_unapproved_sensitive_large_upload_high_and_explained(self):
        u=usage(departments=['Finance'],mean_bytes_uploaded=8_000_000,max_bytes_uploaded=12_000_000,unique_users=84)
        r=assess(u,{'approved_status':'unapproved'},POL)
        self.assertEqual(r['risk_score'],55)
        self.assertEqual(r['risk_level'],'high')
        self.assertEqual(sum(x['points'] for x in r['reasons']),55)
        self.assertEqual([x['rule'] for x in r['reasons']].count('large_upload'),1)
        self.assertTrue(all(x['evidence'] and x['explanation'] for x in r['reasons']))

    def test_medium_boundary_and_high_boundary(self):
        for n,expected in [(19,'low'),(20,'medium'),(49,'medium'),(50,'high')]:
            p=copy.deepcopy(POL);p['points']['unapproved']=n
            self.assertEqual(assess(usage(),{'approved_status':'unapproved'},p)['risk_level'],expected)

    def test_policy_signals_and_score_cap(self):
        raw=controls();raw.update(approved_status='unapproved',sso_available=False,training_on_customer_data=True,data_retention='indefinite',admin_controls_available=False,enterprise_plan_available=False)
        r=assess(usage(departments=['Finance'],unique_users=100,mean_bytes_uploaded=6_000_000,max_bytes_uploaded=11_000_000,mean_bytes_downloaded=12_000_000),raw,POL)
        self.assertEqual(r['risk_score'],100)
        self.assertEqual(sum(x['points'] for x in r['reasons']),115)

    def test_missing_uploads_remain_unknown(self):
        u=aggregate([event(bytes_uploaded='',bytes_downloaded='')])[0]
        self.assertIsNone(u['mean_bytes_uploaded'])
        r=assess(u,controls(),POL)
        self.assertIn('bytes_uploaded',r['unknown_fields'])
        self.assertEqual(r['assessment_status'],'incomplete')

    def test_invalid_duplicate_events_rejected(self):
        with self.assertRaises(ValueError):aggregate([event(),event()])
        for field,value in [('bytes_uploaded','-1'),('bytes_downloaded','NaN'),('timestamp','bad'),('employee_id','')]:
            with self.assertRaises(ValueError):aggregate([event(**{field:value})])

    def test_unique_users_and_partial_coverage(self):
        u=aggregate([event(),event(event_id='b',bytes_uploaded='')])[0]
        self.assertEqual(u['unique_users'],1)
        self.assertEqual(u['events'],2)
        self.assertIn('bytes_uploaded',assess(u,{},POL)['unknown_fields'])

    def test_application_paths_and_subdomains_remain_distinct(self):
        self.assertEqual(identity('https://WWW.GitHub.com/copilot/?x=y'),'github.com/copilot')
        self.assertNotEqual(identity('github.com'),identity('github.com/copilot'))
        self.assertNotEqual(identity('docs.google.com'),identity('drive.google.com'))
        self.assertEqual(len(aggregate([event(),event(event_id='b',domain='example.com/agent') ])),2)

    def test_ambiguous_metadata_never_arbitrarily_joined(self):
        self.assertEqual(metadata_index([{'domain':'example.com'},{'domain':'example.com'}]),{})
        self.assertNotIn('example.com/agent',metadata_index([{'domain':'example.com'}]))

    def test_strict_policy_values_and_retention(self):
        self.assertIsNone(boolean('unknown'))
        self.assertIsNone(boolean(0))
        self.assertFalse(boolean('False'))
        self.assertEqual(retention('30_days'),30)
        self.assertIsNone(retention('custom_policy'))
        self.assertIsNone(retention('none'))
        p=copy.deepcopy(POL);p['points']['no_sso']=-5
        with self.assertRaises(ValueError):assess(usage(),{},p)

    def test_approval_is_company_scoped_and_scenarios_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);activity=d/'activity.csv';metadata=d/'metadata.csv';fixture=d/'scenario.json'
            for path,rows in [(activity,[event()]),(metadata,[dict(domain='example.com',approved_status='unapproved')])]:
                with path.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            result=run('A',activity,metadata,POLICY)
            self.assertIsNone(result['applications'][0]['metadata']['approved_status'])
            fixture.write_text(json.dumps({'synthetic':True,'companies':{'A':{'example.com':{'approved_status':'unapproved'}},'B':{'example.com':{'approved_status':'approved'}}}}))
            a=run('A',activity,metadata,POLICY,fixture)['applications'][0]
            b=run('B',activity,metadata,POLICY,fixture)['applications'][0]
            self.assertEqual(a['risk_score']-b['risk_score'],25)
            fixture.write_text(json.dumps({'synthetic':False,'companies':{}}))
            with self.assertRaises(ValueError):run('A',activity,metadata,POLICY,fixture)

    def test_real_synthetic_shadow_ai_case(self):
        r=run('Summit Financial',ROOT/'data/raw/summit_financial_usage (1).csv',ROOT/'data/processed/dataset_b/dataset_b_master.csv',POLICY,ROOT/'governance/synthetic_scenarios.json')
        app=next(x for x in r['applications'] if x['application_key']=='documind-ai.com')
        self.assertEqual(app['events'],120)
        self.assertEqual(app['unique_users'],84)
        self.assertFalse(app['metadata_match'])
        self.assertEqual(app['risk_level'],'high')
        self.assertIn('sso_available',app['unknown_fields'])

if __name__=='__main__':unittest.main()
