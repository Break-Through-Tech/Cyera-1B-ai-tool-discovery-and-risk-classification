import unittest
from discovery.report import identify, run, ROOT


class DiscoveryTests(unittest.TestCase):
    def test_shared_domain_scope_is_unknown(self):
        rows = [{'tool_name':'Platform','is_ai_tool':'0','record_id':'a'},
                {'tool_name':'AI Add-on','is_ai_tool':'1','record_id':'b'}]
        result = identify('example.com', {'example.com':rows},
                          {'example.com':'AI description'}, lambda _: self.fail('Must not override ambiguity'))
        self.assertIsNone(result['is_ai_tool'])
        self.assertEqual(result['method'],'unresolved_catalog')

    def test_no_path_fallback_and_no_name_guessing(self):
        rows = [{'tool_name':'Copilot','is_ai_tool':'1','record_id':'a'}]
        result = identify('github.com', {'github.com/copilot':rows}, {}, None)
        self.assertIsNone(result['is_ai_tool'])

    def test_unseen_description_uses_model(self):
        result = identify('new.example', {}, {'new.example':'Concrete description'},
                          lambda text:dict(is_ai_tool=1,model_ai_score=.7,needs_review=True))
        self.assertEqual(result['method'],'description_model')
        self.assertEqual(result['is_ai_tool'],1)

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


if __name__ == '__main__':
    unittest.main()
