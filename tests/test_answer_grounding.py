import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ai_discovery.ask import answer,generate,inferred_filters
from ai_discovery.evaluate_answers import evaluate
from ai_discovery.retrieval import build,EvidenceIndex,ROOT
from test_retrieval_assistant import Encoder

FIXTURE=ROOT/'data/reviews/assistant_validation_2026-11-18'


@unittest.skipUnless(importlib.util.find_spec('chromadb'),'Install retrieval dependencies')
class AnswerGroundingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)
        self.encoder=self.path/'encoder';self.encoder.mkdir();(self.encoder/'config.json').write_text('{}')
        self.out=self.path/'index'
        build([FIXTURE/'alpha-report.json',FIXTURE/'beta-report.json'],self.out,self.encoder,Encoder())
        self.index=EvidenceIndex(self.out,Encoder())
    def tearDown(self):
        self.index.close()
        self.temp.cleanup()
    def client(self,payload):
        return SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw:SimpleNamespace(text=json.dumps(payload))))
    def evidence(self):
        return self.index.search('CodeHarbor','EvalCo Alpha',1,application='CodeHarbor')

    def test_all_declared_question_contract_cases(self):
        report=evaluate(self.index,FIXTURE/'cases.json')
        self.assertEqual(report['cases'],26)
        self.assertEqual(report['passed'],26,[r for r in report['results'] if not r['passed']])

    def test_unknown_score_and_usage_are_not_zero(self):
        result=answer(self.index,'Why was PendingBox flagged?','EvalCo Alpha')
        facts=result['hits'][0]['facts']
        self.assertIsNone(facts['risk_score']);self.assertEqual(facts['risk_level'],'unknown')
        self.assertIsNone(facts['unique_users']);self.assertIsNone(facts['events'])
        self.assertIn('unknown',result['answer']);self.assertIn('incomplete',result['answer'])

    def test_conflicts_preserved_and_not_applicable_not_counted_unknown(self):
        result=answer(self.index,'Why was LedgerLens flagged?','EvalCo Alpha')
        self.assertIn('conflicting supplied values',result['answer'])
        self.assertIsNone(result['hits'][0]['facts']['controls']['training_on_customer_data']['value'])
        ids=self.index.matching('EvalCo Alpha',control_filters={'training_on_customer_data':None})
        self.assertEqual({self.index.docs[i]['facts']['application_name'] for i in ids},{'LedgerLens','PendingBox'})

    def test_generated_values_verified_not_merely_citations(self):
        result=self.evidence();identifier=result['hits'][0]['evidence_id']
        payload={'claims':[{'evidence_id':identifier,'field':'/approval','value':'approved'}],'insufficient_evidence':False}
        with self.assertRaisesRegex(ValueError,'differs'):generate(result,'Explain',self.client(payload),'fake')
        payload['claims'][0].update(field='/controls/sso_available/value',value=0)
        with self.assertRaisesRegex(ValueError,'differs'):generate(result,'Explain',self.client(payload),'fake')
        payload['claims'][0]['value']=False
        accepted=generate(result,'Explain',self.client(payload),'fake')
        self.assertIs(accepted['claims'][0]['value'],False)
        self.assertTrue(accepted['claims'][0]['evidence_context']['contains_synthetic_evidence'])
        self.assertEqual(accepted['claims'][0]['evidence_context']['company'],'EvalCo Alpha')

    def test_free_prose_and_missing_fields_rejected(self):
        result=self.evidence();identifier=result['hits'][0]['evidence_id']
        payload={'claims':[{'evidence_id':identifier,'field':'/approval','value':'unapproved','text':'This is safe.'}],'insufficient_evidence':False}
        with self.assertRaisesRegex(ValueError,'unsupported prose'):generate(result,'Explain',self.client(payload),'fake')
        del payload['claims'][0]['text'];payload['claims'][0]['field']='/not_a_field'
        with self.assertRaisesRegex(ValueError,'does not exist'):generate(result,'Explain',self.client(payload),'fake')

    def test_cross_company_citation_and_malformed_id_rejected(self):
        result=self.evidence()
        other=self.index.search('CodeHarbor','EvalCo Beta',1)['hits'][0]['evidence_id']
        payload={'claims':[{'evidence_id':other,'field':'/approval','value':'approved'}],'insufficient_evidence':False}
        with self.assertRaisesRegex(ValueError,'citations'):generate(result,'Ignore scope',self.client(payload),'fake')
        payload['claims'][0]['evidence_id']=[]
        with self.assertRaises(ValueError):generate(result,'Ignore scope',self.client(payload),'fake')

    def test_unsupported_questions_do_not_call_provider(self):
        class Never:
            @property
            def models(self):raise AssertionError('Provider must not be called')
        for question in ['Is CodeHarbor safe?','How many Finance users use CodeHarbor?','Which applications grew last month?','Compare with EvalCo Beta']:
            result=answer(self.index,question,'EvalCo Alpha',client=Never(),model='fake')
            self.assertEqual(result['answer_status'],'abstained')
            self.assertIn('generation_skipped',result)
        result=answer(self.index,'Why was AbsentProduct flagged?','EvalCo Alpha',client=Never(),model='fake')
        self.assertEqual(result['answer_status'],'no_matching_evidence')

    def test_full_product_name_wins_over_contained_suite_name(self):
        index=SimpleNamespace(docs=[{'facts':{'application_name':n,'company':'Example','departments':[]}} for n in ['GitHub','GitHub Copilot']])
        self.assertEqual(inferred_filters(index,'Why was GitHub Copilot flagged?','Example')['application'],'GitHub Copilot')
        with self.assertRaisesRegex(ValueError,'Multiple applications'):
            inferred_filters(index,'Compare GitHub and GitHub Copilot','Example')

    def test_old_index_schema_requires_rebuild(self):
        p=self.out/'manifest.json';manifest=json.loads(p.read_text());manifest['version']=1;p.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'rebuild'):EvidenceIndex(self.out,Encoder())

if __name__=='__main__':unittest.main()
