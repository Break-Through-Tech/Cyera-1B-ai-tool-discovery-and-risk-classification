import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

from ai_discovery.retrieval import build,documents,EvidenceIndex
from ai_discovery.ask import answer,generate

AVAILABLE=importlib.util.find_spec('faiss') is not None


class Encoder:
    def encode(self,texts,**kwargs):
        import numpy as np
        result=np.zeros((len(texts),32),dtype='float32')
        for i,text in enumerate(texts):
            for word in re.findall(r'\w+',text.casefold()):result[i,int(hashlib.sha256(word.encode()).hexdigest()[:8],16)%32]+=1
        return result


def app(name,company,score=60,approval='unapproved',ai=False):
    return dict(application_id='id_'+name,application_name=name,endpoint=name.lower()+'.example',is_ai_tool=ai,
        decision_source='catalog',observed_identity=dict(event_count=5,unique_employee_count=3,departments='["Finance", "Engineering"]'),
        categorization=dict(category='Technology And Development' if ai else None,category_source='model' if ai else None,category_status='provisional'),
        governance=dict(company=company,risk_score=score,risk_level='high' if score>=50 else 'low',
            assessment_status='incomplete',needs_review=True,activity_provenance='synthetic',contains_synthetic_evidence=True,
            evidence={'approved_status':{'value':approval,'company':company,'provenance':'synthetic'}},
            reasons=[{'rule':'unapproved','explanation':'Explicitly unapproved in the fictional company.','points':25}],
            missing_evidence_rules=['no_sso'],assessment_basis='prototype',policy_version='demo'))


@unittest.skipUnless(AVAILABLE,'Install modeling/requirements-retrieval.txt for retrieval tests')
class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.encoder=self.root/'encoder';self.encoder.mkdir();(self.encoder/'config.json').write_text('{}')
        self.a=self.root/'a.json';self.b=self.root/'b.json'
        self.a.write_text(json.dumps({'results':[app('Alpha','Company A',60),app('Beta','Company A',80,ai=True)]}))
        self.b.write_text(json.dumps({'results':[app('Alpha','Company B',10,approval='approved')]}))
        self.path=self.root/'index';build([self.a,self.b],self.path,self.encoder,Encoder())
        self.index=EvidenceIndex(self.path,Encoder())
    def tearDown(self):self.temp.cleanup()

    def test_company_filter_before_top_k_and_source_citations(self):
        result=self.index.search('Alpha company approval','Company B',1)
        self.assertEqual(result['total_matches'],1)
        hit=result['hits'][0];self.assertEqual(hit['facts']['company'],'Company B')
        self.assertEqual(hit['source']['json_pointer'],'/results/0')
        self.assertEqual(hit['source']['file'],'b.json')
        self.assertNotIn('employee_id',json.dumps(hit))

    def test_counts_use_full_filtered_snapshot_not_top_k(self):
        result=answer(self.index,'How many unapproved applications are used by Finance?','Company A',1)
        self.assertEqual(result['total_matches'],2);self.assertEqual(len(result['hits']),1)
        self.assertTrue(result['answer'].startswith('2 applications'))
        self.assertEqual(result['applied_filters'],{'department':'Finance','approval':'unapproved'})

    def test_ai_status_and_unknown_approval_do_not_conflate(self):
        result=answer(self.index,'How many unapproved AI tools are used by Finance?','Company A')
        self.assertEqual(result['total_matches'],1)
        self.assertEqual(result['hits'][0]['facts']['application_name'],'Beta')
        result=answer(self.index,'How many applications?','Company A',filters={'approval':'unknown'})
        self.assertEqual(result['total_matches'],0)

    def test_highest_risk_uses_numeric_sort_not_semantic_similarity(self):
        result=answer(self.index,'What are the highest-risk applications?','Company A',1)
        self.assertEqual(result['hits'][0]['facts']['application_name'],'Beta')
        self.assertEqual(result['ranking'],'risk score descending')

    def test_no_matches_does_not_guess_or_call_encoder(self):
        result=answer(self.index,'Show applications','Unknown Company')
        self.assertEqual(result['hits'],[]);self.assertIn('does not establish absence',result['answer'])

    def test_empty_question_and_company_and_bad_k_fail(self):
        for question,company,k in [('', 'Company A',5),('x','',5),('x','Company A',0)]:
            with self.assertRaises(ValueError):answer(self.index,question,company,k)

    def test_no_overwrite_and_no_multiple_company_snapshots(self):
        with self.assertRaises(ValueError):build([self.a],self.path,self.encoder,Encoder())
        with self.assertRaises(ValueError):build([self.a,self.a],self.root/'second',self.encoder,Encoder())

    def test_changed_report_and_corrupt_index_rejected(self):
        self.a.write_text(self.a.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'Source report changed'):EvidenceIndex(self.path,Encoder())
        self.a.write_text(self.a.read_text().rstrip())
        (self.path/'documents.json').write_text('[]')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):EvidenceIndex(self.path,Encoder())

    def test_mixed_company_report_rejected(self):
        self.a.write_text(json.dumps({'results':[app('Alpha','Company A'),app('Beta','Company B')]}))
        with self.assertRaisesRegex(ValueError,'same explicit'):documents([self.a])

    def test_approval_from_other_company_is_not_used(self):
        row=app('Tool','Company A');row['governance']['evidence']['approved_status']['company']='Company B'
        self.a.write_text(json.dumps({'results':[row]}));docs,_=documents([self.a])
        self.assertIsNone(docs[0]['facts']['approval'])
        self.assertIsNone(docs[0]['facts']['controls']['approved_status']['value'])

    def test_synthetic_missingness_and_provisional_category_stay_visible(self):
        result=answer(self.index,'Why was Beta flagged?','Company A')
        self.assertEqual(result['total_matches'],1)
        self.assertIn('fictional',result['answer']);self.assertIn('incomplete',result['answer'])
        self.assertIn('provisional',result['answer']);self.assertIn('across all application departments',result['answer'])
        self.assertIn('['+result['hits'][0]['evidence_id']+']',result['answer'])

    def test_multiple_departments_require_explicit_scope(self):
        with self.assertRaisesRegex(ValueError,'Multiple departments'):answer(self.index,'Compare Finance and Engineering','Company A')

    def test_explicit_scope_overrides_ambiguous_wording_and_provider_failure_is_not_retried(self):
        result=answer(self.index,'Compare Finance and Engineering','Company A',filters={'department':'Finance'})
        self.assertEqual(result['applied_filters']['department'],'Finance')
        class Failing:
            def __init__(self):self.models=self;self.calls=0
            def generate_content(self,**kwargs):
                self.calls+=1
                raise TimeoutError('request details must not be printed')
        client=Failing()
        with self.assertRaisesRegex(ValueError,'TimeoutError') as caught:
            generate(result,'Explain',client,'fake-model')
        self.assertNotIn('request details',str(caught.exception));self.assertEqual(client.calls,1)

    def test_generation_requires_valid_citations_and_counts_never_call_provider(self):
        result=self.index.search('Beta','Company A',1)
        identifier=result['hits'][0]['evidence_id']
        class Client:
            def __init__(self,payload):
                self.calls=0;self.payload=payload;self.models=self
            def generate_content(self,**kwargs):
                self.calls+=1
                return SimpleNamespace(text=json.dumps(self.payload))
        client=Client({'claims':[{'evidence_id':identifier,'field':'/approval','value':'unapproved'}],'insufficient_evidence':False})
        self.assertEqual(generate(result,'Why?',client,'fake-model')['claims'][0]['evidence_id'],identifier)
        client=Client({'claims':[{'evidence_id':'invented','field':'/approval','value':'approved'}],'insufficient_evidence':False})
        with self.assertRaisesRegex(ValueError,'citations'):generate(result,'Why?',client,'fake-model')
        counted=answer(self.index,'How many applications?','Company A',client=client,model='fake-model')
        self.assertEqual(client.calls,1);self.assertEqual(counted['total_matches'],2)

if __name__=='__main__':unittest.main()
