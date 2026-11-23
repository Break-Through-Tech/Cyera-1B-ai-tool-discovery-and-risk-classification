import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ai_discovery import jev,jev_catalog as catalog


def products():
    return [{'application_id':'p'+str(i),'application_name':'Tool '+str(i),'endpoint':str(i)+'.example',
             'descriptions':['Product description'],'source_labels':['ai'],'source_rows':[{'file':'source.csv','row':i+1}]} for i in range(3)]

def reply(payload,key):
    return {'model':catalog.MODEL,'answers':{k:{'choice':'non_ai','probabilities':{'ai':.01,'non_ai':.98,'unknown':.01}}
        for k in payload['questions']},'usage':{'input_tokens':100,'output_tokens':0}}

class CatalogTests(unittest.TestCase):
    def test_products_are_isolated_from_other_products_and_source_labels(self):
        ps=products();data=catalog.payload(ps)
        self.assertNotIn('source_labels',json.dumps(data))
        self.assertNotIn('source_rows',json.dumps(data))
        first=data['questions']['p0']['instructions']['product']
        self.assertEqual(first['application_name'],'Tool 0')
        self.assertNotIn('Tool 1',json.dumps(first))
        self.assertEqual(len(list(catalog.batches(ps,2))),2)

    def test_wrong_response_ids_or_probabilities_fail(self):
        ps=products();data=reply(catalog.payload(ps),'unused');catalog.validate_batch(data,ps)
        data['answers']['other']=data['answers'].pop('p0')
        with self.assertRaises(jev.JevError):catalog.validate_batch(data,ps)
        with self.assertRaises(ValueError):list(catalog.batches(ps,max_bytes=10))

    def test_complete_resume_is_free_and_source_labels_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(catalog,'prepare',return_value=(products(),[])),patch.object(catalog.time,'sleep'):
            root=Path(folder);calls=[]
            def sender(payload,key):calls.append(payload);return reply(payload,key)
            report=catalog.run(root/'out',root/'cache',key='not-a-real-key',sender=sender,batch_size=2)
            self.assertEqual(report['completed_applications'],3)
            self.assertEqual(report['successful_http_requests_total'],2)
            self.assertEqual(report['source_label_disagreements'],3)
            self.assertTrue(catalog.result(products()[0],{'model':catalog.MODEL,'classified_at':'test','answer':reply(catalog.payload(products()),'test')['answers']['p0']})['needs_review'])
            again=catalog.run(root/'out',root/'cache',sender=sender,batch_size=3)
            self.assertEqual(len(calls),2)
            self.assertEqual(again['new_successful_http_requests_this_run'],0)
            self.assertEqual(again['summary'],{'non_ai':3})
            self.assertNotIn('not-a-real-key',(root/'out'/'all_classified.csv').read_text())

    def test_failed_batch_preserves_success_and_does_not_invent_labels(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(catalog,'prepare',return_value=(products(),[])),patch.object(catalog.time,'sleep'):
            root=Path(folder);count=0
            def fail_second(payload,key):
                nonlocal count
                count+=1
                if count==2:raise jev.JevError('Jev HTTP 429')
                return reply(payload,key)
            with self.assertRaises(jev.JevError):catalog.run(root/'out',root/'cache',key='test',sender=fail_second,batch_size=2)
            saved=json.loads((root/'out'/'summary.json').read_text())
            self.assertEqual(saved['completed_applications'],2)
            self.assertEqual(saved['remaining_applications'],1)
            self.assertEqual(saved['run_status'],'stopped_on_error')
            resumed=catalog.run(root/'out',root/'cache',key='test',sender=reply)
            self.assertEqual(resumed['completed_applications'],3)
            self.assertEqual(resumed['new_successful_http_requests_this_run'],1)

if __name__=='__main__':unittest.main()
