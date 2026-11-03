import importlib.util
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'preprocessing_scripts'))
import prepare_full_classification as full


def fixtures():
    catalog=[];reviews=[];negatives=[]
    for i in range(20):
        catalog.append(dict(tool_id=f'ai_{i}',tool_name=f'AI product {i}',domain=f'ai{i}.com',
                            description=f'Distinctive ai task {i}',domain_valid='True',site_status_group='active'))
        reviews.append(dict(tool_id=f'ai_{i}',review_status='accepted',ai_status='ai',description_usable='True',category='Productivity'))
        negatives.append(dict(tool_id=f'non_{i}',tool_name=f'Conventional {i}',domain=f'non{i}.com',
                              description=f'Distinctive business task {i}',is_ai_tool='false'))
    for i in range(20):
        catalog.append(dict(tool_id=f'weak_{i}',tool_name=f'Provisional {i}',domain=f'weak{i}.com',
                            description=f'Provisional task {i}',domain_valid='True',site_status_group='active'))
    return catalog,reviews,negatives


class FullPreparationTests(unittest.TestCase):
    def test_all_rows_accounted_for_and_eval_sources(self):
        c,r,n=fixtures();splits,audit=full.prepare(c,r,n)
        self.assertEqual(len(audit),40)
        self.assertEqual(len({a['source_record_id'] for a in audit}),40)
        self.assertTrue(all(a['reason'] for a in audit))
        self.assertTrue(all(x['label_source']!='provisional_catalog_membership' for s in ('validation','test') for x in splits[s]))
        self.assertEqual(sum(x['label_source']=='provisional_catalog_membership' for x in splits['train']),20)
        self.assertEqual((splits,audit),full.prepare(list(reversed(c)),list(reversed(r)),list(reversed(n))))

    def test_heldout_related_provisional_is_quarantined(self):
        c,r,n=fixtures()
        # Attach one provisional application to every reviewed application's host.
        # Some must follow heldout anchors regardless of the shuffled assignment.
        for i in range(20):
            c[20+i]['domain']=c[i]['domain']
        result,audit=full.prepare(c,r,n)
        train_hosts={x['domain'] for x in result['train']}
        eval_hosts={x['domain'] for s in ('validation','test') for x in result[s]}
        self.assertFalse(train_hosts & eval_hosts)
        self.assertEqual(sum(a['disposition']=='quarantined' for a in audit),6)
        for a in audit:
            if a['review_status']=='unreviewed' and a['domain'] in eval_hosts:
                self.assertEqual(a['disposition'],'quarantined')

    def test_review_exclusions_and_quality_filters(self):
        c,r,n=fixtures()
        c[-1]['description']=''
        c[-2]['site_status_group']='dead'
        c[-3]['domain_valid']='False'
        r.extend([dict(tool_id='weak_0',review_status='needs_review'),dict(tool_id='weak_1',review_status='rejected')])
        _,a=full.prepare(c,r,n)
        reasons={x['source_record_id']:x['reason'] for x in a}
        self.assertEqual(reasons['weak_0'],'review_needs_review')
        self.assertEqual(reasons['weak_1'],'review_rejected')
        self.assertEqual(reasons['weak_19'],'missing_name_or_description')
        self.assertEqual(reasons['weak_18'],'flagged_dead_site')
        self.assertEqual(reasons['weak_17'],'invalid_domain')

    def test_provisional_negative_overlap_excluded(self):
        c,r,n=fixtures();c[-1]['domain']=n[0]['domain']
        _,a=full.prepare(c,r,n)
        self.assertEqual(next(x for x in a if x['source_record_id']=='weak_19')['reason'],
                         'unreviewed_overlap_with_negative_seed')

    def test_duplicate_application_prefer_reviewed(self):
        c,r,n=fixtures();c.append(dict(c[0],tool_id='duplicate'))
        _,a=full.prepare(c,r,n)
        self.assertEqual(next(x for x in a if x['source_record_id']=='duplicate')['reason'],'duplicate_application')

    def test_conflicting_accepted_identity_fails(self):
        c,r,n=fixtures();n.append(dict(n[0],tool_id='conflict',tool_name=c[0]['tool_name'],domain=c[0]['domain']))
        with self.assertRaises(ValueError):full.prepare(c,r,n)

    def test_current_repository_accounting(self):
        s,a=full.prepare(full.load(ROOT/'data/processed/kaggle_clean.csv'),
                        full.load(ROOT/'data/processed/dataset_b/sample/review_results.csv'),
                        full.load(ROOT/'data/raw/non_ai_enterprise_tools.csv'))
        self.assertEqual(len(a),16762)
        self.assertEqual(sum(map(len,s.values())),14792)
        self.assertEqual([len(s[k]) for k in ['train','validation','test']],[14692,50,50])
        self.assertEqual(sum(x['disposition']=='quarantined' for x in a),2)


if __name__=='__main__':unittest.main()
