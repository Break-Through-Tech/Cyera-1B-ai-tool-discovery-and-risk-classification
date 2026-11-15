"""Standard-library guard tests; run from project root with --prep-script if needed."""
import argparse
import copy
import json
import sys
import unittest
from pathlib import Path
import review_discovery_categories as review

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--prep-script',type=Path,default=Path('preprocessing_scripts/prepare_category_data.py'))
parser.add_argument('--discovery',type=Path)
args,rest=parser.parse_known_args()
prep=review.import_file(args.prep_script,'test_preparation')


def record():
    return {'application_name':'Product','endpoint':'product.example','classification_status':'classified',
            'is_ai_tool':True,'identity_supported':True,
            'reports':[{'report_blocks':[{'text':'## Product Identity\nNot model input.\n## Product Description\nEdits software code.\n## AI Functionality\nSuggests bug fixes.\n## Uncertainty\nNot model input either.'}]}]}


class GuardTests(unittest.TestCase):
    def test_actual_purpose_input(self):
        e=review.candidate(record(),prep)
        self.assertEqual(e['status'],'ready')
        self.assertEqual(e['model_text'],'Edits software code.\nSuggests bug fixes.')

    def test_detection_gates(self):
        for label,status,expected in [(False,'classified','not_applicable'),(None,'unresolved','pending_ai_detection'),('true','classified','pending_ai_detection')]:
            r=record();r.update(is_ai_tool=label,classification_status=status)
            e=review.candidate(r,prep)
            self.assertEqual(e['status'],expected);self.assertEqual(e['model_text'],'')

    def test_identity_and_errors(self):
        r=record();r['identity_supported']=False
        self.assertEqual(review.candidate(r,prep)['status'],'pending_identity_review')
        r=record();r['technical_error']=True
        self.assertEqual(review.candidate(r,prep)['status'],'pending_identity_review')

    def test_multiple_reports(self):
        r=record();r['reports'].append(copy.deepcopy(r['reports'][0]))
        self.assertEqual(review.candidate(r,prep)['status'],'needs_report_selection')
        self.assertEqual(review.candidate(r,prep,1)['status'],'ready')
        with self.assertRaises(ValueError):review.candidate(r,prep,2)

    def test_missing_sections(self):
        r=record();r['reports'][0]['report_blocks'][0]['text']='No headed purpose evidence.'
        self.assertEqual(review.candidate(r,prep)['status'],'pending_purpose_evidence')

    def test_overlap_and_split_leakage(self):
        r={'source_id':'r1','group_id':'g1','tool_name':'Product','authority':'product.example',
           'product_path':'','description':'Edits software code.','model_text':'Edits software code.\nSuggests bug fixes.'}
        idx,parts=review.overlap_indexes([r],{'r1':{'partition':'development'}},prep)
        status,matches=review.detect_overlap(record(),review.candidate(record(),prep),idx,parts,prep)
        self.assertEqual(status,'development_overlap');self.assertTrue(matches)
        r2=dict(r,source_id='r2')
        with self.assertRaises(ValueError):
            review.overlap_indexes([r,r2],{'r1':{'partition':'development'},'r2':{'partition':'test'}},prep)

    def test_stale_reviews(self):
        r=record();h=review.fingerprint(r['reports'])
        review.validate_review({'evidence_sha256':h},r,h)
        with self.assertRaises(ValueError):review.validate_review({'evidence_sha256':'old'},r,h)
        r['is_ai_tool']=False
        with self.assertRaises(ValueError):review.validate_review({'evidence_sha256':h},r,h)


if args.discovery:
    data=review.read_json(args.discovery)
    outcomes=[(r['application_name'],review.candidate(r,prep)['status']) for r in data['results']]
    print('Real evidence extraction:',json.dumps(outcomes))

if __name__=='__main__':unittest.main(argv=[sys.argv[0]]+rest)
