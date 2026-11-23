"""Jev category step in ai_discovery.categorize. Jev is simulated; no network calls.

Run: python -m unittest tests.test_component3_jev
"""
import copy
import unittest
import urllib.error

from ai_discovery import categorize as c
from tests import test_component3_pipeline as base
from tests.test_component3_pipeline import StubEngine


def answer(choice, p):
    """A Jev response putting probability p on `choice`, the rest spread evenly."""
    slugs = [c.helpers.import_file(c.ROOT/'category_training/jev_category_eval.py', 'jev_t').slug(x)
             for x in base.CategorizationTests.taxonomy]
    rest = (1 - p) / (len(slugs) - 1)
    return {'choice': choice, 'probabilities': {s: (p if s == choice else rest) for s in slugs},
            'confidence': p, 'returned_model': 'jev-test', 'usage': {}}


class JevCategoryTests(unittest.TestCase):
    # Reuse the Component 3 fixture (temporary catalog, model folder, AI record).
    setUp_base = base.CategorizationTests.setUp
    write_catalog = base.CategorizationTests.write_catalog
    row = base.CategorizationTests.row

    def setUp(self):
        self.setUp_base()
        base.CategorizationTests.taxonomy = c.helpers.import_file(self.args.prep_script, 'tp').TAXONOMY
        self.args.category_api = 'jev'
        self.args.api_review_threshold = 0.5
        self.args.category_jev_model = 'jev-latest'
        self.args.category_jev_criteria = c.ROOT/'artifacts/jev_category_eval/criteria.json'
        self.args.category_jev_cache = self.root/'jev_cache.json'
        self.args.category_jev_script = c.ROOT/'category_training/jev_category_eval.py'
        self.args.api_timeout = 5

    def engine(self, response=None, error=None, key='test-key'):
        e = StubEngine(self.args)
        e.jev.key = key
        calls = []

        def fake(model, text, criteria, timeout, api_key):
            calls.append(text)
            if error:
                raise error
            return response
        e.jev.mod.call_jev = fake
        return e, calls

    def test_confident_answer_is_provisional_api_category(self):
        e, calls = self.engine(answer('technology_and_development', 0.9))
        r = e.categorize(self.record)
        self.assertEqual(r['category'], 'Technology And Development')
        self.assertEqual((r['category_source'], r['category_status']), ('api', 'provisional_prediction'))
        self.assertEqual(r['api_probability'], 0.9)
        self.assertEqual(r['top_three'][0], {'category': 'Technology And Development', 'probability': 0.9})
        self.assertEqual(e.calls, 0)                       # local model not used
        self.assertEqual(len(calls), 1)
        self.assertNotIn('Generates code', calls[0])        # description only, no functionality section

    def test_low_probability_goes_to_human_review(self):
        e, _ = self.engine(answer('media_and_content_creation', 0.3))
        r = e.categorize(self.record)
        self.assertIsNone(r['category'])
        self.assertEqual(r['category_status'], 'needs_category_review')
        self.assertEqual(r['suggested_category'], 'Media And Content Creation')
        self.assertEqual(e.calls, 0)

    def test_threshold_is_inclusive(self):
        e, _ = self.engine(answer('technology_and_development', 0.5))
        self.assertEqual(e.categorize(self.record)['category_status'], 'provisional_prediction')

    def test_request_failure_falls_back_to_local_model(self):
        e, calls = self.engine(error=urllib.error.URLError('offline'))
        r = e.categorize(self.record)
        self.assertEqual((r['category_source'], r['category_status']), ('model', 'provisional_prediction'))
        self.assertIn('URLError', r['api_fallback_reason'])
        self.assertEqual((e.calls, len(calls)), (1, 1))

    def test_missing_key_falls_back_without_calling(self):
        e, calls = self.engine(answer('technology_and_development', 0.9), key='')
        r = e.categorize(self.record)
        self.assertEqual(r['category_source'], 'model')
        self.assertIn('TYPESAFE_API_KEY', r['api_fallback_reason'])
        self.assertEqual(calls, [])

    def test_out_of_taxonomy_answer_falls_back(self):
        bad = answer('technology_and_development', 0.9)
        bad['choice'] = 'not_a_category'
        e, _ = self.engine(bad)
        self.assertEqual(e.categorize(self.record)['category_source'], 'model')

    def test_cached_answer_is_reused_across_runs(self):
        e, calls = self.engine(answer('technology_and_development', 0.9))
        self.assertFalse(e.categorize(self.record)['cache_hit'])
        e.save_cache()
        again, calls2 = self.engine(answer('media_and_content_creation', 0.9))
        r = again.categorize(self.record)
        self.assertTrue(r['cache_hit'])
        self.assertEqual(r['category'], 'Technology And Development')
        self.assertEqual((len(calls), len(calls2)), (1, 0))

    def test_changed_evidence_makes_a_new_request(self):
        e, calls = self.engine(answer('technology_and_development', 0.9))
        e.categorize(self.record)
        changed = copy.deepcopy(self.record)
        changed['reports'][0]['report_blocks'][0]['text'] = changed['reports'][0]['report_blocks'][0]['text'].replace(
            'An editor for writing', 'A changed editor for writing')
        e.categorize(changed)
        self.assertEqual(len(calls), 2)

    def test_catalog_and_human_review_still_take_precedence(self):
        self.write_catalog(self.catalog, [self.row()])
        e, calls = self.engine(answer('media_and_content_creation', 0.99))
        self.assertEqual(e.categorize(self.record)['category_source'], 'catalog')
        self.write_catalog(self.catalog, [])
        e, calls2 = self.engine(answer('media_and_content_creation', 0.99))
        review = {'reviewed_category': 'Technology And Development', 'reviewer': 'Tester', 'reason': 'Reviewed',
                  'evidence_sha256': c.helpers.fingerprint(self.record['reports'])}
        self.assertEqual(e.categorize(self.record, review)['category_source'], 'human_review')
        self.assertEqual((calls, calls2), ([], []))

    def test_non_ai_records_never_reach_jev(self):
        e, calls = self.engine(answer('technology_and_development', 0.9))
        self.assertEqual(e.categorize(dict(self.record, is_ai_tool=False))['category_status'], 'not_applicable')
        self.assertEqual(calls, [])

    def test_off_by_default_and_invalid_threshold_rejected(self):
        self.args.category_api = 'none'
        self.assertIsNone(StubEngine(self.args).jev)
        self.args.category_api = 'jev'
        self.args.api_review_threshold = 1.5
        with self.assertRaisesRegex(ValueError, 'threshold'):
            StubEngine(self.args)


if __name__ == '__main__':
    unittest.main()
