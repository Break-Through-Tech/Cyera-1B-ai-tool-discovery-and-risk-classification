"""Independent expected outcomes for demo-1.0; no network or model required."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from ai_discovery.ingest import extract_applications
from governance.risk import POLICY, assess_records


class GovernanceEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'activity.csv'
        self.policy = json.loads(POLICY.read_text())

    def evaluate(self, controls=None, upload=0, download=0, users=1,
                 department='Engineering', ai=True, provenance='synthetic'):
        rows = [dict(application_name='Fictional Tool', domain='tool.example',
                     employee_id=str(i), department=department,
                     bytes_uploaded=str(upload[i] if isinstance(upload, list) else upload),
                     bytes_downloaded=str(download))
                for i in range(users)]
        with self.path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        ingestion = extract_applications(self.path)
        app = ingestion['applications'][0]
        record = dict(application_id=app['application_id'], observed_identity=app,
                      classification_status='classified', identity_supported=True,
                      is_ai_tool=ai)
        values = dict(approved_status='approved', sso_available=True,
                      data_retention='30_days', training_on_customer_data=False,
                      admin_controls_available=True, enterprise_plan_available=True)
        values.update(controls or {})
        fields = {key: dict(value=value, provenance=provenance,
                            source='Fictional evaluation fixture', company='Demo')
                  for key, value in values.items() if value is not None}
        evidence = dict(version=1, applications={app['application_id']: fields})
        before = copy.deepcopy((record, ingestion, evidence))
        result = assess_records([record], ingestion, self.path, self.policy,
                                company='Demo', evidence=evidence,
                                activity_provenance=provenance)[0]['governance']
        self.assertEqual(before, (record, ingestion, evidence))
        return result

    def check_result(self, result, score, level, status='complete'):
        self.assertEqual((result['risk_score'], result['risk_level'],
                          result['assessment_status']), (score, level, status))
        self.assertEqual(min(100, sum(x['points'] for x in result['reasons'])), score)

    def test_retention_boundaries(self):
        for value, score in [('no_retention', 0), ('30_days', 0),
                             ('90_days', 0), ('91_days', 10), ('indefinite', 10)]:
            with self.subTest(retention=value):
                self.check_result(self.evaluate({'data_retention': value}), score, 'low')

    def test_unknown_retention_is_incomplete_not_long_retention(self):
        result = self.evaluate({'data_retention': None})
        self.check_result(result, 0, 'low', 'incomplete')
        self.assertIn('long_retention', result['missing_evidence_rules'])
        self.assertTrue(result['needs_review'])

    def test_mean_upload_boundary(self):
        for value, score in [(4_999_999, 0), (5_000_000, 15), (5_000_001, 15)]:
            with self.subTest(upload=value):
                self.check_result(self.evaluate(upload=value), score, 'low')

    def test_download_boundary(self):
        for value, score in [(9_999_999, 0), (10_000_000, 5), (10_000_001, 5)]:
            with self.subTest(download=value):
                self.check_result(self.evaluate(download=value), score, 'low')

    def test_single_upload_boundary_with_mean_below_threshold(self):
        for value, score in [(9_999_999, 0), (10_000_000, 15), (10_000_001, 15)]:
            with self.subTest(maximum=value):
                self.check_result(self.evaluate(upload=[value, 0, 0], users=3), score, 'low')

    def test_adoption_boundary(self):
        for users, score in [(49, 0), (50, 5), (51, 5)]:
            with self.subTest(users=users):
                self.check_result(self.evaluate(users=users), score, 'low')

    def test_each_control_in_isolation(self):
        for field, value, score, level in [
            ('approved_status', 'unapproved', 25, 'medium'),
            ('sso_available', False, 10, 'low'),
            ('admin_controls_available', False, 10, 'low'),
            ('enterprise_plan_available', False, 5, 'low'),
            ('training_on_customer_data', True, 20, 'medium')]:
            with self.subTest(field=field):
                self.check_result(self.evaluate({field: value}), score, level)

    def test_training_excluded_for_non_ai(self):
        result = self.evaluate({'training_on_customer_data': True}, ai=False)
        self.check_result(result, 0, 'low')
        check = next(c for c in result['checks'] if c['rule'] == 'customer_data_training')
        self.assertEqual(check['status'], 'not_applicable')

    def test_under_review_remains_incomplete(self):
        result = self.evaluate({'approved_status': 'under_review'})
        self.check_result(result, 0, 'low', 'incomplete')
        self.assertIn('unapproved', result['missing_evidence_rules'])

    def test_natural_score_bands_without_changing_weights(self):
        cases = [({'sso_available': False}, 1, 'Engineering', 10, 'low'),
                 ({'sso_available': False}, 1, 'Finance', 20, 'medium'),
                 ({'approved_status': 'unapproved', 'sso_available': False}, 1, 'Finance', 45, 'medium'),
                 ({'approved_status': 'unapproved', 'sso_available': False}, 50, 'Finance', 50, 'high')]
        for controls, users, department, score, level in cases:
            with self.subTest(score=score):
                self.check_result(self.evaluate(controls, users=users, department=department), score, level)

    def test_integrated_cap_and_upload_counted_once(self):
        result = self.evaluate(dict(approved_status='unapproved', sso_available=False,
            admin_controls_available=False, enterprise_plan_available=False,
            training_on_customer_data=True, data_retention='indefinite'),
            upload=12_000_000, download=10_000_000, users=50, department='Finance')
        self.check_result(result, 100, 'high')
        self.assertEqual(sum(r['points'] for r in result['reasons']), 115)
        self.assertEqual(sum(r['rule'] == 'large_upload' for r in result['reasons']), 1)

    def test_synthetic_and_verified_same_score_distinct_provenance(self):
        for provenance in ['synthetic', 'verified']:
            with self.subTest(provenance=provenance):
                result = self.evaluate({'sso_available': False}, provenance=provenance)
                self.check_result(result, 10, 'low')
                self.assertEqual(result['contains_synthetic_evidence'], provenance == 'synthetic')
                self.assertEqual(result['reasons'][0]['evidence_provenance'], provenance)


if __name__ == '__main__':
    unittest.main()
