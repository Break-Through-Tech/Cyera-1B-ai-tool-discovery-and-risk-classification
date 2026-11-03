import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'preprocessing_scripts/prepare_classification.py'
spec = importlib.util.spec_from_file_location('prep', SCRIPT)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def fixture():
    def row(i, label):
        return dict(tool_id=f'{label}_{i}', tool_name=f'Application {label} {i}',
                    domain=f'app{label}-{i}.example.com', description=f'Unique task {label} {i}',
                    is_ai_tool='true' if label else 'false', category='Productivity')
    return [row(i, 1) for i in range(20)], [row(i, 0) for i in range(20)]


class PreparationTests(unittest.TestCase):
    def test_determinism_and_both_classes(self):
        a, n = fixture()
        first, _ = prep.build(a, n)
        self.assertEqual(first, prep.build(list(reversed(a)), list(reversed(n)))[0])
        self.assertEqual(sum(map(len, first.values())), 40)
        self.assertTrue(all({r['is_ai_tool'] for r in rows} == {0, 1} for rows in first.values()))

    def test_transitive_groups_and_mixed_label_hosts(self):
        a, n = fixture()
        a[1]['domain'] = 'https://www.' + a[0]['domain'] + '/feature'
        a[2]['tool_name'] = a[1]['tool_name']
        a[3]['description'] = a[2]['description']
        n[0]['domain'] = a[0]['domain']
        splits, _ = prep.build(a, n)
        matched = [(s, r) for s, rows in splits.items() for r in rows
                   if r['source_record_id'] in {'1_0','1_1','1_2','1_3','0_0'}]
        self.assertEqual(len(matched), 5)
        self.assertEqual(len({s for s, _ in matched}), 1)
        self.assertEqual(len({r['group_id'] for _, r in matched}), 1)

    def test_duplicate_removal(self):
        a, n = fixture()
        a.append(dict(a[0], tool_id='duplicate'))
        splits, excluded = prep.build(a, n)
        self.assertEqual(sum(map(len, splits.values())), 40)
        self.assertEqual(excluded[0]['reason'], 'duplicate_application')

    def test_conflicting_identity_withheld(self):
        a, n = fixture()
        n.append(dict(a[0], tool_id='conflict', is_ai_tool='false'))
        splits, excluded = prep.build(a, n)
        self.assertEqual(sum(map(len, splits.values())), 39)
        self.assertEqual(sum(e['reason']=='conflicting_identity_labels' for e in excluded), 2)

    def test_missing_and_invalid_inputs_excluded(self):
        a, n = fixture()
        a[0]['description'] = ''
        a[1]['domain'] = 'not a host'
        a[2]['is_ai_tool'] = 'unknown'
        _, excluded = prep.build(a, n)
        self.assertEqual({e['reason'] for e in excluded},
                         {'missing_name_or_description','invalid_domain','unexpected_source_label'})

    def test_feature_isolation_and_provenance(self):
        a, n = fixture()
        a[0]['category'] = 'LABEL_SENTINEL'
        a[0]['risk_label'] = 'RISK_SENTINEL'
        splits, _ = prep.build(a, n)
        row = next(r for rows in splits.values() for r in rows if r['source_record_id']=='1_0')
        self.assertNotIn('SENTINEL', row['text_for_model'])
        self.assertNotIn('risk_label', row)
        self.assertEqual(row['label_quality'], 'weak_supervision')
        self.assertEqual(row['category'], 'LABEL_SENTINEL')

    def test_insufficient_class_fails(self):
        a, n = fixture()
        with self.assertRaises(ValueError):
            prep.build(a, n[:2])

    def test_current_repository(self):
        root = SCRIPT.parents[1]
        splits, excluded = prep.build(
            prep.load(root/'data/processed/dataset_b/sample/accepted_dataset_b.csv'),
            prep.load(root/'data/raw/non_ai_enterprise_tools.csv'))
        self.assertEqual([len(splits[s]) for s in ['train','validation','test']], [235,50,50])
        self.assertEqual(excluded, [])
        prep.validate(splits)


if __name__ == '__main__':
    unittest.main()
