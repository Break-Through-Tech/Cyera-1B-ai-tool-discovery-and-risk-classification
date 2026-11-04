import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'modeling'))
AVAILABLE=importlib.util.find_spec('sklearn') is not None
if AVAILABLE:
    import numpy as np
    from embedding_utils import ENCODER_REVISION, encode_rows, nearest, normalized
    from train_embeddings import choose_default


@unittest.skipUnless(AVAILABLE,'Install modeling/requirements-embeddings.txt')
class EmbeddingUnitTests(unittest.TestCase):
    def test_cosine_ranking_is_normalized_and_stable(self):
        ids,scores=nearest([[2.,0.]],[[9.,0.],[0.,2.],[1.,0.]],k=9)
        self.assertEqual(ids.tolist(),[[0,2,1]])
        np.testing.assert_allclose(scores,[[1,1,0]],atol=1e-6)

    def test_invalid_vectors_and_k_rejected(self):
        for reference in ([[0.,0.]],[[float('nan'),0.]],[[1.,0.,0.]],[]):
            with self.assertRaises(ValueError):nearest([[1.,0.]],reference)
        with self.assertRaises(ValueError):nearest([[1.,0.]],[[1.,0.]],0)

    def test_cache_tracks_actual_text_and_preserves_vectors(self):
        class Encoder:
            calls=0
            def encode(self,texts,**kwargs):
                self.calls+=1
                return np.array([[1.,len(t)+1.,2.] for t in texts],dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            encoder=Encoder();rows=[{'text_for_model':'one'},{'text_for_model':'two'}]
            first=encode_rows(encoder,rows,td);second=encode_rows(encoder,rows,td)
            np.testing.assert_array_equal(first,second)
            self.assertEqual(encoder.calls,1)
            encode_rows(encoder,[{'text_for_model':'changed'}],td)
            self.assertEqual(encoder.calls,2)

    def test_validation_ties_preserve_baseline(self):
        self.assertEqual(choose_default(.8,.8),'tfidf')
        self.assertEqual(choose_default(.79,.8),'tfidf')
        self.assertEqual(choose_default(.81,.8),'embeddings')


@unittest.skipUnless(AVAILABLE and (ROOT/'artifacts/embeddings/model_config.json').exists(),
                     'Run embedding training before artifact checks')
class EmbeddingArtifactTests(unittest.TestCase):
    def test_encoder_revision_and_baseline_inputs_match(self):
        import hashlib
        config=json.loads((ROOT/'artifacts/embeddings/model_config.json').read_text())
        self.assertEqual(config['encoder_revision'],ENCODER_REVISION)
        self.assertTrue(config['encoder_frozen'])
        for name,digest in config['inputs'].items():
            self.assertEqual(hashlib.sha256((ROOT/'data/processed/classification_full'/name).read_bytes()).hexdigest(),digest)
        self.assertEqual(hashlib.sha256((ROOT/'artifacts/baseline/model_config.json').read_bytes()).hexdigest(),
                         config['baseline_config_sha256'])

    def test_similarity_references_have_no_heldout_identity(self):
        import csv
        config=json.loads((ROOT/'artifacts/embeddings/model_config.json').read_text())
        refs=json.loads((ROOT/'artifacts/embeddings/reference_records.json').read_text())
        with np.load(ROOT/'artifacts/embeddings/reference_embeddings.npz',allow_pickle=False) as f:
            vectors=f['embeddings']
        self.assertEqual(vectors.shape,(len(refs),384))
        np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
        heldout=[]
        for split in ('validation','test'):
            with (ROOT/f'data/processed/classification_full/{split}.csv').open(encoding='utf-8-sig') as f:
                heldout.extend(csv.DictReader(f))
        for key in ['group_id','application_id','domain','tool_name','description']:
            self.assertFalse({r[key] for r in refs}&{r[key] for r in heldout})
        self.assertTrue(all(r['split']=='train' and r['label_source']!='provisional_catalog_membership' for r in refs))
        self.assertEqual(len(refs),config['reference_rows'])
        self.assertFalse(set(config['excluded_training_ids'])&{r['source_record_id'] for r in refs})

    def test_selection_depends_on_validation_not_test(self):
        report=json.loads((ROOT/'artifacts/embeddings/metrics.json').read_text())
        config=json.loads((ROOT/'artifacts/embeddings/model_config.json').read_text())
        base=json.loads((ROOT/'artifacts/baseline/metrics.json').read_text())
        variant=max(('reviewed_only','full_catalog'),key=lambda n:report['binary'][n]['validation']['macro_f1'])
        self.assertEqual(config['selected_embedding_variant'],variant)
        self.assertEqual(config['default_binary_family'],choose_default(report['binary'][variant]['validation']['macro_f1'],
            base['binary'][config['baseline_binary_variant']]['validation']['macro_f1']))
        self.assertEqual(config['default_category_family'],choose_default(report['category']['validation']['metrics']['macro_f1'],
            base['category']['validation']['metrics']['macro_f1']))
        self.assertFalse(report['independent_fresh_evaluation'])

    def test_saved_classifiers_are_compatible_with_references(self):
        import joblib
        for name in ['binary_reviewed_only','binary_full_catalog','category']:
            model=joblib.load(ROOT/f'artifacts/embeddings/{name}.joblib')
            self.assertEqual(model.n_features_in_,384)
        config=json.loads((ROOT/'artifacts/embeddings/model_config.json').read_text())
        cat=joblib.load(ROOT/'artifacts/embeddings/category.joblib')
        self.assertEqual(list(cat.classes_),config['category_labels'])


if __name__=='__main__':unittest.main()
