"""Frozen encoder and cosine-similarity helpers for issue #7."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ENCODER_ID = 'sentence-transformers/all-MiniLM-L6-v2'
ENCODER_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def load_encoder(cache_dir):
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('HF_XET_CACHE',str(cache_dir/'xet'))
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    return SentenceTransformer(
        ENCODER_ID, revision=ENCODER_REVISION, device='cpu',
        cache_folder=str(cache_dir), trust_remote_code=False,
        model_kwargs={'use_safetensors': True})


def normalized(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError('Expected a finite 2D embedding matrix')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError('Zero-length vectors cannot define cosine similarity')
    return vectors / norms


def nearest(query, reference, k=3):
    if k < 1:
        raise ValueError('k must be positive')
    query, reference = normalized(query), normalized(reference)
    if len(reference) == 0 or query.shape[1] != reference.shape[1]:
        raise ValueError('Reference embeddings must be nonempty and dimension-compatible')
    scores = query @ reference.T
    ids = np.argsort(-scores, axis=1, kind='stable')[:, :min(k,len(reference))]
    return ids, np.take_along_axis(np.clip(scores,-1,1),ids,axis=1)


def encode_rows(encoder, rows, cache_dir):
    texts = [r['text_for_model'] for r in rows]
    key = hashlib.sha256(json.dumps([ENCODER_ID,ENCODER_REVISION,texts],ensure_ascii=False).encode()).hexdigest()
    cache = Path(cache_dir)/f'{key}.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as saved:
            embeddings = saved['embeddings'].copy()
        if embeddings.ndim != 2 or not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings,axis=1),1,atol=1e-5):
            raise ValueError('Invalid cached embeddings')
        if len(embeddings) != len(rows):
            raise ValueError('Cached embedding row count mismatch')
        return embeddings
    embeddings = normalized(encoder.encode(texts,batch_size=64,show_progress_bar=True,
                                          convert_to_numpy=True,normalize_embeddings=True))
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,embeddings=embeddings)
    return embeddings
