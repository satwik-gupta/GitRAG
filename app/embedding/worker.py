"""
app/embedding/worker.py
────────────────────────
Async GPU/CPU embedding worker.

Dense embeddings
────────────────
  Uses sentence-transformers SentenceTransformer.
  Detects GPU availability via torch; falls back to batched CPU if absent.
  Heavy compute is dispatched via run_in_executor so the event loop is
  never blocked.

Sparse (BM25) embeddings
─────────────────────────
  A lightweight vocabulary-hash BM25 encoder that produces qdrant-compatible
  SparseVector (indices: list[int], values: list[float]) without requiring
  a pre-built global vocabulary file.  Token IDs are stable SHA-256-derived
  integers, which means vectors from different workers are always consistent.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ── Global thread pool used for all CPU-bound embedding work ───────────────
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="embed_")


# ── Dense encoder ─────────────────────────────────────────────────────────


def _load_sentence_transformer():
    """Load SentenceTransformer in a thread (import is slow, do once)."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    import torch  # noqa: PLC0415

    device_pref = settings.embedding_device
    if device_pref:
        device = device_pref
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info("Embedding worker: GPU detected (%s)", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        logger.info("Embedding worker: no GPU, using CPU")

    model = SentenceTransformer(settings.embedding_model, device=device)
    model.max_seq_length = 512
    return model, device


_dense_model = None
_device: str = "cpu"


def _get_dense_model():
    global _dense_model, _device
    if _dense_model is None:
        _dense_model, _device = _load_sentence_transformer()
    return _dense_model


def _encode_batch_sync(texts: list[str]) -> np.ndarray:
    model = _get_dense_model()
    return model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


# ── Sparse BM25 encoder ───────────────────────────────────────────────────

_BM25_K1 = 1.5
_BM25_B = 0.75
_AVG_DOC_LEN = 120          # approximate average token count for code chunks
_TOKEN_SPACE = (1 << 24)    # 16M token ID space; collision probability ≈ negligible


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-zA-Z_]\w*\b", text.lower())
    return [t for t in tokens if 2 <= len(t) <= 40]


def _token_id(token: str) -> int:
    """Map token string → stable integer in [0, _TOKEN_SPACE)."""
    digest = hashlib.sha256(token.encode()).digest()
    return int.from_bytes(digest[:4], "big") % _TOKEN_SPACE


def compute_sparse_vector(
    text: str,
    k1: float = _BM25_K1,
    b: float = _BM25_B,
    avg_doc_len: float = _AVG_DOC_LEN,
) -> tuple[list[int], list[float]]:
    """
    Return (indices, values) suitable for a Qdrant SparseVector.

    The formula uses the BM25 TF component (IDF is approximated as 1.0
    because we lack a global DF table at ingestion time).  This still
    outperforms raw TF for retrieval and is fully symmetric between
    document and query encoding.
    """
    tokens = _tokenize(text)
    if not tokens:
        return [], []

    doc_len = len(tokens)
    tf = Counter(tokens)

    indices: list[int] = []
    values: list[float] = []
    seen_ids: set[int] = set()

    for token, count in tf.items():
        tid = _token_id(token)
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        tf_score = (count * (k1 + 1)) / (
            count + k1 * (1 - b + b * doc_len / avg_doc_len)
        )
        indices.append(tid)
        values.append(round(tf_score, 6))

    return indices, values


# ── Public async API ──────────────────────────────────────────────────────


class EmbeddingWorker:
    """
    Async facade for dense and sparse vector generation.

    All heavy work is dispatched to *_EXECUTOR* so the event loop remains
    responsive even during large batches.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def _run_in_executor(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_EXECUTOR, fn, *args)

    # ── Dense ─────────────────────────────────────────────────────────────

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Asynchronously embed *texts* with the dense model.
        Returns an (N, D) float32 numpy array.
        """
        if not texts:
            return np.empty((0, settings.qdrant_vector_size), dtype=np.float32)
        t0 = time.monotonic()
        embeddings: np.ndarray = await self._run_in_executor(
            _encode_batch_sync, texts
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.debug(
            "Dense-encoded %d texts in %d ms on %s", len(texts), elapsed_ms, _device
        )
        return embeddings

    # ── Sparse ────────────────────────────────────────────────────────────

    async def sparse_encode(
        self, text: str
    ) -> tuple[list[int], list[float]]:
        """Compute BM25 sparse vector asynchronously."""
        return await self._run_in_executor(compute_sparse_vector, text)

    async def sparse_encode_batch(
        self, texts: list[str]
    ) -> list[tuple[list[int], list[float]]]:
        """Compute BM25 sparse vectors for a batch asynchronously."""

        def _batch(texts: list[str]) -> list[tuple[list[int], list[float]]]:
            return [compute_sparse_vector(t) for t in texts]

        return await self._run_in_executor(_batch, texts)

    # ── Combined ──────────────────────────────────────────────────────────

    async def embed_chunks(
        self, chunks  # list[CodeChunk]
    ) -> None:
        """
        Embed a list of CodeChunk objects in-place.
        Populates `chunk.embedding`, `chunk.sparse_indices`, `chunk.sparse_values`.
        """
        texts = [c.content for c in chunks]

        dense_vecs, sparse_results = await asyncio.gather(
            self.embed_texts(texts),
            self.sparse_encode_batch(texts),
        )

        for i, chunk in enumerate(chunks):
            chunk.embedding = dense_vecs[i].tolist()
            idx, vals = sparse_results[i]
            chunk.sparse_indices = idx
            chunk.sparse_values = vals
