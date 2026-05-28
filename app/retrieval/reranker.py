"""
app/retrieval/reranker.py
──────────────────────────
Cross-encoder reranker built on sentence-transformers.

The reranker takes the top-K ANN candidates and re-scores every
(query, chunk) pair with a full cross-encoder pass.  This is
CPU/GPU bound; all work is dispatched to a thread-pool executor
so the event loop stays non-blocking.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.config import settings
from app.ingestion.chunker import CodeChunk

logger = logging.getLogger(__name__)

_RERANK_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rerank_")

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        logger.info("Loading cross-encoder: %s", settings.reranker_model)
        _cross_encoder = CrossEncoder(settings.reranker_model, max_length=512)
    return _cross_encoder


def _rerank_sync(query: str, chunks: list[CodeChunk], top_n: int) -> list[CodeChunk]:
    ce = _get_cross_encoder()
    pairs = [(query, c.content[:1024]) for c in chunks]
    scores = ce.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_n]]


class CrossEncoderReranker:
    """
    Async wrapper around a sentence-transformers CrossEncoder.
    """

    async def rerank(
        self,
        query: str,
        candidates: list[CodeChunk],
        top_n: Optional[int] = None,
    ) -> list[CodeChunk]:
        """
        Re-score *candidates* with the cross-encoder and return the top *top_n*.

        Parameters
        ----------
        query:      Original (un-normalised) query text for best lexical match.
        candidates: ANN / BM25 candidate chunks.
        top_n:      Number of results to return.  Defaults to ``settings.rerank_top_n``.
        """
        if not candidates:
            return []
        n = top_n if top_n is not None else settings.rerank_top_n
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _RERANK_EXECUTOR,
            _rerank_sync,
            query,
            candidates,
            n,
        )
