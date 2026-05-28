"""
app/cache/cag.py
─────────────────
Cache-Augmented Generation (CAG) layer.

Responsibilities
────────────────
1. **Query hashing** – deterministically maps a canonical (normalised) query
   string to a SHA-256 hex digest.  Equal canonical queries always produce
   the same hash, enabling exact-match cache hits.

2. **Context fingerprinting** – given the ordered list of CodeChunks that
   would be sent to the LLM, produces a deterministic SHA-256 fingerprint
   (sorted chunk-ID hashes concatenated).  If underlying chunks change
   (re-ingestion), the fingerprint changes and the cached answer is stale.

3. **Hit / miss logic** – checks the `cag_cache` table, validates staleness
   and TTL, increments hit counters, and returns the cached answer or None.

4. **Write-through** – after the LLM generates a fresh answer, the caller
   calls :meth:`store` to persist the mapping for future requests.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import CagCache
from app.db.repositories import CagCacheRepository
from app.ingestion.chunker import CodeChunk

logger = logging.getLogger(__name__)


# ── Fingerprinting helpers ─────────────────────────────────────────────────


def compute_query_hash(canonical_query: str) -> str:
    """SHA-256 of the canonical (normalised) query string."""
    return hashlib.sha256(canonical_query.encode("utf-8")).hexdigest()


def compute_context_fingerprint(chunks: list[CodeChunk]) -> str:
    """
    Deterministic fingerprint for an ordered set of chunks.

    Sorts chunk IDs before hashing so that different ordering of the same
    retrieval set produces the same fingerprint.
    """
    sorted_ids = sorted(c.chunk_id for c in chunks)
    combined = "|".join(sorted_ids)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ── CAGCache service ──────────────────────────────────────────────────────


class CAGCache:
    """
    Semantic cache service backed by PostgreSQL `cag_cache` table.

    Parameters
    ----------
    session:
        An AsyncSession.  The caller manages the transaction boundary.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._repo = CagCacheRepository(session)

    # ── Public API ─────────────────────────────────────────────────────────

    async def get(
        self,
        canonical_query: str,
    ) -> Optional[CagCache]:
        """
        Look up a cached answer for *canonical_query*.

        Returns
        -------
        The :class:`CagCache` ORM row if a valid, non-stale, non-expired
        entry exists; ``None`` otherwise.
        """
        query_hash = compute_query_hash(canonical_query)
        entry: Optional[CagCache] = await self._repo.get_by_query_hash(query_hash)

        if entry is None:
            logger.debug("CAG miss  hash=%s…", query_hash[:16])
            return None

        if entry.is_stale:
            logger.debug("CAG stale hash=%s…", query_hash[:16])
            return None

        if entry.expires_at and entry.expires_at < datetime.now(timezone.utc):
            logger.debug("CAG expired hash=%s…", query_hash[:16])
            return None

        await self._repo.record_hit(query_hash)
        logger.info(
            "CAG HIT  hash=%s… hits=%d", query_hash[:16], entry.hit_count + 1
        )
        return entry

    async def store(
        self,
        canonical_query: str,
        context_chunks: list[CodeChunk],
        answer: str,
        source_citations: Optional[list[dict]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> CagCache:
        """
        Persist a new (or refreshed) cache entry.

        Parameters
        ----------
        canonical_query:   Normalised query string.
        context_chunks:    The exact chunks sent to the LLM.
        answer:            LLM-generated answer text.
        source_citations:  List of citation dicts to store alongside the answer.
        ttl_seconds:       Override the default TTL from settings.
        """
        query_hash = compute_query_hash(canonical_query)
        fingerprint = compute_context_fingerprint(context_chunks)
        ttl = ttl_seconds if ttl_seconds is not None else settings.cache_ttl_seconds
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)

        entry = await self._repo.upsert(
            query_hash=query_hash,
            canonical_query=canonical_query,
            context_fingerprint=fingerprint,
            answer=answer,
            source_citations=source_citations,
            expires_at=expires_at,
        )
        logger.info(
            "CAG stored hash=%s… fingerprint=%s…", query_hash[:16], fingerprint[:16]
        )
        return entry

    async def invalidate_for_repo(self, repo_url: str) -> int:
        """Mark all entries that cite chunks from *repo_url* as stale."""
        count = await self._repo.mark_stale_for_repo(repo_url)
        logger.info("CAG invalidated %d entries for repo %s", count, repo_url)
        return count
