"""
app/vector/qdrant_manager.py
─────────────────────────────
Async Qdrant client wrapper.

Collection layout
─────────────────
  Named vectors
    "dense"   – BAAI/bge-base-en-v1.5 (768-dim, cosine, HNSW + Scalar Quant)
    "sparse"  – BM25 sparse vectors (dot product)

  Sharding    – configurable (default 4 shards)
  Replication – configurable (default 1)
  On-disk payload storage enabled for large-scale deployments.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app.config import settings

logger = logging.getLogger(__name__)


class QdrantManager:
    """
    Thin async wrapper around :class:`AsyncQdrantClient`.

    All public methods are coroutines.
    """

    def __init__(self) -> None:
        self._client: Optional[AsyncQdrantClient] = None
        self._collection = settings.qdrant_collection

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create the async client and ensure the collection exists."""
        try:
            # Log connection intent (mask api_key presence for safety)
            logger.debug(
                "Initializing AsyncQdrantClient host=%s port=%s grpc_port=%s prefer_grpc=%s use_tls=%s api_key_present=%s",
                settings.qdrant_host,
                settings.qdrant_port,
                settings.qdrant_grpc_port,
                True,
                settings.qdrant_use_tls,
                bool(settings.qdrant_api_key),
            )
            self._client = AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                grpc_port=settings.qdrant_grpc_port,
                prefer_grpc=True,
                https=settings.qdrant_use_tls,
                api_key=settings.qdrant_api_key,
            )
            await self._ensure_collection()
            logger.info(
                "QdrantManager connected to %s:%s (grpc_port=%s), collection=%r, tls=%s",
                settings.qdrant_host,
                settings.qdrant_port,
                settings.qdrant_grpc_port,
                self._collection,
                settings.qdrant_use_tls,
            )
        except Exception:
            logger.exception(
                "Failed to connect to Qdrant at %s:%s (grpc_port=%s), tls=%s",
                settings.qdrant_host,
                settings.qdrant_port,
                settings.qdrant_grpc_port,
                settings.qdrant_use_tls,
            )
            raise

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("QdrantManager.connect() must be called first.")
        return self._client

    # ── Collection setup ──────────────────────────────────────────────────

    async def _ensure_collection(self) -> None:
        try:
            collections = await self.client.get_collections()
            existing = [c.name for c in collections.collections]
            logger.debug("Qdrant collections fetched: %s", existing)
            if self._collection in existing:
                logger.debug("Collection %r already exists.", self._collection)
                return
        except Exception:
            logger.exception("Failed to list Qdrant collections (host=%s grpc_port=%s)", settings.qdrant_host, settings.qdrant_grpc_port)
            raise

        logger.info("Creating Qdrant collection %r …", self._collection)
        await self.client.create_collection(
            collection_name=self._collection,
            vectors_config={
                "dense": qm.VectorParams(
                    size=settings.qdrant_vector_size,
                    distance=qm.Distance.COSINE,
                    hnsw_config=qm.HnswConfigDiff(
                        ef_construct=settings.qdrant_hnsw_ef_construct,
                        m=settings.qdrant_hnsw_m,
                        on_disk=True,
                    ),
                    quantization_config=qm.ScalarQuantization(
                        scalar=qm.ScalarQuantizationConfig(
                            type=qm.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=False,
                        ),
                    ),
                    on_disk=settings.qdrant_on_disk_payload,
                ),
            },
            sparse_vectors_config={
                "sparse": qm.SparseVectorParams(
                    index=qm.SparseIndexParams(on_disk=True),
                ),
            },
            shard_number=settings.qdrant_shard_number,
            replication_factor=settings.qdrant_replication_factor,
            on_disk_payload=settings.qdrant_on_disk_payload,
            optimizers_config=qm.OptimizersConfigDiff(
                indexing_threshold=20_000,
                memmap_threshold=50_000,
            ),
        )
        # Payload indices for metadata-filter pre-filtering
        for field_name, field_type in [
            ("language", qm.PayloadSchemaType.KEYWORD),
            ("file_path", qm.PayloadSchemaType.KEYWORD),
            ("repo_url", qm.PayloadSchemaType.KEYWORD),
            ("doc_type", qm.PayloadSchemaType.KEYWORD),
        ]:
            await self.client.create_payload_index(
                collection_name=self._collection,
                field_name=field_name,
                field_schema=field_type,
            )
        logger.info("Collection %r created with indices.", self._collection)

    # ── Write ─────────────────────────────────────────────────────────────

    async def upsert_points(self, points: list[qm.PointStruct]) -> None:
        """Upsert a batch of PointStruct objects (supports both dense + sparse)."""
        await self.client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,
        )

    def build_point(
        self,
        point_id: str,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        payload: dict[str, Any],
    ) -> qm.PointStruct:
        return qm.PointStruct(
            id=point_id,
            vector={
                "dense": dense_vector,
                "sparse": qm.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
            },
            payload=payload,
        )

    # ── Read / Search ─────────────────────────────────────────────────────

    async def dense_search(
        self,
        query_vector: list[float],
        top_k: int,
        filter_: Optional[qm.Filter] = None,
    ) -> list[qm.ScoredPoint]:
        """ANN search on the dense HNSW index."""
        resp = await self.client.query_points(
            collection_name=self._collection,
            query=query_vector,
            using="dense",
            limit=top_k,
            query_filter=filter_,
            with_payload=True,
            with_vectors=False,
        )
        return resp.points

    async def sparse_search(
        self,
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int,
        filter_: Optional[qm.Filter] = None,
    ) -> list[qm.ScoredPoint]:
        """BM25 sparse vector search."""
        sparse_vec = qm.SparseVector(indices=sparse_indices, values=sparse_values)
        resp = await self.client.query_points(
            collection_name=self._collection,
            query=sparse_vec,
            using="sparse",
            limit=top_k,
            query_filter=filter_,
            with_payload=True,
            with_vectors=False,
        )
        return resp.points

    async def scroll_by_repo(self, repo_url: str, batch_size: int = 100) -> list[str]:
        """Return all point IDs belonging to a given repo (for invalidation)."""
        point_ids: list[str] = []
        offset = None
        filter_ = qm.Filter(
            must=[qm.FieldCondition(key="repo_url", match=qm.MatchValue(value=repo_url))]
        )
        while True:
            records, next_offset = await self.client.scroll(
                collection_name=self._collection,
                scroll_filter=filter_,
                limit=batch_size,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for r in records:
                point_ids.append(str(r.id))
            if next_offset is None:
                break
            offset = next_offset
        return point_ids

    async def scroll_chunks_by_filter(
        self,
        filter_: Optional[qm.Filter] = None,
        limit: int = 500,
    ) -> list[dict]:
        """
        Return point payloads (no vectors) matching *filter_*.
        Used by the macro-context aggregator to build structural overviews
        without running an ANN search.
        """
        all_payloads: list[dict] = []
        offset = None
        while True:
            records, next_offset = await self.client.scroll(
                collection_name=self._collection,
                scroll_filter=filter_,
                limit=min(limit, 250),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                if r.payload:
                    all_payloads.append(r.payload)
            if next_offset is None or len(all_payloads) >= limit:
                break
            offset = next_offset
        return all_payloads[:limit]

    async def delete_by_repo(self, repo_url: str) -> int:
        """Delete all points for *repo_url*. Returns count deleted."""
        ids = await self.scroll_by_repo(repo_url)
        if ids:
            await self.client.delete(
                collection_name=self._collection,
                points_selector=qm.PointIdsList(points=ids),
                wait=True,
            )
        return len(ids)

    # ── Metadata filter builder ────────────────────────────────────────────

    @staticmethod
    def build_filter(
        language: Optional[str] = None,
        file_path_prefix: Optional[str] = None,
        repo_url: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> Optional[qm.Filter]:
        """Compose a Qdrant Filter from optional query constraints."""
        must: list[qm.Condition] = []
        if language:
            must.append(
                qm.FieldCondition(key="language", match=qm.MatchValue(value=language))
            )
        if file_path_prefix:
            must.append(
                qm.FieldCondition(
                    key="file_path",
                    match=qm.MatchText(text=file_path_prefix),
                )
            )
        if repo_url:
            must.append(
                qm.FieldCondition(key="repo_url", match=qm.MatchValue(value=repo_url))
            )
        if doc_type:
            must.append(
                qm.FieldCondition(key="doc_type", match=qm.MatchValue(value=doc_type))
            )
        return qm.Filter(must=must) if must else None
