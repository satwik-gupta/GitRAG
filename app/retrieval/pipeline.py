"""
app/retrieval/pipeline.py
──────────────────────────
Full retrieval pipeline.

Request path (per LLD spec)
────────────────────────────
  1. Normalise query
  2. Metadata filter construction
  3. CAG cache check → return cached answer if valid
  4. Hybrid ANN search (dense + sparse BM25)
  5. Score fusion: FUSED = 0.6*BM25 + 0.3*VECTOR + 0.1*METADATA_BOOST
  6. Cross-encoder rerank → top N chunks
  7. LLM generation with source-first citations
  8. Write-through to CAG cache

LLM is Google Gemini 2.5 Flash-Lite. At most ``settings.max_context_chunks``
chunks are sent; every answer must begin with inline citations.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from google import genai

from app.cache.cag import CAGCache, compute_query_hash
from app.config import settings
from app.db.models import CagCache as CagCacheModel
from app.embedding.worker import EmbeddingWorker, compute_sparse_vector
from app.ingestion.chunker import CodeChunk
from app.retrieval.normalizer import NormalisedQuery, QueryNormaliser
from app.retrieval.reranker import CrossEncoderReranker
from app.vector.qdrant_manager import QdrantManager

logger = logging.getLogger(__name__)


# ── Result containers ─────────────────────────────────────────────────────


@dataclass
class Citation:
    file_path: str
    section_name: str
    repo_url: str
    start_line: int
    end_line: int
    commit_sha: str = ""

    def label(self) -> str:
        sha = self._short_sha()
        return (
            f"[{self.file_path}#{self.section_name}"
            f" L{self.start_line}-{self.end_line}"
            f"{(' @' + sha) if sha else ''}]"
        )

    def _short_sha(self) -> str:
        return self.commit_sha[:8] if self.commit_sha and self.commit_sha != "unknown" else ""

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "section_name": self.section_name,
            "repo_url": self.repo_url,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "commit_sha": self.commit_sha,
        }


@dataclass
class RetrievalResult:
    answer: str
    citations: list[Citation]
    canonical_query: str
    query_hash: str
    cache_hit: bool = False
    chunks_used: list[CodeChunk] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "canonical_query": self.canonical_query,
            "query_hash": self.query_hash,
            "cache_hit": self.cache_hit,
        }


# ── Score fusion ──────────────────────────────────────────────────────────


def _normalise_scores(scored_ids: dict[str, float]) -> dict[str, float]:
    """Min-max normalise a {id: score} dict to [0, 1]."""
    if not scored_ids:
        return {}
    mn, mx = min(scored_ids.values()), max(scored_ids.values())
    if mx == mn:
        return {k: 1.0 for k in scored_ids}
    return {k: (v - mn) / (mx - mn) for k, v in scored_ids.items()}


def _metadata_boost(chunk: CodeChunk, nq: NormalisedQuery) -> float:
    score = 0.0
    if nq.language_filter and chunk.language == nq.language_filter:
        score += 0.7
    if nq.file_path_filter and nq.file_path_filter in chunk.file_path:
        score += 0.3
    return min(score, 1.0)


def fuse_scores(
    vector_hits: dict[str, float],
    bm25_hits: dict[str, float],
    chunks_by_id: dict[str, CodeChunk],
    nq: NormalisedQuery,
) -> list[tuple[float, CodeChunk]]:
    """
    Combine scores using the formula from the LLD:
        FUSED = 0.6*BM25 + 0.3*VECTOR + 0.1*METADATA_BOOST
    """
    norm_vec = _normalise_scores(vector_hits)
    norm_bm25 = _normalise_scores(bm25_hits)

    all_ids = set(norm_vec) | set(norm_bm25)
    fused: list[tuple[float, CodeChunk]] = []

    for cid in all_ids:
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            continue
        vec_score = norm_vec.get(cid, 0.0)
        bm25_score = norm_bm25.get(cid, 0.0)
        meta_boost = _metadata_boost(chunk, nq)
        total = (
            settings.bm25_weight * bm25_score
            + settings.vector_weight * vec_score
            + settings.metadata_boost_weight * meta_boost
        )
        fused.append((total, chunk))

    fused.sort(key=lambda x: x[0], reverse=True)
    return fused


# ── LLM generation ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert code analyst. Answer questions about codebases using ONLY the
provided source code context. Always begin your response with inline citations
in the format [file_path#section_name L<start>-<end>], referencing every source
you draw from. If the answer cannot be determined from the provided context,
say so explicitly rather than hallucinating.
"""


def _build_user_prompt(query: str, chunks: list[CodeChunk]) -> str:
    context_parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        header = (
            f"--- Source {i}: {c.citation_label} ---\n"
            f"Language: {c.language}  |  Type: {c.doc_type}\n"
        )
        if c.docstring:
            header += f"Docstring: {c.docstring}\n"
        context_parts.append(header + c.content)

    context = "\n\n".join(context_parts)
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Provide a precise, technically accurate answer. "
        "Cite every relevant source inline."
    )


async def _call_llm(query: str, chunks: list[CodeChunk]) -> str:
    # Use the public Client and its async interface (`.aio`) for async calls.
    async with genai.Client(api_key=settings.gemini_api_key).aio as client:
        # `generate_content` expects a `config` that may include
        # `system_instruction`; pass it via `GenerateContentConfig`.
        gen_config = genai.types.GenerateContentConfig(
            max_output_tokens=settings.llm_max_tokens,
            system_instruction=_SYSTEM_PROMPT,
        )
        message = await client.models.generate_content(
            model=settings.llm_model,
            contents=_build_user_prompt(query, chunks),
            config=gen_config,
        )
    return message.text


# ── Pipeline ──────────────────────────────────────────────────────────────


class RetrievalPipeline:
    """
    Orchestrates the full retrieval pipeline.

    Parameters
    ----------
    qdrant:   Connected :class:`QdrantManager`.
    embedder: Initialised :class:`EmbeddingWorker`.
    """

    def __init__(
        self,
        qdrant: QdrantManager,
        embedder: EmbeddingWorker,
    ) -> None:
        self._qdrant = qdrant
        self._embedder = embedder
        self._normaliser = QueryNormaliser()
        self._reranker = CrossEncoderReranker()

    # ── Public ─────────────────────────────────────────────────────────────

    async def query(
        self,
        raw_query: str,
        cag_cache: CAGCache,
        repo_url: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Execute the full retrieval pipeline for *raw_query*.

        Parameters
        ----------
        raw_query:  User-facing question string.
        cag_cache:  CAGCache instance bound to an active DB session.
        repo_url:   Optional repo scope restriction.
        top_k:      Override ANN candidate count.
        """
        k = top_k or settings.retrieval_top_k

        # ── Step 1: Normalise ──────────────────────────────────────────────
        loop = asyncio.get_running_loop()
        nq: NormalisedQuery = await loop.run_in_executor(
            None, self._normaliser.normalise, raw_query
        )
        logger.info(
            "Query normalised  hash=%s… lang=%s path=%s",
            nq.query_hash[:16],
            nq.language_filter,
            nq.file_path_filter,
        )

        # ── Step 2: Metadata filter ────────────────────────────────────────
        qdrant_filter = self._qdrant.build_filter(
            language=nq.language_filter,
            file_path_prefix=nq.file_path_filter,
            repo_url=repo_url,
        )

        # ── Step 3: CAG cache check ────────────────────────────────────────
        cached: Optional[CagCacheModel] = await cag_cache.get(nq.canonical)
        if cached:
            citations = [
                Citation(**c) for c in (cached.source_citations or [])
            ]
            return RetrievalResult(
                answer=cached.answer,
                citations=citations,
                canonical_query=nq.canonical,
                query_hash=nq.query_hash,
                cache_hit=True,
            )

        # ── Step 4: Hybrid ANN search ──────────────────────────────────────
        dense_vec, sparse_result = await asyncio.gather(
            self._embedder.embed_texts([raw_query]),
            loop.run_in_executor(None, compute_sparse_vector, raw_query),
        )
        query_dense = dense_vec[0].tolist()
        sparse_idx, sparse_vals = sparse_result

        vector_hits_raw, bm25_hits_raw = await asyncio.gather(
            self._qdrant.dense_search(query_dense, k, qdrant_filter),
            self._qdrant.sparse_search(sparse_idx, sparse_vals, k, qdrant_filter),
        )

        # Build chunk objects from Qdrant payloads + score dicts
        chunks_by_id: dict[str, CodeChunk] = {}
        vector_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}

        for hit in vector_hits_raw:
            chunk = self._payload_to_chunk(hit.payload)
            chunks_by_id[chunk.chunk_id] = chunk
            vector_scores[chunk.chunk_id] = float(hit.score)

        for hit in bm25_hits_raw:
            chunk = self._payload_to_chunk(hit.payload)
            if chunk.chunk_id not in chunks_by_id:
                chunks_by_id[chunk.chunk_id] = chunk
            bm25_scores[chunk.chunk_id] = float(hit.score)

        # ── Step 5: Score fusion ───────────────────────────────────────────
        fused = fuse_scores(vector_scores, bm25_scores, chunks_by_id, nq)
        candidates = [c for _, c in fused[:k]]

        # ── Step 6: Rerank ─────────────────────────────────────────────────
        top_chunks = await self._reranker.rerank(
            raw_query,
            candidates,
            top_n=settings.max_context_chunks,
        )

        if not top_chunks:
            return RetrievalResult(
                answer="No relevant code context found for this query.",
                citations=[],
                canonical_query=nq.canonical,
                query_hash=nq.query_hash,
                cache_hit=False,
            )

        # ── Step 7: LLM generation ─────────────────────────────────────────
        answer = await _call_llm(raw_query, top_chunks)

        citations = [
            Citation(
                file_path=c.file_path,
                section_name=c.qualified_name,
                repo_url=c.repo_url,
                start_line=c.start_line,
                end_line=c.end_line,
                commit_sha=c.commit_sha or "",
            )
            for c in top_chunks
        ]

        # ── Step 8: Write-through cache ────────────────────────────────────
        await cag_cache.store(
            canonical_query=nq.canonical,
            context_chunks=top_chunks,
            answer=answer,
            source_citations=[c.to_dict() for c in citations],
        )

        return RetrievalResult(
            answer=answer,
            citations=citations,
            canonical_query=nq.canonical,
            query_hash=nq.query_hash,
            cache_hit=False,
            chunks_used=top_chunks,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _payload_to_chunk(payload: dict) -> CodeChunk:
        return CodeChunk(
            chunk_id=payload["chunk_id"],
            content=payload.get("content", ""),
            docstring=payload.get("docstring") or None,
            language=payload.get("language", ""),
            doc_type=payload.get("doc_type", "function"),
            section_name=payload.get("section_name", ""),
            parent_name=payload.get("parent_name") or None,
            annotations=payload.get("annotations", []),
            repo_url=payload.get("repo_url", ""),
            file_path=payload.get("file_path", ""),
            start_line=payload.get("start_line", 0),
            end_line=payload.get("end_line", 0),
            commit_sha=payload.get("commit_sha") or None,
        )
