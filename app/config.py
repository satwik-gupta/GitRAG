"""
app/config.py
─────────────
Central settings — loaded once from environment / .env file.
Every other module imports `settings` from here.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── PostgreSQL ─────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:root@localhost:5432/gitrag",
        description="SQLAlchemy async DSN",
    )
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30

    # ── Qdrant ─────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    # Whether to use TLS for Qdrant (both REST and gRPC). Set True for TLS-enabled servers.
    qdrant_use_tls: bool = False
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "code_chunks"
    qdrant_shard_number: int = 4
    qdrant_replication_factor: int = 1
    qdrant_vector_size: int = 768       # matches BAAI/bge-base-en-v1.5
    qdrant_on_disk_payload: bool = True
    qdrant_hnsw_ef_construct: int = 200
    qdrant_hnsw_m: int = 16

    # ── Embedding ──────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_batch_size: int = 64
    embedding_device: Optional[str] = "cpu"  # None → auto-detect GPU/CPU

    # ── Reranker ───────────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── LLM (Google Gemini) ────────────────────────────────────────────────
    gemini_api_key: str = "" # Must be set in environment or .env file to use docs generation features
    llm_model: str = "gemini-2.5-flash-lite"
    llm_max_tokens: int = 4096

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieval_top_k: int = 20          # ANN candidates before rerank
    rerank_top_n: int = 10             # after cross-encoder
    max_context_chunks: int = 8        # sent to LLM
    bm25_weight: float = 0.6
    vector_weight: float = 0.3
    metadata_boost_weight: float = 0.1

    # ── CAG Cache ──────────────────────────────────────────────────────────
    cache_ttl_seconds: int = 86_400    # 24 h default TTL

    # ── GitHub / Cloner ────────────────────────────────────────────────────
    github_token: Optional[str] = None
    clone_timeout_seconds: int = 300
    max_clone_retries: int = 3

    # ── spaCy ──────────────────────────────────────────────────────────────
    spacy_model: str = "en_core_web_sm"

    # ── Documentation Generation ───────────────────────────────────────────
    docs_output_dir: str = "generated_docs"
    templates_dir: str = "app/docs/templates"
    max_mermaid_repair_attempts: int = 3
    doc_section_max_tokens: int = 2048


settings = Settings()
