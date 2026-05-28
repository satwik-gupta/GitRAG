"""
app/main.py
────────────
FastAPI application entry-point.

Lifespan
─────────
  startup : create DB tables (if missing), connect Qdrant, warm ML models
  shutdown: close Qdrant client gracefully

Module-level singletons (qdrant_manager, embedding_worker) are imported by
the endpoint dependency injectors to avoid re-instantiation per request.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.models import Base
from app.db.session import get_engine
import app.docs.models  # noqa: F401 — registers DocGenerationJob with Base.metadata
from app.embedding.worker import EmbeddingWorker
from app.vector.qdrant_manager import QdrantManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────

qdrant_manager: QdrantManager = QdrantManager()
embedding_worker: EmbeddingWorker = EmbeddingWorker()


# ── Lifespan ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("=== GitRAG startup ===")

    # Create PostgreSQL tables (idempotent — skips existing tables)
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("DB schema verified.")

    # Connect Qdrant and ensure collection exists
    await qdrant_manager.connect()

    # Pre-warm dense embedding model in the thread pool (avoid cold-start
    # latency on the first request)
    import asyncio  # noqa: PLC0415
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: __import__("app.embedding.worker", fromlist=["_get_dense_model"])
        ._get_dense_model(),
    )
    logger.info("Embedding model loaded.")

    yield  # ← application runs here

    logger.info("=== GitRAG shutdown ===")
    await qdrant_manager.close()


# ── Application ───────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title="GitRAG",
        description=(
            "Production-grade asynchronous RAG + CAG pipeline "
            "for multi-language GitHub codebases."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.api.endpoints import router  # noqa: PLC0415
    from app.docs.endpoints import docs_router  # noqa: PLC0415

    app.include_router(router, prefix="/api/v1", tags=["GitRAG"])
    app.include_router(docs_router, prefix="/api/v1", tags=["Documentation"])

    return app


app = create_app()
