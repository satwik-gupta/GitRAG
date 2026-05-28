# GitRAG

**A production-grade, fully async RAG + CAG pipeline that ingests any GitHub codebase, builds a temporal knowledge graph over a hybrid vector index, and serves two output modes — source-cited Q&A with cache-first precision, and automated generation of HLD, LLD, SOP, and Mermaid architecture documents.**

---

## Table of Contents

1. [Architecture & System Design](#architecture--system-design)
   - [Technical Stack](#technical-stack)
   - [Two Parallel Processing Paths](#two-parallel-processing-paths)
   - [Path A — RAG Query Pipeline](#path-a--rag-query-pipeline)
   - [Path B — Documentation Generation Pipeline](#path-b--documentation-generation-pipeline)
   - [Shared Ingestion Pipeline](#shared-ingestion-pipeline)
   - [Directory Layout](#directory-layout)
   - [Qdrant Collection Design](#qdrant-collection-design)
   - [PostgreSQL Schema](#postgresql-schema)
2. [Prerequisites & Environment Setup](#prerequisites--environment-setup)
3. [Getting Started & Build Instructions](#getting-started--build-instructions)
4. [API Reference](#api-reference)
   - [Ingestion](#ingestion)
   - [RAG Query](#rag-query)
   - [Documentation Generation](#documentation-generation)
   - [Cache Management](#cache-management)
   - [Endpoint Summary](#endpoint-summary)
5. [Built-in Document Templates](#built-in-document-templates)
6. [Running Tests](#running-tests)
7. [Configuration Reference](#configuration-reference)

---

## Architecture & System Design

### Technical Stack

| Layer | Technology | Version |
|---|---|---|
| **Language & Runtime** | Python | 3.11+ |
| **Web Framework** | FastAPI + Uvicorn | 0.111+ / 0.29+ |
| **Database (State)** | PostgreSQL + asyncpg | 14+ / 0.29+ |
| **ORM** | SQLAlchemy 2.x (async mapped dataclasses) | 2.0+ |
| **Migrations** | Alembic | 1.13+ |
| **Vector DB** | Qdrant (AsyncQdrantClient, gRPC) | 1.9+ |
| **Dense Embedder** | `BAAI/bge-base-en-v1.5` via sentence-transformers | 768-dim |
| **Sparse Embedder** | BM25 (vocabulary-hash, no index file needed) | — |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | — |
| **LLM** | Anthropic claude-sonnet-4-6 | — |
| **AST Parsing** | tree-sitter 0.22+ (Python / Java / Go / C++) | — |
| **Query NLP** | spaCy `en_core_web_sm` | 3.7+ |
| **Retry Logic** | tenacity | 8.3+ |
| **Async File I/O** | aiofiles | 23.2+ |

---

### Two Parallel Processing Paths

GitRAG exposes two independent output modes that share the same ingestion index. A single API request is routed to the appropriate path based on the endpoint called.

```
Ingested GitHub Repository
(Qdrant vectors + PostgreSQL graph)
              │
              ├─────────────────────────────────────────────┐
              │                                             │
              ▼                                             ▼
   ┌──────────────────────┐                   ┌─────────────────────────┐
   │  PATH A — RAG QUERY  │                   │  PATH B — DOC GENERATOR │
   │  POST /query         │                   │  POST /generate-docs    │
   │                      │                   │                         │
   │  micro-retrieval:    │                   │  macro-synthesis:       │
   │  5–10 specific chunks│                   │  full graph topology +  │
   │  → cited Q&A answer  │                   │  signatures → HLD/LLD/  │
   │  → CAG cache         │                   │  SOP/DIAGRAM .md file   │
   └──────────────────────┘                   └─────────────────────────┘
```

---

### Path A — RAG Query Pipeline

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  1. Query Normalisation         │  lowercase → lemmatise (spaCy) → sort tokens
│     + Metadata Filter           │  extract: language, file_path constraints
└──────────────┬──────────────────┘
               │ canonical query + hash
               ▼
┌─────────────────────────────────┐
│  2. CAG Cache Check             │  SHA-256(canonical) → PostgreSQL cag_cache
│                                 │  validates staleness + TTL
└──────────────┬──────────────────┘
        hit ◄──┴──► miss
        │              │
        │              ▼
        │  ┌─────────────────────────────────────────────┐
        │  │  3. Hybrid ANN Search                       │
        │  │     dense:  BAAI/bge cosine (HNSW, INT8)    │
        │  │     sparse: BM25 dot-product (Qdrant sparse) │
        │  └──────────────┬──────────────────────────────┘
        │                 │ up to top_k=20 candidates each
        │                 ▼
        │  ┌─────────────────────────────────────────────┐
        │  │  4. Score Fusion                             │
        │  │     FUSED = 0.6·BM25 + 0.3·VECTOR           │
        │  │                      + 0.1·METADATA_BOOST   │
        │  └──────────────┬──────────────────────────────┘
        │                 │ top-K fused candidates
        │                 ▼
        │  ┌─────────────────────────────────────────────┐
        │  │  5. Cross-Encoder Rerank                     │
        │  │     ms-marco-MiniLM-L-6-v2                   │
        │  │     → top 8 chunks (max_context_chunks)      │
        │  └──────────────┬──────────────────────────────┘
        │                 │ ≤8 chunks
        │                 ▼
        │  ┌─────────────────────────────────────────────┐
        │  │  6. LLM Generation (claude-sonnet-4-6)       │
        │  │     system: cite every source inline         │
        │  │     answer prefixed with [file#section Lx-y] │
        │  └──────────────┬──────────────────────────────┘
        │                 │ answer + citations
        │                 ▼
        │  ┌─────────────────────────────────────────────┐
        │  │  7. Write-Through CAG Cache                  │
        │  │     query_hash → context_fingerprint → answer│
        │  └─────────────────────────────────────────────┘
        │                 │
        └────────►  JSON Response
```

---

### Path B — Documentation Generation Pipeline

```
POST /generate-docs
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  1. Intent Router & Template Resolution                  │
│     • document_type: HLD | LLD | SOP | DIAGRAM          │
│     • template_source: LOCAL (built-in) | CUSTOM (raw)  │
│     • parse sections, {{context:}} keys, {{diagram:}}   │
│     → creates DocGenerationJob row (PENDING)            │
└──────────────────────────┬──────────────────────────────┘
                           │ returns 202 + job_id immediately
                           ▼  (runs as BackgroundTask)
┌─────────────────────────────────────────────────────────┐
│  2. Macro-Context Aggregation   [AGGREGATING_CONTEXT]   │
│     ┌─ PostgreSQL                                       │
│     │   GraphNode + GraphEdge → call graph, imports     │
│     │   IngestionJob list → file tree, language dist.   │
│     └─ Qdrant (payload scroll, no ANN)                  │
│         class + function signatures, docstrings         │
│     → StructuredContext (serialisable by context key)   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  3. Iterative Section-by-Section Generation  [GENERATING]│
│     For each template section:                           │
│       a. select context keys → focused topology string  │
│       b. append last 3 sections (continuity summary)    │
│       c. append section instruction text                │
│       d. call LLM (claude-sonnet-4-6, per-section tokens)│
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  4. Mermaid Syntax Validation & Auto-Repair [VALIDATING] │
│     For every ```mermaid``` block in the document:       │
│       • check diagram-type declaration                  │
│       • check balanced brackets / parens / braces       │
│       • check arrow syntax (no ->, => in flowcharts)    │
│       • check orphaned 'end' / duplicate node IDs       │
│     If invalid → LLM repair pass (up to 3 retries)      │
│     Still invalid → embed <!--VALIDATION_FAILED--> note │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  5. Output & State Finalisation       [COMPLETED]        │
│     • write to generated_docs/{repo}_{type}_{ts}.md     │
│     • persist output_path in doc_generation_jobs        │
│     • available at GET /doc-jobs/{job_id}/download      │
└─────────────────────────────────────────────────────────┘
```

**DocGenerationJob state machine:**
```
PENDING → AGGREGATING_CONTEXT → GENERATING → VALIDATING → COMPLETED
                                                               ↑
                                  FAILED ←──── (any exception at any stage)
```

---

### Shared Ingestion Pipeline

Both output paths consume the same index. Run ingestion once, query or generate docs as many times as needed.

```
GitHub URL
    │
    ▼
AsyncGitHubCloner          git clone --depth=1 (async subprocess, PAT auth, retry)
    │
    ▼
ChunkingEngine             discovers .py / .java / .go / .cpp / .h / .cc / ...
    ├── tree-sitter AST    extracts functions, classes, methods + docstrings
    └── line-window        60-line overlapping fallback when AST yields nothing
    │
    ▼                      CodeChunk{chunk_id, content, language, doc_type,
    │                                section_name, file_path, repo_url, ...}
    ├──► EmbeddingWorker   dense (GPU→CPU fallback) + BM25 sparse, async executor
    │        │
    │        ▼
    │    QdrantManager     upsert named vectors {"dense": [...], "sparse": {i,v}}
    │
    └──► TemporalKnowledgeGraph
             │             extracts: calls / imports / inherits / implements / ffi_calls
             ▼
         PostgreSQL        GraphNode + GraphEdge (commit_sha, valid_from timestamps)
```

---

### Directory Layout

```
GitRAG/
├── app/
│   ├── main.py                      # FastAPI app + lifespan (DB init, Qdrant connect, model warm-up)
│   ├── config.py                    # pydantic-settings Settings — all knobs in one place
│   │
│   ├── db/
│   │   ├── models.py                # ORM: Workflow, IngestionJob, EmbeddingJob,
│   │   │                            #       CagCache, GraphNode, GraphEdge
│   │   ├── repositories.py          # Thin async repos (no raw SQL in service code)
│   │   ├── session.py               # Async engine + get_session() / get_db() dependency
│   │   ├── 001_initial_schema.py    # Alembic migration — core tables
│   │   └── 002_doc_generation_jobs.py  # Alembic migration — doc engine table
│   │
│   ├── ingestion/
│   │   ├── cloner.py                # AsyncGitHubCloner
│   │   ├── ast_parsers.py           # tree-sitter parsers → RawEntity list
│   │   └── chunker.py               # ChunkingEngine → CodeChunk list
│   │
│   ├── vector/
│   │   └── qdrant_manager.py        # Collection setup (HNSW + quant + sparse) + CRUD
│   │
│   ├── embedding/
│   │   └── worker.py                # EmbeddingWorker (dense GPU/CPU + BM25 sparse)
│   │
│   ├── graph/
│   │   └── knowledge_graph.py       # TemporalKnowledgeGraph builder
│   │
│   ├── cache/
│   │   └── cag.py                   # CAGCache (fingerprint, hit/miss, TTL, staleness)
│   │
│   ├── retrieval/
│   │   ├── normalizer.py            # QueryNormaliser (spaCy lemmatise + entity extract)
│   │   ├── reranker.py              # CrossEncoderReranker (thread-pool executor)
│   │   └── pipeline.py              # RetrievalPipeline — full 8-step orchestration
│   │
│   ├── docs/                        # ── Documentation Generation Engine ──
│   │   ├── models.py                # DocGenerationJob ORM model (UUID PK, state machine)
│   │   ├── repository.py            # DocGenerationJobRepository
│   │   ├── schemas.py               # Pydantic models for doc-gen API
│   │   ├── template_manager.py      # TemplateManager (YAML frontmatter + {{context:}} parser)
│   │   ├── context_aggregator.py    # MacroContextAggregator → StructuredContext
│   │   ├── mermaid_validator.py     # MermaidValidator (5 rules + LLM repair loop)
│   │   ├── generation_engine.py     # IterativeDocumentConstructor (section-by-section)
│   │   ├── orchestrator.py          # DocGenerationOrchestrator (state machine controller)
│   │   ├── endpoints.py             # docs_router: /generate-docs, /doc-jobs, /templates
│   │   └── templates/
│   │       ├── hld.md               # High-Level Design (6 sections, flowchart + sequenceDiagram)
│   │       ├── lld.md               # Low-Level Design (7 sections, classDiagram + erDiagram)
│   │       ├── sop.md               # Standard Operating Procedure (7 sections)
│   │       └── diagram.md           # Architecture Diagram Suite (4 Mermaid views)
│   │
│   └── api/
│       ├── schemas.py               # Pydantic v2 models for RAG API
│       └── endpoints.py             # FastAPI router (ingest, query, workflows, cache, health)
│
├── generated_docs/                  # Output directory — generated .md files land here
├── requirements.txt
└── README.md
```

---

### Qdrant Collection Design

The single collection `code_chunks` stores two named vector types per point:

| Vector | Type | Model | Distance | Index |
|---|---|---|---|---|
| `dense` | float32 [768] | BAAI/bge-base-en-v1.5 | Cosine | HNSW (`ef=200, m=16`) + INT8 scalar quantisation, on-disk |
| `sparse` | `{indices, values}` | BM25 (SHA-256 token IDs) | Dot product | Qdrant sparse inverted index, on-disk |

Payload indices on `language`, `file_path`, `repo_url`, `doc_type` enable sub-millisecond metadata pre-filtering before ANN search. The doc engine uses **payload scroll** (not ANN) to retrieve class and function signatures for macro-context assembly.

---

### PostgreSQL Schema

| Table | Migration | Purpose |
|---|---|---|
| `workflows` | 001 | One row per repo ingestion — tracks status through `pending → cloning → parsing → embedding → graphing → completed` |
| `ingestion_jobs` | 001 | One row per source file processed inside a workflow |
| `embedding_jobs` | 001 | One row per embedding batch; records device used, duration, and Qdrant point IDs |
| `cag_cache` | 001 | Semantic cache: `query_hash → context_fingerprint → answer`, with hit counter, TTL, and staleness flag |
| `graph_nodes` | 001 | Code entities (function, class, method, reference) with `first_seen_commit` and `last_seen_commit` |
| `graph_edges` | 001 | Directed temporal relationships: `calls / imports / inherits / implements / ffi_calls` tagged with `commit_sha + valid_from` |
| `doc_generation_jobs` | 002 | One row per doc-gen request — UUID PK, tracks state machine, `current_section`, and `output_path` |

---

## Prerequisites & Environment Setup

### System Requirements

- **Python** 3.11 or higher
- **Git** 2.x (must be on `PATH` — the cloner calls it as a subprocess)
- **PostgreSQL** 14 or higher (local or remote)
- **Qdrant** 1.9 or higher (Docker recommended, see below)
- **GPU** (optional) — CUDA-capable GPU speeds up embedding; falls back to CPU automatically

### Quick-start: External Services via Docker

```bash
# PostgreSQL
docker run -d \
  --name gitrag-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=gitrag \
  -p 5432:5432 \
  postgres:16

# Qdrant
docker run -d \
  --name gitrag-qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant:latest
```

### Environment Variables

Copy the template below to a `.env` file in the project root.

```dotenv
# ── PostgreSQL ──────────────────────────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/gitrag
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# ── Qdrant ──────────────────────────────────────────────────────────────────
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_API_KEY=                        # leave blank for local/unauthenticated
QDRANT_COLLECTION=code_chunks
QDRANT_SHARD_NUMBER=4
QDRANT_VECTOR_SIZE=768                 # must match embedding model output dim

# ── LLM ─────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...           # Required — get from console.anthropic.com
LLM_MODEL=claude-sonnet-4-6
LLM_MAX_TOKENS=4096

# ── Embedding ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_BATCH_SIZE=64
EMBEDDING_DEVICE=                      # leave blank for auto (cuda if available)

# ── Retrieval ────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K=20
RERANK_TOP_N=10
MAX_CONTEXT_CHUNKS=8
BM25_WEIGHT=0.6
VECTOR_WEIGHT=0.3
METADATA_BOOST_WEIGHT=0.1

# ── CAG Cache ────────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS=86400                # 24 hours

# ── GitHub ───────────────────────────────────────────────────────────────────
GITHUB_TOKEN=ghp_...                   # Optional — required for private repos
CLONE_TIMEOUT_SECONDS=300
MAX_CLONE_RETRIES=3

# ── Documentation Generation ─────────────────────────────────────────────────
DOCS_OUTPUT_DIR=generated_docs         # directory where .md files are written
TEMPLATES_DIR=app/docs/templates       # path to built-in template files
MAX_MERMAID_REPAIR_ATTEMPTS=3          # LLM auto-repair retries per broken diagram
DOC_SECTION_MAX_TOKENS=2048            # max LLM tokens per document section
```

> **Note:** `ANTHROPIC_API_KEY` is the only hard requirement. All other values have sensible defaults that work out of the box for a local setup.

---

## Getting Started & Build Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-org/gitrag.git
cd gitrag
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure the environment

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY at minimum
```

### 6. Set up the database

Tables are created automatically on startup via `create_all`. For production, use Alembic migrations:

```bash
# First time only
alembic init alembic

# Apply both migrations (001 core schema + 002 doc engine)
alembic upgrade head
```

### 7. Create the output directory

```bash
mkdir generated_docs
```

### 8. Start the server

```bash
# Development — hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production — 4 workers
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

The API is now live:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **Health check:** `http://localhost:8000/api/v1/health`

---

## API Reference

### Ingestion

Ingest a repository. Returns immediately with a `workflow_id` to poll.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/pallets/flask", "branch": "main"}'
```

```json
{
  "workflow_id": 1,
  "repo_url": "https://github.com/pallets/flask",
  "branch": "main",
  "status": "pending",
  "message": "Ingestion workflow 1 enqueued."
}
```

Poll for completion:

```bash
curl http://localhost:8000/api/v1/workflows/1
```

```json
{
  "workflow_id": 1,
  "status": "completed",
  "total_files": 48,
  "processed_files": 48,
  "total_chunks": 312,
  "commit_sha": "a1b2c3d4e5f6..."
}
```

---

### RAG Query

Query the indexed codebase with natural language. The response always includes inline citations.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does Flask handle request context?",
    "repo_url": "https://github.com/pallets/flask"
  }'
```

```json
{
  "answer": "[src/flask/ctx.py#RequestContext.push L89-134] Flask manages request context using a context-local stack...",
  "citations": [
    {
      "file_path": "src/flask/ctx.py",
      "section_name": "RequestContext.push",
      "start_line": 89,
      "end_line": 134,
      "commit_sha": "a1b2c3d4"
    }
  ],
  "cache_hit": false,
  "query_hash": "e3b0c44298fc1c..."
}
```

A second identical query returns instantly with `"cache_hit": true` — the LLM is bypassed entirely.

---

### Documentation Generation

Generate a full technical document from an ingested repository. Returns a `job_id` immediately; the document is built asynchronously.

#### Trigger generation using a built-in template

```bash
curl -X POST http://localhost:8000/api/v1/generate-docs \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/pallets/flask",
    "document_type": "HLD",
    "template_source": "LOCAL",
    "template_name": "hld"
  }'
```

```json
{
  "job_id": "3f8a1b2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
  "repo_url": "https://github.com/pallets/flask",
  "document_type": "HLD",
  "status": "pending",
  "message": "Documentation generation job 3f8a1b2c enqueued. Poll /api/v1/doc-jobs/3f8a1b2c-... for status."
}
```

#### Trigger generation using a custom template

```bash
curl -X POST http://localhost:8000/api/v1/generate-docs \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/pallets/flask",
    "document_type": "HLD",
    "template_source": "CUSTOM",
    "template_content": "---\nname: my_template\ndocument_type: HLD\nversion: \"1.0\"\ndescription: \"Custom template\"\n---\n\n## Overview\n{{context: repo_name, language_distribution}}\nWrite a brief overview of the system."
  }'
```

#### Poll job status

```bash
curl http://localhost:8000/api/v1/doc-jobs/3f8a1b2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c
```

```json
{
  "job_id": "3f8a1b2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
  "repo_url": "https://github.com/pallets/flask",
  "document_type": "HLD",
  "status": "generating",
  "current_section": "System Architecture",
  "output_path": null,
  "error_log": null
}
```

When `status` is `"completed"`, `output_path` is populated with the path to the generated file on disk.

#### Download the generated document

```bash
curl -O http://localhost:8000/api/v1/doc-jobs/3f8a1b2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c/download
```

Returns the `.md` file as a download. Returns `409 Conflict` if the job is not yet complete.

#### List all doc jobs for a repository

```bash
curl "http://localhost:8000/api/v1/doc-jobs?repo_url=https://github.com/pallets/flask"
```

#### List available built-in templates

```bash
curl http://localhost:8000/api/v1/templates
```

```json
{
  "templates": [
    { "name": "hld", "document_type": "HLD", "description": "High-Level Design Document...", "version": "1.0", "section_count": 6 },
    { "name": "lld", "document_type": "LLD", "description": "Low-Level Design Document...", "version": "1.0", "section_count": 7 },
    { "name": "sop", "document_type": "SOP", "description": "Standard Operating Procedure...", "version": "1.0", "section_count": 7 },
    { "name": "diagram", "document_type": "DIAGRAM", "description": "Architecture Diagram Suite...", "version": "1.0", "section_count": 4 }
  ]
}
```

---

### Cache Management

Invalidate all cached Q&A answers for a repository (e.g. after re-ingesting a new commit):

```bash
curl -X POST http://localhost:8000/api/v1/cache/invalidate \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/pallets/flask"}'
```

---

### Endpoint Summary

#### RAG Pipeline

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/ingest` | Enqueue repo ingestion workflow (returns 202 immediately) |
| `GET` | `/api/v1/workflows` | List recent workflows (`?limit=20`) |
| `GET` | `/api/v1/workflows/{id}` | Status of a specific ingestion workflow |
| `POST` | `/api/v1/query` | RAG + CAG query |
| `POST` | `/api/v1/cache/invalidate` | Mark CAG cache entries for a repo as stale |
| `GET` | `/api/v1/health` | Liveness check — reports DB and Qdrant status |

#### Documentation Engine

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/generate-docs` | Enqueue a doc-gen job (returns 202 + job_id immediately) |
| `GET` | `/api/v1/doc-jobs` | List recent doc jobs (`?repo_url=` to filter) |
| `GET` | `/api/v1/doc-jobs/{job_id}` | Full status including `current_section` and `output_path` |
| `GET` | `/api/v1/doc-jobs/{job_id}/download` | Stream the generated `.md` file (completed jobs only) |
| `GET` | `/api/v1/templates` | List built-in templates with section counts |

---

## Built-in Document Templates

Templates live in `app/docs/templates/`. Each is a markdown file with YAML frontmatter and `## Section` blocks. Section bodies are LLM instruction text, not literal output. Two directives control generation:

- `{{context: key1, key2}}` — which topology components to inject into the section prompt
- `{{diagram: type}}` — forces a Mermaid block of that type (validated + auto-repaired)

| Template | `document_type` | Sections | Diagrams |
|---|---|---|---|
| `hld.md` | `HLD` | Executive Overview, System Architecture, Module Breakdown, Data Flow, External Integrations, Design Decisions | `flowchart TD`, `sequenceDiagram` |
| `lld.md` | `LLD` | Module Overview, Class & Interface Specs, API Contract, Data Models, Algorithm & Control Flow, Error Handling, Concurrency Model | `classDiagram`, `erDiagram`, `sequenceDiagram` |
| `sop.md` | `SOP` | Purpose & Scope, Prerequisites, Deployment Procedure, Operational Workflows, Monitoring, Troubleshooting, Maintenance | `sequenceDiagram` |
| `diagram.md` | `DIAGRAM` | System Overview, Component Interaction, Class Hierarchy, Data & State Flow | `flowchart TD`, `sequenceDiagram`, `classDiagram`, `flowchart LR` |

### Custom Template Format

```markdown
---
name: my_template
document_type: HLD
version: "1.0"
description: "My custom template"
---

## Executive Overview
{{context: repo_name, language_distribution, total_files}}
Write a 2–3 paragraph overview of the system's purpose and scale.

## Architecture
{{context: call_graph, module_structure}}
{{diagram: flowchart}}
Describe the architecture. Include a Mermaid flowchart of the component graph.
```

### Mermaid Auto-Repair

When the LLM generates a Mermaid block, the validator checks for 5 structural rules:

1. First line is a recognised diagram-type declaration (`graph`, `flowchart`, `sequenceDiagram`, etc.)
2. No invalid arrow syntax (`->`, `=>` instead of `-->`)
3. All brackets / parentheses / braces are balanced
4. No orphaned `end` keywords without a matching `subgraph`
5. No duplicate node IDs in graph/flowchart diagrams

Failed blocks are sent back to the LLM with the error description for a targeted repair pass, up to `MAX_MERMAID_REPAIR_ATTEMPTS` times. If all attempts fail, a `<!--MERMAID_VALIDATION_FAILED-->` comment is embedded alongside the block so the failure is visible but the document is still delivered.

---

## Running Tests

### Install test dependencies

```bash
pip install pytest pytest-asyncio pytest-cov httpx
```

### Run the full test suite

```bash
pytest tests/ -v
```

### Run with coverage report

```bash
pytest tests/ --cov=app --cov-report=term-missing --cov-report=html
# Open htmlcov/index.html to browse line-level coverage
```

### Run only unit tests (no external services required)

```bash
pytest tests/unit/ -v -m "not integration"
```

### Run only integration tests (requires live PostgreSQL + Qdrant)

```bash
pytest tests/integration/ -v -m integration
```

### Async test configuration

```ini
# pytest.ini
[pytest]
asyncio_mode = auto
```

### Recommended test layout

```
tests/
├── unit/
│   ├── test_chunker.py              # ChunkingEngine + AST parsers (no DB/network)
│   ├── test_normalizer.py           # QueryNormaliser canonical forms + hash stability
│   ├── test_cag.py                  # Fingerprint + query hash determinism
│   ├── test_score_fusion.py         # fuse_scores formula: weight coefficients
│   ├── test_template_manager.py     # Template parsing: frontmatter, sections, {{directives}}
│   ├── test_mermaid_validator.py    # All 5 validation rules + bracket balancing edge cases
│   └── test_context_aggregator.py   # StructuredContext serialisation by key
└── integration/
    ├── test_ingestion.py            # Full clone → chunk → embed → upsert flow
    ├── test_pipeline.py             # End-to-end RAG query against real Qdrant
    ├── test_doc_generation.py       # Full generate-docs → poll → download flow
    └── test_endpoints.py            # httpx AsyncClient against live FastAPI app
```

---

## Configuration Reference

All settings live in [`app/config.py`](app/config.py) and are loaded from the `.env` file. Every value has a typed default.

### Core Settings

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:password@localhost:5432/gitrag` | SQLAlchemy async DSN |
| `DB_POOL_SIZE` | `10` | Connection pool size |
| `DB_MAX_OVERFLOW` | `20` | Max overflow connections |
| `QDRANT_HOST` | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant HTTP port |
| `QDRANT_GRPC_PORT` | `6334` | Qdrant gRPC port (preferred) |
| `QDRANT_SHARD_NUMBER` | `4` | Collection shards |
| `QDRANT_VECTOR_SIZE` | `768` | Must match embedding model output dim |
| `ANTHROPIC_API_KEY` | _(required)_ | Anthropic API key |
| `LLM_MODEL` | `claude-sonnet-4-6` | Anthropic model ID |
| `LLM_MAX_TOKENS` | `4096` | Max tokens for RAG answers |

### Retrieval Settings

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | HuggingFace dense encoder |
| `EMBEDDING_BATCH_SIZE` | `64` | Chunks per embedding batch |
| `EMBEDDING_DEVICE` | _(auto)_ | `cuda` / `cpu` — blank for auto-detect |
| `RETRIEVAL_TOP_K` | `20` | ANN candidates before reranking |
| `RERANK_TOP_N` | `10` | Candidates after cross-encoder |
| `MAX_CONTEXT_CHUNKS` | `8` | Chunks sent to the LLM per query |
| `BM25_WEIGHT` | `0.6` | Score fusion BM25 coefficient |
| `VECTOR_WEIGHT` | `0.3` | Score fusion dense vector coefficient |
| `METADATA_BOOST_WEIGHT` | `0.1` | Score fusion metadata match coefficient |
| `CACHE_TTL_SECONDS` | `86400` | CAG cache entry lifetime (24 h) |

### Ingestion Settings

| Variable | Default | Description |
|---|---|---|
| `GITHUB_TOKEN` | _(optional)_ | PAT for private repos — injected into clone URL |
| `CLONE_TIMEOUT_SECONDS` | `300` | Per-clone subprocess timeout |
| `MAX_CLONE_RETRIES` | `3` | Exponential-backoff clone retry limit |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy model for query normalisation |

### Documentation Generation Settings

| Variable | Default | Description |
|---|---|---|
| `DOCS_OUTPUT_DIR` | `generated_docs` | Directory where generated `.md` files are written |
| `TEMPLATES_DIR` | `app/docs/templates` | Path to the built-in template files |
| `MAX_MERMAID_REPAIR_ATTEMPTS` | `3` | LLM auto-repair retries per invalid Mermaid block |
| `DOC_SECTION_MAX_TOKENS` | `2048` | Max LLM tokens allocated per document section |
