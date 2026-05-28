"""
app/api/schemas.py
───────────────────
Pydantic v2 request / response models for all API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator


# ── Ingestion ─────────────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub HTTPS repository URL")
    branch: str = Field(default="HEAD", description="Branch or tag to clone")

    @field_validator("repo_url")
    @classmethod
    def validate_github_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not (v.startswith("https://github.com/") or v.startswith("http://github.com/")):
            raise ValueError("repo_url must be a github.com HTTPS URL")
        return v


class IngestResponse(BaseModel):
    workflow_id: int
    repo_url: str
    branch: str
    status: str
    message: str


# ── Workflow status ───────────────────────────────────────────────────────


class WorkflowStatusResponse(BaseModel):
    workflow_id: int
    repo_url: str
    branch: str
    commit_sha: Optional[str]
    status: str
    total_files: int
    processed_files: int
    total_chunks: int
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime


class WorkflowListResponse(BaseModel):
    workflows: list[WorkflowStatusResponse]
    total: int


# ── Query ─────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    repo_url: Optional[str] = Field(
        default=None,
        description="Restrict search to a specific repository URL",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="ANN candidate count override",
    )


class CitationSchema(BaseModel):
    file_path: str
    section_name: str
    repo_url: str
    start_line: int
    end_line: int
    commit_sha: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationSchema]
    canonical_query: str
    query_hash: str
    cache_hit: bool


# ── Cache ─────────────────────────────────────────────────────────────────


class CacheInvalidateRequest(BaseModel):
    repo_url: str = Field(..., description="Invalidate all cache entries for this repo URL")


class CacheInvalidateResponse(BaseModel):
    invalidated_count: int
    repo_url: str


# ── Health ────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    db: str
    qdrant: str
    version: str = "1.0.0"
