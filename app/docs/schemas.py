"""
app/docs/schemas.py
────────────────────
Pydantic v2 request / response models for the documentation-generation API.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── Request ───────────────────────────────────────────────────────────────


class GenerateDocsRequest(BaseModel):
    repo_url: str = Field(
        ...,
        description="GitHub repository URL that has already been ingested.",
    )
    document_type: Literal["HLD", "LLD", "SOP", "DIAGRAM"] = Field(
        default="HLD",
        description="Type of document to generate.",
    )
    template_source: Literal["LOCAL", "CUSTOM"] = Field(
        default="LOCAL",
        description="'LOCAL' uses a built-in template; 'CUSTOM' uses template_content.",
    )
    template_name: Optional[str] = Field(
        default=None,
        description="Name of the built-in template (e.g. 'hld', 'lld'). "
                    "Required when template_source='LOCAL'. Defaults to document_type lower-cased.",
    )
    template_content: Optional[str] = Field(
        default=None,
        description="Raw markdown template string. Required when template_source='CUSTOM'.",
        max_length=32_000,
    )
    workflow_id: Optional[int] = Field(
        default=None,
        description="Optional ID of the ingestion workflow that produced the index.",
    )

    @field_validator("repo_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not v.startswith("https://"):
            raise ValueError("repo_url must be an HTTPS URL.")
        return v

    @field_validator("template_content")
    @classmethod
    def validate_custom_template(cls, v: Optional[str], info) -> Optional[str]:
        if info.data.get("template_source") == "CUSTOM" and not v:
            raise ValueError(
                "template_content is required when template_source='CUSTOM'."
            )
        return v


# ── Response ──────────────────────────────────────────────────────────────


class GenerateDocsResponse(BaseModel):
    job_id: uuid.UUID
    repo_url: str
    document_type: str
    status: str
    message: str


class DocJobStatusResponse(BaseModel):
    job_id: uuid.UUID
    repo_url: str
    document_type: str
    template_source: str
    template_path_or_hash: Optional[str]
    current_section: Optional[str]
    output_path: Optional[str]
    status: str
    error_log: Optional[str]
    created_at: datetime
    updated_at: datetime


class DocJobListResponse(BaseModel):
    jobs: list[DocJobStatusResponse]
    total: int


# ── Template listing ──────────────────────────────────────────────────────


class TemplateInfo(BaseModel):
    name: str
    document_type: str
    description: str
    version: str
    section_count: int


class TemplateListResponse(BaseModel):
    templates: list[TemplateInfo]
