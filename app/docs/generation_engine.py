"""
app/docs/generation_engine.py
──────────────────────────────
Iterative Document Constructor.

Generates large technical documents section-by-section to preserve
context density without hitting LLM context-window limits.

Algorithm
─────────
  For each section in the ParsedTemplate:
    1. Select the relevant context keys from StructuredContext.
    2. Summarise (truncate) all previously generated sections to preserve
       continuity without exploding the prompt.
    3. Build a focused prompt: GLOBAL TOPOLOGY + PREV SECTIONS + SECTION INSTRUCTIONS.
    4. Call the LLM (Google Gemini 2.5 Flash-Lite).
    5. Validate and auto-repair any Mermaid blocks in the response.
    6. Accumulate the section output.
  Assemble the full document and write to disk asynchronously.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
from google import genai

from app.config import settings
from app.docs.context_aggregator import StructuredContext
from app.docs.mermaid_validator import MermaidValidator
from app.docs.template_manager import ParsedTemplate, TemplateSection

logger = logging.getLogger(__name__)

# ── Prompt constants ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a Principal Software Architect generating professional-grade technical \
documentation for a software repository.

Guidelines:
- Be technically precise. Reference actual file paths, class names, and function \
  names from the provided context.
- Every claim must be grounded in the repository context supplied. Do not hallucinate \
  components or relationships not evidenced in the context.
- When asked to produce a Mermaid diagram, output valid Mermaid.js syntax inside \
  a ```mermaid ... ``` fenced block. Use the exact diagram type specified.
- Do NOT include the section title heading — it is added automatically.
- Write in a professional technical style appropriate for an engineering audience.
- Output ONLY the section content. No preamble, no meta-commentary.
"""

_MAX_PREV_SECTION_CHARS = 1200   # characters from each previous section kept in context
_MAX_TOPOLOGY_CHARS = 6000       # characters of global topology passed per section


# ── Prompt builder ────────────────────────────────────────────────────────


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n… [truncated]"


def _build_section_prompt(
    section: TemplateSection,
    section_context: str,
    prev_sections: list[tuple[str, str]],  # [(title, content), ...]
    requires_diagram: bool,
    diagram_type: Optional[str],
) -> str:
    parts: list[str] = []

    # ── Global topology (section-specific keys) ────────────────────────────
    parts.append("=== REPOSITORY CONTEXT ===")
    parts.append(_truncate(section_context, _MAX_TOPOLOGY_CHARS))

    # ── Previously generated sections (abbreviated) ────────────────────────
    if prev_sections:
        parts.append("\n=== PREVIOUSLY GENERATED SECTIONS (for continuity) ===")
        for title, content in prev_sections[-3:]:  # keep last 3 only
            parts.append(f"--- {title} ---")
            parts.append(_truncate(content, _MAX_PREV_SECTION_CHARS))

    # ── Section instructions ───────────────────────────────────────────────
    parts.append(f"\n=== GENERATE SECTION: {section.title.upper()} ===")
    parts.append(section.instruction.strip())

    # ── Diagram enforcement ────────────────────────────────────────────────
    if requires_diagram and diagram_type:
        parts.append(
            f"\nREQUIREMENT: This section MUST include a Mermaid diagram of type "
            f"'{diagram_type}'. Produce it inside a ```mermaid ... ``` fenced block. "
            f"Ensure the first line of the block is a valid '{diagram_type}' declaration."
        )

    return "\n\n".join(parts)


# ── Output writer ─────────────────────────────────────────────────────────


async def _write_document(content: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(output_path, "w", encoding="utf-8") as fh:
        await fh.write(content)
    logger.info("Document written to %s (%d chars)", output_path, len(content))


def _make_output_path(repo_name: str, document_type: str, job_id: uuid.UUID) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{repo_name}_{document_type}_{timestamp}_{str(job_id)[:8]}.md"
    return Path(settings.docs_output_dir) / filename


# ── IterativeDocumentConstructor ──────────────────────────────────────────


class IterativeDocumentConstructor:
    """
    Generates a full documentation artifact section by section.

    Parameters
    ----------
    template:   Parsed template driving the document structure.
    context:    Aggregated repository context.
    job_id:     UUID of the DocGenerationJob (used for output filename).
    on_section: Optional async callback invoked after each section is generated
                with (section_id, section_title) — used to update job state.
    """

    def __init__(
        self,
        template: ParsedTemplate,
        context: StructuredContext,
        job_id: uuid.UUID,
        on_section: Optional[any] = None,
    ) -> None:
        self._template = template
        self._context = context
        self._job_id = job_id
        self._on_section = on_section
        self._validator = MermaidValidator()
        self._client = genai.Client(api_key=settings.gemini_api_key).aio

    # ── Public ─────────────────────────────────────────────────────────────

    async def construct(self) -> tuple[str, Path]:
        """
        Run the full iterative generation loop.

        Returns
        -------
        (markdown_content, output_path)
        """
        repo_name = self._context.repo_name
        logger.info(
            "Starting iterative generation for %s (%s, %d sections)",
            repo_name,
            self._template.document_type,
            len(self._template.sections),
        )

        document_header = (
            f"# {repo_name} — "
            f"{self._template.document_type} Documentation\n\n"
            f"*Generated by GitRAG · Commit: {self._context.commit_sha[:12]}*\n\n"
            f"---\n\n"
        )

        generated_sections: list[tuple[str, str]] = []
        section_parts: list[str] = [document_header]

        for section in self._template.sections:
            logger.info("Generating section: %s", section.title)

            if self._on_section:
                await self._on_section(section.section_id, section.title)

            content = await self._generate_section(section, generated_sections)
            generated_sections.append((section.title, content))

            # Validate and repair Mermaid blocks in this section
            content, repair_log = await self._validator.validate_and_repair(content)
            if repair_log:
                for entry in repair_log:
                    logger.info("Mermaid repair: %s", entry)

            section_parts.append(f"## {section.title}\n\n{content.strip()}")

        full_document = "\n\n---\n\n".join(section_parts)

        # Final document-level Mermaid validation pass
        full_document, final_log = await self._validator.validate_and_repair(full_document)
        if final_log:
            logger.info("Final Mermaid pass: %s", "; ".join(final_log))

        output_path = _make_output_path(
            repo_name, self._template.document_type, self._job_id
        )
        await _write_document(full_document, output_path)

        return full_document, output_path

    # ── Private ────────────────────────────────────────────────────────────

    async def _generate_section(
        self,
        section: TemplateSection,
        prev_sections: list[tuple[str, str]],
    ) -> str:
        """Call the LLM to generate one section of the document."""
        section_ctx = self._context.to_context_string(section.context_keys)
        prompt = _build_section_prompt(
            section=section,
            section_context=section_ctx,
            prev_sections=prev_sections,
            requires_diagram=section.requires_diagram,
            diagram_type=section.diagram_type,
        )

        try:
            gen_config = genai.types.GenerateContentConfig(
                max_output_tokens=settings.doc_section_max_tokens,
                system_instruction=_SYSTEM_PROMPT,
            )
            response = await self._client.models.generate_content(
                model=settings.llm_model,
                contents=prompt,
                config=gen_config,
            )
            return response.text.strip()
        except Exception as exc:
            logger.error("LLM call failed for section %r: %s", section.title, exc)
            return (
                f"*Generation failed for this section: {exc}*\n\n"
                f"Section instructions were:\n\n{section.instruction}"
            )
