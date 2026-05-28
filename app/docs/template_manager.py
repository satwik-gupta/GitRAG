"""
app/docs/template_manager.py
──────────────────────────────
Template Registry & Engine.

Template format
───────────────
Templates are plain markdown files with a YAML frontmatter block:

    ---
    name: hld
    document_type: HLD
    version: "1.0"
    description: "High-Level Design..."
    ---

    ## Section Title
    {{context: key1, key2, key3}}
    {{diagram: flowchart}}
    Instruction text for the LLM — what to write in this section.

Parsing rules
─────────────
  - Everything before the first `## ` header is the preamble (ignored in generation).
  - Each `## Header` starts a new section.
  - `{{context: ...}}` on its own line declares which StructuredContext
    keys should be injected for this section.  Stripped from LLM prompt.
  - `{{diagram: type}}` signals the generation engine to require a Mermaid
    block of that type.  Stripped from LLM prompt.
  - The remaining text becomes the `instruction` field shown to the LLM.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiofiles
import yaml

from app.config import settings

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_CONTEXT_RE = re.compile(r"\{\{context:\s*([^}]+)\}\}")
_DIAGRAM_RE = re.compile(r"\{\{diagram:\s*([^}]+)\}\}")
_SECTION_SPLIT_RE = re.compile(r"^(##\s+.+)$", re.MULTILINE)


# ── Data structures ───────────────────────────────────────────────────────


@dataclass
class TemplateSection:
    """A single `## Header` block parsed from a template."""

    section_id: str            # slug derived from title
    title: str                 # raw header text (without ##)
    context_keys: list[str]    # keys from {{context: ...}}
    requires_diagram: bool
    diagram_type: Optional[str]
    instruction: str           # cleaned LLM-facing instruction text


@dataclass
class ParsedTemplate:
    """The fully parsed in-memory representation of a template file."""

    name: str
    document_type: str
    version: str
    description: str
    sections: list[TemplateSection] = field(default_factory=list)
    raw_content: str = ""

    @property
    def section_ids(self) -> list[str]:
        return [s.section_id for s in self.sections]


# ── Parser helpers ────────────────────────────────────────────────────────


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Strip YAML frontmatter and return (meta_dict, body)."""
    m = _FRONTMATTER_RE.match(raw)
    if m:
        try:
            meta = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        body = raw[m.end():]
    else:
        meta = {}
        body = raw
    return meta, body


def _parse_section_block(header_text: str, body_text: str) -> TemplateSection:
    """Parse a single section block into a TemplateSection."""
    title = header_text.lstrip("#").strip()
    section_id = _slugify(title)

    # Extract context keys
    context_match = _CONTEXT_RE.search(body_text)
    context_keys = (
        [k.strip() for k in context_match.group(1).split(",") if k.strip()]
        if context_match
        else []
    )

    # Extract diagram requirement
    diagram_match = _DIAGRAM_RE.search(body_text)
    requires_diagram = bool(diagram_match)
    diagram_type = diagram_match.group(1).strip() if diagram_match else None

    # Clean instruction: strip the marker lines
    instruction = _CONTEXT_RE.sub("", body_text)
    instruction = _DIAGRAM_RE.sub("", instruction).strip()

    return TemplateSection(
        section_id=section_id,
        title=title,
        context_keys=context_keys,
        requires_diagram=requires_diagram,
        diagram_type=diagram_type,
        instruction=instruction,
    )


def parse_template(raw_content: str, name: str = "custom") -> ParsedTemplate:
    """
    Parse raw template markdown into a :class:`ParsedTemplate`.

    Parameters
    ----------
    raw_content: Full template text (frontmatter + body).
    name:        Fallback name if frontmatter has no `name` key.
    """
    meta, body = _parse_frontmatter(raw_content)

    # Split body into sections by ## headers
    parts = _SECTION_SPLIT_RE.split(body)
    # parts = [preamble, "## Header1", content1, "## Header2", content2, ...]

    sections: list[TemplateSection] = []
    i = 1  # skip preamble (index 0)
    while i < len(parts) - 1:
        header = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append(_parse_section_block(header, content))
        i += 2

    return ParsedTemplate(
        name=meta.get("name", name),
        document_type=meta.get("document_type", "HLD"),
        version=str(meta.get("version", "1.0")),
        description=meta.get("description", ""),
        sections=sections,
        raw_content=raw_content,
    )


# ── TemplateManager ────────────────────────────────────────────────────────


class TemplateManager:
    """
    Manages loading, parsing, and validating documentation templates.

    LOCAL templates are read from *templates_dir* (configured in settings).
    CUSTOM templates are parsed directly from user-supplied strings.
    """

    def __init__(self, templates_dir: Optional[str] = None) -> None:
        self._dir = Path(templates_dir or settings.templates_dir)

    # ── Local template operations ──────────────────────────────────────────

    async def list_local(self) -> list[ParsedTemplate]:
        """Return all local templates found in the templates directory."""
        templates: list[ParsedTemplate] = []
        if not self._dir.exists():
            logger.warning("Templates directory not found: %s", self._dir)
            return templates

        for fp in sorted(self._dir.glob("*.md")):
            try:
                tmpl = await self.load_local(fp.stem)
                templates.append(tmpl)
            except Exception as exc:
                logger.warning("Could not load template %s: %s", fp.name, exc)
        return templates

    async def load_local(self, name: str) -> ParsedTemplate:
        """
        Load and parse a built-in template by *name* (without .md extension).

        Raises
        ------
        FileNotFoundError if the template file does not exist.
        ValueError if the template cannot be parsed.
        """
        fp = self._dir / f"{name}.md"
        if not fp.exists():
            raise FileNotFoundError(
                f"Template '{name}' not found in {self._dir}. "
                f"Available: {[p.stem for p in self._dir.glob('*.md')]}"
            )
        async with aiofiles.open(fp, "r", encoding="utf-8") as fh:
            raw = await fh.read()
        tmpl = parse_template(raw, name=name)
        self._validate_parsed(tmpl)
        return tmpl

    # ── Custom template operations ─────────────────────────────────────────

    def load_custom(self, content: str) -> tuple[ParsedTemplate, str]:
        """
        Parse a user-supplied template string.

        Returns
        -------
        (parsed_template, content_hash)
            The hash can be used as `template_path_or_hash` in the DB.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:32]
        tmpl = parse_template(content, name=f"custom_{content_hash[:8]}")
        self._validate_parsed(tmpl)
        return tmpl, content_hash

    # ── Resolution: pick local by document_type if name not specified ──────

    async def resolve(
        self,
        template_source: str,
        document_type: str,
        template_name: Optional[str] = None,
        template_content: Optional[str] = None,
    ) -> tuple[ParsedTemplate, str]:
        """
        Resolve to a ParsedTemplate from whatever the caller provided.

        Returns (parsed_template, path_or_hash).
        """
        if template_source == "CUSTOM":
            if not template_content:
                raise ValueError("template_content required for CUSTOM source.")
            return self.load_custom(template_content)

        # LOCAL
        name = template_name or document_type.lower()
        tmpl = await self.load_local(name)
        return tmpl, name

    # ── Validation ─────────────────────────────────────────────────────────

    @staticmethod
    def _validate_parsed(tmpl: ParsedTemplate) -> None:
        if not tmpl.sections:
            raise ValueError(
                f"Template '{tmpl.name}' has no sections. "
                "Add ## headers to define sections."
            )
        if not tmpl.document_type:
            raise ValueError(
                f"Template '{tmpl.name}' frontmatter is missing 'document_type'."
            )
