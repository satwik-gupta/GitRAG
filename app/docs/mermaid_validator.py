"""
app/docs/mermaid_validator.py
──────────────────────────────
Mermaid.js Syntax Validator.

Extracts every ```mermaid ... ``` fenced block from a markdown document,
runs a multi-rule structural check against each block, and triggers an
automatic LLM repair loop for any block that fails validation.

Validation rules (in order)
────────────────────────────
  1. Non-empty body after stripping whitespace.
  2. First non-empty line is a recognised Mermaid diagram-type declaration.
  3. No ambiguous single-dash arrows (`->`, `=>`) in graph/flowchart diagrams.
  4. All bracket / parenthesis / brace pairs are balanced.
  5. No bare `end` keywords that would close an unopened subgraph.
  6. No duplicate node IDs that differ only in bracket style (can cause
     silent rendering failures in some Mermaid versions).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from google import genai

from app.config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(
    r"```mermaid\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

_VALID_DIAGRAM_TYPES = {
    "graph", "flowchart", "sequencediagram", "classdiagram",
    "statediagram", "statediagram-v2", "erdiagram", "gantt",
    "pie", "gitgraph", "mindmap", "timeline", "quadrantchart",
    "xychart-beta", "block-beta", "architecture-beta", "requirementdiagram",
    "c4context", "c4container", "c4component", "c4dynamic",
}

# Arrows that are valid in Mermaid flowchart / graph
_VALID_ARROW_RE = re.compile(
    r"--?>|---|-\.->|\.->|==?>|<-->|o--o|x--x|--x|--o|<\.\.>|<==>"
)

# Suspicious single-dash arrows (invalid in graph/flowchart)
_INVALID_ARROW_RE = re.compile(r"(?<![=-])->(?!>)|(?<![=-])=>(?!>)")

# Node ID extractor for duplicate detection
_NODE_ID_RE = re.compile(r"^\s{0,8}([A-Za-z_]\w*)\s*[\[\(\{<]", re.MULTILINE)


# ── Validation result ─────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    diagram_type: Optional[str] = None

    def error_summary(self) -> str:
        return "\n".join(f"  • {e}" for e in self.errors)


# ── Per-block validation ──────────────────────────────────────────────────


def validate_mermaid_block(block: str) -> ValidationResult:
    """
    Validate a single Mermaid diagram body (text INSIDE the fences, no backticks).
    """
    errors: list[str] = []

    stripped = block.strip()
    if not stripped:
        return ValidationResult(valid=False, errors=["Diagram body is empty."])

    lines = stripped.splitlines()
    first_line = lines[0].strip().lower()

    # ── Rule 1: recognised diagram type ───────────────────────────────────
    diagram_type: Optional[str] = None
    for dtype in _VALID_DIAGRAM_TYPES:
        if first_line.startswith(dtype):
            diagram_type = dtype
            break

    if diagram_type is None:
        errors.append(
            f"First line {lines[0]!r} is not a recognised Mermaid diagram type. "
            f"Expected one of: graph, flowchart, sequenceDiagram, classDiagram, etc."
        )

    is_graph_type = diagram_type in ("graph", "flowchart", None)

    # ── Rule 2: invalid arrows in graph/flowchart ──────────────────────────
    if is_graph_type:
        for i, line in enumerate(lines[1:], 2):
            if _INVALID_ARROW_RE.search(line):
                errors.append(
                    f"Line {i}: invalid arrow syntax detected ({line.strip()!r}). "
                    f"Use --> for arrows, not -> or =>."
                )

    # ── Rule 3: balanced brackets ─────────────────────────────────────────
    stack: list[tuple[str, int]] = []
    pair_map = {")": "(", "]": "[", "}": "{"}
    open_chars = set("([{")
    close_chars = set(")]}")
    in_string = False
    string_char = ""

    for line_no, line in enumerate(lines, 1):
        # Skip comment lines
        if line.strip().startswith("%%"):
            continue
        for ch in line:
            if in_string:
                if ch == string_char:
                    in_string = False
            elif ch in ('"', "'"):
                in_string = True
                string_char = ch
            elif ch in open_chars:
                stack.append((ch, line_no))
            elif ch in close_chars:
                expected = pair_map[ch]
                if stack and stack[-1][0] == expected:
                    stack.pop()
                else:
                    errors.append(
                        f"Line {line_no}: unmatched closing '{ch}'. "
                        f"Check bracket nesting."
                    )

    for ch, line_no in stack:
        errors.append(f"Line {line_no}: unclosed '{ch}' — bracket not closed.")

    # ── Rule 4: bare `end` without matching `subgraph` ───────────────────
    subgraph_depth = 0
    for i, line in enumerate(lines, 1):
        ls = line.strip().lower()
        if ls.startswith("subgraph"):
            subgraph_depth += 1
        elif ls == "end":
            if subgraph_depth > 0:
                subgraph_depth -= 1
            else:
                errors.append(
                    f"Line {i}: bare 'end' without a matching 'subgraph'. "
                    f"Remove the orphaned 'end' or wrap content in 'subgraph'."
                )

    # ── Rule 5: duplicate node IDs ────────────────────────────────────────
    if is_graph_type:
        seen_ids: set[str] = set()
        for match in _NODE_ID_RE.finditer(stripped):
            nid = match.group(1)
            if nid.lower() in ("end", "style", "class", "click", "linkstyle"):
                continue
            if nid in seen_ids:
                errors.append(
                    f"Duplicate node ID '{nid}' detected. "
                    f"Each node must have a unique ID."
                )
            seen_ids.add(nid)

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        diagram_type=diagram_type,
    )


# ── Document-level processing ─────────────────────────────────────────────


def extract_mermaid_blocks(markdown: str) -> list[tuple[int, str]]:
    """
    Extract all mermaid fenced blocks from *markdown*.

    Returns list of (start_pos, block_content) where block_content is
    the text between the fences (no backtick lines).
    """
    return [(m.start(), m.group(1)) for m in _FENCE_RE.finditer(markdown)]


# ── LLM repair ────────────────────────────────────────────────────────────


_REPAIR_SYSTEM = (
    "You are a Mermaid.js syntax expert. Your sole task is to fix broken Mermaid "
    "diagram syntax. Return ONLY the corrected diagram content inside a single "
    "```mermaid ... ``` fenced block. Do not add any explanation, prose, or "
    "additional content outside the fences."
)


async def _repair_block(broken_block: str, errors: list[str]) -> Optional[str]:
    """
    Ask the LLM to fix *broken_block*.  Returns the repaired block body
    (without fences) or None if the LLM call fails.
    """
    error_text = "\n".join(f"- {e}" for e in errors)
    prompt = (
        f"The following Mermaid diagram is invalid:\n\n"
        f"```mermaid\n{broken_block}\n```\n\n"
        f"Validation errors:\n{error_text}\n\n"
        f"Fix the Mermaid syntax and return the corrected diagram."
    )

    try:
        async with genai.Client(api_key=settings.gemini_api_key).aio as client:
            gen_config = genai.types.GenerateContentConfig(
                max_output_tokens=1024,
                system_instruction=_REPAIR_SYSTEM,
            )
            response = await client.models.generate_content(
                model=settings.llm_model,
                contents=prompt,
                config=gen_config,
            )
            text = response.text
        # Extract the block from the LLM's response
        m = _FENCE_RE.search(text)
        if m:
            return m.group(1).strip()
        # Fallback: return the whole response trimmed of fences
        return text.strip().lstrip("```mermaid").rstrip("```").strip()
    except Exception as exc:
        logger.warning("LLM repair call failed: %s", exc)
        return None


# ── MermaidValidator ──────────────────────────────────────────────────────


class MermaidValidator:
    """
    Async Mermaid validation + auto-repair service.

    Validates every mermaid block in a markdown document.
    Invalid blocks are sent to the LLM for repair (up to
    ``settings.max_mermaid_repair_attempts`` retries per block).
    """

    async def validate_and_repair(self, markdown: str) -> tuple[str, list[str]]:
        """
        Validate all mermaid blocks in *markdown*.

        Returns
        -------
        (repaired_markdown, repair_log)
            *repaired_markdown* is the document with all fixable blocks repaired.
            *repair_log* is a list of human-readable repair event messages.
        """
        blocks = extract_mermaid_blocks(markdown)
        if not blocks:
            return markdown, []

        repair_log: list[str] = []
        result_markdown = markdown

        # Process blocks in reverse order so string offsets stay valid
        for _start, block_body in reversed(blocks):
            result_markdown, log = await self._process_block(
                result_markdown, block_body
            )
            repair_log.extend(log)

        return result_markdown, repair_log

    async def _process_block(
        self,
        markdown: str,
        block_body: str,
    ) -> tuple[str, list[str]]:
        log: list[str] = []
        current_body = block_body
        max_attempts = settings.max_mermaid_repair_attempts

        for attempt in range(1, max_attempts + 1):
            result = validate_mermaid_block(current_body)
            if result.valid:
                if attempt > 1:
                    log.append(
                        f"Mermaid block repaired in {attempt - 1} attempt(s)."
                    )
                # Replace original block with (possibly repaired) version
                markdown = markdown.replace(
                    f"```mermaid\n{block_body}\n```",
                    f"```mermaid\n{current_body}\n```",
                    1,
                )
                return markdown, log

            log.append(
                f"Attempt {attempt}/{max_attempts}: Mermaid block invalid "
                f"({len(result.errors)} error(s)): {result.errors[0]}"
            )

            if attempt < max_attempts:
                repaired = await _repair_block(current_body, result.errors)
                if repaired:
                    current_body = repaired
                else:
                    log.append("LLM repair returned no content — keeping original.")
                    break

        # All attempts exhausted: embed validation errors as a comment
        error_comment = (
            "<!--\n"
            "MERMAID_VALIDATION_FAILED:\n"
            + "\n".join(f"  {e}" for e in validate_mermaid_block(current_body).errors)
            + "\n-->"
        )
        markdown = markdown.replace(
            f"```mermaid\n{block_body}\n```",
            f"{error_comment}\n```mermaid\n{current_body}\n```",
            1,
        )
        return markdown, log
