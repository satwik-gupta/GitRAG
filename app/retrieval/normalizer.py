"""
app/retrieval/normalizer.py
────────────────────────────
Query normaliser: collapses query variations into a canonical form and
extracts structured constraints (language, file path) for metadata pre-filtering.

Pipeline
────────
  raw query
    → lowercase
    → lemmatise tokens (spaCy)
    → remove stop-words
    → sort tokens for canonical stability
    → join → canonical string
    → SHA-256 → query hash

Extracted filters
─────────────────
  language:       detected from explicit mentions ("in Python", "golang function")
  file_path:      path-like tokens  ("src/", ".java", "controllers/")
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── spaCy lazy-loader ─────────────────────────────────────────────────────

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy  # noqa: PLC0415

        from app.config import settings  # noqa: PLC0415

        try:
            _nlp = spacy.load(settings.spacy_model, disable=["ner", "parser"])
        except OSError:
            raise RuntimeError(
                f"spaCy model '{settings.spacy_model}' not found. "
                f"Run: python -m spacy download {settings.spacy_model}"
            )
    return _nlp


# ── Language keyword mapping ──────────────────────────────────────────────

_LANGUAGE_KEYWORDS: dict[str, str] = {
    "python": "python",
    "py": "python",
    "java": "java",
    "go": "golang",
    "golang": "golang",
    "c++": "cpp",
    "cpp": "cpp",
    "cxx": "cpp",
}

_LANG_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _LANGUAGE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# ── Path-like token pattern ────────────────────────────────────────────────

_PATH_PATTERN = re.compile(
    r"""
    (?:
        [\w\-]+/[\w\-./]+   # token/path.ext
        |[\w\-]+\.\w{1,6}   # file.ext  (e.g. main.py)
        |src/|lib/|pkg/|cmd/ # common prefix dirs
    )
    """,
    re.VERBOSE,
)


# ── Result container ──────────────────────────────────────────────────────


@dataclass
class NormalisedQuery:
    canonical: str
    query_hash: str
    language_filter: Optional[str] = None
    file_path_filter: Optional[str] = None
    raw: str = ""
    tokens: list[str] = field(default_factory=list)


# ── QueryNormaliser ───────────────────────────────────────────────────────


class QueryNormaliser:
    """
    Stateless query normaliser.  spaCy model is loaded lazily on first use
    and cached in the module-level *_nlp* variable.
    """

    def normalise(self, raw_query: str) -> NormalisedQuery:
        """
        Normalise *raw_query* and return a :class:`NormalisedQuery`.

        This method is synchronous and CPU-bound; callers should dispatch it
        via ``run_in_executor`` if called from async context at high load.
        """
        raw = raw_query.strip()
        lower = raw.lower()

        # ── Extract structured filters before lemmatisation ────────────────
        language_filter = self._extract_language(lower)
        file_path_filter = self._extract_file_path(raw)

        # ── Lemmatise ─────────────────────────────────────────────────────
        nlp = _get_nlp()
        doc = nlp(lower)
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and token.lemma_.strip()
        ]

        # ── Canonical: sort tokens for stability across phrasing variants ──
        canonical_tokens = sorted(set(tokens))
        canonical = " ".join(canonical_tokens)

        query_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        return NormalisedQuery(
            canonical=canonical,
            query_hash=query_hash,
            language_filter=language_filter,
            file_path_filter=file_path_filter,
            raw=raw,
            tokens=tokens,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_language(lower: str) -> Optional[str]:
        m = _LANG_PATTERN.search(lower)
        if m:
            return _LANGUAGE_KEYWORDS.get(m.group(1).lower())
        return None

    @staticmethod
    def _extract_file_path(text: str) -> Optional[str]:
        m = _PATH_PATTERN.search(text)
        return m.group(0) if m else None
