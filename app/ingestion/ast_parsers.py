"""
app/ingestion/ast_parsers.py
─────────────────────────────
Language-specific AST parsers built on tree-sitter (>=0.22).

Each parser returns a list of RawEntity dicts describing extractable
code units (functions, classes, methods) from a single source file.
Missing language grammars degrade gracefully: the parser returns an
empty list and logs a warning.

Supported languages: python, java, golang, cpp
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Tree-sitter grammar availability flags ─────────────────────────────────

_TS_PYTHON = _TS_JAVA = _TS_GO = _TS_CPP = False

try:
    import tree_sitter_python as _tspython  # type: ignore[import]
    from tree_sitter import Language as _Language, Parser as _Parser

    _PY_LANGUAGE = _Language(_tspython.language())
    _TS_PYTHON = True
except Exception:
    logger.warning("tree-sitter-python not available; Python AST parsing disabled.")

try:
    import tree_sitter_java as _tsjava  # type: ignore[import]
    from tree_sitter import Language as _Language, Parser as _Parser

    _JAVA_LANGUAGE = _Language(_tsjava.language())
    _TS_JAVA = True
except Exception:
    logger.warning("tree-sitter-java not available; Java AST parsing disabled.")

try:
    import tree_sitter_go as _tsgo  # type: ignore[import]
    from tree_sitter import Language as _Language, Parser as _Parser

    _GO_LANGUAGE = _Language(_tsgo.language())
    _TS_GO = True
except Exception:
    logger.warning("tree-sitter-go not available; Go AST parsing disabled.")

try:
    import tree_sitter_cpp as _tscpp  # type: ignore[import]
    from tree_sitter import Language as _Language, Parser as _Parser

    _CPP_LANGUAGE = _Language(_tscpp.language())
    _TS_CPP = True
except Exception:
    logger.warning("tree-sitter-cpp not available; C++ AST parsing disabled.")


# ── Data structure ─────────────────────────────────────────────────────────


@dataclass
class RawEntity:
    """A single extractable code entity from a source file."""

    entity_type: str            # "function" | "class" | "method"
    name: str
    start_line: int
    end_line: int
    content: str                # full text of this entity
    docstring: Optional[str] = None
    parent_name: Optional[str] = None   # enclosing class name (for methods)
    annotations: list[str] = field(default_factory=list)  # decorators / annotations


# ── Generic tree walker ────────────────────────────────────────────────────


def _node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _node_lines(node) -> tuple[int, int]:
    """Return (start_line, end_line) as 1-based integers."""
    return node.start_point[0] + 1, node.end_point[0] + 1


# ── Python parser ──────────────────────────────────────────────────────────


def _extract_python_docstring(node, source_bytes: bytes) -> Optional[str]:
    """Return the first string-literal in a function/class body, if any."""
    body = None
    for child in node.children:
        if child.type == "block":
            body = child
            break
    if body is None:
        return None
    for stmt in body.children:
        if stmt.type == "expression_statement":
            for child in stmt.children:
                if child.type in ("string", "concatenated_string"):
                    raw = _node_text(child, source_bytes)
                    return raw.strip("\"'").strip('"""').strip("'''").strip()
    return None


def _collect_decorators(node, source_bytes: bytes) -> list[str]:
    decorators = []
    for child in node.children:
        if child.type == "decorator":
            decorators.append(_node_text(child, source_bytes).strip())
    return decorators


def parse_python(source: str) -> list[RawEntity]:
    if not _TS_PYTHON:
        return []

    source_bytes = source.encode("utf-8")
    parser = _Parser(_PY_LANGUAGE)
    tree = parser.parse(source_bytes)

    entities: list[RawEntity] = []

    def walk(node, parent_class: Optional[str] = None) -> None:
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node, source_bytes) if name_node else "<anon>"
            start, end = _node_lines(node)
            doc = _extract_python_docstring(node, source_bytes)
            entities.append(
                RawEntity(
                    entity_type="class",
                    name=name,
                    start_line=start,
                    end_line=end,
                    content=_node_text(node, source_bytes),
                    docstring=doc,
                    annotations=_collect_decorators(node, source_bytes),
                )
            )
            # recurse into class body for methods
            for child in node.children:
                walk(child, parent_class=name)

        elif node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node, source_bytes) if name_node else "<anon>"
            start, end = _node_lines(node)
            doc = _extract_python_docstring(node, source_bytes)
            etype = "method" if parent_class else "function"
            entities.append(
                RawEntity(
                    entity_type=etype,
                    name=name,
                    start_line=start,
                    end_line=end,
                    content=_node_text(node, source_bytes),
                    docstring=doc,
                    parent_name=parent_class,
                    annotations=_collect_decorators(node, source_bytes),
                )
            )
        else:
            for child in node.children:
                walk(child, parent_class=parent_class)

    walk(tree.root_node)
    return entities


# ── Java parser ────────────────────────────────────────────────────────────


def _extract_javadoc(node, source_bytes: bytes) -> Optional[str]:
    """Return a /** ... */ comment immediately preceding `node`."""
    # tree-sitter places block_comment as a sibling before the declaration
    prev = node.prev_sibling
    if prev and prev.type == "block_comment":
        text = _node_text(prev, source_bytes).strip()
        if text.startswith("/**"):
            return re.sub(r"\s*\*\s?", " ", text.strip("/**").rstrip("*/")).strip()
    return None


def _java_modifiers(node, source_bytes: bytes) -> list[str]:
    mods = []
    for child in node.children:
        if child.type == "modifiers":
            mods = [_node_text(c, source_bytes) for c in child.children]
    return mods


def parse_java(source: str) -> list[RawEntity]:
    if not _TS_JAVA:
        return []

    source_bytes = source.encode("utf-8")
    parser = _Parser(_JAVA_LANGUAGE)
    tree = parser.parse(source_bytes)

    entities: list[RawEntity] = []

    def walk(node, parent_class: Optional[str] = None) -> None:
        if node.type in ("class_declaration", "interface_declaration", "enum_declaration"):
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node, source_bytes) if name_node else "<anon>"
            start, end = _node_lines(node)
            doc = _extract_javadoc(node, source_bytes)
            entities.append(
                RawEntity(
                    entity_type="class",
                    name=name,
                    start_line=start,
                    end_line=end,
                    content=_node_text(node, source_bytes),
                    docstring=doc,
                    annotations=_java_modifiers(node, source_bytes),
                )
            )
            for child in node.children:
                walk(child, parent_class=name)

        elif node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node, source_bytes) if name_node else "<anon>"
            start, end = _node_lines(node)
            doc = _extract_javadoc(node, source_bytes)
            etype = "method" if parent_class else "function"
            entities.append(
                RawEntity(
                    entity_type=etype,
                    name=name,
                    start_line=start,
                    end_line=end,
                    content=_node_text(node, source_bytes),
                    docstring=doc,
                    parent_name=parent_class,
                    annotations=_java_modifiers(node, source_bytes),
                )
            )
        else:
            for child in node.children:
                walk(child, parent_class=parent_class)

    walk(tree.root_node)
    return entities


# ── Go parser ──────────────────────────────────────────────────────────────


def _go_comment_above(node, source_bytes: bytes) -> Optional[str]:
    prev = node.prev_sibling
    lines = []
    while prev and prev.type == "comment":
        lines.insert(0, _node_text(prev, source_bytes).lstrip("/ ").strip())
        prev = prev.prev_sibling
    return " ".join(lines) if lines else None


def parse_golang(source: str) -> list[RawEntity]:
    if not _TS_GO:
        return []

    source_bytes = source.encode("utf-8")
    parser = _Parser(_GO_LANGUAGE)
    tree = parser.parse(source_bytes)

    entities: list[RawEntity] = []

    def walk(node) -> None:
        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node, source_bytes) if name_node else "<anon>"
            start, end = _node_lines(node)
            doc = _go_comment_above(node, source_bytes)
            entities.append(
                RawEntity(
                    entity_type="function",
                    name=name,
                    start_line=start,
                    end_line=end,
                    content=_node_text(node, source_bytes),
                    docstring=doc,
                )
            )

        elif node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node, source_bytes) if name_node else "<anon>"
            # receiver type as parent_name
            receiver = node.child_by_field_name("receiver")
            parent = None
            if receiver:
                for child in receiver.children:
                    if child.type in ("type_identifier", "pointer_type"):
                        parent = _node_text(child, source_bytes).lstrip("*")
                        break
            start, end = _node_lines(node)
            doc = _go_comment_above(node, source_bytes)
            entities.append(
                RawEntity(
                    entity_type="method",
                    name=name,
                    start_line=start,
                    end_line=end,
                    content=_node_text(node, source_bytes),
                    docstring=doc,
                    parent_name=parent,
                )
            )

        elif node.type == "type_declaration":
            # struct / interface type declarations
            for spec in node.children:
                if spec.type == "type_spec":
                    name_node = spec.child_by_field_name("name")
                    type_node = spec.child_by_field_name("type")
                    if type_node and type_node.type in ("struct_type", "interface_type"):
                        name = _node_text(name_node, source_bytes) if name_node else "<anon>"
                        start, end = _node_lines(spec)
                        entities.append(
                            RawEntity(
                                entity_type="class",
                                name=name,
                                start_line=start,
                                end_line=end,
                                content=_node_text(spec, source_bytes),
                                docstring=_go_comment_above(node, source_bytes),
                            )
                        )

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return entities


# ── C++ parser ─────────────────────────────────────────────────────────────


def _cpp_comment_above(node, source_bytes: bytes) -> Optional[str]:
    prev = node.prev_sibling
    if prev and prev.type == "comment":
        return _node_text(prev, source_bytes).lstrip("/ ").strip()
    return None


def parse_cpp(source: str) -> list[RawEntity]:
    if not _TS_CPP:
        return []

    source_bytes = source.encode("utf-8")
    parser = _Parser(_CPP_LANGUAGE)
    tree = parser.parse(source_bytes)

    entities: list[RawEntity] = []

    def walk(node, parent_class: Optional[str] = None) -> None:
        if node.type in ("class_specifier", "struct_specifier"):
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node, source_bytes) if name_node else "<anon>"
            start, end = _node_lines(node)
            doc = _cpp_comment_above(node, source_bytes)
            entities.append(
                RawEntity(
                    entity_type="class",
                    name=name,
                    start_line=start,
                    end_line=end,
                    content=_node_text(node, source_bytes),
                    docstring=doc,
                )
            )
            for child in node.children:
                walk(child, parent_class=name)

        elif node.type == "function_definition":
            # declarator → function_declarator → declarator (name)
            decl = node.child_by_field_name("declarator")
            name = "<anon>"
            if decl:
                if decl.type == "function_declarator":
                    inner = decl.child_by_field_name("declarator")
                    name = _node_text(inner, source_bytes) if inner else "<anon>"
                elif decl.type == "pointer_declarator":
                    inner = decl.child_by_field_name("declarator")
                    if inner and inner.type == "function_declarator":
                        name_node = inner.child_by_field_name("declarator")
                        name = _node_text(name_node, source_bytes) if name_node else "<anon>"
                else:
                    name = _node_text(decl, source_bytes).split("(")[0].strip()

            start, end = _node_lines(node)
            doc = _cpp_comment_above(node, source_bytes)
            etype = "method" if parent_class else "function"
            entities.append(
                RawEntity(
                    entity_type=etype,
                    name=name,
                    start_line=start,
                    end_line=end,
                    content=_node_text(node, source_bytes),
                    docstring=doc,
                    parent_name=parent_class,
                )
            )
        else:
            for child in node.children:
                walk(child, parent_class=parent_class)

    walk(tree.root_node)
    return entities


# ── Dispatcher ─────────────────────────────────────────────────────────────

LANGUAGE_PARSERS = {
    "python": parse_python,
    "java": parse_java,
    "golang": parse_golang,
    "cpp": parse_cpp,
}


def parse_file(source: str, language: str) -> list[RawEntity]:
    """
    Parse *source* text for the given *language*.
    Returns an empty list for unsupported or unavailable grammars.
    """
    fn = LANGUAGE_PARSERS.get(language)
    if fn is None:
        logger.debug("No parser registered for language %r", language)
        return []
    try:
        return fn(source)
    except Exception as exc:
        logger.warning("AST parse error (%s): %s", language, exc)
        return []
