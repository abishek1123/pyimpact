"""AST parsing primitives for PyImpact.

This is the lowest layer of the pipeline: turn a Python source file into an AST
and extract its imports. Everything downstream (import graph, call graph) is
built on top of these functions, so they are kept small and independently
testable.
"""

from __future__ import annotations

import ast

# Exceptions that mean "this file can't be analyzed" rather than "PyImpact has a
# bug": invalid syntax (or syntax from a newer Python), undecodable bytes, null
# bytes, or pathological nesting.
PARSE_ERRORS = (SyntaxError, ValueError, UnicodeDecodeError, RecursionError,
                MemoryError)


def parse_file(file_path: str) -> ast.Module:
    """Parse a Python source file into an AST module node.

    The file is read as bytes so ``ast.parse`` honours BOMs and PEP 263
    ``# -*- coding: ... -*-`` declarations instead of assuming UTF-8. The
    ``filename`` is passed through so a ``SyntaxError`` carries the real path,
    which keeps failures on real-world repos legible.
    """
    with open(file_path, "rb") as f:
        source = f.read()
    return ast.parse(source, filename=file_path)


def imports_from_tree(tree: ast.AST) -> list[str]:
    """Return the dotted names imported anywhere in ``tree``.

    Both statement forms are flattened to strings:

    - ``import os`` / ``import os.path as p``  -> ``"os"``, ``"os.path"``
    - ``from x import y`` / ``from x import y as z`` -> ``"x.y"``
    - ``from . import y`` (relative)          -> ``".y"``
    - ``from ..pkg import y``                 -> ``"..pkg.y"``
    - ``from x import *``                     -> ``"x.*"``

    Relative imports keep their leading dots so callers can tell them apart
    from absolute ones; resolving those dots to concrete files is the import
    graph's job, not the parser's.
    """
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            for alias in node.names:
                sep = "." if base and not base.endswith(".") else ""
                imports.append(f"{base}{sep}{alias.name}")
    return imports


def get_imports(file_path: str) -> list[str]:
    """Parse ``file_path`` and return its imports (see ``imports_from_tree``)."""
    return imports_from_tree(parse_file(file_path))
