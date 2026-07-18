"""AST parsing primitives for PyImpact.

This is the lowest layer of the pipeline: turn a Python source file into an AST
and extract its imports. Everything downstream (import graph, call graph) is
built on top of these functions, so they are kept small and independently
testable.
"""

from __future__ import annotations

import ast


def parse_file(file_path: str) -> ast.Module:
    """Parse a Python source file into an AST module node.

    The ``filename`` is passed to ``ast.parse`` so that any ``SyntaxError``
    carries the real path, which makes failures on real-world repos legible.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    return ast.parse(source, filename=file_path)


def get_imports(file_path: str) -> list[str]:
    """Return the dotted names imported by a file, in source order.

    Both statement forms are flattened to strings:

    - ``import os`` / ``import os.path as p``  -> ``"os"``, ``"os.path"``
    - ``from x import y`` / ``from x import y as z`` -> ``"x.y"``
    - ``from . import y`` (relative)          -> ``".y"``
    - ``from ..pkg import y``                 -> ``"..pkg.y"``

    Relative imports keep their leading dots so callers can tell them apart
    from absolute ones; resolving those dots to concrete files is the import
    graph's job, not the parser's.
    """
    tree = parse_file(file_path)

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            module = node.module or ""
            base = f"{prefix}{module}"
            for alias in node.names:
                # `from x import *` -> keep the `*` so it isn't silently dropped.
                sep = "." if base and not base.endswith(".") else ""
                imports.append(f"{base}{sep}{alias.name}")

    return imports
