"""Repo-level import graph construction.

Builds a ``networkx.DiGraph`` where an edge ``A -> B`` means *module A imports
module B*. Only intra-repo edges are recorded — the stdlib and third-party
packages can't be affected by a change inside the repo, so they are dropped.

Resolution is the interesting part. An import names a *dotted path* that mixes
packages, modules and imported symbols (``from pkg.mod import thing`` ->
``"pkg.mod.thing"``). We don't know where the module ends and the symbol begins,
so we try progressively shorter prefixes against the set of known repo modules
and take the longest match. Relative imports are first rebased onto the
importing file's package.
"""

from __future__ import annotations

import ast

import networkx as nx

from .index import RepoIndex, ensure_index, find_python_files, module_name_for

__all__ = [
    "build_import_graph",
    "find_python_files",
    "module_name_for",
    "resolve_relative",
]


def _package_of(module: str) -> str:
    """The package a module lives in: its dotted name minus the last part."""
    return module.rpartition(".")[0]


def resolve_relative(dotted: str, importer_module: str, is_package: bool) -> str:
    """Rebase a relative import string onto an absolute dotted path.

    ``dotted`` still carries its leading dots (``".sibling"``, ``"..pkg.thing"``).
    The number of dots is the level: 1 = current package, 2 = parent, etc.
    """
    level = len(dotted) - len(dotted.lstrip("."))
    remainder = dotted[level:]
    # For a package __init__, level 1 is the package itself; for a regular
    # module it is the module's containing package.
    anchor = importer_module if is_package else _package_of(importer_module)
    for _ in range(level - 1):
        anchor = _package_of(anchor)
    if anchor and remainder:
        return f"{anchor}.{remainder}"
    return anchor or remainder


def _longest_module_prefix(dotted: str, known_modules) -> str | None:
    """The longest prefix of ``dotted`` that names a known repo module."""
    parts = dotted.split(".")
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        if candidate in known_modules:
            return candidate
    return None


def _imports_with_lines(nodes):
    """Yield ``(dotted_name, lineno)`` for each import statement in ``nodes``."""
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            for alias in node.names:
                sep = "." if base and not base.endswith(".") else ""
                yield f"{base}{sep}{alias.name}", node.lineno


def build_import_graph(repo, exclude=()) -> nx.DiGraph:
    """Build the intra-repo import graph.

    ``repo`` is a path or a ``RepoIndex``. Nodes are dotted module names with
    ``path``/``rel_path``/``is_test`` attributes; edge ``A -> B`` (``linenos`` =
    the import statements' lines) means module A imports module B.
    """
    index: RepoIndex = ensure_index(repo, exclude)
    known = index.modules

    graph = nx.DiGraph()
    for sf in index.files.values():
        graph.add_node(
            sf.module, path=sf.path, rel_path=sf.rel_path,
            is_package=sf.is_package, is_test=sf.is_test,
            parse_error=sf.error,
        )

    for sf in index.parsed():
        for dotted, lineno in _imports_with_lines(sf.imports):
            if dotted.startswith("."):
                dotted = resolve_relative(dotted, sf.module, sf.is_package)
            target = _longest_module_prefix(dotted, known)
            if target is None or target == sf.module:
                continue
            if graph.has_edge(sf.module, target):
                graph.edges[sf.module, target]["linenos"].append(lineno)
            else:
                graph.add_edge(sf.module, target, linenos=[lineno])
    return graph
