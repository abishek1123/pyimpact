"""Repo-level import graph construction.

Walks every ``.py`` file under a repo root, maps each file to its dotted module
name, and builds a ``networkx.DiGraph`` where an edge ``A -> B`` means *file A
imports file B*. Only intra-repo edges are recorded — imports of the stdlib or
third-party packages are not repo files, so they can't be impacted by a change
inside the repo and are deliberately dropped.

Resolution is the interesting part. An import string from ``parser.get_imports``
names a *dotted path* that mixes packages, modules, and imported symbols
(``from pkg.mod import thing`` -> ``"pkg.mod.thing"``). We don't know where the
module ends and the symbol begins, so we try progressively shorter prefixes of
the dotted path against the set of known repo modules and take the longest match.
Relative imports are first rebased onto the importing file's package.
"""

from __future__ import annotations

import os
from typing import Optional

import networkx as nx

from .parser import get_imports


def find_python_files(repo_path: str) -> list[str]:
    """Return absolute paths of all ``.py`` files under ``repo_path``.

    Common non-source directories are skipped so we don't graph a project's own
    virtualenv or VCS metadata.
    """
    skip_dirs = {
        ".git",
        "venv",
        ".venv",
        "env",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "build",
        "dist",
        "node_modules",
        ".eggs",
    }
    py_files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            if name.endswith(".py"):
                py_files.append(os.path.join(dirpath, name))
    return sorted(py_files)


def module_name_for(file_path: str, repo_path: str) -> str:
    """Compute the dotted module name of ``file_path`` relative to ``repo_path``.

    ``repo/pkg/mod.py``      -> ``"pkg.mod"``
    ``repo/pkg/__init__.py`` -> ``"pkg"``
    ``repo/top.py``          -> ``"top"``
    """
    rel = os.path.relpath(file_path, repo_path)
    rel_no_ext = os.path.splitext(rel)[0]
    parts = rel_no_ext.split(os.sep)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_of(module: str) -> str:
    """Return the package a module lives in (its dotted name minus the last part).

    Used as the anchor for resolving relative imports. For a package's own
    ``__init__`` (whose module name is the package itself) the caller passes the
    package name directly.
    """
    return module.rpartition(".")[0]


def resolve_relative(dotted: str, importer_module: str, is_package: bool) -> str:
    """Rebase a relative import string onto an absolute dotted path.

    ``dotted`` still carries its leading dots (``".sibling"``, ``"..pkg.thing"``)
    as produced by ``parser.get_imports``. ``level`` is the number of leading
    dots: level 1 means "current package", level 2 means "parent package", etc.
    """
    level = len(dotted) - len(dotted.lstrip("."))
    remainder = dotted[level:]

    # Anchor: for a package __init__, level 1 is the package itself; for a
    # regular module, level 1 is the module's containing package.
    anchor = importer_module if is_package else _package_of(importer_module)
    # Each extra dot beyond the first walks one package upward.
    for _ in range(level - 1):
        anchor = _package_of(anchor)

    if anchor and remainder:
        return f"{anchor}.{remainder}"
    return anchor or remainder


def _longest_module_prefix(dotted: str, known_modules: set[str]) -> Optional[str]:
    """Return the longest prefix of ``dotted`` that is a known repo module.

    ``dotted`` may end in an imported symbol or a submodule name we can't
    distinguish, so we shorten from the right until a prefix names a real module.
    """
    parts = dotted.split(".")
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        if candidate in known_modules:
            return candidate
    return None


def build_import_graph(repo_path: str) -> nx.DiGraph:
    """Build the intra-repo import graph.

    Nodes are dotted module names (one per ``.py`` file). Each node carries a
    ``path`` attribute with its absolute file path. An edge ``A -> B`` means file
    ``A`` imports something that resolves to repo module ``B``.
    """
    repo_path = os.path.abspath(repo_path)
    files = find_python_files(repo_path)

    module_of: dict[str, str] = {}  # file path -> module name
    for path in files:
        module_of[path] = module_name_for(path, repo_path)
    known_modules = set(module_of.values())

    graph = nx.DiGraph()
    for path, module in module_of.items():
        graph.add_node(module, path=path)

    for path, module in module_of.items():
        is_package = os.path.basename(path) == "__init__.py"
        try:
            raw_imports = get_imports(path)
        except SyntaxError:
            # A file we can't parse contributes no edges; skip it rather than
            # aborting analysis of the whole repo.
            continue

        for dotted in raw_imports:
            if dotted.startswith("."):
                dotted = resolve_relative(dotted, module, is_package)
            target = _longest_module_prefix(dotted, known_modules)
            if target is not None and target != module:
                graph.add_edge(module, target)

    return graph
