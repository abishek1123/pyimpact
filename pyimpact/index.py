"""Repository index: discover, name and parse every Python file exactly once.

Every analysis stage (import graph, call graph, search, stats) works off one
``RepoIndex`` so a repo is walked and parsed a single time, and every stage agrees
on the same module names.

Module naming is *package-aware*, mirroring how Python's import system names
modules: starting from a file, walk up while directories contain ``__init__.py``;
the module name is the path relative to the directory *above* the top-most
package. That makes ``src/`` layouts, monorepos and "point it at the package
itself" all produce the real dotted names (``requests.sessions``, not
``src.requests.sessions`` or bare ``sessions``), so absolute imports resolve.
Files that are not inside a package fall back to their path relative to the
repo root (``scripts/build.py`` -> ``scripts.build``).
"""

from __future__ import annotations

import ast
import contextlib
import fnmatch
import gc
import os
from dataclasses import dataclass
from functools import lru_cache

from .parser import PARSE_ERRORS, parse_file

# Directories that never contain first-party source worth analysing. Any
# directory whose name starts with "." is skipped as well.
DEFAULT_EXCLUDE_DIRS = frozenset({
    "venv", "env", "virtualenv", "__pycache__", "build", "dist",
    "node_modules", "site-packages", "__pypackages__",
})

TEST_DIR_NAMES = frozenset({"tests", "test", "testing"})


@contextlib.contextmanager
def paused_gc():
    """Pause the cyclic garbage collector during bulk AST work.

    Parsing a repo allocates millions of AST objects that all stay alive, so the
    collector keeps re-scanning an ever-growing heap for nothing — on a
    2,700-file tree that made indexing ~10x slower. Nothing we allocate here is
    garbage, so it is safe to pause it and restore the previous state after.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def _posix(path: str) -> str:
    return path.replace(os.sep, "/")


def _matches(rel: str, name: str, patterns) -> bool:
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(name, p)
               for p in patterns)


def find_python_files(repo_path: str, exclude=()) -> list[str]:
    """Return absolute paths of all ``.py`` files under ``repo_path``.

    Virtualenvs, caches, build output and hidden directories are skipped.
    ``exclude`` takes extra glob patterns matched against the repo-relative POSIX
    path or the bare name (``"tests"``, ``"*/migrations/*"``, ``"*_pb2.py"``).
    """
    repo_path = os.path.abspath(repo_path)
    patterns = tuple(exclude)
    py_files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_path):
        kept = []
        for d in dirnames:
            if d.startswith(".") or d in DEFAULT_EXCLUDE_DIRS:
                continue
            rel = _posix(os.path.relpath(os.path.join(dirpath, d), repo_path))
            if patterns and _matches(rel, d, patterns):
                continue
            kept.append(d)
        dirnames[:] = sorted(kept)
        for name in filenames:
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            rel = _posix(os.path.relpath(full, repo_path))
            if patterns and _matches(rel, name, patterns):
                continue
            py_files.append(full)
    return sorted(py_files)


@lru_cache(maxsize=4096)
def _is_package_dir(directory: str) -> bool:
    return os.path.isfile(os.path.join(directory, "__init__.py"))


def module_name_for(file_path: str, repo_path: str) -> str:
    """Compute the dotted module name of ``file_path`` (package-aware).

    ``src/pkg/mod.py`` (with ``src/pkg/__init__.py``) -> ``"pkg.mod"``
    ``pkg/__init__.py``                                -> ``"pkg"``
    ``scripts/tool.py`` (no ``__init__.py`` around)    -> ``"scripts.tool"``
    """
    file_path = os.path.abspath(file_path)
    top = None
    cur = os.path.dirname(file_path)
    while _is_package_dir(cur):
        top = cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    base = os.path.dirname(top) if top else os.path.abspath(repo_path)
    rel = os.path.splitext(os.path.relpath(file_path, base))[0]
    parts = [p for p in rel.split(os.sep) if p not in ("", ".")]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def is_test_path(rel_path: str) -> bool:
    """Heuristic: is this repo-relative path part of a test suite?"""
    parts = rel_path.lower().split("/")
    name = parts[-1]
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
        or any(p in TEST_DIR_NAMES for p in parts[:-1])
    )


_STMT_BLOCKS = ("body", "orelse", "finalbody", "handlers", "cases")


def import_nodes(tree: ast.AST) -> list:
    """All ``import``/``from ... import`` statements in ``tree``, in source order.

    Imports are statements, so only statement blocks are visited — expression
    subtrees (the vast majority of nodes) are never touched.
    """
    found, stack = [], [tree]
    while stack:
        node = stack.pop()
        for name in _STMT_BLOCKS:
            block = getattr(node, name, None)
            if type(block) is not list:
                continue
            for child in block:
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    found.append(child)
                elif isinstance(child, ast.AST):
                    stack.append(child)
    found.sort(key=lambda n: (n.lineno, n.col_offset))
    return found


@dataclass
class SourceFile:
    path: str              # absolute path
    rel_path: str          # POSIX path relative to the repo root
    module: str            # dotted module name (the node id in the graphs)
    is_package: bool       # True for a package's __init__.py
    is_test: bool
    tree: ast.Module | None
    error: str | None = None   # why the file couldn't be parsed, if it couldn't
    _imports: list | None = None

    @property
    def imports(self) -> list:
        """Cached import statements of this file (empty if unparseable)."""
        if self._imports is None:
            self._imports = import_nodes(self.tree) if self.tree is not None else []
        return self._imports


class RepoIndex:
    """All Python files of a repo, named and parsed once.

    ``files`` maps module name -> ``SourceFile`` for every discovered file,
    including unparseable ones (``tree`` is ``None`` and ``error`` explains why),
    so they still appear as nodes. ``skipped`` lists just the unparseable ones.
    """

    def __init__(self, root: str, exclude=()):
        self.root = os.path.abspath(root)
        if not os.path.isdir(self.root):
            raise NotADirectoryError(f"not a directory: {root}")
        self.exclude = tuple(exclude)
        self.files: dict[str, SourceFile] = {}
        _is_package_dir.cache_clear()

        with paused_gc():
            self._load()

    def _load(self):
        for path in find_python_files(self.root, self.exclude):
            rel = _posix(os.path.relpath(path, self.root))
            module = module_name_for(path, self.root)
            if not module or module in self.files:
                # Collision (e.g. a root-level foo.py and a src/foo package) or
                # an empty name: fall back to the unambiguous path-based name.
                module = os.path.splitext(rel)[0].replace("/", ".")
                if module.endswith(".__init__"):
                    module = module[: -len(".__init__")]
                while module in self.files:
                    module += "_"
            try:
                tree, error = parse_file(path), None
            except PARSE_ERRORS as exc:
                tree, error = None, f"{type(exc).__name__}: {exc}"
            except OSError as exc:
                tree, error = None, f"unreadable: {exc}"
            self.files[module] = SourceFile(
                path=path,
                rel_path=rel,
                module=module,
                is_package=os.path.basename(path) == "__init__.py",
                is_test=is_test_path(rel),
                tree=tree,
                error=error,
            )

    @property
    def skipped(self) -> list[SourceFile]:
        return [f for f in self.files.values() if f.tree is None]

    @property
    def modules(self) -> set[str]:
        return set(self.files)

    def parsed(self):
        """Iterate over the successfully parsed files."""
        return (f for f in self.files.values() if f.tree is not None)

    def by_path(self, path: str) -> SourceFile | None:
        path = os.path.normcase(os.path.abspath(path))
        for f in self.files.values():
            if os.path.normcase(f.path) == path:
                return f
        return None


def ensure_index(repo, exclude=()) -> RepoIndex:
    """Accept either a path or an existing ``RepoIndex``."""
    return repo if isinstance(repo, RepoIndex) else RepoIndex(repo, exclude)
