"""Change-impact analysis for a git diff.

``pyimpact diff`` answers the question a reviewer actually has: *given everything
this branch (or my working tree) changed, what else could break?*

1. Run ``git diff --unified=0`` against the base and collect changed line numbers
   on both sides of every Python file (plus untracked new files).
2. Map new-side lines onto the *innermost* function, method or class whose span
   contains them in the current code; lines outside any definition are
   module-level changes.
3. Map old-side lines onto symbols of the *base* version of the file (read with
   ``git show``). A touched symbol that no longer exists was removed or renamed.
4. Impact of changed symbols = reverse reachability in the current graph.
   Impact of removed symbols = reverse reachability in a graph built from the
   base revision (via ``git archive``), keeping only dependents that still
   exist — that is exactly the code still calling something that is gone.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field

import networkx as nx

from .call_graph import _FUNC_TYPES, _first_line, _toplevel, node_id
from .impact import Affected, build_unified_graph, find_impact
from .index import RepoIndex, ensure_index
from .parser import PARSE_ERRORS

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # git's well-known empty tree
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class DiffError(RuntimeError):
    """Raised when git is unavailable or the repo/base is invalid."""


@dataclass
class FileDiff:
    old_path: str | None              # git-root-relative POSIX path; None if added
    new_path: str | None              # None if deleted
    old_lines: set = field(default_factory=set)
    new_lines: set = field(default_factory=set)
    whole_file: bool = False          # untracked file: every line is new


@dataclass
class ChangedSymbol:
    node: str                         # node id (module name for module-level changes)
    kind: str                         # module | class | function | method
    change: str                       # modified | added | removed
    rel_path: str
    line: int | None = None


@dataclass
class DiffResult:
    graph: nx.DiGraph
    base_label: str
    files: list                       # changed .py files under the analysed root
    changed: list                     # [ChangedSymbol] present in the current code
    affected: list                    # [Affected] dependents of the changed symbols
    removed: list = field(default_factory=list)            # [ChangedSymbol]
    removed_affected: list = field(default_factory=list)   # [Affected] directly referencing removed code


# -- git plumbing --------------------------------------------------------------

def _git(args, cwd: str, check: bool = True) -> str:
    try:
        proc = subprocess.run(
            ["git", "-c", "core.quotepath=off", *args], cwd=cwd,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError as exc:
        raise DiffError("git is not installed or not on PATH") from exc
    if check and proc.returncode != 0:
        raise DiffError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout


def git_root(path: str) -> str:
    out = _git(["rev-parse", "--show-toplevel"], cwd=path, check=False).strip()
    if not out:
        raise DiffError(f"{path} is not inside a git repository")
    return os.path.abspath(out)


def _unquote(path: str) -> str:
    path = path.rstrip("\t").rstrip()
    if len(path) >= 2 and path[0] == path[-1] == '"':
        path = path[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape")
        path = path.encode("latin-1", "replace").decode("utf-8", "replace")
    return path


def _strip(path: str, prefix: str) -> str | None:
    path = _unquote(path)
    if path == "/dev/null":
        return None
    return path[len(prefix):] if path.startswith(prefix) else path


def parse_unified_diff(text: str) -> list[FileDiff]:
    """Parse ``git diff --unified=0`` output into per-file changed line sets."""
    files: list[FileDiff] = []
    cur: FileDiff | None = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            cur = FileDiff(None, None)
            files.append(cur)
            m = re.match(r"diff --git a/(.*) b/(.*)$", line)
            if m:
                cur.old_path, cur.new_path = m.group(1), m.group(2)
        elif cur is None:
            continue
        elif line.startswith("--- "):
            cur.old_path = _strip(line[4:], "a/")
        elif line.startswith("+++ "):
            cur.new_path = _strip(line[4:], "b/")
        elif line.startswith("new file mode"):
            cur.old_path = None
        elif line.startswith("deleted file mode"):
            cur.new_path = None
        elif line.startswith("@@"):
            m = _HUNK.match(line)
            if m:
                a, b = int(m[1]), int(m[2]) if m[2] is not None else 1
                c, d = int(m[3]), int(m[4]) if m[4] is not None else 1
                cur.old_lines.update(range(a, a + b))
                cur.new_lines.update(range(c, c + d))
    return files


def _base_revision(root: str, base: str | None, staged: bool) -> tuple[str, list, str]:
    """-> (revision to compare against, git-diff args, human label)."""
    has_head = bool(_git(["rev-parse", "--verify", "--quiet", "HEAD"], root, check=False).strip())
    if staged:
        rev = "HEAD" if has_head else EMPTY_TREE
        return rev, ["diff", "--cached", rev], "staged changes"
    if base is None:
        rev = "HEAD" if has_head else EMPTY_TREE
        return rev, ["diff", rev], "uncommitted changes (vs HEAD)"
    if not _git(["rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"], root, check=False).strip():
        raise DiffError(f"unknown base revision: {base!r} (fetch it first, e.g. "
                        f"`git fetch origin {base}`)")
    merge_base = _git(["merge-base", base, "HEAD"], root).strip()
    return merge_base, ["diff", merge_base], f"changes since {base} (merge-base {merge_base[:8]})"


def collect_file_diffs(root: str, base: str | None = None, staged: bool = False):
    """-> (revision, label, [FileDiff]) for every changed ``.py`` file."""
    rev, args, label = _base_revision(root, base, staged)
    text = _git([*args, "--no-color", "--no-ext-diff", "--unified=0", "-M",
                 "--", "*.py"], root)
    files = parse_unified_diff(text)
    if not staged:
        untracked = _git(["ls-files", "--others", "--exclude-standard", "--", "*.py"], root)
        files += [FileDiff(None, p, whole_file=True) for p in untracked.splitlines() if p]
    return rev, label, files


# -- symbol spans ----------------------------------------------------------------

def _symbol_spans(tree: ast.Module) -> list[tuple[str, str, int, int]]:
    """(qualname, kind, start, end) for top-level functions, classes, methods."""
    spans = []
    for stmt in _toplevel(tree.body):
        if isinstance(stmt, _FUNC_TYPES):
            spans.append((stmt.name, "function", _first_line(stmt), stmt.end_lineno))
        elif isinstance(stmt, ast.ClassDef):
            spans.append((stmt.name, "class", _first_line(stmt), stmt.end_lineno))
            for item in _toplevel(stmt.body):
                if isinstance(item, _FUNC_TYPES):
                    spans.append((f"{stmt.name}.{item.name}", "method",
                                  _first_line(item), item.end_lineno))
    return spans


def _innermost(spans, line: int):
    best = None
    for entry in spans:
        start, end = entry[-2], entry[-1]
        if start <= line <= end and (best is None or end - start < best[-1] - best[-2]):
            best = entry
    return best


def _head_spans(graph: nx.DiGraph) -> dict[str, list]:
    """module -> [(node, kind, start, end)] from the current graph."""
    by_module: dict[str, list] = {}
    for node, data in graph.nodes(data=True):
        if data.get("kind") in ("function", "method", "class"):
            for start, end in data.get("spans") or [[data["start_lineno"], data["end_lineno"]]]:
                by_module.setdefault(data["module"], []).append((node, data["kind"], start, end))
    return by_module


def _read_base(root: str, rev: str, path: str) -> ast.Module | None:
    if rev == EMPTY_TREE:
        return None
    proc = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=root, capture_output=True)
    if proc.returncode != 0:
        return None
    try:
        return ast.parse(proc.stdout)
    except PARSE_ERRORS:
        return None


def _base_graph(root: str, rev: str, sub: str, exclude) -> tuple[nx.DiGraph, str]:
    """Build the unified graph of the analysed directory at ``rev``.
    Returns (graph, temp dir) — the caller removes the dir."""
    tmp = tempfile.mkdtemp(prefix="pyimpact-base-")
    proc = subprocess.Popen(["git", "archive", "--format=tar", rev, "--", sub or "."],
                            cwd=root, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".py"):
                dest = os.path.join(tmp, *member.name.split("/"))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with tar.extractfile(member) as src, open(dest, "wb") as out:
                    out.write(src.read())
    proc.wait()
    base_root = os.path.join(tmp, *sub.split("/")) if sub else tmp
    if not os.path.isdir(base_root):
        os.makedirs(base_root, exist_ok=True)
    return build_unified_graph(RepoIndex(base_root, exclude)), tmp


# -- the analysis ------------------------------------------------------------------

def analyze_diff(repo, base: str | None = None, staged: bool = False, exclude=(),
                 max_distance: int | None = None) -> DiffResult:
    """Impact of the changes between ``base`` (merge-base with HEAD; default:
    HEAD) and the working tree. ``repo`` is a path or a ``RepoIndex``."""
    index = ensure_index(repo, exclude)
    root = git_root(index.root)
    sub = os.path.relpath(index.root, root).replace(os.sep, "/")
    sub = "" if sub == "." else sub
    rev, label, file_diffs = collect_file_diffs(root, base, staged)

    graph = build_unified_graph(index)
    head_spans = _head_spans(graph)
    changed: dict[str, ChangedSymbol] = {}
    removed: dict[tuple, ChangedSymbol] = {}
    touched_files: list[str] = []

    def in_scope(path):
        return path is not None and (not sub or path == sub or path.startswith(sub + "/"))

    for fd in file_diffs:
        if not (in_scope(fd.new_path) or in_scope(fd.old_path)):
            continue
        head_sf = index.by_path(os.path.join(root, fd.new_path)) if fd.new_path else None
        base_tree = _read_base(root, rev, fd.old_path) if fd.old_path else None
        base_spans = _symbol_spans(base_tree) if base_tree is not None else []
        base_quals = {q for q, *_ in base_spans}
        if head_sf is None and base_tree is None:
            continue
        touched_files.append(fd.new_path or fd.old_path)

        def mark(node, kind, line, sf=head_sf, fd=fd, base_quals=base_quals):
            if node in changed:
                return
            qual = graph.nodes[node].get("qualname")
            is_new = fd.old_path is None or (kind != "module" and qual not in base_quals)
            change = "added" if is_new else "modified"
            changed[node] = ChangedSymbol(node, kind, change, sf.rel_path, line)

        if head_sf is not None:
            spans = head_spans.get(head_sf.module, [])
            if fd.whole_file or fd.old_path is None:
                lines = set(range(1, _line_count(head_sf.path) + 1))
            else:
                lines = fd.new_lines
            for line in sorted(lines):
                hit = _innermost(spans, line)
                if hit:
                    mark(hit[0], hit[1], hit[2])
                elif _in_statement(head_sf.tree, line):
                    mark(head_sf.module, "module", line)

        if base_tree is None:
            continue
        if fd.new_path is None:
            # Whole file deleted: every symbol in it (and the module) is gone.
            for qual, kind, start, _ in base_spans:
                removed.setdefault((fd.old_path, qual), ChangedSymbol(
                    qual, kind, "removed", fd.old_path, start))
            removed.setdefault((fd.old_path, None), ChangedSymbol(
                "", "module", "removed", fd.old_path, None))
            continue
        for line in sorted(fd.old_lines):
            hit = _innermost(base_spans, line)
            if hit is None:
                if head_sf is not None and _in_statement(base_tree, line):
                    mark(head_sf.module, "module", line)
                continue
            qual = hit[0]
            nid = node_id(head_sf.module, qual) if head_sf is not None else None
            if nid and graph.has_node(nid):
                mark(nid, graph.nodes[nid]["kind"], graph.nodes[nid].get("lineno"))
            else:
                removed.setdefault((fd.old_path, qual), ChangedSymbol(
                    qual, hit[1], "removed", fd.old_path, hit[2]))

    targets = list(changed)
    affected = find_impact(graph, targets, max_distance=max_distance) if targets else []

    removed_list, removed_affected = [], []
    if removed and rev != EMPTY_TREE:
        removed_list, removed_affected, downstream = _removed_impact(
            root, rev, sub, index.exclude, removed, graph, set(changed), max_distance)
        breaking_nodes = {a.node for a in removed_affected}
        best = {a.node: a for a in affected}
        for a in downstream:
            if a.node in breaking_nodes:
                continue
            current = best.get(a.node)
            if current is None or (a.score, -a.distance) > (current.score, -current.distance):
                best[a.node] = a
        affected = sorted((a for a in best.values() if a.node not in breaking_nodes),
                          key=lambda a: (a.distance, -a.score, a.node))

    return DiffResult(graph=graph, base_label=label, files=touched_files,
                      changed=sorted(changed.values(), key=lambda c: (c.rel_path, c.line or 0)),
                      affected=affected, removed=removed_list,
                      removed_affected=removed_affected)


def _removed_impact(root, rev, sub, exclude, removed, head_graph, changed_nodes,
                    max_distance):
    import shutil

    base_graph, tmp = _base_graph(root, rev, sub, exclude)
    try:
        index = base_graph.graph["index"]
        targets, symbols = [], []
        for (path, qual), sym in removed.items():
            sf = index.by_path(os.path.join(tmp, *path.split("/")))
            if sf is None:
                continue
            nid = sf.module if qual is None else node_id(sf.module, qual)
            if not base_graph.has_node(nid):
                continue
            sym.node = nid
            sym.rel_path = path[len(sub) + 1:] if sub and path.startswith(sub + "/") else path
            symbols.append(sym)
            targets.append(nid)
        found = find_impact(base_graph, targets, max_distance=max_distance) if targets else []
        survivors = [a for a in found
                     if head_graph.has_node(a.node) and a.node not in changed_nodes]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # Direct code references to something that no longer exists will break.
    # (The module/class that merely *defined* it doesn't.) Everything further
    # out is affected through those breakages and joins the normal blast radius.
    breaking = [a for a in survivors if a.distance == 1 and a.via not in ("contains", "member")]
    downstream = [a for a in survivors if a not in breaking and a.via not in ("contains", "member")
                  or a.distance > 1]
    return sorted(symbols, key=lambda s: (s.rel_path, s.line or 0)), breaking, downstream


def _line_count(path: str) -> int:
    try:
        with open(path, "rb") as f:
            return f.read().count(b"\n") + 1
    except OSError:
        return 0


def _in_statement(tree: ast.Module | None, line: int) -> bool:
    """True if ``line`` belongs to a module-level statement (not a blank line or
    comment), so whitespace/comment-only edits don't flag the whole module."""
    if tree is None:
        return False
    return any(stmt.lineno <= line <= (stmt.end_lineno or stmt.lineno)
               for stmt in tree.body)


__all__ = [
    "Affected", "ChangedSymbol", "DiffError", "DiffResult", "FileDiff",
    "analyze_diff", "collect_file_diffs", "git_root", "parse_unified_diff",
]
