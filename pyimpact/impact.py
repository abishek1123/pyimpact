"""Change-impact analysis.

This is the query the whole tool exists to answer: *given a target function,
class or file, what could break if I change it?* The deterministic answer is a
reverse-reachability query on a dependency graph.

Every edge in the unified graph points in the "depends-on" direction — ``A -> B``
means *A depends on B*, so a change to B may affect A:

=============  ===============================================  ======
edge kind      meaning                                          weight
=============  ===============================================  ======
call/direct    A calls B at a resolved call site                1.00
call/reference A passes B around (callback, decorator)          0.85
call/virtual   B overrides the method A calls (dynamic dispatch) 0.60
instantiate    A constructs class B                             0.90
uses           A references class B (isinstance, annotations)   0.70
inherits       class A subclasses B                             0.80
member         class A defines method B                         0.40
contains       module A defines function/class B                1.00
import         module A imports module B                        0.50
=============  ===============================================  ======

Impact = every node that can reach the target. ``distance`` is the fewest hops;
``score`` is the strongest path's confidence — the product of edge weights along
it, decayed by 0.85 per extra hop — and maps to a ``severity`` of high (≥ 0.7),
medium (≥ 0.35) or low. So a direct caller is *high*, a file that merely imports
the module is *medium*, and something four import hops away is *low*.
"""

from __future__ import annotations

import difflib
import heapq
import os
from collections import deque
from dataclasses import dataclass

import networkx as nx

from .call_graph import build_call_graph
from .import_graph import build_import_graph
from .index import ensure_index, module_name_for, paused_gc

CALL_WEIGHTS = {"direct": 1.0, "reference": 0.85, "virtual": 0.6}
EDGE_WEIGHTS = {
    "instantiate": 0.9,
    "uses": 0.7,
    "inherits": 0.8,
    "member": 0.4,
    "contains": 1.0,
    "import": 0.5,
}
HOP_DECAY = 0.85
SEVERITY_LEVELS = (("high", 0.7), ("medium", 0.35), ("low", 0.0))
SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
CALLABLE_KINDS = frozenset({"function", "method"})


def edge_weight(data: dict) -> float:
    kind = data.get("kind")
    if kind == "call":
        return CALL_WEIGHTS.get(data.get("strength", "direct"), 1.0)
    return EDGE_WEIGHTS.get(kind, 0.5)


def severity_for(score: float) -> str:
    for name, threshold in SEVERITY_LEVELS:
        if score >= threshold:
            return name
    return "low"


def build_unified_graph(repo, exclude=()) -> nx.DiGraph:
    """Merge the import graph and call graph into one depends-on graph.

    ``repo`` is a path or a ``RepoIndex``. Nodes carry ``kind`` (``module``,
    ``class``, ``function``, ``method``) plus ``path``/``rel_path``/``is_test``.
    ``graph.graph`` holds ``root``, ``index`` and ``skipped`` (unparseable files).
    """
    index = ensure_index(repo, exclude)
    with paused_gc():
        return _build_unified(index)


def _build_unified(index) -> nx.DiGraph:
    import_graph = build_import_graph(index)
    call_graph = build_call_graph(index)

    g = nx.DiGraph(root=index.root, index=index,
                   skipped=[(f.rel_path, f.error) for f in index.skipped])

    for module, data in import_graph.nodes(data=True):
        g.add_node(module, kind="module", qualname=module, module=module, **data)
    for node, data in call_graph.nodes(data=True):
        mod = g.nodes[data["module"]]
        g.add_node(node, path=mod["path"], rel_path=mod["rel_path"], **data)

    # Structural edges first: they win over behavioural duplicates.
    for node, data in call_graph.nodes(data=True):
        if data["kind"] == "method":
            g.add_edge(data["class_id"], node, kind="member")
        else:
            g.add_edge(data["module"], node, kind="contains")
        for base in data.get("bases", ()):
            g.add_edge(node, base, kind="inherits")

    for u, v, data in import_graph.edges(data=True):
        g.add_edge(u, v, kind="import", **data)
    for u, v, data in call_graph.edges(data=True):
        g.add_edge(u, v, **data)
    for u, v, data in call_graph.graph.get("module_edges", ()):
        if not g.has_edge(u, v):
            g.add_edge(u, v, **data)
    return g


# -- target resolution ---------------------------------------------------------

def resolve_target(graph: nx.DiGraph, target: str, repo_path: str | None = None):
    """Map a user-supplied target string to a node id, or ``None``.

    Accepts an exact node id (``app.auth`` / ``app.auth:login``), a file path
    (``app/auth.py``), or a dotted name (``app.auth.login``,
    ``app.models.User.check``) where any dot may be the module boundary.
    """
    target = target.strip()
    if graph.has_node(target):
        return target
    root = repo_path or graph.graph.get("root")

    if target.endswith(".py") or "/" in target or "\\" in target:
        candidate = target if os.path.isabs(target) or root is None else \
            os.path.join(root, target)
        if os.path.isfile(candidate):
            index = graph.graph.get("index")
            sf = index.by_path(candidate) if index is not None else None
            module = sf.module if sf else module_name_for(candidate, root)
            if graph.has_node(module):
                return module

    if ":" not in target and "." in target:
        parts = target.split(".")
        for i in range(len(parts) - 1, 0, -1):
            candidate = f"{'.'.join(parts[:i])}:{'.'.join(parts[i:])}"
            if graph.has_node(candidate):
                return candidate
    return None


def suggest_targets(graph: nx.DiGraph, target: str, limit: int = 5) -> list[str]:
    """Closest node ids to a target that didn't resolve (for error messages)."""
    leaf = target.replace(":", ".").split(".")[-1].lower()
    by_leaf = [n for n, d in graph.nodes(data=True)
               if str(d.get("qualname", n)).split(".")[-1].lower() == leaf]
    if by_leaf:
        return sorted(by_leaf)[:limit]
    return difflib.get_close_matches(target, list(graph.nodes), n=limit, cutoff=0.5)


# -- the impact query ------------------------------------------------------------

@dataclass(frozen=True)
class Affected:
    """One node in the target's blast radius."""

    node: str
    kind: str            # module | class | function | method
    distance: int        # fewest hops from a target (1 = direct dependent)
    via: str             # kind of the edge leading toward the target
    toward: str = ""     # next node on that shortest path (BFS parent)
    score: float = 1.0   # confidence of the strongest path, 0..1
    severity: str = "high"
    is_test: bool = False


def find_impact(graph: nx.DiGraph, targets, max_distance: int | None = None
                ) -> list[Affected]:
    """Reverse-traverse from one or more targets; return everything that depends
    on them, sorted by distance, then strongest score, then node id."""
    sources = [targets] if isinstance(targets, str) else list(dict.fromkeys(targets))
    sources = [t for t in sources if graph.has_node(t)]
    source_set = set(sources)

    # Fewest hops (and the edge that got us there) via multi-source BFS.
    distance = {t: 0 for t in sources}
    via: dict[str, str] = {}
    toward: dict[str, str] = {}
    queue = deque(sources)
    while queue:
        node = queue.popleft()
        if max_distance is not None and distance[node] >= max_distance:
            continue
        for pred in graph.predecessors(node):
            if pred not in distance:
                distance[pred] = distance[node] + 1
                via[pred] = graph.edges[pred, node].get("kind", "")
                toward[pred] = node
                queue.append(pred)

    # Strongest path confidence via max-product Dijkstra (weights are <= 1).
    score = {t: 1.0 for t in sources}
    heap = [(-1.0, t) for t in sources]
    done: set = set()
    while heap:
        neg, node = heapq.heappop(heap)
        if node in done:
            continue
        done.add(node)
        decay = 1.0 if node in source_set else HOP_DECAY
        for pred in graph.predecessors(node):
            if pred not in distance:
                continue
            cand = -neg * edge_weight(graph.edges[pred, node]) * decay
            if cand > score.get(pred, 0.0):
                score[pred] = cand
                heapq.heappush(heap, (-cand, pred))

    results = []
    for n, d in distance.items():
        if n in source_set:
            continue
        s = round(score.get(n, 0.0), 3)
        data = graph.nodes[n]
        results.append(Affected(
            node=n, kind=data.get("kind", "unknown"), distance=d, via=via[n],
            toward=toward[n], score=s, severity=severity_for(s),
            is_test=bool(data.get("is_test")),
        ))
    results.sort(key=lambda a: (a.distance, -a.score, a.node))
    return results


def pytest_id(graph: nx.DiGraph, node: str) -> str | None:
    """``tests/test_x.py::TestA::test_b`` for a test function/method node."""
    data = graph.nodes[node]
    if data.get("kind") not in CALLABLE_KINDS or not data.get("is_test"):
        return None
    qual = data["qualname"].split(".")
    if not qual[-1].startswith("test"):
        return None
    if len(qual) == 2 and not qual[0].startswith("Test"):
        return None
    return f"{data['rel_path']}::{'::'.join(qual)}"


def affected_tests(graph: nx.DiGraph, affected) -> tuple[list[str], list[str]]:
    """(pytest node ids, test files) among the affected nodes."""
    ids, files = [], []
    for a in affected:
        if not a.is_test:
            continue
        pid = pytest_id(graph, a.node)
        if pid:
            ids.append(pid)
        rel = graph.nodes[a.node].get("rel_path")
        if rel and rel not in files and _collectable(rel):
            files.append(rel)
    return sorted(set(ids)), sorted(files)


def _collectable(rel_path: str) -> bool:
    """Would pytest collect tests from this file by default?"""
    name = rel_path.rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py")


def analyze(repo_path, target: str, exclude=()):
    """Build the graph, resolve ``target``, and run the impact query.

    Returns ``(resolved_node, affected_list)``; ``resolved_node`` is ``None``
    when the target matches nothing in the repo.
    """
    graph = build_unified_graph(repo_path, exclude)
    resolved = resolve_target(graph, target)
    if resolved is None:
        return None, []
    return resolved, find_impact(graph, resolved)
