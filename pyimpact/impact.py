"""Change-impact analysis.

This is the query the whole tool exists to answer: *given a target function or
file, what could break if I change it?* The deterministic answer is a reverse
reachability query on a dependency graph.

Every edge in the unified graph points in the "depends-on" direction — ``A -> B``
means *A depends on B*, so a change to B may affect A:

- call edge      ``caller_func -> callee_func``     (caller depends on callee)
- import edge    ``importer_module -> imported_module``
- contains edge  ``module -> function``             (a module's behavior includes
                                                      each function it defines)

The contains edges bridge the two granularities: reverse-reaching from a function
first finds its callers, then reaches the function's own module, then the files
that import that module — yielding both the function-level and file-level blast
radius in one traversal.

Impact = every node that can reach the target by following edges forward, i.e.
every *predecessor* in the transitive sense. We compute it with a BFS over
reversed edges and keep the hop distance, so callers can distinguish "calls the
target directly" (distance 1) from "three hops away".
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import networkx as nx

from .call_graph import build_call_graph
from .import_graph import build_import_graph, module_name_for


def build_unified_graph(repo_path: str) -> nx.DiGraph:
    """Merge the import graph and call graph into one depends-on graph.

    Nodes carry a ``kind`` attribute: ``module`` for files, ``function`` /
    ``method`` for callables. Edges carry a ``kind``: ``import``, ``call``, or
    ``contains``.
    """
    repo_path = os.path.abspath(repo_path)
    import_graph = build_import_graph(repo_path)
    call_graph = build_call_graph(repo_path)

    g = nx.DiGraph()

    for module, data in import_graph.nodes(data=True):
        g.add_node(module, kind="module", **data)
    for u, v in import_graph.edges():
        g.add_edge(u, v, kind="import")

    for fn, data in call_graph.nodes(data=True):
        g.add_node(fn, **data)  # kind is already function/method
    for u, v, data in call_graph.edges(data=True):
        # Preserve call metadata (e.g. linenos) alongside the kind tag.
        g.add_edge(u, v, kind="call", **data)

    # Bridge granularities: a module depends on each function it defines.
    for fn, data in call_graph.nodes(data=True):
        module = data.get("module")
        if module is not None:
            if not g.has_node(module):
                g.add_node(module, kind="module")
            g.add_edge(module, fn, kind="contains")

    return g


def resolve_target(graph: nx.DiGraph, target: str, repo_path: str) -> str | None:
    """Map a user-supplied target string to a node id in the graph.

    Accepts, in order of preference:
    - an exact node id (``app.auth`` or ``app.auth:login``)
    - a file path (``app/auth.py`` or an absolute path)
    - a dotted callable (``app.auth.login`` or ``app.models.User.check``),
      resolved by trying each dot as the module/qualname boundary.
    """
    if graph.has_node(target):
        return target

    if target.endswith(".py") or os.sep in target or "/" in target:
        candidate = os.path.abspath(os.path.join(repo_path, target)) \
            if not os.path.isabs(target) else target
        if os.path.exists(candidate):
            module = module_name_for(candidate, os.path.abspath(repo_path))
            if graph.has_node(module):
                return module

    if ":" not in target and "." in target:
        parts = target.split(".")
        # Try the longest module prefix first so app.models.User.check maps to
        # module app.models + qualname User.check rather than a wrong split.
        for i in range(len(parts) - 1, 0, -1):
            module = ".".join(parts[:i])
            qual = ".".join(parts[i:])
            candidate = f"{module}:{qual}"
            if graph.has_node(candidate):
                return candidate

    return None


@dataclass(frozen=True)
class Affected:
    """One node in the target's blast radius."""

    node: str
    kind: str      # module | function | method
    distance: int  # hops from the target (1 = direct dependent)
    via: str       # edge kind connecting it toward the target: call/import/contains
    toward: str = ""  # the next node on the path toward the target (BFS parent)


def find_impact(graph: nx.DiGraph, target_node: str) -> list[Affected]:
    """Reverse-BFS from ``target_node``; return everything that depends on it.

    Results are sorted by distance (closest first), then by node id, so output
    is deterministic and the most-directly-affected items lead.
    """
    distance = {target_node: 0}
    via: dict[str, str] = {}
    toward: dict[str, str] = {}
    queue = deque([target_node])

    while queue:
        node = queue.popleft()
        for pred in graph.predecessors(node):
            if pred not in distance:
                distance[pred] = distance[node] + 1
                via[pred] = graph.edges[pred, node].get("kind", "")
                toward[pred] = node
                queue.append(pred)

    results = [
        Affected(
            node=n,
            kind=graph.nodes[n].get("kind", "unknown"),
            distance=distance[n],
            via=via[n],
            toward=toward[n],
        )
        for n in distance
        if n != target_node
    ]
    results.sort(key=lambda a: (a.distance, a.node))
    return results


def analyze(repo_path: str, target: str):
    """End-to-end convenience: build the graph, resolve the target, run impact.

    Returns ``(resolved_node, affected_list)``. ``resolved_node`` is ``None`` if
    the target string didn't match anything in the repo.
    """
    graph = build_unified_graph(repo_path)
    resolved = resolve_target(graph, target, repo_path)
    if resolved is None:
        return None, []
    return resolved, find_impact(graph, resolved)
