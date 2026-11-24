"""Repository-wide insights: overview counts, risk hotspots and import cycles.

**Hotspots** are the functions whose change would ripple furthest: ranked by the
number of production functions/methods that transitively *call* them (structural
links such as "module defines function" are not counted). Counts are exact — computed for every node at once by propagating ancestor bitsets
through the graph's condensation (a DAG of strongly connected components) in
topological order, instead of running one traversal per function.

**Import cycles** are strongly connected components of the load-time import
graph — imports under ``if TYPE_CHECKING:`` or inside functions are ignored,
since they don't run when a module is first imported. One concrete cycle is
reported per component.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import networkx as nx

from .impact import CALLABLE_KINDS

EXACT_LIMIT = 20000   # nodes; above this, fall back to per-candidate traversal


@dataclass
class Hotspot:
    node: str
    kind: str
    dependents: int     # production functions/methods that transitively depend on it
    fan_in: int         # direct callers
    fan_out: int        # direct callees
    file: str | None
    line: int | None


def overview(graph: nx.DiGraph) -> dict:
    kinds = Counter(d.get("kind") for _, d in graph.nodes(data=True))
    edges = Counter(d.get("kind") for *_, d in graph.edges(data=True))
    return {
        "files": kinds.get("module", 0),
        "unparsed_files": len(graph.graph.get("skipped", [])),
        "test_files": sum(1 for _, d in graph.nodes(data=True)
                          if d.get("kind") == "module" and d.get("is_test")),
        "classes": kinds.get("class", 0),
        "functions": kinds.get("function", 0),
        "methods": kinds.get("method", 0),
        "edges": dict(sorted(edges.items())),
    }


def _call_degree(graph, node, incoming: bool) -> int:
    edges = graph.in_edges(node, data=True) if incoming else graph.out_edges(node, data=True)
    return sum(1 for *_, d in edges if d.get("kind") == "call")


def _dependent_counts(graph: nx.DiGraph, counted: set) -> dict:
    """node -> number of ``counted`` nodes that can reach it (excluding itself)."""
    if graph.number_of_nodes() <= EXACT_LIMIT:
        cond = nx.condensation(graph)
        bit = {n: 1 << i for i, n in enumerate(sorted(counted))}
        members = cond.graph["mapping"]          # node -> component id
        comp_bits = Counter()
        for n, c in members.items():
            comp_bits[c] |= bit.get(n, 0)
        ancestors: dict = {}
        for c in nx.topological_sort(cond):      # predecessors before successors
            acc = 0
            for p in cond.predecessors(c):
                acc |= ancestors[p] | comp_bits[p]
            ancestors[c] = acc
        result = {}
        for n, c in members.items():
            reach = ancestors[c] | comp_bits[c]
            result[n] = reach.bit_count() - (1 if n in bit else 0)
        return result
    # Very large graphs: exact counts for the most-called candidates only.
    candidates = sorted((n for n, d in graph.nodes(data=True) if d.get("kind") in CALLABLE_KINDS),
                        key=lambda n: -graph.in_degree(n))[:500]
    return {n: len(nx.ancestors(graph, n) & counted) for n in candidates}


def hotspots(graph: nx.DiGraph, top: int = 15, include_tests: bool = False) -> list[Hotspot]:
    production = {n for n, d in graph.nodes(data=True)
                  if d.get("kind") in CALLABLE_KINDS and not d.get("is_test")}
    calls = nx.DiGraph()
    calls.add_nodes_from(graph.nodes(data=True))
    calls.add_edges_from((u, v) for u, v, d in graph.edges(data=True) if d.get("kind") == "call")
    counts = _dependent_counts(calls, production)
    rows = []
    for node, count in counts.items():
        data = graph.nodes[node]
        if data.get("kind") not in CALLABLE_KINDS or count == 0:
            continue
        if data.get("is_test") and not include_tests:
            continue
        rows.append(Hotspot(node, data["kind"], count, _call_degree(graph, node, True),
                            _call_degree(graph, node, False), data.get("rel_path"),
                            data.get("lineno")))
    rows.sort(key=lambda h: (-h.dependents, -h.fan_in, h.node))
    return rows[:top]


def import_cycles(graph: nx.DiGraph, limit: int = 20) -> list[list[str]]:
    """One representative cycle per strongly connected group of modules."""
    imports = nx.DiGraph((u, v) for u, v, d in graph.edges(data=True)
                         if d.get("kind") == "import" and d.get("load_time", True))
    cycles = []
    for component in nx.strongly_connected_components(imports):
        if len(component) < 2:
            continue
        sub = imports.subgraph(component)
        cycle = [u for u, _ in nx.find_cycle(sub, source=min(component))]
        cycles.append(cycle)
    cycles.sort(key=lambda c: (len(c), c))
    return cycles[:limit]
