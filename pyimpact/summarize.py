"""Graph-grounded code summarization (Tier 2).

This is deliberately NOT "summarize this function" — that's commoditized and
against the project's positioning. Instead, a component is summarized *using its
position in the dependency graph* as authoritative context: who calls it, what
it calls, and how big its blast radius is. The result reads like:

    "Validates a login request. Called by 3 places (auth_routes, admin_panel,
     test_auth) and calls hash_password() and db.query(). 8 components depend on
     it — moderate blast radius."

The graph facts (callers, callees, blast radius) are ground truth extracted
deterministically. The model only turns the function's source plus those facts
into prose; it is instructed not to invent relationships beyond the ones given.

As with the explainer, the Gemini call is injectable so everything except the
model wording is unit-testable with no key/network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from .explain import gemini_generate
from .impact import (
    build_unified_graph,
    find_impact,
    resolve_target,
)

MAX_SOURCE_CHARS = 4000  # keep prompts bounded on large functions


@dataclass
class GraphContext:
    """Deterministic facts about a node's position in the dependency graph."""

    node: str
    kind: str
    callers: list[str] = field(default_factory=list)   # who calls / imports this
    callees: list[str] = field(default_factory=list)   # what this calls / imports
    dependents_total: int = 0                           # size of the blast radius
    dependents_functions: int = 0                       # function/method dependents
    source: str = ""                                    # the component's own source


@dataclass
class Summary:
    node: str
    context: GraphContext
    text: str


def _node_file(graph: nx.DiGraph, node: str) -> str | None:
    data = graph.nodes[node]
    if "path" in data:
        return data["path"]
    module = data.get("module")
    if module and graph.has_node(module):
        return graph.nodes[module].get("path")
    return None


def _node_source(graph: nx.DiGraph, node: str) -> str:
    """Read a function/method's source using its recorded line span."""
    data = graph.nodes[node]
    lineno = data.get("lineno")
    end = data.get("end_lineno")
    path = _node_file(graph, node)
    if not (path and lineno):
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return ""
    end = end or lineno
    snippet = "".join(lines[lineno - 1:end])
    return snippet[:MAX_SOURCE_CHARS]


def _incoming(graph: nx.DiGraph, node: str, kind: str) -> list[str]:
    """Predecessors connected to ``node`` by an edge of ``kind`` (who depends on it)."""
    return sorted(u for u in graph.predecessors(node)
                  if graph.edges[u, node].get("kind") == kind)


def _outgoing(graph: nx.DiGraph, node: str, kind: str) -> list[str]:
    """Successors ``node`` connects to by an edge of ``kind`` (what it depends on)."""
    return sorted(v for v in graph.successors(node)
                  if graph.edges[node, v].get("kind") == kind)


def gather_context(graph: nx.DiGraph, node: str) -> GraphContext:
    """Collect the deterministic graph position of ``node``."""
    kind = graph.nodes[node].get("kind", "unknown")
    # Functions relate via call edges; modules via import edges.
    edge_kind = "import" if kind == "module" else "call"
    callers = _incoming(graph, node, edge_kind)
    callees = _outgoing(graph, node, edge_kind)

    affected = find_impact(graph, node)
    dependents_functions = sum(
        1 for a in affected if a.kind in ("function", "method"))

    return GraphContext(
        node=node,
        kind=kind,
        callers=callers,
        callees=callees,
        dependents_total=len(affected),
        dependents_functions=dependents_functions,
        source=_node_source(graph, node),
    )


_PROMPT = """\
You are summarizing a Python {kind} for a developer, using its position in the \
codebase's dependency graph as authoritative context. The graph facts below are \
correct and were derived by static analysis — do NOT contradict them or invent \
callers/callees that are not listed.

COMPONENT: {node}

GRAPH POSITION (ground truth):
- Called by ({n_callers}): {callers}
- Calls ({n_callees}): {callees}
- Total components that depend on it (blast radius): {dependents_total} \
({dependents_functions} of them functions/methods)

SOURCE:
```python
{source}
```

Write 2-4 sentences: first what the component does (from its source), then its \
role in the codebase using the graph facts (who depends on it, what it depends \
on, and whether its blast radius is small/moderate/large). Be concrete and \
concise. Output plain prose only.
"""


def build_summary_prompt(context: GraphContext) -> str:
    return _PROMPT.format(
        kind=context.kind,
        node=context.node,
        n_callers=len(context.callers),
        callers=", ".join(context.callers) or "nothing",
        n_callees=len(context.callees),
        callees=", ".join(context.callees) or "nothing",
        dependents_total=context.dependents_total,
        dependents_functions=context.dependents_functions,
        source=context.source or "(source unavailable)",
    )


def summarize_target(repo_path: str, target: str, *, generate_fn=None,
                     model: str | None = None,
                     api_key: str | None = None) -> Summary | None:
    """Summarize ``target`` grounded in its graph position.

    ``generate_fn(prompt) -> str`` is injectable; when omitted a Gemini-backed
    (plain-text) function is used. Returns ``None`` if the target can't resolve.
    """
    graph = build_unified_graph(repo_path)
    resolved = resolve_target(graph, target, repo_path)
    if resolved is None:
        return None

    context = gather_context(graph, resolved)

    if generate_fn is None:
        def generate_fn(prompt: str) -> str:
            return gemini_generate(
                prompt, api_key=api_key, model=model, json_mode=False)

    text = generate_fn(build_summary_prompt(context)).strip()
    return Summary(node=resolved, context=context, text=text)
