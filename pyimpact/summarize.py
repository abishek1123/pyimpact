"""Graph-grounded code summarization.

Deliberately NOT "summarize this function" — that is commoditized. A component
is summarized *using its position in the dependency graph* as authoritative
context: who depends on it, what it depends on, and how large its blast radius
is. The result reads like:

    "Validates a login request. Called by route_login and route_admin, and calls
     hash_password(). 8 components depend on it — a moderate blast radius."

The graph facts are extracted deterministically, including the blast-radius
label, so the model never estimates risk itself; it only writes the prose and is
told not to invent relationships. The model call is injectable for testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from . import llm
from .impact import CALLABLE_KINDS, build_unified_graph, find_impact, resolve_target

MAX_SOURCE_CHARS = 4000  # keep prompts bounded on large functions


@dataclass
class GraphContext:
    """Deterministic facts about a node's position in the dependency graph."""

    node: str
    kind: str
    callers: list[str] = field(default_factory=list)   # who depends on it directly
    callees: list[str] = field(default_factory=list)   # what it depends on directly
    dependents_total: int = 0                           # size of the blast radius
    dependents_functions: int = 0                       # callable dependents
    high_risk: int = 0                                  # high-severity dependents
    tests: int = 0                                      # test functions/files reached
    blast_label: str = "none"                           # none | small | moderate | large
    location: str = ""                                  # file:line
    signature: str = ""
    source: str = ""


@dataclass
class Summary:
    node: str
    context: GraphContext
    text: str


def _node_source(graph: nx.DiGraph, node: str) -> str:
    data = graph.nodes[node]
    path = data.get("path")
    start, end = data.get("start_lineno") or data.get("lineno"), data.get("end_lineno")
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    if start:
        lines = lines[start - 1:end or start]
    return "".join(lines)[:MAX_SOURCE_CHARS]


def _neighbours(graph, node, kinds, incoming: bool) -> list[str]:
    if incoming:
        pairs = ((u, (u, node)) for u in graph.predecessors(node))
    else:
        pairs = ((v, (node, v)) for v in graph.successors(node))
    return sorted(n for n, edge in pairs if graph.edges[edge].get("kind") in kinds)


def blast_label(callable_dependents: int) -> str:
    if callable_dependents == 0:
        return "none"
    if callable_dependents <= 3:
        return "small"
    if callable_dependents <= 15:
        return "moderate"
    return "large"


def gather_context(graph: nx.DiGraph, node: str) -> GraphContext:
    """Collect the deterministic graph position of ``node``."""
    data = graph.nodes[node]
    kind = data.get("kind", "unknown")
    if kind == "module":
        callers = _neighbours(graph, node, {"import"}, incoming=True)
        callees = _neighbours(graph, node, {"import"}, incoming=False)
    elif kind == "class":
        callers = _neighbours(graph, node, {"instantiate", "uses", "inherits"}, incoming=True)
        callees = _neighbours(graph, node, {"inherits", "member"}, incoming=False)
    else:
        callers = _neighbours(graph, node, {"call"}, incoming=True)
        callees = _neighbours(graph, node, {"call", "instantiate"}, incoming=False)

    affected = find_impact(graph, node)
    callable_dependents = sum(1 for a in affected if a.kind in CALLABLE_KINDS)
    location = data.get("rel_path") or ""
    if data.get("lineno"):
        location += f":{data['lineno']}"
    return GraphContext(
        node=node,
        kind=kind,
        callers=callers,
        callees=callees,
        dependents_total=len(affected),
        dependents_functions=callable_dependents,
        high_risk=sum(1 for a in affected if a.severity == "high"),
        tests=sum(1 for a in affected if a.is_test and a.kind in CALLABLE_KINDS),
        blast_label=blast_label(callable_dependents),
        location=location,
        signature=data.get("signature", ""),
        source=_node_source(graph, node) if kind != "module" else "",
    )


_PROMPT = """\
You are summarizing a Python {kind} for a developer, using its position in the \
codebase's dependency graph as authoritative context. The graph facts below are \
correct and were derived by static analysis — do NOT contradict them and do NOT \
invent callers or callees that are not listed.

COMPONENT: {node}  ({location})

GRAPH POSITION (ground truth):
- Depended on directly by ({n_callers}): {callers}
- Depends directly on ({n_callees}): {callees}
- Blast radius: {dependents_total} components depend on it, {dependents_functions} \
of them functions/methods ({high_risk} high-severity, {tests} tests). \
Classified as: {blast_label}.

SOURCE:
```python
{source}
```

Write 2-4 sentences: first what the component does (from its source), then its \
role in the codebase using the graph facts — who depends on it, what it depends \
on, and its blast radius exactly as classified above. Be concrete and concise. \
Output plain prose only.
"""


def _listing(names: list[str], limit: int = 20) -> str:
    if not names:
        return "nothing"
    shown = ", ".join(names[:limit])
    return shown + (f", and {len(names) - limit} more" if len(names) > limit else "")


def build_summary_prompt(context: GraphContext) -> str:
    return _PROMPT.format(
        kind=context.kind,
        node=context.node,
        location=context.location or "unknown location",
        n_callers=len(context.callers),
        callers=_listing(context.callers),
        n_callees=len(context.callees),
        callees=_listing(context.callees),
        dependents_total=context.dependents_total,
        dependents_functions=context.dependents_functions,
        high_risk=context.high_risk,
        tests=context.tests,
        blast_label=context.blast_label.upper(),
        source=context.source or "(module: see its imports above)",
    )


def summarize_node(graph: nx.DiGraph, node: str, *, generate_fn=None,
                   model: str | None = None, api_key: str | None = None) -> Summary:
    context = gather_context(graph, node)
    if generate_fn is None:
        def generate_fn(prompt: str) -> str:
            return llm.generate(prompt, api_key=api_key, model=model)
    text = (generate_fn(build_summary_prompt(context)) or "").strip()
    return Summary(node=node, context=context, text=text)


def summarize_target(repo_path, target: str, *, generate_fn=None,
                     model: str | None = None, api_key: str | None = None,
                     exclude=()) -> Summary | None:
    """Summarize ``target`` grounded in its graph position (``None`` if the
    target can't be resolved)."""
    graph = build_unified_graph(repo_path, exclude)
    resolved = resolve_target(graph, target)
    if resolved is None:
        return None
    return summarize_node(graph, resolved, generate_fn=generate_fn, model=model,
                          api_key=api_key)
