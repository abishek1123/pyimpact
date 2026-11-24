"""AI explanation layer.

Grounding principle — the reason this project isn't another LLM wrapper: **the
graph is ground truth.** This module never asks the model *what* is affected; the
deterministic reverse-reachability analysis already decided that. Each affected
component is handed to the model together with the exact edge, line number,
source line and severity that connect it to the change, and the model is asked
for one thing only: a readable sentence explaining *why*.

The model narrates facts it is given. It cannot add an impact the graph didn't
find, because it is never asked to reason about impact — only to phrase a
relationship that already exists. The model call is injectable
(``generate_fn``) so evidence gathering and prompting are testable offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import networkx as nx

from . import llm
from .impact import Affected, build_unified_graph, find_impact, resolve_target

DEFAULT_MODEL = llm.DEFAULT_MODEL


@dataclass
class Evidence:
    """Deterministic, model-free facts about one affected component."""

    affected: Affected
    relationship: str          # e.g. "calls app.core:helper (line 12)"
    snippet: str = ""          # the source line at that site, if available


@dataclass
class Explanation:
    affected: Affected
    why: str


@dataclass
class ImpactExplanation:
    target: str
    items: list[Explanation] = field(default_factory=list)

    def as_dict(self) -> dict[str, str]:
        return {item.affected.node: item.why for item in self.items if item.why}


# -- evidence gathering (deterministic) -------------------------------------------

def _node_file(graph: nx.DiGraph, node: str) -> str | None:
    data = graph.nodes[node]
    if data.get("path"):
        return data["path"]
    module = data.get("module")
    if module and graph.has_node(module):
        return graph.nodes[module].get("path")
    return None


def _source_line(path: str | None, lineno: int) -> str:
    if not path or lineno <= 0:
        return ""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for number, line in enumerate(f, 1):
                if number == lineno:
                    return line.strip()[:200]
    except OSError:
        pass
    return ""


_PHRASES = {
    ("call", "direct"): "calls {t}",
    ("call", "reference"): "references {t} as a value (callback, decorator or handler)",
    ("call", "virtual"): "calls a method that {t} overrides, so dynamic dispatch may reach it",
    ("instantiate", None): "constructs {t}",
    ("uses", None): "uses the class {t}",
    ("inherits", None): "subclasses {t}",
    ("member", None): "is the class that defines {t}",
    ("contains", None): "is the module that defines {t}",
    ("import", None): "imports the module {t}",
}


def gather_evidence(graph: nx.DiGraph, a: Affected) -> Evidence:
    """Turn one ``Affected`` into concrete facts, using the edge that put it in
    the blast radius (``a.node -> a.toward``) and its recorded line numbers."""
    edge = graph.edges[a.node, a.toward] if graph.has_edge(a.node, a.toward) else {}
    strength = edge.get("strength") if a.via == "call" else None
    template = _PHRASES.get((a.via, strength)) or _PHRASES.get((a.via, None)) \
        or "depends on {t}"
    relationship = template.format(t=a.toward)
    linenos = edge.get("linenos") or []
    snippet = ""
    if linenos:
        relationship += f" (line {linenos[0]})"
        snippet = _source_line(_node_file(graph, a.node), linenos[0])
    return Evidence(affected=a, relationship=relationship, snippet=snippet)


# -- prompt ---------------------------------------------------------------------------

_PROMPT_HEADER = """\
You are narrating the results of a DETERMINISTIC static-analysis impact report \
for a Python codebase. A change is being made to:

    TARGET: {target}

A static call/import graph has ALREADY determined which components are affected \
and exactly why (the connecting edge, line number and severity are given). Your \
job is NOT to decide what is affected — that is already decided and correct.

For each item, write ONE concise sentence (max ~25 words) explaining why a \
change to the target could affect it, based ONLY on the facts given. Do not \
introduce impacts that are not listed. Do not hedge or add caveats. Refer to \
components by their short name.

Return ONLY a JSON array of objects with keys "index" (int) and "why" (string), \
one per item, in the same order. No prose outside the JSON.

ITEMS:
"""


def build_prompt(target: str, evidence: list[Evidence]) -> str:
    lines = [_PROMPT_HEADER.format(target=target)]
    for i, ev in enumerate(evidence):
        a = ev.affected
        parts = [
            f"[{i}] {a.node}",
            f"kind={a.kind}",
            f"distance={a.distance} hop(s)",
            f"severity={a.severity}",
            f"relationship: this component {ev.relationship}",
        ]
        if ev.snippet:
            parts.append(f"source: `{ev.snippet}`")
        lines.append("  " + " | ".join(parts))
    return "\n".join(lines)


def gemini_generate(prompt: str, *, api_key: str | None = None,
                    model: str | None = None, json_mode: bool = True) -> str:
    """Backwards-compatible alias for :func:`pyimpact.llm.generate`."""
    return llm.generate(prompt, api_key=api_key, model=model, json_mode=json_mode)


def _parse_response(raw: str, count: int) -> dict[int, str]:
    """Parse the model's JSON into ``{index: why}``, tolerating minor drift."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("["):] if "[" in text else text
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    if isinstance(data, dict):
        data = data.get("items") or data.get("explanations") or []
    result: dict[int, str] = {}
    if isinstance(data, list):
        for pos, item in enumerate(data):
            if isinstance(item, dict):
                idx, why = item.get("index", pos), item.get("why", "")
            else:
                idx, why = pos, str(item)
            if isinstance(idx, int) and 0 <= idx < count and str(why).strip():
                result[idx] = str(why).strip()
    return result


def explain_affected(graph: nx.DiGraph, target_label: str, affected, *,
                     generate_fn=None, model: str | None = None,
                     api_key: str | None = None, limit: int | None = None
                     ) -> ImpactExplanation:
    """Narrate an already-computed impact list. ``limit`` caps how many items
    (highest severity first) are sent, which also bounds token cost."""
    from .impact import SEVERITY_RANK

    ranked = sorted(affected, key=lambda a: (SEVERITY_RANK[a.severity], a.distance, a.node))
    if limit is not None:
        ranked = ranked[:limit]
    evidence = [gather_evidence(graph, a) for a in ranked]
    if not evidence:
        return ImpactExplanation(target=target_label, items=[])
    if generate_fn is None:
        def generate_fn(prompt: str) -> str:
            return llm.generate(prompt, api_key=api_key, model=model, json_mode=True)
    whys = _parse_response(generate_fn(build_prompt(target_label, evidence)), len(evidence))
    return ImpactExplanation(
        target=target_label,
        items=[Explanation(ev.affected, whys.get(i, "")) for i, ev in enumerate(evidence)],
    )


def explain_impact(repo_path, target: str, *, generate_fn=None, model: str | None = None,
                   api_key: str | None = None, limit: int | None = None,
                   exclude=()) -> ImpactExplanation | None:
    """Build the graph, run the impact query for ``target`` and narrate it.
    Returns ``None`` if the target can't be resolved."""
    graph = build_unified_graph(repo_path, exclude)
    resolved = resolve_target(graph, target)
    if resolved is None:
        return None
    return explain_affected(graph, resolved, find_impact(graph, resolved),
                            generate_fn=generate_fn, model=model, api_key=api_key,
                            limit=limit)
