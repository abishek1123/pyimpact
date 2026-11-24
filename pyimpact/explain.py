"""AI explanation layer (Tier 2).

Grounding principle — the whole reason this project isn't just another
LLM-wrapper: **the graph is ground truth.** This module never asks the model
what is affected; the deterministic reverse-reachability analysis in
``impact.py`` already decided that. Here we take each affected component together
with the exact edge and line number that connect it toward the target, and ask
the model for one thing only: a human-readable sentence explaining *why*.

So the model narrates facts it is handed. It cannot introduce an impact the
graph didn't find, because it is never asked to reason about impact — only to
phrase a relationship that already exists.

The Gemini call is isolated behind an injectable ``generate_fn`` so the evidence
gathering and prompt building are fully unit-testable with no network or API key.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import networkx as nx

from .impact import (
    Affected,
    build_unified_graph,
    find_impact,
    resolve_target,
)

DEFAULT_MODEL = "gemini-2.0-flash"


@dataclass
class Evidence:
    """Deterministic, model-free facts about one affected component."""

    affected: Affected
    relationship: str          # e.g. "calls requests.sessions:Session.request (line 602)"
    snippet: str = ""          # the source line at the call site, if available


@dataclass
class Explanation:
    """One affected component paired with the model's narration of why."""

    affected: Affected
    why: str


@dataclass
class ImpactExplanation:
    target: str
    items: list[Explanation] = field(default_factory=list)


# --- evidence gathering (deterministic) ------------------------------------

def _node_file(graph: nx.DiGraph, node: str) -> str | None:
    """Return the source file backing a node (module node, or a function's module)."""
    data = graph.nodes[node]
    if "path" in data:
        return data["path"]
    module = data.get("module")
    if module and graph.has_node(module):
        return graph.nodes[module].get("path")
    return None


def _source_line(path: str | None, lineno: int) -> str:
    if not path or lineno <= 0:
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        if lineno <= len(lines):
            return lines[lineno - 1].strip()
    except OSError:
        pass
    return ""


def gather_evidence(graph: nx.DiGraph, a: Affected) -> Evidence:
    """Turn one ``Affected`` into concrete, human/model-readable facts.

    Uses the edge ``a.node -> a.toward`` (the step that put ``a`` in the blast
    radius) plus any recorded call-site line numbers.
    """
    edge = graph.edges[a.node, a.toward] if graph.has_edge(a.node, a.toward) else {}
    if a.via == "call":
        linenos = edge.get("linenos") or []
        where = f" (line {linenos[0]})" if linenos else ""
        relationship = f"calls {a.toward}{where}"
        snippet = _source_line(_node_file(graph, a.node), linenos[0]) if linenos else ""
    elif a.via == "import":
        relationship = f"imports the module {a.toward}"
        snippet = ""
    elif a.via == "contains":
        # contains edge is module -> function: a.node is the module.
        relationship = f"is the module that defines {a.toward}"
        snippet = ""
    else:
        relationship = f"depends on {a.toward}"
        snippet = ""
    return Evidence(affected=a, relationship=relationship, snippet=snippet)


# --- prompt building -------------------------------------------------------

_PROMPT_HEADER = """\
You are narrating the results of a DETERMINISTIC static-analysis impact report \
for a Python codebase. A change is being made to the target:

    TARGET: {target}

A static call/import graph has ALREADY determined which components are affected \
and exactly why (the connecting edge and line number are given). Your job is \
NOT to decide what is affected — that is already decided and correct.

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
            f"distance={a.distance} hop(s) from target",
            f"relationship: this component {ev.relationship}",
        ]
        if ev.snippet:
            parts.append(f"source: `{ev.snippet}`")
        lines.append("  " + " | ".join(parts))
    return "\n".join(lines)


# --- Gemini backend --------------------------------------------------------

def gemini_generate(prompt: str, *, api_key: str | None = None,
                    model: str | None = None, json_mode: bool = True) -> str:
    """Call Gemini and return the raw text response.

    ``json_mode`` requests a JSON response (used by the explainer); set it False
    for free-text (used by the summarizer). Imported lazily so the rest of the
    module works without google-genai installed. Raises RuntimeError with
    actionable guidance if no key is set.
    """
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY), "
            "or pass api_key=..."
        )
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "google-genai is not installed. Install the AI extra: "
            "pip install 'pyimpact[ai]'"
        ) from exc

    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        response_mime_type="application/json" if json_mode else "text/plain"
    )
    resp = client.models.generate_content(
        model=model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        contents=prompt,
        config=config,
    )
    return resp.text or ""


# --- response parsing + orchestration --------------------------------------

def _parse_response(raw: str, count: int) -> dict[int, str]:
    """Parse the model's JSON into ``{index: why}``, tolerating minor drift."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}

    result: dict[int, str] = {}
    if isinstance(data, list):
        for pos, item in enumerate(data):
            if isinstance(item, dict):
                idx = item.get("index", pos)
                why = item.get("why", "")
            else:
                idx, why = pos, str(item)
            if isinstance(idx, int) and 0 <= idx < count:
                result[idx] = str(why).strip()
    return result


def explain_impact(repo_path: str, target: str, *, generate_fn=None,
                   model: str | None = None, api_key: str | None = None,
                   limit: int | None = None) -> ImpactExplanation | None:
    """Run the impact query, then narrate each affected component with the model.

    ``generate_fn(prompt) -> str`` is injectable; when omitted, a Gemini-backed
    function is used (needs an API key). Returns ``None`` if the target can't be
    resolved. ``limit`` caps how many (closest-first) items are explained, which
    also bounds token cost.
    """
    graph = build_unified_graph(repo_path)
    resolved = resolve_target(graph, target, repo_path)
    if resolved is None:
        return None

    affected = find_impact(graph, resolved)
    if limit is not None:
        affected = affected[:limit]

    evidence = [gather_evidence(graph, a) for a in affected]
    if not evidence:
        return ImpactExplanation(target=resolved, items=[])

    if generate_fn is None:
        def generate_fn(prompt: str) -> str:
            return gemini_generate(prompt, api_key=api_key, model=model)

    prompt = build_prompt(resolved, evidence)
    whys = _parse_response(generate_fn(prompt), len(evidence))

    items = [
        Explanation(affected=ev.affected, why=whys.get(i, ""))
        for i, ev in enumerate(evidence)
    ]
    return ImpactExplanation(target=resolved, items=items)
