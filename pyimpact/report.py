"""Turn impact results into output: a JSON-ready payload, plain text, Markdown.

Every command builds one payload dict (``build_payload``) and a renderer turns it
into text for the terminal, Markdown for pull-request comments, or JSON for
tooling — so all three formats always agree.
"""

from __future__ import annotations

import json

from . import __version__
from .impact import SEVERITY_RANK, affected_tests

MARKER = "<!-- pyimpact-report -->"

_REASON = {
    ("call", "direct"): "calls",
    ("call", "reference"): "references",
    ("call", "virtual"): "may dispatch to",
    ("call", None): "calls",
    ("instantiate", None): "constructs",
    ("uses", None): "uses",
    ("inherits", None): "subclasses",
    ("member", None): "defines",
    ("contains", None): "defines",
    ("import", None): "imports",
}
_SEVERITY_ICON = {"high": ":red_circle:", "medium": ":orange_circle:", "low": ":white_circle:"}
_CHANGE_MARK = {"modified": "~", "added": "+", "removed": "-"}


def reason(graph, node: str, via: str, toward: str) -> str:
    """Short human phrase for why ``node`` is affected: 'calls compute'."""
    strength = None
    if via == "call" and graph is not None and graph.has_edge(node, toward):
        strength = graph.edges[node, toward].get("strength", "direct")
    verb = _REASON.get((via, strength)) or _REASON.get((via, None)) or via
    if strength is None and via == "call":
        verb = "calls"
    return f"{verb} {short_name(toward)}"


def short_name(node: str) -> str:
    return node.split(":", 1)[1] if ":" in node else node


def _location(graph, node: str) -> tuple[str | None, int | None]:
    if graph is None or not graph.has_node(node):
        return None, None
    data = graph.nodes[node]
    return data.get("rel_path"), data.get("lineno")


def _affected_dict(graph, a, explanations=None) -> dict:
    file, line = _location(graph, a.node)
    item = {
        "node": a.node, "kind": a.kind, "file": file, "line": line,
        "distance": a.distance, "severity": a.severity, "score": a.score,
        "via": a.via, "toward": a.toward, "is_test": a.is_test,
        "reason": reason(graph, a.node, a.via, a.toward),
    }
    if explanations and explanations.get(a.node):
        item["explanation"] = explanations[a.node]
    return item


def _symbol(item) -> dict:
    """Normalise a target/removed entry (dict or ``ChangedSymbol``) to a dict."""
    if isinstance(item, dict):
        return item
    return {"node": item.node, "kind": item.kind, "change": item.change,
            "file": item.rel_path, "line": item.line}


def build_payload(graph, *, mode: str, targets: list, affected,
                  base: str | None = None, files=None, removed=None,
                  removed_affected=None, explanations=None,
                  min_severity: str | None = None) -> dict:
    """Assemble the canonical report dict for ``analyze`` / ``diff``."""
    if min_severity:
        cutoff = SEVERITY_RANK[min_severity]
        affected = [a for a in affected if SEVERITY_RANK[a.severity] <= cutoff]
        removed_affected = [a for a in (removed_affected or [])
                            if SEVERITY_RANK[a.severity] <= cutoff]
    removed_affected = removed_affected or []
    targets = [_symbol(t) for t in targets]
    removed = [_symbol(r) for r in (removed or [])]
    ids, test_files = affected_tests(graph, [*affected, *removed_affected])
    counts = {s: sum(1 for a in affected if a.severity == s) for s in SEVERITY_RANK}
    return {
        "tool": "pyimpact",
        "version": __version__,
        "mode": mode,
        "root": graph.graph.get("root"),
        "base": base,
        "files_changed": list(files or []),
        "targets": targets,
        "summary": {
            "targets": len(targets),
            "affected": len(affected),
            **counts,
            "tests": len(ids),
            "removed": len(removed),
            "breaking": len(removed_affected),
        },
        "affected": [_affected_dict(graph, a, explanations) for a in affected],
        "removed": removed,
        "breaking": [_affected_dict(graph, a) for a in removed_affected],
        "tests": {"ids": ids, "files": test_files},
        "skipped_files": [{"file": f, "error": e} for f, e in graph.graph.get("skipped", [])],
    }


def target_dict(graph, node: str, change: str | None = None) -> dict:
    file, line = _location(graph, node)
    kind = graph.nodes[node].get("kind") if graph.has_node(node) else "module"
    return {"node": node, "kind": kind, "change": change, "file": file, "line": line}


def to_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def pytest_command(tests: dict, limit: int = 25) -> str | None:
    ids, files = tests["ids"], tests["files"]
    if ids and len(ids) <= limit:
        return "pytest " + " ".join(ids)
    if files:
        return "pytest " + " ".join(files[:limit]) + (" ..." if len(files) > limit else "")
    return None


# -- plain text ------------------------------------------------------------------

def _plain(text, **_):
    return text


def render_text(payload: dict, style=_plain, max_items: int = 100) -> str:
    """Terminal report (ASCII only, so it survives any console encoding)."""
    out: list[str] = []
    s = payload["summary"]
    sev_color = {"high": "red", "medium": "yellow", "low": "white"}

    if payload["mode"] == "diff":
        out.append(style(f"PyImpact: {payload['base']}", bold=True))
        if not payload["targets"] and not payload["removed"]:
            out.append("No Python changes found.")
            return "\n".join(out)
        out.append(f"Changed symbols ({len(payload['targets']) + len(payload['removed'])}):")
        for t in [*payload["targets"], *payload["removed"]]:
            mark = _CHANGE_MARK.get(t.get("change"), "*")
            loc = f"{t['file']}:{t['line']}" if t.get("line") else (t.get("file") or "")
            out.append(f"  {mark} {t['node'] or t['file']:<50} {loc}  [{t.get('change')}]")
        out.append("")
    else:
        for t in payload["targets"]:
            loc = f" ({t['file']}:{t['line']})" if t.get("line") else \
                (f" ({t['file']})" if t.get("file") else "")
            out.append(f"Target: {style(t['node'], bold=True)}{loc}")

    if not payload["affected"] and not payload["breaking"]:
        out.append("No affected components found: nothing depends on this.")
    else:
        out.append(
            f"Blast radius: {s['affected']} component{'s' if s['affected'] != 1 else ''} "
            f"({s['high']} high, {s['medium']} medium, {s['low']} low)")
        current = None
        shown = 0
        for item in payload["affected"]:
            if shown >= max_items:
                break
            if item["distance"] != current:
                current = item["distance"]
                hop = "hop" if current == 1 else "hops"
                out.append(style(f"-- {current} {hop} away --", fg="cyan"))
            tag = style(f"{item['severity'].upper():<6}", fg=sev_color[item["severity"]])
            loc = f"{item['file']}:{item['line']}" if item.get("line") else (item.get("file") or "")
            kind = "file" if item["kind"] == "module" else item["kind"]
            out.append(f"  {tag} [{kind:<8}] {item['node']}  {loc}  ({item['reason']})")
            if item.get("explanation"):
                out.append(f"           {item['explanation']}")
            shown += 1
        hidden = len(payload["affected"]) - shown
        if hidden > 0:
            out.append(f"  ... and {hidden} more (use --max-items or --format json)")

    if payload["breaking"]:
        out.append("")
        out.append(style(f"Still referencing removed code ({len(payload['breaking'])}):",
                         fg="red", bold=True))
        for item in payload["breaking"][:max_items]:
            loc = f"{item['file']}:{item['line']}" if item.get("line") else ""
            out.append(f"  {item['node']}  {loc}  ({item['reason']})")

    command = pytest_command(payload["tests"])
    if command:
        out.append("")
        out.append(style(f"Tests to run ({len(payload['tests']['ids']) or len(payload['tests']['files'])}):",
                         bold=True))
        out.append(f"  {command}")
    if payload["skipped_files"]:
        out.append("")
        out.append(f"Note: {len(payload['skipped_files'])} file(s) could not be parsed "
                   "and were skipped (see --format json).")
    return "\n".join(out)


# -- Markdown (pull-request comments) ---------------------------------------------

def _md_code(text) -> str:
    return f"`{str(text).replace('`', '')}`"


def render_markdown(payload: dict, max_items: int = 40) -> str:
    s = payload["summary"]
    out = [MARKER, "### PyImpact: blast radius of this change", ""]

    if payload["mode"] == "diff" and not payload["targets"] and not payload["removed"]:
        out.append("No Python changes detected. :white_check_mark:")
        return "\n".join(out) + "\n"

    risk = (f":red_circle: {s['high']} high · :orange_circle: {s['medium']} medium · "
            f":white_circle: {s['low']} low")
    changed_count = len(payload["targets"]) + len(payload["removed"])
    noun = "symbol" if changed_count == 1 else "symbols"
    lead = (f"**{changed_count}** {noun} changed · " if payload["mode"] == "diff"
            else f"Target {', '.join(_md_code(t['node']) for t in payload['targets'])} · ")
    out.append(f"{lead}**{s['affected']}** components affected ({risk}) · "
               f"**{s['tests']}** tests to run")
    out.append("")

    if payload["breaking"]:
        out.append(f":warning: **{len(payload['breaking'])} component(s) still reference "
                   "code this change removes** — these will likely break:")
        out.append("")
        for item in payload["breaking"][:max_items]:
            loc = f" ({item['file']}:{item['line']})" if item.get("line") else ""
            out.append(f"- {_md_code(item['node'])}{loc} — {item['reason']}")
        out.append("")

    if payload["mode"] == "diff":
        rows = [*payload["targets"], *payload["removed"]]
        out.append(f"<details><summary>Changed symbols ({len(rows)})</summary>")
        out.append("")
        out.append("| Change | Symbol | Location |")
        out.append("|---|---|---|")
        for t in rows[:max_items]:
            loc = f"{t['file']}:{t['line']}" if t.get("line") else (t.get("file") or "")
            out.append(f"| {t.get('change')} | {_md_code(t['node'] or t['file'])} | {_md_code(loc)} |")
        if len(rows) > max_items:
            out.append(f"| | _… and {len(rows) - max_items} more_ | |")
        out.append("")
        out.append("</details>")
        out.append("")

    if payload["affected"]:
        ranked = sorted(payload["affected"],
                        key=lambda i: (SEVERITY_RANK[i["severity"]], -i["score"], i["distance"]))
        out.append("| Risk | Affected component | Location | Why |")
        out.append("|---|---|---|---|")
        for item in ranked[:max_items]:
            loc = f"{item['file']}:{item['line']}" if item.get("line") else (item.get("file") or "")
            hops = f"{item['distance']} hop{'s' if item['distance'] != 1 else ''}"
            why = item.get("explanation") or f"{item['reason']} ({hops})"
            out.append(f"| {_SEVERITY_ICON[item['severity']]} {item['severity']} | "
                       f"{_md_code(item['node'])} | {_md_code(loc)} | {why.replace('|', '/')} |")
        if len(ranked) > max_items:
            out.append(f"\n_… and {len(ranked) - max_items} more lower-risk components._")
        out.append("")

    command = pytest_command(payload["tests"])
    if command:
        out.append(f"<details><summary>Tests to run ({s['tests'] or len(payload['tests']['files'])})"
                   "</summary>")
        out.append("")
        out.append("```bash")
        out.append(command)
        out.append("```")
        out.append("</details>")
        out.append("")

    out.append(f"<sub>Generated by [PyImpact](https://github.com/abishek1123/pyimpact) "
               f"v{payload['version']} · static analysis, no code was executed</sub>")
    return "\n".join(out) + "\n"
