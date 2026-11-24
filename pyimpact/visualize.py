"""Interactive dependency visualizations as a single self-contained HTML file.

Two views, both rendered with vis-network (loaded from a pinned CDN URL):

- **Blast radius** — the target(s) plus everything that depends on them, coloured
  by severity, with the target's own direct dependencies shown as grey context.
- **Architecture** — the file-level import graph of the whole repository,
  coloured by top-level package and sized by how many files import each one.

Clicking a node shows its file, line, signature, severity and why it is
affected; nodes can be searched and filtered by severity. Nothing is sent
anywhere — the data is embedded in the page.
"""

from __future__ import annotations

import html
import json
from collections import Counter

import networkx as nx

from . import __version__
from .impact import SEVERITY_RANK
from .report import reason

VIS_NETWORK_JS = ("https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/"
                  "standalone/umd/vis-network.min.js")
MAX_CONTEXT = 12


def _node_payload(graph, node, **extra) -> dict:
    data = graph.nodes[node]
    label = data.get("qualname", node) if data.get("kind") != "module" else node
    return {
        "id": node,
        "label": label,
        "kind": data.get("kind", "module"),
        "file": data.get("rel_path"),
        "line": data.get("lineno"),
        "signature": data.get("signature", ""),
        "doc": data.get("doc", ""),
        "is_test": bool(data.get("is_test")),
        **extra,
    }


def blast_radius_view(graph: nx.DiGraph, targets, affected, max_nodes: int = 200):
    """Nodes/edges for the targets and their (capped) blast radius."""
    targets = [t for t in targets if graph.has_node(t)]
    ranked = sorted(affected, key=lambda a: (SEVERITY_RANK[a.severity], a.distance, -a.score))
    kept = ranked[: max(0, max_nodes - len(targets))]
    nodes = {t: _node_payload(graph, t, role="target", severity="target", distance=0)
             for t in targets}
    for a in kept:
        nodes[a.node] = _node_payload(
            graph, a.node, role="affected", severity=a.severity, distance=a.distance,
            score=a.score, via=a.via, reason=reason(graph, a.node, a.via, a.toward))
    for t in targets:                              # what the target itself relies on
        deps = [v for v in graph.successors(t) if v not in nodes
                and graph.edges[t, v].get("kind") in ("call", "instantiate", "inherits")]
        for v in deps[:MAX_CONTEXT]:
            nodes[v] = _node_payload(graph, v, role="context", severity="context", distance=-1)
    edges = [{"from": u, "to": v, "kind": d.get("kind"), "strength": d.get("strength")}
             for u, v, d in graph.subgraph(nodes).edges(data=True)]
    return list(nodes.values()), edges, len(affected) - len(kept)


def architecture_view(graph: nx.DiGraph, max_nodes: int = 400):
    """Module-level import graph, capped to the most connected modules."""
    modules = [n for n, d in graph.nodes(data=True) if d.get("kind") == "module"]
    imports = nx.DiGraph()
    imports.add_nodes_from(modules)
    imports.add_edges_from((u, v) for u, v, d in graph.edges(data=True) if d.get("kind") == "import")
    degree = {m: imports.in_degree(m) * 2 + imports.out_degree(m) for m in modules}
    kept = sorted(modules, key=lambda m: (-degree[m], m))[:max_nodes]
    packages = Counter(m.split(".")[0] for m in kept)
    nodes = [_node_payload(graph, m, role="module", severity="module",
                           package=m.split(".")[0], importers=imports.in_degree(m))
             for m in kept]
    edges = [{"from": u, "to": v, "kind": "import"} for u, v in imports.subgraph(kept).edges]
    return nodes, edges, len(modules) - len(kept), sorted(packages, key=lambda p: -packages[p])


def render_html(*, title: str, subtitle: str, mode: str, nodes, edges,
                hidden: int = 0, packages=()) -> str:
    data = json.dumps({"mode": mode, "nodes": nodes, "edges": edges, "hidden": hidden,
                       "packages": list(packages), "version": __version__},
                      ensure_ascii=False).replace("</", "<\\/")
    page = _TEMPLATE.replace("__TITLE__", html.escape(title))
    page = page.replace("__SUBTITLE__", html.escape(subtitle))
    page = page.replace("__VIS__", VIS_NETWORK_JS)
    return page.replace("__DATA__", data)


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fbfbfc;--panel:#fff;--text:#1c2024;--muted:#60646c;--line:#e0e1e6;
--accent:#6e56cf;--high:#e5484d;--medium:#f5a524;--low:#8b8d98;--context:#b9bbc6}
@media (prefers-color-scheme:dark){:root{--bg:#111113;--panel:#18191b;--text:#edeef0;
--muted:#b0b4ba;--line:#2e3135;--accent:#9e8cfc;--low:#6f7177;--context:#43484e}}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--text);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:14px 18px;border-bottom:1px solid var(--line);display:flex;gap:16px;
align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0}.sub{color:var(--muted);font-size:13px}
.controls{margin-left:auto;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
input[type=search]{padding:6px 10px;border:1px solid var(--line);border-radius:8px;
background:var(--panel);color:var(--text);min-width:200px}
label.chip{display:inline-flex;gap:5px;align-items:center;font-size:12.5px;color:var(--muted);
cursor:pointer}.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
button{padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:var(--panel);
color:var(--text);cursor:pointer}
main{display:flex;height:calc(100% - 61px)}#graph{flex:1;min-width:0}
aside{width:340px;max-width:40vw;border-left:1px solid var(--line);background:var(--panel);
padding:16px;overflow:auto}
aside h2{font-size:14px;margin:0 0 6px;word-break:break-all}
aside .kv{color:var(--muted);font-size:12.5px;margin:3px 0}
aside code{font-size:12px;background:var(--bg);padding:2px 5px;border-radius:5px;word-break:break-all}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;color:#fff}
.empty{color:var(--muted)}footer{position:fixed;bottom:8px;left:12px;color:var(--muted);font-size:11.5px}
@media (max-width:760px){main{flex-direction:column}aside{width:auto;max-width:none;
border-left:0;border-top:1px solid var(--line);height:40%}#graph{height:60%}}
</style>
</head>
<body>
<header>
  <div><h1>__TITLE__</h1><div class="sub">__SUBTITLE__</div></div>
  <div class="controls" id="controls">
    <input type="search" id="find" placeholder="Find a node..." list="names">
    <datalist id="names"></datalist>
    <span id="filters"></span>
    <button id="physics" title="Toggle layout physics">Freeze</button>
    <button id="fit">Fit</button>
  </div>
</header>
<main><div id="graph"></div>
<aside id="details"><p class="empty">Click a node to see why it is affected.</p></aside></main>
<footer id="foot"></footer>
<script src="__VIS__"></script>
<script>
const DATA = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const PALETTE = ["#3e63dd","#12a594","#e54666","#f76b15","#8e4ec6","#0090ff","#30a46c",
  "#d6409f","#ab6400","#0d74ce","#5b5bd6","#ca244d"];
const pkgColor = {};
DATA.packages.forEach((p, i) => pkgColor[p] = PALETTE[i % PALETTE.length]);
const SHAPES = {module: "box", class: "diamond", function: "dot", method: "dot"};
function colorFor(n) {
  if (n.severity === "target") return css("--accent");
  if (n.severity === "module") return pkgColor[n.package] || css("--low");
  return css("--" + n.severity) || css("--low");
}
function visNode(n) {
  const c = colorFor(n);
  const size = n.role === "target" ? 26 : n.role === "module"
    ? 10 + Math.min(30, (n.importers || 0) * 2) : (n.severity === "high" ? 16 : 12);
  // Label only what matters on big graphs; everything else shows on hover.
  const labelled = DATA.nodes.length <= 60 || ["target", "high", "context", "module"].includes(n.severity);
  const text = n.label.length > 42 ? n.label.slice(0, 40) + "..." : n.label;
  return {id: n.id, label: labelled ? text : undefined, title: n.id,
    shape: n.role === "target" ? "star" : SHAPES[n.kind] || "dot", size,
    color: {background: c, border: n.role === "target" ? css("--text") : c,
            highlight: {background: c, border: css("--text")}},
    borderWidth: n.role === "target" ? 2 : 1,
    shapeProperties: {borderDashes: n.is_test ? [4, 3] : false},
    font: {color: css("--text"), size: n.role === "target" ? 15 : 12,
           face: "system-ui", background: n.kind === "module" ? undefined : css("--bg")},
    raw: n};
}
const nodes = new vis.DataSet(DATA.nodes.map(visNode));
const edges = new vis.DataSet(DATA.edges.map((e, i) => ({id: i, from: e.from, to: e.to,
  arrows: {to: {enabled: true, scaleFactor: 0.5}},
  dashes: e.strength === "virtual" || e.kind === "import",
  color: {color: css("--line"), highlight: css("--accent")}, width: e.kind === "call" ? 1.4 : 1})));
const big = DATA.nodes.length > 250;
const EMPTY = DATA.mode === "architecture" ? "Click a module to see what imports it."
  : "Click a node to see why it is affected.";
document.querySelector("#details .empty").textContent = EMPTY;
const network = new vis.Network(document.getElementById("graph"), {nodes, edges}, {
  layout: {improvedLayout: !big},
  physics: {solver: "forceAtlas2Based", stabilization: {iterations: big ? 150 : 400},
    forceAtlas2Based: {gravitationalConstant: -60, springLength: 110, avoidOverlap: 0.4}},
  interaction: {hover: true, tooltipDelay: 150},
  edges: {smooth: big ? false : {type: "continuous"}}});
network.once("stabilizationIterationsDone", () => network.setOptions({physics: false}));
let frozen = false;
document.getElementById("physics").onclick = e => { frozen = !frozen;
  network.setOptions({physics: !frozen}); e.target.textContent = frozen ? "Unfreeze" : "Freeze"; };
document.getElementById("fit").onclick = () => network.fit({animation: true});
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function show(id) {
  const n = nodes.get(id)?.raw; const box = document.getElementById("details");
  if (!n) { box.innerHTML = `<p class="empty">${EMPTY}</p>`; return; }
  const badge = ["high","medium","low","target"].includes(n.severity)
    ? `<span class="badge" style="background:${colorFor(n)}">${n.severity}</span> ` : "";
  box.innerHTML = `<h2>${esc(n.id)}</h2>${badge}<span class="kv">${esc(n.kind)}${n.is_test ? " · test" : ""}</span>
    ${n.file ? `<p class="kv">File: <code>${esc(n.file)}${n.line ? ":" + n.line : ""}</code></p>` : ""}
    ${n.signature ? `<p class="kv">Signature: <code>${esc(n.signature)}</code></p>` : ""}
    ${n.reason ? `<p class="kv">Why: <b>${esc(n.reason)}</b> (${n.distance} hop${n.distance === 1 ? "" : "s"}, confidence ${n.score})</p>` : ""}
    ${n.role === "context" ? `<p class="kv">A dependency of the target (shown for context).</p>` : ""}
    ${n.importers !== undefined ? `<p class="kv">Imported by ${n.importers} file(s)</p>` : ""}
    ${n.doc ? `<p>${esc(n.doc)}</p>` : ""}`;
}
network.on("click", p => show(p.nodes[0]));
const names = document.getElementById("names");
DATA.nodes.forEach(n => { const o = document.createElement("option"); o.value = n.id; names.appendChild(o); });
document.getElementById("find").addEventListener("change", e => {
  const q = e.target.value.trim().toLowerCase();
  const hit = DATA.nodes.find(n => n.id.toLowerCase() === q) || DATA.nodes.find(n => n.id.toLowerCase().includes(q));
  if (hit) { network.selectNodes([hit.id]); network.focus(hit.id, {scale: 1.2, animation: true}); show(hit.id); }
});
const filters = document.getElementById("filters");
const levels = DATA.mode === "architecture" ? [] : ["high", "medium", "low"];
const visible = new Set(levels);
levels.forEach(level => {
  const l = document.createElement("label"); l.className = "chip";
  l.innerHTML = `<input type="checkbox" checked><span class="dot" style="background:${css("--" + level)}"></span>${level}`;
  l.querySelector("input").onchange = ev => { ev.target.checked ? visible.add(level) : visible.delete(level);
    nodes.update(DATA.nodes.filter(n => n.severity === level).map(n => ({id: n.id, hidden: !visible.has(level)}))); };
  filters.appendChild(l);
});
document.getElementById("foot").textContent = `PyImpact ${DATA.version}` +
  (DATA.hidden ? ` · ${DATA.hidden} lower-priority nodes not shown` : "") +
  " · dashed border = test · dashed edge = import or dynamic dispatch";
if (DATA.nodes.find(n => n.role === "target")) show(DATA.nodes.find(n => n.role === "target").id);
</script>
</body>
</html>
"""
