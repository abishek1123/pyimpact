"use strict";
// PyImpact Playground: runs the real pyimpact package in Pyodide.

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";
const MAX_FILES = 2500;
const MAX_FILE_BYTES = 1_500_000;
const MAX_TOTAL_BYTES = 40_000_000;
const SKIP_DIRS = new Set(["venv", "env", "virtualenv", "node_modules", "site-packages",
  "__pycache__", "build", "dist", "__pypackages__"]);

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

let pyodide = null;
let pyReady = null;           // promise for the runtime + package
let runCounter = 0;
let current = null;           // { label, ref }
let pendingTarget = null;
let archDrawn = false;

// ---------------------------------------------------------------------------
// UI helpers

function setProgress(fraction, text) {
  $("#progress").hidden = false;
  $("#bar").style.width = `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
  $("#status").textContent = text;
}

function showError(message) {
  const box = $("#error");
  box.textContent = message;
  box.hidden = !message;
}

function busy(on) {
  $("#go").disabled = on;
  $("#go").textContent = on ? "Working…" : "Analyze";
}

function sevDot(severity) {
  return `<span class="dot" style="background:${css("--" + severity)}"></span>`;
}

function where(item) {
  if (!item.file) return "";
  return item.line ? `${item.file}:${item.line}` : item.file;
}

// ---------------------------------------------------------------------------
// Python runtime

const BRIDGE = String.raw`
import json, sys, time
sys.path.insert(0, "/lib")
from pyimpact import __version__
from pyimpact.impact import (build_unified_graph, find_impact, resolve_target,
                             suggest_targets, SEVERITY_RANK)
from pyimpact.report import build_payload, target_dict
from pyimpact.stats import overview, hotspots, import_cycles
from pyimpact.search import SearchIndex
from pyimpact.visualize import blast_radius_view, architecture_view
from pyimpact.explain import gather_evidence, build_prompt, _parse_response

STATE = {}

def pi_load(root, exclude_json):
    started = time.perf_counter()
    graph = build_unified_graph(root, json.loads(exclude_json))
    STATE.clear()
    STATE["g"] = graph
    names = sorted(n for n, d in graph.nodes(data=True)
                   if d.get("kind") in ("function", "method", "class") and not d.get("is_test"))
    return json.dumps({"overview": overview(graph), "seconds": round(time.perf_counter() - started, 2),
                       "targets": names[:6000], "skipped": graph.graph["skipped"][:20]})

def pi_analyze(target, max_distance):
    g = STATE["g"]
    node = resolve_target(g, target)
    if node is None:
        return json.dumps({"error": f"Could not resolve '{target}'.",
                           "suggestions": suggest_targets(g, target)})
    affected = find_impact(g, node, max_distance=max_distance or None)
    STATE["last"] = (node, affected)
    payload = build_payload(g, mode="analyze", targets=[target_dict(g, node)], affected=affected)
    nodes, edges, hidden = blast_radius_view(g, [node], affected, 150)
    return json.dumps({"payload": payload, "graph": {"nodes": nodes, "edges": edges, "hidden": hidden}})

def pi_stats():
    g = STATE["g"]
    spots = [h.__dict__ for h in hotspots(g, top=25)]
    return json.dumps({"hotspots": spots, "cycles": import_cycles(g)})

def pi_search(query):
    if "search" not in STATE:
        STATE["search"] = SearchIndex(STATE["g"])
    return json.dumps([h.__dict__ for h in STATE["search"].search(query, k=20)])

def pi_architecture():
    nodes, edges, hidden, packages = architecture_view(STATE["g"], 250)
    return json.dumps({"nodes": nodes, "edges": edges, "hidden": hidden, "packages": packages})

def pi_explain_prompt(limit):
    node, affected = STATE["last"]
    ranked = sorted(affected, key=lambda a: (SEVERITY_RANK[a.severity], a.distance, a.node))[:limit]
    STATE["evidence"] = [gather_evidence(STATE["g"], a) for a in ranked]
    return build_prompt(node, STATE["evidence"])

def pi_explain_parse(raw):
    evidence = STATE["evidence"]
    whys = _parse_response(raw, len(evidence))
    return json.dumps({evidence[i].affected.node: why for i, why in whys.items()})

__version__
`;

function loadRuntime() {
  if (pyReady) return pyReady;
  pyReady = (async () => {
    const py = await loadPyodide({ indexURL: PYODIDE_URL });
    await py.loadPackage("networkx");
    const manifest = await (await fetch("py/manifest.json")).json();
    py.FS.mkdirTree("/lib/pyimpact");
    await Promise.all(manifest.files.map(async (name) => {
      const text = await (await fetch(`py/pyimpact/${name}`)).text();
      py.FS.writeFile(`/lib/pyimpact/${name}`, text);
    }));
    const version = py.runPython(BRIDGE);
    $("#version").textContent = `PyImpact ${version}.`;
    pyodide = py;
    return py;
  })();
  pyReady.catch(() => { pyReady = null; });
  return pyReady;
}

function call(fn, ...args) {
  const result = pyodide.globals.get(fn)(...args);
  return typeof result === "string" ? result : String(result);
}

// ---------------------------------------------------------------------------
// Fetching sources

function parseRepo(input) {
  let text = input.trim().replace(/\.git$/, "").replace(/\/+$/, "");
  let ref = "", subdir = "";
  const url = text.match(/github\.com\/([^/\s]+)\/([^/\s#?]+)(?:\/(?:tree|blob)\/([^/\s]+)(?:\/(.+))?)?/i);
  if (url) {
    text = `${url[1]}/${url[2]}`;
    ref = url[3] || "";
    subdir = url[4] || "";
  }
  const m = text.match(/^([A-Za-z0-9_.-]+)\/([A-Za-z0-9_.-]+)$/);
  if (!m) throw new Error("Enter a repository as owner/name or a github.com URL.");
  return { owner: m[1], name: m[2], ref, subdir };
}

function wanted(path, subdir) {
  if (!path.endsWith(".py")) return false;
  if (subdir && !(path === subdir || path.startsWith(subdir.replace(/\/$/, "") + "/"))) return false;
  const parts = path.split("/");
  return !parts.slice(0, -1).some((p) => p.startsWith(".") || SKIP_DIRS.has(p));
}

async function getJSON(url) {
  const resp = await fetch(url, { headers: { Accept: "application/vnd.github+json" } });
  if (!resp.ok) {
    const err = new Error(`${resp.status}`);
    err.status = resp.status;
    throw err;
  }
  return resp.json();
}

async function listGitHub(repo) {
  const base = `https://api.github.com/repos/${repo.owner}/${repo.name}`;
  try {
    let ref = repo.ref;
    if (!ref) ref = (await getJSON(base)).default_branch;
    const tree = await getJSON(`${base}/git/trees/${encodeURIComponent(ref)}?recursive=1`);
    const files = tree.tree.filter((t) => t.type === "blob" && wanted(t.path, repo.subdir))
      .map((t) => ({ path: t.path, size: t.size || 0 }));
    return { ref, files, truncated: tree.truncated };
  } catch (err) {
    if (err.status === 404) throw new Error(`Repository ${repo.owner}/${repo.name} (or ref "${repo.ref}") not found. Only public repositories are supported.`);
    if (err.status !== 403 && err.status !== 429) throw err;
    return listJsDelivr(repo);          // GitHub API rate limit: use the jsDelivr mirror
  }
}

async function listJsDelivr(repo) {
  for (const ref of repo.ref ? [repo.ref] : ["main", "master"]) {
    try {
      const data = await getJSON(`https://data.jsdelivr.com/v1/packages/gh/${repo.owner}/${repo.name}@${encodeURIComponent(ref)}?structure=flat`);
      const files = data.files.map((f) => ({ path: f.name.replace(/^\//, ""), size: f.size || 0 }))
        .filter((f) => wanted(f.path, repo.subdir));
      return { ref, files, truncated: false, mirror: true };
    } catch (_) { /* try the next ref */ }
  }
  throw new Error("GitHub's API rate limit was reached and the jsDelivr mirror could not list this repository. Try again in a few minutes.");
}

function rawUrl(repo, ref, path, mirror) {
  const encoded = path.split("/").map(encodeURIComponent).join("/");
  return mirror
    ? `https://cdn.jsdelivr.net/gh/${repo.owner}/${repo.name}@${encodeURIComponent(ref)}/${encoded}`
    : `https://raw.githubusercontent.com/${repo.owner}/${repo.name}/${encodeURIComponent(ref)}/${encoded}`;
}

async function fetchText(repo, ref, path, mirror) {
  for (const useMirror of mirror ? [true] : [false, true]) {
    try {
      const resp = await fetch(rawUrl(repo, ref, path, useMirror));
      if (resp.ok) return await resp.text();
    } catch (_) { /* fall through to the mirror */ }
  }
  return null;
}

function capFiles(files) {
  const sorted = [...files].sort((a, b) => a.path.localeCompare(b.path));
  const kept = [];
  let total = 0;
  for (const f of sorted) {
    if (f.size > MAX_FILE_BYTES) continue;
    if (kept.length >= MAX_FILES || total + f.size > MAX_TOTAL_BYTES) break;
    kept.push(f);
    total += f.size;
  }
  return { kept, dropped: files.length - kept.length };
}

async function downloadAll(repo, listing, onProgress) {
  const { kept, dropped } = capFiles(listing.files);
  const out = [];
  let done = 0;
  let index = 0;
  async function worker() {
    while (index < kept.length) {
      const f = kept[index++];
      const text = await fetchText(repo, listing.ref, f.path, listing.mirror);
      if (text !== null) out.push({ path: f.path, text });
      onProgress(++done, kept.length);
    }
  }
  await Promise.all(Array.from({ length: 16 }, worker));
  return { files: out, dropped };
}

// ---------------------------------------------------------------------------
// Analysis

function writeFiles(files) {
  const root = `/repo${++runCounter}`;
  for (const f of files) {
    const full = `${root}/${f.path}`;
    pyodide.FS.mkdirTree(full.slice(0, full.lastIndexOf("/")));
    pyodide.FS.writeFile(full, f.text);
  }
  return root;
}

async function analyzeSource(label, getFiles, subdir, meta) {
  showError("");
  busy(true);
  $("#results").hidden = true;
  archDrawn = false;
  try {
    setProgress(0.05, "Loading the Python runtime (first visit downloads ~10 MB)…");
    const runtime = loadRuntime();
    const { files, dropped, ref } = await getFiles();
    if (!files.length) throw new Error("No Python files found there.");
    await runtime;
    setProgress(0.85, `Building the call & import graph of ${files.length} files…`);
    await new Promise((r) => setTimeout(r, 30));
    const root = writeFiles(files);
    const analysisRoot = subdir ? `${root}/${subdir.replace(/^\/|\/$/g, "")}` : root;
    const exclude = JSON.stringify($("#exclude").value.split(/\s+/).filter(Boolean));
    const info = JSON.parse(call("pi_load", analysisRoot, exclude));
    current = { label, ref };
    renderSummary(label, ref, info, files.length, dropped, meta);
    setProgress(1, "Ready.");
    setTimeout(() => { $("#progress").hidden = true; }, 600);
    $("#results").hidden = false;
    renderStats();
    selectTab("impact");
    if (pendingTarget) {
      $("#target").value = pendingTarget;
      pendingTarget = null;
      runImpact();
    }
  } catch (err) {
    $("#progress").hidden = true;
    showError(err.message || String(err));
  } finally {
    busy(false);
  }
}

function analyzeGitHub(input, target) {
  let repo;
  try {
    repo = parseRepo(input);
  } catch (err) {
    showError(err.message);
    return;
  }
  if ($("#ref").value.trim()) repo.ref = $("#ref").value.trim();
  if ($("#subdir").value.trim()) repo.subdir = $("#subdir").value.trim().replace(/^\/|\/$/g, "");
  pendingTarget = target || null;
  const label = `${repo.owner}/${repo.name}`;
  const params = new URLSearchParams({ repo: label });
  if (repo.ref) params.set("ref", repo.ref);
  if (repo.subdir) params.set("path", repo.subdir);
  if (target) params.set("target", target);
  history.replaceState(null, "", `?${params}`);
  analyzeSource(label, async () => {
    setProgress(0.1, `Listing ${label} on GitHub…`);
    const listing = await listGitHub(repo);
    if (!listing.files.length) throw new Error(`No Python files found in ${label}${repo.subdir ? "/" + repo.subdir : ""}.`);
    const result = await downloadAll(repo, listing, (done, total) =>
      setProgress(0.1 + 0.7 * done / total, `Downloading ${done} / ${total} Python files…`));
    return { ...result, ref: listing.ref, truncated: listing.truncated };
  }, repo.subdir, `https://github.com/${label}`);
}

async function analyzeFolder(fileList) {
  const all = Array.from(fileList);
  if (!all.length) return;
  const top = (all[0].webkitRelativePath || all[0].name).split("/")[0];
  const chosen = all.map((f) => ({ file: f, path: (f.webkitRelativePath || f.name).split("/").slice(1).join("/") }))
    .filter((f) => f.path && wanted(f.path, ""));
  history.replaceState(null, "", window.location.pathname);
  analyzeSource(top, async () => {
    const { kept, dropped } = capFiles(chosen.map((f) => ({ ...f, size: f.file.size })));
    const files = [];
    for (const [i, f] of kept.entries()) {
      files.push({ path: f.path, text: await f.file.text() });
      if (i % 25 === 0) setProgress(0.1 + 0.7 * i / kept.length, `Reading ${i} / ${kept.length} files…`);
    }
    return { files, dropped, ref: "local folder" };
  }, "", null);
}

// ---------------------------------------------------------------------------
// Rendering

function renderSummary(label, ref, info, fileCount, dropped, link) {
  const o = info.overview;
  $("#repo-name").innerHTML = link ? `<a href="${esc(link)}">${esc(label)}</a>` : esc(label);
  const notes = [`@ ${ref}`, `analyzed in ${info.seconds}s`];
  if (dropped) notes.push(`${dropped} files skipped (size limits)`);
  if (o.unparsed_files) notes.push(`${o.unparsed_files} unparseable`);
  $("#repo-meta").textContent = notes.join(" · ");
  const stat = (n, label) => `<div class="stat"><b>${Number(n).toLocaleString()}</b><span>${label}</span></div>`;
  $("#stats").innerHTML = stat(o.files, "files") + stat(o.functions + o.methods, "functions") +
    stat(o.classes, "classes") + stat(o.edges.call || 0, "call edges") + stat(o.test_files, "test files");
  $("#targets").innerHTML = info.targets.map((t) => `<option value="${esc(t)}">`).join("");
  $("#impact-out").hidden = true;
  $("#impact-hint").hidden = false;
  $("#hits").innerHTML = "";
}

function renderStats() {
  const data = JSON.parse(call("pi_stats"));
  $("#hotspots tbody").innerHTML = data.hotspots.map((h, i) => `
    <tr class="clickable" data-target="${esc(h.node)}">
      <td>${i + 1}</td><td><div class="name">${esc(h.node)}</div></td>
      <td><b>${h.dependents}</b></td><td>${h.fan_in}</td>
      <td><span class="loc">${esc(where({ file: h.file, line: h.line }))}</span></td></tr>`).join("") ||
    `<tr><td colspan="5" class="muted">No functions with callers found.</td></tr>`;
  $("#cycles").innerHTML = data.cycles.length
    ? data.cycles.map((c) => `<li>${[...c, c[0]].map(esc).join(" → ")}</li>`).join("")
    : `<li class="muted">None found.</li>`;
}

function runImpact() {
  const target = $("#target").value.trim();
  if (!target || !pyodide) return;
  const depth = parseInt($("#depth").value, 10) || 0;
  const data = JSON.parse(call("pi_analyze", target, depth));
  $("#ai-status").textContent = "";
  if (data.error) {
    const hint = data.suggestions.length ? ` Did you mean: ${data.suggestions.join(", ")}?` : "";
    $("#impact-hint").textContent = data.error + hint;
    $("#impact-hint").hidden = false;
    $("#impact-out").hidden = true;
    return;
  }
  const params = new URLSearchParams(window.location.search);
  if (params.get("repo")) {
    params.set("target", target);
    history.replaceState(null, "", `?${params}`);
  }
  $("#impact-hint").hidden = true;
  $("#impact-out").hidden = false;
  renderImpact(data.payload, data.graph);
}

let lastPayload = null;

function renderImpact(payload, graph) {
  lastPayload = payload;
  const s = payload.summary;
  const t = payload.targets[0];
  $("#risk-row").innerHTML = `
    <span class="pill"><b>${esc(t.node)}</b>&nbsp;<span class="muted">${esc(where(t))}</span></span>
    <span class="pill">${sevDot("high")} ${s.high} high</span>
    <span class="pill">${sevDot("medium")} ${s.medium} medium</span>
    <span class="pill">${sevDot("low")} ${s.low} low</span>
    <span class="pill">${s.tests} tests</span>`;
  renderAffectedRows(payload.affected, {});
  const command = payload.tests.ids.length && payload.tests.ids.length <= 25
    ? "pytest " + payload.tests.ids.join(" ")
    : payload.tests.files.length ? "pytest " + payload.tests.files.join(" ") : "";
  $("#tests-card").hidden = !command;
  $("#tests").textContent = command;
  drawGraph($("#impact-graph"), graph, "impact");
}

function renderAffectedRows(items, explanations) {
  const order = { high: 0, medium: 1, low: 2 };
  const ranked = [...items].sort((a, b) => order[a.severity] - order[b.severity] || b.score - a.score || a.distance - b.distance);
  const shown = ranked.slice(0, 200);
  $("#affected tbody").innerHTML = shown.map((a) => `
    <tr class="clickable" data-target="${esc(a.node)}">
      <td><span class="sev">${sevDot(a.severity)}${a.severity}</span></td>
      <td><div class="name">${esc(a.node)}</div><div class="loc">${esc(where(a))}${a.is_test ? " · test" : ""}</div></td>
      <td>${esc(a.reason)}${explanations[a.node] ? `<div class="why-ai">${esc(explanations[a.node])}</div>` : ""}</td>
      <td>${a.distance}</td></tr>`).join("") ||
    `<tr><td colspan="4" class="muted">Nothing depends on this. It is safe to change in isolation.</td></tr>`;
  $("#more").textContent = ranked.length > shown.length ? `Showing the 200 highest-risk of ${ranked.length}. Use the CLI for the full list.` : "";
}

// ---------------------------------------------------------------------------
// Graphs (vis-network)

const PALETTE = ["#3e63dd", "#12a594", "#e54666", "#f76b15", "#8e4ec6", "#0090ff", "#30a46c",
  "#d6409f", "#ab6400", "#0d74ce", "#5b5bd6", "#ca244d"];

function drawGraph(container, data, mode) {
  container.innerHTML = "";
  const pkgColor = {};
  (data.packages || []).forEach((p, i) => { pkgColor[p] = PALETTE[i % PALETTE.length]; });
  const colorFor = (n) => n.severity === "target" ? css("--accent")
    : n.severity === "module" ? (pkgColor[n.package] || css("--low"))
    : css("--" + n.severity) || css("--low");
  const shapes = { module: "box", class: "diamond", function: "dot", method: "dot" };
  const many = data.nodes.length > 60;
  const nodes = new vis.DataSet(data.nodes.map((n) => {
    const c = colorFor(n);
    const labelled = !many || ["target", "high", "context", "module"].includes(n.severity);
    const text = n.label.length > 40 ? n.label.slice(0, 38) + "…" : n.label;
    return {
      id: n.id, label: labelled ? text : undefined, title: n.id,
      shape: n.role === "target" ? "star" : shapes[n.kind] || "dot",
      size: n.role === "target" ? 24 : n.role === "module" ? 10 : n.severity === "high" ? 14 : 10,
      color: { background: c, border: c, highlight: { background: c, border: css("--text") } },
      shapeProperties: { borderDashes: n.is_test ? [4, 3] : false },
      font: { color: css("--text"), size: n.role === "target" ? 14 : 11, face: "system-ui" },
      raw: n,
    };
  }));
  const edges = new vis.DataSet(data.edges.map((e, i) => ({
    id: i, from: e.from, to: e.to, arrows: { to: { enabled: true, scaleFactor: 0.45 } },
    dashes: e.strength === "virtual" || e.kind === "import",
    color: { color: css("--line"), highlight: css("--accent") },
  })));
  const network = new vis.Network(container, { nodes, edges }, {
    layout: { improvedLayout: data.nodes.length < 120 },
    physics: { solver: "forceAtlas2Based", stabilization: { iterations: 250 },
      forceAtlas2Based: { gravitationalConstant: -55, springLength: 100, avoidOverlap: 0.4 } },
    interaction: { hover: true, tooltipDelay: 120 },
    edges: { smooth: false },
  });
  network.once("stabilizationIterationsDone", () => network.setOptions({ physics: false }));
  const info = document.createElement("div");
  info.className = "node-info";
  info.hidden = true;
  container.appendChild(info);
  network.on("click", (p) => {
    const n = p.nodes.length ? nodes.get(p.nodes[0]).raw : null;
    if (!n) { info.hidden = true; return; }
    info.hidden = false;
    const why = n.reason ? ` · ${esc(n.reason)}` : n.role === "context" ? " · used by the target" : "";
    const action = mode === "impact" && n.role !== "target"
      ? `<button class="ghost" data-target="${esc(n.id)}" type="button">Trace from here</button>` : "";
    info.innerHTML = `<div class="row spaced"><span class="name">${esc(n.id)}</span>${action}</div>
      <div class="muted">${esc(n.kind)}${n.severity && n.role === "affected" ? " · " + n.severity : ""}${why}
      ${n.file ? " · " + esc(where(n)) : ""}${n.importers !== undefined ? ` · imported by ${n.importers}` : ""}</div>`;
  });
  return network;
}

// ---------------------------------------------------------------------------
// AI explanations (bring your own key)

async function explainWithGemini(key, model) {
  const prompt = call("pi_explain_prompt", 15);
  const resp = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-goog-api-key": key },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: { responseMimeType: "application/json", temperature: 0.2 },
    }),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error?.message || `Gemini returned ${resp.status}`);
  const text = (data.candidates?.[0]?.content?.parts || []).map((p) => p.text || "").join("");
  return JSON.parse(call("pi_explain_parse", text));
}

// ---------------------------------------------------------------------------
// Tabs, events, startup

function selectTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => { p.hidden = p.id !== `tab-${name}`; });
  if (name === "architecture" && !archDrawn && pyodide) {
    archDrawn = true;
    drawGraph($("#arch-graph"), JSON.parse(call("pi_architecture")), "architecture");
  }
}

function trace(target) {
  $("#target").value = target;
  selectTab("impact");
  runImpact();
  $("#target").scrollIntoView({ behavior: "smooth", block: "center" });
}

$("#repo-form").addEventListener("submit", (e) => {
  e.preventDefault();
  analyzeGitHub($("#repo").value);
});
document.querySelectorAll(".chip[data-repo]").forEach((chip) => chip.addEventListener("click", () => {
  $("#repo").value = chip.dataset.repo;
  $("#ref").value = "";
  $("#subdir").value = "";
  analyzeGitHub(chip.dataset.repo, chip.dataset.target);
}));
$("#folder").addEventListener("change", (e) => analyzeFolder(e.target.files));
$("#target-form").addEventListener("submit", (e) => { e.preventDefault(); runImpact(); });
$("#search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const hits = JSON.parse(call("pi_search", $("#query").value));
  $("#hits").innerHTML = hits.map((h) => `
    <li><span class="name" data-target="${esc(h.node)}">${esc(h.node)}</span>
      <div class="meta">${esc(h.kind)} · ${esc(where(h))}${h.doc ? " · " + esc(h.doc) : ""}</div></li>`).join("") ||
    `<li class="muted">No matches.</li>`;
});
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => selectTab(t.dataset.tab)));
document.addEventListener("click", (e) => {
  const el = e.target.closest("[data-target]");
  if (el && !el.classList.contains("chip")) trace(el.dataset.target);
});
$("#copy-tests").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("#tests").textContent);
    $("#copy-tests").textContent = "Copied";
    setTimeout(() => { $("#copy-tests").textContent = "Copy"; }, 1500);
  } catch (_) { /* clipboard unavailable */ }
});
$("#ai-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!lastPayload) return;
  const status = $("#ai-status");
  status.textContent = "Asking Gemini…";
  try {
    const explanations = await explainWithGemini($("#ai-key").value.trim(), $("#ai-model").value.trim());
    renderAffectedRows(lastPayload.affected, explanations);
    status.textContent = `Explained ${Object.keys(explanations).length} components.`;
  } catch (err) {
    status.textContent = `Gemini error: ${err.message}`;
  }
});

(function start() {
  const params = new URLSearchParams(window.location.search);
  const repo = params.get("repo");
  if (params.get("ref")) $("#ref").value = params.get("ref");
  if (params.get("path")) $("#subdir").value = params.get("path");
  if (repo) {
    $("#repo").value = repo;
    analyzeGitHub(repo, params.get("target"));
  } else {
    // Warm the runtime in the background so the first analysis is quicker.
    setTimeout(() => loadRuntime().catch(() => {}), 1200);
  }
})();
