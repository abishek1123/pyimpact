# Changelog

## 1.0.1 — 2026-11-24

- Each result's reason ("calls X", "defines Y") now comes from the same path that set
  its severity, so the explanation always agrees with the risk level.
- Release workflow only runs for full version tags (the moving `v1` Action tag no
  longer triggers it).

## 1.0.0 — 2026-11-24

First public release.

### Analysis
- Repository index: every file discovered, named (package-aware: `src/` layouts,
  namespace sub-packages, monorepos) and parsed once; unparseable files are reported,
  never fatal.
- Call graph with flow-insensitive type inference: constructors, inheritance/MRO,
  `super()`, `cls`, virtual dispatch to overrides, annotated parameters and returns,
  `with`-bindings, instance attributes, module-level instances, re-exports,
  `getattr` with literals, callbacks and decorators, `@overload` stubs, import-time code.
- Unified depends-on graph with weighted edges; confidence scores and
  high/medium/low severity; multi-target queries; affected pytest node ids.
- `diff`: git-aware change mapping (working tree, staged, or merge-base with a branch),
  including removed/renamed symbols traced through the base revision.
- `stats`: exact transitive-caller hotspots and load-time import cycles.

### Interfaces
- CLI commands `analyze`, `diff`, `visualize`, `stats`, `search`, `summarize`, with
  text/JSON/Markdown output and `--fail-on` CI gating.
- GitHub Action that comments the blast radius on pull requests.
- Browser playground (Pyodide) for any public GitHub repository.
- Optional Gemini features: grounded explanations, graph-aware summaries, semantic
  search with a local embedding cache.

## 0.1.0

Initial prototype: import graph, basic call graph, impact query and CLI.
