# PyImpact

**Predict the downstream impact of a Python code change through static analysis.**

PyImpact parses a Python codebase into an explicit dependency graph — import
edges between files and call edges between functions — then answers the question
every developer asks before a change:

> *If I change this function or file, what else could break?*

It answers that as a **reverse reachability query**: find everything that
(transitively) depends on the thing you're about to touch.

## Why this is different

Most "AI + your codebase" tools (chat-with-your-repo assistants) *summarize and
search*. PyImpact does **rigorous, deterministic dependency tracing**. The graph
traversal is ground truth — it never guesses at impact. When an AI explanation
layer is added (roadmap, Tier 2), the AI only *narrates what the graph already
found*; it never decides impact itself.

Engineering framing: source code contains an implicit, undocumented dependency
structure. PyImpact makes it **explicit** as a directed graph, then treats change
impact as a graph problem (reverse BFS), not a language-model problem.

## Install

```bash
git clone <repo-url>
cd pyimpact
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate    # macOS/Linux
pip install -e .
```

Requires Python 3.11+. Runtime dependencies: `networkx`, `click`.

## Usage

```bash
pyimpact analyze <repo_path> --target <function_or_file>
```

`--target` accepts several forms:

| Form | Example |
|---|---|
| Function (explicit) | `pkg.mod:func` |
| Function (dotted) | `pkg.mod.func` |
| Method | `pkg.mod.Class.method` |
| Module | `pkg.mod` |
| File path | `pkg/mod.py` |

Options:

- `--max-distance N` — only show items within N hops of the target.
- `--kind function|module|all` — restrict the blast radius to callables or files.

You can also run it without installing: `python -m pyimpact analyze ...`.

### Example

Running PyImpact on its own source, asking what depends on `get_imports`:

```
$ pyimpact analyze pyimpact --target parser.get_imports
Target: parser:get_imports
Blast radius: 10 components (max depth 5)

-- 1 hop away --
  [function] import_graph:build_import_graph  (calls the target's chain)
  [file    ] parser  (defines the target's chain)
-- 2 hops away --
  [file    ] call_graph  (imports the target's chain)
  [function] impact:build_unified_graph  (calls the target's chain)
  [file    ] import_graph  (defines the target's chain)
-- 3 hops away --
  [file    ] impact  (defines the target's chain)
  [function] impact:analyze  (calls the target's chain)
...
```

Each item shows **how far** it is from the target and **why** it's connected
(`calls`, `imports`, or `defines`).

## How it works

```
Input: repo path + target function/file
   │
   ▼
[parser]        walks every .py file -> one AST each; extracts imports
   │
   ▼
[import_graph]  file-level DiGraph: A -> B means "file A imports file B"
   │
   ▼
[call_graph]    function-level DiGraph: A -> B means "function A calls function B"
   │            (resolves names using imports + local/class scope, incl. self.method())
   ▼
[impact]        merges both into one "depends-on" graph, then runs a reverse BFS
   │            from the target to collect everything that depends on it
   ▼
Output: blast radius, ordered by hop distance, each labeled by edge kind
```

**Edge direction convention:** every edge points in the *depends-on* direction
(`A -> B` = "A depends on B", so a change to B may affect A). Impact analysis is
therefore a reverse traversal: collect all predecessors of the target.

The unified graph has three edge kinds:

- `call` — a function calls another function
- `import` — a file imports another file
- `contains` — a module depends on each function it defines (this bridges the two
  granularities, so a function change surfaces both its callers *and* the files
  that import its module)

Each pipeline stage is a separate, independently unit-tested module. Node ids are
dotted module names for files (`app.auth`) and `module:qualname` for functions
(`app.auth:login`, `app.models:User.check`).

## Known limitations

Static analysis of a dynamically typed language has hard boundaries. PyImpact is
deliberately honest about where the call graph stops resolving, rather than
guessing and producing false edges:

- **Instance-method calls are not resolved.** `obj.method()` where `obj` is a
  local variable or parameter needs type inference to know `obj`'s class. Only
  `self.method()` (within a class) and `module.func()` (via an import) are
  resolved.
- **Constructors don't link to `__init__`.** `User()` doesn't create a call edge.
- **Dynamic dispatch is invisible.** `getattr(obj, name)()`, calls through
  dictionaries/callbacks, and metaprogramming aren't traced.
- **`from module import *`** is not tracked.
- **Decorators** that replace a function aren't followed.
- **Nested-function calls** are attributed to the enclosing top-level function
  (an over-approximation), and nested functions aren't call targets.
- **File-level import impact is conservative.** Importing a module marks the
  importer as potentially affected even if it doesn't use the specific changed
  function — an upper bound on the blast radius. (Confidence/severity weighting
  is on the roadmap.)

These are documented as tests, so the boundaries are pinned, not accidental.

## Testing

```bash
pytest
```

Every pipeline stage is unit-tested against small, hand-crafted example files
whose correct output is obvious by inspection — the tool's entire value is that
*the analysis is correct*, so correctness is checked at the smallest unit.

## Roadmap

Tier 1 (this release) is a complete, standalone tool. Later tiers add an AI
narration layer and richer views on top of the **same** graph — they never
replace the deterministic core.

- **Tier 1 — done:** AST parsing, import graph, call graph, impact query, CLI.
- **Tier 2:** AI explanations of *why* each item is affected; graph-grounded
  summarization; semantic search over the codebase; confidence/severity scoring.
- **Tier 2.5:** Interactive dependency visualizations (`pyvis`).
- **Tier 3 (optional):** Streamlit web demo, GitHub Action for PR diffs, PyPI
  packaging.

Scope is Python-only by design — keeping the static-analysis correctness bar
achievable is the point.
