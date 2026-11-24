"""Command-line interface for PyImpact.

    pyimpact analyze  [REPO] -t TARGET     what could break if TARGET changes
    pyimpact diff     [REPO] [--base REF]  impact of your git changes / a PR
    pyimpact visualize [REPO] [-t TARGET]  interactive HTML graph
    pyimpact search   [REPO] QUERY         find code by name or meaning
    pyimpact summarize [REPO] -t TARGET    AI summary grounded in the graph
    pyimpact stats    [REPO]               hotspots and import cycles

The CLI is a thin presentation layer; all analysis lives in library modules.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time

import click

from . import __version__
from .impact import (
    CALLABLE_KINDS,
    SEVERITY_RANK,
    build_unified_graph,
    find_impact,
    resolve_target,
    suggest_targets,
)
from .report import build_payload, render_markdown, render_text, target_dict, to_json

FORMATS = click.Choice(["text", "json", "markdown"])
SEVERITIES = click.Choice(["low", "medium", "high"])


# -- shared options -------------------------------------------------------------

def repo_argument(f):
    return click.argument("repo", default=".", required=False,
                          type=click.Path(exists=True, file_okay=False))(f)


def exclude_option(f):
    return click.option("--exclude", "-x", multiple=True, metavar="GLOB",
                        help="Skip files/dirs matching GLOB (repeatable), e.g. "
                             "'tests' or '*/migrations/*'.")(f)


def output_options(f):
    f = click.option("--output", "-o", type=click.Path(dir_okay=False),
                     help="Write the report to a file instead of stdout.")(f)
    return click.option("--format", "-f", "fmt", type=FORMATS, default="text",
                        show_default=True, help="Output format.")(f)


def ai_options(f):
    f = click.option("--model", default=None,
                     help="Gemini model for --explain (default: $GEMINI_MODEL or "
                          "gemini-3.5-flash-lite).")(f)
    f = click.option("--limit", type=int, default=15, show_default=True,
                     help="With --explain, how many items (highest risk first) to narrate.")(f)
    return click.option("--explain", is_flag=True,
                         help="Add an AI explanation of WHY each component is affected "
                              "(needs GEMINI_API_KEY).")(f)


def _load_graph(repo: str, exclude, quiet: bool):
    started = time.perf_counter()
    graph = build_unified_graph(repo, exclude)
    if not quiet:
        kinds = [d.get("kind") for _, d in graph.nodes(data=True)]
        files = kinds.count("module")
        funcs = sum(1 for k in kinds if k in CALLABLE_KINDS)
        skipped = len(graph.graph.get("skipped", []))
        note = f", {skipped} unparseable" if skipped else ""
        click.secho(f"Analyzed {files} files, {funcs} functions in "
                    f"{time.perf_counter() - started:.1f}s{note}", err=True, dim=True)
    return graph


def _resolve_all(graph, targets) -> list[str]:
    resolved = []
    for target in targets:
        node = resolve_target(graph, target)
        if node is None:
            hints = suggest_targets(graph, target)
            hint = ("\nDid you mean: " + ", ".join(hints)) if hints else ""
            raise click.ClickException(
                f"Could not resolve target '{target}'.\nUse a function (pkg.mod:func or "
                f"pkg.mod.func), a method (pkg.mod.Class.method), a class, a module "
                f"(pkg.mod) or a file path (pkg/mod.py).{hint}")
        resolved.append(node)
    return resolved


def _filter_kind(affected, kind: str):
    if kind == "all":
        return affected
    wanted = {"function": CALLABLE_KINDS, "class": {"class"}, "module": {"module"}}[kind]
    return [a for a in affected if a.kind in wanted]


def _explanations(graph, label, affected, explain, limit, model):
    if not explain or not affected:
        return None
    from .explain import explain_affected
    from .llm import LLMError

    try:
        return explain_affected(graph, label, affected, limit=limit, model=model).as_dict()
    except LLMError as exc:
        raise click.ClickException(str(exc)) from None


def _emit(payload: dict, fmt: str, output: str | None, max_items: int) -> None:
    if fmt == "json":
        text = to_json(payload)
    elif fmt == "markdown":
        text = render_markdown(payload, max_items=min(max_items, 60))
    else:
        text = render_text(payload, style=_plain if output else click.style, max_items=max_items)
    _write(text, output)


def _plain(text, **_):
    return text


def _write(text: str, output: str | None) -> None:
    if output:
        with open(output, "w", encoding="utf-8", newline="\n") as f:
            f.write(text if text.endswith("\n") else text + "\n")
        click.secho(f"Wrote {output}", err=True, dim=True)
    else:
        click.echo(text)


# -- commands ----------------------------------------------------------------------

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="pyimpact")
def cli() -> None:
    """PyImpact: predict what a Python code change will break, using static analysis.

    The dependency graph is ground truth; the optional AI features only explain
    what the graph already found."""


@cli.command("analyze")
@repo_argument
@click.option("--target", "-t", "targets", multiple=True, required=True,
              help="Function, method, class, module or file to analyze (repeatable).")
@click.option("--max-distance", type=int, default=None, help="Only follow N hops.")
@click.option("--min-severity", type=SEVERITIES, default=None,
              help="Hide results below this severity.")
@click.option("--kind", type=click.Choice(["all", "function", "class", "module"]),
              default="all", show_default=True, help="Only show this kind of component.")
@click.option("--max-items", type=int, default=100, show_default=True,
              help="Maximum rows in text/markdown output.")
@ai_options
@output_options
@exclude_option
def analyze_cmd(repo, targets, max_distance, min_severity, kind, max_items, explain,
                limit, model, fmt, output, exclude):
    """Show everything that could be affected by changing TARGET."""
    graph = _load_graph(repo, exclude, quiet=fmt != "text")
    resolved = _resolve_all(graph, targets)
    affected = _filter_kind(find_impact(graph, resolved, max_distance=max_distance), kind)
    explanations = _explanations(graph, ", ".join(resolved), affected, explain, limit, model)
    payload = build_payload(graph, mode="analyze",
                            targets=[target_dict(graph, n) for n in resolved],
                            affected=affected, explanations=explanations,
                            min_severity=min_severity)
    _emit(payload, fmt, output, max_items)


@cli.command("diff")
@repo_argument
@click.option("--base", "-b", default=None, metavar="REF",
              help="Compare against the merge-base with REF (e.g. main, origin/main). "
                   "Default: uncommitted changes vs HEAD.")
@click.option("--staged", is_flag=True, help="Only analyze staged changes.")
@click.option("--max-distance", type=int, default=None, help="Only follow N hops.")
@click.option("--min-severity", type=SEVERITIES, default=None,
              help="Hide results below this severity.")
@click.option("--max-items", type=int, default=100, show_default=True,
              help="Maximum rows in text/markdown output.")
@click.option("--fail-on", type=click.Choice(["never", "breaking", "high", "medium", "low"]),
              default="never", show_default=True,
              help="Exit with status 2 if the change has affected components at or above "
                   "this severity (or still-referenced removed code), for CI gating.")
@ai_options
@output_options
@exclude_option
def diff_cmd(repo, base, staged, max_distance, min_severity, max_items, fail_on, explain,
             limit, model, fmt, output, exclude):
    """Impact of your git changes: working tree vs HEAD, or a branch vs --base."""
    from .diff import DiffError, analyze_diff

    started = time.perf_counter()
    try:
        result = analyze_diff(repo, base=base, staged=staged, exclude=exclude,
                              max_distance=max_distance)
    except DiffError as exc:
        raise click.ClickException(str(exc)) from None
    if fmt == "text":
        click.secho(f"Analyzed {len(result.files)} changed file(s) in "
                    f"{time.perf_counter() - started:.1f}s", err=True, dim=True)
    graph = result.graph
    label = "the changed symbols: " + ", ".join(c.node for c in result.changed[:20])
    explanations = _explanations(graph, label, result.affected, explain, limit, model)
    payload = build_payload(graph, mode="diff", targets=result.changed,
                            affected=result.affected, base=result.base_label,
                            files=result.files, removed=result.removed,
                            removed_affected=result.removed_affected,
                            explanations=explanations, min_severity=min_severity)
    _emit(payload, fmt, output, max_items)

    if fail_on != "never":
        breaking = bool(result.removed_affected)
        threshold = None if fail_on == "breaking" else SEVERITY_RANK[fail_on]
        risky = threshold is not None and any(
            SEVERITY_RANK[a.severity] <= threshold for a in result.affected)
        if breaking or risky:
            click.secho(f"pyimpact: failing because of --fail-on {fail_on}", err=True, fg="red")
            sys.exit(2)


@cli.command("summarize")
@repo_argument
@click.option("--target", "-t", required=True, help="Function, method, class or module.")
@click.option("--model", default=None, help="Gemini model (default: $GEMINI_MODEL or "
                                            "gemini-3.5-flash-lite).")
@click.option("--format", "-f", "fmt", type=click.Choice(["text", "json"]), default="text",
              show_default=True)
@exclude_option
def summarize_cmd(repo, target, model, fmt, exclude):
    """AI summary of TARGET grounded in its position in the dependency graph."""
    from .llm import LLMError
    from .summarize import summarize_node

    graph = _load_graph(repo, exclude, quiet=fmt != "text")
    node = _resolve_all(graph, [target])[0]
    try:
        result = summarize_node(graph, node, model=model)
    except LLMError as exc:
        raise click.ClickException(str(exc)) from None
    ctx = result.context
    if fmt == "json":
        import dataclasses
        data = dataclasses.asdict(ctx)
        data.pop("source", None)
        _write(to_json({"node": result.node, "summary": result.text, "context": data}), None)
        return
    click.echo(f"Summary of {click.style(result.node, bold=True)} ({ctx.location})\n")
    click.echo(result.text or "(no summary returned)")
    click.secho(f"\nGraph position: depended on by {len(ctx.callers)}, depends on "
                f"{len(ctx.callees)}, blast radius {ctx.dependents_total} "
                f"({ctx.dependents_functions} functions, {ctx.blast_label})", fg="cyan")


@cli.command("search")
@repo_argument
@click.argument("query", nargs=-1, required=True)
@click.option("--top", "-k", type=int, default=10, show_default=True, help="Number of results.")
@click.option("--semantic", is_flag=True,
              help="Also rank by meaning using Gemini embeddings (needs GEMINI_API_KEY; "
                   "vectors are cached locally).")
@click.option("--include-tests", is_flag=True, help="Search test code too.")
@click.option("--model", default=None, help="Embedding model (default: gemini-embedding-001).")
@click.option("--format", "-f", "fmt", type=click.Choice(["text", "json"]), default="text",
              show_default=True)
@exclude_option
def search_cmd(repo, query, top, semantic, include_tests, model, fmt, exclude):
    """Find functions, methods and classes by name or description."""
    from .search import SearchIndex, semantic_search

    text = " ".join(query)
    graph = _load_graph(repo, exclude, quiet=fmt != "text")
    index = SearchIndex(graph, include_tests=include_tests)
    if semantic:
        from .llm import LLMError
        try:
            hits = semantic_search(index, text, k=top, model=model)
        except LLMError as exc:
            raise click.ClickException(str(exc)) from None
    else:
        hits = index.search(text, k=top)
    if fmt == "json":
        import dataclasses
        _write(to_json({"query": text, "semantic": semantic,
                        "results": [dataclasses.asdict(h) for h in hits]}), None)
        return
    if not hits:
        click.echo(f"No matches for '{text}'.")
        return
    for i, h in enumerate(hits, 1):
        loc = f"{h.file}:{h.line}" if h.line else (h.file or "")
        click.echo(f"{i:>2}. {click.style(h.node, bold=True)}{h.signature if h.kind != 'class' else ''}")
        click.secho(f"    {h.kind} | {loc}" + (f" | {h.doc}" if h.doc else ""), dim=True)


@cli.command("visualize")
@repo_argument
@click.option("--target", "-t", "targets", multiple=True,
              help="Show the blast radius of TARGET (repeatable). Omit for the "
                   "whole-repo architecture view.")
@click.option("--diff", "use_diff", is_flag=True, help="Show the blast radius of your git changes.")
@click.option("--base", default=None, metavar="REF", help="With --diff: compare against REF.")
@click.option("--max-distance", type=int, default=None, help="Only follow N hops.")
@click.option("--max-nodes", type=int, default=200, show_default=True,
              help="Cap on nodes drawn (lowest-risk ones are dropped first).")
@click.option("--output", "-o", type=click.Path(dir_okay=False), default="pyimpact-graph.html",
              show_default=True)
@click.option("--open/--no-open", "open_browser", default=None,
              help="Open the result in a browser (default: when run interactively).")
@exclude_option
def visualize_cmd(repo, targets, use_diff, base, max_distance, max_nodes, output,
                  open_browser, exclude):
    """Write an interactive HTML dependency graph."""
    from .visualize import architecture_view, blast_radius_view, render_html

    name = os.path.basename(os.path.abspath(repo))
    if use_diff:
        from .diff import DiffError, analyze_diff
        try:
            result = analyze_diff(repo, base=base, exclude=exclude, max_distance=max_distance)
        except DiffError as exc:
            raise click.ClickException(str(exc)) from None
        graph, nodes_in = result.graph, [c.node for c in result.changed]
        nodes, edges, hidden = blast_radius_view(graph, nodes_in, result.affected, max_nodes)
        title = f"{name}: impact of {result.base_label}"
        subtitle = f"{len(nodes_in)} changed symbols, {len(result.affected)} affected components"
        html_text = render_html(title=title, subtitle=subtitle, mode="impact", nodes=nodes,
                                edges=edges, hidden=hidden)
    elif targets:
        graph = _load_graph(repo, exclude, quiet=False)
        resolved = _resolve_all(graph, targets)
        affected = find_impact(graph, resolved, max_distance=max_distance)
        nodes, edges, hidden = blast_radius_view(graph, resolved, affected, max_nodes)
        counts = {s: sum(1 for a in affected if a.severity == s) for s in SEVERITY_RANK}
        html_text = render_html(
            title=f"Blast radius of {', '.join(resolved)}",
            subtitle=f"{len(affected)} affected: {counts['high']} high, "
                     f"{counts['medium']} medium, {counts['low']} low",
            mode="impact", nodes=nodes, edges=edges, hidden=hidden)
    else:
        graph = _load_graph(repo, exclude, quiet=False)
        nodes, edges, hidden, packages = architecture_view(graph, max_nodes)
        html_text = render_html(title=f"{name}: module architecture",
                                subtitle=f"{len(nodes)} modules, {len(edges)} import edges",
                                mode="architecture", nodes=nodes, edges=edges,
                                hidden=hidden, packages=packages)
    with open(output, "w", encoding="utf-8") as f:
        f.write(html_text)
    click.echo(f"Wrote {output}")
    if open_browser is None:
        open_browser = sys.stdout.isatty()
    if open_browser:
        click.launch(os.path.abspath(output))


@cli.command("stats")
@repo_argument
@click.option("--top", type=int, default=15, show_default=True, help="Number of hotspots.")
@click.option("--include-tests", is_flag=True, help="Include test functions as hotspots.")
@click.option("--format", "-f", "fmt", type=click.Choice(["text", "json"]), default="text",
              show_default=True)
@exclude_option
def stats_cmd(repo, top, include_tests, fmt, exclude):
    """Repository overview: riskiest functions (hotspots) and import cycles."""
    import dataclasses

    from .stats import hotspots, import_cycles, overview

    graph = _load_graph(repo, exclude, quiet=fmt != "text")
    info = overview(graph)
    spots = hotspots(graph, top=top, include_tests=include_tests)
    cycles = import_cycles(graph)
    if fmt == "json":
        _write(to_json({"overview": info, "hotspots": [dataclasses.asdict(h) for h in spots],
                        "import_cycles": cycles}), None)
        return
    click.echo(click.style("Overview", bold=True))
    click.echo(f"  {info['files']} files ({info['test_files']} test), {info['classes']} classes, "
               f"{info['functions']} functions, {info['methods']} methods")
    click.echo("  edges: " + ", ".join(f"{k} {v}" for k, v in info["edges"].items()))
    if info["unparsed_files"]:
        click.echo(f"  {info['unparsed_files']} file(s) could not be parsed")
    click.echo("")
    click.echo(click.style("Hotspots (changes here ripple furthest)", bold=True))
    if not spots:
        click.echo("  none")
    for i, h in enumerate(spots, 1):
        loc = f"{h.file}:{h.line}" if h.line else (h.file or "")
        click.echo(f"  {i:>2}. {h.node}  {click.style(str(h.dependents), fg='red', bold=True)} "
                   f"transitive callers, {h.fan_in} direct | {loc}")
    click.echo("")
    click.echo(click.style(f"Import cycles ({len(cycles)})", bold=True))
    if not cycles:
        click.echo("  none")
    for cycle in cycles:
        click.echo("  " + " -> ".join([*cycle, cycle[0]]))


def main() -> None:  # console-script entry point
    # Never crash on a console that can't encode a character in a symbol name.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(ValueError, OSError):
                stream.reconfigure(errors="replace")
    cli()


if __name__ == "__main__":
    main()
