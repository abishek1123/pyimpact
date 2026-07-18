"""Command-line interface for PyImpact.

Usage:
    pyimpact analyze <repo_path> --target <function_or_file>

The CLI is a thin presentation layer over ``impact.analyze`` — all analysis
logic lives in the library modules so it stays independently testable.
"""

from __future__ import annotations

import click

from .impact import analyze


# Human-friendly labels for the edge kind that connects an affected node toward
# the target.
_VIA_LABEL = {
    "call": "calls",
    "import": "imports",
    "contains": "defines",
}


@click.group()
@click.version_option(package_name="pyimpact", message="%(prog)s %(version)s")
def cli() -> None:
    """PyImpact — predict the downstream impact of a Python code change."""


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--target",
    "-t",
    required=True,
    help="Function or file to analyze: 'pkg.mod:func', 'pkg.mod.func', "
    "'pkg.mod', or a path like 'pkg/mod.py'.",
)
@click.option(
    "--max-distance",
    type=int,
    default=None,
    help="Only show items within this many hops of the target.",
)
@click.option(
    "--kind",
    type=click.Choice(["all", "function", "module"]),
    default="all",
    help="Restrict the blast radius to functions/methods or to modules.",
)
def analyze_cmd(repo_path: str, target: str, max_distance: int | None,
                kind: str) -> None:
    """Show everything that could be affected by changing TARGET in REPO_PATH."""
    resolved, affected = analyze(repo_path, target)

    if resolved is None:
        raise click.ClickException(
            f"Could not resolve target '{target}' in {repo_path}.\n"
            "Expected a function (pkg.mod:func / pkg.mod.func), a module "
            "(pkg.mod), or a file path (pkg/mod.py)."
        )

    if kind != "all":
        wanted = {"function", "method"} if kind == "function" else {"module"}
        affected = [a for a in affected if a.kind in wanted]
    if max_distance is not None:
        affected = [a for a in affected if a.distance <= max_distance]

    click.echo(f"Target: {click.style(resolved, bold=True)}")
    if not affected:
        click.echo("No affected components found — nothing depends on this target.")
        return

    click.echo(
        f"Blast radius: {len(affected)} component"
        f"{'s' if len(affected) != 1 else ''} "
        f"(max depth {max(a.distance for a in affected)})\n"
    )

    current = None
    for a in affected:
        if a.distance != current:
            current = a.distance
            hop = "hop" if a.distance == 1 else "hops"
            # ASCII-only separator: Windows consoles default to cp1252 and choke
            # on box-drawing characters.
            click.echo(click.style(f"-- {a.distance} {hop} away --", fg="cyan"))
        label = _VIA_LABEL.get(a.via, a.via)
        tag = "file" if a.kind == "module" else a.kind
        click.echo(f"  [{tag:8}] {a.node}  ({label} the target's chain)")


# `analyze` is a nicer command name than the function; register it explicitly.
cli.add_command(analyze_cmd, name="analyze")


def main() -> None:  # console-script entry point
    cli()


if __name__ == "__main__":
    main()
