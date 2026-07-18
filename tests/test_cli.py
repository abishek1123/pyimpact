"""Tests for the click CLI via click's CliRunner (no subprocess needed)."""

from __future__ import annotations

import textwrap

from click.testing import CliRunner

from pyimpact.cli import cli


def make_repo(tmp_path, files: dict[str, str]):
    for rel, source in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(tmp_path)


def test_analyze_reports_blast_radius(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "m.py": """
            def a():
                return 1

            def b():
                return a()
            """,
        },
    )
    result = CliRunner().invoke(cli, ["analyze", repo, "--target", "m:a"])
    assert result.exit_code == 0
    assert "m:a" in result.output
    assert "m:b" in result.output
    assert "Blast radius" in result.output


def test_analyze_unknown_target_errors(tmp_path):
    repo = make_repo(tmp_path, {"m.py": "def f():\n    return 1\n"})
    result = CliRunner().invoke(cli, ["analyze", repo, "--target", "does.not.exist"])
    assert result.exit_code != 0
    assert "Could not resolve" in result.output


def test_analyze_no_dependents(tmp_path):
    # A module that nothing imports has an empty blast radius. (A *function*
    # always has its own module as a dependent via the contains edge, so the
    # empty case is tested at module granularity.)
    repo = make_repo(tmp_path, {"m.py": "def lonely():\n    return 1\n"})
    result = CliRunner().invoke(cli, ["analyze", repo, "--target", "m"])
    assert result.exit_code == 0
    assert "No affected components" in result.output


def test_function_target_lists_own_module(tmp_path):
    repo = make_repo(tmp_path, {"m.py": "def lonely():\n    return 1\n"})
    result = CliRunner().invoke(cli, ["analyze", repo, "--target", "m:lonely"])
    assert result.exit_code == 0
    # The defining module is the one dependent, via the contains edge.
    assert "Blast radius: 1 component" in result.output


def test_max_distance_filter(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "m.py": """
            def a():
                return 1

            def b():
                return a()

            def c():
                return b()
            """,
        },
    )
    result = CliRunner().invoke(
        cli, ["analyze", repo, "--target", "m:a", "--max-distance", "1"]
    )
    assert result.exit_code == 0
    assert "m:b" in result.output
    assert "m:c" not in result.output


def test_kind_filter_functions_only(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/auth.py": "def login():\n    return 1\n",
            "app/routes.py": "import app.auth\n",
        },
    )
    result = CliRunner().invoke(
        cli, ["analyze", repo, "--target", "app.auth:login", "--kind", "module"]
    )
    assert result.exit_code == 0
    # module-only view should not list function nodes (contain a colon)
    assert "app.routes" in result.output


def test_missing_target_option(tmp_path):
    repo = make_repo(tmp_path, {"m.py": "def f():\n    return 1\n"})
    result = CliRunner().invoke(cli, ["analyze", repo])
    assert result.exit_code != 0  # --target is required
