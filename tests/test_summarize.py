"""Tests for graph-grounded summarization.

The model call is injected as a fake, so these assert the deterministic pieces:
the graph context (callers, callees, blast radius, source) is accurate, and the
prompt carries those facts with the anti-invention guardrail.
"""

from __future__ import annotations

import textwrap

from click.testing import CliRunner

from pyimpact.cli import cli
from pyimpact.impact import build_unified_graph, resolve_target
from pyimpact.summarize import (
    build_summary_prompt,
    gather_context,
    summarize_target,
)


def make_repo(tmp_path, files: dict[str, str]):
    for rel, source in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(tmp_path)


REPO = {
    "m.py": """
    def hash_password(p):
        return p[::-1]

    def validate(user):
        return hash_password(user)

    def route_login(user):
        return validate(user)

    def route_admin(user):
        return validate(user)
    """,
}


def test_context_callers_and_callees(tmp_path):
    repo = make_repo(tmp_path, REPO)
    g = build_unified_graph(repo)
    node = resolve_target(g, "m:validate", repo)
    ctx = gather_context(g, node)

    # validate is called by both routes and calls hash_password.
    assert set(ctx.callers) == {"m:route_login", "m:route_admin"}
    assert ctx.callees == ["m:hash_password"]
    assert ctx.kind == "function"


def test_context_blast_radius_and_source(tmp_path):
    repo = make_repo(tmp_path, REPO)
    g = build_unified_graph(repo)
    node = resolve_target(g, "m:validate", repo)
    ctx = gather_context(g, node)

    # Dependents: route_login, route_admin (functions) + module m (contains).
    assert ctx.dependents_functions == 2
    assert ctx.dependents_total >= 2
    # Source is the real function body.
    assert "def validate(user):" in ctx.source
    assert "hash_password(user)" in ctx.source


def test_prompt_has_graph_facts_and_guardrail(tmp_path):
    repo = make_repo(tmp_path, REPO)
    g = build_unified_graph(repo)
    ctx = gather_context(g, resolve_target(g, "m:validate", repo))
    prompt = build_summary_prompt(ctx)

    assert "m:route_login" in prompt and "m:route_admin" in prompt
    assert "m:hash_password" in prompt
    assert "blast radius" in prompt.lower()
    assert "do not" in prompt.lower()  # anti-invention guardrail


def test_summarize_target_uses_generate_fn(tmp_path):
    repo = make_repo(tmp_path, REPO)
    captured = {}

    def fake(prompt: str) -> str:
        captured["prompt"] = prompt
        return "  A grounded summary.  "

    result = summarize_target(repo, "m:validate", generate_fn=fake)
    assert result is not None
    assert result.node == "m:validate"
    assert result.text == "A grounded summary."      # stripped
    assert "GRAPH POSITION" in captured["prompt"]


def test_summarize_unknown_target(tmp_path):
    repo = make_repo(tmp_path, REPO)
    result = summarize_target(repo, "nope.nope", generate_fn=lambda p: "x")
    assert result is None


def test_module_target_uses_imports_as_context(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/db.py": "def connect():\n    return 1\n",
            "app/service.py": "from app import db\n",
        },
    )
    g = build_unified_graph(repo)
    ctx = gather_context(g, "app.db")
    assert ctx.kind == "module"
    # app.service imports app.db -> it's a caller at module granularity.
    assert "app.service" in ctx.callers


def test_cli_summarize_without_key_errors(tmp_path):
    repo = make_repo(tmp_path, REPO)
    result = CliRunner().invoke(
        cli, ["summarize", repo, "--target", "m:validate"],
        env={"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""},
    )
    assert result.exit_code != 0
    assert "Gemini API key" in result.output
