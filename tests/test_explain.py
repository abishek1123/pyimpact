"""Tests for the AI explanation layer.

The model call is always injected as a fake ``generate_fn`` so these tests run
with no network and no API key. They verify the two things that must be correct
regardless of the model: (1) the evidence handed to the model is accurate and
grounded, and (2) the model's response is parsed and paired back to the right
components. The model's *wording* is not asserted — that's not deterministic.
"""

from __future__ import annotations

import json
import textwrap

from click.testing import CliRunner

from pyimpact.cli import cli
from pyimpact.explain import (
    build_prompt,
    explain_impact,
    gather_evidence,
    _parse_response,
)
from pyimpact.impact import build_unified_graph, find_impact, resolve_target


def make_repo(tmp_path, files: dict[str, str]):
    for rel, source in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(tmp_path)


CALLER_REPO = {
    "m.py": """
    def helper():
        return 1

    def main():
        return helper()
    """,
}


def _echo_fake(n: int):
    """A fake generator that returns valid JSON explaining n items generically."""
    def fake(prompt: str) -> str:
        return json.dumps(
            [{"index": i, "why": f"explanation {i}"} for i in range(n)]
        )
    return fake


def test_evidence_call_edge_is_grounded(tmp_path):
    repo = make_repo(tmp_path, CALLER_REPO)
    g = build_unified_graph(repo)
    target = resolve_target(g, "m:helper", repo)
    affected = find_impact(g, target)
    main = next(a for a in affected if a.node == "m:main")

    ev = gather_evidence(g, main)
    assert "calls m:helper" in ev.relationship
    assert "line" in ev.relationship          # a concrete line number was found
    assert "helper()" in ev.snippet           # the actual source line


def test_evidence_contains_edge(tmp_path):
    repo = make_repo(tmp_path, CALLER_REPO)
    g = build_unified_graph(repo)
    target = resolve_target(g, "m:helper", repo)
    affected = find_impact(g, target)
    module = next(a for a in affected if a.node == "m")

    ev = gather_evidence(g, module)
    assert "defines m:helper" in ev.relationship


def test_prompt_contains_grounding_and_guardrails(tmp_path):
    repo = make_repo(tmp_path, CALLER_REPO)
    g = build_unified_graph(repo)
    target = resolve_target(g, "m:helper", repo)
    evidence = [gather_evidence(g, a) for a in find_impact(g, target)]
    prompt = build_prompt("m:helper", evidence)

    # Grounding facts present...
    assert "m:helper" in prompt
    assert "m:main" in prompt
    # ...and the anti-hallucination guardrail is in the instructions.
    assert "already decided" in prompt.lower()
    assert "not to decide what is affected" in prompt.lower()


def test_explain_impact_pairs_responses(tmp_path):
    repo = make_repo(tmp_path, CALLER_REPO)
    affected = find_impact(
        build_unified_graph(repo), "m:helper"
    )
    result = explain_impact(
        repo, "m:helper", generate_fn=_echo_fake(len(affected))
    )
    assert result is not None
    assert result.target == "m:helper"
    assert len(result.items) == len(affected)
    # every item got a non-empty explanation paired to it
    assert all(item.why.startswith("explanation") for item in result.items)


def test_explain_impact_unknown_target(tmp_path):
    repo = make_repo(tmp_path, CALLER_REPO)
    result = explain_impact(repo, "nope.nothing", generate_fn=_echo_fake(0))
    assert result is None


def test_explain_impact_respects_limit(tmp_path):
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
    captured = {}

    def fake(prompt: str) -> str:
        captured["prompt"] = prompt
        return json.dumps([{"index": 0, "why": "only one"}])

    result = explain_impact(repo, "m:a", generate_fn=fake, limit=1)
    assert len(result.items) == 1
    # Only the single closest item should appear in the prompt.
    assert "[1]" not in captured["prompt"]


def test_parse_response_tolerates_plain_strings():
    assert _parse_response(json.dumps(["a", "b"]), 2) == {0: "a", 1: "b"}


def test_parse_response_bad_json_returns_empty():
    assert _parse_response("not json at all", 3) == {}


def test_cli_explain_without_key_errors(tmp_path):
    # With no API key and no injected fake, --explain must fail cleanly, not crash.
    repo = make_repo(tmp_path, CALLER_REPO)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["analyze", repo, "--target", "m:helper", "--explain"],
        env={"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""},
    )
    assert result.exit_code != 0
    assert "Gemini API key" in result.output
