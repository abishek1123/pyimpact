"""Integration tests against real open-source repos.

These run PyImpact's full pipeline over medium-sized real codebases and assert
relationships that a careful human reading the source can confirm. They are the
counterpart to the unit tests: the unit tests prove each stage is correct on
hand-crafted inputs; these prove the whole thing produces sane results on code
nobody wrote for the test.

The repos are cloned into a gitignored ``fixtures/`` dir (see README /
CONTRIBUTING for the clone commands). When the fixtures are absent — e.g. a
fresh checkout or CI without them — every test in this module skips instead of
failing, so the suite stays green without a network dependency.

Clone commands::

    mkdir fixtures && cd fixtures
    git clone --depth 1 https://github.com/psf/requests.git
    git clone --depth 1 https://github.com/pallets/flask.git
    git clone --depth 1 https://github.com/pallets/click.git
    git clone --depth 1 https://github.com/bottlepy/bottle.git
"""

from __future__ import annotations

import os

import pytest

from pyimpact.impact import build_unified_graph, find_impact, resolve_target

FIXTURES = os.path.join(os.path.dirname(__file__), os.pardir, "fixtures")


def _require(rel_path: str) -> str:
    """Return an absolute fixture path, or skip the test if it's not present."""
    path = os.path.abspath(os.path.join(FIXTURES, rel_path))
    if not os.path.exists(path):
        pytest.skip(f"fixture not cloned: {rel_path}")
    return path


def _impact_nodes(repo: str, target: str) -> set[str]:
    g = build_unified_graph(repo)
    resolved = resolve_target(g, target, repo)
    assert resolved is not None, f"could not resolve {target!r} in {repo}"
    return {a.node for a in find_impact(g, resolved)}


def _has_call_edge(repo: str, caller: str, callee: str) -> bool:
    g = build_unified_graph(repo)
    return g.has_edge(caller, callee) and \
        g.edges[caller, callee].get("kind") == "call"


# --- requests: HTTP-verb methods all delegate to self.request() ------------

def test_requests_verb_methods_call_request():
    repo = _require("requests/src")
    impacted = _impact_nodes(repo, "requests.sessions.Session.request")
    # Changing Session.request affects every verb shortcut that calls it.
    for verb in ("get", "post", "put", "delete", "head", "options", "patch"):
        assert f"requests.sessions:Session.{verb}" in impacted


def test_requests_get_calls_request_directly():
    repo = _require("requests/src")
    assert _has_call_edge(
        repo,
        "requests.sessions:Session.get",
        "requests.sessions:Session.request",
    )


# --- flask: __call__ -> wsgi_app -> full_dispatch -> dispatch_request -------

def test_flask_dispatch_chain_is_transitive():
    repo = _require("flask/src")
    impacted = _impact_nodes(repo, "flask.app.Flask.dispatch_request")
    # The WSGI entrypoint transitively depends on dispatch_request.
    assert "flask.app:Flask.full_dispatch_request" in impacted
    assert "flask.app:Flask.wsgi_app" in impacted


# --- click: __call__ -> main -> invoke -------------------------------------

def test_click_command_main_calls_invoke():
    repo = _require("click/src")
    assert _has_call_edge(
        repo, "click.core:Command.main", "click.core:Command.invoke"
    )


# --- bottle: single-file, verb shortcuts call self.route() -----------------

def test_bottle_verb_methods_call_route():
    repo = _require("bottle")
    impacted = _impact_nodes(repo, "bottle.Bottle.route")
    for verb in ("get", "post", "put", "delete", "patch"):
        assert f"bottle:Bottle.{verb}" in impacted


def test_bottle_builds_nontrivial_graph():
    repo = _require("bottle")
    g = build_unified_graph(repo)
    funcs = [n for n, d in g.nodes(data=True)
             if d.get("kind") in ("function", "method")]
    # Sanity: bottle's core is a big single module (plus its test suite); we
    # should extract hundreds of callables and many resolved call edges.
    assert len(funcs) > 100
    assert g.number_of_edges() > 100
