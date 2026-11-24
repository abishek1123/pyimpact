"""Integration tests against real open-source repositories.

The whole pipeline runs over pinned releases of requests, flask, click and
bottle — analysed from the *repository root*, exactly as a user would — and the
assertions are relationships anyone can confirm by reading those projects'
source. Unit tests prove each stage on hand-crafted inputs; these prove the
result is right on code nobody wrote for the test.

Fetch the fixtures once with ``python scripts/fetch_fixtures.py``. When they are
absent every test here skips, so the suite never needs the network.
"""

from __future__ import annotations

import os

import pytest

from pyimpact.impact import affected_tests, build_unified_graph, find_impact
from pyimpact.stats import import_cycles

FIXTURES = os.path.join(os.path.dirname(__file__), os.pardir, "fixtures")
_GRAPHS: dict = {}


def repo_graph(name: str):
    path = os.path.abspath(os.path.join(FIXTURES, name))
    if not os.path.isdir(path):
        pytest.skip(f"fixture not cloned: {name} (run scripts/fetch_fixtures.py)")
    if name not in _GRAPHS:
        _GRAPHS[name] = build_unified_graph(path)
    return _GRAPHS[name]


def edge(g, a, b):
    return g.edges[a, b] if g.has_edge(a, b) else None


def impacted(g, node):
    return {a.node: a for a in find_impact(g, node)}


# --- requests 2.32.3 ----------------------------------------------------------------

def test_requests_src_layout_named_from_repo_root():
    g = repo_graph("requests")
    assert g.has_node("requests.sessions")
    assert not any(n.startswith("src.") for n in g.nodes)


def test_requests_verb_methods_delegate_to_request():
    g = repo_graph("requests")
    for verb in ("get", "post", "put", "delete", "head", "options", "patch"):
        assert edge(g, f"requests.sessions:Session.{verb}",
                    "requests.sessions:Session.request")["strength"] == "direct"


def test_requests_api_reaches_session_through_with_statement():
    # api.request does `with sessions.Session() as session: session.request(...)`
    g = repo_graph("requests")
    assert g.has_edge("requests.api:request", "requests.sessions:Session.request")
    assert g.has_edge("requests.api:request", "requests.sessions:Session")


def test_requests_change_reaches_public_api_and_tests():
    g = repo_graph("requests")
    affected = impacted(g, "requests.sessions:Session.request")
    assert affected["requests.api:get"].severity == "high"
    ids, files = affected_tests(g, affected.values())
    assert "tests/test_requests.py" in files
    assert any(i.startswith("tests/test_requests.py::") for i in ids)


# --- flask 3.1.0 ----------------------------------------------------------------------

def test_flask_wsgi_chain():
    g = repo_graph("flask")
    affected = impacted(g, "flask.app:Flask.dispatch_request")
    assert affected["flask.app:Flask.full_dispatch_request"].distance == 1
    assert affected["flask.app:Flask.wsgi_app"].distance == 2
    assert affected["flask.app:Flask.__call__"].distance == 3
    assert all(affected[n].severity == "high" for n in
               ("flask.app:Flask.full_dispatch_request", "flask.app:Flask.wsgi_app"))


def test_flask_namespace_subpackage_and_no_false_cycles():
    g = repo_graph("flask")
    assert g.has_node("flask.sansio.app")         # flask/sansio has no __init__.py
    assert not any(n.startswith("src.") for n in g.nodes)
    assert not [c for c in import_cycles(g) if c[0].startswith("flask")]


def test_flask_inherited_method_resolves_to_base_class():
    # Flask subclasses sansio.App; add_url_rule is defined on App.
    g = repo_graph("flask")
    assert "flask.sansio.app:App" in g.nodes["flask.app:Flask"]["bases"]
    affected = impacted(g, "flask.sansio.app:App.add_url_rule")
    assert any(n.startswith("tests.") for n in affected)


# --- click 8.1.8 --------------------------------------------------------------------------

def test_click_main_dispatches_virtually_to_overrides():
    # BaseCommand.main calls self.invoke(ctx); Command and MultiCommand override it.
    g = repo_graph("click")
    assert edge(g, "click.core:BaseCommand.main", "click.core:BaseCommand.invoke")["strength"] == "direct"
    assert edge(g, "click.core:BaseCommand.main", "click.core:Command.invoke")["strength"] == "virtual"
    assert edge(g, "click.core:BaseCommand.main", "click.core:MultiCommand.invoke")["strength"] == "virtual"


def test_click_overloads_point_at_implementation():
    g = repo_graph("click")
    node = g.nodes["click.core:BaseCommand.main"]
    assert len(node["spans"]) == 3                 # two @overload stubs + implementation
    assert node["lineno"] == 1014                  # the real `def main(` in click 8.1.8


# --- bottle 0.13.2 -----------------------------------------------------------------------

def test_bottle_verb_shortcuts_call_route():
    g = repo_graph("bottle")
    affected = impacted(g, "bottle:Bottle.route")
    for verb in ("get", "post", "put", "delete", "patch"):
        assert affected[f"bottle:Bottle.{verb}"].distance == 1


def test_all_fixtures_build_cleanly():
    for name in ("requests", "flask", "click", "bottle"):
        g = repo_graph(name)
        callables = sum(1 for _, d in g.nodes(data=True) if d.get("kind") in ("function", "method"))
        assert callables > 200, name
        calls = sum(1 for *_, d in g.edges(data=True) if d.get("kind") == "call")
        assert calls > 200, name
