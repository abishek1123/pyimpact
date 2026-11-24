"""Tests for reports, search, visualization, stats, the LLM helpers and the CLI."""

from __future__ import annotations

import json
import subprocess
import textwrap

import pytest
from click.testing import CliRunner

from pyimpact import llm
from pyimpact.cli import cli
from pyimpact.impact import build_unified_graph, find_impact
from pyimpact.report import (
    MARKER,
    build_payload,
    render_markdown,
    render_text,
    target_dict,
)
from pyimpact.search import EmbeddingCache, SearchIndex, semantic_search, tokenize
from pyimpact.stats import hotspots, import_cycles, overview
from pyimpact.visualize import architecture_view, blast_radius_view, render_html

REPO = {
    "shop/__init__.py": "",
    "shop/security.py": '''
    def hash_password(raw: str) -> str:
        """Hash a password with a salt before storing it."""
        return raw[::-1]

    def verify_password(raw: str, hashed: str) -> bool:
        return hash_password(raw) == hashed
    ''',
    "shop/users.py": '''
    from shop.security import hash_password, verify_password
    from shop import orders

    class UserStore:
        """Persists user accounts."""

        def create(self, name, password):
            return {"name": name, "pw": hash_password(password)}

        def login(self, name, password, stored):
            return verify_password(password, stored)
    ''',
    "shop/orders.py": '''
    from shop import users

    def checkout(cart):
        """Charge the customer's card for everything in the cart."""
        return sum(cart)
    ''',
    "tests/test_users.py": '''
    from shop.users import UserStore

    def test_create():
        assert UserStore().create("a", "b")["pw"] == "b"
    ''',
}


@pytest.fixture
def repo(tmp_path):
    for rel, src in REPO.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(src), encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def graph(repo):
    return build_unified_graph(repo)


# --- reports ------------------------------------------------------------------------

def _payload(graph, target="shop.security:hash_password"):
    affected = find_impact(graph, target)
    return build_payload(graph, mode="analyze", targets=[target_dict(graph, target)],
                         affected=affected)


def test_payload_is_json_and_counts_add_up(graph):
    payload = _payload(graph)
    json.dumps(payload)
    s = payload["summary"]
    assert s["affected"] == s["high"] + s["medium"] + s["low"]
    assert payload["tests"]["ids"] == ["tests/test_users.py::test_create"]
    first = payload["affected"][0]
    assert {"node", "file", "line", "severity", "reason"} <= set(first)


def test_text_and_markdown_render(graph):
    payload = _payload(graph)
    text = render_text(payload)
    assert "Target: shop.security:hash_password" in text
    assert "pytest tests/test_users.py::test_create" in text
    text.encode("ascii")                       # terminal output stays ASCII-safe
    md = render_markdown(payload)
    assert md.startswith(MARKER)
    assert "| Risk | Affected component |" in md
    assert ":red_circle:" in md


def test_min_severity_filter(graph):
    affected = find_impact(graph, "shop.security:hash_password")
    payload = build_payload(graph, mode="analyze", targets=[], affected=affected,
                            min_severity="high")
    assert {a["severity"] for a in payload["affected"]} == {"high"}


# --- search ----------------------------------------------------------------------------

def test_tokenize_splits_identifiers():
    assert tokenize("hashPassword") == tokenize("hash_password") == ["hash", "password"]
    assert tokenize("HTTPServer") == ["http", "server"]


def test_lexical_search_ranks_by_name_and_doc(graph):
    index = SearchIndex(graph)
    assert index.search("password hashing")[0].node == "shop.security:hash_password"
    assert index.search("charge card")[0].node == "shop.orders:checkout"
    assert "tests.test_users:test_create" not in {h.node for h in index.search("create")}
    assert index.search("zzzz") == []


def test_semantic_search_uses_cache(graph, tmp_path):
    calls = []

    def fake_embed(texts, task_type):
        calls.append((task_type, len(texts)))
        return [[1.0 if "checkout" in t or "card" in t else 0.0, 1.0] for t in texts]

    index = SearchIndex(graph)
    cache = EmbeddingCache(str(tmp_path / "vec.sqlite3"))
    hits = semantic_search(index, "billing card", embed_fn=fake_embed, cache=cache,
                           hybrid=False)
    assert hits[0].node == "shop.orders:checkout"
    semantic_search(index, "billing card", embed_fn=fake_embed, cache=cache)
    cache.close()
    documents = [n for task, n in calls if task == "RETRIEVAL_DOCUMENT"]
    assert documents == [len(index)]           # second run embedded nothing new


# --- visualization ------------------------------------------------------------------------

def test_blast_radius_html(graph):
    affected = find_impact(graph, "shop.security:hash_password")
    nodes, edges, hidden = blast_radius_view(graph, ["shop.security:hash_password"], affected)
    roles = {n["id"]: n["role"] for n in nodes}
    assert roles["shop.security:hash_password"] == "target"
    assert hidden == 0 and edges
    page = render_html(title="t", subtitle="s", mode="impact", nodes=nodes, edges=edges)
    assert "vis-network.min.js" in page and "shop.users:UserStore.create" in page
    assert "</script><script" not in page.split("const DATA")[1].split("\n")[0]


def test_architecture_view_caps_nodes(graph):
    nodes, edges, hidden, packages = architecture_view(graph, max_nodes=3)
    assert len(nodes) == 3 and hidden == len([n for n, d in graph.nodes(data=True)
                                              if d["kind"] == "module"]) - 3
    assert packages


# --- stats -------------------------------------------------------------------------------------

def test_overview_hotspots_cycles(graph):
    info = overview(graph)
    assert info["files"] == 5 and info["test_files"] == 1 and info["classes"] == 1
    spots = hotspots(graph, top=3)
    assert spots[0].node == "shop.security:hash_password"
    assert spots[0].dependents == 3            # verify_password, create, login
    cycles = import_cycles(graph)
    assert sorted(cycles[0]) == ["shop.orders", "shop.users"]


# --- LLM key handling -------------------------------------------------------------------------

def test_dotenv_only_when_no_key_variable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OTHER=1\nGEMINI_API_KEY='abc123'\n", encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert llm.get_api_key() == "abc123"
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert llm.get_api_key() is None           # explicitly empty means "no key"


def test_generate_without_key_raises(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    with pytest.raises(llm.LLMError, match="Gemini API key"):
        llm.generate("hi")


# --- CLI ------------------------------------------------------------------------------------------

def run(*args):
    return CliRunner().invoke(cli, list(args), catch_exceptions=False)


def test_cli_analyze_formats(repo, tmp_path):
    out = run("analyze", repo, "-t", "shop.security.hash_password", "-f", "json")
    data = json.loads(out.output)
    assert data["targets"][0]["node"] == "shop.security:hash_password"
    md_file = tmp_path / "r.md"
    run("analyze", repo, "-t", "shop/security.py", "-f", "markdown", "-o", str(md_file))
    assert md_file.read_text(encoding="utf-8").startswith(MARKER)


def test_cli_suggests_targets(repo):
    result = CliRunner().invoke(cli, ["analyze", repo, "-t", "shop.hash_password"])
    assert result.exit_code != 0
    assert "Did you mean: shop.security:hash_password" in result.output


def test_cli_search_stats_visualize(repo, tmp_path):
    assert "shop.security:hash_password" in run("search", repo, "hash", "password").output
    stats = run("stats", repo)
    assert "Hotspots" in stats.output and "shop.orders -> shop.users" in stats.output.replace(
        "shop.users -> shop.orders", "shop.orders -> shop.users")
    html_file = tmp_path / "g.html"
    run("visualize", repo, "-t", "shop.security.hash_password", "-o", str(html_file), "--no-open")
    assert "vis-network" in html_file.read_text(encoding="utf-8")
    run("visualize", repo, "-o", str(html_file), "--no-open")
    assert "module architecture" in html_file.read_text(encoding="utf-8")


def test_cli_diff_and_fail_on(repo):
    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@e.com",
                        "-c", "commit.gpgsign=false", *args], cwd=repo, check=True,
                       capture_output=True)

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    path = f"{repo}/shop/security.py"
    with open(path, encoding="utf-8") as f:
        src = f.read()
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace("raw[::-1]", "raw[::-2]"))
    result = CliRunner().invoke(cli, ["diff", repo, "-f", "json"])
    data = json.loads(result.output)
    assert data["targets"][0]["node"] == "shop.security:hash_password"
    assert data["summary"]["high"] >= 1
    gated = CliRunner().invoke(cli, ["diff", repo, "--fail-on", "high"])
    assert gated.exit_code == 2
    ok = CliRunner().invoke(cli, ["diff", repo, "--fail-on", "breaking"])
    assert ok.exit_code == 0


def test_deferred_and_type_checking_imports_are_not_cycles(tmp_path):
    files = {
        "a.py": "import b\n",
        "b.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import a\n\n"
                "def f():\n    import a\n",
    }
    for rel, src in files.items():
        (tmp_path / rel).write_text(src, encoding="utf-8")
    g = build_unified_graph(str(tmp_path))
    assert g.has_edge("b", "a")                 # still a dependency for impact
    assert g.edges["b", "a"]["load_time"] is False
    assert import_cycles(g) == []
