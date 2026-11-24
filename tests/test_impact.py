"""Unit tests for pyimpact.impact.

Builds small fake repos with a known dependency structure and asserts the
reverse-reachability blast radius, distances, and target resolution.
"""

from __future__ import annotations

import textwrap

from pyimpact.impact import (
    analyze,
    build_unified_graph,
    find_impact,
    resolve_target,
)


def make_repo(tmp_path, files: dict[str, str]):
    for rel, source in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(tmp_path)


def affected_nodes(affected):
    return {a.node for a in affected}


def test_direct_and_transitive_callers(tmp_path):
    # c calls b, b calls a. Changing a should surface both b (dist 1) and c (dist 2).
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
    resolved, affected = analyze(repo, "m:a")
    assert resolved == "m:a"
    by_node = {x.node: x for x in affected}
    assert by_node["m:b"].distance == 1
    assert by_node["m:c"].distance == 2
    assert by_node["m:b"].via == "call"


def test_unaffected_functions_excluded(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "m.py": """
            def target():
                return 1

            def caller():
                return target()

            def unrelated():
                return 99
            """,
        },
    )
    _, affected = analyze(repo, "m:target")
    nodes = affected_nodes(affected)
    assert "m:caller" in nodes
    assert "m:unrelated" not in nodes


def test_file_importers_in_blast_radius(tmp_path):
    # routes imports auth. Changing auth.login should reach the auth module and
    # then the routes module that imports it (file-level blast radius).
    repo = make_repo(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/auth.py": """
            def login():
                return True
            """,
            "app/routes.py": """
            import app.auth

            def handle():
                return 1
            """,
        },
    )
    _, affected = analyze(repo, "app.auth:login")
    nodes = affected_nodes(affected)
    # The module that defines login, and the module importing it, are affected.
    assert "app.auth" in nodes
    assert "app.routes" in nodes


def test_target_module_affects_importers(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/db.py": "def connect():\n    return 1\n",
            "app/service.py": "from app import db\n",
        },
    )
    resolved, affected = analyze(repo, "app.db")
    assert resolved == "app.db"
    assert "app.service" in affected_nodes(affected)


def test_no_impact_for_leaf(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "m.py": """
            def caller():
                return callee()

            def callee():
                return 1
            """,
        },
    )
    # Nothing calls caller() -> empty blast radius (besides its own module link,
    # which points module -> caller, not the reverse).
    _, affected = analyze(repo, "m:caller")
    assert "m:callee" not in affected_nodes(affected)


def test_resolve_target_variants(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/auth.py": """
            class User:
                def check(self):
                    return 1

            def login():
                return 1
            """,
        },
    )
    g = build_unified_graph(repo)
    # exact node id
    assert resolve_target(g, "app.auth:login", repo) == "app.auth:login"
    # dotted callable -> colon form
    assert resolve_target(g, "app.auth.login", repo) == "app.auth:login"
    # dotted method
    assert resolve_target(g, "app.auth.User.check", repo) == "app.auth:User.check"
    # module name
    assert resolve_target(g, "app.auth", repo) == "app.auth"
    # file path
    assert resolve_target(g, "app/auth.py", repo) == "app.auth"
    # nonexistent
    assert resolve_target(g, "does.not.exist", repo) is None


def test_analyze_unknown_target(tmp_path):
    repo = make_repo(tmp_path, {"m.py": "def f():\n    return 1\n"})
    resolved, affected = analyze(repo, "nope.nothing")
    assert resolved is None
    assert affected == []


def test_results_sorted_by_distance(tmp_path):
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
    _, affected = analyze(repo, "m:a")
    distances = [x.distance for x in affected if x.via == "call"]
    assert distances == sorted(distances)


def test_unified_graph_has_both_kinds(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "m.py": "def f():\n    return 1\n",
        },
    )
    g = build_unified_graph(repo)
    kinds = {data.get("kind") for _, data in g.nodes(data=True)}
    assert "module" in kinds
    assert "function" in kinds
    # module -> function containment edge exists
    assert g.has_edge("m", "m:f")
    assert g.edges["m", "m:f"]["kind"] == "contains"


# --- scoring, severity, multi-target, tests ---------------------------------

SCORING_REPO = {
    "app/__init__.py": "",
    "app/core.py": """
    def target():
        return 1

    def direct():
        return target()

    def two_hops():
        return direct()
    """,
    "app/other.py": """
    import app.core

    def unrelated():
        return 2
    """,
    "tests/test_core.py": """
    from app.core import direct

    def test_direct():
        assert direct() == 1

    def helper():
        return 0
    """,
}


def test_severity_reflects_edge_strength(tmp_path):
    repo = make_repo(tmp_path, SCORING_REPO)
    _, affected = analyze(repo, "app.core:target")
    by = {a.node: a for a in affected}
    assert by["app.core:direct"].severity == "high"
    assert by["app.core:direct"].score == 1.0
    assert by["app.core:two_hops"].score < by["app.core:direct"].score
    # A file that only imports the module is a weaker, medium-confidence signal.
    assert by["app.other"].severity == "medium"
    assert by["app.other"].via == "import"


def test_affected_tests_are_flagged(tmp_path):
    from pyimpact.impact import affected_tests

    repo = make_repo(tmp_path, SCORING_REPO)
    g = build_unified_graph(repo)
    affected = find_impact(g, "app.core:target")
    ids, files = affected_tests(g, affected)
    assert ids == ["tests/test_core.py::test_direct"]
    assert files == ["tests/test_core.py"]
    assert all(a.is_test for a in affected if a.node.startswith("tests."))


def test_multi_target_takes_closest(tmp_path):
    repo = make_repo(tmp_path, SCORING_REPO)
    g = build_unified_graph(repo)
    affected = find_impact(g, ["app.core:target", "app.core:direct"])
    nodes = {a.node: a for a in affected}
    assert "app.core:direct" not in nodes          # targets are not "affected"
    assert nodes["app.core:two_hops"].distance == 1


def test_max_distance_limits_traversal(tmp_path):
    repo = make_repo(tmp_path, SCORING_REPO)
    g = build_unified_graph(repo)
    affected = find_impact(g, "app.core:target", max_distance=1)
    assert max(a.distance for a in affected) == 1


def test_method_change_reaches_class_constructors_weakly(tmp_path):
    repo = make_repo(tmp_path, {"m.py": """
    class Cart:
        def total(self):
            return 0

    def build():
        return Cart()
    """})
    _, affected = analyze(repo, "m:Cart.total")
    by = {a.node: a for a in affected}
    # build() never calls total(), but holds a Cart: flagged, but not "high".
    assert by["m:build"].severity != "high"
    assert by["m:Cart"].via == "member"


def test_suggest_targets(tmp_path):
    from pyimpact.impact import suggest_targets

    repo = make_repo(tmp_path, SCORING_REPO)
    g = build_unified_graph(repo)
    assert "app.core:direct" in suggest_targets(g, "wrong.module.direct")
