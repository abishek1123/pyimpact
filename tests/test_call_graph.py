"""Unit tests for pyimpact.call_graph.

Each test builds a tiny fake repo whose call structure is obvious by reading it,
then asserts on the resolved edges. Edge convention: ``A -> B`` means "A calls B".
Node ids are ``"<module>:<qualname>"``.
"""

from __future__ import annotations

import textwrap

from pyimpact.call_graph import build_call_graph


def make_repo(tmp_path, files: dict[str, str]):
    for rel, source in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(tmp_path)


def test_same_module_call(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "m.py": """
            def helper():
                return 1

            def main():
                return helper()
            """,
        },
    )
    g = build_call_graph(repo)
    assert g.has_edge("m:main", "m:helper")
    assert not g.has_edge("m:helper", "m:main")


def test_self_method_call(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "m.py": """
            class Service:
                def check(self):
                    return True

                def run(self):
                    return self.check()
            """,
        },
    )
    g = build_call_graph(repo)
    assert g.has_edge("m:Service.run", "m:Service.check")


def test_cross_module_from_import(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/auth.py": """
            def login():
                return True
            """,
            "app/routes.py": """
            from app.auth import login

            def handle():
                return login()
            """,
        },
    )
    g = build_call_graph(repo)
    assert g.has_edge("app.routes:handle", "app.auth:login")


def test_cross_module_relative_import(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/auth.py": """
            def login():
                return True
            """,
            "app/routes.py": """
            from .auth import login

            def handle():
                return login()
            """,
        },
    )
    g = build_call_graph(repo)
    assert g.has_edge("app.routes:handle", "app.auth:login")


def test_qualified_module_call(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/auth.py": """
            def login():
                return True
            """,
            "main.py": """
            import app.auth

            def go():
                return app.auth.login()
            """,
        },
    )
    g = build_call_graph(repo)
    assert g.has_edge("main:go", "app.auth:login")


def test_aliased_import_call(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/auth.py": """
            def login():
                return True
            """,
            "main.py": """
            from app import auth as a

            def go():
                return a.login()
            """,
        },
    )
    g = build_call_graph(repo)
    assert g.has_edge("main:go", "app.auth:login")


def test_external_call_not_an_edge(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "m.py": """
            import os

            def f():
                return os.getcwd()
            """,
        },
    )
    g = build_call_graph(repo)
    # os.getcwd is external -> no node, no edge; f still exists as a node.
    assert g.has_node("m:f")
    assert g.number_of_edges() == 0


def test_all_functions_become_nodes(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "m.py": """
            def a():
                pass

            class C:
                def m(self):
                    pass
            """,
        },
    )
    g = build_call_graph(repo)
    assert g.has_node("m:a")
    assert g.has_node("m:C.m")


def test_method_calls_top_level_function(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "m.py": """
            def util():
                return 1

            class C:
                def run(self):
                    return util()
            """,
        },
    )
    g = build_call_graph(repo)
    assert g.has_edge("m:C.run", "m:util")


def test_instance_method_call_via_local_inference(tmp_path):
    # obj = C(); obj.target() -> local type inference resolves the method.
    repo = make_repo(
        tmp_path,
        {
            "m.py": """
            class C:
                def target(self):
                    pass

            def f():
                obj = C()
                return obj.target()
            """,
        },
    )
    g = build_call_graph(repo)
    assert g.has_edge("m:f", "m:C.target")
    assert g.edges["m:f", "m:C"]["kind"] == "instantiate"


def test_untyped_parameter_stays_unresolved(tmp_path):
    # Without an annotation or assignment there is no type to go on: the tool
    # must not guess (no edge is better than an invented one).
    repo = make_repo(
        tmp_path,
        {
            "m.py": """
            class C:
                def target(self):
                    pass

            def f(obj):
                return obj.target()
            """,
        },
    )
    g = build_call_graph(repo)
    assert not g.has_edge("m:f", "m:C.target")


def test_syntax_error_repo_does_not_crash(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "good.py": """
            def f():
                return 1
            """,
            "bad.py": "def broken(:\n    pass\n",
        },
    )
    g = build_call_graph(repo)  # must not raise
    assert g.has_node("good:f")


def test_no_recursion_self_edge(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "m.py": """
            def f(n):
                return f(n - 1)
            """,
        },
    )
    g = build_call_graph(repo)
    # A function calling itself is not a downstream-impact edge; skip self-loops.
    assert not g.has_edge("m:f", "m:f")
