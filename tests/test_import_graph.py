"""Unit tests for pyimpact.import_graph.

Each test materializes a tiny fake repo on disk with a known import structure,
builds the graph, and asserts on its nodes and edges. Edge direction convention:
``A -> B`` means "A imports B".
"""

from __future__ import annotations

import textwrap


from pyimpact.import_graph import (
    build_import_graph,
    find_python_files,
    module_name_for,
    resolve_relative,
)


def make_repo(tmp_path, files: dict[str, str]):
    """Create files under tmp_path. Keys are repo-relative POSIX-style paths."""
    for rel, source in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(tmp_path)


# --- module_name_for -------------------------------------------------------

def test_module_name_plain(tmp_path):
    repo = make_repo(tmp_path, {"pkg/mod.py": ""})
    path = str(tmp_path / "pkg" / "mod.py")
    assert module_name_for(path, repo) == "pkg.mod"


def test_module_name_package_init(tmp_path):
    repo = make_repo(tmp_path, {"pkg/__init__.py": ""})
    path = str(tmp_path / "pkg" / "__init__.py")
    assert module_name_for(path, repo) == "pkg"


def test_module_name_top_level(tmp_path):
    repo = make_repo(tmp_path, {"top.py": ""})
    path = str(tmp_path / "top.py")
    assert module_name_for(path, repo) == "top"


# --- resolve_relative ------------------------------------------------------

def test_resolve_relative_sibling_from_module():
    # inside pkg/a.py: `from . import b` -> pkg.b
    assert resolve_relative(".b", "pkg.a", is_package=False) == "pkg.b"


def test_resolve_relative_from_package_init():
    # inside pkg/__init__.py: `from . import b` -> pkg.b
    assert resolve_relative(".b", "pkg", is_package=True) == "pkg.b"


def test_resolve_relative_parent_package():
    # inside pkg/sub/a.py: `from ..other import c` -> pkg.other.c
    assert resolve_relative("..other.c", "pkg.sub.a", is_package=False) == "pkg.other.c"


# --- find_python_files -----------------------------------------------------

def test_find_python_files_skips_noise(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "a.py": "",
            "pkg/b.py": "",
            "venv/lib.py": "",
            ".git/hook.py": "",
            "__pycache__/c.py": "",
        },
    )
    found = {module_name_for(p, repo) for p in find_python_files(repo)}
    assert found == {"a", "pkg.b"}


# --- build_import_graph ----------------------------------------------------

def test_absolute_import_edge(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/auth.py": "def login():\n    pass\n",
            "app/routes.py": "from app.auth import login\n",
        },
    )
    g = build_import_graph(repo)
    assert g.has_edge("app.routes", "app.auth")
    # No spurious reverse edge.
    assert not g.has_edge("app.auth", "app.routes")


def test_relative_import_edge(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/auth.py": "def login():\n    pass\n",
            "app/routes.py": "from .auth import login\n",
        },
    )
    g = build_import_graph(repo)
    assert g.has_edge("app.routes", "app.auth")


def test_external_imports_dropped(tmp_path):
    repo = make_repo(
        tmp_path,
        {"solo.py": "import os\nfrom collections import defaultdict\n"},
    )
    g = build_import_graph(repo)
    assert set(g.nodes) == {"solo"}
    assert g.number_of_edges() == 0


def test_longest_prefix_resolution(tmp_path):
    # `from app.auth import login` -> dotted "app.auth.login"; must resolve to
    # module app.auth, not fail because "app.auth.login" isn't a file.
    repo = make_repo(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/auth.py": "def login():\n    pass\n",
            "main.py": "from app.auth import login\n",
        },
    )
    g = build_import_graph(repo)
    assert g.has_edge("main", "app.auth")


def test_import_submodule_directly(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/auth.py": "",
            "main.py": "import app.auth\n",
        },
    )
    g = build_import_graph(repo)
    assert g.has_edge("main", "app.auth")


def test_syntax_error_file_skipped_not_fatal(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "good.py": "import broken\n",
            "broken.py": "def f(:\n    pass\n",  # unparseable
        },
    )
    # Should not raise; broken.py still appears as a node, just no edges from it.
    g = build_import_graph(repo)
    assert "broken" in g.nodes
    assert "good" in g.nodes
    assert g.has_edge("good", "broken")


def test_no_self_edges(tmp_path):
    repo = make_repo(tmp_path, {"pkg/__init__.py": "from pkg import thing\n"})
    g = build_import_graph(repo)
    assert not g.has_edge("pkg", "pkg")
