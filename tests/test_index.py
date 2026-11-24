"""Tests for RepoIndex: discovery, package-aware naming and robust parsing."""

from __future__ import annotations

import textwrap

from pyimpact.index import RepoIndex, is_test_path, module_name_for


def make_repo(tmp_path, files: dict) -> str:
    for rel, source in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(source, bytes):
            path.write_bytes(source)
        else:
            path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(tmp_path)


def test_src_layout_gets_real_package_names(tmp_path):
    repo = make_repo(tmp_path, {
        "src/pkg/__init__.py": "",
        "src/pkg/core.py": "",
        "setup.py": "",
    })
    assert set(RepoIndex(repo).files) == {"pkg", "pkg.core", "setup"}


def test_pointing_at_the_package_itself(tmp_path):
    make_repo(tmp_path, {"pkg/__init__.py": "", "pkg/core.py": ""})
    index = RepoIndex(str(tmp_path / "pkg"))
    assert set(index.files) == {"pkg", "pkg.core"}


def test_non_package_dirs_use_relative_paths(tmp_path):
    repo = make_repo(tmp_path, {"scripts/build.py": "", "tool.py": ""})
    assert set(RepoIndex(repo).files) == {"scripts.build", "tool"}


def test_module_name_for_namespace_package(tmp_path):
    make_repo(tmp_path, {"ns/sub/__init__.py": "", "ns/sub/mod.py": ""})
    # `ns` has no __init__.py, so the package root is ns/sub.
    assert module_name_for(str(tmp_path / "ns/sub/mod.py"), str(tmp_path)) == "sub.mod"


def test_name_collisions_are_disambiguated(tmp_path):
    repo = make_repo(tmp_path, {
        "a/util.py": "",          # not a package -> "a.util"
        "pkg/__init__.py": "",
        "pkg/a/__init__.py": "",
        "pkg/a/util.py": "",      # -> "pkg.a.util"
    })
    names = set(RepoIndex(repo).files)
    assert {"a.util", "pkg.a.util"} <= names


def test_unparseable_files_are_recorded_not_fatal(tmp_path):
    repo = make_repo(tmp_path, {
        "ok.py": "x = 1\n",
        "bad_syntax.py": "def f(:\n",
        "nulls.py": b"x = 1\x00\n",
        "latin.py": b"# -*- coding: latin-1 -*-\nname = '\xe9'\n",
        "bad_bytes.py": b"name = '\xff\xfe'\n",
    })
    index = RepoIndex(repo)
    skipped = {f.module for f in index.skipped}
    assert skipped == {"bad_syntax", "nulls", "bad_bytes"}
    assert index.files["latin"].tree is not None     # PEP 263 cookie honoured
    assert all(f.error for f in index.skipped)


def test_excludes_and_default_skips(tmp_path):
    repo = make_repo(tmp_path, {
        "app.py": "",
        "venv/lib.py": "",
        ".tox/x.py": "",
        "node_modules/y.py": "",
        "migrations/0001.py": "",
        "gen/api_pb2.py": "",
    })
    index = RepoIndex(repo, exclude=["migrations", "*_pb2.py"])
    assert set(index.files) == {"app"}


def test_is_test_path():
    assert is_test_path("tests/test_api.py")
    assert is_test_path("pkg/tests/helpers.py")
    assert is_test_path("api_test.py")
    assert is_test_path("conftest.py")
    assert not is_test_path("pkg/testing_utils.py")
    assert not is_test_path("pkg/contest.py")


def test_namespace_subpackage_inside_regular_package(tmp_path):
    # flask/sansio/ has no __init__.py but lives inside the regular package flask/.
    repo = make_repo(tmp_path, {
        "src/flask/__init__.py": "",
        "src/flask/sansio/app.py": "",
        "tests/test_x.py": "",
    })
    assert set(RepoIndex(repo).files) == {"flask", "flask.sansio.app", "tests.test_x"}


def test_import_entries_flag_load_time(tmp_path):
    repo = make_repo(tmp_path, {"m.py": """
        import os
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            import json

        def f():
            import csv
    """})
    entries = RepoIndex(repo).files["m"].import_entries
    flags = {node.names[0].name: load_time for node, load_time in entries}
    assert flags == {"os": True, "TYPE_CHECKING": True, "json": False, "csv": False}
