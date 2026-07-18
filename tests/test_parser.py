"""Unit tests for pyimpact.parser.

Each test writes a tiny, hand-crafted source file whose correct import list is
obvious by inspection, then asserts the parser reproduces it. This is the
"analysis is correct" bar the tool lives or dies by, checked at the smallest
possible unit.
"""

from __future__ import annotations

import textwrap

import pytest

from pyimpact.parser import get_imports, parse_file


def write_module(tmp_path, source: str):
    """Write dedented source to a temp .py file and return its path."""
    path = tmp_path / "sample.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def test_plain_import(tmp_path):
    path = write_module(tmp_path, "import os")
    assert get_imports(path) == ["os"]


def test_dotted_and_aliased_import(tmp_path):
    path = write_module(
        tmp_path,
        """
        import os.path as p
        import sys
        """,
    )
    # asname is irrelevant to *what* is imported; the dotted target is.
    assert get_imports(path) == ["os.path", "sys"]


def test_multiple_names_one_statement(tmp_path):
    path = write_module(tmp_path, "import os, sys, json")
    assert get_imports(path) == ["os", "sys", "json"]


def test_from_import(tmp_path):
    path = write_module(tmp_path, "from collections import defaultdict")
    assert get_imports(path) == ["collections.defaultdict"]


def test_from_import_multiple_and_aliased(tmp_path):
    path = write_module(
        tmp_path,
        "from os.path import join, dirname as d",
    )
    assert get_imports(path) == ["os.path.join", "os.path.dirname"]


def test_relative_import_bare(tmp_path):
    path = write_module(tmp_path, "from . import sibling")
    assert get_imports(path) == [".sibling"]


def test_relative_import_with_module(tmp_path):
    path = write_module(tmp_path, "from ..pkg import thing")
    assert get_imports(path) == ["..pkg.thing"]


def test_star_import_preserved(tmp_path):
    path = write_module(tmp_path, "from os import *")
    assert get_imports(path) == ["os.*"]


def test_source_order_preserved(tmp_path):
    path = write_module(
        tmp_path,
        """
        import a
        from b import c
        import d
        """,
    )
    assert get_imports(path) == ["a", "b.c", "d"]


def test_no_imports(tmp_path):
    path = write_module(tmp_path, "x = 1\n\ndef f():\n    return x\n")
    assert get_imports(path) == []


def test_parse_file_returns_module(tmp_path):
    import ast

    path = write_module(tmp_path, "def f():\n    return 1\n")
    tree = parse_file(path)
    assert isinstance(tree, ast.Module)


def test_syntax_error_reports_path(tmp_path):
    path = write_module(tmp_path, "def broken(:\n    pass\n")
    with pytest.raises(SyntaxError) as exc_info:
        get_imports(path)
    # Failures on real repos must be traceable to the offending file.
    assert "sample.py" in str(exc_info.value.filename)
