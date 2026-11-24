"""Tests for git-diff impact analysis, using real temporary git repositories."""

from __future__ import annotations

import subprocess
import textwrap

import pytest

from pyimpact.diff import DiffError, analyze_diff, parse_unified_diff

BASE_FILES = {
    "app/__init__.py": "",
    "app/core.py": """
    def helper():
        return 1

    def compute():
        return helper() + 1

    def legacy():
        return 0
    """,
    "app/api.py": """
    from app.core import compute, legacy

    def endpoint():
        return compute()

    def old_endpoint():
        return legacy()
    """,
    "tests/test_api.py": """
    from app.api import endpoint

    def test_endpoint():
        assert endpoint() == 2
    """,
}


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
                    "-c", "commit.gpgsign=false", *args],
                   cwd=cwd, check=True, capture_output=True)


def write(root, files):
    for rel, src in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(src), encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    write(tmp_path, BASE_FILES)
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def edit(root, rel, old, new):
    path = root / rel
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_parse_unified_diff():
    text = textwrap.dedent("""\
        diff --git a/x.py b/x.py
        index 1..2 100644
        --- a/x.py
        +++ b/x.py
        @@ -3,2 +3,0 @@ def f():
        -    a
        -    b
        @@ -10 +8,3 @@
        -old
        +new1
        +new2
        +new3
        diff --git a/gone.py b/gone.py
        deleted file mode 100644
        --- a/gone.py
        +++ /dev/null
        @@ -1,2 +0,0 @@
    """)
    files = parse_unified_diff(text)
    assert files[0].old_lines == {3, 4, 10}
    assert files[0].new_lines == {8, 9, 10}
    assert files[1].new_path is None and files[1].old_path == "gone.py"


def test_modified_function_and_its_dependents(repo):
    edit(repo, "app/core.py", "return 1", "return 2")
    result = analyze_diff(str(repo))
    changed = {c.node: c for c in result.changed}
    assert changed["app.core:helper"].change == "modified"
    affected = {a.node for a in result.affected}
    assert {"app.core:compute", "app.api:endpoint",
            "tests.test_api:test_endpoint"} <= affected
    assert "app.api:old_endpoint" not in affected


def test_removed_function_flags_remaining_callers(repo):
    edit(repo, "app/core.py", "def legacy():\n    return 0\n", "")
    result = analyze_diff(str(repo))
    assert [r.node for r in result.removed] == ["app.core:legacy"]
    breaking = {a.node for a in result.removed_affected}
    assert "app.api:old_endpoint" in breaking


def test_untracked_new_file_is_added(repo):
    write(repo, {"app/extra.py": "from app.core import helper\n\ndef more():\n    return helper()\n"})
    result = analyze_diff(str(repo))
    changes = {c.node: c.change for c in result.changed}
    assert changes["app.extra:more"] == "added"
    assert changes["app.extra"] == "added"


def test_comment_only_module_edit_is_ignored(repo):
    edit(repo, "app/core.py", "def helper():", "# a comment\n\ndef helper():")
    result = analyze_diff(str(repo))
    assert all(c.kind != "module" for c in result.changed)


def test_branch_diff_against_base(repo):
    git(repo, "checkout", "-q", "-b", "feature")
    edit(repo, "app/api.py", "return compute()", "return compute() * 2")
    git(repo, "commit", "-q", "-am", "change endpoint")
    result = analyze_diff(str(repo), base="main")
    assert "main" in result.base_label
    assert [c.node for c in result.changed] == ["app.api:endpoint"]
    assert "tests.test_api:test_endpoint" in {a.node for a in result.affected}


def test_staged_only(repo):
    edit(repo, "app/core.py", "return 1", "return 3")
    assert analyze_diff(str(repo), staged=True).changed == []
    git(repo, "add", "-A")
    assert analyze_diff(str(repo), staged=True).changed


def test_no_changes(repo):
    result = analyze_diff(str(repo))
    assert result.changed == [] and result.affected == []


def test_subdirectory_root(repo):
    edit(repo, "app/core.py", "return 1", "return 5")
    result = analyze_diff(str(repo / "app"))
    assert "app.core:helper" in {c.node for c in result.changed}


def test_unknown_base_and_non_repo(repo, tmp_path_factory):
    with pytest.raises(DiffError, match="unknown base"):
        analyze_diff(str(repo), base="does-not-exist")
    plain = tmp_path_factory.mktemp("plain")
    (plain / "x.py").write_text("x = 1\n")
    with pytest.raises(DiffError, match="not inside a git repository"):
        analyze_diff(str(plain))
