"""Clone the pinned open-source repos used by the integration tests.

    python scripts/fetch_fixtures.py

Each repo is a shallow clone of a fixed release tag into ``fixtures/`` (which is
gitignored), so the integration tests are reproducible and never break because
an upstream project changed.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys

FIXTURES = {
    "requests": ("https://github.com/psf/requests.git", "v2.32.3"),
    "flask": ("https://github.com/pallets/flask.git", "3.1.0"),
    "click": ("https://github.com/pallets/click.git", "8.1.8"),
    "bottle": ("https://github.com/bottlepy/bottle.git", "0.13.2"),
}
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")


def remove_tree(path: str) -> None:
    """rmtree that also works on Windows, where git marks objects read-only."""
    def make_writable(func, target, _exc):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    if os.path.isdir(path):
        shutil.rmtree(path, onerror=make_writable)


def current_tag(path: str) -> str | None:
    proc = subprocess.run(["git", "-C", path, "describe", "--tags", "--exact-match"],
                          capture_output=True, text=True)
    return proc.stdout.strip() or None


def main() -> int:
    os.makedirs(ROOT, exist_ok=True)
    for name, (url, tag) in FIXTURES.items():
        dest = os.path.join(ROOT, name)
        if os.path.isdir(dest) and current_tag(dest) == tag:
            print(f"{name}: already at {tag}")
            continue
        remove_tree(dest)
        print(f"{name}: cloning {tag} ...")
        subprocess.run(["git", "-c", "advice.detachedHead=false", "clone", "--quiet",
                        "--depth", "1", "--branch", tag, url, dest], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
