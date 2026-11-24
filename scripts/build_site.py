"""Assemble the static web playground (GitHub Pages) into an output directory.

    python scripts/build_site.py _site

Copies ``site/`` and bundles the ``pyimpact`` package source under ``py/`` with a
manifest, so the browser can load the real analysis engine into Pyodide.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Modules the browser never imports (git subprocess, the click CLI, CI glue).
SKIP = {"cli.py", "ci.py", "diff.py", "__main__.py"}


def main(out: str) -> int:
    out = os.path.abspath(out)
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, "site"), out)
    pkg_out = os.path.join(out, "py", "pyimpact")
    os.makedirs(pkg_out)
    files = sorted(f for f in os.listdir(os.path.join(ROOT, "pyimpact"))
                   if f.endswith(".py") and f not in SKIP)
    for name in files:
        shutil.copy2(os.path.join(ROOT, "pyimpact", name), os.path.join(pkg_out, name))
    sys.path.insert(0, ROOT)
    from pyimpact import __version__

    with open(os.path.join(out, "py", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"version": __version__, "files": files}, f, indent=2)
    print(f"Built playground into {out} (pyimpact {__version__}, {len(files)} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "_site"))
