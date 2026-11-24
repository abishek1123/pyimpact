"""Glue for the PyImpact GitHub Action (``action.yml``).

Turns one ``pyimpact diff --format json`` report into everything the action
needs, so the analysis runs only once per job:

- a Markdown report (the pull-request comment),
- the job summary (``$GITHUB_STEP_SUMMARY``),
- step outputs (``$GITHUB_OUTPUT``): affected / high / medium / low / breaking /
  tests counts,
- and the exit status for ``fail-on`` gating (2 = threshold reached).

    python -m pyimpact.ci --json report.json --markdown report.md --fail-on high
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .impact import SEVERITY_RANK
from .report import render_markdown


def should_fail(payload: dict, fail_on: str) -> bool:
    if fail_on == "never":
        return False
    if payload["summary"]["breaking"]:
        return True
    if fail_on == "breaking":
        return False
    threshold = SEVERITY_RANK[fail_on]
    return any(SEVERITY_RANK[a["severity"]] <= threshold for a in payload["affected"])


def _append(path: str | None, text: str) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pyimpact.ci")
    parser.add_argument("--json", required=True, help="pyimpact diff --format json output")
    parser.add_argument("--markdown", help="where to write the Markdown report")
    parser.add_argument("--fail-on", default="never",
                        choices=["never", "breaking", "high", "medium", "low"])
    parser.add_argument("--max-items", type=int, default=40)
    parser.add_argument("--gate-only", action="store_true",
                        help="only apply --fail-on; write no report, outputs or summary")
    args = parser.parse_args(argv)
    if not args.gate_only and not args.markdown:
        parser.error("--markdown is required unless --gate-only is given")

    with open(args.json, encoding="utf-8") as f:
        payload = json.load(f)
    if args.gate_only:
        if should_fail(payload, args.fail_on):
            print(f"::error title=PyImpact::Change reaches the fail-on threshold ({args.fail_on}).")
            return 2
        return 0
    markdown = render_markdown(payload, max_items=args.max_items)
    with open(args.markdown, "w", encoding="utf-8") as f:
        f.write(markdown)

    _append(os.environ.get("GITHUB_STEP_SUMMARY"), markdown)
    s = payload["summary"]
    outputs = {key: s[key] for key in ("affected", "high", "medium", "low", "breaking", "tests")}
    outputs["report"] = os.path.abspath(args.markdown)
    _append(os.environ.get("GITHUB_OUTPUT"),
            "".join(f"{key}={value}\n" for key, value in outputs.items()))

    print(f"PyImpact: {s['targets'] + s['removed']} changed symbols, {s['affected']} affected "
          f"({s['high']} high, {s['medium']} medium, {s['low']} low), "
          f"{s['breaking']} breaking, {s['tests']} tests to run")
    if should_fail(payload, args.fail_on):
        print(f"::error title=PyImpact::Change reaches the fail-on threshold ({args.fail_on}).")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
