"""The terminal summary of a one-sided run."""

from __future__ import annotations

from typing import Any

from search.eval.summary import single_markdown, single_text


def _entry(verdict: str, *head: str, unstable: bool = False) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "head": [{"verdict": v, "detail": f"{v} detail", "infra": False} for v in head],
        "unstable": unstable,
        "fixed": False,
    }


def test_an_unreached_head_sample_reads_as_clean_and_renders_its_own_letter():
    """Every unreached head sample already fails ``disposition``, so a check unreached on
    every sample stays off the list; one unreached beside failures shows as ``~``."""
    checks = {
        "disposition": _entry("new_case_failure", "fail", "fail", "fail"),
        "measure": _entry("pass", "unreached", "unreached", "unreached"),
        "provenance": _entry("pass", "unreached", "fail", "fail", unstable=True),
    }
    case = {"file": "external", "query": "q", "checks": checks, "infra_errors": {}, "samples": {}}

    lines = single_text({"cases": {"c-1": case}, "exit_code": 1}).splitlines()

    rows = [line.split()[:5] for line in lines if line.startswith("  ") and "samples" not in line]
    assert rows == [
        ["FAIL", "disposition", "F", "F", "F"],
        ["unstable", "provenance", "~", "F", "F"],
    ]


def test_a_failed_sample_leads_the_detail_over_an_unreached_one():
    checks = {"measure": _entry("new_case_failure", "unreached", "fail", "fail")}
    case = {"file": "external", "query": "q", "checks": checks, "infra_errors": {}, "samples": {}}
    report = {"cases": {"c-1": case}, "exit_code": 1, "shard": None, "url": "u", "samples": 3}

    rows = [line for line in single_markdown(report).splitlines() if line.startswith("| `c-1`")]

    assert rows == ["| `c-1` | measure | ~ F F | fail detail |"]
