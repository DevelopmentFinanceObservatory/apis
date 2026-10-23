"""Render the eval driver's JSON report as a Markdown summary or terminal text.

``paired_markdown`` lists the blocking verdicts first, then the checks base passed on a
wrong answer, then what the change did to the case set against base. ``single_text`` prints, case by case, every check that did not pass on
every sample. ``single_markdown`` is its Markdown form for a job summary. All three read
the report ``scripts/eval_paired.py`` writes. Both Markdown forms name the report's shard,
and say so when the selection holds no eligible case.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

from search.eval.gate import CONTENT_CHECKS, Outcome

# A list longer than this folds into a <details> block.
_FOLD_AFTER = 15
# Table cells keep this many characters of a check's detail.
_CELL_CHARS = 300

_LETTER = {"pass": "P", "fail": "F", "unverifiable": "U", "unreached": "~", "n/a": "-"}
# Head verdicts the gate reads as a pass: an unreached check's sample already fails
# `disposition`.
_CLEAN = frozenset({"pass", "unreached", "n/a"})
_FAILED = frozenset({"new_case_failure", "structural_failure"})

type Report = Mapping[str, Any]


def paired_markdown(report: Report) -> str:
    """The gate summary: result, blocking verdicts, infrastructure, case-set changes."""
    cases: Mapping[str, Any] = report["cases"]
    checks = list(_checks(cases))
    blocking = [(c, n, e) for c, n, e in checks if e["blocks"]]
    infra = [(c, n, e) for c, n, e in checks if e["verdict"] == "infra"]
    tally = Counter(e["verdict"] for _, _, e in checks)

    lines = [
        "## Search eval gate" + _shard_suffix(report),
        "",
        _result_line(report["exit_code"], blocking, infra),
        "",
    ]
    lines.append(
        f"Head {report['head_url']} against base `{report['base_ref']}` at "
        f"{report['base_url']}: {len(cases)} cases, {report['samples']} samples each."
    )
    lines += _no_cases(cases)
    if not report["base_corpus"]["loaded"]:
        lines.append(
            "The base corpus does not load under the head case model, so every case is new "
            f"({report['base_corpus']['error']})."
        )
    lines += ["", "Verdicts: " + ", ".join(f"{v} {n}" for v, n in sorted(tally.items())), ""]

    if blocking:
        lines += ["### Blocking verdicts", "", "| Case | Check | Verdict | Detail |"]
        lines.append("| --- | --- | --- | --- |")
        lines += [f"| `{c}` | {n} | {e['verdict']} | {_cell(_detail(e))} |" for c, n, e in blocking]
        lines.append("")
    if infra:
        lines += _section(
            "Infrastructure failures",
            [f"`{c}` {n}: {_detail(e) or _infra_error(cases[c])}" for c, n, e in infra],
        )
    lines += _section(
        "Wrong answers at base",
        [
            f"`{c}` {n}: head {_detail(e)}; base {_base_content_detail(cases[c])}"
            for c, n, e in checks
            if e["verdict"] == Outcome.WRONG_ANSWER_AT_BASE
        ],
    )

    changes = report["case_set"]
    lines += _section("Deleted cases", [f"`{c}`" for c in changes["deleted"]])
    lines += _section(
        "Newly ineligible cases",
        [f"`{c}`: {'; '.join(r)}" for c, r in changes["newly_ineligible"].items()],
    )
    lines += _section(
        "New known failures",
        [f"`{c}`: {', '.join(checks)}" for c, checks in changes["newly_known_failure"].items()],
    )
    lines += _section("Fixed known failures", [f"`{c}` {n}" for c, n, e in checks if e["fixed"]])
    lines += _section(
        "Unstable checks",
        [
            f"`{c}` {n} ({e['verdict']}): {_letters(e, 'head')}"
            for c, n, e in checks
            if e["unstable"] or e["verdict"] == "unstable"
        ],
    )
    lines += _section("Degraded samples", list(_degraded(cases)))
    return "\n".join(lines).rstrip() + "\n"


def single_text(report: Report) -> str:
    """Every case with a check that did not pass on every sample, for a terminal.

    Each check line shows its label, name, one letter per sample (P pass, F fail, U
    unverifiable, ~ unreached, - n/a, . not run on that sample) and each distinct detail.
    FAIL marks a check that passed on no sample and is not a known failure of its case.
    """
    cases: Mapping[str, Any] = report["cases"]
    out: list[str] = []
    shown = unmarked = 0
    for case_id, case in cases.items():
        rows = [(n, e) for n, e in case["checks"].items() if _not_clean(e)]
        if not rows and not case["infra_errors"]:
            continue
        shown += 1
        unmarked += any(e["verdict"] in _FAILED for _, e in rows)
        out.append(f"{case_id} ({case['file']}) {case['query']}")
        for name, entry in rows:
            details = _distinct_details(entry["head"])
            label = _single_label(entry)
            first = details[0] if details else ""
            out.append(f"  {label:<9} {name:<16} {_letters(entry, 'head')}  {first}".rstrip())
            out += [f"{'':<28}{detail}" for detail in details[1:]]
        for error in case["infra_errors"].values():
            out.append(f"  INFRA     {error}")
        out.append(f"  samples   {_samples_line(case['samples'].get('head', []))}")
        out.append("")
    out.append(
        f"{len(cases)} cases run, {shown} with a check that did not always pass, "
        f"{unmarked} with an unmarked failure. Exit {report['exit_code']}."
    )
    return "\n".join(out)


def single_markdown(report: Report) -> str:
    """The one-sided run as Markdown: unmarked failures, then counts."""
    cases: Mapping[str, Any] = report["cases"]
    checks = list(_checks(cases))
    failing = [(c, n, e) for c, n, e in checks if e["verdict"] in _FAILED]
    infra = [(c, n, e) for c, n, e in checks if e["verdict"] == "infra"]
    known = sum(1 for _, _, e in checks if e["verdict"] == "known_failure")
    lines = [
        "## Search eval" + _shard_suffix(report),
        "",
        f"{report['url']}: {len(cases)} cases, {report['samples']} samples each. "
        f"{len(failing)} unmarked failing checks, {known} known failures, "
        f"{len(infra)} infrastructure failures.",
        *_no_cases(cases),
        "",
    ]
    if failing:
        lines += ["### Unmarked failures", "", "| Case | Check | Samples | Detail |"]
        lines.append("| --- | --- | --- | --- |")
        lines += [
            f"| `{c}` | {n} | {_letters(e, 'head')} | {_cell(_detail(e))} |" for c, n, e in failing
        ]
        lines.append("")
    if infra:
        lines += _section(
            "Infrastructure failures",
            [f"`{c}` {n}: {_detail(e) or _infra_error(cases[c])}" for c, n, e in infra],
        )
    lines += _section("Degraded samples", list(_degraded(cases)))
    return "\n".join(lines).rstrip() + "\n"


def _shard_suffix(report: Report) -> str:
    return f" (shard {report['shard']})" if report["shard"] else ""


def _no_cases(cases: Mapping[str, Any]) -> list[str]:
    return [] if cases else ["No eligible case falls in this selection, so none ran."]


def _checks(cases: Mapping[str, Any]) -> Iterator[tuple[str, str, Mapping[str, Any]]]:
    for case_id, case in cases.items():
        for name, entry in case["checks"].items():
            yield case_id, name, entry


def _result_line(code: int, blocking: Sequence, infra: Sequence) -> str:
    if code == 1:
        return f"**Blocked:** {_count(len(blocking), 'blocking verdict')} (exit 1)."
    if code == 3:
        return (
            f"**Not judged:** {_count(len(infra), 'check')} hit infrastructure failures (exit 3)."
        )
    return "**Passed:** nothing blocks (exit 0)."


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _section(title: str, items: list[str]) -> list[str]:
    if not items:
        return []
    bullets = [f"- {item}" for item in items]
    if len(items) > _FOLD_AFTER:
        summary = f"<details><summary>{len(items)} entries</summary>"
        bullets = [summary, "", *bullets, "", "</details>"]
    return [f"### {title} ({len(items)})", "", *bullets, ""]


def _rounds(entry: Mapping[str, Any]) -> Iterator[Mapping[str, Any] | None]:
    for round_ in ("head", "confirm_head", "base", "confirm_base"):
        yield from entry.get(round_, ())


def _detail(entry: Mapping[str, Any]) -> str:
    """The entry's own detail when the driver wrote one, as for ``answer``, otherwise the
    first detail of a sample that did not pass, head samples first, a failed or unverifiable
    sample before an unreached or n/a one."""
    if detail := entry.get("detail"):
        return detail
    seen = [r for r in _rounds(entry) if r and r["verdict"] != "pass" and r["detail"]]
    seen.sort(key=lambda r: r["verdict"] in _CLEAN)
    return seen[0]["detail"] if seen else ""


def _base_content_detail(case: Mapping[str, Any]) -> str:
    """The first base-side failure of a content check (``search.eval.gate.CONTENT_CHECKS``),
    named by its check: ``measure: parts[0]: shape population_type: ...``."""
    for name in sorted(CONTENT_CHECKS & case["checks"].keys()):
        entry = case["checks"][name]
        for round_ in ("base", "confirm_base"):
            for result in entry.get(round_, ()):
                if result and result["verdict"] == "fail":
                    return f"{name}: {result['detail']}"
    return ""


def _distinct_details(results: Iterable[Mapping[str, Any] | None]) -> list[str]:
    seen = (r["detail"] for r in results if r and r["verdict"] not in _CLEAN)
    return list(dict.fromkeys(d for d in seen if d))


def _letters(entry: Mapping[str, Any], round_: str) -> str:
    return " ".join(_LETTER[r["verdict"]] if r else "." for r in entry.get(round_, ()))


def _not_clean(entry: Mapping[str, Any]) -> bool:
    return entry["verdict"] == "infra" or any(
        r and r["verdict"] not in _CLEAN for r in entry["head"]
    )


def _single_label(entry: Mapping[str, Any]) -> str:
    match entry["verdict"]:
        case "new_case_failure" | "structural_failure":
            return "FAIL"
        case "known_failure":
            return "fixed" if entry["fixed"] else "known"
        case "infra":
            return "INFRA"
    return "unstable" if entry["unstable"] else "partial"


def _samples_line(samples: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for index, sample in enumerate(samples):
        if sample.get("infra"):
            parts.append(f"#{index} infra")
            continue
        flags = [f for f in ("degraded", "request_rejected") if sample[f]]
        modal = " modal" if sample["modal"] else ""
        flag_text = f" {','.join(flags)}" if flags else ""
        parts.append(f"#{index} {sample['status']} {sample['trace_id']}{modal}{flag_text}")
    return "; ".join(parts)


def _degraded(cases: Mapping[str, Any]) -> Iterator[str]:
    for case_id, case in cases.items():
        for round_, samples in case["samples"].items():
            for index, sample in enumerate(samples):
                if sample.get("degraded"):
                    yield f"`{case_id}` {round_} sample {index} (trace {sample['trace_id']})"


def _infra_error(case: Mapping[str, Any]) -> str:
    return "; ".join(f"{r}: {e}" for r, e in case["infra_errors"].items())


def _cell(text: str) -> str:
    flat = " ".join(text.split()).replace("|", "\\|")
    return flat if len(flat) <= _CELL_CHARS else flat[: _CELL_CHARS - 1] + "…"
