#!/usr/bin/env python3
"""Score an offline flow-direction scanner against hand-labelled cases.

`--scanner module:function` accepts `(query, entities)` and returns one `origin`, `destination`,
or `None` verdict per entity. Cases come from `corpus/direction/role_direction_cases.yaml`.
Run `scripts/eval_role_direction.py --scanner module:function [--verdicts-out path]`.
Put the scanner's module on `sys.path` first. The only in-tree scanner is the frozen
`corpus/direction/role_text_pinned.py`, kept as a baseline.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, TypedDict

import yaml

REPO = Path(__file__).parent.parent
CASES_YAML = REPO / "corpus" / "direction" / "role_direction_cases.yaml"

Axis = Literal["origin", "destination"]
ExpectedAxis = Literal["origin", "destination", "unspecified"]
Scanner = Callable[[str, Sequence[tuple[str | None, str]]], Sequence[Axis | None]]


class Verdict(TypedDict):
    case_id: str
    query: str
    surface: str | None
    dcid: str
    expected: ExpectedAxis
    predicted: ExpectedAxis
    correct: bool


class ScoreResult(TypedDict):
    accuracy: float
    false_confident_rate: float
    confidently_wrong_count: int
    confidently_wrong_rate: float
    entity_count: int
    unspecified_count: int
    by_category: dict[str, dict[str, int]]
    verdicts: list[Verdict]


def load_cases(path: Path = CASES_YAML) -> list[dict]:
    """Load cases and require unique ids.

    `score()` reports malformed record shapes.
    """
    cases = yaml.safe_load(path.read_text()) or []
    ids = [c["id"] for c in cases]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        sys.exit(f"{path}: duplicate case ids: {dupes}")
    return cases


def score(cases: list[dict], scanner: Scanner) -> ScoreResult:
    """Score every entity and return aggregate rates plus per-entity verdicts.

    Accuracy maps `None` to `unspecified`. False-confident rate is the share of expected-
    unspecified entities given a directional answer. Confidently-wrong rate is the share of
    directional entities assigned the opposite axis. `unspecified` predictions count toward
    accuracy only.
    """
    verdicts: list[Verdict] = []
    total = correct = 0
    unspecified_total = unspecified_confident = 0
    directional_total = confidently_wrong = 0
    by_category: dict[str, dict[str, int]] = {}

    for case in cases:
        query = case["query"]
        entities = [(e["surface"], e["dcid"]) for e in case["entities"]]
        predicted = list(scanner(query, entities))
        if len(predicted) != len(entities):
            sys.exit(
                f"{case['id']}: scanner returned {len(predicted)} verdicts "
                f"for {len(entities)} entities"
            )
        bucket = by_category.setdefault(case["category"], {"total": 0, "correct": 0})
        for entity, axis in zip(case["entities"], predicted, strict=True):
            expected: ExpectedAxis = entity["expected_axis"]
            predicted_label: ExpectedAxis = axis if axis is not None else "unspecified"
            is_correct = predicted_label == expected

            total += 1
            bucket["total"] += 1
            if is_correct:
                correct += 1
                bucket["correct"] += 1
            if expected == "unspecified":
                unspecified_total += 1
                if axis is not None:
                    unspecified_confident += 1
            else:
                directional_total += 1
                opposite = "origin" if expected == "destination" else "destination"
                if predicted_label == opposite:
                    confidently_wrong += 1

            verdicts.append(
                {
                    "case_id": case["id"],
                    "query": query,
                    "surface": entity["surface"],
                    "dcid": entity["dcid"],
                    "expected": expected,
                    "predicted": predicted_label,
                    "correct": is_correct,
                }
            )

    return {
        "accuracy": correct / total if total else 0.0,
        "false_confident_rate": (
            unspecified_confident / unspecified_total if unspecified_total else 0.0
        ),
        "confidently_wrong_count": confidently_wrong,
        "confidently_wrong_rate": (
            confidently_wrong / directional_total if directional_total else 0.0
        ),
        "entity_count": total,
        "unspecified_count": unspecified_total,
        "by_category": by_category,
        "verdicts": verdicts,
    }


def _load_scanner(dotted: str) -> Scanner:
    module_name, _, func_name = dotted.partition(":")
    if not func_name:
        sys.exit(f"--scanner must be 'module:function', got {dotted!r}")
    # Direct execution adds `scripts/` to `sys.path`.
    # Add the repository root for package scanner imports.
    repo_str = str(REPO)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scanner",
        required=True,
        help="module:function, e.g. corpus.direction.role_text_pinned:detect_directions",
    )
    parser.add_argument("--cases", type=Path, default=CASES_YAML)
    parser.add_argument(
        "--verdicts-out",
        type=Path,
        default=None,
        help="write every per-entity verdict as JSON to this path",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    scanner = _load_scanner(args.scanner)
    result = score(cases, scanner)

    print(f"entities scored:            {result['entity_count']}")
    print(f"directional accuracy:       {result['accuracy']:.1%}")
    print(f"unspecified-truth entities: {result['unspecified_count']}")
    print(f"false-confident rate:       {result['false_confident_rate']:.1%}")
    print(f"confidently-wrong count:    {result['confidently_wrong_count']}")
    print(f"confidently-wrong rate:     {result['confidently_wrong_rate']:.1%}")
    print()
    print(f"{'category':<26}{'correct':>10}{'total':>10}")
    for cat, bucket in sorted(result["by_category"].items()):
        print(f"{cat:<26}{bucket['correct']:>10}{bucket['total']:>10}")

    if args.verdicts_out:
        args.verdicts_out.write_text(json.dumps(result["verdicts"], indent=2) + "\n")
        print(f"\nper-entity verdicts written to {args.verdicts_out}")


if __name__ == "__main__":
    main()
