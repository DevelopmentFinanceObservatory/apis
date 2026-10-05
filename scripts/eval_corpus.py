#!/usr/bin/env python3
"""Check the eval case corpus, or print its JSON schema.

`check` loads `corpus/*.yaml` with the families under `corpus/families/` and exits 1 on any
schema error, duplicate id, unknown family, `why` over 200 characters, `known_failure`
naming an unknown check, or required warning code outside the allowlist for the case's door
(`search.eval.assertable_warnings`). It prints each ineligible case with its reasons; ineligibility
alone does not fail. `--corpus-dir` points it at another corpus.

`schema` prints the JSON schema of one case file, generated from `search.eval.cases.Case`.

Requires the eval extra (`uv sync --extra eval`). `just corpus-check` and
`just corpus-schema` wrap the two commands.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

from search.eval.assertable_warnings import check_cases_assertable_warnings
from search.eval.cases import (
    CorpusError,
    corpus_schema,
    ineligibility,
    load_cases,
)

CORPUS_DIR = Path(__file__).parent.parent / "corpus"


def implemented_checks() -> frozenset[str]:
    """The check names `search.eval.checks.CHECKS` registers, or none before it exists."""
    try:
        checks = importlib.import_module("search.eval.checks")
    except ModuleNotFoundError as exc:
        if exc.name != "search.eval.checks":
            raise
        return frozenset()
    return frozenset(checks.CHECKS)


def check(corpus_dir: Path) -> int:
    try:
        cases = load_cases(corpus_dir)
    except CorpusError as exc:
        return _failed(exc.errors)
    violations = check_cases_assertable_warnings(cases)
    if violations:
        return _failed(violations)

    implemented = implemented_checks()
    ineligible = {
        case.id: reasons
        for case in cases
        if (reasons := ineligibility(case, implemented_checks=implemented))
    }
    for case_id, reasons in ineligible.items():
        print(f"ineligible {case_id}: {'; '.join(reasons)}")
    print(
        f"CORPUS CHECK PASSED: {len(cases)} cases, "
        f"{len(cases) - len(ineligible)} eligible, {len(ineligible)} ineligible"
    )
    return 0


def _failed(errors: list[str]) -> int:
    print(f"CORPUS CHECK FAILED ({len(errors)} error(s)):")
    for error in errors:
        print(f"  - {error}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check_parser = sub.add_parser("check", help="validate the corpus and list ineligible cases")
    check_parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    sub.add_parser("schema", help="print the case file JSON schema")
    args = parser.parse_args(argv)
    if args.command == "schema":
        print(json.dumps(corpus_schema(), indent=2))
        return 0
    return check(args.corpus_dir)


if __name__ == "__main__":
    sys.exit(main())
