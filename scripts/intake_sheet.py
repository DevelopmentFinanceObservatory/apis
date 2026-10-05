"""Parse a golden-submission CSV into staging YAML for maintainer review.

The expected-answer cell uses `[candidates: ]<dcid>[, ...]; places: <place>[, ...][; date: <date>]`.
DCIDs omit `dcid:`. Dates are `YYYY` or `YYYY-YYYY`. No-data and refuse-redirect rows leave the
cell blank. Malformed answers are rejected.

Origin, answer-verification method, independence, and contributor fields have no schema fields.
Each record gets `status: UNVERIFIED` because the sheet's trust rating may come from the system
under test. Difficulty, source/area, notes, multi-turn and follow-up details, and date feed the
record's notes.
Maintainers fill expected shape, slots, and no-data details from a live engine run. Statvars and
places come from the sheet.

Usage:
    scripts/intake_sheet.py parse <csv> --out corpus/staging/<name>.yaml

`--out` must be outside `corpus/` itself. The compiler reads only `corpus/*.yaml`, so staging
subdirectories stay out of the corpus.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).parent.parent
CORPUS_DIR = REPO / "corpus"

# The 14 columns this script reads, in sheet order. Everything past column 14
# in a real export is spreadsheet-tool padding (blank trailing columns).
N_SHEET_COLUMNS = 14

BEHAVIOUR_TO_STATUS = {
    "value": "definite",
    "candidates": "candidates",
    "no-data": "no_data",
    "answer-redirect": "definite",
}
# refuse-redirect is parked: it has no importable expected_status.
PARKED_BEHAVIOURS = frozenset({"refuse-redirect"})

TRUST_TIER = "UNVERIFIED"

_DATE_RE = re.compile(r"^\d{4}(-\d{4})?$")
_SLUG_WORD_RE = re.compile(r"[a-z0-9]+")
_MAX_SLUG_WORDS = 8

# Function words dropped when deriving an id slug from the question text, so
# two questions differing only in grammar (e.g. "GDP in X" vs "GDP of X")
# don't collapse to the same slug just because both keep every word.
_SLUG_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "did",
        "do",
        "does",
        "for",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "s",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "which",
        "who",
        "why",
        "will",
    }
)

# Exact-token corrections for known invalid place and StatVar dcids.
CORRECTIONS: dict[str, str] = {
    # Vietnam's ISO 3166-1 alpha-3 place dcid is `country/VNM`.
    "country/VTN": "country/VNM",
    # WHO StatVar dcids use a lowercase namespace.
    "WHO/mslv": "who/mslv",
}


class CellParseError(ValueError):
    """Raised when an Expected-answer cell fails the grammar.

    The message gives the reject reason.
    """


@dataclass(frozen=True)
class ParsedAnswer:
    """One cleanly-parsed Expected-answer cell."""

    is_candidates: bool
    stat_vars: list[str]
    places: list[str]
    date: str | None


def parse_expected_answer(cell: str) -> ParsedAnswer:
    """Parse one Expected-answer cell, raising `CellParseError` with the invalid segment.

    Reject dates inside `places:` and trailing years without a `date:` label.
    """
    cell = cell.strip()
    if not cell:
        return ParsedAnswer(is_candidates=False, stat_vars=[], places=[], date=None)

    segments = [s.strip() for s in cell.split(";")]
    first, rest = segments[0], segments[1:]

    is_candidates = first.lower().startswith("candidates:")
    if is_candidates:
        first = first.split(":", 1)[1].strip()

    stat_vars = [d.strip() for d in first.split(",") if d.strip()]
    if not stat_vars:
        raise CellParseError(f"no stat_var dcid(s) found before the first ';' in {cell!r}")

    places: list[str] | None = None
    date: str | None = None
    for seg in rest:
        low = seg.lower()
        if low.startswith("places:"):
            raw_places = [p.strip() for p in seg.split(":", 1)[1].split(",") if p.strip()]
            leaked = [p for p in raw_places if _DATE_RE.match(p)]
            if leaked:
                raise CellParseError(
                    f"places segment {seg!r} contains a bare date {leaked!r} that leaked "
                    "out of a date: field"
                )
            places = raw_places
        elif low.startswith("date:"):
            value = seg.split(":", 1)[1].strip()
            if not _DATE_RE.match(value):
                raise CellParseError(
                    f"date segment {seg!r} is not a year or year-range (YYYY or YYYY-YYYY)"
                )
            date = value
        elif _DATE_RE.match(seg):
            raise CellParseError(
                f"trailing segment {seg!r} looks like a date but is missing the 'date:' label"
            )
        else:
            raise CellParseError(f"unrecognized segment in expected-answer cell: {seg!r}")

    if places is None:
        raise CellParseError(f"no 'places:' segment found in {cell!r}")
    if not places:
        raise CellParseError(f"'places:' segment in {cell!r} has no place dcid(s)")

    return ParsedAnswer(is_candidates=is_candidates, stat_vars=stat_vars, places=places, date=date)


def domain_for(stat_vars: list[str]) -> str:
    """Return `sdg` for an `sdg/` prefix and `standard` for every other prefix."""
    if any(sv.startswith("sdg/") for sv in stat_vars):
        return "sdg"
    return "standard"


def apply_corrections(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Apply exact-token corrections and return corrected tokens with descriptions."""
    corrected = []
    applied = []
    for token in tokens:
        fix = CORRECTIONS.get(token)
        if fix is None:
            corrected.append(token)
            continue
        applied.append(f"corrected {token!r} to {fix!r} (known sheet data error)")
        corrected.append(fix)
    return corrected, applied


def slugify_id(question: str) -> str:
    """Build a deterministic kebab-case id from at most eight question words.

    Drop punctuation and stopwords, but keep all words if stopword removal empties the question.
    """
    words = _SLUG_WORD_RE.findall(question.lower())
    content_words = [w for w in words if w not in _SLUG_STOPWORDS] or words
    return "-".join(content_words[:_MAX_SLUG_WORDS])


@dataclass(frozen=True)
class ParsedGolden:
    row_index: int
    question: str
    record: dict
    corrections: list[str]
    collision_with: str | None


@dataclass(frozen=True)
class RowIssue:
    row_index: int
    question: str
    reason: str


@dataclass(frozen=True)
class ParseResult:
    parsed: list[ParsedGolden] = field(default_factory=list)
    rejected: list[RowIssue] = field(default_factory=list)
    parked: list[RowIssue] = field(default_factory=list)


def _sheet_fields(row: list[str]) -> dict[str, str]:
    """Unpack one CSV row into the 14 named columns, tolerating short/padded rows."""
    padded = row[:N_SHEET_COLUMNS] + [""] * max(0, N_SHEET_COLUMNS - len(row))
    names = (
        "sheet_id",
        "question",
        "behaviour",
        "raw_cell",
        "multi_turn",
        "follow_up",
        "how_established",
        "independent",
        "status_trust",
        "origin",
        "difficulty",
        "source_area",
        "notes",
        "contributor",
    )
    return {name: value.strip() for name, value in zip(names, padded, strict=True)}


def _missing_fields_note(expected_status: str, *, behaviour: str) -> str:
    """Name the schema fields a CSV cannot support, tied to which ones matter for this row."""
    missing = ["expected_slots"]
    if expected_status == "definite":
        missing.insert(0, "expected_shape")
    if expected_status == "no_data":
        missing += ["expected_no_data_reason", "expected_no_data_category"]
    if behaviour == "answer-redirect":
        # The question asks for a comparison, ranking or delta the caller computes.
        missing.append("expected_operations")
    return f"{', '.join(missing)} unset -- fill in from a live engine run before promoting."


def _build_notes(
    *,
    expected_status: str,
    behaviour: str,
    difficulty: str,
    source_area: str,
    sheet_notes: str,
    multi_turn: str,
    follow_up: str,
    date: str | None,
    corrections: list[str],
    collision_with: str | None,
) -> str:
    """Combine sheet notes with parse details the golden schema has no dedicated fields for."""
    parts = []
    if collision_with:
        parts.append(f"ID COLLISION: this id is already used by {collision_with}.")
    if corrections:
        parts.append(f"Corrections applied: {'; '.join(corrections)}.")
    if difficulty:
        parts.append(f"Sheet difficulty: {difficulty}.")
    if source_area:
        parts.append(f"Sheet source/area: {source_area}.")
    if date:
        parts.append(f"Sheet date: {date}.")
    if multi_turn:
        parts.append(f"Sheet multi-turn: {multi_turn}.")
    if follow_up:
        parts.append(f"Sheet follow-up turns & resolution: {follow_up}")
    if sheet_notes:
        parts.append(f"Sheet notes: {sheet_notes}")
    parts.append(_missing_fields_note(expected_status, behaviour=behaviour))
    return " ".join(parts)


def _process_row(row_index: int, row: list[str], seen_ids: set[str]) -> ParsedGolden | RowIssue:
    """Return a golden or a row issue with its index and reason.

    Refuse-redirect issues are parked. Other issues are rejected.
    """
    f = _sheet_fields(row)
    question = f["question"]
    behaviour = f["behaviour"]

    if behaviour in PARKED_BEHAVIOURS:
        return RowIssue(row_index, question, f"parked: {behaviour!r} is not importable")

    if behaviour not in BEHAVIOUR_TO_STATUS:
        return RowIssue(row_index, question, f"unknown behaviour value {behaviour!r}")

    expected_status = BEHAVIOUR_TO_STATUS[behaviour]

    if behaviour == "no-data":
        if f["raw_cell"]:
            return RowIssue(
                row_index,
                question,
                f"no-data row has a non-empty expected-answer cell (expected blank): "
                f"{f['raw_cell']!r}",
            )
        stat_vars, places, date = [], [], None
    else:
        try:
            parsed = parse_expected_answer(f["raw_cell"])
        except CellParseError as exc:
            return RowIssue(row_index, question, str(exc))

        if not f["raw_cell"].strip():
            return RowIssue(
                row_index,
                question,
                f"behaviour is {behaviour!r} but the expected-answer cell is blank",
            )
        if behaviour == "candidates" and not parsed.is_candidates:
            return RowIssue(
                row_index,
                question,
                "behaviour is 'candidates' but the expected-answer cell has no leading "
                "'candidates:' marker",
            )
        if behaviour != "candidates" and parsed.is_candidates:
            return RowIssue(
                row_index,
                question,
                f"behaviour is {behaviour!r} but the expected-answer cell has a leading "
                "'candidates:' marker",
            )
        if behaviour == "candidates" and len(parsed.stat_vars) < 2:
            return RowIssue(
                row_index,
                question,
                f"candidates row has only {len(parsed.stat_vars)} stat_var(s); need >= 2",
            )
        stat_vars, places, date = parsed.stat_vars, parsed.places, parsed.date

    stat_vars, sv_fixes = apply_corrections(stat_vars)
    places, place_fixes = apply_corrections(places)
    corrections = sv_fixes + place_fixes

    domain = domain_for(stat_vars) if behaviour != "no-data" else "standard"

    rid = slugify_id(question)
    collision_with = None
    if rid in seen_ids:
        collision_with = f"another golden id {rid!r} already present in the corpus or this batch"
    seen_ids.add(rid)

    tags = [
        {"behaviour": expected_status},
        {"domain": domain},
        {"directional_flow": "na"},
        {"conjunction": "none"},
    ]

    entities = [
        {"dcid": place, "role_kind": "subject", "role_axis": None, "role_dcid": None}
        for place in places
    ]

    record = {
        "id": rid,
        "query": question,
        "entry_path": "raw_text",
        "tags": tags,
        "expected_status": expected_status,
        "expected_shape": None,
        "expected_slots": [],
        "expected_stat_vars": stat_vars,
        "expected_entities": entities,
        "expected_no_data_reason": None,
        "expected_no_data_category": None,
        "candidate_count": len(stat_vars) if expected_status == "candidates" else None,
        "status": TRUST_TIER,
        "notes": _build_notes(
            expected_status=expected_status,
            behaviour=behaviour,
            difficulty=f["difficulty"],
            source_area=f["source_area"],
            sheet_notes=f["notes"],
            multi_turn=f["multi_turn"],
            follow_up=f["follow_up"],
            date=date,
            corrections=corrections,
            collision_with=collision_with,
        ),
    }
    return ParsedGolden(row_index, question, record, corrections, collision_with)


def parse_rows(rows: list[list[str]], *, existing_ids: set[str]) -> ParseResult:
    """Process data rows without the header, checking collisions against existing and in-batch ids.

    Copy `existing_ids` before tracking in-batch ids.
    """
    result = ParseResult()
    seen_ids = set(existing_ids)
    for row_index, row in enumerate(rows, start=1):
        outcome = _process_row(row_index, row, seen_ids)
        if isinstance(outcome, ParsedGolden):
            result.parsed.append(outcome)
        elif outcome.reason.startswith("parked:"):
            result.parked.append(outcome)
        else:
            result.rejected.append(outcome)
    return result


def _load_existing_ids() -> set[str]:
    """Return ids from top-level corpus YAML files for collision detection."""
    ids: set[str] = set()
    for path in sorted(CORPUS_DIR.glob("*.yaml")):
        records = yaml.safe_load(path.read_text()) or []
        ids.update(r["id"] for r in records)
    return ids


def _validate_out_path(out: Path) -> None:
    """Keep staging output out of the compiler's non-recursive `corpus/*.yaml` glob."""
    if out.resolve().parent == CORPUS_DIR.resolve():
        sys.exit(
            f"refusing to write directly into {CORPUS_DIR} -- write to a subdirectory "
            "such as corpus/staging/ instead"
        )


def _read_csv_rows(csv_path: Path) -> list[list[str]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        sys.exit(f"{csv_path}: empty file, no header row")
    return rows[1:]


def _print_report(result: ParseResult) -> None:
    print(
        f"parsed {len(result.parsed)}, rejected {len(result.rejected)}, "
        f"parked {len(result.parked)} (of {len(result.parsed) + len(result.rejected) + len(result.parked)} rows)"
    )

    corrections = [(g.row_index, g.question, c) for g in result.parsed for c in g.corrections]
    if corrections:
        print(f"\ncorrections applied ({len(corrections)}):")
        for row_index, question, detail in corrections:
            print(f"  row {row_index} ({question!r}): {detail}")

    collisions = [g for g in result.parsed if g.collision_with]
    if collisions:
        print(f"\nID COLLISIONS ({len(collisions)}) -- resolve before promoting:")
        for g in collisions:
            print(
                f"  row {g.row_index} ({g.question!r}): id {g.record['id']!r} -- {g.collision_with}"
            )

    if result.rejected:
        print(f"\nrejected ({len(result.rejected)}):")
        for issue in result.rejected:
            print(f"  row {issue.row_index} ({issue.question!r}): {issue.reason}")

    if result.parked:
        print(f"\nparked ({len(result.parked)}):")
        for issue in result.parked:
            print(f"  row {issue.row_index} ({issue.question!r}): {issue.reason}")


def cmd_parse(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    out_path = Path(args.out)
    _validate_out_path(out_path)

    rows = _read_csv_rows(csv_path)
    result = parse_rows(rows, existing_ids=_load_existing_ids())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = [g.record for g in result.parsed]
    out_path.write_text(yaml.safe_dump(records, sort_keys=False, allow_unicode=True))
    print(f"wrote {out_path}: {len(records)} record(s)\n")

    _print_report(result)

    if result.rejected:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_cmd = subparsers.add_parser("parse", help="Parse a sheet CSV export into a staging YAML")
    parse_cmd.add_argument("csv", help="Path to the sheet's CSV export")
    parse_cmd.add_argument(
        "--out", required=True, help="Output staging YAML path (must not be directly in corpus/)"
    )
    parse_cmd.set_defaults(func=cmd_parse)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
