#!/usr/bin/env python3
"""Trim a sentence panel run into the golden `tests/render/test_sentence_golden.py` reads.

The sentence panel (`scripts/sentence_panel/run.py`) writes one `raw-<slice>.jsonl` per slice,
each line a query and the full `/api/search/resolve` response. This script keeps what the
sentence renderer reads and the plans it wrote, and writes
`tests/fixtures/sentence_panel/<slice>.jsonl.gz`, one JSON line per query:

- `id`, `query`: the panel query.
- `rendered_sentence`, `sentence_plan`: the envelope sentence and plan.
- `parts`: each part with its status, `variable_text`, warnings, departures and own
  sentence; a definite part's Spec, a candidates part's shown Specs, a no-data part's reason.

A Spec keeps its definition and the resolved fields `render_sentence_plan` reads (shape,
slots, StatVars, entities and groups, coverage, scope, facets, slot options, open-axis and
role-axis facts, measure label, the trace's sources) with its recorded sentence. A StatVar
keeps its ref, label, constraint properties and served date range. Observations, timing,
diagnostics, skills, entity and facet options and the rest of the trace go; the model fields
that are required but unread (`slot_values`, the trace's StatVar, entity, filter and step
lists) are written empty, and a role-axis fact's `member_count`, which the wire omits when
it is null, is restored as null. A no-data part keeps only its reason, since the envelope
composer reads no other `no_data` field.

The files are gzipped. A panel run's StatVar lists alone hold about 3 MB of JSON, and the
182-query golden is about 400 kB compressed against 5.7 MB plain. Gzip runs with a zero
mtime, so the same run writes the same bytes. To review a re-recording, the script prints
each query whose envelope sentence differs from the golden it replaces, old then new.

Every trimmed line is parsed with the response models before it is written. The script exits
non-zero on a non-200 row, a non-resolution body or an empty raw file, and writes nothing
for a slice that fails.

`--rerender` rewrites the recorded plans of the committed golden with the current renderer
and leaves every input as recorded, so a renderer change updates the golden with no live
panel run. It re-renders what the test compares: each definite Spec's plan, each candidates
part's candidate plans and its own plan, and the envelope plan, with the sentence of each.
A no-data part's own plan needs inputs the wire does not carry, so it stays as recorded.

Usage:
    just sentence-golden runs/sentence-panel
    uv run python scripts/sentence_golden.py runs/sentence-panel [--out DIR]
    uv run python scripts/sentence_golden.py --rerender [--out DIR]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from search.models import CandidatesPart, DefinitePart, SentenceToken, Spec
from search.models.response import Part
from search.render.candidates import candidate_differences, candidates_sentence_plan
from search.render.envelope import compose_envelope_sentence
from search.render.heading import declared_axis_names, sibling_how_properties
from search.render.plan import render_sentence_plan, sentence_from_plan

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "tests" / "fixtures" / "sentence_panel"

_PART_KEYS = (
    "status",
    "variable_text",
    "warnings",
    "departures",
    "rendered_sentence",
    "sentence_plan",
)
_RESOLVED_KEYS = (
    "shape",
    "slots",
    "stat_vars",
    "entities",
    "entity_groups",
    "coverage",
    "scope",
    "facets",
    "slot_options",
    "open_axis_facts",
    "role_axis_facts",
    "measure_label",
    "measure_label_axes",
    "measure_label_named_values",
    "rendered_sentence",
    "sentence_plan",
)
_STAT_VAR_KEYS = ("ref", "label", "constraint_properties", "data_date_range")

_PARTS = TypeAdapter(list[Part])
_PLAN = TypeAdapter(list[SentenceToken])


def _pick(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: source[key] for key in keys if key in source}


def trim_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """`spec` with only what the renderer reads and the sentence it recorded."""
    resolved = spec["resolved"]
    trimmed = _pick(resolved, _RESOLVED_KEYS)
    stat_vars = resolved["stat_vars"]
    trimmed["stat_vars"] = {
        **{key: value for key, value in stat_vars.items() if key != "items"},
        "items": [
            {**_pick(item, _STAT_VAR_KEYS), "slot_values": []} for item in stat_vars["items"]
        ],
    }
    trimmed["trace"] = {
        "resolved_sources": resolved["trace"]["resolved_sources"],
        "resolved_stat_vars": [],
        "resolved_entities": [],
        "slot_filters": [],
        "pipeline_trace": [],
    }
    if facts := resolved.get("role_axis_facts"):
        # The wire omits null fields and `RoleAxisFact.member_count` is required but
        # nullable, so an unknown count comes back as an absent key.
        trimmed["role_axis_facts"] = [{"member_count": None, **fact} for fact in facts]
    return {"spec_id": spec["spec_id"], "definition": spec["definition"], "resolved": trimmed}


def trim_part(part: dict[str, Any]) -> dict[str, Any]:
    """`part` with its own sentence, departures and warnings, and the inputs the renderer
    reads for its status."""
    trimmed = _pick(part, _PART_KEYS)
    match part["status"]:
        case "definite":
            trimmed["spec"] = trim_spec(part["spec"])
        case "candidates":
            candidates = part["candidates"]
            trimmed["candidates"] = {
                **{key: value for key, value in candidates.items() if key != "specs"},
                "specs": [trim_spec(spec) for spec in candidates["specs"]],
            }
        case "no_data":
            trimmed["no_data"] = {"reason": part["no_data"]["reason"]}
        case "unsupported":
            trimmed["unsupported"] = part["unsupported"]
    return trimmed


def trim_row(row: dict[str, Any]) -> dict[str, Any]:
    """One panel row as a golden line, parsed with the response models before it returns."""
    body = row.get("body")
    if row.get("http_status") != 200 or not isinstance(body, dict):
        raise ValueError(f"{row.get('id')}: HTTP {row.get('http_status')}, no body to record")
    if body.get("kind") != "resolution":
        raise ValueError(f"{row['id']}: body kind {body.get('kind')!r} is not a resolution")
    line = {
        "id": row["id"],
        "query": row["query"],
        "rendered_sentence": body["rendered_sentence"],
        "sentence_plan": body["sentence_plan"],
        "parts": [trim_part(part) for part in body["parts"]],
    }
    _PARTS.validate_python(line["parts"])
    _PLAN.validate_python(line["sentence_plan"] or [])
    return line


def read_golden(path: Path) -> list[dict[str, Any]]:
    """The lines of one golden file."""
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]


def changed_sentences(
    old: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[tuple[str, str | None, str | None]]:
    """Each query id whose envelope sentence differs between `old` and `new`, with both
    sentences. An id on one side only reads None on the other."""
    before = {line["id"]: line["rendered_sentence"] for line in old}
    after = {line["id"]: line["rendered_sentence"] for line in new}
    return [
        (query_id, before.get(query_id), after.get(query_id))
        for query_id in sorted(before.keys() | after.keys())
        if before.get(query_id) != after.get(query_id)
    ]


def write_golden(path: Path, lines: list[dict[str, Any]]) -> None:
    """Write golden lines to `path`, gzipped with a zero mtime."""
    text = "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines)
    path.write_bytes(gzip.compress(text.encode(), compresslevel=9, mtime=0))


def trim_file(raw: Path, out: Path) -> list[dict[str, Any]]:
    """Write the golden for one `raw-<slice>.jsonl` to `out` and return its lines."""
    rows = [json.loads(text) for text in raw.read_text().splitlines() if text.strip()]
    if not rows:
        raise ValueError(f"{raw}: no rows")
    lines = [trim_row(row) for row in rows]
    write_golden(out, lines)
    return lines


def rerendered_candidates(specs: list[Spec]) -> list[list[SentenceToken]]:
    """Each candidate's plan as the candidates part renders it: siblings keep the how
    properties that set them apart, and the tokens that differ are marked."""
    declarations = declared_axis_names()
    keep = sibling_how_properties(specs)
    rendered = [
        spec.model_copy(
            update={
                "resolved": spec.resolved.model_copy(
                    update={
                        "sentence_plan": render_sentence_plan(
                            spec, keep_how_properties=keep, axis_declarations=declarations
                        )
                    }
                )
            }
        )
        for spec in specs
    ]
    candidate_differences(rendered, axis_declarations=declarations)
    return [spec.resolved.sentence_plan or [] for spec in rendered]


def _set_plan(target: dict[str, Any], plan: list[SentenceToken]) -> None:
    """Store `plan` and its sentence on `target` as the wire writes them."""
    target["sentence_plan"] = [token.model_dump(mode="json", exclude_none=True) for token in plan]
    target["rendered_sentence"] = sentence_from_plan(plan)


def rerender_line(line: dict[str, Any]) -> dict[str, Any]:
    """`line` with every plan the golden test compares re-rendered by the current renderer
    and every input unchanged."""
    parts = _PARTS.validate_python(line["parts"])
    declarations = declared_axis_names()
    for part, raw in zip(parts, line["parts"], strict=True):
        if isinstance(part, DefinitePart):
            plan = render_sentence_plan(part.spec, axis_declarations=declarations)
            _set_plan(raw["spec"]["resolved"], plan)
        elif isinstance(part, CandidatesPart):
            specs = part.candidates.specs
            raw_specs = raw["candidates"]["specs"]
            for raw_spec, plan in zip(raw_specs, rerendered_candidates(specs), strict=True):
                _set_plan(raw_spec["resolved"], plan)
            _set_plan(
                raw,
                candidates_sentence_plan(
                    specs, truncated=part.candidates.truncated, departures=part.departures
                ),
            )
    composed = compose_envelope_sentence(_PARTS.validate_python(line["parts"]))
    if composed is None:
        line["rendered_sentence"], line["sentence_plan"] = None, None
    else:
        _set_plan(line, composed[1])
    return line


def rerender_golden(out: Path) -> int:
    """Re-render every golden file under `out` in place and print each changed envelope
    sentence. Return the number of queries."""
    paths = sorted(out.glob("*.jsonl.gz"))
    if not paths:
        raise ValueError(f"no golden files under {out}")
    total = 0
    for path in paths:
        old = read_golden(path)
        lines = [rerender_line(line) for line in read_golden(path)]
        write_golden(path, lines)
        total += len(lines)
        print(f"{path}: {len(lines)} queries re-rendered")
        for query_id, before, after in changed_sentences(old, lines):
            print(f"  {query_id}\n    - {before}\n    + {after}")
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, nargs="?", help="a sentence panel output directory")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--rerender",
        action="store_true",
        help="re-render the committed golden's plans with the current renderer",
    )
    args = parser.parse_args(argv)
    if args.rerender:
        print(f"{rerender_golden(args.out)} queries")
        return 0
    if args.run is None:
        parser.error("a sentence panel output directory is required without --rerender")
    raws = sorted(args.run.glob("raw-*.jsonl"))
    if not raws:
        print(f"no raw-*.jsonl files in {args.run}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for raw in raws:
        out = args.out / f"{raw.stem.removeprefix('raw-')}.jsonl.gz"
        old = read_golden(out) if out.exists() else []
        try:
            lines = trim_file(raw, out)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
        total += len(lines)
        print(f"{out}: {len(lines)} queries, {out.stat().st_size:,} bytes")
        for query_id, before, after in changed_sentences(old, lines):
            print(f"  {query_id}\n    - {before}\n    + {after}")
    print(f"{len(raws)} slices, {total} queries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
