"""Score the `closest_match` rule against recorded panel responses.

Reads `raw-*.jsonl` panel responses (one `{"id", "query", "body"}` record per line, the
body a full search response), runs `search.engine.closest_match.closest_match_fact` over
every reading each response shows (a definite part's Spec, each Spec of a candidates
part), and compares the result with hand labels. A query is predicted positive when any of
its readings raises the fact.

Each reading's inputs come from the recorded body: `variable_text` and the
`retrieval_score_weak` warning from its part, the measure noun from the authored noun
`search.heading.measure_noun` gives the Spec's shape and shared constraint values, else the
first `what` token of the Spec's own recorded `sentence_plan`, the StatVar names from its
members' labels, and the bound codes from the how-slot bindings and the members'
constraint values.

Responses carry neither a Spec's members nor any `alternateName`, so both come from
snapshot files. The members snapshot maps a `spec_id` to the `members` the Datasets
members door (`POST /api/datasets/{id}/members`) lists for the Spec's `definition`: the
StatVars a mint of it serves. `--fetch-members` writes it from the service at `--base` for
every Spec the responses show. The names snapshot maps a StatVar dcid to every `name` and
`alternateName` it holds. `--fetch-alternate-names` writes it from the graph at
`SEARCH_GRAPH_BASE` (the service root, as the engine reads it) for the StatVars of every
weak reading. A run with neither flag reads the snapshots and touches no network, so it
reruns to the same output.

The labels file maps a query id to `{"closest_match": bool, "basis": "assessment" | "own",
"reason": str}`. A query it does not list is a negative whose basis is the assessment.

Usage:
    uv run python scripts/closest_match_confusion.py --raw-dir RUNS --labels LABELS.json \\
        --members MEMBERS.json --fetch-members
    uv run --env-file .env python scripts/closest_match_confusion.py --raw-dir RUNS \\
        --labels LABELS.json --members MEMBERS.json --alternate-names NAMES.json \\
        --fetch-alternate-names
    uv run python scripts/closest_match_confusion.py --raw-dir RUNS --labels LABELS.json \\
        --members MEMBERS.json --alternate-names NAMES.json --out confusion.md
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx

from search.engine.closest_match import bound_code_names, closest_match_fact
from search.heading.measure_noun import measure_noun
from search.heading.words import content_words

_WEAK = "retrieval_score_weak"
_DCID_PREFIX = "dcid:"
DEFAULT_BASE = "https://dfo-dev-apis-zms5lj4zhq-nn.a.run.app"
TIMEOUT_S = 150


@dataclass(frozen=True)
class Reading:
    """One Spec a recorded part shows, with the rule's inputs read off it."""

    query_id: str
    variable_text: str
    weak: bool
    measure_noun: str
    stat_vars: tuple[tuple[str, str | None], ...]
    bound_codes: tuple[str, ...]


@dataclass(frozen=True)
class Query:
    query_id: str
    query: str
    readings: tuple[Reading, ...]


def _load_records(raw_dir: Path) -> list[dict]:
    paths = sorted(raw_dir.glob("raw-*.jsonl"))
    if not paths:
        raise SystemExit(f"no raw-*.jsonl files under {raw_dir}")
    return [json.loads(line) for path in paths for line in path.read_text().splitlines()]


def _parts(record: dict) -> list[dict]:
    return (record.get("body") or {}).get("parts", [])


def _shown_specs(part: dict) -> list[dict]:
    """The Specs a recorded part shows: a definite part's one, a candidates part's each."""
    if part["status"] == "definite":
        return [part["spec"]]
    if part["status"] == "candidates":
        return part["candidates"]["specs"]
    return []


def _load_queries(records: list[dict], members: dict[str, list[dict]]) -> list[Query]:
    return [
        Query(
            record["id"],
            record["query"],
            tuple(
                reading
                for part in _parts(record)
                for reading in _readings(record["id"], part, members)
            ),
        )
        for record in records
    ]


def _readings(query_id: str, part: dict, members: dict[str, list[dict]]) -> Iterator[Reading]:
    weak = any(warning["code"] == _WEAK for warning in part.get("warnings", []))
    for spec in _shown_specs(part):
        resolved = spec["resolved"]
        if spec["spec_id"] not in members:
            raise SystemExit(
                f"the members snapshot holds no {spec['spec_id']} (query {query_id}); "
                "write it with --fetch-members"
            )
        items = members[spec["spec_id"]]
        what = [token["text"] for token in resolved["sentence_plan"] if token.get("axis") == "what"]
        codes = [
            value["ref"]["ref"]
            for slot in resolved["slots"]
            if slot["key"]["axis"] == "how"
            for value in _binding_values(slot["binding"])
        ]
        codes.extend(value for item in items for value in item["constraints"].values() if value)
        noun = _authored_noun(resolved["shape"], items)
        yield Reading(
            query_id=query_id,
            variable_text=part["variable_text"],
            weak=weak,
            measure_noun=noun or (what[0] if what else resolved["shape"]["label"]),
            stat_vars=tuple(
                (item["ref"].removeprefix(_DCID_PREFIX), item["label"]) for item in items
            ),
            bound_codes=tuple(dict.fromkeys(codes)),
        )


def _authored_noun(shape: dict, items: list[dict]) -> str | None:
    """The authored measure noun the heading shows for `shape`, from the
    constraint values every member in `items` shares."""
    bound = [
        {prop: value for prop, value in item["constraints"].items() if value} for item in items
    ]
    shared = (
        {
            prop: value
            for prop, value in bound[0].items()
            if all(other.get(prop) == value for other in bound)
        }
        if bound
        else {}
    )
    noun = measure_noun(
        population_type=(shape.get("population_type") or {}).get("ref"),
        measured_property=(shape.get("measured_property") or {}).get("ref"),
        stat_type=(shape.get("stat_type") or {}).get("ref"),
        constraints=shared,
    )
    return noun.noun if noun else None


def _binding_values(binding: dict) -> list[dict]:
    if binding["kind"] == "value":
        return [binding["value"]]
    if binding["kind"] == "set":
        return list(binding["values"])
    return []


def _stat_var_names(reading: Reading, alternate_names: dict[str, list[str]]) -> list[str]:
    return [
        name
        for dcid, label in reading.stat_vars
        for name in (label, *alternate_names.get(dcid, ()))
        if name
    ]


def _fires(reading: Reading, alternate_names: dict[str, list[str]]) -> bool:
    fact = closest_match_fact(
        variable_text=reading.variable_text,
        retrieval_score_weak=reading.weak,
        measure_noun=reading.measure_noun,
        stat_var_names=_stat_var_names(reading, alternate_names),
        bound_codes=reading.bound_codes,
    )
    return fact is not None


def _shared_words(reading: Reading, alternate_names: dict[str, list[str]]) -> list[str]:
    """The content words of a weak reading's variable text that some name of its measure
    holds, each with the first name that holds it. These words keep the rule silent."""
    if not reading.weak:
        return []
    names = [
        reading.measure_noun,
        *_stat_var_names(reading, alternate_names),
        *bound_code_names(reading.bound_codes),
    ]
    shared: dict[str, str] = {}
    for word in sorted(content_words(reading.variable_text)):
        holder = next((name for name in names if word in content_words(name)), None)
        if holder is not None:
            shared[word] = holder
    return [f"{word!r} in {holder!r}" for word, holder in shared.items()]


def _fetch_members(records: list[dict], path: Path, *, base: str) -> None:
    definitions = {
        spec["spec_id"]: spec["definition"]
        for record in records
        for part in _parts(record)
        for spec in _shown_specs(part)
    }
    if not definitions:
        raise SystemExit("the responses show no Spec; nothing to fetch")
    members: dict[str, list[dict]] = {}
    with httpx.Client(base_url=base.rstrip("/"), timeout=TIMEOUT_S) as client:
        for spec_id, definition in sorted(definitions.items()):
            response = client.post(
                f"/api/datasets/{spec_id}/members",
                json={"schema_version": 1, "definition": definition},
            )
            if response.status_code != 200:
                raise SystemExit(
                    f"the members door refused {spec_id}: {response.status_code} {response.text}"
                )
            members[spec_id] = response.json()["members"]
    path.write_text(json.dumps(members, indent=1) + "\n")
    print(f"wrote the members of {len(members)} Specs to {path}")


def _fetch_alternate_names(queries: list[Query], path: Path) -> None:
    from dc_graph.client import GraphClient

    base = os.environ.get("SEARCH_GRAPH_BASE", "").strip()
    if not base:
        raise SystemExit("SEARCH_GRAPH_BASE must be set to fetch alternate names")
    dcids = sorted(
        {dcid for q in queries for r in q.readings if r.weak for dcid, _label in r.stat_vars}
    )
    if not dcids:
        raise SystemExit("no weak reading names a StatVar; nothing to fetch")
    client = GraphClient(
        base_url=f"{base.rstrip('/')}/core/api", token=os.environ.get("SEARCH_GRAPH_TOKEN") or None
    )
    names = client.node_names_many(dcids)
    path.write_text(json.dumps({dcid: list(names[dcid]) for dcid in dcids}, indent=1) + "\n")
    print(f"wrote names for {len(dcids)} StatVars to {path}")


def _report(queries: list[Query], labels: dict, alternate_names: dict[str, list[str]]) -> str:
    unknown = sorted(set(labels) - {q.query_id for q in queries})
    if unknown:
        raise SystemExit(f"labels name queries the responses do not hold: {unknown}")
    rows = []
    for q in queries:
        label = labels.get(q.query_id, {})
        expected = bool(label.get("closest_match", False))
        predicted = any(_fires(r, alternate_names) for r in q.readings)
        weak = any(r.weak for r in q.readings)
        rows.append((q, expected, predicted, weak, label))

    def count(expected: bool, predicted: bool) -> int:
        return sum(1 for _q, e, p, _w, _l in rows if e == expected and p == predicted)

    tp, fp, fn, tn = count(True, True), count(False, True), count(True, False), count(False, False)
    precision = f"{tp / (tp + fp):.2f}" if tp + fp else "n/a"
    recall = f"{tp / (tp + fn):.2f}" if tp + fn else "n/a"
    own = sum(1 for *_rest, label in rows if label.get("basis") == "own")
    lines = [
        "# Closest-match rule: confusion over the recorded panel",
        "",
        f"{len(rows)} queries, {sum(1 for r in rows if r[3])} with a weak shown reading. "
        f"{own} labels are the labeller's own reading of the query and its assessment; every "
        "other label follows the assessment, and a query the labels file omits is a negative.",
        "",
        "| | labelled closest match | labelled not |",
        "| --- | --- | --- |",
        f"| rule fires | {tp} (TP) | {fp} (FP) |",
        f"| rule silent | {fn} (FN) | {tn} (TN) |",
        "",
        f"Precision {precision}, recall {recall}.",
        "",
    ]
    for title, wanted in (
        ("True positives", (True, True)),
        ("False positives", (False, True)),
        ("False negatives", (True, False)),
    ):
        lines += [f"## {title}", ""]
        selected = [r for r in rows if (r[1], r[2]) == wanted]
        if not selected:
            lines += ["None.", ""]
            continue
        lines += [
            "| id | query | weak | label basis | reason | readings shown | shared words |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for q, _e, _p, weak, label in selected:
            shown = "; ".join(
                dict.fromkeys(f"{r.variable_text!r} -> {r.measure_noun}" for r in q.readings)
            )
            shared = "; ".join(
                dict.fromkeys(w for r in q.readings for w in _shared_words(r, alternate_names))
            )
            lines.append(
                f"| {q.query_id} | {q.query} | {'yes' if weak else 'no'} | "
                f"{label.get('basis', 'assessment')} | {label.get('reason', '')} | {shown} | "
                f"{shared} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--members", type=Path, required=True)
    parser.add_argument("--fetch-members", action="store_true")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--alternate-names", type=Path)
    parser.add_argument("--fetch-alternate-names", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    records = _load_records(args.raw_dir)
    if args.fetch_members:
        _fetch_members(records, args.members, base=args.base)
        return
    if args.alternate_names is None:
        parser.error("--alternate-names is required without --fetch-members")
    queries = _load_queries(records, json.loads(args.members.read_text()))
    if args.fetch_alternate_names:
        _fetch_alternate_names(queries, args.alternate_names)
        return
    alternate_names = json.loads(args.alternate_names.read_text())
    report = _report(queries, json.loads(args.labels.read_text()), alternate_names)
    if args.out is None:
        print(report)
    else:
        args.out.write_text(report + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
