"""Score the `closest_match` rule against recorded panel responses.

Reads `raw-*.jsonl` panel responses (one `{"id", "query", "body"}` record per line, the
body a full search response), runs `search.engine.closest_match.closest_match_fact` over
every reading each response shows (a definite part's Spec, each Spec of a candidates
part), and compares the result with hand labels. A query is predicted positive when any of
its readings raises the fact.

Each reading's inputs come from the recorded body: `variable_text` and the
`retrieval_score_weak` warning from its part, the measure noun from the authored noun
`search.heading.measure_noun` gives the Spec's shape and shared constraint values, else the
first `what` token of the Spec's own recorded `sentence_plan`, the StatVar names from
`resolved.stat_vars.items[].label`, and the bound codes from the how-slot bindings and the
StatVars' `slot_values`. Responses carry no `alternateName`, so those come from a
snapshot file mapping StatVar dcid to every `name` and `alternateName` it holds.
`--fetch-alternate-names` writes that snapshot from the graph at `SEARCH_GRAPH_BASE`
(the service root, as the engine reads it) for the StatVars of every weak reading. A run
without it reads the snapshot and touches no network, so it reruns to the same output.

The labels file maps a query id to `{"closest_match": bool, "basis": "assessment" | "own",
"reason": str}`. A query it does not list is a negative whose basis is the assessment.

Usage:
    uv run python scripts/closest_match_confusion.py --raw-dir RUNS --labels LABELS.json \\
        --alternate-names NAMES.json --out confusion.md
    uv run --env-file .env python scripts/closest_match_confusion.py --raw-dir RUNS \\
        --labels LABELS.json --alternate-names NAMES.json --fetch-alternate-names
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from search.engine.closest_match import bound_code_names, closest_match_fact
from search.heading.measure_noun import measure_noun
from search.heading.words import content_words

_WEAK = "retrieval_score_weak"


@dataclass(frozen=True)
class Reading:
    """One Spec a recorded part shows, with the rule's inputs read off it."""

    query_id: str
    variable_text: str
    weak: bool
    measure_noun: str
    stat_vars: tuple[tuple[str, str], ...]
    bound_codes: tuple[str, ...]


@dataclass(frozen=True)
class Query:
    query_id: str
    query: str
    readings: tuple[Reading, ...]


def _load_queries(raw_dir: Path) -> list[Query]:
    paths = sorted(raw_dir.glob("raw-*.jsonl"))
    if not paths:
        raise SystemExit(f"no raw-*.jsonl files under {raw_dir}")
    queries: list[Query] = []
    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            body = record.get("body") or {}
            readings = tuple(
                reading
                for part in body.get("parts", [])
                for reading in _readings(record["id"], part)
            )
            queries.append(Query(record["id"], record["query"], readings))
    return queries


def _readings(query_id: str, part: dict) -> Iterator[Reading]:
    if part["status"] == "definite":
        specs = [part["spec"]]
    elif part["status"] == "candidates":
        specs = part["candidates"]["specs"]
    else:
        return
    weak = any(warning["code"] == _WEAK for warning in part.get("warnings", []))
    for spec in specs:
        resolved = spec["resolved"]
        items = resolved["stat_vars"]["items"]
        what = [token["text"] for token in resolved["sentence_plan"] if token.get("axis") == "what"]
        codes = [
            value["ref"]["dcid"]
            for slot in resolved["slots"]
            if slot["key"]["axis"] == "how"
            for value in _binding_values(slot["binding"])
        ]
        codes.extend(
            slot_value["value"]["ref"]["dcid"]
            for item in items
            for slot_value in item.get("slot_values", [])
            if slot_value["value"].get("ref")
        )
        noun = _authored_noun(resolved["shape"], items)
        yield Reading(
            query_id=query_id,
            variable_text=part["variable_text"],
            weak=weak,
            measure_noun=noun or (what[0] if what else resolved["shape"]["label"]),
            stat_vars=tuple((item["ref"], item["label"]) for item in items),
            bound_codes=tuple(dict.fromkeys(codes)),
        )


def _authored_noun(shape: dict, items: list[dict]) -> str | None:
    """The authored measure noun the heading shows for `shape`, from the
    constraint values every StatVar in `items` shares."""
    values: dict[str, set[str]] = {}
    for item in items:
        for slot_value in item.get("slot_values", []):
            if (ref := slot_value["value"].get("ref")) and (
                prop := slot_value["key"].get("property")
            ):
                values.setdefault(prop["dcid"], set()).add(ref["dcid"])
    shared = {
        prop: next(iter(found))
        for prop, found in values.items()
        if len(found) == 1
        and all(
            any(
                sv["key"].get("property", {}).get("dcid") == prop
                for sv in item.get("slot_values", [])
            )
            for item in items
        )
    }
    noun = measure_noun(
        population_type=(shape.get("population_type") or {}).get("dcid"),
        measured_property=(shape.get("measured_property") or {}).get("dcid"),
        stat_type=(shape.get("stat_type") or {}).get("dcid"),
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
    parser.add_argument("--alternate-names", type=Path, required=True)
    parser.add_argument("--fetch-alternate-names", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    queries = _load_queries(args.raw_dir)
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
