"""Score the `closest_match` rule against recorded panel responses.

Reads `raw-*.jsonl` panel responses (one `{"id", "query", "body"}` record per line, the
body a full search response), judges every reading each response shows (a definite part's
Spec, each Spec of a candidates part) as assembly judges it, and compares the result with
hand labels. A query is predicted positive when any of its readings raises the fact.

Each reading runs `search.engine.closest_match.spec_closest_match_fact` on its part's
`variable_text`, the measure noun its heading shows (`heading_for`), and the inputs
`closest_match.reading_inputs` builds from the Spec. A response carries no Spec's members,
so they come from a members snapshot, which maps a `spec_id` to the `members` the Datasets
members door (`POST /api/datasets/{id}/members`) lists for the Spec's `definition`: the
StatVars a mint of it serves. `--fetch-members` writes it from the service at `--base` for
every Spec the responses show. A run without the flag reads the snapshot and touches no
network beyond the graph release at `GRAPH_RELEASE_URI`, whose code and measure indexes the
rule and the heading read, so it reruns to the same output on one release.

`search.eval.apis_auth` adds the Cloud Run ID token an `https` base URL needs. Set
`DFO_APIS_ID_TOKEN`, or point the base URL at `gcloud run services proxy` on localhost.

The labels file maps a query id to `{"closest_match": bool, "basis": "assessment" | "own",
"reason": str}`. A query it does not list is a negative whose basis is the assessment.

Usage:
    uv run python scripts/closest_match_confusion.py --raw-dir RUNS --labels LABELS.json \\
        --members MEMBERS.json --fetch-members
    uv run python scripts/closest_match_confusion.py --raw-dir RUNS --labels LABELS.json \\
        --members MEMBERS.json --out confusion.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import httpx

from graph_release.context import bound_served_release
from graph_release.service import get_release_service
from search.engine.closest_match import bound_code_names, reading_inputs, spec_closest_match_fact
from search.eval.apis_auth import apis_auth
from search.heading.words import content_words
from search.models import GraphRef, SentenceFacts, Spec
from search.render.heading import heading_for

_DCID_PREFIX = "dcid:"
DEFAULT_BASE = "https://dfo-dev-apis-zms5lj4zhq-nn.a.run.app"
TIMEOUT_S = 150


@dataclass(frozen=True)
class Reading:
    """One Spec a recorded part shows, stating its members' sentence facts, with the
    request's words for its variable and the measure noun its heading shows."""

    variable_text: str
    measure_noun: str
    spec: Spec

    def fires(self) -> bool:
        fact = spec_closest_match_fact(
            self.spec, variable_text=self.variable_text, measure_noun=self.measure_noun
        )
        return fact is not None

    def shared_words(self) -> list[str]:
        """The content words of the variable text that some name of the measure holds,
        each with the first name that holds it. These words keep the rule silent."""
        evidence, bound_codes, _series = reading_inputs(self.spec, measure_noun=self.measure_noun)
        names = [self.measure_noun, *evidence.member_names, *bound_code_names(bound_codes)]
        shared: dict[str, str] = {}
        for word in sorted(content_words(self.variable_text)):
            holder = next((name for name in names if word in content_words(name)), None)
            if holder is not None:
                shared[word] = holder
        return [f"{word!r} in {holder!r}" for word, holder in shared.items()]


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
                _reading(record["id"], part, spec, members)
                for part in _parts(record)
                for spec in _shown_specs(part)
            ),
        )
        for record in records
    ]


def _reading(query_id: str, part: dict, raw: dict, members: dict[str, list[dict]]) -> Reading:
    """The reading `raw`, a Spec `part` shows, stating the members the snapshot lists."""
    if raw["spec_id"] not in members:
        raise SystemExit(
            f"the members snapshot holds no {raw['spec_id']} (query {query_id}); "
            "write it with --fetch-members"
        )
    items = members[raw["spec_id"]]
    facts = SentenceFacts(
        members=[
            GraphRef(ref=item["ref"].removeprefix(_DCID_PREFIX), label=item["label"])
            for item in items
        ],
        constraint_properties=[
            list(properties)
            for properties in sorted(
                {
                    tuple(sorted(p for p, value in item["constraints"].items() if value))
                    for item in items
                }
            )
        ],
    )
    spec = Spec.model_validate(raw)
    spec = spec.model_copy(update={"resolved": spec.resolved.stating_sentence_facts(facts)})
    return Reading(
        variable_text=part["variable_text"],
        measure_noun=heading_for(spec).heading.noun.inline_text,
        spec=spec,
    )


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
    with httpx.Client(base_url=base.rstrip("/"), timeout=TIMEOUT_S, auth=apis_auth()) as client:
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


def _report(queries: list[Query], labels: dict) -> str:
    unknown = sorted(set(labels) - {q.query_id for q in queries})
    if unknown:
        raise SystemExit(f"labels name queries the responses do not hold: {unknown}")
    rows = []
    for q in queries:
        label = labels.get(q.query_id, {})
        expected = bool(label.get("closest_match", False))
        predicted = any(r.fires() for r in q.readings)
        rows.append((q, expected, predicted, label))

    def count(expected: bool, predicted: bool) -> int:
        return sum(1 for _q, e, p, _l in rows if e == expected and p == predicted)

    tp, fp, fn, tn = count(True, True), count(False, True), count(True, False), count(False, False)
    precision = f"{tp / (tp + fp):.2f}" if tp + fp else "n/a"
    recall = f"{tp / (tp + fn):.2f}" if tp + fn else "n/a"
    own = sum(1 for *_rest, label in rows if label.get("basis") == "own")
    lines = [
        "# Closest-match rule: confusion over the recorded panel",
        "",
        f"{len(rows)} queries. {own} labels are the labeller's own reading of the query and its assessment; every "
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
            "| id | query | label basis | reason | readings shown | shared words |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for q, _e, _p, label in selected:
            shown = "; ".join(
                dict.fromkeys(f"{r.variable_text!r} -> {r.measure_noun}" for r in q.readings)
            )
            shared = "; ".join(dict.fromkeys(w for r in q.readings for w in r.shared_words()))
            lines.append(
                f"| {q.query_id} | {q.query} | {label.get('basis', 'assessment')} | {label.get('reason', '')} | {shown} | "
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
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    records = _load_records(args.raw_dir)
    if args.fetch_members:
        _fetch_members(records, args.members, base=args.base)
        return
    with bound_served_release(get_release_service()):
        queries = _load_queries(records, json.loads(args.members.read_text()))
        report = _report(queries, json.loads(args.labels.read_text()))
    if args.out is None:
        print(report)
    else:
        args.out.write_text(report + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
