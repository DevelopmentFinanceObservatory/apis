"""Record what our mixer answers for the variables recall finds, to check the arc tables against.

The script loads the committed eval catalogue (``tests/fixtures/origin_catalogue_eval``), sends
each corpus query to our mixer's ``/v2/resolve`` as recall sends it (the indicator resolver over
``base_and_custom``), and cuts the candidates as recall cuts them (``cut_hits``). It keeps the hits
the build lists as DFO or base variables until it holds ``--per-origin`` of each, taking at most
``_FIRST_PASS_PER_QUERY`` new hits of an origin from one query on a first pass over the corpus,
so the sample spans many queries, and any hit on a second pass.

For each kept variable it records the ``->*`` arcs ``dc_graph.GraphClient.node_arcs_batch``
returns, as the sorted distinct ``dcid`` or ``value`` of each property, and the label and first type
``node_labels_batch`` and ``node_types_batch`` return. For every node those arcs name it records
the label and type too, since the origin client answers those from the arc tables, and
``arc_names``, the sorted distinct names those arcs carry for it.
``tests/origin_catalogue/test_catalogue_equivalence.py`` compares the eval build's tables with
the recording.

``--keep-sample`` reads the variables and their origins from the existing recording and records
them again, with no resolve and no catalogue.

Usage:
    set -a; . .env; set +a
    uv run python scripts/record_resolver_hit_arcs.py \\
        --base-url https://<mixer>/core/api --token "$SEARCH_GRAPH_TOKEN"
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from dc_graph import GraphClient  # noqa: E402
from origin_catalogue import CatalogueSnapshot  # noqa: E402
from origin_catalogue.artifact import load_catalogue  # noqa: E402
from scripts.stage_origin_catalogue_fixture import stage_fixture  # noqa: E402
from search.engine.clients.graph import cut_hits  # noqa: E402
from search.engine.clients.graph_client import _parse_detect_candidates  # noqa: E402
from search.engine.config import settings  # noqa: E402
from search.eval.cases import load_cases  # noqa: E402

EVAL_FIXTURE_DIR: Final = _REPO_ROOT / "tests" / "fixtures" / "origin_catalogue_eval"
RECORDING: Final = _REPO_ROOT / "tests" / "fixtures" / "recorded_resolver_hit_arcs.json"

_FIRST_PASS_PER_QUERY: Final[int] = 5
_BATCH: Final[int] = 50
_RESOLVES_IN_FLIGHT: Final[int] = 8


def _hits(graph: GraphClient, query: str, *, catalogue: CatalogueSnapshot) -> list[str]:
    """The query's recall hits, as `cut_hits` leaves them, that the build lists."""
    body = graph.post(
        "/v2/resolve",
        {"nodes": [query], "resolver": "indicator", "target": settings().resolve_target},
    )
    cut = cut_hits(
        _parse_detect_candidates(body), catalogue=catalogue, detect_k=settings().detect_k
    )
    return [hit.dcid for hit in cut if _origin(hit.dcid, catalogue) is not None]


def _origin(dcid: str, catalogue: CatalogueSnapshot) -> str | None:
    if dcid in catalogue.tables.dfo_variables:
        return "dfo"
    if dcid in catalogue.tables.base_variables:
        return "base"
    return None


def sample_hits(
    graph: GraphClient, queries: list[str], *, catalogue: CatalogueSnapshot, per_origin: int
) -> dict[str, str]:
    """Up to `per_origin` DFO and `per_origin` base recall hits, each mapped to its origin."""
    with ThreadPoolExecutor(max_workers=_RESOLVES_IN_FLIGHT) as pool:
        found = pool.map(lambda query: _hits(graph, query, catalogue=catalogue), queries)
        hits = dict(zip(queries, found, strict=True))
    kept: dict[str, str] = {}
    counts = {"dfo": 0, "base": 0}
    for limit in (_FIRST_PASS_PER_QUERY, None):
        for query in queries:
            taken = {"dfo": 0, "base": 0}
            for dcid in hits[query]:
                origin = _origin(dcid, catalogue)
                if dcid in kept or origin is None or counts[origin] >= per_origin:
                    continue
                if limit is not None and taken[origin] >= limit:
                    continue
                kept[dcid] = origin
                counts[origin] += 1
                taken[origin] += 1
    print(f"kept {counts} from {len(queries)} queries", file=sys.stderr)
    return kept


def _batches(dcids: list[str]) -> Iterable[list[str]]:
    for start in range(0, len(dcids), _BATCH):
        yield dcids[start : start + _BATCH]


def _value_sets(arcs: Mapping[str, dict] | None) -> dict[str, list[str]] | None:
    if arcs is None:
        return None
    return {
        prop: sorted({node.get("dcid") or node.get("value") for node in entry["nodes"]})
        for prop, entry in arcs.items()
    }


def _labels_and_types(graph: GraphClient, dcids: list[str]) -> dict[str, dict[str, str | None]]:
    found: dict[str, dict[str, str | None]] = {}
    for batch in _batches(dcids):
        labels = graph.node_labels_batch(batch)
        types = graph.node_types_batch(batch)
        found.update(
            {
                dcid: {"label": labels.get(dcid), "type": types[dcid][0] if dcid in types else None}
                for dcid in batch
            }
        )
    return found


def record(graph: GraphClient, variables: Mapping[str, str]) -> dict:
    """The recording: each variable's origin, arcs, label and type, and each node its arcs
    name with its label and type."""
    dcids = sorted(variables)
    arcs: dict[str, dict[str, list[str]] | None] = {}
    arc_names: dict[str, set[str]] = {}
    for batch in _batches(dcids):
        for dcid, found in graph.node_arcs_batch(batch).items():
            arcs[dcid] = _value_sets(found)
            for entry in (found or {}).values():
                for node in entry["nodes"]:
                    if node.get("dcid"):
                        names = arc_names.setdefault(node["dcid"], set())
                        if node.get("name"):
                            names.add(node["name"])
    named = _labels_and_types(graph, dcids)
    target_ids = sorted(arc_names.keys() - variables.keys())
    return {
        "variables": {
            dcid: {"origin": variables[dcid], "arcs": arcs[dcid], **named[dcid]} for dcid in dcids
        },
        "targets": {
            dcid: {**facts, "arc_names": sorted(arc_names[dcid])}
            for dcid, facts in _labels_and_types(graph, target_ids).items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True, help="our mixer's API root, with /core/api")
    parser.add_argument("--token", required=True)
    parser.add_argument("--per-origin", type=int, default=250)
    parser.add_argument(
        "--keep-sample", action="store_true", help="record the existing recording's variables"
    )
    args = parser.parse_args()

    with GraphClient(base_url=args.base_url, token=args.token) as graph:
        if args.keep_sample:
            kept = json.loads(RECORDING.read_text())["variables"]
            variables = {dcid: entry["origin"] for dcid, entry in kept.items()}
        else:
            queries = list(dict.fromkeys(case.query for case in load_cases() if case.query))
            with tempfile.TemporaryDirectory() as root:
                catalogue = load_catalogue(stage_fixture(EVAL_FIXTURE_DIR, Path(root)))
            variables = sample_hits(graph, queries, catalogue=catalogue, per_origin=args.per_origin)
        recording = record(graph, variables)
    RECORDING.write_text(
        "{\n"
        + ",\n".join(
            f"{json.dumps(section)}: {{\n"
            + ",\n".join(
                f"  {json.dumps(dcid)}: {json.dumps(entry, sort_keys=True)}"
                for dcid, entry in recording[section].items()
            )
            + "\n}"
            for section in ("variables", "targets")
        )
        + "\n}\n"
    )
    print(
        f"wrote {len(recording['variables'])} variables and {len(recording['targets'])} "
        f"targets to {RECORDING}"
    )


if __name__ == "__main__":
    main()
