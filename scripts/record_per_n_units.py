"""Record every unit whose ref or label reads as a value per N of something.

Reads the members of the three unit classes through `<-typeOf`: `UnitOfMeasure` and
`SDG_UnitOfMeasure` on the graph `SEARCH_GRAPH_BASE` names, and `undata/UNIT_MEASURE` on the
UN System Data Commons. Writes `{ref: label}`, sorted, to `tests/fixtures/per_n_units.json`.
`tests/unit_words/test_per_n_units.py` holds each recorded ref to the unit registry, so a new
unit in the graph shows up as a failure on the next recording.

Requires `SEARCH_GRAPH_BASE`. `SEARCH_GRAPH_TOKEN` is forwarded when set.

Usage:
    uv run --env-file .env python scripts/record_per_n_units.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from dc_graph.client import GraphClient

_FIXTURE = Path(__file__).resolve().parent.parent / "tests/fixtures/per_n_units.json"
_UN_BASE = "https://unsd-datacommons.gcp.un-icc.cloud/core/api"
_PER_N = re.compile(r"per[ _]?(count_)?(\d|million|hundred|thousand)", re.IGNORECASE)
# Far above any unit class's membership, so a read that reaches it is truncated and refused.
_MAX_UNITS = 10_000


def _labelled_members(graph: GraphClient, unit_class: str) -> dict[str, str | None]:
    """Each member of `unit_class` with its first name."""
    page = graph.node_subjects(unit_class, property="typeOf", max_results=_MAX_UNITS)
    if page.truncated:
        raise RuntimeError(f"{unit_class} has more than {_MAX_UNITS} members")
    names = graph.node_names_many(page.dcids)
    return {dcid: next(iter(names.get(dcid, ())), None) for dcid in page.dcids}


def main() -> int:
    base = os.environ.get("SEARCH_GRAPH_BASE")
    if not base:
        print("SEARCH_GRAPH_BASE is not set", file=sys.stderr)
        return 1
    token = os.environ.get("SEARCH_GRAPH_TOKEN") or None
    with (
        GraphClient(base_url=f"{base.rstrip('/')}/core/api", token=token) as graph,
        GraphClient(base_url=_UN_BASE, user_agent="dfo-apis") as un_graph,
    ):
        members = {
            **_labelled_members(graph, "UnitOfMeasure"),
            **_labelled_members(graph, "SDG_UnitOfMeasure"),
            **_labelled_members(un_graph, "undata/UNIT_MEASURE"),
        }
    recorded = {
        ref: members[ref]
        for ref in sorted(members)
        if _PER_N.search(ref) or _PER_N.search(members[ref] or "")
    }
    _FIXTURE.write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")
    print(f"{len(recorded)} of {len(members)} units recorded in {_FIXTURE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
