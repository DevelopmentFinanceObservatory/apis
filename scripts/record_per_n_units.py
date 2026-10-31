"""Record every unit whose ref or label reads as a value per N of something.

Reads the members of the three unit classes through `<-typeOf`: `UnitOfMeasure` and
`SDG_UnitOfMeasure` on the graph `SEARCH_GRAPH_BASE` names, and `undata/UNIT_MEASURE` on the
UN System Data Commons. Writes `{ref: label}`, sorted, to `tests/fixtures/per_n_units.json`.
`tests/unit_words/test_per_n_units.py` holds each recorded ref to the unit registry, so a new
unit in the graph shows up as a failure on the next recording.

Requires `SEARCH_GRAPH_BASE` and a `gcloud` login: the dev graph service takes an identity token.

Usage:
    uv run --env-file .env python scripts/record_per_n_units.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx

_FIXTURE = Path(__file__).resolve().parent.parent / "tests/fixtures/per_n_units.json"
_UN_BASE = "https://unsd-datacommons.gcp.un-icc.cloud/core/api"
_PER_N = re.compile(r"per[ _]?(count_)?(\d|million|hundred|thousand)", re.IGNORECASE)


def _members(base: str, unit_class: str, *, headers: dict[str, str]) -> dict[str, str | None]:
    """Each member of `unit_class` with its name, paged to the end."""
    members: dict[str, str | None] = {}
    token = None
    while True:
        body = {"nodes": [unit_class], "property": "<-typeOf"}
        if token is not None:
            body["nextToken"] = token
        response = httpx.post(f"{base}/v2/node", json=body, headers=headers, timeout=120)
        response.raise_for_status()
        page = response.json()
        nodes = page["data"][unit_class]["arcs"]["typeOf"]["nodes"]
        members.update({node["dcid"]: node.get("name") for node in nodes})
        token = page.get("nextToken")
        if token is None:
            return members


def main() -> int:
    base = os.environ.get("SEARCH_GRAPH_BASE")
    if not base:
        print("SEARCH_GRAPH_BASE is not set", file=sys.stderr)
        return 1
    identity_token = subprocess.run(
        ["gcloud", "auth", "print-identity-token"], capture_output=True, text=True, check=True
    ).stdout.strip()
    headers = {"Authorization": f"Bearer {identity_token}"}
    graph = f"{base.rstrip('/')}/core/api"
    members = {
        **_members(graph, "UnitOfMeasure", headers=headers),
        **_members(graph, "SDG_UnitOfMeasure", headers=headers),
        **_members(_UN_BASE, "undata/UNIT_MEASURE", headers={"User-Agent": "dfo-apis"}),
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
