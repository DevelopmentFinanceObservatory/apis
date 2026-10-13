"""Record the graph `description` of every World Bank WDI and IDS StatVar.

`description.world_bank.world_bank_definition` cuts the series code the WDI import appends to
each World Development Indicators description, and gives International Debt Statistics
StatVars, whose descriptions are series titles, no definition. The test in
`tests/description/test_world_bank.py` checks it against every description this script
records, in `tests/description/fixtures/world_bank_descriptions.json.gz`:

- `wdi`: every StatVar in the `worldBank/` namespace, which holds no provenance of its own;
- `ids`: every StatVar of `provenance/WorldBankInternationalDebtStatistics`.

Each maps a dcid to its `description` values in the order the graph gives them. The script
pages every `StatisticalVariable` once (about 330,000 nodes, some 600 pages, about 10 minutes),
then reads the descriptions in one batched read.

Requires `SEARCH_GRAPH_BASE`. `SEARCH_GRAPH_TOKEN` is forwarded when set.

Usage:
    uv run --env-file .env python scripts/record_world_bank_descriptions.py
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

from dc_graph import GraphClient
from dc_graph.client import NODE_PATH
from description.world_bank import WORLD_BANK_NAMESPACE

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "description"
    / "fixtures"
    / "world_bank_descriptions.json.gz"
)
_IDS = "provenance/WorldBankInternationalDebtStatistics"
_MAX_PAGES = 5_000


def _stat_vars(client: GraphClient) -> tuple[list[str], list[str]]:
    """The WDI and IDS StatVar dcids, each sorted, from every `StatisticalVariable` paged."""
    wdi: set[str] = set()
    ids: set[str] = set()
    token: str | None = None
    for _ in range(_MAX_PAGES):
        payload: dict = {"nodes": ["StatisticalVariable"], "property": "<-typeOf"}
        if token is not None:
            payload["nextToken"] = token
        body = client.post(NODE_PATH, payload)
        arcs = body.get("data", {}).get("StatisticalVariable", {}).get("arcs", {})
        for node in arcs.get("typeOf", {}).get("nodes", []):
            dcid = node.get("dcid", "")
            if dcid.startswith(WORLD_BANK_NAMESPACE):
                wdi.add(dcid)
            elif node.get("provenanceId") == _IDS:
                ids.add(dcid)
        token = body.get("nextToken")
        if token is None:
            return sorted(wdi), sorted(ids)
    raise SystemExit(f"StatisticalVariable still paging after {_MAX_PAGES} pages")


def main() -> None:
    base = os.environ.get("SEARCH_GRAPH_BASE", "").strip()
    if not base:
        print("SEARCH_GRAPH_BASE must be set; this script has no offline mode.", file=sys.stderr)
        raise SystemExit(1)
    client = GraphClient(base_url=f"{base}/core/api", token=os.environ.get("SEARCH_GRAPH_TOKEN"))
    wdi, ids = _stat_vars(client)
    if not wdi or not ids:
        raise SystemExit(f"read {len(wdi)} WDI and {len(ids)} IDS StatVars; expected both")
    recorded = {}
    for group, dcids in (("wdi", wdi), ("ids", ids)):
        arcs = client.node_arcs_many(dcids, properties=["description"])
        recorded[group] = {
            dcid: list((arcs.get(dcid) or {}).get("description", ())) for dcid in dcids
        }
    _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(recorded, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    _FIXTURE.write_bytes(gzip.compress(body.encode("utf-8"), mtime=0))
    print(f"recorded {len(wdi)} WDI and {len(ids)} IDS StatVars to {_FIXTURE}")


if __name__ == "__main__":
    main()
