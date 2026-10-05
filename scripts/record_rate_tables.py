"""Record live fixtures for one whole ``materialise.rate_tables`` table.

``load_table`` reads a table in three steps, and this script records each one
under ``tests/materialise/fixtures/rate_tables/``:

- ``<stat-var>.availability.json``: the ``/availability`` body listing the
  StatVar's ``observationAbout`` entities;
- ``<stat-var>.csv``: the ``/data`` SDMX-CSV body for every listed entity;
- ``<provenance-id>.json``: the provenance's ``lastDataRefreshDate``, with
  slashes in the id turned into underscores.

Requires ``SEARCH_GRAPH_BASE``. ``SEARCH_GRAPH_TOKEN`` is forwarded when set.

Usage:
    uv run --env-file .env python scripts/record_rate_tables.py \\
        --stat-var ExchangeRate_CurrencyUnit_ConversionRecipeDAC \\
        --provenance-id provenance/ONEConversionTablesDAC
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

from dc_graph import GraphClient
from dc_sdmx import SdmxClient

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES_DIR = _REPO_ROOT / "tests" / "materialise" / "fixtures" / "rate_tables"

# Read raw /data bytes so the fixture preserves the response before CSV parsing.
_DATA_PATH = "/sdmx/v3/data/dataflow/DC/DF_OBS/1.0.0/*"
_ACCEPT_DATA_CSV = "application/vnd.sdmx.data+csv;version=2.0.0"
_SUBJECT_ROLE = "observationAbout"
_REFRESH_PROPERTY = "lastDataRefreshDate"


def _record_availability(
    base_url: str, token: str | None, *, stat_var: str
) -> tuple[list[str], dict]:
    with SdmxClient(base_url=base_url, token=token, timeout=60.0) as sdmx:
        result = sdmx.availability(variables=[stat_var], component=_SUBJECT_ROLE)
    return list(result.components.get(_SUBJECT_ROLE, ())), result.raw


def _record_csv(base_url: str, token: str | None, *, stat_var: str, entities: list[str]) -> bytes:
    headers = {"Accept": _ACCEPT_DATA_CSV}
    if token:
        headers["x-graph-token"] = token
    params = {
        "c[variableMeasured]": stat_var,
        f"c[{_SUBJECT_ROLE}]": ",".join(entities),
    }
    with httpx.Client(base_url=base_url, headers=headers, timeout=60.0) as client:
        response = client.get(_DATA_PATH, params=params)
        response.raise_for_status()
        return response.content


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one whole rate/deflator table")
    parser.add_argument("--stat-var", required=True, help="bare StatVar dcid, no 'dcid:' prefix")
    parser.add_argument("--provenance-id", required=True)
    args = parser.parse_args()

    base_url = os.environ.get("SEARCH_GRAPH_BASE")
    if not base_url:
        print("SEARCH_GRAPH_BASE must be set; this script has no offline mode.", file=sys.stderr)
        raise SystemExit(1)
    token = os.environ.get("SEARCH_GRAPH_TOKEN")

    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    entities, availability_body = _record_availability(base_url, token, stat_var=args.stat_var)
    if not entities:
        print(f"{args.stat_var} serves no {_SUBJECT_ROLE} entities.", file=sys.stderr)
        raise SystemExit(1)
    availability_path = _FIXTURES_DIR / f"{args.stat_var}.availability.json"
    availability_path.write_text(json.dumps(availability_body, indent=2) + "\n")
    print(f"Wrote {availability_path} ({len(entities)} entities)")

    csv_bytes = _record_csv(base_url, token, stat_var=args.stat_var, entities=entities)
    csv_path = _FIXTURES_DIR / f"{args.stat_var}.csv"
    csv_path.write_bytes(csv_bytes)
    print(f"Wrote {csv_path} ({len(csv_bytes)} bytes)")

    graph = GraphClient(base_url=base_url, token=token)
    try:
        arcs = graph.node_arcs_many([args.provenance_id], properties=(_REFRESH_PROPERTY,))
    finally:
        graph.close()
    refresh_dates = arcs.get(args.provenance_id)
    version_path = _FIXTURES_DIR / f"{args.provenance_id.replace('/', '_')}.json"
    version_path.write_text(
        json.dumps(
            {_REFRESH_PROPERTY: list(refresh_dates.get(_REFRESH_PROPERTY, ()))}
            if refresh_dates is not None
            else {_REFRESH_PROPERTY: None},
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {version_path}")


if __name__ == "__main__":
    main()
