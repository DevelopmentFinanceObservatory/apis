"""Mint the readings in `fixtures/readings.json` through the running local stack.

Each reading is posted to the mint door (`POST /api/datasets`) as a `statvar_set` input with
its scope, so its spec, version record, artifacts and inventory rows are the ones a deployed
mint writes, in the stack's Postgres and GCS emulator. The server's upstreams answer from the
same fixture (`upstream.py`).

A mint is idempotent. Minting a reading again answers the version already stored, so a second
run leaves one version per reading. The spec ids and content versions go to
`$LOCAL_STACK_SEEDED` for the smoke script.

Run through `scripts/local-stack/stack.sh seed`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import httpx
from upstream import Reading, load_fixture

MINT_TIMEOUT_SECONDS = 120


def mint_body(reading: Reading) -> dict[str, object]:
    return {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": [f"dcid:{reading.stat_var}"],
            "scope": {"window": {"kind": "all"}, **reading.scope},
        },
        "options": {"bundle": False},
    }


def seed(client: httpx.Client, readings: Sequence[Reading]) -> list[dict[str, object]]:
    """Mint each reading; return its name, spec id, content version, series and rows."""
    seeded = []
    for reading in readings:
        response = client.post("/api/datasets", json=mint_body(reading))
        if response.status_code != 200:
            raise RuntimeError(
                f"minting {reading.name} answered {response.status_code}: {response.text}"
            )
        body = response.json()
        seeded.append(
            {
                "name": reading.name,
                "spec_id": body["definition"]["spec_id"],
                "content_version": body["content_version"],
                "series": len(reading.series),
                "rows": sum(len(series.points) for series in reading.series),
            }
        )
    return seeded


def main() -> None:
    readings = load_fixture().readings
    with httpx.Client(
        base_url=os.environ["LOCAL_STACK_BASE_URL"], timeout=MINT_TIMEOUT_SECONDS
    ) as client:
        seeded = seed(client, readings)
    if len(seeded) != len(readings):
        raise RuntimeError(f"seeded {len(seeded)} of {len(readings)} readings")
    for entry in seeded:
        print(f"{entry['name']}: {entry['spec_id']} {entry['content_version']}")
    Path(os.environ["LOCAL_STACK_SEEDED"]).write_text(json.dumps(seeded, indent=1) + "\n")
    print(f"seeded {len(seeded)} readings")


if __name__ == "__main__":
    main()
