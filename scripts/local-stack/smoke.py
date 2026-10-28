"""Check every seeded reading over HTTP against the running local stack.

For each reading `seed.py` recorded in `$LOCAL_STACK_SEEDED`:

- `GET /api/datasets/{spec_id}/versions` lists its content version;
- `GET /api/datasets/{spec_id}/{content_version}` answers its manifest, with artifacts;
- `GET .../observations` answers every fixture series with the fixture's points;
- each artifact's files link answers 302, and the bytes at the location hash to its digest.

`analysis_step` runs last, over the German ODA and Kenya population readings:

- `POST /api/datasets/analysis` divides German ODA by Kenya's population, pairing the recipient
  with the place, and answers a format 2 ref to an `ops_`/`ocv_` frame over both roots;
- `GET /api/datasets/derived/{ops}/{ocv}` names the frame and its roots, and its Parquet file
  hashes to the ref's `artifact.digest` and holds one Kenya row a year with the quotient the
  fixture gives;
- the bare derived root under `latest` answers the same `ops_` and `ocv_`;
- `POST /api/presentation/components` makes a table card of the frame from the release's
  minimal table intent;
- `POST /api/presentation/shares` makes a live share of the card with its page, embed and data
  URLs and a `shw_` token, which replaces it; a wrong token refuses 403 and a delete takes it
  down.

Exits 1 on the first failed check.

Run through `scripts/local-stack/stack.sh smoke`.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
from upstream import Reading, load_fixture

ANALYSIS_ROUTE = "/api/datasets/analysis"
SHARES_ROUTE = "/api/presentation/shares"
WRITE_TOKEN = "x-dfo-write-token"
ODA_READING = "oda_germany_to_kenya_ghana_nigeria"
POPULATION_READING = "kenya_population"


class SmokeFailure(Exception):
    pass


def _get(client: httpx.Client, path: str) -> httpx.Response:
    response = client.get(path)
    if response.status_code != 200:
        raise SmokeFailure(f"GET {path} answered {response.status_code}: {response.text[:400]}")
    return response


def check_versions(client: httpx.Client, entry: Mapping[str, object]) -> str:
    body = _get(client, f"/api/datasets/{entry['spec_id']}/versions").json()
    listed = [version["content_version"] for version in body["versions"]]
    if entry["content_version"] not in listed:
        raise SmokeFailure(f"versions of {entry['spec_id']} list {listed}")
    return f"versions lists {entry['content_version']} ({len(listed)} listed)"


def check_manifest(client: httpx.Client, entry: Mapping[str, object]) -> list[Mapping]:
    path = f"/api/datasets/{entry['spec_id']}/{entry['content_version']}"
    manifest = _get(client, path).json()
    if manifest["content_version"] != entry["content_version"]:
        raise SmokeFailure(f"manifest at {path} names {manifest['content_version']}")
    if not manifest["artifacts"]:
        raise SmokeFailure(f"manifest at {path} lists no artifacts")
    return manifest["artifacts"]


def _series_key(refs: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(refs))


def check_observations(client: httpx.Client, entry: Mapping[str, object], reading: Reading) -> str:
    path = f"/api/datasets/{entry['spec_id']}/{entry['content_version']}/observations"
    served = {
        _series_key([entity["ref"] for entity in series["entities"]]): [
            (point["date"], point["value"]) for point in series["points"]
        ]
        for series in _get(client, path).json()["series"]
    }
    expected = {
        _series_key(list(series.entities.values())): list(series.points)
        for series in reading.series
    }
    if served != expected:
        raise SmokeFailure(f"observations at {path}: served {served}, expected {expected}")
    rows = sum(len(points) for points in served.values())
    return f"observations: {len(served)} series, {rows} points, equal to the fixture"


def check_files(client: httpx.Client, artifacts: Sequence[Mapping]) -> str:
    for artifact in artifacts:
        link = client.get(artifact["url"])
        if link.status_code != 302:
            raise SmokeFailure(f"GET {artifact['url']} answered {link.status_code}: {link.text}")
        location = link.headers["location"]
        download = client.get(location)
        if download.status_code != 200:
            raise SmokeFailure(f"GET {location} answered {download.status_code}")
        digest = "sha256:" + hashlib.sha256(download.content).hexdigest()
        if digest != artifact["artifact_digest"]:
            raise SmokeFailure(f"{location} holds {digest}, not {artifact['artifact_digest']}")
    media = sorted({artifact["media_type"] for artifact in artifacts})
    return f"files: {len(artifacts)} links redirect to bytes matching their digests ({media})"


def _post(client: httpx.Client, path: str, body: object, *, status: int = 200) -> httpx.Response:
    response = client.post(path, json=body)
    if response.status_code != status:
        raise SmokeFailure(f"POST {path} answered {response.status_code}: {response.text[:400]}")
    return response


def _root(entry: Mapping[str, object]) -> dict[str, object]:
    return {"spec_id": entry["spec_id"], "content_version": entry["content_version"]}


def _expected_quotients(oda: Reading, population: Reading) -> dict[str, float]:
    """German ODA to Kenya over Kenya's population, in each year both readings hold."""
    (to_kenya,) = [
        series for series in oda.series if series.entities["financingRecipient"] == "country/KEN"
    ]
    (people,) = population.series
    persons = dict(people.points)
    return {year: value / persons[year] for year, value in to_kenya.points if year in persons}


def check_quotient(
    client: httpx.Client, oda: Mapping[str, object], population: Mapping[str, object]
) -> Mapping[str, Any]:
    """`quotient_of` of the German ODA reading over Kenya's population, pairing the recipient
    with the place by role. Answers the ref's member naming the derived frame."""
    tree = {
        "verb": "quotient_of",
        "arguments": {"on": {"destination": "place"}},
        "inputs": {"data": _root(oda), "other": _root(population)},
    }
    body = _post(client, ANALYSIS_ROUTE, {"tree": tree}).json()
    ref = body["ref"]
    if body["status"] != "ready" or ref["format_version"] != 2 or len(ref["members"]) != 1:
        raise SmokeFailure(f"analysis answered {body['status']} with ref {ref}")
    (member,) = ref["members"]
    derived = _derived_pair(member)
    if not (
        derived["spec_id"].startswith("ops_") and derived["content_version"].startswith("ocv_")
    ):
        raise SmokeFailure(f"analysis ref names {derived}")
    if [_root(root) for root in member["roots"]] != [_root(oda), _root(population)]:
        raise SmokeFailure(f"analysis ref roots {member['roots']}")
    return member


def _derived_pair(member: Mapping[str, Any]) -> dict[str, str]:
    return {"spec_id": member["spec"]["spec_id"], "content_version": member["content_version"]}


def _derived_path(derived: Mapping[str, str]) -> str:
    return f"/api/datasets/derived/{derived['spec_id']}/{derived['content_version']}"


def check_derived_reads(
    client: httpx.Client,
    member: Mapping[str, Any],
    roots: Sequence[Mapping[str, object]],
    expected: Mapping[str, float],
) -> str:
    """The derived manifest names the frame and its roots, and its Parquet file hashes to the
    ref's artifact digest and holds one row for Kenya in each year with the quotient the
    fixture gives."""
    derived = _derived_pair(member)
    manifest = _get(client, _derived_path(derived)).json()
    named = {"spec_id": manifest["spec_id"], "content_version": manifest["content_version"]}
    if named != derived or [_root(root) for root in manifest["roots"]] != list(map(_root, roots)):
        raise SmokeFailure(f"derived manifest names {named} over {manifest['roots']}")
    path = f"{_derived_path(derived)}/file?format=parquet&bundle=false&layout=frame"
    body = _get(client, path).content
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    if digest != member["artifact"]["digest"]:
        raise SmokeFailure(f"{path} holds {digest}, not the ref's {member['artifact']['digest']}")
    table = pq.read_table(io.BytesIO(body))
    rows = table.select(["financingRecipient", "time_period", "value"]).to_pylist()
    served = {row["time_period"]: row["value"] for row in rows}
    if {row["financingRecipient"] for row in rows} != {"country/KEN"} or served.keys() != set(
        expected
    ):
        raise SmokeFailure(f"derived file rows {rows}, expected Kenya in {sorted(expected)}")
    off = {
        year: served[year] for year in expected if not math.isclose(served[year], expected[year])
    }
    if off:
        raise SmokeFailure(f"derived file values {off} differ from the fixture's {expected}")
    return (
        f"derived: manifest and Parquet file hashing to the ref's digest, {len(rows)} rows equal "
        "to the fixture's quotients"
    )


def check_latest(client: httpx.Client, derived: Mapping[str, str]) -> str:
    """The bare derived root under `latest` computes the same frame again."""
    body = _post(client, ANALYSIS_ROUTE, {"tree": dict(derived), "latest": True}).json()
    (member,) = body["ref"]["members"]
    again = {"spec_id": member["spec"]["spec_id"], "content_version": member["content_version"]}
    if again != derived:
        raise SmokeFailure(f"latest on {derived} answered {again}")
    return "latest: the derived root answers the same ops_ and ocv_"


def create_card(client: httpx.Client, derived: Mapping[str, str]) -> str:
    """A table card on the derived frame, from the release's minimal table intent."""
    intents = _get(client, "/api/presentation/release").json()["minimal_intents"]
    table = {
        **intents["table"],
        "frame_ref": dict(derived),
        "select": {"stat_var": f"dcid:{derived['spec_id']}"},
    }
    card = _post(client, "/api/presentation/components", {"intent": table}, status=201).json()
    if not card["component_id"].startswith("cmp_") or card["revision"] != 1:
        raise SmokeFailure(f"card create answered {card}")
    return card["component_id"]


def check_share(client: httpx.Client, component_id: str) -> str:
    """Create a live share of the card, replace it with its token, refuse a wrong token, and
    take it down."""
    site = os.environ["DFO_PUBLIC_SITE_URL"]
    cards = {"cards": [{"component_id": component_id, "revision": 1}]}
    # A direct call names its caller from the rightmost X-Forwarded-For entry.
    response = client.post(
        SHARES_ROUTE, json={**cards, "latest": True}, headers={"X-Forwarded-For": "127.0.0.1"}
    )
    if response.status_code != 201:
        raise SmokeFailure(f"POST {SHARES_ROUTE} answered {response.status_code}: {response.text}")
    share = response.json()
    share_id, token = share["id"], share["write_token"]
    expected = {
        "url": f"{site}/s/{share_id}",
        "embed_url": f"{site}/embed/{share_id}",
        "data_urls": [f"{site}/api/v1/presentation/shares/{share_id}/slots/1/data"],
        "latest": True,
    }
    if {key: share[key] for key in expected} != expected or not token.startswith("shw_"):
        raise SmokeFailure(f"share create answered {share}")
    path = f"{SHARES_ROUTE}/{share_id}"
    replaced = client.put(path, json=cards, headers={WRITE_TOKEN: token})
    if replaced.status_code != 200 or replaced.json()["revision"] != 2:
        raise SmokeFailure(f"PUT {path} answered {replaced.status_code}: {replaced.text}")
    refused = client.put(path, json=cards, headers={WRITE_TOKEN: "shw_" + "0" * 22})
    if refused.status_code != 403 or refused.json()["code"] != "write_token_invalid":
        raise SmokeFailure(f"PUT {path} with a wrong token answered {refused.status_code}")
    deleted = client.delete(path, headers={WRITE_TOKEN: token})
    if deleted.status_code != 204:
        raise SmokeFailure(f"DELETE {path} answered {deleted.status_code}: {deleted.text}")
    gone = client.get(path)
    if gone.status_code != 404 or gone.json()["code"] != "not_found":
        raise SmokeFailure(f"GET {path} after the delete answered {gone.status_code}")
    return (
        f"share {share_id}: created live with its urls and token, replaced to revision 2, "
        "a wrong token refused 403, deleted, then 404"
    )


def analysis_step(
    client: httpx.Client,
    seeded: Mapping[str, Mapping[str, object]],
    readings: Mapping[str, Reading],
) -> list[str]:
    """Divide the German ODA reading by Kenya's population, read the derived frame back, read
    it again under `latest`, and draw and share a card of it. Answers a line per check."""
    oda, population = seeded[ODA_READING], seeded[POPULATION_READING]
    member = check_quotient(client, oda, population)
    derived = _derived_pair(member)
    expected = _expected_quotients(readings[ODA_READING], readings[POPULATION_READING])
    component_id = create_card(client, derived)
    return [
        f"analysis: quotient_of answered {derived['spec_id']} {derived['content_version']}",
        check_derived_reads(client, member, [oda, population], expected),
        check_latest(client, derived),
        f"card: {component_id} on the derived frame",
        check_share(client, component_id),
    ]


def main() -> int:
    seeded = json.loads(Path(os.environ["LOCAL_STACK_SEEDED"]).read_text())
    readings = {reading.name: reading for reading in load_fixture().readings}
    if not seeded:
        print("no seeded readings; run `stack.sh seed` first", file=sys.stderr)
        return 1
    checks = 0
    with httpx.Client(base_url=os.environ["LOCAL_STACK_BASE_URL"], timeout=60) as client:
        try:
            for entry in seeded:
                print(f"{entry['name']} {entry['spec_id']} {entry['content_version']}")
                print(f"  {check_versions(client, entry)}")
                artifacts = check_manifest(client, entry)
                print(f"  manifest: {len(artifacts)} artifacts")
                print(f"  {check_observations(client, entry, readings[entry['name']])}")
                print(f"  {check_files(client, artifacts)}")
                checks += 4
            by_name = {str(entry["name"]): entry for entry in seeded}
            for line in analysis_step(client, by_name, readings):
                print(line)
                checks += 1
        except SmokeFailure as failure:
            print(f"FAILED: {failure}", file=sys.stderr)
            return 1
    print(f"smoke passed: {checks} checks over {len(seeded)} readings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
