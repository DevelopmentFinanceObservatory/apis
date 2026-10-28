"""Check every seeded reading over HTTP against the running local stack.

For each reading `seed.py` recorded in `$LOCAL_STACK_SEEDED`:

- `GET /api/datasets/{spec_id}/versions` lists its content version;
- `GET /api/datasets/{spec_id}/{content_version}` answers its manifest, with artifacts;
- `GET .../observations` answers every fixture series with the fixture's points;
- each artifact's files link answers 302, and the bytes at the location hash to its digest.

`analysis_step` runs last. Exits 1 on the first failed check.

Run through `scripts/local-stack/stack.sh smoke`.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
from upstream import Reading, load_fixture

ANALYSIS_ROUTE = "/api/datasets/analysis"


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


def analysis_step(client: httpx.Client, seeded: Sequence[Mapping[str, object]]) -> str:
    """Run an analysis tree over a seeded reading through `POST /api/datasets/analysis`.

    This branch has no analysis route, so the step only confirms the route is absent. Once the
    route exists the probe answers something other than 404, and this step fails until it is
    written to post a tree over `seeded[0]` and check the derived frame.
    """
    probe = client.post(ANALYSIS_ROUTE, json={})
    if probe.status_code != 404:
        raise SmokeFailure(
            f"POST {ANALYSIS_ROUTE} answered {probe.status_code}; the route exists, so "
            "analysis_step must run a tree over a seeded reading"
        )
    return f"analysis: skipped, {ANALYSIS_ROUTE} is not on this branch"


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
            print(analysis_step(client, seeded))
        except SmokeFailure as failure:
            print(f"FAILED: {failure}", file=sys.stderr)
            return 1
    print(f"smoke passed: {checks} checks over {len(seeded)} readings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
