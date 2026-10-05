"""`python -m origin_catalogue.verify` checks that the serving image can load
its origin catalogue. It loads the build `current.json` names with
`load_catalogue`, under the memory budget and UN setting the services read at
startup (`CatalogueSettings`), and checks only this image's format root. It
prints the build id, row counts and age, notes a build whose UN rows were
carried forward from an earlier one, and returns a non-zero status when no
build loads.

Pass an artifact URI as an argument, or set `ORIGIN_CATALOGUE_URI`. A job
run that skips or discards its build exits 0 without publishing, so deploys
run this check before moving services to the image.

Usage:
    python -m origin_catalogue.verify                       # reads ORIGIN_CATALOGUE_URI
    python -m origin_catalogue.verify file:///tmp/catalogue  # verifies an explicit URI
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

from origin_catalogue.artifact import load_catalogue
from origin_catalogue.service import CatalogueSettings

_ARTIFACT_URI_ENV_VAR = "ORIGIN_CATALOGUE_URI"


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1:
        print(
            f"usage: python -m origin_catalogue.verify [artifact_uri] "
            f"(reads {_ARTIFACT_URI_ENV_VAR} when omitted)",
            file=sys.stderr,
        )
        return 2
    settings = CatalogueSettings()
    artifact_uri = args[0] if args else settings.artifact_uri
    if not artifact_uri:
        print(
            f"origin_catalogue.verify: no artifact_uri given and {_ARTIFACT_URI_ENV_VAR} is not "
            "set; nothing to verify",
            file=sys.stderr,
        )
        return 1
    try:
        snapshot = load_catalogue(
            artifact_uri,
            max_resident_bytes=settings.max_resident_bytes,
            un_system_enabled=settings.un_system_enabled,
        )
    except Exception as exc:
        print(
            f"origin_catalogue.verify: {artifact_uri!r} did not load a compatible origin "
            f"catalogue: {exc}",
            file=sys.stderr,
        )
        return 1
    manifest = snapshot.manifest
    now = datetime.now(UTC)
    age_days = (now - manifest.built_at).total_seconds() / 86_400
    staleness = "stale" if snapshot.is_stale(now=now) else "fresh"
    print(
        f"origin_catalogue.verify: {artifact_uri!r} loaded build {manifest.build_id!r}, built "
        f"{manifest.built_at.isoformat()} ({age_days:.1f} days ago, {staleness}); row counts "
        f"{json.dumps(dict(manifest.row_counts), sort_keys=True)}"
    )
    carried = manifest.un_carried_forward
    if carried is not None:
        print(
            f"origin_catalogue.verify: build {manifest.build_id!r} carries its UN rows and "
            f"refused places from build {carried.from_build!r}: {carried.reason}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
