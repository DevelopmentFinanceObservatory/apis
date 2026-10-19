"""Pins or releases the serving graph release, the rollback for a bad one:

    python -m graph_release.hold set <release_id>
    python -m graph_release.hold clear

`set` points at a published release and holds it, so the builder leaves the
pointer alone. `clear` lifts the hold. The held release keeps serving until
the builder publishes the next one. Both act on `GRAPH_RELEASE_URI`, or on
`--local-dir`, and take the builder's lease, so pruning cannot delete the
release a hold names. Both refuse while a build holds the lease, and refuse
with nothing written once too little of their own lease remains for the
pointer write to finish inside it.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

from pydantic import Field

from artifact_publication.jobs import (
    AliasOnlySettings,
    acquire_lease,
    emit,
    new_build_id,
    release_lease,
)
from graph_release.format import ReleaseFormatError
from graph_release.publish import HoldError, clear_hold, hold_release

LEASE_TTL_SECONDS: Final[float] = 900.0
"""How long the hold's lease lasts. It leaves 480 s before the write
deadline, which covers the manifest and pointer reads at their worst case
(180 s each, as below), so a slow but healthy hold still lands. A hold
killed mid-run keeps builds waiting this long."""

WRITE_MARGIN_SECONDS: Final[float] = 420.0
"""The lease time that must remain when the pointer write starts. The write
is one conditional upload and, on a failed precondition, one confirming
read. The GCS client retries each for up to 120 s and the last attempt can
run its 60 s request timeout past that, so each takes at most 180 s. The
other 60 s covers clock skew between this host and the builder that judges
the lease lapsed. A builder that reclaims the lapsed lease therefore finds
the pointer write finished, so its prune keeps the release the hold names."""


class HoldSettings(AliasOnlySettings):
    release_uri: str = Field(min_length=1, alias="GRAPH_RELEASE_URI")


def main(argv: Sequence[str] | None = None, *, clock: Callable[[], float] = time.monotonic) -> int:
    parser = argparse.ArgumentParser(prog="python -m graph_release.hold")
    parser.add_argument(
        "--local-dir", type=Path, help="act on this local directory in place of GRAPH_RELEASE_URI"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    held = commands.add_parser("set", help="point at a release and hold it")
    held.add_argument("release_id")
    commands.add_parser("clear", help="lift the hold")
    args = parser.parse_args(argv)

    overrides = {"release_uri": args.local_dir.resolve().as_uri()} if args.local_dir else {}
    root = HoldSettings(**overrides).release_uri
    # Read before the lease is taken, so the deadline errs early.
    write_by = clock() + LEASE_TTL_SECONDS - WRITE_MARGIN_SECONDS
    lease = acquire_lease(root, build_id=new_build_id(), ttl_seconds=LEASE_TTL_SECONDS)
    if lease is None:
        print(
            f"graph_release.hold {args.command}: a build holds the lease at {root}; "
            "nothing was written. Retry once it finishes.",
            file=sys.stderr,
        )
        return 1
    try:
        if args.command == "set":
            pointer = hold_release(root, args.release_id, write_by=write_by, clock=clock)
        else:
            pointer = clear_hold(root, write_by=write_by, clock=clock)
    except (HoldError, ReleaseFormatError) as exc:
        print(f"graph_release.hold {args.command}: {exc}", file=sys.stderr)
        return 1
    finally:
        release_lease(root, lease)
    emit("graph_release_hold", release_id=pointer.release_id, hold=pointer.hold)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["LEASE_TTL_SECONDS", "WRITE_MARGIN_SECONDS", "HoldSettings", "main"]
