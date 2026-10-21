"""`python -m graph_release.verify` checks that the serving image opens the graph release
`GRAPH_RELEASE_URI`'s pointer names for this image's format.

It opens the release as a serving process does at startup, through `ReleaseService`, which
builds the serving views and the origin view (`build_release_views`, `release_origin`), and
logs one `graph_release_verify` line with its outcome:

- `opened`, exit 0: the release opens and its views build.
- `refused`, exit 1: the pointer is unreadable, or the release does not download, map or build
  its views. The line names the exception.
- `no_uri`, exit 2: `GRAPH_RELEASE_URI` is unset or blank.
- `no_pointer`, exit 3: no release is published for this image's format.

The deploy runs it as an execution of the graph release job before the services move, and runs
one build when it fails (deploy/graph-release.md).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Final

from artifact_publication.jobs import emit
from graph_release.context import URI_ENV_VAR, release_uri
from graph_release.format import FORMAT, pointer_uri, read_pointer
from graph_release.service import ReleaseService

OPENED: Final[int] = 0
REFUSED: Final[int] = 1
NO_URI: Final[int] = 2
NO_POINTER: Final[int] = 3


def main() -> int:
    uri = release_uri()
    if uri is None:
        emit(
            "graph_release_verify",
            severity="ERROR",
            outcome="no_uri",
            detail=f"{URI_ENV_VAR} is not set",
        )
        return NO_URI
    try:
        pointer = read_pointer(uri)
        if pointer is None:
            emit(
                "graph_release_verify",
                severity="ERROR",
                outcome="no_pointer",
                format=FORMAT,
                pointer=pointer_uri(uri),
            )
            return NO_POINTER
        with tempfile.TemporaryDirectory(prefix="graph-release-verify-") as scratch:
            release = ReleaseService(uri=uri, local_dir=Path(scratch), recheck_seconds=0).snapshot()
    except Exception as exc:
        emit(
            "graph_release_verify",
            severity="ERROR",
            outcome="refused",
            format=FORMAT,
            pointer=pointer_uri(uri),
            error=type(exc).__name__,
            detail=str(exc),
        )
        return REFUSED
    emit(
        "graph_release_verify",
        outcome="opened",
        format=FORMAT,
        release_id=release.release_id,
        hold=pointer.hold,
    )
    return OPENED


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["NO_POINTER", "NO_URI", "OPENED", "REFUSED", "main"]
