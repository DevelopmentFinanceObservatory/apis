"""`python -m search.catalogue.verify` checks that a loadable search catalogue build is
published at this image's format root under `SEARCH_CATALOGUE_URI`.

It loads the build `current.json` names, as the search surface does (`load_published`), and
logs one `search_catalogue_verify` line with its outcome:

- `opened`, exit 0: the build loads. The line names its version, key, build commit and start.
- `refused`, exit 1: `current.json` is malformed, or the build it names is missing, does not
  parse or carries another version or key. The line names the error.
- `no_uri`, exit 2: `SEARCH_CATALOGUE_URI` is unset.
- `no_pointer`, exit 3: no build is published at this image's format root.

It checks that a build exists, not which run built it, so a deploy after a refused build at an
unchanged format passes on the previous build. The deploy runs it as an execution of the search
catalogue job after the job's own run, before the services move (deploy/search-catalogue.md).
"""

from __future__ import annotations

import sys
from typing import Final

from artifact_publication.jobs import emit
from artifact_publication.uri import ArtifactUriError
from search.catalogue.artifact import (
    FORMAT_ROOT,
    CatalogueArtifactError,
    load_published,
    read_pointer,
)
from search.catalogue.service import SearchCatalogueSettings

OPENED: Final[int] = 0
REFUSED: Final[int] = 1
NO_URI: Final[int] = 2
NO_POINTER: Final[int] = 3


def main() -> int:
    uri = SearchCatalogueSettings().artifact_uri
    if uri is None:
        emit(
            "search_catalogue_verify",
            severity="ERROR",
            outcome="no_uri",
            detail="SEARCH_CATALOGUE_URI is not set",
        )
        return NO_URI
    try:
        pointer, generation = read_pointer(uri)
        if pointer is None and generation is None:
            emit(
                "search_catalogue_verify",
                severity="ERROR",
                outcome="no_pointer",
                format_root=FORMAT_ROOT,
            )
            return NO_POINTER
        artifact = load_published(uri)
    except (ArtifactUriError, CatalogueArtifactError) as exc:
        emit(
            "search_catalogue_verify",
            severity="ERROR",
            outcome="refused",
            format_root=FORMAT_ROOT,
            error=type(exc).__name__,
            detail=str(exc),
        )
        return REFUSED
    emit(
        "search_catalogue_verify",
        outcome="opened",
        format_root=FORMAT_ROOT,
        catalogue_version=artifact.catalogue_version,
        key=artifact.key,
        build_commit=artifact.inputs.build_commit,
        started_at=artifact.started_at.isoformat(),
    )
    return OPENED


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["NO_POINTER", "NO_URI", "OPENED", "REFUSED", "main"]
