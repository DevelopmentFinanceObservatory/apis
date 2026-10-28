"""Publish the graph release and origin catalogue the local stack serves from.

Every surface refuses to start without `GRAPH_RELEASE_URI` and `ORIGIN_CATALOGUE_URI`. This
publishes the test suite's own fixtures under the two URIs: the release `tests/conftest.py`
serves (the recorded code index and coverage slice) and the sample origin catalogue. A URI
that already holds a release or catalogue is left as it is.

Run from the repository root: `python scripts/local-stack/publish_inputs.py`.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from tests.graph_release.support import build_session_release
from tests.origin_catalogue.support import publish_sample


def _published(uri: str) -> bool:
    root = Path(urlparse(uri).path)
    return root.is_dir() and any(root.iterdir())


def main() -> None:
    release_uri = os.environ["GRAPH_RELEASE_URI"]
    catalogue_uri = os.environ["ORIGIN_CATALOGUE_URI"]
    if _published(release_uri):
        print(f"graph release already published at {release_uri}")
    else:
        build_session_release(release_uri)
        print(f"published graph release at {release_uri}")
    if _published(catalogue_uri):
        print(f"origin catalogue already published at {catalogue_uri}")
    else:
        publish_sample(catalogue_uri)
        print(f"published origin catalogue at {catalogue_uri}")


if __name__ == "__main__":
    main()
