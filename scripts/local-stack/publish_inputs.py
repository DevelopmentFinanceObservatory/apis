"""Publish the graph release and origin catalogue the local stack serves from.

Every surface refuses to start without `GRAPH_RELEASE_URI` and `ORIGIN_CATALOGUE_URI`. This
publishes the test suite's own fixtures under the two URIs: the release `tests/conftest.py`
serves (the recorded code index and coverage slice) and the sample origin catalogue. A URI
that already holds a release or catalogue is left as it is.

The sample catalogue holds the population variable's node empty, which a label read takes for
a node with no name, so a card on a quotient by population would name its denominator
`Count_Person`. The local catalogue gives each StatVar of the stack's fixture that the sample
holds empty its fixture name (`_named_fixture_variables`), as a production build copies a base
variable's `name` arc into its base table.

Run from the repository root: `python scripts/local-stack/publish_inputs.py`.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

from upstream import load_fixture

from origin_catalogue.arc_tables import ArcTables
from origin_catalogue.un_mirror import UnArc
from tests.graph_release.support import build_session_release
from tests.origin_catalogue.support import publish_sample, sample_arc_tables


def _published(uri: str) -> bool:
    root = Path(urlparse(uri).path)
    return root.is_dir() and any(root.iterdir())


def _named_fixture_variables(arcs: ArcTables) -> ArcTables:
    """`arcs` with each fixture StatVar its base table holds empty given its fixture name."""
    names = {dcid: stat_var.label for dcid, stat_var in load_fixture().stat_vars.items()}
    named = {
        dcid: (UnArc(property="name", value=names[dcid]),)
        for dcid, held in arcs.base.items()
        if not held and dcid in names
    }
    return replace(arcs, base={**arcs.base, **named})


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
        publish_sample(catalogue_uri, arcs=_named_fixture_variables(sample_arc_tables()))
        print(f"published origin catalogue at {catalogue_uri}")


if __name__ == "__main__":
    main()
