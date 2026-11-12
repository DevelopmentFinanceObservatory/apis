"""Open the staged dev graph release and read its coverage view, code index and declared
method defaults. Shared by the measurement scripts in this directory."""

from __future__ import annotations

import os
import resource
from pathlib import Path

import pyarrow.compute as pc

from dc_graph.client import DEFAULT_FACET_ARCS, declared_facet_policy
from graph_release.format import open_release
from graph_release.service import build_release_views

HERE = Path(os.environ.get("FACET_DIFF_DATA") or Path(__file__).resolve().parent / "data")
RELEASE_DIR = HERE / "graph-release"
LOCAL_DIR = HERE / "graph-release-local"


def peak_rss_mib() -> float:
    # macOS ru_maxrss is bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def load():
    """Open the release and bind the staged origin catalogue for the rest of the process."""
    from origin_catalogue.artifact import load_catalogue
    from origin_catalogue.context import bind_catalogue

    loaded = open_release(RELEASE_DIR.as_uri(), local_dir=LOCAL_DIR)
    views = build_release_views(loaded)
    bind_catalogue(load_catalogue((HERE / "origin-catalogue").as_uri()))
    return loaded, views


def declared_policies(loaded) -> dict:
    """{stat_var: {"measurementMethod": FacetDefaultPolicy}} from the release's edges."""
    arc = DEFAULT_FACET_ARCS["measurementMethod"]
    refs: dict[str, list[str]] = {}
    edges = loaded.tables["edges"]
    hit = edges.filter(pc.equal(edges["predicate"], arc))
    for s, o in zip(hit["subject_id"].to_pylist(), hit["object_id"].to_pylist(), strict=True):
        refs.setdefault(s, [])
        if o not in refs[s]:
            refs[s].append(o)
    out = {}
    for sv, r in refs.items():
        p = declared_facet_policy(sv, arc=arc, refs=r)
        if p is not None:
            out[sv] = {"measurementMethod": p}
    return out
