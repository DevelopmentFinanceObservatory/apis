"""Every production site that links an artifact, or looks one up by its link key, passes the
serving code index's `enrichment_digest`.

A file that can hold a party-kind cell is linked under a key salted with the digest
(`version_store.enriched_rendering_key`). A site that computes the key without it links or
looks up the unsalted one, so after the digest rotates it serves the old bytes or misses the
new ones, with no error. This test reads the source of `src/` and lists every such site that
passes no digest, except the few that handle a file no party kind reaches.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"

# Callees that take `enrichment_digest` as a keyword and must be given it.
_DIGEST_CALLEES = frozenset(
    {
        "artifact_rendering_keys",
        "ArtifactRecord",
        "stored_artifact_entry",
    }
)
# `version_store.rendering_key` hashes a rendering unsalted. Only the function that decides
# salting may call it for a link or lookup key.
_UNSALTED = "rendering_key"

type Site = tuple[str, str, str]
"""A call as `(path under src, enclosing function, callee)`."""

_EXEMPT: dict[Site, str] = {
    ("datasets_api/version_store.py", "enriched_rendering_key", "rendering_key"): (
        "the one function that decides whether a rendering is salted"
    ),
    ("datasets_api/maintenance.py", "_pinned_frame1_versions", "rendering_key"): (
        "looks up the canonical frame, which has no party-kind cell"
    ),
    ("datasets_api/maintenance.py", "run_backfill_frame1", "ArtifactRecord"): (
        "re-links a canonical frame document, which has no party-kind cell"
    ),
    ("datasets_api/render.py", "_file_shape", "rendering_key"): (
        "a six-character filename tag, not a link or lookup key"
    ),
    ("datasets_api/render.py", "render_and_store_canonical_frame", "ArtifactRecord"): (
        "links the canonical frame, which has no party-kind cell"
    ),
    ("datasets_api/operation_store.py", "compute_operation_id", "rendering_key"): (
        "an operation id, not an artifact link key"
    ),
}


def _callee(call: ast.Call) -> str | None:
    match call.func:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
    return None


def _calls(tree: ast.AST) -> Iterator[tuple[str, ast.Call]]:
    """Each call in `tree` with the name of the function enclosing it (`""` at module level)."""

    def walk(node: ast.AST, scope: str) -> Iterator[tuple[str, ast.Call]]:
        for child in ast.iter_child_nodes(node):
            inner = (
                child.name if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) else scope
            )
            if isinstance(child, ast.Call):
                yield scope, child
            yield from walk(child, inner)

    yield from walk(tree, "")


def link_sites_without_a_digest(source: str, *, path: str) -> list[Site]:
    """The calls in `source` that compute a link or lookup key, or build a link, with no
    digest: a call to `rendering_key`, or to one of `_DIGEST_CALLEES` with no
    `enrichment_digest` keyword."""
    found: list[Site] = []
    for scope, call in _calls(ast.parse(source)):
        callee = _callee(call)
        if callee == _UNSALTED or (
            callee in _DIGEST_CALLEES
            and not any(keyword.arg == "enrichment_digest" for keyword in call.keywords)
        ):
            found.append((path, scope, callee))
    return found


def _production_sites() -> list[Site]:
    return [
        site
        for file in sorted(_SRC.rglob("*.py"))
        for site in link_sites_without_a_digest(
            file.read_text(encoding="utf-8"), path=file.relative_to(_SRC).as_posix()
        )
    ]


def test_every_link_and_lookup_site_passes_the_enrichment_digest() -> None:
    sites = set(_production_sites())

    assert sites - _EXEMPT.keys() == set(), (
        "pass `enrichment_digest=<serving code index>.enrichment_digest`, or list the site "
        "in _EXEMPT with the reason a party kind never reaches it"
    )


def test_every_exemption_still_names_a_site_that_passes_no_digest() -> None:
    assert _EXEMPT.keys() - set(_production_sites()) == set()


@pytest.mark.parametrize(
    "source",
    [
        "def f(options, site):\n    return artifact_rendering_keys(options, site=site)\n",
        "def f(rendering):\n    return rendering_key(rendering)\n",
        "def f(x):\n    return ArtifactRecord(artifact_digest=x, rendering={}, media_type='m')\n",
        "def f(location):\n    return stored_artifact_entry(location, subject_id='s', content_version='c')\n",
    ],
    ids=["artifact-key", "unsalted-key", "artifact-record", "stored-entry"],
)
def test_the_scan_flags_a_site_that_passes_no_digest(source: str) -> None:
    ((_, scope, _),) = link_sites_without_a_digest(source, path="x.py")
    assert scope == "f"


def test_the_scan_passes_a_site_that_passes_the_digest() -> None:
    source = (
        "def f(options, site, digest):\n"
        "    return artifact_rendering_keys(options, site=site, enrichment_digest=digest)\n"
    )
    assert link_sites_without_a_digest(source, path="x.py") == []
