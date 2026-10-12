"""Mints a materialisation's `content_version` from its canonical frame through
`dfo_frame_grammar.content_version`. The token hashes the frame alone, so identical content is
one version whatever facets, revisions, or conversion produced it. The trace and the
materialisation row record that lineage beside the token."""

from __future__ import annotations

from collections.abc import Iterable

from dfo_frame_grammar import (
    Frame,
    mint_content_version,
    recompute_content_version,
    stream_frame_sha256,
)

from materialise.recipe_bodies import RECIPE_BODIES, recipe_body_digest


def recipe_dependency_fingerprint(recipe_id: str) -> tuple[str, str] | None:
    """Returns `(recipe_id, recipe_body_digest)` for the current body, or `None` when
    `RECIPE_BODIES` has no entry. Materialisation and currency rechecks use this same rule."""
    recipe = RECIPE_BODIES.get(recipe_id)
    if recipe is None:
        return None
    return recipe_id, recipe_body_digest(recipe)


def mint_version(frame: Frame) -> str:
    """Single-frame convenience wrapper around `mint_version_from_chunks`."""
    return mint_version_from_chunks((frame,))


def mint_version_from_chunks(chunks: Iterable[Frame]) -> str:
    """Hashes validated grammar `Frame` chunks in iteration order, then mints their version.
    Accepts grammar `Frame` chunks built by `Frame.build`, which normalises each cell once. A
    single chunk hashes like `frame_sha256`. Multiple chunks hash like their concatenated rows
    while holding at most one chunk's rows in memory. Use `mint_version_from_digest` when the
    frame digest is already computed."""
    return mint_version_from_digest(stream_frame_sha256(chunks))


def mint_version_from_digest(digest: str) -> str:
    """Mints from an existing frame digest. Group generation hashes during its single fetch,
    and `_finish_group` uses this function to avoid rereading rows."""
    return mint_content_version(frame_sha256=digest)


def recompute_version(canonical_bytes: bytes) -> str:
    """Recomputes the content version from canonical frame bytes."""
    return recompute_content_version(canonical_bytes)
