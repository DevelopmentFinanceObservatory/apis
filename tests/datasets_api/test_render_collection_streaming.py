"""Collection rendering releases each member's table before requesting the next, checked by
weakrefs at the next pull. CPython frees the cycle-free table at once, so the check is exact.

Within a member, generator loop variables retain a bounded number of prior chunks until reassignment. Tests track live markers across 20 chunks and assert the peak stays below half the total.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator, Sequence

from datasets_api.mint_models import MintOptions
from datasets_api.render import CollectionMemberRendering, render_and_store_collection
from materialise.enrichment import EnrichmentLookup
from materialise.reshape import Table

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._render_inputs import PREPARED_AT, SITE

_COLUMNS = ("stat_var", "recipient", "time_period", "value", "unit", "_live_marker")
_GRAPH = FakeMintGraphReader()


class _LiveMarker:
    """Weakly referenceable cell used to track whether a rendered row remains alive."""


def _members_with_liveness_check(spec_ids: Sequence[str]) -> tuple[CollectionMemberRendering, ...]:
    """Builds members whose frame factories check that the previous member is released."""
    previous_marker: list[weakref.ReferenceType[_LiveMarker] | None] = [None]

    def _factory_for(spec_id: str) -> Callable[[], Iterator[Table]]:
        def _factory() -> Iterator[Table]:
            held = previous_marker[0]
            if held is not None:
                assert held() is None, (
                    f"member {spec_id!r} pulled its own frame_chunks() while a prior "
                    "member's own table was still alive"
                )
            marker = _LiveMarker()
            previous_marker[0] = weakref.ref(marker)
            yield Table(
                columns=_COLUMNS,
                rows=((f"dcid:{spec_id}", "country/KEN", "2020", 1.0, "USD", marker),),
            )

        return _factory

    return tuple(
        CollectionMemberRendering(
            spec_id=spec_id,
            frame_chunks=_factory_for(spec_id),
            provenance_refs=("p1",),
            cited_refs=("p1",),
            content_version=f"cv_{spec_id}",
            prepared_at=PREPARED_AT,
            entity_columns=("recipient",),
            resolved_stat_var_refs=(),
        )
        for spec_id in spec_ids
    )


def test_bare_collection_never_pulls_a_members_frame_while_a_prior_one_is_still_alive():
    members = _members_with_liveness_check(("member_a", "member_b", "member_c"))

    rendered = render_and_store_collection(
        members=members,
        options=MintOptions(format="csv", bundle=False),
        spec={"collection_id": "col_bare_streaming_bound"},
        citations={},
        reproducing_call=None,
        content_version="sha256:" + "0" * 64,
        artifact_store=FakeArtifactStore(),
        enrichment=EnrichmentLookup(_GRAPH),
        pivot_max_output_cells=5_000_000,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    artifacts, warnings = rendered.artifacts, rendered.warnings

    assert len(artifacts) == 3
    assert warnings == ()


def test_bundled_collection_never_pulls_a_members_frame_while_a_prior_one_is_still_alive():
    spec_ids = ("member_a", "member_b", "member_c")
    members = _members_with_liveness_check(spec_ids)

    rendered = render_and_store_collection(
        members=members,
        options=MintOptions(format="csv", bundle=True),
        spec={"collection_id": "col_bundle_streaming_bound", "members": list(spec_ids)},
        citations={},
        reproducing_call=None,
        content_version="sha256:" + "1" * 64,
        artifact_store=FakeArtifactStore(),
        enrichment=EnrichmentLookup(_GRAPH),
        pivot_max_output_cells=5_000_000,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    artifacts, warnings = rendered.artifacts, rendered.warnings

    assert len(artifacts) == 1
    assert warnings == ()


_CHUNK_COUNT = 20
"""Twenty chunks expose whether peak liveness stays bounded."""


def _member_with_chunk_liveness_check(
    spec_id: str, *, peak: list[int]
) -> CollectionMemberRendering:
    """Builds a member whose chunks carry weakly tracked markers and records peak liveness.

    The test checks the peak after rendering.
    """
    alive = [0]

    def _factory() -> Iterator[Table]:
        for i in range(_CHUNK_COUNT):
            marker = _LiveMarker()
            alive[0] += 1
            peak[0] = max(peak[0], alive[0])

            def _on_collected(alive: list[int] = alive) -> None:
                alive[0] -= 1

            weakref.finalize(marker, _on_collected)
            yield Table(
                columns=_COLUMNS,
                rows=((f"dcid:{spec_id}", "country/KEN", f"20{i:02d}", 1.0, "USD", marker),),
            )
            del marker

    return CollectionMemberRendering(
        spec_id=spec_id,
        frame_chunks=_factory,
        provenance_refs=("p1",),
        cited_refs=("p1",),
        content_version=f"cv_{spec_id}",
        prepared_at=PREPARED_AT,
        entity_columns=("recipient",),
        resolved_stat_var_refs=(),
    )


def test_bare_collection_never_holds_a_multi_chunk_members_raw_rows_all_at_once():
    peak = [0]
    member = _member_with_chunk_liveness_check("member_wide", peak=peak)

    rendered = render_and_store_collection(
        members=[member],
        options=MintOptions(format="csv", bundle=False),
        spec={"collection_id": "col_bare_chunk_streaming_bound"},
        citations={},
        reproducing_call=None,
        content_version="sha256:" + "2" * 64,
        artifact_store=FakeArtifactStore(),
        enrichment=EnrichmentLookup(_GRAPH),
        pivot_max_output_cells=5_000_000,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    artifacts, warnings = rendered.artifacts, rendered.warnings

    assert len(artifacts) == 1
    assert warnings == ()
    assert peak[0] < _CHUNK_COUNT // 2, (
        f"expected the member's own raw rows to stream in bounded slices, not all at "
        f"once: peak simultaneously-alive chunk count was {peak[0]} out of {_CHUNK_COUNT}"
    )


def test_bundled_collection_never_holds_a_multi_chunk_members_raw_rows_all_at_once():
    peak = [0]
    member = _member_with_chunk_liveness_check("member_wide", peak=peak)

    rendered = render_and_store_collection(
        members=[member],
        options=MintOptions(format="csv", bundle=True),
        spec={"collection_id": "col_bundle_chunk_streaming_bound", "members": ["member_wide"]},
        citations={},
        reproducing_call=None,
        content_version="sha256:" + "3" * 64,
        artifact_store=FakeArtifactStore(),
        enrichment=EnrichmentLookup(_GRAPH),
        pivot_max_output_cells=5_000_000,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    artifacts, warnings = rendered.artifacts, rendered.warnings

    assert len(artifacts) == 1
    assert warnings == ()
    assert peak[0] < _CHUNK_COUNT // 2, (
        f"expected the member's own raw rows to stream in bounded slices, not all at "
        f"once: peak simultaneously-alive chunk count was {peak[0]} out of {_CHUNK_COUNT}"
    )
