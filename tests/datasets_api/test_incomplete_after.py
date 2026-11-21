"""The mint's record of the rows past the last complete year of their source
(`coverage.incomplete_after`, `datasets_api.manifest.incomplete_after`), read from the
`partialObservationDate` the origin catalogue carries for each source.

The rig mints aid to Malawi from nine donors, two of which report 2025
(`tests.datasets_api.test_elected_year`). Its source takes the arcs the dev graph's Provenance
nodes held on 2026-11-21 (`tests/fixtures/dev_provenance_nodes.json`): DAC2A marks 2025 partial,
DAC1 marks nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import cast

from assembler.mint.provenance import _partial_period
from assembler.ports import FrameRef
from assembler.testing import build_resolved_frame
from datasets_api.manifest_models import Manifest
from materialise.window import RequestedWindow
from origin_catalogue import bound_catalogue
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, dev_provenance_nodes, documenting

from ._version_record_store import FakeVersionRecordDatasetStore
from .test_elected_year import _AID, _MALAWI, _MWI, _PROVENANCE, _Rig

_ALL = RequestedWindow(kind="all")
_DAC2A = "provenance/OECDDAC2A"
_DAC1 = "provenance/OECDDAC1"


def _source_as(node: str, *, marked: bool = True) -> FakeCatalogue:
    """A catalogue documenting the rig's source with the arcs the dev graph held for `node`,
    less its `partialObservationDate` when not `marked`, as the prod graph holds them."""
    arcs = dict(dev_provenance_nodes()[node])
    if not marked:
        arcs.pop("partialObservationDate", None)
    return documenting({_PROVENANCE: arcs})


def _recorded(minted: Manifest) -> list[dict[str, object]] | None:
    held = minted.coverage.incomplete_after
    return None if held is None else [entry.model_dump() for entry in held]


def _flagged(minted: Manifest) -> bool:
    """`partial_period` of a card drawing a 2025 row of the minted version."""
    frame = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_x", content_version="cv_x"),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": f"dcid:{_AID}",
                "recipient": _MWI,
                "time_period": "2025",
                "value": 1.0,
                "unit": None,
                "provenance": _PROVENANCE,
            }
        ],
        manifest={"coverage": minted.coverage.model_dump(mode="json")},
    )
    return _partial_period(resolved_frame=frame, rows=None)


def test_a_frame_reaching_its_source_s_partial_year_records_it(
    version_store: FakeVersionRecordDatasetStore,
) -> None:
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    with bound_catalogue(_source_as(_DAC2A)):
        minted, _ = rig.mint(_ALL)

    assert _recorded(minted) == [
        {
            "provenance": _PROVENANCE,
            "stat_vars": [f"dcid:{_AID}"],
            "source": _PROVENANCE,
            "last_complete_year": 2024,
        }
    ]
    assert _flagged(minted)


def test_a_frame_ending_before_its_source_s_partial_year_records_none(
    version_store: FakeVersionRecordDatasetStore,
) -> None:
    """A latest window elects 2024, since 2025 holds two of nine donors."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    with bound_catalogue(_source_as(_DAC2A)):
        minted, _ = rig.mint()

    assert minted.coverage.window is not None and minted.coverage.window["end_year"] == 2024
    assert _recorded(minted) == []


def test_the_prod_graph_s_unmarked_sources_record_an_empty_list(
    version_store: FakeVersionRecordDatasetStore,
) -> None:
    """Prod states no `partialObservationDate` and serves no 2025 rows: the version is
    judged, and nothing is partial."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    with bound_catalogue(_source_as(_DAC2A, marked=False)):
        minted, _ = rig.mint()

    assert _recorded(minted) == []
    assert not _flagged(minted)


def test_a_dac1_frame_reaching_2025_is_not_flagged_and_keeps_2025(
    version_store: FakeVersionRecordDatasetStore,
) -> None:
    """The pipeline marks no DAC1 year partial, so DAC1's newest year, preliminary but not
    partial, stands as every other year does."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    with bound_catalogue(_source_as(_DAC1)):
        minted, _ = rig.mint(_ALL)

    assert minted.coverage.window is not None and minted.coverage.window["end_year"] == 2025
    assert _recorded(minted) == []
    assert not _flagged(minted)


def test_a_mint_with_no_catalogue_bound_records_nothing_and_reads_unknown(
    version_store: FakeVersionRecordDatasetStore,
) -> None:
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    with bound_catalogue(FakeCatalogue()):
        minted, _ = rig.mint(_ALL)

    assert _recorded(minted) == [], "a catalogue that lists no source marks nothing"


def test_a_later_mint_of_the_same_version_replaces_its_partial_years(
    version_store: FakeVersionRecordDatasetStore,
) -> None:
    """The full release completes 2025 while the frame's rows stay the same, so the mint
    after the head's day-long lifetime mints the same version again, and the store takes that mint's judgement."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})
    with bound_catalogue(_source_as(_DAC2A)):
        first, _ = rig.mint(_ALL)
    assert cast("Mapping[str, object]", rig.stored_coverage())["incomplete_after"]

    rig.clock.at += timedelta(hours=25)
    with bound_catalogue(_source_as(_DAC2A, marked=False)):
        second, _ = rig.mint(_ALL)

    assert second.trace["served_from"] == "materialise"
    assert second.content_version == first.content_version
    assert rig.stored_coverage()["incomplete_after"] == []
    assert _recorded(second) == []
