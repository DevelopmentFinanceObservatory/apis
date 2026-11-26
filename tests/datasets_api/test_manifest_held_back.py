"""The get_data manifest's `coverage.held_back` (`datasets_api.manifest.held_back_coverage`):
the partly reported years a frame's source holds, whatever its window, served when the window
opts in and holds a year of them.

The rig mints aid to Malawi from nine donors, eight reporting 2024 and two 2025
(`tests.datasets_api.test_elected_year`), from a source documented as the dev graph's DAC2A
node, which marks 2025 partial (`tests.datasets_api.test_incomplete_after`)."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import cast

import pytest

from datasets_api.manifest_models import Manifest
from materialise.window import RequestedWindow
from origin_catalogue import bound_catalogue

from ._version_record_store import FakeVersionRecordDatasetStore
from .test_elected_year import _AID, _MALAWI, _PROVENANCE, _Rig
from .test_incomplete_after import _DAC2A, _source_as


def _held_back(minted: Manifest) -> object:
    return minted.model_dump(mode="json")["coverage"].get("held_back", "absent")


def _run(*, served: bool, share: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "kind": "partial",
            "served": served,
            "source": _PROVENANCE,
            "source_name": _PROVENANCE,
            "start_year": 2025,
            "end_year": 2025,
            "reported_share": share,
        }
    ]


# Each donor reports a value of 1, so 2025's two over 2024's eight.
_VALUE_SHARE = {"basis": "value", "share": 0.25, "against": 2024}


@pytest.mark.parametrize(
    ("window", "end_year", "share"),
    [
        pytest.param(RequestedWindow(kind="latest"), 2024, _VALUE_SHARE, id="latest"),
        pytest.param(RequestedWindow(kind="all"), 2024, _VALUE_SHARE, id="all"),
        pytest.param(
            RequestedWindow(kind="range", start_year=2021, end_year=2024),
            2024,
            _VALUE_SHARE,
            id="to-2024",
        ),
        pytest.param(
            RequestedWindow(kind="range", start_year=2021, end_year=2022),
            2022,
            _VALUE_SHARE,
            id="to-2022",
        ),
        pytest.param(
            RequestedWindow(
                kind="range", start_year=2021, end_year=2022, include=frozenset({"partial"})
            ),
            2022,
            _VALUE_SHARE,
            id="opted-in-to-2022",
        ),
    ],
)
def test_a_frame_states_the_partial_year_held_back_whatever_its_window(
    version_store: FakeVersionRecordDatasetStore,
    window: RequestedWindow,
    end_year: int,
    share: dict[str, object],
) -> None:
    """A window ending before 2025 states it too, and one opting in to partial years that
    holds none of 2025 serves none of it. The mint counts the share over the rows it read of
    2024 and 2025 before its cuts, by value over their one currency. Under `latest`, its
    2024 rows come from its read of that year alone."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    with bound_catalogue(_source_as(_DAC2A)):
        minted, _ = rig.mint(window)

    assert minted.coverage.window is not None and minted.coverage.window["end_year"] == end_year
    assert _held_back(minted) == _run(served=False, share=share)


def test_a_frame_opting_in_to_the_partial_year_states_it_served(
    version_store: FakeVersionRecordDatasetStore,
) -> None:
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    with bound_catalogue(_source_as(_DAC2A)):
        minted, _ = rig.mint(RequestedWindow(kind="all", include=frozenset({"partial"})))

    assert minted.coverage.window is not None and minted.coverage.window["end_year"] == 2025
    assert _held_back(minted) == _run(served=True, share=_VALUE_SHARE)


def test_a_frame_of_a_source_marking_no_partial_year_states_none(
    version_store: FakeVersionRecordDatasetStore,
) -> None:
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    with bound_catalogue(_source_as(_DAC2A, marked=False)):
        minted, _ = rig.mint(RequestedWindow(kind="latest"))

    assert _held_back(minted) == "absent"


def _stored_held_back(store: FakeVersionRecordDatasetStore) -> object:
    with store.transaction() as tx:
        row = tx.fetch_one("SELECT coverage FROM datasets.version_record")
    assert row is not None
    return cast("dict[str, object]", row["coverage"]).get("held_back", "absent")


def test_a_later_mint_of_the_same_version_states_its_current_runs(
    version_store: FakeVersionRecordDatasetStore,
) -> None:
    """A window ending in 2024 mints the same rows whatever the source marks. A version
    stored without runs takes the run a later mint states, and the run goes
    once the release completes 2025."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})
    window = RequestedWindow(kind="range", start_year=2021, end_year=2024)
    with bound_catalogue(_source_as(_DAC2A)):
        first, _ = rig.mint(window)
    with version_store.transaction() as tx:
        row = tx.fetch_one("SELECT coverage FROM datasets.version_record")
        assert row is not None
        coverage = cast("dict[str, object]", row["coverage"])
        coverage.pop("held_back")
        tx.execute("UPDATE datasets.version_record SET coverage = %s", (json.dumps(coverage),))
    assert _stored_held_back(version_store) == "absent"

    rig.clock.at += timedelta(hours=25)
    with bound_catalogue(_source_as(_DAC2A)):
        stated, _ = rig.mint(window)

    assert stated.trace["served_from"] == "materialise"
    assert stated.content_version == first.content_version
    assert _held_back(stated) == _run(served=False, share=_VALUE_SHARE)
    assert _stored_held_back(version_store) == _run(served=False, share=_VALUE_SHARE)

    rig.clock.at += timedelta(hours=25)
    with bound_catalogue(_source_as(_DAC2A, marked=False)):
        completed, _ = rig.mint(window)

    assert completed.content_version == first.content_version
    assert _held_back(completed) == "absent"
    assert _stored_held_back(version_store) == "absent"
