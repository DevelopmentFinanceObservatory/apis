"""An open-ended window means recorded data: `materialise.window.election_window` ends an
open range at the current year, and the datasets adapter applies it where it builds a
request, so one window governs the request's facet elections and the rows it reads.
"""

from __future__ import annotations

import pytest

from datasets_api.adapter import _requested_window
from materialise import window as window_module
from materialise.generation import _select_member_facets
from materialise.graph import Facet
from materialise.pipeline import MaterialisationRequest
from materialise.window import RequestedWindow, election_window

from .conftest import FakeGraphReader

_SV = "TESTFIXTURE/ElectionWindow/Amount"
_PLACE = "country/KEN"


def _range(start: int | None, end: int | None, *, end_unstated: bool = False) -> RequestedWindow:
    return RequestedWindow(kind="range", start_year=start, end_year=end, end_unstated=end_unstated)


@pytest.mark.parametrize(
    ("requested", "last_year", "elects_under"),
    [
        (_range(2016, None), None, _range(2016, 2026, end_unstated=True)),
        (_range(None, None), None, _range(None, 2026, end_unstated=True)),
        (_range(2026, None), 2032, _range(2026, 2026, end_unstated=True)),
        (_range(2027, None), None, _range(2027, None, end_unstated=True)),
        (_range(2027, None), 2032, _range(2027, 2032, end_unstated=True)),
        (_range(2016, 2030), None, _range(2016, 2030)),
        (_range(2027, 2029), None, _range(2027, 2029)),
        (RequestedWindow(kind="latest"), None, RequestedWindow(kind="latest")),
        (RequestedWindow(kind="all"), None, RequestedWindow(kind="all")),
    ],
)
def test_an_open_end_stops_at_the_current_year_and_every_other_window_is_kept(
    requested: RequestedWindow, last_year: int | None, elects_under: RequestedWindow
) -> None:
    """Under a 2026 clock an open end stops at 2026, unless the range starts later: then it
    runs to the caller's last year, or stays open. Each window it ends is marked as
    stating no end year. A closed range, `latest` and `all` are kept."""
    assert election_window(requested, clock_year=2026, last_year=last_year) == elects_under


def _facet(facet_id: str, provenance: str, years: range) -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id=provenance,
        unit="CurrencyUSD",
        obs_count=len(years),
        earliest_date=str(years[0]),
        latest_date=str(years[-1]),
    )


_RECORDED = _facet("fRecorded", "dc/base/Recorded", range(2000, 2025))
_PROJECTED = _facet("fProjected", "dc/base/Projected", range(2027, 2052))


@pytest.mark.parametrize(
    ("identity_window", "reads", "elected"),
    [
        (
            {"kind": "range", "start_year": 2016},
            _range(2016, 2026, end_unstated=True),
            "fRecorded",
        ),
        (
            {"kind": "range", "start_year": 2027, "end_year": 2029},
            _range(2027, 2029),
            "fProjected",
        ),
        (
            {"kind": "range", "start_year": 2027},
            _range(2027, None, end_unstated=True),
            "fProjected",
        ),
    ],
    ids=["since_2016", "2027_to_2029", "since_2027"],
)
def test_a_datasets_request_elects_and_reads_under_one_clamped_window(
    monkeypatch: pytest.MonkeyPatch,
    identity_window: dict[str, object],
    reads: RequestedWindow,
    elected: str,
) -> None:
    """The projection holds as many observations as the recorded series and reaches further.
    Under a 2026 clock the datasets adapter turns an identity's "since 2016" into 2016 to
    2026, the one window its request elects and reads under, so the recorded series, the
    only one with a point to 2026, is elected and no later year is read. A closed future
    range and an open one starting in the future keep their years and elect the
    projection."""
    monkeypatch.setattr(window_module, "current_year", lambda: 2026)
    request = MaterialisationRequest(
        stat_var=f"dcid:{_SV}",
        role_bindings={"observationAbout": [_PLACE]},
        declared_roles=("observationAbout",),
        facet_bindings={},
        window=_requested_window(identity_window),
    )

    selections = _select_member_facets(
        request, [_PROJECTED, _RECORDED], graph=FakeGraphReader(facets={})
    ).selections

    assert request.window == reads
    assert selections["provenance"].dispatch_facet_id == elected
