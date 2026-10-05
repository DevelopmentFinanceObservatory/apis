"""A definition whose window leaves its end unstated reads to the current year
(`materialise.window.election_window`), so its stored head is current only while a read would
end at the same year. Its currency signal carries the window it read, and a new calendar year
re-reads the head on every path: a fresh mint, a GET from a confirmed head, a GET inside the
recheck interval and the options route. The re-read keeps its content version when the data it
reads is unchanged.
A window stating its end, and one with no years, hold no clock.
"""

from __future__ import annotations

import pytest

from materialise import window as window_module
from materialise.graph import Facet

from .test_definition_door import _CSV_HEADER, _RECIPIENT, _SCOPE, _STAT_VAR, _STAT_VAR_REF
from .test_definition_door_currency import (
    _PAST_INTERVAL,
    _client,
    _counting_transport,
    _CoverageFromFacets,
    _FakeHeadStore,
    _mutable_graph,
    _row,
)

_SINCE_2016 = {"kind": "range", "start_year": 2016}
_2016_TO_2030 = {"kind": "range", "start_year": 2016, "end_year": 2030}


def _years_facet(years: range) -> Facet:
    return Facet(
        facet_id="facet1",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=len(years),
        earliest_date=str(years[0]),
        latest_date=str(years[-1]),
    )


class _Dataset:
    """One spec over a series holding a point in each of `years`, minted under `window` with
    the clock at `clock`, and a client to read it back."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, *, years: range, window: dict | None, clock: int
    ):
        self._monkeypatch = monkeypatch
        self.set_clock(clock)
        self.facets = {(_STAT_VAR, _RECIPIENT): [_years_facet(years)]}
        self.body = {"text": self._rows(years)}
        transport, self.sdmx_calls = _counting_transport(self.body)
        self.head_store = _FakeHeadStore()
        self.client = _client(
            graph=_mutable_graph(self.facets),
            sdmx_transport=transport,
            head_writer=self.head_store,
            coverage=_CoverageFromFacets(self.facets),
        )
        scope = {**_SCOPE, **({"window": window} if window is not None else {})}
        self.minted = self.mint(scope)
        self.spec_id = self.minted["definition"]["spec_id"]
        self._scope = scope

    @staticmethod
    def _rows(years: range) -> str:
        return _CSV_HEADER + "".join(
            _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000", period=str(year)) for year in years
        )

    def set_clock(self, year: int) -> None:
        self._monkeypatch.setattr(window_module, "current_year", lambda: year)

    def set_data(self, years: range) -> None:
        self.facets[(_STAT_VAR, _RECIPIENT)] = [_years_facet(years)]
        self.body["text"] = self._rows(years)

    def mint(self, scope: dict | None = None) -> dict:
        response = self.client.post(
            "/api/datasets",
            json={
                "schema_version": 1,
                "input": {
                    "kind": "statvar_set",
                    "statvars": [_STAT_VAR_REF],
                    "scope": scope if scope is not None else self._scope,
                },
                "options": {"bundle": False},
            },
        )
        assert response.status_code == 200, response.text
        return response.json()

    def get(self) -> dict:
        response = self.client.get(f"/api/datasets/{self.spec_id}")
        assert response.status_code == 200, response.text
        return response.json()

    def options(self) -> dict:
        response = self.client.get(f"/api/datasets/{self.spec_id}/options")
        assert response.status_code == 200, response.text
        return response.json()

    def window_read(self) -> tuple[int | None, int | None] | None:
        (signal,) = self.head_store.head_for(self.spec_id).signal
        return signal.clock_window


def _served(body: dict) -> tuple[int, int]:
    window = body["coverage"]["window"]
    return window["start_year"], window["end_year"]


@pytest.mark.parametrize(
    ("data", "served_2027", "same_content"),
    [(range(2016, 2022), (2016, 2021), True), (range(2016, 2031), (2016, 2027), False)],
    ids=["data_ended_before_the_clock", "data_runs_past_the_clock"],
)
def test_a_fresh_mint_in_a_new_year_reads_to_that_year(
    monkeypatch: pytest.MonkeyPatch,
    data: range,
    served_2027: tuple[int, int],
    same_content: bool,
) -> None:
    dataset = _Dataset(monkeypatch, years=data, window=_SINCE_2016, clock=2026)
    assert dataset.window_read() == (2016, 2026)

    dataset.set_clock(2027)
    reminted = dataset.mint()

    assert _served(reminted) == served_2027
    assert (reminted["content_version"] == dataset.minted["content_version"]) is same_content
    assert dataset.window_read() == (2016, 2027)


@pytest.mark.parametrize("confirmed", ["past_the_recheck_interval", "inside_it"])
@pytest.mark.parametrize(
    ("data", "served_2027", "same_content"),
    [(range(2016, 2022), (2016, 2021), True), (range(2016, 2031), (2016, 2027), False)],
    ids=["data_ended_before_the_clock", "data_runs_past_the_clock"],
)
def test_a_get_in_a_new_year_re_reads_an_open_ended_head(
    monkeypatch: pytest.MonkeyPatch,
    data: range,
    served_2027: tuple[int, int],
    same_content: bool,
    confirmed: str,
) -> None:
    """A head minted in 2026 is re-read by the first GET of 2027, whether the head was last
    confirmed before the recheck interval (the probe path) or inside it (the fast path). With
    data ending in 2021 the re-read serves the same content version; with data running to
    2030 it serves 2027 too."""
    dataset = _Dataset(monkeypatch, years=data, window=_SINCE_2016, clock=2026)
    before = dataset.get()
    assert before["trace"]["served_from"] == "head"
    if confirmed == "past_the_recheck_interval":
        dataset.head_store.age_head(dataset.spec_id, age=_PAST_INTERVAL)
    sdmx_reads = dataset.sdmx_calls.n

    dataset.set_clock(2027)
    after = dataset.get()

    assert after["trace"]["served_from"] == "materialise"
    assert dataset.sdmx_calls.n == sdmx_reads + 1
    assert _served(after) == served_2027
    assert (after["content_version"] == before["content_version"]) is same_content
    assert dataset.head_store.head_for(dataset.spec_id).content_version == after["content_version"]
    assert dataset.window_read() == (2016, 2027)
    # The re-read head is current for the rest of 2027.
    assert dataset.get()["trace"]["served_from"] == "head"


def test_the_options_route_re_reads_a_head_the_clock_moved_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In 2027 the first options call re-reads a head minted in 2026 and, finding the same
    content version, advances its window. The second options call and a GET serve the head."""
    dataset = _Dataset(monkeypatch, years=range(2016, 2022), window=_SINCE_2016, clock=2026)
    sdmx_reads = dataset.sdmx_calls.n

    dataset.set_clock(2027)
    first = dataset.options()
    second = dataset.options()

    assert first["trace"]["served_from"] == "materialise"
    assert second["trace"]["served_from"] == "head"
    assert dataset.sdmx_calls.n == sdmx_reads + 1
    assert dataset.window_read() == (2016, 2027)
    head = dataset.head_store.head_for(dataset.spec_id)
    assert head.content_version == dataset.minted["content_version"]
    assert dataset.get()["trace"]["served_from"] == "head"
    assert dataset.sdmx_calls.n == sdmx_reads + 1


def test_the_options_route_leaves_a_head_whose_re_read_holds_new_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With data running to 2030, the 2027 re-read holds a year the stored version lacks. The
    options route records no version, so the head keeps its 2026 window and each options call
    re-reads it until a GET records the new version."""
    dataset = _Dataset(monkeypatch, years=range(2016, 2031), window=_SINCE_2016, clock=2026)
    sdmx_reads = dataset.sdmx_calls.n

    dataset.set_clock(2027)
    first = dataset.options()
    second = dataset.options()

    assert [first["trace"]["served_from"], second["trace"]["served_from"]] == [
        "materialise",
        "materialise",
    ]
    assert dataset.sdmx_calls.n == sdmx_reads + 2
    assert dataset.window_read() == (2016, 2026)
    assert (
        dataset.head_store.head_for(dataset.spec_id).content_version
        == dataset.minted["content_version"]
    )
    assert dataset.get()["content_version"] != dataset.minted["content_version"]
    assert dataset.window_read() == (2016, 2027)


@pytest.mark.parametrize("confirmed", ["past_the_recheck_interval", "inside_it"])
@pytest.mark.parametrize("window", [_2016_TO_2030, None], ids=["closed", "dateless"])
def test_a_window_holding_no_clock_serves_its_head_in_a_new_year(
    monkeypatch: pytest.MonkeyPatch, window: dict | None, confirmed: str
) -> None:
    dataset = _Dataset(monkeypatch, years=range(2016, 2031), window=window, clock=2026)
    assert dataset.window_read() is None
    if confirmed == "past_the_recheck_interval":
        dataset.head_store.age_head(dataset.spec_id, age=_PAST_INTERVAL)
    sdmx_reads = dataset.sdmx_calls.n

    dataset.set_clock(2027)
    after = dataset.get()

    assert after["trace"]["served_from"] == "head"
    assert after["content_version"] == dataset.minted["content_version"]
    assert dataset.sdmx_calls.n == sdmx_reads


def test_a_window_starting_after_the_clock_is_re_read_once_the_clock_reaches_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Since 2028" asks about the future and reads every year from 2028 in 2026 and 2027.
    In 2028 it reads 2028 alone, so the head is re-read then and not before."""
    dataset = _Dataset(
        monkeypatch,
        years=range(2016, 2031),
        window={"kind": "range", "start_year": 2028},
        clock=2026,
    )
    assert dataset.window_read() == (2028, None)
    assert _served(dataset.minted) == (2028, 2030)

    dataset.set_clock(2027)
    assert dataset.get()["trace"]["served_from"] == "head"

    dataset.set_clock(2028)
    after = dataset.get()
    assert after["trace"]["served_from"] == "materialise"
    assert _served(after) == (2028, 2028)
    assert dataset.window_read() == (2028, 2028)
