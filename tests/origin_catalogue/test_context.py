"""Tests the request-scoped catalogue binding and the suite's fake catalogue."""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pytest

from origin_catalogue import (
    BASE,
    UN_SYSTEM,
    Catalogue,
    CatalogueNotBound,
    SeriesAdmission,
    UnReadFailure,
    UnUnserved,
    bind_catalogue,
    bound_catalogue,
    current_catalogue,
    note_un_blocked_places,
    note_un_unserved,
    recording_un_blocked_places,
    unbind_catalogue,
)
from tests.origin_catalogue.fake_catalogue import (
    FakeCatalogue,
    provenance_row,
    restrictive_catalogue,
)

_WDI = "dc/base/WorldDevelopmentIndicators"


def test_suite_binds_a_fake_catalogue_that_admits_everything() -> None:
    catalogue = current_catalogue()
    assert isinstance(catalogue, FakeCatalogue)
    assert catalogue.is_admitted("dc/base/UN_SDG")
    assert catalogue.admit_series(["a", "b", "a"]) == SeriesAdmission(
        admitted=("a", "b"), withheld=()
    )
    assert catalogue.admit_candidate("Count_Person") == BASE
    assert catalogue.admit_candidate("undata/who/X") == UN_SYSTEM
    assert catalogue.data_commons("undata/who/X") == (UN_SYSTEM,)
    assert catalogue.provenance("provenance/OECDCreditorReportingSystem") is None
    assert catalogue.provenance_rows() == ()


def test_unbound_context_raises() -> None:
    with pytest.raises(CatalogueNotBound):
        contextvars.Context().run(current_catalogue)


def test_bound_catalogue_nests_and_restores() -> None:
    outer = current_catalogue()
    inner = restrictive_catalogue()
    with bound_catalogue(inner) as bound:
        assert bound is inner
        assert current_catalogue() is inner
    assert current_catalogue() is outer


def test_bind_and_unbind_restore_the_previous_binding() -> None:
    outer = current_catalogue()
    token = bind_catalogue(restrictive_catalogue())
    assert current_catalogue() is not outer
    unbind_catalogue(token)
    assert current_catalogue() is outer


def test_pool_threads_see_the_binding_only_under_a_copied_context() -> None:
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(CatalogueNotBound):
            pool.submit(current_catalogue).result()
        assert pool.submit(contextvars.copy_context().run, current_catalogue).result() is (
            current_catalogue()
        )


def test_bind_catalogue_fixture_installs_a_restrictive_catalogue(
    bind_catalogue: Callable[[Catalogue], None],
) -> None:
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row(_WDI, BASE)], withheld_variables=["Count_Person"]
        )
    )
    catalogue = current_catalogue()
    assert catalogue.is_admitted(_WDI)
    assert not catalogue.is_admitted("dc/base/UN_SDG")
    assert catalogue.admit_candidate("Count_Person") is None


def test_a_record_snapshot_joins_another_record_once() -> None:
    failure = UnReadFailure(variables=("undata/who/X",), places=("country/KEN",), detail="503")
    with recording_un_blocked_places() as probe_record:
        note_un_blocked_places(["country/GRC", "country/SVN"])
        probe_record.add_failure(failure)
    unserved = probe_record.snapshot()
    assert unserved == UnUnserved(places=("country/GRC", "country/SVN"), failures=(failure,))

    with recording_un_blocked_places() as record:
        note_un_blocked_places(["country/ESH"])
        note_un_unserved(unserved)
        note_un_unserved(unserved)

    assert record.places() == ("country/ESH", "country/GRC", "country/SVN")
    assert record.failures() == (failure,)


def test_a_snapshot_noted_outside_a_record_is_dropped() -> None:
    note_un_unserved(UnUnserved(places=("country/GRC",)))

    with recording_un_blocked_places() as record:
        pass
    assert record.places() == ()
