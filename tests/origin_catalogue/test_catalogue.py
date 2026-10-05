"""Tests admission over a snapshot: provenances, DFO precedence, candidates, origins, places."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import Any

import pytest

from origin_catalogue import (
    BASE,
    DFO,
    UN_SYSTEM,
    CatalogueManifest,
    CatalogueSnapshot,
    CatalogueTableError,
    CatalogueTables,
    SeriesAdmission,
)
from origin_catalogue.catalogue import BaseVariable, CardFacts
from origin_catalogue.topic_walk import TOPIC_EXPAND_MAX_MEMBERS
from tests.origin_catalogue.fake_catalogue import provenance_row
from tests.origin_catalogue.support import (
    BUILT_AT,
    CENSUS_POVERTY,
    CRS,
    CRS_VARIABLE,
    EUROSTAT_GDP,
    GDP,
    POPULATION,
    UN_WHO,
    WDI,
    sample_tables,
)

_NOT_ADMITTED = "dc/base/UN_SDG"


def _snapshot(*, un_system_enabled: bool = False) -> CatalogueSnapshot:
    manifest = CatalogueManifest(
        build_id="b",
        built_at=BUILT_AT,
        build_seconds=0.0,
        identity_inputs={},
        registry_hash="",
        row_counts={},
        resident_bytes=0,
        overlaps=(),
        files={},
    )
    return CatalogueSnapshot(
        manifest=manifest, tables=sample_tables(), un_system_enabled=un_system_enabled
    )


def test_provenance_returns_origin_source_and_labels() -> None:
    row = _snapshot().provenance(WDI)
    assert row is not None
    assert row.data_commons == BASE
    assert (row.source.ref, row.data_commons.label) == (
        "dc/s/WorldBank",
        "Data Commons by Google",
    )
    assert _snapshot().provenance(_NOT_ADMITTED) is None


def test_un_row_keeps_its_labels_while_the_un_origin_is_off() -> None:
    row = _snapshot().provenance(UN_WHO)
    assert row is not None
    assert (row.data_commons, row.segment) == (UN_SYSTEM, "who")


@pytest.mark.parametrize(
    ("provenance_id", "off", "on"),
    [
        (CRS, True, True),
        (WDI, True, True),
        (UN_WHO, False, True),
        (_NOT_ADMITTED, False, False),
        ("undata/p/DPO", False, False),
    ],
)
def test_is_admitted_gates_the_un_origin_on_the_setting(
    provenance_id: str, off: bool, on: bool
) -> None:
    assert _snapshot().is_admitted(provenance_id) is off
    assert _snapshot(un_system_enabled=True).is_admitted(provenance_id) is on


def test_the_un_origin_is_on_by_default() -> None:
    snapshot = CatalogueSnapshot(manifest=_snapshot().manifest, tables=sample_tables())
    assert snapshot.is_admitted(UN_WHO)


def test_admit_series_withholds_unadmitted_provenances_in_order() -> None:
    admission = _snapshot().admit_series([_NOT_ADMITTED, WDI, UN_WHO, WDI, EUROSTAT_GDP])
    assert admission == SeriesAdmission(
        admitted=(WDI, EUROSTAT_GDP), withheld=(_NOT_ADMITTED, UN_WHO)
    )


def test_admit_series_gives_dfo_precedence_at_a_place_it_covers() -> None:
    admission = _snapshot(un_system_enabled=True).admit_series([WDI, CRS, UN_WHO, _NOT_ADMITTED])
    assert admission == SeriesAdmission(admitted=(CRS,), withheld=(WDI, UN_WHO, _NOT_ADMITTED))


def test_admit_series_with_no_facets_admits_nothing() -> None:
    assert _snapshot().admit_series([]) == SeriesAdmission(admitted=(), withheld=())


@pytest.mark.parametrize(
    ("dcid", "off", "on"),
    [
        (CRS_VARIABLE, DFO, DFO),
        # In both the DFO variable set and the base table: DFO takes it.
        (POPULATION, DFO, DFO),
        (GDP, BASE, BASE),
        ("dc/topic/Demographics", BASE, BASE),
        # Known to hold data only under provenances the catalogue does not admit.
        (CENSUS_POVERTY, None, None),
        # Not listed anywhere: a base candidate, whose reads still admit facet by facet.
        ("Count_Person_Female", BASE, BASE),
        ("dc/topic/Unlisted", BASE, BASE),
        ("undata/who/MORT_U5_RATE", None, UN_SYSTEM),
        ("undata/topic/who/GHED_SH.003", None, UN_SYSTEM),
        ("undata/topic/sdgf/SDG_3", None, UN_SYSTEM),
        ("undata/topic/theme/HEALTH", None, UN_SYSTEM),
        ("undata/topic/abas/X", None, UN_SYSTEM),
        ("undata/dpo/X", None, None),
        ("undata/topic/dpo/X", None, None),
        ("undata/p/WHO", None, None),
        ("undata/who", None, None),
        ("undata/who/", None, None),
        ("undata/topic/who", None, None),
        ("undata/topic/sdgf/", None, None),
    ],
)
def test_admit_candidate(dcid: str, off: object, on: object) -> None:
    assert _snapshot().admit_candidate(dcid) == off
    assert _snapshot(un_system_enabled=True).admit_candidate(dcid) == on


def test_topic_admission_follows_its_leaves() -> None:
    tables = replace(
        sample_tables(),
        # CRS_VARIABLE is also withheld, as a DFO variable is when a provenance the catalogue
        # does not admit holds it at a scan place and no admitted base provenance does.
        withheld_variables=frozenset({"sdg/withheld", CRS_VARIABLE}),
        topics={
            "dc/topic/DevelopmentFinance": ("sdg/withheld", CRS_VARIABLE),
            "dc/topic/Economy": (CRS_VARIABLE, GDP),
            "dc/topic/Unlisted": ("sdg/withheld", "Count_Unlisted"),
            "dc/topic/Sdg": ("sdg/withheld",),
        },
    )
    snapshot = replace(_snapshot(), tables=tables)

    # A DFO leaf alone admits the topic as DFO's; any base leaf, listed or
    # unlisted, makes it base; a topic whose leaves are all withheld is refused.
    assert snapshot.admit_candidate(CRS_VARIABLE) == DFO
    assert snapshot.admit_candidate("dc/topic/DevelopmentFinance") == DFO
    assert snapshot.admit_candidate("dc/topic/Economy") == BASE
    assert snapshot.admit_candidate("dc/topic/Unlisted") == BASE
    assert snapshot.admit_candidate("dc/topic/Sdg") is None
    assert tables.admitted_topics == {
        "dc/topic/DevelopmentFinance": DFO,
        "dc/topic/Economy": BASE,
        "dc/topic/Unlisted": BASE,
    }


@pytest.mark.parametrize(
    ("variable", "expected"),
    [
        (POPULATION, (DFO, BASE)),
        (CRS_VARIABLE, (DFO,)),
        (GDP, (BASE,)),
        # Not listed anywhere: admitted as base, so it reports base.
        ("Count_Person_Female", (BASE,)),
        (CENSUS_POVERTY, ()),
    ],
)
def test_data_commons_lists_every_origin_holding_the_variable(
    variable: str, expected: tuple
) -> None:
    assert _snapshot().data_commons(variable) == expected


@pytest.mark.parametrize(
    "variable", [POPULATION, CRS_VARIABLE, GDP, CENSUS_POVERTY, "Count_Person_Female"]
)
def test_data_commons_names_the_origin_that_admits_a_variable(variable: str) -> None:
    snapshot = _snapshot(un_system_enabled=True)
    admitted = snapshot.admit_candidate(variable)
    assert (admitted is None) == (snapshot.data_commons(variable) == ())
    assert admitted is None or admitted in snapshot.data_commons(variable)


def test_data_commons_for_a_un_variable_follows_the_setting() -> None:
    assert _snapshot().data_commons("undata/who/MORT_U5_RATE") == ()
    assert _snapshot(un_system_enabled=True).data_commons("undata/who/MORT_U5_RATE") == (UN_SYSTEM,)
    assert _snapshot(un_system_enabled=True).data_commons("undata/dpo/X") == ()


def test_base_places_decodes_the_bitset_in_scan_order() -> None:
    snapshot = _snapshot()
    assert snapshot.base_places(POPULATION) == ("Earth", "country/KEN", "country/USA")
    assert snapshot.base_places(GDP) == ("country/KEN", "country/USA", "nuts/DE11")
    assert snapshot.base_places(CRS_VARIABLE) is None


def test_is_stale_after_fourteen_days() -> None:
    snapshot = _snapshot()
    assert snapshot.build_id == "b"
    assert snapshot.built_at == BUILT_AT
    assert not snapshot.is_stale(now=BUILT_AT + timedelta(days=14))
    assert snapshot.is_stale(now=BUILT_AT + timedelta(days=14, seconds=1))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"base_variables": {GDP: BaseVariable(provenances=(CRS,), places=1)}},
            "must be a base provenance row",
        ),
        (
            {"base_variables": {GDP: BaseVariable(provenances=(), places=1)}},
            "at least one",
        ),
        (
            {"base_variables": {GDP: BaseVariable(provenances=(WDI,), places=1 << 5)}},
            "past the 5 scan places",
        ),
        ({"provenances": {WDI: provenance_row(WDI, BASE, segment="wb")}}, "segment"),
        ({"provenances": {UN_WHO: provenance_row(UN_WHO, UN_SYSTEM)}}, "segment"),
        ({"provenances": {"x": provenance_row(WDI, BASE)}}, "is keyed as"),
        ({"dataset_facts": {}}, "dataset facts miss 3"),
        (
            {"dataset_facts": {**sample_tables().dataset_facts, UN_WHO: CardFacts()}},
            "hold 1 unknown",
        ),
        ({"topics": {"undata/topic/who/X": ("undata/who/X",)}}, "at least one leaf"),
        ({"topics": {"dc/topic/Empty": ()}}, "at least one leaf"),
        (
            {
                "base_variables": {GDP: BaseVariable(provenances=(WDI,), places=1)},
                "provenance_places": {WDI: 1},
                "withheld_variables": frozenset({GDP}),
            },
            "both admitted and withheld",
        ),
        (
            {
                "base_variables": {GDP: BaseVariable(provenances=(WDI, EUROSTAT_GDP), places=0b11)},
                "provenance_places": {WDI: 0b01},
            },
            "has places none of its provenances",
        ),
        (
            {
                "base_variables": {GDP: BaseVariable(provenances=(WDI,), places=0b01)},
                "provenance_places": {WDI: 0b11},
            },
            "places no base variable naming it has",
        ),
        (
            {
                "base_variables": {GDP: BaseVariable(provenances=(WDI,), places=0b01)},
                "provenance_places": {WDI: 0b01, EUROSTAT_GDP: 0b01},
            },
            "places no base variable naming it has",
        ),
    ],
)
def test_tables_refuse_broken_invariants(change: dict, message: str) -> None:
    fields: dict[str, Any] = {
        "provenances": sample_tables().provenances,
        "dfo_variables": frozenset(),
        "base_variables": {},
        "provenance_places": {},
        "withheld_variables": frozenset(),
        "topics": {},
        "scan_places": sample_tables().scan_places,
        "scan_levels": {},
        "dataset_facts": sample_tables().dataset_facts,
    }
    fields.update(change)
    with pytest.raises(CatalogueTableError, match=message):
        CatalogueTables(**fields)


def test_tables_derive_the_dfo_provenances_and_un_segments() -> None:
    tables = sample_tables()
    assert tables.dfo_provenances == frozenset({CRS})
    assert tables.un_segments == frozenset({"who"})


def test_variable_provenances_names_base_rows_and_the_un_agency() -> None:
    tables = sample_tables()
    on = _snapshot(un_system_enabled=True)

    assert on.variable_provenances(GDP) == (
        tables.provenances[WDI],
        tables.provenances[EUROSTAT_GDP],
    )
    assert on.variable_provenances("undata/who/MORT_U5_RATE") == (tables.provenances[UN_WHO],)
    assert _snapshot().variable_provenances("undata/who/MORT_U5_RATE") == ()
    assert on.variable_provenances("undata/topic/who/X") == ()
    assert on.variable_provenances(CRS_VARIABLE) == ()
    assert on.variable_provenances("Unlisted") == ()


def test_topic_admission_caps_admitted_leaves_as_search_does() -> None:
    unadmitted = tuple(f"sdg/unadmitted_{i}" for i in range(TOPIC_EXPAND_MAX_MEMBERS + 10))
    too_many_admitted = tuple(f"Count_{i}" for i in range(TOPIC_EXPAND_MAX_MEMBERS + 1))
    tables = replace(
        sample_tables(),
        dfo_variables=frozenset({CRS_VARIABLE, *too_many_admitted}),
        withheld_variables=frozenset(unadmitted),
        topics={
            # Over the cap in raw leaves, one admitted: admitted, as in search.
            "dc/topic/OneAmongMany": (*unadmitted, GDP),
            # Admitted leaves exactly at the cap.
            "dc/topic/AtCap": too_many_admitted[:TOPIC_EXPAND_MAX_MEMBERS],
            # One admitted leaf past the cap: search drops the root.
            "dc/topic/OverCap": too_many_admitted,
        },
    )

    assert tables.admitted_topics == {"dc/topic/OneAmongMany": BASE, "dc/topic/AtCap": DFO}
