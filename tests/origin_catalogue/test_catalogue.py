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
    UnTables,
)
from origin_catalogue.catalogue import BaseVariable, CardFacts
from origin_catalogue.families import FamilyKey, FamilySummary
from origin_catalogue.topic_walk import TOPIC_EXPAND_MAX_MEMBERS
from tests.origin_catalogue.fake_catalogue import (
    FakeCatalogue,
    provenance_row,
    restrictive_catalogue,
    un_mirror,
    un_variable,
)
from tests.origin_catalogue.support import (
    BUILT_AT,
    CENSUS_POVERTY,
    CRS,
    CRS_VARIABLE,
    EUROSTAT_GDP,
    GDP,
    POPULATION,
    UN_EXCLUDED_SERIES,
    UN_EXCLUDED_VARIABLE,
    UN_SERIES,
    UN_TOPIC,
    UN_VARIABLE,
    UN_VARIABLE_FEMALE,
    UN_WHO,
    WDI,
    build_arc_tables,
    sample_tables,
    sample_un_mirror,
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
        manifest=manifest,
        tables=sample_tables(),
        un=UnTables(mirror=sample_un_mirror()),
        un_system_enabled=un_system_enabled,
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


def test_admit_series_keeps_every_origin_at_a_place_dfo_covers() -> None:
    """The facet election, not admission, chooses between DFO and base (#1271)."""
    admission = _snapshot(un_system_enabled=True).admit_series([WDI, CRS, UN_WHO, _NOT_ADMITTED])
    assert admission == SeriesAdmission(admitted=(WDI, CRS, UN_WHO), withheld=(_NOT_ADMITTED,))


def test_admit_series_with_no_facets_admits_nothing() -> None:
    assert _snapshot().admit_series([]) == SeriesAdmission(admitted=(), withheld=())


@pytest.mark.parametrize(
    ("allowed", "admission"),
    [
        (
            {WDI, UN_WHO, _NOT_ADMITTED},
            SeriesAdmission(admitted=(WDI, UN_WHO), withheld=(_NOT_ADMITTED,)),
        ),
        # Out-of-scope provenances are neither admitted nor withheld.
        ({CRS}, SeriesAdmission(admitted=(CRS,), withheld=())),
        # A scope does not lift a registry exclusion.
        ({_NOT_ADMITTED}, SeriesAdmission(admitted=(), withheld=(_NOT_ADMITTED,))),
        (set(), SeriesAdmission(admitted=(), withheld=())),
    ],
)
def test_admit_series_filters_by_scope_before_admission(
    allowed: set[str], admission: SeriesAdmission
) -> None:
    scoped = _snapshot(un_system_enabled=True).scoped(frozenset(allowed))
    assert scoped.admit_series([WDI, CRS, UN_WHO, _NOT_ADMITTED]) == admission


def test_scoped_keeps_the_build_and_narrows_an_existing_scope() -> None:
    snapshot = _snapshot()
    scoped = snapshot.scoped(frozenset({WDI, CRS})).scoped(frozenset({CRS, EUROSTAT_GDP}))
    assert snapshot.scope is None
    assert scoped.scope == frozenset({CRS})
    assert (scoped.build_id, scoped.mixer_domains, scoped.tables) == (
        snapshot.build_id,
        snapshot.mixer_domains,
        snapshot.tables,
    )
    assert scoped.provenance_rows() == snapshot.provenance_rows()


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
        # The UN tables hold the variable, its by-sex variant and a topic reaching both.
        (UN_VARIABLE, None, UN_SYSTEM),
        (UN_VARIABLE_FEMALE, None, UN_SYSTEM),
        (UN_TOPIC, None, UN_SYSTEM),
        # A variable of a series the registry excludes.
        (UN_EXCLUDED_VARIABLE, None, None),
        # A registered agency's variable, a topic and a series the UN tables do not hold.
        ("undata/who/NOT_IN_THE_BUILD", None, None),
        ("undata/topic/who/GHED_SH.003", None, None),
        (UN_SERIES, None, None),
        ("undata/p/WHO", None, None),
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
    admitted = {topic: tables.admit_topic(topic, successors=False) for topic in tables.topics}
    assert admitted == {
        "dc/topic/DevelopmentFinance": DFO,
        "dc/topic/Economy": BASE,
        "dc/topic/Unlisted": BASE,
        "dc/topic/Sdg": None,
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
    assert _snapshot().data_commons(UN_VARIABLE) == ()
    assert _snapshot(un_system_enabled=True).data_commons(UN_VARIABLE) == (UN_SYSTEM,)
    assert _snapshot(un_system_enabled=True).data_commons(UN_EXCLUDED_VARIABLE) == ()
    assert _snapshot(un_system_enabled=True).data_commons("undata/dpo/X") == ()


def test_the_un_tables_answer_a_series_whole_and_nothing_with_the_un_off() -> None:
    on, off = _snapshot(un_system_enabled=True), _snapshot()
    mirror = sample_un_mirror()

    assert on.un_variable(UN_VARIABLE_FEMALE) == mirror.variables[UN_VARIABLE_FEMALE]
    assert on.un_variable(UN_EXCLUDED_VARIABLE) is None
    assert on.un_series_members(UN_SERIES) == (UN_VARIABLE, UN_VARIABLE_FEMALE)
    assert on.un_series_members(UN_EXCLUDED_SERIES) == ()
    assert on.un_topic_leaves(UN_TOPIC) == (UN_VARIABLE, UN_VARIABLE_FEMALE)
    assert on.un_topic_leaves("undata/topic/who/GHED_SH.003") is None
    node = on.un_node("undata/SEX-F")
    assert node is not None
    assert (node.name, node.types) == ("Female", ("undata/SEX",))
    assert on.un_node(UN_VARIABLE) is None
    assert on.superseded_by(UN_EXCLUDED_VARIABLE) == CRS
    assert on.superseded_by(UN_VARIABLE) is None

    assert off.un_variable(UN_VARIABLE) is None
    assert off.un_series_members(UN_SERIES) == ()
    assert off.un_topic_leaves(UN_TOPIC) is None
    assert off.un_node("undata/SEX-F") is None


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
        ({"source_documentation": {}}, "source documentation miss 3"),
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
        ({"superseded": {GDP: "undata/who/X"}}, "is superseded"),
        (
            {
                "withheld_variables": frozenset({GDP}),
                "dfo_variables": frozenset({GDP}),
                "superseded": {GDP: "undata/who/X"},
            },
            "is superseded",
        ),
        (
            {"withheld_variables": frozenset({GDP}), "superseded": {GDP: POPULATION}},
            "is superseded",
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
        "source_documentation": sample_tables().source_documentation,
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

    admitted = {topic: tables.admit_topic(topic, successors=False) for topic in tables.topics}
    assert admitted == {
        "dc/topic/OneAmongMany": BASE,
        "dc/topic/AtCap": DFO,
        "dc/topic/OverCap": None,
    }


_OLD_X, _OLD_Y, _OLD_Z = "old/X", "old/Y", "old/Z"
_UN_X = "undata/who/X"
_WITH_SUCCESSORS = "dc/topic/WithSuccessors"
_WITHOUT_SUCCESSORS = "dc/topic/WithoutSuccessors"
_WITHHELD_ELSEWHERE = "dc/topic/WithheldElsewhere"


def _superseding(*, un_system_enabled: bool) -> CatalogueSnapshot:
    """Two superseded variables with one successor, one with none, and a
    variable withheld under a provenance no agency supersedes."""
    return restrictive_catalogue(
        withheld_variables=[_OLD_X, _OLD_Y, _OLD_Z, CENSUS_POVERTY],
        topics={
            _WITH_SUCCESSORS: [_OLD_X, _OLD_Y, _OLD_Z],
            _WITHOUT_SUCCESSORS: [_OLD_Z],
            _WITHHELD_ELSEWHERE: [CENSUS_POVERTY],
        },
        un_mirror=un_mirror({_UN_X: un_variable("undata/who/SERIES-X", name="x")}),
        successors={_OLD_X: _UN_X, _OLD_Y: _UN_X, _OLD_Z: None},
        un_system_enabled=un_system_enabled,
    )


@pytest.mark.parametrize("un_system_enabled", [True, False])
def test_a_superseded_variable_is_withheld_and_names_its_successor_with_the_un_origin_on(
    un_system_enabled: bool,
) -> None:
    catalogue = _superseding(un_system_enabled=un_system_enabled)

    assert catalogue.successor(_OLD_X) == (_UN_X if un_system_enabled else None)
    assert catalogue.successor(_OLD_Z) is None
    assert catalogue.successor(CENSUS_POVERTY) is None
    assert [catalogue.is_superseded(v) for v in (_OLD_X, _OLD_Z)] == [True, True]
    assert not catalogue.is_superseded(CENSUS_POVERTY)
    assert not catalogue.is_superseded(_UN_X)
    assert catalogue.admit_candidate(_OLD_X) is None
    assert catalogue.admit_candidate(_OLD_Z) is None


def test_a_topic_is_admitted_on_a_leaf_with_a_successor_with_the_un_origin_on() -> None:
    on, off = _superseding(un_system_enabled=True), _superseding(un_system_enabled=False)

    assert on.admit_candidate(_WITH_SUCCESSORS) == UN_SYSTEM
    assert not on.is_superseded(_WITH_SUCCESSORS)
    assert off.admit_candidate(_WITH_SUCCESSORS) is None
    assert off.is_superseded(_WITH_SUCCESSORS)
    for catalogue in (on, off):
        assert catalogue.admit_candidate(_WITHOUT_SUCCESSORS) is None
        assert catalogue.is_superseded(_WITHOUT_SUCCESSORS)
        assert catalogue.admit_candidate(_WITHHELD_ELSEWHERE) is None
        assert not catalogue.is_superseded(_WITHHELD_ELSEWHERE)


def test_each_leaf_with_a_successor_counts_toward_the_member_cap() -> None:
    old = [f"old/V{i}" for i in range(TOPIC_EXPAND_MAX_MEMBERS + 1)]
    catalogue = restrictive_catalogue(
        withheld_variables=old,
        topics={"dc/topic/AtCap": old[:-1], "dc/topic/OverCap": old},
        un_mirror=un_mirror({_UN_X: un_variable("undata/who/SERIES-X", name="x")}),
        successors=dict.fromkeys(old, _UN_X),
    )

    # Every leaf shares one successor, and the bound still counts each leaf.
    assert catalogue.admit_candidate("dc/topic/AtCap") == UN_SYSTEM
    assert catalogue.admit_candidate("dc/topic/OverCap") is None
    # The bound refused OverCap before supersession too, so it holds no recall place.
    assert not catalogue.is_superseded("dc/topic/OverCap")


@pytest.mark.parametrize("un_system_enabled", [True, False])
def test_the_bound_counts_superseded_leaves_as_the_base_leaves_they_were(
    un_system_enabled: bool,
) -> None:
    admitted = [f"Count_V{i}" for i in range(TOPIC_EXPAND_MAX_MEMBERS)]
    catalogue = restrictive_catalogue(
        withheld_variables=[_OLD_Z],
        topics={
            # 50 admitted leaves and one superseded leaf with no successor: over the bound,
            # before supersession and since, so neither admitted nor superseded.
            "dc/topic/Broad": [*admitted, _OLD_Z],
            "dc/topic/Narrow": [admitted[0], _OLD_Z],
        },
        successors={_OLD_Z: None},
        un_system_enabled=un_system_enabled,
    )

    assert catalogue.admit_candidate("dc/topic/Broad") is None
    assert not catalogue.is_superseded("dc/topic/Broad")
    assert catalogue.admit_candidate("dc/topic/Narrow") == BASE
    assert not catalogue.is_superseded("dc/topic/Narrow")


def test_the_fake_catalogue_names_successors_and_superseded_dcids() -> None:
    fake = FakeCatalogue(successors={_OLD_X: _UN_X}, superseded=frozenset({_OLD_Z}))

    assert (fake.successor(_OLD_X), fake.successor(_OLD_Z)) == (_UN_X, None)
    assert [fake.is_superseded(v) for v in (_OLD_X, _OLD_Z, GDP)] == [True, True, False]
    off = replace(fake, un_system_enabled=False)
    assert off.successor(_OLD_X) is None
    assert off.is_superseded(_OLD_X)
    assert [fake.admit_candidate(v) for v in (_OLD_X, _OLD_Z)] == [None, None]
    assert fake.admit_candidate(GDP) == BASE


@pytest.mark.parametrize(("un_system_enabled", "expected"), [(True, UN_SYSTEM), (False, None)])
def test_a_cross_agency_un_topic_the_tables_hold_is_admitted_with_the_un_origin_on(
    un_system_enabled: bool, expected: object
) -> None:
    topic = "undata/topic/sdgf/3.2.1"
    catalogue = restrictive_catalogue(
        un_mirror=un_mirror(
            {_UN_X: un_variable("undata/who/SERIES-X", name="x")}, topics={topic: [_UN_X]}
        ),
        un_system_enabled=un_system_enabled,
    )

    assert catalogue.admit_candidate(topic) == expected


def _family_variable(
    qualifier: dict | None, *, name: str | None = None, constraints: dict[str, str] | None = None
) -> dict:
    """A `->*` answer of a variable of population `Debt`, measured property `amount`,
    with `qualifier` as its `measurementQualifier` target, or none, `name` as its name and
    `constraints` mapping each constraint property to its value."""
    arcs: dict = {
        "populationType": {"nodes": [{"dcid": "Debt", "types": ["Class"]}]},
        "measuredProperty": {"nodes": [{"dcid": "amount", "types": ["Property"]}]},
        "statType": {"nodes": [{"dcid": "measuredValue", "types": ["StatType"]}]},
    }
    if qualifier is not None:
        arcs["measurementQualifier"] = {"nodes": [qualifier]}
    if name is not None:
        arcs["name"] = {"nodes": [{"value": name}]}
    if constraints:
        arcs["constraintProperties"] = {
            "nodes": [{"dcid": prop, "types": ["Property"]} for prop in constraints]
        }
        for prop, value in constraints.items():
            arcs[prop] = {"nodes": [{"dcid": value, "types": ["Thing"]}]}
    return arcs


_DEBT = FamilyKey("Debt", "amount", "measuredValue", None, None, False, False)
_UN_MORTALITY = FamilyKey(UN_SERIES, "value", "", None, None, False, False)


def test_a_snapshot_reads_each_family_summary_the_job_computed_from_its_tables() -> None:
    """A DFO and a base variable whose only difference is a qualifier typed only
    ProvisionalNode, which search reads as none, share a family, summarised once; a real
    qualifier makes another. A node held empty belongs to none. The UN family reads from
    the UN tables, and with the UN origin off it is not held."""
    dfo, base, qualified, empty = "Debt_Dfo", "Debt_Base", "Debt_Real", "Debt_Empty"
    snapshot = restrictive_catalogue(
        provenances=[provenance_row(WDI, BASE)],
        dfo_variables=[dfo],
        base_variables={base: ([WDI], []), qualified: ([WDI], []), empty: ([WDI], [])},
        un_mirror=sample_un_mirror(),
        arcs=build_arc_tables(
            dfo={
                dfo: _family_variable(
                    {"dcid": "dangling", "types": ["ProvisionalNode"]}, name="Debt"
                )
            },
            base={
                base: _family_variable(None, name="Debt owed", constraints={"creditor": "IMF"}),
                qualified: _family_variable({"dcid": "Real", "types": ["Qualifier"]}),
                empty: None,
            },
            schema={},
        ),
    )

    assert snapshot.held_family(_DEBT) == FamilySummary(
        members=(dfo, base),
        values={"creditor": ("IMF",)},
        shared={},
        total=dfo,
        total_name="Debt",
        un_only=False,
    )
    assert snapshot.held_family(_DEBT._replace(measurement_qualifier="Real")) == FamilySummary(
        members=(qualified,),
        values={},
        shared={},
        total=None,
        total_name=None,
        un_only=False,
    )
    assert snapshot.held_family(_DEBT._replace(role_bearing=True)) is None
    assert snapshot.held_family(_UN_MORTALITY) == FamilySummary(
        members=(UN_VARIABLE, UN_VARIABLE_FEMALE),
        values={"undataSex": ("undata/SEX-F",)},
        shared={},
        total=UN_VARIABLE,
        total_name="Under-five mortality rate",
        un_only=True,
    )
    assert replace(snapshot, un_system_enabled=False).held_family(_UN_MORTALITY) is None
