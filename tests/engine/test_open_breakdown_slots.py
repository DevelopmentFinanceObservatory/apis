"""Pins the breakdown reader (`phase_rank_bind._open_breakdown_slots`): a phrase asking for
a slot's values side by side opens that slot as a ``requested_all`` claim, ahead of the
partition binding that would otherwise bind it from recall."""

from __future__ import annotations

import asyncio

import pytest

from search.engine.bind import SlotBindingDraft
from search.engine.binders import Binder, GeminiBinder
from search.engine.quantity_read import QuantityOutput
from search.engine.region.phase_rank_bind import _open_breakdown_slots
from search.engine.shape import ShapeDraft, SlotKeyDraft
from tests.engine._binders import RatingBinder, elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_WEALTH = "undataWealthQuantile"
_SEX = "undataSex"
_AGE = "undataAge"
_REGION = "undataRegion"
_QUINTILES = {
    "undata/WEALTH-Q1": "Lowest quintile",
    "undata/WEALTH-Q2": "Second quintile",
    "undata/WEALTH-Q3": "Middle quintile",
    "undata/WEALTH-Q4": "Fourth quintile",
    "undata/WEALTH-Q5": "Highest quintile",
}

# A recall-sourced shape whose graph labels carry each value's property name, as the
# graph adds it to tell code lists apart.
_SHAPE = ShapeDraft(
    shape_id="undata/sdg/series-mortality",
    label="Under-five mortality",
    pop_type_dcid="Person",
    meas_prop_dcid="mortalityRate",
    stat_type_dcid="measuredValue",
    meas_qual_dcid=None,
    meas_denom_dcid=None,
    slot_keys=(
        SlotKeyDraft(axis="how", property_dcid=_WEALTH, label="Wealth quantile"),
        SlotKeyDraft(axis="how", property_dcid=_SEX, label="Sex"),
        SlotKeyDraft(axis="how", property_dcid=_AGE, label="Age"),
        SlotKeyDraft(axis="how", property_dcid=_REGION, label="Region"),
        SlotKeyDraft(axis="where", property_dcid=None, label="place"),
        SlotKeyDraft(axis="when", property_dcid=None, label="time window"),
    ),
    slot_taxonomy={
        f"how:{_WEALTH}": list(_QUINTILES),
        f"how:{_SEX}": ["undata/SEX-F", "undata/SEX-M"],
        f"how:{_AGE}": ["undata/AGE-Y0T4"],
        f"how:{_REGION}": ["undata/REG-SSA"],
    },
    value_labels={
        _WEALTH: "Wealth quantile",
        _SEX: "Sex",
        _AGE: "Age",
        _REGION: "Region",
        **{dcid: f"{label} (Wealth quantile)" for dcid, label in _QUINTILES.items()},
        "undata/SEX-F": "Female",
        "undata/SEX-M": "Male",
        "undata/AGE-Y0T4": "Under 5 years old",
        "undata/REG-SSA": "Sub-Saharan Africa",
    },
)
_OPEN = [
    SlotBindingDraft(axis="how", property_dcid=prop, kind="unbound")
    for prop in (_WEALTH, _SEX, _AGE, _REGION)
]


def _opened(bindings: list[SlotBindingDraft]) -> dict[str, str | None]:
    """Each slot `bindings` open as ``requested_all``, with its phrase."""
    return {
        b.property_dcid: b.requested_all_phrase
        for b in bindings
        if b.evidence == "requested_all" and b.property_dcid is not None
    }


@pytest.mark.parametrize(
    ("variable", "dropped", "opened"),
    [
        pytest.param(
            "under-five mortality",
            ["by wealth quintile"],
            {_WEALTH: "by wealth quintile"},
            id="a_dropped_phrase_by_wealth_quintile_opens_the_wealth_quantile",
        ),
        pytest.param(
            "under-five mortality by wealth quintile in Kenya",
            [],
            {_WEALTH: "by wealth quintile"},
            id="the_entity_ends_the_span_in_the_variable",
        ),
        pytest.param(
            "under-five mortality across sexes",
            [],
            {_SEX: "across sexes"},
            id="across_opens_a_plural_slot_name",
        ),
        pytest.param(
            "under-five mortality for each sex",
            [],
            {_SEX: "for each sex"},
            id="for_each_opens",
        ),
        pytest.param(
            "under-five mortality by age group",
            [],
            {_AGE: "by age"},
            id="a_filler_noun_is_dropped",
        ),
        pytest.param(
            "under-five mortality by sex in 2015",
            [],
            {_SEX: "by sex"},
            id="a_date_ends_the_span",
        ),
        pytest.param(
            "under-five mortality by wealth", [], {_WEALTH: "by wealth"}, id="the_name_alone"
        ),
        pytest.param(
            "under-five mortality by quintile",
            [],
            {},
            id="value_label_words_alone_name_no_slot",
        ),
        pytest.param(
            "under-five mortality by wealth decile",
            [],
            {},
            id="a_word_neither_name_nor_value_label_holds_opens_nothing",
        ),
        pytest.param(
            "under-five mortality across Africa",
            [],
            {},
            id="a_value_label_word_without_the_slot_name_opens_nothing",
        ),
        pytest.param("deaths per sex", [], {}, id="per_is_no_breakdown"),
        pytest.param("GDP per capita", [], {}, id="per_capita_opens_nothing"),
        pytest.param("under-five mortality by Kenya", [], {}, id="an_entity_span_opens_nothing"),
        pytest.param("under-five mortality by 2015", [], {}, id="a_date_span_opens_nothing"),
        pytest.param(
            "under-five mortality by sex and age",
            [],
            {},
            id="a_span_naming_two_slots_opens_neither",
        ),
        pytest.param(
            "under-five mortality",
            ["by sex", "by wealth quintile"],
            {_SEX: "by sex", _WEALTH: "by wealth quintile"},
            id="two_spans_open_two_slots",
        ),
    ],
)
def test_a_breakdown_phrase_opens_the_slot_it_names(variable, dropped, opened):
    bindings = _open_breakdown_slots(list(_OPEN), variable, dropped, _SHAPE, entities=["Kenya"])

    assert _opened(bindings) == opened
    assert {b.property_dcid for b in bindings} == {b.property_dcid for b in _OPEN}


def test_a_slot_the_bindings_hold_a_value_on_stays_bound():
    female = SlotBindingDraft(
        axis="how", property_dcid=_SEX, kind="value", value_dcids=["undata/SEX-F"]
    )

    bindings = _open_breakdown_slots([female], "deaths by sex", [], _SHAPE, entities=())

    assert bindings == [female]


def test_a_slot_the_bindings_leave_out_is_added_open():
    assert _open_breakdown_slots([], "deaths by sex", [], _SHAPE, entities=()) == [
        SlotBindingDraft(
            axis="how",
            property_dcid=_SEX,
            kind="unbound",
            evidence="requested_all",
            requested_all_phrase="by sex",
        )
    ]


def test_an_index_sourced_slot_reads_its_declared_name():
    """ "by sector" names the CRS purpose slot by its declared plain name."""
    shape = ShapeDraft(
        shape_id="financialflow_amount_measuredvalue_nominal",
        label="Financial flows",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=(SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="CRS purpose"),),
        slot_taxonomy={"how:crsPurpose": ["CRSPurposeCode12220"]},
        slot_taxonomy_index_sourced=True,
    )

    bindings = _open_breakdown_slots([], "ODA disbursements by sector", [], shape, entities=())

    assert _opened(bindings) == {"crsPurpose": "by sector"}


# A held family of under-five mortality with no total: one member per wealth quintile.
_PROVENANCE = "dc/base/UNSDG"


def _member(quintile: str) -> str:
    return f"Mortality_Person_Upto4Years_{quintile.rsplit('-', 1)[-1]}"


def _arcs(quintile: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": "Under-five mortality rate"}]},
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "mortalityRate"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": _WEALTH}]},
        _WEALTH: {"nodes": [{"dcid": quintile, "name": _QUINTILES[quintile]}]},
    }


def _resolve_mortality(
    bind_catalogue,
    variable: str,
    *,
    dropped: list[str],
    binder: Binder | None = None,
    entity_words: dict[str, str] | None = None,
):
    """Resolve `variable` in Kenya over the held family of one under-five mortality member
    per wealth quintile, recall scoring only the highest quintile and the binder leaving the
    wealth slot open, with the bind's `dropped` phrases and the request's `entity_words`."""
    from origin_catalogue import BASE
    from search.engine.region import resolve_variable
    from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue
    from tests.origin_catalogue.support import build_arc_tables

    arcs = {_member(q): _arcs(q) for q in _QUINTILES}
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row(_PROVENANCE, BASE)],
            base_variables={dcid: ([_PROVENANCE], []) for dcid in arcs},
            arcs=build_arc_tables(dfo={}, base=arcs, schema={}),
        )
    )
    graph = FakeGraph(
        nodes={
            **{
                sv: {"label": sv, "type": "StatisticalVariable", "arcs": a}
                for sv, a in arcs.items()
            },
            **{dcid: {"label": label, "type": "Thing"} for dcid, label in _QUINTILES.items()},
            "country/KEN": {"label": "Kenya", "type": "Country"},
        },
        obs={
            f"{sv}|country/KEN": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": _PROVENANCE,
                }
            ]
            for sv in arcs
        },
        detect={variable: {"svs": [_member("undata/WEALTH-Q5")]}},
        resolve={"Kenya": "country/KEN"},
    )
    open_wealth = SlotBindingDraft(axis="how", property_dcid=_WEALTH, kind="unbound")

    class _BindLLM:
        async def generate_structured(self, *, prompt, system, schema):
            if schema is QuantityOutput:
                return NO_QUANTITY
            return elected_reply(0, bindings=[open_wealth], also_plausible=None), None

    return asyncio.run(
        resolve_variable(
            variable,
            entities=["Kenya"],
            date_request=None,
            detect_query=variable,
            role_query=f"{variable} Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLM(),
            binder=binder,
            dropped_qualifier_phrases=dropped,
            entity_words=entity_words,
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


def test_the_reader_opens_the_slot_before_recall_binds_it(bind_catalogue):
    """ "under-five mortality by wealth quintile in Kenya", where recall scores only the
    highest quintile and the binder leaves the slot open. Without the reader the partition
    binding (`bind_partition_values`) binds the slot to that one quintile, since every
    member states one and the scored member is unanimous. The reader runs first, so the
    slot stays open and the part serves every quintile."""
    region = _resolve_mortality(
        bind_catalogue, "under-five mortality", dropped=["by wealth quintile"]
    )

    served = [sv.ref.ref for spec in region.specs for sv in spec.resolved.members]
    assert sorted(served) == sorted(_member(q) for q in _QUINTILES)


@pytest.mark.parametrize(
    ("variable", "entity_words", "asked"),
    [
        pytest.param("under-five mortality by wealth quintile", None, [], id="the_phrase_is_read"),
        pytest.param(
            "rural under-five mortality by wealth quintile",
            None,
            [("rural",)],
            id="a_word_outside_the_phrase_is_rated",
        ),
        pytest.param(
            "Kenyan under-five mortality by wealth quintile",
            {"Kenya": "Kenyan"},
            [],
            id="the_entity_s_own_words_are_read",
        ),
    ],
)
def test_the_words_of_an_opened_slot_phrase_are_read(bind_catalogue, variable, entity_words, asked):
    """The answer serves every value of the slot a breakdown phrase opens, so the coverage
    rule reads the phrase's words and the binder rates none of them: "quintile", which no
    member's name holds, stays off the stated-words read. A word outside the phrase that
    no reader reads, "rural", is still rated. The request's own words for an entity
    ("Kenyan" for Kenya) are the entity's, so the binder rates none of them either."""
    binder = RatingBinder(GeminiBinder())

    region = _resolve_mortality(
        bind_catalogue, variable, dropped=[], binder=binder, entity_words=entity_words
    )

    assert region.status == "definite"
    assert [request.words for request in binder.requests] == asked
