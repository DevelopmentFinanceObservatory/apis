"""Slot menus count the rows each option would serve and drop the options serving none.

The resubmit cases resolve a CRS aid identity from Germany to Guatemala against the packaged
code index, with a coverage index holding rows for a handful of its members. Each count is
the rows a resubmit binding that option would download, one per series and year, so every
assertion below states the fixture rows behind it.
"""

from __future__ import annotations

import functools
from dataclasses import replace

import pytest

from code_index.shared import get_index
from materialise.facets import native_currency_iso
from materialise.rate_tables import RateTableCache
from search.engine.bind import SlotBindingDraft
from search.engine.region import resolve_spec_resubmit
from search.engine.region.option_coverage import (
    _maybe_build_facet_options,
    _stream_slot_options_events,
    count_slot_options,
)
from search.engine.region.state import EngineOptions
from search.engine.region.un_system import NO_LABELS
from search.engine.retrieval.bound_values import OptionServedFact, SeriesRows
from search.engine.same_shape import collapse_same_shape
from search.engine.shape import ShapeDraft, SlotKeyDraft, identity_shape_for
from search.engine.unit.conversion_catalogue import (
    ConversionCatalogue,
    RecipeTables,
    bound_conversion_catalogue,
)
from search.models import (
    BindingUnbound,
    BindingValue,
    GraphRef,
    IdentityBindingValue,
    IdentityFacetBinding,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    RequestedWindow,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotOption,
    SlotOptions,
    SlotValue,
    Spec,
    SpecDefinition,
    TimeWindow,
)
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._axis_doubles import axis_confirming_reader
from tests.engine._base_identity import (
    destination_entity,
    make_base,
    make_identity,
    origin_entity,
    subject_entity,
)
from tests.engine._builders import value_slot as _value_slot
from tests.engine._conjoin_fixtures import _region
from tests.engine._fake_coverage import (
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    MATERIALISE_BASE_URL,
    materialise_double_from_obs,
    walk_options,
    with_member_arcs,
)
from tests.engine.test_conjoin_measure_merge import _recipient_only_shape, _where_bound_spec
from tests.engine.test_resubmit_alternative_measure_round_trip import _arcs_for_row
from tests.fixtures import FakeGraph, _load_json
from tests.fixtures.provenance import index_provenance
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)
from tests.materialise.conftest import FakeGraphReader

_DEU = "country/DEU"
_FRA = "country/FRA"
_GTM = "country/GTM"
_ROLE_NAMES = RoleNames(donor="financingProvider", recipient="financingRecipient")

_MODALITY = "crsCooperationModality"
_PURPOSE = "crsPurpose"
_FINANCE = "crsTypeOfFinance"
_FLOW = "crsTypeOfFlow"
_ODA = "CRSTypeOfFlow10"
_OOF = "CRSTypeOfFlow21"
_SECTOR_BUDGET_SUPPORT = "CRSCooperationModalityA02"


def _member(population: str = "FinancialFlow", **values: str) -> str:
    """The one packaged-index row of the aid five-tuple with exactly `values`."""
    rows = [
        row
        for row in get_index().statvars
        if row.population_type == population
        and row.measured_property == "amount"
        and row.measurement_qualifier == "Nominal"
        and all(row.constraint_properties.get(p) == v for p, v in values.items())
        and set(row.constraint_properties) == set(values)
    ]
    assert len(rows) == 1, (values, [row.dcid for row in rows])
    return rows[0].dcid


def _gross(modality: str, purpose: str, flow: str = _ODA) -> str:
    return _member(
        **{
            _MODALITY: modality,
            _PURPOSE: purpose,
            _FINANCE: "CRSTypeOfFinance110",
            _FLOW: flow,
            "financialFlowType": "Disbursement",
            "nettingBasis": "Gross",
        }
    )


# Gross ODA disbursements from Germany to Guatemala: two project-type interventions, in
# education (11110) and basic health care (12220), and other technical assistance in
# education. No sector budget support.
_PROJECT_EDUCATION = _gross("CRSCooperationModalityC01", "CRSPurposeCode11110")
_PROJECT_HEALTH = _gross("CRSCooperationModalityC01", "CRSPurposeCode12220")
_TA_EDUCATION = _gross("CRSCooperationModalityD02", "CRSPurposeCode11110")
# A non-export credit official flow in education, served only with the flow left open.
_OOF_EDUCATION = _gross("CRSCooperationModalityC01", "CRSPurposeCode11110", flow=_OOF)
# Sector budget support in education from France only, so no answer from Germany serves it.
_FRENCH_BUDGET_SUPPORT = _gross(_SECTOR_BUDGET_SUPPORT, "CRSPurposeCode11110")
# The commitments sibling of the first member, the one commitment the answer's alternative
# would serve.
_COMMITMENT = _member(
    "FinancialObligation",
    **{
        _MODALITY: "CRSCooperationModalityC01",
        _PURPOSE: "CRSPurposeCode11110",
        _FINANCE: "CRSTypeOfFinance110",
        _FLOW: _ODA,
        "financialObligationType": "Commitment",
    },
)
# The dataset the code index records the aid members under, every facet's provenance here.
_CRS_PROVENANCE = index_provenance(_PROJECT_EDUCATION)

_ROWS = [
    (_PROJECT_EDUCATION, "crs", _DEU, _GTM, [2019, 2020]),
    (_PROJECT_HEALTH, "crs", _DEU, _GTM, [2020]),
    (_TA_EDUCATION, "crs", _DEU, _GTM, [2021]),
    (_OOF_EDUCATION, "crs", _DEU, _GTM, [2021]),
    (_FRENCH_BUDGET_SUPPORT, "crs", _FRA, _GTM, [2021]),
    (_COMMITMENT, "crs", _DEU, _GTM, [2020]),
]


# The DAC recipe's euro rates for every year the fixtures hold, so a request in euros at
# current prices converts every row. Germany's deflator lacks 2019, so a constant-price
# request leaves out Germany's 2019 rows.
_RATES = {
    "CurrencyUSD": dict.fromkeys(range(2019, 2022), 1.0),
    "CurrencyEUR": dict.fromkeys(range(2019, 2022), 0.9),
}
_DEFLATORS = {
    _DEU: dict.fromkeys((2020, 2021), 100.0),
    _FRA: dict.fromkeys(range(2019, 2022), 100.0),
}
_RATE_BODIES = recipe_table_bodies("ConversionRecipeDAC", rates=_RATES, deflators=_DEFLATORS)
# The conversion catalogue the app loads from those tables.
_CATALOGUE = ConversionCatalogue(
    recipes={_CRS_PROVENANCE: "ConversionRecipeDAC"},
    tables={
        "ConversionRecipeDAC": RecipeTables(
            rate_years={
                native_currency_iso(dcid) or dcid: frozenset(years)
                for dcid, years in _RATES.items()
            },
            deflator_years={dcid: frozenset(years) for dcid, years in _DEFLATORS.items()},
            base_year_min=2019,
            base_year_max=2021,
        )
    },
    national_currencies={},
    default_base_year=2021,
    entity_names={_DEU: "Germany", _FRA: "France"},
)


def _slot(prop: str, value: str | None) -> Slot:
    key = SlotKey(axis="how", property=GraphRef(dcid=prop, label=prop), label=prop)
    if value is None:
        return Slot(key=key, binding=BindingUnbound())
    return Slot(
        key=key,
        binding=BindingValue(
            value=SlotValue(ref=GraphRef(dcid=value, label=value), value_kind="enum_value")
        ),
    )


def _resubmit(monkeypatch: pytest.MonkeyPatch, **bound: str) -> Spec:
    """Resubmit gross ODA disbursements from Germany to Guatemala with `bound` slots."""
    region = _crs_region(monkeypatch, bound=bound)
    assert region.status == "definite", region.no_data_reason
    return region.specs[0]


def _collapse(regions, coverage: CoverageIndex):
    """`collapse_same_shape` judging the merged facet options over `coverage`."""
    return collapse_same_shape(
        regions,
        facet_options_for=functools.partial(
            _maybe_build_facet_options, graph=FakeGraph(), coverage=coverage, deadline=None
        ),
    )


def _crs_region(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bound: dict[str, str],
    origins: tuple[str, ...] = (_DEU,),
    rows: list = _ROWS,
    coverage: CoverageIndex | None = None,
    window: RequestedWindow | None = None,
    facets: dict[str, IdentityFacetBinding] | None = None,
    facet_metadata: dict[str, FacetMetadata] | None = None,
    definition: SpecDefinition | None = None,
    catalogue: ConversionCatalogue = _CATALOGUE,
):
    """Resubmit gross ODA disbursements from `origins` to Guatemala with `bound` slots,
    `window` and `facets`, against coverage `rows` and `facet_metadata`, or against
    `coverage` when given, which a request's answers share. `definition` posts that Spec's
    identity in place of the edit, as a client posts a measure alternative. A unit request
    reads `catalogue`."""
    shape = identity_shape_for(
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
    )
    entities = [*(origin_entity(o) for o in origins), destination_entity(_GTM)]

    def identity_for(slot_values: dict[str, str]):
        values = {
            _FLOW: _ODA,
            "financialFlowType": "Disbursement",
            "nettingBasis": "Gross",
            **slot_values,
        }
        slots = [
            _slot(prop, values.get(prop))
            for prop in (_MODALITY, _PURPOSE, _FINANCE, _FLOW, "financialFlowType", "nettingBasis")
        ]
        return make_base(shape, slots, entities=entities, requested_window=window)

    _edited_id, edited = identity_for(bound)
    if facets:
        # A client posts its pins on the identity's facets, beside an unpinned base.
        edited = edited.model_copy(update={"facets": facets})
    members = list(dict.fromkeys(row[0] for row in rows))
    index = get_index()
    nodes = {
        dcid: {
            "label": dcid,
            "type": "StatisticalVariable",
            "arcs": _arcs_for_row(next(r for r in index.statvars if r.dcid == dcid)),
        }
        for dcid in members
    }
    nodes |= {o: {"label": o, "type": "Country"} for o in origins}
    nodes |= {_GTM: {"label": "Guatemala", "type": "Country"}}
    # The CRS provenance converts by the DAC recipe into US dollars or euros.
    conversion = conversion_arcs(
        recipe_id="ConversionRecipeDAC",
        provenance_id=_CRS_PROVENANCE,
        currencies=["CurrencyUSD", "CurrencyEUR"],
    )
    nodes |= {
        dcid: {
            "label": dcid,
            "arcs": {
                prop: {
                    "nodes": [
                        {"value": v} if prop == "lastDataRefreshDate" else {"dcid": v}
                        for v in values
                    ]
                }
                for prop, values in arcs.items()
            },
        }
        for dcid, arcs in conversion.items()
    }
    graph = FakeGraph(nodes=nodes, obs={})
    if coverage is None:
        coverage = patch_coverage(
            monkeypatch,
            rows,
            role_names=dict.fromkeys(members, _ROLE_NAMES),
            facet_metadata=facet_metadata,
        )
    converts = isinstance((facets or {}).get("unit"), IdentityFacetValueRequest)
    # The conversion catalogue a unit request's offers and counts read.
    with bound_conversion_catalogue(catalogue if converts else None):
        return resolve_spec_resubmit(
            inp=ResubmitIdentityPayload(
                identity=definition.identity if definition is not None else edited
            ),
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=FakeCoverageService(coverage).snapshot(),
            options=walk_options(
                graph,
                EngineOptions(
                    materialise_graph=with_member_arcs(
                        axis_confirming_reader(
                            *members,
                            graph=graph,
                            inner=FakeGraphReader(
                                arcs={
                                    **dict.fromkeys(members, CONVERTIBLE_STAT_VAR_ARCS),
                                    **conversion,
                                }
                            ),
                        ),
                        graph,
                    ),
                    # A unit request decides its unit before grounding, against the DAC recipe's
                    # exchange rates.
                    materialise_sdmx=RateTableServer(_RATE_BODIES).client(
                        base_url=MATERIALISE_BASE_URL
                    )
                    if converts
                    else None,
                    rate_cache=RateTableCache() if converts else None,
                ),
            ),
        )


def _slot_menu(spec: Spec, prop: str) -> SlotOptions:
    return next(
        so for so in spec.resolved.slot_options if so.key.property and so.key.property.dcid == prop
    )


def _menu(spec: Spec, prop: str) -> dict[str, int | None]:
    menu = _slot_menu(spec, prop)
    return {o.value.ref.dcid: o.served_rows for o in menu.options if o.value.ref is not None}


def _alternatives(spec: Spec) -> dict[str, int | None]:
    return {alt.label: alt.served_rows for alt in spec.resolved.alternative_measures or []}


def _served(**rows: int) -> dict[tuple[str, str], SeriesRows]:
    """Series of StatVars named by `rows`, each holding that many rows, exactly."""
    return {(sv, f"{_GTM}|{_DEU}"): SeriesRows(rows=n, exact=True) for sv, n in rows.items()}


def test_an_unbound_role_bearing_menu_lists_only_the_options_the_answer_serves(monkeypatch):
    """Germany to Guatemala serves four rows: project-type education in 2019 and 2020,
    project-type health care in 2020, and technical assistance in 2021. Sector budget
    support has only a French series, so it is absent, and the root counts sum to the
    answer's download rows."""
    spec = _resubmit(monkeypatch)

    modality = _menu(spec, _MODALITY)
    assert modality == {
        "CRSCooperationModalityC": 3,
        "CRSCooperationModalityC01": 3,
        "CRSCooperationModalityD": 1,
        "CRSCooperationModalityD02": 1,
    }
    assert _SECTOR_BUDGET_SUPPORT not in modality
    menu = _slot_menu(spec, _MODALITY)
    assert all(o.served_rows_exact for o in menu.options)
    roots = sum(o.served_rows or 0 for o in menu.options if o.relation == "root")
    assert spec.resolved.download is not None
    assert roots == spec.resolved.download.rows == 4


def test_an_unbound_hierarchical_menu_counts_each_level_through_its_descendants(monkeypatch):
    """The two-level purpose menu lists roots and their children. Social infrastructure
    (100) holds all four rows, education (110) the three in policy and administration
    (11110, two levels down), health (120) the one in basic health care."""
    spec = _resubmit(monkeypatch)

    assert _menu(spec, _PURPOSE) == {
        "CRSPurposeCode100": 4,
        "CRSPurposeCode110": 3,
        "CRSPurposeCode120": 1,
    }


def test_a_bound_slot_counts_its_options_with_itself_left_open(monkeypatch):
    """The flow menu counts each flow with every other binding kept. ODA serves the four
    rows the answer reads, the non-export credit flow the one OOF row. A bound purpose
    counts its parent over every purpose, so with education bound, social infrastructure
    still counts the health row."""
    spec = _resubmit(monkeypatch)
    assert _menu(spec, _FLOW) == {_ODA: 4, _OOF: 1}

    education = _resubmit(monkeypatch, crsPurpose="CRSPurposeCode110")
    assert _menu(education, _PURPOSE) == {"CRSPurposeCode111": 3, "CRSPurposeCode100": 4}


def test_measure_alternatives_count_the_rows_their_resubmit_would_serve(monkeypatch):
    """Commitments serve the one commitment row. Net disbursements serve none and are not
    offered."""
    spec = _resubmit(monkeypatch)

    assert _alternatives(spec) == {"commitments": 1}
    assert [alt.served_rows_exact for alt in spec.resolved.alternative_measures or []] == [True]


# ---------------------------------------------------------------------------
# count_slot_options: one builder for every producer
# ---------------------------------------------------------------------------


def _options(prop: str, *dcids: str, relation=None) -> SlotOptions:
    return SlotOptions(
        key=SlotKey(axis="how", property=GraphRef(dcid=prop, label=prop), label=prop),
        options=[
            SlotOption(
                value=SlotValue(ref=GraphRef(dcid=d, label=d), value_kind="enum_value"),
                relation=relation,
            )
            for d in dcids
        ],
        truncated=False,
        source="observed",
    )


def test_a_bound_slots_current_value_stays_when_it_counts_none():
    """The current value is the reader's own binding, so it stays on its menu even when the
    leave-one-out scan finds no series for it."""
    fact = OptionServedFact(property=_FLOW, served={_OOF: _served(sv1=1)})
    [menu] = count_slot_options([_options(_FLOW, _ODA, _OOF, "CRSTypeOfFlow30")], [fact], [])
    assert {o.value.ref.dcid: o.served_rows for o in menu.options} == {_OOF: 1}

    [menu] = count_slot_options(
        [_options(_FLOW, _ODA, _OOF, "CRSTypeOfFlow30")], [fact], [_slot(_FLOW, _ODA)]
    )
    assert {o.value.ref.dcid: o.served_rows for o in menu.options} == {_ODA: 0, _OOF: 1}


# ---------------------------------------------------------------------------
# The trailing slot_options events start from the counted menus
# ---------------------------------------------------------------------------


def _gender_options(*genders: str) -> SlotOptions:
    return _options("gender", *genders)


def _stream(monkeypatch, slot_options, bindings, probed: list) -> dict[str, list]:
    """Stream option events over `slot_options`. The probe counts every option it is asked
    about at its `_PROBED` series and records the menus it was handed in `probed`."""

    def probe(*, slot_options, pick_members, **_kwargs):
        probed.extend(so.key.property.dcid for so in slot_options)
        return [
            so.model_copy(
                update={
                    "options": [
                        o.model_copy(
                            update={
                                "served_rows": _PROBED[o.value.ref.dcid]
                                if pick_members[so.key.property.dcid][o.value.ref.dcid] is not None
                                else None
                            }
                        )
                        for o in so.options
                    ]
                }
            )
            for so in slot_options
        ]

    monkeypatch.setattr("search.engine.region.option_coverage.standard_option_coverage", probe)
    shape = ShapeDraft(
        shape_id="s",
        label="s",
        pop_type_dcid="Person",
        meas_prop_dcid="unemploymentRate",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(SlotKeyDraft(axis="how", property_dcid="gender", label="gender"),),
        sv_arc_facts={f"Unemployment_{g}": _gender_arcs(g) for g in _PROBED},
    )
    events, _warnings = _stream_slot_options_events(
        "spec_x",
        shape,
        bindings,
        slot_options,
        entities={"observationAbout": ["country/KEN"]},
        date_request=None,
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        materialise_graph=None,
        coverage=empty_coverage_snapshot().index,
        deadline=None,
        facets_pinned=False,
        variable_labels=NO_LABELS,
        open_axes=frozenset(),
    )
    return {
        ev.key.property.dcid: [(o.value.ref.dcid, o.served_rows) for o in ev.options]
        for ev in events
    }


def _gender_arcs(gender: str) -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "unemploymentRate"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": "gender"}]},
        "gender": {"nodes": [{"dcid": gender}]},
    }


_PROBED = {"Male": 2, "Female": 0, "Other": 1}


def test_trailing_events_carry_the_counted_menus_without_reprobing_them(monkeypatch):
    """A counted menu streams as counted. The probe is not asked about it, so an option the
    count dropped is not restored."""
    counted = count_slot_options(
        [_gender_options("Male", "Female", "Other")],
        [OptionServedFact(property="gender", served={"Male": _served(sv1=1, sv2=1)})],
        [],
    )
    probed: list[str] = []
    events = _stream(monkeypatch, counted, [], probed)

    assert probed == []
    assert events == {"gender": [("Male", 2)]}


def test_trailing_events_count_an_uncounted_bound_menu_from_the_probe(monkeypatch):
    """A subject-place family's bound menu is counted by the probe. An option it finds no
    series for is dropped, except the current value."""
    bindings = [
        SlotBindingDraft(axis="how", property_dcid="gender", kind="value", value_dcids=["Female"])
    ]
    probed: list[str] = []
    events = _stream(monkeypatch, [_gender_options("Male", "Female", "Other")], bindings, probed)

    assert probed == ["gender"]
    assert events == {"gender": [("Male", 2), ("Female", 0), ("Other", 1)]}

    probed.clear()
    events = _stream(monkeypatch, [_gender_options("Male", "Female", "Other")], [], probed)
    assert events == {"gender": [("Male", 2), ("Other", 1)]}


# ---------------------------------------------------------------------------
# A same-shape merge counts its menus from the members' unioned facts
# ---------------------------------------------------------------------------


def test_a_merged_menu_counts_a_series_two_members_serve_once():
    """Kenya and Ethiopia both serve `sv_shared` in education, Kenya in 2019 and 2020 and
    Ethiopia's count of it in 2020 and 2021. The merged menu ORs the series' years, three
    rows, as the merged answer serves them once, where summing would state four."""

    def years(*held: int) -> dict[tuple[str, str], SeriesRows]:
        bits = sum(1 << (year - 2000) for year in held)
        return {("sv_shared", "x"): SeriesRows(rows=len(held), exact=True, years=bits)}

    def region(recipient: str, served: dict, index: int):
        fact = OptionServedFact(property=_PURPOSE, served=served)
        [menu] = count_slot_options(
            [_options(_PURPOSE, "CRSPurposeCode110", "CRSPurposeCode120")], [fact], []
        )
        spec = _where_bound_spec(_recipient_only_shape(), recipient, "sv_shared")
        spec = spec.model_copy(
            update={"resolved": spec.resolved.model_copy(update={"slot_options": [menu]})}
        )
        return replace(
            _region(spec, f"aid to {recipient}", earliest_index=index),
            option_axis_facts=(fact,),
        )

    kenya = region("country/KEN", {"CRSPurposeCode110": years(2019, 2020)}, 0)
    ethiopia = region(
        "country/ETH",
        {"CRSPurposeCode110": years(2020, 2021), "CRSPurposeCode120": _served(sv_health=1)},
        1,
    )

    [merged], _residual = collapse_same_shape([kenya, ethiopia])

    assert _menu(merged.spec, _PURPOSE) == {"CRSPurposeCode110": 3, "CRSPurposeCode120": 1}


# ---------------------------------------------------------------------------
# A family the code index does not decide is counted from the family the mint's walk resolves,
# and every counted option is reachable by a resubmit of the client's member list
# ---------------------------------------------------------------------------

# Debt service, a family the packaged index carries no row of, so the mint's graph walk
# decides it. It has a total. A pick of a creditor reads the parts under it.
_DS_TOTAL = "Amount_FinancialFlow_ServicingPayment_Nominal"
_DS_OFFICIAL = "Amount_FinancialFlow_ServicingPayment_IDSCreditorTypeOFFT_Nominal"
_DS_PRIVATE = "Amount_FinancialFlow_ServicingPayment_IDSCreditorTypePRVT_Nominal"
_DS_OFFICIAL_GG = (
    "Amount_FinancialFlow_ServicingPayment_IDSCreditorTypeOFFT_IDSDebtorTypeGG_Nominal"
)
_DS_CONSTRAINTS = {
    _DS_TOTAL: {},
    _DS_OFFICIAL: {"idsCreditorType": "IDSCreditorTypeOFFT"},
    _DS_PRIVATE: {"idsCreditorType": "IDSCreditorTypePRVT"},
    _DS_OFFICIAL_GG: {"idsCreditorType": "IDSCreditorTypeOFFT", "idsDebtorType": "IDSDebtorTypeGG"},
}
_DS_PROPS = ("financialFlowType", "idsCreditorType", "idsDebtorType")
# Germany to Guatemala holds the total and the official series. The private series has no
# row there.
_DS_ROWS = [
    (_DS_TOTAL, "ids", _DEU, _GTM, [2020, 2021]),
    (_DS_OFFICIAL, "ids", _DEU, _GTM, [2020]),
    (_DS_OFFICIAL_GG, "ids", _DEU, _GTM, [2021]),
    (_DS_PRIVATE, "ids", _FRA, _GTM, [2021]),
]


def _ds_arcs(dcid: str) -> dict:
    constraints = {"financialFlowType": "ServicingPayment", **_DS_CONSTRAINTS[dcid]}
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": dcid}]},
        "populationType": {"nodes": [{"dcid": "FinancialFlow"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraints]},
        "observationProperties": {
            "nodes": [{"dcid": _ROLE_NAMES.donor}, {"dcid": _ROLE_NAMES.recipient}]
        },
        **{p: {"nodes": [{"dcid": v}]} for p, v in constraints.items()},
    }


def _debt_resubmit(monkeypatch: pytest.MonkeyPatch, **bound: str):
    """Resubmit debt service from Germany to Guatemala with `bound` slots."""
    shape = identity_shape_for(
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
    )
    entities = [origin_entity(_DEU), destination_entity(_GTM)]
    values = {"financialFlowType": "ServicingPayment", **bound}
    _edited_id, identity = make_base(
        shape, [_slot(p, values.get(p)) for p in _DS_PROPS], entities=entities
    )
    nodes = {
        dcid: {"label": dcid, "type": "StatisticalVariable", "arcs": _ds_arcs(dcid)}
        for dcid in _DS_CONSTRAINTS
    }
    nodes |= {_DEU: {"label": "Germany", "type": "Country"}}
    nodes |= {_GTM: {"label": "Guatemala", "type": "Country"}}
    nodes |= {
        code: {"label": code, "type": "Thing"}
        for constraints in _DS_CONSTRAINTS.values()
        for code in constraints.values()
    }
    graph = FakeGraph(nodes=nodes, obs={})
    coverage = patch_coverage(
        monkeypatch, _DS_ROWS, role_names=dict.fromkeys(_DS_CONSTRAINTS, _ROLE_NAMES)
    )
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=identity,
        ),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=FakeCoverageService(coverage).snapshot(),
        options=walk_options(
            graph,
            EngineOptions(
                materialise_graph=with_member_arcs(
                    axis_confirming_reader(*_DS_CONSTRAINTS, graph=graph), graph
                )
            ),
        ),
    )


def test_a_walked_family_serving_its_total_counts_the_parts_a_pick_would_serve(monkeypatch):
    """The answer serves the total alone, yet the creditor menu counts the parts. Official
    creditors serve the official series (the general-government part is reduced under it),
    and private creditors, with no row from Germany, are not offered. The debtor menu counts
    the one general-government series."""
    region = _debt_resubmit(monkeypatch)
    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]
    assert [sv.ref.dcid for sv in spec.resolved.members] == [_DS_TOTAL]

    assert _menu(spec, "idsCreditorType") == {"IDSCreditorTypeOFFT": 1}
    assert _menu(spec, "idsDebtorType") == {"IDSDebtorTypeGG": 1}


@pytest.mark.parametrize(
    ("bound", "served"),
    [
        ({"idsCreditorType": "IDSCreditorTypeOFFT"}, [_DS_OFFICIAL]),
        ({"idsDebtorType": "IDSDebtorTypeGG"}, [_DS_OFFICIAL_GG]),
    ],
)
def test_a_pick_elects_the_edited_family(monkeypatch, bound, served):
    """A pick of a counted option elects the edited identity's family, as the Datasets mint
    does, and serves the series the option counted."""
    region = _debt_resubmit(monkeypatch, **bound)

    assert region.status == "definite", region.no_data_reason
    assert [sv.ref.dcid for sv in region.specs[0].resolved.members] == served


# ---------------------------------------------------------------------------
# A count is of rows: a StatVar at each entity it is served at, in each year
# ---------------------------------------------------------------------------

# Germany and France both fund project-type education in Guatemala. Germany alone funds
# project-type health care. France also reports a commitment.
_TWO_DONOR_ROWS = [
    (_PROJECT_EDUCATION, "crs", _DEU, _GTM, [2019, 2020]),
    (_PROJECT_EDUCATION, "crs", _FRA, _GTM, [2020]),
    (_PROJECT_HEALTH, "crs", _DEU, _GTM, [2020]),
    (_COMMITMENT, "crs", _DEU, _GTM, [2020]),
    (_COMMITMENT, "crs", _FRA, _GTM, [2021]),
]


def test_a_menu_counts_each_statvar_once_per_donor_it_is_served_from(monkeypatch):
    """From Germany and France, the education member is served twice, once per donor, so
    project-type interventions serve three series and four rows, as the answer's download
    states, where counting StatVars would say two. The commitments alternative serves two
    rows."""
    region = _crs_region(monkeypatch, bound={}, origins=(_DEU, _FRA), rows=_TWO_DONOR_ROWS)
    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]

    assert spec.resolved.download is not None
    assert (spec.resolved.download.series, spec.resolved.download.rows) == (3, 4)
    assert _menu(spec, _MODALITY) == {
        "CRSCooperationModalityC": 4,
        "CRSCooperationModalityC01": 4,
    }
    assert _menu(spec, _PURPOSE)["CRSPurposeCode110"] == 3
    assert _alternatives(spec) == {"commitments": 2}


# ---------------------------------------------------------------------------
# An edit elects the edited family
# ---------------------------------------------------------------------------


def test_an_edit_the_index_decides_elects_the_index_family():
    """An edited identity the code index decides elects the index's family of it, as the
    Datasets mint does. Here that is the non-export credit flow's members."""
    from types import SimpleNamespace

    from search.engine.region.promote_reread import _elected_members

    shape = identity_shape_for(
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
    )
    values = {_FLOW: _OOF, "financialFlowType": "Disbursement", "nettingBasis": "Gross"}
    slots = [
        _slot(prop, values.get(prop))
        for prop in (_MODALITY, _PURPOSE, _FINANCE, _FLOW, "financialFlowType", "nettingBasis")
    ]
    _spec_id, identity = make_base(
        shape, slots, entities=[origin_entity(_DEU), destination_entity(_GTM)]
    )

    elected = _elected_members(SimpleNamespace(inp=SimpleNamespace(identity=identity)))

    assert _OOF_EDUCATION in elected
    assert _PROJECT_EDUCATION not in elected


# ---------------------------------------------------------------------------
# A menu the coverage index counts at zero throughout is left to the probe
# ---------------------------------------------------------------------------


def test_a_menu_counted_at_zero_for_every_option_is_left_uncounted(monkeypatch, caplog):
    """When the coverage index serves none of an answer's members, as when its facets lack
    metadata, counting finds no series for any option. The answer still serves data, so
    the menus stay, uncounted, for the probe, and a warning names the drift."""
    monkeypatch.setattr(
        "search.engine.region.option_coverage.RoleServedFilter.served",
        lambda self, member_facts: {},
    )
    with caplog.at_level("WARNING", logger="search.engine.region.option_coverage"):
        spec = _resubmit(monkeypatch)

    assert set(_menu(spec, _MODALITY).values()) == {None}
    assert "no served series for any option of 'crsCooperationModality'" in caplog.text


@pytest.mark.parametrize(
    ("picks", "drifted"),
    [
        pytest.param({_OOF: ["sv_oof"]}, True, id="members-carry-an-option"),
        pytest.param({_OOF: []}, False, id="no-member-carries-any-option"),
    ],
)
def test_only_a_menu_whose_options_members_carry_counts_as_drift(picks, drifted):
    """A menu counted at zero throughout is drift when the answer's members carry an option.
    When none carries any, as for a value another family's vocabulary lists, the menu serves
    nothing and is dropped."""
    from search.engine.region.option_coverage import _drifted

    fact = OptionServedFact(property=_FLOW, served={_OOF: {}})

    assert _drifted(fact, picks, shape_id="aid") is drifted


# ---------------------------------------------------------------------------
# Under an open flow axis, a pick whose members leave the axis undeclared counts nothing
# ---------------------------------------------------------------------------

_INSTRUMENT = "idsInstrument"


def _instrument_pick(value: str, *roles: str):
    from code_index.types import StatVarRow
    from search.engine.region.option_coverage import _IndexPick
    from search.engine.retrieval.member_facts import IndexMemberFacts

    row = StatVarRow(
        dcid=f"Debt_{value}",
        population_type="FinancialFlow",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=roles,
        constraint_properties={_INSTRUMENT: value},
    )
    return _IndexPick(dataset="ids", members={row.dcid: IndexMemberFacts(row=row, dataset="ids")})


@pytest.mark.parametrize(
    ("open_axes", "counted"),
    [
        pytest.param(frozenset({"origin"}), {"Loans": 1}, id="open-creditor"),
        pytest.param(frozenset(), {"Bonds": 1, "Loans": 1}, id="no-open-axis"),
    ],
)
def test_a_pick_leaving_an_open_axis_undeclared_counts_no_series(monkeypatch, open_axes, counted):
    """Debt owed by Kenya to every creditor, by instrument. The bonds pick elects a family
    holding only the debtor-level total, whose resubmit refuses `open_role_unavailable`, so
    it counts nothing. The loans pick elects a by-creditor twin and counts its series. With
    no open axis both picks count."""
    from typing import cast

    from search.engine.region.option_coverage import RoleServedFilter, option_axis_facts
    from search.engine.retrieval.read_plan import PairPlan

    picks = {
        "Bonds": _instrument_pick("Bonds", "debtor"),
        "Loans": _instrument_pick("Loans", "creditor", "debtor"),
    }
    monkeypatch.setattr(
        "search.engine.region.option_coverage._index_option_picks",
        lambda query, prop, values: {value: picks[value] for value in values},
    )
    monkeypatch.setattr(
        "search.engine.region.option_coverage.RoleServedFilter.served",
        lambda self, member_facts: {sv: _served(**{sv: 1}) for sv in member_facts},
    )
    shape = ShapeDraft(
        shape_id="debt",
        label="debt",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(SlotKeyDraft(axis="what", property_dcid=_INSTRUMENT, label="instrument"),),
    )

    [fact] = option_axis_facts(
        shape,
        [],
        menus={_INSTRUMENT: ["Bonds", "Loans"]},
        plan=cast(PairPlan, object()),
        member_facts={},
        served_filter=RoleServedFilter(
            probe_entities={"debtor": ["country/KEN"]},
            open_axes=open_axes,
            years="latest",
            coverage=cast(CoverageIndex, None),
        ),
        facets_pinned=False,
    )

    assert {value: len(series) for value, series in fact.served.items() if series} == counted


# ---------------------------------------------------------------------------
# A merged answer recounts its alternatives and offers conversions over every member
# ---------------------------------------------------------------------------


def test_a_merged_answer_counts_its_alternatives_and_offers_conversions_over_its_members(
    monkeypatch,
):
    """Aid for education and aid for basic health care merge into one answer over both
    purposes. Its commitments alternative counts the one commitment series, in education,
    and its conversion offer reads the members both answers serve and both read plans."""
    members = list(dict.fromkeys(row[0] for row in _ROWS))
    coverage = patch_coverage(monkeypatch, _ROWS, role_names=dict.fromkeys(members, _ROLE_NAMES))
    education = _crs_region(monkeypatch, bound={_PURPOSE: "CRSPurposeCode11110"}, coverage=coverage)
    health = _crs_region(monkeypatch, bound={_PURPOSE: "CRSPurposeCode12220"}, coverage=coverage)
    health = replace(health, earliest_index=1)
    offered: list[tuple[int, tuple]] = []

    def record_offer(spec, *, members, plans):
        offered.append((len(members), plans))
        return spec

    monkeypatch.setattr("search.engine.same_shape.with_unit_conversion", record_offer)

    [merged], residual = _collapse([education, health], coverage)

    assert residual == []
    assert _alternatives(merged.spec) == {"commitments": 1}
    assert len(education.served_plans) == len(health.served_plans) == 1
    assert offered == [
        (
            len(education.served_members) + len(health.served_members),
            education.served_plans + health.served_plans,
        )
    ]


# ---------------------------------------------------------------------------
# An option's rows are the rows its pick's own answer states
# ---------------------------------------------------------------------------


def _option(spec: Spec, prop: str, value: str) -> SlotOption:
    return next(o for o in _slot_menu(spec, prop).options if o.value.ref.dcid == value)


def _assert_option_bounds_its_pick(rows: int | None, exact: bool | None, pick: Spec) -> None:
    """`rows` is the pick's own `download.rows` when `exact`, else at least it."""
    download = pick.resolved.download
    assert download is not None and download.rows is not None and rows is not None
    if exact:
        assert rows == download.rows
    else:
        assert rows >= download.rows


def _pick(monkeypatch, *, bound: dict[str, str], option: tuple[str, str], **read) -> tuple:
    """The answer with `bound`, its option `option` (property, value), and the answer a
    pick of that option posts, both under `read` (`_crs_region`'s window, facets, rows
    and facet metadata)."""
    base = _crs_region(monkeypatch, bound=bound, **read)
    assert base.status == "definite", base.no_data_reason
    prop, value = option
    picked = _crs_region(monkeypatch, bound={**bound, prop: value}, **read)
    assert picked.status == "definite", picked.no_data_reason
    found = _option(base.specs[0], prop, value)
    return found.served_rows, found.served_rows_exact, picked.specs[0]


_EUR_CURRENT = IdentityFacetValueRequest(
    request=IdentityUnitRequest(currency="EUR", prices="current")
)
_EUR_CONSTANT = IdentityFacetValueRequest(
    request=IdentityUnitRequest(currency="EUR", prices="constant", base_year=2021)
)

# A second CRS facet with the first one's provenance and another method, so a read elects
# one of the two.
_OTHER_METHOD = {
    "crs": FacetMetadata(provenance_id=_CRS_PROVENANCE),
    "crs_other": FacetMetadata(provenance_id=_CRS_PROVENANCE, measurement_method="Other"),
}
# Germany and France fund project-type education in 2019, Germany again in 2020, and
# Germany project-type health care in 2020. Germany's deflator lacks 2019.
_DONORS_IN_2019 = [
    (_PROJECT_EDUCATION, "crs", _DEU, _GTM, [2019, 2020]),
    (_PROJECT_EDUCATION, "crs", _FRA, _GTM, [2019]),
    (_PROJECT_HEALTH, "crs", _DEU, _GTM, [2020]),
]
# Germany funds project-type education in 2019 and 2020 and France in 2021, Germany
# project-type health care in 2020.
_TWO_DONORS_DIFFERENT_LAST_YEARS = [
    (_PROJECT_EDUCATION, "crs", _DEU, _GTM, [2019, 2020]),
    (_PROJECT_EDUCATION, "crs", _FRA, _GTM, [2021]),
    (_PROJECT_HEALTH, "crs", _DEU, _GTM, [2020]),
]


@pytest.mark.parametrize(
    ("bound", "option", "read", "exact"),
    [
        pytest.param({}, (_MODALITY, "CRSCooperationModalityC01"), {}, True, id="unbound-slot"),
        pytest.param({}, (_FLOW, _OOF), {}, True, id="bound-slot"),
        pytest.param(
            {}, (_MODALITY, "CRSCooperationModalityC01"), {"window": "latest"}, True, id="latest"
        ),
        pytest.param(
            {},
            (_PURPOSE, "CRSPurposeCode110"),
            {"window": TimeWindow(start_year=2020, end_year=2021)},
            True,
            id="ranged-window",
        ),
        pytest.param(
            {},
            (_MODALITY, "CRSCooperationModalityC01"),
            {"facets": {"provenance": IdentityBindingValue(ref=_CRS_PROVENANCE)}},
            False,
            id="source-pinned",
        ),
        pytest.param(
            {},
            (_MODALITY, "CRSCooperationModalityC01"),
            {
                "facets": {"unit": _EUR_CURRENT},
                "facet_metadata": {
                    "crs": FacetMetadata(provenance_id=_CRS_PROVENANCE, unit="CurrencyUSD")
                },
            },
            True,
            id="converted",
        ),
        pytest.param(
            {},
            (_MODALITY, "CRSCooperationModalityC01"),
            {
                "facets": {"unit": _EUR_CONSTANT},
                "origins": (_DEU, _FRA),
                "rows": _DONORS_IN_2019,
                "facet_metadata": {
                    "crs": FacetMetadata(provenance_id=_CRS_PROVENANCE, unit="CurrencyUSD")
                },
            },
            True,
            id="constant-prices-with-a-donor-missing-a-deflator-year",
        ),
        pytest.param(
            {},
            (_PURPOSE, "CRSPurposeCode110"),
            {
                "origins": (_DEU, _FRA),
                "rows": [
                    (_PROJECT_EDUCATION, "crs", _DEU, _GTM, [2019, 2020]),
                    (_PROJECT_EDUCATION, "crs_other", _FRA, _GTM, [2021]),
                    (_PROJECT_HEALTH, "crs", _DEU, _GTM, [2020]),
                ],
                "facet_metadata": _OTHER_METHOD,
            },
            False,
            id="a-member-split-across-facets-by-donor",
        ),
        pytest.param(
            {},
            (_PURPOSE, "CRSPurposeCode110"),
            {
                "rows": [
                    (_PROJECT_EDUCATION, "crs", _DEU, _GTM, [2019]),
                    (_PROJECT_EDUCATION, "crs_more", _DEU, _GTM, [2020]),
                ],
                "facet_metadata": {
                    "crs": FacetMetadata(provenance_id=_CRS_PROVENANCE),
                    "crs_more": FacetMetadata(provenance_id=_CRS_PROVENANCE),
                },
            },
            True,
            id="two-facets-in-one-collapse-group",
        ),
        pytest.param(
            {},
            (_MODALITY, "CRSCooperationModalityC01"),
            {
                "origins": (_DEU, _FRA),
                "rows": _TWO_DONORS_DIFFERENT_LAST_YEARS,
                "window": "latest",
            },
            True,
            id="latest-with-donors-ending-in-different-years",
        ),
    ],
)
def test_an_options_rows_are_its_picks_download_rows(monkeypatch, bound, option, read, exact):
    """Posting a counted option gives an answer whose `download.rows` the option stated:
    equal when exact, at least it when the answer pins a facet value. A unit request counts
    the rows its conversion keeps, so the option and the answer's own download are exact,
    at constant prices without Germany's 2019 rows, whose deflator the tables lack."""
    rows, rows_exact, pick = _pick(monkeypatch, bound=bound, option=option, **read)

    assert rows_exact is exact
    _assert_option_bounds_its_pick(rows, rows_exact, pick)
    if "unit" in read.get("facets", {}):
        assert pick.resolved.download.rows_exact is True


@pytest.mark.parametrize(
    ("option", "exact"),
    [
        pytest.param((_MODALITY, "CRSCooperationModalityC01"), False, id="members-in-two-facets"),
        pytest.param((_PURPOSE, "CRSPurposeCode110"), True, id="members-in-one-facet"),
    ],
)
def test_a_bulk_read_options_rows_are_exact_only_in_one_facet(
    monkeypatch, search_settings, option, exact
):
    """Aid to Guatemala from any donor reads in bulk, one facet for every member. Education
    sits in one facet and health care in another of the same provenance, method and unit,
    so a pick of project-type interventions downloads one facet's rows and its count is an
    upper bound. A pick of education reads its one facet, exactly."""
    search_settings(obs_preview_max_pairs=1)
    rows = [
        (_PROJECT_EDUCATION, "crs", _DEU, _GTM, [2019, 2020]),
        (_PROJECT_EDUCATION, "crs", _FRA, _GTM, [2021]),
        (_PROJECT_HEALTH, "crs_health", _DEU, _GTM, [2020]),
    ]
    metadata = {
        "crs": FacetMetadata(provenance_id=_CRS_PROVENANCE),
        "crs_health": FacetMetadata(provenance_id=_CRS_PROVENANCE),
    }

    rows_counted, rows_exact, pick = _pick(
        monkeypatch, bound={}, option=option, origins=(), rows=rows, facet_metadata=metadata
    )

    assert rows_exact is exact
    _assert_option_bounds_its_pick(rows_counted, rows_exact, pick)


def test_a_merged_options_rows_are_its_parts_download_rows_under_latest(monkeypatch):
    """Aid for education and aid for basic health care from Germany and France merge into
    one answer, read at the latest year. Education's latest year is 2021, from France
    alone, and health care's is 2020, so a pick of project-type interventions downloads
    one row of each, the two the merged option states."""
    rows = _TWO_DONORS_DIFFERENT_LAST_YEARS
    members = list(dict.fromkeys(row[0] for row in rows))
    coverage = patch_coverage(monkeypatch, rows, role_names=dict.fromkeys(members, _ROLE_NAMES))
    read = {"origins": (_DEU, _FRA), "rows": rows, "coverage": coverage, "window": "latest"}
    education = _crs_region(monkeypatch, bound={_PURPOSE: "CRSPurposeCode11110"}, **read)
    health = _crs_region(monkeypatch, bound={_PURPOSE: "CRSPurposeCode12220"}, **read)

    [merged], _residual = _collapse([education, replace(health, earliest_index=1)], coverage)

    option = _option(merged.spec, _MODALITY, "CRSCooperationModalityC01")
    downloads = [part.specs[0].resolved.download for part in (education, health)]
    assert all(d is not None and d.rows_exact for d in downloads)
    assert (option.served_rows, option.served_rows_exact) == (
        sum(d.rows for d in downloads if d is not None),
        True,
    )


def test_a_measure_alternatives_rows_are_its_resubmits_download_rows(monkeypatch):
    """Posting the commitments alternative's definition gives an answer whose download
    states the one row the alternative counted."""
    spec = _resubmit(monkeypatch)
    [alternative] = spec.resolved.alternative_measures or []

    picked = _crs_region(monkeypatch, bound={}, definition=alternative.definition)

    assert picked.status == "definite", picked.no_data_reason
    assert alternative.served_rows_exact is True
    _assert_option_bounds_its_pick(alternative.served_rows, True, picked.specs[0])


_LIFE_EXPECTANCY_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "Person",
    "measured_property": "lifeExpectancy",
    "stat_type": "measuredValue",
}
_KEN = "country/KEN"


def _life_expectancy_obs() -> dict[str, list]:
    """Life expectancy of women and men in Kenya as recorded, men's trimmed to 40 annual
    points from 1984, so the two options count differently."""
    recorded = _load_json("graph_obs.json")
    obs = {
        f"LifeExpectancy_Person_{g}|{_KEN}": recorded[f"LifeExpectancy_Person_{g}|{_KEN}"]
        for g in ("Female", "Male")
    }
    obs[f"LifeExpectancy_Person_Male|{_KEN}"] = [
        {**facet, "earliestDate": "1984", "obsCount": 40}
        if facet["provenanceId"] == "dc/base/WorldDevelopmentIndicators"
        else facet
        for facet in obs[f"LifeExpectancy_Person_Male|{_KEN}"]
    ]
    return obs


def test_a_probed_options_rows_are_its_picks_download_rows():
    """A subject-place family's bound gender menu is counted by the coverage probe from the
    plan a pick of each option reads (`download.pair_extent`). With women's life expectancy
    in Kenya bound, men's counts the 40 rows the men's answer downloads."""
    obs = _life_expectancy_obs()
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    shape = ShapeDraft(
        shape_id="life-expectancy",
        label="life expectancy",
        pop_type_dcid="Person",
        meas_prop_dcid="lifeExpectancy",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(SlotKeyDraft(axis="how", property_dcid="gender", label="gender"),),
        sv_arc_facts={
            f"LifeExpectancy_Person_{g}": {
                **_gender_arcs(g),
                "measuredProperty": {"nodes": [{"dcid": "lifeExpectancy"}]},
            }
            for g in ("Female", "Male")
        },
    )
    events, _warnings = _stream_slot_options_events(
        "spec_x",
        shape,
        [
            SlotBindingDraft(
                axis="how", property_dcid="gender", kind="value", value_dcids=["Female"]
            )
        ],
        [_gender_options("Female", "Male")],
        entities={"observationAbout": [_KEN]},
        date_request=None,
        graph=FakeGraph(obs=obs),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        materialise_graph=materialise_graph,
        coverage=empty_coverage_snapshot().index,
        deadline=None,
        facets_pinned=False,
        variable_labels=NO_LABELS,
        open_axes=frozenset(),
    )
    [male] = [o for event in events for o in event.options if o.value.ref.dcid == "Male"]

    # The men's answer a pick lands on: the resubmit elects the men's StatVar.
    identity = make_identity(
        _LIFE_EXPECTANCY_SHAPE,
        [_value_slot("how", "gender", "gender", "Male", "Male")],
        entities=[subject_entity(_KEN)],
    )
    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=identity,
        ),
        graph=FakeGraph(obs=obs),
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=walk_options(
            FakeGraph(obs=obs),
            EngineOptions(
                materialise_graph=with_member_arcs(materialise_graph, FakeGraph(obs=obs)),
                materialise_sdmx=materialise_sdmx,
            ),
        ),
    )

    assert region.status == "definite", region.no_data_reason
    assert (male.served_rows, male.served_rows_exact) == (40, True)
    _assert_option_bounds_its_pick(male.served_rows, male.served_rows_exact, region.specs[0])


def test_a_series_two_facets_serve_in_different_years_counts_an_upper_bound(monkeypatch):
    """Project-type education holds 2019 in one facet and 2021 in another, each with its
    own method, so a pick's read elects one of them and downloads one row. The option
    counts both years, two rows, and says so: it is an upper bound."""
    rows = [
        (_PROJECT_EDUCATION, "crs", _DEU, _GTM, [2019]),
        (_PROJECT_EDUCATION, "crs_other", _DEU, _GTM, [2021]),
        (_PROJECT_HEALTH, "crs", _DEU, _GTM, [2020]),
    ]
    metadata = {
        "crs": FacetMetadata(provenance_id=_CRS_PROVENANCE),
        "crs_other": FacetMetadata(provenance_id=_CRS_PROVENANCE, measurement_method="Other"),
    }

    served_rows, exact, pick = _pick(
        monkeypatch,
        bound={},
        option=(_PURPOSE, "CRSPurposeCode110"),
        rows=rows,
        facet_metadata=metadata,
    )

    assert pick.resolved.download is not None
    assert (served_rows, exact, pick.resolved.download.rows) == (2, False, 1)


@pytest.mark.parametrize(
    ("names", "named", "unnamed"),
    [
        pytest.param(_CATALOGUE.entity_names, (("Germany", (2019,), (2020,)),), 0, id="named"),
        pytest.param({}, (), 1, id="no-name-held-counts-it"),
    ],
)
def test_a_constant_price_answer_names_the_donor_years_its_deflators_leave_out(
    monkeypatch, names, named, unnamed
):
    """Germany's deflator lacks 2019 and France's holds it, so the answer keeps 2019 for
    France, leaves out Germany's 2019 row, and departs on the unit token naming Germany
    with the year it loses, or counting it when the catalogue holds no name for it. Its
    download counts the three rows its conversion keeps."""
    region = _crs_region(
        monkeypatch,
        bound={},
        origins=(_DEU, _FRA),
        rows=_DONORS_IN_2019,
        facets={"unit": _EUR_CONSTANT},
        facet_metadata={"crs": FacetMetadata(provenance_id=_CRS_PROVENANCE, unit="CurrencyUSD")},
        catalogue=replace(_CATALOGUE, entity_names=names),
    )

    assert region.status == "definite", region.no_data_reason
    assert [w.code for w in region.warnings] == ["deflator_missing_omitted"]
    assert [
        (fact.binding_ref, fact.omitted_years, fact.omitted_unnamed)
        for fact in region.departure_facts.values()
        if fact.code == "year_unavailable"
    ] == [("unit:", named, unnamed)]
    download = region.specs[0].resolved.download
    assert download is not None
    assert (download.rows, download.rows_exact) == (3, True)


def test_a_constant_price_answer_the_tables_cannot_count_warns_rows_may_be_left_out(
    monkeypatch,
):
    """A catalogue that names no recipe for the CRS provenance cannot say whose rows the
    conversion leaves out, so the answer warns that some may be, departs on the unit token
    naming no entity, and its download count is an upper bound."""
    region = _crs_region(
        monkeypatch,
        bound={},
        origins=(_DEU, _FRA),
        rows=_DONORS_IN_2019,
        facets={"unit": _EUR_CONSTANT},
        facet_metadata={"crs": FacetMetadata(provenance_id=_CRS_PROVENANCE, unit="CurrencyUSD")},
        catalogue=replace(_CATALOGUE, recipes={}),
    )

    assert region.status == "definite", region.no_data_reason
    assert [w.code for w in region.warnings] == ["deflator_missing_omitted"]
    assert [
        (fact.binding_ref, fact.omitted_years, fact.omission_undecided)
        for fact in region.departure_facts.values()
        if fact.code == "year_unavailable"
    ] == [("unit:", (), True)]
    download = region.specs[0].resolved.download
    assert download is not None
    assert download.rows_exact is False
