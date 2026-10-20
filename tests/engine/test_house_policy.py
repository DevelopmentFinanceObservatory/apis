"""House policies: fact extraction, the part's context, the six policies and the tie-break."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from code_index.shared import get_index
from code_index.types import HouseDefaultCase
from search.engine.binders.house_policy import (
    Context,
    PolicyName,
    PolicyPool,
    ShapeFacts,
    apply_house_policies,
    apply_policy,
    context_of,
    shape_facts,
)
from search.engine.extract import DateRequest
from search.engine.frame import Frame, door_frame
from search.engine.qualifier_match import QualifierMatch
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.models import TimeWindow

_AUDIT_CASES = Path(__file__).parents[1] / "fixtures" / "house_policy" / "policy_audit_cases.json"

_CASE_SYSTEMS: Mapping[str, HouseDefaultCase] = {"purposeCode": "sector"}
_WANTED: Mapping[HouseDefaultCase, str] = {
    "sector": "provenance/Sector",
    "recipient": "provenance/Recipient",
    "donor": "provenance/Donor",
}


def _facts(index: int, **fields: object) -> ShapeFacts:
    base = ShapeFacts(
        index=index,
        population_type="FinancialFlow",
        measured_property="amount",
        stat_type="measuredValue",
        qualifier="Nominal",
        netting=None,
        flow_types=None,
        datasets=(),
        available=None,
    )
    return replace(base, **fields)


def _ctx(*, pre_ge: bool = False, case: HouseDefaultCase | None = None) -> Context:
    return Context(pre_ge=pre_ge, dataset_case=case, wanted_dataset=_WANTED[case] if case else None)


_OUTPUT = _facts(0, population_type="EconomicActivity")
_VALUE_ADDED = _facts(1, population_type="EconomicProduction", measured_property="valueAdded")
_GE = _facts(0, measured_property="grantEquivalent")
_AMOUNT = _facts(1, netting=("Gross", "Net"))
_BY_DATASET = (
    _facts(0, datasets=(_WANTED["sector"],)),
    _facts(1, datasets=(_WANTED["recipient"],)),
    _facts(2, datasets=(_WANTED["donor"],)),
)


@pytest.mark.parametrize(
    ("facts", "context", "chosen", "expected"),
    [
        pytest.param((_OUTPUT, _VALUE_ADDED), _ctx(), 1, 0, id="output-over-value-added"),
        pytest.param(
            (_OUTPUT, replace(_OUTPUT, index=1)), _ctx(), 1, 1, id="nominal-output-only-unchanged"
        ),
        pytest.param(
            (
                _facts(
                    0, population_type="EconomicActivity", stat_type="growthRate", qualifier=None
                ),
                _facts(1, population_type="EconomicActivity", stat_type="growthRate"),
            ),
            _ctx(),
            0,
            0,
            id="output-growth-rates-without-value-added-unchanged",
        ),
        pytest.param((_GE, _AMOUNT), _ctx(pre_ge=True), 0, 1, id="pre-2018-window-amount"),
        pytest.param((_GE, _AMOUNT), _ctx(), 1, 0, id="later-window-grant-equivalent"),
        pytest.param(
            (replace(_GE, population_type="FinancialObligation"), _AMOUNT),
            _ctx(),
            1,
            1,
            id="grant-equivalent-off-financial-flows-unchanged",
        ),
        pytest.param(
            (_facts(0, netting=("Net",)), _facts(1, netting=("Gross", "Net"))),
            _ctx(),
            0,
            1,
            id="gross-capable-over-net-only",
        ),
        pytest.param(_BY_DATASET, _ctx(case="sector"), 2, 0, id="sector-crs"),
        pytest.param(_BY_DATASET, _ctx(case="recipient"), 0, 1, id="recipient-dac2a"),
        pytest.param(_BY_DATASET, _ctx(case="donor"), 0, 2, id="donor-dac1"),
        pytest.param(_BY_DATASET, _ctx(), 2, 2, id="no-place-unchanged"),
        pytest.param(
            (_BY_DATASET[0], _facts(1, datasets=(_WANTED["sector"], _WANTED["recipient"]))),
            _ctx(case="recipient"),
            0,
            1,
            id="a-family-two-datasets-publish-fits-either",
        ),
        pytest.param(
            (
                _facts(0, population_type="FinancialObligation", flow_types=("Commitment",)),
                _facts(1, flow_types=("Disbursement",)),
            ),
            _ctx(),
            0,
            1,
            id="disbursement-over-commitment",
        ),
        pytest.param(
            (_facts(0, available=False), _facts(1, available=True)),
            _ctx(case="recipient"),
            0,
            1,
            id="data-at-the-named-places",
        ),
        pytest.param((_GE,), _ctx(pre_ge=True), 0, 0, id="single-candidate"),
        # amount_vs_ge runs first, so grant equivalent wins over a gross amount.
        pytest.param((_GE, _AMOUNT), _ctx(), 0, 0, id="grant-equivalent-before-gross"),
    ],
)
def test_a_rated_pool_elects_the_shape_the_house_policies_prefer(
    facts: tuple[ShapeFacts, ...], context: Context, chosen: int, expected: int
):
    pool = PolicyPool(measuring=tuple(f.index for f in facts), rated=True)

    assert apply_house_policies(facts, pool, context, chosen=chosen).survivor == expected


@pytest.mark.parametrize(
    "available", [(None, None), (True, True), (False, False)], ids=["unknown", "both", "neither"]
)
def test_equal_availability_leaves_the_tie_to_the_lower_index(
    available: tuple[bool | None, bool | None],
):
    facts = tuple(_facts(i, available=a) for i, a in enumerate(available))
    pool = PolicyPool(measuring=(1, 0), election={0: 0.4, 1: 0.4}, rated=True)

    choice = apply_house_policies(facts, pool, _ctx(case="recipient"), chosen=1)

    assert (choice.survivor, choice.fired) == (0, ())


def test_an_unrated_pool_keeps_the_binders_pick_on_the_measure_policies():
    facts = (_OUTPUT, _VALUE_ADDED, _facts(2, measured_property="grantEquivalent"))
    pool = PolicyPool(measuring=(1, 0, 2))

    choice = apply_house_policies(facts, pool, _ctx(), chosen=1)

    assert (choice.survivor, choice.fired, choice.dropped) == (1, (), frozenset())


def test_an_unrated_pool_runs_the_dataset_flow_and_place_policies():
    pool = PolicyPool(measuring=(2, 1, 0))

    choice = apply_house_policies(_BY_DATASET, pool, _ctx(case="sector"), chosen=2)

    assert choice.survivor == 0
    assert choice.fired == ("default_dataset",)
    assert choice.dropped == frozenset({1, 2})
    assert [f.index for f in choice.facts] == [2, 1, 0]


@pytest.mark.parametrize(
    ("holds", "survivor", "dropped"),
    [
        pytest.param(None, 1, {0, 2}, id="no-report-every-shape-holds"),
        pytest.param({0: True, 1: False, 2: True}, 0, {1}, id="the-default-lacks-a-constraint"),
        pytest.param({0: True, 2: True}, 0, {1}, id="a-shape-left-out-does-not-hold"),
        pytest.param({0: False, 1: False, 2: False}, 0, {1, 2}, id="the-pick-holds-its-own"),
    ],
)
def test_the_policies_choose_only_among_the_shapes_that_hold_the_picks_constraints(
    holds: Mapping[int, bool] | None, survivor: int, dropped: set[int]
):
    """A recipient question prefers DAC2a, but a DAC2a shape that does not hold a value the
    binder recognised on its CRS pick is set aside before the dataset policy runs."""
    pool = PolicyPool(measuring=(0, 1, 2), rated=True)

    choice = apply_house_policies(_BY_DATASET, pool, _ctx(case="recipient"), chosen=0, holds=holds)

    assert (choice.survivor, choice.dropped) == (survivor, frozenset(dropped))


@pytest.mark.parametrize(
    ("fits", "expected"),
    [
        pytest.param((), [0, 1, 2], id="none-fit"),
        pytest.param((0, 1, 2), [0, 1, 2], id="all-fit"),
        pytest.param((1,), [1], id="some-fit"),
    ],
)
def test_a_policy_narrows_only_when_some_not_all_candidates_fit(
    fits: tuple[int, ...], expected: list[int]
):
    facts = [_facts(i, available=i in fits) for i in range(3)]

    kept = apply_policy("data_at_places", facts, _ctx())

    assert [f.index for f in kept] == expected


def _survivors_of_two() -> tuple[ShapeFacts, ...]:
    """Shapes 0 and 2 have data at the named places; shape 1 does not."""
    return tuple(_facts(i, available=i != 1) for i in range(3))


@pytest.mark.parametrize(
    ("election", "chosen", "expected"),
    [
        pytest.param({0: 0.2, 1: 0.5, 2: 0.3}, 1, 2, id="most-probable-survivor"),
        pytest.param({0: 0.3, 1: 0.1, 2: 0.6}, 0, 2, id="probability-over-the-pick"),
        pytest.param({0: 0.3, 1: 0.4, 2: 0.3}, 2, 0, id="equal-goes-to-lower-index"),
    ],
)
def test_the_election_probability_breaks_the_tie(
    election: dict[int, float], chosen: int, expected: int
):
    pool = PolicyPool(measuring=(0, 1, 2), election=election, rated=True)

    choice = apply_house_policies(_survivors_of_two(), pool, _ctx(), chosen=chosen)

    assert choice.survivor == expected


def test_a_survivor_with_no_probability_or_rating_reads_as_zero():
    unrated_survivor = PolicyPool(measuring=(0, 1, 2), election={0: 0.0, 1: 0.9}, rated=True)
    assert (
        apply_house_policies(_survivors_of_two(), unrated_survivor, _ctx(), chosen=1).survivor == 0
    )

    pool = PolicyPool(measuring=(0, 1, 2), rated=True)
    choice = apply_house_policies(
        _survivors_of_two(), pool, _ctx(), chosen=1, ratings={1: 1.0, 2: 0.1}
    )
    assert choice.survivor == 2


@pytest.mark.parametrize(
    ("ratings", "chosen", "expected"),
    [
        pytest.param({0: 0.9, 1: 1.0, 2: 0.6}, 2, 2, id="surviving-pick-over-higher-rating"),
        pytest.param({0: 0.6, 1: 1.0, 2: 0.9}, 1, 2, id="highest-rated-survivor"),
        pytest.param({0: 0.8, 1: 1.0, 2: 0.8}, 1, 0, id="equal-rating-goes-to-lower-index"),
    ],
)
def test_a_rated_pool_without_probabilities_breaks_the_tie_by_rating(
    ratings: dict[int, float], chosen: int, expected: int
):
    pool = PolicyPool(measuring=(2, 1, 0), rated=True)

    choice = apply_house_policies(_survivors_of_two(), pool, _ctx(), chosen=chosen, ratings=ratings)

    assert choice.survivor == expected


@pytest.mark.parametrize(
    ("chosen", "expected"),
    [pytest.param(0, 0, id="surviving-pick"), pytest.param(1, 2, id="first-survivor-in-order")],
)
def test_an_unrated_pool_breaks_the_tie_by_the_binders_order(chosen: int, expected: int):
    pool = PolicyPool(measuring=(1, 2, 0))

    choice = apply_house_policies(_survivors_of_two(), pool, _ctx(), chosen=chosen)

    assert choice.survivor == expected
    assert choice.fired == ("data_at_places",)


def test_a_pick_outside_the_pool_is_refused():
    with pytest.raises(ValueError, match="not in the pool"):
        apply_house_policies((_GE, _AMOUNT), PolicyPool(measuring=(0, 1)), _ctx(), chosen=2)


def test_shape_facts_read_the_reading_slots_dataset_and_availability():
    netting = SlotKeyDraft(axis="how", property_dcid="nettingBasis", label="netting basis")
    flow = SlotKeyDraft(axis="how", property_dcid="financialFlowType", label="flow type")
    flow_shape = ShapeDraft(
        shape_id="flow",
        label="flow",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=(netting, flow),
        slot_taxonomy={
            "how:nettingBasis": ["Gross", "Net"],
            "how:financialFlowType": ["Disbursement"],
        },
        datasets=("provenance/First", "provenance/Second"),
    )
    plain = replace(
        flow_shape,
        shape_id="plain",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        meas_qual_dcid=None,
        slot_keys=(),
        slot_taxonomy=None,
        datasets=None,
    )

    facts = shape_facts([plain, flow_shape], [True, False], [1, 0], code_index=get_index())

    assert facts == (
        ShapeFacts(
            index=1,
            population_type="FinancialFlow",
            measured_property="amount",
            stat_type="measuredValue",
            qualifier="Nominal",
            netting=("Gross", "Net"),
            flow_types=("Disbursement",),
            datasets=("provenance/First", "provenance/Second"),
            available=False,
        ),
        ShapeFacts(
            index=0,
            population_type="Person",
            measured_property="count",
            stat_type="measuredValue",
            qualifier=None,
            netting=None,
            flow_types=None,
            datasets=(),
            available=True,
        ),
    )
    assert shape_facts([plain], [], [0], code_index=get_index())[0].available is None


def _packaged_flow(measured_property: str, *slot_properties: str) -> ShapeDraft:
    """A FinancialFlow shape with no recalled members and no stamped datasets, so its slot
    properties select its family in the packaged index."""
    return ShapeDraft(
        shape_id=measured_property,
        label=measured_property,
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid=measured_property,
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=tuple(
            SlotKeyDraft(axis="how", property_dcid=prop, label=prop) for prop in slot_properties
        ),
    )


_IMPUTED = _packaged_flow(
    "attributedAmount",
    "crsPurpose",
    "crsTypeOfFlow",
    "dac2aAidType",
    "financialFlowType",
    "nettingBasis",
)
_CRS_AMOUNT = _packaged_flow(
    "amount", "crsPurpose", "crsTypeOfFlow", "financialFlowType", "nettingBasis"
)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        pytest.param(
            _IMPUTED,
            ("provenance/ONEImputedMultilateralSectors",),
            id="crs-keyed-imputed-is-its-own-dataset",
        ),
        pytest.param(
            _packaged_flow("attributedAmount", "dac2aAidType", "financialFlowType", "nettingBasis"),
            ("provenance/OECDDAC2A",),
            id="dac2a-keyed-attributed-amount",
        ),
        pytest.param(_CRS_AMOUNT, ("provenance/OECDCreditorReportingSystem",), id="crs-amount"),
    ],
)
def test_shape_facts_read_the_datasets_of_the_packaged_family(
    shape: ShapeDraft, expected: tuple[str, ...]
):
    (facts,) = shape_facts([shape], [], [0], code_index=get_index())

    assert facts.datasets == expected


def test_a_sector_question_prefers_crs_over_the_imputed_attribution():
    """An opt-in table is its own dataset, so the imputed attribution fits no house default."""
    index = get_index()
    facts = shape_facts([_IMPUTED, _CRS_AMOUNT], [], [0, 1], code_index=index)
    context = Context(
        pre_ge=False, dataset_case="sector", wanted_dataset=index.house_defaults()["sector"]
    )

    assert [f.index for f in apply_policy("default_dataset", facts, context)] == [1]


def _frame(
    *,
    window: TimeWindow | None = None,
    latest: bool = False,
    places: Mapping[str, tuple[str, ...]] | None = None,
) -> Frame:
    places = places or {}
    date_request = DateRequest(window=window, latest=latest) if window or latest else None
    return door_frame(
        door="parsed",
        variable="aid",
        entities=list(places),
        entity_roles={
            name: [DoorRole(axis=axis, span=name) for axis in axes] for name, axes in places.items()
        },
        member_entities=frozenset(),
        pins={},
        date_request=date_request,
        unit_text=None,
        scope=None,
        operations=(),
        open_axes=frozenset(),
    )


def _purpose(code_system: str) -> QualifierMatch:
    return QualifierMatch(
        code_system=code_system,
        code_dcid="Code1",
        label="education",
        source="label",
        matched_text="education",
    )


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        pytest.param(
            _frame(window=TimeWindow(start_year=2015, end_year=2017)), True, id="2015-2017"
        ),
        pytest.param(
            _frame(window=TimeWindow(start_year=2016, end_year=2019)), True, id="2016-2019"
        ),
        pytest.param(
            _frame(window=TimeWindow(start_year=2010, end_year=2022)), True, id="2010-2022"
        ),
        pytest.param(
            _frame(window=TimeWindow(start_year=2018, end_year=2022)), False, id="2018-2022"
        ),
        pytest.param(_frame(window=TimeWindow(start_year=2010)), True, id="open-end"),
        pytest.param(_frame(window=TimeWindow(start_year=2019)), False, id="open-end-from-2019"),
        pytest.param(_frame(window=TimeWindow(end_year=2017)), True, id="open-start-to-2017"),
        pytest.param(_frame(window=TimeWindow(end_year=2022)), False, id="open-start-to-2022"),
        pytest.param(_frame(window=TimeWindow(start_year=2022, end_year=2022)), False, id="2022"),
        pytest.param(_frame(latest=True), False, id="latest"),
        pytest.param(_frame(), True, id="no-window"),
    ],
)
def test_the_window_is_pre_grant_equivalent_when_it_starts_before_2018(
    frame: Frame, expected: bool
):
    assert (
        context_of(frame, [], [0], wanted_datasets=_WANTED, case_systems=_CASE_SYSTEMS).pre_ge
        is expected
    )


@pytest.mark.parametrize(
    ("places", "qualifiers", "expected"),
    [
        pytest.param({"Kenya": ()}, [[], [_purpose("purposeCode")]], "sector", id="sector-named"),
        pytest.param({"Kenya": ()}, [[], [_purpose("flowCode")]], "recipient", id="undirected"),
        pytest.param({"Kenya": ("destination",)}, [], "recipient", id="recipient"),
        pytest.param({"Kenya": ("origin", "destination")}, [], "recipient", id="both-axes"),
        pytest.param({"Japan": ("origin",), "Kenya": ()}, [], "recipient", id="donor-and-place"),
        pytest.param({"Japan": ("origin",)}, [], "donor", id="donor-only"),
        pytest.param({}, [], None, id="no-place"),
    ],
)
def test_the_part_names_the_kind_of_question_its_default_dataset_answers(
    places: Mapping[str, tuple[str, ...]],
    qualifiers: list[list[QualifierMatch]],
    expected: HouseDefaultCase | None,
):
    context = context_of(
        _frame(places=places),
        qualifiers,
        [0, 1],
        wanted_datasets=_WANTED,
        case_systems=_CASE_SYSTEMS,
    )

    assert context.dataset_case == expected
    assert context.wanted_dataset == (_WANTED[expected] if expected else None)


def test_a_packaged_purpose_code_makes_a_sector_question():
    index = get_index()
    context = context_of(
        _frame(places={"Kenya": ()}),
        [[_purpose("crsPurpose")]],
        [0],
        wanted_datasets=index.house_defaults(),
        case_systems=index.house_case_systems(),
    )

    assert (context.dataset_case, context.wanted_dataset) == (
        "sector",
        "provenance/OECDCreditorReportingSystem",
    )


def test_a_sector_admitted_outside_the_pool_names_no_sector():
    context = context_of(
        _frame(),
        [[], [_purpose("purposeCode")]],
        [0],
        wanted_datasets=_WANTED,
        case_systems=_CASE_SYSTEMS,
    )

    assert context.dataset_case is None


def _audit_cases() -> list[dict]:
    return json.loads(_AUDIT_CASES.read_text())


@pytest.mark.parametrize(
    "case", _audit_cases(), ids=lambda case: f"{case['item']}-{case['policy']}"
)
def test_each_policy_keeps_the_shapes_the_label_audit_kept(case: dict):
    """Policy steps the label audit recorded on dev, corpus, planner and synthetic items, each
    with the facts of the shapes it saw and the shapes it kept. `data_at_places` keeps every
    shape with data at the named places, as the engine's policy does."""
    wanted = get_index().house_defaults()
    dataset_case = case["context"]["dataset_case"]
    context = Context(
        pre_ge=case["context"]["pre_ge"],
        dataset_case=dataset_case,
        wanted_dataset=wanted[dataset_case] if dataset_case else None,
    )
    facts = [
        ShapeFacts(
            **{
                **shape,
                "netting": tuple(shape["netting"]) if shape["netting"] is not None else None,
                "flow_types": (
                    tuple(shape["flow_types"]) if shape["flow_types"] is not None else None
                ),
                "datasets": tuple(shape["datasets"]),
            }
        )
        for shape in case["shapes"]
    ]
    policy: PolicyName = case["policy"]

    kept = apply_policy(policy, facts, context)

    assert [f.index for f in kept] == case["expected"]
