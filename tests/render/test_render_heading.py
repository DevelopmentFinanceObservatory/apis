"""Pins the heading: possessive order, flow-family role wording, measure nouns, the size
rule, open roles, and the how slots a heading leaves silent."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from search.engine.retrieval.bound_values import MeasurePartition
from search.engine.spec_build import measure_label_for_partition
from search.heading.words import content_words
from search.models import (
    BindingUnbound,
    BindingValue,
    BreadthDim,
    CoverageBreadth,
    DefinitePart,
    GraphRef,
    IdentityEntityGroup,
    OpenConstraintAxisFact,
    ResolvedFacet,
    Slot,
    SlotOption,
    SlotOptions,
    SlotValue,
    Spec,
    StatVar,
)
from search.render.candidates import candidates_sentence_plan
from search.render.envelope import compose_envelope_sentence
from search.render.heading import heading_for
from search.render.no_data import no_data_sentence_plan
from search.render.plan import render_sentence, render_sentence_plan
from search.render.tokens import heading_tokens, plan_words, sentence_from_plan
from tests.render._helpers import (
    _entity_from,
    _entity_to,
    _ref,
    _slot_key,
    _spec,
    _where_place_slot,
    assert_round_trip,
    recorded_spec,
)


def _how(prop: str, dcid: str, label: str) -> Slot:
    return Slot(
        key=_slot_key("how", prop, prop, prop),
        binding=BindingValue(value=SlotValue(ref=_ref(dcid, label), value_kind="enum_value")),
    )


def _members(*constraint_properties: str) -> list[StatVar]:
    return [
        StatVar(
            ref=_ref("sv1", "SV1"),
            shape_id="test-shape",
            slot_values=[],
            constraint_properties=sorted(constraint_properties),
        )
    ]


def _with(spec: Spec, **resolved: object) -> Spec:
    return spec.model_copy(update={"resolved": spec.resolved.model_copy(update=resolved)})


def _with_measure_label(spec: Spec) -> Spec:
    """mls-001's Spec with the measure label the engine gives its imputed partition."""
    label = measure_label_for_partition(MeasurePartition(values=_MLS_001_PARTITION, members=()))
    labelled = _with(spec, measure_label=label.text, measure_label_axes=label.axes)
    return labelled.model_copy(
        update={
            "resolved": labelled.resolved.stating_measure_label_named_values(label.named_values)
        }
    )


def _aid(*, entities, **kwargs) -> Spec:
    return _spec(
        population_type=_ref("FinancialFlow", "Financial flow"),
        measured_property=_ref("amount", "amount"),
        slots=[
            _how("crsTypeOfFlow", "CRSTypeOfFlow10", "official development assistance"),
            _how("financialFlowType", "Disbursement", "disbursement"),
            _how("nettingBasis", "Gross", "gross"),
        ],
        entities=entities,
        stat_vars=_members("crsTypeOfFlow", "financialFlowType", "nettingBasis"),
        measure_label="gross disbursements",
        measure_label_axes=["financialFlowType", "nettingBasis"],
        **kwargs,
    )


def _debt_stock(**kwargs) -> Spec:
    return _spec(
        population_type=_ref("FinancialPosition", "Financial position"),
        measured_property=_ref("amount", "amount"),
        entities=[_entity_to("country/KEN", "Kenya")],
        stat_vars=_members("financialPositionType"),
        open_axis_facts=[
            OpenConstraintAxisFact(
                property="financialPositionType",
                varies=False,
                value=_ref("AmountOutstanding", "amount outstanding"),
                distinct_count=1,
                undeclared_count=0,
            )
        ],
        **kwargs,
    )


def _heading_cases() -> list[object]:
    debt_service = _spec(
        population_type=_ref("FinancialFlow", "Financial flow"),
        measured_property=_ref("amount", "amount"),
        slots=[_how("financialFlowType", "ServicingPayment", "servicing payment")],
        entities=[_entity_to("country/GHA", "Ghana"), _entity_from("country/CHN", "China")],
        stat_vars=_members("financialFlowType"),
    )
    gdp_per_capita = _spec(
        population_type=_ref("EconomicProduction", "Economic production"),
        measured_property=_ref("valueAdded", "value added"),
        measurement_denominator=_ref("Count_Person", "Total population"),
        slots=[
            _how("currencyConversionBasis", "PurchasingPowerParityRate", "purchasing power parity"),
            _how("fixedCapitalConsumptionBasis", "GrossOfFixedCapitalConsumption", "gross"),
            _how("productTaxBasis", "IncludingProductTaxes", "including taxes on products"),
            _where_place_slot("country/JPN", "Japan"),
        ],
        stat_vars=_members(
            "currencyConversionBasis", "fixedCapitalConsumptionBasis", "productTaxBasis"
        ),
        facets={
            "unit": ResolvedFacet(
                basis="fallback", ref="PPPInternationalDollar", label="PPP international dollar"
            )
        },
    )
    return [
        pytest.param(
            _aid(
                entities=[
                    _entity_to("country/KEN", "Kenya"),
                    _entity_from("country/DEU", "Germany"),
                ]
            ),
            "Germany's gross ODA disbursements to Kenya.",
            id="aid",
        ),
        pytest.param(debt_service, "Ghana's external debt service owed to China.", id="debt"),
        pytest.param(
            gdp_per_capita, "Japan's GDP per capita in PPP international dollars.", id="weo"
        ),
    ]


@pytest.mark.parametrize(("spec", "expected"), _heading_cases())
def test_heading_reads_in_possessive_order_with_family_wording(spec: Spec, expected: str):
    """The possessor role leads, the noun names the defaults that change the number, and
    the counterpart takes the family's preposition. Every how slot the noun states keeps
    a silent token, and every token is in the heading."""
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    assert sentence_from_plan(plan) == expected
    assert all(t.segment == "heading" for t in plan)
    assert all(t.text == "" for t in plan if t.axis == "how")


def test_role_tokens_carry_the_role_refs():
    """Origin and destination tokens carry their role's binding_ref and entity refs."""
    spec = _aid(
        entities=[_entity_to("country/KEN", "Kenya"), _entity_from("country/DEU", "Germany")]
    )
    plan = render_sentence_plan(spec)
    roles = {t.role: (t.text, t.binding_ref, t.entity_refs) for t in plan if t.role}
    assert roles == {
        "origin": ("Germany", "origin:", ["country/DEU"]),
        "destination": ("Kenya", "where:", ["country/KEN"]),
    }


_DONORS = [
    _entity_from("country/AUT", "Austria"),
    _entity_from("country/BEL", "Belgium"),
    _entity_from("country/BGR", "Bulgaria"),
    _entity_from("country/CYP", "Cyprus"),
]


@pytest.mark.parametrize(
    ("donors", "groups", "expected"),
    [
        pytest.param(
            _DONORS[:3],
            None,
            "Gross ODA disbursements to Ethiopia from Austria, Belgium, and Bulgaria.",
            id="three-named",
        ),
        pytest.param(
            _DONORS,
            None,
            "Gross ODA disbursements to Ethiopia from 4 providers.",
            id="four-counted",
        ),
        pytest.param(
            _DONORS,
            {"origin": IdentityEntityGroup(label="EU member states")},
            "Gross ODA disbursements to Ethiopia from the 4 EU member states.",
            id="group",
        ),
        pytest.param(
            _DONORS,
            {"origin": IdentityEntityGroup(label="DAC donors")},
            "Gross ODA disbursements to Ethiopia from the 4 DAC donors.",
            id="group-named-by-its-members",
        ),
    ],
)
def test_a_role_names_at_most_three_entities(donors, groups, expected: str):
    """More than three entities read as a count with the family noun; a group reads by
    its size and name. Neither leads in possessive form."""
    spec = _aid(entities=[_entity_to("country/ETH", "Ethiopia"), *donors])
    spec = _with(spec, entity_groups=groups)
    plan = render_sentence_plan(spec)
    assert sentence_from_plan(plan) == expected
    origin = next(t for t in plan if t.role == "origin")
    assert origin.entity_refs == [e.ref.dcid for e in donors]


@pytest.mark.parametrize(
    ("member_count", "expected"),
    [
        pytest.param(162, "Germany's gross ODA disbursements to 162 recipients.", id="count"),
        pytest.param(1, "Germany's gross ODA disbursements.", id="one-unnamed-says-nothing"),
    ],
)
def test_an_open_role_reads_its_count(member_count: int, expected: str):
    """An open role counted by coverage is a role-only token. A count of one with no
    name says nothing, so it reads nothing, never "1 recipients"."""
    spec = _aid(
        entities=[_entity_from("country/DEU", "Germany")],
        coverage=CoverageBreadth(
            has_data=True, dimensions=[BreadthDim(label="recipients", count=member_count)]
        ),
    )
    plan = render_sentence_plan(spec)
    assert sentence_from_plan(plan) == expected
    tokens = [(t.text, t.binding_ref, t.entity_refs) for t in plan if t.role == "destination"]
    assert tokens == ([] if member_count == 1 else [("162 recipients", "where:", [])])


def test_an_open_role_with_no_count_reads_all_unless_a_how_value_narrows_it():
    """With a flow family and served members, an uncounted open role is the total over
    it. A spoken how value narrows that role, so "all" would overstate it."""
    plan = render_sentence_plan(_debt_stock())
    assert sentence_from_plan(plan) == "Kenya's outstanding external debt owed to all creditors."
    creditors = next(t for t in plan if t.role == "origin")
    assert (creditors.text, creditors.binding_ref) == ("all creditors", "origin:")

    narrowed = _debt_stock(slots=[_how("idsCreditorType", "Bondholders", "bondholders")])
    assert render_sentence(narrowed) == "Kenya's outstanding external debt owed to bondholders."


def _options(prop: str, *, values: int, truncated: bool) -> SlotOptions:
    return SlotOptions(
        key=_slot_key("how", prop, prop, prop),
        options=[
            SlotOption(
                value=SlotValue(ref=GraphRef(dcid=f"v{i}", label=f"v{i}"), value_kind="enum_value")
            )
            for i in range(values)
        ],
        truncated=truncated,
        source="observed",
    )


@pytest.mark.parametrize(
    ("values", "truncated"),
    [
        pytest.param(1, True, id="one-observed"),
        pytest.param(1, False, id="one-in-full-level"),
        pytest.param(3, True, id="several-observed"),
    ],
)
def test_a_bound_how_slot_speaks_whatever_its_observed_options_hold(values: int, truncated: bool):
    """A bound how value speaks beside an authored noun whatever its slot's options hold.
    The one-observed row is a resubmit's menu: the posted members carry only the bound
    value, as the first answer's recalled members need not."""
    spec = _aid(
        entities=[_entity_to("country/KEN", "Kenya"), _entity_from("country/DEU", "Germany")]
    )
    spec = _with(
        spec,
        slots=[*spec.resolved.slots, _how("crsPurpose", "CRSPurposeCode14030", "basic sanitation")],
        slot_options=[_options("crsPurpose", values=values, truncated=truncated)],
    )
    plan = render_sentence_plan(spec)
    assert sentence_from_plan(plan) == (
        "Germany's gross ODA disbursements to Kenya for basic sanitation."
    )
    purpose = next(t for t in plan if t.property_dcid == "crsPurpose")
    assert purpose.binding_ref == "how:crsPurpose"


def test_values_the_members_share_beside_an_authored_noun_are_silent():
    """Beside a measure label, an open axis whose members agree on one value ("standard
    grant") restricts nothing, so its slot stays silent; a sibling-distinguishing
    property still renders it."""
    spec = _aid(
        entities=[_entity_to("country/MWI", "Malawi"), _entity_from("org/GF", "Global Fund")]
    )
    spec = _with(
        spec,
        slots=[
            *spec.resolved.slots,
            Slot(
                key=_slot_key("how", "crsTypeOfFinance", "type of finance", "instrument"),
                binding=BindingUnbound(),
            ),
        ],
        open_axis_facts=[
            OpenConstraintAxisFact(
                property="crsTypeOfFinance",
                varies=False,
                value=_ref("CRSTypeOfFinance110", "standard grant"),
                distinct_count=1,
                undeclared_count=0,
            )
        ],
    )
    plan = render_sentence_plan(spec)
    assert sentence_from_plan(plan) == "Global Fund's gross ODA disbursements to Malawi."
    instrument = next(t for t in plan if t.property_dcid == "crsTypeOfFinance")
    assert (instrument.text, instrument.binding_ref) == ("", "how:crsTypeOfFinance")

    kept = render_sentence_plan(spec, keep_how_properties=frozenset({"crsTypeOfFinance"}))
    assert sentence_from_plan(kept) == (
        "Global Fund's gross ODA disbursements to Malawi as standard grants."
    )


def test_mls_001_names_the_imputed_multilateral_measure_once():
    """mls-001, "Imputed multilateral ODA from Norway", recorded on dev: the code index's
    label for the imputed partition is the noun, so neither the flow type nor the aid
    type it names repeats as a how phrase."""
    recorded = recorded_spec(json.loads(Path(__file__).with_name("mls_001_spec.json").read_text()))
    spec = _with_measure_label(recorded)
    plan = render_sentence_plan(spec)
    heading = sentence_from_plan([t for t in plan if t.segment == "heading"])
    assert heading == "Norway's imputed multilateral ODA to 175 recipients, 2011 to 2024."
    silent = {t.binding_ref for t in plan if t.axis == "how" and not t.text}
    assert {"how:crsTypeOfFlow", "how:dac2aAidType"} <= silent
    noun = heading_for(spec).heading.noun
    assert noun.named_axes == ("dac2aAidType",)
    assert {"crsTypeOfFlow", "dac2aAidType"} <= noun.consumed_axes


def test_every_sentence_surface_leaves_out_a_slot_the_five_tuple_pins():
    """mls-001 as recorded: its measure label names neither pinned slot (`crsTypeOfFlow`,
    `dac2aAidType` on the imputed five-tuple), so only the pins keep "for imputed
    multilateral ODA" out of the Spec's sentence, the envelope's, a candidates reading's
    and a no-data claim."""
    spec = recorded_spec(json.loads(Path(__file__).with_name("mls_001_spec.json").read_text()))
    assert not spec.resolved.measure_label_named_values
    sentence = render_sentence_plan(spec)
    part = DefinitePart(
        status="definite",
        warnings=[],
        spec=spec.model_copy(
            update={"resolved": spec.resolved.model_copy(update={"sentence_plan": sentence})}
        ),
    )
    envelope = compose_envelope_sentence([part])
    assert envelope is not None
    surfaces = {
        "spec": sentence_from_plan(sentence),
        "envelope": envelope[0],
        "candidates": sentence_from_plan(candidates_sentence_plan([spec, spec])),
        "no_data": sentence_from_plan(
            no_data_sentence_plan(
                "no_observations", gate="no_member_has_data_at_place", flagged_match=spec
            )
        ),
    }
    assert {
        name: "imputed multilateral ODA" in text for name, text in surfaces.items()
    } == dict.fromkeys(surfaces, False)


def test_heading_tokens_leave_out_the_period_and_take_claim_form():
    """mls-001's heading without its period, closed or in claim form: a lead keeps its
    case in claim form, and a noun that opens the heading is lowered there."""
    recorded = recorded_spec(json.loads(Path(__file__).with_name("mls_001_spec.json").read_text()))
    plan = heading_for(_with_measure_label(recorded))
    heading = plan.heading
    assert sentence_from_plan(heading_tokens(heading, period=False)) == (
        "Norway's imputed multilateral ODA to 175 recipients."
    )
    assert sentence_from_plan(heading_tokens(heading, period=False, claim=True)) == (
        "Norway's imputed multilateral ODA to 175 recipients"
    )
    unled = dataclasses.replace(
        heading, lead=None, noun=dataclasses.replace(heading.noun, text="Gross disbursements")
    )
    assert sentence_from_plan(heading_tokens(unled, claim=True)) == (
        "gross disbursements to 175 recipients, 2011 to 2024"
    )
    assert {"norway's", "imputed", "oda", "recipients", "2024", "sectors"} <= plan_words(plan)
    assert "sectors" not in content_words(sentence_from_plan(heading_tokens(heading)))


# The measure partition bind elects for mls-001.
_MLS_001_PARTITION = (
    ("populationType", "FinancialFlow"),
    ("measuredProperty", "attributedAmount"),
    ("measurementQualifier", "Nominal"),
    ("financialFlowType", "Disbursement"),
    ("financialObligationType", None),
    ("nettingBasis", "Gross"),
)
