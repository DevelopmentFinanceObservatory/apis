"""The measure-basis axis: a part is counted on the basis the declared conventions give its shape
and window, whatever its words call the measure and whichever menu shape the binder picks;
an explicit basis wins and elects the dataset that publishes it; the axis's options are the
bases the part's dataset publishes; and a switch through dataset_resubmit moves the basis and
back.

The parts resolve through the parsed door over a fixture graph holding OECD DAC1's donor
totals of ODA (gross, net and grant equivalents), DAC2a's ODA receipts (gross and net) and CRS
health members under CRS 120 and 130 (gross, net and grant equivalents of flows 10 and 60),
each with a series from the United States, to Kenya where the table has recipients. The
binder picks one menu shape and nominates every other, so the house policies, not the pick,
elect the basis.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import pytest

from code_index.scope_conventions import BASIS_MEASURES
from code_index.shared import get_index
from search.engine.departures import served_datasets
from search.engine.quantity_read import QuantityOutput
from search.engine.spec_id import identity_spec_id
from search.engine.stated_words_read import StatedWordsOutput
from search.models import (
    BaseIdentity,
    ChoiceOrigin,
    DatasetResubmitInput,
    DefinitePart,
    Entry,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityShapeSchema,
    IdentityWindowRange,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    Resolution,
    ResolveRequest,
    ResubmitIdentityEntry,
    Spec,
)
from search.scope.basis import basis_binding, basis_label, identity_basis_record, window_class
from search.scope.edits import apply_binding
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.engine._quantity import NO_QUANTITY, NO_RATING
from tests.fixtures import FakeGraph

_USA = "country/USA"
_KEN = "country/KEN"
_DAC1 = "provenance/OECDDAC1"
_DAC2A = "provenance/OECDDAC2A"
_CRS = "provenance/OECDCreditorReportingSystem"
_HEADLINE = "provenance/ONEOECDHeadlineODA"
_ODA = "CRSTypeOfFlow10"

_DAC1_MEMBERS = (
    "Amount_FinancialFlow_Gross_Disbursement_DAC1AidType1010_Nominal",
    "Amount_FinancialFlow_Net_Disbursement_DAC1AidType1010_Nominal",
    "GrantEquivalent_FinancialFlow_Disbursement_DAC1AidType1010_Nominal",
)
_DAC2A_MEMBERS = (
    "Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal",
    "Amount_FinancialFlow_Net_Disbursement_DAC2aAidType206_Nominal",
)


def _crs_member(
    code: str, *, measured_property: str, netting: str | None, flow: str, modality: str
) -> str:
    """A CRS member measuring `measured_property` at `netting` of type of flow `flow` and
    co-operation modality `modality`, whose purpose is `code` or a code below it."""
    index = get_index()
    return next(
        row.dcid
        for row in index.statvars
        if (row.population_type, row.measured_property, row.measurement_qualifier)
        == ("FinancialFlow", measured_property, "Nominal")
        and row.constraint_properties.get("nettingBasis") == netting
        and row.constraint_properties.get("crsTypeOfFlow") == flow
        and row.constraint_properties.get("crsCooperationModality") == modality
        and row.constraint_properties.get("crsPurpose") in index.descendants(code)
    )


# Gross and net disbursements of budget support (A01) and grant equivalents of project-type
# interventions (C01), the modalities the fixture index holds both health codes under; the
# private sector instruments of flow 60 only under CRS 120.
_CRS_MEMBERS = (
    *(
        _crs_member(
            code,
            measured_property="amount",
            netting=netting,
            flow=_ODA,
            modality="CRSCooperationModalityA01",
        )
        for code in ("CRSPurposeCode120", "CRSPurposeCode130")
        for netting in ("Gross", "Net")
    ),
    *(
        _crs_member(
            code,
            measured_property="grantEquivalent",
            netting=None,
            flow=flow,
            modality="CRSCooperationModalityC01",
        )
        for code, flow in (
            ("CRSPurposeCode120", _ODA),
            ("CRSPurposeCode130", _ODA),
            ("CRSPurposeCode120", "CRSTypeOfFlow60"),
        )
    ),
)
_YEARS = list(range(2005, 2025))


def _nodes(*dcids: str) -> dict:
    return {"nodes": [{"dcid": dcid} for dcid in dcids]}


def _member_node(dcid: str) -> dict:
    row = get_index().statvar_by_dcid(dcid)
    assert row is not None
    return {
        "label": dcid,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": _nodes("StatisticalVariable"),
            "name": {"nodes": [{"value": dcid}]},
            "populationType": _nodes(row.population_type),
            "measuredProperty": _nodes(row.measured_property),
            "statType": _nodes(row.stat_type),
            "measurementQualifier": _nodes(row.measurement_qualifier),
            "observationProperties": _nodes(*row.observation_properties),
            "constraintProperties": _nodes(*row.constraint_properties),
            **{p: _nodes(v) for p, v in row.constraint_properties.items()},
        },
    }


def _graph(variable: str, members: tuple[str, ...]) -> FakeGraph:
    """The United States, Kenya and `members`, which recall returns for `variable`."""
    return FakeGraph(
        nodes={
            _USA: {"label": "United States", "type": "Country"},
            _KEN: {"label": "Kenya", "type": "Country"},
            **{dcid: _member_node(dcid) for dcid in members},
        },
        detect={variable: {"svs": list(members), "cosine_scores": [0.9] * len(members)}},
        resolve={"United States": _USA, "Kenya": _KEN},
    )


_MENU_SIZE = re.compile(r"CANDIDATE SHAPES \((\d+) total\)")


class _PickingLLM:
    """Picks menu shape `pick` (the last when the menu holds fewer) and nominates every other
    shape, so the whole menu is the house policies' pool; reads no quantity and rates no
    word."""

    def __init__(self, pick: int = 0) -> None:
        self.pick = pick

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        if schema is StatedWordsOutput:
            return NO_RATING
        match = _MENU_SIZE.search(prompt)
        assert match is not None, prompt[:400]
        size = int(match.group(1))
        pick = min(self.pick, size - 1)
        return elected_reply(pick, also_plausible=[i for i in range(size) if i != pick]), None


@pytest.fixture(autouse=True)
def _coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    _cover(monkeypatch, _DAC1_MEMBERS)


def _cover(monkeypatch: pytest.MonkeyPatch, dac1: tuple[str, ...]) -> None:
    """Series for the United States over 2005 to 2024 in every DAC2a and CRS member and in
    the DAC1 members `dac1`."""
    donor_only = RoleNames(donor="financingProvider", recipient=None)
    flow = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            *((dcid, "dac1", _USA, ABSENT_ENTITY, _YEARS) for dcid in dac1),
            *((dcid, "dac2a", _USA, _KEN, _YEARS) for dcid in _DAC2A_MEMBERS),
            *((dcid, "crs", _USA, _KEN, _YEARS) for dcid in _CRS_MEMBERS),
        ],
        role_names={
            **dict.fromkeys(_DAC1_MEMBERS, donor_only),
            **dict.fromkeys((*_DAC2A_MEMBERS, *_CRS_MEMBERS), flow),
        },
        facet_metadata={
            "dac1": FacetMetadata(provenance_id=_DAC1, unit="USD"),
            "dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD"),
            "crs": FacetMetadata(provenance_id=_CRS, unit="USD"),
        },
    )


_ALL_MEMBERS = (*_DAC1_MEMBERS, *_DAC2A_MEMBERS, *_CRS_MEMBERS)


def _resolve(
    request: ResolveRequest,
    variable: str,
    *,
    pick: int = 0,
    members: tuple[str, ...] = _ALL_MEMBERS,
) -> Spec:
    response = offline_resolve(
        request,
        graph=_graph(variable, members),
        llm=_PickingLLM(pick),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    return part.spec


def _ask(
    variable: str,
    *,
    time: str,
    recipient: bool,
    pick: int = 0,
    message: str | None = None,
    carried: list[Entry] | None = None,
    members: tuple[str, ...] = _ALL_MEMBERS,
) -> Spec:
    entities = [ParsedEntity(text="United States", role_text="from")]
    if recipient:
        entities.append(ParsedEntity(text="Kenya", role_text="to"))
    return _resolve(
        ResolveRequest(
            input=ParsedInput(
                parts=[ParsedPart(variable_text=variable, entities=entities, time_text=time)],
                user_message=message,
                carried_entries=carried,
            )
        ),
        variable,
        pick=pick,
        members=members,
    )


class _Basis(NamedTuple):
    measured_property: str
    netting: str | None
    dataset: str


def _served_basis(spec: Spec) -> _Basis:
    """The measured property, netting basis and dataset of the spec's identity."""
    identity = spec.definition.identity
    assert isinstance(identity.shape, IdentityShapeSchema)
    netting = next(
        (
            slot.binding.ref
            for slot in identity.slots
            if slot.property == "nettingBasis" and isinstance(slot.binding, IdentityBindingValue)
        ),
        None,
    )
    (dataset,) = served_datasets(spec)
    assert identity.shape.measured_property is not None
    return _Basis(identity.shape.measured_property, netting, dataset)


def _basis_choice(spec: Spec) -> tuple[str, str, str | None]:
    """The spec's measure-basis choice: its active option, provenance and the text of the
    entry that settled it, None for a default."""
    scope = spec.resolved.scope_choices
    assert scope is not None
    (choice,) = (c for c in scope.choices if c.axis == "measure_basis")
    entry = next((e for e in scope.entries if e.axis == "measure_basis"), None)
    text = getattr(entry.origin, "text", None) if entry is not None else None
    return choice.active, choice.provenance, text


_GE_DAC1 = _Basis("grantEquivalent", None, _DAC1)
_NET_DAC1 = _Basis("amount", "Net", _DAC1)
_NET_DAC2A = _Basis("amount", "Net", _DAC2A)
_GROSS_CRS = _Basis("amount", "Gross", _CRS)


class _Cell(NamedTuple):
    variable: str
    time: str
    recipient: bool
    basis: _Basis
    option: str


_CELLS = {
    "donor-only-2019": _Cell("aid", "2019", False, _GE_DAC1, "measure_basis:grant_equivalent"),
    "donor-only-2016": _Cell("aid", "2016", False, _NET_DAC1, "measure_basis:net"),
    "donor-only-2010-2024": _Cell(
        "aid", "2010-2024", False, _NET_DAC1, "measure_basis:oecd_headline"
    ),
    "recipient-all-sectors": _Cell("aid", "2019", True, _NET_DAC2A, "measure_basis:net"),
    "sector": _Cell("health aid", "2019", True, _GROSS_CRS, "measure_basis:gross"),
}


@pytest.mark.parametrize("cell", list(_CELLS))
def test_the_conventions_elect_the_basis_by_shape_and_window(cell: str):
    row = _CELLS[cell]

    spec = _ask(row.variable, time=row.time, recipient=row.recipient)

    assert _served_basis(spec) == row.basis
    assert _basis_choice(spec) == (row.option, "default", None)


# The phrasings of one measure that moved the basis before the conventions decided it.
_DONOR_WORDINGS = ("aid", "ODA", "official development assistance", "assistance", "total aid")
_SECTOR_WORDINGS = (
    "health aid",
    "ODA for health",
    "health ODA",
    "health assistance",
    "health disbursements",
    "total health aid",
)


@pytest.mark.parametrize(
    ("cell", "wordings"),
    [
        pytest.param("donor-only-2019", _DONOR_WORDINGS, id="donor-only-2019"),
        pytest.param("recipient-all-sectors", _DONOR_WORDINGS, id="recipient-all-sectors"),
        pytest.param("sector", _SECTOR_WORDINGS, id="sector"),
    ],
)
def test_neither_the_wording_nor_the_binders_pick_moves_the_basis(
    cell: str, wordings: tuple[str, ...], search_settings
):
    row = _CELLS[cell]
    # The scripted binder rates no word, so a word no code reads ("assistance") is only
    # logged rather than refused.
    search_settings(qualifier_coverage="log")

    served = {
        (variable, pick): _served_basis(
            _ask(variable, time=row.time, recipient=row.recipient, pick=pick)
        )
        for variable in wordings
        for pick in (0, 1, 2)
    }

    assert set(served.values()) == {row.basis}, served


@pytest.mark.parametrize("variable", ["Total ODA", "ODA", "official development assistance"])
def test_a_donor_total_recalling_only_amounts_is_counted_in_grant_equivalents(
    variable: str, search_settings
):
    """Recall finds DAC1's ODA amounts and none of its grant equivalents for these words;
    recall adds the grant-equivalent family (`clients.graph.with_rival_families`), so the
    convention still elects the headline basis for a window from 2018."""
    search_settings(qualifier_coverage="log")
    amounts = tuple(m for m in _DAC1_MEMBERS if m.startswith("Amount_"))
    graph = FakeGraph(
        nodes={
            _USA: {"label": "United States", "type": "Country"},
            **{dcid: _member_node(dcid) for dcid in _DAC1_MEMBERS},
        },
        detect={variable: {"svs": list(amounts), "cosine_scores": [0.9] * len(amounts)}},
        resolve={"United States": _USA},
    )
    response = offline_resolve(
        ResolveRequest(
            input=ParsedInput(
                parts=[
                    ParsedPart(
                        variable_text=variable,
                        entities=[ParsedEntity(text="United States", role_text="from")],
                        time_text="2019",
                    )
                ]
            )
        ),
        graph=graph,
        llm=_PickingLLM(0),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part

    assert _served_basis(part.spec) == _GE_DAC1
    assert _basis_choice(part.spec) == ("measure_basis:grant_equivalent", "default", None)


@pytest.mark.parametrize(
    ("cell", "message", "basis", "option", "text"),
    [
        pytest.param(
            "donor-only-2019",
            "gross ODA from the United States in 2019",
            _Basis("amount", "Gross", _DAC1),
            "measure_basis:gross",
            "gross",
            id="gross-over-grant-equivalents",
        ),
        pytest.param(
            "donor-only-2019",
            "net ODA from the United States in 2019",
            _NET_DAC1,
            "measure_basis:net",
            "net",
            id="net-over-grant-equivalents",
        ),
        pytest.param(
            "donor-only-2016",
            "ODA grant equivalents from the United States in 2016",
            _Basis("grantEquivalent", None, _DAC1),
            "measure_basis:grant_equivalent",
            "grant equivalents",
            id="grant-equivalents-over-net",
        ),
        pytest.param(
            "sector",
            "net health aid from the United States to Kenya in 2019",
            _Basis("amount", "Net", _CRS),
            "measure_basis:net",
            "net",
            id="net-over-gross",
        ),
    ],
)
def test_an_explicit_basis_wins_over_the_convention(
    cell: str, message: str, basis: _Basis, option: str, text: str
):
    row = _CELLS[cell]

    spec = _ask(row.variable, time=row.time, recipient=row.recipient, message=message)

    assert _served_basis(spec) == basis
    assert _basis_choice(spec) == (option, "explicit", text)


def test_a_stated_grant_equivalent_for_a_recipient_elects_the_crs_that_publishes_it():
    """DAC2a, the recipient convention's dataset, publishes no grant equivalents, so the
    stated basis elects the CRS's, ODA's flows 10 and 60 together."""
    spec = _ask(
        "aid",
        time="2019",
        recipient=True,
        message="grant equivalent aid from the United States to Kenya in 2019",
    )

    assert _served_basis(spec) == _Basis("grantEquivalent", None, _CRS)
    assert _basis_choice(spec) == (
        "measure_basis:grant_equivalent",
        "explicit",
        "grant equivalent",
    )
    (flow,) = (s for s in spec.definition.identity.slots if s.property == "crsTypeOfFlow")
    assert flow.binding == IdentityBindingSet(refs=["CRSTypeOfFlow10", "CRSTypeOfFlow60"])


def test_a_carried_basis_wins_over_the_convention():
    carried = Entry(
        axis="measure_basis",
        option_id="measure_basis:gross",
        provenance="carried",
        origin=ChoiceOrigin(choice_id="earlier:measure_basis"),
    )

    spec = _ask("aid", time="2019", recipient=True, carried=[carried])

    assert _served_basis(spec) == _Basis("amount", "Gross", _DAC2A)
    assert _basis_choice(spec) == ("measure_basis:gross", "carried", None)


def test_the_options_are_the_bases_the_parts_dataset_publishes():
    scope = _ask("aid", time="2019", recipient=True).resolved.scope_choices
    assert scope is not None
    (choice,) = (c for c in scope.choices if c.axis == "measure_basis")

    assert [
        (o.option_id, o.label, o.availability.status, o.availability.code) for o in choice.options
    ] == [
        ("measure_basis:gross", "gross disbursements, current USD", "available", None),
        ("measure_basis:net", "net disbursements, current USD", "available", None),
        (
            "measure_basis:grant_equivalent",
            "grant equivalents (OECD headline), current USD",
            "unavailable",
            "basis_not_published",
        ),
        (
            "measure_basis:commitments",
            "commitments, current USD",
            "unavailable",
            "basis_not_published",
        ),
    ]
    assert choice.options[2].availability.detail == {"dataset": _DAC2A}


def _net_identity(qualifier: str | None, unit_request: dict | None) -> BaseIdentity:
    shape = {
        "kind": "statvar_schema",
        "population_type": "FinancialFlow",
        "measured_property": "amount",
        "stat_type": "measuredValue",
    }
    if qualifier is not None:
        shape["measurement_qualifier"] = qualifier
    document: dict = {
        "identity_version": "spec1",
        "shape": shape,
        "slots": [{"property": "nettingBasis", "binding": {"state": "value", "ref": "Net"}}],
        "entities": {"origin": [_USA]},
        "requested_window": {"kind": "range", "start_year": 2019, "end_year": 2019},
    }
    if unit_request is not None:
        document["facets"] = {"unit": {"state": "value", "request": unit_request}}
    return BaseIdentity.model_validate(document)


@pytest.mark.parametrize(
    ("qualifier", "unit_request", "expected"),
    [
        pytest.param("Nominal", None, "net disbursements, current USD", id="qualifier"),
        pytest.param(None, None, "net disbursements", id="no-price-basis"),
        pytest.param(
            "Nominal",
            {"currency": "USD", "prices": "constant", "base_year": 2024},
            "net disbursements, constant 2024 USD",
            id="constant-request",
        ),
        pytest.param(
            "Nominal",
            {"currency": "EUR", "prices": "current"},
            "net disbursements, current EUR",
            id="currency-request",
        ),
        pytest.param(
            "Nominal",
            {"currency": {"role": "origin"}, "prices": "constant", "base_year": 2023},
            "net disbursements, constant 2023 donor currency",
            id="role-currency-request",
        ),
    ],
)
def test_a_basis_label_states_the_prices_and_currency_the_part_is_served_in(
    qualifier: str | None, unit_request: dict | None, expected: str
):
    """A unit request converts the part, so its prices and currency name every basis option
    over the shape's own price basis."""
    net = next(row for row in BASIS_MEASURES if row.basis == "net")

    assert basis_label(net, _net_identity(qualifier, unit_request)) == expected


@pytest.mark.parametrize("cell", ["sector", "donor-only-2016"])
def test_a_convention_settled_part_offers_no_basis_sibling_as_an_alternative_measure(cell: str):
    """The netting basis and the measured property belong to the scope choice, so no
    alternative measure differs from the answer on either."""
    row = _CELLS[cell]

    spec = _ask(row.variable, time=row.time, recipient=row.recipient)

    for alternative in spec.resolved.alternative_measures or []:
        assert _served_basis_of(alternative.definition.identity) == (
            row.basis.measured_property,
            row.basis.netting,
        )


def _served_basis_of(identity: BaseIdentity) -> tuple[str | None, str | None]:
    assert isinstance(identity.shape, IdentityShapeSchema)
    netting = next(
        (
            slot.binding.ref
            for slot in identity.slots
            if slot.property == "nettingBasis" and isinstance(slot.binding, IdentityBindingValue)
        ),
        None,
    )
    return identity.shape.measured_property, netting


def _switch(
    identity: BaseIdentity,
    option_id: str,
    choice_id: str,
    *,
    members: tuple[str, ...] = _ALL_MEMBERS,
) -> Spec:
    entry = Entry(
        axis="measure_basis",
        option_id=option_id,
        provenance="explicit",
        origin=ChoiceOrigin(choice_id=choice_id),
    )
    return _resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(
                        variable_text="aid", identity=identity, scope_entries=[entry]
                    )
                ]
            )
        ),
        "aid",
        members=members,
    )


@pytest.mark.parametrize(
    ("cell", "option", "switched"),
    [
        pytest.param(
            "donor-only-2019", "measure_basis:net", _NET_DAC1, id="grant-equivalents-to-net"
        ),
        pytest.param(
            "sector",
            "measure_basis:grant_equivalent",
            _Basis("grantEquivalent", None, _CRS),
            id="gross-to-grant-equivalents",
        ),
    ],
)
def test_a_switch_moves_the_basis_and_back_to_the_first_spec_id(
    cell: str, option: str, switched: _Basis
):
    row = _CELLS[cell]
    first = _ask(row.variable, time=row.time, recipient=row.recipient)
    scope = first.resolved.scope_choices
    assert scope is not None
    (choice,) = (c for c in scope.choices if c.axis == "measure_basis")

    moved = _switch(scope.base.identity, option, choice.id)
    restored = _switch(scope.base.identity, row.option, choice.id)

    assert _served_basis(moved) == switched
    assert _basis_choice(moved)[:2] == (option, "explicit")
    assert moved.resolved.scope_choices is not None
    assert moved.resolved.scope_choices.base.spec_id == first.spec_id
    assert restored.spec_id == first.spec_id


def test_a_donor_window_crossing_2018_is_counted_on_the_oecd_headline_per_year():
    """2015 to 2020 for a donor: net flows to 2017 and grant equivalents from 2018, one
    composite of two disjoint periods, and a switch to net removes the composite."""
    spec = _ask("aid", time="2015-2020", recipient=False)
    scope = spec.resolved.scope_choices
    assert scope is not None

    assert _basis_choice(spec) == ("measure_basis:oecd_headline", "default", None)
    binding = (spec.definition.identity.facets or {})["provenance"]
    assert binding == IdentityBindingValue(ref=_HEADLINE)
    assert spec.resolved.facets["provenance"].ref == _HEADLINE
    assert [(c.role, c.provenance, c.periods) for c in scope.compiled.components] == [
        ("headline_net", _DAC1, [(2015, 2017)]),
        ("headline_ge", _DAC1, [(2018, 2020)]),
    ]
    measures = [
        (c.identity.shape.measured_property, c.identity.requested_window.model_dump())
        for c in scope.compiled.components
        if isinstance(c.identity.shape, IdentityShapeSchema)
    ]
    assert measures == [
        ("amount", {"kind": "range", "start_year": 2015, "end_year": 2017}),
        ("grantEquivalent", {"kind": "range", "start_year": 2018, "end_year": 2020}),
    ]
    (choice,) = (c for c in scope.choices if c.axis == "measure_basis")

    net = _switch(scope.base.identity, "measure_basis:net", choice.id)

    assert "provenance" not in (net.definition.identity.facets or {})
    assert _served_basis(net) == _NET_DAC1
    back = _switch(scope.base.identity, "measure_basis:oecd_headline", choice.id)
    assert back.spec_id == spec.spec_id


def test_a_donor_with_no_grant_equivalent_series_keeps_net_flows(monkeypatch):
    """The headline splices grant equivalents from 2018, so a donor whose DAC1 rows hold no
    grant-equivalent series is served net flows over 2015 to 2020 under the rule that says
    the served basis differs from the convention, and the headline is offered unavailable
    with the basis it lacks."""
    _cover(monkeypatch, tuple(m for m in _DAC1_MEMBERS if not m.startswith("GrantEquivalent")))

    spec = _ask("aid", time="2015-2020", recipient=False)

    scope = spec.resolved.scope_choices
    assert scope is not None
    (choice,) = (c for c in scope.choices if c.axis == "measure_basis")
    assert (choice.active, choice.rule) == ("measure_basis:net", "served_differs_from_convention")
    (headline,) = (o for o in choice.options if o.option_id == "measure_basis:oecd_headline")
    assert headline.availability.status == "unavailable"
    assert headline.availability.detail["basis"] == "grant_equivalent"
    assert "provenance" not in (spec.definition.identity.facets or {})
    assert _served_basis(spec) == _NET_DAC1
    assert scope.compiled.spec_id == spec.spec_id == scope.base.spec_id


def test_a_stated_bilateral_on_a_donor_total_with_no_provider_axis_is_reported():
    """A donor's DAC1 total has no provider axis, so a typed "bilateral" is reported as not
    applied."""
    spec = _ask("aid", time="2019", recipient=False, message="bilateral aid from the US in 2019")
    scope = spec.resolved.scope_choices
    assert scope is not None

    assert all(choice.axis != "provider" for choice in scope.choices)
    (dropped,) = (d for d in scope.dropped_entries if d.entry.axis == "provider")
    assert (dropped.entry.option_id, dropped.code, dropped.cause) == (
        "provider:bilateral",
        "stated_choice_not_applied",
        "axis_absent",
    )


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        pytest.param("latest", "from_2018", id="latest"),
        pytest.param("all", "crossing_2018", id="every-year"),
    ],
)
def test_the_window_class_of_a_window_with_no_years(window, expected):
    assert window_class(window) == expected


def test_a_commitments_edit_moves_the_population_and_flow_type_and_back():
    """The commitments option sets the obligation population and its commitment flow type in
    place of the disbursed flow's netting basis and flow type, and gross disbursements set
    them back."""
    base = _ask("health aid", time="2019", recipient=True).definition.identity
    bindings = [basis_binding(row) for row in BASIS_MEASURES]
    commitments, gross = (
        basis_binding(next(row for row in BASIS_MEASURES if row.basis == basis))
        for basis in ("commitments", "gross")
    )

    committed = apply_binding(base, commitments, bindings)
    restored = apply_binding(committed, gross, bindings)

    assert isinstance(committed.shape, IdentityShapeSchema)
    assert committed.shape.population_type == "FinancialObligation"
    assert {s.property: s.binding for s in committed.slots}.get("financialObligationType") == (
        IdentityBindingValue(ref="Commitment")
    )
    assert not {"nettingBasis", "financialFlowType"} & {s.property for s in committed.slots}
    assert identity_spec_id(restored) == identity_spec_id(base)


def test_a_stated_basis_the_dataset_does_not_publish_is_reported_as_not_applied():
    """No series of the menu commits, so a recipient's stated commitments leave the answer
    on the convention's net receipts and the entry is reported as not applied."""
    spec = _ask(
        "aid", time="2019", recipient=True, message="aid commitments from the US to Kenya in 2019"
    )

    assert _served_basis(spec) == _NET_DAC2A
    scope = spec.resolved.scope_choices
    assert scope is not None
    assert [(d.entry.option_id, d.code, d.cause) for d in scope.dropped_entries] == [
        ("measure_basis:commitments", "stated_choice_not_applied", "basis_not_published")
    ]
    assert _basis_choice(spec) == ("measure_basis:net", "default", None)


_CRS_GROSS_AND_NET = _CRS_MEMBERS[:4]
_CRS_GRANT_EQUIVALENTS = _CRS_MEMBERS[4:]


def _flows(spec: Spec) -> object:
    (flow,) = (s for s in spec.definition.identity.slots if s.property == "crsTypeOfFlow")
    return flow.binding


def test_a_part_served_on_grant_equivalents_unelected_binds_what_a_switch_binds():
    """A pool holding only the CRS's grant equivalents serves a health part, whose
    convention is gross disbursements, on grant equivalents. The answer reads ODA as flows
    10 and 60, as a switch to grant equivalents does, so gross and back, twice, returns the
    same two spec_ids; its choice states that the convention did not settle the basis, and
    so does a resubmit of the answer with no entry. A switched answer's choice names the
    convention's rule, since its entry settled the basis."""
    first = _ask("health aid", time="2019", recipient=True, members=_CRS_GRANT_EQUIVALENTS)
    resubmitted = _resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(
                        variable_text="aid", identity=first.definition.identity, scope_entries=[]
                    )
                ]
            )
        ),
        "aid",
        members=_CRS_MEMBERS,
    )
    rules = []
    for spec in (first, resubmitted):
        scope = spec.resolved.scope_choices
        assert scope is not None
        (choice,) = (c for c in scope.choices if c.axis == "measure_basis")
        rules.append((choice.active, choice.provenance, choice.rule))
    assert (
        rules
        == [("measure_basis:grant_equivalent", "default", "served_differs_from_convention")] * 2
    )
    assert _flows(first) == IdentityBindingSet(refs=["CRSTypeOfFlow10", "CRSTypeOfFlow60"])

    ids = []
    identity = first.definition.identity
    for option in ("measure_basis:gross", "measure_basis:grant_equivalent") * 2:
        spec = _switch(identity, option, choice.id, members=_CRS_MEMBERS)
        ids.append(spec.spec_id)
        identity = spec.definition.identity
        assert spec.resolved.scope_choices is not None
        (switched,) = (c for c in spec.resolved.scope_choices.choices if c.axis == "measure_basis")
        assert switched.rule == "measure_basis_convention"

    assert ids[1] == ids[3] == first.spec_id
    assert ids[0] == ids[2] != first.spec_id


def test_a_stated_basis_the_pool_cannot_serve_is_reported_and_not_stated_as_served():
    """The CRS publishes covered grant equivalents for health, but the pool holds only gross
    and net disbursements, so the stated basis is served on the convention's gross: the
    choice states gross and the entry is reported as not applied."""
    spec = _ask(
        "health aid",
        time="2019",
        recipient=True,
        message="grant equivalent health aid from the US to Kenya in 2019",
        members=_CRS_GROSS_AND_NET,
    )

    assert _served_basis(spec) == _GROSS_CRS
    assert _basis_choice(spec) == ("measure_basis:gross", "default", None)
    scope = spec.resolved.scope_choices
    assert scope is not None
    assert [(d.entry.option_id, d.code, d.cause) for d in scope.dropped_entries] == [
        ("measure_basis:grant_equivalent", "stated_choice_not_applied", "not_served")
    ]


@pytest.mark.parametrize(
    ("window", "entries", "expected"),
    [
        pytest.param(2019, [], "served_differs_from_convention", id="net-after-2018"),
        pytest.param(2015, [], None, id="net-before-2018"),
        pytest.param(
            2019,
            [
                Entry(
                    axis="measure_basis",
                    option_id="measure_basis:net",
                    provenance="explicit",
                    origin=ChoiceOrigin(choice_id="earlier:measure_basis"),
                )
            ],
            None,
            id="net-asked-for",
        ),
    ],
)
def test_an_identity_reads_the_basis_rule_its_search_recorded(
    window: int, entries: list[Entry], expected: str | None
):
    """A merge or resubmit, where no binder runs, reads the rule from the identity: a donor's
    net flows in 2019, where the convention counts grant equivalents, were served on a basis
    the convention did not settle; before 2018, or under an entry asking for net, they were
    not."""
    identity = _net_identity("Nominal", None).model_copy(
        update={
            "requested_window": IdentityWindowRange(start_year=window, end_year=window),
        }
    )

    record = identity_basis_record(identity, get_index(), entries)

    assert (None if record is None else (record.rule, record.active)) == (
        None if expected is None else (expected, "measure_basis:net")
    )
