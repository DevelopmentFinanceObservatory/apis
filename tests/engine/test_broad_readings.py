"""The code index's broad readings (`bind_repair.apply_broad_readings`) and the debt stock's
slot default, through a resolve over the fixture index's World Bank IDS debt stocks.

"Publicly guaranteed" debt names `PubliclyGuaranteedClaim`, private-sector debt under a public
guarantee, and the words alone read as public and publicly guaranteed debt. A question naming
the private-sector debtor keeps the narrow reading. Debt named without a position type reads
the amount outstanding (`SHAPE_SLOT_DEFAULTS`), on the by-creditor shape too."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from code_index.shared import get_index
from code_index.types import BroadReading
from search.engine.bind import ShapeRating, SlotBindingDraft, _UniversalBindOutput
from search.engine.bind_repair import apply_broad_readings
from search.engine.core import resolve_async
from search.engine.quantity_read import QuantityOutput
from search.engine.region.codes import SLOT_DEFAULTED
from search.engine.retrieval.bound_values import served_members
from search.engine.shape import ShapeDraft, shape_draft_from
from search.models import ParsedEntity, ParsedInput, ParsedPart, Resolution, ResolveRequest
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_IDS = "provenance/WorldBankInternationalDebtStatistics"
_KEN = "country/KEN"
_CHN = "country/CHN"
_YEARS = [2020, 2021, 2022]
_PPG = "PublicOrPubliclyGuaranteed"
_CLAIM = "PubliclyGuaranteedClaim"
_PRIVATE = "IDSDebtorTypePrivateSector"
_GOVERNMENT = "IDSDebtorTypeGG"
_NOTE = (
    "Publicly guaranteed was read as public and publicly guaranteed debt. For private-sector "
    "debt under a public guarantee, ask for publicly guaranteed private-sector debt."
)
_RECIPIENT = ["financingRecipient"]
_PROVIDER_RECIPIENT = ["financingProvider", "financingRecipient"]
# The fixture index's IDS debt positions on the guarantee status, each with its by-creditor
# twin and the World Bank's name: PPG stock and undisbursed, the private sector's guaranteed
# and nonguaranteed stocks, and the general government's stock.
_STOCKS: dict[str, tuple[str, dict[str, str]]] = {
    "Amount_FinancialPosition_AmountOutstanding_PublicOrPubliclyGuaranteed": (
        "Outstanding public or publicly guaranteed external debt",
        {"financialPositionType": "AmountOutstanding", "guaranteeStatus": _PPG},
    ),
    "Amount_FinancialPosition_AmountUndisbursed_PublicOrPubliclyGuaranteed": (
        "Undisbursed public or publicly guaranteed external debt",
        {"financialPositionType": "AmountUndisbursed", "guaranteeStatus": _PPG},
    ),
    "Amount_FinancialPosition_AmountOutstanding_PubliclyGuaranteedClaim_IDSDebtorTypePrivateSector": (
        "Outstanding publicly guaranteed private sector external debt",
        {
            "financialPositionType": "AmountOutstanding",
            "guaranteeStatus": _CLAIM,
            "idsDebtorType": _PRIVATE,
        },
    ),
    "Amount_FinancialPosition_AmountOutstanding_NotPubliclyGuaranteed_IDSDebtorTypePrivateSector": (
        "Outstanding private nonguaranteed external debt",
        {
            "financialPositionType": "AmountOutstanding",
            "guaranteeStatus": "NotPubliclyGuaranteed",
            "idsDebtorType": _PRIVATE,
        },
    ),
    "Amount_FinancialPosition_AmountOutstanding_IDSDebtorTypeGG": (
        "Outstanding external debt owed by the general government",
        {"financialPositionType": "AmountOutstanding", "idsDebtorType": _GOVERNMENT},
    ),
}
_MEMBERS: dict[str, tuple[str, list[str], dict[str, str]]] = {
    **{f"{stem}_Nominal": (name, _RECIPIENT, values) for stem, (name, values) in _STOCKS.items()},
    **{
        f"{stem}_ByCreditor_Nominal": (f"{name}, by creditor", _PROVIDER_RECIPIENT, values)
        for stem, (name, values) in _STOCKS.items()
    },
}
_VALUE_LABELS = {
    _PPG: "public or publicly guaranteed",
    _CLAIM: "publicly guaranteed",
    "NotPubliclyGuaranteed": "private nonguaranteed",
    _PRIVATE: "private sector",
    _GOVERNMENT: "general government",
    "AmountOutstanding": "amount outstanding",
    "AmountUndisbursed": "amount undisbursed",
}


def _arcs(name: str, roles: list[str], constraints: dict[str, str]) -> dict:
    return {
        "name": {"nodes": [{"value": name}]},
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": "FinancialPosition"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraints]},
        "observationProperties": {"nodes": [{"dcid": r} for r in roles]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in constraints.items()},
    }


def _graph(recalled_as: str, *, named: bool) -> FakeGraph:
    """A graph whose embedding resolve recalls every debt position for `recalled_as`. Each
    series carries its World Bank name, or a placeholder holding no value's words where
    `named` is false."""
    facet = {"earliestDate": "2020", "latestDate": "2022", "obsCount": 3}
    return FakeGraph(
        nodes={
            **{
                dcid: {
                    "label": name if named else "debt position",
                    "arcs": _arcs(name if named else "debt position", roles, values),
                }
                for dcid, (name, roles, values) in _MEMBERS.items()
            },
            **{value: {"label": label} for value, label in _VALUE_LABELS.items()},
            _KEN: {"label": "Kenya", "type": "Country"},
            _CHN: {"label": "China", "type": "Country"},
        },
        obs={f"{sv}|{_KEN}": [facet] for sv in _MEMBERS},
        detect={recalled_as: {"svs": list(_MEMBERS)}},
        resolve={"Kenya": _KEN, "China": _CHN},
    )


_COVERAGE_ROWS = [
    (sv, f"f{i}", _CHN if roles == _PROVIDER_RECIPIENT else "", _KEN, _YEARS)
    for i, (sv, (_name, roles, _values)) in enumerate(_MEMBERS.items())
]
_ROLE_NAMES = {
    sv: RoleNames(
        donor="financingProvider" if roles == _PROVIDER_RECIPIENT else None,
        recipient="financingRecipient",
    )
    for sv, (_name, roles, _values) in _MEMBERS.items()
}


class _BindLLM:
    """Elects shape 0 with `bindings`, standing in for the bind model."""

    def __init__(self, bindings: Sequence[SlotBindingDraft]) -> None:
        self.bindings = list(bindings)

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=self.bindings,
            measuring=[ShapeRating(index=0, rating=1.0)],
        ), None


def _mentioned(values: dict[str, str | list[str] | None]) -> list[SlotBindingDraft]:
    """The bind model's drafts naming `values`: a value for a string, a set for a list, and
    an open slot for `None`."""
    return [
        SlotBindingDraft(
            axis="how",
            property_dcid=prop,
            kind="unbound" if value is None else "set" if isinstance(value, list) else "value",
            value_dcids=[] if value is None else value if isinstance(value, list) else [value],
            evidence="mentioned",
        )
        for prop, value in values.items()
    ]


def _answer(
    variable: str,
    bound: dict[str, str | list[str] | None],
    monkeypatch: pytest.MonkeyPatch,
    *,
    entities: Sequence[ParsedEntity] = (ParsedEntity(text="Kenya", role_text="to"),),
    recalled_as: str | None = None,
    named: bool = True,
):
    """The answer to `variable` at `entities` when the bind model drafts `bound`, recall
    reading `recalled_as` (the variable itself where it is `None`)."""
    metadata = {
        facet_id: FacetMetadata(provenance_id=_IDS, observation_period="P1Y")
        for _, facet_id, *_ in _COVERAGE_ROWS
    }
    patch_coverage(monkeypatch, _COVERAGE_ROWS, role_names=_ROLE_NAMES, facet_metadata=metadata)
    graph = _graph(recalled_as or variable, named=named)
    response = asyncio.run(
        resolve_async(
            ResolveRequest(
                input=ParsedInput(parts=[ParsedPart(variable_text=variable, entities=entities)])
            ),
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLM(_mentioned(bound)),
            sdmx=FakeSdmxClient(observations={}),
        )
    )
    assert isinstance(response, Resolution)
    (answer,) = response.parts
    return answer


def _bound(answer) -> dict[str, list[str]]:
    """Each bound slot of `answer`'s identity, by property, its values sorted."""
    identity = answer.spec.definition.identity.model_dump(mode="json", exclude_none=True)
    bound: dict[str, list[str]] = {}
    for slot in identity["slots"]:
        binding = slot["binding"]
        if binding["state"] == "value":
            bound[slot["property"]] = [binding["ref"]]
        elif binding["state"] == "set":
            bound[slot["property"]] = sorted(binding["refs"])
    return bound


def _notes(answer) -> list[str]:
    return [w.message for w in answer.warnings if w.code == SLOT_DEFAULTED]


@pytest.mark.parametrize(
    ("variable", "bound", "guarantee", "notes", "departed", "named"),
    [
        ("publicly guaranteed debt", {"guaranteeStatus": _CLAIM}, [_PPG], [_NOTE], [], True),
        # No series name holds the words, so only the fired reading meets the phrase.
        ("publicly guaranteed debt", {"guaranteeStatus": _CLAIM}, [_PPG], [_NOTE], [], False),
        (
            "publicly guaranteed private-sector debt",
            {"guaranteeStatus": _CLAIM, "idsDebtorType": _PRIVATE},
            [_CLAIM],
            [],
            [],
            True,
        ),
        (
            "publicly guaranteed government debt",
            {"guaranteeStatus": _CLAIM, "idsDebtorType": _GOVERNMENT},
            [_PPG],
            [_NOTE],
            ["how:idsDebtorType"],
            True,
        ),
        ("public or publicly guaranteed debt", {"guaranteeStatus": _PPG}, [_PPG], [], [], True),
    ],
    ids=[
        "bare-reads-ppg",
        "bare-reads-ppg-unnamed-series",
        "private-sector-keeps-the-claim",
        "another-debtor-reads-ppg",
        "ppg-unchanged",
    ],
)
def test_publicly_guaranteed_debt_reads_ppg_unless_the_private_sector_is_named(
    variable, bound, guarantee, notes, departed, named, monkeypatch
):
    """Only the private-sector debtor names the narrow category. A government debtor names
    no guaranteed claim, so the part reads PPG, and IDS publishes no PPG stock by debtor, so
    the debtor departs."""
    answer = _answer(variable, bound, monkeypatch, named=named)

    assert answer.status == "definite", answer
    assert _bound(answer)["guaranteeStatus"] == guarantee
    assert _notes(answer) == notes
    assert [d.binding_ref for d in answer.departures] == departed


def test_a_value_whose_words_lie_inside_another_values_on_its_slot_is_not_asked_for(
    monkeypatch,
):
    """ "Publicly guaranteed" inside "public or publicly guaranteed" names no second value of
    the guarantee status, so the PPG answer stands though no series name holds the words."""
    answer = _answer(
        "public or publicly guaranteed debt", {"guaranteeStatus": _PPG}, monkeypatch, named=False
    )

    assert answer.status == "definite", answer
    assert _bound(answer)["guaranteeStatus"] == [_PPG]


def test_ppg_debt_owed_to_china_reads_the_stock_on_the_by_creditor_shape(monkeypatch):
    """The role-bearing shape's position type slot, which the bind model drafts open as it
    drafts every slot of the shape it elects, takes the amount outstanding, so the undisbursed
    PPG debt owed to China stays out of the answer."""
    answer = _answer(
        "PPG debt",
        {"financialPositionType": None, "guaranteeStatus": _PPG},
        monkeypatch,
        recalled_as="public and publicly guaranteed debt",
        entities=(
            ParsedEntity(text="Kenya", role_text="to"),
            ParsedEntity(text="China", role_text="from"),
        ),
    )

    assert answer.status == "definite", answer
    assert _bound(answer)["financialPositionType"] == ["AmountOutstanding"]
    assert served_members(answer.spec.definition.identity) == {
        "Amount_FinancialPosition_AmountOutstanding_PublicOrPubliclyGuaranteed_ByCreditor_Nominal"
    }


def _shape(guarantee_values: Sequence[str]) -> ShapeDraft:
    """An IDS debt shape with guarantee status and debtor type slots."""
    return shape_draft_from(
        shape_id="ids_debt",
        label="Debt",
        pop_type_dcid="FinancialPosition",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        constraint_props=["guaranteeStatus", "idsDebtorType"],
        prop_labels={"guaranteeStatus": "guaranteeStatus", "idsDebtorType": "idsDebtorType"},
        prop_observed_values={
            "guaranteeStatus": list(guarantee_values),
            "idsDebtorType": [_PRIVATE],
        },
        sv_arc_facts={},
        slot_taxonomy={
            "how:guaranteeStatus": list(guarantee_values),
            "how:idsDebtorType": [_PRIVATE],
        },
        place_available=True,
        value_labels=None,
        recalled=False,
    )


@pytest.mark.parametrize(
    ("drafted", "taxonomy", "repaired"),
    [
        (
            [_CLAIM, "NotPubliclyGuaranteed"],
            [_PPG, _CLAIM],
            ("set", [_PPG, "NotPubliclyGuaranteed"]),
        ),
        ([_CLAIM, _PPG], [_PPG, _CLAIM], ("value", [_PPG])),
        ([_CLAIM], [_CLAIM], ("set", [_CLAIM])),
    ],
    ids=["set-member", "set-collapses-to-a-value", "broader-value-not-held"],
)
def test_a_set_holding_the_narrow_value_reads_the_broader_one(drafted, taxonomy, repaired):
    (draft,) = _mentioned({"guaranteeStatus": drafted})

    bindings, fired = apply_broad_readings([draft], _shape(taxonomy), index=get_index())

    assert [(b.kind, b.value_dcids) for b in bindings] == [repaired]
    assert len(fired) == (repaired[1] != [_CLAIM])


def test_a_debtor_bound_only_as_the_sole_option_names_no_debtor():
    claim, debtor = _mentioned({"guaranteeStatus": _CLAIM, "idsDebtorType": _PRIVATE})
    sole = debtor.model_copy(update={"evidence": "sole_option"})

    bindings, fired = apply_broad_readings([claim, sole], _shape([_PPG, _CLAIM]), index=get_index())

    assert bindings[0].value_dcids == [_PPG]
    assert fired == [
        BroadReading(
            property_dcid="guaranteeStatus",
            value_dcid=_CLAIM,
            reads_as=_PPG,
            unless_bound="idsDebtorType",
            held_with=_PRIVATE,
            note=_NOTE,
        )
    ]
