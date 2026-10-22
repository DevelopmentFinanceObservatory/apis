"""Test a part's publisher series code (`ParsedPart.source_code`): its StatVars replace
recall's embedding resolve, the open roles choose among them, and a code Search cannot serve
alone refuses with a typed no_data reason.

The codes and StatVars are the fixture code index's: DT.TDS.DEGG.CD names the World Bank
IDS general-government debt service aggregate and its by-creditor twin."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.types import IndexStamp, SourceCodeRow
from datasets_api.adapter import resolve_definition_members
from search.engine.bind import ShapeRating, SlotBindingDraft, _UniversalBindOutput
from search.engine.core import resolve_async
from search.engine.quantity_read import QuantityOutput
from search.engine.region import phase_place_materialise, pipeline
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import ResolveState
from search.engine.retrieval.bound_values import bind_member_values, served_members
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.retrieve import NoDataDraft
from search.engine.source_code import (
    SeriesCodeMembers,
    SourceCodeRefusal,
    compile_part_code,
    scan_source_codes,
    stands_for_code,
    without_code_words,
)
from search.engine.source_scope import SourceReference, SourceScope
from search.models import ParsedEntity, ParsedInput, ParsedPart, ResolveRequest, SourceMention
from series_coverage.schema import FacetMetadata, RoleNames
from statvar_members import MemberEnumerator
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.datasets_api._member_bounds import mint_bounds
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.origin_catalogue.fake_catalogue import FakeCatalogue
from tests.origin_catalogue.support import build_arc_tables

_IDS = "provenance/WorldBankInternationalDebtStatistics"
_CRS = "provenance/OECDCreditorReportingSystem"
_DEGG = "DT.TDS.DEGG.CD"
_AGGREGATE = "Amount_FinancialFlow_ServicingPayment_IDSDebtorTypeGG_Nominal"
_BY_CREDITOR = "Amount_FinancialFlow_ServicingPayment_IDSDebtorTypeGG_ByCreditor_Nominal"
# DT.TDS.DECT.CD's StatVars: total debt service over every debtor, which a DT.TDS.DEGG.CD
# identity leaving the debtor type open would elect.
_TOTAL = "Amount_FinancialFlow_ServicingPayment_Nominal"
_TOTAL_BY_CREDITOR = "Amount_FinancialFlow_ServicingPayment_ByCreditor_Nominal"
# The one StatVar IDS files under both DT.NFL.MIDA.CD and DT.NFL.IMFC.CD.
_NET_FLOW = (
    "Amount_FinancialFlow_Net_Disbursement_PrincipalRepayment_DebtInstrument_"
    "IDSConcessional_Nominal"
)
_CONCESSIONAL_NET = {
    "financialFlowType": "Disbursement",
    "financingResource": "DebtInstrument",
    "idsConcessionality": "IDSConcessional",
    "netOf": "PrincipalRepayment",
    "nettingBasis": "Net",
}
_KEN = "country/KEN"
_YEARS = [2020, 2021, 2022]
_GG = {
    "financialFlowType": "ServicingPayment",
    "idsDebtorType": "IDSDebtorTypeGG",
}
_ALL_DEBTORS = {"financialFlowType": "ServicingPayment"}
# DT.TDS.DECB.CD's aggregate, the central bank's debt service beside the general government's.
_CENTRAL_BANK = "Amount_FinancialFlow_ServicingPayment_IDSDebtorTypeCB_Nominal"
_CB = {"financialFlowType": "ServicingPayment", "idsDebtorType": "IDSDebtorTypeCB"}
# Two debtor types, each with its own code, so a mint serves a StatVar of each.
_TWO_DEBTOR_TYPES = {**_ALL_DEBTORS, "idsDebtorType": ["IDSDebtorTypeGG", "IDSDebtorTypeCB"]}
# DT.DOD.DPPG.CD's StatVars: the public or publicly guaranteed debt stock and its by-creditor
# twin. DT.DOD.VPPG.CD's variable-rate PPG stock declares the interest rate variability
# beside them, so a PPG stock identity leaves that slot open, and its family holds the
# variable-rate stock while the mint serves the aggregate, which dominates it, alone.
_DPPG = "DT.DOD.DPPG.CD"
_PPG_STOCK = "Amount_FinancialPosition_AmountOutstanding_PublicOrPubliclyGuaranteed_Nominal"
_PPG_STOCK_BY_CREDITOR = (
    "Amount_FinancialPosition_AmountOutstanding_PublicOrPubliclyGuaranteed_ByCreditor_Nominal"
)
_VARIABLE_PPG_STOCK = (
    "Amount_FinancialPosition_AmountOutstanding_VariableInterestRate_PublicOrPubliclyGuaranteed_"
    "Nominal"
)
_PPG = {
    "financialPositionType": "AmountOutstanding",
    "guaranteeStatus": "PublicOrPubliclyGuaranteed",
}


def _arcs(
    name: str,
    roles: list[str],
    constraints: dict[str, str],
    code: str,
    population: str = "FinancialFlow",
) -> dict:
    return {
        "name": {"nodes": [{"value": name}]},
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": population}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraints]},
        "sourceIndicatorCode": {"nodes": [{"value": code}]},
        "observationProperties": {"nodes": [{"dcid": r} for r in roles]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in constraints.items()},
    }


_RECIPIENT = ["financingRecipient"]
_PROVIDER_RECIPIENT = ["financingProvider", "financingRecipient"]


def _graph(recalled: Sequence[str] = ()) -> FakeGraph:
    """A graph whose embedding resolve recalls `recalled` for "debt service", so only a code
    supplies candidates when it recalls nothing. It holds DT.TDS.DECT.CD's StatVars beside
    DT.TDS.DEGG.CD's, so an identity that leaves the code's shared values open elects the
    total over every debtor, and DT.DOD.DPPG.CD's debt stocks."""
    facet = {"earliestDate": "2020", "latestDate": "2022", "obsCount": 3}
    return FakeGraph(
        nodes={
            _AGGREGATE: {
                "label": "GG debt service",
                "arcs": _arcs("GG debt service", _RECIPIENT, _GG, _DEGG),
            },
            _BY_CREDITOR: {
                "label": "GG debt service by creditor",
                "arcs": _arcs("GG debt service by creditor", _PROVIDER_RECIPIENT, _GG, _DEGG),
            },
            _TOTAL: {
                "label": "Debt service",
                "arcs": _arcs("Debt service", _RECIPIENT, _ALL_DEBTORS, "DT.TDS.DECT.CD"),
            },
            _TOTAL_BY_CREDITOR: {
                "label": "Debt service by creditor",
                "arcs": _arcs(
                    "Debt service by creditor", _PROVIDER_RECIPIENT, _ALL_DEBTORS, "DT.TDS.DECT.CD"
                ),
            },
            _CENTRAL_BANK: {
                "label": "Central bank debt service",
                "arcs": _arcs("Central bank debt service", _RECIPIENT, _CB, "DT.TDS.DECB.CD"),
            },
            _PPG_STOCK: {
                "label": "PPG debt stock",
                "arcs": _arcs("PPG debt stock", _RECIPIENT, _PPG, _DPPG, "FinancialPosition"),
            },
            _PPG_STOCK_BY_CREDITOR: {
                "label": "PPG debt stock by creditor",
                "arcs": _arcs(
                    "PPG debt stock by creditor",
                    _PROVIDER_RECIPIENT,
                    _PPG,
                    _DPPG,
                    "FinancialPosition",
                ),
            },
            _VARIABLE_PPG_STOCK: {
                "label": "Variable-rate PPG debt stock",
                "arcs": _arcs(
                    "Variable-rate PPG debt stock",
                    _RECIPIENT,
                    {**_PPG, "interestRateVariability": "VariableInterestRate"},
                    "DT.DOD.VPPG.CD",
                    "FinancialPosition",
                ),
            },
            _NET_FLOW: {
                "label": "Concessional net flows",
                "arcs": _arcs(
                    "Concessional net flows",
                    _PROVIDER_RECIPIENT,
                    _CONCESSIONAL_NET,
                    "DT.NFL.MIDA.CD",
                ),
            },
            _KEN: {"label": "Kenya", "type": "Country"},
        },
        obs={f"{sv}|{_KEN}": [facet] for sv in _ROLE_NAMES},
        detect={"debt service": {"svs": list(recalled)}} if recalled else {},
        resolve={"Kenya": _KEN},
    )


_COVERAGE_ROWS = [
    (_AGGREGATE, "f_agg", "", _KEN, _YEARS),
    (_BY_CREDITOR, "f_cr", "country/CHN", _KEN, _YEARS),
    (_TOTAL, "f_tot", "", _KEN, _YEARS),
    (_TOTAL_BY_CREDITOR, "f_tot_cr", "country/CHN", _KEN, _YEARS),
    (_NET_FLOW, "f_net", "country/CHN", _KEN, _YEARS),
    (_CENTRAL_BANK, "f_cb", "", _KEN, _YEARS),
    (_PPG_STOCK, "f_ppg", "", _KEN, _YEARS),
    (_PPG_STOCK_BY_CREDITOR, "f_ppg_cr", "country/CHN", _KEN, _YEARS),
    (_VARIABLE_PPG_STOCK, "f_vppg", "", _KEN, _YEARS),
]
_ROLE_NAMES = {
    _AGGREGATE: RoleNames(donor=None, recipient="financingRecipient"),
    _BY_CREDITOR: RoleNames(donor="financingProvider", recipient="financingRecipient"),
    _TOTAL: RoleNames(donor=None, recipient="financingRecipient"),
    _TOTAL_BY_CREDITOR: RoleNames(donor="financingProvider", recipient="financingRecipient"),
    _NET_FLOW: RoleNames(donor="financingProvider", recipient="financingRecipient"),
    _CENTRAL_BANK: RoleNames(donor=None, recipient="financingRecipient"),
    _PPG_STOCK: RoleNames(donor=None, recipient="financingRecipient"),
    _PPG_STOCK_BY_CREDITOR: RoleNames(donor="financingProvider", recipient="financingRecipient"),
    _VARIABLE_PPG_STOCK: RoleNames(donor=None, recipient="financingRecipient"),
}


def _coverage():
    return FakeCoverageService(
        build_coverage_index(_COVERAGE_ROWS, role_names=_ROLE_NAMES)
    ).snapshot()


def _answer(
    part: ParsedPart,
    monkeypatch: pytest.MonkeyPatch,
    llm: _BindLLM | None = None,
    *,
    recalled: Sequence[str] = (),
    read: str = _IDS,
):
    """`part`'s answer through the parsed door, over a graph recalling `recalled` and
    coverage holding every StatVar's observations under the provenance `read`, IDS as the
    graph holds them."""
    metadata = {
        facet_id: FacetMetadata(provenance_id=read, observation_period="P1Y")
        for _, facet_id, *_ in _COVERAGE_ROWS
    }
    patch_coverage(monkeypatch, _COVERAGE_ROWS, role_names=_ROLE_NAMES, facet_metadata=metadata)
    graph = _graph(recalled)
    response = asyncio.run(
        resolve_async(
            ResolveRequest(input=ParsedInput(parts=[part])),
            graph=graph,
            resolver=make_resolver(graph),
            llm=llm or _BindLLM(),
            sdmx=FakeSdmxClient(observations={}),
        )
    )
    (answer,) = response.parts
    return answer


def _minted_members(answer) -> set[str]:
    """The StatVars the Datasets mint resolves `answer`'s identity to over the fixture
    index, as the orchestrator's fetch by its spec_id reads them."""
    identity = answer.spec.definition.identity.model_dump(mode="json", exclude_none=True)
    resolved = resolve_definition_members(
        identity,
        spec_id=answer.spec.spec_id,
        graph=FakeMintGraphReader(arcs={}),
        enumerator=MemberEnumerator(),
        code_index=get_index(),
        bounds=mint_bounds(),
    )
    return {shape.stat_var.removeprefix("dcid:") for shape in resolved.shapes}


class _BindLLM:
    """Elects shape 0 with `bindings`, standing in for the bind model, or asks with `ask`.
    It records each bind prompt."""

    def __init__(self, ask: str | None = None, bindings: Sequence[SlotBindingDraft] = ()) -> None:
        self.ask = ask
        self.bindings = list(bindings)
        self.prompts: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        self.prompts.append(prompt)
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=self.bindings,
            ask=self.ask,
            measuring=[ShapeRating(index=0, rating=1.0)],
        ), None


_STAMP = IndexStamp(
    graph_endpoint="https://fake.example",
    exported_at="2026-10-13T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="",
)


def _index(rows: list[SourceCodeRow]) -> CodeIndex:
    return CodeIndex(stamp=_STAMP, declarations=(), codes=(), statvars=(), source_codes=rows)


_ROLES = {
    frozenset(): ["financingRecipient"],
    frozenset({"origin"}): ["financingProvider", "financingRecipient"],
}


@pytest.mark.parametrize("open_axes", list(_ROLES), ids=["aggregate", "by-creditor"])
def test_a_codes_statvars_are_the_recall_and_the_open_role_elects_among_them(open_axes):
    members = compile_part_code(_DEGG, scope=None, index=get_index())
    assert members == SeriesCodeMembers(
        code=_DEGG, provenance=_IDS, dcids=tuple(sorted((_AGGREGATE, _BY_CREDITOR)))
    )
    graph = _graph()

    result = asyncio.run(
        resolve_variable(
            "debt service",
            entities=["Kenya"],
            date_request=None,
            detect_query="debt service",
            role_query="debt service",
            open_axes=open_axes,
            source_code=members,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLM(),
            sdmx=FakeSdmxClient(observations={}),
            base_timing={},
            coverage=_coverage(),
        )
    )

    assert result.status == "definite", result.no_data_reason
    (member,) = result.served_members
    assert [n["dcid"] for n in member.arcs["observationProperties"]["nodes"]] == _ROLES[open_axes]


@pytest.mark.parametrize(
    ("source_code", "open_roles", "reason", "roles"),
    [
        (_DEGG, None, None, _ROLES[frozenset()]),
        ("dt.tds.degg.cd ", ["creditor"], None, _ROLES[frozenset({"origin"})]),
        (None, None, "variable_not_resolved", None),
        ("NGDP", None, "source_code_not_held", None),
        ("DT.NFL.MIDA.CD", None, "source_code_shares_variable", None),
    ],
    ids=["aggregate", "by-creditor", "no-code", "not-held", "shares-variable"],
)
def test_the_parsed_door_answers_a_code_from_its_statvars(
    source_code, open_roles, reason, roles, monkeypatch
):
    part = ParsedPart(
        variable_text="debt service",
        entities=[ParsedEntity(text="Kenya")],
        source_code=source_code,
        open_roles=open_roles,
    )

    answer = _answer(part, monkeypatch)

    if reason is None:
        assert answer.status == "definite", answer
        assert answer.restatement.source_code == _DEGG
        assert answer.spec.definition.identity.open_roles == (["origin"] if open_roles else None)
        assert _minted_members(answer) == {_BY_CREDITOR if open_roles else _AGGREGATE}
        return
    assert answer.no_data.reason == reason
    assert answer.no_data.category == "unresolved"
    if source_code is not None:
        assert answer.no_data.source_terms == [source_code]


def _mentioned(values: dict[str, str | list[str]]) -> list[SlotBindingDraft]:
    """The bind model's drafts naming `values`: a value for a string, a set for a list."""
    return [
        SlotBindingDraft(
            axis="how",
            property_dcid=prop,
            kind="set" if isinstance(value, list) else "value",
            value_dcids=value if isinstance(value, list) else [value],
            evidence="mentioned",
        )
        for prop, value in values.items()
    ]


@pytest.mark.parametrize(
    ("source_code", "recalled", "bound", "open_roles", "exact"),
    [
        (_DEGG, (), {}, None, {_DEGG: True}),
        (None, (_AGGREGATE, _BY_CREDITOR), _GG, None, {_DEGG: True}),
        (None, (_AGGREGATE, _TOTAL), _ALL_DEBTORS, None, {"DT.TDS.DECT.CD": True}),
        pytest.param(
            _DPPG,
            (),
            {},
            None,
            {_DPPG: True},
        ),
        (
            None,
            (_AGGREGATE, _CENTRAL_BANK),
            _TWO_DEBTOR_TYPES,
            None,
            {_DEGG: False, "DT.TDS.DECB.CD": False},
        ),
        pytest.param(
            None,
            (_NET_FLOW,),
            _CONCESSIONAL_NET,
            ["creditor"],
            {"DT.NFL.IMFC.CD": False, "DT.NFL.MIDA.CD": False},
        ),
    ],
    ids=[
        "code-search",
        "one-unshared-code",
        "total-over-every-debtor",
        "family-wider-than-the-served-members",
        "served-across-two-codes",
        "shared-variable",
    ],
)
def test_a_served_code_is_exact_where_a_search_for_it_accepts_the_spec(
    source_code, recalled, bound, open_roles, exact, monkeypatch, bind_catalogue
):
    """A Spec lists the codes of the StatVars it serves, and marks one exact only where a
    part naming that code answers with the same Spec: the members a Datasets mint of its
    identity serves sit within the code's StatVars, and no other code names them. The total
    over every debtor, found by its words, leaves the debtor type open, and its total
    dominates each debtor type's series, so the mint serves DT.TDS.DECT.CD's StatVar alone.
    DT.DOD.DPPG.CD's family holds a variable-rate PPG stock outside the code, which its
    aggregate dominates, so the code part answers. Two debtor types, each its own code,
    serve a StatVar of each, so neither code is exact."""
    # The catalogue holds the PPG stock family, as the deployed one does, so a PPG stock
    # shape declares every slot of its family and its identity leaves the variable rate open.
    stocks = (_PPG_STOCK, _PPG_STOCK_BY_CREDITOR, _VARIABLE_PPG_STOCK)
    held = {dcid: _graph().node_arcs(dcid) for dcid in stocks}
    bind_catalogue(FakeCatalogue(arcs=build_arc_tables(dfo={}, base=held, schema={})))
    part = ParsedPart(
        variable_text="debt service",
        entities=[ParsedEntity(text="Kenya", role_text="to")],
        source_code=source_code,
        open_roles=open_roles,
    )

    answer = _answer(part, monkeypatch, _BindLLM(bindings=_mentioned(bound)), recalled=recalled)

    assert answer.status == "definite", answer
    assert served_members(answer.spec.definition.identity) == _minted_members(answer)
    resolved = answer.spec.resolved
    assert resolved.facets["provenance"].ref == _IDS
    assert {(c.provenance, c.code): c.exact for c in resolved.stat_vars.source_codes} == {
        (_IDS, code): flag for code, flag in exact.items()
    }
    wire = answer.spec.model_dump(mode="json", exclude_none=True)["resolved"]["stat_vars"]
    assert ["exact" in entry for entry in wire["source_codes"]] == list(exact.values())


def test_a_code_is_not_exact_for_a_spec_reading_another_provenance(monkeypatch):
    """A code part's reads are not held to the code's provenance, so its Spec can serve the
    code's StatVars from observations another provenance holds."""
    part = ParsedPart(
        variable_text="debt service", entities=[ParsedEntity(text="Kenya")], source_code=_DEGG
    )
    answer = _answer(part, monkeypatch, read="dc/base/OtherSource")

    (code,) = answer.spec.resolved.stat_vars.source_codes

    assert (code.code, code.exact) == (_DEGG, False)


@pytest.mark.parametrize(
    ("source_code", "recalled", "bound", "served", "exact"),
    [
        (_DEGG, (), {}, (_AGGREGATE,), True),
        (_DEGG, (), {}, (_AGGREGATE, _TOTAL), False),
        (None, (_AGGREGATE, _CENTRAL_BANK), _TWO_DEBTOR_TYPES, (_AGGREGATE,), False),
    ],
    ids=["the-codes-statvars", "a-statvar-outside-the-code", "a-mint-serving-another-code"],
)
def test_a_code_is_not_exact_for_a_spec_serving_a_statvar_outside_it(
    source_code, recalled, bound, served, exact, monkeypatch
):
    """The last identity binds two debtor types, so a mint of it serves DT.TDS.DECB.CD's
    aggregate beside DT.TDS.DEGG.CD's, and the code is not exact even for served StatVars
    that are all its own (`bound_values.elects_within`)."""
    part = ParsedPart(
        variable_text="debt service",
        entities=[ParsedEntity(text="Kenya", role_text="to")],
        source_code=source_code,
    )
    llm = _BindLLM(bindings=_mentioned(bound))
    identity = _answer(part, monkeypatch, llm, recalled=recalled).spec.definition.identity

    stands = stands_for_code(
        _DEGG, provenance=_IDS, identity=identity, served=served, read=_IDS, index=get_index()
    )

    assert stands is exact


@pytest.mark.parametrize("open_roles", [None, ["creditor"]], ids=["aggregate", "by-creditor"])
def test_a_part_naming_its_series_by_the_code_alone_answers_as_one_naming_its_words(
    open_roles, monkeypatch
):
    """A caller that has only the code posts it as the variable text too. Once the code
    resolves, the part answers the spec, warnings and departures the measure's words give."""

    def answer(variable_text: str):
        part = ParsedPart(
            variable_text=variable_text,
            entities=[ParsedEntity(text="Kenya")],
            source_code=_DEGG,
            open_roles=open_roles,
        )
        return _answer(part, monkeypatch)

    words, code = answer("debt service"), answer(_DEGG)

    assert code.status == words.status == "definite"
    assert code.spec.spec_id == words.spec.spec_id
    assert [w.code for w in code.warnings] == [w.code for w in words.warnings]
    assert [d.code for d in code.departures] == [d.code for d in words.departures]


@pytest.mark.parametrize("variable_text", ["debt service", _DEGG], ids=["words", "code-only"])
def test_a_code_part_the_bind_model_cannot_place_refuses_not_held(variable_text, monkeypatch):
    """A bind model that asks leaves the part unanswered, and the code fixes its variable,
    so it refuses `source_code_not_held` caused `unanswerable`. A part whose variable text is
    only the code shows the bind model its StatVars' graph name in place of the code."""
    llm = _BindLLM(ask="Variable does not match any candidate shape.")
    part = ParsedPart(
        variable_text=variable_text, entities=[ParsedEntity(text="Kenya")], source_code=_DEGG
    )

    answer = _answer(part, monkeypatch, llm)

    assert (answer.no_data.reason, answer.no_data.source_terms) == (
        "source_code_not_held",
        [_DEGG],
    )
    (prompt,) = llm.prompts
    phrase = prompt.split("Variable phrase: ", 1)[1].split("\n", 1)[0]
    assert phrase == ("debt service" if variable_text == "debt service" else "GG debt service")


def test_a_code_part_materialise_finds_no_variable_for_refuses_not_held(monkeypatch):
    """Materialise ends a part with `variable_not_resolved` when its members cannot be
    admitted. The part's variable is the code's, so that exit refuses as every other does."""

    async def no_variable(state):
        state.mat_result = NoDataDraft("variable_not_resolved")
        return state.mat_result

    monkeypatch.setattr(phase_place_materialise, "_run_materialise_task", no_variable)
    part = ParsedPart(
        variable_text="debt service", entities=[ParsedEntity(text="Kenya")], source_code=_DEGG
    )

    answer = _answer(part, monkeypatch)

    assert (answer.no_data.reason, answer.no_data.source_terms) == ("source_code_not_held", [_DEGG])


def test_a_restated_code_part_posts_back_to_the_same_spec(monkeypatch):
    part = ParsedPart(
        variable_text="debt service DT.TDS.DEGG.CD",
        entities=[ParsedEntity(text="Kenya")],
        source_code=_DEGG,
    )

    first = _answer(part, monkeypatch)

    assert first.restatement.variable_text == "debt service"
    assert _answer(first.restatement, monkeypatch).spec.spec_id == first.spec.spec_id


def test_a_code_none_of_whose_statvars_the_graph_confirms_refuses_not_held():
    members = SeriesCodeMembers(code=_DEGG, provenance=_IDS, dcids=("Absent_StatVar",))
    graph = _graph()

    result = asyncio.run(
        resolve_variable(
            "debt service",
            entities=["Kenya"],
            date_request=None,
            detect_query="debt service",
            role_query="debt service",
            source_code=members,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLM(),
            sdmx=FakeSdmxClient(observations={}),
            base_timing={},
            coverage=_coverage(),
        )
    )

    assert result.no_data_reason == "source_code_not_held"
    assert result.source_terms == (_DEGG,)
    assert result.source_code_refusal is not None
    assert (
        result.source_code_refusal.cause,
        [s.ref for s in result.source_code_refusal.sources],
    ) == ("not_served", [_IDS])


def test_a_code_part_starts_no_un_leg_and_gets_no_un_pass(monkeypatch):
    """A code part answers from its code's StatVars alone, so its refusal stands: it starts
    no UN System Data Commons derivation and the UN pass is skipped."""
    states: list[ResolveState] = []
    skips: list[str | None] = []
    un_pass_skip = pipeline._un_pass_skip

    def record(result, first):
        states.append(first)
        skips.append(un_pass_skip(result, first))
        return skips[-1]

    monkeypatch.setattr(pipeline, "_un_pass_skip", record)
    llm = _BindLLM(ask="Variable does not match any candidate shape.")
    part = ParsedPart(
        variable_text="debt service", entities=[ParsedEntity(text="Kenya")], source_code=_DEGG
    )

    answer = _answer(part, monkeypatch, llm)

    assert answer.no_data.reason == "source_code_not_held"
    assert skips == ["source_code"]
    assert [state.un_leg for state in states] == [None]


_ALL_DEBTORS_PHRASE = "across all debtor types"


def _asked_in_full(property_dcid: str) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how",
        property_dcid=property_dcid,
        kind="unbound",
        value_dcids=[],
        evidence="requested_all",
        requested_all_phrase=_ALL_DEBTORS_PHRASE,
    )


def _resolve_degg(
    bindings: Sequence[SlotBindingDraft] = (),
    *,
    ask: str | None = None,
    scope: SourceScope | None = None,
):
    """The DT.TDS.DEGG.CD part's result under the source scope `scope` when the bind model
    drafts `bindings`, or asks with `ask`."""
    members = compile_part_code(_DEGG, scope=scope, index=get_index())
    assert isinstance(members, SeriesCodeMembers)
    graph = _graph()
    return asyncio.run(
        resolve_variable(
            "debt service",
            entities=["Kenya"],
            date_request=None,
            detect_query="debt service",
            role_query="debt service",
            dropped_qualifier_phrases=[_ALL_DEBTORS_PHRASE],
            source_code=members,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLM(ask, bindings),
            sdmx=FakeSdmxClient(observations={}),
            base_timing={},
            coverage=_coverage(),
            source_scope=scope,
        )
    )


def test_a_code_part_under_an_exclusion_refuses_not_held_as_one_under_no_scope_does(
    monkeypatch,
):
    """A part naming DT.TDS.DEGG.CD excludes the CRS, which holds none of the code's
    StatVars. The bind model declines its menu, and the part refuses `source_code_not_held`
    as the same part under no scope does: the exclusion is not what the refusal names, the
    scope settles nothing, and the UN pass is skipped for the code alone."""
    excluding = SourceScope(
        allowed=(frozenset(get_index().datasets) | {_IDS}) - {_CRS},
        named=False,
        mentions=(SourceMention(text="CRS", ref=_CRS, polarity="exclude"),),
        references=(SourceReference(ref=_CRS, label="CRS", kind="provenance"),),
        expansions=(frozenset({_CRS}),),
    )

    states: list[ResolveState] = []
    skips: list[str | None] = []
    un_pass_skip = pipeline._un_pass_skip

    def record(result, first):
        states.append(first)
        skips.append(un_pass_skip(result, first))
        return skips[-1]

    monkeypatch.setattr(pipeline, "_un_pass_skip", record)

    plain, excluded = (_resolve_degg(ask="Which series?", scope=s) for s in (None, excluding))

    assert plain.no_data_reason == excluded.no_data_reason == "source_code_not_held"
    assert excluded.source_code_refusal == plain.source_code_refusal
    assert not excluded.unbound_sources
    assert [state.scope_settled for state in states] == [False, False]
    assert skips == ["source_code", "source_code"]


def test_a_code_part_asking_every_value_of_a_slot_its_code_fixes_refuses_not_held():
    """Every StatVar of DT.TDS.DEGG.CD is general government debt, so the code cannot answer
    a part that asks for all debtor types."""
    result = _resolve_degg([_asked_in_full("idsDebtorType")])

    assert result.no_data_reason == "source_code_not_held"
    assert result.source_code_refusal is not None
    assert result.source_code_refusal.cause == "unanswerable"


_CODE_MEMBERS = (_AGGREGATE, _BY_CREDITOR)
_GG_BOUND = {prop: ("value", [value], "sole_option") for prop, value in _GG.items()}
_OPEN_IN_FULL = ("unbound", [], "requested_all")


@pytest.mark.parametrize(
    ("members", "drafted", "bound", "refused"),
    [
        (
            _CODE_MEMBERS,
            _asked_in_full("idsDebtorType"),
            {"idsDebtorType": _OPEN_IN_FULL},
            True,
        ),
        (
            (_AGGREGATE, _TOTAL),
            _asked_in_full("idsDebtorType"),
            {
                "idsDebtorType": _OPEN_IN_FULL,
                "financialFlowType": _GG_BOUND["financialFlowType"],
            },
            False,
        ),
        (
            _CODE_MEMBERS,
            SlotBindingDraft(
                axis="how",
                property_dcid="idsDebtorType",
                kind="value",
                value_dcids=["IDSDebtorTypeCB"],
                evidence="mentioned",
            ),
            _GG_BOUND,
            False,
        ),
    ],
    ids=["asked-in-full-and-fixed", "asked-in-full-and-varying", "mentioned-and-fixed"],
)
def test_member_values_bind_over_a_named_value_and_leave_a_slot_asked_in_full(
    members, drafted, bound, refused
):
    """The debtor type is one value over DT.TDS.DEGG.CD's StatVars and varies between one of
    them and the total over every debtor."""
    graph = _graph()
    (shape,) = derive_shapes(confirmed_svs=list(_CODE_MEMBERS), graph=graph)

    bindings, asked_in_full = bind_member_values(shape, [drafted], members)

    assert {b.property_dcid: (b.kind, b.value_dcids, b.evidence) for b in bindings} == bound
    assert asked_in_full == ([drafted] if refused else [])


def test_compile_names_what_each_refusal_offers():
    index = get_index()

    not_held = compile_part_code("NGDP", scope=None, index=index)
    assert isinstance(not_held, SourceCodeRefusal)
    assert (not_held.reason, not_held.cause, [s.ref for s in not_held.sources]) == (
        "source_code_not_held",
        "unknown",
        sorted(index.source_code_provenances),
    )

    shares = compile_part_code("DT.NFL.MIDA.CD", scope=None, index=index)
    assert isinstance(shares, SourceCodeRefusal)
    assert (shares.reason, shares.shared_codes) == (
        "source_code_shares_variable",
        ("DT.NFL.IMFC.CD",),
    )

    outside = compile_part_code(
        _DEGG, scope=SourceScope(allowed=frozenset({"dc/base/Other"}), named=True), index=index
    )
    assert isinstance(outside, SourceCodeRefusal)
    assert (outside.reason, outside.cause, [s.ref for s in outside.sources]) == (
        "source_code_not_held",
        "excluded",
        [_IDS],
    )


@pytest.mark.parametrize(
    ("allowed", "outcome"),
    [
        (None, "source_code_ambiguous"),
        (frozenset({"dc/base/A"}), ("dc/base/A", ("sv/a",))),
    ],
    ids=["two-sources", "one-named"],
)
def test_a_code_two_sources_hold_needs_one_named(allowed, outcome):
    index = _index(
        [
            SourceCodeRow(provenance="dc/base/A", code="X.CODE", dcids=("sv/a",)),
            SourceCodeRow(provenance="dc/base/B", code="X.CODE", dcids=("sv/b",)),
        ]
    )
    scope = None if allowed is None else SourceScope(allowed=allowed, named=True)

    compiled = compile_part_code("X.CODE", scope=scope, index=index)

    if isinstance(compiled, SourceCodeRefusal):
        assert compiled.reason == outcome
        assert [s.ref for s in compiled.sources] == ["dc/base/A", "dc/base/B"]
    else:
        assert (compiled.provenance, compiled.dcids) == outcome


_SCAN_INDEX = _index(
    [
        SourceCodeRow(provenance="dc/base/A", code=_DEGG, dcids=("sv/a",)),
        SourceCodeRow(provenance="dc/base/W", code="LP", dcids=("sv/lp",)),
    ]
)


@pytest.mark.parametrize(
    ("query", "measures", "expected"),
    [
        ("debt service of Kenya", ["debt service"], [None]),
        ("dt.tds.degg.cd for Kenya in 2020.", ["debt service"], [_DEGG]),
        ("Kenya's debt service (DT.TDS.DEGG.CD).", ["debt service"], [_DEGG]),
        ("the lp of Kenya, Lp included", ["population"], [None]),
        ("LP for Kenya", ["LP"], ["LP"]),
        (
            "debt service DT.TDS.DEGG.CD and population",
            ["debt service DT.TDS.DEGG.CD", "population"],
            [_DEGG, None],
        ),
        ("DT.TDS.DEGG.CD by creditor, DT.TDS.DEGG.CD", [_DEGG], [_DEGG]),
    ],
    ids=[
        "no-code",
        "lower-case-prose",
        "bracketed",
        "short-word",
        "short-capitals",
        "two-measures",
        "code-as-measure",
    ],
)
def test_the_query_scan_fills_a_held_code(query, measures, expected):
    assert scan_source_codes(query, measures=measures, index=_SCAN_INDEX) == expected


@pytest.mark.parametrize(
    ("variable", "expected"),
    [
        ("debt service DT.TDS.DEGG.CD", "debt service"),
        ("dt.tds.degg.cd debt service", "debt service"),
        (_DEGG, _DEGG),
    ],
)
def test_a_codes_words_leave_the_variable_text(variable, expected):
    assert without_code_words(variable, _DEGG) == expected
