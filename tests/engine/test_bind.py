"""Test shape selection, bindings, and prompt fields in universal bind."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from code_index import shared as code_index_shared
from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.types import IndexStamp, PartlyStatedValue, StatVarRow
from search.engine.bind import (
    SlotBindingDraft,
    _build_universal_bind_prompt,
    _UniversalBindOutput,
    bind_universal,
    rank_elected,
    split_stray_bindings,
)
from search.engine.binders import BindRequest, GeminiBinder
from search.engine.errors import LLMOutputRefused
from search.engine.extract import DateRequest
from search.engine.frame import door_frame, with_recall
from search.engine.interpretation import Interpretation
from search.engine.quantity import NamedDenominator, quantity_requirement, with_stated_form
from search.engine.quantity_read import QuantityOutput
from search.engine.region.phase_rank_bind import _stray_requirements
from search.engine.region.place_set import PlaceNode, PlaceSet
from search.engine.region.requirements import judge
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.models import GraphRef, QuantityForm
from tests.engine._binders import elected_reply, keep_the_pick
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY

# Helpers


def _run(coro):
    return asyncio.run(coro)


# Minimal recall data for resolve_variable's failed-bind cases.
_RESOLVE_SV = "Count_Person"
_RESOLVE_DETECT_QUERY = "population"

_RESOLVE_NODES: dict = {
    "Person": {"label": "Person"},
    "count": {"label": "count"},
    "measuredValue": {"label": "Measured Value"},
    "observationAbout": {"label": "observation about"},
    "country/KEN": {"label": "Kenya", "type": "Country"},
    _RESOLVE_SV: {
        "label": "Population",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
}
_RESOLVE_DETECT: dict = {_RESOLVE_DETECT_QUERY: {"svs": [_RESOLVE_SV], "entities": []}}
_RESOLVE_RESOLVE: dict = {"Kenya": "country/KEN"}


def _shape(
    shape_id: str,
    *,
    slot_keys: tuple[SlotKeyDraft, ...] = (),
    slot_taxonomy: dict[str, list[str]] | None = None,
    score: float = 0.8,
) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=slot_keys,
        slot_taxonomy=slot_taxonomy or {},
        representative_score=score,
    )


_GENDER_SLOT = SlotKeyDraft(axis="how", property_dcid="gender", label="gender")


async def _bind_through_the_gemini_binder(variable, shapes, *, llm):
    """`bind_universal`'s reply, read through `GeminiBinder.bind`."""
    request = BindRequest(
        variable=variable, shapes=shapes, query_text=None, apply_policies=keep_the_pick
    )
    result, usage = await GeminiBinder().bind(request, llm=llm)
    assert (result.binder, result.also_plausible_support) == ("gemini", None)
    return result.output, usage


@pytest.fixture(
    params=[bind_universal, _bind_through_the_gemini_binder], ids=["bind_universal", "binder"]
)
def bind(request):
    """Each stub-LLM case runs on `bind_universal` and through the Gemini binder, which
    answers it alike."""
    return request.param


class _StubLLM:
    """Return a fixed universal-bind result."""

    def __init__(self, output: _UniversalBindOutput):
        self._output = output

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return self._output, {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}


# chosen_shape_index / bindings


def test_split_stray_bindings_drops_drafts_on_another_shapes_slots():
    """An output electing one shape and binding the slots of another keeps the drafts its
    elected shape holds and splits off the rest; the same bindings on the shape that holds
    the slot are all kept."""
    shapes = [
        _shape(
            "count_person_by_gender",
            slot_keys=(_GENDER_SLOT,),
            slot_taxonomy={"how:gender": ["Male", "Female"]},
        ),
        _shape("count_person"),
    ]
    gender = SlotBindingDraft(
        axis="how", property_dcid="gender", kind="value", value_dcids=["Female"]
    )
    open_age = SlotBindingDraft(axis="how", property_dcid="age", kind="unbound")

    def output(index: int) -> _UniversalBindOutput:
        return elected_reply(index, bindings=[gender, open_age])

    kept, stray = split_stray_bindings(output(1), shapes)
    assert (kept.bindings, stray) == ([open_age], (gender,))
    kept, stray = split_stray_bindings(output(0), shapes)
    assert (kept.bindings, stray) == ([gender, open_age], ())


def test_bind_universal_chooses_shape_and_binds_slots(bind):
    shapes = [
        _shape(
            "count_person_female",
            slot_keys=(_GENDER_SLOT,),
            slot_taxonomy={"how:gender": ["Male", "Female"]},
        ),
    ]
    output = elected_reply(
        0,
        bindings=[
            SlotBindingDraft(
                axis="how", property_dcid="gender", kind="value", value_dcids=["Female"]
            ),
        ],
        also_plausible=None,
    )
    result, usage = _run(bind("female population", shapes, llm=_StubLLM(output)))
    assert result.chosen_shape_index == 0
    assert result.ask is None
    assert len(result.bindings) == 1
    assert result.bindings[0].value_dcids == ["Female"]
    assert usage is not None


def test_bind_universal_unbinds_a_source_draft_the_model_writes(bind):
    """Only the engine names a dataset, so a source-axis value from the model comes back
    unbound, the engine's draft, and every other draft comes back as written, stamped as
    the reply's."""
    shapes = [_shape("count_person_gender", slot_keys=(_GENDER_SLOT,), score=0.9)]
    gender = SlotBindingDraft(
        axis="how", property_dcid="gender", kind="value", value_dcids=["Female"]
    )
    output = elected_reply(
        0,
        bindings=[
            gender,
            SlotBindingDraft(
                axis="source", property_dcid=None, kind="value", value_dcids=["provenance/Made_Up"]
            ),
        ],
        also_plausible=None,
    )

    result, _ = _run(bind("female population", shapes, llm=_StubLLM(output)))

    assert result.bindings == [
        gender.model_copy(update={"drafted_by": "reply"}),
        SlotBindingDraft(axis="source", property_dcid=None, kind="unbound", drafted_by="engine"),
    ]


_DEBT = "provenance/DebtStatistics"
_RESOURCE_SLOT = SlotKeyDraft(axis="how", property_dcid="financingResource", label="resource")
_SERVICE_TOTAL = "Amount_DebtFlow_ServicingPayment"
_SERVICE_TO_A_LENDER = "Amount_DebtFlow_ServicingPayment_DebtInstrument"
_SERVICE_ELSEWHERE = "Amount_DebtFlow_ServicingPayment_DebtInstrument_Elsewhere"
_SERVICE_UNLISTED = "Amount_DebtFlow_ServicingPayment_DebtInstrument_Unlisted"
_COMMITMENT = "Amount_DebtCommitment_DebtInstrument"
_GRANT = "Amount_DebtFlow_Disbursement_UnrequitedTransfer"
_SERVICE_ON_A_LOAN = "Amount_DebtFlow_ServicingPayment_Loan"


def _debt_row(dcid: str, population_type: str, provenance: str, **constraints: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type=population_type,
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=("financingRecipient",),
        constraint_properties=constraints,
        provenances=(provenance,),
    )


_DEBT_ROWS = {
    row.dcid: row
    for row in (
        _debt_row(_SERVICE_TOTAL, "DebtFlow", _DEBT, financialFlowType="ServicingPayment"),
        _debt_row(
            _SERVICE_TO_A_LENDER,
            "DebtFlow",
            _DEBT,
            financialFlowType="ServicingPayment",
            financingResource="DebtInstrument",
        ),
        _debt_row(
            _SERVICE_ELSEWHERE,
            "DebtFlow",
            "provenance/Elsewhere",
            financialFlowType="ServicingPayment",
            financingResource="DebtInstrument",
        ),
        _debt_row(_COMMITMENT, "DebtCommitment", _DEBT, financingResource="DebtInstrument"),
        _debt_row(
            _GRANT,
            "DebtFlow",
            _DEBT,
            financialFlowType="Disbursement",
            financingResource="UnrequitedTransfer",
        ),
        _debt_row(
            _SERVICE_ON_A_LOAN,
            "DebtFlow",
            _DEBT,
            financialFlowType="ServicingPayment",
            financingResource="Loan",
        ),
    )
}


def _debt_arcs(population_type: str, **constraints: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": population_type}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": "financingRecipient"}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
        **{prop: {"nodes": [{"dcid": value}]} for prop, value in constraints.items()},
    }


def _debt_shape(*members: str) -> ShapeDraft:
    """A recalled sample of `members`, each with the constraints its index row states, or a
    lender's for a member the index lists no row for."""
    population_type = "DebtCommitment" if _COMMITMENT in members else "DebtFlow"
    lender = {"financialFlowType": "ServicingPayment", "financingResource": "DebtInstrument"}
    return dataclasses.replace(
        _shape(
            "debt",
            slot_keys=(_RESOURCE_SLOT,),
            slot_taxonomy={
                "how:financingResource": ["DebtInstrument", "Loan", "UnrequitedTransfer"]
            },
        ),
        pop_type_dcid=population_type,
        meas_prop_dcid="amount",
        sv_arc_facts={
            dcid: _debt_arcs(
                population_type,
                **(_DEBT_ROWS[dcid].constraint_properties if dcid in _DEBT_ROWS else lender),
            )
            for dcid in members
        },
    )


_INSTRUMENT = ["DebtInstrument"]


@pytest.mark.parametrize(
    ("members", "values", "kept"),
    [
        pytest.param(
            (_SERVICE_TOTAL, _SERVICE_TO_A_LENDER), _INSTRUMENT, False, id="beside-a-silent-total"
        ),
        pytest.param(
            (_SERVICE_TOTAL, _SERVICE_TO_A_LENDER, _GRANT),
            _INSTRUMENT,
            False,
            id="beside-a-sibling-value",
        ),
        pytest.param(
            (_SERVICE_TOTAL, _SERVICE_TO_A_LENDER, _SERVICE_ON_A_LOAN),
            ["DebtInstrument", "Loan"],
            False,
            id="a-set-of-partly-stated-values",
        ),
        pytest.param(
            (_SERVICE_TOTAL, _SERVICE_TO_A_LENDER, _GRANT),
            ["DebtInstrument", "UnrequitedTransfer"],
            True,
            id="a-set-holding-a-value-stated-in-full",
        ),
        pytest.param((_COMMITMENT,), _INSTRUMENT, True, id="every-row-of-the-five-tuple-states-it"),
        pytest.param(
            (_SERVICE_TOTAL, _SERVICE_TO_A_LENDER, _SERVICE_ELSEWHERE),
            _INSTRUMENT,
            True,
            id="another-dataset-states-it",
        ),
        pytest.param(
            (_SERVICE_TOTAL, _SERVICE_TO_A_LENDER, _SERVICE_UNLISTED),
            _INSTRUMENT,
            True,
            id="a-member-the-index-lists-no-row-for-states-it",
        ),
        pytest.param((_SERVICE_TOTAL,), _INSTRUMENT, True, id="no-member-states-it"),
    ],
)
def test_bind_universal_unbinds_a_value_a_dataset_states_on_only_some_of_its_series(
    bind, monkeypatch, members, values, kept
):
    """ "Debt service on external debt" names no resource. The debt statistics state the
    debt instrument on the service owed to a named lender and on no total, so the draft
    comes back unbound, the engine's, where every member stating it is theirs, beside a
    grant stating another resource too. It binds as written on the commitments, whose
    every row states it, and in a set holding a value stated in full."""
    index = CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="partly-stated-values",
        ),
        declarations=(),
        codes=(),
        statvars=(),
        listed_statvars=tuple(_DEBT_ROWS.values()),
        partly_stated_values=tuple(
            PartlyStatedValue(provenance=_DEBT, property_dcid="financingResource", value_dcid=value)
            for value in ("DebtInstrument", "Loan")
        ),
    )
    packaged = code_index_shared.get_index
    for module in list(sys.modules.values()):
        if vars(module).get("get_index") is packaged:
            monkeypatch.setattr(module, "get_index", lambda: index)
    resource = SlotBindingDraft(
        axis="how",
        property_dcid="financingResource",
        kind="set" if len(values) > 1 else "value",
        value_dcids=values,
    )
    output = elected_reply(0, bindings=[resource], also_plausible=None)

    result, _ = _run(bind("debt service", [_debt_shape(*members)], llm=_StubLLM(output)))

    assert result.bindings == [
        resource.model_copy(update={"drafted_by": "reply"})
        if kept
        else SlotBindingDraft(
            axis="how", property_dcid="financingResource", kind="unbound", drafted_by="engine"
        )
    ]


_AGE_SLOT = SlotKeyDraft(axis="how", property_dcid="undataAge", label="age")


@pytest.mark.parametrize(
    ("written", "taxonomy", "expected"),
    [
        pytest.param(
            ["undata/AGE-Y0T4"],
            ["undata/AGE-Y0T4", "undata/AGE-Y5T9"],
            SlotBindingDraft(
                axis="how",
                property_dcid="undataAge",
                kind="value",
                value_dcids=["undata/AGE-Y0T4"],
                drafted_by="reply",
            ),
            id="exact-id-binds",
        ),
        pytest.param(
            ["undataAGE-Y0T4"],
            ["undata/AGE-Y0T4", "undata/AGE-Y5T9"],
            SlotBindingDraft(
                axis="how",
                property_dcid="undataAge",
                kind="value",
                value_dcids=["undata/AGE-Y0T4"],
                drafted_by="reply",
            ),
            id="separator-dropped-binds-the-taxonomy-value",
        ),
        pytest.param(
            ["CRSTypeOfFlow10"],
            ["undata/AGE-Y0T4"],
            SlotBindingDraft(
                axis="how",
                property_dcid="undataAge",
                kind="value",
                value_dcids=["CRSTypeOfFlow10"],
                drafted_by="reply",
            ),
            id="index-code-outside-taxonomy-binds",
        ),
        pytest.param(
            ["undataAGE-Y0T4"],
            ["undata/AGE-Y0T4", "undata/AGE_Y0T4"],
            SlotBindingDraft(
                axis="how", property_dcid="undataAge", kind="unbound", drafted_by="engine"
            ),
            id="ambiguous-normalised-match-unbinds",
        ),
        pytest.param(
            ["undataAGE-Y0T4", "Female"],
            ["undata/AGE-Y0T4", "undata/AGE-Y5T9"],
            SlotBindingDraft(
                axis="how",
                property_dcid="undataAge",
                kind="set",
                value_dcids=["undata/AGE-Y0T4", "Female"],
                drafted_by="reply",
            ),
            id="unmatched-value-binds-as-written",
        ),
    ],
)
def test_bind_universal_grounds_values_in_the_chosen_shapes_taxonomy(
    bind, written, taxonomy, expected
):
    """under5-mortality-ethiopia-2015-2020: the model wrote ``undataAGE-Y0T4`` for
    ``undata/AGE-Y0T4``. A value grounds by exact or unique normalised match, an ambiguous
    one opens its slot while the elected shape stays chosen, and one matching nothing binds
    as written for materialise to refuse as a value no member holds. A value the reply
    drafted stays the reply's draft once grounded, and a slot the engine opens is the
    engine's."""
    shapes = [
        _shape("other_shape", slot_keys=(_AGE_SLOT,), slot_taxonomy={"how:undataAge": ["x"]}),
        _shape("under5", slot_keys=(_AGE_SLOT,), slot_taxonomy={"how:undataAge": taxonomy}),
    ]
    output = elected_reply(
        1,
        bindings=[
            SlotBindingDraft(
                axis="how",
                property_dcid="undataAge",
                kind="value" if len(written) == 1 else "set",
                value_dcids=written,
            )
        ],
        also_plausible=None,
    )

    result, _ = _run(bind("under-five mortality", shapes, llm=_StubLLM(output)))

    assert result.chosen_shape_index == 1
    assert result.bindings == [expected]


def test_who_drafted_a_binding_stays_out_of_the_schema_the_model_fills_and_of_dumps():
    """`drafted_by` is the engine's own record: the model is not asked for it, and no dump
    of a draft, which cache keys and the wire read, carries it."""
    draft = SlotBindingDraft(
        axis="how", property_dcid="gender", kind="value", value_dcids=["Female"], drafted_by="reply"
    )

    assert "drafted_by" not in SlotBindingDraft.model_json_schema()["properties"]
    assert "drafted_by" not in json.dumps(_UniversalBindOutput.model_json_schema())
    assert "drafted_by" not in draft.model_dump()
    assert "drafted_by" not in draft.model_dump_json()


def test_bind_universal_picks_non_first_shape(bind):
    shapes = [
        _shape("wrong_shape", score=0.9),
        _shape("right_shape", score=0.6),
    ]
    output = elected_reply(1, bindings=[], also_plausible=None)
    result, _ = _run(bind("some variable", shapes, llm=_StubLLM(output)))
    assert result.chosen_shape_index == 1


def test_bind_universal_also_plausible_ranked(bind):
    shapes = [_shape("a"), _shape("b"), _shape("c")]
    output = elected_reply(0, bindings=[], also_plausible=[2, 1])
    result, _ = _run(bind("ambiguous variable", shapes, llm=_StubLLM(output)))
    assert result.also_plausible == [2, 1]


# CANNOT-BIND (ask) path


def test_bind_universal_off_topic_sets_ask(bind):
    shapes = [_shape("unrelated_shape")]
    output = _UniversalBindOutput(
        chosen_shape_index=0,
        bindings=[],
        also_plausible=None,
        ask="Variable 'weather in London' does not match any candidate shape.",
    )
    result, _ = _run(bind("weather in London", shapes, llm=_StubLLM(output)))
    assert result.ask is not None
    assert result.bindings == []


def test_bind_ask_triggers_no_data_in_resolve_variable():
    """When bind_universal returns ask, resolve_variable returns no_data(variable_not_resolved)."""
    import asyncio

    from search.engine.region import resolve_variable
    from tests.fixtures import FakeGraph, make_resolver

    class _AskLLM:
        async def generate_structured(self, *, prompt, system, schema):
            if schema.__name__ == "_UniversalBindOutput":
                return _UniversalBindOutput(
                    chosen_shape_index=0,
                    bindings=[],
                    also_plausible=None,
                    ask="Variable is completely off-topic.",
                ), None
            if schema is QuantityOutput:
                return NO_QUANTITY
            raise AssertionError(f"unexpected schema {schema.__name__!r}")

    graph = FakeGraph(
        nodes=_RESOLVE_NODES, detect=_RESOLVE_DETECT, resolve=_RESOLVE_RESOLVE, obs={}
    )

    result = asyncio.run(
        resolve_variable(
            "population",
            entities=["Kenya"],
            date_request=None,
            detect_query="population",
            role_query="population of Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_AskLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "variable_not_resolved"


class _OutOfRangeLLM:
    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        return _UniversalBindOutput(
            chosen_shape_index=99, bindings=[], also_plausible=None, ask=None
        ), None


class _SchemaRefusedLLM:
    """Answers as Gemini does when the model's JSON names a binding with no value: the
    schema refuses it, and the parsed response is empty."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        raise LLMOutputRefused(
            "LLM output failed schema validation; response.parsed is None", usage=_USAGE
        )


class _ForeignSlotLLM:
    """Elects the population shape, which has no slots, and binds `drafts` on it."""

    def __init__(self, drafts: list[SlotBindingDraft]) -> None:
        self.drafts = drafts

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        return elected_reply(0, bindings=self.drafts, also_plausible=None), _USAGE


_USAGE = {"input_tokens": 7, "output_tokens": 3, "cached_tokens": 0}


def _resolve_population(llm):
    from search.engine.region import resolve_variable
    from tests.fixtures import FakeGraph, make_resolver

    graph = FakeGraph(
        nodes=_RESOLVE_NODES,
        detect=_RESOLVE_DETECT,
        resolve=_RESOLVE_RESOLVE,
        obs={
            f"{_RESOLVE_SV}|country/KEN": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}
            ]
        },
    )
    return asyncio.run(
        resolve_variable(
            "population",
            entities=["Kenya"],
            date_request=None,
            detect_query="population",
            role_query="population of Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=llm,
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )


@pytest.mark.parametrize(
    "llm",
    [_OutOfRangeLLM(), _SchemaRefusedLLM()],
    ids=["out-of-range-index", "schema-refused"],
)
def test_a_bind_answer_naming_no_reading_refuses_variable_not_resolved(llm):
    """An out-of-range chosen_shape_index and an answer the bind schema refuses at parse
    each end the part `variable_not_resolved`, not an infrastructure failure."""
    result = _resolve_population(llm)
    assert result.status == "no_data"
    assert result.no_data_reason == "variable_not_resolved"


def test_a_bind_answer_the_schema_refuses_carries_its_usage_and_names_its_detail(caplog):
    """A binding that names no value refuses the part with the call's usage, warns
    `degenerate_binding_detected`, and logs the variable and the refusal's detail."""
    with caplog.at_level(logging.INFO, logger="search.engine.region.phase_rank_bind"):
        result = _resolve_population(_SchemaRefusedLLM())

    assert (result.status, result.no_data_reason) == ("no_data", "variable_not_resolved")
    assert result.llm_usage == _USAGE
    assert [w.code for w in result.warnings] == ["degenerate_binding_detected"]
    (record,) = [r for r in caplog.records if "bind answer refused" in r.getMessage()]
    assert "'population'" in record.getMessage()
    assert "response.parsed is None" in record.getMessage()


def _value(prop: str, value: str) -> SlotBindingDraft:
    return SlotBindingDraft(axis="how", property_dcid=prop, kind="value", value_dcids=[value])


@pytest.mark.parametrize(
    "drafts",
    [
        [_value("gender", "Female")],
        [_value("expenditureType", "EducationExpenditure"), _value("expensor", "Government")],
    ],
    ids=["one-stray-value", "every-draft-stray"],
)
def test_a_value_bound_on_a_slot_the_elected_shape_lacks_refuses_naming_it(drafts):
    """The bind elects the population shape, which has no slots, and binds values on
    properties it lacks. Each stray value is dropped and named as a term the answer
    leaves unapplied, so the part refuses `qualifier_not_bound` with the population
    reading flagged, never definite. The refusal names exactly the terms the flagged
    reading's trace records, and none of them departs beside it."""
    result = _resolve_population(_ForeignSlotLLM(drafts))

    assert (result.status, result.no_data_reason) == ("no_data", "qualifier_not_bound")
    assert result.flagged_match is not None
    (flagged,) = result.flagged_match
    assert [m.ref.ref for m in flagged.resolved.members] == [_RESOLVE_SV]
    assert [
        (term.disposition, term.evidence, [ref.ref for ref in term.named])
        for term in result.unbound_qualifiers or ()
    ] == [("unbound", "model", draft.value_dcids) for draft in drafts]
    assert list(flagged.resolved.trace.query_qualifiers) == list(result.unbound_qualifiers or ())
    assert result.departure_facts == {}


def test_a_stray_value_is_met_by_an_answer_that_binds_it():
    """A value the bind drafted on a slot the elected shape lacks is judged as a value the
    request names (`_stray_requirements`): the elected population answer leaves "Female"
    unmet, and an answer by sex that binds it meets it."""
    population, by_sex = _shape("population"), _shape("by_sex", slot_keys=(_GENDER_SLOT,))
    state = SimpleNamespace(capped_shapes=[population], label_dcids=lambda values: values)
    stray = [_value("gender", "Female")]

    requirements = _run(_stray_requirements(state, stray))
    judgement = judge(requirements, [(population, []), (by_sex, stray)])

    assert [t.text for t in judgement.terms(0)] == ["Female"]
    assert judgement.terms(1) == ()


# The prompt's member sample uses recall members, even when the index has more.


def test_bind_prompt_sample_stays_on_the_recall_pool_for_an_index_carried_shape():
    """The prompt sample contains one recalled member despite hundreds in the index."""
    index = get_index()
    target = ("FinancialFlow", "grantEquivalent", "measuredValue", "Nominal", None)
    population = [
        row.dcid
        for row in index.statvars
        if (
            row.population_type or "",
            row.measured_property,
            row.stat_type or "",
            row.measurement_qualifier,
            row.measurement_denominator,
        )
        == target
    ]
    assert len(population) > 100
    recalled_sv = population[0]

    shape = ShapeDraft(
        shape_id="test_bind_prompt_sample_shape",
        label="test shape",
        pop_type_dcid=target[0],
        meas_prop_dcid=target[1],
        stat_type_dcid=target[2],
        meas_qual_dcid=target[3],
        meas_denom_dcid=target[4],
        slot_keys=(),
        sv_arc_facts={recalled_sv: {}},
        sv_arc_facts_complete=False,
    )

    prompt = _build_universal_bind_prompt("test variable", [shape])

    assert f"sample_member_dcids: ['{recalled_sv}']" in prompt
    # Index-only members stay out of the prompt.
    for other_sv in population[1:20]:
        assert other_sv not in prompt


# Prompt controls for withheld axes and dropped qualifiers.

_MATERIAL_SLOT = SlotKeyDraft(axis="what", property_dcid="material", label="material")
_FINISH_SLOT = SlotKeyDraft(axis="how", property_dcid="finish", label="finish")


_HEAD_WIDGET_PROMPT = (
    "Variable phrase: widget variable\n\n"
    "CANDIDATE SHAPES (1 total):\n\n"
    "Shape 0:\n"
    "  label: widget\n"
    "  origin: our Data Commons\n"
    "  family: standard Data Commons\n"
    "  five_tuple: populationType=Person, measuredProperty=count, statType=measuredValue\n"
    "  constraint_slots:\n"
    "    axis=what, property_dcid=material (material)\n"
    "      taxonomy: ['Plastic', 'Metal']\n"
    "    axis=how, property_dcid=finish (finish)\n"
    "      taxonomy: ['Matte', 'Glossy']\n"
)


def test_no_dropped_phrases_keeps_a_recall_sourced_prompt_byte_identical_to_head():
    """An empty dropped-phrase list preserves the prompt bytes."""
    shape = _shape(
        "widget",
        slot_keys=(_MATERIAL_SLOT, _FINISH_SLOT),
        slot_taxonomy={"what:material": ["Plastic", "Metal"], "how:finish": ["Matte", "Glossy"]},
    )
    assert _build_universal_bind_prompt("widget variable", [shape]) == _HEAD_WIDGET_PROMPT
    # Explicit empty dropped_qualifier_phrases is the exact same call.
    assert (
        _build_universal_bind_prompt("widget variable", [shape], dropped_qualifier_phrases=())
        == _HEAD_WIDGET_PROMPT
    )


def test_a_recall_sourced_shapes_slot_shows_its_value_labels():
    """A walked or held-family shape labels its values from `value_labels`, as an
    index-sourced shape does from its code rows; a label repeating its dcid is left out."""
    shape = dataclasses.replace(
        _shape(
            "widget",
            slot_keys=(_MATERIAL_SLOT,),
            slot_taxonomy={"what:material": ["undata/MAT-P", "undata/MAT-M"]},
        ),
        value_labels={"undata/MAT-P": "Plastic", "undata/MAT-M": "undata/MAT-M"},
    )

    prompt = _build_universal_bind_prompt("widget variable", [shape])

    assert (
        "      taxonomy: ['undata/MAT-P', 'undata/MAT-M']\n      labels: undata/MAT-P = Plastic\n"
    ) in prompt


def test_each_menu_shape_states_whether_it_has_data_at_the_named_places():
    """One line per shape, after its origin: yes, no, or unknown where no read settled it.
    A part naming no place passes no availability, and the shapes carry no such line."""
    shapes = [_shape("a"), _shape("b"), _shape("c")]
    prompt = _build_universal_bind_prompt("widget", shapes, availability=[True, False, None])

    assert [
        line for line in prompt.splitlines() if line.startswith("  data at the named places:")
    ] == [
        "  data at the named places: yes",
        "  data at the named places: no",
        "  data at the named places: unknown",
    ]
    assert "data at the named places" not in _build_universal_bind_prompt("widget", shapes)


@pytest.mark.parametrize(
    ("denominator", "unresolved", "line"),
    [
        ("Count_Person", False, "measurementDenominator=per person"),
        ("PerCapita", True, "measurementDenominator=per person"),
        ("area", False, "measurementDenominator=per area"),
        ("Count_Person_15OrMoreYears", False, "measurementDenominator=Count_Person_15OrMoreYears"),
        (None, True, "measurementDenominator=(present, unresolved)"),
    ],
    ids=["count-person", "per-capita-provisional", "area", "statvar", "dcid-less"],
)
def test_a_schema_term_denominator_shows_its_meaning(denominator, unresolved, line):
    """`Count_Person` and the ProvisionalNode `PerCapita` show the model one per-person
    base, so neither spelling reads as the more precise reading. A StatVar denominator
    shows its dcid, and an arc with no dcid stays unresolved."""
    shape = dataclasses.replace(
        _shape("gdp"),
        meas_qual_dcid="Nominal",
        meas_denom_dcid=denominator,
        meas_denom_unresolved=unresolved,
    )
    prompt = _build_universal_bind_prompt("GDP per capita", [shape])

    assert f"    measurementQualifier=Nominal, {line}" in prompt.splitlines()


_SOURCE_SLOT = SlotKeyDraft(axis="how", property_dcid="activitySource", label="activitySource")
_BASIS_SLOT = SlotKeyDraft(axis="how", property_dcid="priceBasis", label="priceBasis")


_GDP_VALUE = "GrossDomesticProduction"


@pytest.mark.parametrize(
    ("top", "picked", "bound", "availability", "elected"),
    [
        pytest.param({}, {}, _GDP_VALUE, (), 0, id="stated-alike"),
        pytest.param({}, {}, _GDP_VALUE, (True, True), 0, id="stated-alike-with-data"),
        pytest.param(
            {"meas_denom_dcid": "Count_Person_15OrMoreYears", "meas_denom_unresolved": False},
            {},
            _GDP_VALUE,
            (),
            1,
            id="statvar-denominator",
        ),
        pytest.param(
            {"meas_denom_dcid": "PerHousehold"},
            {"meas_denom_dcid": "PerDwelling", "meas_denom_unresolved": True},
            _GDP_VALUE,
            (),
            1,
            id="unresolved-denominators",
        ),
        pytest.param({"meas_qual_dcid": "RealValue"}, {}, _GDP_VALUE, (), 1, id="qualifier"),
        pytest.param({"label": "Regional GDP per capita"}, {}, _GDP_VALUE, (), 1, id="label"),
        pytest.param(
            {"declared_roles": ("observationAbout", "financingRecipient")},
            {},
            _GDP_VALUE,
            (),
            1,
            id="roles",
        ),
        pytest.param({"slot_keys": (_SOURCE_SLOT, _BASIS_SLOT)}, {}, _GDP_VALUE, (), 1, id="slot"),
        pytest.param({}, {}, "NetDomesticProduction", (), 1, id="bound-value-it-lacks"),
        pytest.param(
            {"slot_taxonomy": {"how:activitySource": ["CRSCooperationModalityA01"]}},
            {
                "slot_taxonomy": {
                    "how:activitySource": ["CRSCooperationModalityA", "CRSCooperationModalityA01"]
                }
            },
            "CRSCooperationModalityA",
            (),
            1,
            id="bound-parent-it-holds-only-a-child-of",
        ),
        pytest.param({}, {}, None, (), 1, id="open-slot-on-another-taxonomy"),
        pytest.param(
            {"slot_taxonomy": {"how:activitySource": ["NetDomesticProduction", _GDP_VALUE]}},
            {},
            None,
            (),
            0,
            id="open-slot-on-one-taxonomy",
        ),
        pytest.param({}, {}, _GDP_VALUE, (False, True), 1, id="availability"),
    ],
)
def test_the_rank_elects_among_shapes_the_menu_states_alike(
    top, picked, bound, availability, elected
):
    """The model elects the second of two shapes. The first divides by the ProvisionalNode
    `PerCapita` and the second by `Count_Person`, which the menu prints as one per-person
    base, and the value the model binds is in both taxonomies, so its choice between them
    says nothing and the higher-ranked shape is elected. A difference the menu states
    leaves the model's choice: another denominator meaning, qualifier, label, role class
    or slot, a bound value the first shape lacks or holds only a child code of, where the
    binding would read the child alone, a slot left open over another taxonomy,
    or another availability, as where only the model's choice has data at the named
    places. Two unresolved arcs naming no schema term print alike and are one reading
    only as the same arc."""
    chosen = dataclasses.replace(
        _shape(
            "gdp-per-capita",
            slot_keys=(_SOURCE_SLOT,),
            slot_taxonomy={"how:activitySource": [_GDP_VALUE, "NetDomesticProduction"]},
        ),
        meas_qual_dcid="Nominal",
        meas_denom_dcid="Count_Person",
    )
    first = dataclasses.replace(
        chosen,
        **{
            "slot_taxonomy": {"how:activitySource": [_GDP_VALUE]},
            "meas_denom_dcid": "PerCapita",
            "meas_denom_unresolved": True,
        }
        | top,
    )
    chosen = dataclasses.replace(chosen, **picked)
    output = elected_reply(
        1, bindings=[_value("activitySource", bound)] if bound else [], also_plausible=None
    )

    assert rank_elected([first, chosen], output, availability=availability) == elected


def test_dropped_qualifiers_block_renders_only_when_phrases_exist():
    shape = _shape("widget")
    without = _build_universal_bind_prompt("widget variable", [shape])
    assert "DROPPED QUALIFIERS" not in without

    with_phrases = _build_universal_bind_prompt(
        "widget variable", [shape], dropped_qualifier_phrases=("across all flows",)
    )
    assert "DROPPED QUALIFIERS" in with_phrases
    assert "across all flows" in with_phrases
    assert "requested_all" in with_phrases
    assert "requested_all_phrase" in with_phrases


def test_withheld_measure_axes_drop_an_index_sourced_shapes_slots_and_keep_the_default():
    shape = dataclasses.replace(
        _shape(
            "widget",
            slot_keys=(_MATERIAL_SLOT, _FINISH_SLOT),
            slot_taxonomy={
                "what:material": ["Plastic", "Metal"],
                "how:finish": ["Matte", "Glossy"],
            },
        ),
        slot_taxonomy_index_sourced=True,
    )

    withheld = _build_universal_bind_prompt(
        "widget variable", [shape], withheld_measure_axes=frozenset({"material"})
    )
    assert "property_dcid=material" not in withheld
    assert "property_dcid=finish" in withheld

    with_default = _build_universal_bind_prompt("widget variable", [shape])
    with_explicit_empty = _build_universal_bind_prompt(
        "widget variable", [shape], withheld_measure_axes=frozenset()
    )
    assert with_default == with_explicit_empty
    assert "property_dcid=material" in with_default
    assert "property_dcid=finish" in with_default


def test_withheld_measure_axes_leave_a_recall_sourced_shapes_slots_for_the_binder():
    """No election reads a recall-sourced shape's members, so the binder binds the axis."""
    shape = _shape(
        "widget",
        slot_keys=(_MATERIAL_SLOT, _FINISH_SLOT),
        slot_taxonomy={"what:material": ["Plastic", "Metal"], "how:finish": ["Matte", "Glossy"]},
    )

    withheld = _build_universal_bind_prompt(
        "widget variable", [shape], withheld_measure_axes=frozenset({"material"})
    )

    assert withheld == _HEAD_WIDGET_PROMPT


# The bind response refuses a value or set binding that names no dcid at parse.


@pytest.mark.parametrize(
    ("kind", "value_dcids", "refused"),
    [
        pytest.param("value", [], True, id="value-naming-nothing"),
        pytest.param("set", [], True, id="set-naming-nothing"),
        pytest.param("value", [""], True, id="value-naming-an-empty-dcid"),
        pytest.param("value", ["Female"], False, id="value-naming-one"),
        pytest.param("set", ["Male", "Female"], False, id="set-naming-two"),
        pytest.param("unbound", [], False, id="unbound"),
        pytest.param("absent", [], False, id="absent"),
    ],
)
def test_the_bind_response_refuses_a_binding_naming_no_value(kind, value_dcids, refused):
    """The degenerate gender draft of "unemployment rate by gender" (kind value, no dcid)
    fails the response's parse, so no stage reads it as an open slot."""
    response = {
        "chosen_shape_index": 0,
        "bindings": [
            {"axis": "how", "property_dcid": "gender", "kind": kind, "value_dcids": value_dcids}
        ],
    }
    if refused:
        with pytest.raises(ValidationError, match="must name its value dcids"):
            _UniversalBindOutput.model_validate(response)
    else:
        _UniversalBindOutput.model_validate(response)


# The frame block, the role-bearing family and the not-electable list.

_GHANA_DEBT_LEVEL = ShapeDraft(
    shape_id="ids-debt-gg-level",
    label="Outstanding public or publicly guaranteed external debt owed by the general "
    "government (current US$)",
    pop_type_dcid="FinancialPosition",
    meas_prop_dcid="amount",
    stat_type_dcid="measuredValue",
    meas_qual_dcid="Nominal",
    meas_denom_dcid=None,
    slot_keys=(),
)


def _ghana_debt_frame(
    variable: str = "debt to GDP ratio",
    *,
    stated_form: QuantityForm | None = None,
    named_denominator: NamedDenominator | None = None,
):
    """The frame of the parsed part "debt to GDP ratio" for Ghana, latest year (design
    worked example, door replay A15), or of `variable` with the form and base the quantity
    read states."""
    frame = door_frame(
        door="parsed",
        variable=variable,
        entities=["Ghana"],
        entity_roles={},
        member_entities=frozenset(),
        pins={},
        date_request=DateRequest(window=None, latest=True),
        unit_text=None,
        scope=None,
        operations=(),
        open_axes=frozenset(),
    )
    return with_recall(
        frame,
        dropped_phrases=(),
        matches={},
        ties=(),
        readings=(),
        requirement=with_stated_form(quantity_requirement([variable]), stated_form),
        stated_form=stated_form,
        named_denominator=named_denominator,
        measure_class=None,
        place_sets={},
        role_bearing=False,
        slot_defaults=(),
    )


@pytest.mark.parametrize(
    ("variable", "stated_form", "named_denominator", "normalisation"),
    [
        ("debt to GDP ratio", None, None, "  form: rate [reader]\n"),
        (
            "debt to GDP",
            "rate",
            NamedDenominator(ref=GraphRef(ref="Amount_GDP", label="GDP"), base_words="GDP"),
            "  form: rate [model]\n  base: GDP [model]\n",
        ),
        (
            "debt to export earnings",
            "rate",
            NamedDenominator(ref=None, base_words="export earnings"),
            "  form: rate [model]\n  base: export earnings [model]\n",
        ),
    ],
    ids=["reader-form", "model-form-and-listed-base", "model-form-and-unlisted-base"],
)
def test_the_frame_opens_the_prompt_as_fixed_context(
    variable, stated_form, named_denominator, normalisation
):
    """The form line says who stated the form: a wording pattern ("ratio"), or the quantity
    read where no pattern names one, which also names the base by its graph name, else by
    the phrase's words."""
    frame = _ghana_debt_frame(
        variable, stated_form=stated_form, named_denominator=named_denominator
    )
    prompt = _build_universal_bind_prompt(variable, [_shape("widget")], frame=frame)

    assert prompt.startswith(
        f"Variable phrase: {variable}\n\n"
        "FRAME (what the query asks besides the variable phrase; [who stated it]):\n"
        "  place: Ghana (direction undirected, not stated)\n"
        "  years: latest [caller]\n"
        f"{normalisation}"
        "  operations: none [caller]\n"
        "  open roles: none [caller]\n"
        "\n"
        "CANDIDATE SHAPES (1 total):\n"
    )


def test_a_stated_direction_and_a_rank_operation_reach_the_frame_block():
    """ "Kenya's top 5 creditors" posted with Kenya as recipient and the creditor axis open
    (design worked example)."""
    frame = door_frame(
        door="parsed",
        variable="external debt by creditor",
        entities=["Kenya"],
        entity_roles={"Kenya": (DoorRole(axis="destination", span="Kenya", stated=True),)},
        member_entities=frozenset(),
        pins={},
        date_request=None,
        unit_text=None,
        scope=None,
        operations=(),
        open_axes=frozenset({"origin"}),
    )

    prompt = _build_universal_bind_prompt("external debt by creditor", [_shape("w")], frame=frame)

    assert "  place: Kenya (direction destination, stated)\n" in prompt
    assert "  years: all [default]\n" in prompt
    assert "  open roles: origin [caller]\n" in prompt


_AFRICAN_COUNTRIES = PlaceSet(
    name="African countries",
    kind="group",
    node=None,
    members=tuple(f"country/A{i:02d}" for i in range(54)),
    reading="members",
)
_GHANA = PlaceSet(name="Ghana", kind="place", node=PlaceNode("country/GHA", "Country"))


@pytest.mark.parametrize(
    ("place_set", "line"),
    [
        (
            _AFRICAN_COUNTRIES,
            "  place: African countries (54 members, read as members; "
            "direction destination, stated)\n",
        ),
        (_GHANA, "  place: country/GHA (direction destination, stated)\n"),
    ],
    ids=["set-by-name-and-count", "single-place-by-dcid"],
)
def test_a_frame_place_shows_a_set_by_its_name_and_count_and_one_place_by_its_dcid(place_set, line):
    """A set of places reaches the FRAME block as its name, member count and reading, with
    no member dcid, so the block stays within a few hundred tokens for a 54-member set."""
    frame = door_frame(
        door="parsed",
        variable="aid",
        entities=[place_set.name],
        entity_roles={
            place_set.name: (DoorRole(axis="destination", span=place_set.name, stated=True),)
        },
        member_entities=frozenset(),
        pins={},
        date_request=None,
        unit_text=None,
        scope=None,
        operations=(),
        open_axes=frozenset(),
    )
    frame = with_recall(
        frame,
        dropped_phrases=(),
        matches={},
        ties=(),
        readings=(),
        requirement=quantity_requirement(["aid"]),
        measure_class=None,
        place_sets={place_set.name: place_set},
        role_bearing=False,
        slot_defaults=(),
    )

    prompt = _build_universal_bind_prompt("aid", [_shape("w")], frame=frame)

    assert line in prompt
    assert "country/A00" not in prompt


def test_a_role_bearing_shape_shows_its_roles_and_no_shape_shows_a_score():
    flows = dataclasses.replace(
        _shape("crs-oda"), declared_roles=("financingProvider", "financingRecipient")
    )

    prompt = _build_universal_bind_prompt("aid to Kenya", [flows, _shape("population")])

    assert "  family: flow with roles financingProvider, financingRecipient\n" in prompt
    assert prompt.count("  family: standard Data Commons\n") == 1
    assert "retrieval_score" not in prompt


def test_readings_refused_on_a_pinned_part_follow_the_menu_as_not_electable():
    """The Ghana debt-to-GDP menu: the dollar level fails the rate form "ratio" names."""
    prompt = _build_universal_bind_prompt(
        "debt to GDP ratio",
        [_shape("widget")],
        refused=[
            Interpretation(
                shape=_GHANA_DEBT_LEVEL,
                reason="denominator_not_available",
                pinned_failure="form",
            )
        ],
    )

    assert prompt.endswith(
        "NOT ELECTABLE (each fails a part the query pins; never elect one, and when you ask, "
        "nearest_refused_index names one):\n"
        "  0: Outstanding public or publicly guaranteed external debt owed by the general "
        "government (current US$) (fails the form the query asks for)\n"
    )
