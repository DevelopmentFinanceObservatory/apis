"""The binder seam at `_phase_bind`: the engine elects through `ResolveState.binder`, any
binder answering alike leaves the part in the same state, and the Gemini binder sends the
request `bind_universal` sends."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import cast

import pytest

from code_index.types import BroadReading
from entity_resolution import Admission, EntityResolver
from search.engine.bind import (
    ShapeRating,
    SlotBindingDraft,
    _UniversalBindOutput,
    bind_schema,
    bind_universal,
    ground_bind_output,
)
from search.engine.binders import (
    Binder,
    FallbackBinder,
    GeminiBinder,
    JevBinder,
    QuantityRequest,
    ShadowBinder,
)
from search.engine.binders.house_policy import PolicyPool
from search.engine.binders.jev_client import JevError
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.errors import GraphInfraError
from search.engine.interpret import Recall
from search.engine.interpretation import Interpretation
from search.engine.qualifier_match import declared_measures_named
from search.engine.quantity import serves_every_form
from search.engine.quantity_read import QuantityOutput
from search.engine.region import candidate_collapse, phase_rank_bind
from search.engine.region.candidate_collapse import _record_unheld_values, _unheld_requirement
from search.engine.region.phase_rank_bind import (
    _logged_verdicts,
    _phase_bind,
    _read_quantity_answer,
    _read_stated_quantity,
    _stand_in_declined,
    judge_value_requirements,
)
from search.engine.region.requirements import settle_answer, settle_candidate_set
from search.engine.region.state import (
    Requirement,
    ResolveState,
    SlotDefaultDisclosure,
    ValueRule,
)
from search.engine.retrieval.retrieve import NoDataDraft
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.engine.source_code import SeriesCodeMembers
from search.models import GraphRef, Operation, ResolvedQualifier
from tests.engine._assemble_fixtures import _make_spec
from tests.engine._binders import FakeAsk, FixedBinder, elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._quantity import NO_QUANTITY

_GENDER_SLOT = SlotKeyDraft(axis="how", property_dcid="gender", label="gender")
_USAGE = {"input_tokens": 11, "output_tokens": 5, "cached_tokens": 0}


def _shape(shape_id: str, *slots: SlotKeyDraft) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=slots,
        slot_taxonomy={"how:gender": ["Male", "Female"]} if slots else {},
    )


_MENU = [_shape("population"), _shape("population_by_sex", _GENDER_SLOT)]

# The reply elects the shape by sex, binds "Female" on it, rates it, and names the other as
# plausible.
_REPLY = elected_reply(
    1,
    bindings=[
        SlotBindingDraft(axis="how", property_dcid="gender", kind="value", value_dcids=["Female"])
    ],
    also_plausible=[0],
)


class _RecordingLLM:
    """Answers the bind call with `_REPLY` and records each bind request it is sent."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, type]] = []

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        self.requests.append((prompt, system, schema))
        return _REPLY, _USAGE


class _NoCallLLM:
    async def generate_structured(self, *, prompt, system, schema):
        raise AssertionError(f"no model call expected; got {schema.__name__!r}")


def _state(
    *,
    llm: SupportsLLM,
    binder: Binder | None = None,
    variable: str = "female population",
    operations: tuple[Operation, ...] = (),
) -> ResolveState:
    """A part whose seated menu is `_MENU`, asking for `variable` under `operations`. The
    services bind does not read are unset."""
    state = ResolveState(
        variable=variable,
        entities=[],
        date_request=None,
        detect_query=variable,
        role_query=variable,
        operations=operations,
        pac=True,
        graph=cast(EngineGraphClient, None),
        resolver=cast(EntityResolver, None),
        admission=cast(Admission, None),
        llm=llm,
        sdmx=cast(EngineSdmxClient, None),
        deadline=None,
        pre_resolved=None,
        variable_index=0,
        emit=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        unit_text=None,
        materialise_graph=None,
        materialise_sdmx=None,
        timing={},
        warnings=[],
        rcl=Recall(candidates=[], resolutions={}),
        coverage=empty_coverage_snapshot(),
    )
    if binder is not None:
        state.binder = binder
    state.capped_shapes = list(_MENU)
    return state


def test_a_binder_answering_as_gemini_leaves_the_part_in_the_same_state():
    """A binder whose reply, with the shared post-steps applied, is the Gemini stub's
    drives `_phase_bind` to the same bind result, usage and menu state, with no model
    call of its own. The request it gets carries the part's menu and query text."""
    gemini = _state(llm=_RecordingLLM())
    binder = FixedBinder(ground_bind_output(_REPLY, _MENU), usage=_USAGE)
    fixed = _state(llm=_NoCallLLM(), binder=binder)

    assert asyncio.run(_phase_bind(gemini)) is None
    assert asyncio.run(_phase_bind(fixed)) is None

    assert fixed.bind_result == gemini.bind_result
    assert gemini.bind_result.bindings[0].drafted_by == "reply"
    assert (fixed.var_usage, fixed.menu_availability, fixed.stray_requirements) == (
        gemini.var_usage,
        gemini.menu_availability,
        gemini.stray_requirements,
    )
    (request,) = binder.requests
    assert (request.variable, request.shapes, request.query_text) == (
        "female population",
        _MENU,
        "female population",
    )
    assert [s.bind_decision.policy_pool for s in (gemini, fixed) if s.bind_decision] == [
        PolicyPool(measuring=(1,)),
        PolicyPool(measuring=(1,)),
    ]
    assert [s.bind_decision.binder for s in (gemini, fixed) if s.bind_decision] == [
        "gemini",
        "fixed",
    ]


def test_the_gemini_binder_sends_the_request_bind_universal_sends():
    """Through the seam, the bind call's prompt, system text and schema are the ones
    `bind_universal` sends for the same menu and frame, and the schema is the bind schema
    bounded to the menu."""
    llm = _RecordingLLM()
    state = _state(llm=llm)
    assert asyncio.run(_phase_bind(state)) is None
    direct = _RecordingLLM()
    asyncio.run(bind_universal("female population", list(_MENU), frame=state.frame, llm=direct))

    assert llm.requests == direct.requests
    (request,) = llm.requests
    assert request[2] is bind_schema(len(_MENU))


class _ReplyLLM:
    """Answers the bind call with `reply`."""

    def __init__(self, reply: _UniversalBindOutput) -> None:
        self.reply = reply

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        return self.reply, _USAGE


@pytest.mark.parametrize(
    ("measuring", "declines"),
    [
        ([], True),
        ([ShapeRating(index=0, rating=1.0)], True),
        ([ShapeRating(index=1, rating=0.0), ShapeRating(index=0, rating=1.0)], True),
        ([ShapeRating(index=1, rating=0.11)], True),
        ([ShapeRating(index=1, rating=0.5)], True),
        ([ShapeRating(index=1, rating=0.6)], False),
    ],
    ids=[
        "rates-nothing",
        "pick-unrated",
        "pick-rated-0",
        "pick-just-above-the-pool-floor",
        "pick-below-the-decline-floor",
        "pick-at-the-decline-floor",
    ],
)
def test_a_reply_rating_its_pick_below_the_decline_floor_declines(measuring, declines):
    """A reply that leaves its pick out of `measuring`, an empty one included, or rates it
    below `DECLINE_FLOOR`, says the pick measures something else: the part refuses
    `variable_not_resolved` as a reply that asked does. A pick rated at the floor stands."""
    reply = _REPLY.model_copy(update={"measuring": measuring})
    state = _state(llm=_ReplyLLM(reply))

    result = asyncio.run(_phase_bind(state))

    if declines:
        assert result is not None
        assert result.no_data_reason == "variable_not_resolved"
        assert state.bind_result.ask is not None
        assert state.bind_result.bindings == []
    else:
        assert result is None
        assert state.bind_result.ask is None
        assert state.bind_result.chosen_shape_index == 1


class _SlowRecordingLLM(_RecordingLLM):
    """`_RecordingLLM` answering after a pause, so a call started beside it ends first."""

    async def generate_structured(self, *, prompt, system, schema):
        await asyncio.sleep(0.01)
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


@pytest.mark.parametrize(
    ("binder", "served_by"),
    [
        pytest.param(
            FallbackBinder(
                primary=JevBinder(ask=FakeAsk(fail=JevError(None, "timeout"))),
                fallback=GeminiBinder(),
            ),
            "gemini",
            id="jev-failed",
        ),
        pytest.param(
            FallbackBinder(primary=JevBinder(ask=FakeAsk()), fallback=GeminiBinder()),
            "jev",
            id="jev",
        ),
        pytest.param(
            ShadowBinder(served=GeminiBinder(), shadow=JevBinder(ask=FakeAsk()), sample=1.0),
            "gemini",
            id="shadow",
        ),
    ],
)
def test_every_binder_runs_the_house_policies_once_per_election(caplog, binder, served_by):
    """`_phase_bind` holds the house policies to one run on a call that elects a shape,
    which it asserts, under the fallback, Jev and the shadow, whose Jev call answers
    before Gemini's and elects on its own copy of the policy step."""
    caplog.set_level(logging.INFO, logger="search.engine.binders")
    caplog.set_level(logging.INFO, logger="search.engine.region.phase_rank_bind")
    state = _state(llm=_SlowRecordingLLM(), binder=binder)

    assert asyncio.run(_phase_bind(state)) is None

    assert state.bind_decision is not None
    assert state.bind_decision.binder == served_by
    assert state.bind_decision.policy_choice is not None
    if isinstance(binder, ShadowBinder):
        # The shadow line joins the part's bind_decision line on its part and pass.
        (shadow,), (decision,) = (
            [
                json.loads(r.getMessage().removeprefix(prefix))
                for r in caplog.records
                if r.getMessage().startswith(prefix)
            ]
            for prefix in ("binder_shadow ", "bind_decision ")
        )
        assert (shadow["call"], shadow["failure"], shadow["menu"]) == (
            "bind",
            None,
            decision["menu"],
        )
        assert (shadow["part"], shadow["pass"]) == (decision["part"], decision["pass"])


class _QuantityRecordingBinder(GeminiBinder):
    """Records each quantity request and reads no form."""

    def __init__(self) -> None:
        self.quantity_requests: list[QuantityRequest] = []

    async def read_quantity(self, request, *, llm):
        self.quantity_requests.append(request)
        return NO_QUANTITY


_GNI_REF = GraphRef(ref="GrossNationalIncome", label="Gross national income")
_EXPORTS_REF = GraphRef(ref="Exports", label="Exports")
_LISTED = [_GNI_REF, _EXPORTS_REF]


def _flow(shape_id: str, denominator: str | None = None) -> ShapeDraft:
    return dataclasses.replace(
        _shape(shape_id),
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        meas_denom_dcid=denominator,
    )


# ODA, and ODA divided by GNI and by exports, whose denominators the quantity read lists.
_FLOW_SHAPES = [
    _flow("oda"),
    _flow("oda_per_gni", _GNI_REF.ref),
    _flow("oda_per_exports", _EXPORTS_REF.ref),
]
_SHARE_OF_GNI = Operation(verb="share_of", phrase="share of GNI")
_NONE_OF_THESE = "none of these"


def _read(form: str | None = None, base: str | None = None, index: int | None = None):
    return QuantityOutput(stated_form=form, base_words=base, denominator_index=index)


def _denominator(state: ResolveState) -> str | None:
    """The ref of the part's named denominator, `_NONE_OF_THESE` for one the list does not
    hold, None for none."""
    named = state.named_denominator
    if named is None:
        return None
    return named.ref.ref if named.ref is not None else _NONE_OF_THESE


@pytest.mark.parametrize(
    ("variable", "operations", "read", "named", "denominator", "base"),
    [
        # The operation's phrase names the share and the base, so the part is ODA.
        pytest.param(
            "ODA as a share of GNI",
            (_SHARE_OF_GNI,),
            _read("share", "GNI", 0),
            set(),
            None,
            "GNI",
            id="share-and-base-in-the-phrase",
        ),
        pytest.param(
            "share of bilateral ODA",
            (Operation(verb="share_of", phrase="share of bilateral ODA"),),
            _read("share"),
            set(),
            None,
            None,
            id="whole-variable-phrase",
        ),
        # "rate" lies outside the phrase, so the series keeps it.
        pytest.param(
            "unemployment rate of women relative to men",
            (Operation(verb="share_of", phrase="relative to"),),
            _read(),
            {"rate"},
            None,
            None,
            id="series-rate-outside-the-phrase",
        ),
        # "rate" is the growth's; the share of GNI, outside its phrase, is the series'.
        pytest.param(
            "growth rate of ODA as a share of GNI",
            (Operation(verb="growth", phrase="growth rate"),),
            _read("share", "GNI", 0),
            {"share"},
            _GNI_REF.ref,
            "GNI",
            id="growth-rate-of-a-published-ratio",
        ),
        pytest.param(
            "growth rate of ODA as a share of GNI",
            (Operation(verb="growth", phrase="growth rate"), _SHARE_OF_GNI),
            _read("share", "GNI", 0),
            set(),
            None,
            "GNI",
            id="growth-rate-of-a-computed-share",
        ),
        pytest.param(
            "growth of the unemployment rate",
            (Operation(verb="growth", phrase="growth of"),),
            _read("rate"),
            {"rate"},
            None,
            None,
            id="growth-of-a-published-rate",
        ),
        pytest.param(
            "debt service as a percentage of exports, relative to Ghana",
            (Operation(verb="share_of", phrase="relative to"),),
            _read("share", "exports", 1),
            {"share"},
            _EXPORTS_REF.ref,
            "exports",
            id="published-ratio-beside-the-phrase",
        ),
        pytest.param(
            "aid in total aid",
            (Operation(verb="share_of", phrase="aid in total aid"),),
            _read("share", "total aid", 2),
            set(),
            None,
            "total aid",
            id="stated-share-on-a-base-in-the-phrase",
        ),
        # A phrase that leaves the whole's words outside gives the base to the series,
        # which a level then cannot serve.
        pytest.param(
            "aid as a share of aid to all developing countries",
            (Operation(verb="share_of", phrase="share"),),
            _read("share", "aid to all developing countries", 2),
            {"share"},
            _NONE_OF_THESE,
            "aid to all developing countries",
            id="whole-outside-a-short-phrase",
        ),
        pytest.param(
            "ODA as a share of GNI",
            (Operation(verb="share_of", phrase="proportion of GNI"),),
            _read(),
            {"share"},
            None,
            None,
            id="phrase-absent-from-the-variable",
        ),
        pytest.param(
            "share of aid per capita",
            (Operation(verb="share_of", phrase="share of aid per capita"),),
            _read(),
            {"per_person"},
            None,
            None,
            id="per-capita-in-the-phrase",
        ),
        pytest.param(
            "ODA growth rate",
            (),
            _read("rate"),
            {"rate"},
            None,
            None,
            id="published-growth-rate",
        ),
    ],
)
def test_an_operation_owns_the_forms_and_base_its_phrase_names_in_the_variable(
    variable: str,
    operations: tuple[Operation, ...],
    read: QuantityOutput,
    named: set[str],
    denominator: str | None,
    base: str | None,
):
    """An operation owns the words of its phrase where the phrase occurs in the variable,
    and the series owns the rest: a form or base named inside the phrase is the
    operation's, so the series is not held to it, and one named outside stays the
    series'. A phrase the variable does not hold owns nothing."""
    state = _state(llm=_NoCallLLM(), variable=variable, operations=operations)

    assert _read_quantity_answer(state, read, _LISTED, _FLOW_SHAPES)

    assert set(state.quantity_requirement.named) == named
    assert _denominator(state) == denominator
    assert state.base_words == base


def _failing_read(*_args, **_kwargs):
    raise GraphInfraError("graph down")


@pytest.mark.parametrize(
    ("variable", "operations", "read", "raises_at"),
    [
        # The series names a rate of its own, so the denominator read cannot be deferred.
        pytest.param(
            "unemployment rate of women relative to men",
            (Operation(verb="share_of", phrase="relative to"),),
            _read(),
            "denominators",
            id="series-form",
        ),
        # The base is the operation's, so nothing about the series is judged on the reads.
        pytest.param(
            "ODA as a share of GNI",
            (_SHARE_OF_GNI,),
            _read("share", "GNI", 0),
            None,
            id="operation-owned-base",
        ),
        # An index past the list the failed read left empty is the read's failure, not a
        # malformed output.
        pytest.param(
            "ODA relative to GNI",
            (Operation(verb="share_of", phrase="relative to"),),
            _read(None, "GNI", 1),
            "answer",
            id="series-base-past-an-empty-list",
        ),
    ],
)
def test_a_failed_denominator_read_is_raised_wherever_the_series_needs_it(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    operations: tuple[Operation, ...],
    read: QuantityOutput,
    raises_at: str | None,
):
    """A graph failure reading the denominators raises where the series' own form or base
    needs them, and is held where only the operation's whole does."""
    monkeypatch.setattr(phase_rank_bind, "read_denominators", _failing_read)
    state = _state(llm=_NoCallLLM(), variable=variable, operations=operations)

    if raises_at == "denominators":
        with pytest.raises(GraphInfraError):
            asyncio.run(phase_rank_bind._read_denominators(state, _FLOW_SHAPES))
        return
    asyncio.run(phase_rank_bind._read_denominators(state, _FLOW_SHAPES))
    denominators = phase_rank_bind._quantity_denominators(state, _FLOW_SHAPES)
    if raises_at == "answer":
        with pytest.raises(GraphInfraError):
            _read_quantity_answer(state, read, denominators, _FLOW_SHAPES)
        return
    assert _read_quantity_answer(state, read, denominators, _FLOW_SHAPES)
    assert not state.quantity_requirement.named
    assert serves_every_form(state.quantity_requirement, _FLOW_SHAPES[0])


def test_a_code_only_part_sends_the_quantity_read_the_codes_graph_name():
    """The quantity read runs before the menu is seated, and a part whose text is only its
    series code, with two admissible shapes, sends the bind phrase the bind call reads: the
    graph name of the code's member among the admissible shapes. With one admissible shape
    the part is code-pinned (`phase_rank_bind._pins_on_code`) and the read sends nothing,
    since the code fixes the measure."""
    code = SeriesCodeMembers(code="DT.DOD.DECT.CD", provenance="WorldBank/IDS", dcids=("sv_debt",))
    name = "External debt stocks, total (DOD, current US$)"
    shape = dataclasses.replace(
        _shape("debt"), sv_arc_facts={"sv_debt": {"name": {"nodes": [{"value": name}]}}}
    )
    binder = _QuantityRecordingBinder()
    state = _state(llm=_NoCallLLM(), binder=binder)
    state.variable = "DT.DOD.DECT.CD"
    state.source_code = code
    state.capped_shapes = []

    assert asyncio.run(_read_stated_quantity(state, [shape, _shape("debt_share")])) is None

    (request,) = binder.quantity_requests
    assert (request.variable, request.bind_variable) == ("DT.DOD.DECT.CD", name)

    binder.quantity_requests.clear()
    state.code_pinned = phase_rank_bind._pins_on_code(state, [shape])
    assert asyncio.run(_read_stated_quantity(state, [shape])) is None
    assert (state.code_pinned, binder.quantity_requests) == (True, [])

    # A dropped qualifier phrase Jev reads in the bind keeps the read.
    state.dropped_qualifier_phrases = ["across all debtor types"]
    assert not phase_rank_bind._pins_on_code(state, [shape])


_DEBTOR_SLOT = SlotKeyDraft(axis="how", property_dcid="idsDebtorType", label="debtor type")
_GUARANTEE_SLOT = SlotKeyDraft(axis="how", property_dcid="guaranteeStatus", label="guarantee")
# The IDS stock by debtor type and guarantee status, and a commitments series that binds
# guarantee status and has no debtor-type slot.
_DEBT_MENU = [
    dataclasses.replace(
        _shape("stock_by_debtor", _DEBTOR_SLOT, _GUARANTEE_SLOT),
        slot_taxonomy={
            "how:idsDebtorType": ["IDSDebtorTypePS", "IDSDebtorTypePrivate"],
            "how:guaranteeStatus": ["PublicOrPubliclyGuaranteed", "PrivateNonguaranteed"],
        },
        value_labels={
            "IDSDebtorTypePS": "Public sector",
            "PublicOrPubliclyGuaranteed": "Public and publicly guaranteed",
        },
    ),
    dataclasses.replace(
        _shape("commitments_ppg", _GUARANTEE_SLOT),
        slot_taxonomy={
            "how:guaranteeStatus": ["PublicOrPubliclyGuaranteed", "PrivateNonguaranteed"]
        },
    ),
]
_DEBT_DRAFTS = [
    SlotBindingDraft(
        axis="how", property_dcid=prop, kind="value", value_dcids=[value], evidence="mentioned"
    )
    for prop, value in (
        ("idsDebtorType", "IDSDebtorTypePS"),
        ("guaranteeStatus", "PublicOrPubliclyGuaranteed"),
    )
]
_DEBT_JEV_ANSWERS = {
    "m|0": {"type": "noul", "noul": 0.9},
    "m|1": {"type": "noul", "noul": 0.9},
    "elect": {"type": "choice", "probabilities": {"Shape 0": 0.51, "Shape 1": 0.49}},
    "named|0": {"type": "noul", "noul": 0.9},
    "named|1": {"type": "noul", "noul": 0.05},
    "part|how:idsDebtorType": {"type": "noul", "noul": 0.9},
    "n|how:idsDebtorType|0": {"type": "noul", "noul": 0.9},
    "n|how:idsDebtorType|1": {"type": "noul", "noul": 0.05},
    "part|how:guaranteeStatus": {"type": "noul", "noul": 0.9},
    "n|how:guaranteeStatus|0": {"type": "noul", "noul": 0.9},
    "n|how:guaranteeStatus|1": {"type": "noul", "noul": 0.05},
}


@pytest.mark.parametrize(
    ("binder", "llm"),
    [
        pytest.param(
            GeminiBinder(),
            _ReplyLLM(elected_reply(0, bindings=_DEBT_DRAFTS, also_plausible=[1])),
            id="gemini",
        ),
        pytest.param(JevBinder(ask=FakeAsk(_DEBT_JEV_ANSWERS)), _NoCallLLM(), id="jev"),
    ],
)
def test_a_nominated_reading_that_cannot_hold_a_typed_value_departs_it(binder, llm):
    """The phrase "public sector PPG debt" elects the stock by debtor type, binding both
    typed values, and nominates commitments, which have no debtor-type slot: Gemini's pool
    sets it aside but its rating keeps the nomination, and Jev's dataset-named pool leaves
    it out while its election keeps it plausible. The candidate set discloses "Public
    sector" as a departure (`candidate_collapse._record_unheld_values`). A set whose
    readings hold every typed value, in a slot or on each member under the value's own
    property, records and departs nothing, as does a value the request's words do not
    name. A member carrying the value under another property holds nothing of it."""
    state = _state(llm=llm, binder=binder)
    state.variable = state.detect_query = "public sector PPG debt"
    state.capped_shapes = list(_DEBT_MENU)

    assert asyncio.run(_phase_bind(state)) is None

    bound = state.bind_result
    assert (bound.chosen_shape_index, bound.also_plausible) == (0, [1])
    assert {(b.property_dcid, b.evidence) for b in bound.bindings} == {
        ("idsDebtorType", "mentioned"),
        ("guaranteeStatus", "mentioned"),
    }
    elected = (_DEBT_MENU[0], list(bound.bindings))
    nominated = (_DEBT_MENU[1], [b for b in bound.bindings if b.property_dcid == "guaranteeStatus"])

    def settle(answers):
        state.shape_draft, state.bindings = elected
        state.requirements = ()
        asyncio.run(_record_unheld_values(state, answers))
        settle_candidate_set(state, answers)
        assert state.judgement is not None
        return [fact.asked for _warning, fact in state.judgement.departures()]

    assert settle([elected, nominated]) == ["Public sector"]

    # Each member of a public-sector commitments series carries the debtor type beside the
    # guarantee status its slot binds, so the reading holds it without a debtor-type slot.
    def commitments(debtor_property: str) -> ShapeDraft:
        return dataclasses.replace(
            _DEBT_MENU[1],
            shape_id="public_commitments",
            sv_arc_facts={
                "sv_public_commitments": {
                    "constraintProperties": {
                        "nodes": [{"dcid": "guaranteeStatus"}, {"dcid": debtor_property}]
                    },
                    "guaranteeStatus": {"nodes": [{"dcid": "PublicOrPubliclyGuaranteed"}]},
                    debtor_property: {"nodes": [{"dcid": "IDSDebtorTypePS"}]},
                }
            },
        )

    assert settle([elected, (commitments("idsDebtorType"), nominated[1])]) == []
    assert state.requirements == ()
    # The same code under a property of another name is no debtor type.
    assert settle([elected, (commitments("debtorSector"), nominated[1])]) == ["Public sector"]
    assert (
        settle([elected, (dataclasses.replace(_DEBT_MENU[0], shape_id="flows"), elected[1])]) == []
    )
    assert state.requirements == ()
    # A value no word of the request names is the bind model's reading, as LiveBirth is of
    # "birth rate", and the user did not type it.
    state.variable = state.detect_query = "PPG debt"
    assert settle([elected, nominated]) == []
    assert state.requirements == ()


def _member(sv: str, prop: str, value: str) -> dict[str, dict]:
    return {
        sv: {
            "constraintProperties": {"nodes": [{"dcid": prop}]},
            prop: {"nodes": [{"dcid": value}]},
        }
    }


_SECTOR = SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="sector")
_HEALTH, _EDUCATION = "CRSPurposeCode120", "CRSPurposeCode110"
# Aid by CRS sector, and an imputed multilateral health series with no sector slot whose
# member carries the health sector.
_AID_MENU = [
    dataclasses.replace(
        _shape("aid_by_sector", _SECTOR),
        slot_taxonomy={"how:crsPurpose": [_EDUCATION, _HEALTH]},
        value_labels={_EDUCATION: "Education", _HEALTH: "Health"},
    ),
    dataclasses.replace(
        _shape("imputed_health"), sv_arc_facts=_member("sv_health", "crsPurpose", _HEALTH)
    ),
]
_AID_JEV_ANSWERS = {
    "m|0": {"type": "noul", "noul": 0.9},
    "m|1": {"type": "noul", "noul": 0.9},
    "elect": {"type": "choice", "probabilities": {"Shape 0": 0.51, "Shape 1": 0.49}},
    "named|0": {"type": "noul", "noul": 0.9},
    "named|1": {"type": "noul", "noul": 0.05},
    "part|how:crsPurpose": {"type": "noul", "noul": 0.9},
    "n|how:crsPurpose|0": {"type": "noul", "noul": 0.9},
    "n|how:crsPurpose|1": {"type": "noul", "noul": 0.9},
    "several|how:crsPurpose": {"type": "noul", "noul": 0.9},
}


@pytest.mark.parametrize(
    ("binder", "llm"),
    [
        pytest.param(
            GeminiBinder(),
            _ReplyLLM(
                elected_reply(
                    0,
                    bindings=[
                        SlotBindingDraft(
                            axis="how",
                            property_dcid="crsPurpose",
                            kind="set",
                            value_dcids=[_EDUCATION, _HEALTH],
                            evidence="mentioned",
                        )
                    ],
                    also_plausible=[1],
                )
            ),
            id="gemini",
        ),
        pytest.param(JevBinder(ask=FakeAsk(_AID_JEV_ANSWERS)), _NoCallLLM(), id="jev"),
    ],
)
def test_a_nominated_reading_holding_part_of_a_typed_set_departs_the_rest(binder, llm):
    """The phrase "aid to health and education" elects aid by sector, binding both sectors
    as a set, and nominates an imputed health series whose members carry health alone. The
    candidate set departs "Education" and not "Health"
    (`phase_rank_bind.judge_value_requirements`). A reading whose members carry each of the
    two departs nothing. Beside a reading holding
    neither, each value is one requirement and departs once."""
    state = _state(llm=llm, binder=binder)
    state.variable = state.detect_query = "aid to health and education"
    state.capped_shapes = list(_AID_MENU)

    assert asyncio.run(_phase_bind(state)) is None

    bound = state.bind_result
    assert (bound.chosen_shape_index, bound.also_plausible) == (0, [1])
    assert [(b.kind, sorted(b.value_dcids), b.evidence) for b in bound.bindings] == [
        ("set", [_EDUCATION, _HEALTH], "mentioned")
    ]
    elected = (_AID_MENU[0], list(bound.bindings))

    def settle(shape):
        state.shape_draft, state.bindings = elected
        state.requirements = ()
        asyncio.run(_record_unheld_values(state, [elected, (shape, [])]))
        settle_candidate_set(state, [elected, (shape, [])])
        assert state.judgement is not None
        return [fact.asked for _warning, fact in state.judgement.departures()]

    assert settle(_AID_MENU[1]) == ["Education"]
    both = dataclasses.replace(
        _AID_MENU[1],
        shape_id="imputed_health_and_education",
        sv_arc_facts={
            **_member("sv_health", "crsPurpose", _HEALTH),
            **_member("sv_education", "crsPurpose", _EDUCATION),
        },
    )
    assert settle(both) == []
    assert state.requirements == ()
    # Three readings, holding both, health alone and neither: each value departs once.
    neither = dataclasses.replace(_AID_MENU[1], shape_id="general_budget", sv_arc_facts={})
    state.shape_draft, state.bindings = elected
    state.requirements = ()
    answers = [elected, (both, []), (_AID_MENU[1], []), (neither, [])]
    asyncio.run(_record_unheld_values(state, answers))
    settle_candidate_set(state, answers)
    assert state.judgement is not None
    assert [fact.asked for _warning, fact in state.judgement.departures()] == [
        "Education",
        "Health",
    ]
    assert [r.term.text for r in state.requirements] == ["Education", "Health"]


@pytest.mark.parametrize(
    ("binder", "llm"),
    [
        pytest.param(
            GeminiBinder(),
            _ReplyLLM(elected_reply(0, bindings=_DEBT_DRAFTS, also_plausible=[1])),
            id="gemini",
        ),
        pytest.param(JevBinder(ask=FakeAsk(_DEBT_JEV_ANSWERS)), _NoCallLLM(), id="jev"),
    ],
)
def test_a_lone_nominated_reading_answering_alone_departs_a_typed_value_it_cannot_hold(
    monkeypatch, binder, llm
):
    """The phrase "public sector PPG debt" elects the stock by debtor type, which has data
    but misses the quantity the question names, and nominates commitments, which serve it
    and have no debtor-type slot. The quantity filter leaves commitments to answer alone,
    and the definite answer departs "Public sector" (`candidate_collapse.assemble_candidates`)."""
    # Imported here: the rule-one module builds its grounded doubles at import.
    from tests.engine.test_candidate_collapse_rule_one import _grounded, _settle

    state = _state(llm=llm, binder=binder)
    state.variable = state.detect_query = "public sector PPG debt"
    state.capped_shapes = list(_DEBT_MENU)
    assert asyncio.run(_phase_bind(state)) is None
    assert state.bind_result.also_plausible == [1]
    state.shape_draft, state.bindings = _DEBT_MENU[0], list(state.bind_result.bindings)
    state.elected_has_data = True
    state.quantity_refusal = NoDataDraft(reason="no_observations")
    monkeypatch.setattr(candidate_collapse, "start_answer_related_read", lambda *_: None)
    monkeypatch.setattr(
        candidate_collapse, "serves_leg_quantity", lambda _state, shape: shape is _DEBT_MENU[1]
    )
    elected = _make_spec("sv_stock", "stock", "country/KEN")
    commitments = _make_spec("sv_commitments", "commitments", "country/KEN")
    nominee = [b for b in state.bindings if b.property_dcid == "guaranteeStatus"]

    specs, state = _settle(
        monkeypatch,
        state,
        _grounded(
            (elected, "sv_stock", _DEBT_MENU[0], None),
            (commitments, "sv_commitments", _DEBT_MENU[1], "nominated"),
        ),
    )

    assert [spec.spec_id for spec in specs] == [commitments.spec_id]
    settle_answer(state, _DEBT_MENU[1], nominee)
    assert state.judgement is not None
    assert [fact.asked for _warning, fact in state.judgement.departures()] == ["Public sector"]


@pytest.mark.parametrize(
    ("twin", "departed"),
    [
        pytest.param(_DEBT_MENU[1], ["Public sector"], id="twin-without-the-slot"),
        pytest.param(_DEBT_MENU[0], [], id="twin-holding-every-value"),
    ],
)
@pytest.mark.parametrize(
    ("binder", "llm"),
    [
        pytest.param(
            GeminiBinder(),
            _ReplyLLM(elected_reply(0, bindings=_DEBT_DRAFTS, also_plausible=[1])),
            id="gemini",
        ),
        pytest.param(JevBinder(ask=FakeAsk(_DEBT_JEV_ANSWERS)), _NoCallLLM(), id="jev"),
    ],
)
def test_a_lone_twin_is_judged_on_the_typed_values_as_any_reading_is(
    monkeypatch, binder, llm, twin, departed
):
    """With no data on the elected stock, a registry twin answers for it alone, and is
    judged as every answering reading is: a twin whose shape cannot hold "Public sector"
    departs it, and one holding every typed value records nothing."""
    from tests.engine.test_candidate_collapse_rule_one import (
        _ELECTED_SV,
        _TWIN_SV,
        _grounded,
        _settle,
    )

    state = _state(llm=llm, binder=binder)
    state.variable = state.detect_query = "public sector PPG debt"
    state.capped_shapes = list(_DEBT_MENU)
    assert asyncio.run(_phase_bind(state)) is None
    state.shape_draft, state.bindings = _DEBT_MENU[0], list(state.bind_result.bindings)
    state.mat_result = NoDataDraft(reason="no_observations", sv_dcids=[_ELECTED_SV])
    twin_spec = _make_spec(_TWIN_SV, "twin", "country/KEN")
    slots = {slot.property_dcid for slot in twin.slot_keys}
    held = [b for b in state.bindings if b.property_dcid in slots]

    grounded = _grounded((twin_spec, _TWIN_SV, twin, "twin"))
    # The twin answers on the drafts its slots hold, as grounding binds them.
    entry = grounded[twin_spec.spec_id]
    grounded = dataclasses.replace(
        grounded,
        entries={
            twin_spec.spec_id: dataclasses.replace(
                entry, candidate=dataclasses.replace(entry.candidate, bindings=held)
            )
        },
    )

    specs, state = _settle(monkeypatch, state, grounded)

    assert [spec.spec_id for spec in specs] == [twin_spec.spec_id]
    assert [r.term.text for r in state.requirements if r.origin == "unheld"] == departed
    settle_answer(state, twin, held)
    assert state.judgement is not None
    assert [fact.asked for _warning, fact in state.judgement.departures()] == departed


def test_a_stand_in_s_verdict_is_stored_when_it_joins_the_menu(monkeypatch):
    """A stand-in for a declined election joins the menu after rule 4 judged the admissible
    shapes, so rule 4 judges it as it joins and the `bind_decision` line logs the stored
    verdicts of every shape without judging one."""
    state = _state(llm=_NoCallLLM())
    state.variable = "GDP"
    state.named_measures = tuple(declared_measures_named("GDP"))
    state.measure_verdicts = {"population": ["base"], "population_by_sex": ["holds"]}
    state.bind_result = elected_reply(0, bindings=[], also_plausible=[])
    stand_in = _shape("population_stand_in")

    _stand_in_declined(state, Interpretation(shape=stand_in, reason="variable_not_resolved"))

    def judged(*_args: object) -> list[str]:
        raise AssertionError("the line judged rule 4")

    monkeypatch.setattr(phase_rank_bind, "rule_four_verdicts", judged)
    assert _logged_verdicts(state) == [{"GDP": "base"}, {"GDP": "holds"}, {"GDP": "unknown"}]


@pytest.mark.parametrize("fired", [True, False], ids=["broad-reading", "no-reading"])
def test_a_value_drafted_in_a_set_is_applied_by_the_broad_reading_that_fired(fired):
    """ "girls and men" drafts Girls in a set. The part's broad reading of Girls as Female
    fired, and the answer binds Female, so Girls is applied as the reading says; with no
    reading the answer leaves it unmet."""
    state = _state(llm=_NoCallLLM())
    if fired:
        state.slot_defaults = {
            "gender": SlotDefaultDisclosure(
                reading=BroadReading(
                    property_dcid="gender",
                    value_dcid="Girls",
                    reads_as="Female",
                    unless_bound="age",
                    held_with="Adult",
                    note="girls read as female",
                )
            )
        }
    requirement = Requirement(
        term=ResolvedQualifier(
            text="girls", named=[GraphRef(ref="Girls", label="Girls")], disposition="unbound"
        ),
        origin="value",
        rule=ValueRule(
            values=frozenset({("gender", "Girls")}),
            drafted_set=frozenset({("gender", "Girls"), ("gender", "Male")}),
        ),
    )
    bindings = [
        SlotBindingDraft(axis="how", property_dcid="gender", kind="value", value_dcids=["Female"])
    ]

    unmet = judge_value_requirements([requirement], _MENU[1], bindings, state)

    assert unmet == (() if fired else (requirement,))


@pytest.mark.parametrize(
    ("members", "reads"),
    [
        pytest.param({_HEALTH, _EDUCATION}, False, id="every-answer-holds-both"),
        pytest.param({_HEALTH}, True, id="one-answer-holds-health-alone"),
    ],
)
def test_labels_are_read_only_for_a_value_some_answer_leaves_unmet(monkeypatch, members, reads):
    """An answer whose members carry each drafted sector applies both by value, so recording
    the unheld values reads no label. One holding health alone leaves education unmet, and
    only education's label is read."""
    read: list[list[str]] = []

    async def labels(state, values):
        read.append(list(values))
        return {value: value for value in values}

    monkeypatch.setattr(candidate_collapse, "draft_value_labels", labels)
    state = _state(llm=_NoCallLLM())
    drafts = [
        SlotBindingDraft(
            axis="how",
            property_dcid="crsPurpose",
            kind="set",
            value_dcids=[_EDUCATION, _HEALTH],
            evidence="mentioned",
        )
    ]
    state.variable = state.detect_query = "aid to health and education"
    state.shape_draft, state.bindings = _AID_MENU[0], drafts
    nominated = dataclasses.replace(
        _AID_MENU[1],
        sv_arc_facts={
            key: value
            for member in members
            for key, value in _member(f"sv_{member}", "crsPurpose", member).items()
        },
    )

    asyncio.run(_record_unheld_values(state, [(_AID_MENU[0], drafts), (nominated, [])]))

    assert read == ([[_EDUCATION]] if reads else [])


def test_a_value_is_judged_by_its_label_and_never_by_the_words_of_its_dcid(monkeypatch):
    """The series states "person", a word of the dcid `Person`, and not "women", the label
    the value is read by. The value rules leave the value unmet on the answer without a
    gender slot, so its label is read, and the label's words leave it unmet as well."""
    read: list[list[str]] = []

    async def labels(state, values):
        read.append(list(values))
        return dict.fromkeys(values, "Women")

    monkeypatch.setattr(candidate_collapse, "draft_value_labels", labels)
    state = _state(llm=_NoCallLLM())
    state.variable = state.detect_query = "women population"
    drafts = [
        SlotBindingDraft(
            axis="how",
            property_dcid="gender",
            kind="value",
            value_dcids=["Person"],
            evidence="mentioned",
        )
    ]
    state.shape_draft, state.bindings = _MENU[1], drafts

    asyncio.run(_record_unheld_values(state, [(_MENU[1], drafts), (_MENU[0], [])]))

    assert read == [["Person"]]
    assert [requirement.term.text for requirement in state.requirements] == ["Women"]


_IDS_GNI = ShapeDraft(
    shape_id="ids_gni",
    label="Gross national income (current US$)",
    pop_type_dcid="EconomicActivity",
    meas_prop_dcid="amount",
    stat_type_dcid="measuredValue",
    meas_qual_dcid="CurrentPricesScenario",
    meas_denom_dcid=None,
    slot_keys=(),
    sv_arc_facts={
        "Amount_EconomicActivity_GrossNationalIncome_NominalScenario": {
            "name": {"nodes": [{"value": "Gross national income (current US$)"}]},
            "constraintProperties": {"nodes": [{"dcid": "economicActivityType"}]},
            "economicActivityType": {"nodes": [{"dcid": "GrossNationalIncome"}]},
        }
    },
)


@pytest.mark.parametrize("read_words", [True, False], ids=["with-words", "values-alone"])
def test_gross_national_income_on_another_property_is_met_by_its_label(read_words):
    """The placeless-reelect IDS reading of "gross national income": the bind drafts
    `activitySource` GrossNationalIncome, which the PPP and Atlas series carry, and the IDS
    series carries GrossNationalIncome on `economicActivityType` instead. No value rule
    applies the drafted value across properties; the member's name holds every word of the
    value's label, so the label meets it."""
    requirement = _unheld_requirement(
        "activitySource",
        "GrossNationalIncome",
        "Gross national income",
        frozenset(),
    )

    unmet = judge_value_requirements(
        [requirement], _IDS_GNI, [], _state(llm=_NoCallLLM()), read_words=read_words
    )

    assert unmet == (() if read_words else (requirement,))
