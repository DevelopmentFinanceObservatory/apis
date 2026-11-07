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
from search.engine.interpret import Recall
from search.engine.quantity_read import QuantityOutput
from search.engine.region import candidate_collapse
from search.engine.region.candidate_collapse import _record_unheld_values
from search.engine.region.phase_rank_bind import _phase_bind, _read_stated_quantity
from search.engine.region.requirements import settle_answer, settle_candidate_set
from search.engine.region.state import ResolveState
from search.engine.retrieval.retrieve import NoDataDraft
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.engine.source_code import SeriesCodeMembers
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


def _state(*, llm: SupportsLLM, binder: Binder | None = None) -> ResolveState:
    """A part whose seated menu is `_MENU`. The services bind does not read are unset."""
    state = ResolveState(
        variable="female population",
        entities=[],
        date_request=None,
        detect_query="female population",
        role_query="female population",
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


def test_a_code_only_part_sends_the_quantity_read_the_codes_graph_name():
    """The quantity read runs before the menu is seated, and a part whose text is only its
    series code still sends the bind phrase the bind call reads: the graph name of the
    code's member among the admissible shapes."""
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

    assert asyncio.run(_read_stated_quantity(state, [shape])) is None

    (request,) = binder.quantity_requests
    assert (request.variable, request.bind_variable) == ("DT.DOD.DECT.CD", name)


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
    readings hold every typed value, in a slot or on each member under another property,
    records and departs nothing, as does a value the request's words do not name."""
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
        asyncio.run(_record_unheld_values(state, [shape for shape, _ in answers]))
        settle_candidate_set(state, answers)
        assert state.judgement is not None
        return [fact.asked for _warning, fact in state.judgement.departures()]

    assert settle([elected, nominated]) == ["Public sector"]
    # Each member of a public-sector commitments series carries the debtor type on a
    # property of its own, so the reading holds it without a debtor-type slot.
    public_commitments = dataclasses.replace(
        _DEBT_MENU[1],
        shape_id="public_commitments",
        sv_arc_facts={
            "sv_public_commitments": {
                "constraintProperties": {"nodes": [{"dcid": "debtorSector"}]},
                "debtorSector": {"nodes": [{"dcid": "IDSDebtorTypePS"}]},
            }
        },
    )
    assert settle([elected, (public_commitments, nominated[1])]) == []
    assert state.requirements == ()
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
# member carries health under a property of its own.
_AID_MENU = [
    dataclasses.replace(
        _shape("aid_by_sector", _SECTOR),
        slot_taxonomy={"how:crsPurpose": [_EDUCATION, _HEALTH]},
        value_labels={_EDUCATION: "Education", _HEALTH: "Health"},
    ),
    dataclasses.replace(
        _shape("imputed_health"), sv_arc_facts=_member("sv_health", "aidPurpose", _HEALTH)
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
    candidate set departs "Education" and not "Health" (`election.unheld_mentioned`). A
    reading whose members carry each of the two departs nothing. Beside a reading holding
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
        asyncio.run(_record_unheld_values(state, [elected[0], shape]))
        settle_candidate_set(state, [elected, (shape, [])])
        assert state.judgement is not None
        return [fact.asked for _warning, fact in state.judgement.departures()]

    assert settle(_AID_MENU[1]) == ["Education"]
    both = dataclasses.replace(
        _AID_MENU[1],
        shape_id="imputed_health_and_education",
        sv_arc_facts={
            **_member("sv_health", "aidPurpose", _HEALTH),
            **_member("sv_education", "aidPurpose", _EDUCATION),
        },
    )
    assert settle(both) == []
    assert state.requirements == ()
    # Three readings, holding both, health alone and neither: each value departs once.
    neither = dataclasses.replace(_AID_MENU[1], shape_id="general_budget", sv_arc_facts={})
    state.shape_draft, state.bindings = elected
    state.requirements = ()
    answers = [elected, (both, []), (_AID_MENU[1], []), (neither, [])]
    asyncio.run(_record_unheld_values(state, [shape for shape, _ in answers]))
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
    monkeypatch.setattr(candidate_collapse, "start_elected_related_read", lambda *_: None)
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

    specs, state = _settle(monkeypatch, state, _grounded((twin_spec, _TWIN_SV, twin, "twin")))

    assert [spec.spec_id for spec in specs] == [twin_spec.spec_id]
    assert [r.term.text for r in state.requirements if r.origin == "unheld"] == departed
    settle_answer(state, twin, held)
    assert state.judgement is not None
    assert [fact.asked for _warning, fact in state.judgement.departures()] == departed
