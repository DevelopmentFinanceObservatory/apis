"""The binder seam at `_phase_bind`: the engine elects through `ResolveState.binder`, any
binder answering alike leaves the part in the same state, and the Gemini binder sends the
request `bind_universal` sends."""

from __future__ import annotations

import asyncio
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
from search.engine.binders import Binder
from search.engine.binders.house_policy import PolicyPool
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.interpret import Recall
from search.engine.quantity_read import QuantityOutput
from search.engine.region.phase_rank_bind import _phase_bind
from search.engine.region.state import ResolveState
from search.engine.shape import ShapeDraft, SlotKeyDraft
from tests.engine._binders import FixedBinder, elected_reply
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
    """A part whose seated menu is `_MENU`; the services bind never reads are unset."""
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
        PolicyPool(measuring=(1,), rated=True),
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
