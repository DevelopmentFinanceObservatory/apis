"""The house-policy step at `_phase_bind`: the Gemini binder's rated pool, the step
running once per bind call that elects a shape, the drafts a moved election keeps, and the
`bind_decision` line."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

import pytest

from code_index.shared import get_index
from entity_resolution import Admission, EntityResolver
from search.engine.bind import (
    _MEASURING_BIND_SYSTEM_PROMPT,
    ShapeRating,
    SlotBindingDraft,
    _UniversalBindOutput,
    bind_schema,
    bind_universal,
    rank_elected,
)
from search.engine.binders import BindRequest, BindResult, GeminiBinder
from search.engine.binders.base import PolicyContractError
from search.engine.binders.gemini import (
    RATED_POOL_FRACTION,
    rated_pool,
    reelect,
    shape_ratings,
)
from search.engine.binders.house_policy import PolicyChoice, PolicyPool
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.errors import LLMOutputRefused
from search.engine.frame import Frame, door_frame
from search.engine.interpret import Recall
from search.engine.qualifier_match import QualifierMatch
from search.engine.quantity_read import QuantityOutput
from search.engine.region.phase_rank_bind import _elected_index, _HousePolicyStep, _phase_bind
from search.engine.region.state import ResolveState
from search.engine.role_text import DoorRole
from search.engine.shape import Pool, ShapeDraft, SlotKeyDraft
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._quantity import NO_QUANTITY

_HOUSE = get_index().house_defaults()
_AID_TYPE = SlotKeyDraft(axis="how", property_dcid="dac2aAidType", label="aid type")
_CHANNEL = SlotKeyDraft(axis="how", property_dcid="crsChannel", label="channel")


def _shape(
    shape_id: str,
    *slots: SlotKeyDraft,
    taxonomy: Mapping[str, list[str]] | None = None,
    datasets: tuple[str, ...] | None = None,
    population_type: str = "FinancialFlow",
    measured_property: str = "amount",
) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid=population_type,
        meas_prop_dcid=measured_property,
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=slots,
        slot_taxonomy=dict(taxonomy or {}),
        datasets=datasets,
    )


# Aid served from DAC2a, the CRS and DAC1: one reading, three datasets.
_DAC2A = _shape(
    "dac2a",
    _AID_TYPE,
    taxonomy={"how:dac2aAidType": ["DAC2aAidTypeODA"]},
    datasets=(_HOUSE["recipient"],),
)
_CRS = _shape(
    "crs",
    _CHANNEL,
    taxonomy={"how:crsChannel": ["ChannelA", "ChannelB"]},
    datasets=(_HOUSE["sector"],),
)
_DAC1 = _shape("dac1", datasets=(_HOUSE["donor"],))


def _value(prop: str, value: str, evidence: str = "mentioned") -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how", property_dcid=prop, kind="value", value_dcids=[value], evidence=evidence
    )


_SOLE_AID_TYPE = _value("dac2aAidType", "DAC2aAidTypeODA", "sole_option")


def _reply(
    chosen: int, *bindings: SlotBindingDraft, rated: Sequence[int] = (), **fields: object
) -> _UniversalBindOutput:
    """A bind reply electing `chosen` that rates each menu shape in `rated` at 1."""
    return _UniversalBindOutput(
        chosen_shape_index=chosen,
        bindings=list(bindings),
        measuring=[ShapeRating(index=i, rating=1.0) for i in rated],
        **fields,
    )


class _StubLLM:
    """Answers the bind call with `reply`, or refuses it at parse when `reply` is None."""

    def __init__(self, reply: _UniversalBindOutput | None) -> None:
        self.reply = reply

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        if self.reply is None:
            raise LLMOutputRefused("response.parsed is None")
        return self.reply, None


def _frame(places: Mapping[str, tuple[str, ...]]) -> Frame:
    return door_frame(
        door="parsed",
        variable="aid",
        entities=list(places),
        entity_roles={
            name: [DoorRole(axis=axis, span=name) for axis in axes] for name, axes in places.items()
        },
        member_entities=frozenset(),
        pins={},
        date_request=None,
        unit_text=None,
        scope=None,
        operations=(),
        open_axes=frozenset(),
    )


_TO_UGANDA = _frame({"Uganda": ("destination",)})
_NO_PLACE = _frame({})

_EDUCATION = QualifierMatch(
    code_system="crsPurpose",
    code_dcid="CRSPurpose110",
    label="education",
    source="label",
    matched_text="education",
)


def _bind(
    shapes: list[ShapeDraft],
    reply: _UniversalBindOutput,
    *,
    frame: Frame,
    admitted_qualifiers: Sequence[tuple[QualifierMatch, ...]] = (),
) -> BindResult:
    """The Gemini binder's answer to `reply`, through the engine's house-policy step."""
    qualifiers = list(admitted_qualifiers) or [() for _ in shapes]
    step = _HousePolicyStep(
        shapes=shapes,
        frame=frame,
        admitted_qualifiers=qualifiers,
        availability=[],
        code_index=get_index(),
    )
    request = BindRequest(
        variable="aid",
        shapes=shapes,
        admitted_qualifiers=qualifiers,
        frame=frame,
        query_text=None,
        apply_policies=step,
    )
    result, _usage = asyncio.run(GeminiBinder().bind(request, llm=_StubLLM(reply)))
    assert step.calls == 1
    return result


def test_education_aid_to_uganda_elects_the_crs_shape():
    """A sector named on aid to a recipient moves Gemini's DAC2a pick to the CRS, the house
    default for a sector question. The DAC2a draft bound only as its slot's sole option
    leaves with the shape, and the set-aside shapes leave `also_plausible`."""
    shapes = [_DAC2A, _CRS, _DAC1]
    reply = _reply(0, _SOLE_AID_TYPE, also_plausible=[1, 2], rated=(0, 1, 2))

    result = _bind(shapes, reply, frame=_TO_UGANDA, admitted_qualifiers=[(), (_EDUCATION,), ()])

    assert result.policy_pool == PolicyPool(measuring=(0, 1, 2), rated=True)
    assert result.policy_choice is not None
    assert (result.policy_choice.survivor, result.policy_choice.fired) == (1, ("default_dataset",))
    assert result.output.chosen_shape_index == 1
    assert result.output.bindings == []
    assert result.output.also_plausible == []


_OUTPUT = _shape("output", population_type="EconomicActivity")
_VALUE_ADDED = _shape(
    "value_added", population_type="EconomicProduction", measured_property="valueAdded"
)
_GRANT_EQUIVALENT = _shape("grant_equivalent", measured_property="grantEquivalent")


def test_where_no_policy_fires_the_reply_is_gemini_s():
    """Aid served from three datasets keeps Gemini's pick when the part names no place and
    no sector, since the dataset policy has nothing to read. The answer is the one
    `bind_universal` gives."""
    shapes = [_DAC2A, _CRS, _DAC1]
    reply = _reply(2, also_plausible=[0, 1], rated=(0, 1, 2))

    result = _bind(shapes, reply, frame=_NO_PLACE)

    assert result.policy_pool == PolicyPool(measuring=(2, 0, 1), rated=True)
    assert result.policy_choice is not None and result.policy_choice.fired == ()
    direct, _usage = asyncio.run(
        bind_universal("aid", shapes, frame=_NO_PLACE, llm=_StubLLM(reply))
    )
    assert result.output == direct


_FLOW = SlotKeyDraft(axis="how", property_dcid="flowKind", label="flow kind")


def _rated_alike(shapes: Sequence[ShapeDraft], reply: _UniversalBindOutput) -> PolicyPool:
    """The rated pool of `reply` with every menu shape rated at the best."""
    return rated_pool(shapes, reply, dict.fromkeys(range(len(shapes)), 1.0))


def test_the_pool_admits_a_shape_only_with_a_slot_for_each_mentioned_value():
    """A value the phrase names keeps every shape without a slot holding it out of the
    pool, a shape whose slot lacks the value as one with no slot; a value drafted as a
    slot's sole option, or an open slot, keeps none out."""
    shapes = [_CRS, _DAC2A, _DAC1]
    channel = _value("crsChannel", "ChannelA")
    open_channel = SlotBindingDraft(axis="how", property_dcid="crsChannel", kind="unbound")
    grants = _shape("grants", _FLOW, taxonomy={"how:flowKind": ["Grant", "Loan"]})
    loans = _shape("loans", _FLOW, taxonomy={"how:flowKind": ["Loan"]})
    both = _shape("both", _FLOW, taxonomy={"how:flowKind": ["Loan", "Grant"]})

    assert _rated_alike(shapes, _reply(0, channel)).measuring == (0,)
    assert _rated_alike([grants, loans, both], _reply(0, _value("flowKind", "Grant"))) == (
        PolicyPool(measuring=(0, 2), rated=True)
    )
    assert _rated_alike(shapes, _reply(1, _SOLE_AID_TYPE)).measuring == (1, 0, 2)
    assert _rated_alike(shapes, _reply(0, open_channel)).measuring == (0, 1, 2)
    assert _rated_alike(shapes, _reply(0, _value("crsChannel", "ChannelA", "sole_option"))) == (
        PolicyPool(measuring=(0, 1, 2), rated=True)
    )


def test_grant_aid_to_uganda_keeps_the_crs_shape_whose_slot_holds_the_grant():
    """Aid to a recipient prefers DAC2a, but the DAC2a shape's flow slot holds no grant, so
    it stays out of the pool and the CRS shape Gemini elected with the grant stands."""
    crs = _shape(
        "crs", _FLOW, taxonomy={"how:flowKind": ["Grant", "Loan"]}, datasets=(_HOUSE["sector"],)
    )
    dac2a = _shape(
        "dac2a", _FLOW, taxonomy={"how:flowKind": ["Loan"]}, datasets=(_HOUSE["recipient"],)
    )

    reply = _reply(0, _value("flowKind", "Grant"), rated=(0, 1))

    result = _bind([crs, dac2a], reply, frame=_TO_UGANDA)

    assert result.policy_pool == PolicyPool(measuring=(0,), rated=True)
    assert result.output.chosen_shape_index == 0
    assert [(b.property_dcid, b.value_dcids) for b in result.output.bindings] == [
        ("flowKind", ["Grant"])
    ]


def test_a_moved_election_keeps_only_the_drafts_the_new_shape_holds():
    """On a move a value the phrase names stays, a sole-option draft stays where the new
    shape's slot holds its value, and leaves where the slot lacks the value or the shape
    lacks the slot."""
    sole = SlotKeyDraft(axis="how", property_dcid="soleKind", label="sole kind")
    old = _shape(
        "old",
        _FLOW,
        sole,
        _AID_TYPE,
        taxonomy={
            "how:flowKind": ["Grant"],
            "how:soleKind": ["Only"],
            "how:dac2aAidType": ["DAC2aAidTypeODA"],
        },
    )
    new = _shape(
        "new", _FLOW, sole, taxonomy={"how:flowKind": ["Grant", "Loan"], "how:soleKind": ["Only"]}
    )
    other = replace(
        new, shape_id="other", slot_taxonomy={"how:flowKind": ["Grant"], "how:soleKind": ["Else"]}
    )
    grant = _value("flowKind", "Grant")
    output = _reply(
        0, grant, _value("soleKind", "Only", "sole_option"), _SOLE_AID_TYPE, also_plausible=[1, 2]
    )

    def moved_to(index: int) -> _UniversalBindOutput:
        choice = PolicyChoice(
            survivor=index, fired=("default_dataset",), dropped=frozenset({0}), facts=()
        )
        return reelect(output, choice, [old, new, other])

    assert moved_to(1).chosen_shape_index == 1
    assert [(b.property_dcid, b.value_dcids) for b in moved_to(1).bindings] == [
        ("flowKind", ["Grant"]),
        ("soleKind", ["Only"]),
    ]
    assert [(b.property_dcid, b.value_dcids) for b in moved_to(2).bindings] == [
        ("flowKind", ["Grant"])
    ]
    assert moved_to(1).also_plausible == [2]


def test_a_move_to_a_shape_without_a_named_value_is_refused():
    """The pools admit no shape that lacks a value the phrase names, so a move to one is a
    fault in the binder and breaks the house-policy contract."""
    old = _shape("old", _FLOW, taxonomy={"how:flowKind": ["Grant"]})
    new = _shape("new", _FLOW, taxonomy={"how:flowKind": ["Loan"]})
    choice = PolicyChoice(survivor=1, fired=("default_dataset",), dropped=frozenset({0}), facts=())

    with pytest.raises(PolicyContractError, match="does not hold"):
        reelect(_reply(0, _value("flowKind", "Grant")), choice, [old, new])


def test_a_moved_election_judges_unplaced_on_the_new_shape():
    """A phrase the reply left unplaced on the DAC2a shape is one the CRS shape admits a
    qualifier for, so after the sector moves the election to the CRS it is no longer
    unplaced, and the answer reports no missing vocabulary for it; a phrase the CRS shape
    admits nothing for stays."""
    grant = QualifierMatch(
        code_system="crsFlowKind",
        code_dcid="Grant",
        label="grant",
        source="label",
        matched_text="grant",
    )
    reply = _reply(0, _SOLE_AID_TYPE, unplaced=["grant", "blended"], rated=(0, 1))

    result = _bind(
        [_DAC2A, _CRS], reply, frame=_TO_UGANDA, admitted_qualifiers=[(), (_EDUCATION, grant)]
    )

    assert result.output.chosen_shape_index == 1
    assert result.output.unplaced == ["blended"]


def _state(
    shapes: list[ShapeDraft], *, llm: SupportsLLM, pool: Pool = "ours", frame: Frame = _NO_PLACE
) -> ResolveState:
    """A part whose seated menu is `shapes`; the services bind never reads are unset."""
    state = ResolveState(
        variable="aid",
        entities=[],
        date_request=None,
        detect_query="aid",
        role_query="aid",
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
        pool=pool,
    )
    state.frame = frame
    state.capped_shapes = list(shapes)
    return state


@pytest.mark.parametrize(
    ("evidence", "elected", "bindings"),
    [
        pytest.param("sole_option", 1, [], id="sole-option-moves"),
        pytest.param("mentioned", 0, [("crsChannel", ["ChannelA"])], id="mentioned-stays"),
    ],
)
def test_a_part_moved_to_the_recipient_dataset_keeps_no_stray_value(
    evidence: str, elected: int, bindings: list[tuple[str, list[str]]]
):
    """Aid to a recipient prefers DAC2a. Gemini's CRS pick moves there when its channel
    draft bound only as a sole option, and the part settles with no stray requirement and
    no refusal; a channel the phrase names keeps the CRS shape."""
    reply = _reply(0, _value("crsChannel", "ChannelA", evidence), rated=(0, 1))
    state = _state([_CRS, _DAC2A], llm=_StubLLM(reply), frame=_TO_UGANDA)

    assert asyncio.run(_phase_bind(state)) is None

    assert state.bind_result.chosen_shape_index == elected
    assert [(b.property_dcid, b.value_dcids) for b in state.bind_result.bindings] == bindings
    assert state.stray_requirements == ()
    assert not [r for r in state.requirements if r.origin == "stray"]


def test_the_menu_rank_never_undoes_a_policy_move(caplog: pytest.LogCaptureFixture):
    """Two shapes the menu states alike, served from the sector and the recipient default:
    on aid to a recipient the dataset policy moves Gemini's pick to the recipient one, and
    the menu-rank tie-break, which reads only the policy's survivors, keeps it there. The
    line reports the election the part answers with."""
    first = replace(_shape("a", datasets=(_HOUSE["sector"],)), label="aid")
    second = replace(_shape("b", datasets=(_HOUSE["recipient"],)), label="aid")
    caplog.set_level(logging.INFO, logger="search.engine.region.phase_rank_bind")
    state = _state([first, second], llm=_StubLLM(_reply(0, rated=(0, 1))), frame=_TO_UGANDA)

    assert asyncio.run(_phase_bind(state)) is None

    assert rank_elected([first, second], state.bind_result) == 0
    assert _elected_index(state) == 1
    [decision] = _decisions(caplog)
    assert (decision["chosen"], decision["elected"], decision["answered"]) == (0, 1, 1)


def _decisions(caplog: pytest.LogCaptureFixture) -> list[dict]:
    """The JSON payload of each ``bind_decision`` line."""
    return [
        json.loads(r.getMessage().removeprefix("bind_decision "))
        for r in caplog.records
        if r.getMessage().startswith("bind_decision ")
    ]


class _CountingGemini(GeminiBinder):
    """The Gemini binder, counting its calls of the house-policy step."""

    def __init__(self) -> None:
        self.calls = 0

    async def bind(self, request: BindRequest, *, llm) -> tuple[BindResult, dict | None]:
        step = request.apply_policies

        def counted(
            pool: PolicyPool,
            *,
            ratings: Mapping[int, float] | None = None,
            holds: Mapping[int, bool] | None = None,
        ) -> PolicyChoice:
            self.calls += 1
            return step(pool, ratings=ratings, holds=holds)

        return await super().bind(replace(request, apply_policies=counted), llm=llm)


@pytest.mark.parametrize(
    ("shapes", "reply", "calls"),
    [
        pytest.param([_DAC2A, _CRS], _reply(1, rated=(0, 1)), 1, id="elects"),
        pytest.param([_DAC2A, _CRS], _reply(0, ask="Which aid?"), 0, id="asks"),
        pytest.param([_DAC2A, _CRS], _reply(7), 0, id="index-out-of-range"),
        pytest.param([_DAC2A, _CRS], None, 0, id="schema-refusal"),
        pytest.param([], _reply(0), 0, id="empty-menu"),
    ],
)
def test_the_policy_step_runs_once_per_bind_call_that_elects(
    shapes: list[ShapeDraft], reply: _UniversalBindOutput | None, calls: int
):
    binder = _CountingGemini()
    state = _state(shapes, llm=_StubLLM(reply))
    state.binder = binder

    asyncio.run(_phase_bind(state))

    assert binder.calls == calls
    decision = state.bind_decision
    assert (decision is not None and decision.policy_choice is not None) == bool(calls)


class _MiscountingBinder(GeminiBinder):
    """Answers `output` after calling the house-policy step `calls` times."""

    def __init__(self, output: _UniversalBindOutput, calls: int) -> None:
        self.output = output
        self.calls = calls

    async def bind(self, request: BindRequest, *, llm) -> tuple[BindResult, dict | None]:
        for _ in range(self.calls):
            request.apply_policies(PolicyPool(measuring=(0,)))
        return BindResult(output=self.output, binder="miscounting"), None


@pytest.mark.parametrize(
    ("output", "calls"),
    [
        pytest.param(_reply(0), 0, id="elects-without-the-step"),
        pytest.param(_reply(0), 2, id="elects-with-two-steps"),
        pytest.param(_reply(0, ask="Which aid?"), 1, id="asks-with-a-step"),
    ],
)
def test_a_binder_that_miscounts_the_policy_step_refuses_the_part(
    output: _UniversalBindOutput, calls: int, caplog: pytest.LogCaptureFixture
):
    """A binder that runs the policy step other than once on an election breaks the
    contract: the part refuses ``variable_not_resolved``, the breach is logged at ERROR,
    and no ``bind_decision`` line vouches for the election."""
    caplog.set_level(logging.INFO, logger="search.engine.region.phase_rank_bind")
    state = _state([_DAC2A, _CRS], llm=_StubLLM(None))
    state.binder = _MiscountingBinder(output, calls)

    result = asyncio.run(_phase_bind(state))

    assert result is not None and result.no_data_reason == "variable_not_resolved"
    [error] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert "house policies ran" in error.getMessage()
    assert _decisions(caplog) == []


class _MovingBinder(GeminiBinder):
    """The Gemini binder whose policy step moves every election to menu shape 1."""

    async def bind(self, request: BindRequest, *, llm) -> tuple[BindResult, dict | None]:
        def moves(pool: PolicyPool, **_kwargs: object) -> PolicyChoice:
            return PolicyChoice(
                survivor=1, fired=("default_dataset",), dropped=frozenset(), facts=()
            )

        return await super().bind(replace(request, apply_policies=moves), llm=llm)


def test_a_move_to_a_shape_without_a_named_value_refuses_the_part():
    """An election the policy step moves to a shape that lacks a value the phrase names
    breaks the contract, and the part refuses rather than answer without the value."""
    old = _shape("old", _FLOW, taxonomy={"how:flowKind": ["Grant"]})
    new = _shape("new", _FLOW, taxonomy={"how:flowKind": ["Loan"]})
    state = _state([old, new], llm=_StubLLM(_reply(0, _value("flowKind", "Grant"), rated=(0, 1))))
    state.binder = _MovingBinder()

    result = asyncio.run(_phase_bind(state))

    assert result is not None and result.no_data_reason == "variable_not_resolved"


def test_each_bind_call_logs_one_bind_decision_line(caplog: pytest.LogCaptureFixture):
    """Our pass and the UN pass are a bind call each, and each logs its own line: the
    binder, the menu, the pick before and after the house policies, the pool, the rating
    flag, the policies that fired, the reply's ratings and the facts of each pool shape."""
    reply = _reply(0, _value("crsChannel", "ChannelA", "sole_option"), rated=(0, 1))
    caplog.set_level(logging.INFO, logger="search.engine.region.phase_rank_bind")

    for pool in ("ours", "un_system"):
        state = _state([_CRS, _DAC2A], llm=_StubLLM(reply), pool=pool, frame=_TO_UGANDA)
        asyncio.run(_phase_bind(state))

    decisions = _decisions(caplog)
    assert [{k: v for k, v in d.items() if k != "facts"} for d in decisions] == [
        {
            "part": 0,
            "pass": pool,
            "binder": "gemini",
            "menu": ["crs", "dac2a"],
            "chosen": 0,
            "elected": 1,
            "answered": 1,
            "pool": [0, 1],
            "rated": True,
            "fired": ["default_dataset"],
            "ratings": [{"index": 0, "rating": 1.0}, {"index": 1, "rating": 1.0}],
            "refused": False,
        }
        for pool in ("ours", "un_system")
    ]
    assert [(f["index"], f["datasets"]) for f in decisions[0]["facts"]] == [
        (0, [_HOUSE["sector"]]),
        (1, [_HOUSE["recipient"]]),
    ]


def test_a_schema_refusal_logs_a_refused_bind_decision(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="search.engine.region.phase_rank_bind")

    asyncio.run(_phase_bind(_state([_CRS], llm=_StubLLM(None))))

    [decision] = _decisions(caplog)
    assert decision == {
        "part": 0,
        "pass": "ours",
        "binder": "gemini",
        "menu": ["crs"],
        "chosen": None,
        "elected": None,
        "answered": None,
        "pool": [],
        "rated": False,
        "fired": [],
        "ratings": None,
        "facts": [],
        "refused": True,
    }


# -- The rated pool ------------------------------------------------------------------------


class _RecordingLLM(_StubLLM):
    """`_StubLLM`, keeping the system prompt and schema of each bind call."""

    def __init__(self, reply: _UniversalBindOutput | None) -> None:
        super().__init__(reply)
        self.requests: list[tuple[str, str, type]] = []

    async def generate_structured(self, *, prompt, system, schema):
        if schema is not QuantityOutput:
            self.requests.append((prompt, system, schema))
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


def _rated(chosen: int, ratings: Mapping[int, float], *bindings: SlotBindingDraft):
    return _UniversalBindOutput(
        chosen_shape_index=chosen,
        bindings=list(bindings),
        measuring=[ShapeRating(index=i, rating=r) for i, r in ratings.items()],
    )


def test_the_bind_call_asks_for_ratings_bounded_to_its_menu():
    """The bind call sends the system prompt that asks for `measuring`, in the bind schema
    bounded to one entry per menu shape, so a reply that repeats an entry ends at the
    menu's size."""
    recording = _RecordingLLM(_reply(0))
    asyncio.run(_phase_bind(_state([_DAC2A, _CRS], llm=recording)))

    [(_prompt, system, schema)] = recording.requests
    assert system is _MEASURING_BIND_SYSTEM_PROMPT
    assert schema is bind_schema(2)
    assert schema.model_json_schema()["properties"]["measuring"]["maxItems"] == 2


def _rated_bind(shapes: list[ShapeDraft], reply: _UniversalBindOutput) -> BindResult:
    return _bind(shapes, reply, frame=_NO_PLACE)


_PER_CAPITA_OUTPUT = replace(_OUTPUT, shape_id="output_per_capita", meas_denom_dcid="Count_Person")
_GROSS_AMOUNT = _shape(
    "gross_amount",
    SlotKeyDraft(axis="how", property_dcid="nettingBasis", label="netting basis"),
    taxonomy={"how:nettingBasis": ["Gross", "Net"]},
)


@pytest.mark.parametrize(
    ("shapes", "ratings", "pool", "elected"),
    [
        # The spec's guard test: the best-rated shape is a grant-equivalent series, and a
        # lower-rated amount series that offers Gross stays out of the pool.
        pytest.param([_GRANT_EQUIVALENT, _GROSS_AMOUNT], {0: 0.9, 1: 0.5}, (0,), 0, id="guard"),
        # Rated far below value added, the output series cannot win on policy 1.
        pytest.param([_VALUE_ADDED, _OUTPUT], {0: 0.9, 1: 0.5}, (0,), 0, id="guard-gdp"),
        # Rated near the best, it joins, and nominal output beats value added.
        pytest.param(
            [_VALUE_ADDED, _OUTPUT],
            {0: 0.9, 1: 0.9 * (1 - RATED_POOL_FRACTION)},
            (0, 1),
            1,
            id="near-gdp",
        ),
    ],
)
def test_only_shapes_rated_near_the_best_reach_the_policies(
    shapes: list[ShapeDraft],
    ratings: Mapping[int, float],
    pool: tuple[int, ...],
    elected: int,
):
    result = _rated_bind(shapes, _rated(0, ratings))

    assert result.policy_pool == PolicyPool(measuring=pool, rated=True)
    assert result.output.chosen_shape_index == elected


@pytest.mark.parametrize(
    ("ratings", "pool"),
    [
        pytest.param({0: 1.0, 1: 0.9}, (0, 1), id="at-the-bound"),
        pytest.param({0: 0.8, 1: 0.72}, (0, 1), id="at-a-bound-the-float-product-rounds-up"),
        pytest.param({0: 1.0, 1: 0.99 - RATED_POOL_FRACTION}, (0,), id="below-the-bound"),
        pytest.param({0: 0.1, 1: 0.1}, (0,), id="at-the-floor"),
        pytest.param({0: 1.0}, (0,), id="missing"),
        pytest.param({0: 0.9, 1: 1.5}, (0,), id="out-of-range"),
        pytest.param({0: 0.3, 1: 1.0}, (0, 1), id="chosen-below-the-bound"),
    ],
)
def test_the_rated_pool_reads_each_rating_against_the_bound(
    ratings: Mapping[int, float], pool: tuple[int, ...]
):
    """With the fraction at 0.1 a shape joins at 0.9 of the best, whatever the float product
    of the bound rounds to, and not below, and not at the 0.1 floor. A shape the reply leaves unrated reads 0, as does a rating outside 0 to
    1, which sets no best either. The chosen shape stays whatever its rating."""
    shapes = [_OUTPUT, replace(_OUTPUT, shape_id="output_other", label="other output")]
    reply = _rated(0, ratings)

    assert rated_pool(shapes, reply, shape_ratings(reply, len(shapes))) == PolicyPool(
        measuring=pool, rated=True
    )


@pytest.mark.parametrize(
    "ratings",
    [
        pytest.param({1: 1.0}, id="unrated"),
        pytest.param({1: 1.0, 0: 0.5}, id="below-the-bound"),
        pytest.param({1: 1.0, 0: 0.05}, id="under-the-floor"),
    ],
)
def test_a_twin_of_the_pick_joins_the_pool_whatever_its_rating(ratings: Mapping[int, float]):
    """A shape the menu states as the pick's reading joins the pool, however the reply
    rates it, so the menu rank breaks the tie between the two and elects the first."""
    shapes = [replace(_OUTPUT, shape_id="output_first"), _OUTPUT]
    reply = _rated(1, ratings)

    result = _rated_bind(shapes, reply)

    assert result.policy_pool == PolicyPool(measuring=(1, 0), rated=True)
    among = frozenset(result.policy_pool.measuring)
    assert rank_elected(shapes, result.output, among=among) == 0


def test_the_rated_pool_keeps_the_chosen_denominator_and_the_named_values():
    """A shape rated at the best joins only when it states the chosen shape's denominator
    and holds a slot for each value the phrase names."""
    channel = _value("crsChannel", "ChannelA")
    shapes = [_OUTPUT, _PER_CAPITA_OUTPUT, _CRS, replace(_CRS, shape_id="crs_twin")]
    reply = _rated(2, {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}, channel)

    assert rated_pool(shapes, reply, shape_ratings(reply, 4)).measuring == (2, 3)
    plain = _rated(0, {0: 1.0, 1: 1.0, 2: 1.0})
    assert rated_pool(shapes, plain, shape_ratings(plain, 4)).measuring == (0, 2)


def test_a_chosen_shape_below_the_bound_survives_as_the_pick():
    """The pick rated below the bound stays in the pool, and the policies keep it when it
    survives them, over a higher-rated twin."""
    shapes = [_OUTPUT, replace(_OUTPUT, shape_id="output_twin")]

    result = _rated_bind(shapes, _rated(0, {0: 0.3, 1: 1.0}))

    assert result.policy_pool == PolicyPool(measuring=(0, 1), rated=True)
    assert result.output.chosen_shape_index == 0


def test_the_bind_decision_line_records_the_ratings_as_the_reply_wrote_them(
    caplog: pytest.LogCaptureFixture,
):
    """Value added rated within the fraction of the output series joins its pool, and
    nominal output beats it; the line holds the reply's ratings in its own order."""
    caplog.set_level(logging.INFO, logger="search.engine.region.phase_rank_bind")
    reply = _rated(1, {1: 0.9, 0: 0.85})
    state = _state([_VALUE_ADDED, _OUTPUT], llm=_StubLLM(reply))

    asyncio.run(_phase_bind(state))

    [line] = _decisions(caplog)
    assert (line["pool"], line["rated"], line["fired"], line["elected"]) == (
        [1, 0],
        True,
        ["nominal_vs_va"],
        1,
    )
    assert line["ratings"] == [{"index": 1, "rating": 0.9}, {"index": 0, "rating": 0.85}]
