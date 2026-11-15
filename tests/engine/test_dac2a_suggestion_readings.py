"""A nearest-real suggestion and a flagged match that settle on DAC2a's bilateral composite are
minted as it from their own build, and no population read follows them.

Both elect their provider reading at the read they are built over (`reading.elect_at_read`)
and read within the population the reading binds, as the part's definite answer does, so the
composite spec_id they carry is the one a resubmit of their definition answers under.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Callable
from typing import Any

import pytest

from search.engine.degraded_reads import recording_degraded_reads
from search.engine.errors import EngineInputError
from search.engine.interpretation import Interpretation
from search.engine.region import (
    RegionResult,
    grounding,
    phase_place_materialise,
    phase_shared,
    reading,
    resubmit,
)
from search.engine.region.state import ResolveState
from search.engine.retrieval import role_resolve
from search.engine.spec_id import identity_spec_id
from search.models import (
    IdentityBindingValue,
    NearestRealSpec,
    NoDataPart,
    Relaxation,
    Spec,
    Warning,
)
from tests.engine._harness import offline_resolve
from tests.engine.test_dac2a_destination_populations import (
    _AT_TOTAL,
    _BILATERAL,
    _DEVELOPING,
    _FRA,
    _KEN,
    _VARIABLE,
    _counted,
    _coverage,  # noqa: F401 (autouse fixture)
    _ended_state,
    _graph_with_parties,
    _identity,
    _patched_coverage,
    _resolve_resubmit,
    _resubmit,
    _ScriptedLLM,
    _sdmx,
    _table_reads,
    _TableRead,
    _years_short,
)


def _provenance(spec: Spec) -> str | None:
    pin = (spec.definition.identity.facets or {}).get("provenance")
    return pin.ref if isinstance(pin, IdentityBindingValue) else None


@pytest.fixture
def population_reads(monkeypatch: pytest.MonkeyPatch) -> list[_TableRead]:
    """Every read of a part's table from here on (`_table_reads`). It holds the fresh H1
    answer's one read within the population and none by a suggestion or flagged match built
    over it."""
    return _table_reads(monkeypatch)


# The fresh H1 answer's own read, the only table read a suggestion or match beside it leaves.
_ANSWER_READ = [("answer", True)]


@pytest.fixture
def fresh_state(monkeypatch: pytest.MonkeyPatch) -> Callable[[], ResolveState]:
    """The state a fresh H1 answer (France, every recipient open, over the bilateral
    composite) ends its pass with: its shape, bindings and place, and the read that elected
    the composite."""

    def run() -> ResolveState:
        state = _ended_state(monkeypatch, "H1")
        assert state.reading is not None and state.reading.table is not None
        return state

    return run


def _level_suggestion(state: ResolveState, monkeypatch: pytest.MonkeyPatch) -> NearestRealSpec:
    """The one level suggestion of `state`'s elected shape under the same bindings, the aid
    type the bilateral default reads among them, read as the part's read is."""
    read = state.mat_result
    monkeypatch.setattr(role_resolve, "graph_confirm_resolve", lambda **kwargs: read)
    level = (
        state.shape_draft,
        Relaxation(axis="what", property_dcid="measurementDenominator", dropped_value_dcids=["x"]),
    )
    near = asyncio.run(phase_place_materialise._nearest_real_for(state, level=level))
    assert near is not None
    (suggestion,) = near
    return suggestion


def _coded(state: ResolveState, code: str) -> list[Warning]:
    return [warning for warning in state.warnings if warning.code == code]


def test_a_flagged_match_settling_on_the_bilateral_composite_is_minted_as_it(
    fresh_state: Callable[[], ResolveState], population_reads: list[Spec]
) -> None:
    """The match carries the composite it is minted as on its own result, and its whole-total
    warnings read the identity its rows were read for. The part's own answer's composite
    (`ResolveState.minted`) stays as the answer phase minted it."""
    state = fresh_state()
    answered = state.minted
    assert answered is not None

    match = asyncio.run(phase_shared._elected_flagged_match(state))

    flagged = match.spec
    assert _provenance(flagged) == _BILATERAL
    assert flagged.spec_id == identity_spec_id(flagged.definition.identity)
    assert _counted(population_reads) == _ANSWER_READ
    assert state.minted is answered
    assert match.minted is not None
    assert [minted.mint.spec_id for minted in match.matched] == [flagged.spec_id]
    assert state.minted_as(flagged, match.matched) is match.minted


def test_a_flagged_match_minted_as_the_composite_states_the_scope_warning_as_its_own(
    fresh_state: Callable[[], ResolveState],
) -> None:
    """The part's read is the match's, so its `role_scope_defaulted` warning, counted over the
    composite's parties, is unkeyed until the match is built and keyed to it afterwards."""
    state = fresh_state()
    (stated,) = _coded(state, "role_scope_defaulted")
    assert stated.spec_id is None

    match = asyncio.run(phase_shared._elected_flagged_match(state))

    (keyed,) = _coded(state, "role_scope_defaulted")
    assert (keyed.spec_id, keyed.params, keyed.message) == (
        match.spec.spec_id,
        stated.params,
        stated.message,
    )


def test_a_flagged_match_elects_and_decides_off_the_event_loop(
    fresh_state: Callable[[], ResolveState], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The match's election and scope read coverage and scan families, so they run on a
    worker thread, off the event loop the refusal is built on."""
    state = fresh_state()
    state.reading = None
    ran: list[tuple[str, bool]] = []

    def on_loop() -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    for name in ("elect_at_read", "decide_read_scope"):
        real = getattr(phase_shared, name)

        def recorded(*args: Any, _real: Any = real, _name: str = name, **kwargs: Any) -> Any:
            ran.append((_name, on_loop()))
            return _real(*args, **kwargs)

        monkeypatch.setattr(phase_shared, name, recorded)

    asyncio.run(phase_shared._elected_flagged_match(state))

    assert ran == [("elect_at_read", False), ("decide_read_scope", False)]


def test_a_flagged_match_whose_composite_refuses_its_read_says_it_reads_total_oda(
    fresh_state: Callable[[], ResolveState],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A flagged match whose elected composite refuses its table read is read nowhere, so it
    is the table's Spec, and it says it reads total ODA because its scope could not be
    decided, keyed to its own spec_id. The refusal is logged once and noted as a degraded
    read once, at the election. The match's identity chooses the bilateral composite, as the
    part's did, and the reading's undecided choice differs from it by design, which is no
    election fault (`reading._checked`), so the check notes nothing more."""
    state = fresh_state()
    assert state.reading is not None
    bilateral = state.reading.choice
    state.reading = None
    monkeypatch.setattr(reading, "choose_provider", lambda *_args: bilateral)
    degraded: list[str] = []

    def refusing(*_args: object, **_kwargs: object) -> object:
        raise EngineInputError("refused", code="requested_option_unavailable", status=422)

    monkeypatch.setattr(reading, "composite_table_identity", refusing)
    monkeypatch.setattr(reading, "note_degraded_read", degraded.append)

    with caplog.at_level(logging.ERROR, logger=reading.__name__):
        match = asyncio.run(phase_shared._elected_flagged_match(state))

    assert [r.levelno for r in caplog.records if r.name == reading.__name__] == [logging.ERROR]
    assert degraded == ["reading election"]

    assert match.minted is None
    assert _provenance(match.spec) != _BILATERAL
    withheld = [
        w.params
        for w in _coded(state, "bilateral_reading_unavailable")
        if w.spec_id == match.spec.spec_id
    ]
    assert withheld == [{"why": "scope_undecided"}]


def test_a_flagged_match_whose_composite_cannot_be_minted_fails_closed(
    fresh_state: Callable[[], ResolveState], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flagged match read within the bilateral population whose composite cannot be
    minted has no label for its rows to stand under, so the table's Spec, which would say
    they are total ODA, is not built. The refusal fails closed as the definite answer does
    (`phase_answer.failed_closed`): no_data `facet_not_available`, with nothing the match
    raised left on the part."""
    state = fresh_state()
    before = list(state.warnings)

    def raising(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("mint failed")

    monkeypatch.setattr(resubmit, "composite_mint", raising)

    refused = asyncio.run(phase_shared._elected_flagged_match(state))

    assert isinstance(refused, RegionResult), refused
    assert (refused.status, refused.no_data_reason) == ("no_data", "facet_not_available")
    assert refused.flagged_match is None
    assert state.warnings == before


@pytest.mark.parametrize(
    ("reason", "refuse"),
    [
        pytest.param(
            "role_incompatible", phase_shared._role_incompatible_no_data, id="role-incompatible"
        ),
        pytest.param(
            "denominator_not_available",
            lambda state: phase_place_materialise._denominator_refusal(state, None),
            id="denominator",
        ),
        pytest.param(
            "qualifier_not_bound",
            phase_place_materialise._declined_refusal,
            id="pinned-code",
        ),
    ],
)
def test_a_refusal_that_read_nothing_keeps_its_own_reason_when_no_composite_can_be_minted(
    reason: str,
    refuse: Callable[[ResolveState], Any],
    fresh_state: Callable[[], ResolveState],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal raised before the part reads (`ResolveState.reading` is None) labels no rows,
    so where the composite its flagged match elects cannot be minted, the match is minted as
    none and the refusal keeps its own reason and match, noting no degraded read."""
    state = fresh_state()
    state.reading = None
    state.declined = Interpretation(
        shape=state.shape_draft, reason="qualifier_not_bound", pinned_failure="pinned_code"
    )

    async def unsuggested(*_args: object, **_kwargs: object) -> None:
        """No nearest-real suggestion, whose own read within the population would note its
        lost mint."""

    monkeypatch.setattr(phase_place_materialise, "_nearest_real_for", unsuggested)

    def raising(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("mint failed")

    monkeypatch.setattr(resubmit, "composite_mint", raising)

    with recording_degraded_reads() as record:
        refused = asyncio.run(refuse(state))

    assert isinstance(refused, RegionResult)
    assert (refused.status, refused.no_data_reason) == ("no_data", reason)
    (flagged,) = refused.flagged_match or ()
    assert _provenance(flagged) != _BILATERAL
    assert record.reads == []


def test_a_resubmit_refusal_s_flagged_match_is_the_composite() -> None:
    """A composite posted with a destination that holds no row in its population refuses
    `no_observations` with the composite as its flagged match."""
    posted = _identity(_BILATERAL).model_copy(
        update={"entities": {"origin": [_FRA], "destination": [_KEN]}, "open_roles": None}
    )

    response = offline_resolve(
        _resubmit(posted), graph=_graph_with_parties(), llm=_ScriptedLLM(_VARIABLE), sdmx=_sdmx()
    )

    (part,) = response.parts
    assert isinstance(part, NoDataPart)
    flagged = part.no_data.flagged_match
    assert isinstance(flagged, Spec)
    assert flagged.spec_id == identity_spec_id(posted)
    assert _provenance(flagged) == _BILATERAL


def test_a_nearest_real_suggestion_settling_on_the_bilateral_composite_is_minted_as_it(
    fresh_state: Callable[[], ResolveState],
    population_reads: list[Spec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The suggestion states its own `role_scope_defaulted` warning, keyed to it, as a
    resubmit of its definition states it."""
    state = fresh_state()
    read = state.mat_result
    opened_within: list[Any] = []

    def confirmed(**kwargs: Any) -> Any:
        """The relaxed query's own read, answered as the part's read of the population."""
        opened_within.append(kwargs["within"])
        return read

    monkeypatch.setattr(role_resolve, "graph_confirm_resolve", confirmed)

    # The level relaxation reads the elected shape under the same bindings, the aid type the
    # bilateral default reads among them.
    level = (
        state.shape_draft,
        Relaxation(axis="what", property_dcid="measurementDenominator", dropped_value_dcids=["x"]),
    )
    near = asyncio.run(phase_place_materialise._nearest_real_for(state, level=level))

    assert _counted(population_reads) == _ANSWER_READ
    assert near is not None
    (composite,) = near
    assert _provenance(composite) == _BILATERAL
    assert composite.spec_id == identity_spec_id(composite.definition.identity)
    again = _resolve_resubmit(composite.definition.identity)
    assert again.spec.spec_id == composite.spec_id
    assert opened_within and all(within for within in opened_within)
    own = [w for w in _coded(state, "role_scope_defaulted") if w.spec_id == composite.spec_id]
    resubmitted = [w for w in again.warnings if w.code == "role_scope_defaulted"]
    assert [(w.params, w.message) for w in own] == [(w.params, w.message) for w in resubmitted]
    assert resubmitted


def test_each_reading_the_bilateral_default_cannot_reach_says_so_under_its_own_spec_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over every donor with bilateral destinations covering fewer years than the answer, the
    part reads total ODA, and a nearest-real suggestion and a flagged match of it, each
    elected at its own read, each carry `bilateral_reading_unavailable` keyed to its own
    spec_id, though their messages match."""
    _years_short(monkeypatch)
    state = _ended_state(monkeypatch, "H6")
    assert state.reading is not None and state.reading.table is None

    suggestion = _level_suggestion(state, monkeypatch)
    match = asyncio.run(phase_shared._elected_flagged_match(state))

    withheld = _coded(state, "bilateral_reading_unavailable")
    assert suggestion.spec_id != match.spec.spec_id
    assert [(w.spec_id, w.params) for w in withheld] == [
        (suggestion.spec_id, {"why": "option_not_covered"}),
        (match.spec.spec_id, {"why": "option_not_covered"}),
    ]


def test_a_netting_basis_relaxation_elects_the_bilateral_composite_as_its_resubmit_does(
    fresh_state: Callable[[], ResolveState],
    population_reads: list[Spec],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dropping nettingBasis leaves an unbound slot in the suggestion's identity. The election
    reads that slot as the mint does, so the suggestion settles on the bilateral composite,
    reads within its population, passes the after-read monitor with no log, and a resubmit of
    its definition answers under its spec_id."""
    state = fresh_state()
    read = state.mat_result
    opened_within: list[Any] = []

    def confirmed(**kwargs: Any) -> Any:
        opened_within.append(kwargs["within"])
        return read

    monkeypatch.setattr(role_resolve, "graph_confirm_resolve", confirmed)

    with caplog.at_level("WARNING"):
        near = asyncio.run(phase_place_materialise._nearest_real_for(state))

    assert _counted(population_reads) == _ANSWER_READ
    assert near is not None
    (relaxed,) = [
        spec
        for spec in near
        if any(
            slot.property == "nettingBasis" and slot.binding.state == "unbound"
            for slot in spec.definition.identity.slots
        )
    ]
    assert _provenance(relaxed) == _BILATERAL
    again = _resolve_resubmit(relaxed.definition.identity)
    assert again.spec.spec_id == relaxed.spec_id
    assert any(within for within in opened_within)
    monitor = [r for r in caplog.records if r.name == reading.__name__ and r.levelno >= 30]
    assert monitor == []


@pytest.mark.parametrize("scenario", ["D2", "D2-ranking"])
def test_a_part_naming_no_destination_reads_the_total_inside_its_read_alone(
    scenario: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A part naming no destination reads the bilateral composite at the population's total
    (`_DEVELOPING`) inside its read alone (`reading.ReadBinding`): its place after the read and
    the where slot of its bindings name no destination, as before the read. A nearest-real
    suggestion read at that place elects there, reads at the total in its own read and mints
    an identity binding no destination, as the composite the answer is minted as binds
    none, while it lists the total among its entities as the answer does
    (`reading.roles_read`)."""
    _patched_coverage(monkeypatch, _AT_TOTAL)
    before: list[ResolveState] = []
    task = phase_place_materialise._materialise_task

    def reading_from(state: ResolveState) -> Any:
        before.append(dataclasses.replace(state.place))
        return task(state)

    monkeypatch.setattr(phase_place_materialise, "_materialise_task", reading_from)
    state = _ended_state(monkeypatch, scenario, sdmx=_sdmx(_AT_TOTAL))

    assert state.reading is not None and state.reading.table is not None
    (unread,) = before
    for place in (unread, state.place):
        assert place.roles.destination_dcids == []
        assert place.roles.where_dcids == []
        assert place.roles.obs_destination_dcid != _DEVELOPING
    assert not [b for b in state.bindings if b.axis == "where" and b.value_dcids]

    # The relaxation reads at the part's own place, as for a shape installed with its
    # settled context alone (`grounding._relaxation_place`).
    state.unsettled_place = None
    read = state.mat_result
    subjects: list[list[str]] = []

    def confirmed(**kwargs: Any) -> Any:
        subjects.append(list(kwargs["subject_dcids"]))
        return read

    monkeypatch.setattr(role_resolve, "graph_confirm_resolve", confirmed)
    level = (
        state.shape_draft,
        Relaxation(axis="what", property_dcid="measurementDenominator", dropped_value_dcids=["x"]),
    )
    near = asyncio.run(phase_place_materialise._nearest_real_for(state, level=level))

    assert near is not None
    (suggestion,) = near
    assert _provenance(suggestion) == _BILATERAL
    assert "destination" not in suggestion.definition.identity.entities
    assert subjects == [[_DEVELOPING]]
    listed = [(entity.ref, entity.role) for entity in suggestion.resolved.entities]
    answered = [(entity.ref.ref, entity.role.axis) for entity in state.entity_objects]
    assert (_DEVELOPING, "destination") in listed
    assert listed == answered


def test_a_standard_relaxation_reads_with_the_totals_its_reading_binds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relaxation read through the standard route reads the totals the reading it elected
    binds on the axes the part leaves out (`reading.ReadBinding.entities`), as the answer's
    own read does (`phase_place_materialise._materialise_task`), so a suggestion minted as
    the composite reads the rows it is labelled with."""
    _patched_coverage(monkeypatch, _AT_TOTAL)
    state = _ended_state(monkeypatch, "D2", sdmx=_sdmx(_AT_TOTAL))
    assert state.reading is not None and state.reading.totals
    state.unsettled_place = None
    read = state.mat_result
    entities: list[dict[str, list[str]]] = []

    def standard(**kwargs: Any) -> Any:
        entities.append(kwargs["entities"])
        return read

    monkeypatch.setattr(grounding, "is_role_bearing", lambda _shape: False)
    monkeypatch.setattr(grounding, "standard_resolve", standard)
    level = (
        state.shape_draft,
        Relaxation(axis="what", property_dcid="measurementDenominator", dropped_value_dcids=["x"]),
    )
    asyncio.run(phase_place_materialise._nearest_real_for(state, level=level))

    assert entities
    assert all(
        _DEVELOPING in [dcid for dcids in bound.values() for dcid in dcids] for bound in entities
    )
