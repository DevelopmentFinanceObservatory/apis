"""Resubmit timing includes each phase that ran, including phases that return
no_data. Tests assert timing fields and use an increasing fake clock for
phase-by-phase checks; wall-clock magnitudes remain unpinned.
"""

from __future__ import annotations

import itertools

from search.engine.region import (
    promote_bindings,
    promote_edits,
    promote_ground,
    promote_materialise,
    promote_places,
    promote_reread,
    promote_shape,
    promote_survivors,
    promote_validate,
    resolve_spec_resubmit,
)
from search.engine.region.state import EngineOptions
from search.models import (
    RawTextInput,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityPayload,
)
from tests.engine._base_identity import GDP_SHAPE, make_identity, subject_entity
from tests.engine._builders import unbound_slot as _unbound_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve, walk_options
from tests.engine.test_resubmit_regressions import (
    _SOURCE_BOUND_OUT_OF_WINDOW_OBS,
    _WINDOW_REQUESTED,
    _standard_window_inp,
)
from tests.fixtures import FakeGraph

_GDP_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_ENTITY_DCID = "country/IND"


def _standard_promote_input() -> ResubmitIdentityPayload:
    entities = [subject_entity(_ENTITY_DCID)]
    return ResubmitIdentityPayload(
        identity=make_identity(
            GDP_SHAPE,
            # activitySource is this SV's only shape-declared what/how slot
            # key; required present (unbound is fine).
            [_unbound_slot("how", "activitySource", "activity source")],
            entities=entities,
        ),
    )


def _assert_nonneg_int(value: object) -> None:
    assert isinstance(value, int) and not isinstance(value, bool)
    assert value >= 0


# Every module holding a promote-path phase that stamps its own
# state.timing entry (region/resubmit_promote.py's own module docstring
# names each phase's home module). now_ms is patched in all of them so a
# single strictly increasing tick counter reads as every phase's own
# deterministic elapsed time, regardless of which module now owns the phase.
_PROMOTE_PHASE_MODULES = (
    promote_validate,
    promote_reread,
    promote_edits,
    promote_survivors,
    promote_shape,
    promote_places,
    promote_bindings,
    promote_materialise,
    promote_ground,
)


def _patch_promote_now_ms(monkeypatch, tick: itertools.count) -> None:
    """Use one increasing fake clock across the promote-phase modules."""
    for module in _PROMOTE_PHASE_MODULES:
        monkeypatch.setattr(module, "now_ms", lambda: next(tick))


# ---------------------------------------------------------------------------
# Standard-promote resubmit
# ---------------------------------------------------------------------------


def test_standard_promote_resubmit_with_coverage_populates_materialise_answer_options():
    region = resolve_spec_resubmit(
        inp=_standard_promote_input(),
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=walk_options(FakeGraph(), EngineOptions(include_option_coverage=True)),
    )
    assert region.status == "definite"
    by_step = region.timing_by_step
    for key in ("materialise", "answer", "options"):
        assert key in by_step, f"expected {key!r} in timing_by_step, got {by_step!r}"
        _assert_nonneg_int(by_step[key])


def test_standard_promote_resubmit_without_coverage_still_times_materialise_answer():
    region = resolve_spec_resubmit(
        inp=_standard_promote_input(),
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=walk_options(FakeGraph(), EngineOptions(include_option_coverage=False)),
    )
    assert region.status == "definite"
    by_step = region.timing_by_step
    for key in ("materialise", "answer"):
        assert key in by_step
        _assert_nonneg_int(by_step[key])
    if "options" in by_step:
        assert by_step["options"] < 50, "no probe ran; options bucket should stay near-zero"


def test_standard_promote_resubmit_times_every_phase(monkeypatch):
    """Every phase records a nonzero timing entry under the increasing fake clock."""
    tick = itertools.count(1_000)
    _patch_promote_now_ms(monkeypatch, tick)

    region = resolve_spec_resubmit(
        inp=_standard_promote_input(),
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=walk_options(FakeGraph(), EngineOptions(include_option_coverage=False)),
    )
    assert region.status == "definite"
    by_step = region.timing_by_step
    for key in (
        "validate",
        "walk",
        "reread_arcs",
        "detect_edits",
        "resolve_survivors",
        "shape",
        "where_places",
        "bind",
        "materialise",
        "answer",
    ):
        assert key in by_step, f"expected {key!r} in timing_by_step, got {by_step!r}"
        assert by_step[key] >= 1, f"{key!r} should read a nonzero tick, got {by_step[key]!r}"


def test_standard_promote_early_no_data_exit_times_ran_phases(monkeypatch):
    """An identity electing no member preserves timings through the phase that returns
    no_data."""
    tick = itertools.count(1_000)
    _patch_promote_now_ms(monkeypatch, tick)

    entities = [subject_entity(_ENTITY_DCID)]
    inp = ResubmitIdentityPayload(
        identity=make_identity(
            {**GDP_SHAPE, "population_type": "NoSuchActivity"}, entities=entities
        ),
    )

    region = resolve_spec_resubmit(
        inp=inp,
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=walk_options(FakeGraph()),
    )
    assert region.status == "no_data"
    assert region.no_data_reason == "no_matching_slice"
    by_step = region.timing_by_step
    for key in ("validate", "walk", "reread_arcs"):
        assert key in by_step, f"expected {key!r} in timing_by_step, got {by_step!r}"
        assert by_step[key] >= 1, f"{key!r} should read a nonzero tick, got {by_step[key]!r}"


def test_window_no_observations_answer_times_every_ran_phase(monkeypatch):
    """A requested window holding no observation answers `no_observations` in
    `_phase_ground_build`, before a Spec is grounded. Every phase that ran reaches the
    returned RegionResult, as on the definite path, and the answer step, which did not run,
    is absent.
    """
    tick = itertools.count(1_000)
    _patch_promote_now_ms(monkeypatch, tick)

    inp = _standard_window_inp(requested_window=_WINDOW_REQUESTED)
    region = resolve_spec_resubmit(
        inp=inp,
        graph=FakeGraph(obs=_SOURCE_BOUND_OUT_OF_WINDOW_OBS),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(
            FakeGraph(obs=_SOURCE_BOUND_OUT_OF_WINDOW_OBS),
            EngineOptions(include_observations=False),
        ),
    )
    assert region.status == "no_data"
    assert region.no_data_reason == "no_observations"
    by_step = region.timing_by_step
    for key in (
        "validate",
        "walk",
        "reread_arcs",
        "detect_edits",
        "resolve_survivors",
        "shape",
        "where_places",
        "bind",
        "materialise",
    ):
        assert key in by_step, f"expected {key!r} in timing_by_step, got {by_step!r}"
        assert by_step[key] >= 1, f"{key!r} should read a nonzero tick, got {by_step[key]!r}"
    assert "answer" not in by_step


# ---------------------------------------------------------------------------
# raw_text buffered: the options bucket for the inline coverage pass
# ---------------------------------------------------------------------------


def test_raw_text_buffered_with_coverage_populates_options_bucket():
    req = ResolveRequest(
        input=RawTextInput(query="fertility rate in Kenya"),
        response_options=ResolveOptions(include_option_coverage=True),
    )
    part = offline_resolve(req, sdmx=FakeSdmxClient()).parts[0]
    assert part.status == "definite"
    timing = part.timing_ms
    assert timing is not None
    assert timing.by_step is not None
    assert "options" in timing.by_step
    _assert_nonneg_int(timing.by_step["options"])
    # The named steps stay populated too -- "options" is additive, not a replacement.
    for key in ("recall", "shape", "bind", "materialise", "answer"):
        assert key in timing.by_step


def test_raw_text_buffered_without_coverage_options_bucket_absent_or_zero():
    req = ResolveRequest(input=RawTextInput(query="fertility rate in Kenya"))
    part = offline_resolve(req, sdmx=FakeSdmxClient()).parts[0]
    assert part.status == "definite"
    timing = part.timing_ms
    assert timing is not None
    by_step = timing.by_step or {}
    if "options" in by_step:
        assert by_step["options"] < 50, "no probe ran; options bucket should stay near-zero"
