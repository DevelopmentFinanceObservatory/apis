"""Checks when fan-out records a degraded leg, timeout, or request failure."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest

from entity_resolution import Admission
from materialise.place_scope import DefaultPresence
from search.engine.binders import GeminiBinder
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.errors import GraphInfraError
from search.engine.fanout import _FanOutResult, _run_variables_streaming
from search.engine.place import read_default_presence
from search.engine.region import RegionResult
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM, make_resolver


async def _drain(agen: AsyncIterator[object]) -> None:
    async for _ev in agen:
        pass


def _fanout_kwargs(**overrides: Any) -> dict[str, Any]:
    """Shared plumbing for a two-leg `_run_variables_streaming` run: fresh
    fakes, an unbounded deadline, and no semaphore, overridden per test
    where a case needs something else (a short deadline, for instance).
    """
    graph = FakeGraph()
    resolver = make_resolver(graph)
    admission = Admission(graph=graph)
    sdmx = FakeSdmxClient(observations={}, availability={})
    llm = FakeLLM(responses={})
    kwargs: dict[str, Any] = {
        "entities": [],
        "date_request": None,
        "query": "",
        "pac": False,
        "graph": graph,
        "resolver": resolver,
        "admission": admission,
        "llm": llm,
        "binder": GeminiBinder(),
        "sdmx": sdmx,
        "base_timing": {},
        "pre_resolved": None,
        "deadline": time.monotonic() + 3600,
        "use_semaphore": False,
        "coverage": empty_coverage_snapshot(),
    }
    kwargs.update(overrides)
    return kwargs


def _no_data_region(variable: str) -> RegionResult:
    return RegionResult(
        variable_text=variable,
        status="no_data",
        specs=(),
        no_data_reason="variable_not_resolved",
        warnings=(),
        timing_by_step={},
    )


def test_non_infra_exception_under_degrade_sets_degraded_leg():
    """A swallowed application error yields a no-data part and sets `degraded_leg`."""

    async def _stub_resolve_variable(variable: str, **kwargs: object) -> RegionResult:
        if kwargs["variable_index"] == 0:
            raise ValueError("simulated leg failure")
        return _no_data_region(variable)

    async def _scenario() -> _FanOutResult:
        out = _FanOutResult()
        with patch("search.engine.fanout.resolve_variable", new=_stub_resolve_variable):
            await _drain(
                _run_variables_streaming(
                    ["v0", "v1"],
                    degrade_on_error=True,
                    out=out,
                    **_fanout_kwargs(),
                )
            )
        return out

    out = asyncio.run(_scenario())
    assert out.degraded_leg is True
    assert out.regions[0].status == "no_data"
    assert out.regions[0].no_data_reason == "variable_not_resolved"
    assert [region.degraded for region in out.regions] == [True, False]


def test_clean_multi_variable_run_leaves_degraded_leg_false():
    """No leg raises: degraded_leg stays at its default False."""

    async def _stub_resolve_variable(variable: str, **_kwargs: object) -> RegionResult:
        return _no_data_region(variable)

    async def _scenario() -> _FanOutResult:
        out = _FanOutResult()
        with patch("search.engine.fanout.resolve_variable", new=_stub_resolve_variable):
            await _drain(
                _run_variables_streaming(
                    ["v0", "v1"],
                    degrade_on_error=True,
                    out=out,
                    **_fanout_kwargs(),
                )
            )
        return out

    out = asyncio.run(_scenario())
    assert out.degraded_leg is False
    assert out.timed_out is False


def test_a_leg_whose_region_stands_on_less_than_it_read_sets_degraded_leg():
    """A leg that swallowed a member walk's failed read (`RegionResult.degraded`) is degraded,
    so the response is not cached."""

    async def _stub_resolve_variable(variable: str, **kwargs: object) -> RegionResult:
        region = _no_data_region(variable)
        return dataclasses.replace(region, degraded=kwargs["variable_index"] == 1)

    async def _scenario() -> _FanOutResult:
        out = _FanOutResult()
        with patch("search.engine.fanout.resolve_variable", new=_stub_resolve_variable):
            await _drain(
                _run_variables_streaming(
                    ["v0", "v1"],
                    degrade_on_error=True,
                    out=out,
                    **_fanout_kwargs(),
                )
            )
        return out

    out = asyncio.run(_scenario())
    assert out.degraded_leg is True
    assert out.timed_out is False


@pytest.mark.parametrize(
    ("presence", "unsettled"),
    [
        pytest.param(DefaultPresence(countries=(), scopes={}), False, id="settled"),
        pytest.param(
            DefaultPresence(countries=(), scopes={}, unsettled=frozenset({"Count_Person"})),
            True,
            id="unsettled",
        ),
    ],
)
def test_a_default_scope_read_that_left_a_scope_unsettled_is_a_lost_read(
    presence: DefaultPresence, unsettled: bool
) -> None:
    with (
        patch("search.engine.place.place_scope.read_default_presence", return_value=presence),
        recording_degraded_reads() as lost,
    ):
        read_default_presence(graph=FakeGraph(), sdmx=FakeSdmxClient(), stat_vars=[])

    assert lost.degraded is unsettled


def test_infra_error_fails_the_request_and_never_sets_degraded_leg():
    """An infrastructure error fails the request and leaves `degraded_leg` false."""

    async def _stub_resolve_variable(variable: str, **kwargs: object) -> RegionResult:
        if kwargs["variable_index"] == 0:
            raise GraphInfraError("simulated graph transport failure")
        return _no_data_region(variable)

    async def _scenario() -> tuple[_FanOutResult, BaseException | None]:
        out = _FanOutResult()
        raised: BaseException | None = None
        try:
            with patch("search.engine.fanout.resolve_variable", new=_stub_resolve_variable):
                await _drain(
                    _run_variables_streaming(
                        ["v0", "v1"],
                        degrade_on_error=True,
                        out=out,
                        **_fanout_kwargs(),
                    )
                )
        except GraphInfraError as exc:
            raised = exc
        return out, raised

    out, raised = asyncio.run(_scenario())
    assert raised is not None, "expected the injected infra error to surface"
    assert out.degraded_leg is False
    assert out.regions == []


def test_deadline_backfill_sets_timed_out_and_leaves_degraded_leg_false():
    """Deadline backfill sets `timed_out` without marking an exception as degraded."""

    async def _stub_resolve_variable(variable: str, **kwargs: object) -> RegionResult:
        if kwargs["variable_index"] == 0:
            return _no_data_region(variable)
        await asyncio.Event().wait()  # leg 1 never finishes
        raise AssertionError("unreachable")

    async def _scenario() -> _FanOutResult:
        out = _FanOutResult()
        kwargs = _fanout_kwargs(deadline=time.monotonic() + 0.05)
        with patch("search.engine.fanout.resolve_variable", new=_stub_resolve_variable):
            await _drain(
                _run_variables_streaming(
                    ["v0", "v1"],
                    degrade_on_error=True,
                    out=out,
                    **kwargs,
                )
            )
        return out

    out = asyncio.run(_scenario())
    assert out.timed_out is True
    assert out.degraded_leg is False
