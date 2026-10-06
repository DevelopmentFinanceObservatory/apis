"""Tests that assembly includes each `RegionResult.cause` in the wire response when present."""

from __future__ import annotations

from dfo_platform import Cause
from search.engine.conjoin import assemble_region
from search.engine.envelope import now_ms
from search.engine.region.result import RegionResult
from search.models import Unsupported


def _assemble(region: RegionResult) -> dict:
    response = assemble_region(
        region,
        extra_warnings=[],
        start_ms=now_ms(),
        engine_build="test",
        max_candidates=10,
    )
    return response.model_dump(mode="json", exclude_none=True)


def _no_data_region(*, cause: Cause | None) -> RegionResult:
    return RegionResult(
        variable_text="aid",
        status="no_data",
        specs=(),
        no_data_reason="rate_coverage_gap",
        warnings=(),
        timing_by_step={},
        cause=cause,
    )


def _unsupported_region(*, cause: Cause | None) -> RegionResult:
    return RegionResult(
        variable_text="aid",
        status="unsupported",
        specs=(),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        unsupported=Unsupported(
            code="recipe_not_declared",
            pointer="/input/facets/unit",
            detail="no declared conversion recipe",
            definition={},
            cause=cause,
        ),
    )


class TestCauseReachesTheWireThroughAssembly:
    def test_no_data_part_carries_cause_on_the_wire(self):
        cause = Cause(code="rate_missing", entities=["EntityA@2020"], truncated=False)
        body = _assemble(_no_data_region(cause=cause))
        part = body["parts"][0]
        assert part["no_data"]["cause"] == {
            "code": "rate_missing",
            "entities": ["EntityA@2020"],
            "truncated": False,
        }

    def test_no_data_part_omits_cause_when_region_carries_none(self):
        body = _assemble(_no_data_region(cause=None))
        part = body["parts"][0]
        assert "cause" not in part["no_data"]

    def test_unsupported_part_carries_cause_on_the_wire(self):
        cause = Cause(code="nominal_undeclared")
        body = _assemble(_unsupported_region(cause=cause))
        part = body["parts"][0]
        assert part["unsupported"]["cause"] == {
            "code": "nominal_undeclared",
            "entities": [],
            "truncated": False,
        }

    def test_unsupported_part_omits_cause_when_region_carries_none(self):
        body = _assemble(_unsupported_region(cause=None))
        part = body["parts"][0]
        assert "cause" not in part["unsupported"]
