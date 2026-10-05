"""Tests `cause` serialization on Unsupported and NoData.
Also tests propagation from EngineInputError to ProblemDetails.
"""

from __future__ import annotations

from dfo_platform import Cause, Envelope, ProblemSpec
from search.engine.app import classify_search_exception
from search.engine.codes import CODE_CATALOG
from search.engine.errors import EngineInputError
from search.models import NoData, Unsupported


class TestCauseOnUnsupported:
    def test_omitted_when_none(self):
        unsupported = Unsupported(
            code="unit_request_unsupported", pointer="/x", detail="d", definition={}
        )
        dumped = unsupported.model_dump(mode="json", exclude_none=True)
        assert "cause" not in dumped

    def test_round_trips_when_present(self):
        cause = Cause(code="nominal_undeclared", entities=["StatVarX"], truncated=False)
        unsupported = Unsupported(
            code="recipe_not_declared", pointer="/x", detail="d", definition={}, cause=cause
        )
        dumped = unsupported.model_dump(mode="json", exclude_none=True)
        assert dumped["cause"] == {
            "code": "nominal_undeclared",
            "entities": ["StatVarX"],
            "truncated": False,
        }


class TestCauseOnNoData:
    def test_omitted_when_none(self):
        no_data = NoData(reason="no_observations")
        dumped = no_data.model_dump(mode="json", exclude_none=True)
        assert "cause" not in dumped

    def test_round_trips_when_present(self):
        cause = Cause(code="rate_missing", entities=["EntityA@2020"], truncated=True)
        no_data = NoData(reason="rate_coverage_gap", cause=cause)
        dumped = no_data.model_dump(mode="json", exclude_none=True)
        assert dumped["cause"] == {
            "code": "rate_missing",
            "entities": ["EntityA@2020"],
            "truncated": True,
        }


class TestEngineInputErrorCause:
    def test_defaults_to_none(self):
        exc = EngineInputError("bad", code="request_invalid")
        assert exc.cause is None

    def test_classify_search_exception_omits_cause_when_absent(self):
        exc = EngineInputError("bad", code="request_invalid", status=400)
        spec = classify_search_exception(exc)
        assert isinstance(spec, ProblemSpec)
        assert spec.cause is None
        assert "cause" not in spec.to_canonical_data() or spec.to_canonical_data()["cause"] is None

    def test_classify_search_exception_threads_cause(self):
        cause = Cause(code="recipe_undeclared")
        exc = EngineInputError("bad", code="recipe_not_declared", status=422, cause=cause)
        spec = classify_search_exception(exc)
        assert spec.cause == cause

    def test_problem_details_body_emits_cause_when_present(self):
        env = Envelope(catalogue=CODE_CATALOG, capability_version="cap_test")
        body = env.details(
            status=422,
            code="recipe_not_declared",
            retryable=False,
            cause=Cause(code="nominal_undeclared"),
        )
        wire = body.to_json()
        assert wire["cause"] == {"code": "nominal_undeclared", "entities": [], "truncated": False}

    def test_problem_details_body_omits_cause_when_absent(self):
        env = Envelope(catalogue=CODE_CATALOG, capability_version="cap_test")
        body = env.details(status=422, code="recipe_not_declared", retryable=False)
        assert "cause" not in body.to_json()
