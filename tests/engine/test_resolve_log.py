"""Pins the outcome fields of the one `resolve` log line `_log_resolve` writes per request."""

from __future__ import annotations

import hashlib
import json
import logging

import pytest

from search.engine.core import _log_resolve, _ResolveRecord
from search.engine.envelope import make_diagnostics, make_query_echo
from search.models import (
    RESOLVE_RESPONSE_ADAPTER,
    DeclinedAnswer,
    NoData,
    NoDataPart,
    Resolution,
    ResolveRequest,
    Scope,
    Unsupported,
    UnsupportedPart,
)
from tests.engine._harness import make_request

_USAGE = {"input_tokens": 120, "output_tokens": 30, "cached_tokens": 0}


def _logged(request: ResolveRequest, record: _ResolveRecord) -> dict:
    """The JSON object of the one line `_log_resolve` writes for `record`."""
    # A handler on the module's own logger, since the `search` logger may stop propagation.
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    core_logger = logging.getLogger("search.engine.core")
    core_logger.addHandler(handler)
    previous_level = core_logger.level
    core_logger.setLevel(logging.INFO)
    try:
        _log_resolve(request, record, caller=None, transport="buffered")
    finally:
        core_logger.removeHandler(handler)
        core_logger.setLevel(previous_level)
    (line,) = [r.getMessage() for r in records]
    return json.loads(line.removeprefix("resolve "))


def _definite_part(worked_example: dict):
    response = RESOLVE_RESPONSE_ADAPTER.validate_python(worked_example)
    assert isinstance(response, Resolution)
    return response.parts[0]


_NO_DATA = NoDataPart(status="no_data", no_data=NoData(reason="entity_not_resolved"))
_UNSUPPORTED = UnsupportedPart(
    status="unsupported",
    unsupported=Unsupported(
        code="unit_request_unsupported", pointer="/identity", detail="per capita", definition={}
    ),
)


def _resolution(parts: list) -> Resolution:
    return Resolution(
        query_echo=make_query_echo("q", extract_skipped=False),
        diagnostics=make_diagnostics("test-build", [], {}, 5, llm_usage=_USAGE),
        parts=parts,
    )


@pytest.mark.parametrize(
    ("family", "outcome", "part_statuses", "no_data_reasons", "unsupported_codes"),
    [
        ("definite", "definite", ["definite"], [], []),
        (
            "mixed",
            "mixed",
            ["definite", "no_data", "unsupported"],
            ["entity_not_resolved"],
            ["unit_request_unsupported"],
        ),
        ("no_data", "no_data", ["no_data"], ["entity_not_resolved"], []),
        ("declined", "declined", [], [], []),
        ("no_response", None, [], [], []),
    ],
)
def test_the_resolve_line_reports_the_outcome_of_each_answer_family(
    worked_example_response, family, outcome, part_statuses, no_data_reasons, unsupported_codes
):
    definite = _definite_part(worked_example_response)
    response = {
        "definite": _resolution([definite]),
        "mixed": _resolution([definite, _NO_DATA, _UNSUPPORTED]),
        "no_data": _resolution([_NO_DATA]),
        "declined": DeclinedAnswer(
            query_echo=make_query_echo("q", extract_skipped=False),
            diagnostics=make_diagnostics("test-build", [], {}, 5, llm_usage=_USAGE),
            reason="out_of_scope",
            scope=Scope(),
        ),
        "no_response": None,
    }[family]
    record = _ResolveRecord(trace_id="abc123", response=response)

    logged = _logged(make_request("  Fertility Rate in KENYA "), record)

    assert logged["trace_id"] == "abc123"
    assert logged["outcome"] == outcome
    assert logged["part_statuses"] == part_statuses
    assert logged["no_data_reasons"] == no_data_reasons
    assert logged["unsupported_codes"] == unsupported_codes
    assert logged["llm_usage"] == (_USAGE if response is not None else None)
    assert logged["timed_out"] is None
    assert logged["degraded_leg"] is None


def test_the_query_hash_keys_a_casefolded_raw_text_query_and_carries_no_text():
    record = _ResolveRecord(trace_id="abc123")

    logged = _logged(make_request("  Fertility Rate in KENYA "), record)

    assert logged["query_sha256"] == hashlib.sha256(b"fertility rate in kenya").hexdigest()
    assert "fertility" not in json.dumps(logged).casefold()


@pytest.mark.parametrize(
    ("query", "encoded"),
    [("\ud800", b"\xed\xa0\x80"), ("\udfff", b"\xed\xbf\xbf")],
)
def test_the_query_hash_preserves_unpaired_surrogates(query, encoded):
    logged = _logged(make_request(query), _ResolveRecord(trace_id="abc123"))

    assert logged["query_sha256"] == hashlib.sha256(encoded).hexdigest()
