"""ProblemDetails/Envelope: field set, omit-when-absent, Envelope.response,
ProblemSpec/Envelope.render, and ProblemSpec.to_canonical_data.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dfo_platform.problem import (
    Alternative,
    Cause,
    Envelope,
    ProblemDetails,
    ProblemIssue,
    ProblemSpec,
    UnregisteredCode,
)

CATALOGUE = {
    "known_code": ("Known Code", "A known default detail."),
}


def _envelope() -> Envelope:
    return Envelope(catalogue=CATALOGUE, capability_version="v-test")


def test_details_carries_the_full_field_set():
    envelope = _envelope()
    problem = envelope.details(
        status=400,
        code="known_code",
        pointer="/input/query",
        alternatives=[Alternative(ref="a", label="A")],
        truncated=True,
        issues=[ProblemIssue(pointer="/x", member="m1")],
        constraint_id="donor_country_required",
        retryable=False,
    )
    assert isinstance(problem, ProblemDetails)
    assert problem.type == "/codes/known_code"
    assert problem.title == "Known Code"
    assert problem.status == 400
    assert problem.detail == "A known default detail."
    assert problem.code == "known_code"
    assert problem.pointer == "/input/query"
    assert problem.alternatives == [Alternative(ref="a", label="A")]
    assert problem.truncated is True
    assert problem.issues == [ProblemIssue(pointer="/x", member="m1")]
    assert problem.constraint_id == "donor_country_required"
    assert problem.trace_id
    assert problem.capability_version == "v-test"
    assert problem.retryable is False


def test_to_json_omits_absent_optional_members():
    envelope = _envelope()
    problem = envelope.details(status=400, code="known_code", retryable=False)
    body = problem.to_json()
    for absent in ("pointer", "alternatives", "truncated", "issues", "constraint_id"):
        assert absent not in body


def test_constraint_id_round_trips_from_details_through_to_json():
    envelope = _envelope()
    problem = envelope.details(
        status=400,
        code="known_code",
        constraint_id="donor_country_required",
        retryable=False,
    )
    assert problem.constraint_id == "donor_country_required"
    assert problem.to_json()["constraint_id"] == "donor_country_required"


def test_truncated_carried_only_alongside_alternatives():
    envelope = _envelope()

    with_alternatives = envelope.details(
        status=400,
        code="known_code",
        alternatives=[Alternative(ref="a", label="A")],
        truncated=True,
        retryable=False,
    )
    assert with_alternatives.truncated is True

    without_alternatives = envelope.details(
        status=400,
        code="known_code",
        truncated=True,
        retryable=False,
    )
    assert without_alternatives.truncated is None
    assert "truncated" not in without_alternatives.to_json()


def test_issues_round_trip_with_member_set_and_axis_absent():
    envelope = _envelope()
    problem = envelope.details(
        status=400,
        code="known_code",
        issues=[ProblemIssue(pointer="/a", member="donor")],
        retryable=False,
    )
    body = problem.to_json()
    assert body["issues"] == [{"pointer": "/a", "member": "donor"}]


def test_issues_round_trip_with_axis_set_and_member_absent():
    envelope = _envelope()
    problem = envelope.details(
        status=400,
        code="known_code",
        issues=[ProblemIssue(pointer="/b", axis="where")],
        retryable=False,
    )
    body = problem.to_json()
    assert body["issues"] == [{"pointer": "/b", "axis": "where"}]


def test_issue_code_and_detail_serialise_when_set_and_are_omitted_when_unset():
    envelope = _envelope()
    problem = envelope.details(
        status=400,
        code="known_code",
        issues=[
            ProblemIssue(pointer="/a", axis="unit", code="inner_code", detail="inner detail"),
            ProblemIssue(pointer="/b", axis="where"),
        ],
        retryable=False,
    )
    assert problem.to_json()["issues"] == [
        {"pointer": "/a", "axis": "unit", "code": "inner_code", "detail": "inner detail"},
        {"pointer": "/b", "axis": "where"},
    ]


def test_response_status_content_type_and_cache_control():
    envelope = _envelope()
    response = envelope.response(status=422, code="known_code", retryable=False)
    assert response.status_code == 422
    assert response.media_type == "application/problem+json"
    assert response.headers["cache-control"] == "no-store"
    assert response.body


def test_title_falls_back_to_http_reason_phrase_for_unknown_code():
    envelope = _envelope()
    assert envelope.title("nonexistent_code", 404) == "Not Found"


def test_title_falls_back_to_generic_error_for_unknown_status():
    envelope = _envelope()
    assert envelope.title("nonexistent_code", 999) == "Error"


def test_detail_falls_back_to_generic_sentence_for_unknown_code():
    envelope = _envelope()
    assert envelope.detail("nonexistent_code", None) == "Bad request"


def test_detail_prefers_explicit_detail_over_catalogue_default():
    envelope = _envelope()
    assert envelope.detail("known_code", "explicit detail") == "explicit detail"


def test_details_raises_unregistered_code_for_a_code_absent_from_the_catalogue():
    envelope = _envelope()
    with pytest.raises(UnregisteredCode) as exc_info:
        envelope.details(status=400, code="nonexistent_code", retryable=False)
    assert exc_info.value.code == "nonexistent_code"
    assert "nonexistent_code" in str(exc_info.value)


def test_response_raises_unregistered_code_for_a_code_absent_from_the_catalogue():
    envelope = _envelope()
    with pytest.raises(UnregisteredCode):
        envelope.response(status=400, code="nonexistent_code", retryable=False)


def test_response_merges_extra_headers():
    envelope = _envelope()
    response = envelope.response(
        status=405, code="known_code", retryable=False, headers={"Allow": "GET, POST"}
    )
    assert response.headers["allow"] == "GET, POST"
    assert response.headers["cache-control"] == "no-store"


def test_response_header_merge_does_not_let_a_caller_override_no_store():
    envelope = _envelope()
    response = envelope.response(
        status=405,
        code="known_code",
        retryable=False,
        headers={"Allow": "GET", "Cache-Control": "max-age=60"},
    )
    assert response.headers["cache-control"] == "no-store"
    _assert_exactly_one_cache_control_header(response, "no-store")


def test_response_header_merge_displaces_a_lowercase_cache_control():
    """A lowercase cache-control key is replaced by `no-store`."""
    envelope = _envelope()
    response = envelope.response(
        status=405,
        code="known_code",
        retryable=False,
        headers={"Allow": "GET", "cache-control": "max-age=60"},
    )
    _assert_exactly_one_cache_control_header(response, "no-store")


def _assert_exactly_one_cache_control_header(response, expected_value: str) -> None:
    """Checks raw headers for exactly one `cache-control` line."""
    cache_control_lines = [
        value.decode("latin-1")
        for name, value in response.raw_headers
        if name.decode("latin-1").lower() == "cache-control"
    ]
    assert cache_control_lines == [expected_value]


def test_response_header_merge_tolerates_none_and_empty_headers():
    envelope = _envelope()
    # None is the default; an empty mapping must be just as harmless.
    assert envelope.response(status=400, code="known_code", retryable=False, headers=None)
    assert envelope.response(status=400, code="known_code", retryable=False, headers={})


def test_render_round_trips_a_bare_spec_through_response():
    """`Envelope.render` matches `Envelope.response` for the same fields."""
    envelope = _envelope()
    spec = ProblemSpec(status=422, code="known_code", retryable=False)
    rendered = envelope.render(spec)
    direct = envelope.response(status=422, code="known_code", retryable=False)
    assert rendered.status_code == direct.status_code == 422
    assert rendered.body == direct.body
    assert dict(rendered.headers) == dict(direct.headers)


def test_render_round_trips_a_fully_populated_spec():
    envelope = _envelope()
    spec = ProblemSpec(
        status=400,
        code="known_code",
        retryable=False,
        detail="explicit detail",
        pointer="/input/query",
        alternatives=[Alternative(ref="a", label="A")],
        truncated=True,
        issues=[ProblemIssue(pointer="/x", member="m1")],
        constraint_id="donor_country_required",
        headers={"Allow": "GET, POST"},
    )
    rendered = envelope.render(spec)
    body = json.loads(bytes(rendered.body))
    assert body["detail"] == "explicit detail"
    assert body["pointer"] == "/input/query"
    assert body["alternatives"] == [{"ref": "a", "label": "A"}]
    assert body["truncated"] is True
    assert body["issues"] == [{"pointer": "/x", "member": "m1"}]
    assert body["constraint_id"] == "donor_country_required"
    assert rendered.headers["allow"] == "GET, POST"
    assert rendered.headers["cache-control"] == "no-store"


def test_to_canonical_data_is_json_serializable_with_every_field_populated():
    spec = ProblemSpec(
        status=400,
        code="known_code",
        retryable=True,
        detail="a detail",
        pointer="/a/b",
        alternatives=[Alternative(ref="ref", label="label")],
        truncated=True,
        issues=[ProblemIssue(pointer="/c", member="m", axis="ax", code="inner", detail="why")],
        constraint_id="donor_country_required",
        headers={"B-Header": "2", "A-Header": "1"},
    )
    data = spec.to_canonical_data()
    json.dumps(data)  # must not raise
    assert data["status"] == 400
    assert data["code"] == "known_code"
    assert data["retryable"] is True
    assert data["detail"] == "a detail"
    assert data["pointer"] == "/a/b"
    assert data["alternatives"] == [{"ref": "ref", "label": "label"}]
    assert data["truncated"] is True
    assert data["issues"] == [
        {
            "pointer": "/c",
            "member": "m",
            "axis": "ax",
            "alternatives": None,
            "truncated": None,
            "code": "inner",
            "detail": "why",
        }
    ]
    assert data["constraint_id"] == "donor_country_required"
    # Sorted for a stable digest input, regardless of construction order.
    assert list(data["headers"].keys()) == ["A-Header", "B-Header"]


_EVERY_FIELD_SPEC = ProblemSpec(
    status=422,
    code="known_code",
    retryable=True,
    detail="a detail",
    pointer="/a/b",
    alternatives=[Alternative(ref="ref", label="label")],
    truncated=True,
    issues=[
        ProblemIssue(
            pointer="/c",
            member="m",
            axis="ax",
            alternatives=[Alternative(ref="r1", label="One")],
            truncated=False,
        ),
        ProblemIssue(member="m2"),
    ],
    cause=Cause(code="rate_missing", entities=["country/KEN@2020"], truncated=True),
    constraint_id="donor_country_required",
    ceiling_kind="candidate_window",
    count=4000,
    cap=4000,
    headers={"A-Header": "1", "B-Header": "2"},
)


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(_EVERY_FIELD_SPEC, id="every_field"),
        pytest.param(ProblemSpec(status=500, code="internal_error", retryable=False), id="bare"),
    ],
)
def test_from_canonical_data_inverts_to_canonical_data_through_json(spec: ProblemSpec):
    data = json.loads(json.dumps(spec.to_canonical_data()))

    rebuilt = ProblemSpec.from_canonical_data(data)

    assert rebuilt == spec
    assert rebuilt.to_canonical_data() == spec.to_canonical_data()


def test_every_problem_spec_field_survives_the_round_trip():
    # A field added to `ProblemSpec` and left out of `to_canonical_data` fails here. The log line
    # fields are the one exception: they stay out of the wire body and the canonical form.
    data = _EVERY_FIELD_SPEC.to_canonical_data()

    wire_fields = set(ProblemSpec._fields) - {"log", "log_level", "log_traceback"}
    assert set(data) == wire_fields
    assert all(
        getattr(_EVERY_FIELD_SPEC, field) != ProblemSpec._field_defaults[field]
        for field in wire_fields & set(ProblemSpec._field_defaults)
    )


def test_from_canonical_data_defaults_a_key_dumped_before_the_field_existed():
    data = _EVERY_FIELD_SPEC.to_canonical_data()
    del data["ceiling_kind"]

    assert ProblemSpec.from_canonical_data(data) == _EVERY_FIELD_SPEC._replace(ceiling_kind=None)


@pytest.mark.parametrize(
    ("key", "value"),
    [("severity", "high"), ("log", "exemplar"), ("log_level", 40), ("log_traceback", True)],
)
def test_from_canonical_data_refuses_a_key_to_canonical_data_never_writes(key, value):
    data = {**_EVERY_FIELD_SPEC.to_canonical_data(), key: value}

    with pytest.raises(ValidationError):
        ProblemSpec.from_canonical_data(data)


def test_from_canonical_data_refuses_a_value_of_the_wrong_type():
    data = {**_EVERY_FIELD_SPEC.to_canonical_data(), "issues": [{"alternatives": "USD"}]}

    with pytest.raises(ValidationError, match="issues"):
        ProblemSpec.from_canonical_data(data)


def test_to_canonical_data_is_json_serializable_with_nothing_populated():
    spec = ProblemSpec(status=500, code="internal_error", retryable=False)
    data = spec.to_canonical_data()
    json.dumps(data)  # must not raise
    assert data["detail"] is None
    assert data["pointer"] is None
    assert data["alternatives"] is None
    assert data["truncated"] is False
    assert data["issues"] is None
    assert data["constraint_id"] is None
    assert data["headers"] is None
