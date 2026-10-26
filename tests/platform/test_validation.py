"""Checks validation error pointers and the 400/422 refusal split."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

import pytest
from fastapi import APIRouter, Body
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

from datasets_api.mint_models import MintRequest
from dfo_platform.app import Surface, create_app
from dfo_platform.named import MAX_VALUE_CHARS
from dfo_platform.validation import (
    pointer_for_error,
    refusal_with_alternatives,
    reset_request_body_model,
    set_request_body_model,
    validation_problem,
)
from search.models import ResolveRequest
from search.models.identity import (
    IdentityFacetBinding,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
)


def _identity() -> dict:
    return {
        "shape": {"kind": "statvar_schema", "population_type": "df-flow"},
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
    }


def _errors_for(model: type, body: dict) -> list[dict]:
    """Return real Pydantic errors with FastAPI's `body` location prefix.

    Real errors preserve discriminator locations that hand-written tuples miss.
    """
    try:
        model.model_validate(body)
    except ValidationError as exc:
        return [{**e, "loc": ("body", *e["loc"])} for e in exc.errors()]
    raise AssertionError("expected body to fail validation")


def test_pointer_strips_discriminated_union_tag_from_path():
    # Pydantic inserts the `raw_text` union tag into `loc`; it is not a body key.
    body = {"input": {"kind": "raw_text", "bogus": True}}
    exc = RequestValidationError([], body=body)
    error = {"loc": ("body", "input", "raw_text", "bogus"), "type": "bool_type"}
    assert pointer_for_error(exc, error) == "/input/bogus"


def test_pointer_keeps_terminal_segment_for_missing_required_field():
    # Keep the missing terminal segment so the pointer names the absent field.
    body = {"input": {"kind": "raw_text"}}
    exc = RequestValidationError([], body=body)
    error = {"loc": ("body", "input", "query"), "type": "missing"}
    assert pointer_for_error(exc, error) == "/input/query"


def test_pointer_resolves_tag_colliding_with_a_real_sibling_field():
    # #581: The `spec_id` union tag collides with a sibling field and hides `scope`.
    body = {
        "input": {
            "kind": "spec_id",
            "spec_id": "spec_abc123",
            "scope": {"window": {"kind": "since"}},
        }
    }
    errors = _errors_for(MintRequest, body)
    exc = RequestValidationError(errors, body=body)
    assert pointer_for_error(exc, errors[0], model=MintRequest) == "/input/scope/window"


def test_pointer_resolves_tag_colliding_with_a_same_named_list_field():
    # #628: Pydantic repeats the `candidates` union tag and list key in `loc`.
    # The pointer must treat `max_candidates` as a sibling, not a list member.
    body = {
        "input": {
            "kind": "dataset_resubmit",
            "entries": [
                {
                    "entry_kind": "identity",
                    "variable_text": "ODA",
                    "identity": _identity(),
                },
                {
                    "entry_kind": "candidates",
                    "variable_text": "ODA",
                    "ordering": "broadest_first",
                    "truncated": False,
                    "candidates": [{"identity": _identity()}, {"identity": _identity()}],
                    "max_candidates": 6,
                },
            ],
        }
    }
    errors = _errors_for(ResolveRequest, body)
    assert pointer_for_error(
        RequestValidationError(errors, body=body), errors[0], model=ResolveRequest
    ) == ("/input/entries/1/max_candidates")


def test_pointer_keeps_missing_field_matching_the_discriminator_tag_name():
    # Valid control: confirm this union arm accepts `spec_id` before testing its omission.
    MintRequest.model_validate({"input": {"kind": "spec_id", "spec_id": "spec_abc123"}})

    # #581: `loc` repeats the `spec_id` tag and missing field; keep the terminal field.
    body = {"input": {"kind": "spec_id"}}
    errors = _errors_for(MintRequest, body)
    exc = RequestValidationError(errors, body=body)
    assert pointer_for_error(exc, errors[0], model=MintRequest) == "/input/spec_id"


def test_pointer_does_not_misalign_a_genuine_error_onto_the_wrong_entry():
    # #581/#628: Two entries expose index drift; the genuine error is under entry 1.
    bad_identity = _identity()
    bad_identity["slots"] = "not-a-list"
    body = {
        "input": {
            "kind": "dataset_resubmit",
            "entries": [
                {
                    "entry_kind": "identity",
                    "variable_text": "ODA",
                    "identity": _identity(),
                },
                {
                    "entry_kind": "candidates",
                    "variable_text": "ODA",
                    "ordering": "broadest_first",
                    "truncated": False,
                    "candidates": [{"identity": bad_identity}, {"identity": _identity()}],
                },
            ],
        }
    }
    errors = _errors_for(ResolveRequest, body)
    assert (
        pointer_for_error(
            RequestValidationError(errors, body=body), errors[0], model=ResolveRequest
        )
        == "/input/entries/1/candidates/0/identity/slots"
    )


def test_pointer_drops_union_type_tag_on_search_identity_currency():
    # Pydantic adds the failed union arm name to `loc`, though `usd` has no such key.
    body = {"currency": "usd", "prices": "current"}
    errors = _errors_for(IdentityUnitRequest, body)
    exc = RequestValidationError(errors, body=body)
    assert pointer_for_error(exc, errors[0], model=IdentityUnitRequest) == "/currency"


def test_pointer_drops_union_type_tag_when_the_posted_value_is_an_object():
    # Drop the synthetic string-arm tag when the posted currency is an object.
    body = {"currency": {"role": "nowhere"}, "prices": "current"}
    errors = _errors_for(IdentityUnitRequest, body)
    exc = RequestValidationError(errors, body=body)
    assert pointer_for_error(exc, errors[0], model=IdentityUnitRequest) == "/currency"


def test_pointer_drops_union_type_tag_when_the_posted_value_is_an_object_nested():
    # The same synthetic tag occurs one level deeper under `request`.
    body = {"state": "value", "request": {"currency": {"role": "nowhere"}, "prices": "current"}}
    errors = _errors_for(IdentityFacetValueRequest, body)
    exc = RequestValidationError(errors, body=body)
    assert pointer_for_error(exc, errors[0], model=IdentityFacetValueRequest) == "/request/currency"


class RoleMember(BaseModel):
    role: str


class _PatternStrOrModel(BaseModel):
    """Local stand-in for `IdentityUnitRequest.currency`'s shape -- a plain
    `pattern str | model` union -- confirming the union-type-tag fix
    generalises beyond Search's own model."""

    model_config = ConfigDict(extra="forbid")

    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] | RoleMember


def test_pointer_drops_union_type_tag_on_a_local_pattern_str_or_model_field():
    body = {"currency": "usd"}
    errors = _errors_for(_PatternStrOrModel, body)
    exc = RequestValidationError(errors, body=body)
    assert pointer_for_error(exc, errors[0], model=_PatternStrOrModel) == "/currency"


class _PatternStrOrModelWithCollidingField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] | RoleMember
    RoleMember: str


def test_pointer_keeps_a_real_member_literally_named_like_a_union_member_class():
    # Keep the missing top-level `RoleMember` field despite the matching union class name.
    body = {"currency": "USD"}
    errors = _errors_for(_PatternStrOrModelWithCollidingField, body)
    exc = RequestValidationError(errors, body=body)
    assert (
        pointer_for_error(exc, errors[0], model=_PatternStrOrModelWithCollidingField)
        == "/RoleMember"
    )


class _FacetBearingBody(BaseModel):
    """A body whose `facets` map is keyed by an open-ended axis name and whose
    values are `IdentityFacetBinding` -- the callable-`Discriminator` union
    (`Tag("ref")`/`Tag("set")`/`Tag("unit_request")`) whose tags no field-name
    guess-list can name."""

    model_config = ConfigDict(extra="forbid")

    facets: dict[str, IdentityFacetBinding] | None = None


def test_pointer_drops_callable_discriminator_tag_from_a_dict_value():
    # #666: The callable discriminator injects `ref` into `loc`; it is not a facet key.
    body = {"facets": {"provenance": {"state": "value", "ref": "ok", "bogus": 1}}}
    errors = _errors_for(_FacetBearingBody, body)
    exc = RequestValidationError(errors, body=body)
    assert errors[0]["loc"][-2:] == ("ref", "bogus")
    assert pointer_for_error(exc, errors[0], model=_FacetBearingBody) == "/facets/provenance/bogus"


def test_pointer_drops_callable_discriminator_tag_for_the_unit_request_arm():
    # `unit_request` and `constrained-str` are synthetic tags in this location.
    body = {
        "facets": {
            "provenance": {"state": "value", "request": {"currency": "usd", "prices": "current"}}
        }
    }
    errors = _errors_for(_FacetBearingBody, body)
    exc = RequestValidationError(errors, body=body)
    assert "unit_request" in errors[0]["loc"]
    assert pointer_for_error(exc, errors[0], model=_FacetBearingBody) == (
        "/facets/provenance/request/currency"
    )


def test_pointer_drops_callable_discriminator_tag_for_the_set_arm():
    # Drop the `set` arm tag while keeping the real `refs` field.
    body = {"facets": {"provenance": {"state": "set", "refs": []}}}
    errors = _errors_for(_FacetBearingBody, body)
    exc = RequestValidationError(errors, body=body)
    assert errors[0]["loc"][-2:] == ("set", "refs")
    assert pointer_for_error(exc, errors[0], model=_FacetBearingBody) == "/facets/provenance/refs"


def test_pointer_keeps_a_real_facet_key_named_like_an_arm_tag():
    # Here `ref` is a real facet key, not a discriminator tag.
    body = {"facets": {"ref": {"state": "value", "ref": "ok"}}, "bogus": 1}
    errors = _errors_for(_FacetBearingBody, body)
    exc = RequestValidationError(errors, body=body)
    assert errors[0]["loc"] == ("body", "bogus")
    assert pointer_for_error(exc, errors[0], model=_FacetBearingBody) == "/bogus"


def test_pointer_without_a_model_falls_back_to_the_body_walk_alone():
    # Without union metadata, the body walk follows `ref` as a real key.
    body = {"facets": {"provenance": {"state": "value", "ref": "ok", "bogus": 1}}}
    errors = _errors_for(_FacetBearingBody, body)
    exc = RequestValidationError(errors, body=body)
    assert pointer_for_error(exc, errors[0]) == "/facets/provenance/ref"


def test_pointer_reads_the_model_from_request_body_model_when_none_is_passed():
    # The same body, with the model bound through the contextvar the way
    # `dfo_platform.app`'s dispatch binds it, resolves identically to the
    # explicit `model=` call.
    body = {"facets": {"provenance": {"state": "value", "ref": "ok", "bogus": 1}}}
    errors = _errors_for(_FacetBearingBody, body)
    exc = RequestValidationError(errors, body=body)
    token = set_request_body_model(_FacetBearingBody)
    try:
        assert pointer_for_error(exc, errors[0]) == "/facets/provenance/bogus"
    finally:
        reset_request_body_model(token)
    assert pointer_for_error(exc, errors[0]) == "/facets/provenance/ref"


def test_validation_problem_reads_the_model_from_request_body_model():
    body = {"facets": {"provenance": {"state": "value", "ref": "ok", "bogus": 1}}}
    errors = _errors_for(_FacetBearingBody, body)
    exc = RequestValidationError(errors, body=body)
    token = set_request_body_model(_FacetBearingBody)
    try:
        spec = validation_problem(exc)
    finally:
        reset_request_body_model(token)
    assert spec.pointer == "/facets/provenance/bogus"


def _known_colour(value: str) -> str:
    if value in ("red", "blue"):
        return value
    raise refusal_with_alternatives(
        "colour_unknown", f"unknown colour {value!r}", alternatives=["red", "blue"], axis=value
    )


class _ColourBody(BaseModel):
    colours: dict[Annotated[str, AfterValidator(_known_colour)], int]


def test_a_refusal_with_alternatives_maps_to_one_issue_at_the_key():
    # The posted key names a context key, which must reach the detail unsubstituted.
    body = {"colours": {"{alternatives}": 1}}
    exc = RequestValidationError(_errors_for(_ColourBody, body), body=body)

    spec = validation_problem(exc, model=_ColourBody)

    assert (spec.code, spec.pointer) == ("request_invalid", "/colours/{alternatives}")
    assert spec.detail == "unknown colour '{alternatives}'"
    (issue,) = spec.issues
    assert (issue.pointer, issue.axis) == ("/colours/{alternatives}", "{alternatives}")
    assert [a.ref for a in issue.alternatives] == ["red", "blue"]
    assert issue.truncated is False


def test_pointer_is_none_for_non_body_location():
    exc = RequestValidationError([], body={})
    error = {"loc": ("query", "limit"), "type": "int_parsing"}
    assert pointer_for_error(exc, error) is None


def test_pointer_is_none_for_malformed_json_body():
    # Malformed JSON leaves raw text in `exc.body`, with no object to walk.
    exc = RequestValidationError([], body="{not valid json")
    error = {"loc": ("body",), "type": "json_invalid"}
    assert pointer_for_error(exc, error) is None


def test_validation_problem_defaults_to_request_invalid_400():
    body = {"input": {"kind": "raw_text"}}
    errors = [{"loc": ("body", "input", "query"), "type": "missing", "msg": "Field required"}]
    exc = RequestValidationError(errors, body=body)
    spec = validation_problem(exc)
    assert spec.status == 400
    assert spec.code == "request_invalid"


def test_validation_problem_maps_schema_version_to_422():
    body = {"schema_version": 2}
    errors = [
        {
            "loc": ("body", "schema_version"),
            "type": "literal_error",
            "msg": "Input should be 1",
        }
    ]
    exc = RequestValidationError(errors, body=body)
    spec = validation_problem(exc)
    assert spec.status == 422
    assert spec.code == "unsupported_schema_version"


def test_validation_problem_detail_carries_extra_field_message():
    body = {"query": "Kenya", "extra_field": "nope"}
    errors = [
        {
            "loc": ("body", "extra_field"),
            "type": "extra_forbidden",
            "msg": "Extra inputs are not permitted",
        }
    ]
    exc = RequestValidationError(errors, body=body)
    spec = validation_problem(exc)
    assert spec.code == "request_invalid"
    assert spec.detail == "Extra inputs are not permitted"


@pytest.mark.parametrize(
    ("key_chars", "kept"),
    [(MAX_VALUE_CHARS, True), (MAX_VALUE_CHARS + 1, False)],
    ids=["at-the-bound", "past-the-bound"],
)
def test_pointer_ends_at_the_object_holding_an_oversized_key(key_chars: int, kept: bool):
    """A key within the bound is a pointer segment as posted. A longer one, and every
    segment under it, is left out, so the pointer names the object that holds the key."""
    key = "k" * key_chars
    body = {"input": {"entries": [{key: {"nested": 1}}]}}
    error = {"loc": ("body", "input", "entries", 0, key, "nested"), "type": "int_type"}

    pointer = pointer_for_error(RequestValidationError([error], body=body), error)

    assert pointer == (f"/input/entries/0/{key}/nested" if kept else "/input/entries/0")


def test_pointer_is_none_for_an_oversized_top_level_key():
    key = "k" * (MAX_VALUE_CHARS + 1)
    error = {"loc": ("body", key), "type": "extra_forbidden"}

    assert pointer_for_error(RequestValidationError([error], body={key: 1}), error) is None


@pytest.mark.parametrize(
    ("tag_chars", "cut_mark"),
    [(8, ""), (MAX_VALUE_CHARS, ""), (4000, "...")],
    ids=["short", "at-the-bound", "long"],
)
def test_validation_problem_detail_cuts_a_posted_union_tag(tag_chars: int, cut_mark: str):
    """An unknown discriminator tag is quoted whole up to the bound and cut past it."""
    tag = "k" * tag_chars
    body = {"input": {"kind": tag}}
    errors = _errors_for(ResolveRequest, body)

    spec = validation_problem(RequestValidationError(errors, body=body), model=ResolveRequest)

    assert errors[0]["type"] == "union_tag_invalid"
    assert spec.detail == errors[0]["msg"].replace(tag, tag[:MAX_VALUE_CHARS] + cut_mark)


@pytest.mark.parametrize(
    ("tag", "json_type"),
    [
        ({"a": ["b"]}, "object"),
        (["a", "b"], "array"),
        (3, "number"),
        (1.5, "number"),
        (True, "boolean"),
        (None, "null"),
    ],
    ids=["object", "array", "int", "float", "bool", "null"],
)
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda tag: {"input": {"kind": tag}}, id="input-kind"),
        pytest.param(
            lambda tag: {"input": {"kind": "dataset_resubmit", "entries": [{"entry_kind": tag}]}},
            id="entry-kind",
        ),
    ],
)
def test_validation_problem_detail_names_a_non_string_union_tag_by_its_json_type(
    build, tag: object, json_type: str
):
    """A discriminator posted as a list, object, number, boolean or null is named by its
    JSON type, and the detail carries no repr of it."""
    body = build(tag)
    errors = _errors_for(ResolveRequest, body)

    spec = validation_problem(RequestValidationError(errors, body=body), model=ResolveRequest)

    assert errors[0]["type"] == "union_tag_invalid"
    assert json_type in spec.detail
    assert not {"{", "}", "[", "]"} & set(spec.detail)


def test_validation_problem_detail_carries_missing_field_message():
    body = {"input": {"kind": "raw_text"}}
    errors = [{"loc": ("body", "input", "query"), "type": "missing", "msg": "Field required"}]
    exc = RequestValidationError(errors, body=body)
    spec = validation_problem(exc)
    assert spec.code == "request_invalid"
    assert spec.detail == "Field required"


def test_validation_problem_detail_carries_json_decode_message_with_no_pointer():
    errors = [
        {
            "loc": ("body",),
            "type": "json_invalid",
            "msg": "JSON decode error",
        }
    ]
    exc = RequestValidationError(errors, body="{not valid json")
    spec = validation_problem(exc)
    assert spec.code == "request_invalid"
    assert spec.detail == "JSON decode error"
    assert spec.pointer is None


def test_validation_problem_schema_version_detail_stays_none():
    body = {"schema_version": 2}
    errors = [
        {
            "loc": ("body", "schema_version"),
            "type": "literal_error",
            "msg": "Input should be '1'",
        }
    ]
    exc = RequestValidationError(errors, body=body)
    spec = validation_problem(exc)
    assert spec.code == "unsupported_schema_version"
    assert spec.detail is None


def _surface(router: APIRouter) -> Surface:
    return Surface(
        name="probe",
        prefix="/api/probe",
        router=router,
        capability_version="v1",
        codes={},
    )


def _client(router: APIRouter) -> TestClient:
    return TestClient(create_app([_surface(router)]), raise_server_exceptions=False)


def test_embedded_second_body_param_pointer_uses_that_params_own_model():
    # #666: Use the failing parameter's annotation; the first parameter lacks this `bogus` key.
    router = APIRouter()

    @router.post("/api/probe/pair")
    async def pair(first: _FacetBearingBody = Body(...), second: _FacetBearingBody = Body(...)):
        return {"ok": True}

    resp = _client(router).post(
        "/api/probe/pair",
        json={
            "first": {"facets": {}},
            "second": {"facets": {"provenance": {"state": "value", "ref": "ok", "bogus": 1}}},
        },
    )
    assert resp.status_code == 400
    assert resp.json()["pointer"] == "/second/facets/provenance/bogus"


def test_single_non_embedded_body_pointer_still_names_the_real_path():
    """A non-embedded body has no parameter name in `loc` and resolves via `BodyParams.fallback`."""
    router = APIRouter()

    @router.post("/api/probe/one")
    async def one(payload: _FacetBearingBody = Body(...)):
        return {"ok": True}

    resp = _client(router).post(
        "/api/probe/one",
        json={"facets": {"provenance": {"state": "value", "ref": "ok", "bogus": 1}}},
    )
    assert resp.status_code == 400
    assert resp.json()["pointer"] == "/facets/provenance/bogus"


class _ArmA(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["a"] = "a"
    a: str
    scope: int


class _ArmB(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["b"] = "b"


class _SelfNamedFieldBody(BaseModel):
    """A model with a top-level field named "x", for a route whose single
    non-embedded body parameter is also named "x"."""

    model_config = ConfigDict(extra="forbid")

    x: Annotated[_ArmA | _ArmB, Field(discriminator="kind")]


def test_single_non_embedded_body_param_named_like_its_own_field_still_resolves():
    # The parameter and field share `x`; only `BodyParams.fallback` reaches the union correctly.
    router = APIRouter()

    @router.post("/api/probe/self_named")
    async def one(x: _SelfNamedFieldBody = Body(...)):
        return {"ok": True}

    resp = _client(router).post(
        "/api/probe/self_named",
        json={"x": {"kind": "a", "a": "sibling-value", "scope": "not-an-int"}},
    )
    assert resp.status_code == 400
    assert resp.json()["pointer"] == "/x/scope"


def test_list_body_pointer_walks_the_element_model():
    # The pointer includes the JSON array's element index.
    router = APIRouter()

    @router.post("/api/probe/items")
    async def items(payload: list[_FacetBearingBody] = Body(...)):
        return {"ok": True}

    resp = _client(router).post(
        "/api/probe/items",
        json=[{"facets": {"provenance": {"state": "value", "ref": "ok", "bogus": 1}}}],
    )
    assert resp.status_code == 400
    assert resp.json()["pointer"] == "/0/facets/provenance/bogus"


def test_embedded_body_with_no_walkable_annotation_warns_once_and_still_400s(caplog):
    # An `Any` annotation logs once and still returns a 400 with a body-walk pointer.
    router = APIRouter()

    @router.post("/api/probe/opaque")
    async def opaque(payload: Any = Body(..., embed=True)):
        return {"ok": True}

    client = _client(router)
    with caplog.at_level(logging.WARNING, logger="dfo_platform.app"):
        first = client.post("/api/probe/opaque", json={})
        second = client.post("/api/probe/opaque", json={})

    assert first.status_code == 400
    assert first.json()["code"] == "request_invalid"
    assert first.json()["pointer"] == "/payload"
    assert second.status_code == 400
    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and "/api/probe/opaque" in record.getMessage()
    ]
    assert len(warnings) == 1
