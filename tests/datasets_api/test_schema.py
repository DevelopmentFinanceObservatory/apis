"""The published mint schema and runtime `MintRequest` agree on field types, nullability,
required fields, and the closed `scope` shape.
"""

from __future__ import annotations

import pytest

from datasets_api.mint_models import MintRequest as RuntimeMintRequest
from datasets_api.schema import MintRequest as PublishedMintRequest
from datasets_api.schema import build_schema_document
from dfo_platform.named import MAX_VALUE_CHARS
from spec_registry import LiveSpecRegistry

from ._app_factories import make_client
from .conftest import FakeDatasetStore

_STATVAR_INPUT = {"kind": "statvar_set", "statvars": ["dcid:X"]}


class TestOptionsNullRejectedByBothLayers:
    def test_runtime_model_rejects_an_explicit_null(self):
        try:
            RuntimeMintRequest.model_validate(
                {"schema_version": 1, "input": _STATVAR_INPUT, "options": None}
            )
        except Exception as exc:  # pydantic.ValidationError
            assert "options" in str(exc)
        else:
            raise AssertionError("explicit 'options': null must be rejected")

    def test_published_schema_does_not_admit_null_for_options(self):
        options_schema = PublishedMintRequest.model_json_schema()["properties"]["options"]

        assert options_schema.get("type") != "null"
        assert options_schema.get("anyOf") is None
        assert "null" not in str(options_schema.get("type", ""))

    def test_the_door_refuses_an_explicit_null_as_request_invalid(self):
        client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))
        body = {"schema_version": 1, "input": _STATVAR_INPUT, "options": None}

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"


class TestScopeIsClosedInThePublishedSchema:
    """Published `Scope` uses the runtime's closed model, so both layers accept the same shape."""

    def test_published_scope_def_is_closed_with_exactly_the_three_members(self):
        schema = PublishedMintRequest.model_json_schema()
        scope_def = schema["$defs"]["Scope"]

        assert scope_def["additionalProperties"] is False
        assert set(scope_def["properties"]) == {"window", "entities", "facets"}

    def test_every_input_kind_references_the_shared_scope_def(self):
        schema = PublishedMintRequest.model_json_schema()
        defs = schema["$defs"]

        for input_kind in ("SpecIdInput", "StatVarSetInput"):
            scope_schema = defs[input_kind]["properties"]["scope"]
            ref_variant = next(v for v in scope_schema["anyOf"] if "$ref" in v)
            assert ref_variant["$ref"] == "#/$defs/Scope"


class TestStatvarsCarryTheDcidPrefixPatternInThePublishedSchema:
    """`statvar_set.statvars` is `dcid:`-prefix constrained on the wire, not
    only by a runtime validator: the published and runtime `$defs` agree on
    the item pattern.
    """

    def test_published_statvars_items_declare_the_dcid_prefix_pattern(self):
        schema = PublishedMintRequest.model_json_schema()
        items_schema = schema["$defs"]["StatVarSetInput"]["properties"]["statvars"]["items"]

        assert items_schema["pattern"] == "^dcid:"


class TestColumnsUniquenessReflectedInSchema:
    """Runtime validation and the generated schema both enforce unique `columns` entries."""

    def test_published_and_runtime_columns_schema_both_declare_unique_items(self):
        for schema_cls in (PublishedMintRequest, RuntimeMintRequest):
            columns_schema = schema_cls.model_json_schema()["$defs"]["MintOptions"]["properties"][
                "columns"
            ]
            assert columns_schema.get("uniqueItems") is True

    def test_the_runtime_rejection_still_works(self):
        try:
            RuntimeMintRequest.model_validate(
                {
                    "schema_version": 1,
                    "input": _STATVAR_INPUT,
                    "options": {"columns": ["a", "a"]},
                }
            )
        except Exception as exc:  # pydantic.ValidationError
            assert "duplicate" in str(exc)
        else:
            raise AssertionError("a duplicate columns entry must be rejected")


class TestOptionsOmittedIsValidInBothLayers:
    def test_runtime_model_applies_defaults_when_options_is_omitted(self):
        request = RuntimeMintRequest.model_validate({"schema_version": 1, "input": _STATVAR_INPUT})

        assert request.options.format == "csv"
        assert request.options.bundle is True

    def test_published_schema_does_not_require_options(self):
        schema = PublishedMintRequest.model_json_schema()

        assert "options" not in schema.get("required", [])


class TestMintPreviewIsRetired:
    def test_neither_layer_declares_a_preview_field(self):
        assert "preview" not in RuntimeMintRequest.model_fields
        assert "preview" not in PublishedMintRequest.model_json_schema()["properties"]
        assert "preview_envelope" not in build_schema_document()

    def test_the_door_refuses_a_posted_preview_as_request_invalid(self):
        client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))
        body = {"schema_version": 1, "input": _STATVAR_INPUT, "preview": {"limit": 10}}

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"


_LONG = "a" * 4000


def _scoped(**scope) -> dict:
    return {"input": {**_STATVAR_INPUT, "scope": scope}}


def _optioned(**options) -> dict:
    return {"input": _STATVAR_INPUT, "options": options}


def _sidecar(spec_id: str) -> dict:
    return {"schema_version": 1, "spec_id": spec_id, "identity": {}}


_DASHES = "-" * 4000
_SPEC_ID = f"spec_{_LONG}"
_CALLER_STRING_REFUSALS: dict[str, tuple[dict, str]] = {
    # body, the string it posts
    "invalid-entity-role": (_scoped(entities={_DASHES: ["country/KEN"]}), _DASHES),
    "entity-role-without-values": (_scoped(entities={_LONG: []}), _LONG),
    "entity-role-with-an-empty-value": (_scoped(entities={_LONG: [""]}), _LONG),
    "entity-role-over-a-reserved-value": (_scoped(entities={_LONG: ["a+b"]}), _LONG),
    "reserved-value": (_scoped(entities={"recipient": [f"{_LONG}+"]}), _LONG),
    "unknown-facet-axis": (_scoped(facets={_LONG: {"ref": "x"}}), _LONG),
    "duplicate-groupby-axis": (_optioned(groupby={"op": "sum", "by": [_LONG, _LONG]}), _LONG),
    "groupby-level-outside-by": (
        _optioned(groupby={"op": "sum", "by": ["recipient"], "levels": {_LONG: 1}}),
        _LONG,
    ),
    "duplicate-column": (_optioned(columns=[_LONG, _LONG]), _LONG),
    "sidecar-spec-id-differs": (
        {
            "input": {
                "kind": "spec_id",
                "spec_id": _SPEC_ID,
                "definition": _sidecar(f"spec_x{_LONG}"),
            }
        },
        _SPEC_ID,
    ),
}


@pytest.mark.parametrize(
    ("body", "posted"), _CALLER_STRING_REFUSALS.values(), ids=_CALLER_STRING_REFUSALS.keys()
)
def test_a_refusal_names_a_posted_string_by_its_first_characters(body: dict, posted: str):
    """A 4,000-character role, axis, ref or spec id comes back as its first 256 characters,
    and a list of names comes back without a list repr."""
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))

    resp = client.post("/api/datasets", json={"schema_version": 1, **body})

    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "request_invalid"
    assert posted[:MAX_VALUE_CHARS] in resp.json()["detail"]
    assert posted[: MAX_VALUE_CHARS + 1] not in resp.text
    assert "[" not in resp.json()["detail"]


# The fields the doors leave out of a body when `None` (`exclude_if=is_none`), by schema and
# def. Every other field of a response schema is required.
_OMITTED_WHEN_NONE = {
    ("manifest", "Manifest"): {
        "series_count",
        "output_options",
        "series_coverage_stamp",
    },
    ("options_envelope", "OptionsEnvelope"): {"output_options"},
    ("operation_resource", "OperationEnvelope"): {"poll", "retry_after_ms", "progress"},
    ("operation_resource", "OperationResource"): {"manifest", "error"},
}
_RESPONSE_SCHEMAS = (
    "register_response",
    "manifest",
    "options_envelope",
    "members_envelope",
    "pin_response",
    "operation_resource",
)


def _schema_nodes(schema: dict) -> dict[str, dict]:
    return {schema["title"]: schema, **schema.get("$defs", {})}


def test_a_response_schema_requires_every_field_the_server_emits():
    document = build_schema_document()

    for name in _RESPONSE_SCHEMAS:
        for title, node in _schema_nodes(document[name]).items():
            optional = set(node.get("properties", {})) - set(node.get("required", []))
            assert optional == _OMITTED_WHEN_NONE.get((name, title), set()), (name, title)


def test_a_request_schema_keeps_a_defaulted_field_optional():
    document = build_schema_document()

    assert document["mint_request"] == PublishedMintRequest.model_json_schema()
    assert document["mint_request"]["required"] == ["input"]
    assert document["pin_request"]["required"] == ["artifact_digest"]
