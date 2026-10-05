"""The published mint schema and runtime `MintRequest` agree on field types, nullability,
required fields, and the closed `scope` shape. Tests also cover preview version fields.
"""

from __future__ import annotations

from datasets_api.mint_models import MintRequest as RuntimeMintRequest
from datasets_api.preview_models import PreviewMemberModel
from datasets_api.schema import MintRequest as PublishedMintRequest
from datasets_api.schema import PreviewEnvelope as PreviewEnvelopeSchema
from datasets_api.schema import build_schema_document
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

        for input_kind in (
            "SpecIdInput",
            "CollectionIdInput",
            "StatVarSetInput",
            "SourceCodesInput",
        ):
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


class TestPreviewEnvelopeCarriesTheMintVersion:
    """`datasets_api.schema.PreviewEnvelope` publishes `content_version` on
    each member, the same field `datasets_api.preview_models.
    PreviewMemberModel` serialises: a client validating a preview response
    against the published schema can read the version the preview reports.
    """

    def test_published_preview_envelope_member_declares_content_version(self):
        schema = PreviewEnvelopeSchema.model_json_schema()
        member_schema = schema["$defs"]["PreviewMember"]

        assert "content_version" in member_schema["properties"]
        assert "content_version" in member_schema["required"]

    def test_the_published_schema_route_serves_that_member_schema(self):
        served = build_schema_document()["preview_envelope"]

        assert "content_version" in served["$defs"]["PreviewMember"]["properties"]

    def test_the_runtime_preview_member_model_declares_the_same_field(self):
        # Published and runtime models share the same member fields.
        assert "content_version" in PreviewMemberModel.model_fields

    def test_the_runtime_preview_member_holds_a_null_version(self):
        # Search returns `content_version=None`, which the runtime model accepts.
        member = PreviewMemberModel(
            stat_var="dcid:X",
            content_version=None,
            coverage={"rows": 0, "dimensions": [], "window": None},
            citation={"text": "t", "sources": []},
            sample={"columns": [], "rows": []},
            total_rows=0,
            truncated=False,
        )

        assert member.content_version is None
