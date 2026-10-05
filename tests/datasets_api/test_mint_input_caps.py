"""Tests request limits for collection members and resolved StatVar seeds.
Schema and capability limits must match the enforced model constraints.
"""

from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from code_index.shared import get_index
from datasets_api.capabilities import build_capabilities
from datasets_api.config import DatasetsBuildSettings
from datasets_api.mint_models import (
    DefinitionOptionsRequest,
    SourceCodesInput,
    SpecIdInput,
    StatVarSetInput,
)
from datasets_api.models import MAX_COLLECTION_MEMBERS
from datasets_api.revision import SentinelRevisionReader
from datasets_api.schema import build_schema_document
from search.models import _RESUBMIT_MAX_RESOLVED_STAT_VARS
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from statvar_members import DEFAULT_MEMBER_CAP, POSTED_MEMBER_CAP

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter


def _field_max_length(model: type, field_name: str) -> int | None:
    for constraint in model.model_fields[field_name].metadata:
        max_length = getattr(constraint, "max_length", None)
        if max_length is not None:
            return max_length
    return None


def _refusing_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"SDMX must not be called for this request: {request.url}")

    return httpx.MockTransport(handler)


def _client(*, graph_arcs: dict | None = None) -> object:
    from dc_sdmx import SdmxClient

    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=FakeMintGraphReader(arcs=graph_arcs or {}),
        sdmx=SdmxClient(
            base_url="https://datacommons.one.org/core/api", transport=_refusing_transport()
        ),
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
    )


class TestPublishedCapMatchesEnforcedConstraint:
    def test_capability_document_reads_the_models_actual_constraint(self):
        statvars_cap = _field_max_length(StatVarSetInput, "statvars")
        codes_cap = _field_max_length(SourceCodesInput, "codes")
        assert statvars_cap == codes_cap == MAX_COLLECTION_MEMBERS

        caps = build_capabilities(
            capability_version="dset_test",
            build="test-build",
            code_index_stamp=get_index().stamp,
            declarations=get_index().declarations,
            settings=DatasetsBuildSettings(),
        ).caps
        assert caps.max_collection_members == statvars_cap

    def test_published_json_schema_max_items_matches_the_same_constraint(self):
        statvars_cap = _field_max_length(StatVarSetInput, "statvars")
        codes_cap = _field_max_length(SourceCodesInput, "codes")

        document = build_schema_document()
        defs = document["mint_request"]["$defs"]
        assert defs["StatVarSetInput"]["properties"]["statvars"]["maxItems"] == statvars_cap
        assert defs["SourceCodesInput"]["properties"]["codes"]["maxItems"] == codes_cap


class TestOverCapRequestsRefuseBeforeDispatch:
    def test_statvar_set_over_cap_refuses_request_invalid(self):
        client = _client()
        body = {
            "schema_version": 1,
            "input": {
                "kind": "statvar_set",
                "statvars": [f"dcid:StatVar{i}" for i in range(MAX_COLLECTION_MEMBERS + 1)],
            },
        }

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"

    def test_source_codes_over_cap_refuses_request_invalid(self):
        client = _client()
        body = {
            "schema_version": 1,
            "input": {
                "kind": "source_codes",
                "codes": [f"CODE{i}" for i in range(MAX_COLLECTION_MEMBERS + 1)],
            },
        }

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"


class TestAtCapRequestsClearTheLengthCheck:
    """At exactly `MAX_COLLECTION_MEMBERS`, the length check passes.
    The empty-arcs graph leaves every member unresolvable, a resolution-stage refusal.
    """

    def test_statvar_set_at_cap_is_not_refused_for_length(self):
        client = _client(graph_arcs={})
        body = {
            "schema_version": 1,
            "input": {
                "kind": "statvar_set",
                "statvars": [f"dcid:StatVar{i}" for i in range(MAX_COLLECTION_MEMBERS)],
            },
        }

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code != 400 or resp.json().get("code") != "request_invalid"
        assert resp.status_code == 422, resp.text
        assert resp.json()["code"] == "member_not_resolvable"

    def test_source_codes_at_cap_is_not_refused_for_length(self):
        client = _client(graph_arcs={})
        body = {
            "schema_version": 1,
            "input": {
                "kind": "source_codes",
                "codes": [f"CODE{i}" for i in range(MAX_COLLECTION_MEMBERS)],
            },
        }

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code != 400 or resp.json().get("code") != "request_invalid"
        assert resp.status_code == 404, resp.text
        assert resp.json()["code"] == "code_not_found"


class TestPostedMemberCeilingMatchesSearch:
    @pytest.mark.parametrize("model", [SpecIdInput, DefinitionOptionsRequest])
    def test_field_constraint_equals_the_search_resubmit_ceiling(self, model):
        """Search sends up to this many members; the mint and options inputs take them whole,
        above the enumeration cap."""
        cap = _field_max_length(model, "resolved_stat_var_dcids")
        assert cap == POSTED_MEMBER_CAP == _RESUBMIT_MAX_RESOLVED_STAT_VARS
        assert cap > DEFAULT_MEMBER_CAP

    def test_a_1626_member_list_validates_and_one_over_the_ceiling_refuses(self):
        """ODA from Germany resolves 1,626 members."""
        dcids = [f"StatVar{i}" for i in range(1_626)]
        seed = SpecIdInput.model_validate(
            {"kind": "spec_id", "spec_id": "spec_x", "resolved_stat_var_dcids": dcids}
        )
        assert seed.resolved_stat_var_dcids == dcids

        with pytest.raises(ValidationError) as excinfo:
            SpecIdInput.model_validate(
                {
                    "kind": "spec_id",
                    "spec_id": "spec_x",
                    "resolved_stat_var_dcids": ["StatVar"] * (POSTED_MEMBER_CAP + 1),
                }
            )
        assert [error["type"] for error in excinfo.value.errors()] == ["too_long"]

    def test_schema_document_max_items_matches_the_same_constraint(self):
        seed_cap = _field_max_length(SpecIdInput, "resolved_stat_var_dcids")

        document = build_schema_document()
        defs = document["mint_request"]["$defs"]
        # The optional seed publishes `maxItems` on the array branch of its `anyOf` schema.
        property_schema = defs["SpecIdInput"]["properties"]["resolved_stat_var_dcids"]
        array_branch = next(
            branch for branch in property_schema["anyOf"] if branch["type"] == "array"
        )
        assert array_branch["maxItems"] == seed_cap


class TestSpecIdSeedOverCapRefusesRequestInvalid:
    def test_resolved_stat_var_dcids_over_cap_refuses_request_invalid(self):
        client = _client()
        body = {
            "schema_version": 1,
            "input": {
                "kind": "spec_id",
                "spec_id": "spec_unregistered",
                "resolved_stat_var_dcids": [f"StatVar{i}" for i in range(POSTED_MEMBER_CAP + 1)],
            },
        }

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"


class TestSpecIdSeedAtCapClearsTheLengthCheck:
    def test_resolved_stat_var_dcids_at_cap_is_not_refused_for_length(self):
        """At the cap, validation passes and an unregistered id returns `spec_not_found`."""
        client = _client()
        body = {
            "schema_version": 1,
            "input": {
                "kind": "spec_id",
                "spec_id": "spec_unregistered",
                "resolved_stat_var_dcids": [f"StatVar{i}" for i in range(POSTED_MEMBER_CAP)],
            },
        }

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code != 400 or resp.json().get("code") != "request_invalid"
        assert resp.status_code == 404, resp.text
        assert resp.json()["code"] == "spec_not_found"
