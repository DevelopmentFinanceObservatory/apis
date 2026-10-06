"""Tests request limits for `statvar_set` and the member walk's published ceiling.
Schema and capability limits must match the enforced model and settings constraints.
"""

from __future__ import annotations

import httpx
import pytest

from code_index.shared import get_index
from datasets_api.capabilities import build_capabilities
from datasets_api.config import DatasetsBuildSettings
from datasets_api.mint_models import StatVarSetInput
from datasets_api.revision import SentinelRevisionReader
from datasets_api.schema import build_schema_document
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader

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


class TestStatVarSetNamesOneStatVar:
    def test_published_json_schema_max_items_matches_the_model(self):
        document = build_schema_document()
        statvars = document["mint_request"]["$defs"]["StatVarSetInput"]["properties"]["statvars"]
        assert statvars["maxItems"] == _field_max_length(StatVarSetInput, "statvars") == 1

    @pytest.mark.parametrize(
        "mint_input",
        [
            {"kind": "statvar_set", "statvars": ["dcid:StatVarA", "dcid:StatVarB"]},
            {"kind": "source_codes", "codes": ["CODE1"]},
            {"kind": "collection_id", "collection_id": "col_" + "0" * 32},
        ],
        ids=["several-statvars", "source-codes", "collection-id"],
    )
    def test_a_multi_statvar_or_retired_kind_refuses_request_invalid(self, mint_input):
        resp = _client().post("/api/datasets", json={"schema_version": 1, "input": mint_input})

        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"


class TestPublishedSubjectCeilingMatchesSettings:
    def test_capability_document_states_the_member_walk_ceiling(self):
        settings = DatasetsBuildSettings(member_subject_ceiling=12_345)
        caps = build_capabilities(
            capability_version="dset_test",
            build="test-build",
            code_index_stamp=get_index().stamp,
            declarations=get_index().declarations,
            settings=settings,
        ).caps
        assert caps.definition_subject_ceiling == settings.member_subject_ceiling
