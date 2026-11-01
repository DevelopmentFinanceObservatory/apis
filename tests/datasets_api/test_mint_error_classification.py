"""Tests mint error classification for infrastructure failures, request refusals, and
conversion errors.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging

import httpx
import pyarrow.parquet as pq
import pytest
from dfo_frame_grammar import FrameError

from artifact_store import (
    ARTIFACT_INVENTORY_WRITE_FAILED_MARKER,
    ArtifactInventoryError,
    ArtifactRecordConflictError,
    ArtifactSigningError,
    ArtifactWriteError,
    Tier,
)
from datasets_api.codes_materialise import (
    InvalidRoleRef,
    RoleIncompatible,
    RoleSignatureIncompatible,
)
from datasets_api.doors.mint_errors import classify_mint_exception
from dc_sdmx import (
    SdmxClient,
    SdmxContainmentFilterError,
    SdmxHttpError,
    SdmxIncompatibleVariablesError,
    SdmxMissingFilterError,
    SdmxQueryError,
    SdmxTransportError,
    SdmxUnsupportedFilterError,
)
from materialise.errors import (
    DatasetDisagreementError,
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    MembersNotHeldError,
    UpstreamResponseError,
)
from materialise.formats import ExcelLimitError, ParquetSizeBoundError
from materialise.graph import Facet, GraphDeadlineExceededError, GraphReadError
from materialise.group_facets import _as_issue
from materialise.reshape import (
    ColumnNotDroppableError,
    PivotNotHomogeneousError,
    PivotNotUniqueError,
)
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._identity_builders import pinned_identity, register_body
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore, wire_definition_for
from .test_pin_door import FakeArtifactReferenceWriter
from .test_resolve_definition_members_guards import (
    _ODA_GRANTS,
    _ODA_LOANS,
    _SCHEME_PROPERTY,
    _InconsistentReadGraphReader,
    _member_arcs,
    _two_member_identity,
)

_STAT_VAR = "MalariaGrantDisbursements"
_STAT_VAR_REF = f"dcid:{_STAT_VAR}"
_BASE_URL = "https://datacommons.one.org/core/api"
_DAC_RECIPE = "ConversionRecipeDAC"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_CSV_ROW = (
    f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,USD,GrantEquivalent,"
    "NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2015",
    latest_date="2015",
)

_SHAPE_ARCS = {
    _STAT_VAR: {
        "typeOf": ("StatisticalVariable",),
        "observationProperties": ("recipient",),
        "measuredProperty": ("amount",),
    },
}

_SCOPE = {
    "entities": {"recipient": ["country/KEN"]},
    "facets": {
        "provenance": {"ref": "OECD_DAC2A"},
        "measurementMethod": {"ref": "GrantEquivalent"},
        "unit": {"ref": "USD"},
    },
}


def _default_graph(*, extra_arcs: dict | None = None) -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={**_SHAPE_ARCS, **(extra_arcs or {})},
        facets={(_STAT_VAR, "country/KEN"): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a", "dcid:a": "A", "dcid:b": "B", "dcid:c": "C"},
    )


def _sdmx_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_CSV_HEADER + _CSV_ROW,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


def _coverage_service(
    *, donor_role: str | None = None, donor: str | None = None, unit: str = "USD"
) -> FakeCoverageService:
    """Coverage matching `_FACET` and the 2015 SDMX row.
    Donor arguments support the role-cap test.
    """
    coverage_donor = "country/USA" if donor is None and donor_role is not None else donor
    coverage_donor = ABSENT_ENTITY if coverage_donor is None else coverage_donor
    index = build_coverage_index(
        [(_STAT_VAR, "facet1", coverage_donor, "country/KEN", [2015])],
        role_names={_STAT_VAR: RoleNames(donor=donor_role, recipient="recipient")},
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit=unit
            )
        },
    )
    return FakeCoverageService(index)


def _client(*, graph=None, artifact_store=None, store=None, sdmx=None, coverage=None) -> object:
    store = store if store is not None else FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph if graph is not None else _default_graph(),
        sdmx=sdmx
        if sdmx is not None
        else SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
        artifact_store=artifact_store if artifact_store is not None else FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=coverage if coverage is not None else _coverage_service(),
    )


def _spec_id_body(spec_id: str, *, definition: dict) -> dict:
    """Builds a `spec_id` mint request with a described-arm definition."""
    return {
        "schema_version": 1,
        "input": {"kind": "spec_id", "spec_id": spec_id, "definition": definition},
        "options": {"bundle": False},
    }


class TestInfraFailuresClassifyAsHonestAndRetryable:
    """Graph, SDMX, and transient artifact-store failures map to retryable responses."""

    def test_graph_read_failure_maps_to_upstream_error(self):
        exc = GraphReadError("graph transport error calling '/v2/node': connection reset")

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 502
        assert spec.code == "upstream_error"
        assert spec.retryable is True

    def test_graph_deadline_exceeded_maps_to_timeout_not_upstream_error(self):
        """The deadline subclass maps to 504 before the general graph-read 502 arm."""
        exc = GraphDeadlineExceededError("the graph read exceeded its absolute caller deadline")

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 504
        assert spec.code == "timeout"
        assert spec.retryable is True

    def test_sdmx_read_failure_maps_to_upstream_error(self):
        exc = SdmxTransportError("SDMX transport error calling '/sdmx/v3/data/...': timed out")

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 502
        assert spec.code == "upstream_error"
        assert spec.retryable is True

    def test_transient_artifact_write_failure_maps_to_service_unavailable(self):
        exc = ArtifactWriteError(object_name="sha256:deadbeef", digest="sha256:deadbeef")

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 503
        assert spec.code == "service_unavailable"
        assert spec.retryable is True

    def test_transient_artifact_inventory_failure_maps_to_service_unavailable(self):
        exc = ArtifactInventoryError(digest="sha256:deadbeef")

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 503
        assert spec.code == "service_unavailable"
        assert spec.retryable is True

    def test_base_artifact_signing_error_maps_to_service_unavailable(self):
        # An unwrapped signing error maps to a retryable outage.
        exc = ArtifactSigningError(
            bucket="dev-fin-observatory-datasets-working",
            object_name="sha256:deadbeef",
            digest="sha256:deadbeef",
        )

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 503
        assert spec.code == "service_unavailable"
        assert spec.retryable is True

    def test_upstream_response_inconsistency_maps_to_upstream_error(self):
        exc = UpstreamResponseError(
            "SDMX observation variable 'dcid:Other' does not match the requested stat_var"
        )

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 502
        assert spec.code == "upstream_error"
        assert spec.retryable is True


class TestSdmxQueryErrorIsRequestInvalid:
    """Client-side SDMX query-shape errors map to non-retryable 400 refusals."""

    def test_sdmx_query_error_maps_to_request_invalid(self):
        exc = SdmxQueryError("entity role 'recipient' has no values")

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 400
        assert spec.code == "request_invalid"
        assert spec.retryable is False
        assert spec.detail == "entity role 'recipient' has no values"


class TestRoleRefusalsAreNonRetryable422:
    """Scope role errors map to non-retryable 422 responses with one issue per role."""

    def test_invalid_role_ref_maps_to_a_422_with_a_pointer_at_the_role(self):
        exc = InvalidRoleRef(stat_var="dcid:MalariaGrantDisbursements", role="donor")

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 422
        assert spec.code == "invalid_role_ref"
        assert spec.retryable is False
        assert len(spec.issues) == 1
        assert spec.issues[0].pointer == "/input/scope/entities/donor"
        assert spec.issues[0].member == "donor"

    def test_role_incompatible_maps_to_a_422_with_a_pointer_at_the_role(self):
        exc = RoleIncompatible(
            stat_var="dcid:Health-ODAGrants-ETH",
            role="DevelopmentFinanceRecipient",
            pinned_value="country/ETH",
        )

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 422
        assert spec.code == "role_incompatible"
        assert spec.retryable is False
        assert len(spec.issues) == 1
        assert spec.issues[0].pointer == "/input/scope/entities/DevelopmentFinanceRecipient"
        assert spec.issues[0].member == "DevelopmentFinanceRecipient"


class TestUnknownFacetAxisListsTheAxes:
    """An unknown `scope.facets` key is refused at that key, listing the facet axes and leading
    with the nearest one when the posted name is close to it."""

    @pytest.mark.parametrize(
        ("axis", "alternatives"),
        [
            ("units", ["unit", "provenance", "measurementMethod"]),
            ("dacTypeOfFinance", ["provenance", "measurementMethod", "unit"]),
        ],
    )
    def test_refuses_request_invalid_with_the_axes_as_alternatives(self, axis, alternatives):
        client = _client()
        scope = {**_SCOPE, "facets": {axis: {"ref": "USD"}}}
        body = {
            "schema_version": 1,
            "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": scope},
        }

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code == 400, resp.text
        problem = resp.json()
        pointer = f"/input/scope/facets/{axis}"
        assert (problem["code"], problem["pointer"]) == ("request_invalid", pointer)
        (issue,) = problem["issues"]
        assert (issue["pointer"], issue["axis"]) == (pointer, axis)
        assert [a["ref"] for a in issue["alternatives"]] == alternatives
        assert issue["truncated"] is False


class TestRoleSignatureIncompatibleIsNonRetryable422:
    """Described-arm members with different role signatures map to non-retryable 422 responses."""

    def test_maps_to_a_422_with_one_issue_per_disagreeing_member(self):
        exc = RoleSignatureIncompatible(
            spec_id="spec_exemplar",
            roles_by_stat_var={
                "ODAGrantsAmount": ("recipient",),
                "ODALoansAmount": ("provider",),
            },
        )

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 422
        assert spec.code == "role_signature_incompatible"
        assert spec.retryable is False
        assert {issue.member for issue in spec.issues} == {"ODAGrantsAmount", "ODALoansAmount"}

    def test_role_disagreement_refuses_role_signature_incompatible_through_the_mint_door(self):
        """A described-arm enumeration with differing roles returns `role_signature_incompatible`."""
        store = FakeDatasetStore()
        graph = FakeMintGraphReader(
            arcs={
                _ODA_GRANTS: _member_arcs(scheme="ODAGrants", roles=("recipient",)),
                _ODA_LOANS: _member_arcs(scheme="ODALoans", roles=("provider",)),
            }
        )
        client = _client(graph=graph, store=store)
        identity = _two_member_identity()
        spec_id = recompute_identity(identity).spec_id
        definition = wire_definition_for(spec_id=spec_id, identity=identity)

        resp = client.post("/api/datasets", json=_spec_id_body(spec_id, definition=definition))

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "role_signature_incompatible"
        assert problem["retryable"] is False
        assert store.state.spec_registry == {}

    def test_role_disagreement_refuses_role_signature_incompatible_through_the_definition_door(
        self,
    ):
        graph = FakeMintGraphReader(
            arcs={
                _ODA_GRANTS: _member_arcs(scheme="ODAGrants", roles=("recipient",)),
                _ODA_LOANS: _member_arcs(scheme="ODALoans", roles=("provider",)),
            }
        )
        client = _client(graph=graph)
        identity = _two_member_identity()
        spec_id = recompute_identity(identity).spec_id
        reg = client.post(
            "/api/datasets/register",
            json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
        )
        assert reg.status_code == 200, reg.text

        resp = client.get(f"/api/datasets/{spec_id}")

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "role_signature_incompatible"
        assert problem["retryable"] is False


class TestInconsistentCandidatesRefuseMemberNotResolvableNotASilentNarrowing:
    """Candidates whose forward and reverse graph arcs disagree map to 422 responses. The
    reads are a live graph's, which a later read can settle, so the refusal is retryable."""

    def test_inconsistent_candidate_refuses_member_not_resolvable_through_the_mint_door(self):
        store = FakeDatasetStore()
        graph = _InconsistentReadGraphReader(
            FakeMintGraphReader(
                arcs={
                    _ODA_GRANTS: _member_arcs(scheme="ODAGrants"),
                    _ODA_LOANS: _member_arcs(scheme="ODALoans"),
                }
            ),
            stale_dcids=(_ODA_GRANTS,),
        )
        client = _client(graph=graph, store=store)
        identity = _two_member_identity()
        spec_id = recompute_identity(identity).spec_id
        definition = wire_definition_for(spec_id=spec_id, identity=identity)

        resp = client.post("/api/datasets", json=_spec_id_body(spec_id, definition=definition))

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "member_not_resolvable"
        assert problem["retryable"] is True
        assert [issue["pointer"] for issue in problem["issues"]] == ["/input/spec_id"]
        assert [issue["member"] for issue in problem["issues"]] == [f"dcid:{_ODA_GRANTS}"]
        assert store.state.spec_registry == {}

    def test_inconsistent_candidate_refuses_member_not_resolvable_through_the_definition_door(
        self,
    ):
        graph = _InconsistentReadGraphReader(
            FakeMintGraphReader(
                arcs={
                    _ODA_GRANTS: _member_arcs(scheme="ODAGrants"),
                    _ODA_LOANS: _member_arcs(scheme="ODALoans"),
                }
            ),
            stale_dcids=(_ODA_GRANTS,),
        )
        client = _client(graph=graph)
        identity = _two_member_identity()
        spec_id = recompute_identity(identity).spec_id
        reg = client.post(
            "/api/datasets/register",
            json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
        )
        assert reg.status_code == 200, reg.text

        resp = client.get(f"/api/datasets/{spec_id}")

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "member_not_resolvable"
        assert problem["retryable"] is True
        assert [issue["member"] for issue in problem["issues"]] == [f"dcid:{_ODA_GRANTS}"]


class TestInconsistentCandidatesOverTheNamingCapRefuseTruncated:
    """When more than 10 candidates disagree, both doors name 10 and add the truncation suffix."""

    def _truncated_identity_and_graph(self) -> tuple[dict, _InconsistentReadGraphReader]:
        schemes = [f"Scheme{i}" for i in range(12)]
        stat_vars = [f"Amount{i}" for i in range(12)]
        pairs = zip(stat_vars, schemes, strict=True)
        arcs = {stat_var: _member_arcs(scheme=scheme) for stat_var, scheme in pairs}
        graph = _InconsistentReadGraphReader(FakeMintGraphReader(arcs=arcs), stale_dcids=stat_vars)
        identity = _two_member_identity()
        identity["slots"] = [
            {"property": _SCHEME_PROPERTY, "binding": {"state": "set", "refs": schemes}}
        ]
        return identity, graph

    def test_inconsistent_candidates_over_the_naming_cap_refuse_truncated_through_the_mint_door(
        self,
    ):
        identity, graph = self._truncated_identity_and_graph()
        client = _client(graph=graph)
        spec_id = recompute_identity(identity).spec_id
        definition = wire_definition_for(spec_id=spec_id, identity=identity)

        resp = client.post("/api/datasets", json=_spec_id_body(spec_id, definition=definition))

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "member_not_resolvable"
        assert len(problem["issues"]) == 10
        assert problem["detail"].endswith("; more members disagreed than are named here")

    def test_inconsistent_candidates_over_the_naming_cap_refuse_truncated_through_the_definition_door(
        self,
    ):
        identity, graph = self._truncated_identity_and_graph()
        client = _client(graph=graph)
        spec_id = recompute_identity(identity).spec_id
        reg = client.post(
            "/api/datasets/register",
            json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
        )
        assert reg.status_code == 200, reg.text

        resp = client.get(f"/api/datasets/{spec_id}")

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "member_not_resolvable"
        assert len(problem["issues"]) == 10
        assert problem["detail"].endswith("; more members disagreed than are named here")


class TestEnumeratedCandidateOwnArcRefusalsMapToMemberNotResolvable:
    """Degenerate observation properties and multi-valued constraints map to non-retryable 422s."""

    @pytest.mark.parametrize(
        "arcs",
        [
            pytest.param(
                {
                    _ODA_GRANTS: _member_arcs(
                        scheme="ODAGrants", roles=("observationAbout", "recipient")
                    ),
                    _ODA_LOANS: _member_arcs(scheme="ODALoans"),
                },
                id="degenerate-observation-properties",
            ),
            pytest.param(
                {
                    _ODA_GRANTS: {
                        **_member_arcs(scheme="ODAGrants"),
                        _SCHEME_PROPERTY: ("ODAGrants", "ODALoans"),
                    },
                    _ODA_LOANS: _member_arcs(scheme="ODALoans"),
                },
                id="multi-valued-constraint",
            ),
        ],
    )
    def test_enumerated_candidate_own_arc_refusal_maps_to_member_not_resolvable_through_the_mint_door(
        self, arcs: dict
    ):
        store = FakeDatasetStore()
        graph = FakeMintGraphReader(arcs=arcs)
        client = _client(graph=graph, store=store)
        identity = _two_member_identity()
        spec_id = recompute_identity(identity).spec_id
        definition = wire_definition_for(spec_id=spec_id, identity=identity)

        resp = client.post("/api/datasets", json=_spec_id_body(spec_id, definition=definition))

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "member_not_resolvable"
        assert problem["retryable"] is False
        assert store.state.spec_registry == {}

    @pytest.mark.parametrize(
        "arcs",
        [
            pytest.param(
                {
                    _ODA_GRANTS: _member_arcs(
                        scheme="ODAGrants", roles=("observationAbout", "recipient")
                    ),
                    _ODA_LOANS: _member_arcs(scheme="ODALoans"),
                },
                id="degenerate-observation-properties",
            ),
            pytest.param(
                {
                    _ODA_GRANTS: {
                        **_member_arcs(scheme="ODAGrants"),
                        _SCHEME_PROPERTY: ("ODAGrants", "ODALoans"),
                    },
                    _ODA_LOANS: _member_arcs(scheme="ODALoans"),
                },
                id="multi-valued-constraint",
            ),
        ],
    )
    def test_enumerated_candidate_own_arc_refusal_maps_to_member_not_resolvable_through_the_definition_door(
        self, arcs: dict
    ):
        graph = FakeMintGraphReader(arcs=arcs)
        client = _client(graph=graph)
        identity = _two_member_identity()
        spec_id = recompute_identity(identity).spec_id
        reg = client.post(
            "/api/datasets/register",
            json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
        )
        assert reg.status_code == 200, reg.text

        resp = client.get(f"/api/datasets/{spec_id}")

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "member_not_resolvable"
        assert problem["retryable"] is False


class TestSdmxDeterministicRejectionsAreNotRetryable:
    """Typed SDMX query errors map to request or internal refusals.
    Generic HTTP errors remain 502.
    """

    def test_missing_filter_error_is_non_retryable_internal_error(self):
        exc = SdmxMissingFilterError(
            "missing required SDMX component filter: variableMeasured", upstream_status=400
        )

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 500
        assert spec.code == "internal_error"
        assert spec.retryable is False

    def test_unsupported_filter_error_is_non_retryable_internal_error(self):
        exc = SdmxUnsupportedFilterError(
            "unsupported SDMX component filter: scalingFactor", upstream_status=400
        )

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 500
        assert spec.code == "internal_error"
        assert spec.retryable is False

    def test_containment_filter_error_is_non_retryable_internal_error(self):
        exc = SdmxContainmentFilterError(
            "require containedInPlace+ and typeOf", upstream_status=400
        )

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 500
        assert spec.code == "internal_error"
        assert spec.retryable is False

    def test_incompatible_variables_error_is_non_retryable_internal_error(self):
        exc = SdmxIncompatibleVariablesError(
            "incompatible observationProperties", upstream_status=400
        )

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 500
        assert spec.code == "internal_error"
        assert spec.retryable is False

    def test_members_not_held_lists_every_absent_stat_var(self):
        absent = [f"AbsentStatVar{i}" for i in range(3)]
        exc = MembersNotHeldError(stat_vars=list(reversed(absent)))

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 500
        assert spec.code == "members_not_held"
        assert spec.retryable is False
        assert [issue.member for issue in spec.issues or ()] == absent

    def test_a_dataset_disagreement_is_a_server_fault_whose_detail_names_the_dataset(self):
        exc = DatasetDisagreementError(
            "'dcid:Member' is a member of dataset 'provenance/Elected', and its facet "
            "selection chose provenance 'provenance/Other'",
            dataset="provenance/Elected",
            stat_var="dcid:Member",
        )

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 500
        assert spec.code == "internal_error"
        assert spec.retryable is False
        assert "provenance/Elected" in spec.detail
        assert [issue.member for issue in spec.issues or ()] == ["dcid:Member"]

    def test_a_frame_the_grammar_refuses_is_a_typed_server_fault(self):
        exc = FrameError(
            "unknown_column", "row carries unknown column(s): ['stat_var_exampleAxis']"
        )

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 500
        assert spec.code == "internal_error"
        assert spec.retryable is False
        assert "stat_var_exampleAxis" in spec.detail

    def test_plain_sdmx_http_error_still_classifies_as_upstream_error(self):
        """An untyped `SdmxHttpError` remains a retryable 502."""
        exc = SdmxHttpError(
            "some other mixer rejection this client doesn't type", upstream_status=400
        )

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 502
        assert spec.code == "upstream_error"
        assert spec.retryable is True

    def test_role_ceiling_refuses_request_invalid_through_the_real_door(self):
        """Four bound roles exceed the mixer's limit of three and return a non-retryable 400."""
        graph = FakeMintGraphReader(
            arcs={
                _STAT_VAR: {
                    "typeOf": ("StatisticalVariable",),
                    "observationProperties": ("recipient", "donor", "sector", "channel"),
                    "measuredProperty": ("amount",),
                }
            },
            facets={(_STAT_VAR, "country/KEN"): [_FACET]},
            labels={"OECD_DAC2A": "OECD DAC2a"},
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400, json={"message": "max supported entity slots is 3", "code": 3}
            )

        sdmx = SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler))
        store = FakeDatasetStore()
        client = make_client(
            registry=LiveSpecRegistry(store=store),
            spec_reader=LiveSpecReader(store=store),
            graph=graph,
            sdmx=sdmx,
            artifact_store=FakeArtifactStore(),
            snapshot_reader=FakeSnapshotReader(),
            reference_writer=FakeArtifactReferenceWriter(),
            coverage=_coverage_service(donor_role="donor", donor="country/USA"),
        )
        body = {
            "schema_version": 1,
            "input": {
                "kind": "statvar_set",
                "statvars": [_STAT_VAR_REF],
                "scope": {
                    "entities": {
                        "recipient": ["country/KEN"],
                        "donor": ["country/USA"],
                        "sector": ["sector/Health"],
                        "channel": ["channel/Bilateral"],
                    },
                    "facets": _SCOPE["facets"],
                },
            },
            "options": {"bundle": False},
        }

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code == 400, resp.text
        problem = resp.json()
        assert problem["code"] == "request_invalid"
        assert problem["retryable"] is False


class TestFormatSizeRefusalsAreNonRetryable:
    """Excel and Parquet size limits map to non-retryable 422 refusals."""

    def test_excel_limit_maps_to_excel_limit_422(self):
        exc = ExcelLimitError("17000 columns exceeds Excel's 16384-column limit")

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 422
        assert spec.code == "excel_limit"
        assert spec.retryable is False
        assert spec.detail == "17000 columns exceeds Excel's 16384-column limit"

    def test_parquet_size_bound_maps_to_parquet_size_bound_422(self):
        exc = ParquetSizeBoundError("parquet output exceeds its 1000-byte size bound")

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 422
        assert spec.code == "parquet_size_bound"
        assert spec.retryable is False
        assert spec.detail == "parquet output exceeds its 1000-byte size bound"


def test_pivot_not_unique_names_each_duplicate_key_then_each_varying_column():
    exc = PivotNotUniqueError(
        "pivot on index='recipient', columns='time_period' is not unique",
        duplicate_keys=(("country/KEN", "2015"),),
        truncated=False,
        varying=(("provider", 30), ("sector", 4)),
    )

    spec = classify_mint_exception(exc)

    assert spec is not None
    assert spec.code == "pivot_not_unique"
    assert [(i.pointer, i.member, i.axis) for i in spec.issues] == [
        ("/options/pivot", "('country/KEN', '2015')", None),
        ("/options/pivot", None, "provider"),
        ("/options/pivot", None, "sector"),
    ]


@pytest.mark.parametrize(
    ("aggregate_type", "code"),
    [
        (DefaultFacetUndefinedError, "default_facet_undefined"),
        (FacetNotAvailableError, "facet_not_available"),
    ],
)
def test_a_refusal_on_three_axes_serves_each_issue_with_its_own_code_and_detail(
    aggregate_type, code
):
    errors = {
        "unit": DefaultFacetUndefinedError("unit has no default", facet_pointer="/facets/unit"),
        "provenance": FacetNotAvailableError(
            "provenance bound to nothing", facet_pointer="/facets/provenance"
        ),
        "measurementMethod": FacetNotAvailableError(
            "method has no observations", facet_pointer="/facets/measurementMethod"
        ),
    }
    aggregate = aggregate_type(
        "one or more facet axes could not be resolved",
        issues=tuple(_as_issue(axis, error) for axis, error in errors.items()),
    )

    spec = classify_mint_exception(aggregate)

    assert spec is not None
    assert spec.code == code
    assert spec.detail == str(aggregate)
    wire = [issue.model_dump(mode="json", exclude_none=True) for issue in spec.issues or ()]
    assert [(issue["axis"], issue["code"], issue["detail"]) for issue in wire] == [
        (axis, error.code, str(error)) for axis, error in errors.items()
    ]


class TestPivotNotHomogeneousCarriesPointerIssuesAndTruncated:
    """`PivotNotHomogeneousError` maps to `/options/pivot` with one issue per offender."""

    def test_pivot_not_homogeneous_maps_to_422_with_pointer_issues_and_truncated(self):
        exc = PivotNotHomogeneousError(
            "pivot on index='recipient' would merge rows that differ on column(s) it drops",
            offenders=(("unit", "country/KEN", ("EUR", "USD")),),
            truncated=False,
        )

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 422
        assert spec.code == "pivot_not_homogeneous"
        assert spec.retryable is False
        assert spec.pointer is None
        assert spec.truncated is False
        assert len(spec.issues) == 1
        assert spec.issues[0].pointer == "/options/pivot"
        assert spec.issues[0].member == "unit='country/KEN'"

    def test_pivot_not_homogeneous_truncated_flag_passes_through(self):
        exc = PivotNotHomogeneousError(
            "pivot on index='recipient' would merge rows that differ on column(s) it drops "
            "(truncated)",
            offenders=(("unit", "country/KEN", ("EUR", "USD")),),
            truncated=True,
        )

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.truncated is True


class TestColumnNotDroppableCarriesOneIssuePerColumn:
    """`ColumnNotDroppableError` maps to `/options/columns`, with one issue per column."""

    def test_column_not_droppable_maps_to_422_with_one_issue_per_column(self):
        exc = ColumnNotDroppableError(
            "column(s) ('stat_var', 'unit') are grain, unit, or provenance and cannot be dropped",
            columns=("stat_var", "unit"),
        )

        spec = classify_mint_exception(exc)

        assert spec is not None
        assert spec.status == 422
        assert spec.code == "column_not_droppable"
        assert spec.retryable is False
        assert len(spec.issues) == 2
        assert [issue.pointer for issue in spec.issues] == ["/options/columns", "/options/columns"]
        assert [issue.member for issue in spec.issues] == ["stat_var", "unit"]
        assert "stat_var" in spec.detail
        assert "unit" in spec.detail


def test_artifact_record_conflict_stays_non_retryable_and_names_a_reconciliation_warning():
    exc = ArtifactRecordConflictError(
        digest="sha256:deadbeef",
        tier=Tier.working,
        stored_bucket="dev-fin-observatory-datasets-working",
        stored_object_name="sha256:deadbeef",
        stored_size_bytes=100,
        incoming_bucket="dev-fin-observatory-datasets-working-v2",
        incoming_object_name="sha256:deadbeef",
        incoming_size_bytes=120,
    )
    exc.subject_id = "spec_abc123"
    exc.content_version = "cv_deadbeef"

    spec = classify_mint_exception(exc)

    assert spec is not None
    assert spec.status == 500
    assert spec.retryable is False
    # The request path logs the line the problem names (`dfo_platform.problem.log_problem`).
    assert spec.log is not None
    assert spec.log_level == logging.WARNING
    assert all(part in spec.log for part in ("spec_abc123", "cv_deadbeef", "sha256:deadbeef"))


class _ConflictingArtifactStore(FakeArtifactStore):
    """Raises `ArtifactRecordConflictError` on every `put()` -- stands in
    for a bucket-rotation race where a fresh upload's digest is already
    recorded against a different bucket/object_name.
    """

    def put(self, data: bytes, *, tier: Tier):
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        raise ArtifactRecordConflictError(
            digest=digest,
            tier=tier,
            stored_bucket="dev-fin-observatory-datasets-working",
            stored_object_name=digest,
            stored_size_bytes=len(data) + 1,
            incoming_bucket="dev-fin-observatory-datasets-working-v2",
            incoming_object_name=digest,
            incoming_size_bytes=len(data),
        )


def test_mint_time_artifact_record_conflict_names_the_subject_through_the_real_door(caplog):
    store = FakeDatasetStore()
    body = {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
        "options": {"bundle": False},
    }

    # The baseline call establishes this frame's content version on the same registry store.
    working_client = _client(artifact_store=FakeArtifactStore(), store=store)
    baseline = working_client.post("/api/datasets", json=body)
    assert baseline.status_code == 200, baseline.text
    content_version = baseline.json()["content_version"]

    client = _client(artifact_store=_ConflictingArtifactStore(), store=store)
    with caplog.at_level(logging.WARNING):
        resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 500, resp.text
    problem = resp.json()
    assert problem["retryable"] is False
    minted_spec_ids = list(store.state.spec_registry)
    assert len(minted_spec_ids) == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        minted_spec_ids[0] in r.getMessage() and content_version in r.getMessage() for r in warnings
    )


class _InventoryFailingArtifactStore(FakeArtifactStore):
    """Raises `ArtifactInventoryError` on every `put()`, as an inventory write that fails after
    the object landed does."""

    def put(self, data: bytes, *, tier: Tier):
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        raise ArtifactInventoryError(
            digest=digest,
            tier=tier,
            bucket="dev-fin-observatory-datasets-working",
            object_name=digest,
        )


def test_a_mint_whose_inventory_write_fails_logs_the_reconciliation_marker_once(caplog):
    """The classifier names the marker line and the request path logs it, once."""
    body = {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
        "options": {"bundle": False},
    }
    client = _client(artifact_store=_InventoryFailingArtifactStore(), store=FakeDatasetStore())

    with caplog.at_level(logging.ERROR):
        resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 503, resp.text
    marked = [r for r in caplog.records if ARTIFACT_INVENTORY_WRITE_FAILED_MARKER in r.getMessage()]
    assert len(marked) == 1
    assert marked[0].levelno == logging.ERROR


def test_artifact_inventory_write_failure_names_the_reconciliation_marker():
    """A failed inventory write after object storage logs the orphan marker."""
    exc = ArtifactInventoryError(
        digest="sha256:deadbeef",
        tier=Tier.working,
        bucket="dev-fin-observatory-datasets-working",
        object_name="sha256:deadbeef",
    )

    spec = classify_mint_exception(exc)

    assert spec is not None
    assert spec.status == 503
    assert spec.retryable is True
    assert spec.log is not None
    assert spec.log_level == logging.ERROR
    assert spec.log.startswith(ARTIFACT_INVENTORY_WRITE_FAILED_MARKER)
    assert "sha256:deadbeef" in spec.log


def test_artifact_inventory_lookup_failure_does_not_name_the_reconciliation_marker():
    """A failed lookup leaves the object intact. The orphan marker applies to inventory writes."""
    exc = ArtifactInventoryError(digest="sha256:deadbeef", tier=Tier.working)

    spec = classify_mint_exception(exc)

    assert spec is not None
    assert spec.status == 503
    assert ARTIFACT_INVENTORY_WRITE_FAILED_MARKER not in (spec.log or "")


def test_artifact_write_failure_does_not_name_the_reconciliation_marker():
    """A failed object write leaves no orphan to reconcile."""
    exc = ArtifactWriteError(object_name="sha256:deadbeef", digest="sha256:deadbeef")

    spec = classify_mint_exception(exc)

    assert spec is not None
    assert spec.status == 503
    assert ARTIFACT_INVENTORY_WRITE_FAILED_MARKER not in (spec.log or "")


def test_no_bound_entity_mints_instead_of_refusing_ahead_of_the_facet_axes():
    fetches: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetches.append(request)
        return httpx.Response(
            200,
            text=_CSV_HEADER + _CSV_ROW,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = _client(sdmx=SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler)))
    body = {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF]},
    }

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["definition"]["identity"]["entities"] == {}
    assert manifest["coverage"]["rows"] == 1
    assert {f["axis"]: f["basis"] for f in manifest["trace"]["facets"]} == dict.fromkeys(
        ("provenance", "measurementMethod", "unit"), "deployment_fallback"
    )
    # The fetch covers every series coverage holds and names its recipient axis.
    (fetch,) = fetches
    assert fetch.url.params["c[recipient]"] == "country/KEN"
    assert "c[provider]" not in fetch.url.params


class TestConversionRefusalsMapTo422ThroughTheRealMintDoor:
    """Request-form unit conversion errors return 422 through the real mint door.
    The refusals occur during facet selection, before the SDMX fetch.
    """

    def _currency_facet(self, unit: str) -> Facet:
        return Facet(
            facet_id="facet1",
            provenance_id="OECD_DAC2A",
            measurement_method="GrantEquivalent",
            unit=unit,
            obs_count=1,
            earliest_date="2015",
            latest_date="2015",
        )

    def _currency_graph(self, *, unit: str, extra_arcs: dict | None = None) -> FakeMintGraphReader:
        return FakeMintGraphReader(
            arcs={**_SHAPE_ARCS, **(extra_arcs or {})},
            facets={(_STAT_VAR, "country/KEN"): [self._currency_facet(unit)]},
            labels={"OECD_DAC2A": "OECD DAC2a"},
        )

    def _currency_sdmx(self, unit: str, *, rates: dict[str, dict[int, float]] | None = None):
        # The head-priming probe reads SDMX, so its unit must match the graph fixture's unit.
        # `rates` adds the DAC exchange-rate table a converting request loads.
        row = (
            f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,{unit},GrantEquivalent,"
            "NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
        )

        def observations(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text=_CSV_HEADER + row,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )

        bodies = recipe_table_bodies(_DAC_RECIPE, rates=rates) if rates is not None else {}
        return RateTableServer(bodies, fallback=observations).client(base_url=_BASE_URL)

    def _unit_request_identity(self, request: dict) -> dict:
        return pinned_identity(
            stat_var=_STAT_VAR,
            entities={"recipient": ["country/KEN"]},
            facets={"unit": {"state": "value", "request": request}},
        )

    def _post(
        self,
        *,
        graph: FakeMintGraphReader,
        identity: dict,
        unit: str,
        artifact_store=None,
        rates: dict[str, dict[int, float]] | None = None,
    ):
        client = _client(
            graph=graph,
            sdmx=self._currency_sdmx(unit, rates=rates),
            coverage=_coverage_service(unit=unit),
            artifact_store=artifact_store,
        )
        spec_id = recompute_identity(identity).spec_id
        definition = wire_definition_for(spec_id=spec_id, identity=identity)
        return client.post("/api/datasets", json=_spec_id_body(spec_id, definition=definition))

    def test_non_currency_native_unit_refuses_facet_binding_unsupported(self):
        # A request-form binding requires `CurrencyUSD`. This facet carries plain `USD`.
        graph = _default_graph()
        identity = self._unit_request_identity({"currency": "USD", "prices": "current"})

        resp = self._post(graph=graph, identity=identity, unit="USD")

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "facet_binding_unsupported"
        assert problem["retryable"] is False
        assert "pointer" not in problem
        assert "cause" not in problem

    def test_declared_non_nominal_qualifier_refuses_recipe_not_declared_with_its_cause(self):
        # The StatVar declares `RealValue`, so nominal currency conversion is refused.
        graph = self._currency_graph(
            unit="CurrencyUSD",
            extra_arcs={
                "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
                _STAT_VAR: {**_SHAPE_ARCS[_STAT_VAR], "measurementQualifier": ("RealValue",)},
            },
        )
        identity = self._unit_request_identity({"currency": "USD", "prices": "current"})

        resp = self._post(graph=graph, identity=identity, unit="CurrencyUSD")

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "recipe_not_declared"
        assert problem["retryable"] is False
        assert "pointer" not in problem
        assert problem["cause"]["code"] == "nominal_undeclared"
        assert problem["cause"]["entities"] == []

    def test_no_qualifier_arc_at_all_natively_satisfied_serves_the_native_unit(self):
        # With no qualifier arc, the native `CurrencyUSD` facet satisfies the request directly.
        graph = self._currency_graph(
            unit="CurrencyUSD",
            extra_arcs={"CurrencyUSD": {"typeOf": ("CurrencyUnit",)}},
        )
        identity = self._unit_request_identity({"currency": "USD", "prices": "current"})
        artifact_store = FakeArtifactStore()

        resp = self._post(
            graph=graph, identity=identity, unit="CurrencyUSD", artifact_store=artifact_store
        )

        assert resp.status_code == 200, resp.text
        manifest = resp.json()
        statistics_artifact = next(
            a for a in manifest["artifacts"] if a["rendering"] == {"statistics": "series"}
        )
        table = pq.read_table(io.BytesIO(artifact_store.fetch(statistics_artifact["url"])))
        assert json.loads(table.to_pylist()[0]["facet"])["unit"] == {
            "currency": "USD",
            "prices": "current",
        }

    def test_legacy_native_currency_refuses_rate_coverage_gap_with_its_cause(self):
        # `CurrencyFRF` is a withdrawn currency, so the exchange-rate table holds no row for it.
        graph = self._currency_graph(
            unit="CurrencyFRF",
            extra_arcs={
                _STAT_VAR: {**_SHAPE_ARCS[_STAT_VAR], **CONVERTIBLE_STAT_VAR_ARCS},
                **conversion_arcs(
                    recipe_id=_DAC_RECIPE,
                    provenance_id="OECD_DAC2A",
                    currencies=["CurrencyUSD", "CurrencyEUR", "CurrencyFRF"],
                ),
            },
        )
        identity = self._unit_request_identity({"currency": "EUR", "prices": "current"})

        resp = self._post(
            graph=graph,
            identity=identity,
            unit="CurrencyFRF",
            rates={"CurrencyUSD": {2015: 1.0}, "CurrencyEUR": {2015: 0.9}},
        )

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "rate_coverage_gap"
        assert problem["retryable"] is False
        assert "pointer" not in problem
        assert problem["cause"]["code"] == "rate_missing"
        assert problem["cause"]["entities"] == ["CurrencyFRF@2015"]
        assert problem["cause"]["truncated"] is False


# A subject-only StatVar whose SDMX body names no place on its last row.
_SUBJECT_STAT_VARS = ("LifeExpectancy_Person",)
_SUBJECT_ENTITY = "country/KEN"
_SUBJECT_FACET = Facet(
    facet_id="facet1",
    provenance_id="WorldBank",
    measurement_method="dc-aggregate",
    unit="Years",
    obs_count=1,
    earliest_date="2020",
    latest_date="2020",
)


def _subject_row(stat_var: str, place: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{place},Years,dc-aggregate,"
        "NotApplicable,WorldBank,2020,61.4,,facet1\r\n"
    )


def test_a_row_without_a_place_mid_stream_refuses_upstream_error_through_the_mint_door():
    body = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
        "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
        + "".join(_subject_row(sv, _SUBJECT_ENTITY) for sv in _SUBJECT_STAT_VARS)
        + _subject_row(_SUBJECT_STAT_VARS[0], "NotApplicable")
    )
    graph = FakeMintGraphReader(
        arcs={
            sv: {
                "typeOf": ("StatisticalVariable",),
                "measuredProperty": ("lifeExpectancy",),
                "populationType": ("Person",),
            }
            for sv in _SUBJECT_STAT_VARS
        },
        facets={(sv, _SUBJECT_ENTITY): [_SUBJECT_FACET] for sv in _SUBJECT_STAT_VARS},
        labels={"WorldBank": "World Bank", _SUBJECT_ENTITY: "Kenya"},
    )
    sdmx = SdmxClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text=body,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )
        ),
    )
    client = _client(graph=graph, sdmx=sdmx)

    resp = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {
                "kind": "statvar_set",
                "statvars": [f"dcid:{sv}" for sv in _SUBJECT_STAT_VARS],
                "scope": {
                    "entities": {"observationAbout": [_SUBJECT_ENTITY]},
                    "facets": {
                        "provenance": {"ref": "WorldBank"},
                        "measurementMethod": {"ref": "dc-aggregate"},
                        "unit": {"ref": "Years"},
                    },
                },
            },
            "options": {"bundle": False},
        },
    )

    assert resp.status_code == 502, resp.text
    problem = resp.json()
    assert problem["code"] == "upstream_error"
    assert problem["retryable"] is True
