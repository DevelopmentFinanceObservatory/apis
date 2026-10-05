"""Tests `statvar_set` minting, with and without bound entities, and its
`member_not_resolvable` refusal. Registry and materialisation are real. Graph, SDMX, and artifact
storage are faked.
"""

from __future__ import annotations

import hashlib

import httpx
import pydantic
import pytest

from datasets_api.mint_models import Scope
from datasets_api.revision import SentinelRevisionReader
from dc_sdmx import SdmxClient, SdmxQueryError
from materialise.graph import Facet
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.materialise._scope_availability import ScopeAvailability, ScopeTransport

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_STAT_VAR = "MalariaGrantDisbursements"
_STAT_VAR_REF = f"dcid:{_STAT_VAR}"
_BASE_URL = "https://datacommons.one.org/core/api"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _csv_row(*, recipient: str, year: str, value: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{recipient},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,{year},{value},,facet1\r\n"
    )


_CSV_BODY = (
    _CSV_HEADER
    + _csv_row(recipient="country/KEN", year="2015", value="1000")
    + _csv_row(recipient="country/KEN", year="2016", value="2000")
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=2,
    earliest_date="2015",
    latest_date="2016",
)


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            },
            # Distinguishes `role_incompatible` from `invalid_role_ref`.
            "provider": {"typeOf": ("Property",)},
        },
        facets={(_STAT_VAR, "country/KEN"): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a", "country/KEN": "Kenya"},
    )


def _sdmx_transport(requests: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


_HAPPY_PATH_BODY = {
    "schema_version": 1,
    "input": {
        "kind": "statvar_set",
        "statvars": [_STAT_VAR_REF],
        "scope": {
            "entities": {"recipient": ["country/KEN"]},
            "facets": {
                "provenance": {"ref": "OECD_DAC2A"},
                "measurementMethod": {"ref": "GrantEquivalent"},
                "unit": {"ref": "USD"},
            },
        },
    },
    "options": {"bundle": False},
}


def _coverage_service() -> FakeCoverageService:
    """Builds `_STAT_VAR` coverage for the 2015 and 2016 CSV rows."""
    index = build_coverage_index(
        [(_STAT_VAR, "facet1", "country/USA", "country/KEN", [2015, 2016])],
        role_names={_STAT_VAR: RoleNames(donor="provider", recipient="recipient")},
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


def _client(
    *,
    artifact_store,
    sdmx_requests: list[httpx.Request] | None = None,
    sdmx: SdmxClient | None = None,
    revision_reader=None,
    store: FakeDatasetStore | None = None,
    graph: FakeMintGraphReader | None = None,
    coverage: FakeCoverageService | None = None,
):
    store = store if store is not None else FakeDatasetStore()
    if sdmx is None:
        requests = sdmx_requests if sdmx_requests is not None else []
        sdmx = SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport(requests))
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph if graph is not None else _graph(),
        sdmx=sdmx,
        revision_reader=revision_reader
        if revision_reader is not None
        else SentinelRevisionReader(),
        artifact_store=artifact_store,
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=coverage if coverage is not None else _coverage_service(),
    )


class TestMilestoneStatVarSetMintServesACorrectCsvFromASignedUrl:
    """Mints CSV whose signed artifact bytes match its digest and repeat mint."""

    def test_milestone(self):
        artifact_store = FakeArtifactStore()
        client = _client(artifact_store=artifact_store)

        first = client.post("/api/datasets", json=_HAPPY_PATH_BODY)
        assert first.status_code == 200, first.text
        manifest = first.json()

        assert manifest["definition"]["spec_id"].startswith("spec_")
        assert manifest["content_version"].startswith("cv_")
        # C79: mint responses include data and statistics artifacts.
        # Frame1 is recorded on the version record.
        assert len(manifest["artifacts"]) == 2
        artifact = next(
            a for a in manifest["artifacts"] if a["rendering"] != {"statistics": "series"}
        )
        assert artifact["artifact_digest"].startswith("sha256:")
        assert artifact["url"]

        fetched = artifact_store.fetch(artifact["url"])
        assert fetched.decode("utf-8").splitlines()[0].split(",") == [
            "stat_var",
            "recipient",
            "time_period",
            "value",
            "unit",
            "measurementMethod",
            "provenance",
            "status",
        ]
        assert hashlib.sha256(fetched).hexdigest() == artifact["artifact_digest"].removeprefix(
            "sha256:"
        )

        second = client.post("/api/datasets", json=_HAPPY_PATH_BODY)
        assert second.status_code == 200, second.text
        repeat_manifest = second.json()
        assert repeat_manifest["content_version"] == manifest["content_version"]
        assert (
            repeat_manifest["artifacts"][0]["artifact_digest"]
            == manifest["artifacts"][0]["artifact_digest"]
        )


class TestRevisionUnavailableWarning:
    """A sentinel revision reader adds `revision_unavailable`; an injected revision removes it."""

    def test_manifest_carries_the_warning_while_the_sentinel_is_active(self):
        client = _client(artifact_store=FakeArtifactStore())

        resp = client.post("/api/datasets", json=_HAPPY_PATH_BODY)

        assert resp.status_code == 200, resp.text
        warnings = resp.json()["warnings"]
        assert any(w["code"] == "revision_unavailable" for w in warnings)

    def test_manifest_omits_the_warning_under_an_injected_live_style_reader(self):
        from materialise.revision import RevisionSnapshot

        from ._graph_doubles import FakeRevisionReader

        snapshot = RevisionSnapshot(graph_revision="live-rev-2026-08-03", source_revisions=())
        client = _client(
            artifact_store=FakeArtifactStore(),
            revision_reader=FakeRevisionReader([snapshot, snapshot]),
        )

        resp = client.post("/api/datasets", json=_HAPPY_PATH_BODY)

        assert resp.status_code == 200, resp.text
        warnings = resp.json()["warnings"]
        assert not any(w["code"] == "revision_unavailable" for w in warnings)


class TestFacetTruthTable:
    def test_no_scope_at_all_mints_every_series_coverage_holds(self):
        # Each axis takes the deployment fallback, and the spec_id is the pinned hash
        # of this definition.
        client = _client(artifact_store=FakeArtifactStore())
        body = {
            "schema_version": 1,
            "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF]},
        }

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code == 200, resp.text
        manifest = resp.json()
        assert manifest["definition"]["spec_id"] == "spec_c9f997498f1674ca75e0082f69eddae9"
        assert manifest["definition"]["identity"]["entities"] == {}
        assert manifest["coverage"]["rows"] == 2
        assert {f["basis"] for f in manifest["trace"]["facets"]} == {"deployment_fallback"}

    def test_no_scope_at_all_mints_a_role_bearing_stat_var_across_every_recipient(self):
        # An unbound role means every entity. The fetch filters on every covered donor, the
        # smaller axis, and leaves recipients open.
        sdmx_requests: list[httpx.Request] = []
        client = _client(artifact_store=FakeArtifactStore(), sdmx_requests=sdmx_requests)
        body = {
            "schema_version": 1,
            "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF]},
            "options": {"bundle": False},
        }

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code == 200, resp.text
        assert resp.json()["definition"]["identity"]["entities"] == {}
        assert resp.json()["coverage"]["rows"] == 2
        assert [request.url.params["c[provider]"] for request in sdmx_requests] == ["country/USA"]
        assert all("c[recipient]" not in request.url.params for request in sdmx_requests)

    def test_facets_bound_entities_unbound_mints_under_the_bound_facets(self):
        client = _client(artifact_store=FakeArtifactStore())
        body = {
            "schema_version": 1,
            "input": {
                "kind": "statvar_set",
                "statvars": [_STAT_VAR_REF],
                "scope": {
                    "facets": {
                        "provenance": {"ref": "OECD_DAC2A"},
                        "measurementMethod": {"ref": "GrantEquivalent"},
                        "unit": {"ref": "USD"},
                    }
                },
            },
        }

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code == 200, resp.text
        manifest = resp.json()
        assert manifest["definition"]["identity"]["entities"] == {}
        assert manifest["coverage"]["rows"] == 2
        assert {f["basis"] for f in manifest["trace"]["facets"]} == {"bound"}

    def test_subject_only_statvar_with_no_scope_mints_every_place_that_holds_data(self):
        # A StatVar with no `observationProperties` arc is subject-only. With no place bound
        # it covers every place that holds data, and the fetch names them.
        stat_var = "Count_Person"
        graph = FakeMintGraphReader(
            arcs={stat_var: {"typeOf": ("StatisticalVariable",), "measuredProperty": ("count",)}},
            facets={(stat_var, "country/KEN"): [_FACET], (stat_var, "country/TGO"): [_FACET]},
            labels={"OECD_DAC2A": "OECD DAC2a"},
        )
        availability = ScopeAvailability(
            data=[(stat_var, "country/KEN"), (stat_var, "country/TGO")]
        )
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if (response := availability.respond(request)) is not None:
                return response
            requests.append(request)
            rows = "".join(
                f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{place},USD,GrantEquivalent,"
                f"NotApplicable,OECD_DAC2A,2015,10,,facet1\r\n"
                for place in ("country/KEN", "country/TGO")
            )
            return httpx.Response(
                200,
                text=_CSV_HEADER.replace(",recipient,", ",observationAbout,") + rows,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )

        client = _client(
            artifact_store=FakeArtifactStore(),
            graph=graph,
            sdmx=SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler)),
        )
        body = {
            "schema_version": 1,
            "input": {"kind": "statvar_set", "statvars": [f"dcid:{stat_var}"]},
        }

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code == 200, resp.text
        manifest = resp.json()
        assert manifest["definition"]["identity"]["entities"] == {}
        assert manifest["coverage"]["rows"] == 2
        assert {r.url.params["c[observationAbout]"] for r in requests} == {
            "country/KEN,country/TGO"
        }

    def test_subject_only_statvar_with_no_scope_and_no_data_refuses_default_facet_undefined(
        self,
    ):
        # The refusal a mint binding no entity can still meet: no place holds data.
        stat_var = "Count_Person"
        graph = FakeMintGraphReader(
            arcs={stat_var: {"typeOf": ("StatisticalVariable",), "measuredProperty": ("count",)}},
        )
        sdmx_requests: list[httpx.Request] = []
        no_data = ScopeAvailability(data=())
        client = _client(
            artifact_store=FakeArtifactStore(),
            graph=graph,
            sdmx=SdmxClient(
                base_url=_BASE_URL,
                transport=ScopeTransport(_sdmx_transport(sdmx_requests), no_data),
            ),
        )
        body = {
            "schema_version": 1,
            "input": {"kind": "statvar_set", "statvars": [f"dcid:{stat_var}"]},
        }

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code == 422, resp.text
        assert resp.json()["code"] == "default_facet_undefined"
        assert sdmx_requests == []

    def test_unbound_provenance_with_no_declared_policy_refuses_default_facet_undefined(self):
        # The probed facet has no provenance or declared default policy.
        # Provenance needs a ref; unit and measurement method may bind null.
        # `NotApplicable` maps to `None`; an empty cell stays `""`.
        graph = FakeMintGraphReader(
            arcs={
                _STAT_VAR: {
                    "typeOf": ("StatisticalVariable",),
                    "observationProperties": ("recipient",),
                }
            },
        )
        no_provenance_csv = _CSV_HEADER + (
            f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,USD,GrantEquivalent,"
            "NotApplicable,NotApplicable,2015,1000,,facet1\r\n"
        )
        sdmx = SdmxClient(
            base_url=_BASE_URL,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    text=no_provenance_csv,
                    headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
                )
            ),
        )
        # Omit provenance to match the probed facet.
        no_provenance_index = build_coverage_index(
            [(_STAT_VAR, "facet1", "country/USA", "country/KEN", [2015])],
            role_names={_STAT_VAR: RoleNames(donor="provider", recipient="recipient")},
            facet_metadata={
                "facet1": FacetMetadata(measurement_method="GrantEquivalent", unit="USD")
            },
        )
        client = _client(
            artifact_store=FakeArtifactStore(),
            graph=graph,
            sdmx=sdmx,
            coverage=FakeCoverageService(no_provenance_index),
        )
        body = {
            "schema_version": 1,
            "input": {
                "kind": "statvar_set",
                "statvars": [_STAT_VAR_REF],
                "scope": {
                    "entities": {"recipient": ["country/KEN"]},
                    "facets": {
                        "measurementMethod": {"ref": "GrantEquivalent"},
                        "unit": {"ref": "USD"},
                    },
                },
            },
        }

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code == 422, resp.text
        assert resp.json()["code"] == "default_facet_undefined"


def test_role_bearing_mint_with_no_facet_axis_bound_mints_a_dataset():
    """A role-bearing StatVar with one coverage facet mints when no facet axis is bound."""
    client = _client(artifact_store=FakeArtifactStore())
    body = {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": [_STAT_VAR_REF],
            "scope": {"entities": {"recipient": ["country/KEN"]}},
        },
    }

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["definition"]["spec_id"].startswith("spec_")
    facets = {facet["axis"]: facet for facet in manifest["trace"]["facets"]}
    assert facets["provenance"]["ref"] == "OECD_DAC2A"
    assert facets["provenance"]["selection_reason"] == "sole_candidate"


def test_two_identical_statvars_dedup_to_a_one_member_collection_with_the_warning():
    """Duplicate StatVars collapse to one collection member and add `members_deduplicated`."""
    client = _client(artifact_store=FakeArtifactStore())
    body = {
        **_HAPPY_PATH_BODY,
        "input": {**_HAPPY_PATH_BODY["input"], "statvars": [_STAT_VAR_REF, _STAT_VAR_REF]},
    }

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert "definition" not in manifest
    assert manifest["collection"]["collection_id"].startswith("col_")
    assert len(manifest["collection"]["members"]) == 1
    assert any(w["code"] == "members_deduplicated" for w in manifest["warnings"])


def test_member_not_resolvable_names_every_failing_statvar():
    client = _client(artifact_store=FakeArtifactStore())
    body = {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": [f"{_STAT_VAR_REF}", "dcid:UnknownShape"],
            "scope": {
                "entities": {"recipient": ["country/KEN"]},
                "facets": {
                    "provenance": {"ref": "OECD_DAC2A"},
                    "measurementMethod": {"ref": "GrantEquivalent"},
                    "unit": {"ref": "USD"},
                },
            },
        },
    }

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "member_not_resolvable"
    assert {issue["member"] for issue in problem["issues"]} == {"dcid:UnknownShape"}


class TestMalformedReshapeOptionsRefuseRequestInvalid:
    """Malformed reshape options return `request_invalid` at validation."""

    @staticmethod
    def _body(options: dict) -> dict:
        return {**_HAPPY_PATH_BODY, "options": {**_HAPPY_PATH_BODY["options"], **options}}

    @pytest.mark.parametrize(
        ("options", "pointer"),
        [
            pytest.param(
                {"groupby": {"by": ["recipient"], "op": "avg"}},
                "/options/groupby/op",
                id="groupby-op-not-sum",
            ),
            pytest.param(
                {"groupby": {"by": ["recipient"], "op": "sum", "extra": True}},
                "/options/groupby/extra",
                id="groupby-extra-key",
            ),
            pytest.param(
                {"groupby": {"by": [], "op": "sum"}},
                "/options/groupby/by",
                id="groupby-empty-by",
            ),
            pytest.param(
                {"groupby": {"by": ["recipient", "recipient"], "op": "sum"}},
                "/options/groupby/by",
                id="groupby-duplicate-by-entry",
            ),
            pytest.param(
                {"pivot": {"index": "year", "columns": "recipient", "extra": True}},
                "/options/pivot/extra",
                id="pivot-extra-key",
            ),
            pytest.param(
                {"pivot": {"index": "year", "columns": ""}},
                "/options/pivot/columns",
                id="pivot-empty-columns",
            ),
            pytest.param({"columns": ["year", ""]}, "/options/columns/1", id="columns-empty-entry"),
            pytest.param(
                {"columns": ["year", 2020]}, "/options/columns/1", id="columns-non-string-entry"
            ),
            pytest.param(
                {"columns": ["year", "recipient", "year"]},
                "/options/columns",
                id="columns-duplicate-entry",
            ),
        ],
    )
    def test_malformed_options_refuse_request_invalid(self, options: dict, pointer: str):
        client = _client(artifact_store=FakeArtifactStore())
        body = self._body(options)

        resp = client.post("/api/datasets", json=body)

        assert resp.status_code == 400, resp.text
        problem = resp.json()
        assert problem["code"] == "request_invalid"
        assert problem["pointer"] == pointer


def test_a_non_dcid_prefixed_statvar_is_request_invalid():
    client = _client(artifact_store=FakeArtifactStore())
    body = {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": ["MalariaGrantDisbursements"]},
    }

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "request_invalid"


class TestScopeEntitiesRefusals:
    """Wire validation rejects malformed roles, empty values, and reserved characters before
    graph or SDMX calls.
    """

    @staticmethod
    def _body(entities: dict) -> dict:
        return {
            "schema_version": 1,
            "input": {
                "kind": "statvar_set",
                "statvars": [_STAT_VAR_REF],
                "scope": {"entities": entities},
            },
        }

    @pytest.mark.parametrize(
        "entities",
        [
            pytest.param({"recipient": []}, id="empty-role-values"),
            pytest.param({"recipient": [""]}, id="empty-value"),
            pytest.param({"bad role": ["country/KEN"]}, id="invalid-role-name"),
        ],
    )
    def test_malformed_entities_refuse_request_invalid(self, entities: dict):
        client = _client(artifact_store=FakeArtifactStore())

        resp = client.post("/api/datasets", json=self._body(entities))

        assert resp.status_code == 400, resp.text
        problem = resp.json()
        assert problem["code"] == "request_invalid"
        assert problem["pointer"] == "/input/scope"

    @pytest.mark.parametrize("char", ["+", ","])
    def test_a_reserved_character_value_refuses_request_invalid(self, char):
        client = _client(artifact_store=FakeArtifactStore())

        resp = client.post(
            "/api/datasets", json=self._body({"recipient": [f"country/KEN{char}extra"]})
        )

        assert resp.status_code == 400, resp.text
        problem = resp.json()
        assert problem["code"] == "request_invalid"
        assert problem["pointer"] == "/input/scope"


class TestScopeEntitiesRoleRefusals:
    """After shape resolution, unknown properties return `invalid_role_ref`.
    Undeclared roles and `observationAbout` on a multi-role shape return `role_incompatible`.
    Issue pointers name the offending scope key.
    """

    @staticmethod
    def _body(entities: dict) -> dict:
        return {
            "schema_version": 1,
            "input": {
                "kind": "statvar_set",
                "statvars": [_STAT_VAR_REF],
                "scope": {"entities": entities},
            },
        }

    def test_an_undeclared_role_refuses_invalid_role_ref(self):
        client = _client(artifact_store=FakeArtifactStore())

        resp = client.post("/api/datasets", json=self._body({"donor": ["country/FRA"]}))

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "invalid_role_ref"
        assert len(problem["issues"]) == 1
        assert problem["issues"][0]["pointer"] == "/input/scope/entities/donor"
        assert problem["issues"][0]["member"] == "donor"

    def test_observation_about_on_a_role_declaring_statvar_refuses_role_incompatible(self):
        client = _client(artifact_store=FakeArtifactStore())

        resp = client.post("/api/datasets", json=self._body({"observationAbout": ["country/KEN"]}))

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "role_incompatible"
        assert len(problem["issues"]) == 1
        assert problem["issues"][0]["pointer"] == "/input/scope/entities/observationAbout"
        assert problem["issues"][0]["member"] == "observationAbout"

    def test_a_real_but_undeclared_property_refuses_role_incompatible(self):
        client = _client(artifact_store=FakeArtifactStore())

        resp = client.post("/api/datasets", json=self._body({"provider": ["org/USAID"]}))

        assert resp.status_code == 422, resp.text
        problem = resp.json()
        assert problem["code"] == "role_incompatible"
        assert len(problem["issues"]) == 1
        assert problem["issues"][0]["pointer"] == "/input/scope/entities/provider"
        assert problem["issues"][0]["member"] == "provider"


@pytest.mark.parametrize("char", ["+", ","])
def test_the_wire_boundary_and_the_sdmx_client_refuse_the_same_reserved_characters(char):
    """Both the wire model and `SdmxClient.data()` reject `+` and `,` filter values.
    The test names these characters directly to compare the independent validators.
    """
    value = f"country/KEN{char}extra"

    with pytest.raises(SdmxQueryError):
        SdmxClient(
            base_url=_BASE_URL, transport=httpx.MockTransport(lambda r: httpx.Response(200))
        ).data(variables=["Count_Person"], entities={"recipient": [value]})

    with pytest.raises(pydantic.ValidationError):
        Scope.model_validate({"entities": {"recipient": [value]}})
