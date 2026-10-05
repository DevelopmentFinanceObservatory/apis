"""Pins standard resolution from the first `observationAbout` entity and the empty-entity path."""

from __future__ import annotations

from search.engine.region import resolve_spec_resubmit
from search.engine.retrieval.retrieve import Materialised
from search.engine.retrieval.standard import standard_resolve
from search.engine.shape import ShapeDraft, entities_by_role
from search.models import CoverageBare, ResubmitIdentityPayload
from series_coverage.schema import RoleNames
from tests.engine._base_identity import (
    PERSON_COUNT_SHAPE,
    destination_entity,
    make_base,
)
from tests.engine._fake_coverage import FakeCoverageService, empty_coverage_snapshot, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph


def _standard_shape(members: list[str]) -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_shape",
        label="test_shape",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={m: {} for m in members},
        representative_score=0.9,
    )


def _obs_graph(sv: str, entity: str, obs_count: int) -> FakeGraph:
    obs = {
        f"{sv}|{entity}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": obs_count}]
    }
    return FakeGraph(nodes={}, obs=obs, detect={}, resolve={})


class TestStandardResolverEntitiesMap:
    def test_observation_about_entities_resolves_the_same_as_the_named_place(self):
        shape = _standard_shape(["sv_a"])
        graph = _obs_graph("sv_a", "country/KEN", 1)

        result = standard_resolve(
            shape=shape,
            bindings=[],
            entities={"observationAbout": ["country/KEN"]},
            graph=graph,
        )

        assert isinstance(result, Materialised)
        assert result.has_data is True
        assert result.sv_dcids == ["sv_a"]

    def test_empty_entities_map_is_the_no_entity_bare_coverage_path(self):
        """An empty entity map uses bare coverage without a place probe."""
        shape = _standard_shape(["sv_a"])

        result = standard_resolve(
            shape=shape,
            bindings=[],
            entities={},
            graph=FakeGraph(),
        )

        assert isinstance(result, Materialised)
        assert result.has_data is True
        assert isinstance(result.coverage, CoverageBare)


class TestBoundaryHelperMatchesReplacedLiterals:
    """Subject shapes use `observationAbout`. Directional shapes map places to declared roles."""

    def test_standard_places_stay_on_the_subject_key(self):
        shape = _standard_shape(["sv_a"])
        graph = _obs_graph("sv_a", "country/KEN", 1)
        entities = entities_by_role(shape=shape, subject=["country/KEN"])
        assert entities == {"observationAbout": ["country/KEN"]}

        result = standard_resolve(
            shape=shape,
            bindings=[],
            entities=entities,
            graph=graph,
        )

        assert isinstance(result, Materialised)
        assert result.has_data is True
        assert result.sv_dcids == ["sv_a"]

    def test_standard_promote_places_a_posted_subject_on_the_destination_role(self):
        """A role-bearing StatVar maps the posted subject ref to its recipient role."""
        sv_dcid = "Count_Person_RoleBearingStandardPromote"
        nodes = {
            sv_dcid: {
                "label": "Population by role",
                "type": "StatisticalVariable",
                "arcs": {
                    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                    "name": {"nodes": [{"value": "Population by role"}]},
                    "populationType": {"nodes": [{"dcid": "Person"}]},
                    "measuredProperty": {"nodes": [{"dcid": "count"}]},
                    "statType": {"nodes": [{"dcid": "measuredValue"}]},
                    "observationProperties": {
                        "nodes": [{"dcid": "provider"}, {"dcid": "recipient"}]
                    },
                },
            },
            "country/KEN": {"label": "Kenya", "type": "Country"},
        }
        base_spec_id, base_identity = make_base(PERSON_COUNT_SHAPE)
        inp = ResubmitIdentityPayload(
            identity=make_base(
                PERSON_COUNT_SHAPE, entities=[destination_entity("country/KEN", "Kenya")]
            )[1],
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[sv_dcid],
        )

        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(nodes=nodes),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert region.status == "no_data", f"expected no_data, got {region.status!r}: {region}"
        assert region.no_data_reason == "no_observations"

    def test_standard_promote_labels_the_placed_entity_directional(self, monkeypatch):
        """A role-bearing StatVar labels the posted place as `destination`."""
        sv_dcid = "Count_Person_RoleBearingStandardPromoteLabelled"
        nodes = {
            sv_dcid: {
                "label": "Population by role",
                "type": "StatisticalVariable",
                "arcs": {
                    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                    "name": {"nodes": [{"value": "Population by role"}]},
                    "populationType": {"nodes": [{"dcid": "Person"}]},
                    "measuredProperty": {"nodes": [{"dcid": "count"}]},
                    "statType": {"nodes": [{"dcid": "measuredValue"}]},
                    "observationProperties": {
                        "nodes": [{"dcid": "provider"}, {"dcid": "recipient"}]
                    },
                },
            },
            "country/KEN": {"label": "Kenya", "type": "Country"},
        }
        base_spec_id, base_identity = make_base(PERSON_COUNT_SHAPE)
        inp = ResubmitIdentityPayload(
            identity=make_base(
                PERSON_COUNT_SHAPE, entities=[destination_entity("country/KEN", "Kenya")]
            )[1],
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[sv_dcid],
        )

        _auto_coverage_index_1 = patch_coverage(
            monkeypatch,
            [(sv_dcid, "f1", "country/USA", "country/KEN", [2020])],
            role_names={sv_dcid: RoleNames(donor="provider", recipient="recipient")},
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(nodes=nodes),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )

        assert region.status == "definite", f"expected definite, got {region.status!r}: {region}"
        roles_by_ref = {e.ref: e.role for e in region.specs[0].resolved.entities}
        assert roles_by_ref == {"country/KEN": "destination"}

    def test_standard_promote_refuses_a_posted_subject_on_an_unplaceable_role(self):
        """A subject ref cannot map to a declared role outside the axis vocabulary."""
        sv_dcid = "Count_Person_UnplaceableRoleStandardPromote"
        nodes = {
            sv_dcid: {
                "label": "Population by role",
                "type": "StatisticalVariable",
                "arcs": {
                    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                    "name": {"nodes": [{"value": "Population by role"}]},
                    "populationType": {"nodes": [{"dcid": "Person"}]},
                    "measuredProperty": {"nodes": [{"dcid": "count"}]},
                    "statType": {"nodes": [{"dcid": "measuredValue"}]},
                    "observationProperties": {"nodes": [{"dcid": "contributor"}]},
                },
            },
            "country/KEN": {"label": "Kenya", "type": "Country"},
        }
        base_spec_id, base_identity = make_base(PERSON_COUNT_SHAPE)
        inp = ResubmitIdentityPayload(
            identity=make_base(
                PERSON_COUNT_SHAPE, entities=[destination_entity("country/KEN", "Kenya")]
            )[1],
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[sv_dcid],
        )

        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(nodes=nodes),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert region.status == "no_data", f"expected no_data, got {region.status!r}: {region}"
        assert region.no_data_reason == "role_incompatible"
