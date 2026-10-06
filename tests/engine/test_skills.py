"""Tests spec-to-skill matching and response attachment at assembly time.

Hand-built specs and indexes avoid graph and LLM calls.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from search.engine.assemble import build_no_data_part
from search.engine.conjoin import assemble_region
from search.engine.envelope import make_diagnostics, make_response, now_ms
from search.engine.region import RegionResult
from search.engine.skills import project_spec_dcids
from search.engine.spec_build import build_spec
from search.models import (
    BindingValue,
    CandidatesPart,
    CoverageBare,
    DefinitePart,
    Entity,
    EntityRoleSubject,
    Expansion,
    NearestRealSpec,
    QueryEcho,
    Relaxation,
    ResolvedFacet,
    Shape,
    Slot,
    SlotKey,
    SlotValue,
    Spec,
    StatVar,
    StatVarSlotValue,
    TimeWindow,
)
from skills_index import SkillEntry, SkillFrontMatter, SkillMatch, SkillReach, SkillsIndex
from skills_index.manifest import build_manifest, content_version
from tests.engine._builders import (
    binding_set_slot,
    gref,
    shape_five_tuple,
    unbound_slot,
    value_slot,
)
from tests.engine._conjoin_fixtures import _region
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _stat_var(dcid: str, slot_values: list[StatVarSlotValue] | None = None) -> StatVar:
    slot_values = slot_values or []
    return StatVar(
        ref=gref(dcid),
        shape_id="test_shape",
        slot_values=slot_values,
        constraint_properties=sorted(
            sv.key.property.dcid for sv in slot_values if sv.key.property is not None
        ),
    )


def _slot_value(prop_dcid: str, val_dcid: str) -> StatVarSlotValue:
    return StatVarSlotValue(
        key=SlotKey(axis="how", property=gref(prop_dcid), label=prop_dcid),
        value=SlotValue(ref=gref(val_dcid), value_kind="enum_value"),
    )


def _entity(dcid: str) -> Entity:
    return Entity(ref=gref(dcid), entity_type=None, role=EntityRoleSubject())


def _spec_with(
    *,
    shape: Shape,
    stat_vars: list[StatVar],
    slots: list[Slot] | None = None,
    entities: list[Entity] | None = None,
    facets: dict[str, ResolvedFacet] | None = None,
    equivalent_readings: list[Spec] | None = None,
) -> Spec:
    return build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=slots or [],
        stat_vars=stat_vars,
        entities=entities or [],
        coverage=CoverageBare(has_data=True),
        facets=facets,
        equivalent_readings=equivalent_readings,
    )


def _nearest_real(shape: Shape, *, sv_dcid: str) -> NearestRealSpec:
    spec = _spec_with(shape=shape, stat_vars=[_stat_var(sv_dcid)])
    return NearestRealSpec(
        spec_id=spec.spec_id,
        definition=spec.definition,
        resolved=spec.resolved,
        relaxation=Relaxation(axis="how", property_dcid="TestProp", dropped_value_dcids=["X"]),
    )


def _skill_entry(
    skill_id: str, *, name: str = "Name", description: str = "Description"
) -> SkillEntry:
    text = f"---\nname: {skill_id}\ndescription: {description}\n---\nBody.\n"
    manifest = build_manifest(text, ())
    return SkillEntry(
        front_matter=SkillFrontMatter(
            id=skill_id,
            name=name,
            description=description,
            content_version=content_version(manifest),
        ),
        frontmatter={"name": skill_id, "description": description},
        url=None,
        attachments=(),
        reach=SkillReach(skill_id=skill_id, nodes=(), excluded=False),
        statvar_count=1,
        statvar_count_truncated=False,
        files=manifest,
    )


def _index(node_to_skills: dict[str, tuple[SkillMatch, ...]], skill_ids: list[str]) -> SkillsIndex:
    return SkillsIndex(
        skills={skill_id: _skill_entry(skill_id) for skill_id in skill_ids},
        node_to_skills=node_to_skills,
        issues=(),
        may_omit_published_skills=False,
    )


def _assemble_kwargs(**overrides):
    return {
        "query": "test query",
        "extra_warnings": [],
        "start_ms": now_ms(),
        "engine_build": "test",
        "max_candidates": 10,
    } | overrides


# ---------------------------------------------------------------------------
# project_spec_dcids: the shape's five anchors plus, for each slot with a
# constraint property and a value or set binding, the property and each bound
# enum_value dcid
# ---------------------------------------------------------------------------


def _where_entity_slot(entity_dcid: str) -> Slot:
    return Slot(
        key=SlotKey(axis="where", property=None, label="recipient"),
        binding=BindingValue(value=SlotValue(ref=gref(entity_dcid), value_kind="entity")),
    )


class TestProjectSpecDcids:
    def test_projects_the_bound_code_and_not_the_stat_vars_narrower_codes(self):
        """A spec bound to purpose 930 draws StatVars coded 93010. The bound 930
        projects, so the index's downward reach matches skills attached at or
        above 930; the StatVars' 93010 and their own dcids never project."""
        shape = shape_five_tuple("PopType", "measuredProp")
        stat_var = _stat_var(
            "sv/93010", slot_values=[_slot_value("crsPurpose", "crsPurpose/93010")]
        )
        spec = _spec_with(
            shape=shape,
            slots=[value_slot("how", "crsPurpose", "purpose", "crsPurpose/930")],
            stat_vars=[stat_var],
        )

        assert project_spec_dcids(spec) == {
            "PopType",
            "measuredProp",
            "measuredValue",
            "crsPurpose",
            "crsPurpose/930",
        }

    def test_projects_all_five_shape_anchors(self):
        shape = shape_five_tuple("PopType", "measuredProp", "TestQual", denom="TestDenom")
        spec = _spec_with(shape=shape, stat_vars=[_stat_var("sv/1")])

        assert project_spec_dcids(spec) == {
            "PopType",
            "measuredProp",
            "measuredValue",
            "TestQual",
            "TestDenom",
        }

    def test_bound_where_entity_projects_nothing(self):
        shape = shape_five_tuple("PopType", "measuredProp")
        spec = _spec_with(
            shape=shape,
            slots=[_where_entity_slot("country/FRA")],
            stat_vars=[_stat_var("sv/1")],
            entities=[_entity("country/FRA")],
        )

        assert project_spec_dcids(spec) == {"PopType", "measuredProp", "measuredValue"}

    def test_where_axis_constraint_slot_projects_nothing(self):
        """A place bound on a where-axis constraint property adds neither the
        property nor the place."""
        shape = shape_five_tuple("PopType", "measuredProp")
        where_slot = Slot(
            key=SlotKey(axis="where", property=gref("comparisonRegion"), label="region"),
            binding=BindingValue(value=SlotValue(ref=gref("country/FRA"), value_kind="entity")),
        )
        spec = _spec_with(shape=shape, slots=[where_slot], stat_vars=[_stat_var("sv/1")])

        assert project_spec_dcids(spec) == {"PopType", "measuredProp", "measuredValue"}

    def test_when_axis_constraint_slot_projects_nothing(self):
        """A time window bound on a date-valued constraint property adds nothing."""
        shape = shape_five_tuple("PopType", "measuredProp")
        when_slot = Slot(
            key=SlotKey(axis="when", property=gref("dateOfEvent"), label="date"),
            binding=BindingValue(
                value=SlotValue(
                    ref=None,
                    value_kind="time_window",
                    time_window=TimeWindow(start_year=2010, end_year=2020),
                )
            ),
        )
        spec = _spec_with(shape=shape, slots=[when_slot], stat_vars=[_stat_var("sv/1")])

        assert project_spec_dcids(spec) == {"PopType", "measuredProp", "measuredValue"}

    def test_set_binding_projects_the_property_and_each_enum_value(self):
        shape = shape_five_tuple("PopType", "measuredProp")
        spec = _spec_with(
            shape=shape,
            slots=[
                binding_set_slot(
                    "how", "crsPurpose", "purpose", ["crsPurpose/930", "crsPurpose/720"]
                )
            ],
            stat_vars=[_stat_var("sv/1")],
        )

        assert project_spec_dcids(spec) == {
            "PopType",
            "measuredProp",
            "measuredValue",
            "crsPurpose",
            "crsPurpose/930",
            "crsPurpose/720",
        }

    def test_unbound_slot_projects_nothing_not_even_its_property(self):
        shape = shape_five_tuple("PopType", "measuredProp")
        spec = _spec_with(
            shape=shape,
            slots=[unbound_slot("how", "crsPurpose", "purpose")],
            stat_vars=[
                _stat_var("sv/1", slot_values=[_slot_value("crsPurpose", "crsPurpose/93010")])
            ],
        )

        assert project_spec_dcids(spec) == {"PopType", "measuredProp", "measuredValue"}

    def test_expansion_added_codes_project_nothing(self):
        shape = shape_five_tuple("PopType", "measuredProp")
        slot = value_slot("how", "crsPurpose", "purpose", "crsPurpose/930").model_copy(
            update={"expansion": Expansion(added=[gref("crsPurpose/93010")])}
        )
        spec = _spec_with(shape=shape, slots=[slot], stat_vars=[_stat_var("sv/1")])

        dcids = project_spec_dcids(spec)

        assert "crsPurpose/930" in dcids
        assert "crsPurpose/93010" not in dcids

    @pytest.mark.parametrize(
        ("facets", "projected"),
        [
            pytest.param(
                {"unit": ResolvedFacet(basis="bound", ref="fact/1", label="Fact")},
                set(),
                id="unit-ref-adds-nothing",
            ),
            pytest.param(
                {"measurementMethod": ResolvedFacet(basis="fallback", ref="method/1")},
                set(),
                id="method-ref-adds-nothing",
            ),
            pytest.param(
                {
                    "provenance": ResolvedFacet(
                        basis="fallback", ref="provenance/OECDCreditorReportingSystem"
                    ),
                    "unit": ResolvedFacet(basis="bound", ref="fact/1"),
                },
                {"provenance/OECDCreditorReportingSystem"},
                id="selected-provenance-adds-its-dcid",
            ),
            pytest.param(
                {"provenance": ResolvedFacet(basis="fallback", selection="per_series")},
                set(),
                id="per-series-provenance-adds-nothing",
            ),
        ],
    )
    def test_facets_add_only_the_selected_provenance_and_entities_add_nothing(
        self, facets, projected
    ):
        shape = shape_five_tuple("PopType", "measuredProp")
        spec = _spec_with(
            shape=shape,
            stat_vars=[_stat_var("sv/1")],
            entities=[_entity("entity/1")],
            facets=facets,
        )

        assert project_spec_dcids(spec) == {"PopType", "measuredProp", "measuredValue"} | projected

    def test_reads_a_twins_own_positions_too(self):
        """Called directly on a twin, the function returns that twin's own
        positions. assemble.py's with_skills calls it per twin; this function
        does not recurse into equivalent_readings."""
        twin_shape = shape_five_tuple("TwinPop", "twinMeasuredProp")
        twin = _spec_with(shape=twin_shape, stat_vars=[_stat_var("sv/twin")])

        assert project_spec_dcids(twin) == {
            "TwinPop",
            "twinMeasuredProp",
            "measuredValue",
        }


# ---------------------------------------------------------------------------
# The attach: every position with_sentence reaches, guarded on include_skills
# ---------------------------------------------------------------------------


class TestSkillsAttach:
    def test_definite_spec_gets_skills(self):
        shape = shape_five_tuple("PopType", "measuredProp")
        spec = _spec_with(shape=shape, stat_vars=[_stat_var("sv/1")])
        index = _index(
            {
                "PopType": (
                    SkillMatch(skill_id="skill-a", hop=0, breadth=1, breadth_truncated=False),
                )
            },
            ["skill-a"],
        )

        region = _region(spec, "population")
        response = assemble_region(
            region,
            **_assemble_kwargs(include_skills=True, skills_index=index),
        )

        part = response.parts[0]
        assert isinstance(part, DefinitePart)
        assert part.spec.resolved.skills == ["skill-a"]
        assert response.skills is not None
        assert set(response.skills) == {"skill-a"}
        skill = response.skills["skill-a"]
        assert skill.id == "skill-a"
        assert skill.name == "Name"
        assert skill.description == "Description"
        assert skill.content_version == index.skills["skill-a"].front_matter.content_version
        assert skill.content_version.startswith("sha256:")

    def test_candidates_sharing_one_skill_carry_it_once(self):
        shape = shape_five_tuple("PopType", "measuredProp")
        spec_a = _spec_with(shape=shape, stat_vars=[_stat_var("sv/a")])
        spec_b = _spec_with(shape=shape, stat_vars=[_stat_var("sv/b")])
        index = _index(
            {
                "PopType": (
                    SkillMatch(skill_id="skill-a", hop=0, breadth=1, breadth_truncated=False),
                )
            },
            ["skill-a"],
        )

        region = RegionResult(
            variable_text="population",
            status="candidates",
            specs=(spec_a, spec_b),
            no_data_reason=None,
            warnings=(),
            timing_by_step={},
        )
        response = assemble_region(
            region,
            **_assemble_kwargs(include_skills=True, skills_index=index),
        )

        part = response.parts[0]
        assert isinstance(part, CandidatesPart)
        for candidate in part.candidates.specs:
            assert candidate.resolved.skills == ["skill-a"]
        assert response.skills is not None
        assert list(response.skills) == ["skill-a"]

    def test_nearest_real_and_flagged_match_get_skills(self):
        shape = shape_five_tuple("PopType", "measuredProp")
        near = _nearest_real(shape, sv_dcid="sv/near")
        flagged = _spec_with(shape=shape, stat_vars=[_stat_var("sv/flagged")])
        index = _index(
            {
                "PopType": (
                    SkillMatch(skill_id="skill-a", hop=0, breadth=1, breadth_truncated=False),
                )
            },
            ["skill-a"],
        )

        part = build_no_data_part(
            "no_observations",
            variable_text="population",
            warnings=[],
            nearest_real=[near],
            flagged_match=flagged,
            gate="no_member_has_data_at_place",
            include_skills=True,
            skills_index=index,
        )

        assert part.no_data.nearest_real is not None
        assert part.no_data.nearest_real[0].resolved.skills == ["skill-a"]
        assert part.no_data.flagged_match is not None
        assert part.no_data.flagged_match.resolved.skills == ["skill-a"]

    def test_alternate_match_gets_skills(self):
        shape = shape_five_tuple("PopType", "measuredProp")
        alternate = _spec_with(shape=shape, stat_vars=[_stat_var("sv/alt")])
        index = _index(
            {
                "PopType": (
                    SkillMatch(skill_id="skill-a", hop=0, breadth=1, breadth_truncated=False),
                )
            },
            ["skill-a"],
        )

        part = build_no_data_part(
            "no_observations",
            variable_text="population",
            warnings=[],
            alternate_match=alternate,
            gate="only_alternate_has_data",
            include_skills=True,
            skills_index=index,
        )

        assert part.no_data.alternate_match is not None
        assert part.no_data.alternate_match.resolved.skills == ["skill-a"]

    def test_twin_gets_its_own_positions_and_parent_rebuilds_equivalent_readings(self):
        parent_shape = shape_five_tuple("ParentPop", "measuredProp")
        twin_shape = shape_five_tuple("TwinPop", "measuredProp")
        twin = _spec_with(shape=twin_shape, stat_vars=[_stat_var("sv/twin")])
        parent = _spec_with(
            shape=parent_shape, stat_vars=[_stat_var("sv/parent")], equivalent_readings=[twin]
        )
        index = _index(
            {
                "ParentPop": (
                    SkillMatch(skill_id="skill-parent", hop=0, breadth=1, breadth_truncated=False),
                ),
                "TwinPop": (
                    SkillMatch(skill_id="skill-twin", hop=0, breadth=1, breadth_truncated=False),
                ),
            },
            ["skill-parent", "skill-twin"],
        )

        region = _region(parent, "population")
        response = assemble_region(
            region,
            **_assemble_kwargs(include_skills=True, skills_index=index),
        )

        part = response.parts[0]
        assert isinstance(part, DefinitePart)
        assert part.spec.resolved.skills == ["skill-parent"]
        assert part.spec.resolved.equivalent_readings is not None
        assert part.spec.resolved.equivalent_readings[0].resolved.skills == ["skill-twin"]
        assert response.skills is not None
        assert set(response.skills) == {"skill-parent", "skill-twin"}


# ---------------------------------------------------------------------------
# Option absent / on-with-no-match: the None vs. [] / {} split
# ---------------------------------------------------------------------------


class TestSkillsFieldTwoValueContract:
    def test_option_absent_skills_none_everywhere(self):
        spec = _spec_with(
            shape=shape_five_tuple("PopType", "measuredProp"), stat_vars=[_stat_var("sv/1")]
        )
        region = _region(spec, "population")

        response = assemble_region(region, **_assemble_kwargs())

        part = response.parts[0]
        assert isinstance(part, DefinitePart)
        assert part.spec.resolved.skills is None
        assert response.skills is None

    def test_option_on_with_no_match_is_empty_list_and_empty_map(self):
        spec = _spec_with(
            shape=shape_five_tuple("PopType", "measuredProp"), stat_vars=[_stat_var("sv/1")]
        )
        index = _index({}, [])

        region = _region(spec, "population")
        response = assemble_region(
            region, **_assemble_kwargs(include_skills=True, skills_index=index)
        )

        part = response.parts[0]
        assert isinstance(part, DefinitePart)
        assert part.spec.resolved.skills == []
        assert response.skills == {}

    def test_option_on_with_a_spec_less_response_is_empty_map_not_none(self):
        """A no_data part with no nearest_real/flagged_match/alternate_match
        carries no Spec at all -- the map must still read {} rather than
        inferring "not requested" from the absence of any resolved.skills."""
        index = _index({}, [])
        part = build_no_data_part(
            "variable_not_resolved",
            variable_text="unemployment rate",
            warnings=[],
            include_skills=True,
            skills_index=index,
        )

        response = make_response(
            QueryEcho(
                entry_path="raw_text",
                raw_query="unemployment rate",
                normalized_query=None,
                extract_skipped=False,
            ),
            make_diagnostics("test", [], {}, 0),
            [part],
            include_skills=True,
            skills_index=index,
        )

        assert response.skills == {}


# ---------------------------------------------------------------------------
# match_dcids' own hop-ascending order survives onto resolved.skills unchanged
# ---------------------------------------------------------------------------


class TestMatchOrderSurvivesOntoTheWire:
    def test_hop_ascending_order_is_preserved(self):
        shape = shape_five_tuple("PopType", "measuredProp")
        spec = _spec_with(shape=shape, stat_vars=[_stat_var("sv/1")])
        index = _index(
            {
                "PopType": (
                    SkillMatch(skill_id="far", hop=3, breadth=1, breadth_truncated=True),
                    SkillMatch(skill_id="near", hop=0, breadth=1, breadth_truncated=False),
                )
            },
            ["far", "near"],
        )

        region = _region(spec, "population")
        response = assemble_region(
            region, **_assemble_kwargs(include_skills=True, skills_index=index)
        )

        part = response.parts[0]
        assert isinstance(part, DefinitePart)
        assert part.spec.resolved.skills == ["near", "far"]


# ---------------------------------------------------------------------------
# The envelope's own skills member on the published response schema
# ---------------------------------------------------------------------------


def _schema_client() -> TestClient:
    app = search_app(
        graph=FakeGraph(),
        resolver=make_resolver(FakeGraph()),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
    )
    return TestClient(app, raise_server_exceptions=True)


def test_skills_field_appears_in_the_published_response_schema():
    resp = _schema_client().get("/api/search/schema")
    assert resp.status_code == 200
    resolution_schema = resp.json()["response"]["$defs"]["Resolution"]
    assert "skills" in resolution_schema["properties"]
