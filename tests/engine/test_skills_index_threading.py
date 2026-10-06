"""Tests `include_skills` and skills-index warnings at the response envelope.

Hand-built results cover assembly; real indexes cover budget and rejected-resource cases.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import skills_index.build as build_module
from search.engine.assemble import (
    SKILLS_INDEX_INCOMPLETE,
    SKILLS_INDEX_UNAVAILABLE,
    SKILLS_NOT_MATCHABLE,
    build_no_data_part,
    build_unsupported_part,
    skills_index_incomplete_warning,
    skills_index_unavailable_warning,
)
from search.engine.conjoin import assemble_region
from search.engine.envelope import now_ms
from search.engine.region import RegionResult
from search.models import (
    CandidatesPart,
    DefinitePart,
    NoDataPart,
    ResolveOptions,
    Unsupported,
    UnsupportedPart,
)
from skills_index import SkillsIndex
from tests.engine._builders import conjoin_spec as _spec
from tests.engine._conjoin_fixtures import _PERSON_SHAPE, _region
from tests.skills_index.conftest import FakeSkillsGraph, skill_node

_EMPTY_INDEX = SkillsIndex(skills={}, node_to_skills={}, issues=(), may_omit_published_skills=False)
_INCOMPLETE_INDEX = SkillsIndex(
    skills={}, node_to_skills={}, issues=(), may_omit_published_skills=True
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
# ResolveOptions.include_skills round-trip
# ---------------------------------------------------------------------------


class TestIncludeSkillsOption:
    def test_absent_defaults_none(self):
        assert ResolveOptions().include_skills is None

    def test_false_round_trips(self):
        assert ResolveOptions(include_skills=False).include_skills is False

    def test_true_round_trips(self):
        assert ResolveOptions(include_skills=True).include_skills is True

    def test_misspelled_field_rejected(self):
        with pytest.raises(ValidationError):
            ResolveOptions(include_skil=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# skills_index_unavailable_warning (assemble.py's own pure helper)
# ---------------------------------------------------------------------------


class TestSkillsIndexUnavailableWarningHelper:
    def test_none_when_flag_false(self):
        assert skills_index_unavailable_warning(include_skills=False, skills_index=None) is None

    def test_none_when_snapshot_present(self):
        assert (
            skills_index_unavailable_warning(include_skills=True, skills_index=_EMPTY_INDEX) is None
        )

    def test_present_when_flag_true_and_snapshot_none(self):
        warning = skills_index_unavailable_warning(include_skills=True, skills_index=None)
        assert warning is not None
        assert warning.code == SKILLS_INDEX_UNAVAILABLE
        assert warning.severity == "warn"


# ---------------------------------------------------------------------------
# skills_index_unavailable on the envelope, via assemble_region
# ---------------------------------------------------------------------------


class TestSkillsIndexUnavailableOnEnvelope:
    def test_appears_when_option_set_and_snapshot_none(self):
        spec = _spec(_PERSON_SHAPE, "any")
        region = _region(spec, "population")
        response = assemble_region(
            region, **_assemble_kwargs(include_skills=True, skills_index=None)
        )
        codes = [w.code for w in response.diagnostics.warnings]
        assert SKILLS_INDEX_UNAVAILABLE in codes

    def test_absent_when_option_absent(self):
        spec = _spec(_PERSON_SHAPE, "any")
        region = _region(spec, "population")
        response = assemble_region(region, **_assemble_kwargs())
        codes = [w.code for w in response.diagnostics.warnings]
        assert SKILLS_INDEX_UNAVAILABLE not in codes

    def test_absent_when_snapshot_already_available(self):
        spec = _spec(_PERSON_SHAPE, "any")
        region = _region(spec, "population")
        response = assemble_region(
            region, **_assemble_kwargs(include_skills=True, skills_index=_EMPTY_INDEX)
        )
        codes = [w.code for w in response.diagnostics.warnings]
        assert SKILLS_INDEX_UNAVAILABLE not in codes


# ---------------------------------------------------------------------------
# skills_index_incomplete_warning (assemble.py's own pure helper)
# ---------------------------------------------------------------------------


class TestSkillsIndexIncompleteWarningHelper:
    def test_none_when_flag_false(self):
        assert (
            skills_index_incomplete_warning(include_skills=False, skills_index=_INCOMPLETE_INDEX)
            is None
        )

    def test_none_when_snapshot_absent(self):
        # skills_index_unavailable is the one that fires here, not this code.
        assert skills_index_incomplete_warning(include_skills=True, skills_index=None) is None

    def test_none_when_snapshot_complete(self):
        assert (
            skills_index_incomplete_warning(include_skills=True, skills_index=_EMPTY_INDEX) is None
        )

    def test_present_when_flag_true_and_snapshot_incomplete(self):
        warning = skills_index_incomplete_warning(
            include_skills=True, skills_index=_INCOMPLETE_INDEX
        )
        assert warning is not None
        assert warning.code == SKILLS_INDEX_INCOMPLETE
        assert warning.severity == "warn"


# ---------------------------------------------------------------------------
# skills_index_incomplete on the envelope, via assemble_region: the flag set
# by hand and by a budget-exhausted corpus reaches the envelope, and a
# rejected resource leaves it false. tests/skills_index/test_build.py covers
# the flag itself; this covers only that it reaches the response envelope.
# ---------------------------------------------------------------------------


class TestSkillsIndexIncompleteOnEnvelope:
    def test_appears_when_option_set_and_snapshot_incomplete(self):
        spec = _spec(_PERSON_SHAPE, "any")
        region = _region(spec, "population")
        response = assemble_region(
            region, **_assemble_kwargs(include_skills=True, skills_index=_INCOMPLETE_INDEX)
        )
        codes = [w.code for w in response.diagnostics.warnings]
        assert SKILLS_INDEX_INCOMPLETE in codes

    def test_absent_when_option_absent(self):
        spec = _spec(_PERSON_SHAPE, "any")
        region = _region(spec, "population")
        response = assemble_region(region, **_assemble_kwargs(skills_index=_INCOMPLETE_INDEX))
        codes = [w.code for w in response.diagnostics.warnings]
        assert SKILLS_INDEX_INCOMPLETE not in codes

    def test_absent_when_snapshot_complete(self):
        spec = _spec(_PERSON_SHAPE, "any")
        region = _region(spec, "population")
        response = assemble_region(
            region, **_assemble_kwargs(include_skills=True, skills_index=_EMPTY_INDEX)
        )
        codes = [w.code for w in response.diagnostics.warnings]
        assert SKILLS_INDEX_INCOMPLETE not in codes

    def test_absent_when_snapshot_none(self):
        """skills_index_unavailable fires instead; this code never does
        without an available snapshot to read the flag off of."""
        spec = _spec(_PERSON_SHAPE, "any")
        region = _region(spec, "population")
        response = assemble_region(
            region, **_assemble_kwargs(include_skills=True, skills_index=None)
        )
        codes = [w.code for w in response.diagnostics.warnings]
        assert SKILLS_INDEX_INCOMPLETE not in codes
        assert SKILLS_INDEX_UNAVAILABLE in codes

    def test_budget_exhausted_corpus_triggers_the_warning(self, monkeypatch):
        """A valid skill left out under skills_corpus_budget_exhausted is one
        search cannot match, so a real index built off a corpus that trips the
        budget reaches the envelope with skills_index_incomplete."""
        monkeypatch.setattr(build_module, "_FRONT_MATTER_CHUNK_SIZE", 1)
        monkeypatch.setattr(build_module, "SKILLS_CORPUS_CONTENT_BUDGET_BYTES", 5)
        graph = FakeSkillsGraph(
            subjects={("AgentSkill", "typeOf"): ["skill/skill-a", "skill/skill-b"]},
            arcs={
                "skill/skill-a": skill_node("skill-a"),
                "skill/skill-b": skill_node("skill-b"),
            },
        )
        index = build_module.build_index(graph=graph)
        assert any(issue.code == "skills_corpus_budget_exhausted" for issue in index.issues)
        assert index.may_omit_published_skills is True

        spec = _spec(_PERSON_SHAPE, "any")
        region = _region(spec, "population")
        response = assemble_region(
            region, **_assemble_kwargs(include_skills=True, skills_index=index)
        )
        codes = [w.code for w in response.diagnostics.warnings]
        assert SKILLS_INDEX_INCOMPLETE in codes

    def test_rejected_resource_does_not_trigger_the_warning(self):
        """A rejected resource omits no skill either, so a real index
        built off a corpus with one must still reach the envelope with no
        skills_index_incomplete warning."""
        graph = FakeSkillsGraph(
            subjects={("AgentSkill", "typeOf"): ["skill/skill-a"]},
            arcs={
                "skill/skill-a": skill_node("skill-a")
                | {"agentSkillReferences": ("skill/skill-a/notes",)},
                "skill/skill-a/notes": {"name": ("references/notes.md",)},
            },
        )
        index = build_module.build_index(graph=graph)
        assert any(issue.code == "skill_resource_rejected" for issue in index.issues)
        assert index.may_omit_published_skills is False

        spec = _spec(_PERSON_SHAPE, "any")
        region = _region(spec, "population")
        response = assemble_region(
            region, **_assemble_kwargs(include_skills=True, skills_index=index)
        )
        codes = [w.code for w in response.diagnostics.warnings]
        assert SKILLS_INDEX_INCOMPLETE not in codes

    def test_skill_excluded_by_a_failed_reach_read_triggers_the_warning(self):
        """A reach read that stops at its page budget excludes the skill under
        skill_build_unconfirmed, so search may miss a skill that applies and
        the envelope carries skills_index_incomplete."""
        graph = FakeSkillsGraph(
            subjects={
                ("AgentSkill", "typeOf"): ["skill/skill-a"],
                ("Person", "populationType"): ["sv-1"],
            },
            budget_exhausted={("Person", "isPartOf")},
            arcs={"skill/skill-a": skill_node("skill-a") | {"skillAppliesTo": ("Person",)}},
        )
        index = build_module.build_index(graph=graph)
        assert index.skills["skill-a"].reach.excluded is True

        spec = _spec(_PERSON_SHAPE, "any")
        region = _region(spec, "population")
        response = assemble_region(
            region, **_assemble_kwargs(include_skills=True, skills_index=index)
        )
        codes = [w.code for w in response.diagnostics.warnings]
        assert SKILLS_INDEX_INCOMPLETE in codes


# ---------------------------------------------------------------------------
# skills_not_matchable, part-scoped, on the two spec-less builders
# ---------------------------------------------------------------------------


class TestSkillsNotMatchableOnNoDataPart:
    def test_present_when_flag_set(self):
        part = build_no_data_part(
            "variable_not_resolved",
            variable_text="foo",
            warnings=[],
            include_skills=True,
        )
        codes = [w.code for w in part.warnings]
        assert SKILLS_NOT_MATCHABLE in codes

    def test_absent_when_flag_off(self):
        part = build_no_data_part(
            "variable_not_resolved",
            variable_text="foo",
            warnings=[],
            include_skills=False,
        )
        codes = [w.code for w in part.warnings]
        assert SKILLS_NOT_MATCHABLE not in codes

    def test_absent_by_default(self):
        part = build_no_data_part(
            "variable_not_resolved",
            variable_text="foo",
            warnings=[],
        )
        codes = [w.code for w in part.warnings]
        assert SKILLS_NOT_MATCHABLE not in codes


class TestSkillsNotMatchableOnUnsupportedPart:
    def test_present_when_flag_set(self):
        part = build_unsupported_part(
            "facet_binding_unsupported",
            "/identity/facets/0",
            "detail",
            {"schema_version": 1},
            variable_text="foo",
            warnings=[],
            include_skills=True,
        )
        codes = [w.code for w in part.warnings]
        assert SKILLS_NOT_MATCHABLE in codes

    def test_absent_when_flag_off(self):
        part = build_unsupported_part(
            "facet_binding_unsupported",
            "/identity/facets/0",
            "detail",
            {"schema_version": 1},
            variable_text="foo",
            warnings=[],
            include_skills=False,
        )
        codes = [w.code for w in part.warnings]
        assert SKILLS_NOT_MATCHABLE not in codes


# ---------------------------------------------------------------------------
# skills_not_matchable through the real assemble_region seam (RegionResult in,
# Part out), and its absence on a spec-bearing part under the same flag.
# ---------------------------------------------------------------------------


class TestSkillsNotMatchableThroughAssembleRegion:
    def test_no_data_part_carries_skills_not_matchable(self):
        region = _region(
            None, "population", status="no_data", no_data_reason="variable_not_resolved"
        )
        response = assemble_region(region, **_assemble_kwargs(include_skills=True))
        part = response.parts[0]
        assert isinstance(part, NoDataPart)
        codes = [w.code for w in part.warnings]
        assert SKILLS_NOT_MATCHABLE in codes

    def test_no_data_part_omits_skills_not_matchable_when_flag_off(self):
        region = _region(
            None, "population", status="no_data", no_data_reason="variable_not_resolved"
        )
        response = assemble_region(region, **_assemble_kwargs())
        part = response.parts[0]
        assert isinstance(part, NoDataPart)
        codes = [w.code for w in part.warnings]
        assert SKILLS_NOT_MATCHABLE not in codes

    def test_unsupported_part_carries_skills_not_matchable(self):
        region = RegionResult(
            variable_text="foo",
            status="unsupported",
            specs=(),
            no_data_reason=None,
            warnings=(),
            timing_by_step={},
            unsupported=Unsupported(
                code="facet_binding_unsupported",
                pointer="/identity/facets/0",
                detail="detail",
                definition={"schema_version": 1},
            ),
        )
        response = assemble_region(region, **_assemble_kwargs(include_skills=True))
        part = response.parts[0]
        assert isinstance(part, UnsupportedPart)
        codes = [w.code for w in part.warnings]
        assert SKILLS_NOT_MATCHABLE in codes

    def test_definite_part_never_carries_skills_not_matchable(self):
        """A definite part always answers with a spec, so it is never a
        skills_not_matchable candidate regardless of the flag."""
        spec = _spec(_PERSON_SHAPE, "any")
        region = _region(spec, "population")
        response = assemble_region(region, **_assemble_kwargs(include_skills=True))
        part = response.parts[0]
        assert isinstance(part, DefinitePart)
        codes = [w.code for w in part.warnings]
        assert SKILLS_NOT_MATCHABLE not in codes

    def test_candidates_part_never_carries_skills_not_matchable(self):
        """A candidates part's specs each carry their own bound shape, so it
        is never a skills_not_matchable candidate either."""
        spec_a = _spec(_PERSON_SHAPE, "any", sv_dcid="sv_a")
        spec_b = _spec(_PERSON_SHAPE, "other", sv_dcid="sv_b")
        region = RegionResult(
            variable_text="population",
            status="candidates",
            specs=(spec_a, spec_b),
            no_data_reason=None,
            warnings=(),
            timing_by_step={},
        )
        response = assemble_region(region, **_assemble_kwargs(include_skills=True))
        part = response.parts[0]
        assert isinstance(part, CandidatesPart)
        codes = [w.code for w in part.warnings]
        assert SKILLS_NOT_MATCHABLE not in codes
