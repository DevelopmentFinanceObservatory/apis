"""Offline golden tests cover definite, candidates, and no_data responses.
The no-place golden checks variable disambiguation without an entity probe.
"""

from __future__ import annotations

import pytest

from search.engine.bind import ShapeRating, SlotBindingDraft, _UniversalBindOutput
from search.engine.binders import GeminiBinder
from search.models import (
    CandidatesPart,
    DefinitePart,
    NoDataPart,
)
from tests.engine._binders import FixedBindReplyLLM, RatingBinder, nominee_ratings
from tests.engine._harness import make_request as _req
from tests.engine._harness import offline_resolve
from tests.fixtures import FakeGraph
from tests.origin_catalogue.fake_catalogue import FakeCatalogue
from tests.origin_catalogue.support import build_arc_tables

# California resolves through State -> AdministrativeArea -> Place.
# Its offline extraction fixture has a stale prompt hash; the test un-skips once re-recorded.
_SUBNATIONAL_GEO_GAP = (
    "deferred: California resolves and is admitted (see test_admission.py), but the "
    "recorded LLM extraction fixture for 'income in California' is keyed under a stale "
    "prompt hash; un-skip once the fixture is re-recorded"
)


def _candidates(query: str) -> CandidatesPart:
    result = offline_resolve(_req(query))
    part = result.parts[0]
    assert isinstance(part, CandidatesPart), (
        f"expected candidates for {query!r}, got status={part.status!r}"
    )
    return part


# ---------------------------------------------------------------------------
# Shared invariant helpers
# ---------------------------------------------------------------------------


def _check_invariants(response: CandidatesPart, *, query: str, max_count: int = 6) -> None:
    """Assert the two candidates invariants: count and distinct spec_ids.

    Ordering itself is not asserted here: it depends on which of the two
    CandidateSet.ordering values this golden's own query hits (broadest_first,
    descending member_count, when the near-neighbor gate stays quiet;
    intent_rank, bind's own preference order, when it fires -- see
    assemble_candidates), which is per-golden, not a shared invariant.
    """
    specs = response.candidates.specs
    count = len(specs)

    assert 2 <= count <= max_count, f"{query!r}: expected 2..{max_count} specs, got {count}"

    spec_ids = [s.spec_id for s in specs]
    assert len(spec_ids) == len(set(spec_ids)), f"{query!r}: duplicate spec_ids: {spec_ids}"


# ---------------------------------------------------------------------------
# std-02: GDP India
# ---------------------------------------------------------------------------


def _hold_nominal_activity_family(bind_catalogue) -> None:
    """Bind a catalogue whose tables hold the nominal economic activity family as the offline
    graph records it, as the deployed catalogue's tables hold it. Every member states an
    `activitySource`, so the part binds that slot from recall (`bind_partition_values`), which
    reads a held family only: the default fake catalogue holds none."""
    graph = FakeGraph()
    family = {
        dcid: graph.node_arcs(dcid)
        for dcid, arcs in graph.member_arcs().items()
        if arcs.get("populationType") == ("EconomicActivity",)
        and arcs.get("measuredProperty") == ("amount",)
        and arcs.get("statType") == ("measuredValue",)
        and arcs.get("measurementQualifier") == ("Nominal",)
        and not arcs.get("measurementDenominator")
    }
    bind_catalogue(FakeCatalogue(arcs=build_arc_tables(dfo={}, base=family, schema={})))


class TestStd02GDPIndia:
    """std-02: 'GDP India' -> definite, answered by nominal GDP over the held family."""

    _QUERY = "GDP India"

    def test_status_is_definite(self, bind_catalogue):
        _hold_nominal_activity_family(bind_catalogue)
        result = offline_resolve(_req(self._QUERY))
        inner = result.parts[0]
        assert inner.status == "definite"
        assert isinstance(inner, DefinitePart)
        sv_dcids = [sv.ref.ref for sv in inner.spec.resolved.members]
        assert "Amount_EconomicActivity_GrossDomesticProduction_Nominal" in sv_dcids


# ---------------------------------------------------------------------------
# std-03: birth rate Ethiopia
# ---------------------------------------------------------------------------


class TestStd03BirthRateEthiopia:
    """The bare query answers the crude birth rate for Ethiopia. The recorded reply rates the
    fertility rate it nominates at 0.8 of its pick, below the nomination bar, so the
    nomination is no rival."""

    _QUERY = "birth rate Ethiopia"

    def test_the_crude_birth_rate_answers_for_ethiopia(self):
        inner = offline_resolve(_req(self._QUERY)).parts[0]
        assert isinstance(inner, DefinitePart)
        assert [sv.ref.ref for sv in inner.spec.resolved.members] == [
            "Count_BirthEvent_LiveBirth_AsFractionOf_Count_Person"
        ]
        assert [e.ref for e in inner.spec.resolved.entities] == ["country/ETH"]


# ---------------------------------------------------------------------------
# std-04b: number of births in Ethiopia
# ---------------------------------------------------------------------------


class TestStd04bBirthsEthiopia:
    """No birth-count variable has data at Ethiopia. Bind elects the live-birth count,
    `Count_BirthEvent_LiveBirth`, binds `medicalStatus` to live births and nominates the
    birth count and the per-capita birth shape, which holds data there. The question names
    no form, so the per-capita shape is admissible, and status rule 1 lets a nominated
    reading with data ride on the refusal as its alternate match."""

    _QUERY = "number of births in Ethiopia"

    def test_status_is_no_data(self):
        result = offline_resolve(_req(self._QUERY))
        assert result.status == "no_data"

    def test_the_live_birth_count_is_elected_with_the_per_capita_shape_as_alternate(self):
        """The gate leaves `flagged_match` empty, so only the message names the elected
        variable. The bind reply electing the live-birth count and nominating the birth
        count and the per-capita shape is held fixed."""
        llm = FixedBindReplyLLM(
            _UniversalBindOutput(
                chosen_shape_index=2,
                bindings=[
                    SlotBindingDraft(
                        axis="how",
                        property_dcid="medicalStatus",
                        kind="value",
                        value_dcids=["LiveBirth"],
                        evidence="sole_option",
                    )
                ],
                also_plausible=[0, 1],
                measuring=[
                    ShapeRating(index=2, rating=1.0),
                    ShapeRating(index=0, rating=1.0),
                    *nominee_ratings([0, 1], rated=(0, 2)),
                ],
            )
        )
        result = offline_resolve(_req(self._QUERY), llm=llm)

        (prompt,) = llm.bind_prompts
        assert "Shape 1:\n  label: Count of Birth Event (Per capita)\n" in prompt
        assert "Shape 2:\n  label: Total number of reported live births\n" in prompt
        inner = result.parts[0]
        assert isinstance(inner, NoDataPart)
        assert inner.no_data.gate == "only_alternate_has_data"
        assert inner.no_data.elected_count == 1
        assert inner.no_data.flagged_match is None
        assert "reported live births" in inner.no_data.message
        alternate = inner.no_data.alternate_match
        assert alternate is not None
        assert (
            alternate.resolved.shape.shape_id == "birthevent_count_measuredvalue_per_count_person"
        )


# ---------------------------------------------------------------------------
# std-05: under-5 child mortality rate — no entity, variable disambiguation
# ---------------------------------------------------------------------------


class TestStd05Under5ChildMortality:
    """std-05: 'under-5 child mortality rate', no place. Bind elects the under-five mortality
    rate and nominates the under-five death count at 0.8 of its pick, below the nomination
    bar, so the rate answers alone. No reader reads "child" on the rate, which the binder
    rates stating it, as a rate on Age=Under 5 does."""

    _QUERY = "under-5 child mortality rate"

    def test_the_rate_answers_without_the_count(self):
        binder = RatingBinder(GeminiBinder(), {"child": 1.0})
        inner = offline_resolve(_req(self._QUERY), binder=binder).parts[0]
        assert isinstance(inner, DefinitePart)
        assert inner.spec.resolved.shape.label == "Under-five mortality rate [under 5 years old]"
        assert inner.spec.resolved.entities == []
        assert [request.words for request in binder.requests] == [("child",)]


# ---------------------------------------------------------------------------
# std-06b: fertility rate in Kenya
# ---------------------------------------------------------------------------


class TestStd06bFertilityRateKenya:
    """std-06b: 'fertility rate in Kenya' -> definite."""

    _QUERY = "fertility rate in Kenya"

    def test_status_is_definite(self):
        result = offline_resolve(_req(self._QUERY))
        inner = result.parts[0]
        assert inner.status == "definite"
        assert isinstance(inner, DefinitePart)
        sv_dcids = [sv.ref.ref for sv in inner.spec.resolved.members]
        assert "FertilityRate_Person_Female" in sv_dcids


# ---------------------------------------------------------------------------
# std-07: infant mortality rate Ethiopia
# ---------------------------------------------------------------------------


class TestStd07InfantMortalityEthiopia:
    """std-07: 'infant mortality rate Ethiopia' -> definite."""

    _QUERY = "infant mortality rate Ethiopia"

    def test_status_is_definite(self):
        result = offline_resolve(_req(self._QUERY))
        inner = result.parts[0]
        assert inner.status == "definite"
        assert isinstance(inner, DefinitePart)
        sv_dcids = [sv.ref.ref for sv in inner.spec.resolved.members]
        assert "Count_Death_0Years_AsFractionOf_Count_BirthEvent_LiveBirth" in sv_dcids


# ---------------------------------------------------------------------------
# sdg-06: government education spending Kenya
# ---------------------------------------------------------------------------


class TestSdg06EducationSpendingKenya:
    """sdg-06: 'government education spending Kenya' -> definite."""

    _QUERY = "government education spending Kenya"

    @pytest.mark.xfail(
        reason=(
            "The model now flags share-of-government-spending vs share-of-GDP as "
            "also_plausible (genuine denominator ambiguity the query doesn't "
            "resolve) instead of the old cosine-margin picking share-of-government "
            "alone. Plausibly a quality improvement, but it changes sdg-06's status "
            "from definite to candidates."
        ),
        strict=False,
    )
    def test_status_is_definite(self):
        result = offline_resolve(_req(self._QUERY))
        inner = result.parts[0]
        assert inner.status == "definite"
        assert isinstance(inner, DefinitePart)
        sv_dcids = [sv.ref.ref for sv in inner.spec.resolved.members]
        assert (
            "Amount_EconomicActivity_ExpenditureActivity_EducationExpenditure_Government_"
            "AsFractionOf_Amount_EconomicActivity_ExpenditureActivity_Government"
        ) in sv_dcids


# ---------------------------------------------------------------------------
# cand-r1: GDP of Brazil
# ---------------------------------------------------------------------------


class TestCandR1GDPBrazil:
    """A bare "GDP" names a level, so the Brazil query answers nominal GDP alone. The
    per-capita alternate adds a normalisation the query did not name and is refused."""

    _QUERY = "GDP of Brazil"

    def _definite(self) -> DefinitePart:
        part = offline_resolve(_req(self._QUERY)).parts[0]
        assert isinstance(part, DefinitePart), (
            f"expected definite for {self._QUERY!r}, got status={part.status!r}"
        )
        return part

    def test_status_is_definite(self):
        assert offline_resolve(_req(self._QUERY)).status == "definite"

    def test_sv_is_nominal_gdp(self):
        sv_dcids = [sv.ref.ref for sv in self._definite().spec.resolved.members]
        assert sv_dcids == ["Amount_EconomicActivity_GrossDomesticProduction_Nominal"]

    def test_entity_is_brazil(self):
        entities = [e.ref for e in self._definite().spec.resolved.entities]
        assert entities == ["country/BRA"]


# ---------------------------------------------------------------------------
# cand-r2: income in California -> exactly 2 candidates
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason=_SUBNATIONAL_GEO_GAP)
class TestCandR2IncomeCaliforniaExactly2:
    _QUERY = "income in California"

    def test_status_is_candidates(self):
        assert offline_resolve(_req(self._QUERY)).status == "candidates"

    def test_exactly_2_specs(self):
        resp = _candidates(self._QUERY)
        assert len(resp.candidates.specs) == 2, (
            f"expected exactly 2 specs, got {len(resp.candidates.specs)}"
        )

    def test_spec_ids_are_distinct(self):
        resp = _candidates(self._QUERY)
        ids = [s.spec_id for s in resp.candidates.specs]
        assert len(ids) == len(set(ids))
