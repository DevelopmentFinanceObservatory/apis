"""Tests query-aware measure collapse in ranking, binding admission, and election warnings.

Real packaged CRS StatVars supply candidates to minimal `ResolveState` fixtures. The index carries
only `Nominal` measurement qualifiers, so tests vary population types and constraint values.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import cast

import pytest

from code_index.shared import get_index, get_measure_index
from code_index.types import MeasureAxis
from entity_resolution import Admission, EntityResolver
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.departures import warning_key
from search.engine.qualifier_match import QualifierMatch
from search.engine.reading_coverage import ObservedYears
from search.engine.region.alternate_guard import leaves_price_basis_unread
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.region.phase_election import _phase_vet_election
from search.engine.region.phase_rank_bind import _measure_query_matches, _phase_prune_and_rank
from search.engine.region.qualifier_readings import match_query_readings
from search.engine.region.state import ResolveState
from search.engine.retrieval.bound_values import _named_free_values
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.heading.departure_fact import DepartureFact
from search.models import TraceDefaultReading
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._index_slots import index_slot_keys
from tests.engine.test_admissibility_table import _node, _resolve_gdp_of_kenya

_ROLES = ("financingProvider", "financingRecipient")

# Real rows share constraints except populationType; `measuredProperty=amount` keeps them in one family.
_FF_NOMINAL_DCID = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode11330_Nominal"
)
_FO_NOMINAL_DCID = (
    "Amount_FinancialObligation_Commitment_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode11330_Nominal"
)
_GRANT_EQ_NOMINAL_DCID = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode51010_Nominal"
)

_REAL_INDEX = get_index()


def _fixture_cross_check() -> None:
    """Confirm fixture dcids and measure values against the packaged index."""
    by_dcid = {r.dcid: r for r in _REAL_INDEX.statvars}
    for dcid in (_FF_NOMINAL_DCID, _FO_NOMINAL_DCID, _GRANT_EQ_NOMINAL_DCID):
        assert dcid in by_dcid, dcid
    assert (
        by_dcid[_FO_NOMINAL_DCID].constraint_properties["financialObligationType"] == "Commitment"
    )
    assert "nettingBasis" not in by_dcid[_GRANT_EQ_NOMINAL_DCID].constraint_properties


def test_fixture_cross_check():
    _fixture_cross_check()


def _shape(shape_id: str, *, pop_type: str, meas_prop: str, meas_qual: str) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid=pop_type,
        meas_prop_dcid=meas_prop,
        stat_type_dcid="measuredValue",
        meas_qual_dcid=meas_qual,
        meas_denom_dcid=None,
        slot_keys=index_slot_keys(
            population_type=pop_type, measured_property=meas_prop, measurement_qualifier=meas_qual
        ),
        declared_roles=_ROLES,
    )


_FF_NOMINAL = _shape(
    "ff_nominal", pop_type="FinancialFlow", meas_prop="amount", meas_qual="Nominal"
)
_FO_NOMINAL = _shape(
    "fo_nominal", pop_type="FinancialObligation", meas_prop="amount", meas_qual="Nominal"
)
_GRANT_EQ_NOMINAL = _shape(
    "grant_eq_nominal", pop_type="FinancialFlow", meas_prop="grantEquivalent", meas_qual="Nominal"
)


def _state(
    *,
    variable: str,
    shapes_found: list[ShapeDraft],
    dropped_qualifier_phrases: list[str] | None = None,
) -> ResolveState:
    """Build a minimal state for ranking and election.

    Kenya and default child places skip graph probes; role-bearing shapes bypass score gating."""
    return ResolveState(
        variable=variable,
        entities=["Kenya"],
        date_request=None,
        detect_query=variable,
        role_query=variable,
        dropped_qualifier_phrases=dropped_qualifier_phrases or [],
        pac=True,
        graph=cast(EngineGraphClient, None),
        resolver=cast(EntityResolver, None),
        admission=cast(Admission, None),
        llm=cast(SupportsLLM, None),
        sdmx=cast(EngineSdmxClient, None),
        deadline=None,
        pre_resolved=None,
        variable_index=0,
        emit=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        unit_text=None,
        materialise_graph=None,
        materialise_sdmx=None,
        timing={},
        warnings=[],
        shapes_found=shapes_found,
        coverage=empty_coverage_snapshot(),
    )


def _run_prune_and_elect(state: ResolveState) -> None:
    """Run ranking and vetting with one selected shape."""
    result = asyncio.run(_phase_prune_and_rank(state))
    assert result is None, "expected _phase_prune_and_rank to fall through"
    assert len(state.capped_shapes) == 1
    state.bind_result = _UniversalBindOutput(
        chosen_shape_index=0, bindings=[], also_plausible=[], ask=None
    )
    result = asyncio.run(_phase_vet_election(state))
    assert result is None, "expected _phase_vet_election to fall through"


def test_commitments_survives_as_financial_obligation_not_financial_flow():
    # `populationType` is absent from `_family_key`, so `commitments` selects FinancialObligation
    # (#743) and counts as satisfied, suppressing an unapplied-qualifier warning (#745).
    state = _state(variable="commitments to Kenya", shapes_found=[_FF_NOMINAL, _FO_NOMINAL])
    _run_prune_and_elect(state)

    assert state.capped_shapes[0].shape_id == "fo_nominal"
    assert not any(w.code == QUERY_QUALIFIER_DROPPED for w in state.warnings)
    assert "commitments" in state.satisfied_measure_phrases


def test_query_naming_no_measure_elects_the_declared_default_unchanged():
    state = _state(variable="aid to Kenya", shapes_found=[_FF_NOMINAL, _FO_NOMINAL])
    _run_prune_and_elect(state)

    assert state.capped_shapes[0].shape_id == "ff_nominal"
    assert not any(w.code == QUERY_QUALIFIER_DROPPED for w in state.warnings)
    assert state.satisfied_measure_phrases == frozenset()


def test_net_grant_equivalent_keeps_grant_equivalent_shape_and_warns():
    # Grant-equivalent rows omit nettingBasis, so `Nominal` remains elected and unmatched `net`
    # warns once without entering `satisfied_measure_phrases`.
    state = _state(variable="net grant equivalent aid to Kenya", shapes_found=[_GRANT_EQ_NOMINAL])
    _run_prune_and_elect(state)

    assert state.capped_shapes[0].shape_id == "grant_eq_nominal"
    dropped = [w for w in state.warnings if w.code == QUERY_QUALIFIER_DROPPED]
    assert len(dropped) == 1
    assert '"net"' in dropped[0].message
    assert "net" not in state.satisfied_measure_phrases
    assert state.departure_facts == {
        warning_key(dropped[0]): DepartureFact(
            code="measure_not_applied", binding_ref="how:nettingBasis", asked="net"
        )
    }


# Basis matches can warn or be admitted, but do not steer collapse.


def test_named_grant_equivalent_against_face_value_shape_warns_once():
    # The basis phrase `grant equivalent` mismatches this shape's `amount` and warns once.
    state = _state(variable="grant equivalent aid to Kenya", shapes_found=[_FF_NOMINAL])
    _run_prune_and_elect(state)

    assert state.capped_shapes[0].shape_id == "ff_nominal"
    dropped = [w for w in state.warnings if w.code == QUERY_QUALIFIER_DROPPED]
    assert len(dropped) == 1
    assert '"grant equivalent"' in dropped[0].message
    assert "grant equivalent" not in state.satisfied_measure_phrases


def test_named_grant_equivalent_against_grant_equivalent_shape_is_satisfied():
    state = _state(variable="grant equivalent aid to Kenya", shapes_found=[_GRANT_EQ_NOMINAL])
    _run_prune_and_elect(state)

    assert state.capped_shapes[0].shape_id == "grant_eq_nominal"
    assert not any(w.code == QUERY_QUALIFIER_DROPPED for w in state.warnings)
    assert "grant equivalent" in state.satisfied_measure_phrases


def test_basis_match_never_steers_the_collapse():
    # `grant equivalent` matches neither candidate; `commitments` elects FinancialObligation.
    # The winner matches the control case without the basis phrase.
    state = _state(
        variable="grant equivalent commitments to Kenya",
        shapes_found=[_FF_NOMINAL, _FO_NOMINAL],
    )
    _run_prune_and_elect(state)

    assert state.capped_shapes[0].shape_id == "fo_nominal"
    dropped = [w for w in state.warnings if w.code == QUERY_QUALIFIER_DROPPED]
    assert len(dropped) == 1
    assert '"grant equivalent"' in dropped[0].message
    assert "grant equivalent" not in state.satisfied_measure_phrases
    assert "commitments" in state.satisfied_measure_phrases


# `entry/raw_text.py` warns for `dropped_qualifier_phrases`; this region warns only for measure
# phrases in `variable`.


def test_dropped_phrase_measure_match_unsatisfied_is_not_warned_by_the_region():
    state = _state(
        variable="grant equivalent aid to Kenya",
        dropped_qualifier_phrases=["net"],
        shapes_found=[_GRANT_EQ_NOMINAL],
    )
    _run_prune_and_elect(state)

    assert state.capped_shapes[0].shape_id == "grant_eq_nominal"
    assert not any(w.code == QUERY_QUALIFIER_DROPPED for w in state.warnings)
    assert "net" not in state.satisfied_measure_phrases


def test_dropped_phrase_basis_match_unsatisfied_is_not_warned_by_the_region():
    state = _state(
        variable="aid to Kenya",
        dropped_qualifier_phrases=["grant equivalent"],
        shapes_found=[_FF_NOMINAL],
    )
    _run_prune_and_elect(state)

    assert state.capped_shapes[0].shape_id == "ff_nominal"
    assert not any(w.code == QUERY_QUALIFIER_DROPPED for w in state.warnings)
    assert "grant equivalent" not in state.satisfied_measure_phrases


def test_two_conflicting_dropped_phrases_are_not_warned_by_the_region():
    # Conflicting `gross` and `net` values populate conflict phrases without region warnings.
    state = _state(
        variable="aid to Kenya",
        dropped_qualifier_phrases=["gross", "net"],
        shapes_found=[_FF_NOMINAL, _FO_NOMINAL],
    )
    _run_prune_and_elect(state)

    assert state.capped_shapes[0].shape_id == "ff_nominal"
    assert state.measure_qualifier_conflict_phrases == frozenset({"gross", "net"})
    assert not any(w.code == QUERY_QUALIFIER_DROPPED for w in state.warnings)


def test_dropped_phrase_measure_match_satisfied_still_suppresses_the_entry_warning():
    state = _state(
        variable="aid to Kenya",
        dropped_qualifier_phrases=["commitments"],
        shapes_found=[_FF_NOMINAL, _FO_NOMINAL],
    )
    _run_prune_and_elect(state)

    assert state.capped_shapes[0].shape_id == "fo_nominal"
    assert not any(w.code == QUERY_QUALIFIER_DROPPED for w in state.warnings)
    assert "commitments" in state.satisfied_measure_phrases


# Tests declared-axis filtering and conflicting matches in `_measure_query_matches`.


def _measure_match(code_system: str, code_dcid: str, matched_text: str) -> QualifierMatch:
    return QualifierMatch(
        code_system=code_system,
        code_dcid=code_dcid,
        label=code_dcid,
        source="label",
        matched_text=matched_text,
    )


def test_measure_query_matches_reports_a_basis_axis_match():
    axes = get_measure_index().axes
    matches = (_measure_match("measuredProperty", "grantEquivalent", "grant equivalent"),)

    named, conflicts = _measure_query_matches(matches, axes)

    assert named == matches
    assert conflicts == frozenset()


def test_measure_query_matches_refuses_two_values_of_one_axis():
    axes = get_measure_index().axes
    matches = (
        _measure_match("nettingBasis", "Gross", "gross"),
        _measure_match("nettingBasis", "Net", "net"),
    )

    named, conflicts = _measure_query_matches(matches, axes)

    assert named == ()
    assert conflicts == frozenset({"gross", "net"})


def test_measure_query_matches_keeps_an_unambiguous_free_axis_match():
    axes = get_measure_index().axes
    matches = (_measure_match("nettingBasis", "Net", "net"),)

    named, conflicts = _measure_query_matches(matches, axes)

    assert named == matches
    assert conflicts == frozenset()


def test_measure_query_matches_refuses_two_values_of_one_basis_axis():
    # Same-axis conflicts apply to `measuredProperty` basis values as to free-axis values.
    axes = get_measure_index().axes
    matches = (
        _measure_match("measuredProperty", "amount", "face value"),
        _measure_match("measuredProperty", "grantEquivalent", "grant equivalent"),
    )

    named, conflicts = _measure_query_matches(matches, axes)

    assert named == ()
    assert conflicts == frozenset({"face value", "grant equivalent"})


def test_measure_query_matches_reports_a_synthetic_constraint_basis_axis_match():
    # CRS lacks a constraint/basis axis; this synthetic axis isolates role filtering from axis kind.
    synthetic_axis = MeasureAxis(axis="syntheticBasis", kind="constraint", role="basis", prefer=())
    axes = (*get_measure_index().axes, synthetic_axis)
    match = _measure_match("syntheticBasis", "SyntheticValue", "synthetic phrase")

    named, conflicts = _measure_query_matches((match,), axes)

    assert named == (match,)
    assert conflicts == frozenset()
    # Reported basis matches stay out of collapse input.
    assert _named_free_values(named, axes) == ()


# A shape the index does not carry is judged on its own qualifier, read through
# `same_reading_as`, so a World Bank real GDP series satisfies "real".


def _gdp_shape(meas_qual: str) -> ShapeDraft:
    return ShapeDraft(
        shape_id=f"gdp_{meas_qual}",
        label=f"GDP {meas_qual}",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=meas_qual,
        meas_denom_dcid=None,
        slot_keys=(),
        representative_score=0.9,
        recalled=True,
    )


@pytest.mark.parametrize(
    ("meas_qual", "warned"),
    [("RealValue", False), ("InflationAdjusted", False), ("Nominal", True)],
)
def test_real_gdp_of_zambia_warns_only_when_the_unindexed_shape_lacks_the_reading(
    meas_qual: str, warned: bool
):
    state = _state(variable="real GDP of Zambia", shapes_found=[_gdp_shape(meas_qual)])
    _run_prune_and_elect(state)

    dropped = [w for w in state.warnings if w.code == QUERY_QUALIFIER_DROPPED]
    assert bool(dropped) is warned
    assert ("real" in state.satisfied_measure_phrases) is not warned


# "PPP" names CRS purpose 25010 and the PPP price basis. The elected shape keeps the reading
# it carries alone, and a shape carrying neither leaves the phrase unapplied.

_CRS_FLOW_WITH_PURPOSE = replace(
    _FF_NOMINAL,
    slot_keys=(SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="purpose"),),
)


def _elect_reading_ppp(variable: str, shape: ShapeDraft) -> ResolveState:
    """Run ranking, record the "PPP" readings the rank phase records, then vet the election."""
    state = _state(variable=variable, shapes_found=[shape])
    assert asyncio.run(_phase_prune_and_rank(state)) is None
    state.query_qualifier_readings, owned = match_query_readings(state.variable)
    state.variable_matched_qualifier_phrases |= owned
    state.bind_result = _UniversalBindOutput(
        chosen_shape_index=0, bindings=[], also_plausible=[], ask=None
    )
    assert asyncio.run(_phase_vet_election(state)) is None
    return state


def test_ppp_reads_the_price_basis_on_a_gdp_ppp_shape():
    state = _elect_reading_ppp("GDP PPP", _gdp_shape("PurchasingPowerParity"))

    assert not any(w.code == QUERY_QUALIFIER_DROPPED for w in state.warnings)
    assert "ppp" in state.satisfied_measure_phrases
    assert state.admitted_qualifiers == ()


def test_ppp_reads_purpose_25010_on_a_crs_flow():
    state = _elect_reading_ppp("PPP aid", _CRS_FLOW_WITH_PURPOSE)

    assert not any(w.code == QUERY_QUALIFIER_DROPPED for w in state.warnings)
    assert [(m.code_system, m.code_dcid) for m in state.admitted_qualifiers] == [
        ("crsPurpose", "CRSPurposeCode25010")
    ]
    assert any(
        b.property_dcid == "crsPurpose" and b.value_dcids == ["CRSPurposeCode25010"]
        for b in state.bind_result.bindings
    )


def test_ppp_is_unapplied_on_a_shape_carrying_neither_reading():
    state = _elect_reading_ppp("GDP PPP", _gdp_shape("Nominal"))

    dropped = [w for w in state.warnings if w.code == QUERY_QUALIFIER_DROPPED]
    assert [w.message for w in dropped] == ['The requested qualifier "ppp" was not applied.']
    assert state.admitted_qualifiers == ()


# The default reading: before bind, a shape the index does not carry loses the members a
# better-ranked price basis covers at the query's places, unless a price-basis phrase of the
# leg went unread.


def _gdp_reading(meas_qual: str, score: float) -> ShapeDraft:
    """A one-member GDP shape under `meas_qual`, observed at Kenya."""
    sv = f"Amount_EconomicActivity_GrossDomesticProduction_{meas_qual}"
    return replace(
        _gdp_shape(meas_qual),
        representative_score=score,
        sv_arc_facts={
            sv: {
                "constraintProperties": {"nodes": [{"dcid": "activitySource"}]},
                "activitySource": {"nodes": [{"dcid": "GrossDomesticProduction"}]},
            }
        },
        member_coverage={sv: {"country/KEN": ObservedYears(first=2015, last=2022, years=None)}},
    )


@pytest.mark.parametrize(
    ("variable", "dropped", "ranked"),
    [
        pytest.param("GDP", [], ["gdp_Nominal"], id="gdp-of-kenya"),
        pytest.param("real GDP", [], ["gdp_RealValue"], id="real-gdp"),
        pytest.param(
            "GDP",
            ["at 2015 base-year prices"],
            ["gdp_InflationAdjusted", "gdp_Nominal", "gdp_RealValue"],
            id="unread-price-basis",
        ),
        pytest.param("GDP", ["by sector"], ["gdp_Nominal"], id="unrelated-dropped-phrase"),
    ],
)
def test_the_default_reading_collapses_unindexed_price_bases_before_bind(
    variable: str, dropped: list[str], ranked: list[str]
):
    state = _state(
        variable=variable,
        dropped_qualifier_phrases=dropped,
        shapes_found=[
            _gdp_reading("InflationAdjusted", 0.9),
            _gdp_reading("Nominal", 0.86),
            _gdp_reading("RealValue", 0.8),
        ],
    )

    assert asyncio.run(_phase_prune_and_rank(state)) is None
    assert sorted(s.shape_id for s in state.ranked_shapes) == ranked
    unread = dropped == ["at 2015 base-year prices"]
    assert state.unread_price_basis is unread
    assert state.default_reading == TraceDefaultReading(
        applied=not unread,
        removed_stat_vars=[
            f"Amount_EconomicActivity_GrossDomesticProduction_{meas_qual}"
            for meas_qual in ("InflationAdjusted", "Nominal", "RealValue")
            if f"gdp_{meas_qual}" not in ranked
        ],
    )


@pytest.mark.parametrize(
    ("phrase", "unread"),
    [
        ("at 2015 base-year prices", True),
        ("at 2015 prices", True),
        ("at constant prices", False),
        ("current prices", False),
        ("constant 2015 US dollars", False),
        ("in US dollars", False),
        ("inflation-adjusted", False),
        ("PPP", False),
        ("in naira", False),
        ("for East African countries", False),
        ("by sector", False),
    ],
)
def test_a_dropped_phrase_is_unread_when_it_names_a_price_basis_word_no_match_reads(
    phrase: str, unread: bool
):
    """ "PPP" is read through its two-reading span, and function words such as "in", "at"
    and "for" are ignored. A currency names the unit, so "US dollars" is ignored too."""
    assert leaves_price_basis_unread([phrase], operation_phrases=()) is unread


def test_an_operation_phrase_is_never_an_unread_price_basis():
    assert not leaves_price_basis_unread(
        ["at 2015 base-year prices"], operation_phrases=["At 2015 base-year prices "]
    )


def test_the_answer_trace_names_the_members_the_default_reading_removed():
    """An inflation-adjusted GDP that nominal GDP covers at Kenya leaves before bind, and the
    served Spec's trace says so."""
    nominal, adjusted = "NominalGdpTrace", "InflationAdjustedGdpTrace"
    result = _resolve_gdp_of_kenya(
        {
            adjusted: _node(
                "Inflation-adjusted gross domestic product",
                "InflationAdjusted",
                {"activitySource": "GrossDomesticProduction"},
            ),
            nominal: _node(
                "Nominal gross domestic product",
                "Nominal",
                {"activitySource": "GrossDomesticProduction"},
            ),
        },
        [0.9, 0.85],
        {adjusted: None, nominal: None},
        also_plausible=None,
    )

    assert result.status == "definite", result
    (spec,) = result.specs
    assert [sv.ref.dcid for sv in spec.resolved.members] == [nominal]
    assert spec.resolved.trace.default_reading == TraceDefaultReading(
        applied=True, removed_stat_vars=[adjusted]
    )
