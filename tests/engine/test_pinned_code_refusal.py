"""A code the question names is required of every shape whose slots hold its property.

Each code term the bind leaves unapplied on a property is recorded once as a code
requirement (`region.state.Requirement`), and every final answer is judged against each
requirement alone (`region.requirements.judge`). An answer that does not apply one refuses
`qualifier_not_bound` with the answer as its flagged match, and the refused part names
the code once, as a term, with no departure beside it. A price basis, a qualifier no one
slot clears, and a code on a property the shape lacks depart.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import cast

import pytest

from entity_resolution import Admission, EntityResolver
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.departures import (
    measure_not_applied,
    price_basis_not_applied,
)
from search.engine.qualifier_match import QualifierMatch
from search.engine.region.phase_answer import _assemble_carve_outs
from search.engine.region.requirements import (
    record_reading,
    refusal_terms,
    settle_answer,
    settle_candidate_set,
    with_requirement_departures,
)
from search.engine.region.result import RegionResult
from search.engine.region.state import (
    ResolveState,
    code_requirement,
    qualifier_dropped_warning,
    with_requirement,
)
from search.engine.region.unrealisable import unbind_unrealisable
from search.engine.retrieval.retrieve import BoundValuesUnrealisable, Materialised, MemberConflict
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.heading.departure_fact import DepartureFact
from search.models import CoverageBare, GraphRef, ResolvedQualifier, Spec, StatVar, Warning
from tests.conftest import minimal_spec
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.fixtures import FakeGraph

_SV = "worldBank/SE_XPD_TOTL_GD_ZS"
_ASKED = "health expenditure"
_HEALTH = QualifierMatch(
    code_system="expenditureType",
    code_dcid="HealthcareExpenditure",
    label="Healthcare expenditure",
    source="label",
    matched_text=_ASKED,
)
_CLOSEST = DepartureFact(
    code="closest_match",
    binding_ref=None,
    asked="Current health expenditure as share of GDP",
    shown="percent of nominal GDP spent on education",
)
_CLOSEST_WARNING = Warning(
    code="elected_shape_no_data", severity="degraded", message="closest reading"
)


def _education_share() -> ShapeDraft:
    """The education share of GDP: its one constraint slot is `expenditureType`."""
    return ShapeDraft(
        shape_id="education_share_of_gdp",
        label="percent of nominal GDP spent on education",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid="Amount_EconomicActivity_GrossDomesticProduction_Nominal",
        slot_keys=(SlotKeyDraft(axis="how", property_dcid="expenditureType", label="type"),),
    )


def _state(
    *unapplied: tuple[str, str | None], departed: tuple[tuple[Warning, DepartureFact], ...] = ()
) -> ResolveState:
    """The state the definite door reads after the Kenya ratio's election: the education
    share holds data, the bind leaves each of `unapplied`, a phrase and the property that
    clears it, unapplied (`ResolveState.require_unapplied`), and `departed` were raised on
    it."""
    state = ResolveState(
        variable="Current health expenditure as share of GDP",
        entities=[],
        date_request=None,
        detect_query="Current health expenditure as share of GDP",
        role_query="Current health expenditure as share of GDP",
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
        shape_draft=_education_share(),
        mat_result=_materialised(),
        coverage=empty_coverage_snapshot(),
    )
    state.measure_query_matches = (_HEALTH,)
    state.stat_vars = [
        StatVar(
            ref=GraphRef(ref=_SV, label="education share"),
            shape_id="education_share_of_gdp",
            slot_values=[],
            constraint_properties=[],
        )
    ]
    for warning, fact in departed:
        state.depart(warning, fact)
    for asked, prop in unapplied:
        state.require_unapplied(asked, asked=asked, property_dcid=prop, origin="bind")
    return state


def _materialised() -> Materialised:
    return Materialised(sv_dcids=[_SV], has_data=True, coverage=CoverageBare(has_data=True))


def _carve_out(state: ResolveState, spec: Spec) -> RegionResult | None:
    """The definite door's carve-outs on `spec`, with the departures the part emits."""
    result = asyncio.run(_assemble_carve_outs(state, _materialised(), spec))
    if result is None:
        return None
    return with_requirement_departures(result, state.requirements, state.judgement)


def _departed(state: ResolveState) -> list[DepartureFact]:
    """The departure facts the judged part emits beside those raised on it."""
    assert state.judgement is not None
    return [
        *state.departure_facts.values(),
        *(fact for _warning, fact in state.judgement.departures()),
    ]


def test_education_share_refuses_the_health_code_it_cannot_bind():
    """The education share holds `expenditureType` and does not bind HealthcareExpenditure,
    so the part refuses with the education share flagged, and its `closest_match` stays."""
    state = _state((_ASKED, "expenditureType"), departed=((_CLOSEST_WARNING, _CLOSEST),))
    spec = Spec.model_validate(minimal_spec())

    result = _carve_out(state, spec)

    assert result is not None
    assert result.no_data_reason == "qualifier_not_bound"
    assert result.flagged_match == (spec,)
    assert result.unbound_qualifiers == (
        ResolvedQualifier(
            text=_ASKED,
            named=[GraphRef(ref="HealthcareExpenditure", label="Healthcare expenditure")],
            slot="expenditureType",
            disposition="unbound",
        ),
    )
    # The refusal names the code; no departure repeats it.
    assert list(result.departure_facts.values()) == [_CLOSEST]
    assert [w.code for w in result.warnings] == ["elected_shape_no_data"]


@pytest.mark.parametrize(
    ("unapplied", "departed", "fact"),
    [
        # A code on a property the shape lacks is the model's judgement of the measure.
        (
            ("bilateral", "crsChannel"),
            (),
            measure_not_applied(asked="bilateral", property_dcid="crsChannel"),
        ),
        # A price basis is the unit token's to clear.
        (
            None,
            (
                (
                    qualifier_dropped_warning("constant prices"),
                    price_basis_not_applied(asked="constant prices", shown="current US dollars"),
                ),
            ),
            price_basis_not_applied(asked="constant prices", shown="current US dollars"),
        ),
        # A qualifier no one slot clears.
        (
            ("fragile states", None),
            (),
            measure_not_applied(asked="fragile states", property_dcid=None),
        ),
    ],
    ids=["property-not-held", "unit-basis", "no-slot"],
)
def test_departure_the_shape_cannot_hold_stays_definite(unapplied, departed, fact):
    state = _state(*([unapplied] if unapplied else []), departed=departed)

    assert _carve_out(state, Spec.model_validate(minimal_spec())) is None
    assert _departed(state) == [fact]


_HEALTH_RECORD = ResolvedQualifier(
    text=_ASKED,
    named=[GraphRef(ref="HealthcareExpenditure", label="Healthcare expenditure")],
    disposition="unbound",
)


def test_unrealisable_term_refuses_once_and_candidate_sets_depart_it_once():
    """A term `unbind_unrealisable` dropped is a code requirement, and the re-read's unbound
    record of it adds nothing: the refusal names it once, as the requirement, and a
    candidate set that leaves it unmet departs it once."""
    state = _state()
    state.measure_query_matches = ()
    departure = (
        qualifier_dropped_warning(_ASKED),
        measure_not_applied(asked=_ASKED, property_dcid="expenditureType"),
    )
    state.requirements = with_requirement(
        state.requirements,
        code_requirement(
            property_dcid="expenditureType",
            codes=("HealthcareExpenditure",),
            asked=_ASKED,
            named=_HEALTH_RECORD.named,
            departure=departure,
            origin="unbinding",
        ),
    )
    record_reading(state, [_HEALTH_RECORD])

    assert [r.origin for r in state.requirements] == ["unbinding"]
    assert refusal_terms(state, state.shape_draft, state.bindings) == (
        _HEALTH_RECORD.model_copy(update={"slot": "expenditureType"}),
    )
    settle_candidate_set(state, [(state.shape_draft, state.bindings)])
    assert state.judgement is not None
    assert state.judgement.departures() == (departure,)


def _health_share(property_dcid: str) -> ShapeDraft:
    """A health share of GDP whose one constraint slot is `property_dcid`."""
    return ShapeDraft(
        shape_id=f"health_share_{property_dcid}",
        label="health expenditure as a share of GDP",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid="Amount_EconomicActivity_GrossDomesticProduction_Nominal",
        slot_keys=(SlotKeyDraft(axis="how", property_dcid=property_dcid, label="type"),),
    )


@pytest.mark.parametrize(
    ("survivor", "bound", "refuses"),
    [
        (_health_share("expenditureType"), "HealthcareExpenditure", False),
        (_health_share("expenditureType"), "EducationExpenditure", True),
        (_health_share("sector"), "HealthcareExpenditure", False),
    ],
    ids=["binds-the-code", "binds-another-code", "lacks-the-property"],
)
def test_an_alternate_survivor_is_judged_on_its_own_shape_and_bindings(survivor, bound, refuses):
    """The elected education share dropped "health expenditure". A single surviving
    alternate binds its values from its own representative, so it refuses on the code only
    when its own slots hold the property and its own bindings leave the code out."""
    state = _state((_ASKED, "expenditureType"))
    prop = survivor.slot_keys[0].property_dcid
    bindings = [SlotBindingDraft(axis="how", property_dcid=prop, kind="value", value_dcids=[bound])]

    assert bool(refusal_terms(state, survivor, bindings)) is refuses
    # The elected shape itself still refuses on the code it dropped.
    assert refusal_terms(state, state.shape_draft, state.bindings)


_EDUCATION_ASKED = "education expenditure"
_EDUCATION = QualifierMatch(
    code_system="expenditureType",
    code_dcid="EducationExpenditure",
    label="Education expenditure",
    source="label",
    matched_text=_EDUCATION_ASKED,
)


def _binding(*codes: str) -> list[SlotBindingDraft]:
    return [
        SlotBindingDraft(
            axis="how", property_dcid="expenditureType", kind="set", value_dcids=list(codes)
        )
    ]


def _dropped_state() -> ResolveState:
    """The elected education share's bind leaves "health expenditure" unapplied."""
    return _state((_ASKED, "expenditureType"))


def _tied_state() -> ResolveState:
    """As `_dropped_state`, where "health expenditure" is a tie the elected bindings leave
    unresolved, so the elected reading also records it as an unbound query term."""
    state = _dropped_state()
    record_reading(state, [_HEALTH_RECORD])
    return state


def _named_terms_state() -> ResolveState:
    """The question names health and education expenditure, and the elected shape's bind
    leaves both unapplied."""
    state = _state()
    state.measure_query_matches = (_HEALTH, _EDUCATION)
    for asked in (_ASKED, _EDUCATION_ASKED):
        state.require_unapplied(asked, asked=asked, property_dcid="expenditureType", origin="bind")
    return state


def _bound_without_terms_state() -> ResolveState:
    """The elected shape binds health and education expenditure, which the words name by
    their names and no recorded term names, and no member holds them, so unbinding drops
    both, under one departure."""
    state = _state()
    state.variable = "Healthcare expenditure and Education expenditure as share of GDP"
    state.measure_query_matches = ()
    state.graph = cast(
        EngineGraphClient,
        FakeGraph(
            nodes={
                "HealthcareExpenditure": {"label": "Healthcare expenditure", "type": "Thing"},
                "EducationExpenditure": {"label": "Education expenditure", "type": "Thing"},
            },
            obs={},
            detect={},
            resolve={},
        ),
    )
    state.bindings = _binding("HealthcareExpenditure", "EducationExpenditure")
    state.bind_result = _UniversalBindOutput(
        chosen_shape_index=0, bindings=[], also_plausible=None, ask=None
    )
    unbind_unrealisable(
        state,
        BoundValuesUnrealisable(
            conflicts=(MemberConflict(properties=frozenset({"expenditureType"}), fully_bound=True),)
        ),
    )
    return state


_BOTH_ASKED = "Healthcare expenditure and Education expenditure"


@pytest.mark.parametrize(
    ("requested", "other", "departed"),
    [
        (_dropped_state, "HealthcareExpenditure", []),
        (_dropped_state, "EducationExpenditure", [_ASKED]),
        # The tie's unbound query record is the code requirement's phrase, so it departs
        # only as the requirement does.
        (_tied_state, "HealthcareExpenditure", []),
        # Healthcare is met by both answers and Education is not, so Education departs
        # alone: their shared departure would disclose Healthcare too.
        (_bound_without_terms_state, "HealthcareExpenditure", ["Education expenditure"]),
        (_bound_without_terms_state, "DefenceExpenditure", [_BOTH_ASKED]),
    ],
    ids=[
        "every-answer-binds-the-code",
        "one-answer-binds-another-code",
        "tie-every-answer-binds-the-code",
        "shared-departure-one-value-unmet",
        "shared-departure-both-values-unmet",
    ],
)
def test_a_candidate_set_keeps_a_code_departure_one_answer_leaves_unmet(requested, other, departed):
    """A candidate set of two health shares, one binding health and education expenditure
    and one binding `other`, departs each requirement one of them leaves unmet, once, and
    nothing for a requirement both meet."""
    state = requested()
    survivor = _health_share("expenditureType")

    settle_candidate_set(
        state,
        [
            (survivor, _binding("HealthcareExpenditure", "EducationExpenditure")),
            (replace(survivor, shape_id="health_share_other"), _binding(other)),
        ],
    )

    assert [(f.code, f.asked) for f in _departed(state)] == [
        ("measure_not_applied", asked) for asked in departed
    ]


@pytest.mark.parametrize(
    ("requested", "bound", "unbound"),
    [
        (_named_terms_state, ("HealthcareExpenditure",), [_EDUCATION_ASKED]),
        (_named_terms_state, ("EducationExpenditure",), [_ASKED]),
        (_named_terms_state, ("HealthcareExpenditure", "EducationExpenditure"), []),
        (_bound_without_terms_state, ("HealthcareExpenditure",), ["Education expenditure"]),
    ],
    ids=["binds-health-only", "binds-education-only", "binds-both", "unnamed-binds-health-only"],
)
def test_each_requested_code_is_judged_alone(requested, bound, unbound):
    """The request asks for health and education expenditure on one property, by two terms
    or by two bound values its words name and no recorded term names. Applying one code does not satisfy the other:
    each requirement is met only by its own codes."""
    state = requested()

    terms = refusal_terms(state, _health_share("expenditureType"), _binding(*bound))

    assert [t.text for t in terms] == unbound


@pytest.mark.parametrize(
    ("bound", "unbound"),
    [
        ("EducationExpenditure", [("expenditureType", _ASKED)]),
        ("HealthcareExpenditure", []),
    ],
    ids=["stand-in-binds-another-code", "stand-in-binds-the-code"],
)
def test_a_stand_in_is_judged_on_the_code_the_request_named(bound, unbound):
    """The elected shape dropped "health expenditure" and one stand-in answers in its
    place. Definite assembly judges the stand-in's own shape and bindings
    (`candidate_collapse._assemble_definite`): one binding education on the same property
    leaves the health requirement unmet and refuses on it, naming it once as a term; one
    binding health meets it. In neither case does it depart."""
    state = _state((_ASKED, "expenditureType"))
    stand_in = _education_share()

    refused = settle_answer(state, stand_in, _binding(bound))

    assert [(t.slot, t.text) for t in refused] == unbound
    assert _departed(state) == []
