"""Tests cover standard resolution, bound-constraint filtering, and member selection."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest

from search.engine.bind import SlotBindingDraft
from search.engine.conjoin import assemble_region
from search.engine.errors import GraphInfraError
from search.engine.extract import DateRequest
from search.engine.retrieval.retrieve import (
    BoundValuesUnrealisable,
    FamilyTotalAbsent,
    Materialised,
    MemberConflict,
    NoDataDraft,
    materialise,
)
from search.engine.shape import ShapeDraft
from search.models import CoverageBare, TimeWindow
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.fixtures import FakeGraph, make_resolver

# Declaration-free standard shapes probe the subject role.

# The graph names a provenance on every facet it serves; `select_facet` refuses one without.
_SOURCE = "dc/base/StandardFixture"


def test_a_shape_with_no_declarations_resolves_by_subject_probe():
    """A declaration-free shape probes subject facets and propagates GraphInfraError."""
    shape = ShapeDraft(
        shape_id="test_subject_probe_shape",
        label="test shape",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={"sv_a": {}},
        representative_score=0.9,
    )

    with pytest.raises(GraphInfraError):
        materialise(
            shape,
            [],
            {"observationAbout": ["country/KEN"]},
            graph=FakeGraph(raise_on_call=True),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )


# Bound-constraint filtering and its refusal.

_MALE_SV = "UnemploymentRate_Person_Male"
_RECIPIENT = "country/USA"


def _gender_arcs(gender: str) -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "unemploymentRate"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": "gender"}]},
        "gender": {"nodes": [{"dcid": gender}]},
    }


def _male_only_shape() -> ShapeDraft:
    """A shape whose only member asserts gender=Male."""
    return ShapeDraft(
        shape_id="test_unemployment_shape",
        label="test shape",
        pop_type_dcid="Person",
        meas_prop_dcid="unemploymentRate",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_MALE_SV: _gender_arcs("Male")},
        representative_score=0.9,
    )


def _graph_with_male_data() -> FakeGraph:
    obs = {
        f"{_MALE_SV}|{_RECIPIENT}": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1, "provenanceId": _SOURCE}
        ]
    }
    return FakeGraph(nodes={}, obs=obs, detect={}, resolve={})


_MISMATCH_GENDER_SV = "UnemploymentRate_Person_Male_NoResidence"
_MISMATCH_RESIDENCE_SV = "UnemploymentRate_Person_Rural_NoGender"


def _two_axis_arcs(*, gender: str | None, residence: str | None) -> dict:
    props = [p for p, v in (("gender", gender), ("residence", residence)) if v is not None]
    arcs: dict = {
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "unemploymentRate"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in props]},
    }
    if gender is not None:
        arcs["gender"] = {"nodes": [{"dcid": gender}]}
    if residence is not None:
        arcs["residence"] = {"nodes": [{"dcid": residence}]}
    return arcs


def _gender_binding(kind: str = "value", *values: str) -> SlotBindingDraft:
    return SlotBindingDraft(axis="how", property_dcid="gender", kind=kind, value_dcids=list(values))


def _residence_binding(value: str) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how", property_dcid="residence", kind="value", value_dcids=[value]
    )


def _two_axis_shape() -> ShapeDraft:
    """Two members, each asserting one of gender and residence."""
    return ShapeDraft(
        shape_id="test_two_axis_shape",
        label="test two-axis shape",
        pop_type_dcid="Person",
        meas_prop_dcid="unemploymentRate",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            _MISMATCH_GENDER_SV: _two_axis_arcs(gender="Male", residence=None),
            _MISMATCH_RESIDENCE_SV: _two_axis_arcs(gender=None, residence="Rural"),
        },
        representative_score=0.9,
    )


def _conflict(*properties: str, fully_bound: bool = False) -> MemberConflict:
    return MemberConflict(properties=frozenset(properties), fully_bound=fully_bound)


class TestBoundValuesUnrealisable:
    """Bound values that together exclude every member refuse `binding_mismatch`, naming
    each member's conflicting bound properties in recall order."""

    @pytest.mark.parametrize(
        ("shape", "bindings", "conflicts"),
        [
            pytest.param(
                _male_only_shape(),
                [_gender_binding("value", "Female")],
                (_conflict("gender", fully_bound=True),),
            ),
            # The member is silent on residence, so residence is no conflict, and the
            # member does not assert every bound property.
            pytest.param(
                _male_only_shape(),
                [_gender_binding("value", "Female"), _residence_binding("Urban")],
                (_conflict("gender"),),
                id="unasserted-property-not-named",
            ),
            # Each member contradicts a different bound value.
            pytest.param(
                _two_axis_shape(),
                [_gender_binding("value", "Female"), _residence_binding("Urban")],
                (_conflict("gender"), _conflict("residence")),
                id="two-axes",
            ),
        ],
    )
    def test_values_no_member_holds_refuse(self, shape, bindings, conflicts):
        result = materialise(
            shape,
            bindings,
            {"observationAbout": [_RECIPIENT]},
            graph=_graph_with_data_for(*shape.sv_arc_facts),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, BoundValuesUnrealisable)
        assert result.reason == "binding_mismatch"
        assert result.conflicts == conflicts

    @pytest.mark.parametrize(
        "binding", [_gender_binding("value", "Male"), _gender_binding("unbound")]
    )
    def test_a_value_a_member_holds_reads_it(self, binding):
        result = materialise(
            _male_only_shape(),
            [binding],
            {"observationAbout": [_RECIPIENT]},
            graph=_graph_with_male_data(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, Materialised)
        assert result.has_data is True
        assert result.sv_dcids == [_MALE_SV]


class TestRelaxation:
    """A refusal unbinds one member's conflict set: fewest request-named properties, then
    fewest engine-chosen, then a member asserting every bound property, then recall order."""

    @pytest.mark.parametrize(
        ("conflicts", "engine_chosen", "relaxed"),
        [
            # Male+urban over male-rural and female-rural: only residence goes.
            pytest.param(
                (_conflict("residence"), _conflict("gender", "residence")),
                set(),
                {"residence"},
                id="smallest-set",
            ),
            # A default gives way before a value the request named, whatever the order.
            pytest.param(
                (_conflict("gender"), _conflict("residence")),
                {"residence"},
                {"residence"},
                id="default-before-explicit",
            ),
            # One request-named property and a default beats two request-named ones.
            pytest.param(
                (_conflict("gender", "age"), _conflict("gender", "residence")),
                {"residence"},
                {"gender", "residence"},
                id="explicit-count-first",
            ),
            pytest.param(
                (_conflict("gender"), _conflict("residence")),
                set(),
                {"gender"},
                id="tie-takes-recall-order",
            ),
            # The first member is silent on a bound property, so a probe would drop it.
            pytest.param(
                (_conflict("gender"), _conflict("residence", fully_bound=True)),
                set(),
                {"residence"},
                id="tie-takes-fully-bound",
            ),
        ],
    )
    def test_picks_by_the_documented_policy(self, conflicts, engine_chosen, relaxed):
        unrealisable = BoundValuesUnrealisable(conflicts=conflicts)

        assert unrealisable.relaxation(engine_chosen=engine_chosen) == relaxed

    @pytest.mark.parametrize("conflicts", [(), (_conflict(),)], ids=["none", "empty-set"])
    def test_refuses_to_unbind_nothing(self, conflicts):
        with pytest.raises(ValueError, match="binding_mismatch"):
            BoundValuesUnrealisable(conflicts=conflicts).relaxation()


_UNRESOLVED_GENDER_SV = "UnemploymentRate_Person_UnresolvedGender"


def _unresolved_gender_arcs() -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "unemploymentRate"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": "gender"}]},
        "gender": {"nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]},
    }


def _unresolved_gender_shape() -> ShapeDraft:
    """A shape whose only member's gender arc is present but unresolved."""
    return ShapeDraft(
        shape_id="test_unresolved_gender_shape",
        label="test shape",
        pop_type_dcid="Person",
        meas_prop_dcid="unemploymentRate",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_UNRESOLVED_GENDER_SV: _unresolved_gender_arcs()},
        representative_score=0.9,
    )


def _graph_with_unresolved_gender_data() -> FakeGraph:
    obs = {
        f"{_UNRESOLVED_GENDER_SV}|{_RECIPIENT}": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}
        ]
    }
    return FakeGraph(nodes={}, obs=obs, detect={}, resolve={})


class TestFullyBoundMembersExcludesUnresolved:
    """An unresolved bound property excludes a member that has real observations."""

    def test_unresolved_bound_property_excludes_sole_member(self):
        shape = _unresolved_gender_shape()
        bindings = [
            SlotBindingDraft(
                axis="how", property_dcid="gender", kind="value", value_dcids=["Male"]
            ),
        ]
        result = materialise(
            shape,
            bindings,
            {"observationAbout": [_RECIPIENT]},
            graph=_graph_with_unresolved_gender_data(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, Materialised)
        assert result.has_data is False
        assert result.available_at_place is False


class TestDegenerateBindingFailsClosed:
    """Empty value/set bindings return variable_not_resolved despite broader Male data."""

    def test_degenerate_value_binding_refuses_despite_positive_data_for_broadened_member(self):
        shape = _male_only_shape()
        bindings = [
            SlotBindingDraft(axis="how", property_dcid="gender", kind="value", value_dcids=[]),
        ]
        result = materialise(
            shape,
            bindings,
            {"observationAbout": [_RECIPIENT]},
            graph=_graph_with_male_data(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, NoDataDraft)
        assert result.reason == "variable_not_resolved"

    def test_degenerate_set_binding_also_refuses(self):
        shape = _male_only_shape()
        bindings = [
            SlotBindingDraft(axis="how", property_dcid="gender", kind="set", value_dcids=[]),
        ]
        result = materialise(
            shape,
            bindings,
            {"observationAbout": [_RECIPIENT]},
            graph=_graph_with_male_data(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, NoDataDraft)
        assert result.reason == "variable_not_resolved"


def _unemployment_graph() -> FakeGraph:
    """The Male-only unemployment series with data in the USA, and a Female node."""
    return FakeGraph(
        nodes={
            _MALE_SV: {
                "label": "Unemployment rate (male)",
                "type": "StatisticalVariable",
                "arcs": _gender_arcs("Male"),
            },
            "Female": {"label": "Female", "type": "GenderType"},
            "country/USA": {"label": "United States", "type": "Country"},
        },
        obs={
            f"{_MALE_SV}|{_RECIPIENT}": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}
            ]
        },
        detect={"unemployment rate": {"svs": [_MALE_SV]}},
        resolve={"USA": _RECIPIENT},
    )


def _resolve_unemployment(
    binding: SlotBindingDraft,
    monkeypatch=None,
    *,
    default=None,
    dropped: tuple[str, ...] = (),
    also: tuple[SlotBindingDraft, ...] = (),
    graph: FakeGraph | None = None,
    query: str = "female unemployment rate",
):
    """Resolve `query` in USA over the Male-only shape, or `graph`'s, the binder drafting
    `binding` on gender beside `also`. `default` is a binding the declared defaults supply
    in place of an open slot on its property. `dropped` are the phrases extraction
    dropped."""
    import asyncio

    from search.engine.bind import _UniversalBindOutput
    from search.engine.region import phase_election, resolve_variable

    detect_query = "unemployment rate"
    graph = graph or _unemployment_graph()

    class _GenderLLM:
        async def generate_structured(self, *, prompt, system, schema):
            assert schema.__name__ == "_UniversalBindOutput"
            return _UniversalBindOutput(
                chosen_shape_index=0, bindings=[binding, *also], also_plausible=None, ask=None
            ), None

    if default is not None:

        def _defaults(bindings, *_args, **_kwargs):
            kept = [b for b in bindings if b.property_dcid != default.property_dcid]
            return [*kept, default], [default]

        monkeypatch.setattr(phase_election, "apply_declared_defaults", _defaults)

    return asyncio.run(
        resolve_variable(
            query,
            entities=["USA"],
            date_request=None,
            detect_query=detect_query,
            role_query=f"{query} in USA",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_GenderLLM(),
            dropped_qualifier_phrases=list(dropped),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


def _departures(region):
    (part,) = assemble_region(
        region,
        extra_warnings=[],
        start_ms=0,
        engine_build="test",
        max_candidates=5,
    ).parts
    return [(d.code, d.binding_ref, d.asked) for d in part.departures]


class TestUnrealisableValuesUnbound:
    """Bound values no member holds together are unbound and the shape resolved again, so
    the Spec describes what was read."""

    def test_a_request_named_value_no_member_holds_departs_and_is_not_read(self):
        from search.render.heading import heading_for
        from search.render.tokens import plan_words

        without = _resolve_unemployment(_gender_binding("unbound"))
        result = _resolve_unemployment(_gender_binding("value", "Female"))

        assert result.status == without.status == "definite"
        (spec,) = result.specs
        assert spec.spec_id == without.specs[0].spec_id
        assert "female" not in plan_words(heading_for(spec))
        assert _departures(result) == [("measure_not_applied", "how:gender", "Female")]

    def test_a_dropped_phrase_the_raw_text_bound_departs_once_in_its_own_words(self, monkeypatch):
        """Extraction dropped "for women" and the raw text bound it to Female, which no
        member holds. The filter departs once, in the request's words, on its token; the
        phrase stays admitted, so the request-wide dropped-qualifier warning does not
        return beside it."""
        from search.engine.assemble import _surviving_qualifier_warnings
        from search.engine.qualifier_match import QualifierMatch
        from search.engine.region import phase_rank_bind
        from search.models import Warning

        phrase = "for women"
        match_query_qualifiers = phase_rank_bind._match_query_qualifiers

        def _matching_women(sources, **kwargs):
            read = match_query_qualifiers(sources, **kwargs)
            women = QualifierMatch(
                code_system="gender",
                code_dcid="Female",
                label="Female",
                source="synonym",
                matched_text=phrase,
            )
            matches = {vocabulary: (*own, women) for vocabulary, own in read.matches.items()}
            return replace(read, matches=matches)

        monkeypatch.setattr(phase_rank_bind, "_match_query_qualifiers", _matching_women)
        result = _resolve_unemployment(_gender_binding("unbound"), monkeypatch, dropped=(phrase,))
        dropped = Warning(code="query_qualifier_dropped", severity="info", message=phrase)

        assert result.status == "definite"
        assert _departures(result) == [("measure_not_applied", "how:gender", phrase)]
        assert _surviving_qualifier_warnings([dropped], [phrase], [result]) == []

    def test_an_engine_chosen_value_no_member_holds_is_dropped_silently(self, monkeypatch):
        without = _resolve_unemployment(_gender_binding("unbound"))
        result = _resolve_unemployment(
            _gender_binding("unbound"), monkeypatch, default=_gender_binding("value", "Female")
        )

        assert result.status == "definite"
        assert result.specs[0].spec_id == without.specs[0].spec_id
        assert _departures(result) == []
        assert [w.code for w in result.warnings] == [w.code for w in without.warnings]

    def test_an_alternate_reads_without_values_no_member_holds(self):
        """An alternate's values come from its representative, none from the request, so
        values no member holds together are unbound silently and the alternate read."""
        from search.engine.region.candidates import _Alternate, _materialise_alt_shapes
        from tests.engine._builders import bound_place_context, subject_place_roles

        results, unsupported = _materialise_alt_shapes(
            [_Alternate(_male_only_shape(), [_gender_binding("value", "Female")])],
            places=[bound_place_context(subject_place_roles(_RECIPIENT))],
            named_place_dcids=[_RECIPIENT],
            graph=_graph_with_male_data(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            date_request=None,
            coverage=empty_coverage_snapshot(),
        )

        assert unsupported is None
        (candidate,) = results
        assert candidate.mat.sv_dcids == [_MALE_SV]
        assert [(b.property_dcid, b.kind) for b in candidate.bindings] == [
            ("gender", "unbound"),
            (None, "value"),
        ]

    def test_a_resubmit_binding_values_no_member_holds_together_answers_no_data(self):
        """An old share link posts gender=Male and residence=Rural, and no member holds both.
        The resubmit elects the identity's members as the mint does, finds none, and
        answers no_data(no_matching_slice)."""
        region = _resubmit_unemployment(
            {"gender": "Male", "residence": "Rural"},
            {
                _MISMATCH_GENDER_SV: _two_axis_arcs(gender="Male", residence=None),
                _MISMATCH_RESIDENCE_SV: _two_axis_arcs(gender=None, residence="Rural"),
            },
        )

        assert (region.status, region.no_data_reason) == ("no_data", "no_matching_slice")

    def test_a_resubmit_keeps_the_posted_value_a_member_holds(self):
        """A resubmit posts gender=Male and residence=Urban over male-rural and
        female-rural. Both members contradict urban and one also male, so the resubmit
        unbinds residence alone, departing it, and keeps Male."""
        from search.engine.departures import DepartureFact
        from search.engine.region.unrealisable import unbind_posted

        members = {
            _MALE_RURAL_SV: _two_axis_arcs(gender="Male", residence="Rural"),
            _FEMALE_RURAL_SV: _two_axis_arcs(gender="Female", residence="Rural"),
        }
        bindings = [_gender_binding("value", "Male"), _residence_binding("Urban")]
        graph = _two_axis_graph(members)
        refusal = materialise(
            replace(_two_axis_shape(), sv_arc_facts=members),
            bindings,
            {"observationAbout": [_RECIPIENT]},
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
        assert isinstance(refusal, BoundValuesUnrealisable)

        @dataclass
        class _Resubmit:
            bindings: list[SlotBindingDraft]
            graph: FakeGraph
            warnings: list[object] = field(default_factory=list)
            departed: list[DepartureFact] = field(default_factory=list)

            def depart(self, warning: object, fact: DepartureFact) -> object:
                self.departed.append(fact)
                return warning

        state = _Resubmit(bindings=bindings, graph=graph)
        unbind_posted(state, refusal)

        assert [(b.property_dcid, b.kind, b.value_dcids) for b in state.bindings] == [
            ("gender", "value", ["Male"]),
            ("residence", "unbound", []),
        ]
        assert [(fact.binding_ref, fact.asked) for fact in state.departed] == [
            ("how:residence", "Urban")
        ]

    def test_a_request_keeps_the_value_a_member_holds(self):
        """The reviewer's case: male and urban over male-rural and female-rural. Only
        urban departs, and male-rural answers."""
        result = _resolve_unemployment(
            _gender_binding("value", "Male"),
            also=(_residence_binding("Urban"),),
            graph=_two_axis_graph(
                {
                    _MALE_RURAL_SV: _two_axis_arcs(gender="Male", residence="Rural"),
                    _FEMALE_RURAL_SV: _two_axis_arcs(gender="Female", residence="Rural"),
                }
            ),
            query="male urban unemployment rate",
        )

        assert result.status == "definite"
        assert _served(result) == [_MALE_RURAL_SV]
        assert _departures(result) == [("measure_not_applied", "how:residence", "Urban")]

    def test_a_default_a_member_holds_is_kept_when_a_named_value_goes(self, monkeypatch):
        """Female is named and Rural defaulted, over male-rural and male-urban. Dropping
        Female alone admits male-rural, so the default stays bound."""
        result = _resolve_unemployment(
            _gender_binding("value", "Female"),
            monkeypatch,
            default=_residence_binding("Rural"),
            graph=_two_axis_graph(
                {
                    _MALE_RURAL_SV: _two_axis_arcs(gender="Male", residence="Rural"),
                    _MALE_URBAN_SV: _two_axis_arcs(gender="Male", residence="Urban"),
                }
            ),
        )

        assert result.status == "definite"
        assert _served(result) == [_MALE_RURAL_SV]
        assert _departures(result) == [("measure_not_applied", "how:gender", "Female")]


_MALE_RURAL_SV = "UnemploymentRate_Person_Male_Rural"
_FEMALE_RURAL_SV = "UnemploymentRate_Person_Female_Rural"
_MALE_URBAN_SV = "UnemploymentRate_Person_Male_Urban"


def _two_axis_graph(members: dict[str, dict]) -> FakeGraph:
    """`members` with data in the USA, and a node for every gender and residence value."""
    return FakeGraph(
        nodes={
            **{
                sv: {"label": "Unemployment rate", "type": "StatisticalVariable", "arcs": arcs}
                for sv, arcs in members.items()
            },
            **{value: {"label": value, "type": "GenderType"} for value in ("Male", "Female")},
            **{value: {"label": value, "type": "ResidenceType"} for value in ("Rural", "Urban")},
            _RECIPIENT: {"label": "United States", "type": "Country"},
        },
        obs={
            f"{sv}|{_RECIPIENT}": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                }
            ]
            for sv in members
        },
        detect={"unemployment rate": {"svs": list(members)}},
        resolve={"USA": _RECIPIENT},
    )


def _resubmit_unemployment(posted: dict[str, str], members: dict[str, dict]):
    """Resubmit an unemployment identity posting `posted` values on their properties,
    over `members`."""
    from search.engine.region import resolve_spec_resubmit
    from search.models import ResubmitIdentityPayload
    from tests.engine._base_identity import make_identity, subject_entity
    from tests.engine._builders import value_slot

    identity = make_identity(
        {
            "kind": "statvar_schema",
            "population_type": "Person",
            "measured_property": "unemploymentRate",
            "stat_type": "measuredValue",
        },
        [value_slot("how", prop, prop, value) for prop, value in posted.items()],
        entities=[subject_entity(_RECIPIENT)],
    )
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=identity,
        ),
        graph=_two_axis_graph(members),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(_two_axis_graph(members)),
    )


def _served(region) -> list[str]:
    """The StatVars the first Spec serves."""
    return [sv.ref.dcid for sv in region.specs[0].resolved.members]


def _bound_slot(region, prop: str):
    """The binding of the first Spec's slot on `prop`."""
    (slot,) = [
        slot
        for slot in region.specs[0].resolved.slots
        if slot.key.property is not None and slot.key.property.dcid == prop
    ]
    return slot.binding


# ---------------------------------------------------------------------------
# Standard resolution collects multi-member set bindings.
# ---------------------------------------------------------------------------

_SET_SV_A = "TestExpenditure_SourceA"
_SET_SV_B = "TestExpenditure_SourceB"


def _financing_source_arcs(source: str) -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "TestExpenditureActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "testExpenditure"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": "financingSource"}]},
        "financingSource": {"nodes": [{"dcid": source}]},
    }


def _two_member_set_shape() -> ShapeDraft:
    """A shape with two members that each assert a different value of the same
    constraint property (e.g. who-06's real bug, healthFinancingSource=
    {DomesticPrivateHealthFinancing, DomesticGeneralGovernmentHealthFinancing}).
    SourceA scores higher and is inserted first (derive_shapes' insertion order
    is score order), so it is the representative, the single member a
    no-set-binding resolve collapses to (see the unbound-binding test below).
    """
    return ShapeDraft(
        shape_id="test_set_shape",
        label="test set shape",
        pop_type_dcid="TestExpenditureActivity",
        meas_prop_dcid="testExpenditure",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            _SET_SV_A: _financing_source_arcs("SourceA"),
            _SET_SV_B: _financing_source_arcs("SourceB"),
        },
        representative_score=0.9,
    )


def _graph_with_data_for(*svs: str) -> FakeGraph:
    obs = {
        f"{sv}|{_RECIPIENT}": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1, "provenanceId": _SOURCE}
        ]
        for sv in svs
    }
    return FakeGraph(nodes={}, obs=obs, detect={}, resolve={})


class TestSetBindingMultiMemberCollection:
    """A set binding collects every matching member with data in score order."""

    def test_both_members_with_data_are_both_collected_in_score_order(self):
        shape = _two_member_set_shape()
        bindings = [
            SlotBindingDraft(
                axis="how",
                property_dcid="financingSource",
                kind="set",
                value_dcids=["SourceA", "SourceB"],
            ),
        ]
        graph = _graph_with_data_for(_SET_SV_A, _SET_SV_B)
        result = materialise(
            shape,
            bindings,
            {"observationAbout": [_RECIPIENT]},
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, Materialised)
        assert result.has_data is True
        assert result.sv_dcids == [_SET_SV_A, _SET_SV_B]
        assert result.facets_by_sv is not None
        assert set(result.facets_by_sv) == {_SET_SV_A, _SET_SV_B}
        assert result.destination_confirmed is None

    def test_only_the_member_with_data_is_collected(self):
        """Only the set member with observations is emitted."""
        shape = _two_member_set_shape()
        bindings = [
            SlotBindingDraft(
                axis="how",
                property_dcid="financingSource",
                kind="set",
                value_dcids=["SourceA", "SourceB"],
            ),
        ]
        graph = _graph_with_data_for(_SET_SV_B)  # only the lower-scored member has data
        result = materialise(
            shape,
            bindings,
            {"observationAbout": [_RECIPIENT]},
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, Materialised)
        assert result.has_data is True
        assert result.sv_dcids == [_SET_SV_B]


# Bind-aware member specificity.

_BARE_SV = "TestIncidence_Bare"
_FEMALE_SV = "TestIncidence_Female"


def _incidence_arcs(sex: str | None) -> dict:
    """Build a bare or Female-specific incidence member."""
    arcs: dict = {
        "populationType": {"nodes": [{"dcid": "TestIncidenceEvent"}]},
        "measuredProperty": {"nodes": [{"dcid": "testIncidence"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if sex is None:
        arcs["constraintProperties"] = {"nodes": []}
    else:
        arcs["constraintProperties"] = {"nodes": [{"dcid": "testSex"}]}
        arcs["testSex"] = {"nodes": [{"dcid": sex}]}
    return arcs


def _bare_plus_female_shape() -> ShapeDraft:
    """Insert the bare aggregate before its Female-specific sibling."""
    return ShapeDraft(
        shape_id="test_incidence_shape",
        label="test incidence shape",
        pop_type_dcid="TestIncidenceEvent",
        meas_prop_dcid="testIncidence",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            _BARE_SV: _incidence_arcs(None),
            _FEMALE_SV: _incidence_arcs("Female"),
        },
        representative_score=0.9,
    )


@pytest.mark.parametrize(
    ("shape", "bindings", "graph", "expected_sv_dcids"),
    [
        pytest.param(
            _bare_plus_female_shape(),
            [
                SlotBindingDraft(
                    axis="what",
                    property_dcid="testSex",
                    kind="unbound",
                    value_dcids=[],
                ),
            ],
            _graph_with_data_for(_BARE_SV, _FEMALE_SV),
            [_BARE_SV],
            id="bare-plus-female",
        ),
        pytest.param(
            _two_member_set_shape(),
            [
                SlotBindingDraft(
                    axis="how",
                    property_dcid="financingSource",
                    kind="unbound",
                    value_dcids=[],
                ),
            ],
            _graph_with_data_for(_SET_SV_A, _SET_SV_B),
            [_SET_SV_A],
            id="source-a-plus-source-b",
        ),
    ],
)
def test_unbound_entity_probe_preserves_insertion_order_and_single_match(
    shape, bindings, graph, expected_sv_dcids
):
    result = materialise(
        shape,
        bindings,
        {"observationAbout": [_RECIPIENT]},
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == expected_sv_dcids


@pytest.mark.parametrize(
    ("shape", "bindings", "graph", "expected_sv_dcids"),
    [
        pytest.param(
            _bare_plus_female_shape(),
            [
                SlotBindingDraft(
                    axis="what",
                    property_dcid="testSex",
                    kind="value",
                    value_dcids=["Female"],
                ),
            ],
            _graph_with_data_for(),
            [_FEMALE_SV],
            id="explicit-member-wins",
        ),
        pytest.param(
            _bare_plus_female_shape(),
            [
                SlotBindingDraft(
                    axis="what",
                    property_dcid="testSex",
                    kind="value",
                    value_dcids=["Female"],
                ),
            ],
            _graph_with_data_for(),
            [_FEMALE_SV],
            id="silent-sibling-is-not-representative",
        ),
    ],
)
def test_no_entity_bound_member_is_representative(shape, bindings, graph, expected_sv_dcids):
    result = materialise(
        shape,
        bindings,
        {},
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == expected_sv_dcids


class TestBindAwareSpecificityTieBreak:
    """The probe serves full asserters even when a bare sibling passes fail-open filtering."""

    def test_explicit_member_wins_over_bare_aggregate_when_both_have_data(self):
        shape = _bare_plus_female_shape()
        bindings = [
            SlotBindingDraft(
                axis="what",
                property_dcid="testSex",
                kind="value",
                value_dcids=["Female"],
            ),
        ]
        graph = _graph_with_data_for(_BARE_SV, _FEMALE_SV)
        result = materialise(
            shape,
            bindings,
            {"observationAbout": [_RECIPIENT]},
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, Materialised)
        assert result.has_data is True
        assert result.sv_dcids == [_FEMALE_SV]

    def test_asserting_member_flags_unavailable_when_only_silent_sibling_has_data(self):
        """The Female member is required when only the bare aggregate has data."""
        shape = _bare_plus_female_shape()
        bindings = [
            SlotBindingDraft(
                axis="what",
                property_dcid="testSex",
                kind="value",
                value_dcids=["Female"],
            ),
        ]
        graph = _graph_with_data_for(_BARE_SV)  # only the bare aggregate has data
        result = materialise(
            shape,
            bindings,
            {"observationAbout": [_RECIPIENT]},
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, Materialised)
        assert result.has_data is False
        assert result.available_at_place is False
        assert result.sv_dcids == [_FEMALE_SV]
        assert result.coverage == CoverageBare(has_data=False)


class TestNoEntitySpecificityTieBreak:
    """Explicit bindings select the representative in the no-place, no-scope branch."""

    @pytest.mark.parametrize(
        "place_dcids",
        [
            pytest.param(["country/A1", "country/A2"], id="a1_a2_default_scope"),
            pytest.param(["country/A1"], id="a1_default_scope"),
        ],
    )
    def test_explicit_member_wins_with_defaulted_scope(self, place_dcids):
        """A defaulted scope probes data and selects the Female-specific member."""
        shape = _bare_plus_female_shape()
        bindings = [
            SlotBindingDraft(
                axis="what",
                property_dcid="testSex",
                kind="value",
                value_dcids=["Female"],
            ),
        ]
        graph = FakeGraph(
            nodes={},
            obs={
                f"{_FEMALE_SV}|country/A1": [
                    {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}
                ],
            },
            detect={},
            resolve={},
        )
        result = materialise(
            shape,
            bindings,
            {},
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            place_dcids=place_dcids,
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, Materialised)
        assert result.has_data is True
        assert result.sv_dcids == [_FEMALE_SV]

    def test_unbound_axis_keeps_original_insertion_order_with_no_place(self):
        shape = _bare_plus_female_shape()
        bindings = [
            SlotBindingDraft(
                axis="what",
                property_dcid="testSex",
                kind="unbound",
                value_dcids=[],
            ),
        ]
        graph = _graph_with_data_for()
        result = materialise(
            shape,
            bindings,
            {},
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == [_BARE_SV]


# Require full constraint assertion in the no-entity branch.

_SPLIT_GENDER_SV = "TestIncidence_SplitGenderOnly"
_SPLIT_RESIDENCE_SV = "TestIncidence_SplitResidenceOnly"


def _split_ownership_arcs(*, prop: str, value: str) -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "TestIncidenceEvent"}]},
        "measuredProperty": {"nodes": [{"dcid": "testIncidence"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": prop}]},
        prop: {"nodes": [{"dcid": value}]},
    }


def _split_ownership_shape() -> ShapeDraft:
    """Each sibling asserts one of two bound values; both tie at one explicit match."""
    return ShapeDraft(
        shape_id="test_split_ownership_shape",
        label="test split-ownership shape",
        pop_type_dcid="TestIncidenceEvent",
        meas_prop_dcid="testIncidence",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            _SPLIT_GENDER_SV: _split_ownership_arcs(prop="testSex", value="Female"),
            _SPLIT_RESIDENCE_SV: _split_ownership_arcs(prop="placeOfResidence", value="Urban"),
        },
        representative_score=0.9,
    )


class TestNoEntityEligibilityRestriction:
    """The no-entity branch requires one member to assert every bound property."""

    def test_split_ownership_no_member_asserts_both_degrades_honestly(self):
        """Split asserters return an unavailable carrier without a place."""
        shape = _split_ownership_shape()
        bindings = [
            SlotBindingDraft(
                axis="what", property_dcid="testSex", kind="value", value_dcids=["Female"]
            ),
            SlotBindingDraft(
                axis="how",
                property_dcid="placeOfResidence",
                kind="value",
                value_dcids=["Urban"],
            ),
        ]
        result = materialise(
            shape,
            bindings,
            {},
            graph=_graph_with_data_for(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, Materialised)
        assert result.has_data is False
        # With no place entity, available_at_place is None.
        assert result.available_at_place is None
        assert result.coverage == CoverageBare(has_data=False)
        # _bind_aware_order picks the first of two partial asserters, both matching one binding.
        assert result.sv_dcids == [_SPLIT_GENDER_SV]

    def test_split_ownership_no_member_asserts_both_degrades_honestly_with_defaulted_scope(self):
        """A defaulted scope returns the unavailable carrier when no member asserts both axes.

        The graph has Female data, while the request also binds Urban.
        """
        shape = _split_ownership_shape()
        bindings = [
            SlotBindingDraft(
                axis="what", property_dcid="testSex", kind="value", value_dcids=["Female"]
            ),
            SlotBindingDraft(
                axis="how",
                property_dcid="placeOfResidence",
                kind="value",
                value_dcids=["Urban"],
            ),
        ]
        graph = FakeGraph(
            nodes={},
            obs={
                f"{_SPLIT_GENDER_SV}|country/A1": [
                    {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}
                ],
            },
            detect={},
            resolve={},
        )
        result = materialise(
            shape,
            bindings,
            {},
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            place_dcids=["country/A1"],
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, Materialised)
        assert result.has_data is False
        assert result.available_at_place is None
        assert result.coverage == CoverageBare(has_data=False)
        assert result.sv_dcids == [_SPLIT_GENDER_SV]


# The variables coverage dimension.


class TestVariablesCoverageDimension:
    """The fixture shapes use their own member lists for coverage counts."""

    def test_single_member_resolve_reports_the_shapes_own_member_count(self):
        shape = _male_only_shape()
        bindings = [
            SlotBindingDraft(
                axis="what", property_dcid="gender", kind="value", value_dcids=["Male"]
            )
        ]
        result = materialise(
            shape,
            bindings,
            {"observationAbout": [_RECIPIENT]},
            graph=_graph_with_male_data(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
        assert isinstance(result, Materialised)
        dims = result.coverage.dimensions
        assert dims is not None
        variables_dim = next(d for d in dims if d.label == "variables")
        assert variables_dim.count == 1

    def test_multi_member_set_binding_resolve_counts_every_collected_member(self):
        shape = _two_member_set_shape()
        bindings = [
            SlotBindingDraft(
                axis="how",
                property_dcid="financingSource",
                kind="set",
                value_dcids=["SourceA", "SourceB"],
            ),
        ]
        result = materialise(
            shape,
            bindings,
            {"observationAbout": [_RECIPIENT]},
            graph=_graph_with_data_for(_SET_SV_A, _SET_SV_B),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
        assert isinstance(result, Materialised)
        dims = result.coverage.dimensions
        assert dims is not None
        variables_dim = next(d for d in dims if d.label == "variables")
        assert variables_dim.count == 2


# A UN System family serves its total on an unbound property.

_UN_MALE_U5 = "undata/unicef/CME_MR.AGE--Y0T4__SEX--M"
_UN_NEONATAL = "undata/unicef/CME_MR.AGE--M0"
_UN_U5 = "undata/unicef/CME_MR.AGE--Y0T4"
_UN_FEMALE_U5 = "undata/unicef/CME_MR.AGE--Y0T4__SEX--F"


def _un_arcs(*, age: str, sex: str | None) -> dict:
    """A child-mortality member asserting `age` and, when given, `sex`."""
    props = {"undataAge": age} | ({"undataSex": sex} if sex is not None else {})
    return {
        "populationType": {"nodes": [{"dcid": "undata/unicef/SERIES-CME_MR"}]},
        "measuredProperty": {"nodes": [{"dcid": "value"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in props]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in props.items()},
    }


def _child_mortality_shape(members: dict[str, dict]) -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_child_mortality_shape",
        label="Child mortality rate",
        pop_type_dcid="undata/unicef/SERIES-CME_MR",
        meas_prop_dcid="value",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts=members,
        representative_score=0.9,
    )


def _un_family_with_total() -> dict[str, dict]:
    """Recall order puts the male under-5 member first and its both-sexes total last."""
    return {
        _UN_MALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-M"),
        _UN_NEONATAL: _un_arcs(age="undata/AGE-M0", sex=None),
        _UN_U5: _un_arcs(age="undata/AGE-Y0T4", sex=None),
    }


def _unbound(prop: str) -> SlotBindingDraft:
    return SlotBindingDraft(axis="how", property_dcid=prop, kind="unbound", value_dcids=[])


def _resolve_at_recipient(shape: ShapeDraft, bindings: list[SlotBindingDraft], graph: FakeGraph):
    return materialise(
        shape,
        bindings,
        {"observationAbout": [_RECIPIENT]},
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )


def test_un_family_serves_the_both_sexes_member_of_the_first_reading_with_a_notice():
    """The mint keeps the two age-only members. The under-5 one holds recall's first
    member, so it answers ahead of the neonatal one, and since the two differ on the
    unbound age neither is the family's total."""
    result = _resolve_at_recipient(
        _child_mortality_shape(_un_family_with_total()),
        [_unbound("undataAge"), _unbound("undataSex")],
        _graph_with_data_for(_UN_MALE_U5, _UN_NEONATAL, _UN_U5),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_UN_U5]
    assert result.family_total_absent == FamilyTotalAbsent(
        total=None, shown=_UN_U5, places=(_RECIPIENT,)
    )


def test_un_family_whose_members_share_an_unbound_value_serves_its_total_without_a_notice():
    """Every member asserts the under-5 age, so the both-sexes member narrows the family on
    nothing and is its total, whether or not the age slot is bound."""
    members = {
        _UN_MALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-M"),
        _UN_FEMALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-F"),
        _UN_U5: _un_arcs(age="undata/AGE-Y0T4", sex=None),
    }
    result = _resolve_at_recipient(
        _child_mortality_shape(members),
        [_unbound("undataAge"), _unbound("undataSex")],
        _graph_with_data_for(*members),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_UN_U5]
    assert result.family_total_absent is None


def test_un_family_without_the_total_at_the_place_shows_the_closest_member_with_a_notice():
    """The mint keeps the two age-only members and drops the male under-5 member they
    dominate. With the under-5 member missing at the place, the neonatal member answers,
    and the result records that it is not the family's total."""
    result = _resolve_at_recipient(
        _child_mortality_shape(_un_family_with_total()),
        [_unbound("undataAge"), _unbound("undataSex")],
        _graph_with_data_for(_UN_MALE_U5, _UN_NEONATAL),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_UN_NEONATAL]
    assert result.family_total_absent == FamilyTotalAbsent(
        total=None, shown=_UN_NEONATAL, places=(_RECIPIENT,)
    )


def test_un_family_with_only_a_dominated_member_at_the_place_flags_the_first_kept_member():
    """Only the male under-5 member has data, and the mint drops it, so the answer is
    no data with the under-5 member flagged and recorded as not the family's total."""
    result = _resolve_at_recipient(
        _child_mortality_shape(_un_family_with_total()),
        [_unbound("undataAge"), _unbound("undataSex")],
        _graph_with_data_for(_UN_MALE_U5),
    )

    assert isinstance(result, Materialised)
    assert result.has_data is False
    assert result.available_at_place is False
    assert result.sv_dcids == [_UN_U5]
    assert result.family_total_absent == FamilyTotalAbsent(
        total=None, shown=_UN_U5, places=(_RECIPIENT,)
    )


def test_un_family_serves_the_matching_member_on_a_bound_slot():
    bindings = [
        _unbound("undataAge"),
        SlotBindingDraft(
            axis="how", property_dcid="undataSex", kind="value", value_dcids=["undata/SEX-M"]
        ),
    ]
    result = _resolve_at_recipient(
        _child_mortality_shape(_un_family_with_total()),
        bindings,
        _graph_with_data_for(_UN_MALE_U5, _UN_NEONATAL, _UN_U5),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_UN_MALE_U5]


def test_un_family_without_a_total_keeps_recall_order():
    members = {
        _UN_MALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-M"),
        _UN_FEMALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-F"),
    }
    result = _resolve_at_recipient(
        _child_mortality_shape(members),
        [_unbound("undataSex")],
        _graph_with_data_for(_UN_MALE_U5, _UN_FEMALE_U5),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_UN_MALE_U5]


def test_un_family_without_a_place_names_the_first_kept_member_with_a_notice():
    """The family holds under-5 and neonatal members, so the under-5 one it names is
    not the total."""
    result = materialise(
        _child_mortality_shape(_un_family_with_total()),
        [_unbound("undataSex")],
        {},
        graph=_graph_with_data_for(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_UN_U5]
    assert result.family_total_absent == FamilyTotalAbsent(total=None, shown=_UN_U5, places=())


def test_base_family_elects_the_kept_member_holding_the_first_reading():
    """Election order applies to every shape: the male member is dominated, and of the
    two age-only members the under-5 one holds recall's first pick, so it answers."""
    base_male, base_neonatal, base_total = (
        "ChildMortality_Under5_Male",
        "ChildMortality_Neonatal",
        "ChildMortality_Under5",
    )
    members = dict(
        zip(
            (base_male, base_neonatal, base_total),
            _un_family_with_total().values(),
            strict=True,
        )
    )
    result = _resolve_at_recipient(
        _child_mortality_shape(members),
        [_unbound("undataAge"), _unbound("undataSex")],
        _graph_with_data_for(base_male, base_neonatal, base_total),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [base_total]
    assert result.family_total_absent == FamilyTotalAbsent(
        total=None, shown=base_total, places=(_RECIPIENT,)
    )


# Election prefers the family's root, the member the datasets mint keeps.

_PERSON_FEMALE = "Count_Person_Female"
_PERSON_FEMALE_URBAN = "Count_Person_Female_Urban"
_PERSON_MALE = "Count_Person_Male"
_PERSON_FEMALE_DISABLED = "Count_Person_Female_Disabled"


def _person_arcs(**constraints: str) -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraints]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in constraints.items()},
    }


def _person_shape() -> ShapeDraft:
    """Recall order puts the urban member ahead of the root it is dominated by."""
    return ShapeDraft(
        shape_id="test_person_shape",
        label="Population",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            _PERSON_FEMALE_URBAN: _person_arcs(
                gender="Female", placeOfResidenceClassification="Urban"
            ),
            _PERSON_FEMALE: _person_arcs(gender="Female"),
            _PERSON_MALE: _person_arcs(gender="Male"),
        },
        representative_score=0.9,
    )


def _gender(kind: str, *values: str) -> SlotBindingDraft:
    return SlotBindingDraft(axis="how", property_dcid="gender", kind=kind, value_dcids=list(values))


def _resolve_in_years(graph: FakeGraph):
    """A Female value binding at the recipient, read for 2022 to 2024, after every point
    `_graph_with_data_for` holds."""
    return materialise(
        _person_shape(),
        [_gender("value", "Female")],
        {"observationAbout": [_RECIPIENT]},
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=DateRequest(window=TimeWindow(start_year=2022, end_year=2024), latest=False),
        coverage=empty_coverage_snapshot(),
    )


class TestRootElection:
    def test_a_value_binding_elects_the_root_over_a_dominated_member_first_in_recall(self):
        result = _resolve_at_recipient(
            _person_shape(),
            [_gender("value", "Female")],
            _graph_with_data_for(_PERSON_FEMALE_URBAN, _PERSON_FEMALE),
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == [_PERSON_FEMALE]
        assert result.population is not None
        assert result.population.confirmed == 2

    def test_a_value_binding_flags_the_root_when_only_a_dominated_member_has_data(self):
        """The mint serves the root, so the dominated urban member cannot answer."""
        result = _resolve_at_recipient(
            _person_shape(),
            [_gender("value", "Female")],
            _graph_with_data_for(_PERSON_FEMALE_URBAN),
        )

        assert isinstance(result, Materialised)
        assert result.has_data is False
        assert result.available_at_place is False
        assert result.sv_dcids == [_PERSON_FEMALE]

    def test_a_value_binding_with_no_data_flags_the_root(self):
        result = _resolve_at_recipient(
            _person_shape(), [_gender("value", "Female")], _graph_with_data_for()
        )

        assert isinstance(result, Materialised)
        assert result.has_data is False
        assert result.available_at_place is False
        assert result.sv_dcids == [_PERSON_FEMALE]

    def test_a_window_answer_never_names_a_member_the_mint_drops(self):
        """Only the dominated urban member holds data at the place, and only before the
        years read, so the root is flagged unavailable: no answer for the window names a
        member a mint of the Spec cannot serve."""
        result = _resolve_in_years(_graph_with_data_for(_PERSON_FEMALE_URBAN))

        assert isinstance(result, Materialised)
        assert result.has_data is False
        assert result.available_at_place is False
        assert result.window_gap is None
        assert result.sv_dcids == [_PERSON_FEMALE]

    def test_a_root_holding_data_before_the_years_read_answers_for_the_window(self):
        result = _resolve_in_years(_graph_with_data_for(_PERSON_FEMALE, _PERSON_FEMALE_URBAN))

        assert isinstance(result, Materialised)
        assert result.has_data is False
        assert result.available_at_place is True
        assert result.window_gap is not None
        assert result.sv_dcids == [_PERSON_FEMALE]
        assert result.family_total_absent is None

    def test_a_value_binding_elects_a_kept_member_over_one_the_mint_drops(self, monkeypatch):
        """Read as not_declared, an absent disabilityStatus keeps the disabled member in the
        mint's family beside the root, while the urban member stays dominated. With the
        root missing at the place, the disabled member answers with a notice."""
        from search.engine.retrieval import standard

        monkeypatch.setattr(
            standard, "absent_readings", lambda _index: {"disabilityStatus": "not_declared"}
        )
        shape = _person_shape()
        shape.sv_arc_facts[_PERSON_FEMALE_DISABLED] = _person_arcs(
            gender="Female", disabilityStatus="Disabled"
        )
        result = _resolve_at_recipient(
            shape,
            [_gender("value", "Female")],
            _graph_with_data_for(_PERSON_FEMALE_URBAN, _PERSON_FEMALE_DISABLED),
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == [_PERSON_FEMALE_DISABLED]
        assert result.family_total_absent == FamilyTotalAbsent(
            total=_PERSON_FEMALE, shown=_PERSON_FEMALE_DISABLED, places=(_RECIPIENT,)
        )

    def test_the_no_place_path_elects_the_root(self):
        result = materialise(
            _person_shape(),
            [_gender("value", "Female")],
            {},
            graph=_graph_with_data_for(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == [_PERSON_FEMALE]

    def test_a_set_binding_drops_the_member_another_confirmed_match_dominates(self):
        result = _resolve_at_recipient(
            _person_shape(),
            [_gender("set", "Female", "Male")],
            _graph_with_data_for(_PERSON_FEMALE_URBAN, _PERSON_FEMALE, _PERSON_MALE),
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == [_PERSON_FEMALE, _PERSON_MALE]

    def test_a_set_binding_flags_the_root_when_only_a_dominated_member_has_data(self):
        result = _resolve_at_recipient(
            _person_shape(),
            [_gender("set", "Female", "Male")],
            _graph_with_data_for(_PERSON_FEMALE_URBAN),
        )

        assert isinstance(result, Materialised)
        assert result.has_data is False
        assert result.available_at_place is False
        assert result.sv_dcids == [_PERSON_FEMALE]

    def test_a_set_binding_keeps_a_member_whose_extra_property_reads_not_declared(
        self, monkeypatch
    ):
        from search.engine.retrieval import standard

        monkeypatch.setattr(
            standard,
            "absent_readings",
            lambda _index: {"placeOfResidenceClassification": "not_declared"},
        )
        result = _resolve_at_recipient(
            _person_shape(),
            [_gender("set", "Female", "Male")],
            _graph_with_data_for(_PERSON_FEMALE_URBAN, _PERSON_FEMALE, _PERSON_MALE),
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == [_PERSON_FEMALE_URBAN, _PERSON_FEMALE, _PERSON_MALE]


class TestFamilyTotalAbsentWarning:
    """A door that elects a member says when the member shown is not its family's total.
    The sentence does not repeat it."""

    def test_resolve_variable_warns_when_the_answering_member_is_not_the_total(self):
        from search.engine.region.codes import FAMILY_TOTAL_ABSENT

        result = _resolve_child_mortality_in_chad(
            _un_family_with_total(), with_data=(_UN_MALE_U5, _UN_NEONATAL)
        )

        assert result.status == "definite"
        assert [sv.ref.dcid for sv in result.spec.resolved.members] == [_UN_NEONATAL]
        assert [w.message for w in result.warnings if w.code == FAMILY_TOTAL_ABSENT] == [
            "No total for this family of variables was found, so the answer shows "
            "Neonatal mortality rate."
        ]

    def test_a_resubmit_warns_exactly_when_it_runs_the_election(self):
        """The warning belongs to the door that elects. The resubmit's walk settles the
        family, so the resubmit elects again among it and warns as the first answer does."""
        from search.engine.region import resolve_spec_resubmit
        from search.engine.region.codes import FAMILY_TOTAL_ABSENT, MEMBERSHIP_UNVERIFIED
        from search.engine.region.state import EngineOptions
        from search.models import ResubmitIdentityPayload
        from tests.engine._harness import member_walk_reader

        members = _un_family_with_total()
        with_data = (_UN_MALE_U5, _UN_NEONATAL)
        graph = _chad_graph(members, with_data=with_data)
        first = _resolve_child_mortality_in_chad(members, with_data=with_data)
        identity = first.spec.definition.identity
        region = resolve_spec_resubmit(
            inp=ResubmitIdentityPayload(identity=identity),
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(materialise_graph=member_walk_reader(graph)),
        )

        warned = [
            "No total for this family of variables was found, so the answer shows "
            "Neonatal mortality rate."
        ]
        assert region.status == "definite"
        assert [sv.ref.dcid for sv in region.specs[0].resolved.members] == [_UN_NEONATAL]
        assert [w.message for w in region.warnings if w.code == FAMILY_TOTAL_ABSENT] == warned
        assert MEMBERSHIP_UNVERIFIED not in [w.code for w in region.warnings]
        # The resubmit holds no words for the measure, so its closest match states only the
        # member shown, and the lead stays after an edit.
        (part,) = assemble_region(
            region,
            extra_warnings=[],
            start_ms=0,
            engine_build="test",
            max_candidates=5,
        ).parts
        closest = [(d.asked, d.text) for d in part.departures if d.code == "closest_match"]
        assert closest == [
            (None, "This shows neonatal mortality rate, the closest measure the data holds.")
        ]

    def test_a_resubmit_over_a_settled_family_of_one_admitted_member_warns(self):
        """Age is bound to under 5 and sex to the set {male, female}. The family holds the
        male under-5 member and a female neonatal one, so the bindings admit the male member
        alone, which states one value of the two the set names: the first answer warns. The
        resubmit's walk settles the family, elects among it, one member admitted, and warns
        as the first answer does."""
        import asyncio

        from search.engine.bind import _UniversalBindOutput
        from search.engine.region import resolve_spec_resubmit, resolve_variable
        from search.engine.region.codes import FAMILY_TOTAL_ABSENT
        from search.engine.region.state import EngineOptions
        from search.models import ResubmitIdentityPayload
        from tests.engine._harness import member_walk_reader

        female_neonatal = "undata/unicef/CME_MR.AGE--M0__SEX--F"
        members = {
            _UN_MALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-M"),
            female_neonatal: _un_arcs(age="undata/AGE-M0", sex="undata/SEX-F"),
        }
        labels = {**_UN_LABELS, female_neonatal: "Neonatal mortality rate, female"}
        bindings = [
            SlotBindingDraft(
                axis="how", property_dcid="undataAge", kind="value", value_dcids=["undata/AGE-Y0T4"]
            ),
            SlotBindingDraft(
                axis="how",
                property_dcid="undataSex",
                kind="set",
                value_dcids=["undata/SEX-M", "undata/SEX-F"],
            ),
        ]
        chad = "country/TCD"
        graph = FakeGraph(
            nodes={
                **{
                    sv: {"label": labels[sv], "type": "StatisticalVariable", "arcs": arcs}
                    for sv, arcs in members.items()
                },
                chad: {"label": "Chad", "type": "Country"},
            },
            obs={
                f"{sv}|{chad}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}]
                for sv in members
            },
            detect={"child mortality rate": {"svs": list(members)}},
            resolve={"Chad": chad},
        )

        class _BindLLM:
            async def generate_structured(self, *, prompt, system, schema):
                return _UniversalBindOutput(
                    chosen_shape_index=0, bindings=bindings, also_plausible=None, ask=None
                ), None

        first = asyncio.run(
            resolve_variable(
                "child mortality rate",
                entities=["Chad"],
                date_request=None,
                detect_query="child mortality rate",
                role_query="child mortality rate in Chad",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                llm=_BindLLM(),
                base_timing={},
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        )
        assert [w.message for w in first.warnings if w.code == FAMILY_TOTAL_ABSENT] == [
            "No total for this family of variables was found, so the answer shows "
            "Under-five mortality rate, male."
        ]
        identity = first.spec.definition.identity
        region = resolve_spec_resubmit(
            inp=ResubmitIdentityPayload(identity=identity),
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(materialise_graph=member_walk_reader(graph)),
        )

        assert region.status == "definite"
        assert [sv.ref.dcid for sv in region.specs[0].resolved.members] == [_UN_MALE_U5]
        assert [w.message for w in region.warnings if w.code == FAMILY_TOTAL_ABSENT] == [
            "No total for this family of variables was found, so the answer shows "
            "Under-five mortality rate, male."
        ]

    @pytest.mark.parametrize(
        ("fact", "message"),
        [
            pytest.param(
                FamilyTotalAbsent(total="total", shown="shown", places=("country/TCD",)),
                "No data for Total was found for Chad, so the answer shows Shown.",
                id="the_total_has_no_data",
            ),
            pytest.param(
                FamilyTotalAbsent(total=None, shown="shown", places=("country/TCD",)),
                "No total for this family of variables was found, so the answer shows Shown.",
                id="the_family_holds_no_total",
            ),
            # `labels` holds neither place, so a read that labelled them would raise.
            pytest.param(
                FamilyTotalAbsent(total="total", shown="shown", places=("country/A", "country/B")),
                "No data for Total was found for any of the 2 places, so the answer shows Shown.",
                id="several_places_are_counted_and_not_labelled",
            ),
        ],
    )
    def test_the_message_names_the_total_only_when_the_family_holds_one(self, fact, message):
        from search.engine.region.materialise_step import resubmit_family_total_absent_departure

        labels = {"total": "Total", "shown": "Shown", "country/TCD": "Chad"}

        warning, _departure = resubmit_family_total_absent_departure(
            fact, spec=None, label_of=lambda dcids: [labels[dcid] for dcid in dcids]
        )

        assert warning.message == message

    def test_a_no_data_part_warns_when_the_flagged_member_is_not_the_total(self):
        """No kept member has data in Chad, so the part is no_data. Its flagged member is the
        male under-5 one, which the part says is not the family's total."""
        result = _resolve_child_mortality_in_chad(
            {
                _UN_MALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-M"),
                _UN_FEMALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-F"),
            },
            with_data=(),
        )

        assert result.status == "no_data"
        assert [w.message for w in result.warnings if w.code == "family_total_absent"] == [
            "No total for this family of variables was found, so the answer shows "
            "Under-five mortality rate, male."
        ]


_UN_LABELS = {
    _UN_MALE_U5: "Under-five mortality rate, male",
    _UN_FEMALE_U5: "Under-five mortality rate, female",
    _UN_NEONATAL: "Neonatal mortality rate",
    _UN_U5: "Under-five mortality rate",
}


def _chad_graph(members: dict[str, dict], *, with_data: tuple[str, ...]) -> FakeGraph:
    """The child-mortality `members` in Chad, those in `with_data` holding a point in 2020."""
    chad = "country/TCD"
    return FakeGraph(
        nodes={
            **{
                sv: {"label": _UN_LABELS[sv], "type": "StatisticalVariable", "arcs": arcs}
                for sv, arcs in members.items()
            },
            chad: {"label": "Chad", "type": "Country"},
        },
        obs={
            f"{sv}|{chad}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}]
            for sv in with_data
        },
        detect={"child mortality rate": {"svs": list(members)}},
        resolve={"Chad": chad},
    )


def _resolve_child_mortality_in_chad(members: dict[str, dict], *, with_data: tuple[str, ...]):
    """Resolve "child mortality rate" in Chad with age and sex left unbound."""
    import asyncio

    from search.engine.bind import _UniversalBindOutput
    from search.engine.region import resolve_variable

    detect_query = "child mortality rate"
    graph = _chad_graph(members, with_data=with_data)

    class _UnboundAgeAndSexLLM:
        async def generate_structured(self, *, prompt, system, schema):
            assert schema.__name__ == "_UniversalBindOutput"
            return _UniversalBindOutput(
                chosen_shape_index=0,
                bindings=[_unbound("undataAge"), _unbound("undataSex")],
                also_plausible=None,
                ask=None,
            ), None

    return asyncio.run(
        resolve_variable(
            "child mortality rate",
            entities=["Chad"],
            date_request=None,
            detect_query=detect_query,
            role_query="child mortality rate in Chad",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_UnboundAgeAndSexLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


class TestFamilyTotalRule:
    """The single member a result names is the family's total only when every member
    asserts each value it asserts on an unbound property."""

    def test_a_lone_member_of_a_set_binding_warns_that_it_is_not_the_total(self):
        """Gender is bound to Female and Male and the family holds the female member alone.
        It asserts one value of the two the binding states, so the answer is narrower than
        the question and says so."""
        shape = _person_shape()
        shape = replace(shape, sv_arc_facts={_PERSON_FEMALE: shape.sv_arc_facts[_PERSON_FEMALE]})

        result = _resolve_at_recipient(
            shape, [_gender("set", "Female", "Male")], _graph_with_data_for(_PERSON_FEMALE)
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == [_PERSON_FEMALE]
        assert result.family_total_absent == FamilyTotalAbsent(
            total=None, shown=_PERSON_FEMALE, places=(_RECIPIENT,)
        )

    def test_a_family_without_a_both_sexes_member_warns(self):
        """Age is bound and the family holds only the male and female members, so the
        male one answers as one part of the family."""
        members = {
            _UN_MALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-M"),
            _UN_FEMALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-F"),
        }
        age = SlotBindingDraft(
            axis="how", property_dcid="undataAge", kind="value", value_dcids=["undata/AGE-Y0T4"]
        )
        result = _resolve_at_recipient(
            _child_mortality_shape(members),
            [age, _unbound("undataSex")],
            _graph_with_data_for(*members),
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == [_UN_MALE_U5]
        assert result.family_total_absent == FamilyTotalAbsent(
            total=None, shown=_UN_MALE_U5, places=(_RECIPIENT,)
        )

    def test_a_narrower_member_listed_first_by_recall_warns(self):
        """Recall lists the neonatal member ahead of the under-5 one. Both carry the same
        constraint keys, so recall order elects the neonatal member, with the notice."""
        members = {
            _UN_NEONATAL: _un_arcs(age="undata/AGE-M0", sex=None),
            _UN_U5: _un_arcs(age="undata/AGE-Y0T4", sex=None),
            _UN_MALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-M"),
        }
        result = _resolve_at_recipient(
            _child_mortality_shape(members),
            [_unbound("undataAge"), _unbound("undataSex")],
            _graph_with_data_for(*members),
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == [_UN_NEONATAL]
        assert result.family_total_absent == FamilyTotalAbsent(
            total=None, shown=_UN_NEONATAL, places=(_RECIPIENT,)
        )

    def test_a_recall_sample_without_the_root_warns(self):
        shape = _person_shape()
        del shape.sv_arc_facts[_PERSON_FEMALE]
        rural = "Count_Person_Female_Rural"
        shape.sv_arc_facts[rural] = _person_arcs(
            gender="Female", placeOfResidenceClassification="Rural"
        )
        result = _resolve_at_recipient(
            shape,
            [_gender("value", "Female")],
            _graph_with_data_for(_PERSON_FEMALE_URBAN, rural),
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == [_PERSON_FEMALE_URBAN]
        assert result.family_total_absent == FamilyTotalAbsent(
            total=None, shown=_PERSON_FEMALE_URBAN, places=(_RECIPIENT,)
        )

    def test_the_root_of_a_value_binding_answers_without_a_notice(self):
        result = _resolve_at_recipient(
            _person_shape(),
            [_gender("value", "Female")],
            _graph_with_data_for(_PERSON_FEMALE_URBAN, _PERSON_FEMALE),
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == [_PERSON_FEMALE]
        assert result.family_total_absent is None

    @pytest.mark.parametrize(
        ("with_data", "has_data"),
        [
            pytest.param((_UN_MALE_U5, _UN_FEMALE_U5), False, id="only_dropped_members_have_data"),
            pytest.param((_UN_MALE_U5, _UN_FEMALE_U5, _UN_NEONATAL), True, id="all_have_data"),
        ],
    )
    def test_a_kept_member_that_rolls_up_none_of_the_dropped_ones_warns(self, with_data, has_data):
        """The neonatal member's age-only keys dominate both under-5 members, so the mint
        keeps it alone. It answers or is flagged, and either way the result says it is not
        the family's total."""
        members = {
            _UN_MALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-M"),
            _UN_FEMALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-F"),
            _UN_NEONATAL: _un_arcs(age="undata/AGE-M0", sex=None),
        }
        result = _resolve_at_recipient(
            _child_mortality_shape(members),
            [_unbound("undataAge"), _unbound("undataSex")],
            _graph_with_data_for(*with_data),
        )

        assert isinstance(result, Materialised)
        assert result.has_data is has_data
        assert result.sv_dcids == [_UN_NEONATAL]
        assert result.family_total_absent == FamilyTotalAbsent(
            total=None, shown=_UN_NEONATAL, places=(_RECIPIENT,)
        )

    def test_the_no_place_path_warns_when_the_named_member_is_not_the_total(self):
        members = {
            _UN_MALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-M"),
            _UN_FEMALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-F"),
        }
        result = materialise(
            _child_mortality_shape(members),
            [_unbound("undataSex")],
            {},
            graph=_graph_with_data_for(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == [_UN_MALE_U5]
        assert result.family_total_absent == FamilyTotalAbsent(
            total=None, shown=_UN_MALE_U5, places=()
        )

    def test_an_unconstrained_root_with_arcs_answers_as_the_total_without_a_notice(self):
        """The root's arcs carry no constraintProperties arc, which reads as no constraint,
        so it dominates the female member recall lists first and is the family's total."""
        shape = replace(
            _person_shape(),
            sv_arc_facts={
                _PERSON_FEMALE: _person_arcs(gender="Female"),
                "Count_Person": {
                    "populationType": {"nodes": [{"dcid": "Person"}]},
                    "measuredProperty": {"nodes": [{"dcid": "count"}]},
                    "statType": {"nodes": [{"dcid": "measuredValue"}]},
                },
            },
        )
        result = _resolve_at_recipient(
            shape, [_unbound("gender")], _graph_with_data_for(_PERSON_FEMALE, "Count_Person")
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == ["Count_Person"]
        assert result.family_total_absent is None

    def test_a_member_with_an_empty_arcs_mapping_answers_in_recall_order_with_a_notice(self):
        shape = replace(
            _person_shape(),
            sv_arc_facts={"sv_unknown": {}, _PERSON_FEMALE: _person_arcs(gender="Female")},
        )
        result = _resolve_at_recipient(
            shape, [_unbound("gender")], _graph_with_data_for("sv_unknown", _PERSON_FEMALE)
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == ["sv_unknown"]
        assert result.family_total_absent == FamilyTotalAbsent(
            total=None, shown="sv_unknown", places=(_RECIPIENT,)
        )

    @pytest.mark.parametrize(
        ("with_data", "has_data"),
        [
            pytest.param((_PERSON_FEMALE_URBAN,), True, id="one_member_has_data"),
            pytest.param((), False, id="no_member_has_data"),
        ],
    )
    def test_a_set_binding_naming_one_member_warns(self, with_data, has_data):
        """The set binding collects or flags a single urban member, which is narrower than
        its family, as under a value binding."""
        shape = replace(
            _person_shape(),
            sv_arc_facts={
                _PERSON_FEMALE_URBAN: _person_arcs(
                    gender="Female", placeOfResidenceClassification="Urban"
                ),
                "Count_Person_Female_Rural": _person_arcs(
                    gender="Female", placeOfResidenceClassification="Rural"
                ),
            },
        )
        result = _resolve_at_recipient(
            shape, [_gender("set", "Female", "Male")], _graph_with_data_for(*with_data)
        )

        assert isinstance(result, Materialised)
        assert result.has_data is has_data
        assert result.sv_dcids == [_PERSON_FEMALE_URBAN]
        assert result.family_total_absent == FamilyTotalAbsent(
            total=None, shown=_PERSON_FEMALE_URBAN, places=(_RECIPIENT,)
        )

    def test_a_set_binding_naming_several_members_carries_no_notice(self):
        result = _resolve_at_recipient(
            _person_shape(),
            [_gender("set", "Female", "Male")],
            _graph_with_data_for(_PERSON_FEMALE, _PERSON_MALE),
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == [_PERSON_FEMALE, _PERSON_MALE]
        assert result.family_total_absent is None

    def test_one_member_answering_a_binding_that_states_several_values_warns(self):
        """The binding states Female and Male, and the answer names the male member alone."""
        shape = replace(
            _person_shape(),
            sv_arc_facts={
                _PERSON_MALE: _person_arcs(gender="Male"),
                _PERSON_FEMALE: _person_arcs(gender="Female"),
            },
        )
        result = _resolve_at_recipient(
            shape,
            [_gender("value", "Female", "Male")],
            _graph_with_data_for(_PERSON_MALE, _PERSON_FEMALE),
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == [_PERSON_MALE]
        assert result.family_total_absent == FamilyTotalAbsent(
            total=None, shown=_PERSON_MALE, places=(_RECIPIENT,)
        )

    def test_a_member_asserting_the_stated_value_is_the_total_ahead_of_a_descendant(self):
        """A hierarchical binding admits the stated value and its descendants. Only the
        member asserting the stated value is the total, and it sorts first."""
        from search.engine.retrieval.member_facts import ArcsMemberFacts
        from search.engine.retrieval.standard import _elect_family

        member_facts = {
            "child": ArcsMemberFacts(_person_arcs(sector="Child")),
            "parent": ArcsMemberFacts(_person_arcs(sector="Parent")),
        }

        family = _elect_family(list(member_facts), member_facts, {"sector": ["Parent"]})

        assert family.order == ("parent", "child")
        assert family.totals == frozenset({"parent"})

    def test_a_member_with_an_unresolved_constraint_keeps_its_keys_and_dominates(self):
        """Domination compares property keys, as in the mint, so the member with an
        unresolved gender drops the one asserting gender and residence. It is no total."""
        from search.engine.retrieval.member_facts import ArcsMemberFacts
        from search.engine.retrieval.standard import _elect_family

        member_facts = {
            "unresolved": ArcsMemberFacts(_unresolved_gender_arcs()),
            "urban": ArcsMemberFacts(
                _person_arcs(gender="Female", placeOfResidenceClassification="Urban")
            ),
        }

        family = _elect_family(list(member_facts), member_facts, {})

        assert family.order == ("unresolved",)
        assert family.totals == frozenset()

    def test_a_member_with_no_arcs_mapping_is_unknown(self):
        from search.engine.retrieval.member_facts import ArcsMemberFacts
        from search.engine.retrieval.standard import _elect_family

        member_facts = {"sv_none": ArcsMemberFacts(arcs=None)}  # ty: ignore[invalid-argument-type]

        family = _elect_family(list(member_facts), member_facts, {})

        assert family.order == ("sv_none",)
        assert family.totals == frozenset()

    def test_a_member_with_unknown_constraints_is_no_total_and_dominates_nothing(self):
        """A member with no facts stays electable in recall order. It never stands for the
        family, and it leaves the constrained member beside it in the kept set."""
        from search.engine.retrieval.member_facts import ArcsMemberFacts
        from search.engine.retrieval.standard import _elect_family

        member_facts = {
            "sv_unknown": ArcsMemberFacts(arcs={}),
            _UN_MALE_U5: ArcsMemberFacts(_un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-M")),
        }

        family = _elect_family(list(member_facts), member_facts, {})

        assert family.order == ("sv_unknown", _UN_MALE_U5)
        assert family.totals == frozenset()


# The member a single-collect answer elects is confirmed on the years the plan reads.


def _old_and_new_members(date_request):
    def facet(facet_id: str, years: range) -> dict:
        return {
            "facetId": facet_id,
            "provenanceId": _SOURCE,
            "earliestDate": str(years[0]),
            "latestDate": str(years[-1]),
            "obsCount": len(years),
            "dates": [str(y) for y in years],
            "values": [1.0] * len(years),
        }

    shape = ShapeDraft(
        shape_id="test_old_and_new_members",
        label="test shape",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={"sv_old": {}, "sv_new": {}},
        representative_score=0.9,
    )
    graph = FakeGraph(
        obs={
            "sv_old|country/KEN": [facet("fOld", range(2000, 2016))],
            "sv_new|country/KEN": [facet("fNew", range(2015, 2025))],
        }
    )
    return materialise(
        shape,
        [],
        {"observationAbout": ["country/KEN"]},
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=date_request,
        coverage=empty_coverage_snapshot(),
    )


def test_a_single_collect_elects_the_first_member_with_points_in_the_years_read():
    """`sv_old` holds 2000 to 2015 and `sv_new` 2015 to 2024 at Kenya. Dateless elects
    `sv_old`, the first confirmed; "since 2021" confirms only `sv_new`, which holds points
    there, and serves it."""
    dateless = _old_and_new_members(None)
    assert isinstance(dateless, Materialised) and dateless.sv_dcids == ["sv_old"]

    since = _old_and_new_members(
        DateRequest(window=TimeWindow(start_year=2021, end_year=None), latest=False)
    )
    assert isinstance(since, Materialised) and since.has_data
    assert since.sv_dcids == ["sv_new"]
    assert since.coverage.window == TimeWindow(start_year=2021, end_year=2024)
