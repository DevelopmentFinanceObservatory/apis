"""Standard promote resubmits preserve resolved sources and apply concrete
requested windows to coverage, observations, and no-data responses.
"""

from __future__ import annotations

import pytest

from search.engine.conjoin import _build_part
from search.engine.region import (
    NO_MEMBER_HAS_DATA_AT_PLACE,
    PLACE_SCOPE_DEFAULTED,
    resolve_spec_resubmit,
)
from search.engine.region.state import EngineOptions
from search.models import (
    BindingValue,
    CoverageExact,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
    TimeWindow,
)
from tests.engine._base_identity import (
    GDP_SHAPE,
    PERSON_COUNT_SHAPE,
    make_base,
    make_identity,
    subject_entity,
)
from tests.engine._builders import (
    gref as _gref,
)
from tests.engine._builders import (
    unbound_slot as _unbound_slot,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import materialise_double_from_obs
from tests.fixtures import FakeGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# GDP standard SV/shape constants (mirrors test_path_c.py)
_GDP_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_GDP_ENTITY_DCID = "country/IND"


def _when_slot_value(start: int, end: int) -> Slot:
    """A posted when-slot Slot bound to a concrete TimeWindow (mirrors
    test_path_c.py's own helper of the same name)."""
    return Slot(
        key=SlotKey(axis="when", property=None, label="time window"),
        binding=BindingValue(
            value=SlotValue(
                ref=None,
                value_kind="time_window",
                time_window=TimeWindow(start_year=start, end_year=end),
            )
        ),
    )


# ---------------------------------------------------------------------------
# Multi-recipient BindingSet
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# resolved_sources propagated on resubmit
# ---------------------------------------------------------------------------


class TestResubmitResolvedSources:
    """source_refs must be forwarded to build_spec in both resubmit functions."""

    def test_standard_promote_no_provenance_empty_sources(self):
        """Standard promote: observations with no provenanceId → resolved_sources == [].

        Amount_EconomicActivity_GrossDomesticProduction_Nominal|country/IND in
        graph_obs.json has no provenanceId, so resolved_sources must be [].
        """
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            GDP_SHAPE,
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(
                GDP_SHAPE,
                # activitySource is this SV's only shape-declared what/how
                # slot key; required present (unbound is fine).
                [_unbound_slot("how", "activitySource", "activity source")],
                entities=entities,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        region = resolve_spec_resubmit(
            inp=inp, graph=FakeGraph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
        )
        assert region.status == "definite"
        assert region.specs[0].resolved.trace.resolved_sources == [], (
            "resolved_sources must be [] when no provenanceId is present in facets"
        )


# ---------------------------------------------------------------------------
# A concrete requested_window (not the "latest" sentinel) reaches retrieval
# ---------------------------------------------------------------------------

# Custom GDP observations spanning 2010-2020, wider than the requested window
# under test (2015-2018) -- proves the window actually narrows what's returned
# rather than the request being ignored.
_WINDOW_GDP_OBS = {
    f"{_GDP_SV_DCID}|{_GDP_ENTITY_DCID}": [
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 4,
            "provenanceId": "dc/base/WorldBank",
            "observations": [
                {"date": "2010", "value": 1.0},
                {"date": "2015", "value": 2.0},
                {"date": "2018", "value": 3.0},
                {"date": "2020", "value": 4.0},
            ],
        }
    ],
}

_WINDOW_REQUESTED = TimeWindow(start_year=2015, end_year=2018)

_ACTIVITY_SOURCE_SLOT = _unbound_slot("how", "activitySource", "activity source")


def _source_slot_value(dcid: str) -> Slot:
    """A posted source-axis Slot bound to `dcid` (mirrors
    test_resubmit_source_switch.py's own helper of the same name)."""
    return Slot(
        key=SlotKey(axis="source", property=None, label="source"),
        binding=BindingValue(value=SlotValue(ref=_gref(dcid, "IGNORED"), value_kind="source")),
    )


# A single facet, bound entirely OUTSIDE _WINDOW_REQUESTED (2015-2018) --
# provenanceId is a real graph_nodes.json dcid so the source-slot rebind
# resolves a real label.
_SOURCE_BOUND_OUT_OF_WINDOW_OBS = {
    f"{_GDP_SV_DCID}|{_GDP_ENTITY_DCID}": [
        {
            "earliestDate": "2005",
            "latestDate": "2010",
            "obsCount": 2,
            "provenanceId": "dc/base/WorldDevelopmentIndicators",
            "observations": [
                {"date": "2005", "value": 1.0},
                {"date": "2010", "value": 2.0},
            ],
        }
    ],
}


def _resolve_standard_with_window(
    inp: ResubmitIdentityPayload, *, include_observations: bool = False
):
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_WINDOW_GDP_OBS)
    return resolve_spec_resubmit(
        inp=inp,
        graph=FakeGraph(obs=_WINDOW_GDP_OBS),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(
            include_observations=include_observations,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        ),
    )


def _standard_window_inp(
    *,
    requested_window: TimeWindow | None,
    extra_slots: list[Slot] | None = None,
) -> ResubmitIdentityPayload:
    entities = [subject_entity(_GDP_ENTITY_DCID)]
    base_spec_id, base_identity = make_base(GDP_SHAPE, entities=entities)
    return ResubmitIdentityPayload(
        identity=make_identity(
            GDP_SHAPE,
            [_ACTIVITY_SOURCE_SLOT, *(extra_slots or [])],
            entities=entities,
            requested_window=requested_window,
        ),
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        resolved_stat_var_dcids=[_GDP_SV_DCID],
    )


def test_resubmit_property_label_miss_is_humanized_in_slots() -> None:
    """A promote-path property-label miss never exposes a raw property dcid."""

    property_dcid = "placeOfResidenceClassification"

    class MissingResidenceLabelGraph(FakeGraph):
        def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
            return {
                dcid: label
                for dcid, label in super().node_labels_batch(dcids).items()
                if dcid != property_dcid
            }

    entities = [subject_entity(_GDP_ENTITY_DCID)]
    residence_slot = _unbound_slot("how", property_dcid, property_dcid)
    base_spec_id, base_identity = make_base(
        PERSON_COUNT_SHAPE, slots=[residence_slot], entities=entities
    )
    inp = ResubmitIdentityPayload(
        identity=make_identity(PERSON_COUNT_SHAPE, [residence_slot], entities=entities),
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        resolved_stat_var_dcids=["Count_Person_Urban"],
    )
    region = resolve_spec_resubmit(
        inp=inp,
        graph=MissingResidenceLabelGraph(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )

    assert region.status == "definite", region.no_data_reason
    assert region.specs[0].spec_id == base_spec_id
    slot = next(
        slot
        for slot in region.specs[0].resolved.slots
        if slot.key.property and slot.key.property.dcid == property_dcid
    )
    assert slot.key.property.label == property_dcid
    assert slot.key.label == "place of residence classification"


class TestRequestedWindowAppliedToStandardRetrieval:
    """ResubmitIdentityPayload.requested_window carrying a concrete TimeWindow (not
    the "latest" sentinel) with no matching when-slot binding must apply
    directly to retrieval, not resolve against the full history."""

    def test_coverage_and_trace_reflect_the_requested_window(self):
        inp = _standard_window_inp(requested_window=_WINDOW_REQUESTED)
        region = _resolve_standard_with_window(inp)
        assert region.status == "definite", region.no_data_reason
        spec = region.specs[0]
        assert spec.resolved.coverage.window == _WINDOW_REQUESTED
        assert spec.resolved.trace.applied_window == _WINDOW_REQUESTED
        assert spec.resolved.trace.date_source == "query"
        when_slot = next(s for s in spec.resolved.slots if s.key.axis == "when")
        assert isinstance(when_slot.binding, BindingValue)
        assert when_slot.binding.value.time_window == _WINDOW_REQUESTED

    def test_observations_are_filtered_to_the_requested_window(self):
        inp = _standard_window_inp(requested_window=_WINDOW_REQUESTED)
        region = _resolve_standard_with_window(inp, include_observations=True)
        assert region.status == "definite", region.no_data_reason
        spec = region.specs[0]
        assert spec.resolved.observations is not None
        series = spec.resolved.observations.series[0]
        assert [p.date for p in series.points] == ["2015", "2018"]
        for point in series.points:
            year = int(point.date[:4])
            assert _WINDOW_REQUESTED.start_year <= year <= _WINDOW_REQUESTED.end_year

    def test_spec_id_matches_independent_identity_hash(self):
        """The spec id hashes requested_window directly, not the when-slot."""
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        inp = _standard_window_inp(requested_window=_WINDOW_REQUESTED)
        region = _resolve_standard_with_window(inp)
        assert region.status == "definite", region.no_data_reason
        expected_spec_id, _ = make_base(
            GDP_SHAPE,
            [_ACTIVITY_SOURCE_SLOT],
            entities=entities,
            requested_window=_WINDOW_REQUESTED,
        )
        assert region.specs[0].spec_id == expected_spec_id

    def test_requested_window_wins_over_a_differing_posted_when_slot(self):
        """The explicit requested_window takes precedence over a conflicting
        when-slot."""
        differing_when_slot = _when_slot_value(2000, 2005)
        inp = _standard_window_inp(
            requested_window=_WINDOW_REQUESTED, extra_slots=[differing_when_slot]
        )
        region = _resolve_standard_with_window(inp)
        assert region.status == "definite", region.no_data_reason
        spec = region.specs[0]
        assert spec.resolved.coverage.window == _WINDOW_REQUESTED
        assert spec.resolved.trace.applied_window == _WINDOW_REQUESTED

    def test_reversed_requested_window_refuses_in_the_identity(self):
        """Identity normalisation rejects a reversed window before
        materialisation (#572)."""
        with pytest.raises(ValueError, match="start_year 2018 is greater than end_year 2015"):
            _standard_window_inp(requested_window=TimeWindow(start_year=2018, end_year=2015))

    @pytest.mark.parametrize(
        ("extra_slots", "include_observations", "use_materialise_double"),
        [
            pytest.param(
                [_source_slot_value("dc/base/WorldDevelopmentIndicators")],
                True,
                True,
                id="facet_filtered_preview",
            ),
            pytest.param([], True, False, id="unfiltered_preview"),
            pytest.param([], False, False, id="unfiltered_no_preview"),
        ],
    )
    def test_requested_window_exhaustion_flips_to_no_data(
        self,
        extra_slots: list[Slot],
        include_observations: bool,
        use_materialise_double: bool,
    ):
        """Window exhaustion returns no_observations with filtered and
        unfiltered previews, or with previews disabled."""
        inp = _standard_window_inp(
            requested_window=_WINDOW_REQUESTED,
            extra_slots=extra_slots,
        )
        if use_materialise_double:
            materialise_graph, materialise_sdmx = materialise_double_from_obs(
                _SOURCE_BOUND_OUT_OF_WINDOW_OBS
            )
            region = resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(obs=_SOURCE_BOUND_OUT_OF_WINDOW_OBS),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
                options=EngineOptions(
                    include_observations=include_observations,
                    materialise_graph=materialise_graph,
                    materialise_sdmx=materialise_sdmx,
                ),
            )
        else:
            region = resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(obs=_SOURCE_BOUND_OUT_OF_WINDOW_OBS),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
                options=EngineOptions(include_observations=include_observations),
            )
        assert region.status == "no_data"
        assert region.no_data_reason == "no_observations"

    def test_requested_window_exhaustion_names_the_span_the_series_runs(self):
        """The series holds 2005 and 2010 and the resubmit asks for 2015 to 2018: the
        sentence names the years read and the span, as the first resolve's does."""
        region = resolve_spec_resubmit(
            inp=_standard_window_inp(requested_window=_WINDOW_REQUESTED),
            graph=FakeGraph(obs=_SOURCE_BOUND_OUT_OF_WINDOW_OBS),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(include_observations=False),
        )
        part = _build_part(
            region,
            representative=region,
            folded=False,
            include_sentence=True,
            max_candidates=5,
            include_observations=False,
            include_skills=False,
            skills_index=None,
            envelope_warnings=[],
        )
        assert region.no_data_reason == "no_observations"
        assert region.gate is None
        assert region.window_gap_fact is not None
        assert region.window_gap_fact.data_years == (2005, 2010)
        # A resubmit holds no words of the request, so the text quotes no years asked.
        assert [(d.code, d.asked, d.text) for d in part.departures] == [
            ("year_unavailable", None, "The data covers only 2005 to 2010.")
        ]

    def test_no_where_slot_still_reports_the_requested_window(self):
        """The entity-less resubmit uses the Earth/Country scope. Its concrete
        requested window applies to the entity probe."""
        base_spec_id, base_identity = make_base(GDP_SHAPE, entities=[])
        inp = ResubmitIdentityPayload(
            identity=make_identity(
                GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], requested_window=_WINDOW_REQUESTED
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        region = _resolve_standard_with_window(inp)
        assert region.status == "definite", region.no_data_reason
        assert any(w.code == "place_scope_defaulted" for w in region.warnings)
        spec = region.specs[0]
        assert spec.resolved.coverage.kind == "exact"
        assert spec.resolved.coverage.window == _WINDOW_REQUESTED
        assert spec.resolved.trace.applied_window == _WINDOW_REQUESTED

    def test_flagged_unavailable_carrier_reports_the_requested_window(self):
        """The flagged unavailable carrier retains the requested window."""
        inp = _standard_window_inp(requested_window=_WINDOW_REQUESTED)
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(obs={}),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
        assert region.status == "no_data"
        assert region.no_data_reason == "no_observations"
        assert region.flagged_match is not None
        flagged = region.flagged_match[0]
        assert region.gate == NO_MEMBER_HAS_DATA_AT_PLACE
        assert flagged.resolved.coverage.window == _WINDOW_REQUESTED
        assert flagged.resolved.trace.applied_window == _WINDOW_REQUESTED

    def test_flagged_unavailable_carrier_skips_coverage_probe(self, monkeypatch):
        """A non-definite carrier skips option-coverage probes."""

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("standard_option_coverage must not be reached")

        monkeypatch.setattr(
            "search.engine.region.option_coverage.standard_option_coverage", _fail_if_called
        )

        def _fail_if_facet_called(*args, **kwargs):
            raise AssertionError("facet_options probe must not be reached")

        monkeypatch.setattr(
            "search.engine.region.option_coverage._facet_options", _fail_if_facet_called
        )

        inp = _standard_window_inp(requested_window=_WINDOW_REQUESTED)
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(obs={}),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(include_option_coverage=True),
        )
        assert region.status == "no_data"
        assert region.no_data_reason == "no_observations"
        assert region.gate == NO_MEMBER_HAS_DATA_AT_PLACE


# ---------------------------------------------------------------------------
# A where-slot bound to the literal place "Earth" is an ordinary bound
# single-place query
# ---------------------------------------------------------------------------

_EARTH_BOUND_GDP_OBS = {
    f"{_GDP_SV_DCID}|Earth": [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/WorldBank",
            "observations": [{"date": "2020", "value": 1.0}],
        }
    ],
}


class TestResubmitBoundToLiteralEarth:
    """A posted where-slot naming the literal place dcid "Earth" is a genuine
    bound single place, not the default scope's own earth-mode fallback
    (which only applies when NO where-slot is posted at all, the placeless
    branch of region/place_context.py's `place_context`). It must resolve
    exactly like any other single bound country."""

    def test_earth_bound_where_slot_resolves_as_ordinary_bound_place(self):
        entities = [subject_entity("Earth", label="World")]
        base_spec_id, base_identity = make_base(GDP_SHAPE, entities=entities)
        inp = ResubmitIdentityPayload(
            identity=make_identity(GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], entities=entities),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(obs=_EARTH_BOUND_GDP_OBS),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert region.status == "definite", region.no_data_reason
        spec = region.specs[0]
        assert isinstance(spec.resolved.coverage, CoverageExact)
        labels = {d.label for d in spec.resolved.coverage.dimensions}
        assert "sources" in labels
        assert "worldwide" not in labels
        assert not any(w.code == PLACE_SCOPE_DEFAULTED for w in region.warnings)
