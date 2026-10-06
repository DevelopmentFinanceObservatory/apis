"""Test per-part restatements across all statuses and both input doors (#593).

Round trips preserve identity entities, requested windows, and spec IDs; folded
parts retain their own restatements.
"""

from __future__ import annotations

import httpx
import pytest

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise.rate_tables import RateTableCache
from search.engine.conjoin import BINDING_FOLDED, assemble_region, combine_regions
from search.engine.envelope import now_ms
from search.engine.extract import DateRequest, dates_to_request
from search.engine.fanout import (
    _date_request_to_time_text,
    _leg_restatement,
    _placeholder_no_data,
    _region_has_directional_entity,
    _success_directional,
)
from search.engine.region import RegionResult
from search.engine.role_text import DoorRole
from search.engine.time_bind import parse_time_text
from search.models import (
    Entity,
    EntityRoleDirectional,
    EntityRoleSubject,
    GraphRef,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    ResolveRequest,
    TimeWindow,
    Unsupported,
)
from tests.engine._builders import conjoin_spec as _spec
from tests.engine._builders import shape_five_tuple as _shape
from tests.engine._harness import (
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    make_request,
    materialise_double_from_obs,
    offline_resolve,
)
from tests.fixtures import FakeGraph, _load_json
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)

_PERSON_SHAPE = _shape("Person", "count")

_SDMX_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"
_RECIPE = "ConversionRecipeDAC"
_EUR_SOURCE_PROVENANCE = "dc/base/FertilityEurSource"


def _sdmx_with_rate_anchor(obs: dict) -> MaterialiseSdmxClient:
    """Serve fixture observations and an exchange-rate table holding only `CurrencyUSD`.

    Omit the EUR rate so conversion raises `RateCoverageGapError`.
    """
    bodies: dict[str, str] = {}
    for key, facet_dicts in obs.items():
        sv, entity = key.split("|", 1)
        body = bodies.get(sv, _csv_header(["observationAbout"]))
        for i, facet_dict in enumerate(facet_dicts):
            fixture = ObsFacetFixture(
                facet_id=facet_dict.get("facetId") or f"f{i}",
                unit=facet_dict.get("unit"),
                provenance_id=facet_dict.get("provenanceId"),
            )
            body += _csv_row(
                stat_var=sv,
                role_dcids=[entity],
                fixture=fixture,
                date_str=facet_dict["dates"][0],
                value=facet_dict["values"][0],
            )
        bodies[sv] = body

    def observations(request: httpx.Request) -> httpx.Response:
        sv = request.url.params.get("c[variableMeasured]", "")
        body = bodies.get(sv, _csv_header(["observationAbout"]))
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    server = RateTableServer(
        recipe_table_bodies(_RECIPE, rates={"CurrencyUSD": {2020: 1.0}}),
        fallback=observations,
    )
    return server.client(base_url=MATERIALISE_BASE_URL)


def _gr(dcid: str) -> GraphRef:
    return GraphRef(dcid=dcid, label=dcid)


def _subject_entity(dcid: str) -> Entity:
    return Entity(ref=_gr(dcid), role=EntityRoleSubject())


def _directional_entity(dcid: str, axis: str) -> Entity:
    return Entity(ref=_gr(dcid), role=EntityRoleDirectional(role=_gr("recipient"), axis=axis))


# ---------------------------------------------------------------------------
# _date_request_to_time_text: canonical, not verbatim
# ---------------------------------------------------------------------------


class TestDateRequestToTimeText:
    @pytest.mark.parametrize(
        ("date_request", "expected"),
        [
            pytest.param(None, None, id="dateless"),
            pytest.param(
                DateRequest(window=TimeWindow(start_year=2010, end_year=None), latest=False),
                "since 2010",
                id="open-end",
            ),
            # The grammar has "since YYYY" for an open end, not an open start.
            pytest.param(
                DateRequest(window=TimeWindow(start_year=None, end_year=2010), latest=False),
                None,
                id="open-start",
            ),
            # The parser rejects reversed "YYYY to YYYY" ranges.
            pytest.param(
                DateRequest(window=TimeWindow(start_year=2020, end_year=2010), latest=False),
                None,
                id="reversed-window",
            ),
        ],
    )
    def test_time_text_for_absent_open_or_reversed_windows(self, date_request, expected):
        assert _date_request_to_time_text(date_request) == expected

    def test_latest(self):
        assert _date_request_to_time_text(DateRequest(window=None, latest=True)) == "latest"

    def test_point_window(self):
        dr = DateRequest(window=TimeWindow(start_year=2020, end_year=2020), latest=False)
        assert _date_request_to_time_text(dr) == "2020"

    def test_closed_range(self):
        dr = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)
        assert _date_request_to_time_text(dr) == "2015 to 2020"

    @pytest.mark.parametrize(
        "date_request",
        [
            DateRequest(window=TimeWindow(start_year=2020, end_year=2020), latest=False),
            DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False),
            DateRequest(window=TimeWindow(start_year=2010, end_year=None), latest=False),
            DateRequest(window=None, latest=True),
        ],
    )
    def test_round_trips_through_parse_time_text(self, date_request):
        """The parsed door reconstructs the same `DateRequest` from this text."""
        text = _date_request_to_time_text(date_request)
        assert text is not None
        dates, warning = parse_time_text(text, current_year=2024)
        assert warning is None
        assert dates_to_request(dates) == date_request


# ---------------------------------------------------------------------------
# _leg_restatement: the generic per-leg ParsedPart builder
# ---------------------------------------------------------------------------


class TestLegRestatement:
    def test_builds_the_parsed_part_shape(self):
        date_request = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)
        restatement = _leg_restatement(
            "GDP",
            entities=["Brazil", "Angola"],
            date_request=date_request,
            unit_text="US dollars",
            source_text="World Bank",
        )
        assert restatement == ParsedPart(
            variable_text="GDP",
            entities=[ParsedEntity(text="Brazil"), ParsedEntity(text="Angola")],
            time_text="2015 to 2020",
            unit_text="US dollars",
            source_text="World Bank",
        )

    def test_no_entities_is_none_not_an_empty_list(self):
        restatement = _leg_restatement(
            "GDP", entities=[], date_request=None, unit_text=None, source_text=None
        )
        assert restatement is not None
        assert restatement.entities is None

    def test_a_leg_input_ParsedPart_cannot_express_yields_no_restatement(self):
        # Empty variable text fails ParsedPart validation.
        restatement = _leg_restatement(
            "", entities=[], date_request=None, unit_text=None, source_text=None
        )
        assert restatement is None

    def test_directional_leg_with_no_axis_for_the_entity_yields_no_restatement(self):
        # Directional wording without a door role cannot be represented in the echo.
        restatement = _leg_restatement(
            "aid",
            entities=["France"],
            date_request=None,
            unit_text=None,
            source_text=None,
            directional=True,
        )
        assert restatement is None

    def test_undirected_leg_with_entities_still_yields_one(self):
        # A subject-bound entity remains representable without a direction.
        restatement = _leg_restatement(
            "population",
            entities=["Brazil"],
            date_request=None,
            unit_text=None,
            source_text=None,
            directional=False,
        )
        assert restatement == ParsedPart(
            variable_text="population", entities=[ParsedEntity(text="Brazil")]
        )


# ---------------------------------------------------------------------------
# _region_has_directional_entity: the SUCCESS-path directional signal
# ---------------------------------------------------------------------------


class TestRegionHasDirectionalEntity:
    def test_false_when_every_spec_entity_is_subject(self):
        spec = _spec(_PERSON_SHAPE, "Val", sv_dcid="sv", entities=[_subject_entity("country/BRA")])
        region = RegionResult(
            variable_text="population",
            status="definite",
            specs=(spec,),
            no_data_reason=None,
            warnings=(),
            timing_by_step={},
        )
        assert _region_has_directional_entity(region) is False

    def test_true_when_a_candidate_spec_binds_a_directional_entity(self):
        subject_spec = _spec(
            _PERSON_SHAPE, "ValA", sv_dcid="sv", entities=[_subject_entity("country/BRA")]
        )
        directional_spec = _spec(
            _PERSON_SHAPE,
            "ValB",
            sv_dcid="sv",
            entities=[_directional_entity("country/FRA", "origin")],
        )
        region = RegionResult(
            variable_text="aid",
            status="candidates",
            specs=(subject_spec, directional_spec),
            no_data_reason=None,
            warnings=(),
            timing_by_step={},
        )
        assert _region_has_directional_entity(region) is True

    def test_true_from_flagged_match_when_specs_is_empty(self):
        # A no_data outcome carries no specs at all, but flagged_match is
        # still a Spec resolved against the same entities -- the directional
        # misrepresentation this guards against can hide there too.
        flagged = _spec(
            _PERSON_SHAPE,
            "Val",
            sv_dcid="sv",
            entities=[_directional_entity("country/FRA", "destination")],
        )
        region = RegionResult(
            variable_text="aid",
            status="no_data",
            specs=(),
            no_data_reason="no_observations",
            warnings=(),
            timing_by_step={},
            flagged_match=(flagged,),
        )
        assert _region_has_directional_entity(region) is True

    def test_none_when_nothing_ever_bound_anything(self):
        # No Spec means direction is unknown, which triggers conservative withholding.
        region = RegionResult(
            variable_text="aid",
            status="no_data",
            specs=(),
            no_data_reason="entity_not_resolved",
            warnings=(),
            timing_by_step={},
            unresolved_entity_names=("Nowhereland",),
        )
        assert _region_has_directional_entity(region) is None


# ---------------------------------------------------------------------------
# _success_directional combines bound entity roles with the role-query fallback.
# ---------------------------------------------------------------------------


class TestSuccessDirectional:
    def test_no_spec_anywhere_withholds_the_restatement_when_entities_are_bound(self):
        # Direction remains possible, so the echo is withheld as unnameable.
        region = RegionResult(
            variable_text="aid",
            status="no_data",
            specs=(),
            no_data_reason="entity_not_resolved",
            warnings=(),
            timing_by_step={},
            unresolved_entity_names=("Atlantis",),
        )
        directional = _success_directional(region, role_query="aid from USA to Atlantis")
        assert directional is True
        restatement = _leg_restatement(
            "aid",
            entities=["USA"],
            date_request=None,
            unit_text=None,
            source_text=None,
            directional=directional,
        )
        assert restatement is None

    def test_no_spec_anywhere_still_carries_the_restatement_without_entities(self):
        # With no entities, there is no direction to misrepresent.
        region = RegionResult(
            variable_text="aid",
            status="no_data",
            specs=(),
            no_data_reason="entity_not_resolved",
            warnings=(),
            timing_by_step={},
            unresolved_entity_names=("Atlantis",),
        )
        directional = _success_directional(region, role_query="aid from USA to Atlantis")
        assert directional is True
        restatement = _leg_restatement(
            "aid",
            entities=[],
            date_request=None,
            unit_text=None,
            source_text=None,
            directional=directional,
        )
        assert restatement is not None
        assert restatement.entities is None

    def test_subject_only_specs_still_carry_the_restatement(self):
        # Subject-bound entities remain representable.
        spec = _spec(_PERSON_SHAPE, "Val", sv_dcid="sv", entities=[_subject_entity("country/BRA")])
        region = RegionResult(
            variable_text="population",
            status="definite",
            specs=(spec,),
            no_data_reason=None,
            warnings=(),
            timing_by_step={},
        )
        directional = _success_directional(region, role_query="population of Brazil")
        assert directional is False
        restatement = _leg_restatement(
            "population",
            entities=["Brazil"],
            date_request=None,
            unit_text=None,
            source_text=None,
            directional=directional,
        )
        assert restatement == ParsedPart(
            variable_text="population", entities=[ParsedEntity(text="Brazil")]
        )

    def test_directional_binding_found_in_a_spec_withholds_the_echo(self):
        # A directional Spec prevents an ambiguous echo.
        directional_spec = _spec(
            _PERSON_SHAPE,
            "ValB",
            sv_dcid="sv",
            entities=[_directional_entity("country/FRA", "origin")],
        )
        region = RegionResult(
            variable_text="aid",
            status="candidates",
            specs=(directional_spec,),
            no_data_reason=None,
            warnings=(),
            timing_by_step={},
        )
        directional = _success_directional(region, role_query="aid from France")
        assert directional is True
        restatement = _leg_restatement(
            "aid",
            entities=["France"],
            date_request=None,
            unit_text=None,
            source_text=None,
            directional=directional,
        )
        assert restatement is None

    @pytest.mark.parametrize(
        ("status", "answered_roles", "echoes"),
        [
            # A recipient-only debt total places the bare place by structure.
            ("definite", ("financingRecipient",), True),
            # A two-role answer placed it from the wording.
            ("definite", ("financingProvider", "financingRecipient"), False),
            # Candidates can hold a two-role reading beside the first one.
            ("candidates", ("financingRecipient",), False),
        ],
    )
    def test_a_definite_one_role_answer_echoes_a_place_named_without_a_direction(
        self, status, answered_roles, echoes
    ):
        spec = _spec(
            _PERSON_SHAPE,
            "ValB",
            sv_dcid="sv",
            entities=[_directional_entity("country/GHA", "destination")],
        )
        region = RegionResult(
            variable_text="debt service",
            status=status,
            specs=(spec,),
            no_data_reason=None,
            warnings=(),
            timing_by_step={},
            answered_roles=answered_roles,
        )

        restatement = _leg_restatement(
            "debt service",
            entities=["Ghana"],
            date_request=None,
            unit_text=None,
            source_text=None,
            directional=_success_directional(region, role_query="debt service of Ghana"),
        )

        if echoes:
            assert restatement == ParsedPart(
                variable_text="debt service", entities=[ParsedEntity(text="Ghana")]
            )
        else:
            assert restatement is None


# ---------------------------------------------------------------------------
# _placeholder_no_data: the conservative rule for a leg that never resolved
# ---------------------------------------------------------------------------


class TestPlaceholderNoDataDirectionalRule:
    def test_entities_plus_role_query_with_no_door_axis_yields_no_restatement(self):
        region = _placeholder_no_data(
            "aid", 0, entities=["France"], role_query="how much aid comes from France"
        )
        assert region.restatement is None

    def test_no_role_query_still_yields_a_restatement(self):
        region = _placeholder_no_data("population", 0, entities=["Brazil"], role_query=None)
        assert region.restatement == ParsedPart(
            variable_text="population", entities=[ParsedEntity(text="Brazil")]
        )

    def test_a_door_role_supplies_the_axis_even_with_a_role_query(self):
        # The door named France's axis, so the placeholder echoes it. The
        # role_query alone does not decide.
        region = _placeholder_no_data(
            "aid",
            0,
            entities=["France"],
            role_query="how much aid comes from France",
            entity_roles={"France": [DoorRole(axis="origin", span="from France")]},
        )
        assert region.restatement == ParsedPart(
            variable_text="aid",
            entities=[ParsedEntity(text="France", role_text="from")],
        )

    def test_no_entities_yields_a_restatement_even_with_a_role_query(self):
        region = _placeholder_no_data(
            "aid", 0, entities=[], role_query="how much aid comes from France"
        )
        assert region.restatement is not None
        assert region.restatement.entities is None


# ---------------------------------------------------------------------------
# Part construction: restatement reaches every status, and survives folding
# as the ORIGINAL part's own value, never the representative's.
# ---------------------------------------------------------------------------


def _rr(
    variable_text: str,
    status: str,
    spec=None,
    *,
    restatement: ParsedPart | None = None,
    earliest_index: int = 0,
    no_data_reason: str | None = None,
    extra_specs: tuple = (),
    unsupported: Unsupported | None = None,
) -> RegionResult:
    if status == "definite":
        specs: tuple = (spec,)
    elif status == "candidates":
        specs = (spec, *extra_specs)
    else:
        specs = ()
    return RegionResult(
        variable_text=variable_text,
        status=status,
        specs=specs,
        no_data_reason=no_data_reason,
        warnings=(),
        timing_by_step={},
        earliest_index=earliest_index,
        restatement=restatement,
        unsupported=unsupported,
    )


def _assemble(region: RegionResult):
    return assemble_region(
        region,
        extra_warnings=[],
        start_ms=now_ms(),
        engine_build="test",
        max_candidates=10,
    )


class TestPartCarriesRestatementOnEveryStatus:
    def test_definite_part(self):
        restatement = ParsedPart(variable_text="GDP", entities=[ParsedEntity(text="Brazil")])
        spec = _spec(_PERSON_SHAPE, "Val", sv_dcid="sv")
        response = _assemble(_rr("GDP", "definite", spec, restatement=restatement))
        assert response.parts[0].restatement == restatement

    def test_no_data_part(self):
        # This part has no Spec to derive its restatement from.
        restatement = ParsedPart(variable_text="GDP", entities=[ParsedEntity(text="Nowhereland")])
        response = _assemble(
            _rr(
                "GDP",
                "no_data",
                restatement=restatement,
                no_data_reason="variable_not_resolved",
            )
        )
        assert response.parts[0].restatement == restatement

    def test_candidates_part(self):
        restatement = ParsedPart(variable_text="GDP")
        spec_a = _spec(_PERSON_SHAPE, "ValA", sv_dcid="sv")
        spec_b = _spec(_PERSON_SHAPE, "ValB", sv_dcid="sv")
        response = _assemble(
            _rr("GDP", "candidates", spec_a, extra_specs=(spec_b,), restatement=restatement)
        )
        assert response.parts[0].restatement == restatement

    def test_unsupported_part(self):
        restatement = ParsedPart(variable_text="GDP", unit_text="PPP dollars")
        response = _assemble(
            _rr(
                "GDP",
                "unsupported",
                restatement=restatement,
                unsupported=Unsupported(
                    code="unit_request_unsupported",
                    pointer="/identity/facets/unit",
                    detail="PPP terms cannot be expressed.",
                    definition={},
                ),
            )
        )
        assert response.parts[0].restatement == restatement


class TestFoldedPartKeepsItsOwnRestatement:
    def test_fold_does_not_overwrite_restatement_with_the_representatives(self):
        # Same shape and StatVar with different values collapse into one spec.
        spec_a = _spec(_PERSON_SHAPE, "ValA", sv_dcid="sv")
        spec_b = _spec(_PERSON_SHAPE, "ValB", sv_dcid="sv")
        restatement_a = ParsedPart(variable_text="var A", entities=[ParsedEntity(text="Brazil")])
        restatement_b = ParsedPart(variable_text="var B", entities=[ParsedEntity(text="Angola")])
        response, _effective = combine_regions(
            [
                _rr("var A", "definite", spec_a, earliest_index=0, restatement=restatement_a),
                _rr("var B", "definite", spec_b, earliest_index=1, restatement=restatement_b),
            ],
            query="test",
            extra_warnings=[],
            start_ms=now_ms(),
            engine_build="test",
            max_candidates=10,
        )
        first, second = response.parts
        # Confirm the parts share the folded spec.
        assert first.spec.spec_id == second.spec.spec_id
        assert any(w.code == BINDING_FOLDED for w in second.warnings)
        assert first.restatement == restatement_a
        assert second.restatement == restatement_b


# ---------------------------------------------------------------------------
# End-to-end round trip: real engine, both doors, shared fixtures only.
# ---------------------------------------------------------------------------


class TestRawTextRestatementRoundTrip:
    def test_multi_part_query_echoes_broadcast_entities_and_each_leg_own_variable_text(self):
        # Raw text shares query-level entities across legs, but each echo keeps its variable text.
        result = offline_resolve(make_request("population and fertility rate in Brazil"))
        population, fertility = result.parts
        assert population.restatement == ParsedPart(
            variable_text="population",
            entities=[ParsedEntity(text="Brazil")],
        )
        assert fertility.restatement == ParsedPart(
            variable_text="fertility rate",
            entities=[ParsedEntity(text="Brazil")],
        )

    def test_restatement_round_trips_through_the_parsed_door(self):
        """Feeding a raw_text part's own restatement back through ParsedInput
        reproduces the SAME resolution -- identical identity entities,
        requested_window, and spec_id -- even though the two doors never
        shared a single word of query text."""
        raw = offline_resolve(make_request("population and fertility rate in Brazil"))
        for part in raw.parts:
            assert part.restatement is not None
            reparsed = offline_resolve(ResolveRequest(input=ParsedInput(parts=[part.restatement])))
            assert reparsed.status == "definite"
            reparsed_def = reparsed.parts[0].spec.definition
            original_def = part.spec.definition
            assert reparsed_def.identity.entities == original_def.identity.entities
            assert reparsed_def.identity.requested_window == original_def.identity.requested_window
            assert reparsed_def.spec_id == original_def.spec_id


class TestParsedDoorRestatementRoundTrip:
    def test_differing_entities_and_window_each_echo_their_own(self):
        """The parsed door has genuine per-part entities and time windows to
        begin with, so each part's echo carries its OWN values rather than
        the query's, and role_text round-trips too."""
        parts = [
            ParsedPart(
                variable_text="population",
                entities=[ParsedEntity(text="Brazil")],
                time_text="2015 to 2020",
            ),
            ParsedPart(
                variable_text="fertility rate",
                entities=[ParsedEntity(text="Kenya", role_text="to")],
                time_text="since 2010",
            ),
        ]
        result = offline_resolve(ResolveRequest(input=ParsedInput(parts=parts)))
        first, second = result.parts
        assert first.restatement == parts[0]
        assert second.restatement == parts[1]

    def test_no_data_part_echoes_its_unit_text_too(self):
        """A part that never reaches a spec (rate_coverage_gap, here) still
        carries its own restatement, unit_text included -- the case this
        field exists for."""
        # The EUR facet and declared nominal recipe pass conversion checks.
        # The absent EUR exchange rate refuses the served year as `rate_coverage_gap`.
        obs = {
            "FertilityRate_Person_Female|country/KEN": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": _EUR_SOURCE_PROVENANCE,
                    "dates": ["2020"],
                    "values": [1.0],
                    "unit": "CurrencyEUR",
                }
            ]
        }
        materialise_graph, _unused_sdmx = materialise_double_from_obs(obs)
        materialise_sdmx = _sdmx_with_rate_anchor(obs)
        materialise_graph._arcs = {
            "FertilityRate_Person_Female": dict(CONVERTIBLE_STAT_VAR_ARCS),
            **conversion_arcs(
                recipe_id=_RECIPE,
                provenance_id=_EUR_SOURCE_PROVENANCE,
                currencies=["CurrencyUSD", "CurrencyEUR"],
            ),
        }
        parts = [
            ParsedPart(
                variable_text="fertility rate",
                entities=[ParsedEntity(text="Kenya")],
                unit_text="US dollars",
            )
        ]
        result = offline_resolve(
            ResolveRequest(input=ParsedInput(parts=parts)),
            # The unit decision reads the facets the answer read, so both graphs hold them.
            graph=FakeGraph(obs={**_load_json("graph_obs.json"), **obs}),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=RateTableCache(),
        )
        assert result.status == "no_data"
        assert result.parts[0].no_data.reason == "rate_coverage_gap"
        assert result.parts[0].restatement == parts[0]
