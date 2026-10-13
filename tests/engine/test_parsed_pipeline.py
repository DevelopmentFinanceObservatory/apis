"""Tests parsed-input resolution through the shared pipeline, including units, roles, and pins.

Fixtures use offline graph and SDMX doubles. Scenarios reuse the "fertility rate"/"Kenya"
text so the recorded LLM fixtures still match."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch

import httpx
import pytest

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from entity_resolution import Resolution
from materialise.rate_tables import RateTableCache
from materialise.recipes import RecipeNotDeclaredError
from search.engine.bind import _UniversalBindOutput
from search.engine.config import settings
from search.engine.core import resolve_async, resolve_events
from search.engine.entry.parsed import (
    _dedupe_parts_with_indices,
    confirm_part_pins,
    interpret_parts,
)
from search.engine.errors import EngineInputError, GraphInfraError
from search.engine.interpret import PinnedResolved
from search.engine.quantity_read import QuantityOutput
from search.engine.region import RegionResult
from search.engine.unit.base_year import TABLE_PROVENANCES
from search.models import (
    IdentityUnitRequest,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    ResolveRequest,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    make_request,
    materialise_double_from_obs,
    offline_resolve,
)
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, FakeLLM, _load_json, make_resolver
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)

_FERTILITY_SV = "FertilityRate_Person_Female"
_FERTILITY_KEN_KEY = f"{_FERTILITY_SV}|country/KEN"

_SDMX_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"
_RECIPE = "ConversionRecipeDAC"
_EUR_SOURCE_PROVENANCE = "dc/base/FertilityEurSource"


def _sdmx_with_rate_anchor(obs: dict) -> MaterialiseSdmxClient:
    """Serves per-StatVar CSVs and an exchange-rate table holding only `CurrencyUSD`.

    The absent EUR rate produces `RateCoverageGapError`."""
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


# A bound unit request runs region/convertibility.py's own
# materialise-side decision -- a native `{USD,current}` match needs
# check_nominal to pass, which needs this real live StatVar declared
# Nominal, and a conversion needs its populationType in the recipe's
# `flow_population_types` (materialise_double_from_obs's own FakeGraphReader
# carries no arcs at all by default). Applied to every materialise_graph double this
# file builds for a "US dollars"-bound part.
_FERTILITY_MATERIALISE_ARCS = {
    _FERTILITY_SV: dict(CONVERTIBLE_STAT_VAR_ARCS),
    "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
    "CurrencyEUR": {"typeOf": ("CurrencyUnit",)},
}


def _resolve_parsed(
    parts: list[ParsedPart],
    *,
    raw_query: str | None = None,
    graph: FakeGraph | None = None,
    sdmx: FakeSdmxClient | None = None,
    materialise_graph=None,
    materialise_sdmx=None,
    rate_cache: RateTableCache | None = None,
):
    request = ResolveRequest(input=ParsedInput(parts=parts, raw_query=raw_query))
    return asyncio.run(
        resolve_async(
            request,
            graph=graph if graph is not None else FakeGraph(),
            llm=FakeLLM(),
            sdmx=sdmx if sdmx is not None else FakeSdmxClient(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=rate_cache,
        )
    )


def _fertility_ken_usd_obs() -> dict:
    """Builds a USD facet matching `US dollars` and one 2020 point for window filtering."""
    return {
        _FERTILITY_KEN_KEY: [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": "dc/base/FertilityUsdSource",
                "dates": ["2020"],
                "values": [123.4],
                "unit": "CurrencyUSD",
            }
        ],
    }


def _graph_with_usd_unit_on_fertility_kenya() -> FakeGraph:
    """Adds a USD facet to the Kenya fertility fixture."""
    obs = _load_json("graph_obs.json")
    return FakeGraph(obs={**obs, **_fertility_ken_usd_obs()})


def _fertility_kenya_part(
    *,
    time_text: str | None = None,
    unit_text: str | None = None,
    source_text: str | None = None,
    role_text: str | None = None,
) -> ParsedPart:
    return ParsedPart(
        variable_text="fertility rate",
        entities=[ParsedEntity(text="Kenya", role_text=role_text)],
        time_text=time_text,
        unit_text=unit_text,
        source_text=source_text,
    )


class TestSinglePartResolvesThroughSharedPipeline:
    def test_single_part_resolves_definite(self):
        result = _resolve_parsed([_fertility_kenya_part()])
        assert result.status == "definite"
        assert result.query_echo.entry_path == "parsed"
        assert result.query_echo.extract_skipped is True
        assert [p.variable_text for p in result.parts] == ["fertility rate"]

    def test_convergence_with_raw_text(self):
        """The same utterance through raw_text and parsed converges: same
        requested_window, same identity entity buckets, and (as a
        consequence of both) the same spec_id."""
        raw = offline_resolve(make_request("fertility rate in Kenya"), sdmx=FakeSdmxClient())
        parsed = _resolve_parsed([_fertility_kenya_part()], raw_query="fertility rate in Kenya")
        raw_def = raw.parts[0].spec.definition
        parsed_def = parsed.parts[0].spec.definition
        assert raw_def.identity.requested_window == parsed_def.identity.requested_window
        assert raw_def.identity.entities == parsed_def.identity.entities
        assert raw_def.spec_id == parsed_def.spec_id

    def test_raw_query_is_echo_only(self):
        """Different `raw_query` values echo verbatim and mint the same spec id."""
        one = _resolve_parsed([_fertility_kenya_part()], raw_query="first phrasing")
        two = _resolve_parsed(
            [_fertility_kenya_part()], raw_query="a completely different phrasing"
        )
        assert one.parts[0].spec.definition.spec_id == two.parts[0].spec.definition.spec_id
        assert one.query_echo.raw_query == "first phrasing"
        assert two.query_echo.raw_query == "a completely different phrasing"


class TestUnsupportedIdentityEntitiesGolden:
    """Unsupported definitions preserve entity refs from the definite identity."""

    def test_unsupported_definition_entities_match_definite_path(self):
        definite = _resolve_parsed([_fertility_kenya_part()])
        # PPP dollars cannot be represented by the request form, so this part is unsupported.
        # Table-expressible requests resolve against their data and may be definite or no_data.
        unsupported = _resolve_parsed([_fertility_kenya_part(unit_text="PPP dollars")])
        assert definite.status == "definite"
        assert unsupported.status == "unsupported"
        expected = {"observationAbout": ["country/KEN"]}
        assert definite.parts[0].spec.definition.identity.entities == expected
        assert unsupported.parts[0].unsupported.definition["identity"]["entities"] == expected


class TestRoleTextLexicon:
    def test_unrecognised_role_text_never_refuses(self):
        result = _resolve_parsed([_fertility_kenya_part(role_text="bogus")])
        assert result.status in ("definite", "candidates", "no_data")
        warnings = {w.code for w in result.parts[0].warnings}
        assert "role_text_not_recognised" in warnings

    def test_recognised_role_text_emits_no_warning(self):
        result = _resolve_parsed([_fertility_kenya_part(role_text="to")])
        assert not any(w.code == "role_text_not_recognised" for w in result.parts[0].warnings)

    def test_french_and_spanish_role_words_now_warn(self):
        """Issue #575: French and Spanish role words produce `role_text_not_recognised`."""
        for word in ("de", "à", "donateur", "donante", "destinataire", "destinatario"):
            result = _resolve_parsed([_fertility_kenya_part(role_text=word)])
            warnings = {w.code for w in result.parts[0].warnings}
            assert "role_text_not_recognised" in warnings, word

    def test_debt_and_trade_role_words_emit_no_warning(self):
        """English debt and trade role terms are recognised."""
        for word in ("creditor", "lender", "debtor", "borrower", "exporter", "importer"):
            result = _resolve_parsed([_fertility_kenya_part(role_text=word)])
            warnings = {w.code for w in result.parts[0].warnings}
            assert "role_text_not_recognised" not in warnings, word

    def test_unrecognised_role_text_warning_names_the_grammar(self):
        """The warning names accepted role words so callers can correct their phrasing."""
        result = _resolve_parsed([_fertility_kenya_part(role_text="bogus")])
        warnings = {w.code: w for w in result.parts[0].warnings}
        message = warnings["role_text_not_recognised"].message
        assert "'from'" in message and "'to'" in message
        assert "creditor" in message and "debtor" in message


class TestEntityLikePhrasingWarning:
    """Issue #586: entity-like variable text warns when the part has no structured entities."""

    def test_embedded_capitalised_word_warns_with_no_entities(self):
        result = _resolve_parsed([ParsedPart(variable_text="fertility rate in France")])
        warnings = {w.code: w for w in result.parts[0].warnings}
        assert "variable_text_entity_like" in warnings
        assert "France" in warnings["variable_text_entity_like"].message

    def test_structured_entities_suppress_the_warning(self):
        # The unrecorded text follows the same path in both cases; structured entities alone differ.
        # This isolates the input that suppresses the warning.
        result = _resolve_parsed(
            [
                ParsedPart(
                    variable_text="fertility rate in France", entities=[ParsedEntity(text="France")]
                )
            ]
        )
        assert not any(w.code == "variable_text_entity_like" for w in result.parts[0].warnings)

    def test_leading_word_capitalisation_alone_never_warns(self):
        result = _resolve_parsed([ParsedPart(variable_text="Fertility rate")])
        assert not any(w.code == "variable_text_entity_like" for w in result.parts[0].warnings)

    def test_all_caps_acronym_never_warns(self):
        result = _resolve_parsed([ParsedPart(variable_text="GDP as reported by OECD")])
        assert not any(w.code == "variable_text_entity_like" for w in result.parts[0].warnings)


class TestTimeTextBinding:
    def test_recognised_range_narrows_requested_window(self):
        result = _resolve_parsed([_fertility_kenya_part(time_text="2015 to 2024")])
        assert result.status == "definite"
        window = result.parts[0].spec.definition.identity.requested_window
        assert window.kind == "range"
        assert window.start_year == 2015
        assert window.end_year == 2024
        assert not any(w.code == "time_text_not_parsed" for w in result.parts[0].warnings)

    def test_unparseable_time_text_degrades_to_dateless(self):
        result = _resolve_parsed([_fertility_kenya_part(time_text="not a real date")])
        assert result.status == "definite"
        window = result.parts[0].spec.definition.identity.requested_window
        assert window.kind == "all"
        warnings = {w.code for w in result.parts[0].warnings}
        assert "time_text_not_parsed" in warnings
        # The parse warning identifies this case.
        # The window-default warning marks missing time text.
        assert "time_window_defaulted" not in warnings

    def test_french_and_spanish_since_phrasing_binds_an_open_window(self):
        """Issue #575: `depuis YYYY` and `desde YYYY` parse as open-ended ranges."""
        for phrase in ("depuis 2015", "desde 2015"):
            result = _resolve_parsed([_fertility_kenya_part(time_text=phrase)])
            assert result.status == "definite"
            window = result.parts[0].spec.definition.identity.requested_window
            assert window.kind == "range", phrase
            assert window.start_year == 2015, phrase
            warnings = {w.code for w in result.parts[0].warnings}
            assert "time_text_not_parsed" not in warnings, phrase

    @pytest.mark.parametrize(
        ("phrase", "window"),
        [
            ("between 2010 and the latest", ("range", 2010, None)),
            ("between 2010 and 2020", ("range", 2010, 2020)),
            ("from 2010 to 2020", ("range", 2010, 2020)),
            ("from 2010 until the most recent year available", ("range", 2010, None)),
            ("2010 to latest", ("range", 2010, None)),
            ("latest year available", ("latest", None, None)),
            ("the latest", ("latest", None, None)),
            ("between 2010 to 2020", None),
            ("from 2010 and 2020", None),
            ("2010 and 2020", None),
            ("2010 until 2020", None),
            ("since 2010 to 2020", None),
            ("between 2020 and 2010", None),
            ("2010 to present", None),
            ("today", None),
        ],
    )
    def test_lead_separator_and_end_grammar(self, phrase, window):
        """A lead fixes its separator, a latest end leaves the range open, and a phrase
        outside the grammar resolves dateless with `time_text_not_parsed`."""
        result = _resolve_parsed([_fertility_kenya_part(time_text=phrase)])
        assert result.status == "definite"
        requested = result.parts[0].spec.definition.identity.requested_window
        warnings = {w.code for w in result.parts[0].warnings}
        if window is None:
            assert requested.kind == "all"
            assert "time_text_not_parsed" in warnings
            return
        # A latest window carries no years.
        years = (getattr(requested, "start_year", None), getattr(requested, "end_year", None))
        assert (requested.kind, *years) == window
        assert "time_text_not_parsed" not in warnings

    def test_no_time_text_at_all_emits_time_window_defaulted(self):
        """Issue #577: missing `time_text` defaults the window and emits its warning."""
        result = _resolve_parsed([_fertility_kenya_part()])
        assert result.status == "definite"
        window = result.parts[0].spec.definition.identity.requested_window
        assert window.kind == "all"
        warnings = {w.code for w in result.parts[0].warnings}
        assert "time_window_defaulted" in warnings


class TestUnitTextGate:
    """Tests matched, unmatched, and inexpressible bound-unit requests."""

    def test_matched_unit_resolves_definite_with_request_basis(self):
        """The fixture's USD facet matches the `(USD, current)` request from `US dollars`."""
        materialise_graph, materialise_sdmx = materialise_double_from_obs(_fertility_ken_usd_obs())
        materialise_graph._arcs = _FERTILITY_MATERIALISE_ARCS
        result = _resolve_parsed(
            [_fertility_kenya_part(unit_text="US dollars")],
            graph=_graph_with_usd_unit_on_fertility_kenya(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=RateTableCache(),
        )
        assert result.status == "definite"
        facets = result.parts[0].spec.resolved.facets
        assert facets["unit"].ref == "CurrencyUSD"
        assert facets["unit"].basis == "bound"
        assert facets["unit"].request_basis == {"currency": "bound", "prices": "declared_default"}
        # Prices complete from the contract's declared rule, so nothing warns.
        assert not any(w.code == "default_not_declared" for w in result.parts[0].warnings)

    def test_matched_unit_zero_in_window_points_refuses_no_data_no_observations(self):
        """The USD facet has one 2020 point, outside the requested 2010-2012 window."""
        materialise_graph, materialise_sdmx = materialise_double_from_obs(_fertility_ken_usd_obs())
        materialise_graph._arcs = _FERTILITY_MATERIALISE_ARCS
        result = _resolve_parsed(
            [_fertility_kenya_part(unit_text="US dollars", time_text="2010 to 2012")],
            graph=_graph_with_usd_unit_on_fertility_kenya(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=RateTableCache(),
        )
        assert result.status == "no_data"
        assert result.parts[0].no_data.reason == "no_observations"
        assert [(d.code, d.asked) for d in result.parts[0].departures] == [
            ("year_unavailable", "2010 to 2012")
        ]

    def test_an_unmatched_unit_outside_the_years_asked_answers_for_the_window_first(self):
        """The only facet is EUR, with one 2020 point, and the request asks for US dollars in
        2010 to 2012. The unit decision reads every year and finds a declared conversion, and
        `no_observations` ranks before `facet_not_available` in the resolution order, so the
        answer is the window's."""
        obs = {
            _FERTILITY_KEN_KEY: [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": "dc/base/FertilityEurSource",
                    "dates": ["2020"],
                    "values": [1.0],
                    "unit": "CurrencyEUR",
                }
            ]
        }
        materialise_graph, _unused_sdmx = materialise_double_from_obs(obs)
        materialise_graph._arcs = {
            **_FERTILITY_MATERIALISE_ARCS,
            **conversion_arcs(
                recipe_id=_RECIPE,
                provenance_id=_EUR_SOURCE_PROVENANCE,
                currencies=["CurrencyUSD", "CurrencyEUR"],
            ),
        }
        result = _resolve_parsed(
            [_fertility_kenya_part(unit_text="US dollars", time_text="2010 to 2012")],
            graph=FakeGraph(obs={**_load_json("graph_obs.json"), **obs}),
            materialise_graph=materialise_graph,
            materialise_sdmx=_sdmx_with_rate_anchor(obs),
            rate_cache=RateTableCache(),
        )
        assert result.parts[0].no_data.reason == "no_observations"
        assert [(d.code, d.asked) for d in result.parts[0].departures] == [
            ("year_unavailable", "2010 to 2012")
        ]

    def test_unmatched_unit_refuses_no_data_rate_coverage_gap(self):
        """`US dollars` requests USD, while the only facet is EUR and the rate table holds no
        EUR rate. The unit decision finds the served year's missing rate before any read and
        refuses with the cause a conversion of the same rows raises."""
        obs = {
            _FERTILITY_KEN_KEY: [
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
            **_FERTILITY_MATERIALISE_ARCS,
            **conversion_arcs(
                recipe_id=_RECIPE,
                provenance_id=_EUR_SOURCE_PROVENANCE,
                currencies=["CurrencyUSD", "CurrencyEUR"],
            ),
        }
        result = _resolve_parsed(
            [_fertility_kenya_part(unit_text="US dollars")],
            # The unit decision reads the facets the answer read, so both graphs hold them.
            graph=FakeGraph(obs={**_load_json("graph_obs.json"), **obs}),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=RateTableCache(),
        )
        assert result.status == "no_data"
        no_data = result.parts[0].no_data
        assert no_data.reason == "rate_coverage_gap"
        assert no_data.cause is not None
        assert (no_data.cause.code, no_data.cause.entities) == (
            "rate_missing",
            ["CurrencyEUR@2020"],
        )

    def test_no_unit_at_all_is_unsupported_facet_binding_unsupported(self):
        """The facet has no unit, so the USD binding fails on the unit axis.

        This case returns `facet_binding_unsupported`. The prior case carries EUR."""
        obs = {
            _FERTILITY_KEN_KEY: [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": "dc/base/FertilityUsdSource",
                    "dates": ["2020"],
                    "values": [1.0],
                }
            ]
        }
        materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
        materialise_graph._arcs = _FERTILITY_MATERIALISE_ARCS
        result = _resolve_parsed(
            [_fertility_kenya_part(unit_text="US dollars")],
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        )
        assert result.status == "unsupported"
        assert result.parts[0].unsupported.code == "facet_binding_unsupported"
        assert result.parts[0].unsupported.pointer == "/identity/facets/unit"

    def test_a_currency_no_native_unit_carries_reaches_the_conversion_decision(self):
        """Japanese yen is expressible though no native unit carries it. The per-member
        conversion decision then refuses the fertility rate, which declares no Nominal."""
        materialise_graph, materialise_sdmx = materialise_double_from_obs(_fertility_ken_usd_obs())
        materialise_graph._arcs = {**_FERTILITY_MATERIALISE_ARCS, _FERTILITY_SV: {}}
        result = _resolve_parsed(
            [_fertility_kenya_part(unit_text="Japanese yen")],
            graph=_graph_with_usd_unit_on_fertility_kenya(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=RateTableCache(),
        )
        assert result.status == "unsupported"
        unsupported = result.parts[0].unsupported
        assert unsupported.code == "recipe_not_declared"
        assert unsupported.cause.code == "nominal_undeclared"
        # The pointer resolves into the returned definition and names the unit request.
        assert unsupported.pointer == "/identity/facets/unit"
        assert unsupported.definition["identity"]["facets"]["unit"] == {
            "state": "value",
            "request": {"currency": "JPY", "prices": "current"},
        }

    def test_unresolved_unit_refuses_no_data(self):
        """The malformed unit phrase fails parsing and returns `unit_not_resolved`."""
        result = _resolve_parsed([_fertility_kenya_part(unit_text="xyz garbled nonsense")])
        assert result.status == "no_data"
        assert result.parts[0].no_data.reason == "unit_not_resolved"

    @pytest.mark.parametrize("unit_text", ["PPP dollars", "per capita"], ids=["ppp", "per-capita"])
    def test_unrepresentable_unit_is_unsupported_not_definite(self, unit_text):
        """PPP terms are unrepresentable, so the request returns unsupported. So is a
        per-capita unit (dq-crs-043): the unit gate answers it, not the denominator check
        a per-capita variable meets at materialise."""
        result = _resolve_parsed([_fertility_kenya_part(unit_text=unit_text)])
        assert result.status == "unsupported"
        assert result.parts[0].unsupported.code == "unit_request_unsupported"
        assert result.parts[0].unsupported.pointer == "/identity/facets/unit"


def _two_unit_facets_obs() -> dict:
    """Builds USD (3 points, 2018-2020) and EUR (2 points, 2015-2016) facets.

    USD wins unconstrained by observation count, matching the `US dollars` binding."""
    return {
        _FERTILITY_KEN_KEY: [
            {
                "earliestDate": "2018",
                "latestDate": "2020",
                "obsCount": 3,
                "dates": ["2018", "2019", "2020"],
                "values": [100.0, 110.0, 123.4],
                "unit": "CurrencyUSD",
                "provenanceId": "dc/base/UsdSource",
                "importName": "UsdSource",
            },
            {
                "earliestDate": "2015",
                "latestDate": "2016",
                "obsCount": 2,
                "dates": ["2015", "2016"],
                "values": [10.0, 20.0],
                "unit": "CurrencyEUR",
                "provenanceId": "dc/base/EurSource",
                "importName": "EurSource",
            },
        ],
    }


def _graph_with_two_unit_facets_on_fertility_kenya() -> FakeGraph:
    """Adds the two unit facets and their provenance labels to the Kenya fixture."""
    nodes = _load_json("graph_nodes.json")
    obs = _load_json("graph_obs.json")
    return FakeGraph(
        nodes={
            **nodes,
            "dc/base/UsdSource": {"label": "USD Source"},
            "dc/base/EurSource": {"label": "EUR Source"},
        },
        obs={**obs, **_two_unit_facets_obs()},
    )


class TestUnitFacetNarrowsResponseMetadata:
    """A bound unit narrows coverage, sources, and date range to the served facets."""

    def test_matched_unit_narrows_coverage_sources_and_date_range(self):
        graph = _graph_with_two_unit_facets_on_fertility_kenya()
        materialise_graph, materialise_sdmx = materialise_double_from_obs(_two_unit_facets_obs())
        materialise_graph._arcs = _FERTILITY_MATERIALISE_ARCS
        result = _resolve_parsed(
            [_fertility_kenya_part(unit_text="US dollars")],
            graph=graph,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=RateTableCache(),
        )
        assert result.status == "definite"
        resolved = result.parts[0].spec.resolved

        # Coverage uses the USD facet's 3 observations.
        assert resolved.coverage.kind == "exact"
        assert resolved.coverage.observation_count == 3

        # The source list contains the USD facet's provenance.
        sources = {s.ref for s in resolved.trace.resolved_sources}
        assert sources == {"dc/base/UsdSource"}

        # The date range follows the USD facet's 2018-2020 span.
        date_range = resolved.members[0].data_date_range
        assert date_range is not None
        assert date_range.start == "2018"
        assert date_range.end == "2020"

    def test_no_unit_bound_counts_and_cites_the_served_facet(self):
        """Without `unit_text`, coverage counts the facet the preview serves, USD's three
        points, the sources list its provenance alone, and the StatVar's date range spans
        the served facet's 2018 to 2020."""
        graph = _graph_with_two_unit_facets_on_fertility_kenya()
        result = _resolve_parsed(
            [_fertility_kenya_part()],
            graph=graph,
        )
        assert result.status == "definite"
        resolved = result.parts[0].spec.resolved

        assert resolved.coverage.observation_count == 3

        sources = [s.ref for s in resolved.trace.resolved_sources]
        assert sources == ["dc/base/UsdSource"]

        date_range = resolved.members[0].data_date_range
        assert date_range is not None
        assert date_range.start == "2018"
        assert date_range.end == "2020"


class TestUnitBoundSentenceEventSpecIdAgreement:
    """Both sentence events and the final Spec share one id for a bound-unit request."""

    @staticmethod
    def _stream_unit_bound() -> tuple[list[str], str]:
        async def _body() -> tuple[list[str], str]:
            sentence_ids: list[str] = []
            final_id = ""
            materialise_graph, materialise_sdmx = materialise_double_from_obs(
                _fertility_ken_usd_obs()
            )
            materialise_graph._arcs = _FERTILITY_MATERIALISE_ARCS
            async for ev in resolve_events(
                ResolveRequest(
                    input=ParsedInput(parts=[_fertility_kenya_part(unit_text="US dollars")])
                ),
                graph=_graph_with_usd_unit_on_fertility_kenya(),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(),
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=RateTableCache(),
            ):
                if ev.kind == "sentence":
                    sentence_ids.append(ev.spec_id)
                elif ev.kind == "response":
                    final_id = ev.response.parts[0].spec.spec_id
            return sentence_ids, final_id

        return asyncio.run(_body())

    def test_first_and_definite_sentence_events_agree_with_the_final_spec(self):
        sentence_ids, final_id = self._stream_unit_bound()
        assert sentence_ids, "expected at least one sentence event"
        assert final_id, "expected a definite response carrying a Spec"
        assert len(set(sentence_ids)) == 1, (
            f"every sentence event for this unit-bound variable must carry one id, "
            f"got {sentence_ids}"
        )
        assert sentence_ids[0] == final_id


class TestSourceText:
    def test_source_text_reads_as_a_source_mention(self):
        """The deprecated `source_text` folds into one included mention, which compiles
        against the catalogue like any other: words naming no admitted source refuse
        source_not_resolved, quoting them."""
        result = _resolve_parsed([_fertility_kenya_part(source_text="OECD")])
        assert result.status == "no_data"
        (part,) = result.parts
        assert part.no_data.reason == "source_not_resolved"
        assert part.no_data.source_terms == ["OECD"]
        assert part.restatement is not None
        assert [m.text for m in part.restatement.sources] == ["OECD"]
        assert part.restatement.source_text is None


class _CountingGraph(FakeGraph):
    """Counts `node_arcs_many` reads of the conversion-table provenances."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.table_reads = 0

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        if set(dcids) == set(TABLE_PROVENANCES):
            self.table_reads += 1
        return super().node_arcs_many(dcids, properties=properties, deadline=deadline)


def _graph_with_conversion_tables(base_year: str) -> _CountingGraph:
    obs = _load_json("graph_obs.json")
    nodes = _load_json("graph_nodes.json")
    table = {
        "label": "Conversion tables",
        "arcs": {
            "lastDataRefreshDate": {"nodes": [{"value": "2026-10-01"}]},
            "deflatorBaseYear": {"nodes": [{"value": base_year}]},
        },
    }
    return _CountingGraph(
        nodes={**nodes, **dict.fromkeys(TABLE_PROVENANCES, table)},
        obs={**obs, **_fertility_ken_usd_obs()},
    )


class TestUnitCompletionFromDeclaredSources:
    """A constant-price phrase that names no year completes from the tables' base year."""

    def _decided_requests(self, graph, unit_texts):
        decided = []

        def _capture(*args, unit_request, **kwargs):
            decided.append(unit_request)
            raise RecipeNotDeclaredError("captured", cause="nominal_undeclared")

        materialise_graph, materialise_sdmx = materialise_double_from_obs(_fertility_ken_usd_obs())
        materialise_graph._arcs = _FERTILITY_MATERIALISE_ARCS
        with patch("search.engine.region.phase_answer.decide_part_unit", _capture):
            for unit_text in unit_texts:
                _resolve_parsed(
                    [_fertility_kenya_part(unit_text=unit_text)],
                    graph=graph,
                    materialise_graph=materialise_graph,
                    materialise_sdmx=materialise_sdmx,
                    rate_cache=RateTableCache(),
                )
        return decided

    def test_constant_prices_complete_to_the_declared_base_year_and_default_currency(self):
        graph = _graph_with_conversion_tables("2024")
        (request,) = self._decided_requests(graph, ["constant prices"])
        assert request == IdentityUnitRequest(currency="USD", prices="constant", base_year=2024)

    def test_the_declared_base_year_is_read_once_per_graph_client(self):
        graph = _graph_with_conversion_tables("2024")
        decided = self._decided_requests(graph, ["constant prices", "constant euros"])
        assert [r.base_year for r in decided] == [2024, 2024]
        assert graph.table_reads == 1

    def test_a_named_year_never_reads_the_tables(self):
        graph = _graph_with_conversion_tables("2024")
        (request,) = self._decided_requests(graph, ["constant 2015 prices"])
        assert request.base_year == 2015
        assert graph.table_reads == 0

    def test_the_default_currency_comes_from_search_configuration(self, monkeypatch):
        monkeypatch.setattr(settings(), "unit_default_currency", "EUR")
        graph = _graph_with_conversion_tables("2024")
        (request,) = self._decided_requests(graph, ["current prices"])
        assert request == IdentityUnitRequest(currency="EUR", prices="current")


class TestUnitSourcePrecedence:
    def test_an_unresolved_source_refuses_before_the_unit_is_read(self):
        """A part's sources compile before anything else resolves, so words naming no
        source refuse ahead of a unit the request grammar cannot express (PPP dollars)."""
        result = _resolve_parsed(
            [_fertility_kenya_part(unit_text="PPP dollars", source_text="OECD")]
        )
        assert result.status == "no_data"
        assert result.parts[0].no_data.reason == "source_not_resolved"
        assert not [w for w in result.parts[0].warnings if w.code == "facet_binding_unsupported"]


class TestPartDedupe:
    def test_byte_identical_parts_collapse_to_one_leg(self):
        result = _resolve_parsed([_fertility_kenya_part(), _fertility_kenya_part()])
        assert result.status == "definite"
        # Collapsed before the fan-out: only one leg ever ran, so parts
        # (built from the deduped part list) carries a single entry and no
        # ambiguity warning is possible (there is nothing to disambiguate).
        assert [p.variable_text for p in result.parts] == ["fertility rate"]
        assert result.diagnostics.warnings == []

    @pytest.mark.parametrize("separate", [False, True])
    def test_a_part_marked_separate_never_folds_into_a_same_shape_part(self, separate):
        """Kenya's and Nigeria's fertility rate share a shape and fold into one spec, unless
        the second part is marked separate. Then each answers its own spec, neither
        carries binding_folded, and the second's restatement states the mark."""
        kenya = _fertility_kenya_part()
        nigeria = ParsedPart(
            variable_text="fertility rate",
            entities=[ParsedEntity(text="Nigeria")],
            separate=separate,
        )
        first, second = _resolve_parsed([kenya, nigeria]).parts
        assert [first.status, second.status] == ["definite", "definite"]
        assert (first.spec.spec_id == second.spec.spec_id) is not separate
        assert any(w.code == "binding_folded" for w in second.warnings) is not separate
        assert second.restatement.separate is separate

    def test_parts_sharing_variable_text_but_differing_fields_stay_separate(self):
        """Different time windows keep same-variable parts separate and produce distinct specs."""
        result = _resolve_parsed(
            [
                _fertility_kenya_part(time_text="2015 to 2020"),
                _fertility_kenya_part(time_text="2021 to 2024"),
            ]
        )
        assert [p.variable_text for p in result.parts] == ["fertility rate", "fertility rate"]
        first, second = result.parts
        assert first.spec.spec_id != second.spec.spec_id
        first_window = first.spec.definition.identity.requested_window
        second_window = second.spec.definition.identity.requested_window
        assert (first_window.start_year, first_window.end_year) == (2015, 2020)
        assert (second_window.start_year, second_window.end_year) == (2021, 2024)
        assert not any(w.code == "binding_folded" for w in first.warnings)
        assert not any(w.code == "binding_folded" for w in second.warnings)

    def test_parts_sharing_everything_but_unit_text_stay_separate(self):
        """Different `unit_text` values stay separate and preserve their distinct refusals.

        "Francs" names three currencies. PPP dollars cannot be expressed by the request
        grammar."""
        result = _resolve_parsed(
            [
                _fertility_kenya_part(unit_text="francs"),
                _fertility_kenya_part(unit_text="PPP dollars"),
            ]
        )
        assert [p.variable_text for p in result.parts] == ["fertility rate", "fertility rate"]
        assert result.status == "mixed"
        assert result.parts[0].no_data.reason == "unit_not_resolved"
        assert result.parts[1].unsupported.code == "unit_request_unsupported"

    def test_byte_identical_dedupe_indices_reflect_deduped_ordinals(self):
        """With submitted parts `[A, A, B]`, B's warning uses its deduplicated index, 1."""
        result = _resolve_parsed(
            [
                _fertility_kenya_part(),
                _fertility_kenya_part(),
                ParsedPart(
                    variable_text="population",
                    entities=[ParsedEntity(text="Kenya", role_text="bogus")],
                ),
            ]
        )
        assert [p.variable_text for p in result.parts] == ["fertility rate", "population"]
        assert any(w.code == "role_text_not_recognised" for w in result.parts[1].warnings)

    def test_dcid_pin_pointer_uses_posted_index_not_deduped_ordinal(self):
        """For posted parts `[A, A, B]`, B's refusal pointer keeps submitted index 2 after dedupe."""
        with pytest.raises(EngineInputError) as exc_info:
            _resolve_parsed(
                [
                    _fertility_kenya_part(),
                    _fertility_kenya_part(),
                    ParsedPart(
                        variable_text="population",
                        entities=[ParsedEntity(text="Kenya", dcid="country/DOES_NOT_EXIST")],
                    ),
                ]
            )
        assert exc_info.value.code == "invalid_role_ref"
        assert exc_info.value.pointer == "/input/parts/2/entities/0/dcid"


class TestMultiPartConjunction:
    def test_multi_part_request_resolves_per_part(self):
        result = _resolve_parsed([_fertility_kenya_part(), ParsedPart(variable_text="population")])
        assert result.status == "definite"
        assert [p.variable_text for p in result.parts] == ["fertility rate", "population"]
        assert result.parts[0].status == "definite"
        assert result.parts[1].status == "definite"
        assert result.parts[0].spec.spec_id != result.parts[1].spec.spec_id


class TestUnsupportedConjunction:
    """Each part keeps its status; the envelope uses `mixed` when statuses differ."""

    def test_no_data_part_surfaces_alongside_a_definite_part(self):
        """Kenya's fertility rate holds one EUR facet: the bare part is definite, and the
        "US dollars" part refuses because the rate table holds no EUR rate for its year."""
        obs = {
            _FERTILITY_KEN_KEY: [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": "dc/base/FertilityEurSource",
                    "dates": ["2020"],
                    "values": [1.0],
                    "unit": "CurrencyEUR",
                }
            ]
        }
        materialise_graph, _unused_sdmx = materialise_double_from_obs(obs)
        materialise_graph._arcs = {
            **_FERTILITY_MATERIALISE_ARCS,
            **conversion_arcs(
                recipe_id=_RECIPE,
                provenance_id=_EUR_SOURCE_PROVENANCE,
                currencies=["CurrencyUSD", "CurrencyEUR"],
            ),
        }
        result = _resolve_parsed(
            [_fertility_kenya_part(), _fertility_kenya_part(unit_text="US dollars")],
            graph=FakeGraph(obs={**_load_json("graph_obs.json"), **obs}),
            materialise_graph=materialise_graph,
            materialise_sdmx=_sdmx_with_rate_anchor(obs),
            rate_cache=RateTableCache(),
        )
        assert result.status == "mixed"
        assert [p.variable_text for p in result.parts] == ["fertility rate", "fertility rate"]
        assert result.parts[0].status == "definite"
        assert result.parts[1].status == "no_data"
        assert result.parts[1].no_data.reason == "rate_coverage_gap"

    def test_every_refused_leg_reports_its_own_code(self):
        result = _resolve_parsed(
            [
                # PPP dollars parse but the request grammar cannot express
                # them, so this leg refuses unsupported ahead of any data lookup.
                _fertility_kenya_part(unit_text="PPP dollars"),
                ParsedPart(variable_text="population", source_text="OECD"),
            ]
        )
        assert result.status == "mixed"
        assert result.parts[0].unsupported.code == "unit_request_unsupported"
        assert result.parts[1].no_data.reason == "source_not_resolved"


class TestEntityPreResolutionBounding:
    """The multi-part fan-out resolves its deduplicated entity set once before its legs run."""

    def test_one_resolve_call_precedes_the_fan_out_with_deduplicated_entities(self):
        """Overlapping entities resolve once before the per-variable legs start."""
        fake_graph = FakeGraph()
        resolver = make_resolver(fake_graph)
        order: list[str] = []
        resolve_calls: list[list[str]] = []

        _original_resolve = resolver.resolve

        async def _tracked_resolve(texts: list[str]) -> list[Resolution]:
            resolve_calls.append(list(texts))
            order.append("resolve")
            return await _original_resolve(texts)

        resolver.resolve = _tracked_resolve  # ty: ignore[invalid-assignment]

        async def _ordering_stub_resolve_variable(variable: str, **_kwargs: object) -> RegionResult:
            order.append(f"leg:{variable}")
            return RegionResult(
                variable_text=variable,
                status="no_data",
                specs=(),
                no_data_reason="variable_not_resolved",
                warnings=(),
                timing_by_step={},
            )

        # "Kenya" repeats across the two parts. Pre-resolution dedupes it
        # into a single resolve call.
        parts = [
            ParsedPart(
                variable_text="v0",
                entities=[ParsedEntity(text="Kenya"), ParsedEntity(text="Uganda")],
            ),
            ParsedPart(variable_text="v1", entities=[ParsedEntity(text="Kenya")]),
        ]
        request = ResolveRequest(input=ParsedInput(parts=parts))

        with patch("search.engine.fanout.resolve_variable", new=_ordering_stub_resolve_variable):
            asyncio.run(
                resolve_async(
                    request,
                    graph=fake_graph,
                    resolver=resolver,
                    llm=FakeLLM(),
                    sdmx=FakeSdmxClient(observations={}, availability={}),
                )
            )

        assert resolve_calls == [["Kenya", "Uganda"]]
        assert order == ["resolve", "leg:v0", "leg:v1"]


_NOR = "country/NOR"
_KEN = "country/KEN"
_ROLE_BEARING_SV = "TESTFIXTURE/ParsedDoorRoleBearing/Flow"


def _role_bearing_arcs(*, observation_properties: list[str]) -> dict:
    """Builds arcs that declare the StatVar's roles."""
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": p} for p in observation_properties]},
    }


class _FixedBindLLM:
    """Elects shape 0 with no bindings for the unrecorded role-bearing fixture."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


class TestRoleBearingThroughTheParsedDoor:
    """Tests role-bearing input through the parsed door and shared pipeline."""

    def test_roleless_entity_among_a_named_pair_refuses_role_not_resolved(self):
        """Only Norway has `role_text="from"`; Kenya gets no axis and the pair refuses `role_not_resolved`.

        The parsed door assigns axes from each entity's `role_text`."""
        graph = FakeGraph(
            nodes={
                _NOR: {"label": "Norway", "type": "Country"},
                _KEN: {"label": "Kenya", "type": "Country"},
                _ROLE_BEARING_SV: {
                    "label": "Test flow",
                    "arcs": _role_bearing_arcs(observation_properties=["creditor", "debtor"]),
                },
            },
            detect={"aid": {"svs": [_ROLE_BEARING_SV], "cosine_scores": [0.9]}},
            resolve={"Norway": _NOR, "Kenya": _KEN},
            obs={},
        )
        request = ResolveRequest(
            input=ParsedInput(
                parts=[
                    ParsedPart(
                        variable_text="aid",
                        entities=[
                            ParsedEntity(text="Norway", role_text="from"),
                            ParsedEntity(text="Kenya"),
                        ],
                    )
                ]
            )
        )
        result = asyncio.run(
            resolve_async(
                request,
                graph=graph,
                resolver=make_resolver(graph),
                llm=_FixedBindLLM(),
                sdmx=FakeSdmxClient(observations={}),
            )
        )
        assert result.status == "no_data", result
        assert result.parts[0].no_data.reason == "role_not_resolved"


class TestPinnedDcidC2:
    """Tests pinned entity confirmation, graph labels, and per-part resolution."""

    @pytest.mark.parametrize(
        "text, dcid, graph_extra",
        [
            pytest.param("Kenyaville", "country/KEN", None, id="text_resolves_to_nothing"),
            pytest.param(
                "Congo",
                "country/COD",
                {
                    "FertilityRate_Person_Female|country/COD": [
                        {"earliestDate": "1960", "latestDate": "2023", "obsCount": 64}
                    ]
                },
                id="website_ambiguous_pick_payload",
            ),
        ],
    )
    def test_pinned_dcid_resolves_definite_for_that_entity(self, text, dcid, graph_extra):
        """The pin resolves even when the text is unknown (`Kenyaville`) or ambiguous (`Congo`)."""
        graph = FakeGraph(obs={**_load_json("graph_obs.json"), **(graph_extra or {})})
        part = ParsedPart(
            variable_text="fertility rate",
            entities=[ParsedEntity(text=text, dcid=dcid)],
        )
        result = _resolve_parsed([part], graph=graph)
        assert result.status == "definite", result
        assert result.parts[0].spec.definition.identity.entities == {"observationAbout": [dcid]}

    def test_pinned_dcid_grounds_with_the_graph_label_and_restates_the_caller_text(self):
        """Pinning "Kenya" to the USA uses the graph label in the Spec and keeps "Kenya" in restatement."""
        graph = FakeGraph(
            obs={
                **_load_json("graph_obs.json"),
                "FertilityRate_Person_Female|country/USA": [
                    {"earliestDate": "1960", "latestDate": "2023", "obsCount": 64}
                ],
            }
        )
        usa_label = graph.node_labels_batch(["country/USA"])["country/USA"]
        part = ParsedPart(
            variable_text="fertility rate",
            entities=[ParsedEntity(text="Kenya", dcid="country/USA")],
        )
        result = _resolve_parsed([part], graph=graph)

        assert result.status == "definite", result
        resolved_part = result.parts[0]
        assert resolved_part.spec.definition.identity.entities == {
            "observationAbout": ["country/USA"]
        }
        assert usa_label != "Kenya"
        spec_json = resolved_part.spec.model_dump_json()
        assert usa_label in spec_json
        assert "Kenya" not in spec_json
        assert [(e.text, e.dcid) for e in resolved_part.restatement.entities] == [
            ("Kenya", "country/USA")
        ]

    def test_confirm_part_pins_names_each_pin_by_its_graph_label_or_dcid(self):
        """confirm_part_pins files each pin under the caller's text as a
        PinnedResolved named by the graph's label, and falls back to the dcid
        for a typed node the graph gives no label."""

        graph = FakeGraph()
        part = ParsedPart(
            variable_text="fertility rate",
            entities=[
                ParsedEntity(text="Kenya", dcid="country/USA"),
                ParsedEntity(text="Somewhere", dcid="country/XYZ"),
            ],
        )
        with (
            patch.object(
                graph,
                "node_types_batch",
                side_effect=lambda dcids: dict.fromkeys(dcids, "Country"),
            ),
            patch.object(
                graph,
                "node_labels_batch",
                return_value={"country/USA": "United States of America"},
            ),
        ):
            [pins] = asyncio.run(
                confirm_part_pins(
                    [part],
                    graph=graph,
                    resolver=make_resolver(graph),
                    pointer_prefixes=["/input/parts/0"],
                )
            )

        named: dict[str, tuple[str, str]] = {}
        for text, pin in pins.items():
            assert isinstance(pin, PinnedResolved)
            named[text] = (pin.entity.dcid, pin.entity.name)
        assert named == {
            "Kenya": ("country/USA", "United States of America"),
            "Somewhere": ("country/XYZ", "country/XYZ"),
        }

    def test_confirm_part_pins_falls_back_to_dcid_when_label_read_fails(self, caplog):
        """A label transport failure keeps a valid pin, uses its dcid as the name, and logs a warning."""

        graph = FakeGraph()
        part = ParsedPart(
            variable_text="fertility rate",
            entities=[ParsedEntity(text="Kenya", dcid="country/USA")],
        )
        with (
            patch.object(
                graph,
                "node_types_batch",
                side_effect=lambda dcids: dict.fromkeys(dcids, "Country"),
            ),
            patch.object(
                graph,
                "node_labels_batch",
                side_effect=GraphInfraError("labels unavailable"),
            ),
            caplog.at_level(logging.WARNING, logger="search.engine.entry.parsed"),
        ):
            [pins] = asyncio.run(
                confirm_part_pins(
                    [part],
                    graph=graph,
                    resolver=make_resolver(graph),
                    pointer_prefixes=["/input/parts/0"],
                )
            )

        assert pins["Kenya"].entity.name == "country/USA"
        [record] = caplog.records
        message = record.getMessage()
        assert "GraphInfraError" in message
        assert "country/USA" in message

    def test_confirm_part_pins_raises_invalid_pin_even_when_label_read_also_fails(self):
        """A missing pin returns 422 even when its label read fails."""

        graph = FakeGraph()
        part = ParsedPart(
            variable_text="fertility rate",
            entities=[ParsedEntity(text="Kenya", dcid="country/DOES_NOT_EXIST")],
        )
        with (
            patch.object(graph, "node_types_batch", return_value={}),
            patch.object(
                graph,
                "node_labels_batch",
                side_effect=GraphInfraError("labels unavailable"),
            ),
            pytest.raises(EngineInputError) as exc_info,
        ):
            asyncio.run(
                confirm_part_pins(
                    [part],
                    graph=graph,
                    resolver=make_resolver(graph),
                    pointer_prefixes=["/input/parts/0"],
                )
            )
        assert exc_info.value.code == "invalid_role_ref"
        assert exc_info.value.status == 422

    def test_pinned_dcid_resolves_definite_across_multiple_parts(self):
        """The N>=2 fan-out path (per_part_pre_resolved) excludes a text
        pinned in every one of its occurrences from its own shared
        resolver.resolve() batch and overlays the pin back in per part --
        exercised here alongside an ordinarily-resolved second part."""
        parts = [
            ParsedPart(
                variable_text="fertility rate",
                entities=[ParsedEntity(text="Kenyaville", dcid="country/KEN")],
            ),
            _fertility_kenya_part(),
        ]
        result = _resolve_parsed(parts)
        assert [p.status for p in result.parts] == ["definite", "definite"]
        assert result.parts[0].spec.definition.identity.entities == {
            "observationAbout": ["country/KEN"]
        }

    def test_unconfirmable_pinned_dcid_refuses_invalid_role_ref(self):
        part = ParsedPart(
            variable_text="fertility rate",
            entities=[ParsedEntity(text="Kenya", dcid="country/DOES_NOT_EXIST")],
        )
        with pytest.raises(EngineInputError) as exc_info:
            _resolve_parsed([part])
        assert exc_info.value.code == "invalid_role_ref"
        assert exc_info.value.status == 422
        assert exc_info.value.pointer == "/input/parts/0/entities/0/dcid"

    def test_two_parts_pinning_same_text_to_different_dcids_resolve_different_places(self):
        """Issue F1: each part keeps its own Congo pin. COD has data; COG is confirmed with no observations."""
        graph = FakeGraph(
            obs={
                **_load_json("graph_obs.json"),
                "FertilityRate_Person_Female|country/COD": [
                    {"earliestDate": "1960", "latestDate": "2023", "obsCount": 64}
                ],
                "FertilityRate_Person_Female|country/COG": [],
            }
        )
        parts = [
            ParsedPart(
                variable_text="fertility rate",
                entities=[ParsedEntity(text="Congo", dcid="country/COD")],
            ),
            ParsedPart(
                variable_text="fertility rate",
                entities=[ParsedEntity(text="Congo", dcid="country/COG")],
            ),
        ]
        result = _resolve_parsed(parts, graph=graph)
        assert result.parts[0].status == "definite", result.parts[0]
        assert result.parts[0].spec.definition.identity.entities == {
            "observationAbout": ["country/COD"]
        }
        assert result.parts[1].status == "no_data", result.parts[1]
        assert result.parts[1].no_data.reason == "no_observations"

    def test_one_part_pinning_one_text_to_two_dcids_refuses_request_invalid(self):
        """Two pins for the same text in one part refuse `request_invalid`."""
        part = ParsedPart(
            variable_text="aid from Congo to Congo",
            entities=[
                ParsedEntity(text="Congo", role_text="from", dcid="country/COD"),
                ParsedEntity(text="Congo", role_text="to", dcid="country/COG"),
            ],
        )
        with pytest.raises(EngineInputError) as exc_info:
            _resolve_parsed([part])
        assert exc_info.value.code == "request_invalid"
        assert exc_info.value.pointer == "/input/parts/0/entities"

    def test_one_part_posting_one_text_with_both_members_values_refuses_request_invalid(self):
        """One text asked for as its members and as a whole in one part refuses
        `request_invalid`."""
        part = ParsedPart(
            variable_text="aid from Africa to Africa",
            entities=[
                ParsedEntity(text="Africa", role_text="from", members=True),
                ParsedEntity(text="Africa", role_text="to"),
            ],
        )
        with pytest.raises(EngineInputError) as exc_info:
            _resolve_parsed([part])
        assert exc_info.value.code == "request_invalid"
        assert exc_info.value.pointer == "/input/parts/0/entities"

    def test_parts_differing_only_in_members_stay_distinct(self):
        """Two parts alike but for one entity's `members` are two asks, and each leg is
        handed its own part's member entities."""
        parts = [
            ParsedPart(variable_text="population", entities=[ParsedEntity(text="Africa")]),
            ParsedPart(
                variable_text="population", entities=[ParsedEntity(text="Africa", members=True)]
            ),
        ]
        assert [part for _, part in _dedupe_parts_with_indices(parts)] == parts
        interpretation = interpret_parts(parts, pointer_prefixes=["/p/0", "/p/1"])
        assert interpretation.member_entities == [frozenset(), frozenset({"Africa"})]

    def test_unpinned_text_shared_across_parts_resolves_once(self):
        """Overlapping unpinned texts share one pre-resolution call before the legs run.

        Each leg then resolves an empty miss set."""
        graph = FakeGraph()
        resolver = make_resolver(graph)
        resolve_calls: list[list[str]] = []
        _original_resolve = resolver.resolve

        async def _tracked_resolve(texts: list[str]) -> list[Resolution]:
            resolve_calls.append(list(texts))
            return await _original_resolve(texts)

        resolver.resolve = _tracked_resolve  # type: ignore[method-assign]

        parts = [
            _fertility_kenya_part(),
            ParsedPart(
                variable_text="fertility rate",
                entities=[ParsedEntity(text="Kenya"), ParsedEntity(text="Uganda")],
            ),
        ]
        request = ResolveRequest(input=ParsedInput(parts=parts))
        result = asyncio.run(
            resolve_async(
                request, graph=graph, llm=FakeLLM(), sdmx=FakeSdmxClient(), resolver=resolver
            )
        )
        assert result.status == "definite", result
        assert resolve_calls[0] == ["Kenya", "Uganda"]
        assert resolve_calls[1:] == [[], []]
