"""Checks envelope sentence composition, qualifier hoisting, and token conservation."""

from __future__ import annotations

import itertools
from typing import cast

import pytest

from search.models import (
    METHOD_REF,
    UNIT_REF,
    Assumption,
    BindingUnbound,
    BindingValue,
    BreadthDim,
    CandidateSet,
    CandidatesPart,
    CoverageBreadth,
    DefinitePart,
    Departure,
    IdentityBindingValue,
    NoData,
    NoDataGate,
    NoDataPart,
    ResolvedFacet,
    ResolvedQualifier,
    SentenceToken,
    Slot,
    SlotKey,
    SlotValue,
    Spec,
    StatVar,
    TimeWindow,
    Unsupported,
    UnsupportedPart,
    Warning,
)
from search.render.envelope import compose_envelope_sentence
from search.render.no_data import no_data_phrase, no_data_sentence_plan
from search.render.plan import plain_sentence_plan, render_sentence_plan
from search.render.tokens import sentence_from_plan
from tests.render._helpers import (
    _entity_from,
    _entity_subject,
    _entity_to,
    _gender_slot,
    _how_slot,
    _ref,
    _slot_key,
    _source_slot,
    _spec,
    _unbound_source_slot,
    _what_slot,
    _where_place_slot,
    _where_slot,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _sentenced(spec: Spec) -> Spec:
    """Attach a rendered sentence and plan to `spec`."""
    plan = render_sentence_plan(spec)
    rendered = sentence_from_plan(plan)
    return spec.model_copy(
        update={
            "resolved": spec.resolved.model_copy(
                update={"rendered_sentence": rendered, "sentence_plan": plan}
            )
        }
    )


def _definite(
    spec: Spec, *, variable_text: str | None = None, warnings: list[Warning] | None = None
) -> DefinitePart:
    return DefinitePart(
        status="definite",
        variable_text=variable_text,
        warnings=warnings or [],
        spec=_sentenced(spec),
    )


def _unresolved(variable_text: str | None) -> NoDataPart:
    """Build an unresolved part; composition uses its variable text, not its sentence."""
    return NoDataPart(
        status="no_data",
        variable_text=variable_text,
        warnings=[],
        no_data=NoData(reason="no_observations"),
        rendered_sentence="No data.",
        sentence_plan=plain_sentence_plan("No data."),
    )


def _candidates(variable_text: str, spec_a: Spec, spec_b: Spec) -> CandidatesPart:
    return CandidatesPart(
        status="candidates",
        variable_text=variable_text,
        warnings=[],
        candidates=CandidateSet(
            ordering="broadest_first", max_candidates=5, specs=[spec_a, spec_b], truncated=False
        ),
        rendered_sentence="2 readings.",
        sentence_plan=plain_sentence_plan("2 readings."),
    )


def _no_data_with_reason(reason: str, variable_text: str) -> NoDataPart:
    """Build a no-data part with the standard interpolated phrase."""
    phrase = no_data_phrase(reason, variable_term=variable_text)
    return NoDataPart(
        status="no_data",
        variable_text=variable_text,
        warnings=[],
        no_data=NoData(reason=reason),
        rendered_sentence=phrase,
        sentence_plan=plain_sentence_plan(phrase),
    )


def _no_data_with_plan(reason: str, variable_text: str, plan: list[SentenceToken]) -> NoDataPart:
    return NoDataPart(
        status="no_data",
        variable_text=variable_text,
        warnings=[],
        no_data=NoData(reason=reason),
        rendered_sentence=sentence_from_plan(plan),
        sentence_plan=plan,
    )


def _folded(spec: Spec) -> DefinitePart:
    return DefinitePart(
        status="definite",
        variable_text=None,
        warnings=[Warning(code="binding_folded", severity="info", message="folded onto part 0")],
        spec=_sentenced(spec),
    )


def _unbound_when_slot() -> Slot:
    return Slot(
        key=SlotKey(axis="when", property=None, label="time window"), binding=BindingUnbound()
    )


def _unbound_where_slot() -> Slot:
    return Slot(key=SlotKey(axis="where", property=None, label="place"), binding=BindingUnbound())


def _strip_part_index(tokens: list[SentenceToken]) -> list[SentenceToken]:
    return [t.model_copy(update={"part_index": None}) for t in tokens]


# Single-part plan round trips.


def _round_trip_specs() -> list[Spec]:
    return [
        _spec(slots=[_what_slot("ODA Grants")]),
        _spec(slots=[_what_slot("ODA Grants"), _how_slot("health")]),
        _spec(slots=[_where_slot("Ethiopia")]),
        _spec(entities=[_entity_to("country/ETH", "Ethiopia"), _entity_from("country/USA", "USA")]),
        _spec(window=TimeWindow(start_year=2015, end_year=2023)),
        _spec(slots=[_source_slot("wdi", "World Development Indicators")]),
        _spec(
            slots=[
                _what_slot("ODA Grants"),
                _how_slot("health"),
                _where_slot("Ethiopia"),
                _source_slot("wdi", "World Development Indicators"),
            ],
            entities=[_entity_from("country/USA", "USA")],
            window=TimeWindow(start_year=2015, end_year=2023),
        ),
        _spec(slots=[_unbound_source_slot()]),
        _spec(slots=[_unbound_when_slot()]),
    ]


def test_general_path_round_trips_every_single_part_fixture():
    for spec in _round_trip_specs():
        part = _definite(spec)
        rendered, plan = compose_envelope_sentence([part])
        assert rendered == part.spec.resolved.rendered_sentence
        assert _strip_part_index(plan) == _strip_part_index(part.spec.resolved.sentence_plan)


def _token_identity(
    token: SentenceToken,
) -> tuple[str, str, str | None, str | None, str | None, list[str]]:
    """Normalize initial capitalization while preserving token metadata."""
    text = token.text[0].lower() + token.text[1:] if token.text else token.text
    return (token.kind, text, token.axis, token.role, token.binding_ref, token.entity_refs)


def _assert_token_conservation(specs: list[Spec]) -> None:
    """Every token survives composition under its part index or as a hoisted token."""
    parts = [_definite(spec, variable_text=f"var{i}") for i, spec in enumerate(specs)]
    _rendered, composed = compose_envelope_sentence(parts)
    assert composed is not None
    composed_identities = [(token.part_index, _token_identity(token)) for token in composed]

    for position, part in enumerate(parts):
        original = list(part.spec.resolved.sentence_plan or [])
        if original and original[-1].kind == "text" and original[-1].text == ".":
            original = original[:-1]
        for token in original:
            identity = _token_identity(token)
            assert any(
                composed_identity == identity and part_index in (position, None)
                for part_index, composed_identity in composed_identities
            ), (position, token)


def test_token_conservation_pairwise_and_triple_over_round_trip_specs():
    """Check token conservation for every fixture pair and one three-part set."""
    specs = _round_trip_specs()
    for pair in itertools.combinations(specs, 2):
        _assert_token_conservation(list(pair))
    _assert_token_conservation(specs[:3])


# No sentence to compose.


def test_a_part_without_a_sentence_composes_to_none():
    part = DefinitePart(
        status="definite",
        variable_text="population",
        warnings=[],
        spec=_spec(slots=[_what_slot("Population")]),
    )
    assert compose_envelope_sentence([part]) is None


# Single-part outcomes.


def test_single_definite_part():
    spec = _spec(
        slots=[
            _what_slot("Total population"),
            _where_slot("Ethiopia"),
            _source_slot("wdi", "World Development Indicators"),
        ],
        window=TimeWindow(start_year=2015, end_year=2023),
    )
    part = _definite(spec)
    rendered, plan = compose_envelope_sentence([part])
    assert rendered == part.spec.resolved.rendered_sentence
    assert [token.part_index for token in plan if token.axis == "what"] == [0]
    # The possessive place lead belongs to the part's own phrase; the rest hoists.
    assert [token.part_index for token in plan if token.axis == "where"] == [0]
    assert all(token.part_index is None for token in plan if token.axis in {"when", "source"})
    assert all(
        token.part_index is None
        for token in plan
        if token.kind == "text" and token.text in {" in ", ", ", ", according to ", "."}
    )


def test_single_no_data_part():
    part = _unresolved("GDP")
    rendered, plan = compose_envelope_sentence([part])
    assert rendered == "No data."
    assert plan == [SentenceToken(kind="text", text="No data.", part_index=0)]


def test_single_no_data_part_carries_unbound_qualifiers_without_disturbing_composition():
    """NoData.unbound_qualifiers (the qualifier_not_bound mirror of
    entity_terms) round-trips on the model and plays no part in envelope
    composition -- it exists for a client to read the unbound terms
    structurally, not for the composer to consume."""
    qualifiers = [ResolvedQualifier(text="grants", named=[], slot=None, disposition="unbound")]
    phrase = no_data_phrase("qualifier_not_bound", qualifier_terms=qualifiers)
    part = NoDataPart(
        status="no_data",
        variable_text=None,
        warnings=[],
        no_data=NoData(reason="qualifier_not_bound", unbound_qualifiers=qualifiers),
        rendered_sentence=phrase,
        sentence_plan=plain_sentence_plan(phrase),
    )
    assert part.no_data.unbound_qualifiers == qualifiers
    rendered, plan = compose_envelope_sentence([part])
    assert rendered == phrase
    assert plan == [SentenceToken(kind="text", text=phrase, part_index=0)]


def test_single_candidates_part():
    spec_a = _spec(slots=[_what_slot("A")])
    spec_b = _spec(slots=[_what_slot("B")])
    part = _candidates("population", spec_a, spec_b)
    rendered, plan = compose_envelope_sentence([part])
    assert rendered == "2 readings."
    assert all(t.part_index == 0 for t in plan)


def test_single_unsupported_part_preserves_its_own_provenance():
    part = UnsupportedPart(
        status="unsupported",
        variable_text="GDP in PPP terms",
        warnings=[],
        rendered_sentence="PPP terms are not supported.",
        sentence_plan=plain_sentence_plan("PPP terms are not supported."),
        unsupported=Unsupported(
            code="unit_request_unsupported",
            pointer="/identity/slots/0",
            detail="PPP terms are not supported.",
            definition={},
        ),
    )
    rendered, plan = compose_envelope_sentence([part])
    assert rendered == part.rendered_sentence
    assert plan == [SentenceToken(kind="text", text="PPP terms are not supported.", part_index=0)]


# Folded parts.


def test_folded_part_excluded_from_composition():
    """crs-048: a part folded into an earlier one states that part's heading and byline,
    so they carry no part index, while the earlier part's departures keep it."""
    modelled = Departure(code="modelled_values", text="These values are modelled estimates.")
    primary = _definite(_spec(slots=[_what_slot("Total population")])).model_copy(
        update={"departures": [modelled]}
    )
    folded = _folded(_spec(slots=[_what_slot("Total population")]))
    rendered, plan = compose_envelope_sentence([primary, folded])
    # The remaining definite part uses its own plan.
    assert rendered == f"{primary.spec.resolved.rendered_sentence} {modelled.text}"
    assert [(token.text, token.part_index) for token in plan] == [
        ("Total population", None),
        (".", None),
        (" ", 0),
        (modelled.text, 0),
    ]


def test_folded_part_excluded_from_composition_single_definite_carries_true_index():
    """The sole definite contributor keeps its true position when a folded
    predecessor leaves it at parts[1]."""
    folded = _folded(_spec(slots=[_what_slot("Total population")]))
    primary = _definite(_spec(slots=[_what_slot("Total population")]))
    rendered, plan = compose_envelope_sentence([folded, primary])
    assert rendered == primary.spec.resolved.rendered_sentence
    assert [(token.text, token.part_index) for token in plan] == [
        ("Total population", 1),
        (".", None),
    ]


def test_fold_before_a_later_contributing_part_keeps_its_true_index():
    """A fold that precedes another, distinct contributing part must not
    shift that later part's part_index: part_index is a position in
    ResolveResponse.parts, and a folded part stays in parts (C51a)."""
    population = _definite(_spec(slots=[_what_slot("Total population")]))
    folded = _folded(_spec(slots=[_what_slot("Total population")]))
    gdp = _definite(_spec(slots=[_what_slot("Nominal GDP")]))
    rendered, plan = compose_envelope_sentence([population, folded, gdp])
    assert rendered == "Total population and nominal GDP."
    gdp_tokens = [t for t in plan if t.text == "nominal GDP"]
    assert len(gdp_tokens) == 1
    assert gdp_tokens[0].part_index == 2


# Shared qualifiers hoist once.


def test_agree_hoists_shared_qualifiers_once():
    common = {
        "slots": [_where_place_slot("country/BRA", "Brazil"), _source_slot("wdi", "WDI")],
        "window": TimeWindow(start_year=1960, end_year=2024),
    }
    population = _definite(
        _spec(slots=[_what_slot("Total population"), *common["slots"]], window=common["window"])
    )
    gdp = _definite(
        _spec(slots=[_what_slot("Nominal GDP"), *common["slots"]], window=common["window"])
    )
    rendered, plan = compose_envelope_sentence([population, gdp])
    assert rendered == ("Brazil's total population and nominal GDP, 1960 to 2024. WDI.")
    # The shared possessive place lead opens the sentence once; when hoists and the one
    # byline follows.
    assert [t.part_index for t in plan if t.axis == "where"] == [None]
    assert [(t.part_index, t.segment) for t in plan if t.axis == "source"] == [(None, "byline")]


def _usd_part(
    what: str, *, qualifier: str, method: str | None = None, qualifier_slot: bool = False
) -> DefinitePart:
    """A Kenya US-dollar part, its identity binding ``method`` when given. Under
    ``qualifier_slot`` the shape declares a measurementQualifier how slot, so the unit
    token states that slot's ref in place of the bare axis ref."""
    how_slots = [_crs_how("measurementQualifier", qualifier)] if qualifier_slot else []
    method_facets = (
        {"measurementMethod": ResolvedFacet(basis="bound", ref=method, label=f"{method} label")}
        if method is not None
        else {}
    )
    return _definite(
        _spec(
            slots=[_what_slot(what), _where_place_slot("country/KEN", "Kenya"), *how_slots],
            window=TimeWindow(start_year=1960, end_year=2024),
            measurement_qualifier=_ref(qualifier, qualifier),
            facets={
                "unit": ResolvedFacet(
                    basis="fallback", ref="CurrencyUSD", label="US Dollar", currency="USD"
                ),
                **method_facets,
            },
            resolved_sources=[_ref("wdi", "World Development Indicators")],
            identity_facets=(
                {"measurementMethod": IdentityBindingValue(ref=method)}
                if method is not None
                else None
            ),
        )
    )


def test_identical_bylines_follow_the_composed_heading_once():
    """Parts with the same unit and source share one byline, unindexed, after the
    composed heading's period."""
    rendered, plan = compose_envelope_sentence(
        [
            _usd_part("Total exports", qualifier="Nominal"),
            _usd_part("Total imports", qualifier="Nominal"),
        ]
    )
    assert rendered == (
        "Kenya's total exports and total imports, 1960 to 2024. "
        "Current US dollars · World Development Indicators."
    )
    byline = [t for t in plan if t.segment == "byline"]
    assert [t.text for t in byline] == [
        " ",
        "Current US dollars",
        " · ",
        "World Development Indicators",
        ".",
    ]
    assert {t.part_index for t in byline} == {None}


def test_bylines_that_differ_in_unit_compose_one_sentence_per_part():
    """Current and constant US dollars differ, so each heading keeps its own byline."""
    rendered, plan = compose_envelope_sentence(
        [
            _usd_part("Total exports", qualifier="Nominal"),
            _usd_part("Total imports", qualifier="RealValue"),
        ]
    )
    assert rendered == (
        "Kenya's total exports, 1960 to 2024. Current US dollars · World Development "
        "Indicators. Kenya's total imports, 1960 to 2024. Constant US dollars · World "
        "Development Indicators."
    )
    assert [t.part_index for t in plan if t.axis == "unit"] == [0, 1]


def test_bylines_whose_unit_states_differ_compose_one_sentence_per_part():
    """Two parts read "Current US dollars", but one edits the price basis through its how
    slot and the other through the bare axis. A shared unit token could state only one
    of those refs, so each part keeps its own byline and unit token."""
    _, plan = compose_envelope_sentence(
        [
            _usd_part("Total exports", qualifier="Nominal", qualifier_slot=True),
            _usd_part("Total imports", qualifier="Nominal"),
        ]
    )
    units = [t for t in plan if t.axis == "unit"]
    assert [(t.part_index, t.text, t.states) for t in units] == [
        (0, "Current US dollars", [UNIT_REF, "how:measurementQualifier"]),
        (1, "Current US dollars", [UNIT_REF, "axis:measurementQualifier"]),
    ]


@pytest.mark.parametrize(
    ("methods", "owners"),
    [
        pytest.param(("Census", "Census"), [None], id="shared"),
        pytest.param(("Census", "Survey"), [0, 1], id="differing"),
        pytest.param(("Census", None), [0], id="one-bound"),
    ],
)
def test_a_bound_method_is_shared_like_the_source(methods, owners):
    """Parts that bind one method alike share the byline and its method token, stated
    once with no part index. A method only one part binds, or two that differ, keep one
    sentence per part, each method token under its own part."""
    _, plan = compose_envelope_sentence(
        [
            _usd_part("Total exports", qualifier="Nominal", method=methods[0]),
            _usd_part("Total imports", qualifier="Nominal", method=methods[1]),
        ]
    )
    stating = [t for t in plan if METHOD_REF in t.states]
    assert [t.part_index for t in stating] == owners
    assert all((t.axis, t.binding_ref) == ("method", METHOD_REF) for t in stating)


# Disagreement uses each part's sentence.


def test_disagree_composes_one_sentence_per_part():
    where = _where_place_slot("country/BRA", "Brazil")
    population = _definite(
        _spec(
            slots=[_what_slot("Total population"), where, _source_slot("s1", "6 sources")],
            window=TimeWindow(start_year=1940, end_year=2025),
        )
    )
    gdp = _definite(
        _spec(
            slots=[
                _what_slot("Nominal GDP"),
                where,
                _source_slot("wdi", "World Development Indicators"),
            ],
            window=TimeWindow(start_year=1960, end_year=2024),
        )
    )
    rendered, plan = compose_envelope_sentence([population, gdp])
    assert rendered == (
        "Brazil's total population, 1940 to 2025. 6 sources. "
        "Brazil's nominal GDP, 1960 to 2024. World Development Indicators."
    )
    assert "(" not in rendered and ")" not in rendered
    separators = [t for t in plan if t.kind == "text" and t.text == " " and t.part_index is None]
    assert len(separators) == 1
    assert {t.part_index for t in plan if t is not separators[0]} == {0, 1}
    where_tokens = [t for t in plan if t.axis == "where"]
    assert len(where_tokens) == 2 and {t.part_index for t in where_tokens} == {0, 1}


def _aid_part(what_text: str, donor: tuple[str, str], where: Slot) -> DefinitePart:
    """A definite part whose donor leads and whose where slot names the subject place."""
    return _definite(_spec(slots=[_what_slot(what_text), where], entities=[_entity_from(*donor)]))


# Qualifier metadata must agree even when text matches.


def test_binding_ref_divergence_prevents_hoist():
    """Different slot keys keep identical place text inline (C130)."""
    health = _aid_part(
        "Health ODA", ("country/USA", "USA"), _where_place_slot("country/ETH", "Ethiopia")
    )
    education = _aid_part("Education ODA", ("country/GBR", "Britain"), _where_slot("Ethiopia"))
    rendered, plan = compose_envelope_sentence([health, education])
    assert rendered == "USA's health ODA in Ethiopia and Britain's education ODA in Ethiopia."
    where_tokens = [t for t in plan if t.axis == "where"]
    assert [t.part_index for t in where_tokens] == [0, 1]
    assert {t.binding_ref for t in where_tokens} == {"where:", "where:DevelopmentFinanceRecipient"}


def test_connective_divergence_prevents_hoist():
    """A counterpart and a subject place with the same text stay in their own phrases."""
    finance = _definite(
        _spec(
            slots=[_what_slot("Health ODA")],
            entities=[_entity_from("country/USA", "USA"), _entity_to("country/ETH", "Ethiopia")],
        )
    )
    education = _aid_part("Education ODA", ("country/GBR", "Britain"), _where_slot("Ethiopia"))
    rendered, plan = compose_envelope_sentence([finance, education])
    assert rendered == "USA's health ODA to Ethiopia and Britain's education ODA in Ethiopia."
    where_tokens = [t for t in plan if t.axis == "where"]
    assert [(t.part_index, t.role) for t in where_tokens] == [(0, "destination"), (1, None)]


def test_all_empty_placeholder_binding_ref_divergence_prevents_hoist():
    """Different empty-slot references remain separate and editable (C130)."""
    a = _definite(_spec(slots=[_what_slot("Total population"), _unbound_when_slot()]))
    b = _definite(
        _spec(
            slots=[
                _what_slot("Nominal GDP"),
                Slot(
                    key=_slot_key("when", "requestedYear", "Requested Year", "time window"),
                    binding=BindingUnbound(),
                ),
            ]
        )
    )
    rendered, plan = compose_envelope_sentence([a, b])
    assert rendered == "Total population and nominal GDP."
    when_tokens = [t for t in plan if t.axis == "when"]
    assert len(when_tokens) == 2
    assert {t.part_index for t in when_tokens} == {0, 1}
    assert {t.binding_ref for t in when_tokens} == {"when:", "when:requestedYear"}


def test_full_agreement_still_hoists_regression_guard():
    """Matching qualifier metadata still hoists once."""
    health = _aid_part("Health ODA", ("country/USA", "USA"), _where_slot("Ethiopia"))
    education = _aid_part("Education ODA", ("country/GBR", "Britain"), _where_slot("Ethiopia"))
    rendered, plan = compose_envelope_sentence([health, education])
    assert rendered == "USA's health ODA and Britain's education ODA in Ethiopia."
    where_tokens = [t for t in plan if t.axis == "where"]
    assert [(t.part_index, t.binding_ref) for t in where_tokens] == [
        (None, "where:DevelopmentFinanceRecipient")
    ]


def test_per_part_mode_preserves_each_parts_own_token_order_including_empty_how_placeholder():
    """Per-part mode preserves each source plan's token order."""
    unbound_how = Slot(
        key=_slot_key("how", "DevelopmentFinancePurpose", "Purpose", "purpose"),
        binding=BindingUnbound(),
    )
    population = _definite(
        _spec(
            slots=[_what_slot("Total population"), unbound_how],
            window=TimeWindow(start_year=1940, end_year=2025),
        )
    )
    gdp = _definite(
        _spec(slots=[_what_slot("Nominal GDP")], window=TimeWindow(start_year=1960, end_year=2024))
    )
    _rendered, plan = compose_envelope_sentence([population, gdp])
    part0_tokens = [t for t in plan if t.part_index == 0]
    how_placeholder = next(i for i, t in enumerate(part0_tokens) if t.axis == "how")
    what_token = next(i for i, t in enumerate(part0_tokens) if t.axis == "what")
    when_token = next(i for i, t in enumerate(part0_tokens) if t.axis == "when")
    assert what_token < how_placeholder < when_token


def test_no_composed_plan_ever_mints_a_parenthesis():
    """Neither composition branch introduces parenthesis tokens."""
    where = _where_place_slot("country/BRA", "Brazil")
    agree = [
        _definite(_spec(slots=[_what_slot("Total population"), where])),
        _definite(_spec(slots=[_what_slot("Nominal GDP"), where])),
    ]
    disagree = [
        _definite(
            _spec(
                slots=[_what_slot("Total population")],
                window=TimeWindow(start_year=1940, end_year=2025),
            )
        ),
        _definite(
            _spec(
                slots=[_what_slot("Nominal GDP")], window=TimeWindow(start_year=1960, end_year=2024)
            )
        ),
    ]
    for parts in (agree, disagree):
        _rendered, plan = compose_envelope_sentence(parts)
        texts = {token.text for token in plan}
        assert " (" not in texts and ")" not in texts


# Route unhoisted axes by connective.


def test_origin_inlines_while_where_hoists_reordering_relative_to_each_parts_own_plan():
    """A possessive origin stays in each part's phrase while an agreeing where qualifier
    hoists."""
    where = _where_place_slot("country/ETH", "Ethiopia")
    health = _definite(
        _spec(
            slots=[_what_slot("Health ODA"), where], entities=[_entity_from("country/USA", "USA")]
        )
    )
    education = _definite(
        _spec(
            slots=[_what_slot("Education ODA"), where],
            entities=[_entity_from("country/GBR", "United Kingdom")],
        )
    )
    rendered, plan = compose_envelope_sentence([health, education])
    assert rendered == "USA's health ODA and United Kingdom's education ODA in Ethiopia."
    where_tokens = [t for t in plan if t.axis == "where"]
    assert len(where_tokens) == 1 and where_tokens[0].part_index is None
    origin_tokens = [t for t in plan if t.role == "origin"]
    assert len(origin_tokens) == 2 and {t.part_index for t in origin_tokens} == {0, 1}


def test_parts_that_differ_only_in_actor_keep_each_possessive_lead():
    """Two parts that differ only in their possessive actor join their leads, each with
    the entity's casing and its own part index, before one shared noun and counterpart
    in the family's wording."""
    parts = [
        _definite(
            _spec(
                slots=[_what_slot("Health ODA")],
                entities=[_entity_to("country/ETH", "Ethiopia"), donor],
            )
        )
        for donor in (_entity_from("country/USA", "USA"), _entity_from("country/GBR", "Britain"))
    ]
    rendered, plan = compose_envelope_sentence(parts)
    assert rendered == "USA's and Britain's health ODA to Ethiopia."
    leads = [(t.text, t.part_index, t.binding_ref) for t in plan if t.role == "origin"]
    assert leads == [("USA", 0, "origin:"), ("Britain", 1, "origin:")]
    shared = [(t.text, t.part_index) for t in plan if t.axis in ("what", "where")]
    assert shared == [("health ODA", None), ("Ethiopia", None)]


def test_parts_that_differ_only_in_place_keep_each_possessive_before_a_silent_modifier():
    """A gender modifier folded into the StatVar name is silent; the possessive leads
    still join with each suffix, before one shared noun."""
    parts = [
        _definite(
            _spec(
                slots=[_gender_slot("Female", "Female"), _where_place_slot(dcid, name)],
                entities=[_entity_subject(dcid, name)],
                population_type=_ref("Person", "Person"),
                measured_property=_ref("count", "count"),
                stat_vars=[
                    StatVar(
                        ref=_ref("Count_Person_Female", "Female population"),
                        shape_id="test-shape",
                        slot_values=[],
                        constraint_properties=[],
                    )
                ],
            )
        )
        for dcid, name in (("country/KEN", "Kenya"), ("country/UGA", "Uganda"))
    ]
    assert [part.spec.resolved.rendered_sentence for part in parts] == [
        "Kenya's female population.",
        "Uganda's female population.",
    ]
    rendered, plan = compose_envelope_sentence(parts)
    assert rendered == "Kenya's and Uganda's female population."
    assert [(t.text, t.part_index) for t in plan if t.axis == "where"] == [
        ("Kenya", 0),
        ("Uganda", 1),
    ]


def test_differing_bylines_compose_one_sentence_per_part():
    """Parts whose bylines differ (here, the source) keep each heading with its own byline,
    one sentence per part."""
    where = _where_place_slot("country/KEN", "Kenya")
    window = TimeWindow(start_year=1960, end_year=2024)
    population = _definite(
        _spec(
            slots=[
                _what_slot("Total population"),
                where,
                _source_slot("wdi", "World Development Indicators"),
            ],
            window=window,
        ),
        variable_text="population",
    )
    urban = _definite(
        _spec(
            slots=[
                _what_slot("Total urban population"),
                where,
                _source_slot("kenya_census", "Kenya Census"),
            ],
            window=window,
        ),
        variable_text="urban population",
    )
    rendered, plan = compose_envelope_sentence([population, urban])
    assert rendered == (
        "Kenya's total population, 1960 to 2024. World Development Indicators. "
        "Kenya's total urban population, 1960 to 2024. Kenya Census."
    )
    assert [(t.part_index, t.segment) for t in plan if t.axis == "source"] == [
        (0, "byline"),
        (1, "byline"),
    ]
    assert [t.part_index for t in plan if t.axis == "where"] == [0, 1]


def test_disagreeing_range_windows_compose_one_sentence_per_part():
    """A period span opens with a comma, so spans that differ would read as list items
    inside the join; each part keeps its own sentence (C131)."""
    where = _where_place_slot("country/BRA", "Brazil")
    population = _definite(
        _spec(
            slots=[_what_slot("Total population"), where],
            window=TimeWindow(start_year=1960, end_year=2024),
        ),
        variable_text="population",
    )
    fertility = _definite(
        _spec(
            slots=[_what_slot("Female fertility rate"), where],
            window=TimeWindow(start_year=1960, end_year=2023),
        ),
        variable_text="fertility rate",
    )
    rendered, plan = compose_envelope_sentence([population, fertility])
    assert rendered == (
        "Brazil's total population, 1960 to 2024. Brazil's female fertility rate, 1960 to 2023."
    )
    when_tokens = [t for t in plan if t.axis == "when"]
    assert [t.part_index for t in when_tokens] == [0, 1]


def test_a_byline_only_one_part_states_composes_one_sentence_per_part():
    """A part with no source and a part with one have different bylines, so each keeps
    its own, the empty source placeholder included."""
    where = _where_place_slot("country/KEN", "Kenya")
    window = TimeWindow(start_year=1960, end_year=2024)
    population = _definite(
        _spec(slots=[_what_slot("Total population"), where, _unbound_source_slot()], window=window),
        variable_text="population",
    )
    gdp = _definite(
        _spec(
            slots=[
                _what_slot("Nominal GDP"),
                where,
                _source_slot("wdi", "World Development Indicators"),
            ],
            window=window,
        ),
        variable_text="GDP",
    )
    rendered, plan = compose_envelope_sentence([population, gdp])
    assert rendered == (
        "Kenya's total population, 1960 to 2024. "
        "Kenya's nominal GDP, 1960 to 2024. World Development Indicators."
    )
    source_tokens = [t for t in plan if t.axis == "source"]
    assert len(source_tokens) == 2
    assert {t.text for t in source_tokens} == {"", "World Development Indicators"}


def test_two_disagreeing_comma_axes_falls_back_to_per_part():
    """Two disagreeing comma axes trigger per-part composition."""
    where = _where_place_slot("country/BRA", "Brazil")
    population = _definite(
        _spec(
            slots=[
                _what_slot("Total population"),
                where,
                _source_slot("wdi", "World Development Indicators"),
            ],
            window=TimeWindow(start_year=1960, end_year=2024),
        ),
        variable_text="population",
    )
    fertility = _definite(
        _spec(
            slots=[_what_slot("Female fertility rate"), where, _source_slot("s2", "3 sources")],
            window=TimeWindow(start_year=1960, end_year=2023),
        ),
        variable_text="fertility rate",
    )
    rendered, plan = compose_envelope_sentence([population, fertility])
    assert rendered == (
        "Brazil's total population, 1960 to 2024. World Development Indicators. "
        "Brazil's female fertility rate, 1960 to 2023. 3 sources."
    )
    joins = [t for t in plan if t.kind == "text" and t.text == " " and t.part_index is None]
    assert len(joins) == 1


def test_null_variable_text_in_attribution_set_falls_back_to_per_part():
    """Without variable text, disagreeing sources use per-part sentences."""
    where = _where_place_slot("country/KEN", "Kenya")
    window = TimeWindow(start_year=1960, end_year=2024)
    population = _definite(
        _spec(
            slots=[
                _what_slot("Total population"),
                where,
                _source_slot("wdi", "World Development Indicators"),
            ],
            window=window,
        )
    )
    urban = _definite(
        _spec(
            slots=[
                _what_slot("Total urban population"),
                where,
                _source_slot("kenya_census", "Kenya Census"),
            ],
            window=window,
        )
    )
    rendered, _plan = compose_envelope_sentence([population, urban])
    assert rendered == (
        "Kenya's total population, 1960 to 2024. World Development Indicators. "
        "Kenya's total urban population, 1960 to 2024. Kenya Census."
    )


# Unresolved siblings prevent qualifier hoisting.


def test_asked_qualifier_does_not_hoist_across_unresolved_sibling():
    """A bound where slot stays in its part when a sibling has no data (C130)."""
    population = _definite(
        _spec(slots=[_what_slot("Total population"), _where_place_slot("country/BRA", "Brazil")])
    )
    gdp = _unresolved("GDP")
    rendered, plan = compose_envelope_sentence([population, gdp])
    assert rendered == "Answered 1 of 2 parts. Brazil's total population. No data."
    where_tokens = [t for t in plan if t.axis == "where"]
    assert len(where_tokens) == 1
    assert where_tokens[0].part_index == 0
    assert [t.part_index for t in plan if t.text == "No data."] == [1]


def test_read_back_qualifier_does_not_hoist_across_unresolved_sibling():
    """A coverage-derived where qualifier stays with its part across an unresolved sibling."""
    population = _definite(
        _spec(
            slots=[_what_slot("Total population"), _unbound_where_slot()],
            coverage=CoverageBreadth(
                has_data=True, window=None, dimensions=[BreadthDim(label="countries", count=3)]
            ),
        )
    )
    gdp = _unresolved("GDP")
    rendered, plan = compose_envelope_sentence([population, gdp])
    assert rendered == "Answered 1 of 2 parts. Total population across 3 countries. No data."
    where_tokens = [t for t in plan if t.axis == "where"]
    assert len(where_tokens) == 1
    assert where_tokens[0].part_index == 0


def test_unresolved_part_outcome_wording_by_status():
    """Beside another part, a candidates or unsupported part names its variable as a
    slot; a no-data plan that names what was checked is mirrored. Every token carries
    the part's index."""
    population = _definite(_spec(slots=[_what_slot("Total population")]))

    spec_a = _spec(slots=[_what_slot("A")])
    spec_b = _spec(slots=[_what_slot("B")])
    candidates = _candidates("GDP", spec_a, spec_b)
    _rendered, plan = compose_envelope_sentence([population, candidates])
    tokens = [t for t in plan if t.part_index == 1]
    assert [(t.kind, t.text) for t in tokens] == [("slot", "GDP"), ("text", " has 2 readings.")]
    assert all(t.part_index == 1 for t in tokens)

    no_data = NoDataPart(
        status="no_data",
        variable_text="fertility rate",
        warnings=[],
        no_data=NoData(reason="no_observations"),
        rendered_sentence="No data found for this query.",
        sentence_plan=plain_sentence_plan("No data found for this query."),
    )
    _rendered, plan = compose_envelope_sentence([population, no_data])
    tokens = [t for t in plan if t.part_index == 1]
    assert [(t.kind, t.text) for t in tokens] == [("text", "No data found for this query.")]

    unsupported = UnsupportedPart(
        status="unsupported",
        variable_text="GDP in PPP terms",
        warnings=[],
        unsupported=Unsupported(
            code="unit_request_unsupported",
            pointer="/identity/slots/0",
            detail="PPP terms are not supported.",
            definition={},
        ),
    )
    _rendered, plan = compose_envelope_sentence([population, unsupported])
    tokens = [t for t in plan if t.part_index == 1]
    assert [(t.kind, t.text) for t in tokens] == [
        ("slot", "GDP in PPP terms"),
        ("text", " PPP terms are not supported."),
    ]


def test_candidates_empty_variable_text_degrades_to_suffix_alone():
    """With no variable text, the candidates part's own sentence is mirrored."""
    population = _definite(_spec(slots=[_what_slot("Total population")]))
    spec_a = _spec(slots=[_what_slot("A")])
    spec_b = _spec(slots=[_what_slot("B")])
    candidates = CandidatesPart(
        status="candidates",
        variable_text=None,
        warnings=[],
        candidates=CandidateSet(
            ordering="broadest_first", max_candidates=5, specs=[spec_a, spec_b], truncated=False
        ),
        rendered_sentence="2 readings.",
        sentence_plan=plain_sentence_plan("2 readings."),
    )
    _rendered, plan = compose_envelope_sentence([population, candidates])
    tokens = [t for t in plan if t.part_index == 1]
    assert [(t.kind, t.text) for t in tokens] == [("text", "2 readings.")]


def test_unsupported_empty_variable_text_degrades_to_suffix_alone():
    """With no variable text, the unsupported detail stands alone as a sentence."""
    population = _definite(_spec(slots=[_what_slot("Total population")]))
    unsupported = UnsupportedPart(
        status="unsupported",
        variable_text=None,
        warnings=[],
        unsupported=Unsupported(
            code="unit_request_unsupported",
            pointer="/identity/slots/0",
            detail="The requested unit phrase is not supported.",
            definition={},
        ),
    )
    _rendered, plan = compose_envelope_sentence([population, unsupported])
    tokens = [t for t in plan if t.part_index == 1]
    assert [(t.kind, t.text) for t in tokens] == [
        ("text", "The requested unit phrase is not supported.")
    ]


# Avoid repeating variable terms in composed refusals.

_COMPOSED_TERM_REASONS_AND_OUTCOMES = {
    "variable_not_resolved": (
        "No match for ",
        ". Try a different wording or a related indicator.",
    ),
    "denominator_not_available": ("No series for ", " is in the form it asks for."),
    "no_matching_slice": (
        "No published series for ",
        " matches the combination this query asks for.",
    ),
    "binding_mismatch": (
        "No published series for ",
        " carries every condition this query asks for.",
    ),
    "role_incompatible": (
        "No data for ",
        " as a flow to or from any place named in this query.",
    ),
}


def test_composed_outcome_names_variable_once_for_each_term_quoting_reason():
    population = _definite(_spec(slots=[_what_slot("Total population")]))
    for reason in _COMPOSED_TERM_REASONS_AND_OUTCOMES:
        no_data = _no_data_with_reason(reason, "unemployment rate")
        rendered, _plan = compose_envelope_sentence([population, no_data])
        assert rendered.lower().count("unemployment rate") == 1, (reason, rendered)


def test_composed_outcome_wording_for_each_term_quoting_reason():
    """Pin the composed wording for each variable-term refusal reason."""
    population = _definite(_spec(slots=[_what_slot("Total population")]))
    for reason, (prefix, suffix) in _COMPOSED_TERM_REASONS_AND_OUTCOMES.items():
        no_data = _no_data_with_reason(reason, "Unemployment rate")
        _rendered, plan = compose_envelope_sentence([population, no_data])
        text = "".join(t.text for t in plan if t.part_index == 1)
        assert text == f"{prefix}Unemployment rate{suffix}", reason


def test_slot_keeps_its_own_casing_when_a_prefix_precedes_it():
    """Capitalize the prefix, not the user's variable text."""
    population = _definite(_spec(slots=[_what_slot("Total population")]))
    no_data = _no_data_with_reason("variable_not_resolved", "unemployment rate")
    _rendered, plan = compose_envelope_sentence([population, no_data])
    slot = next(t for t in plan if t.part_index == 1 and t.kind == "slot")
    assert slot.text == "unemployment rate"
    prefix = next(t for t in plan if t.part_index == 1 and t.kind == "text")
    assert prefix.text == "No match for "


def test_entity_reason_composed_outcome_unchanged():
    """Entity refusals are mirrored with their entity term and do not repeat the
    variable."""
    population = _definite(_spec(slots=[_what_slot("Total population")]))
    phrase = no_data_phrase("entity_not_resolved", entity_terms=["Atlantis"])
    no_data = NoDataPart(
        status="no_data",
        variable_text="GDP",
        warnings=[],
        no_data=NoData(reason="entity_not_resolved"),
        rendered_sentence=phrase,
        sentence_plan=plain_sentence_plan(phrase),
    )
    _rendered, plan = compose_envelope_sentence([population, no_data])
    assert [t.text for t in plan if t.part_index == 1] == [
        'Could not use "Atlantis" as a place for this query. Try a different place name.'
    ]


@pytest.mark.parametrize(
    ("reason", "entity_terms", "variable_text", "ambiguous_candidates"),
    [
        pytest.param(
            "entity_ambiguous",
            ["Congo"],
            "GDP",
            ["Congo [DRC]", "Congo [Republic]"],
            id="ambiguous",
        ),
        pytest.param(
            "entity_role_not_admitted",
            ["African Development Bank"],
            "GDP",
            None,
            id="role-not-admitted",
        ),
    ],
)
def test_composed_entity_outcome_keeps_offending_entity(
    reason: str,
    entity_terms: list[str],
    variable_text: str,
    ambiguous_candidates: list[str] | None,
):
    """The refusal names the unresolved entity beside the resolved sibling."""
    population = _definite(_spec(slots=[_what_slot("Total population")]))
    phrase = no_data_phrase(
        reason,  # type: ignore[arg-type]  # parametrized literal is narrowed at the model boundary
        variable_term=variable_text,
        entity_terms=entity_terms,
        ambiguous_candidates=ambiguous_candidates,
    )
    no_data = NoDataPart(
        status="no_data",
        variable_text=variable_text,
        warnings=[],
        no_data=NoData(reason=reason, entity_terms=entity_terms),  # type: ignore[arg-type]
        rendered_sentence=phrase,
        sentence_plan=plain_sentence_plan(phrase),
    )

    rendered, _plan = compose_envelope_sentence([population, no_data])

    assert rendered is not None
    assert phrase in rendered
    assert f'"{entity_terms[0]}"' in rendered


@pytest.mark.parametrize(
    "reason",
    [
        "variable_not_resolved",
        "denominator_not_available",
        "no_matching_slice",
        "binding_mismatch",
        "role_incompatible",
    ],
)
@pytest.mark.parametrize(
    "gate",
    ["no_member_has_data_at_place", "only_alternate_has_data"],
)
def test_composed_variable_outcome_keeps_structured_near_match_continuation(reason: str, gate: str):
    """Replacing the quoted head preserves the structured near-match continuation."""
    variable = "unemployment rate"
    near_match = _sentenced(
        _spec(entities=[_entity_to("country/ETH", "Ethiopia")])
        if gate == "no_member_has_data_at_place"
        else _spec()
    )
    if gate == "no_member_has_data_at_place":
        source_plan = no_data_sentence_plan(
            reason,
            gate=gate,
            flagged_match=near_match,
            variable_term=variable,
        )
    else:
        source_plan = no_data_sentence_plan(
            reason,
            gate=gate,
            alternate_match=near_match,
            variable_term=variable,
        )
    no_data = _no_data_with_plan(reason, variable, source_plan)

    rendered, plan = compose_envelope_sentence(
        [_definite(_spec(slots=[_what_slot("Total population")])), no_data]
    )

    continuation = source_plan[1:]
    assert rendered.endswith(sentence_from_plan(continuation))
    assert plan[-len(continuation) :] == [
        token.model_copy(update={"part_index": 1}) for token in continuation
    ]


def test_no_observations_keeps_its_place_gap_sentence_intact():
    """A place-gap no-data plan states the checked series in claim form and is mirrored
    whole beside another part."""
    source_plan = no_data_sentence_plan(
        "no_observations",
        gate="no_member_has_data_at_place",
        flagged_match=_sentenced(_spec(entities=[_entity_to("country/ETH", "Ethiopia")])),
        variable_term="unemployment rate",
    )
    no_data = _no_data_with_plan("no_observations", "unemployment rate", source_plan)

    rendered, _plan = compose_envelope_sentence(
        [_definite(_spec(slots=[_what_slot("Total population")])), no_data]
    )

    assert rendered.endswith(" No data for development finance flow to Ethiopia.")
    assert rendered.endswith(sentence_from_plan(source_plan))


@pytest.mark.parametrize(
    "gate",
    [
        "no_member_has_data_at_place",
        "only_alternate_has_data",
        "no_member_asserts_bound_properties",
    ],
)
def test_empty_variable_text_keeps_near_match_continuation(gate: str):
    """With empty variable text the part's whole refusal sentence is mirrored."""
    reason = "variable_not_resolved"
    near_match = (
        _sentenced(_spec(entities=[_entity_to("country/ETH", "Ethiopia")]))
        if gate == "no_member_has_data_at_place"
        else _sentenced(_spec())
    )
    kwargs = (
        {"alternate_match": near_match}
        if gate == "only_alternate_has_data"
        else {"flagged_match": near_match}
    )
    source_plan = no_data_sentence_plan(
        reason, gate=cast(NoDataGate, gate), variable_term="", **kwargs
    )
    no_data = _no_data_with_plan(reason, "", source_plan)

    rendered, plan = compose_envelope_sentence(
        [_definite(_spec(slots=[_what_slot("Total population")])), no_data]
    )

    assert ". ." not in rendered
    assert rendered.endswith(sentence_from_plan(source_plan))
    part_tokens = [t for t in plan if t.part_index == 1]
    assert _strip_part_index(part_tokens) == source_plan


@pytest.mark.parametrize(
    "reason",
    [
        "variable_not_resolved",
        "denominator_not_available",
        "no_matching_slice",
        "binding_mismatch",
        "role_incompatible",
    ],
)
@pytest.mark.parametrize(
    "gate",
    [
        "no_member_has_data_at_place",
        "only_alternate_has_data",
        "no_member_asserts_bound_properties",
    ],
)
def test_composed_variable_outcome_continuation_appears_exactly_once(reason: str, gate: str):
    """The continuation appears once in both sentence text and token plan."""
    variable = "unemployment rate"
    near_match = _sentenced(
        _spec(entities=[_entity_to("country/ETH", "Ethiopia")])
        if gate == "no_member_has_data_at_place"
        else _spec()
    )
    kwargs = (
        {"alternate_match": near_match}
        if gate == "only_alternate_has_data"
        else {"flagged_match": near_match}
    )
    source_plan = no_data_sentence_plan(
        reason, gate=cast(NoDataGate, gate), variable_term=variable, **kwargs
    )
    no_data = _no_data_with_plan(reason, variable, source_plan)

    rendered, plan = compose_envelope_sentence(
        [_definite(_spec(slots=[_what_slot("Total population")])), no_data]
    )

    continuation = source_plan[1:]
    if not continuation:
        pytest.skip("role_incompatible's bare-refusal branch carries no continuation to double")
    continuation_text = sentence_from_plan(continuation)
    assert rendered.count(continuation_text) == 1

    stamped = [token.model_copy(update={"part_index": 1}) for token in continuation]
    run_length = len(stamped)
    occurrences = sum(
        1 for i in range(len(plan) - run_length + 1) if plan[i : i + run_length] == stamped
    )
    assert occurrences == 1


def test_elected_variable_flagged_match_continuation_survives_composition():
    """Matching elected StatVars use "matching series" in the continuation."""
    variable = "population"
    flagged = _sentenced(_spec())
    elected = [_ref("sv1", "SV1")]
    source_plan = no_data_sentence_plan(
        "no_matching_slice",
        flagged_match=flagged,
        elected_stat_vars=elected,
        variable_term=variable,
    )
    no_data = _no_data_with_plan("no_matching_slice", variable, source_plan)

    rendered, plan = compose_envelope_sentence(
        [_definite(_spec(slots=[_what_slot("Total population")])), no_data]
    )

    assert "The matching series is development finance flow." in rendered
    continuation = source_plan[1:]
    assert continuation
    assert rendered.count(sentence_from_plan(continuation)) == 1
    assert plan[-len(continuation) :] == [
        token.model_copy(update={"part_index": 1}) for token in continuation
    ]


# Three-variable Oxford-comma join.


def test_three_variables_oxford_join():
    where = _where_place_slot("country/BRA", "Brazil")
    parts = [
        _definite(_spec(slots=[_what_slot(label), where]))
        for label in ("Total population", "Nominal GDP", "Life expectancy at birth")
    ]
    rendered, _plan = compose_envelope_sentence(parts)
    assert rendered == ("Brazil's total population, nominal GDP, and life expectancy at birth.")


# Empty qualifier placeholders.


def test_empty_placeholder_shared_when_every_part_leaves_it_empty():
    a = _definite(_spec(slots=[_what_slot("Total population"), _unbound_source_slot()]))
    b = _definite(_spec(slots=[_what_slot("Nominal GDP"), _unbound_source_slot()]))
    rendered, plan = compose_envelope_sentence([a, b])
    assert rendered == "Total population and nominal GDP."
    assert "(" not in rendered and ")" not in rendered
    source_tokens = [t for t in plan if t.axis == "source"]
    assert len(source_tokens) == 1
    assert source_tokens[0].text == ""
    assert source_tokens[0].part_index is None


def test_source_axis_declared_by_only_one_part_does_not_hoist():
    """A source axis absent from one part stays inline as an empty placeholder."""
    a = _definite(_spec(slots=[_what_slot("Total population"), _unbound_source_slot()]))
    b = _definite(_spec(slots=[_what_slot("Nominal GDP")]))
    rendered, plan = compose_envelope_sentence([a, b])
    assert rendered == "Total population and nominal GDP."
    source_tokens = [t for t in plan if t.axis == "source"]
    assert len(source_tokens) == 1
    assert source_tokens[0].text == ""
    assert source_tokens[0].part_index == 0


# ---------------------------------------------------------------------------
# A null variable_text never yields a leading/stray " and ".
# ---------------------------------------------------------------------------


def test_null_variable_text_does_not_produce_stray_connective():
    population = _definite(_spec(slots=[_what_slot("Total population")]))
    unnamed = _unresolved(None)
    rendered, plan = compose_envelope_sentence([population, unnamed])
    assert rendered == "Answered 1 of 2 parts. Total population. No data."
    assert " and " not in rendered
    assert not [t for t in plan if t.part_index == 1 and t.kind == "slot"]


def test_an_ambiguous_wording_keeps_its_slot_token_beside_a_resolved_sibling():
    """Per-part composition keeps the ambiguous wording's slot token, under the part's
    index, and the part's own sentence is mirrored whole."""
    population = _definite(_spec(slots=[_what_slot("Total population")]))
    plan = no_data_sentence_plan(
        "entity_ambiguous",
        entity_terms=["Congo"],
        ambiguous_candidates=["Congo [DRC]", "Congo [Republic]"],
        ambiguous_roles={"Congo": "destination"},
    )
    ambiguous = _no_data_with_plan("entity_ambiguous", "GDP", plan)

    rendered, composed = compose_envelope_sentence([population, ambiguous])

    (token,) = [t for t in composed if t.part_index == 1 and t.kind == "slot" and t.axis != "what"]
    assert token.text == "Congo"
    assert token.role == "destination"
    assert rendered is not None
    assert rendered.endswith(f". {sentence_from_plan(plan)}")


# Departures close the composed sentence.


def _with_departures(part: DefinitePart, *texts: str) -> DefinitePart:
    departures = [Departure(code="year_unavailable", binding_ref="when:", text=t) for t in texts]
    return part.model_copy(update={"departures": departures})


def _departure_tokens(plan: list[SentenceToken]) -> list[tuple[str, int | None]]:
    return [(t.text, t.part_index) for t in plan if t.segment == "departure" and t.text.strip()]


def test_a_single_part_composes_to_its_spec_sentence_then_its_departures():
    """C129: the spec sentence, then each departure as a departure-segment token under the
    part's index; the spec's own plan carries none."""
    part = _with_departures(
        _definite(
            _spec(
                slots=[_what_slot("ODA grants")],
                entities=[
                    _entity_to("country/KEN", "Kenya"),
                    _entity_from("country/DEU", "Germany"),
                ],
                resolved_sources=[_ref("crs", "OECD CRS")],
            )
        ),
        "You asked for 2030. The data ends in 2024.",
    )
    rendered, plan = compose_envelope_sentence([part])
    assert rendered == (
        f"{part.spec.resolved.rendered_sentence} You asked for 2030. The data ends in 2024."
    )
    assert _departure_tokens(plan) == [("You asked for 2030. The data ends in 2024.", 0)]
    assert not [t for t in part.spec.resolved.sentence_plan or [] if t.segment == "departure"]


def test_departures_follow_the_joined_headings_and_byline_in_part_order():
    where = _where_place_slot("country/BRA", "Brazil")
    population = _with_departures(
        _definite(_spec(slots=[_what_slot("Total population"), where, _source_slot("wdi", "WDI")])),
        "First.",
    )
    gdp = _definite(_spec(slots=[_what_slot("Nominal GDP"), where, _source_slot("wdi", "WDI")]))
    gdp = _with_departures(gdp, "Second.", "Third.")
    rendered, plan = compose_envelope_sentence([population, gdp])
    assert rendered == "Brazil's total population and nominal GDP. WDI. First. Second. Third."
    assert _departure_tokens(plan) == [("First.", 0), ("Second.", 1), ("Third.", 1)]


def test_per_part_sentences_keep_each_parts_departures_after_it():
    population = _with_departures(
        _definite(_spec(slots=[_what_slot("Total population"), _source_slot("wdi", "WDI")])),
        "First.",
    )
    gdp = _definite(_spec(slots=[_what_slot("Nominal GDP"), _source_slot("imf", "IMF")]))
    rendered, plan = compose_envelope_sentence([population, gdp])
    assert rendered == "Total population. WDI. First. Nominal GDP. IMF."
    assert _departure_tokens(plan) == [("First.", 0)]


def test_a_shared_counterpart_is_stated_once_in_the_family_wording():
    """Parts that differ in actor and noun share the counterpart, stated once after the
    joined phrases; each possessive lead keeps its own casing and part index."""
    parts = [
        _definite(
            _spec(
                slots=[_what_slot(noun)],
                entities=[_entity_to("country/ETH", "Ethiopia"), donor],
            )
        )
        for noun, donor in (
            ("Health ODA", _entity_from("country/USA", "USA")),
            ("Education ODA", _entity_from("country/GBR", "Britain")),
        )
    ]
    rendered, plan = compose_envelope_sentence(parts)
    assert rendered == "USA's health ODA and Britain's education ODA to Ethiopia."
    assert [(t.text, t.part_index) for t in plan if t.role == "destination"] == [("Ethiopia", None)]


# Recorded queries, rebuilt as the Specs the engine served.

_SHARE_DEPARTURE = Departure(
    code="operation_not_applied",
    text="You asked for a share. This shows the amounts behind it, because Search does not compute shares.",
)
_CRS_BYLINE = " OECD Creditor Reporting System (CRS)."


def _crs_how(prop: str, label: str) -> Slot:
    code = f"{prop}/{label.split()[0]}"
    return Slot(
        key=_slot_key("how", prop, prop, prop),
        binding=BindingValue(value=SlotValue(ref=_ref(code, label), value_kind="entity")),
    )


def _crs_part(
    donor: str,
    recipient: str,
    years: tuple[int, int],
    *departures: Departure,
    what: str = "gross ODA disbursements",
    how: tuple[Slot, ...] = (),
) -> DefinitePart:
    """A CRS definite part: donor lead, measure noun, recipient, how values, span and
    source."""
    spec = _spec(
        slots=[_what_slot(what), *how],
        entities=[
            _entity_from(f"country/{donor}", donor),
            _entity_to(f"country/{recipient}", recipient),
        ],
        window=TimeWindow(start_year=years[0], end_year=years[1]),
        resolved_sources=[_ref("CRS", "OECD Creditor Reporting System (CRS)")],
    )
    return _definite(spec).model_copy(update={"departures": list(departures)})


def test_crs_040_equal_headings_are_stated_once():
    """crs-040: two parts serve the same heading. It is stated once, unindexed, never
    joined to itself ("gross ODA disbursements and gross ODA disbursements")."""
    rendered, plan = compose_envelope_sentence(
        [
            _crs_part("Germany", "Kenya", (2015, 2020), _SHARE_DEPARTURE),
            _crs_part("Germany", "Kenya", (2015, 2020)),
        ]
    )
    assert rendered == (
        f"Germany's gross ODA disbursements to Kenya, 2015 to 2020.{_CRS_BYLINE} "
        f"{_SHARE_DEPARTURE.text}"
    )
    assert [t.part_index for t in plan if t.axis in ("what", "where")] == [None, None]
    assert _departure_tokens(plan) == [(_SHARE_DEPARTURE.text, 0)]


def test_equal_nouns_with_different_counterparts_keep_their_own_headings():
    """Parts whose noun phrases are equal keep their own counterparts, one heading each,
    and the byline they share follows once."""
    rendered, plan = compose_envelope_sentence(
        [
            _crs_part("Germany", "Kenya", (2015, 2020)),
            _crs_part("Germany", "Ghana", (2015, 2020)),
        ]
    )
    assert rendered == (
        "Germany's gross ODA disbursements to Kenya, 2015 to 2020. "
        f"Germany's gross ODA disbursements to Ghana, 2015 to 2020.{_CRS_BYLINE}"
    )
    assert [t.part_index for t in plan if t.axis == "where"] == [0, 1]
    assert {t.part_index for t in plan if t.segment == "byline"} == {None}


def test_mls_034_a_byline_both_parts_share_is_stated_once():
    """mls-034: headings that cannot join (their spans differ) each state their own
    sentence; the byline they share follows once, then the departures."""
    rendered, plan = compose_envelope_sentence(
        [
            _crs_part("Germany", "Ghana", (1973, 2024), _SHARE_DEPARTURE),
            _crs_part(
                "Germany",
                "Ghana",
                (2004, 2024),
                how=(_crs_how("crsPurpose", "core contribution and pooled programme and fund"),),
            ),
        ]
    )
    assert rendered == (
        "Germany's gross ODA disbursements to Ghana, 1973 to 2024. Germany's gross ODA "
        "disbursements to Ghana for core contribution and pooled programme and fund, "
        f"2004 to 2024.{_CRS_BYLINE} {_SHARE_DEPARTURE.text}"
    )
    assert [t.part_index for t in plan if t.axis == "when"] == [0, 1]
    assert {t.part_index for t in plan if t.segment == "byline"} == {None}


def test_a_departure_two_parts_carry_is_stated_once():
    """The same departure text from two parts is stated once, with no part index."""
    rendered, plan = compose_envelope_sentence(
        [
            _crs_part("Germany", "Kenya", (2015, 2020), _SHARE_DEPARTURE),
            _crs_part("France", "Kenya", (2015, 2020), _SHARE_DEPARTURE),
        ]
    )
    assert rendered is not None and rendered.count(_SHARE_DEPARTURE.text) == 1
    assert _departure_tokens(plan) == [(_SHARE_DEPARTURE.text, None)]


def test_weo_041_a_no_data_heading_two_parts_state_is_stated_once():
    """weo-041: both parts checked the same series. Its no-data heading is stated once,
    unindexed, and each part's departure follows under its own index."""
    gross = 'You asked for "gross". This check left it out.'
    net = 'You asked for "net". This check left it out.'

    def part(text: str) -> NoDataPart:
        plan = [
            SentenceToken(kind="text", text="No data for "),
            SentenceToken(kind="text", text="Ghana"),
            SentenceToken(kind="text", text="'s"),
            SentenceToken(kind="text", text=" "),
            SentenceToken(kind="text", text="government debt"),
            SentenceToken(kind="text", text="."),
            SentenceToken(kind="text", text=" ", segment="departure"),
            SentenceToken(kind="text", text=text, segment="departure"),
        ]
        return _no_data_with_plan("no_observations", "debt", plan)

    rendered, plan = compose_envelope_sentence([part(gross), part(net)])
    assert rendered == f"No data for Ghana's government debt. {gross} {net}"
    assert {t.part_index for t in plan if t.segment == "heading" and t.text.strip()} == {None}
    assert _departure_tokens(plan) == [(gross, 0), (net, 1)]


def test_mls_037_mixed_parts_keep_each_departure_after_its_own_sentence():
    """mls-037: a no-data part and a definite part carry the same departure. Each follows
    its own part's sentence, under that part's index, so the reader can tell which part
    it belongs to."""
    growth = "You asked for a growth rate. Search does not compute growth rates."
    refusal = '"education" names a category the search could not apply to this query.'
    no_data_plan = [
        SentenceToken(kind="text", text=refusal),
        SentenceToken(kind="text", text=" ", segment="departure"),
        SentenceToken(kind="text", text=growth, segment="departure"),
    ]
    no_data = _no_data_with_plan("qualifier_not_bound", "education", no_data_plan)
    definite = _crs_part(
        "Denmark",
        "Kenya",
        (2015, 2024),
        Departure(code="operation_not_applied", text=growth),
        how=(_crs_how("crsPurpose", "health"),),
    )

    rendered, plan = compose_envelope_sentence([no_data, definite])

    assert rendered == (
        f"Answered 1 of 2 parts. {refusal} {growth} Denmark's gross ODA disbursements to "
        "Kenya for health, "
        f"2015 to 2024.{_CRS_BYLINE} {growth}"
    )
    assert _departure_tokens(plan) == [(growth, 0), (growth, 1)]


def test_ids_028_no_data_parts_with_one_reason_state_it_once():
    """ids-028: two parts refused for the same reason name both variables in one clause;
    each part's departure follows."""
    debt_share = _no_data_with_plan(
        "role_incompatible",
        "debt share",
        no_data_sentence_plan(
            "role_incompatible", variable_term="debt share", departures=[_SHARE_DEPARTURE]
        ),
    )
    debt = _no_data_with_plan(
        "role_incompatible",
        "debt",
        no_data_sentence_plan("role_incompatible", variable_term="debt"),
    )
    rendered, plan = compose_envelope_sentence([debt_share, debt])
    assert rendered == (
        "No data for debt share and debt as a flow to or from any place named in this "
        f"query. {_SHARE_DEPARTURE.text}"
    )
    assert [(t.text, t.part_index) for t in plan if t.axis == "what"] == [
        ("debt share", 0),
        ("debt", 1),
    ]


def _crs_033_part(what: str) -> DefinitePart:
    """crs-033's Spec: the counterpart precedes the purpose and finance type."""
    return _crs_part(
        "Japan",
        "Vietnam",
        (1994, 2024),
        what=what,
        how=(
            _crs_how("crsPurpose", "economic infrastructure and services"),
            _crs_how("crsTypeOfFinance", "debt instruments"),
        ),
    )


def test_crs_033_a_single_part_envelope_keeps_its_counterpart_before_the_how_values():
    """crs-033: the envelope of one part reads as its spec sentence; the counterpart
    stays before the purpose and finance type."""
    part = _crs_033_part("gross ODA disbursements")
    rendered, _plan = compose_envelope_sentence([part])
    assert rendered == part.spec.resolved.rendered_sentence
    assert rendered == (
        "Japan's gross ODA disbursements to Vietnam for economic infrastructure and services "
        "as debt instruments, 1994 to 2024." + _CRS_BYLINE
    )


def test_a_counterpart_stated_before_how_values_stays_in_each_phrase():
    """A counterpart two parts share but state before their how values stays in each
    phrase, in its place; the period both state last still hoists."""
    gross = _crs_033_part("gross ODA disbursements")
    net = _crs_033_part("net ODA disbursements")
    rendered, plan = compose_envelope_sentence([gross, net])
    assert rendered == (
        "Japan's gross ODA disbursements to Vietnam for economic infrastructure and services "
        "as debt instruments and net ODA disbursements to Vietnam for economic infrastructure "
        "and services as debt instruments, 1994 to 2024." + _CRS_BYLINE
    )
    assert [t.part_index for t in plan if t.axis == "where"] == [0, 1]
    assert [t.part_index for t in plan if t.axis == "when"] == [None]


def test_leads_with_one_label_for_two_entities_are_each_stated():
    """Two donors that share a label but not a dcid are two leads, each under its own
    part, before the noun the parts share."""
    parts = [
        _definite(
            _spec(
                slots=[_what_slot("Health ODA")],
                entities=[_entity_to("country/ETH", "Ethiopia"), _entity_from(dcid, "Georgia")],
            )
        )
        for dcid in ("country/GEO", "geoId/13")
    ]
    rendered, plan = compose_envelope_sentence(parts)
    assert rendered == "Georgia's and Georgia's health ODA to Ethiopia."
    leads = [(t.part_index, t.entity_refs) for t in plan if t.role == "origin"]
    assert leads == [(0, ["country/GEO"]), (1, ["geoId/13"])]


# ---------------------------------------------------------------------------
# The assumptions clause and the mixed opener.
# ---------------------------------------------------------------------------

_ASSUMED_ODA = Assumption(kind="declared_default", binding_ref="how:crsTypeOfFlow", text="ODA")
_ASSUMED_PROVIDERS = Assumption(kind="open_role", binding_ref="origin:", text="every provider")


def _assuming(part: DefinitePart, *assumptions: Assumption) -> DefinitePart:
    return part.model_copy(update={"assumptions": list(assumptions)})


def _clause_tokens(plan: list[SentenceToken]) -> list[tuple[str, int | None]]:
    return [(t.text, t.part_index) for t in plan if t.segment == "assumption"]


def test_a_joined_sentence_closes_with_one_clause_naming_each_part_item():
    """Two parts under one byline write one clause after their departures. An item both
    carry is named once with no part index, and an item of one part carries its index."""
    rendered, plan = compose_envelope_sentence(
        [
            _assuming(_crs_part("Germany", "Kenya", (2015, 2020)), _ASSUMED_ODA),
            _assuming(
                _crs_part("Germany", "Ghana", (2015, 2020), _SHARE_DEPARTURE),
                _ASSUMED_ODA,
                _ASSUMED_PROVIDERS,
            ),
        ]
    )
    assert rendered.endswith(f"{_SHARE_DEPARTURE.text} Assumed: ODA; every provider.")
    assert rendered.count("Assumed: ") == 1
    assert _clause_tokens(plan) == [
        (" ", None),
        ("Assumed: ", None),
        ("ODA", None),
        ("; ", None),
        ("every provider", 1),
        (".", None),
    ]


def test_a_part_by_part_sentence_closes_each_part_with_its_own_clause():
    """Beside a part that is not definite, the definite part's clause follows its own
    departures, under its index, before the next part's sentence."""
    definite = _assuming(
        _crs_part("Germany", "Kenya", (2015, 2020), _SHARE_DEPARTURE), _ASSUMED_ODA
    )
    rendered, plan = compose_envelope_sentence([definite, _unresolved("GDP")])
    assert f"{_SHARE_DEPARTURE.text} Assumed: ODA. No data." in rendered
    assert {index for _text, index in _clause_tokens(plan)} == {0}


@pytest.mark.parametrize(
    ("second", "sentences", "owners"),
    [
        # The second part's clause differs, so it keeps its own sentence before it.
        pytest.param(
            (_ASSUMED_ODA,),
            ["Assumed: every provider.", "Assumed: ODA."],
            [0, 1],
            id="own-clauses",
        ),
        # The whole statement repeats, so it is stated once, unowned.
        pytest.param((_ASSUMED_PROVIDERS,), ["Assumed: every provider."], [None], id="alike"),
    ],
)
def test_a_part_whose_sentence_repeats_keeps_it_when_its_clause_differs(
    second: tuple[Assumption, ...], sentences: list[str], owners: list[int | None]
):
    """Two definite parts that read alike, beside a part that is not definite, each state
    their sentence and clause unless the whole statement repeats."""
    heading = "Germany's gross ODA disbursements to Kenya, 2015 to 2020." + _CRS_BYLINE
    rendered, plan = compose_envelope_sentence(
        [
            _assuming(_crs_part("Germany", "Kenya", (2015, 2020)), _ASSUMED_PROVIDERS),
            _assuming(_crs_part("Germany", "Kenya", (2015, 2020)), *second),
            _unresolved("GDP"),
        ]
    )
    statements = " ".join(f"{heading} {clause}" for clause in sentences)
    assert rendered == f"Answered 2 of 3 parts. {statements} No data."
    assert rendered == sentence_from_plan(plan)
    assert [t.part_index for t in plan if t.text == "Assumed: "] == owners


@pytest.mark.parametrize(
    ("folded_assumptions", "clause"),
    [
        # Both carry the item, so the clause belongs to both and carries no part index.
        pytest.param(
            [_ASSUMED_ODA],
            [(" ", None), ("Assumed: ", None), ("ODA", None), (".", None)],
            id="shared",
        ),
        # Only the part folded into carries it, so the whole clause is that part's.
        pytest.param([], [(" ", 0), ("Assumed: ", 0), ("ODA", 0), (".", 0)], id="own"),
        # The folded part alone carries the providers, so that item is its own.
        pytest.param(
            [_ASSUMED_ODA, _ASSUMED_PROVIDERS],
            [
                (" ", None),
                ("Assumed: ", None),
                ("ODA", None),
                ("; ", None),
                ("every provider", 1),
                (".", None),
            ],
            id="folded-own",
        ),
    ],
)
def test_a_part_others_fold_into_states_its_clause_for_all_of_them(
    folded_assumptions: list[Assumption], clause: list[tuple[str, int | None]]
):
    """A part and the parts folded into it state one clause naming every item any of
    them carries. An item all carry has no part index, and an item one alone carries has
    its part's."""
    representative = _assuming(_crs_part("Germany", "Kenya", (2015, 2020)), _ASSUMED_ODA)
    folded = _assuming(representative, *folded_assumptions).model_copy(
        update={"warnings": [Warning(code="binding_folded", severity="info", message="folded")]}
    )
    _rendered, plan = compose_envelope_sentence([representative, folded])
    assert _clause_tokens(plan) == clause


@pytest.mark.parametrize(
    "after", [pytest.param([], id="joined"), pytest.param([_unresolved("GDP")], id="per-part")]
)
def test_a_folded_part_keeps_its_items_beside_a_separate_part_of_the_same_spec(
    after: list[NoDataPart],
):
    """A separate part can carry the Spec a part folded into. The folded part's own item
    stays in the clause of the part it folded into."""
    representative = _assuming(_crs_part("Germany", "Kenya", (2015, 2020)), _ASSUMED_ODA)
    folded = _assuming(representative, _ASSUMED_ODA, _ASSUMED_PROVIDERS).model_copy(
        update={"warnings": [Warning(code="binding_folded", severity="info", message="folded")]}
    )
    compared = representative.model_copy(update={"separate": True})
    _rendered, plan = compose_envelope_sentence([representative, folded, compared, *after])
    assert ("every provider", 1) in _clause_tokens(plan)


@pytest.mark.parametrize(
    ("folds", "opener"),
    [
        pytest.param(0, "Answered 1 of 2 parts.", id="definite-and-no-data"),
        # A part folded into the definite one is answered with it.
        pytest.param(1, "Answered 2 of 3 parts.", id="with-a-folded-part"),
    ],
)
def test_a_mixed_answer_opens_with_how_many_parts_it_answered(folds: int, opener: str):
    """A definite part beside a no-data part gives a plan that opens with two unowned
    coverage tokens, the opener and a space, and no other token takes that segment. The
    count of answered parts includes the parts folded into a definite one."""
    spec = _spec(slots=[_what_slot("Total population")])
    rendered, plan = compose_envelope_sentence(
        [_definite(spec), *[_folded(spec)] * folds, _unresolved("GDP")]
    )
    assert rendered == f"{opener} Total population. No data."
    assert [(t.text, t.segment, t.part_index) for t in plan[:2]] == [
        (opener, "coverage", None),
        (" ", "coverage", None),
    ]
    assert all(t.segment != "coverage" for t in plan[2:])


def test_a_mixed_answer_with_no_definite_part_has_no_opener():
    """Candidates beside no data answer nothing yet, and each part's sentence states its
    own outcome."""
    a = _spec(slots=[_what_slot("GDP")])
    b = _spec(slots=[_what_slot("GDP per capita")])
    rendered, _plan = compose_envelope_sentence([_candidates("GDP", a, b), _unresolved("debt")])
    assert not rendered.startswith("Answered")
