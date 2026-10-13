"""Pins request and extraction fields that distinguish response-cache keys.

A collision serves one query's cached answer to a different query."""

from __future__ import annotations

from typing import Literal

import pytest

from search.engine.config import settings
from search.engine.extract import ExtractedDate, ExtractedEntity
from search.engine.qualifier_match import CodedQualifier, QualifierMatch, recall_text
from search.engine.response_cache_flow import compute_layer2_signals
from search.engine.response_cache_keys import (
    layer1_key,
    layer2_key,
    normalise_query,
    parsed_key,
    scope_document,
)
from search.models import (
    Operation,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    ResolveOptions,
    SourceMention,
)


def _scope(
    *,
    schema_version: int = 1,
    options: ResolveOptions | None = None,
    max_candidates: int = 6,
    seam_default: bool = True,
    qualifier_coverage: Literal["log", "refuse"] = "log",
    build_commit: str = "commit-a",
    model: str = "model-a",
    seed: int | None = 1,
    today: str = "2026-09-20",
    graph_generation: str = "gen-a",
    coverage_stamp: str = "coverage-a",
    conversion_catalogue: str | None = "catalogue-a",
    origin_catalogue: str = "build-a",
    un_system_enabled: bool = True,
) -> dict:
    return scope_document(
        schema_version=schema_version,
        options=options,
        max_candidates=max_candidates,
        seam_default=seam_default,
        qualifier_coverage=qualifier_coverage,
        build_commit=build_commit,
        model=model,
        seed=seed,
        today=today,
        graph_generation=graph_generation,
        coverage_stamp=coverage_stamp,
        conversion_catalogue=conversion_catalogue,
        origin_catalogue=origin_catalogue,
        un_system_enabled=un_system_enabled,
    )


def _entity(**overrides: object) -> ExtractedEntity:
    fields: dict[str, object] = {
        "text": "Kenya",
        "span": "Kenya",
        "flow_role": "destination",
        "role_span": "to",
    }
    fields.update(overrides)
    return ExtractedEntity(**fields)


def _date(**overrides: object) -> ExtractedDate:
    fields: dict[str, object] = {"kind": "point", "start": "2010", "end": None}
    fields.update(overrides)
    return ExtractedDate(**fields)


def _coded_qualifier(**overrides: object) -> CodedQualifier:
    fields: dict[str, object] = {
        "phrase": "type of finance 110",
        "spans": (),
        "matches": (
            QualifierMatch(
                code_system="crsTypeOfFinance",
                code_dcid="CRSTypeOfFinance110",
                label="Standard grant",
                source="code",
                matched_text="110",
            ),
        ),
    }
    fields.update(overrides)
    return CodedQualifier(**fields)


def _layer2_key(
    *,
    variables: list[str] | None = None,
    entities: list[ExtractedEntity] | None = None,
    dates: list[ExtractedDate] | None = None,
    dropped_qualifiers: list[str] | None = None,
    variable_qualifiers: list[tuple[CodedQualifier, ...]] | None = None,
    operations: list[tuple[Operation, ...]] | None = None,
    sources: list[tuple[SourceMention, ...]] | None = None,
    role_contradictions: list[dict[str, object]] | None = None,
    query_text_qualifier_matches: list[dict[str, object]] | None = None,
    query_text_qualifier_ties: list[dict[str, object]] | None = None,
    query_text_qualifier_readings: list[dict[str, object]] | None = None,
    query_text_qualifier_occurrences: list[dict[str, object]] | None = None,
    unsupported_qualifier_phrases: list[str] | None = None,
    declared_default_trigger_words: list[dict[str, str]] | None = None,
    declared_default_slots_named: list[str] | None = None,
    scope: dict | None = None,
    source_codes: list[str | None] | None = None,
    separate: list[bool] | None = None,
) -> str:
    """`layer2_key` with every raw-text signal defaulting to empty, for a
    test that means to vary exactly one argument."""
    resolved_variables = variables if variables is not None else ["GDP"]
    return layer2_key(
        variables=resolved_variables,
        entities=entities if entities is not None else [_entity()],
        dates=dates if dates is not None else [_date()],
        dropped_qualifiers=dropped_qualifiers if dropped_qualifiers is not None else [],
        variable_qualifiers=(
            variable_qualifiers
            if variable_qualifiers is not None
            else [() for _ in resolved_variables]
        ),
        operations=operations if operations is not None else [() for _ in resolved_variables],
        sources=sources if sources is not None else [() for _ in resolved_variables],
        role_contradictions=role_contradictions if role_contradictions is not None else [],
        query_text_qualifier_matches=(
            query_text_qualifier_matches if query_text_qualifier_matches is not None else []
        ),
        query_text_qualifier_ties=(
            query_text_qualifier_ties if query_text_qualifier_ties is not None else []
        ),
        query_text_qualifier_readings=(
            query_text_qualifier_readings if query_text_qualifier_readings is not None else []
        ),
        query_text_qualifier_occurrences=(
            query_text_qualifier_occurrences if query_text_qualifier_occurrences is not None else []
        ),
        unsupported_qualifier_phrases=(
            unsupported_qualifier_phrases if unsupported_qualifier_phrases is not None else []
        ),
        declared_default_trigger_words=(
            declared_default_trigger_words if declared_default_trigger_words is not None else []
        ),
        declared_default_slots_named=(
            declared_default_slots_named if declared_default_slots_named is not None else []
        ),
        scope=scope if scope is not None else _scope(),
        source_codes=source_codes if source_codes is not None else (),
        separate=separate if separate is not None else (),
    )


def test_normalise_query_collapses_whitespace_and_nfkc_but_preserves_case() -> None:
    """Layer 1 preserves case because extraction can change; both layers collapse whitespace."""
    a = normalise_query("What is  GDP   in Kenya?")
    b = normalise_query("What is GDP in Kenya?")
    assert a == b  # only whitespace differs

    assert normalise_query("Show US aid") != normalise_query("show us aid")


def test_two_wordings_with_identical_extraction_share_layer2_key() -> None:
    """Different wordings with the same extraction and raw-text signals share a key."""
    key_a = _layer2_key()
    key_b = _layer2_key()
    assert key_a == key_b


def test_flow_role_alone_changes_layer2_key() -> None:
    """Changing an entity's flow role changes the key."""
    origin_kenya = _layer2_key(
        variables=["aid"],
        entities=[_entity(text="Kenya", flow_role="origin", role_span="from")],
        dates=[],
    )
    destination_kenya = _layer2_key(
        variables=["aid"],
        entities=[_entity(text="Kenya", flow_role="destination", role_span="to")],
        dates=[],
    )
    assert origin_kenya != destination_kenya


def test_an_entitys_members_flag_alone_changes_layer2_key() -> None:
    """Africa read as its members and Africa read as one place keep separate entries."""
    one_place = _layer2_key(entities=[_entity(text="Africa")])
    its_members = _layer2_key(entities=[_entity(text="Africa", members=True)])
    assert its_members != one_place
    assert one_place == _layer2_key(entities=[_entity(text="Africa", members=False)])


def test_variable_qualifiers_alone_changes_layer2_key_at_n_ge_2() -> None:
    """Multi-variable keys include coded qualifiers because they lack a raw `query_text` signal."""
    finance_110 = _layer2_key(
        variables=["grant equivalent", "aid"],
        dates=[],
        variable_qualifiers=[
            (
                _coded_qualifier(
                    phrase="type of finance 110",
                    matches=(
                        QualifierMatch(
                            code_system="crsTypeOfFinance",
                            code_dcid="CRSTypeOfFinance110",
                            label="Standard grant",
                            source="code",
                            matched_text="110",
                        ),
                    ),
                ),
            ),
            (),
        ],
    )
    finance_421 = _layer2_key(
        variables=["grant equivalent", "aid"],
        dates=[],
        variable_qualifiers=[
            (
                _coded_qualifier(
                    phrase="type of finance 421",
                    matches=(
                        QualifierMatch(
                            code_system="crsTypeOfFinance",
                            code_dcid="CRSTypeOfFinance421",
                            label="Standard loan",
                            source="code",
                            matched_text="421",
                        ),
                    ),
                ),
            ),
            (),
        ],
    )
    assert finance_110 != finance_421

    # Matching qualifiers also share a key.
    finance_110_again = _layer2_key(
        variables=["grant equivalent", "aid"],
        dates=[],
        variable_qualifiers=[
            (
                _coded_qualifier(
                    phrase="type of finance 110",
                    matches=(
                        QualifierMatch(
                            code_system="crsTypeOfFinance",
                            code_dcid="CRSTypeOfFinance110",
                            label="Standard grant",
                            source="code",
                            matched_text="110",
                        ),
                    ),
                ),
            ),
            (),
        ],
    )
    assert finance_110 == finance_110_again


def test_variable_qualifiers_phrase_alone_changes_layer2_key() -> None:
    """The qualifier phrase appears in warnings and resolved text, so it remains part of the key."""
    same_match = QualifierMatch(
        code_system="crsTypeOfFinance",
        code_dcid="CRSTypeOfFinance110",
        label="Standard grant",
        source="code",
        matched_text="110",
    )
    phrase_a = _layer2_key(
        variable_qualifiers=[
            (_coded_qualifier(phrase="type of finance 110", matches=(same_match,)),)
        ]
    )
    phrase_b = _layer2_key(
        variable_qualifiers=[(_coded_qualifier(phrase="finance type 110", matches=(same_match,)),)]
    )
    assert phrase_a != phrase_b


def test_variable_qualifiers_order_changes_layer2_key_when_it_changes_recall_text() -> None:
    """Qualifier order changes the `recall_text` sent to retrieval, so the layer-2 key changes."""
    finance = _coded_qualifier(
        phrase="type of finance 110",
        matches=(
            QualifierMatch(
                code_system="crsTypeOfFinance",
                code_dcid="CRSTypeOfFinance110",
                label="Standard grant",
                source="code",
                matched_text="110",
            ),
        ),
    )
    flow = _coded_qualifier(
        phrase="type of flow 10",
        matches=(
            QualifierMatch(
                code_system="crsTypeOfFlow",
                code_dcid="CRSTypeOfFlow10",
                label="ODA",
                source="code",
                matched_text="10",
            ),
        ),
    )
    variable = "GDP"
    assert recall_text(variable, (finance, flow)) != recall_text(variable, (flow, finance))

    finance_then_flow = _layer2_key(variables=[variable], variable_qualifiers=[(finance, flow)])
    flow_then_finance = _layer2_key(variables=[variable], variable_qualifiers=[(flow, finance)])
    assert finance_then_flow != flow_then_finance


def test_variable_qualifiers_order_does_not_change_layer2_key_when_recall_text_agrees() -> None:
    """If `variable` already contains both phrases, qualifier order leaves retrieval text and key unchanged."""
    finance = _coded_qualifier(
        phrase="type of finance 110",
        matches=(
            QualifierMatch(
                code_system="crsTypeOfFinance",
                code_dcid="CRSTypeOfFinance110",
                label="Standard grant",
                source="code",
                matched_text="110",
            ),
        ),
    )
    flow = _coded_qualifier(
        phrase="type of flow 10",
        matches=(
            QualifierMatch(
                code_system="crsTypeOfFlow",
                code_dcid="CRSTypeOfFlow10",
                label="ODA",
                source="code",
                matched_text="10",
            ),
        ),
    )
    variable = "GDP type of finance 110 type of flow 10"
    assert (
        recall_text(variable, (finance, flow)) == recall_text(variable, (flow, finance)) == variable
    )

    finance_then_flow = _layer2_key(variables=[variable], variable_qualifiers=[(finance, flow)])
    flow_then_finance = _layer2_key(variables=[variable], variable_qualifiers=[(flow, finance)])
    assert finance_then_flow == flow_then_finance


def test_role_contradiction_verdict_alone_changes_layer2_key() -> None:
    """Opposing raw-text role signals change the key even when extraction matches."""
    supports = _layer2_key(
        role_contradictions=[{"span": "Kenya", "axis": "destination", "contradicts": False}]
    )
    contradicts = _layer2_key(
        role_contradictions=[{"span": "Kenya", "axis": "destination", "contradicts": True}]
    )
    assert supports != contradicts

    # Matching verdicts also share a key.
    supports_again = _layer2_key(
        role_contradictions=[{"span": "Kenya", "axis": "destination", "contradicts": False}]
    )
    assert supports == supports_again


def test_query_text_qualifier_match_alone_changes_layer2_key() -> None:
    """The raw word "agriculture" adds a purpose-code match absent from the extracted phrase."""
    match = {
        "code_system": "crsPurpose",
        "code_dcid": "CRSPurposeCode311",
        "label": "agriculture",
        "source": "label",
        "matched_text": "agriculture",
    }
    with_match = _layer2_key(query_text_qualifier_matches=[match])
    without_match = _layer2_key(query_text_qualifier_matches=[])
    assert with_match != without_match


def test_query_text_qualifier_tie_alone_changes_layer2_key() -> None:
    """The raw phrase "budget support" ties two code systems and changes the key."""
    tie = {
        "matched_text": "budget support",
        "rows": [
            {
                "code_system": "crsCooperationModality",
                "code_dcid": "CRSCooperationModalityA",
                "label": "budget support",
                "source": "label",
                "matched_text": "budget support",
            },
            {
                "code_system": "crsPurpose",
                "code_dcid": "CRSPurposeCode510",
                "label": "budget support",
                "source": "label",
                "matched_text": "budget support",
            },
        ],
    }
    with_tie = _layer2_key(query_text_qualifier_ties=[tie])
    without_tie = _layer2_key(query_text_qualifier_ties=[])
    assert with_tie != without_tie


def test_query_text_qualifier_reading_alone_changes_layer2_key() -> None:
    """ "PPP" in the raw query is a two-reading span that `match_query_readings` reads from
    `query_text`; it feeds the availability prune and bind, so its presence must change
    the key."""
    match = {"source": "synonym", "matched_text": "PPP"}
    reading = {
        "matched_text": "PPP",
        "code": {
            **match,
            "code_system": "crsPurpose",
            "code_dcid": "CRSPurposeCode25010",
            "label": "business policy and administration",
        },
        "measure": {
            **match,
            "code_system": "measurementQualifier",
            "code_dcid": "PurchasingPowerParity",
            "label": "purchasing power parity",
        },
    }
    with_reading = _layer2_key(query_text_qualifier_readings=[reading])
    without_reading = _layer2_key(query_text_qualifier_readings=[])
    assert with_reading != without_reading


def _query_text_key(query_text: str) -> str:
    """The layer-2 key of one extraction under `query_text`'s raw-text signals."""
    signals = compute_layer2_signals(query=query_text, entity_roles={}, query_text=query_text)
    return _layer2_key(
        query_text_qualifier_matches=list(signals.query_text_qualifier_matches),
        query_text_qualifier_ties=list(signals.query_text_qualifier_ties),
        query_text_qualifier_readings=list(signals.query_text_qualifier_readings),
        query_text_qualifier_occurrences=list(signals.query_text_qualifier_occurrences),
        unsupported_qualifier_phrases=list(signals.unsupported_qualifier_phrases),
        declared_default_trigger_words=list(signals.declared_default_trigger_words),
        declared_default_slots_named=list(signals.declared_default_slots_named),
    )


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        # A DAC1 vocabulary reads "investment projects" (1310) and "debt buybacks" (1623) in
        # the first text of each pair only; read in every system together, the pairs agree.
        (
            "agricultural investment projects in Kenya",
            "agricultural investment plans in Kenya",
            False,
        ),
        ("senior debt buybacks for Kenya", "senior debt markets for Kenya", False),
        # A word no row names changes no reading, before the first label or after the last.
        (
            "agricultural investment projects in Kenya",
            "agricultural investment projects for Kenya",
            True,
        ),
        ("ODA to Kenya", "show me ODA to Kenya", True),
        ("health ODA to Kenya", "total health ODA to Kenya", True),
    ],
)
def test_query_text_some_vocabulary_reads_apart_changes_layer2_key(first, second, same) -> None:
    signals = [
        compute_layer2_signals(query=text, entity_roles={}, query_text=text)
        for text in (first, second)
    ]
    assert signals[0].query_text_qualifier_matches == signals[1].query_text_qualifier_matches
    assert (_query_text_key(first) == _query_text_key(second)) is same


def test_unsupported_qualifier_phrase_alone_changes_layer2_key() -> None:
    """ "non-concessional" in the raw query is a closed unsupported-vocabulary
    term (qualifier_match._UNSUPPORTED_QUALIFIER_TERMS) that
    `_resolve_query_qualifiers` reads from `query_text` and turns into a
    `no_vocabulary` disposition, refusing the whole answer via
    `qualifier_not_bound`. Same Extraction, the phrase present on one side
    and absent on the other, must be a different key."""
    with_phrase = _layer2_key(unsupported_qualifier_phrases=["non-concessional"])
    without_phrase = _layer2_key(unsupported_qualifier_phrases=[])
    assert with_phrase != without_phrase


def test_variables_content_alone_changes_layer2_key() -> None:
    """Extracted variable text already keys declared defaults through `variables`."""
    aid = _layer2_key(variables=["aid to Kenya"])
    loans = _layer2_key(variables=["loans to Kenya"])
    assert aid != loans


def test_raw_text_declared_default_trigger_alone_changes_layer2_key() -> None:
    """The raw word "aid" triggers the declared default even when extraction matches."""
    variable = "development assistance to Kenya"
    keys = []
    for raw in ("How much aid did Kenya get?", "How much development assistance did Kenya get?"):
        signals = compute_layer2_signals(query=raw, entity_roles={}, query_text=raw)
        keys.append(
            _layer2_key(
                variables=[variable],
                declared_default_trigger_words=list(signals.declared_default_trigger_words),
            )
        )
    assert keys[0] != keys[1]


def test_raw_text_naming_a_defaulted_slot_alone_changes_layer2_key() -> None:
    """ "flows" in the raw query decides whether the flow default warns, so equal
    extractions of "aid to Kenya" and "aid flows to Kenya" key apart."""
    keys = []
    for raw in ("aid to Kenya", "aid flows to Kenya"):
        signals = compute_layer2_signals(query=raw, entity_roles={}, query_text=raw)
        keys.append(
            _layer2_key(
                variables=["aid"],
                declared_default_trigger_words=list(signals.declared_default_trigger_words),
                declared_default_slots_named=list(signals.declared_default_slots_named),
            )
        )
    assert keys[0] != keys[1]


def test_dropped_qualifiers_content_alone_changes_layer2_key() -> None:
    """A dropped "across all flows" phrase changes the key because it suppresses a default."""
    with_breadth = _layer2_key(dropped_qualifiers=["across all flows"])
    without_breadth = _layer2_key(dropped_qualifiers=[])
    assert with_breadth != without_breadth


def test_operations_change_layer2_key() -> None:
    """Operations ride on the part and reach its bind, so two extractions differing only
    in them, or in their order, key apart."""
    rank = Operation(verb="rank", phrase="top 5 donors", over="origin")
    explain = Operation(verb="explain", phrase="why did fall")
    without = _layer2_key(variables=["education aid"])
    with_rank = _layer2_key(variables=["education aid"], operations=[(rank,)])
    with_both = _layer2_key(variables=["education aid"], operations=[(rank, explain)])
    reordered = _layer2_key(variables=["education aid"], operations=[(explain, rank)])
    assert len({without, with_rank, with_both, reordered}) == 4
    assert with_rank == _layer2_key(variables=["education aid"], operations=[(rank,)])


def test_a_separate_variable_changes_layer2_key() -> None:
    """A variable compared with the others does not fold, so marking one keys apart.
    Marking none keys as an extraction without the field."""
    variables = ["health ODA", "education ODA"]
    unmarked = _layer2_key(variables=variables)
    second = _layer2_key(variables=variables, separate=[False, True])
    assert len({unmarked, second, _layer2_key(variables=variables, separate=[True, False])}) == 3
    assert unmarked == _layer2_key(variables=variables, separate=[False, False])


def test_a_series_code_changes_layer2_key() -> None:
    """A code the query writes names the variable's StatVars, so a variable with one keys
    apart from the same variable without, and a code on another variable keys apart too.
    Writing none keys as an extraction without the field."""
    variables = ["debt service", "population"]
    plain = _layer2_key(variables=variables)
    first = _layer2_key(variables=variables, source_codes=["DT.TDS.DEGG.CD", None])
    second = _layer2_key(variables=variables, source_codes=[None, "DT.TDS.DEGG.CD"])
    assert len({plain, first, second}) == 3
    assert plain == _layer2_key(variables=variables, source_codes=[None, None])


def test_both_spellings_of_a_series_code_share_the_parsed_key() -> None:
    def key(code: str | None) -> str:
        part = ParsedPart(variable_text="debt service", source_code=code)
        return parsed_key(parsed=ParsedInput(parts=[part]), scope=_scope())

    assert key(" dt.tds.degg.cd") == key("DT.TDS.DEGG.CD") != key(None)
    assert key(" ") == key(None)


def test_a_separate_part_changes_the_parsed_key() -> None:
    """A part marked separate answers unfolded, so it keys apart from the same part
    unmarked, and an unmarked part keys as a part that omits the field."""

    def key(*, separate: bool) -> str:
        parts = [
            ParsedPart(variable_text="health ODA"),
            ParsedPart(variable_text="education ODA", separate=separate),
        ]
        return parsed_key(parsed=ParsedInput(parts=parts), scope=_scope())

    assert key(separate=True) != key(separate=False)
    assert key(separate=False) == parsed_key(
        parsed=ParsedInput.model_validate(
            {"parts": [{"variable_text": "health ODA"}, {"variable_text": "education ODA"}]}
        ),
        scope=_scope(),
    )


def test_source_mentions_change_layer2_key() -> None:
    """The source mentions each part compiles change its answer, so an extraction that
    excludes a source, includes it, names none, or attaches it to another variable keys
    apart."""
    crs = SourceMention(text="CRS")
    excluded = crs.model_copy(update={"polarity": "exclude"})
    two = ["aid to Kenya", "GDP of Kenya"]
    keys = {
        _layer2_key(variables=["aid to Kenya"]),
        _layer2_key(variables=["aid to Kenya"], sources=[(crs,)]),
        _layer2_key(variables=["aid to Kenya"], sources=[(excluded,)]),
        _layer2_key(variables=two, sources=[(crs,), ()]),
        _layer2_key(variables=two, sources=[(), (crs,)]),
        _layer2_key(variables=two, sources=[(crs,), (crs,)]),
    }
    assert len(keys) == 6


def test_an_entitys_members_flag_changes_the_parsed_key() -> None:
    """Two parsed requests alike but for one entity's `members` ask for different places."""

    def key(*, members: bool) -> str:
        entity = ParsedEntity(text="Africa", members=members)
        part = ParsedPart(variable_text="population", entities=[entity])
        return parsed_key(parsed=ParsedInput(parts=[part]), scope=_scope())

    assert key(members=True) != key(members=False)
    assert key(members=False) == parsed_key(
        parsed=ParsedInput.model_validate(
            {"parts": [{"variable_text": "population", "entities": [{"text": "Africa"}]}]}
        ),
        scope=_scope(),
    )


def test_include_observations_changes_key() -> None:
    """Requests with different observation options produce different keys."""
    off = layer1_key(query="gdp kenya", scope=_scope(options=None))
    on = layer1_key(
        query="gdp kenya",
        scope=_scope(options=ResolveOptions(include_observations=True)),
    )
    assert off != on


def test_include_skills_changes_key() -> None:
    off = layer1_key(query="gdp kenya", scope=_scope(options=None))
    on = layer1_key(
        query="gdp kenya",
        scope=_scope(options=ResolveOptions(include_skills=True)),
    )
    assert off != on


def test_different_max_candidates_changes_key() -> None:
    """The key folds in the server's candidate cap (`SEARCH_MAX_CANDIDATES`), which
    changes answers without moving the build commit."""
    lower = layer1_key(query="gdp kenya", scope=_scope(max_candidates=2))
    higher = layer1_key(query="gdp kenya", scope=_scope(max_candidates=settings().max_candidates))
    assert lower != higher


def test_different_seam_default_changes_key() -> None:
    """SEARCH_SEAM_DEFAULT gates directional binding on every request with no
    per-request override on the wire, so it changes the answer without
    moving build_commit. It must be part of the key."""
    on = layer1_key(query="gdp kenya", scope=_scope(seam_default=True))
    off = layer1_key(query="gdp kenya", scope=_scope(seam_default=False))
    assert on != off


def test_different_qualifier_coverage_changes_key() -> None:
    """SEARCH_QUALIFIER_COVERAGE decides whether an uncovered word refuses, with no
    per-request override, so it changes the answer without moving build_commit."""
    log = layer1_key(query="gdp kenya", scope=_scope(qualifier_coverage="log"))
    refuse = layer1_key(query="gdp kenya", scope=_scope(qualifier_coverage="refuse"))
    assert log != refuse


def test_different_today_changes_key() -> None:
    """The current day changes an open-ended window's result and belongs in the key."""
    day_one = layer1_key(query="gdp kenya", scope=_scope(today="2026-09-20"))
    day_two = layer1_key(query="gdp kenya", scope=_scope(today="2026-09-21"))
    assert day_one != day_two


def test_different_build_commit_changes_key() -> None:
    a = layer1_key(query="gdp kenya", scope=_scope(build_commit="commit-a"))
    b = layer1_key(query="gdp kenya", scope=_scope(build_commit="commit-b"))
    assert a != b


def test_different_model_changes_key() -> None:
    a = layer1_key(query="gdp kenya", scope=_scope(model="model-a"))
    b = layer1_key(query="gdp kenya", scope=_scope(model="model-b"))
    assert a != b


def test_different_seed_changes_key() -> None:
    a = layer1_key(query="gdp kenya", scope=_scope(seed=1))
    b = layer1_key(query="gdp kenya", scope=_scope(seed=2))
    assert a != b


def test_different_graph_generation_changes_key() -> None:
    """Known and unreadable generation stamps each produce distinct keys."""
    known = layer1_key(query="gdp kenya", scope=_scope(graph_generation="gen-a"))
    known_other = layer1_key(query="gdp kenya", scope=_scope(graph_generation="gen-b"))
    unknown = layer1_key(query="gdp kenya", scope=_scope(graph_generation="unknown"))
    assert known != known_other
    assert known != unknown
    assert known_other != unknown


def test_different_coverage_stamp_changes_key() -> None:
    """A new coverage stamp changes both keys for role-bearing answers."""
    a = layer1_key(query="gdp kenya", scope=_scope(coverage_stamp="stamp-a"))
    b = layer1_key(query="gdp kenya", scope=_scope(coverage_stamp="stamp-b"))
    assert a != b


@pytest.mark.parametrize(
    "changed",
    [{"origin_catalogue": "build-b"}, {"un_system_enabled": False}],
    ids=["origin_catalogue", "un_system_enabled"],
)
def test_a_different_origin_catalogue_changes_both_keys(changed: dict) -> None:
    """The origin catalogue decides which provenances an answer admits, and it reloads
    without a new graph generation, so its build and UN System setting key both layers."""
    assert layer1_key(query="gdp kenya", scope=_scope(**changed)) != layer1_key(
        query="gdp kenya", scope=_scope()
    )
    assert _layer2_key(scope=_scope(**changed)) != _layer2_key(scope=_scope())


def test_layer1_and_layer2_keys_never_collide_for_the_same_scope() -> None:
    scope = _scope()
    l1 = layer1_key(query="gdp kenya", scope=scope)
    l2 = _layer2_key(scope=scope)
    assert l1 != l2
    assert l1.startswith("l1_")
    assert l2.startswith("l2_")


def test_keys_are_stable_across_independently_built_equal_inputs() -> None:
    """Equal scope documents hash identically despite nested dictionary order."""
    scope_forward = scope_document(
        schema_version=1,
        options=ResolveOptions(include_skills=True),
        max_candidates=5,
        seam_default=True,
        qualifier_coverage="log",
        build_commit="commit-a",
        model="model-a",
        seed=1,
        today="2026-09-20",
        graph_generation="gen-a",
        coverage_stamp="coverage-a",
        conversion_catalogue="catalogue-a",
        origin_catalogue="build-a",
        un_system_enabled=True,
    )
    scope_reversed = scope_document(
        today="2026-09-20",
        seed=1,
        model="model-a",
        build_commit="commit-a",
        seam_default=True,
        qualifier_coverage="log",
        max_candidates=5,
        options=ResolveOptions(include_skills=True),
        schema_version=1,
        graph_generation="gen-a",
        coverage_stamp="coverage-a",
        conversion_catalogue="catalogue-a",
        origin_catalogue="build-a",
        un_system_enabled=True,
    )
    key_forward = layer1_key(query="gdp kenya", scope=scope_forward)
    key_reversed = layer1_key(query="gdp kenya", scope=scope_reversed)
    assert key_forward == key_reversed
