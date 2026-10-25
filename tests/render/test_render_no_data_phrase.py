"""Checks refusal wording across reasons and variable, entity, and qualifier terms."""

from __future__ import annotations

from typing import get_args

import pytest

from search.engine.source_code import SourceCodeRefusal
from search.engine.source_scope import SourceReference
from search.models import GraphRef, NoDataReason, ResolvedQualifier
from search.render.no_data import (
    _GENERIC_NO_DATA,
    _NO_DATA_PHRASES,
    no_data_phrase,
    no_data_sentence_plan,
)
from search.render.tokens import sentence_from_plan

# Refusal wording by reason.


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("no_observations", "No data found for this query."),
        (
            "source_not_resolved",
            "Could not match the data source named in this query. Try a publisher or "
            "dataset name, or leave the source out.",
        ),
        (
            "source_ambiguous",
            "A data source named in this query matches more than one source. Name one of them.",
        ),
        (
            "entity_not_resolved",
            "Could not use the place named in this query. Try a different place name.",
        ),
        (
            "entity_ambiguous",
            "An entity named in this query matches more than one entity in the graph. "
            "Be more specific.",
        ),
        (
            "entity_role_not_admitted",
            "An entity named in this query is not the kind of entity this query's data "
            "is reported for.",
        ),
        (
            "entity_set_too_large",
            "This query names more places, counting the members of any group or region it "
            "names, than this deployment reads in one request. Try naming fewer places or a "
            "narrower group.",
        ),
        (
            "variable_not_resolved",
            "Could not tell which indicator this query asks for. "
            "Try adding an indicator such as population or GDP.",
        ),
        (
            "role_not_resolved",
            "Could not tell who is giving and who is receiving in this query. "
            'Put "from" or "to" before a name.',
        ),
        (
            "no_matching_slice",
            "The combination this query asks for was never published as a series.",
        ),
        (
            "binding_mismatch",
            "No published series carries every condition this query asks for.",
        ),
        (
            "denominator_not_available",
            "No series for this query is in the form it asks for.",
        ),
        ("unit_not_resolved", "Could not read the requested unit in this query."),
        (
            "facet_not_available",
            "Data exists for this query, but none from the requested source, method, or unit.",
        ),
        (
            "role_incompatible",
            "The matched series is not published as a flow to or from any place named in this query.",
        ),
        (
            "role_signature_incompatible",
            "The matching series disagree about which place gives and which receives, "
            "so no single answer fits this query.",
        ),
        (
            "qualifier_not_bound",
            "A category or condition named in this query could not be applied to the "
            "matching series.",
        ),
    ],
)
def test_no_data_phrase_by_reason(reason, expected):
    assert no_data_phrase(reason) == expected


def test_no_data_phrase_is_distinct_for_every_reason():
    reasons = get_args(NoDataReason)
    assert len({no_data_phrase(reason) for reason in reasons}) == len(reasons)


def test_no_data_phrase_unknown_code_falls_back():
    phrase = no_data_phrase("some_future_unknown_code")  # ty: ignore[invalid-argument-type]  # intentionally out-of-type to exercise runtime fallback
    assert phrase == _GENERIC_NO_DATA
    assert phrase not in _NO_DATA_PHRASES.values()


# ---------------------------------------------------------------------------
# no_data_phrase term interpolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "variable_term", "expected"),
    [
        (
            "variable_not_resolved",
            "crop yield",
            'Could not match "crop yield" to a supported indicator. '
            "Try a different wording or a related indicator.",
        ),
        (
            "denominator_not_available",
            "school enrollment",
            'No series for "school enrollment" is in the form it asks for.',
        ),
        (
            "no_matching_slice",
            "net ODA received",
            'No published series for "net ODA received" matches the combination this '
            "query asks for.",
        ),
        (
            "binding_mismatch",
            "maternal mortality ratio",
            'No published series for "maternal mortality ratio" carries every condition '
            "this query asks for.",
        ),
        (
            "role_incompatible",
            "bilateral aid disbursements",
            'No data for "bilateral aid disbursements" as a flow to or from any place '
            "named in this query.",
        ),
    ],
)
def test_no_data_phrase_variable_term_interpolates(reason, variable_term, expected):
    assert no_data_phrase(reason, variable_term=variable_term) == expected


def test_no_data_phrase_whitespace_only_variable_term_falls_back_to_plain_phrase():
    assert no_data_phrase("variable_not_resolved", variable_term="   ") == no_data_phrase(
        "variable_not_resolved", variable_term=None
    )


def test_no_data_phrase_variable_term_strips_surrounding_whitespace():
    assert no_data_phrase("variable_not_resolved", variable_term=" gdp ") == (
        'Could not match "gdp" to a supported indicator. '
        "Try a different wording or a related indicator."
    )


@pytest.mark.parametrize(
    ("reason", "entity_terms", "expected"),
    [
        (
            "entity_not_resolved",
            ["Ruritania"],
            'Could not use "Ruritania" as a place for this query. Try a different place name.',
        ),
        (
            "entity_not_resolved",
            ["Ruritania", "Elbonia"],
            'Could not use "Ruritania" and "Elbonia" as places for this query. Try a different place name.',
        ),
        (
            "entity_ambiguous",
            ["World Bank"],
            '"World Bank" matches more than one entity in the graph. Be more specific.',
        ),
        (
            "entity_ambiguous",
            ["World Bank", "IFC"],
            '"World Bank" and "IFC" match more than one entity in the graph. Be more specific.',
        ),
        (
            "entity_role_not_admitted",
            ["African Development Bank"],
            '"African Development Bank" is not the kind of entity this query\'s data is '
            "reported for.",
        ),
        (
            "entity_role_not_admitted",
            ["African Development Bank", "Asian Infrastructure Investment Bank"],
            '"African Development Bank" and "Asian Infrastructure Investment Bank" are not '
            "the kind of entity this query's data is reported for.",
        ),
        # crs-010, ids-013: organisations hit it too, so the frame says who, not place.
        (
            "role_not_resolved",
            ["Gates Foundation", "Nigeria"],
            'Could not tell who is giving and who is receiving among "Gates Foundation" and '
            '"Nigeria". Put "from" or "to" before a name.',
        ),
        (
            "role_not_resolved",
            ["IMF"],
            'Could not tell whether "IMF" is giving or receiving. Put "from" or "to" before a '
            "name.",
        ),
    ],
)
def test_no_data_phrase_entity_term_interpolates(reason, entity_terms, expected):
    assert no_data_phrase(reason, entity_terms=entity_terms) == expected


@pytest.mark.parametrize(
    ("entity_terms", "ambiguous_candidates", "expected"),
    [
        (
            ["World Bank"],
            [
                "International Bank for Reconstruction and Development",
                "International Development Association",
            ],
            '"World Bank" matches more than one entity: "International Bank for '
            'Reconstruction and Development" and "International Development '
            'Association". Be more specific.',
        ),
    ],
)
def test_no_data_phrase_entity_ambiguous_lists_candidates(
    entity_terms, ambiguous_candidates, expected
):
    assert (
        no_data_phrase(
            "entity_ambiguous", entity_terms=entity_terms, ambiguous_candidates=ambiguous_candidates
        )
        == expected
    )


@pytest.mark.parametrize(
    ("reason", "source_terms", "candidates", "expected"),
    [
        (
            "source_not_resolved",
            ["XYZ statistics"],
            None,
            'Could not match "XYZ statistics" to a data source. Try a publisher or dataset '
            "name, or leave the source out.",
        ),
        (
            "source_not_resolved",
            ["XYZ", " ", "ABC"],
            None,
            'Could not match "XYZ" and "ABC" to data sources. Try a publisher or dataset '
            "name, or leave the source out.",
        ),
        (
            "source_ambiguous",
            ["WHO"],
            ["World Health Organization (WHO)", "World Health Organization (WHO)"],
            '"WHO" matches more than one source: "World Health Organization (WHO)" and '
            '"World Health Organization (WHO)". Name one of them.',
        ),
        (
            "source_ambiguous",
            ["WHO"],
            None,
            '"WHO" matches more than one source. Name one of them.',
        ),
    ],
)
def test_no_data_phrase_quotes_the_source_wordings(reason, source_terms, candidates, expected):
    assert (
        no_data_phrase(reason, source_terms=source_terms, ambiguous_candidates=candidates)
        == expected
    )


_IDS = SourceReference(
    ref="provenance/WorldBankInternationalDebtStatistics",
    label="World Bank International Debt Statistics",
    kind="provenance",
)


@pytest.mark.parametrize(
    ("refusal", "expected"),
    [
        (
            SourceCodeRefusal(
                reason="source_code_not_held", code="NGDP", cause="unknown", sources=(_IDS,)
            ),
            'No source this search reads holds the series code "NGDP". The search holds the '
            'series codes of "World Bank International Debt Statistics". Ask for the '
            "indicator in words.",
        ),
        (
            SourceCodeRefusal(
                reason="source_code_not_held",
                code="DT.TDS.DEGG.CD",
                cause="excluded",
                sources=(_IDS,),
            ),
            'The series code "DT.TDS.DEGG.CD" is a code of "World Bank International Debt '
            "Statistics\", which this query's sources leave out. Ask for the indicator in words.",
        ),
        (
            SourceCodeRefusal(
                reason="source_code_not_held",
                code="DT.TDS.DEGG.CD",
                cause="not_served",
                sources=(_IDS,),
            ),
            'The series code "DT.TDS.DEGG.CD" is a code of "World Bank International Debt '
            'Statistics", but this search serves none of its series. Ask for the indicator in '
            "words.",
        ),
        (
            SourceCodeRefusal(
                reason="source_code_not_held",
                code="DT.TDS.DEGG.CD",
                cause="unanswerable",
                sources=(_IDS,),
            ),
            'The series of the code "DT.TDS.DEGG.CD" of "World Bank International Debt '
            'Statistics" cannot answer this query as asked. Ask for the indicator in words.',
        ),
        (
            SourceCodeRefusal(
                reason="source_code_ambiguous",
                code="X.CODE",
                sources=(
                    SourceReference(ref="dc/base/A", label="Source A", kind="provenance"),
                    SourceReference(ref="dc/base/B", label="Source B", kind="provenance"),
                ),
            ),
            '"X.CODE" is a series code of more than one source: "Source A" and "Source B". '
            "Name one of them.",
        ),
        (
            SourceCodeRefusal(
                reason="source_code_shares_variable",
                code="DT.NFL.MIDA.CD",
                shared_codes=("DT.NFL.IMFC.CD",),
            ),
            'The series code "DT.NFL.MIDA.CD" shares its variable with "DT.NFL.IMFC.CD", so '
            "this search cannot serve it alone. Ask for the indicator in words.",
        ),
    ],
    ids=["unknown", "excluded", "not-served", "unanswerable", "ambiguous", "shares-variable"],
)
def test_a_source_code_phrase_states_its_refusal(refusal, expected):
    assert no_data_phrase(refusal.reason, source_code_refusal=refusal) == expected


def test_a_bare_source_refusal_plan_is_its_phrase():
    plan = no_data_sentence_plan("source_not_resolved", source_terms=["XYZ"])
    assert sentence_from_plan(plan) == no_data_phrase("source_not_resolved", source_terms=["XYZ"])


def test_no_data_phrase_entity_role_not_admitted_names_the_variable():
    assert (
        no_data_phrase(
            "entity_role_not_admitted",
            variable_term="population",
            entity_terms=["African Development Bank"],
        )
        == '"population" data is not reported for "African Development Bank".'
    )


def test_no_data_phrase_entity_term_containing_quote_renders_unescaped():
    assert no_data_phrase("entity_not_resolved", entity_terms=['At"lantis']) == (
        'Could not use "At"lantis" as a place for this query. Try a different place name.'
    )


def test_no_data_phrase_whitespace_only_entity_term_falls_back_to_plain_phrase():
    assert (
        no_data_phrase("entity_not_resolved", entity_terms=["   "])
        == "Could not use the place named in this query. Try a different place name."
    )


@pytest.mark.parametrize(
    ("reason", "terms", "expected"),
    [
        pytest.param(
            "unit_not_resolved",
            {"variable_term": "metric tons"},
            "Could not read the requested unit in this query.",
            id="variable-term-ignored",
        ),
        pytest.param(
            "no_observations",
            {"entity_terms": ["Ruritania"]},
            "No data found for this query.",
            id="entity-terms-ignored",
        ),
    ],
)
def test_no_data_phrase_ignores_terms_without_interpolated_form(reason, terms, expected):
    assert no_data_phrase(reason, **terms) == expected


# ---------------------------------------------------------------------------
# role_axis: a role-bearing query's own open flow axis, named directly for
# reason=="no_observations" instead of the generic phrase.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("role_axis", "expected"),
    [
        pytest.param("destination", "No recipient has data for this query.", id="recipient"),
        pytest.param("origin", "No provider has data for this query.", id="provider"),
    ],
)
def test_no_data_phrase_names_open_role_axis(role_axis, expected):
    assert no_data_phrase("no_observations", role_axis=role_axis) == expected


def test_no_data_phrase_role_axis_ignored_by_every_other_reason():
    assert (
        no_data_phrase("role_incompatible", role_axis="destination")
        == "The matched series is not published as a flow to or from any place named in this query."
    )


# ---------------------------------------------------------------------------
# no_observations: naming the concept that came up empty. variable_term
# names the query's own raw wording; elected_stat_vars, when supplied,
# names the graph-confirmed StatVar(s) instead when each has a graph name.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variable_term", "expected"),
    [("imputed multilateral ODA", 'No data for "imputed multilateral ODA".'), (None, None)],
)
def test_no_data_phrase_no_observations_never_names_a_stat_var_by_its_dcid(
    variable_term: str | None, expected: str | None
):
    """A StatVar whose label is its dcid has no graph name: the request's words stand in,
    else the term-free phrase."""
    dcid = "AttributedAmount_FinancialFlow_Gross_Disbursement_Nominal"
    phrase = no_data_phrase(
        "no_observations",
        variable_term=variable_term,
        elected_stat_vars=[GraphRef(ref=dcid, label=dcid)],
    )
    assert phrase == (expected or no_data_phrase("no_observations"))


def test_no_data_phrase_no_observations_names_variable_term():
    assert no_data_phrase("no_observations", variable_term="climate finance") == (
        'No data for "climate finance".'
    )


def test_no_data_phrase_no_observations_prefers_elected_stat_vars_over_variable_term():
    """Graph-confirmed StatVar labels take precedence over raw query wording."""
    assert (
        no_data_phrase(
            "no_observations",
            variable_term="climate finance",
            elected_stat_vars=[
                GraphRef(
                    ref="sdg/DC_FIN_CLIMT",
                    label="Total climate-specific financial support provided",
                )
            ],
        )
        == "No data for total climate-specific financial support provided."
    )


def test_no_data_phrase_no_observations_joins_multiple_elected_stat_vars():
    assert (
        no_data_phrase(
            "no_observations",
            elected_stat_vars=[
                GraphRef(ref="Count_Person_Male", label="Male population"),
                GraphRef(ref="Count_Person_Female", label="Female population"),
            ],
        )
        == "No data for male population and female population."
    )


def test_no_data_phrase_no_observations_bounds_a_large_elected_set():
    """A 10,000-member set stays bounded. Naming 3,808 variants once produced 547,111 characters."""
    small = no_data_phrase(
        "no_observations",
        elected_stat_vars=[GraphRef(ref="SV_0", label="flow variant 0")],
    )
    huge = no_data_phrase(
        "no_observations",
        elected_stat_vars=[
            GraphRef(ref=f"SV_{i}", label=f"flow variant {i}") for i in range(10_000)
        ],
    )
    # Length may grow for the count, not once per member.
    assert len(huge) < len(small) + 500
    assert huge.endswith("9,997 more.")
    assert huge.count("flow variant") == 3


def test_no_data_phrase_no_observations_prefers_variable_term_over_a_large_elected_set():
    """When 10,000 StatVars exceed the cap, use the query term (3,812 CRS variants in Q05)."""
    assert (
        no_data_phrase(
            "no_observations",
            variable_term="official development assistance disbursed",
            elected_stat_vars=[
                GraphRef(ref=f"SV_{i}", label=f"flow variant {i}") for i in range(10_000)
            ],
        )
        == 'No data for "official development assistance disbursed".'
    )


def test_no_data_phrase_no_observations_names_every_member_at_or_under_the_cap():
    """Name every member when the set fits within the cap."""
    assert (
        no_data_phrase(
            "no_observations",
            elected_stat_vars=[
                GraphRef(ref="Count_Person_Male", label="Male population"),
                GraphRef(ref="Count_Person_Female", label="Female population"),
                GraphRef(ref="Count_Person_Nonbinary", label="Nonbinary population"),
            ],
        )
        == "No data for male population, female population, and nonbinary population."
    )


def test_no_data_phrase_no_observations_role_axis_combines_with_concept():
    assert (
        no_data_phrase("no_observations", role_axis="destination", variable_term="climate finance")
        == 'No recipient has data for "climate finance".'
    )


def test_no_data_phrase_no_observations_no_concept_no_axis_falls_back_to_plain_phrase():
    assert no_data_phrase("no_observations", variable_term="   ") == "No data found for this query."


# ---------------------------------------------------------------------------
# no_data_phrase qualifier_terms interpolation (reason=="qualifier_not_bound")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "named", "disposition", "expected"),
    [
        pytest.param(
            "budget support",
            [GraphRef(ref="CRSCooperationModalityA", label="Budget support")],
            "unbound",
            '"budget support" names a category the search could not apply to this query.',
            id="unbound-form",
        ),
        pytest.param(
            "untied",
            [],
            "no_vocabulary",
            '"untied" is not a distinction this data records. Try the question without it.',
            id="no-vocabulary-form",
        ),
    ],
)
def test_no_data_phrase_qualifier_not_bound_single_term(text, named, disposition, expected):
    qualifier = ResolvedQualifier(text=text, named=named, slot=None, disposition=disposition)
    assert no_data_phrase("qualifier_not_bound", qualifier_terms=[qualifier]) == expected


def test_no_data_phrase_qualifier_not_bound_no_vocabulary_form_has_no_advice_on_unbound():
    """The advice sentence ("Try the question without it") is specific to
    no_vocabulary: there is nothing the reader can do about an unbound term,
    since the vocabulary does carry it and this binder just didn't apply
    it."""
    qualifier = ResolvedQualifier(text="debt relief", named=[], slot=None, disposition="unbound")
    phrase = no_data_phrase("qualifier_not_bound", qualifier_terms=[qualifier])
    assert phrase == '"debt relief" names a category the search could not apply to this query.'
    assert "Try" not in phrase


@pytest.mark.parametrize(
    ("texts", "disposition", "expected"),
    [
        pytest.param(
            ["budget support", "debt relief"],
            "unbound",
            '"budget support" and "debt relief" name categories the search could not apply '
            "to this query.",
            id="multiple-unbound-terms",
        ),
        pytest.param(
            ["untied", "non-concessional"],
            "no_vocabulary",
            '"untied" and "non-concessional" are not a distinction this data records. '
            "Try the question without them.",
            id="multiple-no-vocabulary-terms",
        ),
    ],
)
def test_no_data_phrase_qualifier_not_bound_multiple_terms(texts, disposition, expected):
    qualifiers = [
        ResolvedQualifier(text=text, named=[], slot=None, disposition=disposition) for text in texts
    ]
    assert no_data_phrase("qualifier_not_bound", qualifier_terms=qualifiers) == expected


def test_no_data_phrase_qualifier_not_bound_unbound_wins_over_mixed_dispositions():
    """An unbound term selects the unbound wording for a mixed set."""
    qualifiers = [
        ResolvedQualifier(text="budget support", named=[], slot=None, disposition="unbound"),
        ResolvedQualifier(text="untied", named=[], slot=None, disposition="no_vocabulary"),
    ]
    phrase = no_data_phrase("qualifier_not_bound", qualifier_terms=qualifiers)
    assert phrase == '"budget support" names a category the search could not apply to this query.'


def test_no_data_phrase_qualifier_not_bound_ignores_bound_entries():
    qualifiers = [
        ResolvedQualifier(
            text="ODA",
            named=[GraphRef(ref="CRSTypeOfFlow10", label="ODA")],
            slot="crsTypeOfFlow",
            disposition="bound",
        )
    ]
    assert (
        no_data_phrase("qualifier_not_bound", qualifier_terms=qualifiers)
        == (_NO_DATA_PHRASES["qualifier_not_bound"])
    )


def test_no_data_phrase_qualifier_terms_ignored_by_every_other_reason():
    qualifier = ResolvedQualifier(text="untied", named=[], slot=None, disposition="no_vocabulary")
    assert no_data_phrase("role_incompatible", qualifier_terms=[qualifier]) == (
        "The matched series is not published as a flow to or from any place named in this query."
    )


@pytest.mark.parametrize(
    ("entity_terms", "roles"),
    [
        (["World Bank"], ["origin"]),
        (["World Bank", "IFC"], ["origin", "destination"]),
        (["World Bank", "IFC", "Georgia"], ["origin", "destination", None]),
        (["Congo", "Congo"], ["origin", "destination"]),
    ],
    ids=["one", "two", "three", "one-wording-on-both-ends"],
)
def test_entity_ambiguous_plan_marks_each_wording_as_a_slot_token(entity_terms, roles):
    """Each ambiguous wording is its own slot token, carrying the role the request gave
    it, and the plan still joins to the phrase."""
    candidates = ["International Bank for Reconstruction and Development", "IDA"]
    plan = no_data_sentence_plan(
        "entity_ambiguous",
        entity_terms=entity_terms,
        ambiguous_candidates=candidates,
        ambiguous_roles=roles,
    )

    assert sentence_from_plan(plan) == no_data_phrase(
        "entity_ambiguous", entity_terms=entity_terms, ambiguous_candidates=candidates
    )
    slots = [t for t in plan if t.kind == "slot"]
    assert [t.text for t in slots] == entity_terms
    axis = {"origin": None, "destination": "where", None: "where"}
    assert [(t.role, t.axis) for t in slots] == [(role, axis[role]) for role in roles]
    for token in slots:
        assert token.entity_refs == []
        assert token.binding_ref is None
