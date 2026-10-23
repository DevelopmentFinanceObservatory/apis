"""Render no-data phrases and compose them with confirmed near-match sentences."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from materialise.roles import axis_noun
from search.models import (
    Departure,
    EntityRoleKey,
    GraphRef,
    NearestRealSpec,
    NoDataDenominator,
    NoDataGate,
    NoDataReason,
    ResolvedQualifier,
    RoleAxis,
    SentenceToken,
    Spec,
)

from .departures import departure_tokens
from .heading import Heading, heading_for, sibling_how_properties
from .labels import _join_labels
from .tokens import _SENTENCE_ENDS, heading_tokens
from .what import _lower_first_word

if TYPE_CHECKING:
    from search.engine.source_code import SourceCodeRefusal

# The next step a role_not_resolved refusal gives.
_NAME_A_DIRECTION = 'Put "from" or "to" before a name.'

# The next step a source_not_resolved refusal gives.
_NAME_A_SOURCE = "Try a publisher or dataset name, or leave the source out."

# The next step a source_code_not_held or source_code_shares_variable refusal gives.
_ASK_IN_WORDS = "Ask for the indicator in words."

# The source-code refusals, which `_source_code_phrase` words.
_SOURCE_CODE_REASONS = frozenset(
    {"source_code_not_held", "source_code_ambiguous", "source_code_shares_variable"}
)


@dataclass(frozen=True, slots=True)
class ScopeSources:
    """The sources a part's no-data outcome was read under, by label: those it names
    (`include`), else those it leaves out (`exclude`)."""

    polarity: Literal["include", "exclude"]
    labels: tuple[str, ...]


# Term-free wording by NoDataReason. Unknown codes use _GENERIC_NO_DATA.
# no_data_phrase adds query terms for supported reasons.
_NO_DATA_PHRASES: dict[str, str] = {
    "no_observations": "No data found for this query.",
    "source_not_resolved": (
        f"Could not match the data source named in this query. {_NAME_A_SOURCE}"
    ),
    "source_ambiguous": (
        "A data source named in this query matches more than one source. Name one of them."
    ),
    "source_code_not_held": (
        f"No source this search reads holds the series code in this query. {_ASK_IN_WORDS}"
    ),
    "source_code_ambiguous": (
        "The series code in this query is held by more than one source. Name one of them."
    ),
    "source_code_shares_variable": (
        "The series code in this query shares its variable with other codes of its source, "
        f"so this search cannot serve it alone. {_ASK_IN_WORDS}"
    ),
    "entity_not_resolved": "Could not use the place named in this query. Try a different place name.",
    "entity_ambiguous": (
        "An entity named in this query matches more than one entity in the graph. Be more specific."
    ),
    "entity_role_not_admitted": (
        "An entity named in this query is not the kind of entity this query's data is reported for."
    ),
    "entity_set_too_large": (
        "This query names more places, counting the members of any group or region it names, "
        "than this deployment reads in one request. Try naming fewer places or a narrower group."
    ),
    "variable_not_resolved": (
        "Could not tell which indicator this query asks for. "
        "Try adding an indicator such as population or GDP."
    ),
    "role_not_resolved": (
        f"Could not tell who is giving and who is receiving in this query. {_NAME_A_DIRECTION}"
    ),
    "no_matching_slice": "The combination this query asks for was never published as a series.",
    "binding_mismatch": "No published series carries every condition this query asks for.",
    "denominator_not_available": "No series for this query is in the form it asks for.",
    "unit_not_resolved": "Could not read the requested unit in this query.",
    "facet_not_available": (
        "Data exists for this query, but none from the requested source, method, or unit."
    ),
    "role_incompatible": (
        "The matched series is not published as a flow to or from any place named in this query."
    ),
    "role_signature_incompatible": (
        "The matching series disagree about which place gives and which receives, so no "
        "single answer fits this query."
    ),
    "qualifier_not_bound": (
        "A category or condition named in this query could not be applied to the matching series."
    ),
}
_GENERIC_NO_DATA = "No data is available for this query."


# Variable-term templates for the five supported reasons. no_data_phrase
# formats {t}. Other reasons use their term-free phrase.
_VARIABLE_TERM_PHRASES: dict[str, str] = {
    "variable_not_resolved": (
        'Could not match "{t}" to a supported indicator. Try a different wording or a related '
        "indicator."
    ),
    "denominator_not_available": 'No series for "{t}" is in the form it asks for.',
    "no_matching_slice": (
        'No published series for "{t}" matches the combination this query asks for.'
    ),
    "binding_mismatch": (
        'No published series for "{t}" carries every condition this query asks for.'
    ),
    "role_incompatible": ('No data for "{t}" as a flow to or from any place named in this query.'),
}


# Prefix/suffix forms for the five variable-term reasons. The envelope slot
# already names the variable, so these forms avoid repeating it.
_COMPOSED_VARIABLE_TERM_PHRASES: dict[str, tuple[str, str]] = {
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


def composed_variable_outcome(reason: NoDataReason) -> tuple[str, str] | None:
    """Return the slot-framing phrase pair for a supported reason, or None."""
    return _COMPOSED_VARIABLE_TERM_PHRASES.get(reason)


# Enumerate at most three elected labels. Larger sets prefer the query term.
_MAX_NAMED_CONCEPTS = 3


def _no_observations_concept(
    elected_stat_vars: list[GraphRef] | None, *, variable_term: str | None
) -> str | None:
    """Return what a no_observations outcome names: the elected StatVars' graph names, or
    the request's own words in quotes, or None.

    Up to three elected StatVars are named in full, their first words lowered. Above
    that cap, the request's words come first, then the first three names and the
    omitted count. A StatVar with no graph name (its label is its dcid) is left unnamed,
    since a name built from its dcid is no one's wording. The request's words stand in.
    """
    quoted = f'"{variable_term}"' if variable_term else None
    if not elected_stat_vars or any(
        not ref.label or ref.label == ref.ref for ref in elected_stat_vars
    ):
        return quoted
    labels = [_lower_first_word(ref.label) for ref in elected_stat_vars]
    if len(labels) <= _MAX_NAMED_CONCEPTS:
        return _join_labels(labels)
    if quoted:
        return quoted
    remaining = len(labels) - _MAX_NAMED_CONCEPTS
    return _join_labels([*labels[:_MAX_NAMED_CONCEPTS], f"{remaining:,} more"])


def no_data_phrase(
    reason: NoDataReason,
    *,
    variable_term: str | None = None,
    entity_terms: list[str] | None = None,
    entity_groups: Sequence[Sequence[str]] | None = None,
    role_axis: EntityRoleKey | None = None,
    ambiguous_candidates: list[str] | None = None,
    elected_stat_vars: list[GraphRef] | None = None,
    qualifier_terms: list[ResolvedQualifier] | None = None,
    source_terms: list[str] | None = None,
    scope: ScopeSources | None = None,
    source_code_refusal: SourceCodeRefusal | None = None,
) -> str:
    """Map a reason to its phrase, using query terms where supported.

    Unknown reasons use the generic phrase.

    ``variable_term`` and ``entity_terms`` contain extracted text, which may
    be normalised or translated. Strip terms and ignore blank entity entries.
    Variable terms are quoted for variable_not_resolved,
    denominator_not_available, no_matching_slice, binding_mismatch, and
    role_incompatible. Entity terms are quoted and joined for
    entity_not_resolved, entity_ambiguous, entity_role_not_admitted, and
    role_not_resolved, which asks who among them is giving and who receiving.
    ``entity_groups`` holds role_not_resolved's terms grouped by the place each names, and
    the phrase counts and names places from it, one place named by several texts as one
    (`_untold_places`); without it each term is a place.
    ``ambiguous_candidates`` adds graph matches to
    entity_ambiguous. For entity_role_not_admitted, ``variable_term`` names
    the indicator whose data is unavailable for the entity.

    ``source_terms`` are quoted and joined for source_not_resolved and
    source_ambiguous, where ``ambiguous_candidates`` holds the labels of the
    sources the wording matches. The source-code reasons are worded from
    ``source_code_refusal`` (`_source_code_phrase`).

    ``elected_stat_vars`` is used only for no_observations
    (`_no_observations_concept`). Up to three graph names take precedence, unquoted.
    Otherwise ``variable_term`` is quoted.

    For no_observations, ``role_axis`` names an open origin or destination
    axis. It combines with the concept term when both exist. Either alone is
    used by itself, and neither returns the term-free phrase.

    For qualifier_not_bound, any ``unbound`` terms take precedence and
    identify categories the search could not apply. Otherwise, ``no_vocabulary``
    terms name distinctions the data does not record and advise omitting them.
    With neither, a part read under a source ``scope`` says no series comes from the
    sources it names, or from any source but those it leaves out (`_scope_verdict`).
    Empty terms and unsupported reasons use the term-free phrase. Embedded
    double quotes pass through unchanged.
    """
    if variable_term is not None:
        variable_term = variable_term.strip() or None
    if reason in ("source_not_resolved", "source_ambiguous"):
        return _source_phrase(reason, source_terms, ambiguous_candidates)
    if reason in _SOURCE_CODE_REASONS:
        return _source_code_phrase(reason, source_code_refusal)
    if reason == "no_observations":
        concept = _no_observations_concept(elected_stat_vars, variable_term=variable_term)
        axis = axis_noun(role_axis) if role_axis is not None else None
        if axis and concept:
            return f"No {axis} has data for {concept}."
        if axis:
            return f"No {axis} has data for this query."
        if concept:
            return f"No data for {concept}."
        return _NO_DATA_PHRASES["no_observations"]
    if reason == "qualifier_not_bound" and qualifier_terms:
        unbound = [q.text for q in qualifier_terms if q.disposition == "unbound"]
        if unbound:
            joined = _join_labels([f'"{t}"' for t in unbound])
            verb = "name" if len(unbound) != 1 else "names"
            noun = "categories" if len(unbound) != 1 else "a category"
            return f"{joined} {verb} {noun} the search could not apply to this query."
        no_vocabulary = [q.text for q in qualifier_terms if q.disposition == "no_vocabulary"]
        if no_vocabulary:
            joined = _join_labels([f'"{t}"' for t in no_vocabulary])
            verb = "are" if len(no_vocabulary) != 1 else "is"
            pronoun = "them" if len(no_vocabulary) != 1 else "it"
            return (
                f"{joined} {verb} not a distinction this data records. "
                f"Try the question without {pronoun}."
            )
    if reason == "qualifier_not_bound" and scope is not None:
        return _scope_verdict(scope)
    if variable_term and reason in _VARIABLE_TERM_PHRASES:
        return _VARIABLE_TERM_PHRASES[reason].format(t=variable_term)
    if entity_terms:
        terms = [t.strip() for t in entity_terms if t and t.strip()]
        if terms:
            plural = len(terms) != 1
            joined = _join_labels([f'"{t}"' for t in terms])
            if reason == "entity_not_resolved":
                noun = "places" if plural else "a place"
                return (
                    f"Could not use {joined} as {noun} for this query. Try a different place name."
                )
            if reason == "entity_ambiguous":
                verb = "match" if plural else "matches"
                candidates = [c.strip() for c in ambiguous_candidates or () if c and c.strip()]
                if candidates:
                    candidate_list = _join_labels([f'"{c}"' for c in candidates])
                    return (
                        f"{joined} {verb} more than one entity: {candidate_list}. Be more specific."
                    )
                return f"{joined} {verb} more than one entity in the graph. Be more specific."
            if reason == "role_not_resolved":
                places = _untold_places(entity_groups or [[t] for t in terms])
                place_list = _join_labels(places)
                if len(places) != 1:
                    return (
                        f"Could not tell who is giving and who is receiving among {place_list}. "
                        f"{_NAME_A_DIRECTION}"
                    )
                return (
                    f"Could not tell whether {place_list} is giving or receiving. "
                    f"{_NAME_A_DIRECTION}"
                )
            if reason == "entity_role_not_admitted":
                if variable_term:
                    return f'"{variable_term}" data is not reported for {joined}.'
                verb = "are" if plural else "is"
                return f"{joined} {verb} not the kind of entity this query's data is reported for."
    return _NO_DATA_PHRASES.get(reason, _GENERIC_NO_DATA)


def _untold_places(groups: Sequence[Sequence[str]]) -> list[str]:
    """Each untold place as a role_not_resolved phrase names it, its first text quoted
    (`untold_place_name`). A group left with no text after stripping names no place."""
    return [name for group in groups if (name := untold_place_name(group, quote_first=True))]


def untold_place_name(texts: Sequence[str], *, quote_first: bool) -> str | None:
    """One place a `role_not_resolved` refusal names, from the request texts that name it in
    query order: its first text, then its further texts, each once, quoted in parentheses.
    The message quotes the first text as well, `"the union" ("European Union")`, and an
    option label leaves it bare, `the union ("European Union")`. ``None`` when no text is
    left after stripping."""
    stripped = list(dict.fromkeys(t.strip() for t in texts if t and t.strip()))
    if not stripped:
        return None
    first = f'"{stripped[0]}"' if quote_first else stripped[0]
    others = ", ".join(f'"{t}"' for t in stripped[1:])
    return f"{first} ({others})" if others else first


def _source_phrase(
    reason: NoDataReason, source_terms: list[str] | None, candidates: list[str] | None
) -> str:
    """The source_not_resolved or source_ambiguous phrase, quoting the wordings and,
    for an ambiguous one, the sources they match."""
    terms = [t.strip() for t in source_terms or () if t and t.strip()]
    if not terms:
        return _NO_DATA_PHRASES[reason]
    plural = len(terms) != 1
    joined = _join_labels([f'"{t}"' for t in terms])
    if reason == "source_not_resolved":
        noun = "data sources" if plural else "a data source"
        return f"Could not match {joined} to {noun}. {_NAME_A_SOURCE}"
    verb = "match" if plural else "matches"
    labels = [c.strip() for c in candidates or () if c and c.strip()]
    if labels:
        candidate_list = _join_labels([f'"{c}"' for c in labels])
        return f"{joined} {verb} more than one source: {candidate_list}. Name one of them."
    return f"{joined} {verb} more than one source. Name one of them."


def _source_code_phrase(reason: NoDataReason, refusal: SourceCodeRefusal | None) -> str:
    """The phrase of a source-code refusal, quoting the code and the sources or codes
    `refusal` names. A `source_code_not_held` phrase states its cause: no source this search
    reads holds the code (`unknown`), the query's sources leave out its holders
    (`excluded`), the search serves none of its series (`not_served`), or its series cannot
    answer the query as asked (`unanswerable`)."""
    if refusal is None:
        return _NO_DATA_PHRASES[reason]
    code = f'"{refusal.code}"'
    sources = _join_labels([f'"{source.label}"' for source in refusal.sources])
    if reason == "source_code_ambiguous":
        among = f": {sources}" if sources else ""
        return f"{code} is a series code of more than one source{among}. Name one of them."
    if reason == "source_code_shares_variable":
        shared = _join_labels([f'"{c}"' for c in refusal.shared_codes])
        return (
            f"The series code {code} shares its variable with "
            f"{shared or 'other codes of its source'}, so this search cannot serve it alone. "
            f"{_ASK_IN_WORDS}"
        )
    of = f" of {sources}" if sources else ""
    match refusal.cause:
        case "excluded":
            verdict = f"The series code {code} is a code{of}, which this query's sources leave out."
        case "not_served":
            verdict = (
                f"The series code {code} is a code{of}, but this search serves none of its series."
            )
        case "unanswerable":
            verdict = f"The series of the code {code}{of} cannot answer this query as asked."
        case _:
            held = f" The search holds the series codes{of}." if sources else ""
            verdict = f"No source this search reads holds the series code {code}.{held}"
    return f"{verdict} {_ASK_IN_WORDS}"


def _scope_verdict(scope: ScopeSources) -> str:
    """The qualifier_not_bound verdict of a part whose scope leaves no series for it."""
    joined = _join_labels(list(scope.labels))
    if scope.polarity == "include":
        return f"No series for this query comes from {joined}."
    return f"No series for this query comes from a source other than {joined}."


def scope_sentence(
    reason: NoDataReason,
    scope: ScopeSources | None,
    qualifier_terms: list[ResolvedQualifier] | None = None,
) -> str | None:
    """The sentence a no-data outcome read under `scope` closes on, naming the sources that
    limited it: "The search read only ..." or "The search left out ...". `None` without a
    scope, and for a qualifier_not_bound outcome naming no term, whose verdict already names
    the sources (`_scope_verdict`)."""
    if scope is None:
        return None
    if reason == "qualifier_not_bound" and not any(
        q.disposition in ("unbound", "no_vocabulary") for q in qualifier_terms or ()
    ):
        return None
    joined = _join_labels(list(scope.labels))
    if scope.polarity == "include":
        return f"The search read only {joined}."
    return f"The search left out {joined}."


# Reasons whose no_data_phrase already closes on an imperative ("Try ...").
# _compose_nearest_series drops its own trailing "Check that it measures..."
# clause for these: an envelope carries one call to action, not two.
_REASONS_WITH_TRAILING_IMPERATIVE = frozenset(
    {
        "source_not_resolved",
        "source_ambiguous",
        "source_code_not_held",
        "source_code_ambiguous",
        "source_code_shares_variable",
        "entity_not_resolved",
        "entity_ambiguous",
        "entity_set_too_large",
        "variable_not_resolved",
        "role_not_resolved",
    }
)


def _text(text: str) -> SentenceToken:
    return SentenceToken(kind="text", text=text)


# Guidance added to a bare no_observations phrase, the only reason without
# its own "Try ..." instruction.
_NO_OBSERVATIONS_BARE_GUIDANCE = "Try a different place, time period, or indicator."


def _bare_no_data_text(phrase: str, reason: NoDataReason) -> str:
    """Return the bare no-data phrase, adding guidance only for no_observations."""
    if reason == "no_observations":
        return f"{phrase} {_NO_OBSERVATIONS_BARE_GUIDANCE}"
    return phrase


def _bare_no_data_plan(phrase: str, reason: NoDataReason) -> list[SentenceToken]:
    return [_text(_bare_no_data_text(phrase, reason))]


def no_data_message(
    reason: NoDataReason,
    *,
    variable_term: str | None = None,
    entity_terms: list[str] | None = None,
    entity_groups: Sequence[Sequence[str]] | None = None,
    role_axis: EntityRoleKey | None = None,
    ambiguous_candidates: list[str] | None = None,
    elected_stat_vars: list[GraphRef] | None = None,
    qualifier_terms: list[ResolvedQualifier] | None = None,
    source_terms: list[str] | None = None,
    scope: ScopeSources | None = None,
    denominator: NoDataDenominator | None = None,
    source_code_refusal: SourceCodeRefusal | None = None,
) -> str:
    """The bare no-data sentence: `no_data_phrase` with every term given, plus the guidance
    a bare no_observations sentence adds, the denominators a denominator_not_available
    refusal compared (`denominator_sentence`), and the sources a scoped part was read
    under (`scope_sentence`), as `no_data_sentence_plan` renders it when it names no near
    match."""
    phrase = no_data_phrase(
        reason,
        variable_term=variable_term,
        entity_terms=entity_terms,
        entity_groups=entity_groups,
        role_axis=role_axis,
        ambiguous_candidates=ambiguous_candidates,
        elected_stat_vars=elected_stat_vars,
        qualifier_terms=qualifier_terms,
        source_terms=source_terms,
        scope=scope,
        source_code_refusal=source_code_refusal,
    )
    text = _bare_no_data_text(phrase, reason)
    if denominator is not None:
        text += denominator_sentence(denominator)
    closing = scope_sentence(reason, scope, qualifier_terms)
    return f"{text} {closing}" if closing else text


def denominator_sentence(denominator: NoDataDenominator) -> str:
    """The sentence, with its leading space, naming the denominator a question named and
    the one the closest series divides by: " The question names GDP as the base, and the
    closest series divides by gross national income." Empty when it names neither."""
    asked = _lower_first_word(denominator.asked.label) if denominator.asked else None
    found = _lower_first_word(denominator.found.label) if denominator.found else None
    if asked and found:
        return (
            f" The question names {asked} as the base, and the closest series divides by {found}."
        )
    if asked:
        return f" The question names {asked} as the base, and the closest series states none."
    if found:
        return f" The closest series divides by {found}."
    return ""


def _denominator_plan(
    phrase: str,
    flagged_match: Spec,
    nearest_real: list[NearestRealSpec] | None,
    denominator: NoDataDenominator | None,
) -> list[SentenceToken]:
    """A denominator_not_available refusal's plan: the phrase, the denominators compared
    (`denominator_sentence`), the flagged match offered as the closest series, and the
    first nearest_real level as data that exists. ``phrase`` stays the first token, which
    the envelope reframes."""
    plan = [_text(phrase)]
    if denominator is not None and (said := denominator_sentence(denominator)):
        plan.append(_text(said))
    if lead := _near_match_lead_plan(_near_match_heading(flagged_match)):
        plan.extend([_text(" The closest series is "), *lead])
    if nearest_real and (
        level := _near_match_lead_plan(
            _near_match_heading(
                nearest_real[0], keep_how_properties=sibling_how_properties(nearest_real)
            )
        )
    ):
        plan.extend([_text(" Data exists for "), *level])
    return plan


def _join_separators(count: int) -> list[str]:
    """The separators `_join_labels` puts between `count` labels, in order."""
    if count < 2:
        return []
    if count == 2:
        return [" and "]
    return [", "] * (count - 2) + [", and "]


def _ambiguous_entity_plan(
    entity_terms: list[str] | None,
    *,
    ambiguous_candidates: list[str] | None,
    ambiguous_roles: Mapping[str, RoleAxis] | None,
) -> list[SentenceToken] | None:
    """Return the bare entity_ambiguous phrase with each ambiguous wording as a slot token.

    The tokens join to `no_data_phrase`'s text. Each wording's token carries the flow
    axis the door gave it as `role`, and the place axis unless that role is origin, as
    an entity read-back does; it has no `entity_refs`, since the wording names no node.
    Return `None` when no wording is left after stripping.
    """
    terms = [t.strip() for t in entity_terms or () if t and t.strip()]
    if not terms:
        return None
    phrase = no_data_phrase(
        "entity_ambiguous", entity_terms=terms, ambiguous_candidates=ambiguous_candidates
    )
    roles = ambiguous_roles or {}
    tokens = [_text('"')]
    for term, separator in zip(terms, [*_join_separators(len(terms)), None], strict=True):
        role = roles.get(term)
        tokens.append(
            SentenceToken(
                kind="slot", text=term, axis=None if role == "origin" else "where", role=role
            )
        )
        if separator is not None:
            tokens.append(_text(f'"{separator}"'))
    joined = _join_labels([f'"{t}"' for t in terms])
    tokens.append(_text(f'"{phrase.removeprefix(joined)}'))
    return tokens


def _compose_nearest_series(
    phrase: str, reason: NoDataReason, lead: list[SentenceToken]
) -> list[SentenceToken]:
    """Name a series that holds data after the phrase ("Data exists for Kenya's GDP,
    1960 to 2024."), or return the bare phrase when ``lead`` is empty.

    The series is the search's nearest by ranking, not by meaning, so add a caution
    unless the phrase already gives the reader a "Try ..." instruction.
    """
    if not lead:
        return _bare_no_data_plan(phrase, reason)
    caution = (
        ""
        if reason in _REASONS_WITH_TRAILING_IMPERATIVE
        else " Check that it measures what you asked for."
    )
    plan = [_text(phrase), _text(" Data exists for "), *lead]
    if caution:
        plan.append(_text(caution))
    return plan


def _compose_matching_series(
    phrase: str, reason: NoDataReason, lead: list[SentenceToken]
) -> list[SentenceToken]:
    """Compose a flagged match for the elected variable as a matching series, not a substitute."""
    if not lead:
        return _bare_no_data_plan(phrase, reason)
    caution = (
        ""
        if reason in _REASONS_WITH_TRAILING_IMPERATIVE
        else " Check that it measures what you asked for."
    )
    plan = [_text(phrase), _text(" The matching series is "), *lead]
    if caution:
        plan.append(_text(caution))
    return plan


def _checked_match_plan(
    phrase: str,
    reason: NoDataReason,
    flagged_match: Spec,
    elected_stat_vars: list[GraphRef] | None,
    *,
    checked: bool,
) -> list[SentenceToken]:
    """The plan for a flagged match. A ``checked`` match, the series the gate read, is
    stated in claim form (`_claim_plan`) unless the reason quotes the variable
    (`composed_variable_outcome`). The envelope swaps such a plan's first token for its
    own framing, so the plan keeps ``phrase`` first and names the match after it, as a
    matching series when its StatVars are the elected ones and a nearest one otherwise.
    """
    heading = _near_match_heading(flagged_match)
    if checked and composed_variable_outcome(reason) is None:
        claim = _claim_plan(heading)
        if claim:
            return claim
    lead = _near_match_lead_plan(heading)
    if _is_elected_variable(flagged_match, elected_stat_vars):
        return _compose_matching_series(phrase, reason, lead)
    return _compose_nearest_series(phrase, reason, lead)


def _near_match_heading(
    spec: Spec, *, keep_how_properties: frozenset[str] = frozenset()
) -> Heading:
    """The heading a near-match states (`heading_for`), as its own sentence plan does."""
    return heading_for(spec, keep_how_properties=keep_how_properties).heading


def _claim_plan(heading: Heading) -> list[SentenceToken]:
    """The checked series' heading in claim form, "No data for {heading}.", as text
    tokens, or [] when the heading states nothing. The heading names the entity,
    measure, counterpart, and span that were checked. Its byline is left out."""
    claim = _text_tokens(heading_tokens(heading, claim=True))
    if not claim:
        return []
    return [_text("No data for "), *claim, _text(".")]


def _is_elected_variable(flagged_match: Spec, elected_stat_vars: list[GraphRef] | None) -> bool:
    """Whether a flagged match and the elected request name identical StatVars."""
    if not elected_stat_vars:
        return False
    flagged_stat_vars = {stat_var.ref.ref for stat_var in flagged_match.resolved.members}
    elected = {stat_var.ref for stat_var in elected_stat_vars}
    return bool(flagged_stat_vars) and flagged_stat_vars == elected


def alternate_preserves_bound_role(
    alternate: Spec, bound_role_dcids: Mapping[EntityRoleKey, Sequence[str]] | None
) -> bool:
    """Return whether an alternate's directional entities match the query's bound roles.

    Subject-only alternates are safe. Every origin or destination entity in
    a directional alternate must match a dcid bound on the same axis. Otherwise the
    alternate could reverse donor and recipient roles.
    """
    directional = [
        (e.role, e.ref) for e in alternate.resolved.entities if e.role != "observationAbout"
    ]
    if not directional:
        return True
    if not bound_role_dcids:
        return False
    return all(dcid in bound_role_dcids.get(role, ()) for role, dcid in directional)


def no_data_sentence_plan(
    reason: NoDataReason,
    *,
    gate: NoDataGate | None = None,
    flagged_match: Spec | None = None,
    alternate_match: Spec | None = None,
    nearest_real: list[NearestRealSpec] | None = None,
    elected_stat_vars: list[GraphRef] | None = None,
    variable_term: str | None = None,
    entity_terms: list[str] | None = None,
    entity_groups: Sequence[Sequence[str]] | None = None,
    role_axis: EntityRoleKey | None = None,
    bound_role_dcids: Mapping[EntityRoleKey, Sequence[str]] | None = None,
    ambiguous_candidates: list[str] | None = None,
    ambiguous_roles: Mapping[str, RoleAxis] | None = None,
    qualifier_terms: list[ResolvedQualifier] | None = None,
    source_terms: list[str] | None = None,
    scope: ScopeSources | None = None,
    denominator: NoDataDenominator | None = None,
    departures: Sequence[Departure] = (),
    source_code_refusal: SourceCodeRefusal | None = None,
) -> list[SentenceToken]:
    """Build a no-data plan: a verdict, a near-match's sentence when one is available,
    then the part's departures.

    ``gate`` identifies the failed quality gate. A near-match is named by its heading
    (`heading_for`) in claim form, as plain text, and its byline is left out.

    Apply these branches in order:
    - A place-availability gate with ``flagged_match`` states, in claim form, the
      heading of the series that was checked: "No data for Gates Foundation's gross ODA
      disbursements to Nigeria." (`_claim_plan`), naming the entity, measure, and
      counterpart.
    - ``only_alternate_has_data`` shows ``alternate_match`` only when its
      directional roles match the query (`alternate_preserves_bound_role`). Otherwise
      return the bare phrase.
    - ``role_incompatible`` or ``qualifier_not_bound`` with a flagged match
      returns the bare refusal to avoid contradicting it with the match.
    - ``denominator_not_available`` with a flagged match names the denominator asked and
      found (``denominator``), offers the flagged match as the closest series and the
      first ``nearest_real`` entry, its level reading, as data that exists
      (`_denominator_plan`).
    - Another flagged match whose StatVars equal the elected set is the checked
      series too and takes the same claim form, unless the reason quotes the
      variable (`composed_variable_outcome`), whose phrase the envelope reframes.
      Other flagged matches and a ``nearest_real`` entry use nearest-series wording.
    - With no match, return the bare phrase. The bare entity_ambiguous phrase
      marks each ambiguous wording as a slot token, so a client can hang the
      candidate menu on it.

    Every composed form starts with the no-data phrase. A match whose heading states
    nothing falls back to the bare phrase.

    The first token is the complete no-data phrase, except in the bare
    entity_ambiguous form, which splits the phrase around its slot tokens. A
    near-match lead-in, its plain-text sentence tokens, and its place-gap or
    nearest-series tail follow as later tokens. This lets the envelope
    composer replace a term-quoting phrase without parsing rendered prose.

    ``entity_terms`` are omitted for the place-availability gate to avoid
    naming the place twice. ``bound_role_dcids`` checks alternate roles.
    ``ambiguous_candidates`` and ``ambiguous_roles`` apply only to the bare
    entity-ambiguity form; ``ambiguous_roles`` maps a wording to its flow axis.
    ``elected_stat_vars`` are added when the verdict would not repeat a
    variable already named by the match. ``qualifier_terms`` apply only to
    qualifier_not_bound, ``source_terms`` to the two source reasons, whose
    ``ambiguous_candidates`` are source labels, and ``source_code_refusal`` to the
    source-code reasons (`no_data_phrase`). A part read under a source ``scope``
    closes its verdict on the sources it was read under (`scope_sentence`). The part's
    ``departures`` follow as departure-segment tokens. On the place-availability gate, a
    ``direction_discarded`` departure leads in place of the checked heading and the verdict
    follows it ("You named Bangladesh as a recipient, but this measure has no recipients, so
    this checked Bangladesh itself. No data was found there."), since the heading's "No data
    for Bangladesh's ..." would read as a gap in that place's data.
    """
    discarded = next((d for d in departures if d.code == "direction_discarded"), None)
    if gate == "no_member_has_data_at_place" and flagged_match is not None and discarded:
        plan = [_text(discarded.text), _text(" No data was found there.")]
        departures = [d for d in departures if d is not discarded]
    else:
        plan = _no_data_verdict_plan(
            reason,
            gate=gate,
            flagged_match=flagged_match,
            alternate_match=alternate_match,
            nearest_real=nearest_real,
            elected_stat_vars=elected_stat_vars,
            variable_term=variable_term,
            entity_terms=entity_terms,
            entity_groups=entity_groups,
            role_axis=role_axis,
            bound_role_dcids=bound_role_dcids,
            ambiguous_candidates=ambiguous_candidates,
            ambiguous_roles=ambiguous_roles,
            qualifier_terms=qualifier_terms,
            source_terms=source_terms,
            scope=scope,
            denominator=denominator,
            source_code_refusal=source_code_refusal,
        )
    if closing := scope_sentence(reason, scope, qualifier_terms):
        plan.append(_text(f" {closing}" if plan else closing))
    plan.extend(departure_tokens([d.text for d in departures], after_text=bool(plan)))
    return plan


def _no_data_verdict_plan(
    reason: NoDataReason,
    *,
    gate: NoDataGate | None,
    flagged_match: Spec | None,
    alternate_match: Spec | None,
    nearest_real: list[NearestRealSpec] | None,
    elected_stat_vars: list[GraphRef] | None,
    variable_term: str | None,
    entity_terms: list[str] | None,
    entity_groups: Sequence[Sequence[str]] | None,
    role_axis: EntityRoleKey | None,
    bound_role_dcids: Mapping[EntityRoleKey, Sequence[str]] | None,
    ambiguous_candidates: list[str] | None,
    ambiguous_roles: Mapping[str, RoleAxis] | None,
    qualifier_terms: list[ResolvedQualifier] | None,
    source_terms: list[str] | None,
    scope: ScopeSources | None,
    denominator: NoDataDenominator | None,
    source_code_refusal: SourceCodeRefusal | None,
) -> list[SentenceToken]:
    """The no-data plan before its departures and its scope sentence
    (`no_data_sentence_plan`)."""
    if gate == "no_member_has_data_at_place" and flagged_match is not None:
        plan = _checked_match_plan(
            no_data_phrase(reason, variable_term=variable_term, role_axis=role_axis),
            reason,
            flagged_match,
            elected_stat_vars,
            checked=True,
        )
    elif gate == "only_alternate_has_data" and alternate_match is not None:
        # Name the elected concept in the verdict before the different-variable alternate.
        phrase = no_data_phrase(
            reason,
            variable_term=variable_term,
            entity_terms=entity_terms,
            entity_groups=entity_groups,
            role_axis=role_axis,
            elected_stat_vars=elected_stat_vars,
        )
        if alternate_preserves_bound_role(alternate_match, bound_role_dcids):
            lead = _near_match_lead_plan(_near_match_heading(alternate_match))
            plan = _compose_nearest_series(phrase, reason, lead)
        else:
            # Withhold unsafe directional matches.
            plan = _bare_no_data_plan(phrase, reason)
    elif reason in ("role_incompatible", "qualifier_not_bound") and flagged_match is not None:
        plan = [
            _text(
                no_data_phrase(
                    reason,
                    variable_term=variable_term,
                    entity_terms=entity_terms,
                    entity_groups=entity_groups,
                    role_axis=role_axis,
                    qualifier_terms=qualifier_terms,
                    scope=scope,
                )
            )
        ]
    elif reason == "denominator_not_available" and flagged_match is not None:
        plan = _denominator_plan(
            no_data_phrase(reason, variable_term=variable_term),
            flagged_match,
            nearest_real,
            denominator,
        )
    elif flagged_match is not None:
        phrase = no_data_phrase(
            reason,
            variable_term=variable_term,
            entity_terms=entity_terms,
            entity_groups=entity_groups,
            role_axis=role_axis,
        )
        plan = _checked_match_plan(
            phrase,
            reason,
            flagged_match,
            elected_stat_vars,
            checked=_is_elected_variable(flagged_match, elected_stat_vars),
        )
    elif nearest_real:
        phrase = no_data_phrase(
            reason,
            variable_term=variable_term,
            entity_terms=entity_terms,
            entity_groups=entity_groups,
            role_axis=role_axis,
        )
        # Siblings keep the how properties that set them apart, as their own plans do.
        lead = _near_match_lead_plan(
            _near_match_heading(
                nearest_real[0], keep_how_properties=sibling_how_properties(nearest_real)
            )
        )
        plan = _compose_nearest_series(phrase, reason, lead)
    elif reason == "entity_ambiguous" and (
        ambiguous_plan := _ambiguous_entity_plan(
            entity_terms,
            ambiguous_candidates=ambiguous_candidates,
            ambiguous_roles=ambiguous_roles,
        )
    ):
        plan = ambiguous_plan
    else:
        # Include the elected concept when the bare verdict has no match to name.
        plan = _bare_no_data_plan(
            no_data_phrase(
                reason,
                variable_term=variable_term,
                entity_terms=entity_terms,
                entity_groups=entity_groups,
                role_axis=role_axis,
                ambiguous_candidates=ambiguous_candidates,
                elected_stat_vars=elected_stat_vars,
                qualifier_terms=qualifier_terms,
                source_terms=source_terms,
                scope=scope,
                source_code_refusal=source_code_refusal,
            ),
            reason,
        )
    return plan


def _near_match_lead_plan(heading: Heading) -> list[SentenceToken]:
    """The near-match's heading as plain text tokens, to follow a lead-in: claim form
    (`heading_tokens`), closed by a period unless it already ends a sentence."""
    lead = _text_tokens(heading_tokens(heading, claim=True))
    if lead and not lead[-1].text.endswith(_SENTENCE_ENDS):
        lead.append(_text("."))
    return lead


def _text_tokens(tokens: Sequence[SentenceToken]) -> list[SentenceToken]:
    """The non-empty tokens' text as text tokens, without slot or entity metadata."""
    return [_text(token.text) for token in tokens if token.text]
