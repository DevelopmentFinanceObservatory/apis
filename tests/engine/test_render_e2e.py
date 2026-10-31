"""Pins sentences returned by `resolve`."""

from __future__ import annotations

import pytest

from search.engine.quantity_read import QuantityOutput
from search.engine.stated_words_read import StatedWordsOutput
from search.models import (
    CandidatesPart,
    DatasetResubmitInput,
    DefinitePart,
    EntityRefs,
    NoDataPart,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    Part,
    RawTextInput,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityEntry,
    SentenceToken,
)
from search.render.tokens import sentence_from_plan
from tests.engine._base_identity import GDP_SHAPE, make_identity, subject_entity
from tests.engine._binders import elected_reply
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.engine._quantity import NO_QUANTITY, NO_RATING
from tests.fixtures import FakeGraph
from tests.render._helpers import assert_round_trip


def _resolve(query: str, *, pac: bool | None = None, sdmx: FakeSdmxClient | None = None) -> Part:
    request = ResolveRequest(
        input=RawTextInput(query=query),
        response_options=ResolveOptions(),
    )
    return offline_resolve(request, pac=pac, sdmx=sdmx or FakeSdmxClient()).parts[0]


# ---------------------------------------------------------------------------
# Case 5: standard place, no spurious directional voice
# ---------------------------------------------------------------------------


def test_standard_place_population_sentence():
    """A standard count uses the StatVar label and voices its subject place.

    Its open date window comes from the confirmed facet range, 1960 to 2026.
    """
    r = _resolve("total population India")
    assert isinstance(r, DefinitePart)
    rendered_sentence = r.spec.resolved.rendered_sentence
    # The when-slot resolves from the resolved stat_var's own confirmed
    # facet range (1960-2026, no explicit date named in the query).
    assert rendered_sentence == "India's total population, 1960 to 2026."
    # The interpretation clause itself (before the resolved-window suffix, which
    # legitimately contains " to " as part of "1960 to 2026") carries no
    # directional "from"/"to" phrase.
    base_clause = rendered_sentence.split(",")[0]
    assert " from " not in base_clause
    assert " to " not in base_clause


def test_definite_sentence_plan_round_trips():
    """The definite response's sentence_plan joins to exactly its rendered_sentence."""
    r = _resolve("total population India")
    assert isinstance(r, DefinitePart)
    spec = r.spec
    assert spec.resolved.rendered_sentence is not None
    assert spec.resolved.sentence_plan is not None
    assert_round_trip(spec.resolved.sentence_plan, spec.resolved.rendered_sentence)


# ---------------------------------------------------------------------------
# Case 6: candidates — count summary with trailing period
# ---------------------------------------------------------------------------


def test_candidates_count_summary_sentence():
    """Candidates: rendered_sentence counts the readings, then states each one's
    heading and a byline led by its source and span.

    The elected reading comes first, then its rival, the share the model nominated
    (status rule 2(b)).
    """
    r = _resolve("remittances as percentage of GDP in Nigeria")
    assert isinstance(r, CandidatesPart)
    assert r.rendered_sentence == (
        "2 readings. "
        "Nigeria's inward remittances as a percentage of nominal gross domestic production. "
        "World Development Indicators · 1977 to 2024. "
        "Nigeria's volume of remittances as a proportion of total GDP. UN SDG · 2000 to 2022."
    )


# ---------------------------------------------------------------------------
# Case 10: sentence_plan (structured sentence), present under the same
# condition as rendered_sentence, and derives it exactly, for every status.
# ---------------------------------------------------------------------------


def test_sentence_plan_present_on_the_resolved_spec():
    """Sentence fields are on the resolved spec, not the definite part."""
    r = offline_resolve(ResolveRequest(input=RawTextInput(query="total population India"))).parts[0]
    assert isinstance(r, DefinitePart)
    assert r.rendered_sentence is None
    assert r.sentence_plan is None
    spec = r.spec
    assert spec.resolved.rendered_sentence is not None
    assert spec.resolved.sentence_plan is not None


def test_sentence_plan_wraps_candidates_summary():
    """The part's plan states the readings as text tokens that join to its sentence;
    each candidate keeps its own sentence plan."""
    r = _resolve("remittances as percentage of GDP in Nigeria")
    assert isinstance(r, CandidatesPart)
    assert r.rendered_sentence is not None
    assert r.sentence_plan is not None
    assert sentence_from_plan(r.sentence_plan) == r.rendered_sentence
    assert all(token.kind == "text" for token in r.sentence_plan)
    # And each candidate carries its own full sentence structure.
    for spec in r.candidates.specs:
        assert spec.resolved.rendered_sentence is not None
        assert spec.resolved.sentence_plan is not None


def test_sentence_plan_wraps_no_data_phrase():
    """no_data: sentence_plan is a single text token wrapping rendered_sentence."""
    r = _resolve("   ")
    assert isinstance(r, NoDataPart)
    assert r.rendered_sentence is not None
    assert r.sentence_plan == [SentenceToken(kind="text", text=r.rendered_sentence)]


# ---------------------------------------------------------------------------
# no_data: the envelope sentence names the reader's own unresolved place term
# ---------------------------------------------------------------------------


def test_no_data_sentence_names_unresolved_entity():
    """The refusal quotes an entity that the graph cannot resolve."""
    r = _resolve("health ODA grants from USA to Atlantis")
    assert isinstance(r, NoDataPart)
    assert r.no_data.reason == "entity_not_resolved"
    assert r.no_data.entity_terms == ["Atlantis"]
    assert r.rendered_sentence == (
        'Could not use "Atlantis" as a place for this query. Try a different place name.'
    )


class _FixedBindLLM:
    """Elects shape index 0 unconditionally, with no constraint bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        if schema is StatedWordsOutput:
            return NO_RATING
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0), None


def test_place_other_kind_resolution_reaches_no_observations():
    """A State resolves through the graph's place hierarchy and reaches materialisation.

    With no observations, the request returns `no_observations` after resolution.
    """
    graph = FakeGraph(resolve_any={"California": "geoId/06"})
    request = ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(variable_text="income", entities=[ParsedEntity(text="California")]),
            ]
        ),
        response_options=ResolveOptions(),
    )
    r = offline_resolve(
        request,
        graph=graph,
        llm=_FixedBindLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    ).parts[0]
    assert isinstance(r, NoDataPart)
    assert r.no_data.reason == "no_observations"
    # The checked series is stated in claim form. Its open_axis_facts state the
    # age/income constraints it declares but this shape's grammar never modelled as
    # how-slots: a constant axis speaks instead of staying silent. The fixture code
    # index carries no label for either dcid (it holds only the four CRS code
    # systems), so each value renders through _humanize_ref's dcid fallback
    # ("Years15Onwards" -> "years 15 onwards", "WithIncome" -> "with income") rather
    # than the raw dcid or silence: the served members really do agree on both
    # constraints. The checked member is not its family's total, so the part departs
    # with the measure it checked.
    assert r.rendered_sentence == (
        "No data for California's person for years 15 onwards with income. You asked for "
        "income. This checked person, the closest measure the data holds."
    )


@pytest.mark.parametrize(
    ("entity", "term"),
    [
        (ParsedEntity(text="African Development Bank"), "African Development Bank"),
        (ParsedEntity(text="AfDB", dcid="AfricanDevelopmentBank"), "AfDB"),
    ],
    ids=["named", "pinned"],
)
def test_no_data_sentence_names_role_not_admitted_entity(entity, term):
    """A subject-only population shape refuses an Organization as its place, quoted in the
    request's words, a pinned row's included."""
    graph = FakeGraph()
    request = ResolveRequest(
        input=ParsedInput(parts=[ParsedPart(variable_text="population", entities=[entity])]),
        response_options=ResolveOptions(),
    )
    r = offline_resolve(
        request,
        graph=graph,
        llm=_FixedBindLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    ).parts[0]
    assert isinstance(r, NoDataPart)
    assert r.no_data.reason == "entity_role_not_admitted"
    assert r.no_data.entity_terms == [term]
    assert r.rendered_sentence == f'"population" data is not reported for "{term}".'


def test_no_data_sentence_names_ambiguous_entity_candidates():
    """The refusal names both Congo matches and returns them as structured candidates.

    No variable candidate is recalled, so ambiguity is resolved before role admission.
    """
    graph = FakeGraph()
    request = ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(
                    variable_text="zzz nonexistent indicator zzz",
                    entities=[ParsedEntity(text="Congo")],
                ),
            ]
        ),
        response_options=ResolveOptions(),
    )
    r = offline_resolve(
        request,
        graph=graph,
        llm=_FixedBindLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    ).parts[0]
    assert isinstance(r, NoDataPart)
    assert r.no_data.reason == "entity_ambiguous"
    assert r.no_data.entity_terms == ["Congo"]
    assert r.rendered_sentence == (
        '"Congo" matches more than one entity: "Congo [DRC]" and "Congo [Republic]". '
        "Be more specific."
    )

    # Each candidate carries its dcid, label, and kind.
    assert [g.text for g in r.no_data.ambiguous_entities] == ["Congo"]
    group = r.no_data.ambiguous_entities[0]
    assert {c.ref for c in group.candidates} == {"country/COD", "country/COG"}
    assert {c.label for c in group.candidates} == {"Congo [DRC]", "Congo [Republic]"}
    assert all(c.kind == "country" for c in group.candidates)

    # Ambiguous terms also appear in the restatement.
    assert r.restatement is not None
    restated_texts = {e.text for e in (r.restatement.entities or [])}
    assert all(g.text in restated_texts for g in r.no_data.ambiguous_entities)


def test_no_data_restatement_for_directional_ambiguous_entity():
    """An ambiguous directional entity keeps its role text and structured matches."""
    graph = FakeGraph()
    request = ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(
                    variable_text="zzz nonexistent indicator zzz",
                    entities=[ParsedEntity(text="Congo", role_text="to")],
                ),
            ]
        ),
        response_options=ResolveOptions(),
    )
    r = offline_resolve(
        request,
        graph=graph,
        llm=_FixedBindLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    ).parts[0]
    assert isinstance(r, NoDataPart)
    assert r.no_data.reason == "entity_ambiguous"
    assert r.restatement is not None
    assert [e.text for e in r.restatement.entities] == ["Congo"]
    assert r.restatement.entities[0].role_text == "to"
    # An ambiguous text binds no place.
    assert r.entity_refs == [EntityRefs(text="Congo", refs=[])]
    restated_texts = {e.text for e in r.restatement.entities}
    assert all(g.text in restated_texts for g in r.no_data.ambiguous_entities)
    assert {c.ref for c in r.no_data.ambiguous_entities[0].candidates} == {
        "country/COD",
        "country/COG",
    }


def test_no_data_refused_before_binding_states_its_resolved_texts_refs():
    """A part refused before its places bind states the node each resolved text names,
    and [] for the ambiguous one."""
    request = ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(
                    variable_text="zzz nonexistent indicator zzz",
                    entities=[
                        ParsedEntity(text="Netherlands", role_text="from"),
                        ParsedEntity(text="Congo", role_text="to"),
                    ],
                ),
            ]
        ),
        response_options=ResolveOptions(),
    )
    r = offline_resolve(
        request,
        graph=FakeGraph(),
        llm=_FixedBindLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    ).parts[0]
    assert isinstance(r, NoDataPart)
    assert r.no_data.reason == "entity_ambiguous"
    assert r.entity_refs == [
        EntityRefs(text="Netherlands", refs=["country/NLD"]),
        EntityRefs(text="Congo", refs=[]),
    ]


# ---------------------------------------------------------------------------
# no_data: a resubmit quotes its posted entry text, never a display slug
# ---------------------------------------------------------------------------


def test_no_data_sentence_from_a_resubmitted_identity_quotes_the_posted_text():
    """A failed resubmit quotes its posted variable text, not the display slug."""
    entities = [subject_entity("country/IND")]
    request = ResolveRequest(
        input=DatasetResubmitInput(
            entries=[
                ResubmitIdentityEntry(
                    variable_text="nominal GDP of India",
                    # No StatVar of the graph carries this shape.
                    identity=make_identity(
                        {**GDP_SHAPE, "population_type": "NoSuchActivity"}, entities=entities
                    ),
                )
            ]
        ),
        response_options=ResolveOptions(),
    )
    r = offline_resolve(request, graph=FakeGraph(), sdmx=FakeSdmxClient()).parts[0]
    assert isinstance(r, NoDataPart)
    assert r.no_data.reason == "no_matching_slice"
    assert r.rendered_sentence == (
        'No published series for "nominal GDP of India" matches the combination this query '
        "asks for."
    )
    assert "economicactivity_amount_measuredvalue" not in r.rendered_sentence
