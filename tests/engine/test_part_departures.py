"""Departures reach the part on every sentence path, and never the Spec's own sentence.

Each part's `departures` carry the text `render.departures.departure_text` writes for its
fact, in `DEPARTURE_ORDER`. A candidates part speaks for every reading it shows, so a
fact raised beside a warning stamped on one reading stays off it.
"""

from __future__ import annotations

import dataclasses

import pytest

from search.engine.closest_match import substituted_measure_fact
from search.engine.conjoin import assemble_region
from search.engine.departures import (
    measure_not_applied,
    price_basis_not_applied,
    warning_key,
)
from search.engine.envelope import now_ms
from search.engine.region import RETRIEVAL_SCORE_WEAK, RegionResult
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.region.materialise_step import Departed, family_total_absent_departure
from search.engine.region.result import with_entity_words
from search.engine.retrieval.retrieve import FamilyTotalAbsent
from search.heading.departure_fact import DepartureFact
from search.models import (
    Entity,
    EntityGroup,
    EntityRoleSubject,
    GraphRef,
    Operation,
    Part,
    ResolvedFacet,
    Spec,
    Warning,
)
from search.render.departures import departure_text
from search.render.heading import heading_for
from tests.engine._builders import conjoin_spec as _spec
from tests.engine._conjoin_fixtures import _ECON_SHAPE, _PERSON_SHAPE, _region

_SHARE = Operation(verb="share_of", phrase="share of aid going to Kenya")
_PRICES = Warning(
    code=QUERY_QUALIFIER_DROPPED,
    severity="degraded",
    message='The requested qualifier "constant 2022 dollars" was not applied.',
)
_WEAK = Warning(code=RETRIEVAL_SCORE_WEAK, severity="info", message="weak")


def _places() -> list[Entity]:
    return [
        Entity(ref=GraphRef(ref=dcid, label=label), role=EntityRoleSubject())
        for dcid, label in (("country/KEN", "Kenya"), ("country/SOM", "Somaliland"))
    ]


def _unserved(spec: Spec, *dcids: str) -> Spec:
    """`spec` stating each of `dcids` unserved as its subject."""
    pairs = tuple(("observationAbout", dcid) for dcid in dcids)
    return spec.model_copy(update={"resolved": spec.resolved.stating_unserved(pairs)})


def _assemble(region: RegionResult) -> Part:
    response = assemble_region(
        region,
        extra_warnings=[],
        start_ms=now_ms(),
        engine_build="test",
        max_candidates=10,
    )
    (part,) = response.parts
    return part


def _spec_sentences(part: Part) -> list[Spec]:
    """Every Spec whose own sentence the part carries."""
    if part.status == "definite":
        return [part.spec]
    if part.status == "candidates":
        return list(part.candidates.specs)
    no_data = part.no_data
    return [
        spec
        for spec in (no_data.flagged_match, no_data.alternate_match, *(no_data.nearest_real or []))
        if spec is not None
    ]


def _assert_departures(part: Part, expected: list[DepartureFact]) -> None:
    """No Spec sentence holds a departure token, and the part's departures are `expected`
    in order, each with the text its fact renders to on a part that shows data or, on a
    no-data part, on one that shows none."""
    for spec in _spec_sentences(part):
        assert spec.resolved.sentence_plan
        assert not [t for t in spec.resolved.sentence_plan if t.segment == "departure"]
    assert [(d.code, d.binding_ref, d.asked, d.shown, d.text) for d in part.departures] == [
        (
            f.code,
            f.binding_ref,
            f.asked,
            f.shown,
            departure_text(f, family=None, shows_data=part.status != "no_data"),
        )
        for f in expected
    ]


def test_a_definite_part_carries_every_departure_its_answer_and_request_raise():
    """A weak reading of "cost of living" answered with a population count, a share not
    run, and Somaliland unserved: three departures, in order. The dropped price basis
    departs not at all, since a population count has no price basis to show."""
    spec = _unserved(_spec(_PERSON_SHAPE, "ValA", entities=_places()), "country/SOM")
    region = dataclasses.replace(
        _region(spec, "cost of living"),
        warnings=(_PRICES, _WEAK),
        operations=(_SHARE,),
        departure_facts={
            warning_key(_PRICES): price_basis_not_applied(asked="constant 2022 dollars", shown=None)
        },
    )

    part = _assemble(region)

    measure_noun = part.departures[0].shown
    assert measure_noun
    _assert_departures(
        part,
        [
            DepartureFact(
                code="closest_match",
                binding_ref=None,
                asked="cost of living",
                shown=measure_noun,
            ),
            DepartureFact(
                code="operation_not_applied",
                binding_ref=None,
                asked=_SHARE.phrase,
                verb="share_of",
            ),
            DepartureFact(
                code="entity_unavailable",
                binding_ref="where:",
                asked="Somaliland",
                entities=("Somaliland",),
            ),
        ],
    )
    assert part.departures[-1].text == (
        "You asked for Somaliland. The data holds no values for it."
    )


def test_a_candidates_part_keeps_only_the_departures_every_reading_carries():
    """The price basis was dropped for the whole request; a direction was discarded for
    one reading only, stamped with its spec_id. Neither reaches the part: the readings
    share no byline to state the prices shown, and the direction holds for one reading."""
    first = _spec(_PERSON_SHAPE, "ValA", sv_dcid="sv_a")
    second = _spec(_PERSON_SHAPE, "ValB", sv_dcid="sv_b")
    one_reading = Warning(
        code="direction_discarded",
        severity="info",
        message="Kenya was named as a donor.",
        spec_id=second.spec_id,
    )
    region = dataclasses.replace(
        _region(first, "population", status="candidates", extra_specs=(second,)),
        warnings=(_PRICES, one_reading),
        operations=(_SHARE,),
        departure_facts={
            warning_key(_PRICES): price_basis_not_applied(
                asked="constant 2022 dollars", shown=None
            ),
            warning_key(one_reading): DepartureFact(
                code="direction_discarded", binding_ref="where:", asked="Kenya", over="origin"
            ),
        },
    )

    part = _assemble(region)

    assert part.status == "candidates"
    _assert_departures(
        part,
        [
            DepartureFact(
                code="operation_not_applied",
                binding_ref=None,
                asked=_SHARE.phrase,
                verb="share_of",
            ),
        ],
    )


def test_a_no_data_part_carries_its_raised_and_operation_departures():
    spec = _spec(_PERSON_SHAPE, "ValA")
    region = dataclasses.replace(
        _region(
            None,
            "population",
            status="no_data",
            no_data_reason="no_observations",
            gate="only_alternate_has_data",
            alternate_match=spec,
        ),
        warnings=(_PRICES,),
        operations=(_SHARE,),
        departure_facts={
            warning_key(_PRICES): price_basis_not_applied(asked="constant 2022 dollars", shown=None)
        },
    )

    part = _assemble(region)

    assert part.status == "no_data"
    _assert_departures(
        part,
        [
            DepartureFact(
                code="operation_not_applied",
                binding_ref=None,
                asked=_SHARE.phrase,
                verb="share_of",
            ),
        ],
    )


@pytest.mark.parametrize("reason", ["entity_ambiguous", "role_not_resolved"])
def test_a_no_data_part_that_ran_no_check_departs_only_for_search_itself(reason):
    """mls-027 "How much goes through the World Bank?": "World Bank" is ambiguous, so no
    check ran to leave "financial flows" out; an operation Search does not compute still
    departs."""
    dropped = Warning(
        code=QUERY_QUALIFIER_DROPPED,
        severity="degraded",
        message='The requested qualifier "financial flows" was not applied.',
    )
    region = dataclasses.replace(
        _region(None, "financial flows", status="no_data", no_data_reason=reason),
        warnings=(dropped,),
        operations=(_SHARE,),
        departure_facts={
            warning_key(dropped): measure_not_applied(
                asked="financial flows", property_dcid="populationType"
            )
        },
    )

    part = _assemble(region)

    assert part.status == "no_data"
    _assert_departures(
        part,
        [
            DepartureFact(
                code="operation_not_applied",
                binding_ref=None,
                asked=_SHARE.phrase,
                verb="share_of",
            ),
        ],
    )


def test_a_dropped_price_basis_shows_the_unit_the_byline_states():
    """The raise site does not know the served unit; assembly fills `shown` from it."""
    spec = _spec(_ECON_SHAPE, "ValA")
    spec = spec.model_copy(
        update={
            "resolved": spec.resolved.model_copy(
                update={
                    "facets": {
                        "unit": ResolvedFacet(
                            basis="fallback", ref="USDollar", label="US Dollar", currency="USD"
                        )
                    }
                }
            )
        }
    )
    region = dataclasses.replace(
        _region(spec, "gdp"),
        warnings=(_PRICES,),
        departure_facts={
            warning_key(_PRICES): price_basis_not_applied(asked="constant 2022 dollars", shown=None)
        },
    )

    part = _assemble(region)

    _assert_departures(
        part, [price_basis_not_applied(asked="constant 2022 dollars", shown="current US dollars")]
    )
    assert part.departures[0].text == (
        "You asked for constant 2022 dollars. This shows current US dollars."
    )


def test_a_qualifier_the_heading_states_does_not_depart():
    """weo-009: "gross" was dropped, but the heading's measure noun ("gross capital
    formation") already states it; a qualifier the noun lacks still departs."""
    spec = _spec(_ECON_SHAPE, "ValA")
    noun_word = heading_for(spec).heading.noun.inline_text.split()[0]
    stated = Warning(
        code=QUERY_QUALIFIER_DROPPED,
        severity="degraded",
        message=f'The requested qualifier "{noun_word}" was not applied.',
    )
    region = dataclasses.replace(
        _region(spec, "gdp"),
        warnings=(stated, _PRICES),
        departure_facts={
            warning_key(stated): measure_not_applied(
                asked=noun_word.upper(), property_dcid="nettingBasis"
            ),
            warning_key(_PRICES): measure_not_applied(
                asked="grant equivalent", property_dcid="crsFinanceType"
            ),
        },
    )

    part = _assemble(region)

    assert [d.asked for d in part.departures] == ["grant equivalent"]


def test_a_part_states_one_closest_match():
    """An answer on the elected shape's measure sibling raises its `closest_match` beside
    its warning; under a weak score the lexical rule would raise a second, which the part
    does not state."""
    spec = _spec(_PERSON_SHAPE, "ValA")
    sibling = Warning(code="elected_shape_no_data", severity="warn", message="sibling")
    raised = substituted_measure_fact(
        variable_text="cost of living", measure_noun=heading_for(spec).heading.noun.inline_text
    )
    region = dataclasses.replace(
        _region(spec, "cost of living"),
        warnings=(sibling, _WEAK),
        departure_facts={warning_key(sibling): raised},
    )

    part = _assemble(region)

    _assert_departures(part, [raised])


@pytest.mark.parametrize(
    ("region_words", "door_words"),
    [
        # crs-006 "aid from EU institutions to Ethiopia": extraction names the group
        # "European Union", the graph's label for it too.
        ({}, {"European Union": "EU institutions", "Ethiopia": "Ethiopia"}),
        # Extraction names the group "EU" and the answer labels it as the graph does
        # (`place_roles.group_request_names`).
        ({"European Union": "EU"}, {"EU": "EU institutions", "Ethiopia": "Ethiopia"}),
    ],
    ids=["named_as_labelled", "named_otherwise"],
)
def test_a_group_s_departures_ask_in_the_request_s_words(region_words, door_words):
    """Both a group's departures ask for it in the query's own words, and the members
    without values are counted as the heading names the group."""
    spec = _spec(_PERSON_SHAPE, "ValA", entities=_places())
    spec = spec.model_copy(
        update={
            "resolved": spec.resolved.model_copy(
                update={"entity_groups": {"observationAbout": EntityGroup(label="European Union")}}
            )
        }
    )
    region = with_entity_words(
        dataclasses.replace(
            _region(_unserved(spec, "country/SOM"), "aid"), entity_words=region_words
        ),
        door_words,
    )

    part = _assemble(region)

    _assert_departures(
        part,
        [
            DepartureFact(
                code="entity_unavailable",
                binding_ref="where:",
                asked="EU institutions",
                entities=("Somaliland",),
                group="European Union",
                group_size=2,
            ),
            DepartureFact(
                code="group_read_as_members", binding_ref="where:", asked="EU institutions"
            ),
        ],
    )
    assert part.departures[0].text == (
        "The data holds no values for 1 of the 2 European Union members."
    )


def test_a_resubmit_s_departures_ask_for_nothing():
    """A resubmit of crs-006 "aid from EU institutions to Ethiopia" holds no words of the
    request, so every departure's `asked` is null. The group read as members is named as the
    heading names it, the posted value left out by its name, and the members without
    values keep their text, which does not quote `asked`."""
    spec = _spec(_PERSON_SHAPE, "ValA", entities=_places())
    spec = spec.model_copy(
        update={
            "resolved": spec.resolved.model_copy(
                update={"entity_groups": {"observationAbout": EntityGroup(label="European Union")}}
            )
        }
    )
    dropped = Warning(
        code=QUERY_QUALIFIER_DROPPED,
        severity="degraded",
        message='The requested qualifier "debt instruments" was not applied.',
    )
    region = dataclasses.replace(
        _region(_unserved(spec, "country/SOM"), "aid"),
        entity_words=None,
        warnings=(dropped,),
        departure_facts={
            warning_key(dropped): measure_not_applied(
                asked="debt instruments", property_dcid="crsTypeOfFinance"
            )
        },
    )

    part = _assemble(region)

    assert [(d.code, d.asked, d.text) for d in part.departures] == [
        (
            "measure_not_applied",
            None,
            "This shows the measure without debt instruments.",
        ),
        (
            "entity_unavailable",
            None,
            "The data holds no values for 1 of the 2 European Union members.",
        ),
        (
            "group_read_as_members",
            None,
            "This shows the 2 European Union members, each on its own.",
        ),
    ]


def test_a_role_not_resolved_frame_names_the_place_in_the_query_s_words():
    """crs-010 "Gates Foundation grants to Nigeria": the frame names the one name whose
    direction the query does not tell, in the query's own words for it."""
    region = with_entity_words(
        dataclasses.replace(
            _region(None, "grants", status="no_data", no_data_reason="role_not_resolved"),
            unresolved_entity_names=("Bill & Melinda Gates Foundation",),
        ),
        {"Bill & Melinda Gates Foundation": "Gates Foundation", "Nigeria": "Nigeria"},
    )

    part = _assemble(region)

    assert part.status == "no_data"
    assert part.no_data.entity_terms == ["Gates Foundation"]
    assert part.rendered_sentence == (
        'Could not tell whether "Gates Foundation" is giving or receiving. '
        'Put "from" or "to" before a name.'
    )


def test_a_role_states_one_entity_unavailable_whichever_reads_raise_it():
    """The read plan serves no point at Somaliland, and the UN front door refused Kenya,
    so one departure on the role names both, the Spec's entity first."""
    spec = _unserved(_spec(_PERSON_SHAPE, "ValA", entities=_places()), "country/SOM")
    refused = Warning(code="un_system_unavailable", severity="warn", message="refused")
    region = dataclasses.replace(
        _region(spec, "population"),
        warnings=(refused,),
        departure_facts={
            warning_key(refused): DepartureFact(
                code="entity_unavailable", binding_ref="where:", asked="Kenya", entities=("Kenya",)
            )
        },
    )

    part = _assemble(region)

    _assert_departures(
        part,
        [
            DepartureFact(
                code="entity_unavailable",
                binding_ref="where:",
                asked="Somaliland, Kenya",
                entities=("Somaliland", "Kenya"),
            )
        ],
    )


def _family_total_absent(spec: Spec) -> Departed:
    """The `family_total_absent` warning and departure for `spec`'s one member."""
    return family_total_absent_departure(
        FamilyTotalAbsent(total=None, shown="sv_a", places=()),
        spec=spec,
        variable_text="child mortality",
        label_of=lambda dcids: [f"label of {dcid}" for dcid in dcids],
    )


def test_a_definite_part_showing_one_member_of_a_family_states_the_closest_match():
    spec = _spec(_PERSON_SHAPE, "ValA", sv_dcid="sv_a")
    warning, fact = _family_total_absent(spec)
    region = dataclasses.replace(
        _region(spec, "child mortality"),
        warnings=(warning,),
        departure_facts={warning_key(warning): fact},
    )

    part = _assemble(region)

    assert warning.spec_id == spec.spec_id
    _assert_departures(
        part,
        [
            DepartureFact(
                code="closest_match",
                binding_ref=None,
                asked="child mortality",
                shown=heading_for(spec).heading.noun.inline_text,
            )
        ],
    )


_ELECTED_NO_DATA = Warning(
    code="elected_shape_no_data",
    severity="info",
    message=(
        "the originally chosen reading had no observations at this place; a sibling "
        "reading confirmed with real data was elected instead."
    ),
)


def _labelled(spec: Spec, label: str) -> Spec:
    """`spec` with its one StatVar named `label`."""
    (stat_var,) = spec.resolved.members
    ref = stat_var.ref.model_copy(update={"label": label})
    return spec.model_copy(
        update={
            "resolved": spec.resolved.stating_members([stat_var.model_copy(update={"ref": ref})])
        }
    )


def test_a_candidates_part_offered_for_an_elected_measure_without_data_states_one_closest_match():
    """The elected measure holds no data and no reading names "child mortality", so the
    readings stand in for it, with one plural `closest_match`. One reading shows a member
    of a family without its total. That departure holds for that reading alone, so the part
    leaves it out."""
    first = _spec(_PERSON_SHAPE, "ValA", sv_dcid="sv_a")
    second = _spec(_PERSON_SHAPE, "ValB", sv_dcid="sv_b")
    family_warning, family_fact = _family_total_absent(first)
    region = dataclasses.replace(
        _region(first, "child mortality", status="candidates", extra_specs=(second,)),
        warnings=(_ELECTED_NO_DATA, family_warning),
        departure_facts={warning_key(family_warning): family_fact},
    )

    part = _assemble(region)

    assert part.status == "candidates"
    _assert_departures(
        part,
        [
            DepartureFact(
                code="closest_match", binding_ref=None, asked="child mortality", readings=True
            )
        ],
    )
    assert part.departures[0].text == (
        "You asked for child mortality. These readings are the closest measures the data holds."
    )


@pytest.mark.parametrize(
    ("variable_text", "first", "departs"),
    [
        # weo-002 "GDP of Kenya in current US dollars": the first reading's measure noun
        # is GDP.
        pytest.param("GDP", _spec(_ECON_SHAPE, "GDP", sv_dcid="sv_a"), False, id="noun"),
        # weo-003 "the economy of Kenya": the first reading's StatVar name holds the words.
        pytest.param(
            "the economy",
            _labelled(_spec(_ECON_SHAPE, "ValA", sv_dcid="sv_a"), "Size of the economy"),
            False,
            id="stat-var-name",
        ),
        pytest.param("GDP", _spec(_ECON_SHAPE, "ValA", sv_dcid="sv_a"), True, id="none"),
    ],
)
def test_a_candidates_part_elected_in_place_of_a_measure_without_data_departs_unless_a_reading_names_it(
    variable_text: str, first: Spec, departs: bool
):
    """Readings with data were elected in place of a measure without, and the site that
    elected them raised no departure. The part states the plural `closest_match` only when
    no reading names the measure asked, by the definite rule's test (`closest_match_fact`)
    run on each reading."""
    second = _spec(_ECON_SHAPE, "ValB", sv_dcid="sv_b")
    region = dataclasses.replace(
        _region(first, variable_text, status="candidates", extra_specs=(second,)),
        warnings=(_ELECTED_NO_DATA,),
    )

    part = _assemble(region)

    assert part.status == "candidates"
    expected = DepartureFact(
        code="closest_match", binding_ref=None, asked=variable_text, readings=True
    )
    _assert_departures(part, [expected] if departs else [])
