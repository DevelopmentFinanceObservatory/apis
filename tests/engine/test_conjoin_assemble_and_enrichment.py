"""Tests response assembly, echo handling, candidate ordering, and same-shape merges.
Fake tasks check enrichment retargets merged specs and preserves untouched tasks.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time

import pytest

from search.engine import conjoin
from search.engine.conjoin import BINDING_FOLDED, assemble_region, combine_regions
from search.engine.deadline import _cancel_and_await
from search.engine.enrichment import _reconcile_after_collapse, _stream_slot_options
from search.engine.envelope import now_ms
from search.engine.events import SlotOptionsEvent
from search.engine.region import RegionResult
from search.engine.region.facet_probe import FacetProbe
from search.engine.region.result import (
    AmbiguousEntity,
    AmbiguousEntityCandidate,
    RoleNotAdmittedEntity,
)
from search.engine.same_shape import collapse_same_shape
from search.engine.spec_build import build_spec
from search.models import (
    BindingSet,
    CandidatesPart,
    CoverageBare,
    NoDataPart,
    ParsedEntity,
    ParsedPart,
    QueryEcho,
    RoleAxis,
    SlotKey,
    SlotOptions,
    StatVar,
    TimeWindow,
    Warning,
)
from search.render.no_data import UnlistedPopulation
from tests.engine._builders import conjoin_spec as _spec
from tests.engine._builders import constraint_value_slot as _value_slot
from tests.engine._builders import gref as _gr
from tests.engine._conjoin_fixtures import _ECON_SHAPE, _PERSON_SHAPE, _region

# ---------------------------------------------------------------------------
# echo kwarg passthrough
# ---------------------------------------------------------------------------


def _assemble_kwargs(region: RegionResult, **overrides):
    """Minimal keyword args for assemble_region / combine_regions."""
    return {
        "query": "test query",
        "extra_warnings": [],
        "start_ms": now_ms(),
        "engine_build": "test",
        "max_candidates": 10,
    } | overrides


def _resubmit_echo(variable_text: str = "aid to ethiopia") -> QueryEcho:
    return QueryEcho(
        entry_path="dataset_resubmit",
        raw_query=None,
        normalized_query=None,
        extract_skipped=True,
    )


def test_assemble_region_default_echo_uses_raw_text():
    """Without an explicit echo, assemble_region builds entry_path='raw_text'."""
    spec = _spec(_PERSON_SHAPE, "any")
    region = _region(spec, "population")
    response = assemble_region(region, **_assemble_kwargs(region))
    assert response.query_echo.entry_path == "raw_text"
    assert response.query_echo.extract_skipped is False


@pytest.mark.parametrize("assemble", ["one", "many"])
def test_the_envelope_total_covers_building_the_parts(monkeypatch, assemble):
    """The request's total is stamped once its parts are built, so it covers their
    assembly."""
    clock = [0]
    monkeypatch.setattr(conjoin, "now_ms", lambda: clock[0])
    build_part = conjoin._build_part

    def slow_build_part(*args, **kwargs):
        clock[0] += 250
        return build_part(*args, **kwargs)

    monkeypatch.setattr(conjoin, "_build_part", slow_build_part)
    region = _region(_spec(_PERSON_SHAPE, "any"), "population")
    kwargs = _assemble_kwargs(region, start_ms=0)

    if assemble == "one":
        response = assemble_region(region, **kwargs)
    else:
        response, _effective = combine_regions([region], **kwargs)

    assert response.diagnostics.timing_ms is not None
    assert response.diagnostics.timing_ms.total == 250


def test_assemble_region_supplied_echo_used_verbatim():
    """When echo is supplied, assemble_region uses it verbatim (entry_path preserved)."""
    spec = _spec(_PERSON_SHAPE, "any")
    region = _region(spec, "aid to ethiopia")
    supplied = _resubmit_echo("aid to ethiopia")
    response = assemble_region(region, echo=supplied, **_assemble_kwargs(region))
    assert response.query_echo.entry_path == "dataset_resubmit"
    assert response.query_echo.extract_skipped is True
    assert response.query_echo.raw_query is None


def test_a_parts_total_leaves_out_the_stages_nested_inside_shape():
    """`shape` spans its `shape_*` stages, so the part's total sums only the top-level
    steps, and the shared `extract` step stays on the envelope."""
    stages = {
        "shape_arcs": 30,
        "shape_probe": 20,
        "shape_labels": 5,
        "shape_groups": 10,
        "shape_admission": 25,
    }
    timing = {"extract": 50, "recall": 10, "shape": 100, **stages, "bind": 7}
    region = dataclasses.replace(
        _region(_spec(_PERSON_SHAPE, "any"), "population"), timing_by_step=timing
    )

    part = assemble_region(region, **_assemble_kwargs(region)).parts[0]

    assert part.timing_ms is not None
    assert part.timing_ms.total == 10 + 100 + 7
    assert part.timing_ms.by_step == {"recall": 10, "shape": 100, **stages, "bind": 7}


def test_assemble_region_candidates_keep_the_status_rules_order():
    """A candidates region states the elected reading first and its rivals in menu order,
    and assembly keeps that order whatever the retrieval scores, as `intent_rank`."""
    spec_a = _spec(_PERSON_SHAPE, "any", sv_dcid="sv_a")
    spec_b = _spec(_ECON_SHAPE, "any", sv_dcid="sv_b")
    spec_c = _spec(_PERSON_SHAPE, "other", sv_dcid="sv_c")

    region = RegionResult(
        variable_text="test variable",
        status="candidates",
        specs=(spec_b, spec_c, spec_a),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
    )

    response = assemble_region(region, **_assemble_kwargs(region))
    part = response.parts[0]
    assert isinstance(part, CandidatesPart)
    assert [s.spec_id for s in part.candidates.specs] == [
        spec_b.spec_id,
        spec_c.spec_id,
        spec_a.spec_id,
    ]
    assert part.candidates.ordering == "intent_rank"


def test_combine_regions_default_echo_uses_raw_text():
    """Without an explicit echo, combine_regions builds entry_path='raw_text'."""
    spec_a = _spec(_PERSON_SHAPE, "any", sv_dcid="sv_a")
    spec_b = _spec(_ECON_SHAPE, "any", sv_dcid="sv_b")
    regions = [
        _region(spec_a, "population", earliest_index=0),
        _region(spec_b, "GDP", earliest_index=1),
    ]
    response, _effective = combine_regions(
        regions,
        query="population and GDP",
        extra_warnings=[],
        start_ms=now_ms(),
        engine_build="test",
        max_candidates=10,
    )
    assert response.query_echo.entry_path == "raw_text"


def test_combine_regions_no_data_part_carries_gate_and_alternate_match():
    """The first no_data part keeps its gate and alternate match when a second part is combined."""
    alternate = _spec(_ECON_SHAPE, "any", sv_dcid="sdg/SH_HIV_INCD")
    regions = [
        _region(
            _spec(_PERSON_SHAPE, "any", sv_dcid="sv_a"),
            "population",
            status="no_data",
            no_data_reason="no_matching_slice",
            earliest_index=0,
            gate="only_alternate_has_data",
            alternate_match=alternate,
        ),
        _region(
            _spec(_ECON_SHAPE, "any", sv_dcid="sv_b"),
            "GDP",
            status="no_data",
            no_data_reason="no_observations",
            earliest_index=1,
        ),
    ]

    response, _effective = combine_regions(
        regions,
        query="population and GDP",
        extra_warnings=[],
        start_ms=now_ms(),
        engine_build="test",
        max_candidates=10,
    )

    first = response.parts[0]
    assert first.status == "no_data"
    assert first.no_data.gate == "only_alternate_has_data"
    assert first.no_data.alternate_match is not None
    assert first.no_data.alternate_match.spec_id == alternate.spec_id


def _entity_terms_part(region: RegionResult) -> NoDataPart:
    response = assemble_region(region, **_assemble_kwargs(region))
    part = response.parts[0]
    assert isinstance(part, NoDataPart)
    return part


def test_assemble_region_threads_every_unresolved_entity_term():
    region = RegionResult(
        variable_text="population",
        status="no_data",
        specs=(),
        no_data_reason="entity_not_resolved",
        warnings=(),
        timing_by_step={},
        unresolved_entity_names=("Atlantis", "El Dorado"),
    )

    part = _entity_terms_part(region)

    assert part.no_data.entity_terms == ["Atlantis", "El Dorado"]
    assert part.rendered_sentence == (
        'Could not use "Atlantis" and "El Dorado" as places for this query. '
        "Try a different place name."
    )


_KEN = "country/KEN"


def test_assemble_region_entity_not_resolved_keeps_ambiguous_candidates_off_the_wire():
    """The entity check names an ambiguous entity with its candidates whatever reason it
    elects. The wire carries candidates only on entity_ambiguous, the one reason whose part
    a client can post back with a candidate pinned."""
    region = RegionResult(
        variable_text="population",
        status="no_data",
        specs=(),
        no_data_reason="entity_not_resolved",
        warnings=(),
        timing_by_step={},
        unresolved_entity_names=("Congo", "Atlantis"),
        parsed_form=ParsedPart(
            variable_text="population",
            entities=[ParsedEntity(text="Congo"), ParsedEntity(text="Atlantis")],
        ),
        ambiguous_entities=(
            AmbiguousEntity(
                text="Congo",
                candidates=(
                    AmbiguousEntityCandidate(
                        dcid="country/COD", name="Congo [DRC]", kind="Country"
                    ),
                    AmbiguousEntityCandidate(
                        dcid="country/COG", name="Congo [Republic]", kind="Country"
                    ),
                ),
            ),
        ),
    )

    part = _entity_terms_part(region)

    assert part.no_data.entity_terms == ["Congo", "Atlantis"]
    assert part.no_data.ambiguous_entities == []
    assert part.no_data.options == []


def test_assemble_region_threads_every_ambiguous_entity_term():
    """Guinea precedes Congo in query order but not alphabetically, so a
    sorted() regression in _no_data_entity_terms would reorder entity_terms
    and this assertion would catch it.
    """
    region = RegionResult(
        variable_text="population",
        status="no_data",
        specs=(),
        no_data_reason="entity_ambiguous",
        warnings=(),
        timing_by_step={},
        ambiguous_entities=(
            AmbiguousEntity(
                text="Guinea",
                candidates=(
                    AmbiguousEntityCandidate(dcid="country/GIN", name="Guinea", kind="Country"),
                ),
            ),
            AmbiguousEntity(
                text="Congo",
                candidates=(
                    AmbiguousEntityCandidate(
                        dcid="country/COD", name="Congo [DRC]", kind="Country"
                    ),
                    AmbiguousEntityCandidate(
                        dcid="country/COG", name="Congo [Republic]", kind="Country"
                    ),
                ),
            ),
        ),
    )

    part = _entity_terms_part(region)

    assert part.no_data.entity_terms == ["Guinea", "Congo"]
    assert part.rendered_sentence == (
        '"Guinea" and "Congo" match more than one entity: "Guinea", '
        '"Congo [DRC]", and "Congo [Republic]". Be more specific.'
    )


_DRC_NAME = "Democratic Republic of the Congo"
_ROC_NAME = "Republic of the Congo"


def _congo_to_congo(roles: tuple[RoleAxis | None, RoleAxis | None]) -> RegionResult:
    """An entity_ambiguous region of two names the request writes "Congo", the door giving
    each name the axis in `roles`."""
    return RegionResult(
        variable_text="aid",
        status="no_data",
        specs=(),
        no_data_reason="entity_ambiguous",
        warnings=(),
        timing_by_step={},
        entity_surfaces={_DRC_NAME: "Congo", _ROC_NAME: "Congo"},
        ambiguous_entities=tuple(
            AmbiguousEntity(
                text=name,
                candidates=(
                    AmbiguousEntityCandidate(dcid="country/COD", name=_DRC_NAME, kind="Country"),
                    AmbiguousEntityCandidate(dcid="country/COG", name=_ROC_NAME, kind="Country"),
                ),
                role=role,
            )
            for name, role in zip((_DRC_NAME, _ROC_NAME), roles, strict=True)
        ),
    )


@pytest.mark.parametrize(
    ("roles", "wordings"),
    [
        (("origin", "destination"), '"Congo" (from) and "Congo" (to)'),
        ((None, None), 'The first "Congo" and the second "Congo"'),
        (("origin", "origin"), 'The first "Congo" and the second "Congo"'),
    ],
    ids=["axis-words", "no-roles-by-order", "equal-roles-by-order"],
)
def test_one_wording_naming_two_groups_is_told_apart_in_the_message(roles, wordings):
    """Two groups both written "Congo" keep the one word in every term and slot, and the
    message and sentence tell them apart outside the quotes."""
    part = _entity_terms_part(_congo_to_congo(roles))

    expected = (
        f"{wordings} each match more than one entity: "
        f'"{_DRC_NAME}" and "{_ROC_NAME}". Be more specific.'
    )
    assert part.no_data.entity_terms == ["Congo", "Congo"]
    assert part.no_data.message == expected
    assert part.rendered_sentence == expected
    assert [
        (token.text, token.entity_choice) for token in part.sentence_plan if token.kind == "slot"
    ] == [("Congo", 0), ("Congo", 1)]


def test_a_refused_role_quotes_the_requests_words_for_the_entity():
    """A role-not-admitted entity is named by its identity, and the refusal quotes the
    request's words for it, keying an unlisted population by those words too."""
    population = UnlistedPopulation(axis="origin", left_out=("private",))
    region = RegionResult(
        variable_text="aid",
        status="no_data",
        specs=(),
        no_data_reason="entity_role_not_admitted",
        warnings=(),
        timing_by_step={},
        entity_words={"DACOfficialDonors": "official donors"},
        role_not_admitted_entities=(
            RoleNotAdmittedEntity(
                text="DACOfficialDonors",
                dcid="DACOfficialDonors",
                graph_type=None,
                unlisted=population,
            ),
        ),
    )

    part = _entity_terms_part(region)

    assert part.no_data.entity_terms == ["official donors"]
    assert part.no_data.message == (
        '"official donors" cannot be listed donor by donor: every source of the "aid" data '
        'also holds private donors, which "official donors" leaves out. Ask for its total, '
        "or name the donors to list."
    )


_TWO_AMBIGUOUS = RegionResult(
    variable_text="ODA",
    status="no_data",
    specs=(),
    no_data_reason="entity_ambiguous",
    warnings=(),
    timing_by_step={},
    parsed_form=ParsedPart(
        variable_text="ODA",
        entities=[
            ParsedEntity(text="Korea", role_text="from"),
            ParsedEntity(text="Congo", role_text="to"),
            ParsedEntity(text="Kenya", role_text="to"),
        ],
    ),
    resolved_entity_dcids={"Kenya": "country/KEN"},
    ambiguous_entities=(
        AmbiguousEntity(
            text="Korea",
            candidates=(
                AmbiguousEntityCandidate(dcid="country/KOR", name="South Korea", kind="Country"),
                AmbiguousEntityCandidate(dcid="country/PRK", name="North Korea", kind="Country"),
            ),
            role="origin",
        ),
        AmbiguousEntity(
            text="Congo",
            candidates=(
                AmbiguousEntityCandidate(dcid="country/COD", name="Congo [DRC]", kind="Country"),
                AmbiguousEntityCandidate(
                    dcid="country/COG", name="Congo [Republic]", kind="Country"
                ),
            ),
            role="destination",
        ),
    ),
)


def test_each_option_pins_one_ambiguous_wording_and_leaves_the_other_bare():
    """Two ambiguous wordings give one option per wording and candidate, in query order.
    Each pins its wording and every entity that resolved to one node, and leaves the other
    wording unpinned for a second refusal."""
    part = _entity_terms_part(_TWO_AMBIGUOUS)

    def pins(option) -> list[tuple[str, str | None]]:
        return [(e.text, e.dcid) for e in option.data_query.parts[0].entities]

    assert [(o.mentions, o.label, pins(o)) for o in part.no_data.options] == [
        (["Korea"], "South Korea", [("Korea", "country/KOR"), ("Congo", None), ("Kenya", _KEN)]),
        (["Korea"], "North Korea", [("Korea", "country/PRK"), ("Congo", None), ("Kenya", _KEN)]),
        (["Congo"], "Congo [DRC]", [("Korea", None), ("Congo", "country/COD"), ("Kenya", _KEN)]),
        (
            ["Congo"],
            "Congo [Republic]",
            [("Korea", None), ("Congo", "country/COG"), ("Kenya", _KEN)],
        ),
    ]


@pytest.mark.parametrize(
    ("group_name", "group_option"),
    [
        ("European Union", [[ParsedEntity(text="European Union", members=True)]]),
        (None, []),
    ],
    ids=["named", "no-name-resolves-to-the-group-alone"],
)
def test_a_group_candidate_on_a_members_row_posts_the_group_name_without_a_pin(
    group_name, group_option
):
    """A pin on a group's own node cannot ask for its members, so the candidate's option
    posts the group's registered name with the members flag and no dcid. A group node with
    no name that resolves to its group alone has no option."""
    region = dataclasses.replace(
        _TWO_AMBIGUOUS,
        parsed_form=ParsedPart(
            variable_text="ODA", entities=[ParsedEntity(text="the Union", members=True)]
        ),
        resolved_entity_dcids={},
        ambiguous_entities=(
            AmbiguousEntity(
                text="the Union",
                candidates=(
                    AmbiguousEntityCandidate(
                        dcid="EuropeanUnion",
                        name="European Union",
                        kind="continental_union",
                        group_node=True,
                        group_name=group_name,
                    ),
                    AmbiguousEntityCandidate(dcid="africa", name="Africa", kind="continent"),
                ),
            ),
        ),
    )

    part = _entity_terms_part(region)

    assert [o.data_query.parts[0].entities for o in part.no_data.options] == [
        *group_option,
        [ParsedEntity(text="the Union", members=True, dcid="africa")],
    ]
    assert {tuple(o.mentions) for o in part.no_data.options} == {("the Union",)}


def test_an_option_the_parsed_door_would_refuse_is_not_offered():
    """The group candidate's option would post "Africa" asked for its members beside the
    row "Africa" asked as a whole, which the parsed door refuses, so only the continent's
    pin is offered."""
    region = dataclasses.replace(
        _TWO_AMBIGUOUS,
        parsed_form=ParsedPart(
            variable_text="ODA",
            entities=[ParsedEntity(text="the Union", members=True), ParsedEntity(text="Africa")],
        ),
        resolved_entity_dcids={},
        ambiguous_entities=(
            AmbiguousEntity(
                text="the Union",
                candidates=(
                    AmbiguousEntityCandidate(
                        dcid="AfricanUnion",
                        name="African Union",
                        kind="continental_union",
                        group_node=True,
                        group_name="Africa",
                    ),
                    AmbiguousEntityCandidate(dcid="africa", name="Africa", kind="continent"),
                ),
            ),
        ),
    )

    part = _entity_terms_part(region)

    assert [o.data_query.parts[0].entities for o in part.no_data.options] == [
        [ParsedEntity(text="the Union", members=True, dcid="africa"), ParsedEntity(text="Africa")]
    ]


def test_assemble_region_non_entity_reason_ignores_unresolved_entity_names():
    """C50b: entity_terms is empty for every no_data reason other than
    entity_not_resolved, entity_ambiguous, and entity_role_not_admitted, even
    when unresolved_entity_names is (spuriously) populated. This must go
    through assemble_region/build_no_data_part -- a bare NoData(...) never
    reaches _no_data_entity_terms, the function that decides this, and would
    pass vacuously.
    """
    region = RegionResult(
        variable_text="population",
        status="no_data",
        specs=(),
        no_data_reason="no_observations",
        warnings=(),
        timing_by_step={},
        unresolved_entity_names=("Atlantis", "El Dorado"),
    )

    part = _entity_terms_part(region)

    assert part.no_data.entity_terms == []


def test_assemble_region_threads_every_role_not_admitted_entity_term():
    region = RegionResult(
        variable_text="population",
        status="no_data",
        specs=(),
        no_data_reason="entity_role_not_admitted",
        warnings=(),
        timing_by_step={},
        role_not_admitted_entities=(
            RoleNotAdmittedEntity(
                text="African Development Bank", dcid="org/ADB", graph_type="Organization"
            ),
            RoleNotAdmittedEntity(text="World Bank", dcid="org/WB", graph_type="Organization"),
        ),
    )

    part = _entity_terms_part(region)

    assert part.no_data.entity_terms == ["African Development Bank", "World Bank"]
    assert part.rendered_sentence == (
        '"population" data is not reported for "African Development Bank" and "World Bank".'
    )


def test_combine_regions_supplied_echo_used_verbatim():
    """When echo is supplied, combine_regions uses it verbatim across all branches."""
    spec_a = _spec(_PERSON_SHAPE, "any", sv_dcid="sv_a")
    spec_b = _spec(_ECON_SHAPE, "any", sv_dcid="sv_b")
    regions = [
        _region(spec_a, "population", earliest_index=0),
        _region(spec_b, "GDP", earliest_index=1),
    ]
    supplied = _resubmit_echo("population and GDP")
    response, _effective = combine_regions(
        regions,
        query="population and GDP",
        extra_warnings=[],
        start_ms=now_ms(),
        engine_build="test",
        max_candidates=10,
        echo=supplied,
    )
    assert response.query_echo.entry_path == "dataset_resubmit"
    assert response.query_echo.extract_skipped is True


def test_combine_regions_full_collapse_threads_echo():
    """A same-shape collapse reducing every part to one shared spec still
    threads the supplied echo through combine_regions' own per-part loop."""
    # Two regions with the same five-tuple collapse to one shared spec.
    spec_a = _spec(_PERSON_SHAPE, "grants", sv_dcid="sv_grants")
    spec_b = _spec(_PERSON_SHAPE, "loans", sv_dcid="sv_loans")
    regions = [
        _region(spec_a, "grants", earliest_index=0),
        _region(spec_b, "loans", earliest_index=1),
    ]
    supplied = _resubmit_echo("grants and loans")
    response, _effective = combine_regions(
        regions,
        query="grants and loans",
        extra_warnings=[],
        start_ms=now_ms(),
        engine_build="test",
        max_candidates=10,
        echo=supplied,
    )
    assert response.query_echo.entry_path == "dataset_resubmit"
    assert response.query_echo.extract_skipped is True
    assert response.parts[0].spec.spec_id == response.parts[1].spec.spec_id
    assert BINDING_FOLDED not in [warning.code for warning in response.parts[0].warnings]
    assert BINDING_FOLDED in [warning.code for warning in response.parts[1].warnings]


# ---------------------------------------------------------------------------
# collapse_same_shape propagates value_kind from source binding
# ---------------------------------------------------------------------------


def test_collapse_same_shape_propagates_enum_value_kind():
    """Merged BindingSet carries value_kind from the source slot, not hardcoded 'entity'.

    _value_slot builds what-axis slots with value_kind='enum_value'. After collapse,
    the merged BindingSet values must also carry 'enum_value', not 'entity'.
    """
    spec_a = _spec(_PERSON_SHAPE, "ValA")
    spec_b = _spec(_PERSON_SHAPE, "ValB")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged_binding = effective[0].spec.resolved.slots[0].binding
    assert isinstance(merged_binding, BindingSet)
    # value_kind must come from the source slot (enum_value), not hardcoded entity
    for sv in merged_binding.values:
        assert sv.value_kind == "enum_value", (
            f"expected 'enum_value' from source binding, got {sv.value_kind!r}"
        )


# ---------------------------------------------------------------------------
# A merge creates a new spec id, so reconciliation cancels stale probes and launches one
# for the merged region. Untouched regions keep their task; fake probes use asyncio Tasks.
# ---------------------------------------------------------------------------

_PROBE_KEY = SlotKey(axis="what", property=_gr("constraintProp"), label="constraintProp")


def _fake_slot_options_event(
    spec_id: str,
) -> tuple[tuple[SlotOptionsEvent, ...], tuple[Warning, ...]]:
    return (SlotOptionsEvent(spec_id=spec_id, key=_PROBE_KEY, options=[], truncated=False),), ()


async def _probe_coro(spec_id: str) -> tuple[tuple[SlotOptionsEvent, ...], tuple[Warning, ...]]:
    return _fake_slot_options_event(spec_id)


def _make_relaunch(call_log: list[str], tag: str):
    """Return a closure that logs its target spec id and emits an event for it."""

    def _relaunch(
        spec_id: str, slot_options: list[SlotOptions], facet_probe: FacetProbe | None = None
    ) -> asyncio.Task:
        call_log.append(f"{tag}:{spec_id}")
        return asyncio.create_task(_probe_coro(spec_id))

    return _relaunch


def test_reconcile_after_collapse_relaunches_merged_region_against_its_own_spec_id():
    """A merged region gets a probe keyed to its new spec id.
    An untouched region keeps its task.
    """

    async def _body():
        call_log: list[str] = []

        spec_a = _spec(_PERSON_SHAPE, "ValA")
        spec_b = _spec(_PERSON_SHAPE, "ValB")
        spec_c = _spec(_ECON_SHAPE, "any", sv_dcid="sv_c")

        task_a = asyncio.create_task(_probe_coro(spec_a.spec_id))
        task_b = asyncio.create_task(_probe_coro(spec_b.spec_id))
        task_c = asyncio.create_task(_probe_coro(spec_c.spec_id))
        await asyncio.gather(task_a, task_b, task_c)

        r_a = dataclasses.replace(
            _region(spec_a, "var A", earliest_index=0),
            slot_options_task=task_a,
            slot_options_relaunch=_make_relaunch(call_log, "a"),
        )
        r_b = dataclasses.replace(
            _region(spec_b, "var B", earliest_index=1),
            slot_options_task=task_b,
            slot_options_relaunch=_make_relaunch(call_log, "b"),
        )
        r_c = dataclasses.replace(
            _region(spec_c, "var C", earliest_index=2),
            slot_options_task=task_c,
            slot_options_relaunch=_make_relaunch(call_log, "c"),
        )

        _response, effective = combine_regions(
            [r_a, r_b, r_c],
            query="var A and var B and var C",
            extra_warnings=[],
            start_ms=now_ms(),
            engine_build="test",
            max_candidates=10,
        )
        # Sanity: A/B actually collapsed into one merged region distinct from both.
        effective_spec_ids = {r.spec.spec_id for r in effective if r.status == "definite"}
        assert spec_a.spec_id not in effective_spec_ids
        assert spec_b.spec_id not in effective_spec_ids
        assert spec_c.spec_id in effective_spec_ids

        streaming_regions, stale_tasks = _reconcile_after_collapse([r_a, r_b, r_c], effective)

        assert set(stale_tasks) == {task_a, task_b}
        await asyncio.gather(*(_cancel_and_await(t) for t in stale_tasks), return_exceptions=True)

        survivor_c = next(r for r in streaming_regions if r.spec.spec_id == spec_c.spec_id)
        assert survivor_c.slot_options_task is task_c
        assert not any(entry.startswith("c:") for entry in call_log), (
            "an untouched region's probe must never be relaunched"
        )

        merged = next(
            r
            for r in streaming_regions
            if r.spec.spec_id not in {spec_a.spec_id, spec_b.spec_id, spec_c.spec_id}
        )
        assert merged.slot_options_task not in (None, task_a, task_b)
        events, warns = await merged.slot_options_task
        assert warns == ()
        assert len(events) == 1
        assert events[0].spec_id == merged.spec.spec_id, (
            "relaunched enrichment must be keyed to the MERGED spec's own spec_id, "
            "never a stale pre-merge one"
        )
        assert f"a:{merged.spec.spec_id}" in call_log or f"b:{merged.spec.spec_id}" in call_log

    asyncio.run(_body())


def test_stream_slot_options_never_yields_event_for_a_spec_id_absent_from_response():
    """Every streamed options event must name a spec in the final response."""

    async def _body():
        spec_a = _spec(_PERSON_SHAPE, "ValA")
        spec_b = _spec(_PERSON_SHAPE, "ValB")

        task_a = asyncio.create_task(_probe_coro(spec_a.spec_id))
        task_b = asyncio.create_task(_probe_coro(spec_b.spec_id))
        await asyncio.gather(task_a, task_b)

        call_log: list[str] = []
        r_a = dataclasses.replace(
            _region(spec_a, "var A", earliest_index=0),
            slot_options_task=task_a,
            slot_options_relaunch=_make_relaunch(call_log, "a"),
        )
        r_b = dataclasses.replace(
            _region(spec_b, "var B", earliest_index=1),
            slot_options_task=task_b,
            slot_options_relaunch=_make_relaunch(call_log, "b"),
        )

        response, effective = combine_regions(
            [r_a, r_b],
            query="var A and var B",
            extra_warnings=[],
            start_ms=now_ms(),
            engine_build="test",
            max_candidates=10,
        )
        streaming_regions, stale_tasks = _reconcile_after_collapse([r_a, r_b], effective)
        await asyncio.gather(*(_cancel_and_await(t) for t in stale_tasks), return_exceptions=True)

        response_spec_ids = {part.spec.spec_id for part in response.parts}

        warnings_out: list[Warning] = []
        events = [
            ev
            async for ev in _stream_slot_options(
                streaming_regions,
                deadline=time.monotonic() + 5,
                warnings_out=warnings_out,
                late_departures_out=[],
                incomplete_out=[],
                response=None,
                answered=(),
            )
        ]
        assert events, "expected the merged region's relaunched probe to yield an event"
        for ev in events:
            assert isinstance(ev, SlotOptionsEvent)
            assert ev.spec_id in response_spec_ids, (
                f"slot_options event spec_id {ev.spec_id!r} is absent from the "
                f"response's specs {response_spec_ids!r} -- undeliverable enrichment"
            )

    asyncio.run(_body())


def test_window_refusal_keeps_original_enrichment_tasks_joinable_to_visible_parts():
    """Different windows keep both specs visible, so their original probes remain valid."""

    async def _body():
        call_log: list[str] = []
        spec_a = build_spec(
            open_roles=(),
            measure_elected_properties=frozenset(),
            shape=_PERSON_SHAPE,
            slots=[_value_slot("constraintProp", "ValA")],
            stat_vars=[
                StatVar(
                    ref=_gr("sv_a"),
                    shape_id=_PERSON_SHAPE.shape_id,
                    slot_values=[],
                    constraint_properties=[],
                )
            ],
            entities=[],
            coverage=CoverageBare(has_data=True),
            requested_window=TimeWindow(start_year=2015, end_year=2020),
        )
        spec_b = build_spec(
            open_roles=(),
            measure_elected_properties=frozenset(),
            shape=_PERSON_SHAPE,
            slots=[_value_slot("constraintProp", "ValB")],
            stat_vars=[
                StatVar(
                    ref=_gr("sv_b"),
                    shape_id=_PERSON_SHAPE.shape_id,
                    slot_values=[],
                    constraint_properties=[],
                )
            ],
            entities=[],
            coverage=CoverageBare(has_data=True),
            requested_window=TimeWindow(start_year=2018, end_year=2020),
        )
        task_a = asyncio.create_task(_probe_coro(spec_a.spec_id))
        task_b = asyncio.create_task(_probe_coro(spec_b.spec_id))
        await asyncio.gather(task_a, task_b)
        r_a = dataclasses.replace(
            _region(spec_a, "var A", earliest_index=0),
            slot_options_task=task_a,
            slot_options_relaunch=_make_relaunch(call_log, "a"),
        )
        r_b = dataclasses.replace(
            _region(spec_b, "var B", earliest_index=1),
            slot_options_task=task_b,
            slot_options_relaunch=_make_relaunch(call_log, "b"),
        )

        response, effective = combine_regions(
            [r_a, r_b],
            query="var A and var B",
            extra_warnings=[],
            start_ms=now_ms(),
            engine_build="test",
            max_candidates=10,
        )
        streaming_regions, stale_tasks = _reconcile_after_collapse([r_a, r_b], effective)

        assert effective == [r_a, r_b]
        assert streaming_regions == [r_a, r_b]
        assert effective[0] is r_a
        assert effective[1] is r_b
        assert streaming_regions[0] is r_a
        assert streaming_regions[1] is r_b
        assert stale_tasks == []
        assert call_log == []

        response_spec_ids = {part.spec.spec_id for part in response.parts}
        warnings_out: list[Warning] = []
        events = [
            event
            async for event in _stream_slot_options(
                streaming_regions,
                deadline=time.monotonic() + 5,
                warnings_out=warnings_out,
                late_departures_out=[],
                incomplete_out=[],
                response=None,
                answered=(),
            )
        ]
        assert {event.spec_id for event in events} == response_spec_ids

    asyncio.run(_body())


def test_stale_slot_options_task_cancelled_before_it_runs():
    """Cancelling before the event loop steps a task prevents its coroutine body from running."""

    async def _body():
        call_log: list[str] = []

        async def _tracked_probe(spec_id: str):
            call_log.append(spec_id)
            return _fake_slot_options_event(spec_id)

        task = asyncio.create_task(_tracked_probe("never-should-run"))
        # No await in between task creation and cancellation: the loop has not
        # yet had a chance to schedule this task's first step.
        await _cancel_and_await(task)

        assert task.cancelled()
        assert call_log == [], (
            "a task cancelled before the loop ever scheduled it must never "
            "execute its body — the probe function must not have been called"
        )

    asyncio.run(_body())


def test_each_part_of_a_fold_gets_its_own_late_departures_over_sse():
    """Two parts of one shape fold into one merged Spec ("aid for health and education").
    Its streamed read used projected rates, so `done` carries a late list for each part,
    named by `part_index`. Each list is the part's shipped list plus `modelled_values`.
    The folded part's list lacks the dropped-filter departure that part 0 raised."""
    import json

    from search.engine.departures import warning_key
    from search.engine.events import DoneEvent, DoneTelemetry, ObservationsEvent, serialize_sse
    from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
    from search.heading.departure_fact import DepartureFact
    from search.models import (
        ObservationSeries,
        ObservationSeriesConversion,
        SeriesEntityRef,
        SeriesFacetRef,
        SeriesStatVarRef,
        SpecObservations,
    )

    dropped = Warning(code=QUERY_QUALIFIER_DROPPED, severity="info", message="malaria")
    dropped_fact = DepartureFact(code="measure_not_applied", binding_ref=None, asked="malaria")

    def _projected(spec_id: str) -> SpecObservations:
        conversion = ObservationSeriesConversion(
            recipe_id="recipe/Test",
            table_provenance=SeriesFacetRef(ref="provenance/TestRateTables"),
            exchange_rate_table_version="2026-01-01T00:00:00Z",
            fallback_count=0,
            projected_count=2,
        )
        return SpecObservations(
            series=[
                ObservationSeries(
                    stat_var=SeriesStatVarRef(ref=spec_id),
                    entities=[SeriesEntityRef(role="observationAbout", ref="country/KEN")],
                    points=[],
                    total_obs=0,
                    conversion=conversion,
                )
            ],
            total_obs=0,
            truncated=False,
        )

    async def _read(spec_id: str):
        return (ObservationsEvent(spec_id=spec_id, observations=_projected(spec_id)),), ()

    def _relaunch(spec_id: str) -> asyncio.Task:
        return asyncio.create_task(_read(spec_id))

    async def _body():
        spec_a = _spec(_PERSON_SHAPE, "ValA")
        spec_b = _spec(_PERSON_SHAPE, "ValB")
        regions = [
            dataclasses.replace(
                _region(spec_a, "health", earliest_index=0),
                warnings=(dropped,),
                departure_facts={warning_key(dropped): dropped_fact},
                observations_relaunch=_relaunch,
            ),
            dataclasses.replace(
                _region(spec_b, "education", earliest_index=1),
                observations_relaunch=_relaunch,
            ),
        ]
        response, effective = combine_regions(regions, **_assemble_kwargs(regions[0]))
        streaming, stale = _reconcile_after_collapse(regions, effective)
        await asyncio.gather(*(_cancel_and_await(t) for t in stale), return_exceptions=True)
        late: list = []
        async for _event in _stream_slot_options(
            streaming,
            deadline=time.monotonic() + 5,
            warnings_out=[],
            late_departures_out=late,
            incomplete_out=[],
            response=response,
            answered=regions,
        ):
            pass
        return response, late

    response, late = asyncio.run(_body())
    first, folded = response.parts
    assert first.spec.spec_id == folded.spec.spec_id, "the two parts must fold"
    assert [w.code for w in folded.warnings] == [BINDING_FOLDED]

    done = DoneEvent(
        elapsed_ms=1,
        degraded_leg=False,
        trace_id="t",
        telemetry=DoneTelemetry(),
        late_departures=late,
    )
    frame = json.loads(serialize_sse(done).split("data: ", 1)[1])
    modelled = {"code": "modelled_values", "text": "Some of these values are modelled estimates."}
    by_index = {entry["part_index"]: entry for entry in frame["late_departures"]}
    assert sorted(by_index) == [0, 1]
    for index, part in enumerate(response.parts):
        shipped = [d.model_dump(mode="json", exclude_none=True) for d in part.departures]
        assert by_index[index]["spec_id"] == part.spec.spec_id
        assert sorted(by_index[index]["departures"], key=lambda d: d["code"]) == sorted(
            [*shipped, modelled], key=lambda d: d["code"]
        )
    assert "measure_not_applied" in {d.code for d in first.departures}
    assert "measure_not_applied" not in {d["code"] for d in by_index[1]["departures"]}
