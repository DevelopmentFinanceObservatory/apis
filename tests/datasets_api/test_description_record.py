"""Tests for the description record builders, over versions the Datasets fixtures mint and
publish.

A mint record is built from a version's stored facts, its series figures and the code index;
a collapse's from its base version's record; a publish's from its lineage, its posted steps and
its frame. Each record round-trips through `describe_version` into a description the kit's
schema accepts.
"""

from __future__ import annotations

import copy
import datetime
import json
from collections.abc import Mapping
from dataclasses import replace
from functools import cache
from typing import Any

import pytest
from pydantic.experimental.missing_sentinel import MISSING

from code_index.shared import get_index, get_measure_index
from code_index.types import MeasureAxis, MeasureIndex, MeasurePartitionLabel
from dataset_store.snapshot import FrameLineage, ManifestFacts
from datasets_api.description_record import (
    CHOICE_RANKING,
    ROLE_VALUES_MAX,
    WARNING_DESTINATIONS,
    bounded_text,
    collapsed_record,
    method_status_build,
    mint_record,
    named_refs,
    publish_record,
    unit_from_cell,
    with_labels,
)
from datasets_api.version_descriptions import mint_title
from datasets_api.versions import VersionRecord
from description.compose import describe_version, version_title
from description.facts import (
    CaveatFact,
    CodeIndexStamp,
    DescriptionRecord,
    DimensionValue,
    InputVersion,
    LabelEvidence,
    PipelineStep,
    RoleCount,
    SourceEvidence,
    SourceUnitCount,
    Span,
    StatusCounts,
)
from description.steps import ComposedUnit, LeftOut, SimpleUnit, Step, VariableSide
from description.vocabulary import Vocabulary, build_vocabulary
from materialise.frame_build import build_frame
from materialise.statistics import (
    EntityFigure,
    RoleFigure,
    SeriesFigures,
    SeriesStatisticsSink,
    SourceUnitFigure,
    StatusFigures,
    compute_series_statistics_table,
)
from tests.description._kit import validate_description
from tests.description.test_accept_compose import _PROVIDERS as _WORKED_PROVIDERS
from tests.description.test_accept_compose import _WORKED_DIMENSIONS

from . import test_serve_groupby_collapse as family_fixture
from ._description_doubles import (
    DENOMINATOR,
    FAMILY_AXIS,
    GROUPBY_OPTIONS,
    NUMERATOR,
    OCV,
    OPS_ID,
    datasets_client,
    mint_family,
    publish_body,
    quotient_steps,
)
from ._version_facts import facts_from_body
from ._version_record_store import FakeVersionRecordDatasetStore

_COMPUTED_AT = datetime.datetime(2026, 10, 11, 9, 0, tzinfo=datetime.UTC)
_FIXTURE_STAMP = CodeIndexStamp(
    content_hash="sha256:fixture",
    exported_at="2026-09-15T00:00:00Z",
    graph_endpoint="https://fixture.example.org",
)
_TITLE = "Amount of development finance to Kenya by test axis"


def _figures(*, series: int = 2, roles: tuple[RoleFigure, ...] | None = None) -> SeriesFigures:
    """The family fixture's figures: one year, every series reaching it, no method classed.
    Its one recipient, which the definition binds, is held by both variables unless `roles`
    replaces it."""
    return SeriesFigures(
        series=series,
        span=("2020", "2020"),
        grain="year",
        latest_period="2020",
        series_reaching_latest=series,
        entities_by_role=roles
        if roles is not None
        else (
            RoleFigure(role="recipient", values=(EntityFigure(ref="country/KEN", variables=2),)),
        ),
        by_source_unit=(SourceUnitFigure(source="OECD_DAC2A", unit="USD", series=series),),
        status=StatusFigures(estimated=None, provisional=None),
    )


def _version(manifest: Mapping[str, Any], **facts: Any) -> VersionRecord:
    stored = facts_from_body(manifest)
    assert isinstance(stored, ManifestFacts)
    return VersionRecord(
        subject_id=manifest["definition"]["spec_id"],
        subject_kind="spec",
        content_version=manifest["content_version"],
        schema_version=1,
        facts=replace(stored, **facts),
    )


@cache
def _family_manifests() -> tuple[dict[str, Any], dict[str, Any]]:
    """The family fixture's raw mint and its groupby collapse, minted once."""
    client = datasets_client(FakeVersionRecordDatasetStore(), fixture="family")
    raw = mint_family(client, options={"bundle": False})
    collapsed = mint_family(client, options=GROUPBY_OPTIONS)
    return raw, collapsed


def _family_raw() -> dict[str, Any]:
    return copy.deepcopy(_family_manifests()[0])


def _mint(
    manifest: Mapping[str, Any], *, figures: SeriesFigures | None = None, **facts: Any
) -> DescriptionRecord:
    return mint_record(
        _version(manifest, **facts),
        figures=figures if figures is not None else _figures(),
        code_index=family_fixture._code_index(),
        title=_TITLE,
    )


def _with_trace(manifest: Mapping[str, Any], **members: Any) -> dict[str, Any]:
    return {**manifest["trace"], **members}


# Mint


def family_raw_bound(binding: Mapping[str, Any] | None) -> dict[str, Any]:
    """The family fixture's raw mint, its test axis bound by `binding` in place of the set of
    both its members when `binding` is given."""
    manifest = _family_raw()
    if binding is not None:
        (slot,) = manifest["definition"]["identity"]["slots"]
        slot["binding"] = dict(binding)
    return manifest


# The family fixture binds its test axis to a set; an unbound axis narrows to nothing.
STAMPED_BINDINGS = [
    pytest.param(None, 2, id="set-bound"),
    pytest.param({"state": "unbound"}, 1, id="unbound"),
]


@pytest.mark.parametrize(("binding", "stamp"), STAMPED_BINDINGS)
def test_a_record_is_stamped_the_lowest_schema_version_that_describes_it(
    binding: Mapping[str, Any] | None, stamp: int
) -> None:
    """Version 1 lacks `within_set`, so only a record narrowing a dimension to a set is
    stamped 2, and an image of version 1 reads every other record. A collapse is stamped by
    what it holds, not by its base, which a read of a version 1 row lifts to version 2."""
    raw = _mint(family_raw_bound(binding))
    collapsed = _collapse(raw.model_copy(update={"schema_version": 2}))

    assert (raw.schema_version, collapsed.schema_version) == (stamp, stamp)


def test_a_mint_record_holds_the_versions_variables_series_and_evidence() -> None:
    manifest = _family_raw()

    record = _mint(manifest)

    assert (record.id, record.content_version, record.basis, record.schema_version) == (
        manifest["definition"]["spec_id"],
        manifest["content_version"],
        "mint",
        2,
    )
    assert record.computed is None
    assert record.index_stamp == _FIXTURE_STAMP
    variables = record.variables
    assert variables is not None
    assert [(item.property, item.value) for item in variables.fixed] == [
        ("recipient", "country/KEN")
    ]
    assert (variables.admitted_at_mint, variables.admitted_from, variables.with_data) == (
        2,
        "family_member_count",
        2,
    )
    assert variables.dimensions is not None
    (axis,) = variables.dimensions
    assert (axis.property, axis.within, axis.partial, axis.undeclared_variables) == (
        FAMILY_AXIS,
        None,
        False,
        0,
    )
    assert axis.values == [
        DimensionValue(value="TestAxisA", variables=1),
        DimensionValue(value="TestAxisB", variables=1),
    ]
    series = record.series
    assert (series.series, series.rows, series.periods) == (manifest["series_count"], 2, 1)
    assert (series.span, series.grain, series.latest_period, series.series_reaching_latest) == (
        Span(start="2020", end="2020"),
        "year",
        "2020",
        2,
    )
    assert series.entities_by_role == [RoleCount(role="recipient", entities=1)]
    assert series.by_source_unit == [
        SourceUnitCount(source="OECD_DAC2A", unit=SimpleUnit(ref="USD"), series=2)
    ]
    assert series.status == StatusCounts(estimated=None, provisional=None)
    # The fixture's provenance is the sole candidate, so the mint made no choice.
    assert (record.steps, record.caveats, record.late_fields) == ([], [], [])
    evidence = record.evidence
    assert evidence.prepared == datetime.date.fromisoformat(manifest["prepared_at"][:10])
    assert evidence.title == _TITLE
    assert evidence.sources == [
        SourceEvidence(
            ref="OECD_DAC2A", name="OECD_DAC2A", licence=None, downloaded=None, pipeline_steps=None
        )
    ]
    assert (evidence.source_release, evidence.membership, evidence.labels) == (None, None, [])
    assert evidence.builds.origin_catalogue == "fake-catalogue"
    assert evidence.builds.code_index == _FIXTURE_STAMP
    assert evidence.builds.method_status == method_status_build()
    assert method_status_build().startswith("sha256:")


def test_a_mint_record_of_variables_the_index_does_not_hold_has_no_dimensions() -> None:
    manifest = _family_raw()
    trace = _with_trace(
        manifest,
        resolved_stat_vars=[
            {"ref": "dcid:Count_Person", "roles": ["recipient"], "observation_count": 3}
        ],
    )

    record = _mint(manifest, trace=trace)

    assert record.variables is not None
    assert record.variables.dimensions is None
    assert record.index_stamp is None


@pytest.mark.parametrize(
    ("counts", "member_count", "expected"),
    [
        pytest.param(
            [1, 1], 2, (2, "family_member_count", 2), id="control-every-variable-holds-data"
        ),
        pytest.param([3, 0], 2, (2, "family_member_count", 1), id="a-zero-count-is-known"),
        pytest.param([None, None], 2, (2, "family_member_count", None), id="no-counts-is-unknown"),
        pytest.param([1, 1], None, (2, "resolved_variables", 2), id="no-member-count"),
    ],
)
def test_with_data_is_known_only_where_the_trace_counts_observations(
    counts: list[int | None], member_count: int | None, expected: tuple[object, ...]
) -> None:
    manifest = _family_raw()
    entries = []
    for entry, count in zip(manifest["trace"]["resolved_stat_vars"], counts, strict=True):
        entry = {key: value for key, value in entry.items() if key != "observation_count"}
        if count is not None:
            entry["observation_count"] = count
        entries.append(entry)
    trace = _with_trace(manifest, resolved_stat_vars=entries, family_member_count=member_count)

    variables = _mint(manifest, trace=trace).variables

    assert variables is not None
    assert (variables.admitted_at_mint, variables.admitted_from, variables.with_data) == expected


def test_a_dimension_is_partial_when_a_variable_with_data_is_missing_from_the_index() -> None:
    manifest = _family_raw()
    trace = _with_trace(
        manifest,
        resolved_stat_vars=[
            *manifest["trace"]["resolved_stat_vars"],
            {"ref": "dcid:FamilyMemberGone", "roles": ["recipient"], "observation_count": 1},
        ],
    )

    variables = _mint(manifest, trace=trace).variables

    assert variables is not None
    assert variables.dimensions is not None
    assert [dimension.partial for dimension in variables.dimensions] == [True]
    assert variables.with_data == 3


def _role(role: str, *pairs: tuple[str, int]) -> RoleFigure:
    return RoleFigure(
        role=role,
        values=tuple(EntityFigure(ref=ref, variables=variables) for ref, variables in pairs),
    )


# The frame's roles, in the order its statistics fold them: providers before recipients.
_PROVIDERS = _role("financingProvider", ("dcid:dc/o/p1", 2), ("dcid:dc/o/p2", 1))
_RECIPIENTS = _role("financingRecipient", ("dcid:country/NGA", 2), ("dcid:country/KEN", 1))


def _with_entities(manifest: Mapping[str, Any], entities: Mapping[str, list[str]]):
    definition = manifest["definition"]
    return {**definition, "identity": {**definition["identity"], "entities": dict(entities)}}


@pytest.mark.parametrize(
    ("entities", "fixed", "dimensions"),
    [
        pytest.param(
            {"destination": ["country/NGA"]},
            [("financingRecipient", "country/NGA")],
            [
                ("financingProvider", [("dc/o/p1", 2), ("dc/o/p2", 1)]),
                (FAMILY_AXIS, [("TestAxisA", 1), ("TestAxisB", 1)]),
            ],
            id="control-a-role-bound-to-one-entity-is-fixed",
        ),
        pytest.param(
            {"destination": ["country/NGA", "country/KEN"], "origin": ["dc/o/p1", "dc/o/p2"]},
            [],
            [
                ("financingRecipient", [("country/KEN", 1), ("country/NGA", 2)]),
                ("financingProvider", [("dc/o/p1", 2), ("dc/o/p2", 1)]),
                (FAMILY_AXIS, [("TestAxisA", 1), ("TestAxisB", 1)]),
            ],
            id="roles-bound-to-several-entities-keep-the-definitions-order",
        ),
        pytest.param(
            {},
            [],
            [
                ("financingProvider", [("dc/o/p1", 2), ("dc/o/p2", 1)]),
                ("financingRecipient", [("country/KEN", 1), ("country/NGA", 2)]),
                (FAMILY_AXIS, [("TestAxisA", 1), ("TestAxisB", 1)]),
            ],
            id="unbound-roles-follow-the-frame",
        ),
    ],
)
def test_open_roles_are_dimensions_counting_the_variables_holding_each_entity(
    entities: dict[str, list[str]],
    fixed: list[tuple[str, str]],
    dimensions: list[tuple[str, list[tuple[str, int]]]],
) -> None:
    manifest = _family_raw()

    variables = _mint(
        manifest,
        figures=_figures(roles=(_PROVIDERS, _RECIPIENTS)),
        definition=_with_entities(manifest, entities),
    ).variables

    assert variables is not None
    assert [(item.property, item.value) for item in variables.fixed] == fixed
    assert variables.dimensions is not None
    assert [
        (dimension.property, [(value.value, value.variables) for value in dimension.values])
        for dimension in variables.dimensions
    ] == dimensions
    roles = [dim for dim in variables.dimensions if dim.property.startswith("financing")]
    assert [(dim.within, dim.undeclared_variables, dim.partial) for dim in roles] == [
        (None, 0, False)
    ] * len(roles)


@pytest.mark.parametrize(
    ("held", "partial"),
    [
        pytest.param(ROLE_VALUES_MAX, False, id="control-at-the-bound-every-value-is-kept"),
        pytest.param(ROLE_VALUES_MAX + 1, True, id="past-the-bound-the-fewest-held-are-dropped"),
    ],
)
def test_a_role_past_the_bound_keeps_the_values_most_variables_hold_and_is_partial(
    held: int, partial: bool
) -> None:
    """Every provider but the last is held by three variables; the last by one."""
    manifest = _family_raw()
    providers = [(f"dc/o/p{n:04d}", 3) for n in range(held - 1)] + [("dc/o/last", 1)]

    variables = _mint(
        manifest,
        figures=_figures(roles=(_role("financingProvider", *providers),)),
        definition=_with_entities(manifest, {}),
    ).variables

    assert variables is not None
    assert variables.dimensions is not None
    (provider,) = [dim for dim in variables.dimensions if dim.property == "financingProvider"]
    assert provider.partial is partial
    kept = providers[:ROLE_VALUES_MAX] if partial else providers
    assert sorted((value.value, value.variables) for value in provider.values) == sorted(kept)


_CHOICES = [
    {
        "axis": "provenance",
        "candidates": [
            {"ref": "OECD_DAC2A", "places": 3, "years": 10, "observations": 30},
            {"ref": "dcid:IMF_WEO", "places": 3, "years": 10, "observations": 12},
            {"ref": None, "places": 1, "years": 2, "observations": 2},
        ],
    }
]


_CONTINUATIONS = [{"axis": "measurementMethod", "method": "Estimated", "periods": ["2025", "2030"]}]


@pytest.mark.parametrize(
    ("choices", "continuations", "others"),
    [
        pytest.param(None, None, [], id="trace-before-choices-names-no-others"),
        pytest.param(_CHOICES, None, [{"ref": "IMF_WEO"}], id="trace-choices-name-the-others"),
        pytest.param(
            _CHOICES,
            _CONTINUATIONS,
            [{"ref": "IMF_WEO"}],
            id="trace-continuation-follows-the-choice",
        ),
    ],
)
def test_a_source_chosen_over_disagreeing_candidates_is_a_step_naming_the_ranking(
    choices: list[dict[str, Any]] | None,
    continuations: list[dict[str, Any]] | None,
    others: list[dict[str, str]],
) -> None:
    """The others are the trace's non-elected candidates in rank order, by bare dcid; a
    candidate carrying no value on the axis names nothing. Each continuation of the chosen
    series by the same source's estimates is a step after the choice, naming the method and
    the periods it supplies; a trace without one writes none."""
    manifest = _family_raw()
    facets = copy.deepcopy(manifest["trace"]["facets"])
    facets[0]["selection_reason"] = "most_observations_over_disagreement"
    trace = _with_trace(manifest, facets=facets)
    if choices is not None:
        trace["choices"] = choices
    if continuations is not None:
        trace["continuations"] = continuations

    record = _mint(manifest, trace=trace)

    assert record.steps is not None
    continued = [
        {
            "id": "m2",
            "op": "continue_series",
            "policy": "continue_with_source_estimates",
            "params": {"method": "Estimated", "periods": ["2025", "2030"]},
        }
    ]
    assert [step.model_dump(mode="json") for step in record.steps] == [
        {
            "id": "m1",
            "op": "choose_source",
            "policy": "facet_default_by_coverage",
            "choice": {
                "chosen": {"ref": "OECD_DAC2A"},
                "others": others,
                "ranking": list(CHOICE_RANKING),
            },
        },
        *(continued if continuations is not None else []),
    ]


def _converted(
    manifest: dict[str, Any], *, projected: int, left_out: list[dict[str, object]] | None = None
) -> dict[str, Any]:
    """`manifest` as converted into constant 2023 US dollars from euros, leaving out what
    `left_out` counts."""
    manifest["definition"]["identity"]["facets"] = {
        "unit": {
            "state": "value",
            "request": {"currency": "USD", "prices": "constant", "base_year": 2023},
        }
    }
    manifest["trace"]["conversion"] = [
        {
            "recipe": "ConversionRecipeIMF",
            "rate_table_version": "rates-1",
            "deflator_table_version": "deflators-1",
            "members": [
                {"stat_var": family_fixture._MEMBER_A, "native_unit": "CurrencyEUR"},
                {"stat_var": family_fixture._MEMBER_B, "native_unit": "CurrencyEUR"},
            ],
            "fallback": {"count": 0, "examples": [], "truncated": False},
            "projected": {"count": projected, "examples": [], "truncated": False},
        }
    ]
    if left_out is not None:
        manifest["trace"]["conversion"][0]["left_out"] = left_out
    return manifest


_DEFLATOR_LEFT_OUT = [
    {"reason": "deflator_missing", "count": 2, "of": "entities"},
    {"reason": "deflator_missing", "count": 5, "of": "entity_periods"},
]


@pytest.mark.parametrize(
    ("projected", "left_out", "codes", "caveats"),
    [
        pytest.param(0, None, [], [], id="control-no-projection-no-gap"),
        pytest.param(
            4,
            None,
            [],
            [CaveatFact(kind="estimated", step="m1", count=4)],
            id="projected-estimated",
        ),
        pytest.param(
            0,
            None,
            ["currency_unmapped_omitted"],
            [CaveatFact(kind="coverage_gap", step="m1")],
            id="trace-before-counts-currency-unmapped-gap",
        ),
        pytest.param(
            0,
            None,
            ["deflator_missing_omitted"],
            [CaveatFact(kind="coverage_gap", step="m1")],
            id="trace-before-counts-deflator-missing-gap",
        ),
        pytest.param(
            0,
            _DEFLATOR_LEFT_OUT,
            ["deflator_missing_omitted"],
            [CaveatFact(kind="coverage_gap", step="m1")],
            id="counted-deflator-missing-gap",
        ),
        pytest.param(
            0,
            [{"reason": "currency_unmapped", "count": 3, "of": "entities"}],
            [],
            [CaveatFact(kind="coverage_gap", step="m1")],
            id="counted-gap-without-its-warning",
        ),
    ],
)
def test_a_conversion_is_a_step_and_its_estimates_and_gaps_are_caveats(
    projected: int,
    left_out: list[dict[str, object]] | None,
    codes: list[str],
    caveats: list[CaveatFact],
) -> None:
    """A conversion step counts what the trace says it left out, by reason; a trace from
    before conversions counted it leaves only the gap its warnings name."""
    manifest = _converted(_family_raw(), projected=projected, left_out=left_out)
    warnings = [*manifest["warnings"], *({"code": code, "severity": "warn"} for code in codes)]

    record = _mint(manifest, warnings=warnings)

    step = Step(
        id="m1",
        op="convert",
        policy="convert_by_rate_and_deflator",
        converted_from=SimpleUnit(ref="CurrencyEUR"),
        unit=SimpleUnit(currency="USD", prices="constant", base_year=2023),
    )
    if left_out is not None:
        step = step.model_copy(
            update={"left_out": [LeftOut.model_validate(item) for item in left_out]}
        )
    assert record.steps == [step]
    assert record.caveats == caveats


_GAP_PLACES = [f"dcid:country/C{index:02d}" for index in range(51)]


@pytest.mark.parametrize(
    ("gaps", "caveats"),
    [
        pytest.param([], [], id="control-no-gap"),
        pytest.param(
            [
                {"role": "recipient", "reason": "no_national_currency", "entities": ["Earth"]},
                {
                    "role": "recipient",
                    "reason": "no_facet_in_national_currency",
                    "entities": ["country/FRA", "country/KEN"],
                },
            ],
            [
                CaveatFact(kind="coverage_gap", entities=["Earth"]),
                CaveatFact(kind="coverage_gap", entities=["country/FRA", "country/KEN"]),
            ],
            id="one-caveat-per-reason",
        ),
        pytest.param(
            [{"role": "recipient", "reason": "no_national_currency", "entities": _GAP_PLACES}],
            [
                CaveatFact(
                    kind="coverage_gap",
                    entities=[place.removeprefix("dcid:") for place in _GAP_PLACES],
                )
            ],
            id="names-every-place",
        ),
    ],
)
def test_each_national_currency_gap_is_a_coverage_gap_naming_its_places(
    gaps: list[dict[str, object]], caveats: list[CaveatFact]
) -> None:
    manifest = _family_raw()

    record = _mint(manifest, trace=_with_trace(manifest, coverage_gaps=gaps))

    assert record.caveats == caveats


@pytest.mark.parametrize(
    ("gaps", "caveats"),
    [
        pytest.param([], [], id="control-no-gap"),
        pytest.param(
            [
                {
                    "code": "withheld_by_registry",
                    "variable": "dcid:GrantA",
                    "entities": ["country/NGA"],
                    "provenances": ["dc/base/UN_SDG"],
                }
            ],
            [CaveatFact(kind="coverage_gap", series=["GrantA"], entities=["country/NGA"])],
            id="withheld-names-variable-and-places",
        ),
        pytest.param(
            [{"code": "un_system_unavailable", "entities": ["country/GRC", "country/SVN"]}],
            [CaveatFact(kind="coverage_gap", entities=["country/GRC", "country/SVN"])],
            id="un-unread-names-places",
        ),
        pytest.param(
            [{"code": "un_system_unavailable", "entities": _GAP_PLACES}],
            [
                CaveatFact(
                    kind="coverage_gap",
                    entities=[place.removeprefix("dcid:") for place in _GAP_PLACES],
                )
            ],
            id="un-unread-names-every-place",
        ),
    ],
)
def test_places_the_origin_reads_left_without_data_are_coverage_gaps(
    gaps: list[dict[str, object]], caveats: list[CaveatFact]
) -> None:
    manifest = _family_raw()

    record = _mint(manifest, trace=_with_trace(manifest, origin_gaps=gaps))

    assert record.caveats == caveats


def test_a_mint_records_the_documentation_it_read_and_nothing_for_a_source_it_read_none_for() -> (
    None
):
    manifest = _family_raw()
    citation = {
        "text": manifest["citation"]["text"],
        "sources": [
            {
                "ref": "OECD_DAC2A",
                "label": "OECD DAC2a",
                "license": "OECD Terms and Conditions",
                "documentation": {
                    "last_update": "2026-10-08",
                    "transformation_log": [
                        {"transformation": "Filter", "description": "Keeps bilateral rows."}
                    ],
                },
            },
            {"ref": "OECD_DAC2A", "label": "repeated at another processing level"},
            {"ref": "OECD_CRS", "label": "OECD_CRS", "documentation": {"last_update": None}},
        ],
    }

    sources = _mint(manifest, citation=citation).evidence.sources

    assert sources == [
        SourceEvidence(
            ref="OECD_DAC2A",
            name="OECD DAC2a",
            licence="OECD Terms and Conditions",
            downloaded=datetime.date(2026, 10, 8),
            pipeline_steps=[PipelineStep(title="Filter", description="Keeps bilateral rows.")],
        ),
        SourceEvidence(
            ref="OECD_CRS", name="OECD_CRS", licence=None, downloaded=None, pipeline_steps=[]
        ),
    ]


def test_a_mint_that_found_its_version_recorded_cites_and_dates_as_the_stored_row() -> None:
    manifest = _family_raw()
    stored = datetime.datetime(2026, 1, 2, 6, 0, tzinfo=datetime.UTC)

    record = _mint(manifest, prepared_at=stored)

    assert record.evidence.prepared == datetime.date(2026, 1, 2)


# Collapse


def _collapse(base: DescriptionRecord) -> DescriptionRecord:
    return collapsed_record(base, _version(_family_manifests()[1]), figures=_figures(series=1))


def test_a_collapsed_record_keeps_its_bases_facts_and_adds_one_step_naming_what_was_summed() -> (
    None
):
    """The groupby sums the test axis away. The definition binds it to the set of A and B, so
    the collapse is that set's total: it keeps the axis narrowed to the set, varying over no
    value, and only the labels of what it still names."""
    read = datetime.date(2026, 10, 11)
    base = with_labels(
        _mint(_family_raw()), {"TestAxisA": "Axis value A", "country/KEN": "Kenya"}, read=read
    )
    collapsed_manifest = _family_manifests()[1]

    record = _collapse(base)

    assert record.content_version == collapsed_manifest["content_version"]
    assert record.content_version != base.content_version
    assert base.variables is not None
    assert base.variables.dimensions
    (axis,) = base.variables.dimensions
    assert axis.within_set == ["TestAxisA", "TestAxisB"]
    assert (record.id, record.basis, record.variables, record.caveats) == (
        base.id,
        base.basis,
        base.variables.model_copy(
            update={
                "dimensions": [
                    axis.model_copy(
                        update={"values": [], "undeclared_variables": 0, "partial": False}
                    )
                ]
            }
        ),
        base.caveats,
    )
    assert record.evidence == base.evidence.model_copy(
        update={
            "labels": [
                LabelEvidence(ref="TestAxisA", label="Axis value A", read=read),
                LabelEvidence(ref="country/KEN", label="Kenya", read=read),
            ]
        }
    )
    assert (record.series.series, record.series.rows) == (1, 1)
    assert record.steps is not None
    assert [step.model_dump(mode="json") for step in record.steps] == [
        {
            "id": "m1",
            "op": "sum_variables",
            "policy": "sum_family_members",
            "summed": {
                "count": 2,
                "variables": [
                    {"ref": family_fixture._MEMBER_A},
                    {"ref": family_fixture._MEMBER_B},
                ],
            },
        }
    ]
    assert base.steps == []


_RECIPIENT = _role("recipient", ("country/KEN", 2))
_COLLAPSED_RECIPIENT = _role("recipient", ("country/KEN", 1))
_COLLAPSED_PROVIDERS = _role("financingProvider", ("dcid:dc/o/p1", 1), ("dcid:dc/o/p2", 1))


_UNNARROWED: dict[str, Any] = {"within": None, "within_set": MISSING}
_NARROWED_TO_ALL: dict[str, Any] = {"within": "TestAxisAll", "within_set": MISSING}


@pytest.mark.parametrize(
    ("narrowing", "kept_axis", "dimensions", "fixed"),
    [
        pytest.param(
            _UNNARROWED,
            (),
            [("financingProvider", None, [("dc/o/p1", 1), ("dc/o/p2", 1)])],
            [],
            id="axis-summed-away",
        ),
        pytest.param(
            {},
            (),
            [
                ("financingProvider", None, [("dc/o/p1", 1), ("dc/o/p2", 1)]),
                ("testAxis", None, []),
            ],
            [],
            id="set-bound-axis-summed-to-the-sets-total",
        ),
        pytest.param(
            _NARROWED_TO_ALL,
            (),
            [("financingProvider", None, [("dc/o/p1", 1), ("dc/o/p2", 1)])],
            [("testAxis", "TestAxisAll")],
            id="narrowed-axis-summed-to-its-code",
        ),
        pytest.param(
            _NARROWED_TO_ALL,
            (_role("stat_var_testAxis", ("dcid:TestAxisA", 1)),),
            [
                ("financingProvider", None, [("dc/o/p1", 1), ("dc/o/p2", 1)]),
                ("testAxis", "TestAxisAll", [("TestAxisA", 1)]),
            ],
            [],
            id="axis-kept",
        ),
    ],
)
def test_a_collapse_keeps_the_dimensions_its_frame_holds_with_its_own_values(
    narrowing: dict[str, Any],
    kept_axis: tuple[RoleFigure, ...],
    dimensions: list[tuple[str, str | None, list[tuple[str, int]]]],
    fixed: list[tuple[str, str]],
) -> None:
    """Each role stays a column of the collapse, a kept code axis is its `stat_var_` column,
    and every other axis is summed away; a kept dimension's values are the collapse's. An
    axis summed away is dropped, becomes a fixed value when narrowed to one code, and stays
    narrowed to the set, over no value, when the definition binds a set (the fixture's)."""
    base = _mint(_family_raw(), figures=_figures(roles=(_RECIPIENT, _PROVIDERS)))
    assert base.variables is not None
    assert base.variables.dimensions is not None
    base = base.model_copy(
        update={
            "variables": base.variables.model_copy(
                update={
                    "dimensions": [
                        dimension.model_copy(update=narrowing)
                        if dimension.property == FAMILY_AXIS
                        else dimension
                        for dimension in base.variables.dimensions
                    ]
                }
            )
        }
    )

    record = collapsed_record(
        base,
        _version(_family_manifests()[1]),
        figures=_figures(series=1, roles=(_COLLAPSED_RECIPIENT, _COLLAPSED_PROVIDERS, *kept_axis)),
    )

    assert record.variables is not None
    assert record.variables.dimensions is not None
    assert [
        (
            dimension.property,
            dimension.within,
            [(value.value, value.variables) for value in dimension.values or ()],
        )
        for dimension in record.variables.dimensions
    ] == dimensions
    assert [(item.property, item.value) for item in record.variables.fixed] == [
        (item.property, item.value) for item in base.variables.fixed
    ] + fixed


@pytest.mark.parametrize("collapsed", [False, True], ids=["minted", "summed-collapse"])
def test_a_set_binding_names_its_codes_in_the_version_title(collapsed: bool) -> None:
    """The fixture's definition binds the test axis to the set of A and B, so its title names
    both codes, and so does its multi-contributor collapse, the set's total, which a computed
    version names an input by."""
    base = with_labels(
        _mint(_family_raw()),
        {"TestAxisA": "Axis value A", "TestAxisB": "Axis value B", "country/KEN": "Kenya"},
        read=datetime.date(2026, 10, 11),
    )
    record = _collapse(base) if collapsed else base
    vocabulary = build_vocabulary(
        index=family_fixture._code_index(),
        measure_index=MeasureIndex(
            axes=(
                MeasureAxis(axis="populationType", kind="shape", role="free", prefer=()),
                MeasureAxis(axis="measuredProperty", kind="shape", role="basis", prefer=()),
            ),
            values=(),
            partition_labels=(
                MeasurePartitionLabel(
                    axis_values=(
                        ("measuredProperty", "amount"),
                        ("populationType", "DevelopmentFinance"),
                    ),
                    label="development finance",
                ),
            ),
        ),
        labels={},
        sources={},
        graph_generation="gen-1",
        catalogue_build="fake-catalogue",
    )

    assert version_title(record, vocabulary) == (
        "Development finance to Kenya for Axis value A and Axis value B"
    )


_A, _B = family_fixture._MEMBER_A, family_fixture._MEMBER_B
_NO_DATA_ANYWHERE = CaveatFact(kind="coverage_gap", entities=["country/NGA"])
# Stands for the collapse's one variable, its subject id, in an expected caveat.
_SUM = "the-sum"


@pytest.mark.parametrize(
    ("caveats", "collapsed"),
    [
        pytest.param(
            [
                CaveatFact(kind="coverage_gap", series=[_A], entities=["country/KEN"]),
                CaveatFact(kind="coverage_gap", series=[_B], entities=["country/UGA"]),
                _NO_DATA_ANYWHERE,
            ],
            [_NO_DATA_ANYWHERE],
            id="each-place-missing-from-one-member-holds-a-sum",
        ),
        pytest.param(
            [
                CaveatFact(
                    kind="coverage_gap",
                    series=[_A],
                    entities=["country/UGA", "country/KEN", "country/NGA"],
                    count=60,
                ),
                _NO_DATA_ANYWHERE,
                CaveatFact(
                    kind="coverage_gap", series=[_B], entities=["country/KEN", "country/UGA"]
                ),
            ],
            [
                CaveatFact(
                    kind="coverage_gap", series=[_SUM], entities=["country/UGA", "country/KEN"]
                ),
                _NO_DATA_ANYWHERE,
            ],
            id="places-missing-from-every-member-without-the-members-count",
        ),
        pytest.param(
            [CaveatFact(kind="estimated", series=[_A], periods=["2020"])],
            [CaveatFact(kind="estimated", series=[_SUM], periods=["2020"])],
            id="a-member-estimate-holds-for-the-sum",
        ),
    ],
)
def test_a_collapse_caveat_holds_for_the_sum_of_its_members(
    caveats: list[CaveatFact], collapsed: list[CaveatFact]
) -> None:
    """A gap at a place only some summed members lack is no gap in the sum, which holds the
    other members' values there; a place every member lacks is."""
    base = _mint(_family_raw()).model_copy(update={"caveats": caveats})

    record = _collapse(base)

    assert record.caveats == [
        caveat.model_copy(update={"series": [base.id]}) if caveat.series == [_SUM] else caveat
        for caveat in collapsed
    ]


def test_a_collapse_of_a_base_whose_steps_are_unknown_holds_its_sum_after_them() -> None:
    """The base's steps stay unknown and the sum is known to follow them, so the description
    serves the sum with the version's own steps as a gap, and it and the manifest both cite
    the summed figures as processed by the DFO."""
    base = _mint(_family_raw()).model_copy(update={"steps": None})

    record = _collapse(base)
    served = _described(record)

    assert record.steps is None
    assert [step.op for step in record.known_steps] == ["sum_variables"]
    assert DescriptionRecord.model_validate(record.model_dump(mode="json")) == record
    assert [step["op"] for step in served["steps"]] == ["sum_variables"]
    assert "steps.version" in served["gaps"]
    assert "processed by" in served["cite"]["text"]
    assert "processed by" in _family_manifests()[1]["citation"]["text"]


# Publish


def _publish(steps: list[dict[str, object]] | None) -> DescriptionRecord:
    body = publish_body(steps=steps)
    version = VersionRecord(
        subject_id=OPS_ID,
        subject_kind="ops",
        content_version=OCV,
        schema_version=1,
        facts=FrameLineage(
            input_content_versions=body["input_content_versions"],
            composed_stat_vars=[
                {
                    "stat_var": "dcid:Exemplar",
                    "relation": "quotient",
                    "first": [f"dcid:{NUMERATOR[0]}"],
                    "second": ["dcid:GNI"],
                }
            ],
        ),
    )
    return publish_record(
        version,
        steps=None if steps is None else [Step.model_validate(step) for step in steps],
        frame=build_frame(role_keys=body["role_keys"], rows=body["rows"]),
        computed_at=_COMPUTED_AT,
    )


def test_a_publish_record_names_each_retrieved_input_by_its_step() -> None:
    steps = quotient_steps()

    record = _publish(steps)

    assert (record.id, record.content_version, record.basis, record.variables) == (
        OPS_ID,
        OCV,
        "publish",
        None,
    )
    assert record.steps is not None
    assert [step.model_dump(mode="json") for step in record.steps] == steps
    assert record.computed is not None
    assert record.computed.inputs == [
        InputVersion(
            id=NUMERATOR[0], content_version=NUMERATOR[1], label="Gross ODA disbursements"
        ),
        InputVersion(
            id=DENOMINATOR[0], content_version=DENOMINATOR[1], label="Gross national income"
        ),
    ]
    assert record.computed.composed is not None
    assert [(item.variable, item.first, item.second) for item in record.computed.composed] == [
        ("Exemplar", [NUMERATOR[0]], ["GNI"])
    ]
    assert record.caveats == [CaveatFact(kind="coverage_gap", step="s3")]
    assert (record.series.series, record.series.rows, record.series.periods) == (1, 1, 1)
    assert record.series.entities_by_role == [RoleCount(role="recipient", entities=1)]
    assert record.evidence.prepared == _COMPUTED_AT.date()
    assert record.evidence.builds.method_status == method_status_build()


def test_a_publish_without_steps_keeps_each_input_without_an_id() -> None:
    """Its inputs are the posted lineage with unknown ids, and what its steps would state, the
    caveats and label evidence, is unknown."""
    record = _publish(None)

    assert record.computed is not None
    assert record.computed.inputs == [
        InputVersion(id=None, content_version=NUMERATOR[1]),
        InputVersion(id=None, content_version=DENOMINATOR[1]),
    ]
    assert (record.steps, record.caveats, record.evidence.labels) == (None, None, None)


def test_a_publish_whose_steps_retrieve_nothing_has_no_inputs() -> None:
    """The inputs are the versions the `retrieve` steps name, never the posted lineage."""
    record = _publish([])

    assert record.computed is not None
    assert (record.computed.inputs, record.steps, record.caveats) == ([], [], [])


# Warning destinations

_WARNING_ROWS = [
    ("revision_unavailable", "gap", "freshness.source_release"),
    ("processing_level_undeclared", "diagnostics", "diagnostics"),
    ("origin_catalogue_stale", "diagnostics", "diagnostics"),
    ("stat_vars_summed", "step", "steps.summed"),
    ("facet_default_selected", "step", "steps.choice"),
    ("deflator_projected", "caveat", "caveats.estimated"),
    ("currency_unmapped_omitted", "step", "steps.left_out"),
    ("deflator_missing_omitted", "step", "steps.left_out"),
    ("withheld_by_registry", "caveat", "caveats.coverage_gap"),
    ("un_system_unavailable", "caveat", "caveats.coverage_gap"),
    ("coverage_below_scope", "warning", "warnings"),
    ("empty_result", "warning", "warnings"),
    ("enrichment_unavailable", "warning", "warnings"),
    ("observation_attributes_in_notes", "warning", "warnings"),
    ("members_deduplicated", "retired", None),
]


@pytest.mark.parametrize(("code", "destination", "member"), _WARNING_ROWS)
def test_each_datasets_warning_code_goes_to_its_destination(
    code: str, destination: str, member: str | None
) -> None:
    assert WARNING_DESTINATIONS[code] == (destination, member)


def test_the_destinations_cover_the_fifteen_datasets_warning_codes() -> None:
    assert set(WARNING_DESTINATIONS) == {code for code, _, _ in _WARNING_ROWS}


# Labels


def test_labels_are_kept_for_the_refs_the_facts_name_on_the_day_they_were_read() -> None:
    record = _mint(_family_raw())
    read = datetime.date(2026, 10, 11)

    labelled = with_labels(
        record,
        {
            "TestAxisA": "Axis\x00 value A",
            "TestAxisB": "\x00",
            "country/KEN": "Kenya",
            "country/NGA": "Nigeria",
        },
        read=read,
    )

    assert named_refs(record) == ["TestAxisA", "TestAxisB", "country/KEN"]
    assert labelled.evidence.labels == [
        LabelEvidence(ref="TestAxisA", label="Axis value A", read=read),
        LabelEvidence(ref="country/KEN", label="Kenya", read=read),
    ]
    again = with_labels(labelled, {"country/KEN": "Republic of Kenya"}, read=read)
    assert again == labelled


# The definition of gross ODA disbursements to Kenya by donor, as Search mints it.
_ODA_VARIABLE = "Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal"
_ODA_TO_KENYA = {
    "shape": {
        "kind": "statvar_schema",
        "stat_type": "measuredValue",
        "population_type": "FinancialFlow",
        "measured_property": "amount",
        "measurement_qualifier": "Nominal",
    },
    "slots": [
        {"binding": {"ref": "DAC2aAidType206", "state": "value"}, "property": "dac2aAidType"},
        {"binding": {"state": "unbound"}, "property": "dacTypeOfFinance"},
        {"binding": {"ref": "Disbursement", "state": "value"}, "property": "financialFlowType"},
        {"binding": {"ref": "Gross", "state": "value"}, "property": "nettingBasis"},
    ],
    "entities": {"destination": ["country/KEN"]},
    "open_roles": ["origin"],
    "identity_version": "spec1",
    "requested_window": {"kind": "range", "start_year": 2015},
}


def test_a_definitions_shape_names_its_measure_through_the_fixture_indexes() -> None:
    """Over the fixture code and measure indexes, the record keeps the shape the measure
    partition is keyed on, so the title names the measure. The `dacTypeOfFinance` slot the
    definition leaves open, which its one variable does not declare, varies over nothing and
    is not served."""
    manifest = _family_raw()
    trace = _with_trace(
        manifest,
        resolved_stat_vars=[
            {
                "ref": f"dcid:{_ODA_VARIABLE}",
                "roles": ["financingProvider", "financingRecipient"],
                "observation_count": 10,
            }
        ],
        family_member_count=1,
    )
    roles = (
        RoleFigure(
            role="financingProvider",
            values=(EntityFigure(ref="AdaptationFund", variables=1),),
        ),
        RoleFigure(
            role="financingRecipient", values=(EntityFigure(ref="country/KEN", variables=1),)
        ),
    )
    index = get_index()
    version = _version(
        manifest, definition={**manifest["definition"], "identity": _ODA_TO_KENYA}, trace=trace
    )
    figures = _figures(series=1, roles=roles)
    labels = {"country/KEN": "Kenya", "AdaptationFund": "Adaptation Fund"}
    provisional = mint_record(version, figures=figures, code_index=index, title=None)
    title = mint_title(provisional, labels=labels, code_index=index)
    record = mint_record(version, figures=figures, code_index=index, title=title)
    vocabulary = build_vocabulary(
        index=index,
        measure_index=get_measure_index(),
        labels=labels,
        sources={},
        graph_generation=None,
        catalogue_build=None,
    )

    served = describe_version(record, vocabulary)

    assert record.variables is not None
    assert [(item.property, item.value) for item in record.variables.shape] == [
        ("populationType", "FinancialFlow"),
        ("measuredProperty", "amount"),
        ("statType", "measuredValue"),
        ("measurementQualifier", "Nominal"),
    ]
    assert title == "Gross disbursements to Kenya for official development assistance"
    assert (served["title"], served["cite"]["title"]) == (title, title)
    assert served["about"]["measure"] == {
        "label": "gross disbursements",
        "definition": (
            "The total amount released to recipients, or spent on goods and services for them, "
            "in a period, before any repayments are subtracted."
        ),
    }
    assert [entry["property"]["ref"] for entry in served["about"]["varies"]] == [
        "financingProvider"
    ]
    assert not {"title", "about.measure", "about.measure.definition", "cite.title"} & set(
        served["gaps"]
    )


# The e2e's base Data Commons definitions, as Search mints them: no measure label or noun
# covers their shapes, and each resolved one variable.
_POPULATION_OF_KENYA = {
    "shape": {
        "kind": "statvar_schema",
        "stat_type": "measuredValue",
        "population_type": "Person",
        "measured_property": "count",
    },
    "slots": [{"binding": {"state": "unbound"}, "property": "gender"}],
    "entities": {"observationAbout": ["country/KEN"]},
    "identity_version": "spec1",
    "requested_window": {"kind": "all"},
}
_LIFE_EXPECTANCY_IN_EAST_AFRICA = {
    "shape": {
        "kind": "statvar_schema",
        "stat_type": "measuredValue",
        "population_type": "Person",
        "measured_property": "lifeExpectancy",
    },
    "slots": [{"binding": {"state": "unbound"}, "property": "gender"}],
    "entities": {"observationAbout": ["country/KEN", "country/TZA", "country/UGA"]},
    "identity_version": "spec1",
    "requested_window": {"kind": "range", "start_year": 2000},
}


@pytest.mark.parametrize(
    ("identity", "variable", "places", "labels", "title"),
    [
        pytest.param(
            _POPULATION_OF_KENYA,
            "Count_Person",
            ["country/KEN"],
            {"Count_Person": "Total population", "country/KEN": "Kenya"},
            "Total population in Kenya",
            id="population-at-one-place",
        ),
        pytest.param(
            _LIFE_EXPECTANCY_IN_EAST_AFRICA,
            "LifeExpectancy_Person",
            ["country/KEN", "country/TZA", "country/UGA"],
            {"LifeExpectancy_Person": "Life Expectancy"},
            "Life Expectancy",
            id="life-expectancy-at-three-places",
        ),
        pytest.param(
            _POPULATION_OF_KENYA,
            "Count_Person",
            ["country/KEN"],
            {"country/KEN": "Kenya"},
            None,
            id="variable-without-a-label",
        ),
    ],
)
def test_a_base_variables_title_leads_with_its_label(
    identity: dict[str, Any],
    variable: str,
    places: list[str],
    labels: dict[str, str],
    title: str | None,
) -> None:
    """A definition no measure label or noun covers that resolved one variable is titled by
    that variable's graph label, with the place it fixes; without the label the title is a
    gap."""
    manifest = _family_raw()
    trace = _with_trace(
        manifest,
        resolved_stat_vars=[
            {"ref": f"dcid:{variable}", "roles": ["observationAbout"], "observation_count": 66}
        ],
        family_member_count=None,
    )
    roles = (
        RoleFigure(
            role="observationAbout",
            values=tuple(EntityFigure(ref=place, variables=1) for place in places),
        ),
    )
    index = get_index()
    version = _version(
        manifest, definition={**manifest["definition"], "identity": identity}, trace=trace
    )
    figures = _figures(series=len(places), roles=roles)
    provisional = mint_record(version, figures=figures, code_index=index, title=None)

    assert provisional.variables is not None
    assert provisional.variables.variable == variable
    assert variable in named_refs(provisional)
    assert mint_title(provisional, labels=labels, code_index=index) == title


@pytest.mark.parametrize(
    ("member", "value", "ref"),
    [
        pytest.param(
            "unit",
            {"numerator": {"ref": "USDollar"}, "denominator": {"stat_var": "Count_Person"}},
            "Count_Person",
            id="variable-a-per-capita-unit-counts",
        ),
        pytest.param(
            "sides",
            {"denominator": {}, "numerator": {"donor": "dcid:country/FRA"}},
            "country/FRA",
            id="entity-a-side-holds",
        ),
        pytest.param(
            "params",
            {"numerator_entity": {"donor": "dcid:country/GBR"}},
            "country/GBR",
            id="entity-a-side-is-narrowed-to",
        ),
    ],
)
def test_a_ref_a_step_names_outside_its_inputs_is_a_named_ref(
    member: str, value: dict[str, object], ref: str
) -> None:
    """Its label is read with the others, so the unit's denominator and the entity a side
    holds or is narrowed to are worded."""
    steps = quotient_steps()
    steps[2][member] = value

    record = _publish(steps)

    assert ref in named_refs(record)


# Units


@pytest.mark.parametrize(
    ("cell", "unit"),
    [
        pytest.param(None, None, id="none-declared"),
        pytest.param("dcid:USDollar", SimpleUnit(ref="USDollar"), id="ref"),
        pytest.param(
            {"currency": "USD", "prices": "constant", "base_year": 2023},
            SimpleUnit(currency="USD", prices="constant", base_year=2023),
            id="converted",
        ),
        pytest.param(
            {"numerator": "USDollar", "denominator": {"stat_var": "dcid:Count_Person"}},
            ComposedUnit(
                numerator=SimpleUnit(ref="USDollar"),
                denominator=VariableSide(stat_var="Count_Person"),
            ),
            id="per-capita",
        ),
    ],
)
def test_a_unit_cell_becomes_a_stored_unit_by_bare_dcid(
    cell: str | dict[str, object] | None, unit: object
) -> None:
    assert unit_from_cell(cell) == unit


# Round trip through the composer


@cache
def _vocabulary() -> Vocabulary:
    return build_vocabulary(
        index=family_fixture._code_index(),
        measure_index=MeasureIndex(axes=(), values=(), partition_labels=()),
        labels={},
        sources={},
        graph_generation="gen-1",
        catalogue_build="fake-catalogue",
    )


def _described(record: DescriptionRecord) -> dict[str, Any]:
    served = describe_version(record, _vocabulary())
    assert json.loads(json.dumps(served)) == served
    validate_description(served)
    return served


def test_a_mint_record_is_described_as_the_kit_schema_requires() -> None:
    record = with_labels(
        _mint(_family_raw()), {"country/KEN": "Kenya"}, read=datetime.date(2026, 10, 11)
    )

    served = _described(record)

    assert served["describes"]["kind"] == "dataset_version"
    assert [dimension["property"]["ref"] for dimension in served["about"]["varies"]] == [
        FAMILY_AXIS
    ]


def test_the_worked_examples_providers_fold_and_mint_into_its_provider_dimension() -> None:
    """The design's Nigeria health version: provider n holds data under 3 + n % 5 variables,
    each with two series. The fold and the mint give the acceptance record's provider
    dimension, which serves providers 04 and 09 at 7 variables and 61 more."""
    rows = [
        {
            "stat_var": f"dcid:Variable{variable}",
            "financingProvider": f"dcid:{provider}",
            "financingRecipient": "dcid:country/NGA",
            "time_period": period,
            "value": 1.0,
            "measurementMethod": method,
        }
        for n, (provider, _label) in enumerate(_WORKED_PROVIDERS)
        for variable in range(3 + n % 5)
        for method in ("Reported", "Estimated")
        for period in ("2023", "2024")
    ]
    sink = SeriesStatisticsSink(continuing=())
    sink.write(
        compute_series_statistics_table(
            build_frame(role_keys=("financingProvider", "financingRecipient"), rows=rows)
        )
    )
    manifest = _family_raw()

    record = _mint(
        manifest,
        figures=sink.finish().figures,
        definition=_with_entities(manifest, {"destination": ["country/NGA"]}),
    )

    assert record.variables is not None
    recipient = record.variables.fixed[0]
    assert (recipient.property, recipient.value) == ("financingRecipient", "country/NGA")
    assert record.variables.dimensions is not None
    provider = record.variables.dimensions[0]
    assert provider == _WORKED_DIMENSIONS[0]
    served = _described(
        with_labels(record, dict(_WORKED_PROVIDERS), read=datetime.date(2026, 9, 28))
    )
    (varies,) = [
        entry
        for entry in served["about"]["varies"]
        if entry["property"]["ref"] == "financingProvider"
    ]
    top = [(entry["label"], entry["variables"]) for entry in varies["top"]]
    assert (varies["count"], top, varies["more"]) == (
        63,
        [("provider 04", 7), ("provider 09", 7)],
        61,
    )
    assert varies["values"].endswith("/description/values?property=financingProvider")


def test_a_record_of_a_choice_a_counted_conversion_and_currency_gaps_is_described() -> None:
    """The kit schema accepts the steps and caveats the trace's structured facts give: a
    choice naming the others, a conversion counting what it left out, and coverage gaps
    naming places and a variable."""
    manifest = _converted(_family_raw(), projected=2, left_out=_DEFLATOR_LEFT_OUT)
    facets = copy.deepcopy(manifest["trace"]["facets"])
    facets[0]["selection_reason"] = "most_observations_over_disagreement"
    trace = _with_trace(
        manifest,
        facets=facets,
        choices=_CHOICES,
        coverage_gaps=[
            {"role": "recipient", "reason": "no_national_currency", "entities": ["Earth"]}
        ],
        origin_gaps=[
            {
                "code": "withheld_by_registry",
                "variable": family_fixture._MEMBER_A,
                "entities": ["country/KEN"],
                "provenances": ["dc/base/UN_SDG"],
            }
        ],
    )

    served = _described(_mint(manifest, trace=trace))

    assert [step["op"] for step in served["steps"]] == ["choose_source", "convert"]
    assert [caveat["kind"] for caveat in served["caveats"]] == [
        "estimated",
        "coverage_gap",
        "coverage_gap",
        "coverage_gap",
    ]


def test_a_collapsed_record_is_described_with_its_sum() -> None:
    served = _described(_collapse(_mint(_family_raw())))

    assert [step["op"] for step in served["steps"]] == ["sum_variables"]


def test_a_publish_record_with_steps_is_described_with_its_inputs_ids() -> None:
    served = _described(_publish(quotient_steps()))

    assert [entry["id"] for entry in served["freshness"]["inputs"]] == [
        NUMERATOR[0],
        DENOMINATOR[0],
    ]
    assert not [gap for gap in served["gaps"] if gap.endswith(".id") or gap.endswith(".ref")]


def test_a_publish_record_without_steps_is_described_with_null_input_ids_and_gaps() -> None:
    served = _described(_publish(None))

    assert [(entry["id"], entry["content_version"]) for entry in served["freshness"]["inputs"]] == [
        (None, NUMERATOR[1]),
        (None, DENOMINATOR[1]),
    ]
    assert [entry["id"] for entry in served["cite"]["inputs"]] == [None, None]
    assert [(source["ref"], "description" in source) for source in served["sources"]] == [
        (None, False),
        (None, False),
    ]
    assert {
        "freshness.inputs.0.id",
        "freshness.inputs.1.id",
        "cite.inputs.0.id",
        "sources.0.ref",
    } <= set(served["gaps"])


# Text bounds


@pytest.mark.parametrize(
    ("value", "limit", "expected"),
    [
        pytest.param("a" * 512, 512, "a" * 512, id="text-at-its-bound"),
        pytest.param("a" * 513, 512, "a" * 511 + "…", id="text-past-its-bound"),
        pytest.param("a" * 8192, 8192, "a" * 8192, id="long-text-at-its-bound"),
        pytest.param("a" * 8193, 8192, "a" * 8191 + "…", id="long-text-past-its-bound"),
        pytest.param("a\x00" * 512, 512, "a" * 512, id="u0000-does-not-count"),
        pytest.param("\x00", 512, None, id="only-u0000-is-unknown"),
        pytest.param(None, 512, None, id="not-text-is-unknown"),
    ],
)
def test_text_read_at_mint_is_clipped_to_its_bound(
    value: str | None, limit: int, expected: str | None
) -> None:
    assert bounded_text(value, limit=limit) == expected


def test_overlong_text_read_at_mint_is_clipped_and_never_fails_the_record() -> None:
    manifest = _family_raw()
    citation = {
        "text": manifest["citation"]["text"],
        "sources": [
            {
                "ref": "OECD_DAC2A",
                "label": "n" * 600,
                "license": "l" * 600,
                "documentation": {
                    "transformation_log": [
                        {"transformation": "t" * 600, "description": "d" * 9000},
                        {"transformation": "\x00", "description": "left out"},
                        {"transformation": "Untitled body", "description": ""},
                    ]
                },
            }
        ],
    }

    record = mint_record(
        _version(
            manifest,
            citation=citation,
            definition=_with_entities(manifest, {"r" * 600: ["country/" + "K" * 600]}),
        ),
        figures=_figures(),
        code_index=family_fixture._code_index(),
        title="T" * 600,
    )
    labelled = with_labels(record, {"country/KEN": "K" * 600}, read=datetime.date(2026, 10, 11))

    (source,) = labelled.evidence.sources
    assert (source.name, source.licence) == ("n" * 511 + "…", "l" * 511 + "…")
    assert source.pipeline_steps == [
        PipelineStep(title="t" * 511 + "…", description="d" * 8191 + "…"),
        PipelineStep(title="Untitled body", description="Untitled body"),
    ]
    assert labelled.evidence.title == "T" * 511 + "…"
    assert record.variables is not None
    assert [(item.property, item.value) for item in record.variables.fixed] == [
        ("r" * 511 + "…", "country/" + "K" * 503 + "…")
    ]
    assert [label.label for label in labelled.evidence.labels] == ["K" * 511 + "…"]
