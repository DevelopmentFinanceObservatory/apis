"""`compose_components` composes a composite's component results into one frame.

Each component result here is built from rows written for the rule under test, so these
tests prove the composition rules over years and cells, not any recorded figure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import pytest
from dfo_frame_grammar import (
    FRAME_GRAMMAR_VERSION,
    derive_columns,
    frame_sha256,
    stream_frame_sha256,
)

from code_index.composites import DECLARED_COMPOSITES
from code_index.declarations import COMPOSITES, CRS_INCLUSIVE
from code_index.types import ComponentRecipe, ComponentRole, CompositeDeclaration
from materialise import MaterialisationRequest, MaterialisationResult
from materialise.composite import ComponentResult, compose_components
from materialise.content_version import mint_version_from_chunks
from materialise.dependencies import Dependencies
from materialise.enrichment import EnrichmentLookup
from materialise.errors import MaterialisationUsageError
from materialise.finish import _coverage_below_scope_warning
from materialise.frame_build import build_frame
from materialise.profile import compute_output_profile, compute_output_profile_from_chunks
from materialise.statistics import SeriesStatisticsSink, compute_series_statistics_table
from materialise.trace import ManifestTrace
from materialise.window import Coverage, RequestedWindow

from .conftest import FakeGraphReader

_ROLES = ("financingProvider", "financingRecipient")
_DONOR = "country/USA"
_CRS = "provenance/OECDCreditorReportingSystem"
_IMPUTED = "provenance/ONEImputedMultilateralSectors"
_DAC1 = "provenance/OECDDAC1"
_STACKED = DECLARED_COMPOSITES[CRS_INCLUSIVE]
_SPLICED = next(c for c in COMPOSITES if c.kind == "spliced")
_ENRICHMENT = EnrichmentLookup(FakeGraphReader(), parties=lambda: {})

# CRS and imputed members of one sector. The imputed dcid sorts before the CRS one, so the
# composed frame leads with the zero-filling component's rows.
_B = "dcid:Amount_Health"
_I = "dcid:AttributedAmount_Health"

type _Row = tuple[str, str, int, float]
"""A row's member, recipient, year and value."""


def _recipe(composite: CompositeDeclaration, role: ComponentRole) -> ComponentRecipe:
    return next(recipe for recipe in composite.components if recipe.role == role)


def _component(
    composite: CompositeDeclaration,
    role: ComponentRole,
    rows: Sequence[_Row],
    *,
    provenance: str,
    members: Sequence[str],
    recipients: Sequence[str] = ("country/KEN", "country/GHA", "country/NGA"),
    method: str | None = None,
    years: tuple[int, int] = (2005, 2024),
    unserved: Sequence[str] = (),
    units: Mapping[str, object] | None = None,
    covered: Sequence[int] | None = None,
) -> ComponentResult:
    """One component's result over `rows`, as `pipeline.materialise_group` returns it, with
    `unserved` the members its facet group election left out and its warning counting them.
    Each row's unit is its recipient's in `units`, USD when `units` is None. Its table covers
    `covered`, the years of `rows` when None."""
    frame = build_frame(
        role_keys=_ROLES,
        rows=[
            {
                "stat_var": stat_var,
                "financingProvider": _DONOR,
                "financingRecipient": recipient,
                "time_period": str(year),
                "value": value,
                "unit": "USD" if units is None else units[recipient],
                "measurementMethod": method,
                "provenance": provenance,
            }
            for stat_var, recipient, year, value in rows
        ],
    )
    window = RequestedWindow(kind="range", start_year=years[0], end_year=years[1])
    requests = [
        MaterialisationRequest(
            stat_var=member,
            role_bindings={"financingProvider": [_DONOR], "financingRecipient": list(recipients)},
            declared_roles=_ROLES,
            facet_bindings={},
            window=window,
            component=role,
        )
        for member in members
    ]
    statistics = SeriesStatisticsSink(continuing=())
    result = MaterialisationResult(
        frame_chunks=lambda: iter((frame,)),
        frame_sha256=frame_sha256(frame),
        content_version=mint_version_from_chunks((frame,)),
        trace=ManifestTrace(
            applied_window=None,
            graph_revision=None,
            source_revisions=({"source": provenance, "revision": "r1"},),
            resolved_stat_vars=tuple(
                {"ref": member, "roles": list(_ROLES), "observation_count": 0} for member in members
            ),
            facets=(
                {
                    "axis": "provenance",
                    "ref": provenance,
                    "basis": "deployment_fallback",
                    "default_policy": None,
                    "default_policy_version": None,
                    "selection_reason": None,
                    "dispatch_facet_id": "facet",
                },
                {
                    "axis": "measurementMethod",
                    "ref": method,
                    "basis": "deployment_fallback",
                    "default_policy": None,
                    "default_policy_version": None,
                    "selection_reason": None,
                    "dispatch_facet_id": "facet",
                },
            ),
            normalisations=(),
            conversion=None,
            frame_grammar_version=FRAME_GRAMMAR_VERSION,
            materialiser_build="build-1",
            capability_version="cap-1",
        ),
        coverage=Coverage(rows=frame.num_rows, dimensions=(), window=None),
        warnings=(
            (
                _coverage_below_scope_warning(
                    (),
                    countries_fallback=False,
                    off_cadence=(),
                    out_of_window=(),
                    unserved_members=unserved,
                    member_count=len(members),
                ),
            )
            if unserved
            else ()
        ),
        unserved_members=tuple(unserved),
        series_statistics=statistics.finish(),
        output_profile=compute_output_profile(frame, role_columns=_ROLES, enrichment=_ENRICHMENT),
        conversion_record=None,
        member_conversion_records=dict.fromkeys(members),
        dependencies=Dependencies(
            members="sha256:x",
            sources=((provenance, "2026-10-01"),),
            conversion_sources=(),
            recipes=(),
            window=None,
        ),
    )
    return ComponentResult(
        role=role,
        recipe=_recipe(composite, role),
        requests=requests,
        result=result,
        covered_years=frozenset((year for _, _, year, _ in rows) if covered is None else covered),
    )


def _compose(
    components: Sequence[ComponentResult],
    *,
    composite: CompositeDeclaration = _STACKED,
    years: tuple[int, int] = (2005, 2024),
) -> MaterialisationResult:
    return compose_components(
        components,
        composite=composite,
        window=RequestedWindow(kind="range", start_year=years[0], end_year=years[1]),
        role_columns=_ROLES,
        enrichment=_ENRICHMENT,
    )


def _rows(result: MaterialisationResult) -> list[tuple[str, str, str, float, str | None]]:
    """Each composed row's member, recipient, period, value and provenance."""
    return [
        (
            row["stat_var"],
            row["financingRecipient"],
            row["time_period"],
            row["value"],
            row["provenance"],
        )
        for frame in result.frame_chunks()
        for row in frame.table.to_pylist()
    ]


def _summary(result: MaterialisationResult, role: str) -> dict:
    return next(entry for entry in result.trace.components if entry["role"] == role)


def test_a_leading_missing_year_drops_the_years_one_component_does_not_cover():
    bilateral = _component(
        _STACKED,
        "bilateral",
        [(_B, "country/KEN", year, 10.0) for year in (2009, 2010, 2011, 2012)],
        provenance=_CRS,
        members=[_B],
    )
    imputed = _component(
        _STACKED,
        "imputed",
        [(_I, "country/KEN", year, 1.0) for year in (2011, 2012)],
        provenance=_IMPUTED,
        members=[_I],
    )

    result = _compose([bilateral, imputed], years=(2009, 2012))

    assert {period for _, _, period, _, _ in _rows(result)} == {"2011", "2012"}
    assert _summary(result, "bilateral")["dropped_years"] == [2009, 2010]
    assert _summary(result, "bilateral")["periods"] == [[2011, 2012]]
    assert _summary(result, "imputed")["zero_rows"] == 0
    warning = next(w for w in result.warnings if w["code"] == "coverage_below_scope")
    assert warning["requested_years"] == {"start": 2009, "end": 2012}
    assert warning["served_years"] == {"start": 2011, "end": 2012}


def test_an_internal_gap_leaves_the_year_out_of_every_component():
    bilateral = _component(
        _STACKED,
        "bilateral",
        [(_B, "country/KEN", year, 10.0) for year in (2011, 2012, 2013)],
        provenance=_CRS,
        members=[_B],
    )
    imputed = _component(
        _STACKED,
        "imputed",
        [(_I, "country/KEN", year, 1.0) for year in (2011, 2013)],
        provenance=_IMPUTED,
        members=[_I],
    )

    result = _compose([bilateral, imputed])

    assert {period for _, _, period, _, _ in _rows(result)} == {"2011", "2013"}
    assert _summary(result, "bilateral")["dropped_years"] == [2012]


def test_a_year_the_imputed_table_covers_and_this_read_lacks_keeps_the_bilateral_figure():
    """The imputed table covers 2011 to 2013, so the composite serves every one of them, and
    a year this read holds no imputed row in is filled with zeros instead of dropped."""
    bilateral = _component(
        _STACKED,
        "bilateral",
        [(_B, "country/KEN", year, 10.0) for year in (2011, 2012, 2013)],
        provenance=_CRS,
        members=[_B],
    )
    imputed = _component(
        _STACKED,
        "imputed",
        [(_I, "country/KEN", year, 1.0) for year in (2011, 2013)],
        provenance=_IMPUTED,
        members=[_I],
        covered=(2011, 2012, 2013),
    )

    result = _compose([bilateral, imputed])

    rows = _rows(result)
    assert (_B, "country/KEN", "2012", 10.0, _STACKED.provenance) in rows
    assert (_I, "country/KEN", "2012", 0.0, _STACKED.provenance) in rows
    assert _summary(result, "bilateral")["dropped_years"] == []
    assert _summary(result, "imputed")["periods"] == [[2011, 2013]]


def test_an_empty_imputed_read_in_years_its_table_covers_writes_zeros_beside_the_bilateral():
    """No imputed row reaches this read, but the table covers the year: the composite keeps
    the bilateral figure and writes each imputed member's zero row with the method its facet
    selection names."""
    bilateral = _component(
        _STACKED,
        "bilateral",
        [(_B, "country/KEN", 2012, 10.0), (_B, "country/GHA", 2012, 20.0)],
        provenance=_CRS,
        members=[_B],
    )
    imputed = _component(
        _STACKED,
        "imputed",
        [],
        provenance=_IMPUTED,
        members=[_I],
        method="ONEImputed",
        covered=(2012,),
    )

    result = _compose([bilateral, imputed])

    assert sorted(
        (row["stat_var"], row["financingRecipient"], row["value"], row["measurementMethod"])
        for frame in result.frame_chunks()
        for row in frame.table.to_pylist()
    ) == [
        (_B, "country/GHA", 20.0, None),
        (_B, "country/KEN", 10.0, None),
        (_I, "country/GHA", 0.0, "ONEImputed"),
        (_I, "country/KEN", 0.0, "ONEImputed"),
    ]
    assert _summary(result, "imputed")["zero_rows"] == 2


def test_a_covered_cell_the_imputed_table_lacks_gets_one_zero_row():
    bilateral = _component(
        _STACKED,
        "bilateral",
        [(_B, "country/KEN", 2012, 10.0), (_B, "country/GHA", 2012, 20.0)],
        provenance=_CRS,
        members=[_B],
    )
    imputed = _component(
        _STACKED,
        "imputed",
        [(_I, "country/GHA", 2012, 2.0)],
        provenance=_IMPUTED,
        members=[_I],
    )

    result = _compose([bilateral, imputed])

    rows = _rows(result)
    assert (_I, "country/KEN", "2012", 0.0, _STACKED.provenance) in rows
    assert [row for row in rows if row[0] == _I and row[1] == "country/KEN"] == [
        (_I, "country/KEN", "2012", 0.0, _STACKED.provenance)
    ]
    assert sum(row[3] for row in rows) == 10.0 + 20.0 + 2.0
    assert _summary(result, "imputed")["zero_rows"] == 1
    assert _summary(result, "bilateral")["zero_rows"] == 0


def test_a_recipient_only_the_imputed_table_serves_keeps_its_rows_and_gets_no_bilateral_row():
    bilateral = _component(
        _STACKED,
        "bilateral",
        [(_B, "country/KEN", 2012, 10.0)],
        provenance=_CRS,
        members=[_B],
    )
    imputed = _component(
        _STACKED,
        "imputed",
        [(_I, "country/KEN", 2012, 1.0), (_I, "country/NGA", 2012, 3.0)],
        provenance=_IMPUTED,
        members=[_I],
    )

    result = _compose([bilateral, imputed])

    rows = _rows(result)
    assert (_I, "country/NGA", "2012", 3.0, _STACKED.provenance) in rows
    assert not [row for row in rows if row[0] == _B and row[1] == "country/NGA"]
    warning = next(w for w in result.warnings if w["code"] == "coverage_below_scope")
    assert warning["unserved_entities"]["refs"] == ["dcid:country/GHA", "dcid:country/NGA"]


def test_every_row_carries_the_composite_provenance_and_the_trace_names_each_component():
    bilateral = _component(
        _STACKED, "bilateral", [(_B, "country/KEN", 2012, 10.0)], provenance=_CRS, members=[_B]
    )
    imputed = _component(
        _STACKED, "imputed", [(_I, "country/KEN", 2012, 1.0)], provenance=_IMPUTED, members=[_I]
    )

    result = _compose([bilateral, imputed])

    assert {row[4] for row in _rows(result)} == {_STACKED.provenance}
    assert [facet["ref"] for facet in result.trace.facets if facet["axis"] == "provenance"] == [
        _STACKED.provenance
    ]
    assert [(entry["role"], entry["provenance"]) for entry in result.trace.components] == [
        ("bilateral", _CRS),
        ("imputed", _IMPUTED),
    ]
    assert _summary(result, "bilateral")["content_version"] == bilateral.result.content_version
    assert result.content_version == mint_version_from_chunks(result.frame_chunks())
    assert [entry["ref"] for entry in result.trace.resolved_stat_vars] == [_B, _I]


def test_the_composed_frame_records_each_components_measurement_method_under_one_unit():
    bilateral = _component(
        _STACKED,
        "bilateral",
        [(_B, "country/KEN", 2012, 10.0), (_B, "country/GHA", 2012, 4.0)],
        provenance=_CRS,
        members=[_B],
        method="OECDCRSReported",
    )
    imputed = _component(
        _STACKED,
        "imputed",
        [(_I, "country/GHA", 2012, 1.0)],
        provenance=_IMPUTED,
        members=[_I],
        method="ONEImputed",
    )

    result = _compose([bilateral, imputed])

    table = next(iter(result.frame_chunks())).table
    frames = list(result.frame_chunks())
    units = {row["unit"]["ref"] for frame in frames for row in frame.table.to_pylist()}
    methods = {
        (row["stat_var"], row["measurementMethod"])
        for frame in frames
        for row in frame.table.to_pylist()
    }
    assert table.num_rows > 0
    assert units == {"USD"}
    # The zero row copies its member's method.
    assert methods == {(_B, "OECDCRSReported"), (_I, "ONEImputed")}


def test_a_member_holding_no_row_copies_its_components_unit_and_method():
    bilateral = _component(
        _STACKED, "bilateral", [(_B, "country/KEN", 2012, 10.0)], provenance=_CRS, members=[_B]
    )
    other = "dcid:AttributedAmount_Other"
    imputed = _component(
        _STACKED,
        "imputed",
        [(_I, "country/KEN", 2012, 1.0)],
        provenance=_IMPUTED,
        members=[_I, other],
        method="ONEImputed",
    )

    result = _compose([bilateral, imputed])

    zero = [row for row in _rows(result) if row[0] == other]
    assert zero == [(other, "country/KEN", "2012", 0.0, _STACKED.provenance)]


def test_a_member_its_facet_group_left_out_gets_no_zero_row_and_is_counted():
    """A member the component's facet group election leaves out holds its series under
    another source, unit or method, so its absence from the frame is not a zero."""
    bilateral = _component(
        _STACKED, "bilateral", [(_B, "country/KEN", 2012, 10.0)], provenance=_CRS, members=[_B]
    )
    other = "dcid:AttributedAmount_Other"
    imputed = _component(
        _STACKED,
        "imputed",
        [(_I, "country/KEN", 2012, 1.0)],
        provenance=_IMPUTED,
        members=[_I, other],
        unserved=[other],
    )

    result = _compose([bilateral, imputed])

    assert not [row for row in _rows(result) if row[0] == other]
    assert _summary(result, "imputed")["zero_rows"] == 0
    warning = next(w for w in result.warnings if w["code"] == "coverage_below_scope")
    assert warning["unserved_members"] == {"count": 1, "refs": [other]}
    assert warning["member_count"] == 2


def test_the_merge_rests_on_stat_var_leading_the_sort_key_and_refuses_a_shared_member():
    # One pass emits each member's rows and zero rows together only because `stat_var` leads
    # the row sort key and no member belongs to two components.
    assert derive_columns(_ROLES)[0] == "stat_var"
    bilateral = _component(
        _STACKED, "bilateral", [(_B, "country/KEN", 2012, 10.0)], provenance=_CRS, members=[_B]
    )
    shared = _component(
        _STACKED, "imputed", [(_B, "country/KEN", 2012, 1.0)], provenance=_IMPUTED, members=[_B]
    )

    with pytest.raises(MaterialisationUsageError, match="both serve"):
        _compose([bilateral, shared])


def test_the_composed_frame_is_the_same_whichever_order_the_components_sort_in():
    later = "dcid:ZAttributedAmount_Health"
    bilateral = _component(
        _STACKED,
        "bilateral",
        [(_B, "country/KEN", 2012, 10.0), (_B, "country/GHA", 2012, 4.0)],
        provenance=_CRS,
        members=[_B],
    )
    before = _component(
        _STACKED, "imputed", [(_I, "country/GHA", 2012, 1.0)], provenance=_IMPUTED, members=[_I]
    )
    after = _component(
        _STACKED,
        "imputed",
        [(later, "country/GHA", 2012, 1.0)],
        provenance=_IMPUTED,
        members=[later],
    )

    leading = _rows(_compose([bilateral, before]))
    trailing = _rows(_compose([bilateral, after]))

    assert [row[1:] for row in leading if row[0] == _I] == [
        row[1:] for row in trailing if row[0] == later
    ]
    assert [row[0] for row in leading] == sorted(row[0] for row in leading)
    assert [row[0] for row in trailing] == sorted(row[0] for row in trailing)


def test_a_spliced_composite_serves_each_component_in_its_own_years_alone():
    net, ge = "dcid:Amount_Net_ODA", "dcid:GrantEquivalent_ODA"
    headline_net = _component(
        _SPLICED,
        "headline_net",
        [(net, "country/KEN", year, 5.0) for year in (2015, 2016, 2017, 2018)],
        provenance=_DAC1,
        members=[net],
        years=(2015, 2017),
    )
    headline_ge = _component(
        _SPLICED,
        "headline_ge",
        [(ge, "country/KEN", year, 6.0) for year in (2018, 2019, 2020)],
        provenance=_DAC1,
        members=[ge],
        years=(2018, 2020),
    )

    result = _compose([headline_net, headline_ge], composite=_SPLICED, years=(2015, 2020))

    by_year = {}
    for stat_var, _recipient, period, _value, provenance in _rows(result):
        by_year.setdefault(period, set()).add(stat_var)
        assert provenance == _SPLICED.provenance
    assert by_year == {
        "2015": {net},
        "2016": {net},
        "2017": {net},
        "2018": {ge},
        "2019": {ge},
        "2020": {ge},
    }
    assert _summary(result, "headline_net")["periods"] == [[2015, 2017]]
    assert _summary(result, "headline_net")["dropped_years"] == [2018]
    assert _summary(result, "headline_ge")["periods"] == [[2018, 2020]]


def test_no_component_row_in_the_requested_years_is_an_empty_result():
    bilateral = _component(
        _STACKED, "bilateral", [(_B, "country/KEN", 2012, 10.0)], provenance=_CRS, members=[_B]
    )
    imputed = _component(_STACKED, "imputed", [], provenance=_IMPUTED, members=[_I])

    result = _compose([bilateral, imputed])

    assert result.coverage.rows == 0
    assert [w["code"] for w in result.warnings] == ["empty_result"]
    assert _summary(result, "bilateral")["dropped_years"] == [2012]


def test_a_composite_whose_component_declares_other_roles_is_refused():
    bilateral = _component(
        _STACKED, "bilateral", [(_B, "country/KEN", 2012, 10.0)], provenance=_CRS, members=[_B]
    )
    imputed = _component(
        _STACKED, "imputed", [(_I, "country/KEN", 2012, 1.0)], provenance=_IMPUTED, members=[_I]
    )
    other_roles = replace(
        imputed,
        requests=[
            replace(request, declared_roles=("financingProvider",)) for request in imputed.requests
        ],
    )

    with pytest.raises(MaterialisationUsageError, match="declares roles"):
        _compose([bilateral, other_roles])


def test_zero_rows_take_their_place_in_the_members_row_order():
    imputed = _component(
        _STACKED,
        "imputed",
        [(_I, "country/KEN", 2011, 1.0), (_I, "country/KEN", 2012, 1.0)],
        provenance=_IMPUTED,
        members=[_I],
        years=(2011, 2012),
    )
    bilateral = _component(
        _STACKED,
        "bilateral",
        [
            (_B, "country/GHA", 2011, 3.0),
            (_B, "country/GHA", 2012, 4.0),
            (_B, "country/KEN", 2012, 10.0),
        ],
        provenance=_CRS,
        members=[_B],
    )

    result = _compose([bilateral, imputed], years=(2011, 2012))

    assert [row[1:3] for row in _rows(result) if row[0] == _I] == [
        ("country/GHA", "2011"),
        ("country/GHA", "2012"),
        ("country/KEN", "2011"),
        ("country/KEN", "2012"),
    ]
    assert _summary(result, "imputed")["zero_rows"] == 2


# Each recipient's national currency, the unit a mint binding national currencies serves.
_NATIONAL = {
    "country/KEN": {"currency": "KES", "prices": "current"},
    "country/GHA": {"currency": "GHS", "prices": "current"},
}


def _units_by_cell(result: MaterialisationResult) -> dict[tuple[str, str], set[str]]:
    """The currencies each recipient and period's rows state, over every member: one per cell
    is what a read summing the members of a cell needs."""
    cells: dict[tuple[str, str], set[str]] = {}
    for frame in result.frame_chunks():
        for row in frame.table.to_pylist():
            key = (row["financingRecipient"], row["time_period"])
            cells.setdefault(key, set()).add(row["unit"]["request"]["currency"])
    return cells


def test_a_zero_row_states_its_own_cells_unit():
    """The imputed member holds Ghana alone, in cedis. Its Kenya zero is in shillings, the
    unit of the Kenyan cell, so every cell states one unit."""
    bilateral = _component(
        _STACKED,
        "bilateral",
        [(_B, "country/KEN", 2012, 10.0), (_B, "country/GHA", 2012, 20.0)],
        provenance=_CRS,
        members=[_B],
        units=_NATIONAL,
    )
    imputed = _component(
        _STACKED,
        "imputed",
        [(_I, "country/GHA", 2012, 2.0)],
        provenance=_IMPUTED,
        members=[_I],
        units=_NATIONAL,
    )

    result = _compose([bilateral, imputed])

    (zero,) = (
        row
        for frame in result.frame_chunks()
        for row in frame.table.to_pylist()
        if row["stat_var"] == _I and row["financingRecipient"] == "country/KEN"
    )
    assert (zero["value"], zero["unit"]["request"]["currency"]) == (0.0, "KES")
    assert _units_by_cell(result) == {
        ("country/KEN", "2012"): {"KES"},
        ("country/GHA", "2012"): {"GHS"},
    }


def test_a_member_holding_no_row_in_a_frame_of_several_currencies_takes_each_cells_unit():
    other = "dcid:AttributedAmount_Other"
    bilateral = _component(
        _STACKED,
        "bilateral",
        [(_B, "country/KEN", 2012, 10.0), (_B, "country/GHA", 2012, 20.0)],
        provenance=_CRS,
        members=[_B],
        units=_NATIONAL,
    )
    imputed = _component(
        _STACKED,
        "imputed",
        [(_I, "country/KEN", 2012, 1.0), (_I, "country/GHA", 2012, 2.0)],
        provenance=_IMPUTED,
        members=[_I, other],
        method="ONEImputed",
        units=_NATIONAL,
    )

    result = _compose([bilateral, imputed])

    zeros = sorted(
        (row["financingRecipient"], row["unit"]["request"]["currency"], row["measurementMethod"])
        for frame in result.frame_chunks()
        for row in frame.table.to_pylist()
        if row["stat_var"] == other
    )
    assert zeros == [("country/GHA", "GHS", "ONEImputed"), ("country/KEN", "KES", "ONEImputed")]
    assert all(len(units) == 1 for units in _units_by_cell(result).values())


def test_the_frame_is_composed_once_and_every_later_pass_reads_the_same_rows():
    """The pass that hashes the composed frame also folds its statistics and profile, so
    each component is read twice (the plan and the composition) however often the result is
    read after, and what that pass folded equals what separate passes over the result read."""
    members_b = [f"dcid:Amount_Health_{i}" for i in range(3)]
    members_i = [f"dcid:AttributedAmount_Health_{i}" for i in range(2)]
    reads: dict[str, int] = {}

    def counted(component: ComponentResult) -> ComponentResult:
        chunks = component.result.frame_chunks

        def frame_chunks():
            reads[component.role] = reads.get(component.role, 0) + 1
            return chunks()

        return replace(component, result=replace(component.result, frame_chunks=frame_chunks))

    bilateral = _component(
        _STACKED,
        "bilateral",
        [
            (member, recipient, year, float(position + year - 2000))
            for position, member in enumerate(members_b)
            for recipient in ("country/KEN", "country/GHA")
            for year in (2011, 2012)
        ],
        provenance=_CRS,
        members=members_b,
        units=_NATIONAL,
    )
    imputed = _component(
        _STACKED,
        "imputed",
        [(members_i[0], "country/GHA", 2012, 2.0)],
        provenance=_IMPUTED,
        members=members_i,
        units=_NATIONAL,
        covered=(2011, 2012),
    )

    result = _compose([counted(bilateral), counted(imputed)])
    passes = [list(result.frame_chunks()) for _ in range(3)]

    assert reads == {"bilateral": 2, "imputed": 2}
    assert all(
        [frame.table for frame in chunks] == [frame.table for frame in passes[0]]
        for chunks in passes
    )
    assert [frame.table["stat_var"].to_pylist()[0] for frame in passes[0]] == [
        *members_b,
        *members_i,
    ]
    assert result.frame_sha256 == stream_frame_sha256(iter(passes[0]))
    assert result.output_profile == compute_output_profile_from_chunks(
        result.frame_chunks, role_columns=_ROLES, enrichment=_ENRICHMENT
    )
    statistics = SeriesStatisticsSink(continuing=())
    for frame in passes[0]:
        statistics.write(compute_series_statistics_table(frame))
    assert result.series_statistics.data == statistics.finish().data
