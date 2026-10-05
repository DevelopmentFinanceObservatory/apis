"""Pins how `derive_output_options` combines a frame profile with deployment capabilities."""

from __future__ import annotations

from code_index.shared import get_index
from datasets_api.capabilities import CapabilityFormat, build_capabilities
from datasets_api.config import DatasetsBuildSettings
from datasets_api.output_options import derive_output_options
from materialise.profile import ColumnProfile, EnrichmentCoverage, OutputProfile, PivotFeasibility

_CAPABILITY_VERSION = "cap-v1"
_BUILD = "build-123"


def _capabilities(**overrides):
    caps = build_capabilities(
        capability_version=_CAPABILITY_VERSION,
        build=_BUILD,
        code_index_stamp=get_index().stamp,
        declarations=get_index().declarations,
        settings=DatasetsBuildSettings(),
    )
    return caps.model_copy(update=overrides) if overrides else caps


def _column(name: str, kind: str, *, droppable: bool, distinct: int | None) -> ColumnProfile:
    return ColumnProfile(name=name, kind=kind, droppable=droppable, distinct=distinct)


def _pivot(
    index: str,
    columns: str,
    *,
    unique: bool = True,
    homogeneous: bool = True,
    width: int = 1,
    cells: int = 1,
    varying: tuple[tuple[str, int], ...] = (),
    duplicate_keys: bool = False,
) -> PivotFeasibility:
    return PivotFeasibility(
        index=index,
        columns=columns,
        duplicate_keys=duplicate_keys,
        homogeneous=homogeneous,
        unique=unique,
        width=width,
        cells=cells,
        varying=varying,
    )


def _coverage(
    column: str,
    field: str,
    *,
    entities: int,
    covered: int,
    missing: tuple[tuple[str, str], ...] = (),
    truncated: bool = False,
) -> EnrichmentCoverage:
    return EnrichmentCoverage(
        column=column,
        field=field,
        entities=entities,
        covered=covered,
        missing=missing,
        truncated=truncated,
    )


def _profile(
    *,
    rows: int = 1,
    columns: tuple[ColumnProfile, ...] = (),
    pivots: tuple[PivotFeasibility, ...] = (),
    enrichments: tuple[EnrichmentCoverage, ...] = (),
) -> OutputProfile:
    return OutputProfile(
        schema_version=1, rows=rows, columns=columns, pivots=pivots, enrichments=enrichments
    )


class TestCapabilityVersionAndColumns:
    def test_capability_version_comes_from_the_document(self):
        document = derive_output_options(_profile(), capabilities=_capabilities())
        assert document.capability_version == _CAPABILITY_VERSION

    def test_columns_carry_the_profile_verbatim(self):
        columns = (
            _column("stat_var", "grain", droppable=False, distinct=1),
            _column("value", "value", droppable=True, distinct=None),
        )
        document = derive_output_options(_profile(columns=columns), capabilities=_capabilities())

        assert [(c.name, c.kind, c.droppable, c.distinct) for c in document.columns] == [
            ("stat_var", "grain", False, 1),
            ("value", "value", True, None),
        ]


class TestFormatsAndBundle:
    def test_a_live_format_is_available_and_a_dark_one_is_unavailable(self):
        capabilities = _capabilities(
            formats=[
                CapabilityFormat(name="csv", live=True),
                CapabilityFormat(name="xlsx", live=False),
            ]
        )
        document = derive_output_options(_profile(), capabilities=capabilities)

        by_name = {f.name: f for f in document.formats}
        assert by_name["csv"].state == "available"
        assert by_name["csv"].reason is None
        assert by_name["xlsx"].state == "unavailable"
        assert by_name["xlsx"].reason == "capability_unavailable"

    def test_bundle_state_follows_bundle_active(self):
        active = derive_output_options(_profile(), capabilities=_capabilities(bundle_active=True))
        assert active.bundle.state == "available"
        assert active.bundle.reason is None

        inactive = derive_output_options(
            _profile(), capabilities=_capabilities(bundle_active=False)
        )
        assert inactive.bundle.state == "unavailable"
        assert inactive.bundle.reason == "capability_unavailable"


class TestEnrichments:
    def test_full_coverage_is_available_with_no_missing(self):
        coverage = (_coverage("recipient", "iso3", entities=3, covered=3),)
        document = derive_output_options(
            _profile(enrichments=coverage), capabilities=_capabilities()
        )

        assert len(document.enrichments) == 1
        entry = document.enrichments[0]
        assert entry.column == "recipient"
        assert entry.field == "iso3"
        assert entry.token == "recipient_iso3"
        assert entry.state == "available"
        assert entry.reason is None
        assert entry.coverage.entities == 3
        assert entry.coverage.covered == 3
        assert entry.missing == []

    def test_partial_coverage_names_the_reason_and_the_missing_refs(self):
        coverage = (
            _coverage(
                "recipient",
                "iso3",
                entities=3,
                covered=2,
                missing=(("country/UGA", "Uganda"),),
            ),
        )
        document = derive_output_options(
            _profile(enrichments=coverage), capabilities=_capabilities()
        )

        entry = document.enrichments[0]
        assert entry.state == "partial"
        assert entry.reason == "enrichment_unavailable"
        assert entry.coverage.covered == 2
        assert [(m.ref, m.label) for m in entry.missing] == [("country/UGA", "Uganda")]

    def test_a_field_the_deployment_has_not_enabled_is_omitted(self):
        coverage = (_coverage("recipient", "iso3", entities=3, covered=3),)
        capabilities = _capabilities(enrichment_fields=["label"])

        document = derive_output_options(_profile(enrichments=coverage), capabilities=capabilities)

        assert document.enrichments == []

    def test_truncated_flag_passes_through(self):
        coverage = (_coverage("recipient", "iso3", entities=12, covered=1, truncated=True),)
        document = derive_output_options(
            _profile(enrichments=coverage), capabilities=_capabilities()
        )

        assert document.enrichments[0].truncated is True


class TestPivots:
    def test_not_unique_with_varying_is_unavailable_and_names_varying_columns(self):
        pivots = (
            _pivot(
                "recipient",
                "time_period",
                unique=False,
                duplicate_keys=True,
                varying=(("provider", 30),),
            ),
        )
        document = derive_output_options(_profile(pivots=pivots), capabilities=_capabilities())

        assert len(document.pivots) == 1
        entry = document.pivots[0]
        assert entry.index == "recipient"
        assert entry.columns == "time_period"
        assert entry.state == "unavailable"
        assert entry.reason == "pivot_not_unique"
        assert [(v.column, v.distinct) for v in entry.varying] == [("provider", 30)]
        assert entry.width is None
        assert entry.cells is None

    def test_not_unique_with_duplicate_keys_and_no_varying_reports_empty_varying(self):
        # Identical duplicate rows have no varying column but still refuse as non-unique.
        pivots = (
            _pivot(
                "recipient",
                "time_period",
                unique=False,
                duplicate_keys=True,
                varying=(),
            ),
        )
        document = derive_output_options(_profile(pivots=pivots), capabilities=_capabilities())

        assert len(document.pivots) == 1
        entry = document.pivots[0]
        assert entry.state == "unavailable"
        assert entry.reason == "pivot_not_unique"
        assert entry.varying == []

    def test_not_unique_with_empty_varying_and_no_duplicate_keys_is_pruned(self):
        # A stringification collision without duplicate keys is pruned.
        pivots = (_pivot("recipient", "time_period", unique=False, varying=()),)
        document = derive_output_options(_profile(pivots=pivots), capabilities=_capabilities())

        assert document.pivots == []

    def test_collision_and_non_homogeneous_reports_not_homogeneous_not_pruned(self):
        # Homogeneity fails first, so this collision reports `pivot_not_homogeneous`.
        pivots = (
            _pivot(
                "recipient",
                "time_period",
                unique=False,
                homogeneous=False,
                duplicate_keys=False,
                varying=(),
            ),
        )
        document = derive_output_options(_profile(pivots=pivots), capabilities=_capabilities())

        assert len(document.pivots) == 1
        entry = document.pivots[0]
        assert entry.state == "unavailable"
        assert entry.reason == "pivot_not_homogeneous"
        assert entry.varying is None

    def test_not_homogeneous_is_unavailable_with_no_varying(self):
        pivots = (_pivot("provider", "recipient", unique=True, homogeneous=False),)
        document = derive_output_options(_profile(pivots=pivots), capabilities=_capabilities())

        entry = document.pivots[0]
        assert entry.state == "unavailable"
        assert entry.reason == "pivot_not_homogeneous"
        assert entry.varying is None

    def test_feasible_pivot_is_available_with_width_and_cells(self):
        pivots = (_pivot("observationAbout", "time_period", width=11, cells=594),)
        document = derive_output_options(_profile(pivots=pivots), capabilities=_capabilities())

        entry = document.pivots[0]
        assert entry.state == "available"
        assert entry.reason is None
        assert entry.width == 11
        assert entry.cells == 594
        assert entry.unavailable_formats is None

    def test_width_over_the_excel_column_cap_is_partial_for_xlsx_only(self):
        pivots = (_pivot("recipient", "time_period", width=16_385, cells=16_385),)
        document = derive_output_options(_profile(pivots=pivots), capabilities=_capabilities())

        entry = document.pivots[0]
        assert entry.state == "partial"
        formats = {f.format: f.reason for f in entry.unavailable_formats}
        assert formats == {"xlsx": "pivot_over_column_cap"}

    def test_cells_over_the_deployment_cap_is_partial_for_every_format(self):
        capabilities = _capabilities(
            caps=_capabilities().caps.model_copy(update={"pivot_max_output_cells": 100})
        )
        pivots = (_pivot("recipient", "time_period", width=5, cells=101),)
        document = derive_output_options(_profile(pivots=pivots), capabilities=capabilities)

        entry = document.pivots[0]
        assert entry.state == "partial"
        formats = {f.format: f.reason for f in entry.unavailable_formats}
        assert formats == {
            "csv": "pivot_over_cell_cap",
            "xlsx": "pivot_over_cell_cap",
            "parquet": "pivot_over_cell_cap",
        }


class TestGroupby:
    def test_state_is_unavailable_while_summability_declarations_are_not_live(self):
        capabilities = _capabilities(summability_declarations_live=False)
        document = derive_output_options(_profile(), capabilities=capabilities)

        assert document.groupby.state == "unavailable"
        assert document.groupby.reason == "capability_unavailable"
        assert document.groupby.axes == []

    def test_state_is_available_once_summability_declarations_are_live(self):
        capabilities = _capabilities(
            summability_declarations_live=True, summable_axes=["crsPurpose"]
        )
        document = derive_output_options(_profile(), capabilities=capabilities)

        assert document.groupby.state == "available"
        assert document.groupby.reason is None
        assert document.groupby.axes == [{"name": "crsPurpose"}]

    def test_axes_are_empty_while_summability_declarations_are_not_live_even_with_axes_declared(
        self,
    ):
        # Axes render only when summability declarations are live.
        capabilities = _capabilities(
            summability_declarations_live=False, summable_axes=["crsPurpose"]
        )
        document = derive_output_options(_profile(), capabilities=capabilities)

        assert document.groupby.state == "unavailable"
        assert document.groupby.axes == []


class TestZeroRowProfile:
    def test_zero_row_profile_yields_empty_pivots_and_enrichments(self):
        columns = (_column("stat_var", "grain", droppable=False, distinct=1),)
        document = derive_output_options(
            _profile(rows=0, columns=columns), capabilities=_capabilities()
        )

        assert document.pivots == []
        assert document.enrichments == []
        assert [c.name for c in document.columns] == ["stat_var"]


class TestDesignWorkedExamples:
    """Two worked examples of a fully-populated output-options derivation,
    expressed as hand-built profiles.
    """

    def test_crs_two_role_frame(self):
        columns = (
            _column("stat_var", "grain", droppable=False, distinct=1),
            _column("provider", "role", droppable=False, distinct=30),
            _column("recipient", "role", droppable=False, distinct=54),
            _column("time_period", "grain", droppable=False, distinct=10),
            _column("value", "value", droppable=True, distinct=None),
            _column("unit", "unit", droppable=False, distinct=1),
            _column("measurementMethod", "facet", droppable=True, distinct=1),
            _column("provenance", "facet", droppable=False, distinct=1),
            _column("status", "status", droppable=True, distinct=None),
        )
        pivots = (
            _pivot(
                "recipient",
                "time_period",
                unique=False,
                duplicate_keys=True,
                varying=(("provider", 30),),
            ),
            _pivot(
                "time_period",
                "recipient",
                unique=False,
                duplicate_keys=True,
                varying=(("provider", 30),),
            ),
            _pivot(
                "provider",
                "recipient",
                unique=False,
                duplicate_keys=True,
                varying=(("time_period", 10),),
            ),
        )
        enrichments = (
            _coverage(
                "recipient",
                "iso3",
                entities=54,
                covered=51,
                missing=(("dcid:africa", "Africa"),),
            ),
            _coverage("recipient", "label", entities=54, covered=54),
        )
        document = derive_output_options(
            _profile(columns=columns, pivots=pivots, enrichments=enrichments),
            capabilities=_capabilities(),
        )

        assert len(document.pivots) == 3
        assert all(
            p.state == "unavailable" and p.reason == "pivot_not_unique" for p in document.pivots
        )

        by_field = {e.field: e for e in document.enrichments}
        assert by_field["iso3"].state == "partial"
        assert by_field["iso3"].coverage.entities == 54
        assert by_field["iso3"].coverage.covered == 51
        assert by_field["label"].state == "available"

        assert document.groupby.state == "available"
        assert document.groupby.reason is None
        assert document.groupby.axes == [
            {"name": "crsCooperationModality"},
            {"name": "crsPurpose"},
            {"name": "crsTypeOfFinance"},
            {"name": "crsTypeOfFlow"},
        ]

    def test_observation_about_single_role_frame_is_the_common_case(self):
        pivots = (
            _pivot("observationAbout", "time_period", width=11, cells=594),
            _pivot("time_period", "observationAbout", width=55, cells=550),
        )
        enrichments = (
            _coverage("observationAbout", "iso3", entities=54, covered=54),
            _coverage("observationAbout", "label", entities=54, covered=54),
        )
        document = derive_output_options(
            _profile(pivots=pivots, enrichments=enrichments), capabilities=_capabilities()
        )

        assert all(p.state == "available" for p in document.pivots)
        assert all(e.state == "available" for e in document.enrichments)
        assert all(f.state == "available" for f in document.formats)
        assert document.groupby.axes == [
            {"name": "crsCooperationModality"},
            {"name": "crsPurpose"},
            {"name": "crsTypeOfFinance"},
            {"name": "crsTypeOfFlow"},
        ]
