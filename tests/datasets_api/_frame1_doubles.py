"""Build frame1 mint fixtures for the version-record artifact tests.
Includes one 2015 CSV row, matching coverage metadata, and artifact-row readers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import httpx

from materialise.graph import Facet
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import FakeVersionRecordDatasetStore

_STAT_VAR = "MalariaGrantDisbursements"
_STAT_VAR_REF = f"dcid:{_STAT_VAR}"
_BASE_URL = "https://datacommons.one.org/core/api"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_CSV_BODY = (
    _CSV_HEADER + f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/KEN,USD,GrantEquivalent,"
    "NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2015",
    latest_date="2015",
)

_SCOPE = {
    "entities": {"recipient": ["country/KEN"]},
    "facets": {
        "provenance": {"ref": "OECD_DAC2A"},
        "measurementMethod": {"ref": "GrantEquivalent"},
        "unit": {"ref": "USD"},
    },
}


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            }
        },
        facets={(_STAT_VAR, "country/KEN"): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a", "country/KEN": "Kenya"},
    )


def _coverage_service() -> FakeCoverageService:
    """Build coverage matching `_FACET`'s role, facet, provenance, method, and unit.

    The single 2015 CSV row supplies its observation count and latest year.
    """
    index = build_coverage_index(
        [(_STAT_VAR, "facet1", "country/USA", "country/KEN", [2015])],
        role_names={_STAT_VAR: RoleNames(donor="provider", recipient="recipient")},
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


def _sdmx_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )


def _body(*, format: str, bundle: bool) -> dict:
    return {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
        "options": {"format": format, "bundle": bundle},
    }


def _version_artifact_rows(
    version_store: FakeVersionRecordDatasetStore, *, subject_id: str, content_version: str
) -> list[Mapping[str, object]]:
    with version_store.transaction() as tx:
        return tx.fetch_all(
            "SELECT artifact_digest, rendering, media_type FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s",
            (subject_id, content_version),
        )


def _frame1_row(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    matches = [row for row in rows if row["rendering"] == {"format": "frame1"}]
    assert len(matches) == 1
    return matches[0]


def _statistics_row(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    matches = [row for row in rows if row["rendering"] == {"statistics": "series"}]
    assert len(matches) == 1
    return matches[0]
