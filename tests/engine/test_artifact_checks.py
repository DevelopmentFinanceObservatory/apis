"""The `artifact_pair_stale` and `catalogue_index_drift` warnings: the pair check between
the bound catalogue and coverage builds, the drift check between the catalogue's arcs and
the code index's rows, their log lines, the settings that silence them, and their place
on a resolve response."""

from __future__ import annotations

import dataclasses
import logging

import pytest

from artifact_publication.jobs import Ingestion
from code_index.query import CodeIndex
from code_index.types import IndexStamp, StatVarRow
from origin_catalogue.arc_tables import ArcTables
from origin_catalogue.un_mirror import UnArc
from search.engine import core as search_core
from search.engine.artifact_checks import (
    index_drift,
    index_drift_warning,
    pair_stale_warning,
)
from search.engine.region.codes import ARTIFACT_PAIR_STALE, CATALOGUE_INDEX_DRIFT
from series_coverage.shared import CoverageSnapshot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import make_request, offline_resolve
from tests.origin_catalogue.fake_catalogue import FakeCatalogue
from tests.origin_catalogue.support import named_labels

_MAY = Ingestion(workflow_execution_id="exec-may", creation_timestamp="2026-05-01T00:00:00Z")
_JUNE = Ingestion(workflow_execution_id="exec-june", creation_timestamp="2026-06-01T00:00:00Z")

_ROW = StatVarRow(
    dcid="sv/Oda",
    population_type="FinancialFlow",
    measured_property="amount",
    stat_type="measuredValue",
    measurement_qualifier=None,
    measurement_denominator=None,
    observation_properties=("financingProvider", "financingRecipient"),
    constraint_properties={"crsFlowType": "CRSFlowType11"},
)

_ARCS = (
    UnArc(property="populationType", dcid="FinancialFlow"),
    UnArc(property="measuredProperty", dcid="amount"),
    UnArc(property="statType", dcid="measuredValue"),
    UnArc(property="constraintProperties", dcid="crsFlowType"),
    # The import adds this one; the graph client and the export both leave it out.
    UnArc(property="constraintProperties", dcid="sourceIndicatorCode"),
    UnArc(property="crsFlowType", dcid="CRSFlowType11"),
    UnArc(property="observationProperties", dcid="financingProvider"),
    UnArc(property="observationProperties", dcid="financingRecipient"),
    UnArc(property="typeOf", dcid="StatisticalVariable"),
)


def _coverage(ingestion: Ingestion | None) -> CoverageSnapshot:
    snapshot = empty_coverage_snapshot()
    return dataclasses.replace(
        snapshot, manifest=dataclasses.replace(snapshot.manifest, ingestion=ingestion)
    )


def _index(*rows: StatVarRow) -> CodeIndex:
    stamp = IndexStamp(
        graph_endpoint="test",
        exported_at="2026-10-11T00:00:00Z",
        code_counts={},
        statvar_shape_counts={},
        content_hash="index-hash",
    )
    return CodeIndex(stamp=stamp, declarations=(), codes=(), statvars=rows)


def _catalogue(
    arcs: dict[str, tuple[UnArc, ...]], *, ingestion: Ingestion | None = None
) -> FakeCatalogue:
    return FakeCatalogue(
        build_id="catalogue-build",
        ingestion=ingestion,
        dfo_variables=frozenset(arcs),
        arcs=ArcTables(dfo=arcs, base={}, schema={}, labels=named_labels(arcs)),
    )


# ---------------------------------------------------------------------------
# artifact_pair_stale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("catalogue_ingestion", "coverage_ingestion"),
    [(_MAY, _MAY), (None, _JUNE), (_MAY, None), (None, None)],
    ids=["same", "catalogue-unknown", "coverage-unknown", "both-unknown"],
)
def test_a_pair_from_one_ingestion_or_an_unknown_one_raises_no_warning(
    catalogue_ingestion, coverage_ingestion, caplog
):
    catalogue = FakeCatalogue(ingestion=catalogue_ingestion)
    with caplog.at_level(logging.WARNING, logger="search.engine.artifact_checks"):
        warning = pair_stale_warning(catalogue, _coverage(coverage_ingestion).manifest, warn=True)
    assert warning is None
    assert not caplog.records


def test_a_pair_from_two_ingestions_warns_naming_both_builds(caplog):
    catalogue = FakeCatalogue(build_id="catalogue-build", ingestion=_MAY)
    coverage = _coverage(_JUNE).manifest
    with caplog.at_level(logging.WARNING, logger="search.engine.artifact_checks"):
        warning = pair_stale_warning(catalogue, coverage, warn=True)

    assert warning is not None
    assert warning.code == ARTIFACT_PAIR_STALE
    for named in ("catalogue-build", coverage.build_id, "exec-may", "exec-june"):
        assert named in warning.message
    (record,) = caplog.records
    assert record.getMessage().startswith("artifact_pair_stale ")


def test_silencing_the_pair_warning_keeps_its_log_line(caplog):
    catalogue = FakeCatalogue(ingestion=_MAY)
    with caplog.at_level(logging.WARNING, logger="search.engine.artifact_checks"):
        warning = pair_stale_warning(catalogue, _coverage(_JUNE).manifest, warn=False)
    assert warning is None
    assert [r.getMessage().split()[0] for r in caplog.records] == ["artifact_pair_stale"]


# ---------------------------------------------------------------------------
# catalogue_index_drift
# ---------------------------------------------------------------------------


def _without(property: str) -> tuple[UnArc, ...]:
    return tuple(arc for arc in _ARCS if arc.property != property)


@pytest.mark.parametrize(
    ("arcs", "aspects"),
    [
        (_ARCS, ()),
        ((*_without("statType"), UnArc(property="statType", dcid="meanValue")), ("five_tuple",)),
        (
            (*_ARCS, UnArc(property="measurementDenominator", dcid="Count_Person")),
            ("five_tuple",),
        ),
        (
            (*_without("crsFlowType"), UnArc(property="crsFlowType", dcid="CRSFlowType13")),
            ("constraints",),
        ),
        (
            (
                *_ARCS,
                UnArc(property="constraintProperties", dcid="crsChannel"),
                UnArc(property="crsChannel", dcid="CRSChannel1"),
            ),
            ("constraints",),
        ),
        (_without("observationProperties"), ("roles",)),
        ((), ("five_tuple", "constraints", "roles")),
    ],
    ids=[
        "agree",
        "stat-type",
        "denominator",
        "constraint-value",
        "constraint-added",
        "roles",
        "held-and-empty",
    ],
)
def test_index_drift_names_each_aspect_that_disagrees(arcs, aspects):
    assert index_drift(_ROW, arcs) == aspects


def test_drift_on_a_served_variable_warns_naming_it_and_its_aspects(caplog):
    response = offline_resolve(make_request("population of Kenya"))
    row = dataclasses.replace(_ROW, dcid="Count_Person")
    catalogue = _catalogue({"Count_Person": _without("observationProperties")})
    with caplog.at_level(logging.WARNING, logger="search.engine.artifact_checks"):
        warning = index_drift_warning(response, catalogue=catalogue, index=_index(row), warn=True)

    assert warning is not None
    assert warning.code == CATALOGUE_INDEX_DRIFT
    assert "Count_Person (roles)" in warning.message
    assert "catalogue-build" in warning.message
    (record,) = caplog.records
    assert record.getMessage().startswith("catalogue_index_drift ")


@pytest.mark.parametrize(
    ("index_rows", "arcs"),
    [
        ((), {"Count_Person": ()}),
        ((dataclasses.replace(_ROW, dcid="Count_Person"),), {}),
        ((dataclasses.replace(_ROW, dcid="Count_Person"),), {"Count_Person": _ARCS}),
    ],
    ids=["not-in-the-index", "not-held-by-the-catalogue", "agree"],
)
def test_no_drift_warning_without_two_disagreeing_sources(index_rows, arcs):
    response = offline_resolve(make_request("population of Kenya"))
    warning = index_drift_warning(
        response, catalogue=_catalogue(arcs), index=_index(*index_rows), warn=True
    )
    assert warning is None


def test_silencing_the_drift_warning_keeps_its_log_line(caplog):
    response = offline_resolve(make_request("population of Kenya"))
    row = dataclasses.replace(_ROW, dcid="Count_Person")
    catalogue = _catalogue({"Count_Person": ()})
    with caplog.at_level(logging.WARNING, logger="search.engine.artifact_checks"):
        warning = index_drift_warning(response, catalogue=catalogue, index=_index(row), warn=False)
    assert warning is None
    assert [r.getMessage().split()[0] for r in caplog.records] == ["catalogue_index_drift"]


# ---------------------------------------------------------------------------
# On a resolve response
# ---------------------------------------------------------------------------


class _CoverageService:
    def __init__(self, snapshot: CoverageSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self) -> CoverageSnapshot:
        return self._snapshot


@pytest.fixture
def drifted_stale_pair(bind_catalogue, monkeypatch):
    """A bound catalogue whose ingestion differs from the coverage build's and whose arcs
    for the answer's StatVar disagree with the code index on its roles."""
    bind_catalogue(_catalogue({"Count_Person": _without("observationProperties")}, ingestion=_MAY))
    monkeypatch.setattr(
        search_core, "get_coverage_service", lambda: _CoverageService(_coverage(_JUNE))
    )
    monkeypatch.setattr(
        search_core, "get_index", lambda: _index(dataclasses.replace(_ROW, dcid="Count_Person"))
    )


def _codes() -> list[str]:
    response = offline_resolve(make_request("population of Kenya"))
    return [warning.code for warning in response.diagnostics.warnings]


@pytest.mark.usefixtures("drifted_stale_pair")
def test_a_response_carries_both_warnings_once():
    codes = _codes()
    assert codes.count(ARTIFACT_PAIR_STALE) == 1
    assert codes.count(CATALOGUE_INDEX_DRIFT) == 1


@pytest.mark.usefixtures("drifted_stale_pair")
@pytest.mark.parametrize(
    ("variable", "silenced", "kept"),
    [
        ("ORIGIN_CATALOGUE_WARN_PAIR_STALE", ARTIFACT_PAIR_STALE, CATALOGUE_INDEX_DRIFT),
        ("ORIGIN_CATALOGUE_WARN_INDEX_DRIFT", CATALOGUE_INDEX_DRIFT, ARTIFACT_PAIR_STALE),
    ],
)
def test_the_environment_silences_either_warning(monkeypatch, variable, silenced, kept):
    monkeypatch.setenv(variable, "false")
    codes = _codes()
    assert silenced not in codes
    assert kept in codes
