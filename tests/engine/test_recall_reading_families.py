"""Recall adds a row of each family of a recalled five-tuple that reads a code of the
variable no recalled family of that five-tuple reads (`clients.graph.with_reading_families`,
applied by `interpret._detect`).

The index is built here: the committed fixture index holds no climate code. It mirrors the
dev graph's aid tables on two five-tuples. On the FinancialFlow amount the CRS rows hold
health and agriculture sectors and ONE's climate finance rows the climate total and its
adaptation and mitigation parts; on the FinancialObligation amount DAC1 holds an ODA
commitments total by donor and ONE's climate finance rows the same objectives by donor and
recipient. A World Bank series and
an SDG series stand outside the index.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from code_index.declarations import CODE_SYNONYMS
from code_index.query import CodeIndex
from code_index.types import (
    CodeRow,
    CodeSystemDeclaration,
    DatasetDeclaration,
    IndexStamp,
    StatVarRow,
)
from search.engine.interpret import recall
from search.engine.quantity_read import QuantityOutput
from search.engine.region.pipeline import resolve_variable
from series_coverage.schema import ABSENT_ENTITY, RoleNames
from tests.engine._binders import MenuRecordingBinder, elected_reply
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.graph_release.support import serving_index

_CRS = "provenance/OECDCreditorReportingSystem"
_DAC1 = "provenance/OECDDAC1"
_CLIMATE = "provenance/ONEClimateFinance"

_TOTAL = "ClimateChangePolicyObjective"
_ADAPTATION = "AdaptationOnlyClimateChangePolicyObjective"
_MITIGATION = "MitigationOnlyClimateChangePolicyObjective"

_DEU = "country/DEU"
_KEN = "country/KEN"
_ROLES = ("financingProvider", "financingRecipient")


def _row(
    dcid: str,
    population_type: str,
    roles: tuple[str, ...],
    constraints: dict[str, str],
    provenance: str,
) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type=population_type,
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier="Nominal",
        measurement_denominator=None,
        observation_properties=roles,
        constraint_properties=constraints,
        provenances=(provenance,),
    )


def _flow(dcid: str, constraint: dict[str, str], provenance: str) -> StatVarRow:
    return _row(
        dcid,
        "FinancialFlow",
        _ROLES,
        {"financialFlowType": "Disbursement", "nettingBasis": "Gross", **constraint},
        provenance,
    )


def _obligation(
    dcid: str, constraint: dict[str, str], provenance: str, roles: tuple[str, ...] = _ROLES
) -> StatVarRow:
    return _row(
        dcid,
        "FinancialObligation",
        roles,
        {"financialObligationType": "Commitment", **constraint},
        provenance,
    )


_CRS_HEALTH = _flow("TESTFIXTURE/Closure/CrsHealth", {"crsPurpose": "CRSPurposeCode120"}, _CRS)
_CRS_AGRICULTURE = _flow(
    "TESTFIXTURE/Closure/CrsAgriculture", {"crsPurpose": "CRSPurposeCode311"}, _CRS
)
_CLIMATE_TOTAL = _flow(
    "TESTFIXTURE/Closure/ClimateTotal", {"climatePolicyObjective": _TOTAL}, _CLIMATE
)
_CLIMATE_ADAPTATION = _flow(
    "TESTFIXTURE/Closure/ClimateAdaptation", {"climatePolicyObjective": _ADAPTATION}, _CLIMATE
)
_CLIMATE_MITIGATION = _flow(
    "TESTFIXTURE/Closure/ClimateMitigation", {"climatePolicyObjective": _MITIGATION}, _CLIMATE
)
_DAC1_COMMITMENTS = _obligation(
    "TESTFIXTURE/Closure/Dac1OdaCommitments",
    {"dac1AidType": "DAC1AidType1010"},
    _DAC1,
    roles=("financingProvider",),
)
_CLIMATE_TOTAL_COMMITMENTS = _obligation(
    "TESTFIXTURE/Closure/ClimateTotalCommitments", {"climatePolicyObjective": _TOTAL}, _CLIMATE
)
_CLIMATE_ADAPTATION_COMMITMENTS = _obligation(
    "TESTFIXTURE/Closure/ClimateAdaptationCommitments",
    {"climatePolicyObjective": _ADAPTATION},
    _CLIMATE,
)
_CLIMATE_MITIGATION_COMMITMENTS = _obligation(
    "TESTFIXTURE/Closure/ClimateMitigationCommitments",
    {"climatePolicyObjective": _MITIGATION},
    _CLIMATE,
)
_ROWS = (
    _CRS_HEALTH,
    _CRS_AGRICULTURE,
    _CLIMATE_TOTAL,
    _CLIMATE_ADAPTATION,
    _CLIMATE_MITIGATION,
    _DAC1_COMMITMENTS,
    _CLIMATE_TOTAL_COMMITMENTS,
    _CLIMATE_ADAPTATION_COMMITMENTS,
    _CLIMATE_MITIGATION_COMMITMENTS,
)
_SDG_CLIMATE = "undata/sdg/series-dc_fin_climt_value_measuredvalue"
_WB_AGRICULTURE = "TESTFIXTURE/Closure/AgricultureValueAdded"


def _code(
    dcid: str, system: str, label: str, parent: str | None = None, code: str | None = None
) -> CodeRow:
    return CodeRow(
        dcid=dcid,
        code_system=system,
        code=code,
        label=label,
        synonyms=CODE_SYNONYMS.get(dcid, ()),
        parent=parent,
    )


_CODES = (
    _code(_TOTAL, "climatePolicyObjective", "climate change policy objective"),
    _code(
        _ADAPTATION,
        "climatePolicyObjective",
        "adaptation-only climate change policy objective",
        _TOTAL,
    ),
    _code(
        _MITIGATION,
        "climatePolicyObjective",
        "mitigation-only climate change policy objective",
        _TOTAL,
    ),
    _code("CRSPurposeCode120", "crsPurpose", "health", code="120"),
    _code("CRSPurposeCode311", "crsPurpose", "agriculture", code="311"),
    _code("DAC1AidType1010", "dac1AidType", "official development assistance", code="1010"),
)
_SYSTEMS = {
    "climatePolicyObjective": (True, "climate objective"),
    "crsPurpose": (False, "sector"),
    "dac1AidType": (False, "aid type"),
}


_LABELS = {
    _CRS: "OECD Creditor Reporting System",
    _DAC1: "OECD DAC1",
    _CLIMATE: "ONE Data Climate Finance",
}


def _dataset(provenance: str, rank: int) -> DatasetDeclaration:
    return DatasetDeclaration(
        provenance=provenance,
        label=_LABELS[provenance],
        description="A test table.",
        opt_in=False,
        absent_observation="unknown",
        account="provider",
        rank=rank,
        invariants={},
    )


def _index(codes: tuple[CodeRow, ...] = _CODES) -> CodeIndex:
    """The test index, holding `codes`."""
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={
                system: sum(code.code_system == system for code in codes) for system in _SYSTEMS
            },
            statvar_shape_counts={},
            content_hash="recall-reading-families",
        ),
        declarations=tuple(
            CodeSystemDeclaration(
                code_system=system,
                hierarchical=hierarchical,
                disjoint_from_children=not hierarchical,
                summable=not hierarchical,
                absent_reading="not_declared",
                plain_name=plain_name,
            )
            for system, (hierarchical, plain_name) in _SYSTEMS.items()
        ),
        codes=codes,
        statvars=_ROWS,
        datasets=(_dataset(_CRS, 0), _dataset(_DAC1, 1), _dataset(_CLIMATE, 2)),
    )


def _graph(
    query: str, recalled: tuple[tuple[str, float], ...], series: dict[str, str] | None = None
) -> FakeGraph:
    """The test graph, recalling `recalled` for `query`, with `series` (dcid to label) as
    further series outside the index."""
    return FakeGraph(
        nodes={
            **{row.dcid: _row_node(row) for row in _ROWS},
            _SDG_CLIMATE: _series_node("Climate-specific financial support provided"),
            _WB_AGRICULTURE: _series_node("Agriculture, value added (% of GDP)"),
            **{dcid: _series_node(label) for dcid, label in (series or {}).items()},
            _DEU: {"label": "Germany", "type": "Country"},
            _KEN: {"label": "Kenya", "type": "Country"},
        },
        detect={
            query: {
                "svs": [sv for sv, _score in recalled],
                "cosine_scores": [score for _sv, score in recalled],
            }
        },
        resolve={"Germany": _DEU, "Kenya": _KEN},
        obs={},
    )


def _row_node(row: StatVarRow) -> dict:
    """The graph node of index row `row`, with the roles and constraints it declares."""
    return {
        "label": row.dcid,
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "populationType": {"nodes": [{"dcid": row.population_type}]},
            "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
            "statType": {"nodes": [{"dcid": row.stat_type}]},
            "measurementQualifier": {"nodes": [{"dcid": row.measurement_qualifier}]},
            "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
            **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
            "observationProperties": {"nodes": [{"dcid": r} for r in row.observation_properties]},
        },
    }


def _series_node(label: str) -> dict:
    return {
        "label": label,
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": "value"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    }


@pytest.mark.parametrize(
    ("variable", "recalled", "joined"),
    [
        pytest.param(
            "adaptation disbursements",
            ((_CRS_HEALTH.dcid, 0.8),),
            {_CLIMATE_TOTAL.dcid},
            id="adaptation-beside-crs",
        ),
        pytest.param(
            "climate finance",
            ((_DAC1_COMMITMENTS.dcid, 0.8), (_SDG_CLIMATE, 0.7)),
            {_CLIMATE_TOTAL_COMMITMENTS.dcid},
            id="climate-finance-beside-dac1-and-sdg",
        ),
        pytest.param("health aid", ((_CRS_HEALTH.dcid, 0.8),), set(), id="health-read-by-crs"),
        pytest.param(
            "agriculture value added",
            ((_WB_AGRICULTURE, 0.8),),
            set(),
            id="agriculture-beside-a-series-outside-the-index",
        ),
    ],
)
def test_a_family_reading_an_unread_code_of_a_recalled_five_tuple_joins_the_pool(
    variable, recalled, joined
):
    """The climate row joins as a rival row at the best score of its five-tuple; the
    controls add nothing, since the CRS reads "health" itself and the World Bank series
    belongs to no five-tuple of the index."""
    graph = _graph(variable, recalled)
    with serving_index(_index()):
        result = asyncio.run(
            recall(variable, [], graph=graph, resolver=make_resolver(graph), raw_query=variable)
        )

    recalled_dcids = {dcid for dcid, _score in recalled}
    assert result.hits is not None
    added = {hit.dcid: hit for hit in result.hits if hit.dcid not in recalled_dcids}
    assert set(added) == joined
    assert all(hit.rival and hit.score == recalled[0][1] for hit in added.values())
    assert {c.dcid for c in result.candidates} == recalled_dcids | joined


def _coverage() -> FakeCoverageService:
    """Each index row's series from Germany, to Kenya where the row declares a recipient,
    in 2021 and 2022, one facet per dataset."""
    return FakeCoverageService(
        build_coverage_index(
            [
                (
                    row.dcid,
                    row.provenances[0],
                    _DEU,
                    _KEN if "financingRecipient" in row.observation_properties else ABSENT_ENTITY,
                    [2021, 2022],
                )
                for row in _ROWS
            ],
            role_names={
                row.dcid: RoleNames(
                    donor="financingProvider",
                    recipient="financingRecipient"
                    if "financingRecipient" in row.observation_properties
                    else None,
                )
                for row in _ROWS
            },
        )
    )


class _ElectFirst:
    """Elects the first shape on the menu with no bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0), None


@pytest.mark.parametrize(
    ("query", "variable", "recalled", "climate", "objective"),
    [
        pytest.param(
            "germany adaptation disbursements 2022",
            "adaptation disbursements",
            ((_CRS_HEALTH.dcid, 0.8),),
            _CLIMATE_TOTAL,
            _ADAPTATION,
            id="adaptation-disbursements",
        ),
        pytest.param(
            "climate finance from Germany in 2022",
            "climate finance",
            ((_DAC1_COMMITMENTS.dcid, 0.8), (_SDG_CLIMATE, 0.7)),
            _CLIMATE_TOTAL_COMMITMENTS,
            _TOTAL,
            id="climate-finance",
        ),
    ],
)
def test_the_injected_climate_family_is_elected_with_the_objective_the_query_reads(
    query, variable, recalled, climate, objective, caplog
):
    """The injected family is logged as injected, displaces the recalled family of its
    five-tuple that reads none of the query (`elect_five_tuple_family`), and answers with
    the objective the variable names."""
    binder = MenuRecordingBinder()
    graph = _graph(query, recalled)
    with serving_index(_index()), caplog.at_level(logging.INFO):
        coverage = _coverage()
        result = asyncio.run(
            resolve_variable(
                variable,
                entities=["Germany"],
                date_request=None,
                detect_query=query,
                role_query=query,
                entity_roles=door_roles_for_test_query(query, ["Germany"]),
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                llm=_ElectFirst(),
                binder=binder,
                sdmx=FakeSdmxClient(observations={}),
                base_timing={},
                coverage=coverage.snapshot(),
            )
        )

    pool_logs = [r.getMessage() for r in caplog.records if r.getMessage().startswith("shape pool")]
    assert len(pool_logs) == 1 and "1 injected" in pool_logs[0], pool_logs
    assert len(binder.menus) == 1
    menu = binder.menus[0]
    assert climate.dcid in menu[0], menu
    assert all(recalled[0][0] not in shape for shape in menu), menu
    assert result.status == "definite", result
    identity = result.specs[0].definition.identity
    slots = {slot.property: slot.binding for slot in identity.slots}
    assert slots["climatePolicyObjective"].ref == objective, identity
