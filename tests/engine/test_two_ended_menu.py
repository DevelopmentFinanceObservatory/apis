"""A query naming both flow ends elects among role-bearing shapes only.

A subject-only shape has one subject slot, so it cannot hold an origin and a destination
at once, and `place_roles.bind_place_roles` refuses it ``role_incompatible``. When the
query's places name both ends and a role-bearing shape survives the score floor,
`_phase_prune_and_rank` drops subject-only shapes before the bind menu is built
(`subject_shapes_refused`). A query
naming one end or none keeps every shape on the menu. The fixtures mirror "Kenya's debt to
China" on the dev graph: a subject-only government debt series outranks the by-creditor
series that declares both roles.

A role-bearing shape that declares no role on an axis the query needs, a directed place's
or an operation's, leaves the menu the same way (`lacks_query_axis`): a creditor-only
series, ranked first, cannot hold "to Kenya" or rank debtors.

An axis the query asks every member of removes a shape only when the code index carries
the shape and another role-bearing shape declares the axis (`without_open_axis_gaps`). The
index fixtures are packaged rows: a CRS flow, whose dataset declares donor and recipient,
and a DAC1 provider total, whose dataset declares the donor alone. A synthetic index holds
a provider-only family beside a recipient-only one, for a query that opens both axes.
"""

from __future__ import annotations

import asyncio

import pytest

from code_index import shared as code_index_shared
from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.types import DatasetDeclaration, IndexStamp, StatVarRow
from search.engine.bind import _UniversalBindOutput
from search.engine.region import phase_rank_bind
from search.engine.region.pipeline import resolve_variable
from search.models import Operation, RoleAxis
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.fixtures import FakeGraph, make_resolver

_KEN = "country/KEN"
_CHN = "country/CHN"
_SUBJECT_SV = "TESTFIXTURE/TwoEndedMenu/DebtGovernment"
_FLOW_SV = "TESTFIXTURE/TwoEndedMenu/DebtByCreditor"
_CREDITOR_SV = "TESTFIXTURE/TwoEndedMenu/DebtLentByCreditor"
_DEBTOR_SV = "TESTFIXTURE/TwoEndedMenu/DebtOwedByDebtor"
_FRA = "country/FRA"
_CRS_SV = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSPurposeCode11110_Nominal"
)
_DAC1_SV = "Amount_FinancialFlow_Gross_Disbursement_DAC1AidType1010_Nominal"
# One of the two DAC1 rows that hold a recipient beside the donor.
_DAC1_FLOW_SV = "Amount_FinancialFlow_Net_Disbursement_DAC1AidType8863_Nominal"


_ONE_END_DATASET = "provenance/TwoEndedMenu"
_PROVIDER_ROW = StatVarRow(
    dcid="TESTFIXTURE/TwoEndedMenu/GivenByProvider",
    population_type="FinancialFlow",
    measured_property="amount",
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
    measurement_denominator=None,
    observation_properties=("financingProvider",),
    constraint_properties={"flowType": "Disbursement"},
    provenances=(_ONE_END_DATASET,),
)
_RECIPIENT_ROW = StatVarRow(
    dcid="TESTFIXTURE/TwoEndedMenu/ReceivedByRecipient",
    population_type="FinancialFlow",
    measured_property="amountReceived",
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
    measurement_denominator=None,
    observation_properties=("financingRecipient",),
    constraint_properties={"flowType": "Disbursement"},
    provenances=(_ONE_END_DATASET,),
)


def _one_end_index() -> CodeIndex:
    """An index of two one-row families of one dataset: a provider-only flow and, on
    another measured property, a recipient-only one."""
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="two-ended-menu/one-end",
        ),
        declarations=(),
        codes=(),
        statvars=(_PROVIDER_ROW, _RECIPIENT_ROW),
        datasets=(
            DatasetDeclaration(
                provenance=_ONE_END_DATASET,
                label=_ONE_END_DATASET,
                description="A test table.",
                opt_in=False,
                absent_observation="zero",
                rank=0,
                invariants={},
            ),
        ),
    )


def _debt_arcs(qualifier: str, roles: tuple[str, ...] = ()) -> dict:
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": qualifier}]},
    }
    if roles:
        arcs["observationProperties"] = {"nodes": [{"dcid": role} for role in roles]}
    return arcs


def _index_row_node(found: str | StatVarRow) -> dict:
    """The graph node of the index row `found`, a row or the dcid of a packaged one, with
    the roles and constraints the row declares."""
    row = (
        found
        if isinstance(found, StatVarRow)
        else next(r for r in get_index().statvars if r.dcid == found)
    )
    dcid = row.dcid
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "measurementQualifier": {"nodes": [{"dcid": row.measurement_qualifier}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
        "observationProperties": {"nodes": [{"dcid": r} for r in row.observation_properties]},
    }
    return {"label": dcid, "arcs": arcs}


_NODES = {
    _SUBJECT_SV: {"label": "Government debt", "arcs": _debt_arcs("Government")},
    _FLOW_SV: {
        "label": "Debt by creditor",
        "arcs": _debt_arcs("ByCreditor", ("creditor", "debtor")),
    },
    _CREDITOR_SV: {
        "label": "Debt lent by creditor",
        "arcs": _debt_arcs("LentByCreditor", ("creditor",)),
    },
    _DEBTOR_SV: {
        "label": "Debt owed by debtor",
        "arcs": _debt_arcs("OwedByDebtor", ("debtor",)),
    },
    _CRS_SV: _index_row_node(_CRS_SV),
    _DAC1_SV: _index_row_node(_DAC1_SV),
    _DAC1_FLOW_SV: _index_row_node(_DAC1_FLOW_SV),
    _PROVIDER_ROW.dcid: _index_row_node(_PROVIDER_ROW),
    _RECIPIENT_ROW.dcid: _index_row_node(_RECIPIENT_ROW),
    _FRA: {"label": "France", "type": "Country"},
    _KEN: {"label": "Kenya", "type": "Country"},
    _CHN: {"label": "China", "type": "Country"},
}


class _ElectFirst:
    """Elects the first shape on the menu with no bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


def _resolve(
    monkeypatch,
    query: str,
    entities: list[str],
    *,
    variable: str = "debt",
    recalled: tuple[tuple[str, float], ...] = ((_SUBJECT_SV, 0.9), (_FLOW_SV, 0.8)),
    operations: tuple[Operation, ...] = (),
    open_axes: frozenset[RoleAxis] = frozenset(),
):
    """Resolve `query` over the `recalled` StatVars and their scores, and return the result
    with the StatVars of each bind-menu shape."""
    menus: list[list[tuple[str, ...]]] = []
    real_bind = phase_rank_bind.bind_universal

    async def _spy(variable, shapes, **kwargs):
        menus.append([tuple(shape.sv_arc_facts or ()) for shape in shapes])
        return await real_bind(variable, shapes, **kwargs)

    monkeypatch.setattr(phase_rank_bind, "bind_universal", _spy)
    graph = FakeGraph(
        nodes=_NODES,
        detect={
            query: {
                "svs": [sv for sv, _score in recalled],
                "cosine_scores": [score for _sv, score in recalled],
            }
        },
        resolve={"Kenya": _KEN, "China": _CHN, "France": _FRA},
        obs={},
    )
    coverage = FakeCoverageService(
        build_coverage_index(
            [(_FLOW_SV, "f1", _CHN, _KEN, [2020, 2021, 2022])],
            role_names={_FLOW_SV: RoleNames(donor="creditor", recipient="debtor")},
        )
    )
    result = asyncio.run(
        resolve_variable(
            variable,
            entities=entities,
            date_request=None,
            detect_query=query,
            role_query=query,
            entity_roles=door_roles_for_test_query(query, entities),
            operations=operations,
            open_axes=open_axes,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_ElectFirst(),
            sdmx=FakeSdmxClient(observations={}),
            base_timing={},
            coverage=coverage.snapshot(),
        )
    )
    assert len(menus) == 1, menus
    return result, menus[0]


def test_a_two_ended_query_elects_the_role_bearing_shape(monkeypatch) -> None:
    """With China as origin and Kenya as destination the menu holds only the by-creditor
    series, so the top-ranked government debt series cannot be elected and refuse."""
    result, menu = _resolve(monkeypatch, "debt from China to Kenya", ["China", "Kenya"])

    assert menu == [(_FLOW_SV,)]
    assert result.status == "definite", result
    spec = result.specs[0]
    assert [item.ref.dcid for item in spec.resolved.members] == [_FLOW_SV]
    assert spec.definition.identity.entities == {"origin": [_CHN], "destination": [_KEN]}


@pytest.mark.parametrize(
    ("query", "entities"),
    [
        pytest.param("debt to Kenya", ["Kenya"], id="one-flow-end"),
        pytest.param("debt of Kenya", ["Kenya"], id="no-flow-end"),
        pytest.param("debt of Kenya and China", ["Kenya", "China"], id="two-undirected-places"),
    ],
)
def test_a_query_naming_one_flow_end_or_none_keeps_every_shape(
    monkeypatch, query: str, entities: list[str]
) -> None:
    """The subject-only series stays on the menu, ranked first by its recall score."""
    _result, menu = _resolve(monkeypatch, query, entities)

    assert menu == [(_SUBJECT_SV,), (_FLOW_SV,)]


@pytest.mark.parametrize(
    ("query", "operations"),
    [
        pytest.param("debt to Kenya", (), id="directed-place"),
        pytest.param(
            "top debtors of China",
            (Operation(verb="rank", phrase="top debtors", over="destination"),),
            id="operation-over-the-axis",
        ),
    ],
)
def test_a_shape_without_an_axis_the_query_needs_leaves_the_menu(
    monkeypatch, query: str, operations: tuple[Operation, ...]
) -> None:
    """The creditor-only series ranks first but declares no debtor, so the menu holds the
    by-creditor series that does."""
    entities = ["Kenya"] if "Kenya" in query else ["China"]
    _result, menu = _resolve(
        monkeypatch,
        query,
        entities,
        recalled=((_CREDITOR_SV, 0.95), (_FLOW_SV, 0.8)),
        operations=operations,
    )

    assert menu == [(_FLOW_SV,)]


@pytest.mark.parametrize(
    ("query", "open_axis", "recalled"),
    [
        pytest.param("debt from China", "destination", (_CREDITOR_SV,), id="the-only-shape"),
        pytest.param(
            "debt from China",
            "destination",
            (_CREDITOR_SV, _SUBJECT_SV),
            id="beside-a-subject-only-shape",
        ),
        pytest.param(
            "debt from China",
            "destination",
            (_CREDITOR_SV, _FLOW_SV),
            id="beside-a-shape-declaring-the-axis",
        ),
        pytest.param(
            "debt of Kenya", "origin", (_DEBTOR_SV, _FLOW_SV), id="a-sample-holding-only-totals"
        ),
    ],
)
def test_an_open_axis_keeps_a_shape_the_index_does_not_carry(
    monkeypatch, query: str, open_axis: RoleAxis, recalled: tuple[str, ...]
) -> None:
    """Every member of `open_axis` is asked for and the first-ranked series declares no
    role there. The index does not carry it, so its recalled members do not show what its
    family declares, and it stays whatever is recalled beside it."""
    _result, menu = _resolve(
        monkeypatch,
        query,
        ["Kenya"] if "Kenya" in query else ["China"],
        recalled=tuple((sv, 0.95 - 0.05 * rank) for rank, sv in enumerate(recalled)),
        open_axes=frozenset({open_axis}),
    )

    assert menu == [(sv,) for sv in recalled]


@pytest.mark.parametrize(
    ("open_axes", "recalled", "expected"),
    [
        pytest.param(
            frozenset({"destination"}),
            (_DAC1_SV, _CRS_SV),
            [(_CRS_SV,)],
            id="beside-a-shape-declaring-the-axis",
        ),
        pytest.param(
            frozenset({"destination"}),
            (_DAC1_FLOW_SV, _CRS_SV),
            [(_CRS_SV,)],
            id="recalled-by-a-member-holding-the-axis",
        ),
        pytest.param(
            frozenset({"destination"}),
            (_DAC1_SV, _SUBJECT_SV),
            [(_DAC1_SV,), (_SUBJECT_SV,)],
            id="beside-a-subject-only-shape",
        ),
        pytest.param(
            frozenset(), (_DAC1_SV, _CRS_SV), [(_DAC1_SV,), (_CRS_SV,)], id="no-open-axis"
        ),
    ],
)
def test_an_open_axis_removes_an_index_shape_only_for_one_declaring_it(
    monkeypatch,
    open_axes: frozenset[RoleAxis],
    recalled: tuple[str, ...],
    expected: list[tuple[str, ...]],
) -> None:
    """ "Aid from France to every recipient" recalls a DAC1 shape first. Its dataset
    declares no recipient, whichever member recall confirmed, so it leaves for the CRS flow
    that lists recipients, and stays when no role-bearing shape does or no recipient is
    asked for."""
    _result, menu = _resolve(
        monkeypatch,
        "aid from France",
        ["France"],
        variable="aid",
        recalled=tuple((sv, 0.95 - 0.05 * rank) for rank, sv in enumerate(recalled)),
        open_axes=open_axes,
    )

    assert menu == expected


@pytest.mark.parametrize(
    "recalled",
    [
        pytest.param((_PROVIDER_ROW.dcid, _RECIPIENT_ROW.dcid), id="provider-only-first"),
        pytest.param((_RECIPIENT_ROW.dcid, _PROVIDER_ROW.dcid), id="recipient-only-first"),
    ],
)
def test_two_open_axes_keep_the_index_shape_declaring_the_destination(
    monkeypatch, recalled: tuple[str, ...]
) -> None:
    """ "Aid from every donor to every recipient" names no place and recalls two index
    shapes, each declaring one of the two axes. The axes are read in sorted order, so the
    destination is read first: the provider-only shape leaves for the recipient-only one,
    whichever ranks first, and no shape left declares the origin, so that one stays."""
    with code_index_shared._singleton.poisoned_for_test(_one_end_index()):
        _result, menu = _resolve(
            monkeypatch,
            "aid",
            [],
            variable="aid",
            recalled=tuple((sv, 0.95 - 0.05 * rank) for rank, sv in enumerate(recalled)),
            open_axes=frozenset({"origin", "destination"}),
        )

    assert menu == [(_RECIPIENT_ROW.dcid,)]
