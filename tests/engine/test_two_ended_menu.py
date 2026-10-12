"""A query naming both flow ends elects among role-bearing shapes only.

A subject-only shape has one subject slot, so it cannot hold an origin and a destination
at once, and `place_roles.bind_place_roles` refuses it ``role_incompatible``. When the
query's places name both ends and a role-bearing shape survives the score floor,
`_phase_prune_and_rank` drops subject-only shapes before the bind menu is built
(`subject_shapes_refused`). A query naming one end or none keeps every shape on the menu.
The fixtures mirror "Kenya's debt to
China" on the dev graph: a subject-only government debt series outranks the by-creditor
series that declares both roles.

A role-bearing shape that declares no role on an axis the query needs, a directed place's
or an operation's, leaves the menu the same way (`lacks_query_axis`): a creditor-only
series, ranked first, cannot hold "to Kenya" or rank debtors.

The converse holds once the per-vocabulary readings exist (`without_unasked_axes`). A
role-bearing shape publishing on an axis the query neither needs nor opens leaves for one
that reads every query phrase it reads without that axis, so "aid from France" reads DAC1's
donor totals over CRS flows by recipient.

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
_DAC1_MULTILATERAL_SV = "Amount_FinancialFlow_Gross_Disbursement_DAC1AidType2000_Nominal"
_DAC2A_IMPUTED_SV = "AttributedAmount_FinancialFlow_Gross_Disbursement_DAC2aAidType106_Nominal"
_DAC2A_ODA_SV = "Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal"
_DEU = "country/DEU"
_NOR = "country/NOR"
_JPN = "country/JPN"
_CRS_COMMITMENT_SV = (
    "Amount_FinancialObligation_Commitment_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSPurposeCode11110_Nominal"
)
# The DAC1 commitments total recall adds beside CRS commitments
# (`clients.graph.with_rival_families`).
_DAC1_COMMITMENT_SV = "Amount_FinancialObligation_Commitment_DAC1AidType1010_Nominal"
_SECTORS_SV = (
    "AttributedAmount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_DAC2aAidType106_"
    "CRSPurposeCode11220_Nominal"
)
_CRS_GRANT_EQUIVALENT_SV = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSPurposeCode99810_Nominal"
)


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
                account="provider",
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
    _DAC1_MULTILATERAL_SV: _index_row_node(_DAC1_MULTILATERAL_SV),
    _DAC2A_IMPUTED_SV: _index_row_node(_DAC2A_IMPUTED_SV),
    _DAC2A_ODA_SV: _index_row_node(_DAC2A_ODA_SV),
    _CRS_GRANT_EQUIVALENT_SV: _index_row_node(_CRS_GRANT_EQUIVALENT_SV),
    _SECTORS_SV: _index_row_node(_SECTORS_SV),
    _CRS_COMMITMENT_SV: _index_row_node(_CRS_COMMITMENT_SV),
    _DAC1_COMMITMENT_SV: _index_row_node(_DAC1_COMMITMENT_SV),
    _PROVIDER_ROW.dcid: _index_row_node(_PROVIDER_ROW),
    _RECIPIENT_ROW.dcid: _index_row_node(_RECIPIENT_ROW),
    _FRA: {"label": "France", "type": "Country"},
    _KEN: {"label": "Kenya", "type": "Country"},
    _CHN: {"label": "China", "type": "Country"},
    _DEU: {"label": "Germany", "type": "Country"},
    _NOR: {"label": "Norway", "type": "Country"},
    _JPN: {"label": "Japan", "type": "Country"},
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
        resolve={
            "Kenya": _KEN,
            "China": _CHN,
            "France": _FRA,
            "Germany": _DEU,
            "Norway": _NOR,
            "Japan": _JPN,
        },
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
        # With no recipient asked for, the CRS flow publishes on an axis the query does not
        # name and DAC1 reads the query as well without it (`without_unasked_axes`).
        pytest.param(frozenset(), (_DAC1_SV, _CRS_SV), [(_DAC1_SV,)], id="no-open-axis"),
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
    that lists recipients, and stays when no role-bearing shape does. With no recipient
    asked for, the CRS flow leaves instead."""
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


@pytest.mark.parametrize(
    ("query", "entities", "variable", "recalled", "expected"),
    [
        # DAC1 publishes donor totals and CRS each donor's flows by recipient. Both read
        # "aid" alike, so the query naming no recipient reads DAC1's totals.
        pytest.param(
            "aid from France",
            ["France"],
            "aid",
            (_CRS_SV, _DAC1_SV),
            [(_DAC1_SV,)],
            id="donor-only",
        ),
        pytest.param(
            "aid worldwide", [], "aid", (_CRS_SV, _DAC1_SV), [(_DAC1_SV,)], id="worldwide"
        ),
        # France may be the donor or the recipient, so no axis is unasked.
        pytest.param(
            "France aid",
            ["France"],
            "aid",
            (_CRS_SV, _DAC1_SV),
            [(_CRS_SV,), (_DAC1_SV,)],
            id="undirected-place",
        ),
        # DAC1 reads "multilateral ODA" and DAC2a "imputed multilateral ODA", so DAC1 does
        # not read as much of the query and DAC2a's imputed table stays.
        pytest.param(
            "imputed multilateral ODA from Germany",
            ["Germany"],
            "imputed multilateral ODA",
            (_DAC2A_IMPUTED_SV, _DAC1_MULTILATERAL_SV),
            [(_DAC2A_IMPUTED_SV,), (_DAC1_MULTILATERAL_SV,)],
            id="imputed-from-donor",
        ),
        # No longer label matches, and DAC2a's imputed code reads "imputed" through its
        # declared synonym, which DAC1's donor totals do not read.
        pytest.param(
            "imputed aid from Norway",
            ["Norway"],
            "imputed aid",
            (_DAC2A_IMPUTED_SV, _DAC1_SV),
            [(_DAC2A_IMPUTED_SV,), (_DAC1_SV,)],
            id="vacuous-words",
        ),
        # "Grant equivalent" names the grant-equivalent five-tuple, which DAC1's donor
        # disbursements do not read, so the CRS grant equivalents the query names stay.
        pytest.param(
            "grant equivalent ODA from France",
            ["France"],
            "grant equivalent ODA",
            (_CRS_GRANT_EQUIVALENT_SV, _DAC1_SV),
            [(_CRS_GRANT_EQUIVALENT_SV,), (_DAC1_SV,)],
            id="measure-word",
        ),
        # A query naming no measure reads nothing on any table, so DAC1's donor totals displace
        # a role-bearing table of another five-tuple too, in either recall order.
        *(
            pytest.param(
                query,
                entities,
                "aid",
                recalled,
                [(_DAC1_SV,)],
                id=f"{row_id}-{rival_id}-{'rival' if recalled[0] != _DAC1_SV else 'dac1'}-first",
            )
            for query, entities, row_id in (
                ("aid from France", ["France"], "donor-only"),
                ("aid worldwide", [], "worldwide"),
            )
            for rival, rival_id in (
                (_DAC2A_IMPUTED_SV, "dac2a-imputed"),
                (_SECTORS_SV, "imputed-sectors"),
                (_CRS_GRANT_EQUIVALENT_SV, "crs-grant-equivalent"),
            )
            for recalled in ((rival, _DAC1_SV), (_DAC1_SV, rival))
        ),
        # CRS commitments, another five-tuple, read "ODA" as DAC1 does, so DAC1's donor
        # totals displace them; "commitments" reads only on the commitments five-tuple, where
        # recall adds DAC1's commitments total and the CRS commitments by recipient leave for it.
        # DAC1's disbursements then collapse into its commitments, the measure the query names
        # (`collapse_measure_families`).
        pytest.param(
            "ODA from Japan",
            ["Japan"],
            "ODA",
            (_CRS_COMMITMENT_SV, _DAC1_SV),
            [(_DAC1_SV,)],
            id="oda-across-five-tuples",
        ),
        pytest.param(
            "ODA commitments from Japan",
            ["Japan"],
            "ODA commitments",
            (_CRS_COMMITMENT_SV, _DAC1_SV),
            [(_DAC1_COMMITMENT_SV,)],
            id="commitments-measure-word",
        ),
        # Recall adds DAC1's commitments total for the CRS commitments, which leave for it.
        # The query did not recall it, so the CRS flows of another five-tuple stay.
        pytest.param(
            "aid from France",
            ["France"],
            "aid",
            (_CRS_SV, _CRS_COMMITMENT_SV),
            [(_CRS_SV,), (_DAC1_COMMITMENT_SV,)],
            id="rival-displaces-its-own-five-tuple-only",
        ),
        # "Attributed to donors" names the attributed-amount measure, which the imputed table
        # holds and DAC1's disbursements do not, so the imputed table stays.
        pytest.param(
            "aid attributed to donors via multilaterals",
            [],
            "aid attributed to donors via multilaterals",
            (_SECTORS_SV, _DAC1_SV),
            [(_SECTORS_SV,), (_DAC1_SV,)],
            id="attributed-measure",
        ),
        # Both debt series are recalled samples the index does not carry, so the
        # creditor-only one shows no more than its sample's axes and the by-creditor
        # series stays.
        pytest.param(
            "debt worldwide",
            [],
            "debt",
            (_FLOW_SV, _CREDITOR_SV),
            [(_FLOW_SV,), (_CREDITOR_SV,)],
            id="unindexed-sample",
        ),
        # The by-creditor debt series is a recalled sample the index does not carry, so it
        # reads no index code and its axes are its sample's; it stays beside DAC1.
        pytest.param(
            "debt from France",
            ["France"],
            "debt",
            (_FLOW_SV, _DAC1_SV),
            [(_FLOW_SV,), (_DAC1_SV,)],
            id="unindexed-leaver",
        ),
    ],
)
def test_a_shape_publishing_on_an_axis_the_query_does_not_ask_for_leaves_the_menu(
    monkeypatch,
    query: str,
    entities: list[str],
    variable: str,
    recalled: tuple[str, ...],
    expected: list[tuple[str, ...]],
) -> None:
    """A role-bearing shape leaves for one that declares fewer of the axes the query does
    not ask for and reads every phrase it reads (`without_unasked_axes`), across
    five-tuples, unless a named place has no direction. Both shapes are ones the index
    carries."""
    _result, menu = _resolve(
        monkeypatch,
        query,
        entities,
        variable=variable,
        recalled=tuple((sv, 0.95 - 0.05 * rank) for rank, sv in enumerate(recalled)),
    )

    assert menu == expected
