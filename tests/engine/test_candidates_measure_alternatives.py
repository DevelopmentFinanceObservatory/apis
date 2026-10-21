"""Candidate responses publish measure labels and alternatives for every candidate.

The elected FinancialFlow/grantEquivalent member belongs to a fixture index measure family but has no
alternative measures. The also-plausible FinancialFlow/amount fixture is subject-only, so it does
not fold into the elected role-bearing shape and has no measure family of its own.
"""

from __future__ import annotations

import asyncio

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.models import Spec
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_QUERY = "grant equivalent flows or total aid spending to Kenya"
_KEN_DCID = "country/KEN"

# Elected FinancialFlow/grantEquivalent shape.
_POP_TYPE = "FinancialFlow"
_MEAS_PROP = "grantEquivalent"
_STAT_TYPE = "measuredValue"
_MEAS_QUAL = "Nominal"

_RECALLED_SV = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode11110_Nominal"
)
_EXTRA_SV = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode51010_Nominal"
)
_PROP_PURPOSE = "crsPurpose"
_EXTRA_PURPOSE = "CRSPurposeCode51010"
_ROLES = ("financingProvider", "financingRecipient")
_ROLE_RECIPIENT = "financingRecipient"

# Subject-only FinancialFlow/amount alternate with no measure family.
# Shares the elected shape's populationType (FinancialFlow) but declares no
# non-subject role, so it neither folds in as a measure sibling
# (_shapes_are_measure_siblings) nor trips R5's population conjunct. Its
# RealValue qualifier appears on no row of the fixture index (every row
# there is Nominal), so its five-tuple carries no measure family at all.
_ALT_SV = (
    "Amount_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode51010_RealValue"
)
# The alt's own ->name arc value, which derive_shapes stamps onto its
# ShapeDraft.label and build_spec leaves untouched (no measure family means no
# measure_label to compose the wire label from).
_ALT_SV_LABEL = "Total aid spending"


def _sv_arcs(dcid: str, *, constraint_props: list[str], **constraint_values: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": dcid}]},
        "populationType": {"nodes": [{"dcid": _POP_TYPE}]},
        "measuredProperty": {"nodes": [{"dcid": _MEAS_PROP}]},
        "statType": {"nodes": [{"dcid": _STAT_TYPE}]},
        "measurementQualifier": {"nodes": [{"dcid": _MEAS_QUAL}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraint_props]},
        "observationProperties": {"nodes": [{"dcid": p} for p in _ROLES]},
        **{p: {"nodes": [{"dcid": constraint_values[p]}]} for p in constraint_props},
    }


def _sv_arcs_from_real_row(dcid: str) -> dict:
    index = get_index()
    row = next(r for r in index.statvars if r.dcid == dcid)
    return _sv_arcs(
        dcid, constraint_props=list(row.constraint_properties), **row.constraint_properties
    )


def _std_arcs(*, pop_type: str, meas_prop: str, meas_qual: str, label: str) -> dict:
    """A subject-only StatVar's arcs: no observationProperties arc at all, so
    facts.read_declared_roles reads its roles as ("observationAbout",) and
    is_role_bearing reports False (see that predicate). label becomes this
    node's ->name value, the source derive_shapes stamps ShapeDraft.label
    from (_name_for)."""
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": pop_type}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": meas_qual}]},
    }


_NODES = {
    _KEN_DCID: {"label": "Kenya", "type": "Country"},
    _RECALLED_SV: {
        "label": "test recalled sibling",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs_from_real_row(_RECALLED_SV),
    },
    # Grounding reads this index-only member by name. Its node declares the flow roles.
    _EXTRA_SV: {
        "label": "test index-only sibling",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs_from_real_row(_EXTRA_SV),
    },
    _EXTRA_PURPOSE: {"label": "CRS purpose 51010"},
    _ALT_SV: {
        "label": _ALT_SV_LABEL,
        "type": "StatisticalVariable",
        "arcs": _std_arcs(
            pop_type=_POP_TYPE,
            meas_prop="amount",
            meas_qual="RealValue",
            label=_ALT_SV_LABEL,
        ),
    },
}

_DETECT = {
    _QUERY: {
        "svs": [_RECALLED_SV, _ALT_SV],
        "cosine_scores": [0.9, 0.85],
    }
}

# The plain alt member's own observations (graph path).
_GRAPH_OBS = {
    f"{_ALT_SV}|{_KEN_DCID}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}]
}

# Only _EXTRA_SV survives the crsPurpose bind. Kenya binds the destination.
# No assertion uses the donor dcid.
_ROLE_NAMES = RoleNames(donor=_ROLES[0], recipient=_ROLES[1])
_COVERAGE_ROWS = [(_EXTRA_SV, "f1", "country/USA", _KEN_DCID, [2018, 2019, 2020])]


class _ElectGrantEquivalentAlsoPlausibleAidSpendingLLM:
    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="how",
                    property_dcid=_PROP_PURPOSE,
                    kind="value",
                    value_dcids=[_EXTRA_PURPOSE],
                )
            ],
            also_plausible=[1],
            ask=None,
        ), None


def _run(monkeypatch):
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch, _COVERAGE_ROWS, role_names={_EXTRA_SV: _ROLE_NAMES}
    )
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve={"Kenya": _KEN_DCID}, obs=_GRAPH_OBS)
    return asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            entity_roles=door_roles_for_test_query(_QUERY, ["Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_ElectGrantEquivalentAlsoPlausibleAidSpendingLLM(),
            base_timing={},
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )
    )


def _spec_for(specs: list[Spec], sv_dcid: str) -> Spec:
    return next(
        spec for spec in specs if any(sv.ref.ref == sv_dcid for sv in spec.resolved.members)
    )


def test_the_role_bearing_candidate_publishes_a_label_and_no_real_alternatives(monkeypatch):
    # The elected family has no measure alternatives. The subject-only shape has no family.
    result = _run(monkeypatch)
    # Both the elected and plausible members must reach the response.
    assert result.status == "candidates"
    assert len(result.specs) == 2
    assert {sv.ref.ref for spec in result.specs for sv in spec.resolved.members} == {
        _EXTRA_SV,
        _ALT_SV,
    }

    crs_spec = _spec_for(result.specs, _EXTRA_SV)
    assert crs_spec.resolved.measure_label is not None
    assert crs_spec.resolved.alternative_measures == []


def test_the_role_bearing_candidate_wire_label_names_what_the_group_shares(monkeypatch):
    """The composed label uses the measure label and shared axes.
    It omits each member's StatVar name.
    """
    result = _run(monkeypatch)
    crs_spec = _spec_for(result.specs, _EXTRA_SV)

    label = crs_spec.resolved.shape.label
    assert label != "test index-only sibling"
    assert label != "test recalled sibling"
    assert label.startswith("grant equivalents (")
    assert "general budget support" in label
    assert "standard grant" in label
    assert "official development assistance" in label
    # The shared label appears once even though both axes resolve to it.
    assert label.count("general budget support") == 1


def test_the_alt_candidate_outside_any_measure_family_publishes_null_on_both_fields(monkeypatch):
    result = _run(monkeypatch)
    alt_spec = _spec_for(result.specs, _ALT_SV)

    assert alt_spec.resolved.measure_label is None
    assert alt_spec.resolved.alternative_measures is None
    # Without a measure family, the response keeps the StatVar name.
    assert alt_spec.resolved.shape.label == _ALT_SV_LABEL
