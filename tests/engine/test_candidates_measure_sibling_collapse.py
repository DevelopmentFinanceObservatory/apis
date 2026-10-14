"""A role-bearing FinancialFlow/amount candidate that differs only in measured property folds into
the elected grantEquivalent spec's `alternative_measures` list.
"""

from __future__ import annotations

import asyncio

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.region.codes import ELECTED_SHAPE_NO_DATA
from search.models import Spec
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_QUERY = "grant equivalent flows or gross disbursements to Kenya"
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

# Alternate FinancialFlow/amount measure sibling.
_ALT_SV = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode11110_Nominal"
)


def _sv_arcs_from_real_row(dcid: str) -> dict:
    index = get_index()
    row = next(r for r in index.statvars if r.dcid == dcid)
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": dcid}]},
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "measurementQualifier": {"nodes": [{"dcid": row.measurement_qualifier}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        "observationProperties": {"nodes": [{"dcid": p} for p in row.observation_properties]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }


_NODES = {
    _KEN_DCID: {"label": "Kenya", "type": "Country"},
    _RECALLED_SV: {
        "label": "test recalled sibling",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs_from_real_row(_RECALLED_SV),
    },
    # Grounding reads this index-only member by name. Its node also declares its flow roles.
    _EXTRA_SV: {
        "label": "test index-only sibling",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs_from_real_row(_EXTRA_SV),
    },
    _EXTRA_PURPOSE: {"label": "CRS purpose 51010"},
    "FinancialFlow": {"label": "financial flow"},
    _ALT_SV: {
        "label": "test amount sibling",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs_from_real_row(_ALT_SV),
    },
}

_DETECT = {
    _QUERY: {
        "svs": [_RECALLED_SV, _ALT_SV],
        "cosine_scores": [0.9, 0.85],
    }
}

# The role-bearing elected member's own observations. Only _EXTRA_SV
# survives the crsPurpose bind, the same as test_measure_election_
# inapplicable_slots.py's own fixture.
_SDMX_OBS = {
    _EXTRA_SV: {
        ((_ROLE_RECIPIENT, _KEN_DCID),): [
            Facet(
                earliest_date="2020",
                latest_date="2020",
                obs_count=3,
                dates=("2020",),
                values=(0.0,),
                facet_id="f1",
            )
        ]
    },
    # The alternate uses its recalled StatVar as the construct-and-confirm representative.
    _ALT_SV: {
        ((_ROLE_RECIPIENT, _KEN_DCID),): [
            Facet(
                earliest_date="2020",
                latest_date="2021",
                obs_count=5,
                dates=("2020", "2021"),
                values=(10.0, 12.0),
                facet_id="f2",
            )
        ]
    },
}


class _ElectGrantEquivalentAlsoPlausibleAmountLLM:
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


_ROLE_NAMES = RoleNames(donor=_ROLES[0], recipient=_ROLES[1])
# Both shapes are role-bearing. Kenya binds the destination, and no assertion uses the donor dcid.
_COVERAGE_ROWS = [
    (_EXTRA_SV, "f1", "country/USA", _KEN_DCID, [2018, 2019, 2020]),
    (_ALT_SV, "f2", "country/USA", _KEN_DCID, [2020, 2021]),
]
_COVERAGE_ROWS_ELECTED_HAS_NO_DATA = [
    (_ALT_SV, "f2", "country/USA", _KEN_DCID, [2020, 2021]),
]


def _run(monkeypatch, obs: dict = _SDMX_OBS, coverage_rows: list = _COVERAGE_ROWS):
    role_names = {row[0]: _ROLE_NAMES for row in coverage_rows}
    _auto_coverage_index_1 = patch_coverage(monkeypatch, coverage_rows, role_names=role_names)
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve={"Kenya": _KEN_DCID}, obs={})
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
            sdmx=FakeSdmxClient(observations=obs, availability={}),
            llm=_ElectGrantEquivalentAlsoPlausibleAmountLLM(),
            base_timing={},
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )
    )


def test_a_measure_only_pair_collapses_to_one_definite_answer(monkeypatch):
    result = _run(monkeypatch)
    assert result.status == "definite"
    assert len(result.specs) == 1

    survivor = result.specs[0]
    assert any(sv.ref.ref == _EXTRA_SV for sv in survivor.resolved.members)
    # The region carries the measure facts its Spec was built from, which a same-shape merge
    # re-mints the label from.
    assert result.measure_partition is not None
    assert dict(result.measure_partition.values)["measuredProperty"] == _MEAS_PROP
    assert {fact.property for fact in result.open_axis_facts} >= {"crsCooperationModality"}


def _bound_ref(identity, property_dcid: str) -> str | None:
    """The single bound dcid `identity` carries on the named `IdentitySlot`
    (a `state="value"` binding), or None when the slot is absent/unbound."""
    for slot in identity.slots:
        if slot.property == property_dcid and slot.binding.state == "value":
            return slot.binding.ref
    return None


def test_the_folded_sibling_rides_alternative_measures_not_a_second_menu_entry(monkeypatch):
    result = _run(monkeypatch)
    survivor = result.specs[0]

    alternatives = survivor.resolved.alternative_measures
    assert alternatives is not None
    assert len(alternatives) > 0
    alt_spec_ids = {a.definition.spec_id for a in alternatives}
    assert survivor.spec_id not in alt_spec_ids
    # The folded identity names the amount measure.
    assert any(a.definition.identity.shape.measured_property == "amount" for a in alternatives)
    # The folded entry keeps the survivor's crsPurpose binding.
    assert _bound_ref(survivor.definition.identity, _PROP_PURPOSE) == _EXTRA_PURPOSE
    for alt in alternatives:
        assert _bound_ref(alt.definition.identity, _PROP_PURPOSE) == _EXTRA_PURPOSE


def test_no_stale_candidates_warning_survives_the_fold(monkeypatch):
    result = _run(monkeypatch)
    # A measure fold returns one survivor without a candidates warning.
    assert not any(
        "candidate" in w.message.lower() and "surfac" in w.message.lower() for w in result.warnings
    )


def _spec_for(specs: list[Spec], sv_dcid: str) -> Spec:
    return next(
        spec for spec in specs if any(sv.ref.ref == sv_dcid for sv in spec.resolved.members)
    )


# The elected member has no rows. Only its amount sibling has data.
_SDMX_OBS_ELECTED_HAS_NO_DATA = {
    _ALT_SV: _SDMX_OBS[_ALT_SV],
}


def test_s8_a_measure_sibling_with_data_rides_on_the_refusal(monkeypatch):
    """Status rule 1: when the elected measure has no data, only an equivalent reading or a
    sibling differing on the price basis answers in its place. A sibling of another
    measured property is neither, so the part refuses, and the nominated sibling with data
    rides on the refusal as its alternate match."""
    result = _run(
        monkeypatch,
        obs=_SDMX_OBS_ELECTED_HAS_NO_DATA,
        coverage_rows=_COVERAGE_ROWS_ELECTED_HAS_NO_DATA,
    )
    assert (result.status, result.no_data_reason) == ("no_data", "no_observations")
    assert result.gate == "only_alternate_has_data"
    assert result.alternate_match is not None
    assert any(sv.ref.ref == _ALT_SV for sv in result.alternate_match.resolved.members)
    assert ELECTED_SHAPE_NO_DATA not in [w.code for w in result.warnings]
