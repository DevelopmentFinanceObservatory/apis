"""A standard resubmit accepts an unresolved measurementDenominator arc.
The display id uses `_per_unresolved`; the identity marks the denominator as
unresolved.
"""

from __future__ import annotations

from search.engine.region import resolve_spec_resubmit
from search.models import ResubmitIdentityPayload
from tests.engine._base_identity import make_identity, subject_entity
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.fixtures import FakeGraph, _load_json

_UNRESOLVED_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_PerUnresolved"
# The display shape_id derive_shapes mints for this five-tuple (kept for the
# wire Shape.shape_id assertion below); the identity shape is _UNRESOLVED_SHAPE.
_UNRESOLVED_SHAPE_ID = "economicactivity_amount_measuredvalue_per_unresolved"
# Mirrors _UNRESOLVED_DENOM_ARCS below -- resolve_spec_resubmit's standard
# path re-derives the shape from the posted resolved_stat_var_dcids' real
# arcs and requires it to match the posted shape structurally.
_UNRESOLVED_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "EconomicActivity",
    "measured_property": "amount",
    "stat_type": "measuredValue",
    "measurement_denominator": {"unresolved": True},
}
_GDP_ENTITY_DCID = "country/IND"

_UNRESOLVED_DENOM_ARCS = {
    "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
    "measuredProperty": {"nodes": [{"dcid": "amount"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
    "measurementDenominator": {
        "nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]
    },
}


def _graph_with_unresolved_sv() -> FakeGraph:
    nodes = {
        **_load_json("graph_nodes.json"),
        _UNRESOLVED_SV_DCID: {
            "label": "unresolved-denominator GDP",
            "arcs": _UNRESOLVED_DENOM_ARCS,
        },
    }
    # Facets keyed "sv|entity" (FakeGraph.observation_facets) -- a real GDP
    # entry's own facets, reused so the synthetic SV has observation coverage
    # for _GDP_ENTITY_DCID and the promote path reaches "definite" rather than
    # no_observations.
    real_gdp_key = f"Amount_EconomicActivity_GrossDomesticProduction_Nominal|{_GDP_ENTITY_DCID}"
    synthetic_key = f"{_UNRESOLVED_SV_DCID}|{_GDP_ENTITY_DCID}"
    default_obs = _load_json("graph_obs.json")
    obs = {**default_obs, synthetic_key: default_obs[real_gdp_key]}
    return FakeGraph(nodes=nodes, obs=obs, detect={}, resolve={})


class TestUnresolvedDenominatorPromote:
    """A standard promote/refine resubmit of an unresolved-denominator shape
    is accepted, not rejected as shape_mismatch."""

    def test_unresolved_denominator_promote_is_accepted(self):
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        inp = ResubmitIdentityPayload(
            identity=make_identity(_UNRESOLVED_SHAPE, entities=entities),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_graph_with_unresolved_sv(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(_graph_with_unresolved_sv()),
        )
        assert region.status == "definite", (
            "an unresolved-denominator five-tuple's own minted shape_id must promote "
            f"cleanly, not degrade; got status={region.status!r}"
        )
        assert [sv.ref.ref for sv in region.specs[0].resolved.members] == [_UNRESOLVED_SV_DCID]
        assert region.specs[0].resolved.shape.shape_id == _UNRESOLVED_SHAPE_ID


class TestUnresolvedDenominatorShapeDraftFlag:
    """The ShapeDraft built on the resubmit path carries
    meas_denom_unresolved=True for an unresolved-denominator five-tuple."""

    def test_shape_draft_carries_meas_denom_unresolved(self, monkeypatch):
        import search.engine.region.promote_shape as promote_shape

        captured: dict[str, object] = {}
        real_shape_draft_from = promote_shape.shape_draft_from

        def _recording_shape_draft_from(*args, **kwargs):
            captured["meas_denom_unresolved"] = kwargs.get("meas_denom_unresolved")
            return real_shape_draft_from(*args, **kwargs)

        monkeypatch.setattr(promote_shape, "shape_draft_from", _recording_shape_draft_from)

        entities = [subject_entity(_GDP_ENTITY_DCID)]
        inp = ResubmitIdentityPayload(
            identity=make_identity(_UNRESOLVED_SHAPE, entities=entities),
        )
        resolve_spec_resubmit(
            inp=inp,
            graph=_graph_with_unresolved_sv(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(_graph_with_unresolved_sv()),
        )
        assert captured["meas_denom_unresolved"] is True
