"""Test financing-role resolution and coverage confirmation with synthetic StatVars."""

from __future__ import annotations

from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.retrieve import Materialised, NoDataDraft, materialise
from search.engine.shape import ShapeDraft, role_names_by_axis
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._builders import bound_place_context, flow_place_roles
from tests.engine._fake_coverage import FakeCoverageService, empty_coverage_snapshot, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

_SV_DCID = "Amount_EconomicActivity_FinancingRoleFixture"
_UNPLACEABLE_SV_DCID = "Amount_EconomicActivity_UnplaceableRoleFixture"

_FINANCING_ROLE_NAMES = RoleNames(donor="financingProvider", recipient="financingRecipient")


def _financing_arcs() -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {
            "nodes": [{"dcid": "financingProvider"}, {"dcid": "financingRecipient"}]
        },
    }


def _derived_shape() -> ShapeDraft:
    nodes = {_SV_DCID: {"label": "Financing role fixture", "arcs": _financing_arcs()}}
    graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
    shapes = derive_shapes(confirmed_svs=[_SV_DCID], graph=graph)
    return shapes[0]


def test_role_names_by_axis_places_the_financing_names() -> None:
    shape = _derived_shape()
    assert role_names_by_axis(shape) == {
        "origin": "financingProvider",
        "destination": "financingRecipient",
    }


def test_materialise_resolves_a_financing_role_bearing_shape(monkeypatch) -> None:
    """Bound financing roles reach coverage confirmation and resolve the StatVar."""
    shape = _derived_shape()
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", "country/FRA", "country/KEN", [2018, 2019, 2020, 2021, 2022])],
        role_names={_SV_DCID: _FINANCING_ROLE_NAMES},
    )

    result = materialise(
        shape,
        [],
        {"financingProvider": ["country/FRA"], "financingRecipient": ["country/KEN"]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),  # coverage supplies role facets
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_SV_DCID]
    assert result.facets_by_sv is not None
    assert [f.facet_id for f in result.facets_by_sv[_SV_DCID]] == ["f1"]
    assert result.facets_by_sv[_SV_DCID][0].obs_count == 5


def test_materialise_two_destinations_sharing_one_facet_id_report_one_source(
    monkeypatch,
) -> None:
    """Two destination rows sharing facet_id f1 count as one source."""
    shape = _derived_shape()
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_SV_DCID, "f1", "country/FRA", "country/KEN", [2018, 2019, 2020, 2021, 2022]),
            (_SV_DCID, "f1", "country/FRA", "country/UGA", [2018, 2019, 2020, 2021, 2022]),
        ],
        role_names={_SV_DCID: _FINANCING_ROLE_NAMES},
        facet_metadata={"f1": FacetMetadata(provenance_id="source/FRA")},
    )

    result = materialise(
        shape,
        [],
        {
            "financingProvider": ["country/FRA"],
            "financingRecipient": ["country/KEN", "country/UGA"],
        },
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_SV_DCID]
    assert result.coverage.dimensions[0].count == 1
    assert result.coverage.observation_count == 10


def _unplaceable_role_arcs() -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": "beneficiaryOrganisation"}]},
    }


def test_materialise_refuses_role_not_resolved_for_an_unplaceable_role() -> None:
    """An unplaceable declared role yields role_not_resolved before coverage or SDMX reads."""
    nodes = {
        _UNPLACEABLE_SV_DCID: {
            "label": "Unplaceable role fixture",
            "arcs": _unplaceable_role_arcs(),
        }
    }
    graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
    shape = derive_shapes(confirmed_svs=[_UNPLACEABLE_SV_DCID], graph=graph)[0]

    result = materialise(
        shape,
        [],
        {"beneficiaryOrganisation": ["org/Some"]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage=empty_coverage_snapshot(),
    )

    assert isinstance(result, NoDataDraft)
    assert result.reason == "role_not_resolved"


# Alternate shapes fail or pass role validation independently.

_ALT_SV_DECLARING = "Amount_EconomicActivity_FinancingRoleAltDeclaring"

_ALT_POP_TYPE = "FinancialFlow"
_ALT_MEAS_PROP = "grantEquivalentAlt"
_ALT_STAT_TYPE = "measuredValue"

_KEN = "country/KEN"


def _alt_shape(sv_dcid: str, *, observation_properties: tuple[str, ...]) -> ShapeDraft:
    """Build an alternate shape with the requested member role arcs."""
    arcs: dict = {
        "populationType": {"nodes": [{"dcid": _ALT_POP_TYPE}]},
        "measuredProperty": {"nodes": [{"dcid": _ALT_MEAS_PROP}]},
        "statType": {"nodes": [{"dcid": _ALT_STAT_TYPE}]},
    }
    if observation_properties:
        arcs["observationProperties"] = {
            "nodes": [{"dcid": prop} for prop in observation_properties]
        }
    return ShapeDraft(
        shape_id=f"test_alt_financing_shape_{sv_dcid}",
        label="alt financing flow",
        pop_type_dcid=_ALT_POP_TYPE,
        meas_prop_dcid=_ALT_MEAS_PROP,
        stat_type_dcid=_ALT_STAT_TYPE,
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={sv_dcid: arcs},
        slot_taxonomy={},
        declared_roles=("financingProvider", "financingRecipient"),
    )


def test_materialise_alt_shapes_serves_the_alt_declaring_the_bound_roles(monkeypatch) -> None:
    """An alternate declaring both bound roles materialises at its provider and recipient.

    A subject-only alternate never reaches materialisation for this query: its places name
    both flow ends, so `bind_place_roles` refuses it at admission."""
    from tests.engine._builders import materialise_alternates

    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_ALT_SV_DECLARING, "f1", "country/FRA", _KEN, [2020])],
        role_names={_ALT_SV_DECLARING: _FINANCING_ROLE_NAMES},
    )

    declaring = _alt_shape(
        _ALT_SV_DECLARING,
        observation_properties=("financingProvider", "financingRecipient"),
    )

    results, unsupported = materialise_alternates(
        [declaring],
        places=[
            bound_place_context(
                flow_place_roles(declaring, destination=[_KEN], origin=["country/FRA"]), _KEN
            )
        ],
        elected_bindings=[],
        elected_has_data=False,
        named_place_dcids=[_KEN],
        graph=FakeGraph(nodes={}, detect={}, resolve={}, obs={}),
        sdmx=FakeSdmxClient(observations={}),
        date_request=None,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert unsupported is None
    surviving_svs = [sv for cand in results for sv in cand.mat.sv_dcids]
    assert surviving_svs == [_ALT_SV_DECLARING]


# Preserve the unavailable carrier when the bound role pair has no data.

_TOKEN_SV_DCID = "Amount_EconomicActivity_FinancingRoleTokenFixture"
_TOKEN_SV_ALT_DCID = "Amount_EconomicActivity_FinancingRoleTokenAltFixture"


def _token_shape(*, complete: bool) -> ShapeDraft:
    """Build two role-bearing members. `complete` marks their population as exact."""
    return ShapeDraft(
        shape_id="test_financing_token_shape",
        label="financing flow",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            _TOKEN_SV_DCID: _financing_arcs(),
            _TOKEN_SV_ALT_DCID: _financing_arcs(),
        },
        slot_taxonomy={},
        declared_roles=("financingProvider", "financingRecipient"),
        sv_arc_facts_complete=complete,
    )


def test_zero_observation_role_bearing_resolve_returns_the_flagged_carrier(monkeypatch) -> None:
    """An empty bound pair returns a no-observations carrier with the shape as flagged_match.

    A stocked control confirms the lookup ran successfully.
    """
    entities = {"financingProvider": ["country/FRA"], "financingRecipient": [_KEN]}

    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_TOKEN_SV_DCID, "f1", "country/FRA", _KEN, [2020])],
        role_names={_TOKEN_SV_DCID: _FINANCING_ROLE_NAMES},
    )
    with_data = materialise(
        _token_shape(complete=True),
        [],
        entities,
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert isinstance(with_data, Materialised)
    assert with_data.has_data is True
    assert with_data.sv_dcids == [_TOKEN_SV_DCID]

    _auto_coverage_index_2 = patch_coverage(monkeypatch, [], role_names={})
    result = materialise(
        _token_shape(complete=True),
        [],
        entities,
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage=FakeCoverageService(_auto_coverage_index_2).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.available_at_place is False
    assert result.has_data is False
    assert result.facets_by_sv is None
    assert len(result.sv_dcids) == 1
    assert result.sv_dcids[0] in (_TOKEN_SV_DCID, _TOKEN_SV_ALT_DCID)
    assert result.coverage.has_data is False
    assert result.population is not None
    assert result.population.admitted == 2
    assert result.population.confirmed == 0
    assert result.population.exact is True


def test_zero_observation_carrier_reports_a_capped_population_as_inexact() -> None:
    """A capped recall sample makes admitted a lower bound and the population inexact."""
    result = materialise(
        _token_shape(complete=False),
        [],
        {"financingProvider": ["country/FRA"], "financingRecipient": [_KEN]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage=empty_coverage_snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.available_at_place is False
    assert result.population is not None
    assert result.population.exact is False
