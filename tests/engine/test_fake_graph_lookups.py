"""Tests FakeGraph node, entity, and place lookups against fixtures and injected data."""

import pytest

from search.engine.errors import GraphInfraError
from tests.fixtures import FakeGraph


def test_node_label_known_dcid() -> None:
    g = FakeGraph()
    label = g.node_label("country/ETH")
    assert label == "Ethiopia"


def test_node_label_absent_dcid_returns_none() -> None:
    g = FakeGraph()
    assert g.node_label("nonexistent/FAKE_NODE") is None


def test_node_arcs_known_dcid() -> None:
    g = FakeGraph()
    arcs = g.node_arcs("country/KEN")
    assert arcs is not None
    assert "typeOf" in arcs
    assert "name" in arcs


def test_node_arcs_absent_dcid_returns_none() -> None:
    g = FakeGraph()
    assert g.node_arcs("totally/made/up") is None


def test_node_type_known_dcid() -> None:
    g = FakeGraph()
    assert g.node_type("country/ETH") == "Country"
    assert g.node_type("activitySource") == "Property"
    assert g.node_type("GrossDomesticProduction") == "EconomicActivitySourceEnum"


def test_node_type_absent_returns_none() -> None:
    g = FakeGraph()
    assert g.node_type("fabricated/NODE") is None


@pytest.mark.parametrize(
    ("dcid", "expected_label"),
    [
        pytest.param(
            "FertilityRate_Person_Female",
            "Fertility rate of women",
            id="multi-name-last-value",
        ),
        pytest.param("Nominal", "Nominal", id="single-value"),
    ],
)
def test_node_label_selection(dcid: str, expected_label: str) -> None:
    g = FakeGraph()
    assert g.node_label(dcid) == expected_label


def test_exists_known_dcid() -> None:
    g = FakeGraph()
    assert g.exists("country/USA") is True


def test_exists_absent_dcid() -> None:
    g = FakeGraph()
    assert g.exists("does/not/exist") is False


_COUNTRY_ARC = "<-description{typeOf:Country}->dcid"
_CONTINENT_ARC = "<-description{typeOf:Continent}->dcid"
_UNGEOREGION_ARC = "<-description{typeOf:UNGeoRegion}->dcid"
_BARE_ARC = "<-description->dcid"


def test_resolve_by_arc_country_known_name() -> None:
    g = FakeGraph()
    assert g.resolve_by_arc(["Ethiopia", "Kenya", "Germany"], arc=_COUNTRY_ARC) == {
        "Ethiopia": ["country/ETH"],
        "Kenya": ["country/KEN"],
        "Germany": ["country/DEU"],
    }


def test_resolve_by_arc_country_unknown_name() -> None:
    g = FakeGraph()
    assert g.resolve_by_arc(["Atlantis"], arc=_COUNTRY_ARC) == {"Atlantis": []}


def test_raise_on_call_resolve_by_arc() -> None:
    g = FakeGraph(raise_on_call=True)
    with pytest.raises(GraphInfraError):
        g.resolve_by_arc(["Africa"], arc=_CONTINENT_ARC)


def test_raise_on_call_node_arcs_many() -> None:
    g = FakeGraph(raise_on_call=True)
    with pytest.raises(GraphInfraError):
        g.node_arcs_many(["africa"], properties=("typeOf", "name"))


# Place-resolution fixtures cover UNGeoRegion and ContinentalUnion.


def test_resolve_by_arc_ungeoregion_from_shared_fixture() -> None:
    """ "Southern Africa" resolves via the UNGeoRegion rung -- a covered type
    in entity_resolution.graph's own fallback ladder."""
    g = FakeGraph()
    assert g.resolve_by_arc(["Southern Africa"], arc=_UNGEOREGION_ARC) == {
        "Southern Africa": ["SouthernAfrica"]
    }


def test_resolve_by_arc_exact_dcid_text_for_ungeoregion() -> None:
    """The fixture resolves a UNGeoRegion by display name and dcid."""
    g = FakeGraph()
    assert g.resolve_by_arc(["SouthernAfrica"], arc=_UNGEOREGION_ARC) == {
        "SouthernAfrica": ["SouthernAfrica"]
    }
    assert g.resolve_by_arc(["SouthernAfrica"], arc=_BARE_ARC) == {
        "SouthernAfrica": ["SouthernAfrica"]
    }


def test_resolve_by_arc_continentalunion_is_not_covered() -> None:
    """ "European Union" is a real graph node (ContinentalUnion), but that
    type has no dedicated rung in the fallback ladder -- the Continent and
    UNGeoRegion rungs both miss it."""
    g = FakeGraph()
    assert g.resolve_by_arc(["European Union"], arc=_CONTINENT_ARC) == {"European Union": []}
    assert g.resolve_by_arc(["European Union"], arc=_UNGEOREGION_ARC) == {"European Union": []}


def test_resolve_by_arc_bare_arc_finds_continentalunion() -> None:
    """The bare, unconstrained rung finds "European Union" even though the
    typed rungs cannot -- positive evidence of a real node whose type this
    ladder does not carry a dedicated rung for."""
    g = FakeGraph()
    assert g.resolve_by_arc(["European Union"], arc=_BARE_ARC) == {
        "European Union": ["EuropeanUnion"]
    }


def test_resolve_by_arc_bare_arc_unknown_name_returns_empty_list() -> None:
    """A name absent from every fixture table misses the bare rung too --
    nothing in the graph matches it at all."""
    g = FakeGraph()
    assert g.resolve_by_arc(["Republic of Atlantis"], arc=_BARE_ARC) == {"Republic of Atlantis": []}


def test_child_places_ungeoregion_from_shared_fixture() -> None:
    """The shared fixture holds SouthernAfrica's real `Country` children."""
    g = FakeGraph()
    assert g.child_places_batch(["SouthernAfrica"], "Country") == {
        "SouthernAfrica": [
            "country/BWA",
            "country/LSO",
            "country/NAM",
            "country/SWZ",
            "country/ZAF",
        ]
    }


def test_child_places_of_a_region_with_no_children_from_shared_fixture() -> None:
    """A region the shared fixture gives no `Country` child lists none."""
    g = FakeGraph()
    (dcid,) = g.resolve_by_arc(["Empty Region"], arc=_UNGEOREGION_ARC)["Empty Region"]
    assert g.child_places_batch([dcid], "Country") == {dcid: []}


@pytest.mark.parametrize(
    "lookup",
    [
        pytest.param(
            lambda graph: graph.child_places_batch(["africa"], "Country"),
            id="child_places_batch",
        ),
        pytest.param(lambda graph: graph.node_label("country/ETH"), id="node_label"),
        pytest.param(lambda graph: graph.node_arcs("ODAGrants"), id="node_arcs"),
        pytest.param(
            lambda graph: graph.node_arcs_batch(["country/KEN"]),
            id="node_arcs_batch",
        ),
        pytest.param(
            lambda graph: graph.node_observation_properties_batch(["country/KEN"]),
            id="node_observation_properties_batch",
        ),
        pytest.param(lambda graph: graph.exists("country/KEN"), id="exists"),
    ],
)
def test_raise_on_call_lookup_methods(lookup) -> None:
    graph = FakeGraph(raise_on_call=True)
    with pytest.raises(GraphInfraError):
        lookup(graph)


def test_node_arcs_batch_present_and_absent() -> None:
    """Batch fetch returns arcs for a present dcid and None for an absent dcid."""
    g = FakeGraph()
    result = g.node_arcs_batch(["country/KEN", "totally/made/up"])
    # Present dcid returns arcs dict
    assert result["country/KEN"] is not None
    assert "typeOf" in result["country/KEN"]
    # The absent dcid remains a key with value None.
    assert "totally/made/up" in result
    assert result["totally/made/up"] is None


_ROLE_BEARING_DCID = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSPurposeCode99810_Nominal"
)


def test_node_observation_properties_batch_derives_tuple_from_recorded_arcs() -> None:
    """Derive role names from the recorded `observationProperties` arc."""
    g = FakeGraph()
    result = g.node_observation_properties_batch([_ROLE_BEARING_DCID])
    assert result == {_ROLE_BEARING_DCID: ("financingProvider", "financingRecipient")}


def test_node_observation_properties_batch_unknown_dcid_returns_empty_tuple() -> None:
    g = FakeGraph()
    result = g.node_observation_properties_batch(["totally/made/up"])
    assert result == {"totally/made/up": ()}


def test_node_observation_properties_batch_present_node_without_arc_returns_empty_tuple() -> None:
    """A present standard StatVar without role arcs returns an empty tuple."""
    g = FakeGraph()
    result = g.node_observation_properties_batch(["measuredValue"])
    assert result == {"measuredValue": ()}


def test_per_test_injection_overrides_file() -> None:
    """Explicit dictionaries replace the shared fixture data."""
    node_arcs = {"typeOf": {"nodes": [{"dcid": "Thing"}]}}
    g = FakeGraph(
        nodes={"my/Node": {"label": "My Node", "type": "Thing", "arcs": node_arcs}},
        obs={},
        detect={},
        resolve={},
    )
    assert g.node_label("my/Node") == "My Node"
    assert g.node_label("country/ETH") is None  # absent because custom nodes dict is used
