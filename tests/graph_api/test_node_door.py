"""Node-door tests use in-memory graph fixtures and recorded `/v2/node` responses."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from graph_api import app as graph_app
from graph_api import routing
from graph_api.doors import node as node_door
from graph_api.node_models import NODES_MAX_DCIDS
from graph_api.version import GRAPH_CAPABILITY_VERSION
from graph_generation.fake import FakeGeneration
from materialise.graph import GraphDeadlineExceededError, GraphReadError, parse_arcs
from origin_catalogue import DFO
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue

from .conftest import FakeGraphNodeReader, make_client, send_past_a_full_pool, wait_until

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _wire_arcs(fixture_name: str, dcid: str) -> dict[str, tuple[str, ...]]:
    """Parse a recorded `/v2/node` response with the node door's allowlist."""
    body = json.loads((_FIXTURES_DIR / fixture_name).read_text())
    arcs = body["data"][dcid]["arcs"]
    return parse_arcs(arcs, node_door._ALLOWLIST, mark_unresolved=True)


class TestStatVarShape:
    def test_statvar_with_two_declared_roles_carries_its_definition_and_description(self):
        graph = FakeGraphNodeReader(
            nodes={
                "dc/base/BilateralFlow_Grants": {
                    "typeOf": ("StatisticalVariable",),
                    "name": ("Bilateral flow (grants)",),
                    "definition": ("mq=Nominal,mp=amount,pt=BilateralFlow",),
                    "description": ("Grants between a provider and a recipient",),
                    "populationType": ("BilateralFlow",),
                    "observationProperties": ("provider", "recipient"),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/dc/base/BilateralFlow_Grants")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["kind"] == "stat_var"
        assert body["ref"] == "dc/base/BilateralFlow_Grants"
        assert body["label"] == "Bilateral flow (grants)"
        assert body["types"] == ["StatisticalVariable"]
        assert body["description"] == "Grants between a provider and a recipient"
        stat_var = body["stat_var"]
        assert set(stat_var) == {"definition", "availability"}
        definition = stat_var["definition"]
        assert definition["roles"] == ["provider", "recipient"]
        assert definition["population_type"] == {"ref": "BilateralFlow", "label": None}
        assert definition["definitions"] == [
            {
                "of": {"ref": "dc/base/BilateralFlow_Grants", "label": "Bilateral flow (grants)"},
                "text": "Grants between a provider and a recipient",
            }
        ]
        # A surface with no profile cache reads no availability.
        assert stat_var["availability"] is None
        assert body["links"] == {
            "facets": "/api/graph/node/dc/base/BilateralFlow_Grants/members?relation=facets"
        }
        assert body["gaps"] == [
            "stat_var.definition.population_type.label",
            "stat_var.definition.measured_property",
            "stat_var.definition.stat_type",
            "stat_var.definition.measure",
            "stat_var.availability.not_served",
        ]
        assert resp.headers["cache-control"] == "public, max-age=300"
        # The arc read, then the definition's shape read and its word read.
        assert graph.calls["node_arcs_many"] == 3

    def test_statvar_with_one_declared_role(self):
        graph = FakeGraphNodeReader(
            nodes={
                "dc/base/BilateralFlow_RecipientOnly": {
                    "typeOf": ("StatisticalVariable",),
                    "name": ("Bilateral flow (recipient role only)",),
                    "populationType": ("BilateralFlow",),
                    "observationProperties": ("recipient",),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/dc/base/BilateralFlow_RecipientOnly")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["kind"] == "stat_var"
        assert body["stat_var"]["definition"]["roles"] == ["recipient"]

    def test_public_dc_statvar_without_observation_properties_defaults_to_observation_about(self):
        graph = FakeGraphNodeReader(
            nodes={
                "Count_Person": {
                    "typeOf": ("StatisticalVariable",),
                    "name": ("Count_Person",),
                    "description": ("Population count",),
                    "populationType": ("Person",),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/Count_Person")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["kind"] == "stat_var"
        definition = body["stat_var"]["definition"]
        assert definition["roles"] == ["observationAbout"]
        assert definition["population_type"]["ref"] == "Person"
        assert body["description"] == "Population count"

    def test_statvar_with_no_populationtype_arc_reports_null(self):
        graph = FakeGraphNodeReader(
            nodes={
                "dc/gdp": {
                    "typeOf": ("StatisticalVariable",),
                    "name": ("GDP per capita",),
                    "definition": ("Gross domestic product per capita",),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/dc/gdp")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        definition = body["stat_var"]["definition"]
        assert definition["roles"] == ["observationAbout"]
        assert definition["population_type"] is None
        assert body["description"] is None
        assert definition["source_indicator_codes"] == []

    def test_statvar_lists_every_source_indicator_code(self):
        dcid = (
            "ValueAdded_EconomicProduction_IncludingProductTaxes_"
            "GrossOfFixedCapitalConsumption_Nominal"
        )
        graph = FakeGraphNodeReader(
            nodes={
                dcid: {
                    "typeOf": ("StatisticalVariable",),
                    "name": ("Gross domestic product (GDP), Current prices",),
                    "populationType": ("EconomicProduction",),
                    "constraintProperties": ("fixedCapitalConsumptionBasis", "productTaxBasis"),
                    "sourceIndicatorCode": ("NGDP", "NGDPD"),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get(f"/api/graph/node/{dcid}")

        assert resp.status_code == 200, resp.text
        assert resp.json()["stat_var"]["definition"]["source_indicator_codes"] == ["NGDP", "NGDPD"]

    def test_degenerate_observation_properties_ships_declared_values_verbatim(self):
        graph = FakeGraphNodeReader(
            nodes={
                "dc/base/BilateralFlow_Degenerate": {
                    "typeOf": ("StatisticalVariable",),
                    "name": ("Degenerate declaration",),
                    "observationProperties": ("observationAbout", "provider"),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/dc/base/BilateralFlow_Degenerate")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["kind"] == "stat_var"
        assert body["stat_var"]["definition"]["roles"] == ["observationAbout", "provider"]


class TestSourceShape:
    def test_who_style_provenance_carries_both_freshness_fields(self):
        graph = FakeGraphNodeReader(
            nodes={
                "dc/base/WHO": {
                    "typeOf": ("Provenance",),
                    "name": ("World Health Organization",),
                    "license": ("CC-BY-4.0",),
                    "licenseType": ("open",),
                    "description": ("WHO data provenance",),
                    "url": ("https://who.int",),
                    "sourceReleaseFrequency": ("P1Y",),
                    "automaticRefreshType": ("GCPCloudRunAutomaticRefresh",),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/dc/base/WHO")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["kind"] == "provenance"
        provenance = body["provenance"]
        assert provenance["license"] == "CC-BY-4.0"
        assert provenance["license_type"] == "open"
        assert body["description"] == "WHO data provenance"
        assert provenance["url"] == "https://who.int"
        assert provenance["freshness"] == {
            "release_frequency": "P1Y",
            "release_frequency_label": "Annual",
            "refresh_type": "GCPCloudRunAutomaticRefresh",
        }
        assert provenance["publisher"] is None
        assert provenance["parent"] is None
        assert provenance["documentation"] == "/api/graph/documentation/dc/base/WHO"
        assert body["links"] == {
            "variables": "/api/graph/node/dc/base/WHO/members?relation=variables"
        }
        assert graph.calls["node_arcs_many"] == 1

    def test_bis_style_source_lacks_release_frequency(self):
        graph = FakeGraphNodeReader(
            nodes={
                "dc/base/BIS": {
                    "typeOf": ("Source",),
                    "name": ("Bank for International Settlements",),
                    "license": ("BIS Terms",),
                    "automaticRefreshType": ("P3M",),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/dc/base/BIS")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["kind"] == "source"
        assert body["source"]["freshness"]["release_frequency"] is None
        assert body["source"]["freshness"]["release_frequency_label"] is None
        assert body["source"]["freshness"]["refresh_type"] == "P3M"
        assert body["links"] == {
            "datasets": "/api/graph/node/dc/base/BIS/members?relation=datasets"
        }

    def test_widened_fields_surface_from_the_allowlist(self):
        graph = FakeGraphNodeReader(
            nodes={
                "dc/base/Wide": {
                    "typeOf": ("Provenance",),
                    "name": ("Wide Provenance",),
                    "sourceDataUrl": ("https://example.org/data.csv",),
                    "descriptionUrl": ("https://example.org/catalog",),
                    "dataTransformationLogic": ("https://github.com/example/scripts",),
                    "lastDataRefreshDate": ("2024-05-08",),
                    "nextDataRefreshDate": ("2025-05-08",),
                    "earliestObservationDate": ("1973",),
                    "latestObservationDate": ("2024",),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/dc/base/Wide")

        assert resp.status_code == 200, resp.text
        provenance = resp.json()["provenance"]
        assert provenance["source_data_url"] == "https://example.org/data.csv"
        assert provenance["description_url"] == "https://example.org/catalog"
        assert provenance["transformation_logic_url"] == "https://github.com/example/scripts"
        assert provenance["last_data_refresh_date"] == "2024-05-08"
        assert provenance["next_data_refresh_date"] == "2025-05-08"
        assert provenance["earliest_observation_date"] == "1973"
        assert provenance["latest_observation_date"] == "2024"

    def test_publisher_and_dataset_resolve_via_one_batched_second_call(self):
        graph = FakeGraphNodeReader(
            nodes={
                "dc/base/WDI": {
                    "typeOf": ("Provenance",),
                    "name": ("WorldDevelopmentIndicators",),
                    "source": ("dc/s/WorldBank",),
                    "isPartOf": ("dc/d/WorldBank_WDI",),
                },
                "dc/s/WorldBank": {"name": ("World Bank",)},
                "dc/d/WorldBank_WDI": {"name": ("World Development Indicators",)},
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/dc/base/WDI")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["provenance"]["publisher"] == {"ref": "dc/s/WorldBank", "label": "World Bank"}
        assert body["provenance"]["parent"] == {
            "ref": "dc/d/WorldBank_WDI",
            "label": "World Development Indicators",
        }
        # The parent Dataset's own name is preferred over the provenance's
        # own import-identifier name.
        assert body["label"] == "World Development Indicators"
        assert graph.calls["node_arcs_many"] == 2

    def test_label_falls_back_to_own_name_when_there_is_no_parent_dataset(self):
        graph = FakeGraphNodeReader(
            nodes={
                "provenance/Orphan": {
                    "typeOf": ("Provenance",),
                    "name": ("Orphan Provenance",),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/provenance/Orphan")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["label"] == "Orphan Provenance"
        assert body["provenance"]["parent"] is None
        assert graph.calls["node_arcs_many"] == 1

    def test_publisher_label_rejects_a_name_value_equal_to_its_own_dcid(self):
        """The recorded fixture puts a dcid-echo `name` first on the OECD
        Source node. A first-value lookup would return the dcid as label.
        """
        graph = FakeGraphNodeReader(
            nodes={
                "provenance/OECDCreditorReportingSystem": {
                    "typeOf": ("Provenance",),
                    "name": ("OECD Creditor Reporting System",),
                    "source": ("dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd",),
                },
                "dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd": {
                    "name": (
                        "dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd",
                        "Organisation for Economic Co-operation and Development (OECD)",
                    )
                },
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/provenance/OECDCreditorReportingSystem")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["provenance"]["publisher"] == {
            "ref": "dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd",
            "label": "Organisation for Economic Co-operation and Development (OECD)",
        }


class TestSourceShapeLiveFixtures:
    """Parse recorded `/v2/node` fixtures with `parse_arcs`."""

    def test_world_development_indicators_full_shape(self):
        arcs = _wire_arcs(
            "node_world_development_indicators.json", "dc/base/WorldDevelopmentIndicators"
        )
        graph = FakeGraphNodeReader(
            nodes={
                "dc/base/WorldDevelopmentIndicators": arcs,
                # Publisher and dataset labels come from a second graph read,
                # separate from the provenance arcs.
                "dc/s/WorldBank": {"name": ("World Bank",)},
                "dc/d/WorldBank_WorldDevelopmentIndicators": {
                    "name": ("World Development Indicators",)
                },
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/dc/base/WorldDevelopmentIndicators")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["kind"] == "provenance"
        provenance = body["provenance"]
        # The parent Dataset supplies the display label.
        assert body["label"] == "World Development Indicators"
        assert body["description"] == (
            "The primary World Bank collection of development indicators "
            "encompassing global economic and social data."
        )
        assert provenance["url"] == "https://datatopics.worldbank.org/world-development-indicators/"
        assert (
            provenance["source_data_url"]
            == "http://api.worldbank.org/v2/en/country/AFG?downloadformat=csv"
        )
        assert (
            provenance["description_url"]
            == "https://datacatalog.worldbank.org/search/dataset/0037712"
        )
        assert provenance["transformation_logic_url"] == (
            "https://github.com/datacommonsorg/data/tree/master/scripts/world_bank/wdi/"
        )
        assert provenance["last_data_refresh_date"] is None
        assert provenance["earliest_observation_date"] is None
        assert provenance["publisher"] == {"ref": "dc/s/WorldBank", "label": "World Bank"}
        assert provenance["parent"] == {
            "ref": "dc/d/WorldBank_WorldDevelopmentIndicators",
            "label": "World Development Indicators",
        }
        assert provenance["freshness"] == {
            "release_frequency": "P1M",
            "release_frequency_label": "Monthly",
            "refresh_type": "GCPCloudRunAutomaticRefresh",
        }
        assert graph.calls["node_arcs_many"] == 2

    def test_oecd_crs_provenance_thin_shape_has_no_parent_dataset(self):
        """The CRS provenance covers 1973-2024. Description, licence type,
        and parent Dataset are absent.
        """
        arcs = _wire_arcs("node_oecd_crs_provenance.json", "provenance/OECDCreditorReportingSystem")
        graph = FakeGraphNodeReader(
            nodes={
                "provenance/OECDCreditorReportingSystem": arcs,
                "dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd": _wire_arcs(
                    "node_oecd_source_contested_name.json",
                    "dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd",
                ),
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/provenance/OECDCreditorReportingSystem")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["kind"] == "provenance"
        provenance = body["provenance"]
        assert body["description"] is None
        assert provenance["license_type"] is None
        assert provenance["parent"] is None
        # No parent Dataset, so the label falls back to the provenance's
        # own name.
        assert body["label"] == "OECD Creditor Reporting System"
        assert provenance["earliest_observation_date"] == "1973"
        assert provenance["latest_observation_date"] == "2024"
        assert provenance["freshness"]["release_frequency"] == "P1Y"
        assert provenance["freshness"]["release_frequency_label"] == "Annual"
        assert provenance["freshness"]["refresh_type"] is None
        # The contested-name rejection: two name values on the publisher's
        # own Source node, the bad one (equal to its own dcid) ordered
        # first.
        assert provenance["publisher"] == {
            "ref": "dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd",
            "label": "Organisation for Economic Co-operation and Development (OECD)",
        }
        assert graph.calls["node_arcs_many"] == 2


class TestDataCommons:
    """Admitted provenances name their Data Commons from the request's catalogue."""

    def test_an_admitted_provenance_names_its_data_commons(self, serve_catalogue):
        serve_catalogue(
            restrictive_catalogue(
                provenances=[provenance_row("provenance/OECDCreditorReportingSystem", DFO)]
            )
        )
        graph = FakeGraphNodeReader(
            nodes={"provenance/OECDCreditorReportingSystem": {"typeOf": ("Provenance",)}}
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/provenance/OECDCreditorReportingSystem")

        assert resp.status_code == 200, resp.text
        assert resp.json()["kind"] == "provenance"
        assert resp.json()["provenance"]["data_commons"] == {
            "id": "dfo",
            "label": "Development Finance Observatory",
        }

    def test_a_source_node_the_catalogue_does_not_list_names_none(self, serve_catalogue):
        serve_catalogue(restrictive_catalogue())
        graph = FakeGraphNodeReader(nodes={"dc/s/WorldBank": {"typeOf": ("Source",)}})
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/dc/s/WorldBank")

        assert resp.json()["source"]["data_commons"] is None


class TestPlaceKinds:
    def test_country_entity_resolves_contained_in_labels_via_second_call(self):
        graph = FakeGraphNodeReader(
            nodes={
                "country/KEN": {
                    "typeOf": ("Country",),
                    "name": ("Kenya",),
                    "containedInPlace": ("africa", "worldbank/region1"),
                },
                "africa": {"name": ("Africa",)},
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/country/KEN")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["kind"] == "country"
        assert body["types"] == ["Country"]
        by_dcid = {c["ref"]: c["label"] for c in body["country"]["contained_in"]}
        assert by_dcid == {"africa": "Africa", "worldbank/region1": None}
        assert graph.calls["node_arcs_many"] == 2

    def test_continental_union_classifies_as_its_own_place_kind(self):
        graph = FakeGraphNodeReader(
            nodes={
                "europeanunion/EU": {
                    "typeOf": ("ContinentalUnion",),
                    "name": ("European Union",),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/europeanunion/EU")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["kind"] == "continental_union"
        assert body["continental_union"] == {"contained_in": [], "membership": None}


class TestOtherKind:
    def test_an_unclassified_node_carries_its_types_and_description_and_no_block(self):
        graph = FakeGraphNodeReader(
            nodes={
                "dc/base/Foo": {
                    "typeOf": ("SomeOtherType",),
                    "name": ("Some Node",),
                    "description": ("A node of some other type",),
                    "isPartOf": ("dc/base/Parent",),
                    "provenanceCategory": ("Category1",),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/dc/base/Foo")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["kind"] == "other"
        assert body["label"] == "Some Node"
        assert body["types"] == ["SomeOtherType"]
        assert body["description"] == "A node of some other type"
        assert body["links"] == {}
        assert set(body) == {
            "ref",
            "label",
            "kind",
            "types",
            "description",
            "links",
            "gaps",
            "built_from",
        }
        assert graph.calls["node_arcs_many"] == 1

    def test_label_rejects_a_name_value_equal_to_its_own_dcid(self):
        """A `name` value equal to its node dcid is rejected for every shape."""
        graph = FakeGraphNodeReader(
            nodes={
                "dc/base/Foo": {
                    "typeOf": ("SomeOtherType",),
                    "name": ("dc/base/Foo", "Some Node"),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/dc/base/Foo")

        assert resp.status_code == 200, resp.text
        assert resp.json()["label"] == "Some Node"


class TestAbsentDcid:
    def test_unheld_dcid_refuses_node_not_held(self):
        graph = FakeGraphNodeReader(nodes={})
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/dc/base/Nowhere")

        assert resp.status_code == 404, resp.text
        assert resp.headers["content-type"] == "application/problem+json"
        body = resp.json()
        assert body["code"] == "node_not_held"
        assert body["type"] == "/codes/node_not_held"
        assert body["pointer"] == "/dcid"
        assert body["capability_version"] == GRAPH_CAPABILITY_VERSION
        assert isinstance(body["trace_id"], str) and body["trace_id"]

    def test_node_not_held_code_resolves_on_the_platform_codes_door(self):
        client = make_client()

        resp = client.get("/codes/node_not_held")

        assert resp.status_code == 200, resp.text
        assert resp.json()["code"] == "node_not_held"


class TestDcidLengthCeiling:
    """The handler rejects dcids over 512 characters before any graph read."""

    def test_oversized_dcid_refuses_request_invalid_before_any_read(self):
        graph = FakeGraphNodeReader(nodes={})
        client = make_client(graph=graph)
        dcid = "a" * 513

        with client:
            resp = client.get(f"/api/graph/node/{dcid}")

        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert body["code"] == "request_invalid"
        assert body["pointer"] == "/dcid"
        assert dcid not in body["detail"]
        assert graph.calls["node_arcs_many"] == 0

    def test_dcid_at_the_ceiling_reaches_the_reader(self):
        graph = FakeGraphNodeReader(nodes={})
        client = make_client(graph=graph)
        dcid = "a" * 512

        with client:
            resp = client.get(f"/api/graph/node/{dcid}")

        assert resp.status_code == 404, resp.text
        assert resp.json()["code"] == "node_not_held"
        assert graph.calls["node_arcs_many"] == 1

    def test_oversized_multi_segment_dcid_refuses_the_same_way(self):
        graph = FakeGraphNodeReader(nodes={})
        client = make_client(graph=graph)
        dcid = "/".join(["a" * 100] * 6)  # 605 characters, well over the ceiling
        assert len(dcid) > 512

        with client:
            resp = client.get(f"/api/graph/node/{dcid}")

        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert body["code"] == "request_invalid"
        assert body["pointer"] == "/dcid"
        assert graph.calls["node_arcs_many"] == 0

    def test_classify_node_exception_maps_dcid_too_long_directly(self):
        problem = node_door.classify_node_exception(node_door.DcidTooLong(600))
        assert problem is not None
        assert problem.status == 400
        assert problem.code == "request_invalid"
        assert problem.pointer == "/dcid"
        assert problem.retryable is False


class TestCapabilitiesCoexistWithNodeRoutes:
    """The composed router distinguishes an unheld dcid from a missing route."""

    def test_capabilities_and_unheld_dcid_404_coexist(self):
        graph = FakeGraphNodeReader(nodes={})
        client = make_client(graph=graph)

        with client:
            caps_resp = client.get("/api/graph/capabilities")
            node_resp = client.get("/api/graph/node/dc/base/Nowhere")

        assert caps_resp.status_code == 200, caps_resp.text
        assert node_resp.status_code == 404, node_resp.text
        assert node_resp.json()["code"] == "node_not_held"

    def test_capabilities_and_routing_404_coexist(self):
        client = make_client()

        with client:
            caps_resp = client.get("/api/graph/capabilities")
            routing_resp = client.get("/api/graph/node/")

        assert caps_resp.status_code == 200, caps_resp.text
        assert routing_resp.status_code == 404, routing_resp.text
        assert routing_resp.json()["code"] == "not_found"


class TestRouting:
    def test_multi_segment_dcids_route_to_the_node_door(self):
        graph = FakeGraphNodeReader(
            nodes={
                "country/KEN": {"typeOf": ("Country",), "name": ("Kenya",)},
                "dc/base/Foo": {"typeOf": ("Thing",), "name": ("Foo",)},
                "a/b/c": {"typeOf": ("Thing",), "name": ("ABC",)},
            }
        )
        client = make_client(graph=graph)

        with client:
            for dcid in ("country/KEN", "dc/base/Foo", "a/b/c"):
                resp = client.get(f"/api/graph/node/{dcid}")
                assert resp.status_code == 200, resp.text
                assert resp.json()["ref"] == dcid

    def test_empty_dcid_falls_through_to_routing_404(self):
        client = make_client()

        resp = client.get("/api/graph/node/")

        assert resp.status_code == 404, resp.text
        assert resp.json()["code"] == "not_found"


def test_same_reader_instance_serves_both_the_resolve_and_node_doors():
    graph = FakeGraphNodeReader(nodes={"country/KEN": {"typeOf": ("Country",), "name": ("Kenya",)}})
    client = make_client(graph=graph)

    with client:
        resolve_resp = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})
        node_resp = client.get("/api/graph/node/country/KEN")

    assert resolve_resp.status_code == 200, resolve_resp.text
    assert node_resp.status_code == 200, node_resp.text
    assert graph.calls["node_arcs_many"] == 2


_BATCH_NODES: dict[str, dict[str, tuple[str, ...]]] = {
    "dc/base/BilateralFlow_Grants": {
        "typeOf": ("StatisticalVariable",),
        "name": ("Bilateral flow (grants)",),
        "definition": ("mq=Nominal,mp=amount,pt=BilateralFlow",),
        "description": ("Grants between a provider and a recipient",),
        "observationProperties": ("provider", "recipient"),
    },
    "dc/base/WDI": {
        "typeOf": ("Provenance",),
        "name": ("WorldDevelopmentIndicators",),
        "source": ("dc/s/WorldBank",),
        "isPartOf": ("dc/d/WorldBank_WDI",),
    },
    "dc/s/WorldBank": {"name": ("World Bank",)},
    "dc/d/WorldBank_WDI": {"name": ("World Development Indicators",)},
    "country/KEN": {
        "typeOf": ("Country",),
        "name": ("Kenya",),
        "containedInPlace": ("africa", "worldbank/region1"),
    },
    "africa": {"name": ("Africa",)},
    "dc/base/Foo": {"typeOf": ("SomeOtherType",), "name": ("Some Node",)},
}


class TestNodesBatch:
    def test_each_entry_matches_the_get_body_and_unheld_dcids_read_not_held(self):
        dcids = [
            "dc/base/BilateralFlow_Grants",
            "dc/base/WDI",
            "dc/base/Nowhere",
            "country/KEN",
            "dc/base/Foo",
            "dc/base/WDI",
        ]
        graph = FakeGraphNodeReader(nodes=_BATCH_NODES)
        client = make_client(graph=graph)

        with client:
            resp = client.post("/api/graph/nodes", json={"dcids": dcids})
            assert resp.status_code == 200, resp.text
            # The arc read, the label targets' name read, and the variable's definition read.
            assert graph.calls["node_arcs_many"] == 3
            entries = resp.json()["nodes"]
            singles = [client.get(f"/api/graph/node/{dcid}") for dcid in dcids]

        assert [entry["ref"] for entry in entries] == dcids
        for entry, single in zip(entries, singles, strict=True):
            if single.status_code == 404:
                assert entry == {"ref": entry["ref"], "status": "not_held"}
            else:
                assert single.status_code == 200, single.text
                assert entry == single.json()
        assert [entry.get("kind", entry.get("status")) for entry in entries] == [
            "stat_var",
            "provenance",
            "not_held",
            "country",
            "other",
            "provenance",
        ]
        assert entries[0]["description"] == "Grants between a provider and a recipient"

    def test_a_batch_without_label_targets_makes_one_read(self):
        graph = FakeGraphNodeReader(nodes=_BATCH_NODES)
        client = make_client(graph=graph)

        with client:
            resp = client.post(
                "/api/graph/nodes", json={"dcids": ["dc/base/Foo", "dc/base/Nowhere"]}
            )

        assert resp.status_code == 200, resp.text
        assert graph.calls["node_arcs_many"] == 1

    def test_a_batch_at_the_cap_reads_and_one_over_refuses_before_any_read(self):
        graph = FakeGraphNodeReader(nodes=_BATCH_NODES)
        client = make_client(graph=graph)
        at_cap = [f"dc/base/Node{i}" for i in range(NODES_MAX_DCIDS)]

        with client:
            over = client.post("/api/graph/nodes", json={"dcids": [*at_cap, "dc/base/Extra"]})
            assert graph.calls["node_arcs_many"] == 0
            at = client.post("/api/graph/nodes", json={"dcids": at_cap})

        assert over.status_code == 400, over.text
        assert over.json()["code"] == "request_invalid"
        assert at.status_code == 200, at.text
        assert len(at.json()["nodes"]) == NODES_MAX_DCIDS

    @pytest.mark.parametrize(
        "dcids",
        [[], ["dc/base/"], ["a//b"], ["x" * (routing.DCID_MAX_LENGTH + 1)]],
        ids=["empty", "trailing_slash", "empty_segment", "over_length"],
    )
    def test_an_invalid_dcid_list_refuses_request_invalid(self, dcids):
        graph = FakeGraphNodeReader(nodes=_BATCH_NODES)
        client = make_client(graph=graph)

        with client:
            resp = client.post("/api/graph/nodes", json={"dcids": dcids})

        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"
        assert graph.calls["node_arcs_many"] == 0

    def test_a_batch_holds_one_pool_worker(self):
        """A blocked batch of seven dcids holds one of two workers, so a node read alongside it
        runs without waiting for the batch."""
        delegate = FakeGraphNodeReader(nodes=_BATCH_NODES)
        graph = _BlockingGraphNodeReader(delegate, block_batches_only=True)
        client = make_client(graph=graph, node_pool_size=2)

        with client:
            batch_response: list = []
            batch_thread = threading.Thread(
                target=lambda: batch_response.append(
                    client.post("/api/graph/nodes", json={"dcids": list(_BATCH_NODES)})
                )
            )
            batch_thread.start()
            assert graph.entered.wait(timeout=5.0), "the batch never reached the blocking read"

            alongside = client.get("/api/graph/node/dc/base/Foo")
            assert alongside.status_code == 200, alongside.text

            graph.release.set()
            batch_thread.join(timeout=5.0)

        assert batch_response, "the batch never returned"
        assert batch_response[0].status_code == 200, batch_response[0].text


class _RecordingGraphNodeReader(FakeGraphNodeReader):
    """Record the dcids of every `node_arcs_many` call."""

    def __init__(self, *, nodes) -> None:
        super().__init__(nodes=nodes)
        self.reads: list[list[str]] = []

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        self.reads.append(list(dcids))
        return super().node_arcs_many(dcids, properties=properties, deadline=deadline)


class TestNodesBatchCache:
    def test_a_repeated_batch_makes_no_graph_read(self):
        dcids = list(_BATCH_NODES)
        graph = FakeGraphNodeReader(nodes=_BATCH_NODES)
        client = make_client(graph=graph)

        with client:
            first = client.post("/api/graph/nodes", json={"dcids": dcids})
            reads = graph.calls["node_arcs_many"]
            second = client.post("/api/graph/nodes", json={"dcids": dcids})

        assert first.status_code == second.status_code == 200, second.text
        assert reads > 0
        assert graph.calls["node_arcs_many"] == reads
        assert second.json() == first.json()

    def test_a_batch_reads_only_its_misses_and_answers_in_request_order(self):
        graph = _RecordingGraphNodeReader(nodes=_BATCH_NODES)
        client = make_client(graph=graph)
        dcids = ["country/KEN", "dc/base/Nowhere", "dc/base/WDI", "dc/base/Foo", "country/KEN"]

        with client:
            client.post("/api/graph/nodes", json={"dcids": ["dc/base/Foo", "country/KEN"]})
            graph.reads.clear()
            merged = client.post("/api/graph/nodes", json={"dcids": dcids})
        with make_client(graph=FakeGraphNodeReader(nodes=_BATCH_NODES)) as fresh:
            uncached = fresh.post("/api/graph/nodes", json={"dcids": dcids})

        assert merged.status_code == 200, merged.text
        # The hold read, then the WDI provenance's label read.
        assert graph.reads == [
            ["dc/base/Nowhere", "dc/base/WDI"],
            ["dc/s/WorldBank", "dc/d/WorldBank_WDI"],
        ]
        assert merged.json() == uncached.json()

    def test_a_new_generation_misses(self):
        graph = FakeGraphNodeReader(nodes=_BATCH_NODES)
        client = make_client(graph=graph)

        with client:
            client.post("/api/graph/nodes", json={"dcids": ["dc/base/Foo"]})
            deps = client.app.state.graph_deps
            client.app.state.graph_deps = replace(deps, generation=FakeGeneration(token="rev-2"))
            resp = client.post("/api/graph/nodes", json={"dcids": ["dc/base/Foo"]})

        assert resp.status_code == 200, resp.text
        assert graph.calls["node_arcs_many"] == 2

    @pytest.mark.parametrize(
        ("query", "reads"), [("", 1), ("?as_of=2020-01-01", 2)], ids=["undated", "as-of"]
    )
    def test_a_repeated_get_reads_the_graph_only_when_it_carries_as_of(self, query, reads):
        """A GET without `as_of` reads through the batch's entries; a dated GET always reads."""
        graph = FakeGraphNodeReader(nodes=_BATCH_NODES)
        client = make_client(graph=graph)

        with client:
            first = client.get(f"/api/graph/node/dc/base/Foo{query}")
            second = client.get(f"/api/graph/node/dc/base/Foo{query}")

        assert first.status_code == second.status_code == 200, second.text
        assert second.json() == first.json()
        assert graph.calls["node_arcs_many"] == reads

    def test_a_generation_change_clears_the_entries(self, monkeypatch):
        """The lifespan registers the entries with the generation provider, whose clear
        empties them."""
        generation = FakeGeneration(token="rev-1")
        monkeypatch.setattr(graph_app, "acquire_shared_generation", lambda **_kwargs: generation)
        monkeypatch.setattr(graph_app, "release_shared_generation", lambda _generation: None)
        client = make_client(graph=FakeGraphNodeReader(nodes=_BATCH_NODES))

        with client:
            client.post("/api/graph/nodes", json={"dcids": ["dc/base/Foo"]})
            client.post("/api/graph/nodes", json={"dcids": ["dc/base/Foo"]})
            entries = client.app.state.graph_deps.node_entries
            assert entries.stats() == {"hits": 1, "misses": 1, "entries": 1}
            generation.advance("rev-2")
            wait_until(lambda: entries.stats()["entries"] == 0)

        assert generation.registered_names() == ()

    def test_an_entry_with_a_transient_gap_is_not_stored(self):
        """A coverage index that failed to load leaves the variable's availability with a
        transient gap, so the next batch reads the variable again and serves the other node from
        the cache."""
        graph = _RecordingGraphNodeReader(nodes=_BATCH_NODES)
        client = make_client(graph=graph)
        dcids = ["dc/base/BilateralFlow_Grants", "dc/base/Foo"]

        with client:
            deps = client.app.state.graph_deps
            client.app.state.graph_deps = replace(deps, coverage_load_failed=True)
            first = client.post("/api/graph/nodes", json={"dcids": dcids})
            graph.reads.clear()
            client.post("/api/graph/nodes", json={"dcids": dcids})

        assert first.json()["nodes"][0]["stat_var"]["availability"] is None
        assert graph.reads[0] == ["dc/base/BilateralFlow_Grants"]


class _BlockingGraphNodeReader:
    """Block `node_arcs_many` until the test releases it."""

    def __init__(self, delegate: FakeGraphNodeReader, *, block_batches_only: bool = False) -> None:
        self._delegate = delegate
        self._block_batches_only = block_batches_only
        self.entered = threading.Event()
        self.release = threading.Event()
        self.entries = 0
        self.calls = delegate.calls

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        if self._block_batches_only and len(dcids) == 1:
            return self._delegate.node_arcs_many(dcids, properties=properties, deadline=deadline)
        self.entries += 1
        self.entered.set()
        self.release.wait(timeout=5.0)
        return self._delegate.node_arcs_many(dcids, properties=properties, deadline=deadline)

    def node_names_many(
        self,
        dcids,
        *,
        deadline: float | None = None,
    ):
        return self._delegate.node_names_many(dcids)


def test_a_read_past_the_pool_size_is_refused_at_once():
    """With every worker busy a read answers retryable 503 without reaching the reader. The
    in-pool read and a read after it succeed. The read after it names another node, which no
    cached envelope answers, so it reaches the pool."""
    delegate = FakeGraphNodeReader(
        nodes={"country/KEN": {"typeOf": ("Country",)}, "country/UGA": {"typeOf": ("Country",)}}
    )
    graph = _BlockingGraphNodeReader(delegate)
    client = make_client(graph=graph, node_pool_size=1)

    with client:
        responses = send_past_a_full_pool(
            client,
            lambda: client.get("/api/graph/node/country/KEN"),
            entered=graph.entered,
            release=graph.release,
        )
        after = client.get("/api/graph/node/country/UGA")

    in_pool, refused = responses
    assert in_pool.status_code == 200, in_pool.text
    assert refused.status_code == 503, refused.text
    assert refused.json()["code"] == "service_unavailable"
    assert refused.json()["retryable"] is True
    assert refused.headers["retry-after"] == "1"
    assert after.status_code == 200, after.text
    assert graph.entries == 2


def test_a_read_while_a_timed_out_read_holds_the_worker_is_refused_at_once(monkeypatch):
    """A read that outlives its door timeout answers 504 and keeps its worker until it
    returns, so a read sent meanwhile answers retryable 503 without reaching the reader. The
    next read runs once that worker returns."""
    monkeypatch.setattr(node_door, "GRAPH_NODE_TIMEOUT_SECONDS", 0.2)
    delegate = FakeGraphNodeReader(nodes={"country/KEN": {"typeOf": ("Country",)}})
    graph = _BlockingGraphNodeReader(delegate)
    client = make_client(graph=graph, node_pool_size=1)

    with client:
        running = client.get("/api/graph/node/country/KEN")
        assert graph.entered.wait(timeout=5.0), "the first read never started"
        refused = client.get("/api/graph/node/country/KEN")
        graph.release.set()
        wait_until(lambda: client.app.state.graph_deps.dcp_executor._in_flight == 0)
        recovered = client.get("/api/graph/node/country/KEN")

    assert running.status_code == 504, running.text
    assert "retry-after" not in running.headers
    assert refused.status_code == 503, refused.text
    assert refused.headers["retry-after"] == "1"
    assert refused.json()["code"] == "service_unavailable"
    assert refused.json()["retryable"] is True
    assert recovered.status_code == 200, recovered.text
    assert graph.entries == 2


class TestGraphReadErrorClassification:
    def test_upstream_4xx_maps_to_502_upstream_error(self):
        problem = node_door.classify_node_exception(GraphReadError("exemplar", upstream_status=404))
        assert problem is not None
        assert problem.status == 502
        assert problem.code == "upstream_error"
        assert problem.retryable is True

    def test_no_upstream_status_maps_to_503_service_unavailable(self):
        problem = node_door.classify_node_exception(GraphReadError("exemplar"))
        assert problem is not None
        assert problem.status == 503
        assert problem.code == "service_unavailable"
        assert problem.retryable is True


class TestGraphDeadlineExceededClassification:
    def test_maps_to_504_timeout_before_the_general_graph_read_error_arm(self):
        # `GraphDeadlineExceededError` subclasses `GraphReadError`. Its 504
        # arm must precede the general read-error arm.
        problem = node_door.classify_node_exception(GraphDeadlineExceededError("exemplar"))
        assert problem is not None
        assert problem.status == 504
        assert problem.code == "timeout"
        assert problem.retryable is True


class _RaisingGraphNodeReader:
    """`node_arcs_many()` raises `GraphReadError` outright -- exercises the
    node door's own classify branch end-to-end, through a real request.
    """

    def __init__(self, exc: GraphReadError) -> None:
        self._exc = exc

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        raise self._exc

    def node_names_many(
        self,
        dcids,
        *,
        deadline: float | None = None,
    ):
        raise self._exc


def test_graph_read_error_end_to_end_maps_to_502():
    graph = _RaisingGraphNodeReader(GraphReadError("upstream refused", upstream_status=404))
    client = make_client(graph=graph)

    with client:
        resp = client.get("/api/graph/node/country/KEN")

    assert resp.status_code == 502, resp.text
    assert resp.json()["code"] == "upstream_error"


def test_graph_deadline_exceeded_end_to_end_maps_to_504_timeout():
    graph = _RaisingGraphNodeReader(GraphDeadlineExceededError("exemplar"))
    client = make_client(graph=graph)

    with client:
        resp = client.get("/api/graph/node/country/KEN")

    assert resp.status_code == 504, resp.text
    problem = resp.json()
    assert problem["code"] == "timeout"
    assert problem["retryable"] is True
