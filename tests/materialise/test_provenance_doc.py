"""Tests provenance and dataset source documents with synthetic graphs and recorded Graph API fixtures.

Recorded `/v2/node` responses are parsed through `parse_arcs`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from dfo_provenance_docs import (
    MAX_DOCUMENT_BYTES,
    ChangeKind,
    ChangelogChange,
    ChangelogRelease,
    DocumentErrorReason,
    TransformationStep,
    decode_mcf_text,
    parse_notes,
)

import materialise.provenance_doc as provenance_doc_module
from materialise.graph import UNRESOLVED_NODE, GraphReadError, SubjectPage, parse_arcs
from materialise.provenance_doc import (
    _DATASET_PROPERTIES,
    _NODE_PROPERTIES,
    _PUBLISHER_PROPERTIES,
    _ROLLUP_PROPERTIES,
    _SOLE_CHILD_PROPERTIES,
    CONFLICTING_VALUES,
    UNCONDITIONAL_GAPS,
    CoverageFacts,
    DocumentIssue,
    NotASourceNode,
    ProvenanceDoc,
    SourceDocumentNotFound,
    TooManyChildProvenances,
    _compute_gaps,
    _read_document,
    provenance_names,
    read_provenance_doc,
    read_source_citations,
)
from origin_catalogue import BASE, UN_SYSTEM, SourceRef
from origin_catalogue.card_facts import DOCUMENT_PROPERTIES, LICENCE_PROPERTIES
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue

from .conftest import FakeGraphReader

_GRAPH_API_FIXTURES_DIR = Path(__file__).parent.parent / "graph_api" / "fixtures"


def _wire_arcs(
    fixture_name: str, dcid: str, properties: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    body = json.loads((_GRAPH_API_FIXTURES_DIR / fixture_name).read_text())
    arcs = body["data"][dcid]["arcs"]
    return parse_arcs(arcs, properties, mark_unresolved=True)


def _wire_subject_dcids(fixture_name: str, dcid: str, property: str) -> tuple[str, ...]:
    body = json.loads((_GRAPH_API_FIXTURES_DIR / fixture_name).read_text())
    nodes = body["data"][dcid]["arcs"].get(property, {}).get("nodes", [])
    return tuple(n["dcid"] for n in nodes if "dcid" in n)


class TestOrphanProvenance:
    def test_orphan_with_no_source_arc_serves_its_own_fields_only(self):
        graph = FakeGraphReader(
            arcs={
                "provenance/Orphan": {
                    "typeOf": ("Provenance",),
                    "name": ("Orphan Provenance",),
                    "url": ("https://example.org",),
                    "description": ("An orphan import.",),
                    "licenseType": ("PublicDomain",),
                    "sourceReleaseFrequency": ("P1Y",),
                    "lastDataRefreshDate": ("2024-01-01",),
                    "nextDataRefreshDate": ("2025-01-01",),
                }
            }
        )

        doc = read_provenance_doc(graph, "provenance/Orphan")

        assert doc == ProvenanceDoc(
            dcid="provenance/Orphan",
            provenances=("provenance/Orphan",),
            name_of_source="Orphan Provenance",
            publisher=None,
            source=None,
            data_commons=None,
            description="An orphan import.",
            source_url="https://example.org",
            source_data_url=None,
            license_type="PublicDomain",
            license_url=None,
            update_cadence="P1Y",
            last_update="2024-01-01",
            next_expected_update="2025-01-01",
            years_covered=None,
            citation=None,
            granularity=None,
            currencies=(),
            methodology=None,
            data_notes=(),
            transformation_log=(),
            version_history=(),
            document_errors=(),
            gaps=(
                "details.publisher",
                "details.source",
                "details.data_commons",
                "details.source_data_url",
                "details.citation",
                "license.url",
                "scope_and_units.countries",
                "scope_and_units.years_covered",
                "scope_and_units.granularity",
                "scope_and_units.currencies",
                "methodology.description",
                "data_notes",
                "usage.api_access",
                "usage.python",
                "transformation_log",
                "version_history",
            ),
        )
        # The node's identity read, the batch reading the orphan as its own
        # sole child, then the licence node's name. No isPartOf, so no dataset
        # climb and no children read.
        assert graph.node_arcs_many_calls == [
            (("provenance/Orphan",), _NODE_PROPERTIES),
            (("provenance/Orphan",), (*_ROLLUP_PROPERTIES, *_SOLE_CHILD_PROPERTIES)),
            (("PublicDomain",), LICENCE_PROPERTIES),
        ]
        assert graph.calls["node_subjects"] == 0

    def test_orphan_with_source_arc_resolves_publisher_via_one_more_batch(self):
        graph = FakeGraphReader(
            arcs={
                "provenance/OrphanWithSource": {
                    "typeOf": ("Provenance",),
                    "name": ("Orphan Provenance",),
                    "source": ("dc/s/SomePublisher",),
                },
                "dc/s/SomePublisher": {"name": ("Some Publisher",)},
            }
        )

        doc = read_provenance_doc(graph, "provenance/OrphanWithSource")

        assert doc.publisher == "Some Publisher"
        assert graph.calls["node_arcs_many"] == 2

    def test_orphan_publisher_label_rejects_a_name_value_equal_to_its_own_dcid(self):
        """The CRS contested-name hazard recorded on the OECD Source
        node: a junk `name` triple whose value is the target's own dcid,
        competing with the real label.
        """
        graph = FakeGraphReader(
            arcs={
                "provenance/Orphan": {"typeOf": ("Provenance",), "source": ("dc/s/Contested",)},
                "dc/s/Contested": {"name": ("dc/s/Contested", "Real Publisher Name")},
            }
        )

        doc = read_provenance_doc(graph, "provenance/Orphan")

        assert doc.publisher == "Real Publisher Name"


class TestOrigin:
    def _dataset(self, *children: str) -> FakeGraphReader:
        return FakeGraphReader(
            arcs={"dc/d/Dataset1": {"typeOf": ("Dataset",)}},
            subjects={("dc/d/Dataset1", "isPartOf"): SubjectPage(dcids=children, truncated=False)},
        )

    def test_children_sharing_one_admitted_source_serve_it_ungapped(self):
        catalogue = restrictive_catalogue(
            provenances=[
                provenance_row("dc/base/Child1", BASE, source="dc/s/WorldBank"),
                provenance_row("dc/base/Child2", BASE, source="dc/s/WorldBank"),
            ]
        )

        doc = read_provenance_doc(
            self._dataset("dc/base/Child1", "dc/base/Child2"), "dc/d/Dataset1", catalogue=catalogue
        )

        assert doc.provenances == ("dc/base/Child1", "dc/base/Child2")
        assert doc.source == SourceRef(ref="dc/s/WorldBank", label="dc/s/WorldBank")
        assert doc.data_commons == BASE
        assert "details.source" not in doc.gaps
        assert "details.data_commons" not in doc.gaps

    def test_children_under_two_sources_gap_the_source_and_keep_their_shared_origin(self):
        catalogue = restrictive_catalogue(
            provenances=[
                provenance_row("dc/base/Child1", BASE, source="dc/s/WorldBank"),
                provenance_row("dc/base/Child2", BASE, source="dc/s/OECD"),
            ]
        )

        doc = read_provenance_doc(
            self._dataset("dc/base/Child1", "dc/base/Child2"), "dc/d/Dataset1", catalogue=catalogue
        )

        assert doc.source is None
        assert doc.data_commons == BASE
        assert "details.source" in doc.gaps

    def test_a_dataset_rolls_up_only_the_children_the_catalogue_admits(self):
        catalogue = restrictive_catalogue(
            provenances=[provenance_row("dc/base/Child1", BASE, source="dc/s/WorldBank")]
        )

        doc = read_provenance_doc(
            self._dataset("dc/base/Child1", "dc/base/Child2"), "dc/d/Dataset1", catalogue=catalogue
        )

        assert doc.provenances == ("dc/base/Child1",)
        assert doc.source == SourceRef(ref="dc/s/WorldBank", label="dc/s/WorldBank")
        assert doc.data_commons == BASE

    def test_a_dataset_with_no_admitted_child_is_not_found(self):
        catalogue = restrictive_catalogue(provenances=[provenance_row("dc/base/Other", BASE)])

        with pytest.raises(SourceDocumentNotFound, match="no Provenance child"):
            read_provenance_doc(
                self._dataset("dc/base/Child1", "dc/base/Child2"),
                "dc/d/Dataset1",
                catalogue=catalogue,
            )
        with pytest.raises(SourceDocumentNotFound):
            read_provenance_doc(self._dataset(), "dc/d/Dataset1", catalogue=catalogue)

    def test_without_a_catalogue_a_dataset_rolls_up_every_child(self):
        doc = read_provenance_doc(
            self._dataset("dc/base/Child1", "dc/base/Child2"), "dc/d/Dataset1"
        )

        assert doc.provenances == ("dc/base/Child1", "dc/base/Child2")

    def test_a_un_agency_provenance_is_its_own_source(self):
        graph = FakeGraphReader(arcs={"undata/p/SDG": {"typeOf": ("Provenance",)}})
        row = provenance_row("undata/p/SDG", UN_SYSTEM, segment="sdg")

        doc = read_provenance_doc(
            graph, "undata/p/SDG", catalogue=restrictive_catalogue(provenances=[row])
        )

        assert doc.source == SourceRef(ref="undata/p/SDG", label="undata/p/SDG")
        assert doc.data_commons == UN_SYSTEM

    def test_a_provenance_in_a_dataset_uses_its_own_row_not_its_siblings(self):
        graph = FakeGraphReader(
            arcs={
                "undata/p/SDG": {"typeOf": ("Provenance",), "isPartOf": ("dc/d/UN",)},
                "dc/d/UN": {"typeOf": ("Dataset",)},
            },
            subjects={
                ("dc/d/UN", "isPartOf"): SubjectPage(
                    dcids=("undata/p/SDG", "undata/p/Unlisted"), truncated=False
                )
            },
        )
        # The admitted sibling is its own source, so the Dataset's two children disagree.
        catalogue = restrictive_catalogue(
            provenances=[
                provenance_row("undata/p/SDG", UN_SYSTEM, segment="sdg"),
                provenance_row("undata/p/Unlisted", UN_SYSTEM, segment="unlisted"),
            ]
        )

        doc = read_provenance_doc(graph, "undata/p/SDG", catalogue=catalogue)
        dataset_doc = read_provenance_doc(graph, "dc/d/UN", catalogue=catalogue)

        assert doc.source == SourceRef(ref="undata/p/SDG", label="undata/p/SDG")
        assert doc.data_commons == UN_SYSTEM
        assert dataset_doc.provenances == ("undata/p/SDG", "undata/p/Unlisted")
        assert dataset_doc.source is None
        assert "details.source" in dataset_doc.gaps

    def test_a_provenance_with_a_catalogue_row_is_named_from_the_row(self):
        graph = FakeGraphReader(
            arcs={
                "undata/p/WHO": {"typeOf": ("Provenance",), "isPartOf": ("dc/d/UN",)},
                "undata/p/Withheld": {"typeOf": ("Provenance",), "isPartOf": ("dc/d/UN",)},
                "dc/d/UN": {
                    "typeOf": ("Dataset",),
                    "name": ("United Nations Statistics Division",),
                },
            },
            subjects={
                ("dc/d/UN", "isPartOf"): SubjectPage(
                    dcids=("undata/p/WHO", "undata/p/Withheld"), truncated=False
                )
            },
        )
        row = replace(
            provenance_row("undata/p/WHO", UN_SYSTEM, segment="who"),
            name="World Health Organization (WHO)",
        )
        catalogue = restrictive_catalogue(provenances=[row])

        doc = read_provenance_doc(graph, "undata/p/WHO", catalogue=catalogue)
        dataset_doc = read_provenance_doc(graph, "dc/d/UN", catalogue=catalogue)
        # The catalogue withholds this sibling, so it has no row to be named from.
        rowless = read_provenance_doc(graph, "undata/p/Withheld", catalogue=catalogue)

        assert doc.name_of_source == "World Health Organization (WHO)"
        assert dataset_doc.name_of_source == "United Nations Statistics Division"
        assert rowless.name_of_source == "United Nations Statistics Division"

    def test_with_the_un_origin_off_a_un_provenance_is_not_labelled(self):
        graph = FakeGraphReader(arcs={"undata/p/SDG": {"typeOf": ("Provenance",)}})
        catalogue = restrictive_catalogue(
            provenances=[provenance_row("undata/p/SDG", UN_SYSTEM, segment="sdg")],
            un_system_enabled=False,
        )

        doc = read_provenance_doc(graph, "undata/p/SDG", catalogue=catalogue)

        assert (doc.source, doc.data_commons) == (None, None)


class TestCoverageFacts:
    def test_orphan_passes_itself_and_found_facts_leave_their_paths_ungapped(self):
        graph = FakeGraphReader(arcs={"provenance/Orphan": {"typeOf": ("Provenance",)}})
        seen: list[tuple[str, ...]] = []

        def coverage_facts_of(provenance_dcids: tuple[str, ...]) -> CoverageFacts:
            seen.append(provenance_dcids)
            return CoverageFacts(
                currencies=("EUR", "USD"), granularity="Annual, by provider and recipient"
            )

        doc = read_provenance_doc(graph, "provenance/Orphan", coverage_facts_of=coverage_facts_of)

        assert seen == [("provenance/Orphan",)]
        assert doc.currencies == ("EUR", "USD")
        assert doc.granularity == "Annual, by provider and recipient"
        assert "scope_and_units.currencies" not in doc.gaps
        assert "scope_and_units.granularity" not in doc.gaps

    def test_dataset_passes_every_child_provenance(self):
        graph = FakeGraphReader(
            arcs={"dc/d/Dataset1": {"typeOf": ("Dataset",)}},
            subjects={
                ("dc/d/Dataset1", "isPartOf"): SubjectPage(
                    dcids=("dc/base/Child1", "dc/base/Child2"), truncated=False
                )
            },
        )
        seen: list[tuple[str, ...]] = []

        def coverage_facts_of(provenance_dcids: tuple[str, ...]) -> CoverageFacts:
            seen.append(provenance_dcids)
            return CoverageFacts()

        doc = read_provenance_doc(graph, "dc/d/Dataset1", coverage_facts_of=coverage_facts_of)

        assert seen == [("dc/base/Child1", "dc/base/Child2")]
        assert doc.currencies == ()
        assert doc.granularity is None
        assert "scope_and_units.currencies" in doc.gaps
        assert "scope_and_units.granularity" in doc.gaps

    def test_each_fact_gaps_independently(self):
        graph = FakeGraphReader(arcs={"provenance/Orphan": {"typeOf": ("Provenance",)}})

        doc = read_provenance_doc(
            graph,
            "provenance/Orphan",
            coverage_facts_of=lambda _dcids: CoverageFacts(granularity="Annual, by x and y"),
        )

        assert "scope_and_units.currencies" in doc.gaps
        assert "scope_and_units.granularity" not in doc.gaps

    def test_without_a_lookup_both_facts_are_null_and_gapped(self):
        graph = FakeGraphReader(arcs={"provenance/Orphan": {"typeOf": ("Provenance",)}})

        doc = read_provenance_doc(graph, "provenance/Orphan")

        assert doc.currencies == ()
        assert doc.granularity is None
        assert "scope_and_units.currencies" in doc.gaps
        assert "scope_and_units.granularity" in doc.gaps

    def test_a_stored_data_granularity_arc_is_neither_read_nor_served(self):
        graph = _orphan(dataGranularity=("Annual, by provider and recipient",))

        doc = read_provenance_doc(graph, "provenance/Orphan")

        assert "dataGranularity" not in {*_ROLLUP_PROPERTIES, *_SOLE_CHILD_PROPERTIES}
        assert doc.granularity is None
        assert "scope_and_units.granularity" in doc.gaps


class TestDatasetSingleChild:
    def test_single_child_serves_that_child_s_description_and_dataset_s_own_fields(self):
        graph = FakeGraphReader(
            arcs={
                "dc/d/Dataset1": {
                    "typeOf": ("Dataset",),
                    "name": ("Readable Dataset Name",),
                    "url": ("https://dataset.example.org",),
                    "isPartOf": ("dc/s/Publisher1",),
                },
                "dc/s/Publisher1": {"name": ("Publisher One",)},
                "dc/base/Child1": {
                    "description": ("The child's own description.",),
                    "licenseType": ("PublicDomain",),
                    "sourceReleaseFrequency": ("P1M",),
                    "lastDataRefreshDate": ("2024-06-01",),
                    "nextDataRefreshDate": ("2024-07-01",),
                },
            },
            subjects={
                ("dc/d/Dataset1", "isPartOf"): SubjectPage(
                    dcids=("dc/base/Child1",), truncated=False
                )
            },
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.name_of_source == "Readable Dataset Name"
        assert doc.publisher == "Publisher One"
        assert doc.source_url == "https://dataset.example.org"
        assert doc.description == "The child's own description."
        assert doc.license_type == "PublicDomain"
        assert doc.update_cadence == "P1M"
        assert doc.last_update == "2024-06-01"
        assert doc.next_expected_update == "2024-07-01"
        # The one child carries no earliestObservationDate/latestObservationDate,
        # so years_covered stays a gap even though every other roll-up filled.
        assert doc.years_covered is None
        assert doc.gaps == (
            "details.source",
            "details.data_commons",
            "details.source_data_url",
            "details.citation",
            "license.url",
            "scope_and_units.countries",
            "scope_and_units.years_covered",
            "scope_and_units.granularity",
            "scope_and_units.currencies",
            "methodology.description",
            "data_notes",
            "usage.api_access",
            "usage.python",
            "transformation_log",
            "version_history",
        )
        assert graph.calls["node_subjects"] == 1
        # One node_arcs_many for the Dataset itself, one batched call
        # covering the publisher dcid and the one child dcid together, and one
        # for the licence node's name.
        assert graph.calls["node_arcs_many"] == 3

    def test_dataset_with_no_url_falls_back_to_its_sole_child_s_url(self):
        graph = FakeGraphReader(
            arcs={
                "dc/d/Dataset1": {"typeOf": ("Dataset",), "name": ("Name",)},
                "dc/base/Child1": {"url": ("https://child.example.org",)},
            },
            subjects={
                ("dc/d/Dataset1", "isPartOf"): SubjectPage(
                    dcids=("dc/base/Child1",), truncated=False
                )
            },
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.source_url == "https://child.example.org"

    def test_dataset_with_zero_children_leaves_every_rollup_field_empty(self):
        graph = FakeGraphReader(
            arcs={"dc/d/Empty": {"typeOf": ("Dataset",), "name": ("Empty Dataset",)}},
            subjects={("dc/d/Empty", "isPartOf"): SubjectPage(dcids=(), truncated=False)},
        )

        doc = read_provenance_doc(graph, "dc/d/Empty")

        assert doc.description is None
        assert doc.license_type is None
        assert doc.update_cadence is None
        assert doc.last_update is None
        assert doc.next_expected_update is None
        assert doc.source_url is None


class TestDatasetMultiChildRollup:
    def _dataset_graph(
        self,
        children: dict[str, dict[str, tuple[str, ...]]],
        licences: dict[str, dict[str, tuple[str, ...]]] | None = None,
    ) -> FakeGraphReader:
        """A Dataset with `children`, on a graph that also holds the `licences` nodes."""
        arcs = {"dc/d/Dataset1": {"typeOf": ("Dataset",), "name": ("Multi-Child Dataset",)}}
        arcs.update(children)
        arcs.update(licences or {})
        return FakeGraphReader(
            arcs=arcs,
            subjects={
                ("dc/d/Dataset1", "isPartOf"): SubjectPage(dcids=tuple(children), truncated=False)
            },
        )

    def test_description_is_empty_on_any_multi_child_dataset_even_when_children_agree(self):
        graph = self._dataset_graph(
            {
                "dc/base/ChildA": {"description": ("Same description.",)},
                "dc/base/ChildB": {"description": ("Same description.",)},
            }
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.description is None
        assert "details.description" in doc.gaps

    def test_license_type_agrees_when_every_child_declares_and_all_agree(self):
        graph = self._dataset_graph(
            {
                "dc/base/ChildA": {"licenseType": ("PublicDomain",)},
                "dc/base/ChildB": {"licenseType": ("PublicDomain",)},
            }
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.license_type == "PublicDomain"
        assert "license.type" not in doc.gaps

    def test_license_type_is_the_licence_node_s_graph_name(self):
        graph = self._dataset_graph(
            {
                "dc/base/ChildA": {"licenseType": ("InternationalMonetaryFundCopyrightAndUsage",)},
                "dc/base/ChildB": {"licenseType": ("InternationalMonetaryFundCopyrightAndUsage",)},
            },
            {"InternationalMonetaryFundCopyrightAndUsage": {"name": ("IMF Copyright and Usage",)}},
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.license_type == "IMF Copyright and Usage"
        assert graph.node_arcs_many_calls[-1] == (
            ("InternationalMonetaryFundCopyrightAndUsage",),
            LICENCE_PROPERTIES,
        )

    @pytest.mark.parametrize(
        "licence_node",
        [
            pytest.param({}, id="not_held"),
            pytest.param({"SomeLicense": {"name": ()}}, id="unnamed"),
            pytest.param({"SomeLicense": {"name": ("SomeLicense",)}}, id="named_by_its_dcid"),
        ],
    )
    def test_license_type_is_the_dcid_when_the_graph_has_no_name_for_it(self, licence_node):
        graph = self._dataset_graph(
            {
                "dc/base/ChildA": {"licenseType": ("SomeLicense",)},
                "dc/base/ChildB": {"licenseType": ("SomeLicense",)},
            },
            licence_node,
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.license_type == "SomeLicense"
        assert "license.type" not in doc.gaps

    def test_children_naming_no_licence_cost_no_licence_read(self):
        graph = self._dataset_graph({"dc/base/ChildA": {}, "dc/base/ChildB": {}})

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.license_type is None
        assert all(properties != LICENCE_PROPERTIES for _, properties in graph.node_arcs_many_calls)

    def test_an_unresolved_licence_name_serves_the_licence_dcid(self):
        graph = self._dataset_graph(
            {
                "dc/base/ChildA": {"licenseType": ("SomeLicense",)},
                "dc/base/ChildB": {"licenseType": ("SomeLicense",)},
            },
            {"SomeLicense": {"name": (UNRESOLVED_NODE,)}},
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.license_type == "SomeLicense"

    def test_children_naming_different_licences_cost_no_licence_read(self):
        graph = self._dataset_graph(
            {
                "dc/base/ChildA": {"licenseType": ("PublicDomain",)},
                "dc/base/ChildB": {"licenseType": ("CreativeCommonsAttributionLicense",)},
            }
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.license_type is None
        assert all(properties != LICENCE_PROPERTIES for _, properties in graph.node_arcs_many_calls)

    def test_license_type_is_empty_when_one_child_declares_nothing(self):
        """A child without `licenseType` leaves the dataset field empty."""
        graph = self._dataset_graph(
            {
                "dc/base/ChildA": {"licenseType": ("PublicDomain",)},
                "dc/base/ChildB": {"licenseType": ("PublicDomain",)},
                "dc/base/ChildC": {},
            }
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.license_type is None
        assert "license.type" in doc.gaps

    def test_license_type_is_empty_on_a_real_conflict(self):
        graph = self._dataset_graph(
            {
                "dc/base/ChildA": {"licenseType": ("PublicDomain",)},
                "dc/base/ChildB": {"licenseType": ("CreativeCommonsAttributionLicense",)},
            }
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.license_type is None
        assert "license.type" in doc.gaps

    def test_last_update_is_the_max_and_next_expected_update_is_the_min_across_children(self):
        graph = self._dataset_graph(
            {
                "dc/base/ChildA": {
                    "lastDataRefreshDate": ("2024-01-01",),
                    "nextDataRefreshDate": ("2025-06-01",),
                },
                "dc/base/ChildB": {
                    "lastDataRefreshDate": ("2024-06-01",),
                    "nextDataRefreshDate": ("2025-01-01",),
                },
            }
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.last_update == "2024-06-01"
        assert doc.next_expected_update == "2025-01-01"

    def test_last_update_is_order_independent_within_one_child_s_own_two_values(self):
        """Both asserted dates contribute to the maximum, regardless of arc order."""
        forward = self._dataset_graph(
            {"dc/base/ChildA": {"lastDataRefreshDate": ("2023-07-19", "2025-03-14")}}
        )
        reversed_ = self._dataset_graph(
            {"dc/base/ChildA": {"lastDataRefreshDate": ("2025-03-14", "2023-07-19")}}
        )

        assert read_provenance_doc(forward, "dc/d/Dataset1").last_update == "2025-03-14"
        assert read_provenance_doc(reversed_, "dc/d/Dataset1").last_update == "2025-03-14"

    def test_years_covered_is_the_earliest_start_paired_with_the_latest_end(self):
        graph = self._dataset_graph(
            {
                "dc/base/ChildA": {"earliestObservationDate": ("2010",)},
                "dc/base/ChildB": {
                    "earliestObservationDate": ("2015",),
                    "latestObservationDate": ("2023",),
                },
                # The year range uses available child dates.
                "dc/base/ChildC": {},
            }
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.years_covered == "2010-2023"
        assert "scope_and_units.years_covered" not in doc.gaps

    def test_years_covered_stays_empty_when_no_child_carries_either_date(self):
        graph = self._dataset_graph({"dc/base/ChildA": {"description": ("No dates here.",)}})

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.years_covered is None
        assert "scope_and_units.years_covered" in doc.gaps

    def test_years_covered_projects_mixed_precisions_down_to_a_plain_year_range(self):
        """Mixed ISO date precisions reduce to their leading years before forming a span."""
        graph = self._dataset_graph(
            {
                "dc/base/ChildA": {"earliestObservationDate": ("1940",)},
                "dc/base/ChildB": {"latestObservationDate": ("2024-11-03",)},
                "dc/base/ChildC": {
                    "earliestObservationDate": ("1962-01-02",),
                    "latestObservationDate": ("2026-05-07",),
                },
            }
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.years_covered == "1940-2026"
        assert "scope_and_units.years_covered" not in doc.gaps

    def test_years_covered_is_a_bare_year_when_start_and_end_match(self):
        graph = self._dataset_graph(
            {
                "dc/base/ChildA": {
                    "earliestObservationDate": ("2024-01-01",),
                    "latestObservationDate": ("2024-12-31",),
                }
            }
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.years_covered == "2024"
        assert "scope_and_units.years_covered" not in doc.gaps

    def test_years_covered_skips_a_value_with_no_readable_leading_year(self):
        graph = self._dataset_graph(
            {
                "dc/base/ChildA": {
                    "earliestObservationDate": ("unknown",),
                    "latestObservationDate": ("2023",),
                },
                "dc/base/ChildB": {"earliestObservationDate": ("2015",)},
            }
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.years_covered == "2015-2023"
        assert "scope_and_units.years_covered" not in doc.gaps

    def test_update_cadence_conflict_leaves_it_empty_independent_of_license_type_agreement(self):
        graph = self._dataset_graph(
            {
                "dc/base/ChildA": {
                    "licenseType": ("PublicDomain",),
                    "sourceReleaseFrequency": ("P1M",),
                },
                "dc/base/ChildB": {
                    "licenseType": ("PublicDomain",),
                    "sourceReleaseFrequency": ("P1Y",),
                },
            }
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.license_type == "PublicDomain"
        assert doc.update_cadence is None


class TestProvenanceWithParentDataset:
    def test_non_orphan_provenance_climbs_to_its_dataset_and_rolls_up_every_sibling(self):
        """A child with a parent Dataset rolls up fields across all siblings."""
        graph = FakeGraphReader(
            arcs={
                "dc/base/GivenChild": {
                    "typeOf": ("Provenance",),
                    "isPartOf": ("dc/d/Dataset1",),
                },
                "dc/d/Dataset1": {
                    "typeOf": ("Dataset",),
                    "name": ("Dataset One",),
                    "isPartOf": ("dc/s/Publisher1",),
                },
                "dc/s/Publisher1": {"name": ("Publisher One",)},
                "dc/base/GivenChild2": {"licenseType": ("PublicDomain",)},
                "dc/base/SiblingChild": {"licenseType": ("PublicDomain",)},
            },
            subjects={
                ("dc/d/Dataset1", "isPartOf"): SubjectPage(
                    dcids=("dc/base/GivenChild2", "dc/base/SiblingChild"), truncated=False
                )
            },
        )

        doc = read_provenance_doc(graph, "dc/base/GivenChild")

        assert doc.name_of_source == "Dataset One"
        assert doc.publisher == "Publisher One"
        assert doc.license_type == "PublicDomain"
        # Read #1 (the given node), read #2 (the parent Dataset), one
        # combined batch (#4) and the licence node's name (#5). node_subjects
        # is read #3.
        assert graph.calls["node_arcs_many"] == 4
        assert graph.calls["node_subjects"] == 1

    def test_dataset_with_self_referential_name_falls_through_to_the_given_nodes_own_name(self):
        """A Dataset name equal to its dcid falls back to the child's valid name."""
        graph = FakeGraphReader(
            arcs={
                "dc/base/GivenChild": {
                    "typeOf": ("Provenance",),
                    "isPartOf": ("dc/d/Dataset1",),
                    "name": ("Given Child Provenance",),
                },
                "dc/d/Dataset1": {"typeOf": ("Dataset",), "name": ("dc/d/Dataset1",)},
            },
            subjects={
                ("dc/d/Dataset1", "isPartOf"): SubjectPage(
                    dcids=("dc/base/GivenChild",), truncated=False
                )
            },
        )

        doc = read_provenance_doc(graph, "dc/base/GivenChild")

        assert doc.name_of_source == "Given Child Provenance"

    def test_dataset_name_of_source_rejects_a_name_value_equal_to_its_own_dcid(self):
        """The skip side of the same guard, on a Dataset given directly."""
        graph = FakeGraphReader(
            arcs={
                "dc/d/Contested": {
                    "typeOf": ("Dataset",),
                    "name": ("dc/d/Contested", "Real Dataset Name"),
                }
            }
        )

        doc = read_provenance_doc(graph, "dc/d/Contested")

        assert doc.name_of_source == "Real Dataset Name"


class TestUnresolvedNodeSentinel:
    """Synthetic fixtures exercise `UNRESOLVED_NODE`; none of 1,115 dev Provenances carries it."""

    def test_sentinel_only_description_is_filtered_and_becomes_a_gap(self):
        graph = FakeGraphReader(
            arcs={
                "provenance/Orphan": {
                    "typeOf": ("Provenance",),
                    "description": (UNRESOLVED_NODE,),
                }
            }
        )

        doc = read_provenance_doc(graph, "provenance/Orphan")

        assert doc.description is None
        assert "details.description" in doc.gaps

    def test_sentinel_only_license_type_is_filtered_and_becomes_a_gap(self):
        graph = FakeGraphReader(
            arcs={
                "provenance/Orphan": {
                    "typeOf": ("Provenance",),
                    "licenseType": (UNRESOLVED_NODE,),
                }
            }
        )

        doc = read_provenance_doc(graph, "provenance/Orphan")

        assert doc.license_type is None
        assert "license.type" in doc.gaps

    def test_sentinel_only_isPartOf_takes_the_orphan_path_and_reads_no_bogus_parent(self):
        """Unfiltered, `_first(("\\x00unresolved-node",))` is not `None`,
        so the reader would climb to a bogus, NUL-containing parent dcid
        and spend a `node_arcs_many` call on it. Filtered, the tuple is
        empty, `parent_dcid` is `None`, and the reader takes the true-orphan
        path instead: the identity read and the node's own batch read, no
        `node_subjects` call, and the given node's own `name` serves
        `name_of_source` directly.
        """
        graph = FakeGraphReader(
            arcs={
                "provenance/SentinelParent": {
                    "typeOf": ("Provenance",),
                    "isPartOf": (UNRESOLVED_NODE,),
                    "name": ("Sentinel Parent Provenance",),
                }
            }
        )

        doc = read_provenance_doc(graph, "provenance/SentinelParent")

        assert doc.name_of_source == "Sentinel Parent Provenance"
        assert [dcids for dcids, _ in graph.node_arcs_many_calls] == [
            ("provenance/SentinelParent",),
            ("provenance/SentinelParent",),
        ]
        assert graph.calls["node_subjects"] == 0


class TestRefusals:
    def test_a_dcid_the_graph_does_not_hold_raises_source_document_not_found(self):
        graph = FakeGraphReader(arcs={})

        with pytest.raises(SourceDocumentNotFound):
            read_provenance_doc(graph, "dc/base/Missing")

    def test_a_node_that_is_neither_dataset_nor_provenance_raises_not_a_source_node(self):
        graph = FakeGraphReader(arcs={"country/KEN": {"typeOf": ("Country",)}})

        with pytest.raises(NotASourceNode):
            read_provenance_doc(graph, "country/KEN")

    def test_a_source_dcid_given_directly_raises_not_a_source_node(self):
        """Publisher Sources fall outside the document reader's input types."""
        graph = FakeGraphReader(
            arcs={"dc/s/WorldBank": {"typeOf": ("Source",), "name": ("World Bank",)}}
        )

        with pytest.raises(NotASourceNode):
            read_provenance_doc(graph, "dc/s/WorldBank")

    def test_a_truncated_children_page_raises_too_many_child_provenances(self):
        graph = FakeGraphReader(
            arcs={"dc/d/Dataset1": {"typeOf": ("Dataset",)}},
            subjects={
                ("dc/d/Dataset1", "isPartOf"): SubjectPage(
                    dcids=("dc/base/ChildA",), truncated=True
                )
            },
        )

        with pytest.raises(TooManyChildProvenances):
            read_provenance_doc(graph, "dc/d/Dataset1")


class TestGaps:
    def test_every_field_present_leaves_only_the_unconditional_gaps(self):
        gaps = _compute_gaps(
            name_of_source="x",
            publisher="x",
            has_source=True,
            has_data_commons=True,
            description="x",
            source_url="x",
            source_data_url="x",
            citation="x",
            license_type="x",
            license_url="x",
            update_cadence="x",
            last_update="x",
            next_expected_update="x",
            years_covered="x",
            granularity="x",
            has_currencies=True,
            methodology="x",
            has_data_notes=True,
            has_transformation_log=True,
            has_version_history=True,
        )
        assert gaps == UNCONDITIONAL_GAPS

    def test_unconditional_gaps_are_the_members_with_no_source(self):
        assert UNCONDITIONAL_GAPS == (
            "scope_and_units.countries",
            "usage.api_access",
            "usage.python",
        )

    def test_every_field_absent_gaps_every_path_in_spec_order(self):
        gaps = _compute_gaps(
            name_of_source=None,
            publisher=None,
            has_source=False,
            has_data_commons=False,
            description=None,
            source_url=None,
            source_data_url=None,
            citation=None,
            license_type=None,
            license_url=None,
            update_cadence=None,
            last_update=None,
            next_expected_update=None,
            years_covered=None,
            granularity=None,
            has_currencies=False,
            methodology=None,
            has_data_notes=False,
            has_transformation_log=False,
            has_version_history=False,
        )
        assert gaps == (
            "details.name_of_source",
            "details.publisher",
            "details.source",
            "details.data_commons",
            "details.description",
            "details.source_url",
            "details.source_data_url",
            "details.citation",
            "license.type",
            "license.url",
            "license.update_cadence",
            "license.last_update",
            "license.next_expected_update",
            "scope_and_units.countries",
            "scope_and_units.years_covered",
            "scope_and_units.granularity",
            "scope_and_units.currencies",
            "methodology.description",
            "data_notes",
            "usage.api_access",
            "usage.python",
            "transformation_log",
            "version_history",
        )


def _escape(markdown: str) -> str:
    """The stored form of a Markdown document: `pipelines`'
    `escape_markdown` without its outer quotes, which the graph drops.
    """
    return json.dumps(markdown, ensure_ascii=False)[1:-1].replace('\\"', "\\u0022")


_TRANSFORMATIONS_MD = """## Download the source data

Read the "CRS" parquet file,
  then keep `flow_type = 1`.

## Aggregate totals

Sum by provider and recipient.
"""

_CHANGELOG_MD = """## 2026-09-22 Exclude donor core contributions

### Changed

- Donor core contributions (`bi_multi = 2`) are
  excluded from bilateral flows.

### Fixed

- Recipient codes map to ISO 3166.

## 2026-08-20 First release

### Added

- Dataset added to the graph.
"""

_METHODOLOGY_MD = """The OECD collects activity-level reports
from each DAC member.

A second paragraph.
"""

_NOTES_MD = """- Figures for the latest year are preliminary.
- Private flows are reported by a subset of providers only.
"""

_DOCUMENT_ARCS: dict[str, tuple[str, ...]] = {
    "dataTransformationDescription": (_escape(_TRANSFORMATIONS_MD),),
    "dataTransformationChangelog": (_escape(_CHANGELOG_MD),),
    "sourceMethodologyDescription": (_escape(_METHODOLOGY_MD),),
    "dataNotes": (_escape(_NOTES_MD),),
}


def _orphan(**arcs: tuple[str, ...]) -> FakeGraphReader:
    return FakeGraphReader(arcs={"provenance/Orphan": {"typeOf": ("Provenance",), **arcs}})


def _multi_child_dataset(children: dict[str, dict[str, tuple[str, ...]]]) -> FakeGraphReader:
    return FakeGraphReader(
        arcs={"dc/d/Dataset1": {"typeOf": ("Dataset",)}, **children},
        subjects={
            ("dc/d/Dataset1", "isPartOf"): SubjectPage(dcids=tuple(children), truncated=False)
        },
    )


class TestDocuments:
    def test_orphan_serves_all_four_documents_decoded_and_parsed(self):
        doc = read_provenance_doc(_orphan(**_DOCUMENT_ARCS), "provenance/Orphan")

        assert doc.transformation_log == (
            TransformationStep(
                title="Download the source data",
                description='Read the "CRS" parquet file, then keep `flow_type = 1`.',
            ),
            TransformationStep(
                title="Aggregate totals", description="Sum by provider and recipient."
            ),
        )
        assert doc.version_history == (
            ChangelogRelease(
                date=date(2026, 9, 22),
                title="Exclude donor core contributions",
                changes=(
                    ChangelogChange(
                        kind=ChangeKind.CHANGED,
                        note=(
                            "Donor core contributions (`bi_multi = 2`) are excluded "
                            "from bilateral flows."
                        ),
                    ),
                    ChangelogChange(kind=ChangeKind.FIXED, note="Recipient codes map to ISO 3166."),
                ),
            ),
            ChangelogRelease(
                date=date(2026, 8, 20),
                title="First release",
                changes=(
                    ChangelogChange(kind=ChangeKind.ADDED, note="Dataset added to the graph."),
                ),
            ),
        )
        assert doc.methodology == (
            "The OECD collects activity-level reports from each DAC member.\n\nA second paragraph."
        )
        assert doc.data_notes == (
            "Figures for the latest year are preliminary.",
            "Private flows are reported by a subset of providers only.",
        )
        assert doc.document_errors == ()
        for path in (
            "methodology.description",
            "data_notes",
            "transformation_log",
            "version_history",
        ):
            assert path not in doc.gaps

    def test_single_child_dataset_requests_and_serves_the_documents(self):
        graph = FakeGraphReader(
            arcs={"dc/d/Dataset1": {"typeOf": ("Dataset",)}, "dc/base/Child1": _DOCUMENT_ARCS},
            subjects={
                ("dc/d/Dataset1", "isPartOf"): SubjectPage(
                    dcids=("dc/base/Child1",), truncated=False
                )
            },
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert graph.node_arcs_many_calls[-1] == (
            ("dc/base/Child1",),
            (*_ROLLUP_PROPERTIES, *_SOLE_CHILD_PROPERTIES),
        )
        assert len(doc.transformation_log) == 2
        assert len(doc.version_history) == 2
        assert doc.methodology is not None
        assert len(doc.data_notes) == 2

    def test_multi_child_dataset_never_requests_the_documents(self):
        """Documents come from a sole child only, so a Dataset with several
        children leaves them empty and gapped without an error, and the
        reader never asks the graph for them.
        """
        graph = _multi_child_dataset(
            {"dc/base/ChildA": dict(_DOCUMENT_ARCS), "dc/base/ChildB": dict(_DOCUMENT_ARCS)}
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        requested = {prop for _, properties in graph.node_arcs_many_calls for prop in properties}
        assert requested.isdisjoint(DOCUMENT_PROPERTIES)
        assert doc.transformation_log == ()
        assert doc.version_history == ()
        assert doc.methodology is None
        assert doc.data_notes == ()
        assert doc.document_errors == ()
        for path in (
            "methodology.description",
            "data_notes",
            "transformation_log",
            "version_history",
        ):
            assert path in doc.gaps

    def test_multi_child_dataset_reads_only_the_publisher_name_and_roll_up_fields(self):
        """A Dataset with several children requests the Source's `name` and
        each child's roll-up fields, and no `description`, `url` or document
        that only a sole child would supply.
        """
        child = {
            "name": ("Child",),
            "description": ("A child import.",),
            "url": ("https://example.org/child",),
            "licenseType": ("PublicDomain",),
            **_DOCUMENT_ARCS,
        }
        graph = FakeGraphReader(
            arcs={
                "dc/d/Dataset1": {"typeOf": ("Dataset",), "isPartOf": ("dc/s/Publisher",)},
                "dc/s/Publisher": {"name": ("The Publisher",)},
                "dc/base/ChildA": dict(child),
                "dc/base/ChildB": dict(child),
            },
            subjects={
                ("dc/d/Dataset1", "isPartOf"): SubjectPage(
                    dcids=("dc/base/ChildA", "dc/base/ChildB"), truncated=False
                )
            },
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert graph.node_arcs_many_calls == [
            (("dc/d/Dataset1",), _NODE_PROPERTIES),
            (
                ("dc/s/Publisher", "dc/base/ChildA", "dc/base/ChildB"),
                (*_PUBLISHER_PROPERTIES, *_ROLLUP_PROPERTIES),
            ),
            (("PublicDomain",), LICENCE_PROPERTIES),
        ]
        assert doc.publisher == "The Publisher"
        assert doc.license_type == "PublicDomain"
        assert doc.description is None
        assert doc.source_url is None

    def test_undecodable_document_is_empty_gapped_and_reported(self, caplog):
        # A raw quote ends an MCF value, so a stored value never holds one.
        graph = _orphan(dataTransformationDescription=('## Step\\n\\nSays "hi".',))

        with caplog.at_level(logging.WARNING, logger="materialise.provenance_doc"):
            doc = read_provenance_doc(graph, "provenance/Orphan")

        assert doc.transformation_log == ()
        assert "transformation_log" in doc.gaps
        assert doc.document_errors == (
            DocumentIssue(
                member="transformation_log", reason=DocumentErrorReason.UNDECODABLE, line=None
            ),
        )
        [record] = caplog.records
        assert record.levelno == logging.WARNING
        message = record.getMessage()
        assert "provenance/Orphan" in message
        assert "transformation_log" in message
        assert "undecodable" in message

    def test_template_violation_reports_its_line_and_serves_the_rest(self):
        """The live CRS changelog has bare-date headings. The changelog is
        refused with its line while the transformations still serve.
        """
        changelog = "## 2026-09-22\n\n### Changed\n\n- A change.\n"
        graph = _orphan(
            dataTransformationDescription=_DOCUMENT_ARCS["dataTransformationDescription"],
            dataTransformationChangelog=(_escape(changelog),),
        )

        doc = read_provenance_doc(graph, "provenance/Orphan")

        assert len(doc.transformation_log) == 2
        assert doc.version_history == ()
        assert "version_history" in doc.gaps
        assert "transformation_log" not in doc.gaps
        assert doc.document_errors == (
            DocumentIssue(
                member="version_history",
                reason=DocumentErrorReason.MISSING_RELEASE_TITLE,
                line=1,
            ),
        )

    def test_two_distinct_values_on_one_node_is_conflicting_values(self):
        graph = _orphan(dataNotes=(_escape("- One.\n"), _escape("- Two.\n")))

        doc = read_provenance_doc(graph, "provenance/Orphan")

        assert doc.data_notes == ()
        assert "data_notes" in doc.gaps
        assert doc.document_errors == (
            DocumentIssue(member="data_notes", reason=CONFLICTING_VALUES, line=None),
        )

    def test_identical_repeated_values_collapse_to_one(self):
        issues: list[DocumentIssue] = []
        value = _escape("- One.\n")

        notes = _read_document(
            {"dataNotes": (value, value)}, "dataNotes", parse_notes, dcid="x", issues=issues
        )

        assert notes == ("One.",)
        assert issues == []

    # `\u002d\u0020` spells "- ", so every stored character is part of a
    # six-character escape and the value decodes to exactly
    # `MAX_DOCUMENT_BYTES` bytes, the largest document the grammar accepts.
    _AT_CEILING = "\\u002d\\u0020" + "\\u0041" * (MAX_DOCUMENT_BYTES - 2)

    @pytest.fixture
    def decode_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        """Records the length of every value the reader hands the decoder."""
        calls: list[int] = []

        def spy(value: str) -> str:
            calls.append(len(value))
            return decode_mcf_text(value)

        monkeypatch.setattr(provenance_doc_module, "decode_mcf_text", spy)
        return calls

    def test_a_stored_value_at_the_escaped_ceiling_is_decoded_and_served(
        self, decode_calls: list[int]
    ):
        assert len(self._AT_CEILING) == 6 * MAX_DOCUMENT_BYTES
        issues: list[DocumentIssue] = []

        notes = _read_document(
            {"dataNotes": (self._AT_CEILING,)}, "dataNotes", parse_notes, dcid="x", issues=issues
        )

        assert decode_calls == [6 * MAX_DOCUMENT_BYTES]
        assert notes == ("A" * (MAX_DOCUMENT_BYTES - 2),)
        assert issues == []

    def test_a_stored_value_past_the_escaped_ceiling_is_refused_before_decoding(
        self, decode_calls: list[int], caplog: pytest.LogCaptureFixture
    ):
        graph = _orphan(dataNotes=(self._AT_CEILING + "A",))

        with caplog.at_level(logging.WARNING, logger="materialise.provenance_doc"):
            doc = read_provenance_doc(graph, "provenance/Orphan")

        assert decode_calls == []
        assert doc.data_notes == ()
        assert "data_notes" in doc.gaps
        assert doc.document_errors == (
            DocumentIssue(
                member="data_notes", reason=DocumentErrorReason.DOCUMENT_TOO_LARGE, line=None
            ),
        )
        [record] = caplog.records
        assert "document_too_large" in record.getMessage()

    def test_every_document_error_is_reported_in_spec_order(self):
        graph = _orphan(
            dataTransformationDescription=(_escape("Text before a heading.\n"),),
            dataTransformationChangelog=(_escape("## 2026-09-22\n"),),
            sourceMethodologyDescription=(_escape("## A heading\n"),),
            dataNotes=(_escape("Not a list.\n"),),
        )

        doc = read_provenance_doc(graph, "provenance/Orphan")

        assert [issue.member for issue in doc.document_errors] == [
            "methodology.description",
            "data_notes",
            "transformation_log",
            "version_history",
        ]


class TestPlainTextFacts:
    def test_citation_is_served_as_stored_without_decoding(self):
        # A literal backslash-n in plain text is the publisher's own text.
        graph = _orphan(licenseAttribution=("OECD (2026), CRS \\n edition",))

        doc = read_provenance_doc(graph, "provenance/Orphan")

        assert doc.citation == "OECD (2026), CRS \\n edition"
        assert "details.citation" not in doc.gaps

    def test_citation_rolls_up_when_every_child_agrees(self):
        facts = {"licenseAttribution": ("Cite me.",)}
        graph = _multi_child_dataset({"dc/base/ChildA": facts, "dc/base/ChildB": facts})

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.citation == "Cite me."

    def test_citation_is_empty_when_children_disagree(self):
        graph = _multi_child_dataset(
            {
                "dc/base/ChildA": {"licenseAttribution": ("Cite A.",)},
                "dc/base/ChildB": {"licenseAttribution": ("Cite B.",)},
            }
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.citation is None
        assert "details.citation" in doc.gaps

    def test_source_data_url_is_served_from_a_sole_child(self):
        graph = _orphan(sourceDataUrl=("https://example.org/data.parquet",))

        doc = read_provenance_doc(graph, "provenance/Orphan")

        assert doc.source_data_url == "https://example.org/data.parquet"
        assert "details.source_data_url" not in doc.gaps

    def test_source_data_url_is_empty_when_a_sole_child_carries_two_values(self):
        graph = _orphan(sourceDataUrl=("https://example.org/a.csv", "https://example.org/b.csv"))

        doc = read_provenance_doc(graph, "provenance/Orphan")

        assert doc.source_data_url is None
        assert "details.source_data_url" in doc.gaps

    def test_source_data_url_rolls_up_when_every_child_agrees(self):
        facts = {"sourceDataUrl": ("https://example.org/data.csv",)}
        graph = _multi_child_dataset({"dc/base/ChildA": facts, "dc/base/ChildB": facts})

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.source_data_url == "https://example.org/data.csv"
        assert "details.source_data_url" not in doc.gaps

    def test_source_data_url_is_empty_when_children_disagree(self):
        graph = _multi_child_dataset(
            {
                "dc/base/ChildA": {"sourceDataUrl": ("https://example.org/a.csv",)},
                "dc/base/ChildB": {"sourceDataUrl": ("https://example.org/b.csv",)},
            }
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.source_data_url is None
        assert "details.source_data_url" in doc.gaps

    def test_license_url_rolls_up_when_every_child_agrees(self):
        facts = {"license": ("https://example.org/terms",)}
        graph = _multi_child_dataset({"dc/base/ChildA": facts, "dc/base/ChildB": facts})

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.license_url == "https://example.org/terms"
        assert "license.url" not in doc.gaps

    @pytest.mark.parametrize(
        "child_b",
        [{"license": ("https://example.org/other-terms",)}, {}],
        ids=["disagreeing", "silent"],
    )
    def test_license_url_is_empty_when_a_child_disagrees_or_is_silent(
        self, child_b: dict[str, tuple[str, ...]]
    ):
        graph = _multi_child_dataset(
            {
                "dc/base/ChildA": {"license": ("https://example.org/terms",)},
                "dc/base/ChildB": child_b,
            }
        )

        doc = read_provenance_doc(graph, "dc/d/Dataset1")

        assert doc.license_url is None
        assert "license.url" in doc.gaps

    def test_next_expected_update_reads_next_data_refresh_date_only(self):
        graph = _orphan(nextSourceReleaseDate=("2027-01-01",))

        doc = read_provenance_doc(graph, "provenance/Orphan")

        assert doc.next_expected_update is None
        requested = {prop for _, properties in graph.node_arcs_many_calls for prop in properties}
        assert "nextSourceReleaseDate" not in requested


class TestLiveFixtures:
    def test_world_development_indicators_dataset_with_its_one_real_child(self):
        dataset_arcs = _wire_arcs(
            "dataset_world_bank_wdi.json",
            "dc/d/WorldBank_WorldDevelopmentIndicators",
            _NODE_PROPERTIES,
        )
        assert dataset_arcs["isPartOf"] == ("dc/s/WorldBank",)
        assert "source" not in dataset_arcs or dataset_arcs.get("source") == ()

        publisher_arcs = _wire_arcs("source_world_bank_name.json", "dc/s/WorldBank", ("name",))
        child_arcs = _wire_arcs(
            "node_world_development_indicators.json",
            "dc/base/WorldDevelopmentIndicators",
            (*_ROLLUP_PROPERTIES, *_SOLE_CHILD_PROPERTIES),
        )

        graph = FakeGraphReader(
            arcs={
                "dc/d/WorldBank_WorldDevelopmentIndicators": dataset_arcs,
                "dc/s/WorldBank": publisher_arcs,
                "dc/base/WorldDevelopmentIndicators": child_arcs,
            },
            subjects={
                ("dc/d/WorldBank_WorldDevelopmentIndicators", "isPartOf"): SubjectPage(
                    dcids=("dc/base/WorldDevelopmentIndicators",), truncated=False
                )
            },
        )

        doc = read_provenance_doc(graph, "dc/d/WorldBank_WorldDevelopmentIndicators")

        assert doc.name_of_source == "World Development Indicators"
        assert doc.publisher == "World Bank"
        assert doc.source_url == "https://datacatalog.worldbank.org/search/dataset/003771"
        assert doc.description == (
            "The primary World Bank collection of development indicators "
            "encompassing global economic and social data."
        )
        # The field keeps its raw enum dcid.
        assert doc.license_type == "CreativeCommonsAttributionLicense"
        assert doc.update_cadence == "P1M"
        # The recorded `sourceDataUrl` is one country's http download, served as stored.
        assert (
            doc.source_data_url == "http://api.worldbank.org/v2/en/country/AFG?downloadformat=csv"
        )
        assert doc.license_url == "https://datacatalog.worldbank.org/search/dataset/0037712"
        assert doc.last_update is None
        assert doc.next_expected_update is None
        assert "license.last_update" in doc.gaps
        assert "license.next_expected_update" in doc.gaps

    def test_oecd_crs_orphan_provenance_has_no_isPartOf_and_no_publisher_lookup_batched(self):
        arcs = _wire_arcs(
            "node_oecd_crs_provenance.json",
            "provenance/OECDCreditorReportingSystem",
            (*_NODE_PROPERTIES, *_ROLLUP_PROPERTIES, *_SOLE_CHILD_PROPERTIES),
        )
        assert arcs["isPartOf"] == ()

        graph = FakeGraphReader(arcs={"provenance/OECDCreditorReportingSystem": arcs})

        doc = read_provenance_doc(graph, "provenance/OECDCreditorReportingSystem")

        assert doc.name_of_source == "OECD Creditor Reporting System"
        assert doc.description is None
        assert "details.description" in doc.gaps
        assert doc.update_cadence == "P1Y"
        assert doc.license_url == "https://www.oecd.org/en/about/terms-conditions.html"
        assert doc.years_covered == "1973-2024"
        assert "scope_and_units.years_covered" not in doc.gaps

    def test_us_census_acs_dataset_rolls_up_61_real_children_with_disagreeing_descriptions(self):
        """The live Dataset has 61 children; two children carry sampled `description` and
        `licenseType` values, while the other 59 are silent."""
        dataset_arcs = _wire_arcs(
            "dataset_us_census_acs.json",
            "dc/d/UsCensusBureau_AmericanCommunitySurveyAcs",
            (*_DATASET_PROPERTIES, "typeOf"),
        )
        child_dcids = _wire_subject_dcids(
            "dataset_us_census_acs_children.json",
            "dc/d/UsCensusBureau_AmericanCommunitySurveyAcs",
            "isPartOf",
        )
        assert len(child_dcids) == 61

        arcs: dict[str, dict[str, tuple[str, ...]]] = {
            "dc/d/UsCensusBureau_AmericanCommunitySurveyAcs": dataset_arcs,
            "dc/s/UsCensusBureau": _wire_arcs(
                "source_us_census_bureau_name.json", "dc/s/UsCensusBureau", ("name",)
            ),
            # Two sampled children have different descriptions and matching licences.
            "dc/base/CensusACS1YearSurvey": {
                "description": (
                    "1-Year estimates from the American Community Survey covering "
                    "socio-economic, housing, and demographic data.",
                ),
                "licenseType": ("PublicDomain",),
            },
            "dc/base/CensusACS5YearSurvey": {
                "description": (
                    "5-Year estimates from the American Community Survey providing "
                    "detailed socio-economic, demographic, and housing data for "
                    "small geographic areas.",
                ),
                "licenseType": ("PublicDomain",),
            },
        }
        graph = FakeGraphReader(
            arcs=arcs,
            subjects={
                ("dc/d/UsCensusBureau_AmericanCommunitySurveyAcs", "isPartOf"): SubjectPage(
                    dcids=child_dcids, truncated=False
                )
            },
        )

        doc = read_provenance_doc(graph, "dc/d/UsCensusBureau_AmericanCommunitySurveyAcs")

        assert doc.name_of_source == "American Community Survey (ACS)"
        assert doc.publisher == "U.S. Census Bureau"
        # More than one child, descriptions disagree (and no honest
        # rollup exists regardless) -- always empty for a multi-child
        # Dataset.
        assert doc.description is None
        # Only 2 of 61 children declare licences; the other 59 leave the field empty.
        assert doc.license_type is None
        assert "license.type" in doc.gaps


class _DeadlineRecordingGraphReader(FakeGraphReader):
    """Records deadlines passed to `node_arcs_many` and `node_subjects`."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.deadlines_seen: list[float | None] = []

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        self.deadlines_seen.append(deadline)
        return super().node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )

    def node_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        self.deadlines_seen.append(deadline)
        return super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results, deadline=deadline
        )


class TestDeadlineForwarding:
    def test_deadline_forwarded_to_every_read_the_document_makes(self):
        # Same climb-to-parent-and-roll-up-across-siblings shape as
        # TestProvenanceWithParentDataset's own first test, which exercises
        # all five reads: read #1 (the given node), read #2 (the parent
        # Dataset), read #3 (node_subjects for the sibling children), read #4
        # (the combined publisher+children batch), and read #5 (the licence
        # node's name).
        graph = _DeadlineRecordingGraphReader(
            arcs={
                "dc/base/GivenChild": {
                    "typeOf": ("Provenance",),
                    "isPartOf": ("dc/d/Dataset1",),
                },
                "dc/d/Dataset1": {
                    "typeOf": ("Dataset",),
                    "name": ("Dataset One",),
                    "isPartOf": ("dc/s/Publisher1",),
                },
                "dc/s/Publisher1": {"name": ("Publisher One",)},
                "dc/base/GivenChild2": {"licenseType": ("PublicDomain",)},
                "dc/base/SiblingChild": {"licenseType": ("PublicDomain",)},
            },
            subjects={
                ("dc/d/Dataset1", "isPartOf"): SubjectPage(
                    dcids=("dc/base/GivenChild2", "dc/base/SiblingChild"), truncated=False
                )
            },
        )
        deadline = 12345.0

        read_provenance_doc(graph, "dc/base/GivenChild", deadline=deadline)

        assert graph.calls["node_arcs_many"] == 4
        assert graph.calls["node_subjects"] == 1
        assert graph.deadlines_seen == [deadline, deadline, deadline, deadline, deadline]

    def test_deadline_none_by_default_behaves_exactly_as_before(self):
        # "dc/d/Contested" is a Dataset given directly, with no isPartOf
        # (no publisher batch target) and no registered subjects (no
        # children) -- two reads only: read #1 (the node itself) and
        # read #3 (node_subjects for its, here empty, Provenance children).
        graph = _DeadlineRecordingGraphReader(
            arcs={"dc/d/Contested": {"typeOf": ("Dataset",), "name": ("Real Name",)}}
        )

        doc = read_provenance_doc(graph, "dc/d/Contested")

        assert doc.name_of_source == "Real Name"
        assert graph.deadlines_seen == [None, None]


_WHO = "dc/base/WHO_GHO"
_WHO_NAMING_ARCS = {
    _WHO: {"typeOf": ("Provenance",), "name": ("WHO_GHO",), "isPartOf": ("dc/d/WHO_GHO",)},
    "dc/d/WHO_GHO": {"typeOf": ("Dataset",), "name": ("WHO Global Health Observatory",)},
}


@pytest.mark.parametrize(
    ("dcid", "arcs", "name", "reads"),
    [
        pytest.param("dc/base/Listed", {}, "Listed dataset", 0, id="listed"),
        # A base provenance is named for its import, so its parent Dataset names it, from the
        # name the mixer answers beside the `isPartOf` target.
        pytest.param(
            _WHO, _WHO_NAMING_ARCS, "WHO Global Health Observatory", 1, id="unlisted-with-parent"
        ),
        # A Dataset's `isPartOf` is its Source, so it is named by its own name.
        pytest.param(
            "dc/d/WHO_GHO",
            {
                **_WHO_NAMING_ARCS,
                "dc/d/WHO_GHO": {**_WHO_NAMING_ARCS["dc/d/WHO_GHO"], "isPartOf": ("dc/s/WHO",)},
                "dc/s/WHO": {"typeOf": ("Source",), "name": ("World Health Organization",)},
            },
            "WHO Global Health Observatory",
            1,
            id="dataset-typed",
        ),
        # A parent named only by its own dcid names nothing, so the node's own name stands.
        pytest.param(
            _WHO,
            {
                **_WHO_NAMING_ARCS,
                "dc/d/WHO_GHO": {"typeOf": ("Dataset",), "name": ("dc/d/WHO_GHO",)},
            },
            "WHO_GHO",
            1,
            id="parent-self-named",
        ),
    ],
)
def test_provenance_names_reads_the_least_that_names_a_provenance_as_its_citation(
    dcid, arcs, name, reads
):
    """A listed provenance takes the catalogue's name with no read. An unlisted one is named
    by one read of its arcs and their targets' names and types, and the name equals its
    citation's wherever the citation reads the same nodes."""
    catalogue = restrictive_catalogue(
        provenances=[provenance_row("dc/base/Listed", BASE, name="Listed dataset")]
    )
    graph = FakeGraphReader(arcs=arcs)

    names, read_ok = provenance_names(graph, [dcid, dcid], catalogue=catalogue, deadline=None)

    assert (names, read_ok) == ({dcid: name}, True)
    assert (graph.calls["node_arc_targets_many"], graph.calls["node_labels_batch"]) == (reads, 0)
    if dcid in arcs and name != "WHO_GHO":
        cited = read_source_citations(FakeGraphReader(arcs=arcs), [dcid], catalogue=catalogue)
        assert cited[dcid].name == name


def test_provenance_names_keeps_the_catalogue_names_when_the_graph_read_fails():
    class _Failing(FakeGraphReader):
        def node_arc_targets_many(self, dcids, *, properties, deadline=None):
            raise GraphReadError("transport")

    catalogue = restrictive_catalogue(
        provenances=[provenance_row("dc/base/Listed", BASE, name="Listed dataset")]
    )

    names, read_ok = provenance_names(
        _Failing(), ["dc/base/Listed", _WHO], catalogue=catalogue, deadline=None
    )

    assert (names, read_ok) == ({"dc/base/Listed": "Listed dataset"}, False)
