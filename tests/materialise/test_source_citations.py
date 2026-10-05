"""Tests source citation naming from the origin catalogue and the graph, URL and licence
fallback, and graph read counts."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from dc_graph import GraphCapExceededError, GraphDeadlineExceededError
from materialise import provenance_doc
from materialise.graph import GraphReadError, parse_arcs
from materialise.provenance_doc import (
    _CITATION_LICENCE_PROPERTIES,
    _CITATION_LINKED_PROPERTIES,
    _CITATION_NODE_PROPERTIES,
    _NODE_PROPERTIES,
    NotASourceNode,
    SourceCitation,
    SourceDocumentNotFound,
    TooManyChildProvenances,
    read_provenance_doc,
    read_source_citations,
)
from origin_catalogue import BASE, DFO, SourceRef
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue

from .conftest import FakeGraphReader

# A catalogue listing no provenance, so every name is read from the graph.
_UNLISTED = restrictive_catalogue()

_GRAPH_API_FIXTURES_DIR = Path(__file__).parent.parent / "graph_api" / "fixtures"

_WDI_PROVENANCE = "dc/base/WorldDevelopmentIndicators"
_WDI_DATASET = "dc/d/WorldBank_WorldDevelopmentIndicators"
_OECD_CRS = "provenance/OECDCreditorReportingSystem"

_CC_BY = "CreativeCommonsAttributionLicense"
_OECD_TERMS = "OECDTermsAndConditions"
# The `licenseType` nodes the cited provenances name, with the names the graph holds.
_LICENCE_NODES: dict[str, dict[str, tuple[str, ...]]] = {
    _CC_BY: {"name": ("Creative Commons Attribution License",)},
    _OECD_TERMS: {"name": ("OECD Terms and Conditions",)},
}


def _wire_arcs(
    fixture_name: str, dcid: str, properties: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    body = json.loads((_GRAPH_API_FIXTURES_DIR / fixture_name).read_text())
    return parse_arcs(body["data"][dcid]["arcs"], properties, mark_unresolved=True)


def _wdi_graph() -> FakeGraphReader:
    return FakeGraphReader(
        arcs={
            _WDI_PROVENANCE: _wire_arcs(
                "node_world_development_indicators.json",
                _WDI_PROVENANCE,
                _CITATION_NODE_PROPERTIES,
            ),
            _WDI_DATASET: _wire_arcs(
                "dataset_world_bank_wdi.json", _WDI_DATASET, _CITATION_LINKED_PROPERTIES
            ),
            **_LICENCE_NODES,
        }
    )


class TestLiveFixtures:
    def test_an_unlisted_wdi_provenance_is_named_after_its_dataset(self):
        graph = _wdi_graph()

        citations = read_source_citations(graph, [_WDI_PROVENANCE], catalogue=_UNLISTED)

        assert citations == {
            _WDI_PROVENANCE: SourceCitation(
                dcid=_WDI_PROVENANCE,
                name="World Development Indicators",
                # The Dataset URL returns 404; the Provenance page is live.
                url="https://datatopics.worldbank.org/world-development-indicators/",
                licence="Creative Commons Attribution License",
                licence_type=_CC_BY,
                licence_url="https://datacatalog.worldbank.org/search/dataset/0037712",
            )
        }
        assert graph.node_arcs_many_calls == [
            ((_WDI_PROVENANCE,), _CITATION_NODE_PROPERTIES),
            ((_WDI_DATASET, _CC_BY), _CITATION_LINKED_PROPERTIES),
        ]

    def test_oecd_crs_orphan_takes_its_own_name_and_url_in_one_read(self):
        graph = FakeGraphReader(
            arcs={
                _OECD_CRS: _wire_arcs(
                    "node_oecd_crs_provenance.json", _OECD_CRS, _CITATION_NODE_PROPERTIES
                )
            }
        )

        (citation,) = read_source_citations(graph, [_OECD_CRS], catalogue=_UNLISTED).values()

        assert citation.name == "OECD Creditor Reporting System"
        assert citation.url is not None
        assert citation.url.startswith("https://data-explorer.oecd.org/")
        assert citation.licence is None
        assert len(graph.node_arcs_many_calls) == 1


class TestNaming:
    def test_an_unlisted_provenance_takes_its_dataset_s_graph_name(self):
        graph = FakeGraphReader(
            arcs={
                "dc/base/SomeImport": {
                    "typeOf": ("Provenance",),
                    "name": ("SomeImport",),
                    "isPartOf": ("dc/d/SomePublisher_SomeDataset",),
                },
                "dc/d/SomePublisher_SomeDataset": {
                    "typeOf": ("Dataset",),
                    "name": ("Some Dataset",),
                },
            }
        )

        citations = read_source_citations(graph, ["dc/base/SomeImport"], catalogue=_UNLISTED)

        assert citations["dc/base/SomeImport"].name == "Some Dataset"

    def test_the_node_s_own_url_wins_over_its_dataset_s(self):
        graph = FakeGraphReader(
            arcs={
                "dc/base/SomeImport": {
                    "typeOf": ("Provenance",),
                    "url": ("https://example.org/import",),
                    "isPartOf": ("dc/d/SomePublisher_SomeDataset",),
                },
                "dc/d/SomePublisher_SomeDataset": {
                    "typeOf": ("Dataset",),
                    "url": ("https://example.org/dataset",),
                },
            }
        )

        citations = read_source_citations(graph, ["dc/base/SomeImport"], catalogue=_UNLISTED)

        assert citations["dc/base/SomeImport"].url == "https://example.org/import"

    def test_a_node_with_no_url_takes_its_dataset_s(self):
        graph = FakeGraphReader(
            arcs={
                "dc/base/SomeImport": {
                    "typeOf": ("Provenance",),
                    "isPartOf": ("dc/d/SomePublisher_SomeDataset",),
                },
                "dc/d/SomePublisher_SomeDataset": {
                    "typeOf": ("Dataset",),
                    "url": ("https://example.org/dataset",),
                },
            }
        )

        citations = read_source_citations(graph, ["dc/base/SomeImport"], catalogue=_UNLISTED)

        assert citations["dc/base/SomeImport"].url == "https://example.org/dataset"

    def test_a_parent_dataset_the_graph_does_not_hold_falls_back_to_the_node(self):
        graph = FakeGraphReader(
            arcs={
                "dc/base/SomeImport": {
                    "typeOf": ("Provenance",),
                    "name": ("Some Import",),
                    "url": ("https://example.org/import",),
                    "isPartOf": ("dc/d/Missing",),
                }
            }
        )

        citations = read_source_citations(graph, ["dc/base/SomeImport"], catalogue=_UNLISTED)

        assert citations["dc/base/SomeImport"] == SourceCitation(
            dcid="dc/base/SomeImport",
            name="Some Import",
            url="https://example.org/import",
            licence=None,
        )

    def test_a_source_parent_is_no_publication_and_the_node_keeps_its_own_name_and_url(self):
        graph = FakeGraphReader(
            arcs={
                "dc/base/SomeImport": {
                    "typeOf": ("Provenance",),
                    "name": ("Some Import",),
                    "isPartOf": ("dc/s/SomePublisher",),
                },
                "dc/s/SomePublisher": {
                    "typeOf": ("Source",),
                    "name": ("Some Publisher",),
                    "url": ("https://example.org/publisher",),
                },
            }
        )

        citations = read_source_citations(graph, ["dc/base/SomeImport"], catalogue=_UNLISTED)

        assert citations["dc/base/SomeImport"] == SourceCitation(
            dcid="dc/base/SomeImport", name="Some Import", url=None, licence=None
        )

    def test_the_first_parent_typed_dataset_is_the_publication(self):
        graph = FakeGraphReader(
            arcs={
                "dc/base/SomeImport": {
                    "typeOf": ("Provenance",),
                    "isPartOf": ("dc/s/SomePublisher", "dc/d/SomePublisher_SomeDataset"),
                },
                "dc/s/SomePublisher": {"typeOf": ("Source",), "name": ("Some Publisher",)},
                "dc/d/SomePublisher_SomeDataset": {
                    "typeOf": ("Dataset",),
                    "name": ("Some Dataset",),
                },
            }
        )

        citations = read_source_citations(graph, ["dc/base/SomeImport"], catalogue=_UNLISTED)

        assert citations["dc/base/SomeImport"].name == "Some Dataset"

    def test_a_dataset_cited_directly_does_not_climb_to_its_source(self):
        graph = FakeGraphReader(
            arcs={
                _WDI_DATASET: {
                    "typeOf": ("Dataset",),
                    "name": ("World Development Indicators",),
                    "isPartOf": ("dc/s/WorldBank",),
                }
            }
        )

        citations = read_source_citations(graph, [_WDI_DATASET], catalogue=_UNLISTED)

        assert citations[_WDI_DATASET].name == "World Development Indicators"
        assert len(graph.node_arcs_many_calls) == 1

    def test_a_name_equal_to_the_node_s_own_dcid_is_no_name(self):
        graph = FakeGraphReader(
            arcs={"dc/base/Bare": {"typeOf": ("Provenance",), "name": ("dc/base/Bare",)}}
        )

        citations = read_source_citations(graph, ["dc/base/Bare"], catalogue=_UNLISTED)

        assert citations["dc/base/Bare"].name is None


class TestLicence:
    def test_the_oecd_crs_licence_dcid_reads_as_its_graph_name(self):
        graph = FakeGraphReader(
            arcs={
                _OECD_CRS: {
                    "typeOf": ("Provenance",),
                    "licenseType": (_OECD_TERMS,),
                },
                **_LICENCE_NODES,
            }
        )

        citation = read_source_citations(graph, [_OECD_CRS], catalogue=_UNLISTED)[_OECD_CRS]

        assert citation.licence == "OECD Terms and Conditions"
        assert graph.node_arcs_many_calls == [
            ((_OECD_CRS,), _CITATION_NODE_PROPERTIES),
            ((_OECD_TERMS,), _CITATION_LINKED_PROPERTIES),
        ]

    def test_a_licence_node_the_graph_has_no_name_for_reads_as_its_dcid(self):
        graph = FakeGraphReader(
            arcs={"dc/base/X": {"typeOf": ("Provenance",), "licenseType": ("SomeNewLicense",)}}
        )

        assert (
            read_source_citations(graph, ["dc/base/X"], catalogue=_UNLISTED)["dc/base/X"].licence
            == "SomeNewLicense"
        )

    def test_two_licence_values_on_one_node_is_no_licence(self):
        graph = FakeGraphReader(
            arcs={
                "dc/base/X": {
                    "typeOf": ("Provenance",),
                    "licenseType": ("PublicDomain", "CreativeCommons0"),
                    "license": ("https://a.example/terms", "https://b.example/terms"),
                }
            }
        )

        citation = read_source_citations(graph, ["dc/base/X"], catalogue=_UNLISTED)["dc/base/X"]

        assert (citation.licence, citation.licence_type, citation.licence_url) == (None, None, None)


class TestReads:
    def test_a_dcid_the_graph_does_not_hold_has_no_entry(self):
        graph = FakeGraphReader(arcs={})

        assert read_source_citations(graph, ["dc/base/NotHeld"], catalogue=_UNLISTED) == {}
        assert len(graph.node_arcs_many_calls) == 1

    def test_no_dcids_makes_no_read(self):
        graph = FakeGraphReader(arcs={})

        assert read_source_citations(graph, [], catalogue=_UNLISTED) == {}
        assert graph.node_arcs_many_calls == []

    def test_many_refs_cost_two_reads_with_each_dataset_read_once(self):
        graph = FakeGraphReader(
            arcs={
                "dc/base/A": {"typeOf": ("Provenance",), "isPartOf": ("dc/d/Shared",)},
                "dc/base/B": {"typeOf": ("Provenance",), "isPartOf": ("dc/d/Shared",)},
                "dc/base/C": {"typeOf": ("Provenance",), "name": ("Orphan C",)},
                "dc/d/Shared": {"typeOf": ("Dataset",), "name": ("Shared Dataset",)},
            }
        )

        citations = read_source_citations(
            graph, ["dc/base/A", "dc/base/B", "dc/base/A", "dc/base/C"], catalogue=_UNLISTED
        )

        assert {dcid: citation.name for dcid, citation in citations.items()} == {
            "dc/base/A": "Shared Dataset",
            "dc/base/B": "Shared Dataset",
            "dc/base/C": "Orphan C",
        }
        assert graph.node_arcs_many_calls == [
            (("dc/base/A", "dc/base/B", "dc/base/C"), _CITATION_NODE_PROPERTIES),
            (("dc/d/Shared",), _CITATION_LINKED_PROPERTIES),
        ]

    def test_parent_datasets_and_licence_nodes_share_the_second_read(self):
        graph = FakeGraphReader(
            arcs={
                "dc/base/A": {
                    "typeOf": ("Provenance",),
                    "isPartOf": ("dc/d/Shared",),
                    "licenseType": (_CC_BY,),
                },
                "dc/base/B": {"typeOf": ("Provenance",), "licenseType": (_CC_BY,)},
                "dc/d/Shared": {"typeOf": ("Dataset",), "name": ("Shared Dataset",)},
                **_LICENCE_NODES,
            }
        )

        citations = read_source_citations(graph, ["dc/base/A", "dc/base/B"], catalogue=_UNLISTED)

        assert {dcid: citation.licence for dcid, citation in citations.items()} == {
            "dc/base/A": "Creative Commons Attribution License",
            "dc/base/B": "Creative Commons Attribution License",
        }
        assert graph.node_arcs_many_calls == [
            (("dc/base/A", "dc/base/B"), _CITATION_NODE_PROPERTIES),
            (("dc/d/Shared", _CC_BY), _CITATION_LINKED_PROPERTIES),
        ]


class TestCatalogued:
    def test_a_listed_provenance_takes_all_but_its_licence_from_its_row(self):
        row = provenance_row(_WDI_PROVENANCE, BASE, source="dc/s/WorldBank")
        graph = _wdi_graph()

        citations = read_source_citations(
            graph, [_WDI_PROVENANCE], catalogue=restrictive_catalogue(provenances=[row])
        )

        assert citations == {
            _WDI_PROVENANCE: SourceCitation(
                dcid=_WDI_PROVENANCE,
                name=row.name,
                url=row.url,
                licence="Creative Commons Attribution License",
                source=SourceRef(ref="dc/s/WorldBank", label="dc/s/WorldBank"),
                data_commons=BASE,
                licence_type=_CC_BY,
                licence_url="https://datacatalog.worldbank.org/search/dataset/0037712",
            )
        }
        assert graph.node_arcs_many_calls == [
            ((_WDI_PROVENANCE,), _CITATION_LICENCE_PROPERTIES),
            ((_CC_BY,), _CITATION_LINKED_PROPERTIES),
        ]

    def test_listed_and_unlisted_refs_share_one_node_read(self):
        graph = FakeGraphReader(
            arcs={
                _OECD_CRS: {"typeOf": ("Provenance",), "licenseType": (_OECD_TERMS,)},
                "dc/base/A": {"typeOf": ("Provenance",), "isPartOf": ("dc/d/Shared",)},
                "dc/d/Shared": {"typeOf": ("Dataset",), "name": ("Shared Dataset",)},
                **_LICENCE_NODES,
            }
        )
        catalogue = restrictive_catalogue(provenances=[provenance_row(_OECD_CRS, DFO)])

        citations = read_source_citations(graph, [_OECD_CRS, "dc/base/A"], catalogue=catalogue)

        assert citations[_OECD_CRS].data_commons == DFO
        assert citations[_OECD_CRS].licence == "OECD Terms and Conditions"
        assert citations["dc/base/A"].name == "Shared Dataset"
        assert citations["dc/base/A"].source is None
        assert citations["dc/base/A"].data_commons is None
        assert graph.node_arcs_many_calls == [
            ((_OECD_CRS, "dc/base/A"), _CITATION_NODE_PROPERTIES),
            (("dc/d/Shared", _OECD_TERMS), _CITATION_LINKED_PROPERTIES),
        ]

    def test_a_listed_provenance_the_graph_does_not_hold_is_still_cited(self):
        row = provenance_row("dc/base/NotHeld", BASE)
        graph = FakeGraphReader(arcs={})

        citations = read_source_citations(
            graph, ["dc/base/NotHeld"], catalogue=restrictive_catalogue(provenances=[row])
        )

        assert citations["dc/base/NotHeld"].name == row.name
        assert citations["dc/base/NotHeld"].licence is None


class _DocumentationFailingGraph(FakeGraphReader):
    """Serves citation reads and fails every documentation read."""

    def node_arcs_many(self, dcids, *, properties, deadline=None, **kwargs):
        if tuple(properties) == _NODE_PROPERTIES:
            raise GraphReadError("documentation read failed")
        return super().node_arcs_many(dcids, properties=properties, deadline=deadline, **kwargs)


_DOCUMENTED_ORPHAN = {
    _OECD_CRS: {
        "typeOf": ("Provenance",),
        "name": ("OECD Creditor Reporting System",),
        "licenseType": (_OECD_TERMS,),
        "license": ("https://www.oecd.org/en/about/terms-conditions.html",),
        "licenseAttribution": ("OECD (2026), Creditor Reporting System",),
        "sourceReleaseFrequency": ("P3M",),
    },
    **_LICENCE_NODES,
}


class TestDocumentation:
    def test_documented_citations_carry_the_provenance_document(self):
        graph = FakeGraphReader(arcs=_DOCUMENTED_ORPHAN)

        citation = read_source_citations(graph, [_OECD_CRS], catalogue=_UNLISTED, documented=True)[
            _OECD_CRS
        ]

        assert citation.documentation == read_provenance_doc(
            FakeGraphReader(arcs=_DOCUMENTED_ORPHAN), _OECD_CRS, catalogue=_UNLISTED
        )
        assert citation.documentation is not None
        assert citation.documentation.citation == "OECD (2026), Creditor Reporting System"
        assert citation.licence_type == _OECD_TERMS
        assert citation.licence_url == "https://www.oecd.org/en/about/terms-conditions.html"

    def test_undocumented_citations_make_no_documentation_read(self):
        graph = FakeGraphReader(arcs=_DOCUMENTED_ORPHAN)

        citation = read_source_citations(graph, [_OECD_CRS], catalogue=_UNLISTED)[_OECD_CRS]

        assert citation.documentation is None
        assert [properties for _dcids, properties in graph.node_arcs_many_calls] == [
            _CITATION_NODE_PROPERTIES,
            _CITATION_LINKED_PROPERTIES,
        ]

    def test_a_failed_documentation_read_keeps_the_citation_and_logs_a_warning(self, caplog):
        graph = _DocumentationFailingGraph(arcs=_DOCUMENTED_ORPHAN)

        with caplog.at_level(logging.WARNING, logger="materialise.provenance_doc"):
            citations = read_source_citations(
                graph, [_OECD_CRS], catalogue=_UNLISTED, documented=True
            )

        undocumented = read_source_citations(
            FakeGraphReader(arcs=_DOCUMENTED_ORPHAN), [_OECD_CRS], catalogue=_UNLISTED
        )
        assert citations == {
            _OECD_CRS: replace(undocumented[_OECD_CRS], documentation_read_failed=True)
        }
        assert citations[_OECD_CRS].licence == "OECD Terms and Conditions"
        assert [record.levelno for record in caplog.records] == [logging.WARNING]

    @pytest.mark.parametrize(
        ("raised", "failed"),
        [
            (SourceDocumentNotFound(_OECD_CRS), False),
            (NotASourceNode(_OECD_CRS, ("Organization",)), False),
            (TooManyChildProvenances("dc/d/Wide"), False),
            (GraphReadError("transport"), True),
            (GraphDeadlineExceededError("deadline"), True),
        ],
        ids=["not-found", "not-a-source", "too-many-children", "read-error", "deadline"],
    )
    def test_a_documentation_read_tells_nothing_to_document_from_a_failed_read(
        self, monkeypatch: pytest.MonkeyPatch, raised: Exception, failed: bool
    ):
        """The graph documenting nothing is a lasting answer; only a failed read can change on
        the next read, so only it is marked."""

        def read(*_args, **_kwargs):
            raise raised

        monkeypatch.setattr(provenance_doc, "read_provenance_doc", read)

        citation = read_source_citations(
            FakeGraphReader(arcs=_DOCUMENTED_ORPHAN),
            [_OECD_CRS],
            catalogue=_UNLISTED,
            documented=True,
        )[_OECD_CRS]

        assert citation.documentation is None
        assert citation.documentation_read_failed is failed
        assert citation.licence == "OECD Terms and Conditions"

    @pytest.mark.parametrize(
        "raised",
        [GraphCapExceededError("cap", cap_kind="bytes", cap=1, observed=2), TypeError("bug")],
        ids=["cap-exceeded", "programming-error"],
    )
    def test_a_documentation_read_lets_other_errors_propagate(
        self, monkeypatch: pytest.MonkeyPatch, raised: Exception
    ):
        def read(*_args, **_kwargs):
            raise raised

        monkeypatch.setattr(provenance_doc, "read_provenance_doc", read)

        with pytest.raises(type(raised)):
            read_source_citations(
                FakeGraphReader(arcs=_DOCUMENTED_ORPHAN),
                [_OECD_CRS],
                catalogue=_UNLISTED,
                documented=True,
            )
