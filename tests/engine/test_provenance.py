"""Test provenance source refs and `resolved_sources` in a definite response."""

from __future__ import annotations

from origin_catalogue import BASE, UN_SYSTEM, ProvenanceRow, SourceRef
from search.engine.clients.graph import Facet
from search.engine.region import _build_source_refs
from search.models import DataCommonsRef, ProvenanceRef
from search.models import SourceRef as WireSourceRef
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve
from tests.fixtures import FakeGraph
from tests.origin_catalogue.fake_catalogue import restrictive_catalogue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_graph(*, nodes: dict | None = None, obs: dict | None = None) -> FakeGraph:
    """Build a minimal graph for source-reference tests."""
    return FakeGraph(
        nodes=nodes or {},
        obs=obs or {},
        detect={},
        resolve={},
    )


# ---------------------------------------------------------------------------
# Unit tests: _build_source_refs
# ---------------------------------------------------------------------------


class TestBuildSourceRefs:
    def test_provenance_id_confirmed(self):
        """A facet with a provenanceId that resolves yields one GraphRef."""
        graph = _minimal_graph(
            nodes={"dc/base/HumanCRS": {"label": "Human CRS"}},
        )
        facets = [
            Facet(
                earliest_date="2010",
                latest_date="2020",
                obs_count=10,
                provenance_id="dc/base/HumanCRS",
            )
        ]
        refs = _build_source_refs(facets, graph)
        assert len(refs) == 1
        assert refs[0].ref == "dc/base/HumanCRS"
        assert refs[0].label == "Human CRS"

    def test_unresolvable_provenance_id_omitted(self):
        """A facet whose provenanceId does not resolve is omitted."""
        graph = _minimal_graph(nodes={})
        facets = [
            Facet(
                earliest_date="2010",
                latest_date="2020",
                obs_count=5,
                provenance_id="dc/base/UnknownImport",
            )
        ]
        refs = _build_source_refs(facets, graph)
        assert refs == []

    def test_mixed_facets_only_confirmed_returned(self):
        """Only confirmed sources appear; the unresolvable facet is omitted."""
        graph = _minimal_graph(
            nodes={"dc/base/HumanCRS": {"label": "Human CRS"}},
        )
        facets = [
            Facet(
                earliest_date="2010",
                latest_date="2020",
                obs_count=10,
                provenance_id="dc/base/HumanCRS",
            ),
            Facet(
                earliest_date="2010",
                latest_date="2020",
                obs_count=5,
                provenance_id="dc/base/UnknownImport",
            ),
        ]
        refs = _build_source_refs(facets, graph)
        assert len(refs) == 1
        assert refs[0].ref == "dc/base/HumanCRS"

    def test_deduplication_multiple_facets_same_source(self):
        """Multiple facets pointing to the same provenanceId yield exactly one ref."""
        graph = _minimal_graph(
            nodes={"dc/base/HumanCRS": {"label": "Human CRS"}},
        )
        facets = [
            Facet(
                earliest_date="2010",
                latest_date="2020",
                obs_count=10,
                provenance_id="dc/base/HumanCRS",
            ),
            Facet(
                earliest_date="2018",
                latest_date="2018",
                obs_count=1,
                provenance_id="dc/base/HumanCRS",
            ),
        ]
        refs = _build_source_refs(facets, graph)
        assert len(refs) == 1
        assert refs[0].ref == "dc/base/HumanCRS"

    def test_empty_facets_returns_empty(self):
        """No facets yields no refs."""
        graph = _minimal_graph()
        assert _build_source_refs([], graph) == []

    def test_facets_with_no_provenance_fields_returns_empty(self):
        """Facets without a provenanceId produce no refs."""
        graph = _minimal_graph()
        facets = [
            Facet(earliest_date="2010", latest_date="2020", obs_count=10),
            Facet(earliest_date="2015", latest_date="2020", obs_count=5),
        ]
        assert _build_source_refs(facets, graph) == []


# ---------------------------------------------------------------------------
# Integration test: resolved_sources populated end-to-end
# ---------------------------------------------------------------------------


class TestResolvedSourcesEndToEnd:
    def test_definite_response_has_resolved_sources(self):
        """A definite response includes the fixture's World Development Indicators source."""
        result = offline_resolve(
            make_request("life expectancy in Kenya"), pac=True, sdmx=FakeSdmxClient()
        )
        inner = result.parts[0]
        assert inner.status == "definite"
        sources = inner.spec.resolved.trace.resolved_sources
        assert len(sources) >= 1
        dcids = {s.ref for s in sources}
        assert "dc/base/WorldDevelopmentIndicators" in dcids
        labels = {s.ref: s.label for s in sources}
        assert labels["dc/base/WorldDevelopmentIndicators"] == "WorldDevelopmentIndicators"


# ---------------------------------------------------------------------------
# Catalogue rows: label, source and Data Commons
# ---------------------------------------------------------------------------

_HUMAN_CRS = ProvenanceRow(
    id="dc/base/HumanCRS",
    data_commons=BASE,
    source=SourceRef(ref="dc/s/OECD", label="OECD"),
    name="Human CRS dataset",
    url="https://example.org/crs",
)
_UN_WHO = ProvenanceRow(
    id="undata/p/WHO",
    data_commons=UN_SYSTEM,
    source=SourceRef(ref="undata/p/WHO", label="World Health Organization"),
    name="World Health Organization",
    url="https://example.org/who",
    segment="who",
)


def _facet(provenance_id: str) -> Facet:
    return Facet(
        earliest_date="2010", latest_date="2020", obs_count=10, provenance_id=provenance_id
    )


class TestCatalogueRows:
    def test_listed_provenance_takes_the_row_and_reads_no_label(self, bind_catalogue):
        """A listed provenance carries the row's name, source and Data Commons; only the
        unlisted one is read from the graph."""
        bind_catalogue(restrictive_catalogue(provenances=[_HUMAN_CRS]))
        graph = _LabelSpyGraph(
            nodes={
                "dc/base/HumanCRS": {"label": "Human CRS"},
                "dc/base/Other": {"label": "Other import"},
            }
        )

        refs = _build_source_refs([_facet("dc/base/HumanCRS"), _facet("dc/base/Other")], graph)

        assert refs == [
            ProvenanceRef(
                ref="dc/base/HumanCRS",
                label="Human CRS dataset",
                source=WireSourceRef(ref="dc/s/OECD", label="OECD"),
                data_commons=DataCommonsRef(id="base", label="Data Commons by Google"),
            ),
            ProvenanceRef(ref="dc/base/Other", label="Other import"),
        ]
        assert [ref.model_dump(mode="json")["documentation"] for ref in refs] == [
            "/api/graph/documentation/dc/base/HumanCRS",
            "/api/graph/documentation/dc/base/Other",
        ]
        assert graph.label_reads == [["dc/base/Other"]]

    def test_un_provenance_is_its_own_source(self, bind_catalogue):
        bind_catalogue(restrictive_catalogue(provenances=[_UN_WHO]))
        graph = _LabelSpyGraph(nodes={})

        (ref,) = _build_source_refs([_facet("undata/p/WHO")], graph)

        assert ref.source is not None
        assert ref.source.ref == ref.ref
        assert ref.data_commons == DataCommonsRef(id="un_system", label="UN System Data Commons")
        assert graph.label_reads == []


class _LabelSpyGraph(FakeGraph):
    """Record every `node_labels_batch` call."""

    def __init__(self, *, nodes: dict) -> None:
        super().__init__(nodes=nodes, obs={}, detect={}, resolve={})
        self.label_reads: list[list[str]] = []

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        self.label_reads.append(list(dcids))
        return super().node_labels_batch(dcids)
