"""Test provenance source refs and `resolved_sources` in a definite response."""

from __future__ import annotations

import pytest

from code_index.composites import DECLARED_COMPOSITES
from code_index.declarations import CRS, CRS_INCLUSIVE
from origin_catalogue import BASE, DFO, UN_SYSTEM, ProvenanceRow, SourceRef
from search.engine.region.refs import provenance_labels
from search.models import DataCommonsRef, ProvenanceRef
from search.models import SourceRef as WireSourceRef
from search.observations import composite_facet_ref, provenance_refs
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve
from tests.fixtures import FakeGraph
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue

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


def _source_refs(dcids: list[str], graph: FakeGraph) -> list[ProvenanceRef]:
    """The refs a part's sources carry, built as grounding builds them."""
    return provenance_refs(dcids, provenance_labels(dcids, graph, deadline=None))


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

        refs = _source_refs(["dc/base/HumanCRS", "dc/base/Other"], graph)

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
        assert graph.node_reads == [["dc/base/Other"]]
        assert graph.label_reads == []

    def test_un_provenance_is_its_own_source(self, bind_catalogue):
        bind_catalogue(restrictive_catalogue(provenances=[_UN_WHO]))
        graph = _LabelSpyGraph(nodes={})

        (ref,) = _source_refs(["undata/p/WHO"], graph)

        assert ref.source is not None
        assert ref.source.ref == ref.ref
        assert ref.data_commons == DataCommonsRef(id="un_system", label="UN System Data Commons")
        assert graph.label_reads == []


_IMPUTED = "provenance/ONEImputedMultilateralSectors"


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param(
            [
                provenance_row(CRS, DFO, source="dc/s/OECD"),
                provenance_row(_IMPUTED, DFO, source="dc/s/OECD"),
            ],
            ("dfo", "dc/s/OECD"),
            id="agree",
        ),
        pytest.param(
            [
                provenance_row(CRS, DFO, source="dc/s/OECD"),
                provenance_row(_IMPUTED, BASE, source="dc/s/ONE"),
            ],
            (None, None),
            id="disagree",
        ),
        pytest.param(
            [
                provenance_row(CRS, DFO, source="dc/s/OECD"),
                provenance_row(_IMPUTED, BASE, source="dc/s/OECD"),
            ],
            (None, "dc/s/OECD"),
            id="one-publisher-two-origins",
        ),
        pytest.param([provenance_row(CRS, DFO, source="dc/s/OECD")], (None, None), id="unlisted"),
    ],
)
def test_a_composite_facet_keeps_what_its_component_tables_share(
    bind_catalogue, rows: list[ProvenanceRow], expected: tuple[str | None, str | None]
):
    """A composite's provenance facet keeps the Data Commons and source only where every
    component table's catalogue row names the same one, and carries no document route."""
    bind_catalogue(restrictive_catalogue(provenances=rows))
    composite = DECLARED_COMPOSITES[CRS_INCLUSIVE]

    ref = composite_facet_ref(composite)

    assert (ref.ref, ref.label, ref.documentation) == (CRS_INCLUSIVE, composite.label, None)
    assert (
        ref.data_commons.id if ref.data_commons else None,
        ref.source.ref if ref.source else None,
    ) == expected


class _LabelSpyGraph(FakeGraph):
    """Record every `node_labels_batch` and `node_arc_targets_many` call."""

    def __init__(self, *, nodes: dict) -> None:
        super().__init__(nodes=nodes, obs={}, detect={}, resolve={})
        self.label_reads: list[list[str]] = []
        self.node_reads: list[list[str]] = []

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        self.label_reads.append(list(dcids))
        return super().node_labels_batch(dcids)

    def node_arc_targets_many(self, dcids, *, properties, deadline=None):
        self.node_reads.append(list(dcids))
        return super().node_arc_targets_many(dcids, properties=properties, deadline=deadline)
