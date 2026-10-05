"""Tests the single upstream source a set of variables comes through."""

from __future__ import annotations

import dataclasses

from origin_catalogue import BASE, DFO, UN_SYSTEM, ProvenanceRow, SourceRef
from search.engine.retrieval.derive import upstream_source
from search.engine.shape import UpstreamSource
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue

_WHO_BASE = "dc/base/WHO_GHO"
_WHO_BASE_ALT = "dc/base/WHO_Mortality"
_UNICEF_BASE = "dc/base/UNICEF"
_WHO_UN = "undata/p/WHO"
_DFO = "provenance/OECDCreditorReportingSystem"


def _row(provenance_id, origin, *, source, label, segment=None) -> ProvenanceRow:
    row = provenance_row(provenance_id, origin, source=source, segment=segment)
    return dataclasses.replace(row, source=SourceRef(ref=source, label=label))


def _catalogue():
    return restrictive_catalogue(
        provenances=[
            _row(_WHO_BASE, BASE, source="dc/s/WorldHealthOrganization", label="WHO"),
            _row(_WHO_BASE_ALT, BASE, source="dc/s/WorldHealthOrganization", label="WHO"),
            _row(_UNICEF_BASE, BASE, source="dc/s/UNICEF", label="UNICEF"),
            _row(_WHO_UN, UN_SYSTEM, source=_WHO_UN, label="WHO", segment="who"),
            _row(_DFO, DFO, source=_DFO, label="OECD"),
        ],
        dfo_variables=["Dfo_Flow", "Who_Reimported"],
        base_variables={
            "Who_Reimported": ([_WHO_BASE], []),
            "Who_A": ([_WHO_BASE], []),
            "Who_B": ([_WHO_BASE, _WHO_BASE_ALT], []),
            "Unicef_A": ([_UNICEF_BASE], []),
            "Shared": ([_WHO_BASE, _UNICEF_BASE], []),
        },
    )


def _who(ref: str, origin) -> UpstreamSource:
    return UpstreamSource(source=SourceRef(ref=ref, label="WHO"), data_commons=origin)


class TestUpstreamSource:
    def test_base_shape_with_one_source_names_it(self, bind_catalogue):
        bind_catalogue(_catalogue())
        assert upstream_source(["Who_A", "Who_B"]) == _who("dc/s/WorldHealthOrganization", BASE)

    def test_un_shape_names_its_agency(self, bind_catalogue):
        bind_catalogue(_catalogue())
        assert upstream_source(["undata/who/U5MR"]) == _who(_WHO_UN, UN_SYSTEM)

    def test_shape_with_two_sources_has_none(self, bind_catalogue):
        bind_catalogue(_catalogue())
        assert upstream_source(["Who_A", "Unicef_A"]) is None
        assert upstream_source(["Shared"]) is None

    def test_same_publisher_on_two_origins_has_none(self, bind_catalogue):
        bind_catalogue(_catalogue())
        assert upstream_source(["Who_A", "undata/who/U5MR"]) is None

    def test_a_member_without_a_known_source_gives_none(self, bind_catalogue):
        bind_catalogue(_catalogue())
        assert upstream_source(["Who_A", "Unlisted"]) is None

    def test_dfo_shape_has_none(self, bind_catalogue):
        bind_catalogue(_catalogue())
        assert upstream_source(["Dfo_Flow"]) is None

    def test_a_base_variable_dfo_also_holds_has_none(self, bind_catalogue):
        bind_catalogue(_catalogue())
        assert upstream_source(["Who_Reimported"]) is None
        assert upstream_source(["Who_A", "Who_Reimported"]) is None

    def test_permissive_fake_catalogue_knows_no_sources(self):
        assert upstream_source(["Who_A", "undata/who/U5MR"]) is None
