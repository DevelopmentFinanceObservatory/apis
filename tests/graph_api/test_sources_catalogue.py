"""Tests for `GET /api/graph/sources`: the origin registry joined to the request's catalogue,
and the card facts the catalogue job computes against the documentation page."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from graph_api.doors import sources as sources_door
from graph_api.doors.sources import _to_wire, build_source_catalogue
from graph_api.source_models import DatasetEntry, SourceCatalogue, SourceDocumentation
from materialise.graph import SubjectPage
from materialise.provenance_doc import read_provenance_doc
from origin_catalogue import BASE, DFO, UN_SYSTEM, CardFacts, DatasetRef, load_registry
from origin_catalogue.job import build_provenance_table
from origin_catalogue.registry import parse_registry
from tests.origin_catalogue.fake_catalogue import (
    provenance_row,
    restrictive_catalogue,
    un_mirror,
)

from .conftest import FakeGraphNodeReader, make_client

_REGISTRY = parse_registry(
    b"""
[[base]]
source = "dc/s/WorldBank"
label = "World Bank"
exclusions = [
  { provenance = "dc/base/WorldBank_IDS", reason = "superseded_by_dfo" },
]

[[base]]
source = "dc/s/UnitedNations"
label = "United Nations"
exclusions = []

[[base]]
source = "dc/s/Eurostat"
label = "Eurostat"

[[un_system]]
provenance = "undata/p/SDG"
segment = "sdg"
label = "SDG Global Database"
"""
)

_CATALOGUE_ROWS = (
    provenance_row("provenance/OECDCreditorReportingSystem", DFO, source="dc/s/OECD"),
    provenance_row("dc/base/WorldDevelopmentIndicators", BASE, source="dc/s/WorldBank"),
    provenance_row("dc/base/WorldBank_Subnational", BASE, source="dc/s/WorldBank"),
    provenance_row("dc/base/UN_SDG", BASE, source="dc/s/UnitedNations"),
    provenance_row("undata/p/SDG", UN_SYSTEM, source="undata/p/SDG", segment="sdg"),
)


_NO_FACTS = dict.fromkeys(
    ("description", "license_type", "update_cadence", "years_covered", "source_url")
)


def _entries(catalogue: SourceCatalogue) -> list[tuple[str, str, list[str]]]:
    return [
        (
            entry.data_commons.id,
            entry.source.ref,
            [p.ref for dataset in entry.datasets for p in dataset.provenances],
        )
        for entry in catalogue.sources
    ]


class TestBuildSourceCatalogue:
    def test_dfo_sources_lead_then_registry_entries_in_order_with_their_provenances(self):
        catalogue = build_source_catalogue(
            registry=_REGISTRY, catalogue=restrictive_catalogue(provenances=_CATALOGUE_ROWS)
        )

        assert _entries(catalogue) == [
            ("dfo", "dc/s/OECD", ["provenance/OECDCreditorReportingSystem"]),
            (
                "base",
                "dc/s/WorldBank",
                ["dc/base/WorldDevelopmentIndicators", "dc/base/WorldBank_Subnational"],
            ),
            ("base", "dc/s/UnitedNations", ["dc/base/UN_SDG"]),
            ("base", "dc/s/Eurostat", []),
            ("un_system", "undata/p/SDG", ["undata/p/SDG"]),
        ]

    def test_entries_carry_labels_and_exclusions(self):
        catalogue = build_source_catalogue(
            registry=_REGISTRY, catalogue=restrictive_catalogue(provenances=_CATALOGUE_ROWS)
        )
        body = catalogue.model_dump(mode="json")

        world_bank, united_nations = body["sources"][1], body["sources"][2]
        assert world_bank["source"] == {"ref": "dc/s/WorldBank", "label": "World Bank"}
        assert world_bank["data_commons"] == {"id": "base", "label": "Data Commons by Google"}
        assert world_bank["datasets"][0] == {
            "ref": "dc/base/WorldDevelopmentIndicators",
            "label": "dc/base/WorldDevelopmentIndicators",
            "provenances": [
                {
                    "ref": "dc/base/WorldDevelopmentIndicators",
                    "label": "dc/base/WorldDevelopmentIndicators",
                }
            ],
            **_NO_FACTS,
        }
        assert world_bank["exclusions"] == [
            {"ref": "dc/base/WorldBank_IDS", "reason": "superseded_by_dfo"}
        ]
        assert united_nations["exclusions"] == []
        assert body["sources"][3]["exclusions"] == []

    def test_provenances_group_under_their_dataset_and_a_bare_one_forms_its_own_group(self):
        dataset = DatasetRef(ref="dc/d/WDI", label="World Development Indicators")
        catalogue = build_source_catalogue(
            registry=_REGISTRY,
            catalogue=restrictive_catalogue(
                provenances=[
                    provenance_row("dc/base/A", BASE, source="dc/s/WorldBank", dataset=dataset),
                    provenance_row("dc/base/Bare", BASE, source="dc/s/WorldBank"),
                    provenance_row("dc/base/B", BASE, source="dc/s/WorldBank", dataset=dataset),
                ]
            ),
        )

        world_bank = catalogue.model_dump(mode="json")["sources"][0]
        assert world_bank["datasets"] == [
            {
                "ref": "dc/d/WDI",
                "label": "World Development Indicators",
                "provenances": [
                    {"ref": "dc/base/A", "label": "dc/base/A"},
                    {"ref": "dc/base/B", "label": "dc/base/B"},
                ],
                **_NO_FACTS,
            },
            {
                "ref": "dc/base/Bare",
                "label": "dc/base/Bare",
                "provenances": [{"ref": "dc/base/Bare", "label": "dc/base/Bare"}],
                **_NO_FACTS,
            },
        ]

    def test_the_un_origin_off_leaves_out_its_agencies(self):
        catalogue = build_source_catalogue(
            registry=_REGISTRY,
            catalogue=restrictive_catalogue(provenances=_CATALOGUE_ROWS, un_system_enabled=False),
        )

        assert "un_system" not in {entry.data_commons.id for entry in catalogue.sources}


class TestRoute:
    def test_serves_the_packaged_registry_against_the_request_catalogue(self, serve_catalogue):
        serve_catalogue(restrictive_catalogue(provenances=_CATALOGUE_ROWS))
        client = make_client()

        with client:
            resp = client.get("/api/graph/sources")

        assert resp.status_code == 200, resp.text
        body = SourceCatalogue.model_validate(resp.json())
        registry = load_registry()
        assert len(body.sources) == 1 + len(registry.base) + len(registry.un_system)
        assert body.sources[0].data_commons.id == "dfo"

    def test_makes_no_graph_reads(self):
        graph = FakeGraphNodeReader()
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/sources")

        assert resp.status_code == 200, resp.text
        assert all(count == 0 for count in graph.calls.values())


class TestRouteCaching:
    @pytest.fixture(autouse=True)
    def _fresh_cache(self) -> Iterator[None]:
        sources_door._frozen_source_catalogue.cache_clear()
        yield
        sources_door._frozen_source_catalogue.cache_clear()

    @pytest.fixture
    def builds(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Record the catalogue build each source index is built from."""
        built: list[str] = []

        def recording(*, registry, catalogue):
            built.append(catalogue.build_id)
            return build_source_catalogue(registry=registry, catalogue=catalogue)

        monkeypatch.setattr(sources_door, "build_source_catalogue", recording)
        return built

    def test_serves_public_gzip_with_an_etag_and_answers_304(self, serve_catalogue):
        serve_catalogue(restrictive_catalogue(provenances=_CATALOGUE_ROWS, build_id="build-a"))

        with make_client() as client:
            first = client.get("/api/graph/sources", headers={"accept-encoding": "gzip"})
            again = client.get(
                "/api/graph/sources",
                headers={"accept-encoding": "gzip", "if-none-match": first.headers["etag"]},
            )

        assert first.status_code == 200, first.text
        assert first.headers["cache-control"] == "public, max-age=300"
        assert first.headers["content-encoding"] == "gzip"
        assert first.headers["etag"]
        assert again.status_code == 304

    def test_encodes_the_index_once_per_catalogue_build(self, serve_catalogue, builds):
        serve_catalogue(restrictive_catalogue(provenances=_CATALOGUE_ROWS, build_id="build-a"))
        with make_client() as client:
            first = client.get("/api/graph/sources")
            second = client.get("/api/graph/sources")

        serve_catalogue(restrictive_catalogue(provenances=_CATALOGUE_ROWS[:1], build_id="build-b"))
        with make_client() as client:
            third = client.get("/api/graph/sources")

        assert builds == ["build-a", "build-b"]
        assert first.headers["etag"] == second.headers["etag"] != third.headers["etag"]


_WB = "dc/s/WorldBank"
_MULTI = "dc/d/WorldBank_Multi"
_SOLE = "dc/d/WorldBank_Sole"
_PARTIAL = "dc/d/WorldBank_Partial"
_ORPHAN = "provenance/ONE_Orphan"
_WITHHELD = "dc/base/WorldBank_PartialUpstream"
_CC_BY = "CreativeCommonsAttributionLicense"


def _provenance(
    *, part_of: tuple[str, ...], licence: str, cadence: str, years: tuple[str, str], **extra
) -> dict[str, tuple[str, ...]]:
    return {
        "typeOf": ("Provenance",),
        "source": (_WB,),
        "isPartOf": part_of,
        "url": ("https://data.worldbank.org",),
        "licenseType": (licence,),
        "sourceReleaseFrequency": (cadence,),
        "earliestObservationDate": (years[0],),
        "latestObservationDate": (years[1],),
        **{prop: (value,) for prop, value in extra.items()},
    }


_CARD_GRAPH_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    _WB: {"typeOf": ("Source",), "name": ("World Bank",)},
    # The licence nodes the provenances name. The graph names `PublicDomain` by its dcid.
    _CC_BY: {"typeOf": ("LicenseTypeEnum",), "name": ("Creative Commons Attribution License",)},
    "PublicDomain": {"typeOf": ("LicenseTypeEnum",), "name": ("PublicDomain",)},
    _MULTI: {
        "typeOf": ("Dataset",),
        "name": ("Multi",),
        "url": ("https://wb.org/multi",),
        "isPartOf": (_WB,),
    },
    # Two children. The second names the Source first, so its Dataset is its first
    # parent typed Dataset.
    "dc/base/WorldBank_MultiA": _provenance(
        part_of=(_MULTI,), licence=_CC_BY, cadence="P1Y", years=("1990", "2000-12-31")
    ),
    "dc/base/WorldBank_MultiB": _provenance(
        part_of=(_WB, _MULTI), licence=_CC_BY, cadence="P1Y", years=("1995-01", "2010")
    ),
    _SOLE: {"typeOf": ("Dataset",), "name": ("Sole",), "isPartOf": (_WB,)},
    "dc/base/WorldBank_Sole": {
        **_provenance(
            part_of=(_SOLE,), licence="PublicDomain", cadence="P1M", years=("2001", "2002")
        ),
        "url": ("https://wb.org/sole",),
        "description": ("The sole child.",),
    },
    # One admitted child and one the registry withholds, which disagrees on everything.
    _PARTIAL: {"typeOf": ("Dataset",), "name": ("Partial",), "url": ("https://wb.org/partial",)},
    "dc/base/WorldBank_PartialKept": _provenance(
        part_of=(_PARTIAL,),
        licence=_CC_BY,
        cadence="P1Y",
        years=("2000", "2005"),
        description="The kept child.",
    ),
    _WITHHELD: _provenance(
        part_of=(_PARTIAL,), licence="PublicDomain", cadence="P1D", years=("1950", "2020")
    ),
    # A DFO provenance whose only parent the graph does not hold, so it is an orphan.
    _ORPHAN: _provenance(
        part_of=("dc/s/NotHeld",),
        licence=_CC_BY,
        cadence="P1Y",
        years=("1973", "2024"),
        description="DFO's own import.",
    ),
}

_CARD_REGISTRY = parse_registry(
    f"""
[[base]]
source = "{_WB}"
label = "World Bank"
exclusions = [{{ provenance = "{_WITHHELD}", reason = "superseded_by_dfo" }}]

[[un_system]]
provenance = "undata/p/WHO"
segment = "who"
label = "World Health Organization (WHO)"
""".encode()
)


class _TripleGraph:
    """The catalogue job's reads and the documentation reader's reads over one set of
    triples. A reverse read finds every node whose `property` names `dcid`."""

    def __init__(self, arcs: dict[str, dict[str, tuple[str, ...]]]) -> None:
        self.arcs = arcs

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        return {
            dcid: (
                {prop: self.arcs[dcid].get(prop, ()) for prop in properties}
                if dcid in self.arcs
                else None
            )
            for dcid in dcids
        }

    def node_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        found = tuple(
            sorted(
                node
                for node, arcs in self.arcs.items()
                if dcid in arcs.get(property, ())
                and all(value in arcs.get(key, ()) for key, value in (filters or {}).items())
            )
        )
        return SubjectPage(dcids=found[:max_results], truncated=len(found) > max_results)


def _card(entry: DatasetEntry) -> tuple[str | None, ...]:
    return (
        entry.description,
        entry.license_type,
        entry.update_cadence,
        entry.years_covered,
        entry.source_url,
    )


def _page(doc: SourceDocumentation) -> tuple[str | None, ...]:
    return (
        doc.details.description,
        doc.license.type,
        doc.license.update_cadence,
        doc.scope_and_units.years_covered,
        doc.details.source_url,
    )


class TestCardFacts:
    def test_each_card_equals_its_documentation_page_on_one_graph(self):
        graph = _TripleGraph(_CARD_GRAPH_ARCS)
        stage = build_provenance_table(
            registry=_CARD_REGISTRY, dfo_provenances=frozenset({_ORPHAN}), graph=graph
        )
        assert stage.refusals == ()
        catalogue = restrictive_catalogue(
            provenances=stage.rows.values(), dataset_facts=stage.dataset_facts
        )

        cards = {
            dataset.ref: dataset
            for entry in build_source_catalogue(
                registry=_CARD_REGISTRY, catalogue=catalogue
            ).sources
            for dataset in entry.datasets
        }

        assert set(cards) == {_MULTI, _SOLE, _PARTIAL, _ORPHAN}
        for ref, card in cards.items():
            page = _to_wire(read_provenance_doc(graph, ref, catalogue=catalogue))
            assert _card(card) == _page(page), ref
        assert _card(cards[_MULTI]) == (
            None,
            "Creative Commons Attribution License",
            "Annual",
            "1990-2010",
            "https://wb.org/multi",
        )
        assert _card(cards[_SOLE]) == (
            "The sole child.",
            "PublicDomain",
            "Monthly",
            "2001-2002",
            "https://wb.org/sole",
        )
        # The withheld child counts on neither side, so the kept one is the sole child.
        assert _card(cards[_PARTIAL]) == (
            "The kept child.",
            "Creative Commons Attribution License",
            "Annual",
            "2000-2005",
            "https://wb.org/partial",
        )
        assert _card(cards[_ORPHAN]) == (
            "DFO's own import.",
            "Creative Commons Attribution License",
            "Annual",
            "1973-2024",
            "https://data.worldbank.org",
        )

    def test_un_entries_carry_no_facts(self):
        catalogue = build_source_catalogue(
            registry=_REGISTRY, catalogue=restrictive_catalogue(provenances=_CATALOGUE_ROWS)
        )

        [un] = [entry for entry in catalogue.sources if entry.data_commons.id == "un_system"]
        assert all(value is None for dataset in un.datasets for value in _card(dataset))

    def test_facts_serve_the_stored_licence_label_and_the_label_of_the_raw_cadence(self):
        row = provenance_row("dc/base/WDI", BASE, source="dc/s/WorldBank")
        facts = CardFacts(
            license_type="Creative Commons Attribution License",
            update_cadence="P1Y",
            years_covered="1960",
        )
        catalogue = restrictive_catalogue(provenances=[row], dataset_facts={row.id: facts})

        [dataset] = (
            build_source_catalogue(registry=_REGISTRY, catalogue=catalogue).sources[0].datasets
        )

        assert _card(dataset) == (
            None,
            "Creative Commons Attribution License",
            "Annual",
            "1960",
            None,
        )


_SUPERSESSION_REGISTRY = parse_registry(
    b"""
[[base]]
source = "dc/s/UnitedNations"
label = "United Nations"
exclusions = [
  { provenance = "dc/base/UN_SDG", reason = "superseded_by_un_system", successor = "undata/p/SDG", variable_prefix = "sdg/", code_delimiter = ".", pair_delimiter = "__", value_delimiter = "--" },
  { provenance = "dc/base/UN_Other", reason = "superseded_by_dfo" },
]

[[un_system]]
provenance = "undata/p/SDG"
segment = "sdg"
label = "SDG Global Database"
exclusions = [
  { series = "undata/sdg/SERIES-DC_ODA_TOTL", reason = "superseded_by_dfo", dfo_provenance = "provenance/OECDDAC1" },
  { series = "undata/sdg/SERIES-DC_ODA_TOTG", reason = "superseded_by_dfo", dfo_provenance = "provenance/Unlisted" },
]
"""
)


def _supersession_sources(catalogue) -> dict[str, dict]:
    """The wire entries of `_SUPERSESSION_REGISTRY` against `catalogue`, by source ref."""
    body = build_source_catalogue(registry=_SUPERSESSION_REGISTRY, catalogue=catalogue)
    return {entry["source"]["ref"]: entry for entry in body.model_dump(mode="json")["sources"]}


class TestSupersession:
    def test_a_base_source_lists_its_exclusions_as_ref_and_reason_alone(self):
        sources = _supersession_sources(restrictive_catalogue())

        assert sources["dc/s/UnitedNations"]["exclusions"] == [
            {"ref": "dc/base/UN_SDG", "reason": "superseded_by_un_system"},
            {"ref": "dc/base/UN_Other", "reason": "superseded_by_dfo"},
        ]

    def test_a_un_agency_lists_no_excluded_series(self):
        # The registry excludes two of the agency's series, and the build names both.
        catalogue = restrictive_catalogue(
            un_mirror=un_mirror(
                {},
                excluded={
                    "undata/sdg/TOTL": "undata/sdg/SERIES-DC_ODA_TOTL",
                    "undata/sdg/TOTG": "undata/sdg/SERIES-DC_ODA_TOTG",
                },
            )
        )

        assert _supersession_sources(catalogue)["undata/p/SDG"]["exclusions"] == []

    def test_an_excluded_provenance_is_not_listed_as_admitted(self):
        # A build older than the registry still holds the provenance the registry excludes.
        catalogue = restrictive_catalogue(
            provenances=[
                provenance_row("dc/base/UN_SDG", BASE, source="dc/s/UnitedNations"),
                provenance_row("dc/base/UN_Kept", BASE, source="dc/s/UnitedNations"),
            ]
        )

        sources = _supersession_sources(catalogue)

        assert [d["ref"] for d in sources["dc/s/UnitedNations"]["datasets"]] == ["dc/base/UN_Kept"]

    def test_the_packaged_registry_lists_the_sdg_supersession_in_devs_shape(self):
        body = build_source_catalogue(
            registry=load_registry(), catalogue=restrictive_catalogue()
        ).model_dump(mode="json")

        exclusions = [exclusion for source in body["sources"] for exclusion in source["exclusions"]]
        assert {"ref": "dc/base/UN_SDG", "reason": "superseded_by_un_system"} in exclusions
        assert all(set(exclusion) == {"ref", "reason"} for exclusion in exclusions)
