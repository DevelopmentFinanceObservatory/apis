"""The mint's record of how complete a version's rows are, judged per source
(`ManifestFacts.incomplete_after`, `datasets_api.manifest.incomplete_after`), read from the
`partialObservationDate` the origin catalogue carries for each source and stored in its own
column, which the version's manifest document carries (`version_document`).

The rig mints aid to Malawi from nine donors, two of which report 2025
(`tests.datasets_api.test_elected_year`). Its source takes the arcs the dev graph's Provenance
nodes hold (`tests/fixtures/dev_provenance_nodes.json`): DAC2A marks 2025 partial, DAC1 marks
nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from types import SimpleNamespace
from typing import cast

from assembler.mint.provenance import _partial_period
from assembler.ports import FrameRef
from assembler.testing import build_resolved_frame
from dataset_store.snapshot import ManifestFacts
from datasets_api.manifest import incomplete_after, version_document
from datasets_api.manifest_models import Manifest
from materialise import MaterialisationResult
from materialise.window import RequestedWindow
from origin_catalogue import Catalogue, bound_catalogue
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, dev_provenance_nodes, documenting

from ._description_doubles import FAMILY_MEMBERS, GROUPBY_QUERY, datasets_client, mint_family
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_elected_year import _AID, _MALAWI, _MWI, _PROVENANCE, _Rig

_ALL = RequestedWindow(kind="all")
_DAC2A = "provenance/OECDDAC2A"
_DAC1 = "provenance/OECDDAC1"


def _source_as(node: str, *, marked: bool = True) -> FakeCatalogue:
    """A catalogue documenting the rig's source with the arcs the dev graph holds for `node`,
    less its `partialObservationDate` when not `marked`, as the prod graph holds them."""
    arcs = dict(dev_provenance_nodes()[node])
    if not marked:
        arcs.pop("partialObservationDate", None)
    return documenting({_PROVENANCE: arcs})


def _document(rig: _Rig, minted: Manifest) -> dict[str, object]:
    stored = rig.deps.snapshot_reader.read_version(rig.stored.spec_id, minted.content_version)
    assert stored is not None and isinstance(stored.facts, ManifestFacts)
    return version_document(stored)


def _recorded(rig: _Rig, minted: Manifest) -> object:
    """The version's stored record, `"absent"` when its document carries none. Its
    `coverage` and its wire manifest never hold one."""
    document = _document(rig, minted)
    assert "incomplete_after" not in cast("dict[str, object]", document["coverage"])
    assert "incomplete_after" not in minted.model_dump(mode="json")["coverage"]
    return document.get("incomplete_after", "absent")


def _flagged(rig: _Rig, minted: Manifest) -> bool:
    """`partial_period` of a card drawing a 2025 row of the minted version."""
    frame = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_x", content_version="cv_x"),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": f"dcid:{_AID}",
                "recipient": _MWI,
                "time_period": "2025",
                "value": 1.0,
                "unit": None,
                "provenance": _PROVENANCE,
            }
        ],
        manifest=_document(rig, minted),
    )
    return _partial_period(resolved_frame=frame, rows=None)


_PARTIAL_2025 = [
    {
        "provenance": _PROVENANCE,
        "stat_vars": [f"dcid:{_AID}"],
        "source": _PROVENANCE,
        "status": "partial",
        "partial_period": "2025",
    }
]


def test_a_frame_reaching_its_source_s_partial_year_records_it(
    version_store: FakeVersionRecordDatasetStore,
) -> None:
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    with bound_catalogue(_source_as(_DAC2A)):
        minted, _ = rig.mint(_ALL)

    assert _recorded(rig, minted) == _PARTIAL_2025
    assert _flagged(rig, minted)


def test_a_frame_ending_before_its_source_s_partial_year_records_none(
    version_store: FakeVersionRecordDatasetStore,
) -> None:
    """A latest window elects 2024, since 2025 holds two of nine donors."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    with bound_catalogue(_source_as(_DAC2A)):
        minted, _ = rig.mint()

    assert minted.coverage.window is not None and minted.coverage.window["end_year"] == 2024
    assert _recorded(rig, minted) == []


def test_the_prod_graph_s_unmarked_sources_record_an_empty_list(
    version_store: FakeVersionRecordDatasetStore,
) -> None:
    """Prod states no `partialObservationDate` and serves no 2025 rows: the version is
    judged, and nothing is partial."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    with bound_catalogue(_source_as(_DAC2A, marked=False)):
        minted, _ = rig.mint()

    assert _recorded(rig, minted) == []
    assert not _flagged(rig, minted)


def test_a_dac1_frame_reaching_2025_is_not_flagged_and_keeps_2025(
    version_store: FakeVersionRecordDatasetStore,
) -> None:
    """The pipeline marks no DAC1 year partial, so DAC1's newest year, preliminary but not
    partial, stands as every other year does."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    with bound_catalogue(_source_as(_DAC1)):
        minted, _ = rig.mint(_ALL)

    assert minted.coverage.window is not None and minted.coverage.window["end_year"] == 2025
    assert _recorded(rig, minted) == []
    assert not _flagged(rig, minted)


def test_a_catalogue_that_does_not_document_the_source_records_nothing(
    version_store: FakeVersionRecordDatasetStore,
) -> None:
    """A stale catalogue, or one missing the source's row, cannot say whether the source is
    marked, so the version's completeness is unknown rather than complete."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    with bound_catalogue(FakeCatalogue()):
        minted, _ = rig.mint(_ALL)

    assert _recorded(rig, minted) == "absent"
    assert not _flagged(rig, minted)


def _result(latest_by_source: dict[tuple[str | None, str], str]) -> MaterialisationResult:
    """A mint result whose series reach the periods `latest_by_source` gives, for asking the
    record alone."""
    return cast(
        "MaterialisationResult",
        SimpleNamespace(
            trace=SimpleNamespace(components=()),
            series_statistics=SimpleNamespace(
                figures=SimpleNamespace(latest_by_source=latest_by_source)
            ),
        ),
    )


_CRS = "provenance/OECDCreditorReportingSystem"
_UN = "provenance/UNSDG"


def test_each_source_is_judged_on_its_own() -> None:
    """A version mixing CRS rows reaching CRS's partial 2025 with a UN series and a row with
    no provenance, which no catalogue documents, records CRS's partial cells and lists the
    other two as unknown, and a card drawing the CRS row reads it partial."""
    result = _result({(_CRS, "dcid:X"): "2025", (_UN, "dcid:Y"): "2025", (None, "dcid:Z"): "2025"})

    with bound_catalogue(documenting({_CRS: dev_provenance_nodes()[_CRS]})):
        recorded = incomplete_after(result)

    assert recorded == [
        {
            "provenance": None,
            "stat_vars": ["dcid:Z"],
            "source": None,
            "status": "unknown",
            "partial_period": None,
        },
        {
            "provenance": _CRS,
            "stat_vars": ["dcid:X"],
            "source": _CRS,
            "status": "partial",
            "partial_period": "2025",
        },
        {
            "provenance": _UN,
            "stat_vars": ["dcid:Y"],
            "source": _UN,
            "status": "unknown",
            "partial_period": None,
        },
    ]

    def drawn(provenance: str, stat_var: str) -> bool:
        frame = build_resolved_frame(
            frame_ref=FrameRef(spec_id="spec_x", content_version="cv_x"),
            role_keys=("recipient",),
            rows=[
                {
                    "stat_var": stat_var,
                    "recipient": _MWI,
                    "time_period": "2025",
                    "value": 1.0,
                    "unit": None,
                    "provenance": provenance,
                }
            ],
            manifest={"incomplete_after": recorded},
        )
        return _partial_period(resolved_frame=frame, rows=None)

    assert drawn(_CRS, "dcid:X")
    assert not drawn(_UN, "dcid:Y")
    with bound_catalogue(FakeCatalogue()):
        assert incomplete_after(result) is None, "a catalogue judging no source records nothing"


def test_a_mark_the_frame_grammar_cannot_read_records_its_rows_unknown() -> None:
    """The CRS node states a quarter, which no frame period spells, so nothing says whether
    its 2025 rows reach it; DAC1's unmarked rows keep the record judged."""
    nodes = dev_provenance_nodes()
    catalogue = documenting(
        {_CRS: {**nodes[_CRS], "partialObservationDate": ("2025-Q3",)}, _DAC1: nodes[_DAC1]}
    )
    result = _result({(_CRS, "dcid:X"): "2025", (_DAC1, "dcid:Y"): "2025"})

    with bound_catalogue(catalogue):
        recorded = incomplete_after(result)

    assert recorded == [
        {
            "provenance": _CRS,
            "stat_vars": ["dcid:X"],
            "source": _CRS,
            "status": "unknown",
            "partial_period": None,
        }
    ]


def test_a_mint_with_no_catalogue_bound_records_nothing(
    bind_catalogue: Callable[[Catalogue], None],
) -> None:
    """A mint elsewhere needs a catalogue for admission, so this asks the record alone, over
    a result whose rows reach a partial year."""
    result = _result({(_PROVENANCE, f"dcid:{_AID}"): "2025"})
    with bound_catalogue(_source_as(_DAC2A)):
        assert incomplete_after(result) == _PARTIAL_2025
    bind_catalogue(cast("Catalogue", None))

    assert incomplete_after(result) is None


def _stored_record(rig: _Rig) -> object:
    with rig.version_store.transaction() as tx:
        row = tx.fetch_one("SELECT incomplete_after FROM datasets.version_record")
    assert row is not None
    return row["incomplete_after"]


def test_a_later_mint_of_the_same_version_replaces_its_partial_years(
    version_store: FakeVersionRecordDatasetStore,
) -> None:
    """The full release completes 2025 while the frame's rows stay the same, so the mint
    after the head's day-long lifetime mints the same version again, and the store takes that
    mint's judgement. A mint whose catalogue cannot judge it leaves the stored one."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})
    with bound_catalogue(_source_as(_DAC2A)):
        first, _ = rig.mint(_ALL)
    assert _stored_record(rig) == _PARTIAL_2025

    rig.clock.at += timedelta(hours=25)
    with bound_catalogue(FakeCatalogue()):
        unjudged, _ = rig.mint(_ALL)
    assert unjudged.trace["served_from"] == "materialise"
    assert _stored_record(rig) == _PARTIAL_2025

    rig.clock.at += timedelta(hours=25)
    with bound_catalogue(_source_as(_DAC2A, marked=False)):
        second, _ = rig.mint(_ALL)

    assert second.trace["served_from"] == "materialise"
    assert second.content_version == first.content_version
    assert _stored_record(rig) == []
    assert _recorded(rig, second) == []


def _marked(mark: str) -> dict[str, tuple[str, ...]]:
    """The dev graph's DAC2A arcs, marking `mark` partial."""
    return {**dev_provenance_nodes()[_DAC2A], "partialObservationDate": (mark,)}


def _stored_record_of(store: FakeVersionRecordDatasetStore, content_version: str) -> object:
    with store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT incomplete_after FROM datasets.version_record WHERE content_version = %s",
            (content_version,),
        )
    assert row is not None
    return row["incomplete_after"]


def _partial(stat_vars: list[str], mark: str) -> list[dict[str, object]]:
    return [
        {
            "provenance": "OECD_DAC2A",
            "stat_vars": stat_vars,
            "source": "OECD_DAC2A",
            "status": "partial",
            "partial_period": mark,
        }
    ]


def test_a_groupby_served_from_the_head_stores_the_judgement_of_its_base(
    serve_catalogue: Callable[[Catalogue], None],
) -> None:
    """A plain mint primes the head, so the groupby read that follows collapses the head
    (`current._record_head_hit_derived`), deriving its record from the base version's."""
    serve_catalogue(documenting({"OECD_DAC2A": _marked("1900")}))
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store, fixture="family")
    base = mint_family(client, options={"bundle": False})
    spec_id = base["definition"]["spec_id"]

    response = client.get(f"/api/datasets/{spec_id}?{GROUPBY_QUERY}")

    assert response.status_code == 200, response.text
    collapsed = response.json()
    assert collapsed["trace"]["served_from"] == "head"
    recorded = {
        content_version: _stored_record_of(store, content_version)
        for content_version in (base["content_version"], collapsed["content_version"])
    }
    assert recorded[base["content_version"]] == _partial(
        sorted(f"dcid:{member}" for member in FAMILY_MEMBERS), "1900"
    )
    assert recorded[collapsed["content_version"]] == _partial([f"dcid:{spec_id}"], "1900")
