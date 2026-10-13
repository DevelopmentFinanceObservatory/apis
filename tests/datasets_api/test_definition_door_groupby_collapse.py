"""Head-hit groupby renders must match fresh-mint content versions and bytes, preserve
collapsed warnings with their structured fields, and leave the subject head at the raw version.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast

import httpx
import pytest
from dfo_frame_grammar import frame_sha256

import datasets_api.current as current_module
import datasets_api.render as render_module
from artifact_store import LiveArtifactInventory, Tier
from code_index import CodeIndex
from datasets_api.config import DatasetsBuildSettings
from datasets_api.render import CANONICAL_FRAME_RENDERING
from datasets_api.version_store import rendering_key
from materialise.formats import (
    EXCEL_MAX_ROWS,
    FRAME1_MEDIA_TYPE,
    ExcelLimitError,
    check_excel_rows,
    read_canonical_frame,
    write_frame1,
)
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry.identity import recompute_identity
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._frame_doubles import link_frame1_only
from ._graph_doubles import FakeMintGraphReader
from ._legacy_conversion_trace import MIGRATED_CONVERSION
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door import _client
from .test_definition_door_currency import _FakeHeadStore, _RenderableArtifactStore
from .test_download_links import _RenewingArtifactStore
from .test_serve_groupby_collapse import (
    _CSV_BODY,
    _CSV_HEADER,
    _FACET_ID,
    _IDENTITY,
    _MEASUREMENT_METHOD,
    _MEMBER_A,
    _MEMBER_ARCS,
    _MEMBER_B,
    _PROVENANCE,
    _RECIPIENT,
    _UNIT,
    _code_index,
    _coverage_service,
    _graph,
    _mint,
    _sdmx_transport,
    _sole_member_sdmx_transport,
    _spec_id_and_sidecar,
    _spec_id_and_sidecar_sole_member,
)

_GROUPBY_OPTIONS = {"bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}}
_GROUPBY_QUERY = "bundle=false&groupby.by=recipient&groupby.op=sum"

# The request is satisfied natively because the observed unit is USD. No rate table is needed.
_CONVERTED_IDENTITY = {
    **_IDENTITY,
    "facets": {"unit": {"state": "value", "request": {"currency": "USD", "prices": "current"}}},
}

_CONVERTED_CSV_BODY = (
    _CSV_HEADER
    + f"dataflow,DC:DF_OBS(1.0.0),I,{_MEMBER_A},{_RECIPIENT},CurrencyUSD,GrantEquivalent,"
    f"NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
    + f"dataflow,DC:DF_OBS(1.0.0),I,{_MEMBER_B},{_RECIPIENT},CurrencyUSD,GrantEquivalent,"
    f"NotApplicable,OECD_DAC2A,2020,500,,facet1\r\n"
)


def _converted_graph() -> FakeMintGraphReader:
    return _graph_with_arcs(
        {
            _MEMBER_A: {**_MEMBER_ARCS, "testAxis": ("TestAxisA",)},
            _MEMBER_B: {**_MEMBER_ARCS, "testAxis": ("TestAxisB",)},
            "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
        }
    )


def _graph_with_arcs(arcs: dict) -> FakeMintGraphReader:
    return FakeMintGraphReader(arcs=arcs, labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"})


def _converted_sdmx_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=_CONVERTED_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )


def _converted_spec_id_and_sidecar() -> tuple[str, dict]:
    recomputed = recompute_identity(_CONVERTED_IDENTITY)
    return recomputed.spec_id, {
        "schema_version": 1,
        "spec_id": recomputed.spec_id,
        "identity": _CONVERTED_IDENTITY,
    }


def test_head_hit_mints_the_same_collapsed_content_version_and_bytes_as_the_fresh_path():
    version_store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        head_writer=_FakeHeadStore(),
        artifact_store=_RenderableArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        ),
        code_index=_code_index(),
        coverage=_coverage_service(),
    )
    spec_id, sidecar = _spec_id_and_sidecar()

    # Minting primes `datasets.subject_head` with the raw content version.
    fresh = _mint(client, spec_id=spec_id, sidecar=sidecar, options=_GROUPBY_OPTIONS)

    got = client.get(f"/api/datasets/{spec_id}?{_GROUPBY_QUERY}")
    assert got.status_code == 200, got.text
    served = got.json()

    assert served["trace"]["served_from"] == "head"
    assert served["content_version"] == fresh["content_version"]
    assert served["coverage"]["rows"] == fresh["coverage"]["rows"]
    assert served["artifacts"][0]["artifact_digest"] == fresh["artifacts"][0]["artifact_digest"]
    assert served["artifacts"][0]["bytes"] == fresh["artifacts"][0]["bytes"]

    # Neither path reports the sum as a warning: it is a step of the collapsed version's
    # description record.
    assert "stat_vars_summed" not in {w["code"] for w in fresh["warnings"]}
    assert "stat_vars_summed" not in {w["code"] for w in served["warnings"]}


def test_head_hit_with_a_converted_base_mints_the_same_collapsed_content_version():
    """A base materialisation whose unit axis carried an explicit currency-conversion request:
    the head-hit collapse mints the identical `content_version` the fresh path did, since both
    hash the same collapsed frame."""
    version_store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_converted_graph(),
        sdmx_transport=_converted_sdmx_transport(),
        version_store=version_store,
        head_writer=_FakeHeadStore(),
        artifact_store=_RenderableArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        ),
        code_index=_code_index(),
        coverage=_coverage_service(unit="CurrencyUSD"),
    )
    spec_id, sidecar = _converted_spec_id_and_sidecar()

    fresh = _mint(client, spec_id=spec_id, sidecar=sidecar, options=_GROUPBY_OPTIONS)

    got = client.get(f"/api/datasets/{spec_id}?{_GROUPBY_QUERY}")
    assert got.status_code == 200, got.text
    served = got.json()

    assert served["trace"]["served_from"] == "head"
    assert served["content_version"] == fresh["content_version"]
    assert served["coverage"]["rows"] == fresh["coverage"]["rows"]
    assert served["artifacts"][0]["artifact_digest"] == fresh["artifacts"][0]["artifact_digest"]
    assert served["artifacts"][0]["bytes"] == fresh["artifacts"][0]["bytes"]


def _collapse_client() -> tuple[
    httpx.Client, FakeVersionRecordDatasetStore, _RenderableArtifactStore
]:
    version_store = FakeVersionRecordDatasetStore()
    artifact_store = _RenderableArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        head_writer=_FakeHeadStore(),
        artifact_store=artifact_store,
        code_index=_code_index(),
        coverage=_coverage_service(),
    )
    return client, version_store, artifact_store


def test_a_groupby_head_hit_collapses_a_base_stored_as_a_frame1_document_alone():
    """A base version that links a frame1 document and no canonical frame, as a version minted
    before canonical frames does, collapses on a head hit to the version and bytes a fresh
    groupby mint gives."""
    spec_id, sidecar = _spec_id_and_sidecar()
    fresh_client, _, _ = _collapse_client()
    fresh = _mint(fresh_client, spec_id=spec_id, sidecar=sidecar, options=_GROUPBY_OPTIONS)
    client, version_store, artifact_store = _collapse_client()
    base = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    link_frame1_only(
        version_store, artifact_store, subject_id=spec_id, content_version=base["content_version"]
    )

    got = client.get(f"/api/datasets/{spec_id}?{_GROUPBY_QUERY}")

    assert got.status_code == 200, got.text
    served = got.json()
    assert served["trace"]["served_from"] == "head"
    assert served["content_version"] == fresh["content_version"] != base["content_version"]
    assert served["artifacts"][0]["artifact_digest"] == fresh["artifacts"][0]["artifact_digest"]


def _edit_stored_trace(
    version_store: FakeVersionRecordDatasetStore,
    *,
    subject_id: str,
    content_version: str,
    edit: Callable[[dict], None],
) -> None:
    """Applies `edit` to the version's stored `trace` in place."""
    with version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT trace FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            (subject_id, content_version),
        )
        assert row is not None
        trace = cast(dict, row["trace"])
        edit(trace)
        tx.execute(
            "UPDATE datasets.version_record SET trace = %s "
            "WHERE subject_id = %s AND content_version = %s",
            (trace, subject_id, content_version),
        )


def _groupby_head_hit_over_stored_conversion(conversion: object) -> httpx.Response:
    """Mints the raw version, replaces its stored `trace.conversion`, then requests a groupby
    rendering, which the head hit builds from the stored frame and trace."""
    client, version_store, _ = _collapse_client()
    spec_id, sidecar = _spec_id_and_sidecar()
    base = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    _edit_stored_trace(
        version_store,
        subject_id=spec_id,
        content_version=base["content_version"],
        edit=lambda trace: trace.update(conversion=conversion),
    )

    return client.get(f"/api/datasets/{spec_id}?{_GROUPBY_QUERY}")


def test_a_groupby_head_hit_renders_from_a_migrated_trace_conversion():
    """Migrations 0017 and 0020 rewrite a stored object-shaped `trace.conversion` into
    `MIGRATED_CONVERSION`. A groupby head hit renders from the stored frame and answers 200
    with the list on the wire."""
    got = _groupby_head_hit_over_stored_conversion(MIGRATED_CONVERSION)

    assert got.status_code == 200, got.text
    served = got.json()
    assert served["trace"]["served_from"] == "head"
    assert served["trace"]["conversion"] == MIGRATED_CONVERSION


def test_a_repeated_head_hit_keeps_minting_the_same_collapsed_content_version():
    """Repeated head hits keep the collapsed content version and create no third version record."""
    version_store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        head_writer=_FakeHeadStore(),
        artifact_store=_RenderableArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        ),
        code_index=_code_index(),
        coverage=_coverage_service(),
    )
    spec_id, sidecar = _spec_id_and_sidecar()
    fresh = _mint(client, spec_id=spec_id, sidecar=sidecar, options=_GROUPBY_OPTIONS)

    first = client.get(f"/api/datasets/{spec_id}?{_GROUPBY_QUERY}")
    second = client.get(f"/api/datasets/{spec_id}?{_GROUPBY_QUERY}")

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["content_version"] == fresh["content_version"]
    assert second.json()["content_version"] == fresh["content_version"]

    with version_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT content_version FROM datasets.version_record WHERE subject_id = %s "
            "ORDER BY content_version",
            (spec_id,),
        )
    recorded_versions = {row["content_version"] for row in rows}
    # The table contains the raw and collapsed versions only.
    assert len(recorded_versions) == 2
    assert fresh["content_version"] in recorded_versions


def test_a_collapsed_mint_never_moves_subject_head_to_the_collapsed_version():
    version_store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        head_writer=_FakeHeadStore(),
        artifact_store=_RenderableArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        ),
        code_index=_code_index(),
        coverage=_coverage_service(),
    )
    spec_id, sidecar = _spec_id_and_sidecar()

    fresh = _mint(client, spec_id=spec_id, sidecar=sidecar, options=_GROUPBY_OPTIONS)
    collapsed_head_hit = client.get(f"/api/datasets/{spec_id}?{_GROUPBY_QUERY}")
    assert collapsed_head_hit.status_code == 200, collapsed_head_hit.text

    # A plain GET serves the raw frame named by the subject head.
    raw = client.get(f"/api/datasets/{spec_id}?bundle=false")
    assert raw.status_code == 200, raw.text
    body = raw.json()

    assert body["trace"]["served_from"] == "head"
    assert body["content_version"] != fresh["content_version"]
    assert body["coverage"]["rows"] == 2


_FAMILY_LABEL = "Axis value A/Axis value B"


@pytest.mark.parametrize("structured", [True, False], ids=["structured", "without_fields"])
def test_a_stored_summed_warning_is_served_unchanged_on_a_head_hit_and_a_snapshot(
    structured: bool,
):
    """A collapsed version minted while collapses reported their sum as a warning stores a
    `stat_vars_summed` entry, with the structured fields or, earlier, without them. A head-hit
    GET and a snapshot GET return that entry unchanged, once."""
    version_store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        head_writer=_FakeHeadStore(),
        artifact_store=_RenderableArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        ),
        code_index=_code_index(),
        coverage=_coverage_service(),
    )
    spec_id, sidecar = _spec_id_and_sidecar()
    content_version = _mint(client, spec_id=spec_id, sidecar=sidecar, options=_GROUPBY_OPTIONS)[
        "content_version"
    ]
    stored: dict[str, object] = {
        "code": "stat_vars_summed",
        "severity": "info",
        "message": "The series sums observations from more than one StatVar.",
    }
    if structured:
        stored |= {
            "contributing_member_count": 2,
            "stat_var_ref": f"dcid:{spec_id}",
            "stat_var_label": "Axis value A/Axis value B",
        }
    with version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT warnings FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            (spec_id, content_version),
        )
        assert row is not None
        tx.execute(
            "UPDATE datasets.version_record SET warnings = %s "
            "WHERE subject_id = %s AND content_version = %s",
            (json.dumps([stored, *cast(list, row["warnings"])]), spec_id, content_version),
        )

    head_hit = client.get(f"/api/datasets/{spec_id}?{_GROUPBY_QUERY}")
    assert head_hit.status_code == 200, head_hit.text
    assert head_hit.json()["trace"]["served_from"] == "head"
    snapshot = client.get(f"/api/datasets/{spec_id}/{content_version}")
    assert snapshot.status_code == 200, snapshot.text

    for served in (head_hit.json(), snapshot.json()):
        assert served["content_version"] == content_version
        assert [w for w in served["warnings"] if w["code"] == "stat_vars_summed"] == [stored]


_LEVEL_OPTIONS = {
    "bundle": False,
    "groupby": {"by": ["recipient", "testAxis"], "op": "sum", "levels": {"testAxis": 0}},
}
_LEVEL_QUERY = "bundle=false&groupby.by=recipient,testAxis&groupby.op=sum&groupby.levels=testAxis:0"


_TWO_YEAR_CSV_BODY = _CSV_BODY + "".join(
    f"dataflow,DC:DF_OBS(1.0.0),I,{member},{_RECIPIENT},USD,GrantEquivalent,"
    f"NotApplicable,OECD_DAC2A,2021,{value},,facet1\r\n"
    for member, value in ((_MEMBER_A, 1100), (_MEMBER_B, 600))
)


def _two_year_sdmx_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=_TWO_YEAR_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )


def _two_year_coverage_service() -> FakeCoverageService:
    stat_vars = (_MEMBER_A, _MEMBER_B)
    return FakeCoverageService(
        build_coverage_index(
            [
                (stat_var, _FACET_ID, "country/USA", _RECIPIENT, [2020, 2021])
                for stat_var in stat_vars
            ],
            role_names={
                stat_var: RoleNames(donor="provider", recipient="recipient")
                for stat_var in stat_vars
            },
            facet_metadata={
                _FACET_ID: FacetMetadata(
                    provenance_id=_PROVENANCE, measurement_method=_MEASUREMENT_METHOD, unit=_UNIT
                )
            },
        )
    )


def _hierarchical_code_index() -> CodeIndex:
    """`_code_index` with `testAxis` declared a level axis. Both codes are roots, so level 0
    keeps them apart."""
    base = _code_index()
    return CodeIndex(
        stamp=base.stamp,
        declarations=tuple(
            replace(declaration, hierarchical=True, disjoint_from_children=True)
            for declaration in base.declarations.values()
        ),
        codes=base.codes,
        statvars=base.statvars,
    )


@pytest.mark.parametrize("door", ["spec", "definition", "head_hit"])
def test_a_level_roll_up_frame_describes_its_own_columns_and_pivots(door: str):
    version_store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_two_year_sdmx_transport(),
        version_store=version_store,
        # With no head, every definition GET materialises afresh.
        head_writer=_FakeHeadStore() if door == "head_hit" else None,
        artifact_store=_RenderableArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        ),
        code_index=_hierarchical_code_index(),
        coverage=_two_year_coverage_service(),
    )
    spec_id, sidecar = _spec_id_and_sidecar()

    if door == "spec":
        served = _mint(client, spec_id=spec_id, sidecar=sidecar, options=_LEVEL_OPTIONS)
    else:
        # The spec door registers the definition and records its raw version.
        _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
        got = client.get(f"/api/datasets/{spec_id}?{_LEVEL_QUERY}")
        assert got.status_code == 200, got.text
        served = got.json()
        assert (served["trace"].get("served_from") == "head") == (door == "head_hit")
    snapshot = client.get(f"/api/datasets/{spec_id}/{served['content_version']}")
    assert snapshot.status_code == 200, snapshot.text

    for manifest in (served, snapshot.json()):
        options = manifest["output_options"]
        assert options is not None
        assert "stat_var_testAxis" in {column["name"] for column in options["columns"]}
        # The kept axis is a role of the summed frame, so it pivots against the period.
        pivot = next(
            p
            for p in options["pivots"]
            if (p["index"], p["columns"]) == ("time_period", "stat_var_testAxis")
        )
        assert pivot["state"] == "available"
        # The period column plus one column per code.
        assert (pivot["width"], pivot["cells"]) == (3, 6)


def _pivot_head_hit_client(
    rows: list[tuple[str, str, int]], *, pivot_max_output_cells: int | None = None
):
    """Mints `(member, recipient, year)` rows under a two-member spec scoped to their recipients,
    so the next GET for a pivot rendering is a head hit. Returns the client, spec id and
    artifact store.
    """
    recipients = sorted({recipient for _, recipient, _ in rows})
    csv_body = _CSV_HEADER + "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{member},{recipient},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,{year},{1000 + i},,facet1\r\n"
        for i, (member, recipient, year) in enumerate(rows)
    )
    years: dict[tuple[str, str], list[int]] = {}
    for member, recipient, year in rows:
        years.setdefault((member, recipient), []).append(year)
    coverage = FakeCoverageService(
        build_coverage_index(
            [
                (member, _FACET_ID, "country/USA", recipient, sorted(member_years))
                for (member, recipient), member_years in years.items()
            ],
            role_names={
                member: RoleNames(donor="provider", recipient="recipient")
                for member in (_MEMBER_A, _MEMBER_B)
            },
            facet_metadata={
                _FACET_ID: FacetMetadata(
                    provenance_id=_PROVENANCE, measurement_method=_MEASUREMENT_METHOD, unit=_UNIT
                )
            },
        )
    )
    version_store = FakeVersionRecordDatasetStore()
    artifact_store = _RenderableArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _client(
        graph=_graph(),
        sdmx_transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text=csv_body,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )
        ),
        version_store=version_store,
        head_writer=_FakeHeadStore(),
        artifact_store=artifact_store,
        code_index=_code_index(),
        coverage=coverage,
        settings=(
            DatasetsBuildSettings(pivot_max_output_cells=pivot_max_output_cells)
            if pivot_max_output_cells is not None
            else None
        ),
    )
    identity = {**_IDENTITY, "entities": {"recipient": recipients}}
    spec_id = recompute_identity(identity).spec_id
    sidecar = {"schema_version": 1, "spec_id": spec_id, "identity": identity}
    _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    return client, spec_id, artifact_store


def _spy_on_frame_reads(
    monkeypatch: pytest.MonkeyPatch, artifact_store: _RenderableArtifactStore
) -> list[str]:
    """Records each `get` on `artifact_store` by digest, and each materialise permit a head-hit
    render takes as `"permit"`."""
    reads: list[str] = []
    get = artifact_store.get
    slot = current_module.bounded_materialise_slot

    def spied_get(*, artifact_digest: str, tier: Tier) -> bytes:
        reads.append(artifact_digest)
        return get(artifact_digest=artifact_digest, tier=tier)

    def spied_slot(*args, **kwargs):
        reads.append("permit")
        return slot(*args, **kwargs)

    monkeypatch.setattr(artifact_store, "get", spied_get)
    monkeypatch.setattr(current_module, "bounded_materialise_slot", spied_slot)
    return reads


def test_a_head_hit_pivot_over_the_cell_cap_refuses_from_the_stored_shape_without_reading_the_frame(
    monkeypatch,
):
    # A stat_var by time_period pivot is two rows by three columns, six cells over a cap of five.
    client, spec_id, artifact_store = _pivot_head_hit_client(
        [(member, _RECIPIENT, year) for member in (_MEMBER_A, _MEMBER_B) for year in (2020, 2021)],
        pivot_max_output_cells=5,
    )
    reads = _spy_on_frame_reads(monkeypatch, artifact_store)

    got = client.get(
        f"/api/datasets/{spec_id}?bundle=false&pivot.index=stat_var&pivot.columns=time_period"
    )

    assert got.status_code == 422, got.text
    assert got.json()["code"] == "pivot_over_cell_cap"
    assert reads == []


def _drop_resolved_stat_vars(version_store: FakeVersionRecordDatasetStore, **version: str) -> None:
    _edit_stored_trace(version_store, **version, edit=lambda trace: trace.pop("resolved_stat_vars"))


def _record_excel_overflowing_rows(
    version_store: FakeVersionRecordDatasetStore, *, subject_id: str, content_version: str
) -> None:
    """Records one more row than an `xlsx` sheet holds beside its header in the stored
    coverage. The stored frame keeps its own rows."""
    with version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT coverage FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            (subject_id, content_version),
        )
        assert row is not None
        coverage = cast(dict, row["coverage"])
        coverage["rows"] = EXCEL_MAX_ROWS
        tx.execute(
            "UPDATE datasets.version_record SET coverage = %s "
            "WHERE subject_id = %s AND content_version = %s",
            (coverage, subject_id, content_version),
        )


def _writer_row_detail(total_rows: int) -> str:
    """The `detail` the `xlsx` writer's row check gives for `total_rows`, header included."""
    with pytest.raises(ExcelLimitError) as refused:
        check_excel_rows(total_rows)
    return str(refused.value)


_STORED_RECORD_REFUSALS = [
    (_drop_resolved_stat_vars, "bundle=false&format=parquet", 500, "internal_error", None),
    (
        _record_excel_overflowing_rows,
        "bundle=false&format=xlsx",
        422,
        "excel_limit",
        _writer_row_detail(EXCEL_MAX_ROWS + 1),
    ),
]
_STORED_RECORD_IDS = ["no_resolved_stat_vars", "excel_rows"]


@pytest.mark.parametrize(
    ("edit", "query", "status", "code", "detail"), _STORED_RECORD_REFUSALS, ids=_STORED_RECORD_IDS
)
def test_a_head_hit_the_stored_record_refuses_reads_no_frame_and_takes_no_permit(
    monkeypatch: pytest.MonkeyPatch,
    edit: Callable[..., None],
    query: str,
    status: int,
    code: str,
    detail: str | None,
):
    """A trace without `resolved_stat_vars`, and an `xlsx` request on a version whose stored
    row count is over Excel's limit, refuse from the head's stored record. The `xlsx` refusal
    reads as the writer's own."""
    client, version_store, artifact_store = _collapse_client()
    spec_id, sidecar = _spec_id_and_sidecar()
    base = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    edit(version_store, subject_id=spec_id, content_version=base["content_version"])
    reads = _spy_on_frame_reads(monkeypatch, artifact_store)

    got = client.get(f"/api/datasets/{spec_id}?{query}")

    assert got.status_code == status, got.text
    assert got.json()["code"] == code
    assert detail is None or got.json()["detail"] == detail
    assert reads == []


@pytest.mark.parametrize(
    ("edit", "query"), [row[:2] for row in _STORED_RECORD_REFUSALS], ids=_STORED_RECORD_IDS
)
def test_a_head_hit_with_no_servable_frame_rebuilds_past_the_stored_record_refusals(
    edit: Callable[..., None], query: str
):
    """A head whose every location is past retention has no frame to render from, so it
    rebuilds, and the stored record's refusals do not answer the request."""
    client, version_store, _ = _collapse_client()
    spec_id, sidecar = _spec_id_and_sidecar()
    base = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    edit(version_store, subject_id=spec_id, content_version=base["content_version"])
    with version_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_location SET stored_at = %s",
            (datetime.now(UTC) - timedelta(days=31),),
        )

    got = client.get(f"/api/datasets/{spec_id}?{query}")

    assert got.status_code == 200, got.text


def test_a_plain_xlsx_with_an_unknown_column_on_an_over_limit_version_refuses_the_column():
    """The `columns` refusal outranks the stored row count's `excel_limit`, as the render
    raises it first."""
    client, version_store, _ = _collapse_client()
    spec_id, sidecar = _spec_id_and_sidecar()
    base = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    _record_excel_overflowing_rows(
        version_store, subject_id=spec_id, content_version=base["content_version"]
    )

    got = client.get(
        f"/api/datasets/{spec_id}?bundle=false&format=xlsx&columns=recipient,not_a_real_column"
    )

    assert got.status_code == 422, got.text
    assert got.json()["code"] == "column_unknown"


def test_a_link_with_no_stored_frame_on_a_trace_without_resolved_stat_vars_expires():
    """A rendering past retention whose version links no frame has nothing to render from,
    so its link answers `artifact_expired` before the trace's missing `resolved_stat_vars`
    matters."""
    version_store = FakeVersionRecordDatasetStore()
    store = _RenewingArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        head_writer=_FakeHeadStore(),
        artifact_store=store,
        code_index=_code_index(),
        coverage=_coverage_service(),
    )
    spec_id, sidecar = _spec_id_and_sidecar()
    fresh = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    csv = next(a for a in fresh["artifacts"] if a["media_type"] == "text/csv")
    _drop_resolved_stat_vars(
        version_store, subject_id=spec_id, content_version=fresh["content_version"]
    )
    with version_store.transaction() as tx:
        tx.execute(
            "DELETE FROM datasets.artifact_location WHERE artifact_digest IN ("
            "SELECT artifact_digest FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s AND rendering_key = %s)",
            (spec_id, fresh["content_version"], rendering_key(CANONICAL_FRAME_RENDERING)),
        )
        tx.execute(
            "UPDATE datasets.artifact_location SET stored_at = %s",
            (datetime.now(UTC) - timedelta(days=31),),
        )
    del store._objects[(csv["artifact_digest"], Tier.working.value)]

    resp = client.get(csv["url"], follow_redirects=False)

    assert resp.status_code == 410, resp.text
    assert resp.json()["code"] == "artifact_expired"


def test_a_head_hit_duplicate_key_pivot_refusal_names_the_varying_column_as_an_axis():
    # Both members report Kenya in 2020, so (time_period, recipient) repeats and stat_var varies.
    client, spec_id, _ = _pivot_head_hit_client(
        [
            (_MEMBER_A, _RECIPIENT, 2020),
            (_MEMBER_B, _RECIPIENT, 2020),
            (_MEMBER_A, "country/TGO", 2021),
        ]
    )

    got = client.get(
        f"/api/datasets/{spec_id}?bundle=false&pivot.index=time_period&pivot.columns=recipient"
    )

    assert got.status_code == 422, got.text
    problem = got.json()
    assert problem["code"] == "pivot_not_unique"
    assert [(issue.get("member"), issue.get("axis")) for issue in problem["issues"]] == [
        (f"('2020', '{_RECIPIENT}')", None),
        (None, "stat_var"),
    ]


@pytest.mark.parametrize("stored_frame", ["frame1", "canonical"])
def test_a_groupby_link_past_retention_re_renders_from_its_base_frame_and_redirects(
    stored_frame: str,
):
    """A collapse's rendering whose object is gone re-renders from the base version's frame,
    a canonical frame or a frame1 document alone, as its mint did, never by collapsing the
    collapsed frame again, and the identical bytes renew the clicked digest, which
    redirects."""
    version_store = FakeVersionRecordDatasetStore()
    store = _RenewingArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        head_writer=_FakeHeadStore(),
        artifact_store=store,
        code_index=_code_index(),
        coverage=_coverage_service(),
    )
    spec_id, sidecar = _spec_id_and_sidecar()
    fresh = _mint(client, spec_id=spec_id, sidecar=sidecar, options=_GROUPBY_OPTIONS)
    csv = next(a for a in fresh["artifacts"] if a["media_type"] == "text/csv")
    with version_store.transaction() as tx:
        base = tx.fetch_one(
            "SELECT base_content_version FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            (spec_id, fresh["content_version"]),
        )
        base_frame = tx.fetch_one(
            "SELECT artifact_digest FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s AND rendering_key = %s",
            (spec_id, base["base_content_version"], rendering_key(CANONICAL_FRAME_RENDERING)),
        )
    base_frame_digest = base_frame["artifact_digest"]
    if stored_frame == "frame1":
        base_frame_digest = link_frame1_only(
            version_store,
            store,
            subject_id=spec_id,
            content_version=base["base_content_version"],
        )
    with version_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_location SET stored_at = %s",
            (datetime.now(UTC) - timedelta(days=31),),
        )
    del store._objects[(csv["artifact_digest"], Tier.working.value)]

    resp = client.get(csv["url"], follow_redirects=False)

    assert resp.status_code == 302, resp.text
    assert store.renewals[:2] == [
        f"{base_frame_digest}#r1",
        f"{csv['artifact_digest']}#r1",
    ]
    assert resp.headers["location"] == f"fake://artifact/working/{csv['artifact_digest']}#r1"


_EARLIER_GRAMMAR_VERSION = "1.0.0"
_EARLIER_GRAMMAR_TOKEN = "cv_" + "0" * 32


@pytest.mark.parametrize(
    ("stored_frame", "frames_hashed"),
    [
        ("frame1_document", 1),
        ("canonical_frame", 2),
        ("backfilled_frame1", 1),
        ("recorded_frame_hash", 1),
    ],
)
def test_a_groupby_that_leaves_an_earlier_grammar_versions_frame_unchanged_is_that_version(
    stored_frame: str, frames_hashed: int, monkeypatch: pytest.MonkeyPatch
):
    """A version minted under an earlier frame grammar holds a token the current grammar does
    not mint from its frame. A sole-contributor full collapse leaves that frame unchanged, so
    the head hit answers the base version, records no second version and links the rendering
    under the base. The base frame's hash is the one its record holds, else the digest of
    its frame1 document, which a version `backfill-frame1` converted links beside its canonical
    frame, else the hash of its canonical frame, the one case that hashes a second frame
    beside the collapse."""
    version_store = FakeVersionRecordDatasetStore()
    head_store = _FakeHeadStore()
    artifact_store = _RenderableArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _client(
        graph=_graph(),
        sdmx_transport=_sole_member_sdmx_transport(),
        version_store=version_store,
        head_writer=head_store,
        artifact_store=artifact_store,
        code_index=_code_index(),
        coverage=_coverage_service(stat_vars=(_MEMBER_A,)),
    )
    spec_id, sidecar = _spec_id_and_sidecar_sole_member()
    minted = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})

    frame_key = rendering_key(CANONICAL_FRAME_RENDERING)
    with version_store.transaction() as tx:
        for table in ("version_record", "version_artifact", "materialisation"):
            tx.execute(
                f"UPDATE datasets.{table} SET content_version = %s WHERE subject_id = %s",
                (_EARLIER_GRAMMAR_TOKEN, spec_id),
            )
        tx.execute(
            "UPDATE datasets.version_record SET frame_grammar_version = %s WHERE subject_id = %s",
            (_EARLIER_GRAMMAR_VERSION, spec_id),
        )
        if stored_frame != "recorded_frame_hash":
            tx.execute(
                "UPDATE datasets.version_record SET frame_sha256 = NULL WHERE subject_id = %s",
                (spec_id,),
            )
        canonical = tx.fetch_one(
            "SELECT artifact_digest FROM datasets.version_artifact "
            "WHERE subject_id = %s AND rendering_key = %s",
            (spec_id, frame_key),
        )
    head_store._heads[spec_id] = replace(
        head_store._heads[spec_id], content_version=_EARLIER_GRAMMAR_TOKEN
    )
    if stored_frame in ("frame1_document", "backfilled_frame1"):
        document = io.BytesIO()
        write_frame1(
            (
                read_canonical_frame(
                    artifact_store.get(
                        artifact_digest=cast(str, canonical["artifact_digest"]), tier=Tier.working
                    )
                ),
            ),
            sink=document,
        )
        frame1 = artifact_store.put(document.getvalue(), tier=Tier.working)
        frame1_rendering = {"format": "frame1"}
        with version_store.transaction() as tx:
            if stored_frame == "frame1_document":
                tx.execute(
                    "DELETE FROM datasets.version_artifact "
                    "WHERE subject_id = %s AND rendering_key = %s",
                    (spec_id, frame_key),
                )
            tx.execute(
                "INSERT INTO datasets.version_artifact (subject_id, content_version, "
                "artifact_digest, rendering_key, rendering, media_type) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    spec_id,
                    _EARLIER_GRAMMAR_TOKEN,
                    frame1.digest,
                    rendering_key(frame1_rendering),
                    frame1_rendering,
                    FRAME1_MEDIA_TYPE,
                ),
            )

    hashed: list[object] = []
    hash_frame = render_module.frame_sha256
    monkeypatch.setattr(
        render_module, "frame_sha256", lambda frame: hashed.append(frame) or hash_frame(frame)
    )

    got = client.get(f"/api/datasets/{spec_id}?{_GROUPBY_QUERY}")

    assert got.status_code == 200, got.text
    served = got.json()
    assert served["trace"]["served_from"] == "head"
    assert served["content_version"] == _EARLIER_GRAMMAR_TOKEN
    assert len(hashed) == frames_hashed
    assert served["content_version"] != minted["content_version"]
    (csv,) = [a for a in served["artifacts"] if a["media_type"] == "text/csv"]
    with version_store.transaction() as tx:
        versions = tx.fetch_all(
            "SELECT content_version, base_content_version FROM datasets.version_record "
            "WHERE subject_id = %s",
            (spec_id,),
        )
        linked = tx.fetch_all(
            "SELECT content_version FROM datasets.version_artifact "
            "WHERE subject_id = %s AND artifact_digest = %s",
            (spec_id, csv["artifact_digest"]),
        )
    assert versions == [{"content_version": _EARLIER_GRAMMAR_TOKEN, "base_content_version": None}]
    assert {row["content_version"] for row in linked} == {_EARLIER_GRAMMAR_TOKEN}


@pytest.mark.parametrize("door", ["mint", "head_hit"])
def test_every_recorded_version_holds_the_hash_of_its_canonical_frame(door: str):
    """The base version and the collapse a mint or a head hit records each store
    `frame_sha256`, the hash of the canonical frame linked under them."""
    version_store = FakeVersionRecordDatasetStore()
    artifact_store = _RenderableArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        head_writer=_FakeHeadStore(),
        artifact_store=artifact_store,
        code_index=_code_index(),
        coverage=_coverage_service(),
    )
    spec_id, sidecar = _spec_id_and_sidecar()
    if door == "mint":
        _mint(client, spec_id=spec_id, sidecar=sidecar, options=_GROUPBY_OPTIONS)
    else:
        _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
        got = client.get(f"/api/datasets/{spec_id}?{_GROUPBY_QUERY}")
        assert got.status_code == 200, got.text

    with version_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT vr.frame_sha256 AS frame_sha256, va.artifact_digest AS artifact_digest "
            "FROM datasets.version_record vr JOIN datasets.version_artifact va "
            "ON va.subject_id = vr.subject_id AND va.content_version = vr.content_version "
            "WHERE vr.subject_id = %s AND va.rendering_key = %s",
            (spec_id, rendering_key(CANONICAL_FRAME_RENDERING)),
        )
    assert len(rows) == 2
    for row in rows:
        frame = read_canonical_frame(
            artifact_store.get(artifact_digest=cast(str, row["artifact_digest"]), tier=Tier.working)
        )
        assert row["frame_sha256"] == frame_sha256(frame)
