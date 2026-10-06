"""Head-hit groupby renders must match fresh-mint content versions and bytes, preserve
collapsed warnings with their structured fields, and leave the subject head at the raw version.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import httpx
import pytest

import datasets_api.render as render_module
from artifact_store import LiveArtifactInventory
from assembler.ports import FrameRef
from code_index import CodeIndex
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.config import DatasetsBuildSettings
from presentation_store.frame_store import LiveFrameStore
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._graph_doubles import FakeMintGraphReader
from ._legacy_conversion_trace import MIGRATED_CONVERSION
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_definition_door import _client
from .test_definition_door_currency import _FakeHeadStore, _RenderableArtifactStore
from .test_pin_door import FakeArtifactReferenceWriter
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
    _spec_id_and_sidecar,
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

    # The head-hit collapse mints the same `stat_vars_summed` disclosure
    # the fresh path did.
    assert "stat_vars_summed" in {w["code"] for w in fresh["warnings"]}
    assert "stat_vars_summed" in {w["code"] for w in served["warnings"]}


def test_a_snapshot_of_the_collapsed_version_still_reports_stat_vars_summed():
    """The warning rides `Manifest.warnings` onto the recorded version body
    (`datasets_api.manifest.encode_version_body` excludes only `artifacts`/
    `output_options`), so a later `GET /api/datasets/{id}/{content_version}`
    against the collapsed identity decodes it back unchanged -- the same
    persistence `deflator_projected` already relies on.
    """
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

    snapshot_resp = client.get(f"/api/datasets/{spec_id}/{fresh['content_version']}")
    assert snapshot_resp.status_code == 200, snapshot_resp.text
    snapshot = snapshot_resp.json()

    assert "stat_vars_summed" in {w["code"] for w in snapshot["warnings"]}


def test_head_hit_with_a_converted_base_mints_the_same_collapsed_content_version():
    """The gap `content_version.mint_preimage_from_trace`'s own docstring
    names: a base materialisation whose unit axis carried an explicit
    currency-conversion request. `ManifestTrace.facets` alone (the wider,
    7-key `as_trace_entry()` shape) cannot rebuild that axis's true grammar
    ref -- only `ManifestTrace.mint_facets` (the exact
    `as_grammar_facet()`-shaped preimage captured at finish time) can. This
    pins that the head-hit collapse mints the identical `content_version`
    the fresh path did for exactly this case."""
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


def _groupby_head_hit_over_stored_conversion(conversion: object) -> httpx.Response:
    """Mints the raw version, replaces its stored `trace.conversion`, then requests a groupby
    rendering, which the head hit builds from the stored frame1 and trace."""
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
    base = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})

    with version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT immutable_body FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            (spec_id, base["content_version"]),
        )
        assert row is not None
        body = cast(dict, row["immutable_body"])
        body["trace"]["conversion"] = conversion
        tx.execute(
            "UPDATE datasets.version_record SET immutable_body = %s "
            "WHERE subject_id = %s AND content_version = %s",
            (body, spec_id, base["content_version"]),
        )

    return client.get(f"/api/datasets/{spec_id}?{_GROUPBY_QUERY}")


def test_a_groupby_head_hit_renders_from_a_migrated_trace_conversion():
    """Migrations 0017 and 0020 rewrite a stored object-shaped `trace.conversion` into
    `MIGRATED_CONVERSION`. A groupby head hit rebuilds its preimage from that one entry and
    answers 200 with the list on the wire."""
    got = _groupby_head_hit_over_stored_conversion(MIGRATED_CONVERSION)

    assert got.status_code == 200, got.text
    served = got.json()
    assert served["trace"]["served_from"] == "head"
    assert served["trace"]["conversion"] == MIGRATED_CONVERSION
    assert "stat_vars_summed" in {w["code"] for w in served["warnings"]}


def test_a_groupby_head_hit_over_a_two_entry_trace_conversion_is_an_internal_error():
    """A stored `trace.conversion` list of two entries is corrupt: the head hit refuses it
    as an internal error rather than rendering under a preimage it cannot read."""
    got = _groupby_head_hit_over_stored_conversion([*MIGRATED_CONVERSION, *MIGRATED_CONVERSION])

    assert got.status_code == 500, got.text
    assert got.json()["code"] == "internal_error"


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


def _summed_warning(manifest: dict) -> dict:
    return next(w for w in manifest["warnings"] if w["code"] == "stat_vars_summed")


def test_the_summed_warning_carries_its_structured_fields_on_fresh_head_hit_and_snapshot():
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
    head_hit = client.get(f"/api/datasets/{spec_id}?{_GROUPBY_QUERY}")
    assert head_hit.status_code == 200, head_hit.text
    assert head_hit.json()["trace"]["served_from"] == "head"
    snapshot = client.get(f"/api/datasets/{spec_id}/{fresh['content_version']}")
    assert snapshot.status_code == 200, snapshot.text

    for manifest in (fresh, head_hit.json(), snapshot.json()):
        warning = _summed_warning(manifest)
        assert warning["contributing_member_count"] == 2
        assert warning["stat_var_ref"] == f"dcid:{spec_id}"
        assert warning["stat_var_label"] == _FAMILY_LABEL
        for other in manifest["warnings"]:
            if other["code"] != "stat_vars_summed":
                assert set(other) == {"code", "severity", "message"}


def test_the_summed_warning_label_is_the_presentation_door_label_for_the_collapsed_version():
    store = FakeDatasetStore()
    version_store = FakeVersionRecordDatasetStore()
    artifact_store = _RenderableArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _client(
        graph=_graph(),
        store=store,
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        head_writer=_FakeHeadStore(),
        artifact_store=artifact_store,
        code_index=_code_index(),
        coverage=_coverage_service(),
    )
    spec_id, sidecar = _spec_id_and_sidecar()
    collapsed = _mint(client, spec_id=spec_id, sidecar=sidecar, options=_GROUPBY_OPTIONS)
    warning = _summed_warning(collapsed)

    frame_store = LiveFrameStore(
        snapshots=LiveSnapshotReader(store=version_store),
        artifacts=artifact_store,
        graph=None,
        reference_writer=FakeArtifactReferenceWriter(),
        spec_reader=LiveSpecReader(store=store),
        code_index=_code_index(),
    )
    resolved = frame_store.load(
        frame_ref=FrameRef(spec_id=spec_id, content_version=collapsed["content_version"])
    )

    assert resolved.labels[warning["stat_var_ref"]] == warning["stat_var_label"] == _FAMILY_LABEL


def _relabelled_code_index() -> CodeIndex:
    """`_code_index` with every code relabelled, as after a code-index re-export."""
    base = _code_index()
    return CodeIndex(
        stamp=base.stamp,
        declarations=tuple(base.declarations.values()),
        codes=tuple(replace(code, label=f"Renamed {code.code}") for code in base.codes),
        statvars=base.statvars,
    )


def test_every_door_returns_the_recorded_summed_label_after_a_code_index_relabel():
    """A collapsed version recorded under one code-index label keeps that label on a head-hit
    GET, a snapshot GET, and a repeat mint served under a relabelled index."""
    store = FakeDatasetStore()
    version_store = FakeVersionRecordDatasetStore()
    head_store = _FakeHeadStore()
    artifact_store = _RenderableArtifactStore(inventory=LiveArtifactInventory(store=version_store))

    def _client_with(code_index: CodeIndex):
        return _client(
            graph=_graph(),
            store=store,
            sdmx_transport=_sdmx_transport(),
            version_store=version_store,
            head_writer=head_store,
            artifact_store=artifact_store,
            code_index=code_index,
            coverage=_coverage_service(),
        )

    spec_id, sidecar = _spec_id_and_sidecar()
    recorded = _mint(
        _client_with(_code_index()), spec_id=spec_id, sidecar=sidecar, options=_GROUPBY_OPTIONS
    )
    assert _summed_warning(recorded)["stat_var_label"] == _FAMILY_LABEL

    relabelled = _client_with(_relabelled_code_index())
    head_hit = relabelled.get(f"/api/datasets/{spec_id}?{_GROUPBY_QUERY}")
    assert head_hit.status_code == 200, head_hit.text
    assert head_hit.json()["trace"]["served_from"] == "head"
    snapshot = relabelled.get(f"/api/datasets/{spec_id}/{recorded['content_version']}")
    assert snapshot.status_code == 200, snapshot.text
    repeat = _mint(relabelled, spec_id=spec_id, sidecar=sidecar, options=_GROUPBY_OPTIONS)

    for manifest in (head_hit.json(), snapshot.json(), repeat):
        assert manifest["content_version"] == recorded["content_version"]
        assert [w for w in manifest["warnings"] if w["code"] == "stat_vars_summed"] == [
            _summed_warning(recorded)
        ]
    assert head_hit.json()["warnings"] == snapshot.json()["warnings"]


def test_a_head_hit_returns_a_stored_summed_warning_recorded_without_structured_fields():
    """A collapsed version whose stored `stat_vars_summed` entry predates the structured fields
    returns that entry unchanged, once, on a head-hit GET."""
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
    with version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT immutable_body FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            (spec_id, content_version),
        )
        assert row is not None
        body = cast(dict, row["immutable_body"])
        body["warnings"] = [
            {key: w[key] for key in ("code", "severity", "message")}
            if w["code"] == "stat_vars_summed"
            else w
            for w in body["warnings"]
        ]
        tx.execute(
            "UPDATE datasets.version_record SET immutable_body = %s "
            "WHERE subject_id = %s AND content_version = %s",
            (body, spec_id, content_version),
        )
    (legacy,) = [w for w in body["warnings"] if w["code"] == "stat_vars_summed"]

    head_hit = client.get(f"/api/datasets/{spec_id}?{_GROUPBY_QUERY}")
    assert head_hit.status_code == 200, head_hit.text
    served = head_hit.json()

    assert served["trace"]["served_from"] == "head"
    assert served["content_version"] == content_version
    assert [w for w in served["warnings"] if w["code"] == "stat_vars_summed"] == [legacy]


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
    so the next GET for a pivot rendering is a head hit. Returns the client and spec id.
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
        artifact_store=_RenderableArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        ),
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
    return client, spec_id


def _spy_on_row_reads(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Wraps `render.fold_pivot_chunks`; each call appends whether it read any chunk's rows."""
    rows_read: list[bool] = []
    fold = render_module.fold_pivot_chunks

    class _Proxy:
        def __init__(self, chunk):
            self.columns = chunk.columns
            self._chunk = chunk

        @property
        def rows(self):
            rows_read[-1] = True
            return self._chunk.rows

    def spied(chunks, **kwargs):
        rows_read.append(False)
        return fold((_Proxy(chunk) for chunk in chunks), **kwargs)

    monkeypatch.setattr(render_module, "fold_pivot_chunks", spied)
    return rows_read


def test_a_head_hit_pivot_over_the_cell_cap_refuses_from_the_stored_shape_without_reading_rows(
    monkeypatch,
):
    # A stat_var by time_period pivot is two rows by three columns, six cells over a cap of five.
    client, spec_id = _pivot_head_hit_client(
        [(member, _RECIPIENT, year) for member in (_MEMBER_A, _MEMBER_B) for year in (2020, 2021)],
        pivot_max_output_cells=5,
    )
    rows_read = _spy_on_row_reads(monkeypatch)

    got = client.get(
        f"/api/datasets/{spec_id}?bundle=false&pivot.index=stat_var&pivot.columns=time_period"
    )

    assert got.status_code == 422, got.text
    assert got.json()["code"] == "pivot_over_cell_cap"
    assert rows_read == [False]


def test_a_head_hit_duplicate_key_pivot_refusal_names_the_varying_column_as_an_axis():
    # Both members report Kenya in 2020, so (time_period, recipient) repeats and stat_var varies.
    client, spec_id = _pivot_head_hit_client(
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
