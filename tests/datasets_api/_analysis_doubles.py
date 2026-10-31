"""Doubles for the analysis runner: a root source holding frames by version, and stores over one
SQLite-backed dataset store, so the runner publishes and writes recipes through the live SQL."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from dfo_frame_grammar import Frame

from artifact_store import LiveArtifactInventory, LiveArtifactStore, Tier
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.analysis import (
    AnalysisPending,
    AnalysisStores,
    DatasetRoot,
    LocatedRoot,
    ResolvedRoot,
    RootNotStored,
    RootRendering,
    SpecRoot,
    recorded_dataset_root,
)
from datasets_api.analysis_roots import mint_options
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.version_store import LiveVersionRecordWriter
from dfo_analysis_engine.frame_ref import FrameRef
from materialise.frame_build import build_frame
from tests.artifact_store.conftest import FakeGCSClient

from ._render_inputs import ENRICHMENT, SITE
from ._version_record_store import FakeVersionRecordDatasetStore

AID = "spec_" + "a" * 32
POPULATION = "spec_" + "b" * 32
AID_V1 = "cv_" + "1" * 32
AID_V2 = "cv_" + "2" * 32
POPULATION_V1 = "cv_" + "3" * 32
_STAT_VAR = "dcid:Amount_ODA"
_POPULATION_STAT_VAR = "dcid:Count_Person"


def aid_frame(
    *,
    scale: float = 1.0,
    stat_var: str = _STAT_VAR,
    years: tuple[str, ...] = ("2021", "2022"),
) -> Frame:
    """ODA from one donor to two recipients over 2021 and 2022, each recipient a party."""
    rows = [
        {
            "stat_var": stat_var,
            "financingRecipient": recipient,
            "financingRecipient_party_kind": "party",
            "origin": donor,
            "time_period": year,
            "value": value * scale,
            "unit": "USDollar",
        }
        for recipient, donor, year, value in (
            ("country/KEN", "country/DEU", "2021", 10.0),
            ("country/KEN", "country/DEU", "2022", 12.0),
            ("country/NGA", "country/DEU", "2021", 20.0),
            ("country/NGA", "country/DEU", "2022", 30.0),
        )
        if year in years
    ]
    return build_frame(
        role_keys=["financingRecipient", "financingRecipient_party_kind", "origin"], rows=rows
    )


def population_frame(*, years: tuple[str, ...] = ("2021", "2022")) -> Frame:
    rows = [
        {
            "stat_var": _POPULATION_STAT_VAR,
            "observationAbout": place,
            "time_period": year,
            "value": value,
            "unit": "Person",
        }
        for place, year, value in (
            ("country/KEN", "2021", 50.0),
            ("country/KEN", "2022", 60.0),
            ("country/NGA", "2021", 200.0),
            ("country/NGA", "2022", 210.0),
        )
        if year in years
    ]
    return build_frame(role_keys=["observationAbout"], rows=rows)


def _options(rendering: RootRendering) -> dict[str, Any]:
    """The mint options a root of `rendering` is read with (`analysis_roots.mint_options`), with
    no enrichment columns where `rendering` names none."""
    if rendering.columns is None:
        rendering = rendering.model_copy(update={"columns": ()})
    return mint_options(rendering)


def _record(dataset: FrameRef, options: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "frame": dataset.model_dump(),
        "dataset": dataset.model_dump(),
        "options": dict(options),
        "scope": {},
    }


class FakeRootSource:
    """Frames by `(spec_id, content_version)`, the newest version of each dataset, and the
    recipes of the stores' derived frames. `pending` names datasets whose newest version is
    still building. A root's version records its frame's rows unless `recorded_rows` is false.
    Every resolve is recorded."""

    def __init__(
        self,
        *,
        frames: Mapping[tuple[str, str], Frame],
        newest: Mapping[str, str],
        recipes: LiveRecipeStore | None = None,
        pending: Mapping[str, str] | None = None,
        recorded_rows: bool = True,
    ) -> None:
        self.frames = dict(frames)
        self.recorded_rows = recorded_rows
        self.newest = dict(newest)
        self.recipes = recipes
        self.pending = dict(pending or {})
        self.resolved: list[tuple[str, str | None, bool]] = []
        self.reads = 0

    def resolve(
        self, root: SpecRoot | DatasetRoot, *, latest: bool, trace_id: str | None
    ) -> ResolvedRoot:
        given = root.content_version if isinstance(root, DatasetRoot) else None
        self.resolved.append((root.spec_id, given, latest))
        if root.spec_id in self.pending and (latest or given is None):
            raise AnalysisPending(operations=(self.pending[root.spec_id],), retry_after_s=2.0)
        content_version = given if given is not None and not latest else self.newest[root.spec_id]
        if (root.spec_id, content_version) not in self.frames:
            raise RootNotStored(
                spec_id=root.spec_id, content_version=content_version, reason="not held"
            )
        dataset = FrameRef(spec_id=root.spec_id, content_version=content_version)
        options = _options(root.rendering)
        frame = self.frames[(root.spec_id, content_version)]
        return ResolvedRoot(
            frame=dataset,
            dataset=dataset,
            record=_record(dataset, options),
            rendering=options,
            rows=frame.num_rows if self.recorded_rows else None,
        )

    def locate(self, root: ResolvedRoot) -> LocatedRoot:
        """`root` as it is. The frames are held in memory."""
        return LocatedRoot.of(root)

    def read(self, root: LocatedRoot) -> Frame:
        self.reads += 1
        return self.frames[(root.dataset.spec_id, root.dataset.content_version)]

    def recipe(self, *, spec_id: str, content_version: str) -> Mapping[str, Any] | None:
        if self.recipes is None:
            return None
        record = self.recipes.read_recipe(spec_id=spec_id, content_version=content_version)
        return None if record is None else record.body

    def dataset_root(self, record: Mapping[str, Any]) -> DatasetRoot:
        return recorded_dataset_root(record)


class Stores:
    """The runner's stores over one SQLite-backed dataset store."""

    def __init__(self) -> None:
        self.store = FakeVersionRecordDatasetStore()
        self.now = datetime(2026, 10, 28, tzinfo=UTC)
        self.gcs = FakeGCSClient()
        self.artifact_store = LiveArtifactStore(
            buckets={Tier.working: "working-bucket", Tier.pinned: "pinned-bucket"},
            inventory=LiveArtifactInventory(store=self.store, now=lambda: self.now),
            client=self.gcs,
            now=lambda: self.now,
        )
        self.snapshot_reader = LiveSnapshotReader(store=self.store)
        self.version_writer = LiveVersionRecordWriter(store=self.store, now=lambda: self.now)
        self.recipe_store = LiveRecipeStore(store=self.store, now=lambda: self.now)

    def analysis(self) -> AnalysisStores:
        return AnalysisStores(
            snapshot_reader=self.snapshot_reader,
            version_writer=self.version_writer,
            artifact_store=self.artifact_store,
            recipe_store=self.recipe_store,
            enrichment=ENRICHMENT,
            site=SITE,
            now=lambda: self.now,
        )

    def version_count(self) -> int:
        with self.store.transaction() as tx:
            row = tx.fetch_one("SELECT COUNT(*) AS n FROM datasets.version_record", ())
        assert row is not None
        return int(row["n"])  # ty: ignore[invalid-argument-type] -- the fake returns an int


def source(stores: Stores, **overrides: Any) -> FakeRootSource:
    """A source holding two versions of the aid dataset and one of population."""
    defaults: dict[str, Any] = {
        "frames": {
            (AID, AID_V1): aid_frame(),
            (AID, AID_V2): aid_frame(scale=2.0),
            (POPULATION, POPULATION_V1): population_frame(),
        },
        "newest": {AID: AID_V2, POPULATION: POPULATION_V1},
        "recipes": stores.recipe_store,
    }
    defaults.update(overrides)
    return FakeRootSource(**defaults)


def dataset(spec_id: str, content_version: str) -> dict[str, str]:
    return {"spec_id": spec_id, "content_version": content_version}


def step(verb: str, data: object, other: object | None = None, **arguments: object) -> dict:
    inputs: dict[str, object] = {"data": data}
    if other is not None:
        inputs["other"] = other
    return {"verb": verb, "arguments": arguments, "inputs": inputs}


def ref_of(*members: tuple[str, str]) -> dict[str, Any]:
    """A format 1 ref holding `members`, as a client saves it."""
    return {
        "format_version": 1,
        "members": [
            {
                "parts": [],
                "sentence": "A sentence.",
                "spec": {"spec_id": spec_id, "definition": {}},
                "content_version": content_version,
                "artifact": {
                    "digest": "sha256:" + "0" * 64,
                    "rendering": {"format": "parquet", "bundle": False},
                },
            }
            for spec_id, content_version in members
        ],
    }
