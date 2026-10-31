"""Build the live `FrameStore` shared by the components, frames and shares doors, with the
snapshot, recipe and description readers it reads through. Its pins spend from the allowance
ledger the door passes (`datasets_api.retention_admission`). Beside it comes a store over the
same readers whose pins spend nothing, which the share import pins through. The render routes decide whether a target frame may stand in for a stored one from the snapshot and
recipe readers.
Each door borrows the ref-counted dataset pool independently. This function closes the pool
borrow and graph client if construction fails.
"""

from __future__ import annotations

from dataclasses import dataclass

from artifact_store import LiveArtifactInventory, LiveArtifactStore, Tier
from dataset_store import LiveDatasetStore
from dataset_store import close_shared_pool as close_dataset_pool
from dataset_store import shared_pool as dataset_shared_pool
from dataset_store.snapshot import LiveSnapshotReader, SnapshotReader
from datasets_api.allowance import SharedAllowanceLedger
from datasets_api.description_store import LiveDescriptionStore
from datasets_api.recipe_store import LiveRecipeStore, RecipeReader
from datasets_api.reference_store import LiveArtifactReferenceWriter
from origin_catalogue.graph import OriginGraphClient, build_graph_client
from presentation_api.config import (
    PRESENTATION_COMPONENTS_TIMEOUT_SECONDS,
    PRESENTATION_FRAMES_TIMEOUT_SECONDS,
    PresentationSettings,
)
from presentation_store import LiveFrameStore
from presentation_store.frame_store import DescriptionReader
from spec_registry import LiveSpecReader

__all__ = ["LiveFrames", "build_live_frame_store"]


@dataclass(frozen=True)
class LiveFrames:
    """A live frame store, the graph client to close at teardown, and the readers over the
    datasets pool the frame store was built on. `unmetered_frame_store` pins over the same
    readers and spends no allowance or budget, for the share import; it reads no graph."""

    frame_store: LiveFrameStore
    unmetered_frame_store: LiveFrameStore
    graph: OriginGraphClient
    snapshot_reader: SnapshotReader
    recipe_reader: RecipeReader


def build_live_frame_store(
    *, settings: PresentationSettings, allowances: SharedAllowanceLedger | None
) -> LiveFrames:
    """Build a frame store over the datasets pool and return it with its graph client and
    readers. Its pins spend from `allowances`, `None` for a door that never pins
    (`LiveFrameStore`). Success registers one pool borrow for the caller to release. Failure
    closes that borrow and any graph client already built.
    """
    connections = dataset_shared_pool()
    graph: OriginGraphClient | None = None
    try:
        store = LiveDatasetStore(connections=connections)
        graph = build_graph_client(
            base_url=settings.datasets.graph_base_url,
            token=settings.graph_token,
            un_base_url=settings.datasets.un_graph_base_url,
        )
        snapshot_reader = LiveSnapshotReader(store=store)
        recipe_reader = LiveRecipeStore(store=store)
        description_reader: DescriptionReader = LiveDescriptionStore(store=store)
        artifacts = LiveArtifactStore(
            buckets={
                Tier.working: settings.datasets.artifact_bucket_working,
                Tier.pinned: settings.datasets.artifact_bucket_pinned,
            },
            inventory=LiveArtifactInventory(store=store),
        )
        reference_writer = LiveArtifactReferenceWriter(store=store)
        spec_reader = LiveSpecReader(store=store)
        frame_store = LiveFrameStore(
            snapshots=snapshot_reader,
            artifacts=artifacts,
            graph=graph,
            reference_writer=reference_writer,
            spec_reader=spec_reader,
            recipe_reader=recipe_reader,
            description_reader=description_reader,
            allowances=allowances,
            # Both doors call `load()`, so the shorter wait bounds its graph reads.
            read_budget_seconds=min(
                PRESENTATION_COMPONENTS_TIMEOUT_SECONDS, PRESENTATION_FRAMES_TIMEOUT_SECONDS
            ),
        )
        unmetered_frame_store = LiveFrameStore(
            snapshots=snapshot_reader,
            artifacts=artifacts,
            graph=None,
            reference_writer=reference_writer,
            allowances=None,
            spec_reader=spec_reader,
            recipe_reader=recipe_reader,
        )
    except BaseException:
        if graph is not None:
            graph.close()
        close_dataset_pool()
        raise
    return LiveFrames(
        frame_store=frame_store,
        unmetered_frame_store=unmetered_frame_store,
        graph=graph,
        snapshot_reader=snapshot_reader,
        recipe_reader=recipe_reader,
    )
