"""Build the live `FrameStore` shared by the components and frames doors, with the snapshot,
recipe and description readers it reads through. The render routes decide whether a target frame may stand
in for a stored one from those two readers.
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
    datasets pool the frame store was built on."""

    frame_store: LiveFrameStore
    graph: OriginGraphClient
    snapshot_reader: SnapshotReader
    recipe_reader: RecipeReader


def build_live_frame_store(*, settings: PresentationSettings) -> LiveFrames:
    """Build a frame store over the datasets pool and return it with its graph client and
    readers. Success registers one pool borrow for the caller to release. Failure closes that
    borrow and any graph client already built.
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
        frame_store = LiveFrameStore(
            snapshots=snapshot_reader,
            artifacts=LiveArtifactStore(
                buckets={
                    Tier.working: settings.datasets.artifact_bucket_working,
                    Tier.pinned: settings.datasets.artifact_bucket_pinned,
                },
                inventory=LiveArtifactInventory(store=store),
            ),
            graph=graph,
            reference_writer=LiveArtifactReferenceWriter(store=store),
            spec_reader=LiveSpecReader(store=store),
            recipe_reader=recipe_reader,
            description_reader=description_reader,
            # Both doors call `load()`, so the shorter wait bounds its graph reads.
            read_budget_seconds=min(
                PRESENTATION_COMPONENTS_TIMEOUT_SECONDS, PRESENTATION_FRAMES_TIMEOUT_SECONDS
            ),
        )
    except BaseException:
        if graph is not None:
            graph.close()
        close_dataset_pool()
        raise
    return LiveFrames(
        frame_store=frame_store,
        graph=graph,
        snapshot_reader=snapshot_reader,
        recipe_reader=recipe_reader,
    )
