"""Build the live `FrameStore` shared by the components and frames doors.
Each door borrows the ref-counted dataset pool independently. This function closes the pool
borrow and graph client if construction fails.
"""

from __future__ import annotations

from artifact_store import LiveArtifactInventory, LiveArtifactStore, Tier
from dataset_store import LiveDatasetStore
from dataset_store import close_shared_pool as close_dataset_pool
from dataset_store import shared_pool as dataset_shared_pool
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.reference_store import LiveArtifactReferenceWriter
from origin_catalogue.graph import OriginGraphClient, build_graph_client
from presentation_api.config import (
    PRESENTATION_COMPONENTS_TIMEOUT_SECONDS,
    PRESENTATION_FRAMES_TIMEOUT_SECONDS,
    PresentationSettings,
)
from presentation_store import LiveFrameStore
from spec_registry import LiveSpecReader

__all__ = ["build_live_frame_store"]


def build_live_frame_store(
    *, settings: PresentationSettings
) -> tuple[LiveFrameStore, OriginGraphClient]:
    """Build a frame store over the datasets pool and return its graph client for teardown.
    Success registers one pool borrow for the caller to release. Failure closes that borrow
    and any graph client already built.
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
        frame_store = LiveFrameStore(
            snapshots=LiveSnapshotReader(store=store),
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
    return frame_store, graph
