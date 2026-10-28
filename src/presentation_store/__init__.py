"""Postgres persistence for assembler components, compositions, and frame pins.
Importing reads no environment variables. Settings load through the provider or migration
entrypoint.
"""

from __future__ import annotations

from presentation_store.card_pins import pin_card_frames
from presentation_store.connection import PresentationStoreSettings
from presentation_store.errors import PresentationStoreConfigError, PresentationStoreError
from presentation_store.frame_store import (
    FRAME1_MEDIA_TYPE,
    PRESENTATION_PIN_REFERENCE_KIND,
    LiveFrameStore,
    frame_reads_bounded_by,
)
from presentation_store.provider import close_shared_pool, shared_pool
from presentation_store.store import LivePresentationStore
from presentation_store.store_errors import GeneratedIdCollision, PresentationStoreUnavailable

__all__ = [
    "FRAME1_MEDIA_TYPE",
    "PRESENTATION_PIN_REFERENCE_KIND",
    "GeneratedIdCollision",
    "LiveFrameStore",
    "LivePresentationStore",
    "PresentationStoreConfigError",
    "PresentationStoreError",
    "PresentationStoreSettings",
    "PresentationStoreUnavailable",
    "close_shared_pool",
    "frame_reads_bounded_by",
    "pin_card_frames",
    "shared_pool",
]
