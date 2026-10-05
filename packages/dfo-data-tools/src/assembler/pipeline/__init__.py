"""Assembler operations for writes, frame descriptions, and component readback.

`write_component` validates intent before resolving and minting a candidate. A
create allocates an id and pins its frame before the write transaction, so the
GCS copy does not hold the presentation row lock or a pool connection. A revise
reads the current component first, since tolerant validation needs its `type`.
The transaction rechecks current state and rebuilds the candidate if it changed.
Failures after pinning log the reconciliation marker. Revises reuse the pinned
frame because `frame_ref` is create-only.

A create may carry a client-minted `cmw_` token. `write_component` looks its digest up
before resolving a frame: a stored create with the same request hash is replayed from its
revision 1, and one with another request is refused. A create that loses the race for the
token's unique index replays the winner the same way.

`note_narration` has no frame. `source_card` reads frame provenance without
selecting or binding series. HTTP doors wrap these library operations.
"""

from __future__ import annotations

from assembler.pipeline.describe_component import get_component
from assembler.pipeline.describe_frame import get_frame_view
from assembler.pipeline.write import ORPHANED_PIN_RECONCILIATION_REQUIRED_MARKER, write_component

__all__ = [
    "ORPHANED_PIN_RECONCILIATION_REQUIRED_MARKER",
    "get_component",
    "get_frame_view",
    "write_component",
]
