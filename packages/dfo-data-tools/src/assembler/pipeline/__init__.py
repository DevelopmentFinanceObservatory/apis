"""Assembler operations for writes, frame descriptions, and component readback.

`write_component` validates intent before resolving and minting a candidate. A
create allocates an id before the write transaction. A revise reads the current
component first, since tolerant validation needs its `type`. The transaction
rechecks current state and rebuilds the candidate if it changed. Revises keep the
frame and its stored rendering because `frame_ref` is create-only. No write pins
its frame. A share pins the frames of the cards it includes
(`presentation_store.pin_card_frames`).

A create may carry a client-minted `cmw_` token. `write_component` looks its digest up
before resolving a frame: a stored create with the same request hash is replayed from its
revision 1, and one with another request is refused. A create that loses the race for the
token's unique index replays the winner the same way.

A create's `frame_ref` may name a dataset id alone (`write.bare_frame_ref`). The caller passes
`resolve_spec`, which names the version it stands for, and the revision stores that pair.

`render_component` rebuilds a stored revision's intent on a frame the caller has accepted,
pins nothing, and appends no revision.

`note_narration` has no frame. `source_card` reads frame provenance without
selecting or binding series. HTTP doors wrap these library operations.
"""

from __future__ import annotations

from assembler.pipeline.describe_component import get_component
from assembler.pipeline.describe_frame import get_frame_view
from assembler.pipeline.render import ComponentHoldsNoFrame, render_component
from assembler.pipeline.write import BARE_SPEC_ID, bare_frame_ref, write_component

__all__ = [
    "BARE_SPEC_ID",
    "ComponentHoldsNoFrame",
    "bare_frame_ref",
    "get_component",
    "get_frame_view",
    "render_component",
    "write_component",
]
