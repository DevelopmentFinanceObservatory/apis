"""A refusal of a revision's stored intent, raised by a route that replays it.

The render routes (`presentation_api.doors.render`) and a share slot's reads
(`presentation_api.share_slots.serve_on_slot_frame`) draw a stored revision's intent, which
their caller cannot change. Each runs the draw inside `replaying_stored_intent`, so an
`AssemblerRefusal` leaves it as `StoredIntentRefused`. The components door's classifier
(`presentation_api.doors.components.classify_components_exception`) classifies such a refusal as
the one it wraps and states the `retry_mode` of a refusal of a stored intent
(`presentation_api.refusals.retry_mode`): `do_not_retry` where a request's refusal would state
`correct_and_retry`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from assembler.errors import AssemblerRefusal

__all__ = ["StoredIntentRefused", "replaying_stored_intent"]


class StoredIntentRefused(Exception):
    """A replay refused the revision's stored intent. `refusal` is what the replay raised
    (`replaying_stored_intent`)."""

    def __init__(self, refusal: AssemblerRefusal) -> None:
        self.refusal = refusal
        super().__init__(str(refusal))


@contextmanager
def replaying_stored_intent() -> Iterator[None]:
    """Re-raises an `AssemblerRefusal` raised inside as `StoredIntentRefused`."""
    try:
        yield
    except AssemblerRefusal as exc:
        raise StoredIntentRefused(exc) from exc
