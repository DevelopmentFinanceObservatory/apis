"""The raw presence read a `place.ConsultedPresence` holds, for tests that check what the read
gave. Search reads it only through the wrapper's noting methods."""

from __future__ import annotations

from materialise.place_scope import DefaultPresence
from search.engine.place import ConsultedPresence


def raw_presence(presence: ConsultedPresence) -> DefaultPresence:
    """The `DefaultPresence` `presence` wraps."""
    return presence._presence
