"""Two-phase minting for assembler candidates.

`mint_component` builds and validates the component and provenance before
identity is known. `stamp_envelope` adds the identity and validates the envelope."""

from __future__ import annotations

from assembler.mint.builders import complete_window, default_window
from assembler.mint.dispatch import MintedComponent, mint_component, stamp_envelope

__all__ = [
    "MintedComponent",
    "complete_window",
    "default_window",
    "mint_component",
    "stamp_envelope",
]
