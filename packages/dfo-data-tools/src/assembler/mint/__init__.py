"""Two-phase minting for assembler candidates.

`mint_component` builds and validates the component and provenance before
identity is known. `with_card_description` adds a card revision's description, and
`stamp_envelope` adds the identity and validates the envelope. `selection_facts` gives what a
card draws."""

from __future__ import annotations

from assembler.mint.builders import complete_window, default_window
from assembler.mint.card_description import DescribedRevision, with_card_description
from assembler.mint.dispatch import MintedComponent, mint_component, stamp_envelope
from assembler.mint.provenance import DESCRIPTION_PROVENANCE_MEMBERS
from assembler.mint.selection import selection_facts

__all__ = [
    "DESCRIPTION_PROVENANCE_MEMBERS",
    "DescribedRevision",
    "MintedComponent",
    "complete_window",
    "default_window",
    "mint_component",
    "selection_facts",
    "stamp_envelope",
    "with_card_description",
]
