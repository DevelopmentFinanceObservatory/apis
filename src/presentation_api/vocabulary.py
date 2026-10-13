"""The words Presentation words a card's description from.

`LiveVocabularySource` holds the vocabulary of the process-wide code and measure indexes and the
packaged publisher short names (`description.vocabulary.build_vocabulary`). The labels of the
places, organisations, variables and sources a card draws come from its frame, which the frame
store labelled when it loaded it (`assembler.mint.card_description`), so building the
vocabulary reads no graph and serves under no catalogue, so it carries no graph generation or
catalogue build. A card's description revises its revision with the graph generation and
catalogue build the frame store read the card's frame under
(`assembler.mint.card_description`).
"""

from __future__ import annotations

from code_index.shared import get_index, get_measure_index
from description.vocabulary import Vocabulary, build_vocabulary

__all__ = ["LiveVocabularySource"]


class LiveVocabularySource:
    """`assembler.ports.VocabularySource` over the process-wide code and measure indexes.

    Builds its vocabulary once, at construction, which parses the indexes on first use."""

    def __init__(self) -> None:
        self._vocabulary = build_vocabulary(
            index=get_index(),
            measure_index=get_measure_index(),
            labels={},
            sources={},
            currency_units={},
            graph_generation=None,
            catalogue_build=None,
        )

    def current(self) -> Vocabulary:
        return self._vocabulary
