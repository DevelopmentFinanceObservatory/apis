"""The words of a part's variable that an organisation it binds as the origin of its flow
owns.

"IDA disbursements" with IDA the provider holds "IDA", which the code index also reads as
DAC1 aid type 2103 ("multilateral contribution to the International Development
Association"). The word is the provider's, so no reader of the variable takes it as a code:
recall searches the graph on the text without it too (`without_owned_words`) and pulls in no
family for it (`interpret.recall`), and the bind phase neither elects,
binds nor refuses on it (`phase_rank_bind._without_owned_terms`). Both read the owned words
through `owned_ranges`, from the request's own words for the organisation. Only a
door whose caller states each entity's role marks the owned words
(`origin_organisations`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from code_index.tokens import tokenize, word_tokens
from entity_resolution.types import Kind
from search.engine.qualifier_match import token_occurrences, with_role_preposition
from search.engine.role_text import DoorRole
from search.models import EntryPath

type Ranges = tuple[tuple[int, int], ...]
"""Character ranges of a text, each `(start, end)`."""


def origin_organisations(
    nodes: Mapping[str, tuple[str, Kind | None]],
    entity_roles: Mapping[str, Sequence[DoorRole]],
    *,
    door: EntryPath,
) -> dict[str, str]:
    """The dcid of each organisation of `nodes` (the resolved dcid and kind of each name the
    part gives) that the part binds as the origin of its flow, by the request's name for it:
    every role the door gives it (`entity_roles`) is on the origin axis.

    The parsed door, and the dataset_resubmit door's parsed entries, take each role from the
    caller. The raw-text door names none: its roles are the extraction's reading of the
    request, and its source choice reads the variable before any name resolves
    (`source_choice.read_organisations`), so a part it posts reads every word of its
    variable."""
    if door == "raw_text":
        return {}
    organisations: dict[str, str] = {}
    for name, (dcid, kind) in nodes.items():
        if kind != "organisation":
            continue
        roles = entity_roles.get(name, ())
        if roles and all(role.axis == "origin" for role in roles):
            organisations[name] = dcid
    return organisations


def owned_ranges(
    text: str,
    organisations: Iterable[str],
    *,
    entity_words: Mapping[str, str],
    entity_roles: Mapping[str, Sequence[DoorRole]],
) -> Ranges:
    """The character ranges of `text` that the named `organisations` own: each occurrence
    of an organisation's name, the request's words for it (`entity_words`) or its role
    spans (`DoorRole.span`)."""
    tokens = tokenize(text)
    return tuple(
        occurrence
        for name in organisations
        for words in (
            name,
            entity_words.get(name),
            *(role.span for role in entity_roles.get(name, ())),
        )
        if words
        for occurrence in token_occurrences(tokens, words)
    )


def owns(text: str, owned: Ranges, start: int, end: int) -> bool:
    """Whether `text[start:end]` holds a word and `owned` covers every one."""
    words = word_tokens(text[start:end])
    return bool(words) and all(
        any(low <= start + word.start and start + word.end <= high for low, high in owned)
        for word in words
    )


def without_owned_words(
    text: str,
    organisations: Iterable[str],
    *,
    entity_words: Mapping[str, str],
    entity_roles: Mapping[str, Sequence[DoorRole]],
) -> str | None:
    """`text` less the words the named `organisations` own (`owned_ranges`), each with the
    role preposition and article before it (`qualifier_match.with_role_preposition`), its
    words joined by single spaces: "disbursements" for "IDA disbursements" or
    "disbursements from IDA" with IDA the provider. `None` where they own no word of
    `text` or leave it no word."""
    owned = owned_ranges(text, organisations, entity_words=entity_words, entity_roles=entity_roles)
    if not owned:
        return None
    widened = with_role_preposition(owned, tokenize(text))
    kept = "".join(
        char
        for position, char in enumerate(text)
        if not any(low <= position < high for low, high in widened)
    )
    rest = " ".join(kept.split())
    return rest if word_tokens(rest) else None
