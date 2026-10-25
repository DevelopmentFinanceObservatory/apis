"""The names a door gives the places one part mentions.

A place mention is identified by its text and the flow axis the request gives it. A text
mentioned on at most one axis in a part keeps the text as its name. A text mentioned on
both axes ("aid from Congo to Congo") names two places, and gets one name per axis. The
door invents those names as opaque keys of the engine's state, so everything the request
reads or shows of such a place goes through its text (`DoorNames.texts`), and no invented
name reaches the resolver, a sentence or the wire.

An undirected mention of a text names every place the text names in the part. Both doors
name their places by this one rule, and option building reads a part's parsed form back
through it (`part_names`), so a name a door gave a mention is the name its echoed row
reads back as.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from search.engine.role_text import DoorRole, role_axis_for_text
from search.models import ParsedEntity, RoleAxis

# Joins a text to its axis in an invented name. A control character, so no request wording
# can equal the name.
NAME_SEPARATOR = "\x1f"


@dataclass(frozen=True, slots=True)
class DoorNames:
    """The names of one part's places, in the order the part first mentions them.

    `roles` holds each directed name's roles. `texts` maps every name to the text it names
    a place of: an invented name to its text, any other name to itself. `split` maps each
    text mentioned on both axes to its name per axis."""

    names: tuple[str, ...]
    roles: Mapping[str, list[DoorRole]]
    texts: Mapping[str, str]
    split: Mapping[str, Mapping[RoleAxis, str]]

    def of(self, text: str, axis: RoleAxis | None) -> tuple[str, ...]:
        """The names a mention of `text` on `axis` names: its axis's name for a text on
        both axes, every name of that text for an undirected mention, else the text."""
        by_axis = self.split.get(text)
        if by_axis is None:
            return (text,)
        if axis is None:
            return tuple(by_axis.values())
        return (by_axis[axis],)

    def text_of(self, name: str) -> str:
        """The text `name` names a place of. A name that is not one of `names` raises
        `KeyError`."""
        return self.texts[name]


def door_names(texts: Iterable[str], roles: Mapping[str, Sequence[DoorRole]]) -> DoorNames:
    """Name the places of a part that mentions `texts`, in order, where `roles` holds each
    text's directed mentions in order.

    A text whose roles hold both axes gets one invented name per axis, holding that
    axis's roles, in the order its roles first state each axis. An invented name equal
    to one of `texts` takes another separator, so no name is two places."""
    ordered = list(dict.fromkeys(texts))
    taken = set(ordered)
    names: list[str] = []
    named_roles: dict[str, list[DoorRole]] = {}
    texts_of: dict[str, str] = {}
    split: dict[str, dict[RoleAxis, str]] = {}
    for text in ordered:
        own = list(roles.get(text) or ())
        axes = list(dict.fromkeys(role.axis for role in own))
        if len(axes) < 2:
            names.append(text)
            texts_of[text] = text
            if own:
                named_roles[text] = own
            continue
        by_axis: dict[RoleAxis, str] = {}
        for axis in axes:
            name = f"{text}{NAME_SEPARATOR}{axis}"
            while name in taken:
                name += NAME_SEPARATOR
            taken.add(name)
            by_axis[axis] = name
            names.append(name)
            named_roles[name] = [role for role in own if role.axis == axis]
            texts_of[name] = text
        split[text] = by_axis
    return DoorNames(names=tuple(names), roles=named_roles, texts=texts_of, split=split)


def own_texts(names: Iterable[str]) -> dict[str, str]:
    """The texts of places named by `names` when no door invented any: each name is its own
    text. A caller below the doors that posts its names directly states them by this."""
    return {name: name for name in names}


def mention_roles(entities: Sequence[ParsedEntity] | None) -> dict[str, list[DoorRole]]:
    """Each text of a parsed part's `entities` rows with its directed mentions, in mention
    order, as the parsed door reads them: a row's `role_text` gives its axis, and an
    unrecognised word or none leaves the row undirected, with no role. Each distinct axis
    of a text is one role, its span the text, so a text repeated on opposite axes holds
    both."""
    roles: dict[str, list[DoorRole]] = {}
    for entity in entities or ():
        axis = role_axis_for_text(entity.role_text)
        if axis is None:
            continue
        occurrences = roles.setdefault(entity.text, [])
        role = DoorRole(axis=axis, span=entity.text)
        if role not in occurrences:
            occurrences.append(role)
    return roles


def part_names(entities: Sequence[ParsedEntity] | None) -> DoorNames:
    """The names of a parsed part's places, its `entities` rows read by `mention_roles`."""
    return door_names((entity.text for entity in entities or ()), mention_roles(entities))
