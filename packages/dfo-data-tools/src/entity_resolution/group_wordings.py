"""Query aliases for the common wordings of a group's name, generated from its own names.

Each group the registry holds gets one alias per name the resolver lists for it (canonical name,
aliases and abbreviations) and per wording in `WORDINGS`: "G7 countries", "members of the G7",
"all G-7 countries" and the like. An alias maps its text to the name it was built from, so a
wording resolves to the group whose name it contains and to nothing else; no other text is read
for a group's name. A generated text reads as the group itself, as the group's bare name does;
a curated row states its own reading.

A name is used only when its group is the top candidate past the listed-name gate: "AU" tops on
Australia, so no wording holds it. A generated text is dropped, and recorded in `skipped`, when another
entity lists it as a name or alias, when two groups generate it, or when a curated query alias
already holds it (`query_aliases.json` stays authoritative).

Build on the thread that opened the resolver, since resolvekit's stores are thread-affine."""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from resolvekit import ResolutionContext

from entity_resolution.gate import filter_listed_candidates
from entity_resolution.types import ADMITTED_TYPES

if TYPE_CHECKING:
    from resolvekit import Resolver

    from entity_resolution.groups import GroupRegistry
    from entity_resolution.pack import AliasIndex

WORDINGS: tuple[str, ...] = (
    "{} countries",
    "{} members",
    "{} member countries",
    "{} member states",
    "{} nations",
    "{} economies",
    "{} donors",
    "{} donor countries",
    "{} creditors",
    "{} lenders",
    "all {} countries",
    "all {} members",
    "members of the {}",
    "members of {}",
    "member states of the {}",
    "member countries of the {}",
    "countries in the {}",
    "countries in {}",
)
"""The wordings each group name is put in; `{}` stands for the name."""

SkipReason = Literal["curated", "listed", "shared"]


@dataclass(frozen=True, slots=True)
class Skipped:
    """A generated text left out of the table: `curated` when `query_aliases.json` holds it,
    `listed` when an entity lists it as a name or alias (`owners` names them), `shared` when
    two groups generate it (`owners` names the groups)."""

    text: str
    group: str
    reason: SkipReason
    owners: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GroupWordings:
    """`aliases` maps each normalized generated text to the group name it resolves as.
    `unnamed` lists the names left out because they do not resolve to their group alone."""

    aliases: Mapping[str, str]
    skipped: tuple[Skipped, ...]
    unnamed: tuple[tuple[str, str], ...]


def normalize_key(text: str) -> str:
    """The key a query alias is stored and looked up under: casefolded, whitespace collapsed."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _listed_ids(
    resolver: Resolver, alias_index: AliasIndex, text: str, context: ResolutionContext
) -> list[str]:
    """The entity ids that list `text`, ordered as the resolver ranks its survivors."""
    result = resolver.resolve(text, as_result=True, context=context, include_entity=True)
    candidates = alias_index.with_owners(result.candidates, text)
    return [
        candidate.entity_id for candidate, _ in filter_listed_candidates(resolver, text, candidates)
    ]


def _names(resolver: Resolver, entity_id: str) -> Iterable[str]:
    record = resolver.entity(entity_id)
    if record is None:
        return ()
    return dict.fromkeys([record.canonical_name, *(name.value for name in record.names)])


def build_group_wordings(
    resolver: Resolver,
    alias_index: AliasIndex,
    groups: GroupRegistry,
    *,
    curated: Collection[str],
) -> GroupWordings:
    """Generate the wording aliases of every group in `groups`.

    `curated` holds the normalized keys of the curated query aliases, which win over a generated
    text."""
    context = ResolutionContext(entity_types=ADMITTED_TYPES)
    generated: dict[str, dict[str, str]] = {}
    unnamed: list[tuple[str, str]] = []
    for group in groups:
        for name in _names(resolver, group.entity_id):
            listed = _listed_ids(resolver, alias_index, name, context)
            if not listed or listed[0] != group.entity_id:
                unnamed.append((group.entity_id, name))
                continue
            for wording in WORDINGS:
                owners = generated.setdefault(normalize_key(wording.format(name)), {})
                owners.setdefault(group.entity_id, name)

    aliases: dict[str, str] = {}
    skipped: list[Skipped] = []
    for text, owners in sorted(generated.items()):
        if len(owners) > 1:
            skipped.extend(
                Skipped(text=text, group=group, reason="shared", owners=tuple(sorted(owners)))
                for group in sorted(owners)
            )
            continue
        ((group, name),) = owners.items()
        if text in curated:
            skipped.append(Skipped(text=text, group=group, reason="curated"))
        elif listed := _listed_ids(resolver, alias_index, text, context):
            skipped.append(Skipped(text=text, group=group, reason="listed", owners=tuple(listed)))
        else:
            aliases[text] = name
    return GroupWordings(aliases=aliases, skipped=tuple(skipped), unnamed=tuple(unnamed))
