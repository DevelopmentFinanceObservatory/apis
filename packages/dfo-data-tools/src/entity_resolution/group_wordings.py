"""Query aliases for the common wordings of a group's name, generated from its own names.

Each group the registry holds gets one alias per name the resolver lists for it (canonical name,
aliases and abbreviations) and per wording in `WORDINGS`: "G7 countries", "members of the G7",
"all G-7 countries" and the like. An alias maps its text to the name it was built from, so a
wording resolves to the group whose name it contains and to nothing else; no other text is read
for a group's name. A generated text takes the reading of the curated query alias that holds
the same name in "<name> countries", "<name> members" or "<name> member states" (the members
of a set, as "G20 countries" reads) where one resolves to the same group, and otherwise reads as
the group itself, as the group's bare name does.

A name is used only when its group is the top candidate past the listed-name gate: "AU" tops on
Australia, so no wording holds it. A name that already ends in a set noun ("DAC countries"),
carries a ", Total" suffix or ends in a bracketed qualifier ("Africa (DAC)") is not worded and
is recorded in `unnamed`. A generated text is dropped, and recorded in `skipped`, when another
entity lists it as a name or alias, when two groups generate it, or when a curated query alias
already holds it (`query_aliases.json` stays authoritative). A text only its own group lists
resolves as that group already and is neither an alias nor recorded.

Build on the thread that opened the resolver, since resolvekit's stores are thread-affine."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from resolvekit import ResolutionContext, ResolutionStatus

from entity_resolution import pack
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

MEMBER_WORDINGS: frozenset[str] = frozenset(wording for wording in WORDINGS if "member" in wording)
"""The wordings that name a set's members, which take a curated members reading."""

_SET_NOUNS = frozenset(
    {
        "countries",
        "members",
        "states",
        "nations",
        "economies",
        "donors",
        "creditors",
        "lenders",
        "entities",
        "agencies",
        "providers",
        "regions",
    }
)
_TRAILING_TOTAL_RE = re.compile(r",\s*total\s*$", re.IGNORECASE)
_TRAILING_QUALIFIER_RE = re.compile(r"[(\[][^)\]]*[)\]]\s*$")

SkipReason = Literal["curated", "listed", "shared"]


class EntityResolutionError(Exception):
    """Internal resolvekit failure for one text in a batch.

    Raised when resolvekit returns `ResolutionStatus.ERROR`."""


@dataclass(frozen=True, slots=True)
class QueryAlias:
    """The pack name a query text resolves as. `members` is true when the text names the
    members of the set that name resolves to, each as its own place."""

    name: str
    members: bool = False


def load_query_aliases() -> dict[str, QueryAlias]:
    """The curated query aliases (`query_aliases.json`), keyed by `normalize_key`."""
    rows = json.loads(pack.query_aliases_path().read_text(encoding="utf-8"))
    return {key: QueryAlias(**row) for key, row in rows.items()}


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
class Wording:
    """What a generated text resolves as: the group's `name` it was built from, the `group`
    entity id, and whether it reads as the group's members."""

    name: str
    group: str
    members: bool


@dataclass(frozen=True, slots=True)
class GroupWordings:
    """`aliases` maps each normalized generated text to the `Wording` it resolves as.
    `unnamed` lists the names left out: those that do not resolve to their group alone and
    those that already carry a set noun or qualifier."""

    aliases: Mapping[str, Wording]
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
    if result.status == ResolutionStatus.ERROR:
        raise EntityResolutionError(f"resolvekit reported an internal error resolving {text!r}")
    candidates = alias_index.with_owners(result.candidates, text)
    return [
        candidate.entity_id for candidate, _ in filter_listed_candidates(resolver, text, candidates)
    ]


def _names(resolver: Resolver, entity_id: str) -> Iterable[str]:
    record = resolver.entity(entity_id)
    if record is None:
        return ()
    return dict.fromkeys([record.canonical_name, *(name.value for name in record.names)])


def _is_worded(name: str) -> bool:
    """Whether `name` can take a wording: it does not already end in a set noun, a ", Total"
    suffix or a bracketed qualifier."""
    words = re.findall(r"\w+", name.casefold())
    if not words or words[-1] in _SET_NOUNS:
        return False
    return not (_TRAILING_TOTAL_RE.search(name) or _TRAILING_QUALIFIER_RE.search(name))


def _members_reading(
    resolver: Resolver,
    alias_index: AliasIndex,
    context: ResolutionContext,
    *,
    group: str,
    name: str,
    wording: str,
    curated: Mapping[str, QueryAlias],
) -> bool:
    """Whether `wording` of `name` reads as members: the curated "<name> members", "<name>
    member states" or "<name> countries" row (a member wording tries them in that order, any
    other only "<name> countries") that resolves to `group` states it."""
    forms = (
        ("{} members", "{} member states", "{} countries")
        if wording in MEMBER_WORDINGS
        else ("{} countries",)
    )
    for form in forms:
        row = curated.get(normalize_key(form.format(name)))
        if row is None:
            continue
        listed = _listed_ids(resolver, alias_index, row.name, context)
        if listed and listed[0] == group:
            return row.members
    return False


def build_group_wordings(
    resolver: Resolver,
    alias_index: AliasIndex,
    groups: GroupRegistry,
    *,
    curated: Mapping[str, QueryAlias],
) -> GroupWordings:
    """Generate the wording aliases of every group in `groups`.

    `curated` holds the curated query aliases by normalized key; they win over a generated
    text and give a generated text its reading."""
    context = ResolutionContext(entity_types=ADMITTED_TYPES)
    generated: dict[str, dict[str, Wording]] = {}
    unnamed: list[tuple[str, str]] = []
    for group in groups:
        for name in _names(resolver, group.entity_id):
            if not _is_worded(name):
                unnamed.append((group.entity_id, name))
                continue
            listed = _listed_ids(resolver, alias_index, name, context)
            if not listed or listed[0] != group.entity_id:
                unnamed.append((group.entity_id, name))
                continue
            for wording in WORDINGS:
                owners = generated.setdefault(normalize_key(wording.format(name)), {})
                if group.entity_id not in owners:
                    members = _members_reading(
                        resolver,
                        alias_index,
                        context,
                        group=group.entity_id,
                        name=name,
                        wording=wording,
                        curated=curated,
                    )
                    owners[group.entity_id] = Wording(
                        name=name, group=group.entity_id, members=members
                    )

    aliases: dict[str, Wording] = {}
    skipped: list[Skipped] = []
    for text, owners in sorted(generated.items()):
        if len(owners) > 1:
            skipped.extend(
                Skipped(text=text, group=group, reason="shared", owners=tuple(sorted(owners)))
                for group in sorted(owners)
            )
            continue
        ((group, wording),) = owners.items()
        if text in curated:
            skipped.append(Skipped(text=text, group=group, reason="curated"))
            continue
        listed = _listed_ids(resolver, alias_index, text, context)
        if not listed:
            aliases[text] = wording
        elif set(listed) != {group}:
            skipped.append(Skipped(text=text, group=group, reason="listed", owners=tuple(listed)))
    return GroupWordings(aliases=aliases, skipped=tuple(skipped), unnamed=tuple(unnamed))
