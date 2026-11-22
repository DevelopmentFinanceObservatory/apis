"""Query aliases for the common wordings of a group's name, generated from its own names.

Each group the registry holds gets one alias per name the resolver lists for it (canonical name,
aliases and abbreviations) and per wording in `WORDINGS`: "G7 countries", "members of the G7",
"all G-7 countries" and the like. An alias maps its text to the name it was built from, so a
wording resolves to the group whose name it contains and to nothing else; no other text is read
for a group's name.

A group's wordings share one reading, whichever of its names they use. It comes from the
curated query aliases "<name> countries", "<name> members" and "<name> member states" that
resolve to the group under any of its names: members (the members of a set, as "G20 countries"
reads) when those rows all read members, and the group itself, as its bare name reads, when they
all read the group or none exists. A group whose curated rows disagree reads as the group itself
and is listed in `GroupWordings.conflicts`.

A name is used only when its group is the top candidate past the listed-name gate: "AU" tops on
Australia, so no wording holds it. A name that already ends in a set noun ("DAC countries"),
carries a ", Total" suffix or ends in a bracketed qualifier ("Africa (DAC)") is not worded and
is recorded in `unnamed` with its reason. A generated text is dropped, and recorded in `skipped`,
when another entity lists it as a name or alias, when two groups generate it, when a curated
query alias already holds it (`query_aliases.json` stays authoritative), when resolvekit
reports an error on it, or when only its own group lists it and the group reads as itself (the
text resolves as the group already). A text only its own group lists is kept as an alias when
the group reads as members, so the reading applies.

Build on the thread that opened the resolver, since resolvekit's stores are thread-affine."""

from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)

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

SkipReason = Literal["curated", "listed", "shared", "self_listed", "error"]
UnnamedReason = Literal["gate", "set_noun", "qualifier"]

_READING_FORMS = ("{} countries", "{} members", "{} member states")


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
    two groups generate it (`owners` names the groups), `self_listed` when only its own group lists
    it and the group does not read as members, `error` when resolvekit reported an error on it."""

    text: str
    group: str
    reason: SkipReason
    owners: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Unnamed:
    """A group name left out of the wordings: `gate` when the name does not resolve to its
    group alone, `set_noun` when it ends in a set noun, `qualifier` when it carries a ", Total"
    suffix or a bracketed qualifier."""

    group: str
    name: str
    reason: UnnamedReason


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
    `unnamed` lists the names left out, each with its reason. `conflicts` maps a group whose
    curated rows disagree to those rows' keys; it reads as itself."""

    aliases: Mapping[str, Wording]
    skipped: tuple[Skipped, ...]
    unnamed: tuple[Unnamed, ...]
    conflicts: Mapping[str, tuple[str, ...]]


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


def _unworded_reason(name: str) -> UnnamedReason | None:
    """Why `name` cannot take a wording, or None when it can: it already ends in a set noun, a
    ", Total" suffix or a bracketed qualifier."""
    words = re.findall(r"\w+", name.casefold())
    if not words or words[-1] in _SET_NOUNS:
        return "set_noun"
    if _TRAILING_TOTAL_RE.search(name) or _TRAILING_QUALIFIER_RE.search(name):
        return "qualifier"
    return None


def _group_reading(
    resolver: Resolver,
    alias_index: AliasIndex,
    context: ResolutionContext,
    *,
    group: str,
    names: Iterable[str],
    curated: Mapping[str, QueryAlias],
) -> tuple[bool, tuple[str, ...]]:
    """The reading all of `group`'s wordings share, and the curated keys that disagree.

    Reads every curated "<name> countries", "<name> members" and "<name> member states" row,
    for any of the group's `names`, that resolves to `group`. Returns `(True, ())` when they all
    read members, `(False, ())` when they all read the group or none exists, and
    `(False, keys)` when they disagree."""
    rows: dict[str, QueryAlias] = {}
    for name in names:
        for form in _READING_FORMS:
            key = normalize_key(form.format(name))
            row = curated.get(key)
            if row is None or key in rows:
                continue
            listed = _listed_ids(resolver, alias_index, row.name, context)
            if listed and listed[0] == group:
                rows[key] = row
    readings = {row.members for row in rows.values()}
    if len(readings) > 1:
        return False, tuple(sorted(rows))
    return readings == {True}, ()


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
    unnamed: list[Unnamed] = []
    conflicts: dict[str, tuple[str, ...]] = {}
    for group in groups:
        names = tuple(_names(resolver, group.entity_id))
        worded: list[str] = []
        for name in names:
            reason = _unworded_reason(name)
            if reason is None:
                listed = _listed_ids(resolver, alias_index, name, context)
                if not listed or listed[0] != group.entity_id:
                    reason = "gate"
            if reason is None:
                worded.append(name)
            else:
                unnamed.append(Unnamed(group=group.entity_id, name=name, reason=reason))
        if not worded:
            continue
        members, disagreeing = _group_reading(
            resolver,
            alias_index,
            context,
            group=group.entity_id,
            names=names,
            curated=curated,
        )
        if disagreeing:
            conflicts[group.entity_id] = disagreeing
            logger.warning(
                "group %s: curated readings disagree (%s); its wordings read as the group",
                group.entity_id,
                ", ".join(disagreeing),
            )
        for name in worded:
            for wording in WORDINGS:
                owners = generated.setdefault(normalize_key(wording.format(name)), {})
                owners.setdefault(
                    group.entity_id,
                    Wording(name=name, group=group.entity_id, members=members),
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
        try:
            listed = _listed_ids(resolver, alias_index, text, context)
        except EntityResolutionError:
            logger.warning("resolvekit reported an error on generated text %r; skipped", text)
            skipped.append(Skipped(text=text, group=group, reason="error"))
            continue
        if not listed:
            aliases[text] = wording
        elif set(listed) != {group}:
            skipped.append(Skipped(text=text, group=group, reason="listed", owners=tuple(listed)))
        elif wording.members:
            aliases[text] = wording
        else:
            skipped.append(Skipped(text=text, group=group, reason="self_listed"))
    return GroupWordings(
        aliases=aliases,
        skipped=tuple(skipped),
        unnamed=tuple(unnamed),
        conflicts=conflicts,
    )
