"""Compile a part's source mentions into the provenances it may answer from.

A mention names a source by its words or by a reference. Words match whole against a
dictionary built from the origin catalogue's admitted rows: each provenance's name, its
Dataset's label and its Source's label, each name also without its parenthesised parts,
a parenthesised abbreviation whose letters are the initials of the capitalised words
before it ("World Bank International Debt Statistics (IDS)" yields "IDS"; "ONE
Conversion Tables (IMF)" yields no "IMF"), and the registry's `[[alias]]` names. An
alias name replaces every key the catalogue derives from the same words. Matching
ignores case, punctuation and a leading "the".

A mention resolves to one reference: a provenance, a Dataset, a Source, or a collection
that an alias of several references names. Words that match several references resolve
to the one that contains all the others, as a Source contains its Datasets and
provenances; otherwise they are ambiguous. A provenance expands to itself. A Dataset or
Source expands to its admitted provenances, and an included one leaves out the code index's
opt-in products, unless it publishes nothing else. A collection expands to the union of
its members' expansions.

The part's scope is the union of its included expansions, or every admitted provenance
when it includes none, less the union of its excluded expansions. `named` records
whether it includes any. An excluded mention whose words match nothing removes nothing:
it stays in the scope's mentions with no reference and an empty expansion, so the part
warns that the exclusion was not applied. Included words that match nothing refuse.
"""

from __future__ import annotations

import functools
import logging
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from code_index.query import CodeIndex
from code_index.shared import get_index, per_index
from origin_catalogue import Catalogue, ProvenanceRow, current_catalogue
from origin_catalogue.registry import Alias, load_registry
from search.engine.errors import EngineInputError
from search.models import ParsedPart, SourceMention, SourceRefKind

logger = logging.getLogger(__name__)

type SourceRefusalReason = Literal["source_not_resolved", "source_ambiguous"]


@dataclass(frozen=True, slots=True)
class SourceReference:
    """An admitted source a mention can name, with its catalogue label."""

    ref: str
    label: str
    kind: SourceRefKind


@dataclass(frozen=True, slots=True)
class SourceCollection:
    """The admitted references an alias of several refs names together, labelled with
    the alias's first name. It has no ref of its own, so a client cannot post one back,
    and it is not an ambiguous candidate, since an alias name replaces the catalogue's keys."""

    label: str
    members: tuple[SourceReference, ...]


type NamedSource = SourceReference | SourceCollection


@dataclass(frozen=True, slots=True)
class SourceScope:
    """The provenances a part may answer from.

    `named` is true when the part includes a source, so election may serve an opt-in
    product the scope allows. `mentions` are the part's mentions in posted order, each
    carrying the ref it resolved to (none for a collection, which keeps its words, or for
    an excluded mention whose words match nothing), `references` what they resolved to
    (`None` for such an excluded mention), and `expansions` the provenances each stands
    for under its polarity, empty for an excluded mention that resolved to nothing. `idle` holds the
    positions of the excluded mentions that remove nothing from the shapes the part
    recalled (`region.source_choice.with_idle_exclusions`), empty until candidacy reads
    them, and of the included mentions that serve nothing of the part's answer
    (`region.source_choice.with_idle_inclusions`), marked once the answer is read."""

    allowed: frozenset[str]
    named: bool
    mentions: tuple[SourceMention, ...] = ()
    references: tuple[NamedSource | None, ...] = ()
    expansions: tuple[frozenset[str], ...] = ()
    idle: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class AmbiguousSource:
    """A mention's words that match two or more unrelated references."""

    text: str
    candidates: tuple[SourceReference, ...]


@dataclass(frozen=True, slots=True)
class SourceRefusal:
    """A part whose source words did not resolve, so it answers no_data with `reason`.

    `terms` are the unmatched included words for `source_not_resolved` and the ambiguous
    ones for `source_ambiguous`, in mention order. `ambiguous` pairs each ambiguous
    wording with its candidates. `mentions` are the part's mentions with every resolved
    one carrying its ref, for the part a client posts back with a candidate chosen."""

    reason: SourceRefusalReason
    terms: tuple[str, ...]
    ambiguous: tuple[AmbiguousSource, ...]
    mentions: tuple[SourceMention, ...]


@dataclass(frozen=True, slots=True)
class SourceDictionary:
    """The references one set of admitted rows offers, keyed for matching.

    `keys` maps a normalised name to its references in catalogue order, or to the
    collection an alias gives it. `provenances` maps every reference to its admitted
    provenances, `parents` a reference to the references containing it, and `every`
    holds every admitted provenance."""

    references: Mapping[str, SourceReference]
    keys: Mapping[str, tuple[SourceReference, ...] | SourceCollection]
    provenances: Mapping[str, tuple[str, ...]]
    parents: Mapping[str, frozenset[str]]
    every: frozenset[str]


def source_label(ref: str) -> str:
    """The label of the source `ref`: its code-index dataset label, else its origin
    catalogue name, else the ref."""
    if (dataset := get_index().datasets.get(ref)) is not None:
        return dataset.label
    row = current_catalogue().provenance(ref)
    return row.name if row is not None else ref


def normalise_source_text(text: str) -> str:
    """Casefold `text`, read punctuation as spaces, drop a leading "the" and collapse
    whitespace."""
    folded = "".join(
        " " if unicodedata.category(char).startswith("P") else char for char in text.casefold()
    )
    words = folded.split()
    if len(words) > 1 and words[0] == "the":
        words = words[1:]
    return " ".join(words)


def without_source_words(variable: str, scope: SourceScope) -> str:
    """`variable` less each run of words that repeats the words of a mention of `scope`
    that resolved to a source, compared as `normalise_source_text` reads them: "DAC2a ODA"
    under the mention "DAC2a" reads "ODA". Words read as a source are not the measure's
    too, so recall and vetting read the measure's own words alone. A variable made only of
    source words stays as it is."""
    rest = variable
    for mention, named in zip(scope.mentions, scope.references, strict=True):
        words = normalise_source_text(mention.text or "").split()
        if named is None or not words:
            continue
        pattern = r"(?<!\w)" + r"[\W_]+".join(map(re.escape, words)) + r"(?!\w)"
        rest = re.sub(pattern, " ", rest, flags=re.IGNORECASE)
    return " ".join(rest.split()) or variable


_PARENTHESISED = re.compile(r"\(([^()]*)\)")


def _initials(text: str) -> list[str]:
    """The first letter of each word of `text` that starts with a capital."""
    initials = []
    for word in text.split():
        word = word.lstrip("\"'([{")
        if word[:1].isupper():
            initials.append(word[0])
    return initials


def _name_keys(name: str) -> list[str]:
    """The keys a catalogue name yields: the name, the name without its parenthesised
    parts, and each parenthesised abbreviation whose letters are the initials of the
    capitalised words just before it."""
    keys = [name, _PARENTHESISED.sub(" ", name)]
    for match in _PARENTHESISED.finditer(name):
        letters = [char.upper() for char in match.group(1) if char.isalnum()]
        initials = _initials(name[: match.start()])
        if letters and initials[-len(letters) :] == letters:
            keys.append(match.group(1))
    return [key for key in dict.fromkeys(normalise_source_text(k) for k in keys) if key]


@functools.cache
def _aliases() -> tuple[Alias, ...]:
    return load_registry().alias


@per_index
def _opt_in_products(index: CodeIndex) -> frozenset[str]:
    """The provenances `index` declares opt-in."""
    return frozenset(ref for ref, dataset in index.datasets.items() if dataset.opt_in)


_DICTIONARIES_KEPT = 4
_dictionaries: dict[tuple[str, bool], SourceDictionary] = {}
"""The dictionaries of the catalogues compiled against most recently, keyed by build id and
whether the UN origin is served, the two inputs of a catalogue's admitted rows."""


def source_dictionary(catalogue: Catalogue) -> SourceDictionary:
    """The dictionary of `catalogue`'s admitted rows (`build_source_dictionary`), built
    once per build and UN setting. A scope leaves the rows unchanged."""
    key = (catalogue.build_id, catalogue.un_system_enabled)
    dictionary = _dictionaries.get(key)
    if dictionary is None:
        dictionary = build_source_dictionary(catalogue.provenance_rows())
        if len(_dictionaries) >= _DICTIONARIES_KEPT:
            del _dictionaries[next(iter(_dictionaries))]
        _dictionaries[key] = dictionary
    return dictionary


def clear_source_dictionaries() -> None:
    """Forget every dictionary `source_dictionary` built."""
    _dictionaries.clear()


def build_source_dictionary(rows: Sequence[ProvenanceRow]) -> SourceDictionary:
    """The dictionary of `rows`, a catalogue's admitted provenance rows, and the
    registry's aliases. An alias keeps the refs that name one of the rows, and is skipped
    when none does; one ref left names that reference, several a collection."""
    references: dict[str, SourceReference] = {}
    provenances: dict[str, list[str]] = {}
    parents: dict[str, set[str]] = {}
    names: list[tuple[str, str]] = []

    def add(ref: str, label: str, kind: SourceRefKind, provenance: str) -> None:
        if ref not in references:
            references[ref] = SourceReference(ref=ref, label=label, kind=kind)
            names.append((ref, label))
        provenances.setdefault(ref, []).append(provenance)

    for row in rows:
        add(row.id, row.name, "provenance", row.id)
    for row in rows:
        containers = [row.source.ref]
        if row.dataset is not None:
            add(row.dataset.ref, row.dataset.label, "dataset", row.id)
            containers.append(row.dataset.ref)
            parents.setdefault(row.dataset.ref, set()).add(row.source.ref)
        if row.source.ref != row.id:
            add(row.source.ref, row.source.label, "source", row.id)
        parents.setdefault(row.id, set()).update(ref for ref in containers if ref != row.id)

    catalogue_keys: dict[str, list[SourceReference]] = {}
    for ref, label in names:
        for key in _name_keys(label):
            catalogue_keys.setdefault(key, []).append(references[ref])
    keys: dict[str, tuple[SourceReference, ...] | SourceCollection] = {
        key: tuple(refs) for key, refs in catalogue_keys.items()
    }
    for alias in _aliases():
        members = tuple(references[ref] for ref in alias.refs if ref in references)
        if not members:
            logger.info("source alias skipped: none of %s is admitted", alias.refs)
            continue
        named = members if len(members) == 1 else SourceCollection(alias.names[0], members)
        for name in alias.names:
            keys[normalise_source_text(name)] = named

    return SourceDictionary(
        references=references,
        keys=keys,
        provenances={ref: tuple(dict.fromkeys(ids)) for ref, ids in provenances.items()},
        parents={ref: frozenset(refs) for ref, refs in parents.items()},
        every=frozenset(row.id for row in rows),
    )


def match_source_words(
    dictionary: SourceDictionary, text: str
) -> tuple[SourceReference, ...] | SourceCollection:
    """What `text` names: a collection, or the references it matches, narrowed to the
    one that contains all the others when there is one."""
    named = dictionary.keys.get(normalise_source_text(text), ())
    if isinstance(named, SourceCollection):
        return named
    for reference in named:
        if all(
            reference.ref in dictionary.parents.get(other.ref, ())
            for other in named
            if other != reference
        ):
            return (reference,)
    return named


def _expand(dictionary: SourceDictionary, named: NamedSource, *, include: bool) -> frozenset[str]:
    """The provenances `named` stands for. An included Dataset or Source leaves out its
    opt-in products unless it publishes nothing else."""
    if isinstance(named, SourceCollection):
        return frozenset().union(
            *(_expand(dictionary, member, include=include) for member in named.members)
        )
    reference = named
    provenances = dictionary.provenances.get(reference.ref, (reference.ref,))
    if not include or reference.kind == "provenance":
        return frozenset(provenances)
    opt_in = _opt_in_products(get_index())
    default = frozenset(p for p in provenances if p not in opt_in)
    return default or frozenset(provenances)


def _pinned_reference(dictionary: SourceDictionary, ref: str, *, pointer: str) -> SourceReference:
    """The reference a mention's posted `ref` names: one of the catalogue's admitted rows,
    their Datasets and Sources (`SourceDictionary.references`). Raises `request_invalid` at
    `pointer` for any other."""
    if (reference := dictionary.references.get(ref)) is not None:
        return reference
    raise EngineInputError(
        f"source ref {ref!r} names no provenance, dataset or source the origin catalogue admits",
        code="request_invalid",
        pointer=pointer,
    )


def compile_sources(
    mentions: Sequence[SourceMention],
    *,
    catalogue: Catalogue,
    pointers: Sequence[str],
    refuse_empty: bool = True,
) -> SourceScope | SourceRefusal | None:
    """Compile `mentions` against `catalogue`'s admitted sources.

    `pointers` gives each mention's JSON pointer. Returns `None` for no mentions, a
    `SourceRefusal` when included words match no source or any words match several
    unrelated ones, else the scope. Excluded words that match no source resolve to no
    reference and remove nothing. Raises `request_invalid` at a mention whose ref is not
    admitted, and, when `refuse_empty`, at the excluded mention that leaves the scope
    empty. Without `refuse_empty` such a scope compiles allowing nothing, and its part
    answers `qualifier_not_bound` (`region.result.empty_scope_result`), since the raw-text
    door reads mentions an extraction model wrote, which the caller cannot correct."""
    if not mentions:
        return None
    dictionary = source_dictionary(catalogue)

    resolved: list[NamedSource | None] = []
    unmatched: list[str] = []
    ambiguous: list[AmbiguousSource] = []
    for mention, pointer in zip(mentions, pointers, strict=True):
        if mention.ref is not None:
            resolved.append(_pinned_reference(dictionary, mention.ref, pointer=pointer))
            continue
        text = mention.text or ""
        matches = match_source_words(dictionary, text)
        if isinstance(matches, SourceCollection):
            resolved.append(matches)
            continue
        if len(matches) == 1:
            resolved.append(matches[0])
            continue
        resolved.append(None)
        if matches:
            ambiguous.append(AmbiguousSource(text=text, candidates=matches))
        elif mention.polarity == "include":
            unmatched.append(text)

    pinned = tuple(
        SourceMention(text=mention.text, ref=named.ref, polarity=mention.polarity)
        if isinstance(named, SourceReference)
        else mention
        for mention, named in zip(mentions, resolved, strict=True)
    )
    if unmatched:
        return SourceRefusal(
            reason="source_not_resolved", terms=tuple(unmatched), ambiguous=(), mentions=pinned
        )
    if ambiguous:
        return SourceRefusal(
            reason="source_ambiguous",
            terms=tuple(group.text for group in ambiguous),
            ambiguous=tuple(ambiguous),
            mentions=pinned,
        )

    references = tuple(resolved)
    expansions = tuple(
        frozenset()
        if reference is None
        else _expand(dictionary, reference, include=mention.polarity == "include")
        for mention, reference in zip(mentions, references, strict=True)
    )
    included = [
        expansion
        for mention, expansion in zip(mentions, expansions, strict=True)
        if mention.polarity == "include"
    ]
    named = bool(included)
    allowed = frozenset().union(*included) if named else dictionary.every
    for mention, reference, expansion, pointer in zip(
        mentions, references, expansions, pointers, strict=True
    ):
        if mention.polarity != "exclude" or reference is None:
            continue
        allowed -= expansion
        if not allowed and refuse_empty:
            raise EngineInputError(
                f"excluding {reference.label!r} leaves no source this part may answer from",
                code="request_invalid",
                pointer=pointer,
            )
    return SourceScope(
        allowed=allowed,
        named=named,
        mentions=pinned,
        references=references,
        expansions=expansions,
    )


def compile_part_sources(
    part: ParsedPart, *, catalogue: Catalogue, pointer_prefix: str
) -> SourceScope | SourceRefusal | None:
    """Compile a parsed part's mentions (`compile_sources`), pointing a refusal at the
    field each mention was posted in (`ParsedPart.mention_field`)."""
    pointers = [f"{pointer_prefix}/{part.mention_field(i)}" for i in range(len(part.sources))]
    return compile_sources(part.sources, catalogue=catalogue, pointers=pointers)
