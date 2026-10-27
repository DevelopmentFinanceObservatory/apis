"""The sector axis: how a part's CRS purpose codes were chosen, and the codes it could bind
instead.

A sector concept the text names binds the codes its declared grouping reads it as
(`code_index.scope_conventions.SECTOR_GROUPINGS`): "health" binds CRS 120 and 130, ONE's
analytical grouping, where the code row the word matches is CRS 120 alone. A qualifier match
reads a concept when its phrase is one of the grouping's declared words on the grouping's
code system, and nothing else does: "basic health" (122) and "reproductive health" (130)
match codes of their own. `read_sector` is the binder's reading: it applies an explicit or
carried entry made for the concept over the default grouping, and records the axis. Every
other text-recognised sector binding is an explicit code the text typed or named, recorded
with its words.

`sector_rule` is the axis rule. It reads the base identity alone, so a dataset_resubmit,
where no binder runs, offers what the parsed door offered: a part binding a grouping, or one
code of it, offers the grouping and each of its codes alone; any other bound sector offers
its own codes and nothing else. The held option comes first in the preference, so a part's
derived default is the codes it resolved, and a switch is the only thing that moves them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from code_index.members import identity_family_query, query_family
from code_index.query import CodeIndex
from code_index.scope_conventions import OPTION_LABELS, SECTOR_GROUPINGS, SectorGrouping
from search.models import (
    BaseIdentity,
    Entry,
    IdentityBindingSet,
    IdentityBindingValue,
    Option,
    OptionAvailability,
    SlotBinding,
)
from search.scope.composite import table_document
from search.scope.state import AxisRecord, ScopeState
from search.scope.tables import AxisTable, ScopeFacts, allowed_values
from search.scope.vocabulary import sector_option_id

if TYPE_CHECKING:
    from search.engine.qualifier_match import QualifierMatch

SECTOR_PROPERTIES: frozenset[str] = frozenset(g.code_system for g in SECTOR_GROUPINGS)
"""The code systems the sector axis binds: those the declared groupings bind."""

GROUPING_RULE = "sector_grouping"
"""The rule of a part binding a declared grouping or one of its codes."""
CODE_RULE = "sector_code"
"""The rule of a part binding sector codes no grouping holds."""

_AVAILABLE = OptionAvailability(status="available")
_NOT_COVERED = OptionAvailability(status="unavailable", code="option_not_covered")


def concept_grouping(code_system: str, phrase: str) -> SectorGrouping | None:
    """The grouping a qualifier match on `code_system` whose phrase is `phrase` reads as,
    None when `phrase` is no declared word of a grouping on that system."""
    folded = phrase.casefold()
    return next(
        (g for g in SECTOR_GROUPINGS if g.code_system == code_system and folded in g.words),
        None,
    )


def grouping_options(grouping: SectorGrouping) -> dict[str, tuple[str, ...]]:
    """The options a grouping offers, by option id, each with the codes it binds: the
    grouping, then each of its codes alone."""
    return {
        sector_option_id(grouping.codes): grouping.codes,
        **{sector_option_id((code,)): (code,) for code in grouping.codes},
    }


@dataclass(frozen=True, kw_only=True)
class SectorReading:
    """What the binder reads of one text-recognised sector binding.

    `codes` are the codes the binding binds, `grouped` whether a sector concept bound them
    through its grouping, and `record` the axis record the binder writes."""

    codes: tuple[str, ...]
    grouped: bool
    record: AxisRecord


def read_sector(
    *,
    codes: Sequence[str],
    matches: Sequence[QualifierMatch],
    variable: str,
    scope: ScopeState,
) -> SectorReading:
    """The reading of a sector binding to `codes` that the qualifier `matches` on its code
    system forced, read from the part's `variable`.

    A concept named alone ("health aid") binds the option of the first entry that applies to
    it (`ScopeState.entries_on`, `_applies`), else the grouping itself, the default. A
    concept named beside other codes ("health and education aid") binds the grouping's
    codes beside theirs, by default. Codes no concept names are an explicit entry of the
    words that named them, at their place in the text the part's scope was read from
    (`ScopeState.origin_of`)."""
    named = {
        match.code_dcid: grouping
        for match in matches
        if match.code_dcid in codes
        and (grouping := concept_grouping(match.code_system, match.phrase)) is not None
    }
    if not named:
        entry = Entry(
            axis="sector",
            option_id=sector_option_id(codes),
            provenance="explicit",
            origin=scope.origin_of(_words(matches, variable)),
        )
        return SectorReading(
            codes=tuple(codes),
            grouped=False,
            record=AxisRecord(
                axis="sector",
                rule=CODE_RULE,
                active=entry.option_id,
                provenance="explicit",
                entry=entry,
            ),
        )
    groupings = tuple(dict.fromkeys(named.values()))
    concept = groupings[0].concept if len(groupings) == 1 else None
    if len(groupings) == 1 and set(codes) == named.keys():
        (grouping,) = groupings
        options = grouping_options(grouping)
        entry = next(
            (
                entry
                for entry in scope.entries_on("sector", concept=grouping.concept)
                if _applies(entry, options, named=codes)
            ),
            None,
        )
        if entry is not None:
            return SectorReading(
                codes=options[entry.option_id],
                grouped=True,
                record=AxisRecord(
                    axis="sector",
                    rule=GROUPING_RULE,
                    active=entry.option_id,
                    provenance=entry.provenance,
                    entry=entry,
                    concept=concept,
                ),
            )
    bound = tuple(
        sorted({code for code in codes if code not in named}.union(*(g.codes for g in groupings)))
    )
    return SectorReading(
        codes=bound,
        grouped=True,
        record=AxisRecord(
            axis="sector",
            rule=GROUPING_RULE,
            active=sector_option_id(bound),
            provenance="default",
            concept=concept,
        ),
    )


def _applies(entry: Entry, options: dict[str, tuple[str, ...]], *, named: Sequence[str]) -> bool:
    """Whether `entry` applies to a part naming a grouping alone whose options are
    `options`, with the codes `named` its words matched.

    The grouping must offer the entry's option. An entry made for the grouping's concept then
    applies; one made for no concept, a code entry ("reproductive health aid", CRS 130),
    applies only when the part's words named each of its codes, so a code chosen for other
    words never narrows the grouping."""
    codes = options.get(entry.option_id)
    if codes is None:
        return False
    return entry.concept is not None or set(codes) <= set(named)


def _words(matches: Sequence[QualifierMatch], variable: str) -> str:
    """The words of `variable` the matches read: the span from the first match to the last
    when every match was read from the variable, else the first match's words."""
    starts = [m.start for m in matches]
    ends = [m.end for m in matches]
    if matches and all(s is not None for s in starts) and all(e is not None for e in ends):
        start = min(s for s in starts if s is not None)
        end = max(e for e in ends if e is not None)
        return variable[start:end]
    return matches[0].matched_text if matches else variable


def sector_rule(
    base: BaseIdentity, record: AxisRecord | None, facts: ScopeFacts
) -> AxisTable | None:
    """The sector table of a part whose identity is `base`, None when `base` binds no
    sector code.

    Read from `base` alone, whether a binder recorded the axis (`record`) or not, so a
    resubmit of a served identity offers what its search offered. An option whose codes
    no covered member of the part's family declares is unavailable (`option_not_covered`);
    the option `base` holds is the one the part resolved, so it is available."""
    bound = _sector_binding(base)
    if bound is None:
        return None
    prop, codes = bound
    held = sector_option_id(codes)
    grouping = _grouping_holding(prop, codes)
    options = grouping_options(grouping) if grouping is not None else {held: codes}
    covered = _covered_values(base, prop, facts)
    table_options = tuple(
        Option(
            option_id=option_id,
            label=_label(option_codes, grouping, facts.index),
            binding=SlotBinding(
                slot=prop,
                state="value" if len(option_codes) == 1 else "set",
                refs=sorted(option_codes),
            ),
            availability=_AVAILABLE
            if option_id == held or _serves(option_codes, covered, facts.index)
            else _NOT_COVERED,
        )
        for option_id, option_codes in options.items()
    )
    return AxisTable(
        axis="sector",
        rule=GROUPING_RULE if grouping is not None else CODE_RULE,
        options=table_options,
        preference=(held, *(option_id for option_id in options if option_id != held)),
        held=held,
        concept=grouping.concept if grouping is not None else None,
    )


def _sector_binding(base: BaseIdentity) -> tuple[str, tuple[str, ...]] | None:
    """The sector property `base` binds and the codes it binds there, None when it binds
    none."""
    for slot in base.slots:
        if slot.property not in SECTOR_PROPERTIES:
            continue
        if isinstance(slot.binding, IdentityBindingValue):
            return slot.property, (slot.binding.ref,)
        if isinstance(slot.binding, IdentityBindingSet):
            return slot.property, tuple(sorted(slot.binding.refs))
    return None


def _grouping_holding(prop: str, codes: tuple[str, ...]) -> SectorGrouping | None:
    """The grouping on `prop` whose codes `codes` are, or which holds `codes`' one code."""
    for grouping in SECTOR_GROUPINGS:
        if grouping.code_system != prop:
            continue
        if set(codes) == set(grouping.codes) or (len(codes) == 1 and codes[0] in grouping.codes):
            return grouping
    return None


def _covered_values(base: BaseIdentity, prop: str, facts: ScopeFacts) -> frozenset[str]:
    """The `prop` values of the members of `base`'s family that hold every other binding of
    `base` and a series in the coverage index. A composite base reads the family of the
    identity its option was set on (`composite.table_document`)."""
    document = table_document(base, facts.index)
    if document is None:
        return frozenset()
    query = identity_family_query(facts.index, document)
    family = query_family(facts.index, query)
    if family is None:
        return frozenset()
    others = allowed_values(query.slots, skip=prop)
    properties = tuple(sorted({*others, prop}))
    at = properties.index(prop)
    return frozenset(
        value
        for values in facts.family_values(family, properties).covered
        if (value := values[at]) is not None
        and all(values[i] in others[p] for i, p in enumerate(properties) if p != prop)
    )


def _serves(codes: Sequence[str], covered: frozenset[str], index: CodeIndex) -> bool:
    """Whether a covered member declares one of `codes` or a code below one."""
    return any(not covered.isdisjoint(index.expansion(code)) for code in codes)


def _label(codes: Sequence[str], grouping: SectorGrouping | None, index: CodeIndex) -> str:
    """The reader's name of the option binding `codes` (`OPTION_LABELS`): the grouping's
    own label for its codes together, "health, CRS 120 alone" for one of its codes, else
    each code's label and number, joined: "basic health (CRS 122)"."""
    if grouping is not None and tuple(codes) == grouping.codes:
        return grouping.label
    template = OPTION_LABELS.sector_code if grouping is None else OPTION_LABELS.sector_code_alone
    return " + ".join(
        template.format(label=row.label, code=row.code)
        if (row := index.code_by_dcid(code))
        else code
        for code in sorted(codes)
    )
