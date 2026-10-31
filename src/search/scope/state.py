"""The scope state one part's binders share: the entries they read and the records they
write back."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from search.models import ChoiceProvenance, Entry, ScopeAxis, TextOrigin

_PRECEDENCE = {"explicit": 0, "carried": 1}


@dataclass(frozen=True, kw_only=True)
class AxisRecord:
    """What a binder settled on one axis.

    `rule` names the declared rule it applied, `active` the option id the part's identity
    now holds and `provenance` what settled it. `entry` is the explicit or carried entry the
    binder applied, None for a default, and `concept` the concept the axis was asked for
    ('health'), None when the axis has none."""

    axis: ScopeAxis
    rule: str
    active: str
    provenance: ChoiceProvenance
    entry: Entry | None = None
    concept: str | None = None

    def __post_init__(self) -> None:
        if self.active.partition(":")[0] != self.axis:
            raise ValueError(f"record on {self.axis!r} holds option {self.active!r}")
        if (self.entry is None) != (self.provenance == "default"):
            raise ValueError("a record holds an entry exactly when it is not a default")


@dataclass(frozen=True, slots=True)
class ScopeIntake:
    """The scope a request posts for one part, before its binders run.

    `entries` are the explicit entries read from the part's words and the conversation's
    carried entries. `total` is whether the words ask for a total ("total health aid"),
    which the provider default reads. It is a fact about the request and sits outside the
    entries. `sums_providers` is whether the explicit provider entry was read from words
    summing the bilateral and multilateral lines and naming no imputed share ("bilateral
    plus multilateral ODA"), which the part's dataset reads as the option summing the two
    (`provider.with_summed_provider`). `text` is the text the part's words were read from,
    the one every text origin of the part indexes (`models.TextOrigin`)."""

    entries: tuple[Entry, ...] = ()
    total: bool = False
    sums_providers: bool = False
    text: str = ""

    def state(self) -> ScopeState:
        """A fresh state for one resolution of the part."""
        return ScopeState(
            entries=self.entries,
            total=self.total,
            sums_providers=self.sums_providers,
            text=self.text,
        )


@dataclass(slots=True)
class ScopeState:
    """The scope entries one part resolves with, and the record of every axis its binders
    settle.

    `entries` are the part's explicit and carried entries. A binder reads those on its axis
    (`entries_on`) and applies precedence itself, then writes one `AxisRecord` (`write`).
    `total` is the request's total intent (`ScopeIntake.total`), `sums_providers` whether
    its provider entry sums the bilateral and multilateral lines (`ScopeIntake.sums_providers`)
    and `text` the text the part's words were read from (`ScopeIntake.text`)."""

    entries: tuple[Entry, ...] = ()
    records: dict[ScopeAxis, AxisRecord] = field(default_factory=dict)
    total: bool = False
    sums_providers: bool = False
    text: str = ""

    def origin_of(self, words: str) -> TextOrigin:
        """The origin of an entry read from `words`: their first occurrence in `text`,
        matched without regard to case or runs of whitespace, as `text` writes them. Words
        `text` does not hold have an empty span at its start."""
        pattern = r"\s+".join(map(re.escape, words.split()))
        found = re.search(pattern, self.text, re.IGNORECASE) if pattern else None
        if found is None:
            return TextOrigin(text=words, start=0, end=0)
        return TextOrigin(text=found.group(), start=found.start(), end=found.end())

    def entries_on(self, axis: ScopeAxis, *, concept: str | None = None) -> tuple[Entry, ...]:
        """The entries on `axis`, explicit before carried, each in posted order. With
        `concept`, an entry made for another concept is left out; one made for no concept
        is kept."""
        return tuple(
            sorted(
                (
                    entry
                    for entry in self.entries
                    if entry.axis == axis and entry.concept in (None, concept)
                ),
                key=lambda entry: _PRECEDENCE[entry.provenance],
            )
        )

    def write(self, record: AxisRecord) -> None:
        """Record what a binder settled. Raises `ValueError` when the axis already holds a
        record, since one part settles each axis once."""
        if record.axis in self.records:
            raise ValueError(f"the {record.axis!r} axis is already settled")
        self.records[record.axis] = record

    def attached(self) -> tuple[Entry, ...]:
        """The entries that attach to the part: those on an axis a binder settled, and of
        those made for a concept, the ones whose concept is the one the axis was asked for,
        then, on a part whose binders settled some axis, every explicit entry made for no
        concept on an axis the part has no record of. An entry on another axis the part does
        not have, or for another concept ("health" on an education part, or on a recipient's
        total with no sector axis), is not the part's, so it neither applies nor is reported
        as dropped. A stated word with no concept on an axis the part lacks ("net" on a part
        no basis rule reads) is the part's own, which serves no option for it, so it is
        reported."""
        return tuple(
            entry
            for entry in self.entries
            if (
                (record := self.records.get(entry.axis)) is not None
                and entry.concept in (None, record.concept)
            )
            or (
                bool(self.records)
                and record is None
                and entry.provenance == "explicit"
                and entry.concept is None
            )
        )
