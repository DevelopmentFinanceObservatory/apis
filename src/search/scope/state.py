"""The scope state one part's binders share: the entries they read and the records they
write back."""

from __future__ import annotations

from dataclasses import dataclass, field

from search.models import ChoiceProvenance, Entry, ScopeAxis

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


@dataclass(slots=True)
class ScopeState:
    """The scope entries one part resolves with, and the record of every axis its binders
    settle.

    `entries` are the part's explicit and carried entries. A binder reads those on its axis
    (`entries_on`) and applies precedence itself, then writes one `AxisRecord` (`write`)."""

    entries: tuple[Entry, ...] = ()
    records: dict[ScopeAxis, AxisRecord] = field(default_factory=dict)

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
