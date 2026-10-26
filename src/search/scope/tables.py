"""The inputs compile reads: one table per axis, the conflicts between options of different
axes, and the engine facts the axis rules build the tables from."""

from __future__ import annotations

from dataclasses import dataclass

from code_index.query import CodeIndex
from search.models import Option, ScopeAxis
from search.scope.edits import AXIS_BINDING, ScopeTableError
from series_coverage.shared import CoverageSnapshot


@dataclass(frozen=True, kw_only=True)
class AxisTable:
    """One axis of one part, as its rule reads it.

    `options` are the axis's options in declared order, each with its binding and its
    intrinsic availability (what the part's own data and constraints allow, before any
    other entry is weighed). `preference` is the order a default is derived in: the first
    option no entry blocks. `held` is the option the base identity holds. `rule` names the
    declared rule, and `concept` the concept the axis was asked for, None when it has none.

    Raises `ScopeTableError` for an empty table, a repeated option, an option of another
    axis or binding kind, or a `preference` or `held` naming no option."""

    axis: ScopeAxis
    rule: str
    options: tuple[Option, ...]
    preference: tuple[str, ...]
    held: str
    concept: str | None = None

    def __post_init__(self) -> None:
        ids = [option.option_id for option in self.options]
        if not ids or len(set(ids)) != len(ids):
            raise ScopeTableError(f"{self.axis!r} needs distinct options, got {ids}")
        for option in self.options:
            if option.option_id.partition(":")[0] != self.axis:
                raise ScopeTableError(f"{option.option_id!r} is not an option of {self.axis!r}")
            if option.binding.kind != AXIS_BINDING[self.axis]:
                raise ScopeTableError(f"{option.option_id!r} carries a {option.binding.kind!r}")
        if not self.preference or not set(self.preference) <= set(ids):
            raise ScopeTableError(f"{self.axis!r} preference {self.preference} names no option")
        if self.held not in ids:
            raise ScopeTableError(f"{self.axis!r} holds {self.held!r}, which it does not offer")

    def option(self, option_id: str) -> Option | None:
        """The option `option_id`, None when the axis offers none."""
        return next((o for o in self.options if o.option_id == option_id), None)


@dataclass(frozen=True)
class Conflict:
    """Two options of different axes that cannot hold together, and why."""

    first: str
    second: str
    cause: str


@dataclass(frozen=True, kw_only=True)
class CompileFacts:
    """Everything compile reads beside the base identity and the entries: the table of
    every axis the part settles, in default-derivation order, and the conflicts."""

    axes: tuple[AxisTable, ...]
    conflicts: tuple[Conflict, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ScopeFacts:
    """The engine state the axis rules read: the release's code index and the request's
    coverage snapshot, both in memory. Passed explicitly, so a rule reads no global."""

    index: CodeIndex
    coverage: CoverageSnapshot
