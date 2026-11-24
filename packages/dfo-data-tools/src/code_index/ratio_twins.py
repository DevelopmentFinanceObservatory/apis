"""Ratio families whose role classes divide by role twins of one denominator, read from a
code index's listed rows alone.

IDS publishes a ratio, such as the US-dollar share of the public and publicly guaranteed
debt stock, as one StatVar per role set: the aggregate share divides by the PPG stock and
declares the debtor, and the share by creditor divides by the PPG stock broken down by
creditor and declares both roles. The two StatVars differ in their denominator, so each
role class is a five-tuple of its own. `ratio_twins` maps every such class five-tuple to its
family (`RatioTwins`), so Search derives the family as one shape whatever recall returns.

Only five-tuples of listed rows (`CodeIndex.listed_by_five_tuple`) are candidates. The
export refuses a release unless every StatVar of such a five-tuple is an index row
(`code_index.export._verify_listed_five_tuples`), so the rows of each class are its whole
family and the mapping reads no recall.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import permutations
from typing import TYPE_CHECKING

from code_index.query import row_five_tuple
from code_index.shared import per_index

if TYPE_CHECKING:
    from code_index.query import CodeIndex
    from code_index.types import StatVarRow
    from statvar_members import FiveTuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RatioClass:
    """One role class of a ratio family: the rows of one five-tuple that declare roles."""

    five_tuple: FiveTuple
    """The class's five-tuple, its denominator included."""
    roles: tuple[str, ...]
    """The role set every row of the class declares, in its first row's order."""
    rows: tuple[StatVarRow, ...]
    """Every role-bearing index row of `five_tuple`, in index order."""

    @property
    def denominator(self) -> str:
        """The StatVar the class divides by."""
        denominator = self.five_tuple.measurement_denominator
        assert denominator is not None, "a ratio class divides by a StatVar"
        return denominator


@dataclass(frozen=True, slots=True)
class RatioTwins:
    """One ratio family: its role classes, the base class first, then by role count.

    The base declares the roles every other class declares, and each other class divides by
    the base's denominator broken down by the roles it adds."""

    classes: tuple[RatioClass, ...]

    @property
    def base(self) -> RatioClass:
        """The class whose roles every other class's roles hold."""
        return self.classes[0]

    def class_of(self, roles: Sequence[str]) -> RatioClass | None:
        """The class declaring role set `roles`, compared without regard to order, or None."""
        wanted = frozenset(roles)
        return next((c for c in self.classes if frozenset(c.roles) == wanted), None)


type _BucketKey = tuple[FiveTuple, FiveTuple, frozenset[tuple[str, str]]]


@per_index
def ratio_twins(index: CodeIndex) -> Mapping[FiveTuple, RatioTwins]:
    """Every class five-tuple of a ratio family of `index`, mapped to its family.

    Candidates are the five-tuples of listed rows that divide by a StatVar. They bucket by
    five-tuple with the denominator replaced by the denominator row's own five-tuple and
    constraint map, so a bucket holds the ratios of one numerator over a denominator and its
    role twins. A bucket of two or more becomes one `RatioTwins` when all of these hold:

    1. No class five-tuple has an index family (`CodeIndex.families_by_five_tuple`), and each
       denominator is an index row that declares roles.
    2. Every role-bearing row of a class declares one role set, and the classes' role sets
       are pairwise distinct.
    3. Denominators and classes order alike: one denominator's roles nest in another's
       exactly when the classes' roles do, by the same extra roles.
    4. One class's roles sit inside every other's. That class is the base.

    A bucket that fails any of them yields no entry, so each of its five-tuples derives a
    shape of its own, and is logged once here with the condition it failed."""
    rows_by_five_tuple: dict[FiveTuple, list[StatVarRow]] = defaultdict(list)
    for five_tuple, listed in index.listed_by_five_tuple.items():
        if five_tuple.measurement_denominator is not None and not five_tuple.denominator_unresolved:
            rows_by_five_tuple[five_tuple].extend(listed)
    for row in index.statvars:
        if (five_tuple := row_five_tuple(row)) in rows_by_five_tuple:
            rows_by_five_tuple[five_tuple].append(row)

    buckets: dict[_BucketKey, list[FiveTuple]] = defaultdict(list)
    for five_tuple in rows_by_five_tuple:
        denominator = index.statvar_by_dcid(five_tuple.measurement_denominator or "")
        if denominator is None:
            continue
        key = (
            replace(five_tuple, measurement_denominator=None),
            row_five_tuple(denominator),
            frozenset(denominator.constraint_properties.items()),
        )
        buckets[key].append(five_tuple)

    entries: dict[FiveTuple, RatioTwins] = {}
    for five_tuples in buckets.values():
        if len(five_tuples) < 2:
            continue
        twins, failed = _validated(index, five_tuples, rows_by_five_tuple)
        if twins is None:
            denominators = sorted(str(f.measurement_denominator) for f in five_tuples)
            logger.info(
                "code index: the ratios dividing by %s are no ratio family (%s); each derives "
                "its own shape",
                ", ".join(denominators),
                failed,
                extra={"denominators": denominators, "failed": failed},
            )
            continue
        entries.update((c.five_tuple, twins) for c in twins.classes)
    return entries


def _validated(
    index: CodeIndex,
    five_tuples: Sequence[FiveTuple],
    rows_by_five_tuple: Mapping[FiveTuple, Sequence[StatVarRow]],
) -> tuple[RatioTwins | None, str]:
    """The family of one bucket's `five_tuples`, or None and the condition it fails
    (`ratio_twins`)."""
    classes: list[RatioClass] = []
    denominator_roles: dict[FiveTuple, frozenset[str]] = {}
    for five_tuple in five_tuples:
        if index.families_by_five_tuple.get(five_tuple):
            return None, "a class five-tuple has an index family"
        denominator = index.statvar_by_dcid(five_tuple.measurement_denominator or "")
        if denominator is None or not denominator.observation_properties:
            return None, "a denominator is no role-bearing index row"
        denominator_roles[five_tuple] = frozenset(denominator.observation_properties)
        rows = tuple(row for row in rows_by_five_tuple[five_tuple] if row.observation_properties)
        if not rows or len({frozenset(row.observation_properties) for row in rows}) != 1:
            return None, "a class declares no role set or more than one"
        classes.append(
            RatioClass(five_tuple=five_tuple, roles=rows[0].observation_properties, rows=rows)
        )

    roles = {c.five_tuple: frozenset(c.roles) for c in classes}
    if len(set(roles.values())) != len(classes):
        return None, "two classes declare one role set"
    for first, second in permutations(classes, 2):
        nests = denominator_roles[first.five_tuple] < denominator_roles[second.five_tuple]
        if nests != (roles[first.five_tuple] < roles[second.five_tuple]):
            return None, "classes order unlike their denominators"
        added = denominator_roles[second.five_tuple] - denominator_roles[first.five_tuple]
        if nests and added != roles[second.five_tuple] - roles[first.five_tuple]:
            return None, "classes order unlike their denominators"

    classes.sort(key=lambda c: (len(c.roles), sorted(c.roles)))
    base = frozenset(classes[0].roles)
    if not all(base < frozenset(c.roles) for c in classes[1:]):
        return None, "no class's roles sit inside every other's"
    return RatioTwins(classes=tuple(classes)), ""


__all__ = ["RatioClass", "RatioTwins", "ratio_twins"]
