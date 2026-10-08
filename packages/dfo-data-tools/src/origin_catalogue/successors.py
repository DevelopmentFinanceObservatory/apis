"""Derives which UN variable stands for each superseded base variable.

A base provenance the registry excludes as `superseded_by_un_system` keeps
its variables in the withheld set, and the registry entry says how one of
their ids reads as a series code and a set of constraint pairs
(`registry.UnSupersededProvenance.constraint_key`). `derive_successors`
reads the withheld variables carrying an entry's `variable_prefix` that DFO
does not hold, the superseded variables, and gives each a row naming its
successor: the successor agency's variable of the same code whose
constraint pairs equal the superseded variable's, order ignored, when the
UN tables admit it.

A row names no successor when the agency's variable of that code and pairs
belongs to a series the registry excludes, and when the UN tables hold no
such variable, whether the UN names a dimension or spells a value otherwise
or holds the breakdown not at all. Two agency variables with one code and
one set of pairs make the declared rule pick none, even when one of them
has the superseded variable's own id: the row names no successor and the
variable is listed in `Successors.ambiguous`, which the job refuses. A variable DFO holds has no row, since DFO's data stands.
Every superseded variable stays withheld.

Every successor is a key of `UnMirror.variables`, so no row names an
excluded or absent variable. `unheld_successors` checks a table against UN
tables.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass, field
from typing import Final

from origin_catalogue.registry import ConstraintKey, Registry, UnAgency, UnSupersededProvenance
from origin_catalogue.un_mirror import UnMirror


@dataclass(frozen=True, slots=True)
class Successors:
    """The successor table and how each superseded variable fared.

    `rows` maps each superseded variable to its successor UN variable, or
    to `None`. `twins` counts the rows naming the UN variable of the same
    id and `reordered` those naming one that holds the same pairs in
    another order. `series_excluded` and `unmatched` count the rows naming
    no successor, by cause, and `held_by_dfo` the variables left without a
    row. `ambiguous` maps each variable whose code and pairs two UN
    variables share to those variables.
    """

    rows: Mapping[str, str | None]
    twins: int = 0
    reordered: int = 0
    series_excluded: int = 0
    unmatched: int = 0
    held_by_dfo: int = 0
    ambiguous: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


EMPTY_SUCCESSORS: Final = Successors(rows={})


def derive_successors(
    registry: Registry,
    *,
    withheld_variables: Set[str],
    dfo_variables: Set[str],
    mirror: UnMirror,
) -> Successors:
    """A row for every withheld variable a registry supersession's
    `variable_prefix` starts and DFO does not hold, naming its successor
    from `mirror`. Rows are in variable order."""
    supersessions = registry.un_supersessions()
    if not supersessions:
        return EMPTY_SUCCESSORS
    keyed = {
        supersession.provenance: _by_constraint_key(supersession, agency, mirror)
        for supersession, agency in supersessions
    }
    rows: dict[str, str | None] = {}
    ambiguous: dict[str, tuple[str, ...]] = {}
    twins = reordered = series_excluded = unmatched = held_by_dfo = 0
    for variable in sorted(withheld_variables):
        for supersession, agency in supersessions:
            suffix = supersession.variable_suffix(variable)
            if suffix is None:
                continue
            if variable in dfo_variables:
                held_by_dfo += 1
                break
            matched = keyed[supersession.provenance].get(supersession.constraint_key(suffix), ())
            if len(matched) > 1:
                rows[variable] = None
                ambiguous[variable] = matched
            elif not matched:
                rows[variable] = None
                unmatched += 1
            elif (successor := matched[0]) in mirror.excluded_variables:
                rows[variable] = None
                series_excluded += 1
            else:
                rows[variable] = successor
                if successor == f"{agency.id_prefix}{suffix}":
                    twins += 1
                else:
                    reordered += 1
            break
    return Successors(
        rows=rows,
        twins=twins,
        reordered=reordered,
        series_excluded=series_excluded,
        unmatched=unmatched,
        held_by_dfo=held_by_dfo,
        ambiguous=ambiguous,
    )


def _by_constraint_key(
    supersession: UnSupersededProvenance, agency: UnAgency, mirror: UnMirror
) -> dict[ConstraintKey, tuple[str, ...]]:
    """The agency's variables, admitted and excluded, by code and pairs, each
    key's variables sorted."""
    found: dict[ConstraintKey, list[str]] = {}
    prefix = agency.id_prefix
    for dcid in sorted((*mirror.variables, *mirror.excluded_variables)):
        if dcid.startswith(prefix):
            found.setdefault(supersession.constraint_key(dcid[len(prefix) :]), []).append(dcid)
    return {key: tuple(dcids) for key, dcids in found.items()}


def unheld_successors(rows: Mapping[str, str | None], mirror: UnMirror) -> list[str]:
    """The variables of `rows` naming a successor that is not an admitted
    variable of `mirror`, sorted. A table the same build derived has none."""
    return sorted(
        variable
        for variable, successor in rows.items()
        if successor is not None and successor not in mirror.variables
    )


__all__ = ["EMPTY_SUCCESSORS", "Successors", "derive_successors", "unheld_successors"]
