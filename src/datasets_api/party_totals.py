"""`options.without_party_totals`: a flow frame without the group totals of the party roles its
definition leaves open.

Flow datasets such as OECD DAC2A list group totals (the DAC total, the G7, DAC EU members) as
parties beside their members, so a frame for an open party side mixes members with sums of
members. A party role is a role on a flow axis (`materialise.enrichment.is_party_role`). It is
bound when the served identity's `entities` name an entity on its axis
(`dfo_platform.roles.role_key_axis`), which a request's `scope.entities` replaces whole
(`datasets_api.surgery.apply_scope_surgery`); every other party role of the frame is open.

For each open party role, in column-name order, the rows whose party the code index declares a
`total` in the row's dataset (`materialise.enrichment.party_kind_of` for the row's `provenance`)
are left out, each role reading the rows the roles before it kept. A party can be a total in one
dataset and a party in another (`code_index.types.PartyRow.dataset_kinds`), so the cut keys on
the party and the dataset together. A row with no provenance reads the kind of every dataset,
and keeps a party whose kind differs between datasets. Parties of kind `party` or `residual`, and parties the index declares no
kind for, stay. A role on which every row holds a total keeps its rows, since the totals are
then the only parties there are. A bound role is left alone whatever its parties' kinds.

The cut reads the frame before any reshape, so a `groupby` over it sums members alone. Each pass
holds one chunk at a time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc
from dfo_frame_grammar import TIME_PERIOD_COLUMN, Frame, stream_frame_sha256

from code_index.types import PartyRow
from dfo_platform.roles import role_key_axis
from materialise.enrichment import is_party_role, party_kind_of

_TOTAL_KIND = "total"

PROVENANCE_COLUMN = "provenance"
"""The frame column naming each row's dataset."""

YEARS_DIMENSION = "years"
"""The coverage dimension counting a frame's distinct leading years
(`materialise.columnar.coverage`)."""

TRACE_KEY = "party_totals_left_out"
"""The trace key of the totals a version left out (`PartyTotalsCut.trace_entries`)."""


def open_party_roles(identity: object, entity_columns: Sequence[str]) -> tuple[str, ...]:
    """The party roles among `entity_columns` whose axis `identity`'s `entities` bind no entity
    on, in column-name order."""
    entities = identity.get("entities") if isinstance(identity, Mapping) else None
    bound_axes = (
        {
            role_key_axis(str(role))
            for role, refs in entities.items()
            if isinstance(refs, list | tuple) and refs
        }
        if isinstance(entities, Mapping)
        else set()
    )
    return tuple(
        sorted(
            role
            for role in entity_columns
            if is_party_role(role) and role_key_axis(role) not in bound_axes
        )
    )


@dataclass(frozen=True)
class PartyTotalsCut:
    """The totals a cut leaves out of a frame: each open party role that lost rows, mapped to
    each provenance it lost rows in and the sorted refs of the totals it lost there. The
    provenance is `None` for the rows of a frame without a `provenance` column and the rows
    whose cell is null."""

    left_out: Mapping[str, Mapping[str | None, tuple[str, ...]]]

    def chunks(self, frame_chunks: Callable[[], Iterator[Frame]]) -> Callable[[], Iterator[Frame]]:
        """A fresh pass over `frame_chunks` less the rows `left_out` names, on each call. A
        chunk the cut empties is skipped."""

        def cut() -> Iterator[Frame]:
            for chunk in frame_chunks():
                kept = _without(chunk, self.left_out)
                if kept.table.num_rows:
                    yield kept

        return cut

    def trace_entries(self) -> list[dict[str, object]]:
        """The trace's `party_totals_left_out` entries: one `{role, entities}` per role that lost
        rows, in role order, its entities in ref order, each named once whatever datasets it
        lost rows in."""
        return [
            {
                "role": role,
                "entities": sorted({ref for refs in by_dataset.values() for ref in refs}),
            }
            for role, by_dataset in sorted(self.left_out.items())
        ]


@dataclass(frozen=True)
class CutFrame:
    """The frame a cut keeps. `frame_chunks` returns a fresh pass over it, `frame_sha256` is its
    hash (`dfo_frame_grammar.frame_sha256`), `rows` counts its rows, and `distinct` counts the
    distinct non-null values of each of its role columns and its leading years
    (`YEARS_DIMENSION`), as a materialisation's coverage counts them
    (`materialise.columnar.coverage`).
    `left_out` holds the cut's trace entries (`PartyTotalsCut.trace_entries`)."""

    frame_chunks: Callable[[], Iterator[Frame]]
    frame_sha256: str
    rows: int
    distinct: Mapping[str, int]
    left_out: tuple[dict[str, object], ...]


def cut_frame(
    frame_chunks: Callable[[], Iterator[Frame]],
    cut: PartyTotalsCut,
    *,
    role_columns: Sequence[str],
) -> CutFrame:
    """The frame `frame_chunks` passes over, less what `cut` leaves out, hashed and counted in
    one pass."""
    kept = cut.chunks(frame_chunks)
    rows = 0
    seen: dict[str, set[object]] = {name: set() for name in (*role_columns, YEARS_DIMENSION)}

    def tallied() -> Iterator[Frame]:
        nonlocal rows
        for chunk in kept():
            rows += chunk.table.num_rows
            for name in role_columns:
                seen[name].update(_distinct(chunk.table.column(name)))
            years = pc.utf8_slice_codeunits(chunk.table.column(TIME_PERIOD_COLUMN), 0, 4)
            seen[YEARS_DIMENSION].update(_distinct(years))
            yield chunk

    digest = stream_frame_sha256(tallied())
    return CutFrame(
        frame_chunks=kept,
        frame_sha256=digest,
        rows=rows,
        distinct={name: len(values) for name, values in seen.items()},
        left_out=tuple(cut.trace_entries()),
    )


def cut_party_totals(
    frame_chunks: Callable[[], Iterator[Frame]],
    *,
    roles: Sequence[str],
    parties: Mapping[str, PartyRow],
) -> PartyTotalsCut:
    """The cut of the frame `frame_chunks` passes over, on its open party `roles`, with kinds
    from the code index's `parties` for each row's dataset. Reads the frame once per role, each
    pass after the cuts of the roles before it."""
    left_out: dict[str, dict[str | None, tuple[str, ...]]] = {}
    for role in roles:
        totals: dict[str | None, set[str]] = {}
        others = False
        for chunk in frame_chunks():
            if role not in chunk.columns:
                break
            kept = _without(chunk, left_out)
            for ref, dataset in _parties_by_dataset(kept, role):
                datasets = () if dataset is None else (dataset,)
                if (
                    isinstance(ref, str)
                    and party_kind_of(ref, parties=parties, datasets=datasets) == _TOTAL_KIND
                ):
                    totals.setdefault(dataset, set()).add(ref)
                else:
                    others = True
        if totals and others:
            left_out[role] = {dataset: tuple(sorted(refs)) for dataset, refs in totals.items()}
    return PartyTotalsCut(left_out=left_out)


def _parties_by_dataset(chunk: Frame, role: str) -> list[tuple[object, str | None]]:
    """The distinct `(ref, provenance)` pairs of `chunk`'s `role` column, the provenance `None`
    for a frame without a `provenance` column and for a null cell."""
    table = chunk.table
    if PROVENANCE_COLUMN not in table.column_names:
        return [(ref, None) for ref in pc.unique(table.column(role)).to_pylist()]
    pairs = (
        table.select([role, PROVENANCE_COLUMN]).group_by([role, PROVENANCE_COLUMN]).aggregate([])
    )
    return list(
        zip(
            pairs.column(role).to_pylist(),
            pairs.column(PROVENANCE_COLUMN).to_pylist(),
            strict=True,
        )
    )


def _distinct(values: pa.Array | pa.ChunkedArray) -> list[object]:
    """The distinct non-null values of `values`."""
    return pc.unique(values).drop_null().to_pylist()


def _without(chunk: Frame, left_out: Mapping[str, Mapping[str | None, tuple[str, ...]]]) -> Frame:
    """`chunk` less each row whose party on a role of `left_out` is one of that role's refs. A
    null cell matches no ref. A ref left out in a provenance leaves out its rows of that
    provenance alone, and one left out for `None` its rows with no provenance."""
    if not left_out:
        return chunk
    table = chunk.table
    for role, by_dataset in left_out.items():
        for dataset, refs in by_dataset.items():
            column = table.column(role)
            match = pc.is_in(column, value_set=pa.array(refs, type=column.type))
            if PROVENANCE_COLUMN in table.column_names:
                provenance = table.column(PROVENANCE_COLUMN)
                in_dataset = (
                    pc.is_null(provenance)
                    if dataset is None
                    else pc.equal(pc.fill_null(provenance, pa.scalar("")), pa.scalar(dataset))
                )
                match = pc.and_(match, in_dataset)
            elif dataset is not None:
                continue
            # Two column operands give a column; the stubs widen it to a scalar too.
            table = table.filter(cast("pa.ChunkedArray", pc.invert(match)))
    return Frame(columns=chunk.columns, table=table)


__all__ = [
    "TRACE_KEY",
    "YEARS_DIMENSION",
    "CutFrame",
    "PartyTotalsCut",
    "cut_frame",
    "cut_party_totals",
    "open_party_roles",
]
