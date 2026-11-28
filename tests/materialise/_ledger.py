"""The synthetic `dataset/Ledger`, whose series two member provenances hold, as a register
filed by publisher does, and `dataset/Solo`, a declared Dataset with one member provenance.

`bound_ledger_index` binds a release whose code index declares both, so every reader of a
provenance's dataset (`code_index.shared.dataset_of`) reads them. `provenance/Plain` is a
plain table, its own sole member, in either release.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

from code_index.query import CodeIndex
from code_index.types import DatasetDeclaration, IndexStamp
from graph_release.context import bound_release, current_release

LEDGER = "dataset/Ledger"
NORTH = "provenance/LedgerNorth"
SOUTH = "provenance/LedgerSouth"
SOLO = "dataset/Solo"
SOLO_MEMBER = "provenance/SoloMember"
PLAIN = "provenance/Plain"
MEMBERS = (NORTH, SOUTH, SOLO_MEMBER)


def _declaration(dataset: str, *, rank: int) -> DatasetDeclaration:
    return DatasetDeclaration(
        id=dataset,
        label=dataset.rsplit("/", 1)[-1],
        description=f"The test dataset {dataset}.",
        opt_in=False,
        absent_observation="unknown",
        account="provider",
        rank=rank,
    )


def ledger_index(ledger_members: tuple[str, ...] = (NORTH, SOUTH)) -> CodeIndex:
    """A code index declaring `LEDGER` of `ledger_members`, `NORTH` and `SOUTH` by default,
    `SOLO` of `SOLO_MEMBER`, and the plain table `PLAIN`."""
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="https://fixture.example.org",
            exported_at="2026-10-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="sha256:ledger",
        ),
        declarations=(),
        codes=(),
        statvars=(),
        datasets=(
            _declaration(LEDGER, rank=0),
            _declaration(SOLO, rank=1),
            _declaration(PLAIN, rank=2),
        ),
        dataset_members={LEDGER: ledger_members, SOLO: (SOLO_MEMBER,)},
    )


@contextmanager
def bound_ledger_index(ledger_members: tuple[str, ...] = (NORTH, SOUTH)) -> Iterator[CodeIndex]:
    """Binds the bound release with `ledger_index` of `ledger_members` as its code index for
    the `with` block."""
    release = current_release()
    index = ledger_index(ledger_members)
    with bound_release(replace(release, views=replace(release.views, code_index=index))):
        yield index
