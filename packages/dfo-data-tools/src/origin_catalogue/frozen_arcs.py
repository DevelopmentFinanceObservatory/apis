"""Arc answers the catalogue's tables give, built once per snapshot and held read-only.

A `->*` answer the tables hold (`origin_catalogue.graph._table_arcs`) is a pure function
of the arc rows that answer it, and those rows are fixed for the life of the catalogue
build and the graph release that hold them. `ArcAnswers` builds each dcid's answer once
from its rows and serves the same `FrozenArcs` to every later read of those rows while the
answer is among the `ARC_ANSWERS_HELD` most recently read, so a resolve that reads
thousands of members builds each answer once. A reader of a `FrozenArcs` keeps what it
derives from it on the answer itself (`FrozenArcs.derived`), so it parses each answer once
too.

The answer keeps the wire shape `dc_graph.GraphClient.node_arcs_batch` returns, as
`dict` and `list` subclasses, so every reader of a live answer reads it unchanged. It
refuses every mutation, since one answer is shared by every request of its snapshot.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Final, NoReturn, Self

from origin_catalogue.un_mirror import UnArc, import_added, wire_arcs

_CONSTRAINT_PROPERTIES: Final[str] = "constraintProperties"


def _refuse(self: object, *args: object, **kwargs: object) -> NoReturn:
    raise TypeError(f"{type(self).__name__} is read-only")


class FrozenList(list):
    """A `list` that refuses every mutation."""

    __slots__ = ()

    append = extend = insert = remove = pop = clear = sort = reverse = _refuse
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _refuse

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict) -> Self:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (list(self),))


class FrozenDict(dict):
    """A `dict` that refuses every mutation."""

    __slots__ = ()

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _refuse

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict) -> Self:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (dict(self),))


class FrozenArcs(FrozenDict):
    """A node's `->*` answer, read-only at every level, which keeps what its readers
    derive from it (`derived`)."""

    __slots__ = ("_derived",)

    def __init__(self, arcs: Mapping[str, Any]) -> None:
        super().__init__(arcs)
        self._derived: dict[Callable[[FrozenArcs], object], object] = {}

    def derived[T](self, read: Callable[[FrozenArcs], T]) -> T:
        """`read(self)`, computed on the first call and kept on the answer. `read` must be a
        pure function of the answer that returns an immutable value, since every reader of
        this answer receives the same one."""
        try:
            return self._derived[read]  # ty: ignore[invalid-return-type]
        except KeyError:
            value = self._derived[read] = read(self)
            return value


def frozen_arcs(wire: Mapping[str, Mapping[str, Iterable[Mapping[str, Any]]]]) -> FrozenArcs:
    """`wire`, a `->*` answer, as a `FrozenArcs` of the same content."""
    return FrozenArcs(
        {
            prop: FrozenDict(
                {
                    key: FrozenList(
                        FrozenDict(
                            {
                                name: FrozenList(value) if isinstance(value, list) else value
                                for name, value in node.items()
                            }
                        )
                        for node in nodes
                    )
                    for key, nodes in entry.items()
                }
            )
            for prop, entry in wire.items()
        }
    )


def served_arcs(rows: Iterable[UnArc]) -> FrozenArcs:
    """Table arc rows as `GraphClient.node_arcs_batch` returns them, without the
    `import_added` targets, as the graph client leaves them out. The client keeps the
    `constraintProperties` entry when it leaves out every target, and so does this."""
    held = tuple(rows)
    wire = wire_arcs(arc for arc in held if not import_added(arc))
    if any(arc.property == _CONSTRAINT_PROPERTIES for arc in held):
        wire.setdefault(_CONSTRAINT_PROPERTIES, {"nodes": []})
    return frozen_arcs(wire)


ARC_ANSWERS_HELD: Final[int] = 8_192
"""How many answers `ArcAnswers` keeps, the most recently read. An answer of a CRS variable
holds about 10 KB with what its readers derive from it, so the memo holds at most about
90 MB; the widest CRS total reads 6,310 members, which fit."""


class ArcAnswers:
    """One snapshot's `served_arcs` answers, each built on its dcid's first read and kept
    while it is among the `held` most recently read.

    An answer is kept with the rows it was built from and served again only for those same
    rows (`is`), so a dcid whose rows moved, as a DFO variable's do when a new graph release
    binds under the same catalogue, gets an answer built from its new rows. Safe to share
    between threads."""

    def __init__(self, held: int = ARC_ANSWERS_HELD) -> None:
        self._held = held
        self._answers: OrderedDict[str, tuple[tuple[UnArc, ...], FrozenArcs]] = OrderedDict()
        self._lock = threading.Lock()

    def answer(self, dcid: str, rows: tuple[UnArc, ...]) -> FrozenArcs:
        """`served_arcs(rows)`, `dcid`'s answer, built once for `rows`."""
        with self._lock:
            kept = self._answers.get(dcid)
            if kept is not None and kept[0] is rows:
                self._answers.move_to_end(dcid)
                return kept[1]
        built = served_arcs(rows)
        with self._lock:
            self._answers[dcid] = (rows, built)
            self._answers.move_to_end(dcid)
            while len(self._answers) > self._held:
                self._answers.popitem(last=False)
        return built


__all__ = [
    "ArcAnswers",
    "FrozenArcs",
    "FrozenDict",
    "FrozenList",
    "frozen_arcs",
    "served_arcs",
]
