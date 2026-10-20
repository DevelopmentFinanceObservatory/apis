"""The catalogue version bound to the current request.

A request binds one catalogue at entry and every stage reads it back, so
admission, labels and the stamp an answer reports come from one version.
Context variables do not cross `ThreadPoolExecutor.submit`, so work handed
to a pool either receives the catalogue as an argument or runs under a
copied context. Jobs pass a catalogue explicitly and bind nothing.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass

from graph_release.context import served_origin_key
from origin_catalogue.catalogue import Catalogue

_BOUND: ContextVar[Catalogue | None] = ContextVar("origin_catalogue", default=None)


class CatalogueNotBound(RuntimeError):
    """Admission was asked for with no catalogue bound to the current context."""


def bind_catalogue(catalogue: Catalogue) -> Token[Catalogue | None]:
    """Binds `catalogue` to the current context. Pass the token to
    `unbind_catalogue` to restore the previous binding."""
    return _BOUND.set(catalogue)


def unbind_catalogue(token: Token[Catalogue | None]) -> None:
    _BOUND.reset(token)


@contextmanager
def bound_catalogue(catalogue: Catalogue) -> Iterator[Catalogue]:
    """Binds `catalogue` for the body of the `with` block."""
    token = _BOUND.set(catalogue)
    try:
        yield catalogue
    finally:
        _BOUND.reset(token)


def scoped_reads(allowed: frozenset[str] | None) -> AbstractContextManager[object]:
    """Binds the bound catalogue narrowed to the provenances `allowed`
    (`Catalogue.scoped`) for the body of the `with` block, so every read admits only
    those, before DFO precedence. Binds nothing when `allowed` is `None`."""
    if allowed is None:
        return nullcontext()
    return bound_catalogue(current_catalogue().scoped(allowed))


def current_catalogue() -> Catalogue:
    """The bound catalogue. Raises `CatalogueNotBound` when none is bound."""
    catalogue = _BOUND.get()
    if catalogue is None:
        raise CatalogueNotBound(
            "no origin catalogue is bound to this context; bind one at request entry, or "
            "run pooled work under contextvars.copy_context()"
        )
    return catalogue


def current_build_id(*, default: Callable[[], str]) -> str:
    """The bound catalogue's build id, or `default()` where none is bound."""
    catalogue = _BOUND.get()
    return default() if catalogue is None else catalogue.build_id


def current_words_build() -> str:
    """The bound catalogue's build id, and with `GRAPH_RELEASE_ORIGIN` on the bound release's id
    after `@` (`served_origin_key`), which answers the DFO variables' names and types. A label
    read stamps it, so words read under one release are read again under the next. Raises
    `CatalogueNotBound` when no catalogue is bound."""
    build_id = current_catalogue().build_id
    release = served_origin_key()
    return build_id if release is None else f"{build_id}@{release}"


def current_source_scope() -> frozenset[str] | None:
    """The bound catalogue's source scope (`Catalogue.scope`), or `None` where it has none
    or none is bound."""
    catalogue = _BOUND.get()
    return None if catalogue is None else catalogue.scope


@dataclass(frozen=True, slots=True)
class UnReadFailure:
    """A UN observation read that failed inside a read also naming our variables, which served
    its other part. `detail` is the error's text."""

    variables: tuple[str, ...]
    places: tuple[str, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class UnUnserved:
    """What an `UnBlockedPlaces` record held when it was read (`UnBlockedPlaces.snapshot`):
    the refused places in first-seen order, and the failed reads. A facet probe carries it to
    the materialisation that reuses the probe."""

    places: tuple[str, ...] = ()
    failures: tuple[UnReadFailure, ...] = ()


class UnBlockedPlaces:
    """What UN observation reads could not serve over one `recording_un_blocked_places` block:
    the places they named that the UN front door refuses, in first-seen order, and the reads
    that failed while the rest of their read served (`failures`). Reads on worker threads under
    a copied context add to the same record."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._places: dict[str, None] = {}
        self._failures: list[UnReadFailure] = []

    def add(self, places: Iterable[str]) -> None:
        with self._lock:
            self._places.update(dict.fromkeys(places))

    def add_failure(self, failure: UnReadFailure) -> None:
        with self._lock:
            self._failures.append(failure)

    def places(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._places)

    def failures(self) -> tuple[UnReadFailure, ...]:
        with self._lock:
            return tuple(self._failures)

    def snapshot(self) -> UnUnserved:
        with self._lock:
            return UnUnserved(places=tuple(self._places), failures=tuple(self._failures))

    def merge(self, unserved: UnUnserved) -> None:
        """Adds what `unserved` holds. A failure this record already holds is added once."""
        with self._lock:
            self._places.update(dict.fromkeys(unserved.places))
            self._failures.extend(
                failure for failure in unserved.failures if failure not in self._failures
            )


_UN_BLOCKED: ContextVar[UnBlockedPlaces | None] = ContextVar(
    "origin_catalogue_un_blocked_places", default=None
)


@contextmanager
def recording_un_blocked_places() -> Iterator[UnBlockedPlaces]:
    """Collects, for the body of the `with` block, the refused places UN reads name."""
    record = UnBlockedPlaces()
    token = _UN_BLOCKED.set(record)
    try:
        yield record
    finally:
        _UN_BLOCKED.reset(token)


def un_unavailability_record() -> UnBlockedPlaces | None:
    """The record bound by `recording_un_blocked_places`, or `None` outside one."""
    return _UN_BLOCKED.get()


def record_un_read_failure(
    exc: BaseException,
    *,
    variables: Iterable[str],
    places: Iterable[str],
    alongside_ours: bool,
) -> bool:
    """Record a UN observation read that failed with `exc` in the record bound by
    `recording_un_blocked_places`, so the caller reads the UN part as empty and its read of our
    variables still serves. Only a read made alongside ours (`alongside_ours`) records; with no
    record bound, or nothing of ours alongside, it returns `False` and the caller raises."""
    record = _UN_BLOCKED.get()
    if not alongside_ours or record is None:
        return False
    record.add_failure(
        UnReadFailure(variables=tuple(variables), places=tuple(places), detail=str(exc))
    )
    return True


def note_un_blocked_places(places: Iterable[str]) -> None:
    """Adds `places` to the record bound by `recording_un_blocked_places`. Nothing records
    outside one, because only a caller that reports refused places binds a record."""
    record = _UN_BLOCKED.get()
    if record is not None:
        record.add(places)


def note_un_unserved(unserved: UnUnserved) -> None:
    """Merges `unserved` into the record bound by `recording_un_blocked_places`, so what a
    probe made under another record could not read joins this one. Nothing records outside
    one."""
    record = _UN_BLOCKED.get()
    if record is not None:
        record.merge(unserved)


__all__ = [
    "CatalogueNotBound",
    "UnBlockedPlaces",
    "UnReadFailure",
    "UnUnserved",
    "bind_catalogue",
    "bound_catalogue",
    "current_build_id",
    "current_catalogue",
    "current_source_scope",
    "current_words_build",
    "note_un_blocked_places",
    "note_un_unserved",
    "recording_un_blocked_places",
    "un_unavailability_record",
    "unbind_catalogue",
]
