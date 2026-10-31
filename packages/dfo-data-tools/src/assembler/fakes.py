"""In-memory fakes for the frame, revision, release-registry and vocabulary ports.

They reproduce store refusals without a database or network. `FakeReleaseRegistry`
wraps one `CatalogueRelease`, and `FakeVocabularySource` one `Vocabulary`."""

from __future__ import annotations

import dataclasses
import secrets
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

from assembler.catalogue import CatalogueRelease, load_release
from assembler.ports import (
    CompositionAppend,
    CompositionMembership,
    DraftNotFound,
    EditConflict,
    FrameRef,
    FrameUnavailable,
    PriorCreate,
    ResolvedFrame,
    RevisionRecord,
    WriteContext,
    WriteSubjectKind,
    WriteTokenInUse,
)
from description.vocabulary import Vocabulary
from dfo_platform.caller import Caller

__all__ = ["FakeFrameStore", "FakeReleaseRegistry", "FakeRevisionStore", "FakeVocabularySource"]


class FakeFrameStore:
    """Stores frames by `FrameRef`, registered with `put`.

    `pin` is idempotent and records pins for assertions. Unknown frames raise
    `FrameUnavailable`. `release_pin_error` makes `release_pin` raise."""

    def __init__(
        self,
        *,
        frames: Mapping[FrameRef, ResolvedFrame] | None = None,
        release_pin_error: Exception | None = None,
    ) -> None:
        self._frames: dict[FrameRef, ResolvedFrame] = dict(frames or {})
        self._pins: list[tuple[FrameRef, str]] = []
        self.release_pin_error = release_pin_error

    def put(self, *, resolved_frame: ResolvedFrame) -> None:
        self._frames[resolved_frame.frame_ref] = resolved_frame

    def load(self, *, frame_ref: FrameRef) -> ResolvedFrame:
        try:
            return self._frames[frame_ref]
        except KeyError:
            raise FrameUnavailable(
                reason="io", detail=f"no fake frame registered for {frame_ref!r}"
            ) from None

    def pin(self, *, frame_ref: FrameRef, component_id: str, caller: Caller | None = None) -> bool:
        if frame_ref not in self._frames:
            raise FrameUnavailable(reason="io", detail=f"cannot pin unknown frame {frame_ref!r}")
        entry = (frame_ref, component_id)
        if entry in self._pins:
            return False
        self._pins.append(entry)
        return True

    def release_pin(self, *, frame_ref: FrameRef, component_id: str) -> None:
        if self.release_pin_error is not None:
            raise self.release_pin_error
        entry = (frame_ref, component_id)
        if entry in self._pins:
            self._pins.remove(entry)

    @property
    def pins(self) -> tuple[tuple[FrameRef, str], ...]:
        return tuple(self._pins)


class FakeRevisionStore:
    """Dict-backed `RevisionStore` fake. Holding an id grants read, as in the live store, and
    each draft keeps the write-token digest its create stored.

    An unknown component or composition id raises `DraftNotFound`. `begin_write`
    snapshots `current`, stages appends, and commits on clean exit. A stale prior
    revision raises `EditConflict`. A create whose component write-token digest a
    committed component already stores raises `WriteTokenInUse` on entry, or on
    commit when another create stored it while this one was open. During an open
    write, calls from that thread to `composition_of`, `read_revision`,
    `write_token_digest`, `prior_create`, or `allocate_component_id` raise
    `RuntimeError`. `read_current` is allowed. A committed revision's `created_at` is
    `clock()` at commit, as the live store's column default stamps it at insert."""

    def __init__(self, *, clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self._clock = clock
        self._revisions: dict[str, list[RevisionRecord]] = {}
        self._compositions: dict[str, list[CompositionAppend]] = {}
        # Composition membership is set once, at a component's own creation,
        # and stays fixed after that. This is a plain id map, independent of
        # `_compositions`, which carries no `component_id` of its own.
        self._component_compositions: dict[str, str] = {}
        self._write_token_digests: dict[tuple[WriteSubjectKind, str], bytes | None] = {}
        self._create_request_digests: dict[str, bytes | None] = {}
        self._local = threading.local()

    def _refuse_if_write_open(self, *, method: str) -> None:
        if getattr(self._local, "write_depth", 0) > 0:
            raise RuntimeError(
                f"{method}() called on this store while a write is open on this thread"
            )

    def _require_component(self, *, component_id: str) -> list[RevisionRecord]:
        revisions = self._revisions.get(component_id)
        if revisions is None:
            raise DraftNotFound(subject_kind="component", subject_id=component_id)
        return revisions

    def read_current(self, *, component_id: str) -> RevisionRecord | None:
        revisions = self._require_component(component_id=component_id)
        return revisions[-1] if revisions else None

    def read_revision(self, *, component_id: str, revision: int) -> RevisionRecord | None:
        self._refuse_if_write_open(method="read_revision")
        for record in self._require_component(component_id=component_id):
            if record.revision == revision:
                return record
        return None

    def composition_of(self, *, component_id: str) -> CompositionMembership | None:
        self._refuse_if_write_open(method="composition_of")
        self._require_component(component_id=component_id)
        composition_id = self._component_compositions.get(component_id)
        if composition_id is None:
            return None
        return CompositionMembership(
            composition_id=composition_id, revision=len(self._compositions.get(composition_id, ()))
        )

    def write_token_digest(
        self, *, subject_kind: WriteSubjectKind, subject_id: str
    ) -> bytes | None:
        self._refuse_if_write_open(method="write_token_digest")
        if subject_kind == "component":
            self._require_component(component_id=subject_id)
        elif subject_id not in self._compositions:
            raise DraftNotFound(subject_kind="composition", subject_id=subject_id)
        return self._write_token_digests.get((subject_kind, subject_id))

    def prior_create(self, *, write_token_sha256: bytes) -> PriorCreate | None:
        self._refuse_if_write_open(method="prior_create")
        component_id = self._component_holding(write_token_sha256=write_token_sha256)
        if component_id is None:
            return None
        return PriorCreate(
            component_id=component_id,
            composition_id=self._component_compositions.get(component_id),
            create_request_sha256=self._create_request_digests.get(component_id),
        )

    def _component_holding(self, *, write_token_sha256: bytes) -> str | None:
        for (subject_kind, subject_id), digest in self._write_token_digests.items():
            if subject_kind == "component" and digest == write_token_sha256:
                return subject_id
        return None

    def forget_write_token(self, *, subject_kind: WriteSubjectKind, subject_id: str) -> None:
        """Drop a stored digest, leaving the draft as one created before write tokens."""
        self._write_token_digests[(subject_kind, subject_id)] = None

    def allocate_component_id(self) -> str:
        self._refuse_if_write_open(method="allocate_component_id")
        return f"cmp_{secrets.token_hex(12)}"

    def begin_write(
        self,
        *,
        component_id: str,
        is_create: bool,
        composition_id: str | None,
        write_token_sha256: bytes | None = None,
        composition_write_token_sha256: bytes | None = None,
        create_request_sha256: bytes | None = None,
    ) -> AbstractContextManager[WriteContext]:
        return self._write(
            component_id=component_id,
            is_create=is_create,
            composition_id=composition_id,
            write_token_sha256=write_token_sha256 if is_create else None,
            composition_write_token_sha256=(
                composition_write_token_sha256 if composition_id == "new" else None
            ),
            create_request_sha256=create_request_sha256 if is_create else None,
        )

    @contextmanager
    def _write(
        self,
        *,
        component_id: str,
        is_create: bool,
        composition_id: str | None,
        write_token_sha256: bytes | None,
        composition_write_token_sha256: bytes | None,
        create_request_sha256: bytes | None,
    ) -> Iterator[_WriteContextImpl]:
        # Lock order is composition then component.
        if (
            composition_id is not None
            and composition_id != "new"
            and composition_id not in self._compositions
        ):
            raise DraftNotFound(subject_kind="composition", subject_id=composition_id)
        if is_create:
            self._refuse_taken_write_token(write_token_sha256=write_token_sha256)
            current = None
        else:
            revisions = self._require_component(component_id=component_id)
            current = revisions[-1] if revisions else None

        if composition_id == "new":
            resolved_composition_id = f"cps_{secrets.token_hex(12)}"
        elif composition_id is None and not is_create:
            # Revises preserve the component's creation-time composition membership.
            resolved_composition_id = self._component_compositions.get(component_id)
        else:
            resolved_composition_id = composition_id

        ctx = _WriteContextImpl(
            store=self,
            component_id=component_id,
            composition_id=resolved_composition_id,
            current=current,
            write_token_sha256=write_token_sha256,
            composition_write_token_sha256=composition_write_token_sha256,
            create_request_sha256=create_request_sha256,
        )
        self._local.write_depth = getattr(self._local, "write_depth", 0) + 1
        try:
            yield ctx
        finally:
            self._local.write_depth -= 1
        ctx._commit()

    def _refuse_taken_write_token(self, *, write_token_sha256: bytes | None) -> None:
        if (
            write_token_sha256 is not None
            and self._component_holding(write_token_sha256=write_token_sha256) is not None
        ):
            raise WriteTokenInUse


@dataclass
class _WriteContextImpl:
    store: FakeRevisionStore
    component_id: str
    composition_id: str | None
    current: RevisionRecord | None
    # Stored with the component, and with a newly minted composition, when the write commits.
    write_token_sha256: bytes | None
    composition_write_token_sha256: bytes | None
    create_request_sha256: bytes | None
    _staged: RevisionRecord | None = field(default=None, init=False)
    _staged_composition: CompositionAppend | None = field(default=None, init=False)

    def append(
        self, *, record: RevisionRecord, composition_entry: CompositionAppend | None
    ) -> int | None:
        live = self.store._revisions.get(self.component_id, [])
        live_current = live[-1] if live else None
        expected_prior_revision = self.current.revision if self.current else 0
        actual_prior_revision = live_current.revision if live_current else 0
        if actual_prior_revision != expected_prior_revision or record.revision != (
            expected_prior_revision + 1
        ):
            raise EditConflict
        self._staged = record
        self._staged_composition = composition_entry
        if composition_entry is None:
            return None
        # Return the staged composition revision before the write context commits.
        return len(self.store._compositions.get(composition_entry.composition_id, [])) + 1

    def _commit(self) -> None:
        if self._staged is not None:
            if self.component_id not in self.store._revisions:
                # Another create may have stored this digest while this write was open.
                self.store._refuse_taken_write_token(write_token_sha256=self.write_token_sha256)
                self.store._write_token_digests[("component", self.component_id)] = (
                    self.write_token_sha256
                )
                self.store._create_request_digests[self.component_id] = self.create_request_sha256
            if (
                self.composition_id is not None
                and self.composition_id not in self.store._compositions
            ):
                self.store._write_token_digests[("composition", self.composition_id)] = (
                    self.composition_write_token_sha256
                )
            self.store._revisions.setdefault(self.component_id, []).append(
                dataclasses.replace(self._staged, created_at=self.store._clock())
            )
            if self.composition_id is not None:
                self.store._component_compositions.setdefault(
                    self.component_id, self.composition_id
                )
        if self._staged_composition is not None:
            self.store._compositions.setdefault(self._staged_composition.composition_id, []).append(
                self._staged_composition
            )


class FakeReleaseRegistry:
    """`ReleaseRegistry` fake with a configurable release.

    Loads the baked release lazily."""

    def __init__(self, *, release: CatalogueRelease | None = None) -> None:
        self._release = release

    def _resolved_release(self) -> CatalogueRelease:
        if self._release is None:
            self._release = load_release()
        return self._release

    def pinned(self) -> CatalogueRelease:
        return self._resolved_release()


class FakeVocabularySource:
    """`VocabularySource` fake that gives one `Vocabulary`."""

    def __init__(self, *, vocabulary: Vocabulary) -> None:
        self._vocabulary = vocabulary

    def current(self) -> Vocabulary:
        return self._vocabulary
