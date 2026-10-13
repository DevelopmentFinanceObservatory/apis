"""Creates or revises components, pins create frames, and appends revisions transactionally."""

from __future__ import annotations

import datetime
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from assembler.catalogue import CatalogueRelease
from assembler.errors import (
    BindingInvalid,
    EditConflict,
    ReleaseMismatch,
    ResourceExhausted,
    WriteRefused,
)
from assembler.limits import MAX_INTENT_BYTES
from assembler.mint import DescribedRevision, stamp_envelope
from assembler.pipeline.candidate import _build_candidate
from assembler.ports import (
    CompositionAppend,
    FrameRef,
    FrameStore,
    PriorCreate,
    ReleaseRegistry,
    RevisionRecord,
    RevisionStore,
    VocabularySource,
    WriteSubjectKind,
    WriteTokenInUse,
)
from assembler.reply import WriteReply, build_changed_fields, build_reply
from assembler.resolve import ResolvedSeries
from assembler.write_token import (
    COMPONENT_WRITE_TOKEN_PREFIX,
    COMPOSITION_WRITE_TOKEN_PREFIX,
    CreateRequestInvalid,
    create_request_digest,
    mint_write_token,
    require_client_write_token,
    write_token_digest,
    write_token_matches,
)

logger = logging.getLogger(__name__)

# A create that fails after writing a frame reference logs this marker with
# `component_id`, `spec_id`, and `content_version`. A concurrent writer may use
# the same idempotent reference. Reconciliation determines whether cleanup is
# needed.
ORPHANED_PIN_RECONCILIATION_REQUIRED_MARKER = "presentation_pin_reconciliation_required"


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _series_coordinates_for_storage(
    *, series: Sequence[ResolvedSeries]
) -> tuple[dict[str, object], ...]:
    """Stores one `{id, coordinates}` entry per resolved series, in source order.

    Fact summaries use these coordinates. Table rows use `series_id` to
    recover the same entries on readback."""
    return tuple({"id": one.id, "coordinates": dict(one.coordinates)} for one in series)


def _series_coordinates_equal(
    *,
    candidate: Sequence[Mapping[str, object]],
    stored: Sequence[Mapping[str, object]] | None,
) -> bool:
    """Checks candidate coordinates against stored entries in order.

    Candidate fields must match. Extra stored fields are allowed. `None` is
    empty, so coordinate-free members compare equal while real series require
    stored coordinates."""
    stored_entries = tuple(stored or ())
    if len(candidate) != len(stored_entries):
        return False
    return all(
        all(
            key in stored_entry and stored_entry[key] == value
            for key, value in candidate_entry.items()
        )
        for candidate_entry, stored_entry in zip(candidate, stored_entries, strict=True)
    )


# -- write_component -----------------------------------------------------


def write_component(
    *,
    intent: Mapping[str, object],
    component_id: str | None = None,
    composition_id: str | None = None,
    write_token: str | None = None,
    create_write_token: str | None = None,
    frame_store: FrameStore,
    revision_store: RevisionStore,
    release_registry: ReleaseRegistry,
    vocabulary_source: VocabularySource | None = None,
    clock: Callable[[], datetime.datetime] = _utc_now,
) -> WriteReply:
    """Create a component, or revise the one at `component_id`.

    The revision written carries its description, worded from `vocabulary_source`, when it
    draws series from a frame whose version holds a description record
    (`assembler.pipeline.candidate`). Without a
    `vocabulary_source` no revision carries one. A revise whose envelope equals the head's in
    everything a reader sees is a no-op (`_structurally_equal`). One that gains, loses or
    rewords a description, or changes a source, citation or warning, writes a revision.

    `write_token` is the caller's `x-dfo-write-token`. A revise needs the component's own
    token, and a create into an existing composition needs that composition's token. Both
    checks run before any frame work, so a refused write pins nothing. A create mints the new
    component's token, plus the new composition's token for `composition_id="new"`, and the
    reply carries each once.

    `create_write_token` is a `cmw_` token the client minted for this create; it becomes the
    new component's token and keys the create. A create whose token a stored component already
    holds, sent with the same intent and `composition_id`, returns that component's create
    reply rebuilt from its revision 1, without resolving or pinning a frame. The same token
    with a different request, or on a component created without a request hash, raises
    `CreateRequestInvalid`. So does a malformed token, a token on a revise, or one sent with
    `composition_id="new"`. A create that loses a race to another create with the same token
    returns the winner's reply the same way.

    A revise whose head was minted under a newer catalogue release than this deployment's
    raises `ReleaseMismatch` before any frame work, and again if a concurrent write moves the
    head to a newer release before this one commits."""
    _check_intent_size(intent=intent)
    release = release_registry.pinned()
    is_create = component_id is None
    client_create = (
        _client_create(
            intent=intent,
            composition_id=composition_id,
            write_token=create_write_token,
            is_create=is_create,
        )
        if create_write_token is not None
        else None
    )

    if is_create:
        if composition_id is not None and composition_id != "new":
            _require_write_token(
                revision_store=revision_store,
                subject_kind="composition",
                subject_id=composition_id,
                presented=write_token,
            )
    else:
        _require_write_token(
            revision_store=revision_store,
            subject_kind="component",
            subject_id=component_id,
            presented=write_token,
        )
    if client_create is not None:
        prior = revision_store.prior_create(write_token_sha256=client_create.write_token_sha256)
        if prior is not None:
            return _replay_create(
                prior=prior,
                client_create=client_create,
                revision_store=revision_store,
            )
    if not is_create:
        new_write_token = None
    elif client_create is not None:
        new_write_token = client_create.write_token
    else:
        new_write_token = mint_write_token(prefix=COMPONENT_WRITE_TOKEN_PREFIX)
    new_composition_write_token = (
        mint_write_token(prefix=COMPOSITION_WRITE_TOKEN_PREFIX)
        if is_create and composition_id == "new"
        else None
    )

    current: RevisionRecord | None
    if is_create:
        current = None
    else:
        if composition_id is not None:
            raise BindingInvalid(
                pointer="/composition_id",
                dimension="",
                legal_values=(),
                truncated=False,
                detail="composition membership is set at creation; a revise cannot carry composition_id",
            )
        # An unknown `component_id` raises `DraftNotFound` here, before any frame work.
        current = revision_store.read_current(component_id=component_id)
        if current is None:
            raise EditConflict
        _require_release_not_behind(head=current, release=release)

    # Allocate before the transaction so frame pinning can do network I/O outside
    # the row lock, and before the candidate so its description names the component. The
    # store owns id formats. Revises reuse their pinned frame.
    working_component_id = component_id if not is_create else revision_store.allocate_component_id()

    def described(revision: int) -> DescribedRevision | None:
        if vocabulary_source is None:
            return None
        return DescribedRevision(
            vocabulary_source=vocabulary_source,
            component_id=working_component_id,
            revision=revision,
        )

    revision = (current.revision + 1) if current else 1
    candidate = _build_candidate(
        patch=intent,
        current=current,
        is_create=is_create,
        release=release,
        frame_store=frame_store,
        clock=clock,
        described=described(revision),
    )
    pinned_frame_ref: FrameRef | None = None
    if is_create and candidate.resolved_frame is not None:
        # note_narration resolves no frame (no `frame_ref` on its grammar
        # shape), so there is nothing to pin.
        pin_created = frame_store.pin(
            frame_ref=candidate.resolved_frame.frame_ref, component_id=working_component_id
        )
        # A `None` result leaves creation status unknown, so report the pin if the write fails.
        if pin_created is not False:
            pinned_frame_ref = candidate.resolved_frame.frame_ref

    is_noop = False
    try:
        with revision_store.begin_write(
            component_id=working_component_id,
            is_create=is_create,
            composition_id=composition_id,
            write_token_sha256=_digest_or_none(new_write_token),
            composition_write_token_sha256=_digest_or_none(new_composition_write_token),
            create_request_sha256=(
                client_create.request_sha256 if client_create is not None else None
            ),
        ) as ctx:
            live_current = ctx.current
            _require_release_not_behind(head=live_current, release=release)
            if live_current != current:
                if not is_create and live_current is None:
                    raise EditConflict
                current = live_current
                revision = (current.revision + 1) if current else 1
                candidate = _build_candidate(
                    patch=intent,
                    current=current,
                    is_create=is_create,
                    release=release,
                    frame_store=frame_store,
                    clock=clock,
                    resolved_frame=candidate.resolved_frame,
                    described=described(revision),
                )

            envelope = stamp_envelope(
                minted=candidate.minted,
                component_id=ctx.component_id,
                revision=revision,
                release=release,
            )
            series_coordinates = _series_coordinates_for_storage(
                series=candidate.series,
            )

            if (
                current is not None
                and _structurally_equal(envelope, current.resolved_payload)
                and _series_coordinates_equal(
                    candidate=series_coordinates, stored=current.series_coordinates
                )
                # Two intents can mint one envelope on this frame and differ on a later one:
                # a stated window beside an omitted one, a label beside the relative period
                # resolving to it, a listed select beside the default that draws the same
                # values. The head must store the intent just sent.
                and candidate.stored_intent == current.stored_intent
            ):
                is_noop = True
                noop_component_id = ctx.component_id
                noop_composition_id = ctx.composition_id
                noop_stored_intent = current.stored_intent
                noop_resolved_payload = current.resolved_payload
                noop_formatter_profile = current.formatter_profile
            else:
                record = RevisionRecord(
                    component_id=ctx.component_id,
                    revision=revision,
                    stored_intent=candidate.stored_intent,
                    resolved_payload=envelope,
                    catalog_version=release.catalog_version,
                    schema_version=release.schema_version,
                    formatter_profile=release.formatter_profile,
                    series_coordinates=series_coordinates,
                )
                composition_entry = (
                    CompositionAppend(composition_id=ctx.composition_id, actor="model")
                    if ctx.composition_id is not None
                    else None
                )
                composition_revision = ctx.append(
                    record=record, composition_entry=composition_entry
                )
                minted_composition_id = ctx.composition_id
    except WriteTokenInUse:
        _log_orphaned_pin(
            frame_ref=pinned_frame_ref,
            component_id=working_component_id,
            with_traceback=client_create is None,
        )
        if client_create is None:
            raise
        # Another create with this token committed first. The lookup runs in a fresh
        # transaction, so it sees the winner's component.
        return _replay_create(
            prior=revision_store.prior_create(write_token_sha256=client_create.write_token_sha256),
            client_create=client_create,
            revision_store=revision_store,
        )
    except BaseException:
        _log_orphaned_pin(frame_ref=pinned_frame_ref, component_id=working_component_id)
        raise

    if is_noop:
        # Membership lookup uses a separate transaction, so it follows this commit.
        # A no-op reply uses the stored head envelope and its generation timestamp.
        if noop_composition_id is not None:
            membership = revision_store.composition_of(component_id=noop_component_id)
        else:
            membership = None
        return build_reply(
            composition_id=membership.composition_id if membership is not None else None,
            composition_revision=membership.revision if membership is not None else None,
            changed=(),
            stored_intent=noop_stored_intent,
            resolved_payload=noop_resolved_payload,
            series_coordinates=series_coordinates,
            formatter_profile=noop_formatter_profile,
        )

    changed = build_changed_fields(
        current_intent=current.stored_intent if current else None, patch=intent
    )
    return build_reply(
        composition_id=minted_composition_id,
        composition_revision=composition_revision,
        changed=changed,
        stored_intent=candidate.stored_intent,
        resolved_payload=envelope,
        series_coordinates=series_coordinates,
        formatter_profile=release.formatter_profile,
        write_token=new_write_token,
        composition_write_token=new_composition_write_token,
    )


def _log_orphaned_pin(
    *, frame_ref: FrameRef | None, component_id: str, with_traceback: bool = True
) -> None:
    # Leave the reference for reconciliation. Another create can share this pin and commit.
    # A create that lost the race for its own write token is expected, so it logs no traceback.
    if frame_ref is None:
        return
    logger.error(
        "%s component_id=%s spec_id=%s content_version=%s",
        ORPHANED_PIN_RECONCILIATION_REQUIRED_MARKER,
        component_id,
        frame_ref.spec_id,
        frame_ref.content_version,
        exc_info=with_traceback,
    )


@dataclass(frozen=True)
class _ClientCreate:
    """A create keyed by the write token its client minted."""

    write_token: str
    write_token_sha256: bytes
    request_sha256: bytes


def _client_create(
    *,
    intent: Mapping[str, object],
    composition_id: str | None,
    write_token: str,
    is_create: bool,
) -> _ClientCreate:
    """Check a client write token's format and use, and digest it and its request."""
    require_client_write_token(write_token)
    if not is_create:
        raise CreateRequestInvalid(
            pointer="/write_token",
            detail="a revise carries no write_token in its body; send the component's token "
            "as x-dfo-write-token",
        )
    if composition_id == "new":
        raise CreateRequestInvalid(
            pointer="/write_token",
            detail='write_token cannot be sent with composition_id "new"; omit write_token, or '
            "create the component outside a composition",
        )
    return _ClientCreate(
        write_token=write_token,
        write_token_sha256=write_token_digest(write_token),
        request_sha256=create_request_digest(intent=intent, composition_id=composition_id),
    )


def _replay_create(
    *,
    prior: PriorCreate | None,
    client_create: _ClientCreate,
    revision_store: RevisionStore,
) -> WriteReply:
    """Rebuild the create reply of the component a client write token already names.

    `composition_revision` reports the composition head now, as a no-op reply does. A stored
    request hash that differs, or none at all, raises `CreateRequestInvalid`. A `prior` of
    `None` after `WriteTokenInUse`, or a missing revision 1, raises `EditConflict`."""
    if prior is None:
        raise EditConflict
    if prior.create_request_sha256 != client_create.request_sha256:
        raise CreateRequestInvalid(
            pointer="/write_token",
            detail="write_token already names a draft created from a different request; "
            "mint a new write_token for each create",
        )
    record = revision_store.read_revision(component_id=prior.component_id, revision=1)
    if record is None:
        raise EditConflict
    membership = (
        revision_store.composition_of(component_id=prior.component_id)
        if prior.composition_id is not None
        else None
    )
    return build_reply(
        composition_id=membership.composition_id if membership is not None else None,
        composition_revision=membership.revision if membership is not None else None,
        changed=(),
        stored_intent=record.stored_intent,
        resolved_payload=record.resolved_payload,
        series_coordinates=record.series_coordinates,
        formatter_profile=record.formatter_profile,
        write_token=client_create.write_token,
    )


def _require_release_not_behind(*, head: RevisionRecord | None, release: CatalogueRelease) -> None:
    """Raise `ReleaseMismatch` when `head` was minted under a newer release than `release`.

    Writing would move the head back to an older release, whose intent schema may refuse the
    head's stored intent. This happens on an instance left behind by a rollback. The write
    succeeds once the newer release serves again."""
    if head is None or _calver(head.catalog_version) <= _calver(release.catalog_version):
        return
    raise ReleaseMismatch(
        catalog_version=head.catalog_version,
        detail=(
            f"component {head.component_id} revision {head.revision} was minted under catalogue "
            f"release {head.catalog_version}, newer than this deployment's "
            f"{release.catalog_version}"
        ),
    )


def _calver(catalog_version: str) -> tuple[int, ...]:
    """Split a `YYYY.M.N` catalog version, the envelope schema's pattern, into integers."""
    return tuple(int(part) for part in catalog_version.split("."))


def _digest_or_none(token: str | None) -> bytes | None:
    return write_token_digest(token) if token is not None else None


def _require_write_token(
    *,
    revision_store: RevisionStore,
    subject_kind: WriteSubjectKind,
    subject_id: str,
    presented: str | None,
) -> None:
    """Raise `WriteRefused` unless `presented` matches the subject's stored digest.

    Digests are written once at creation and never change, so this check holds for the
    write transaction that follows. An unknown id raises `DraftNotFound`."""
    stored_digest = revision_store.write_token_digest(
        subject_kind=subject_kind, subject_id=subject_id
    )
    if write_token_matches(presented=presented, stored_digest=stored_digest):
        return
    if stored_digest is None:
        detail = f"{subject_kind} {subject_id!r} predates write tokens and cannot be written"
    elif presented is None:
        detail = f"writing {subject_kind} {subject_id!r} needs its write token in x-dfo-write-token"
    else:
        detail = f"the x-dfo-write-token sent does not grant writing {subject_kind} {subject_id!r}"
    if subject_kind == "component":
        fix = (
            "create a new component with POST /api/presentation/components; read this one "
            f"first with GET /api/presentation/components/{subject_id} to reuse its intent"
        )
    else:
        fix = (
            'create the component outside this composition, or with composition_id "new" '
            "to start a new composition"
        )
    raise WriteRefused(subject_kind=subject_kind, subject_id=subject_id, detail=detail, fix=fix)


def _check_intent_size(*, intent: Mapping[str, object]) -> None:
    size = len(json.dumps(intent).encode("utf-8"))
    if size > MAX_INTENT_BYTES:
        raise ResourceExhausted(
            pointer="/",
            limit_name="MAX_INTENT_BYTES",
            detail=f"submitted intent is {size} bytes, over the maximum of {MAX_INTENT_BYTES}",
        )


_VOLATILE_ENVELOPE_FIELDS = ("revision",)
_VOLATILE_DESCRIPTION_FIELDS = ("built_from",)
_REVISION_NAMING_DESCRIPTION_MEMBERS = ("describes", "cite")


def _excluding_volatile(
    payload: Mapping[str, object], *, volatile: Sequence[str]
) -> dict[str, object]:
    """`payload` without the members that move when its content does not: its `revision`,
    the provenance members in `volatile`, and its description's `built_from` and the revision
    its `describes` and `cite` name."""
    stripped = {
        key: value for key, value in payload.items() if key not in _VOLATILE_ENVELOPE_FIELDS
    }
    provenance = cast("Mapping[str, object]", stripped.get("provenance") or {})
    stripped["provenance"] = {
        key: value for key, value in provenance.items() if key not in volatile
    }
    description = cast("Mapping[str, object] | None", stripped.get("description"))
    if description is not None:
        stripped["description"] = {
            key: _without_revision(value) if key in _REVISION_NAMING_DESCRIPTION_MEMBERS else value
            for key, value in description.items()
            if key not in _VOLATILE_DESCRIPTION_FIELDS
        }
    return stripped


def _without_revision(member: object) -> object:
    """A description member naming the card revision, without that revision."""
    named = cast("Mapping[str, object]", member)
    return {key: value for key, value in named.items() if key != "revision"}


def _structurally_equal(candidate: Mapping[str, object], head: Mapping[str, object]) -> bool:
    """Compares payloads in everything a reader sees, ignoring what moves on every write of the
    same content: `revision`, `provenance.generated_at`, the card revision a description
    names, and the description's `built_from`, whose vocabulary revision moves with every
    graph generation while its words stay put. The words themselves are compared, so a
    description gained, lost or reworded, and a changed source, citation or warning, make the
    revise a revision.

    A head whose `provenance.execution_ref` is null equals a candidate whatever replay key the
    candidate names, since cards issued before provenance named a derived frame's replay key
    store null there. Catalogue and schema versions remain part of the comparison."""
    head_provenance = cast("Mapping[str, object]", head.get("provenance") or {})
    volatile: tuple[str, ...] = ("generated_at",)
    if head_provenance.get("execution_ref") is None:
        volatile = (*volatile, "execution_ref")
    return _excluding_volatile(candidate, volatile=volatile) == _excluding_volatile(
        head, volatile=volatile
    )
