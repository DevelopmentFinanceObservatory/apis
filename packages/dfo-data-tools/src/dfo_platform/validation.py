"""Map request validation errors to problem details with body-aware JSON Pointers."""

from __future__ import annotations

import contextvars
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, Discriminator, Tag
from pydantic.fields import FieldInfo

from dfo_platform.problem import ProblemSpec

if TYPE_CHECKING:
    from fastapi.exceptions import RequestValidationError


request_body_model: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "dfo_request_body_model", default=None
)
"""The request body model a `RequestValidationError` is currently being
classified against, or None outside a request context.

A contextvars.ContextVar for the same reason `dfo_platform.trace`'s own
trace_id is one: the pointer discipline in `pointer_for_error` needs the
model a failing request was validated against, but every door and every
`Surface.classify` already calls `validation_problem(exc)` with no model
argument and no request in hand -- FastAPI's exception handlers run outside
the endpoint's own call chain. `dfo_platform.app` sets this from the matched
route's own body parameters around the classification call, and every
surface's `validation_problem` reads it back out without a signature change
to any door.

The bound value is either one annotation (`model=` passed by a caller) or a
`BodyParams` naming the route's body parameters. A route with several body
parameters -- or one embedded body parameter -- reports a `loc` that names
the failing parameter at index 1, so the `BodyParams` is what lets
`pointer_for_error` pick the annotation the failing segment was actually
validated against; a route with a single non-embedded body parameter has no
such name in its loc, and its `BodyParams.fallback` carries that
annotation for the walk to step into directly.
"""


def set_request_body_model(model: Any) -> contextvars.Token[Any]:
    """Bind a body model and return its reset token."""
    return request_body_model.set(model)


def reset_request_body_model(token: contextvars.Token[Any]) -> None:
    request_body_model.reset(token)


def walkable_body_annotation(state: Any) -> bool:
    """True for models, lists of walkable annotations, and unions with walkable members.

    `Any`, scalars, and missing annotations return `False`."""
    if state is None:
        return False
    annotation = _state_annotation(state)
    if annotation is Any:
        return False
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    origin = typing.get_origin(annotation)
    if origin is list:
        args = typing.get_args(annotation)
        return bool(args) and walkable_body_annotation(args[0])
    if origin in (types.UnionType, typing.Union):
        members = [a for a in typing.get_args(annotation) if a is not type(None)]
        return bool(members) and all(walkable_body_annotation(a) for a in members)
    if origin is Annotated:
        return walkable_body_annotation(typing.get_args(annotation)[0])
    return False


def _unwrap_annotation(annotation: Any) -> Any:
    """Strip `Annotated` and single-member optional wrappers for pointer walks.

    Keep dict/list wrappers and unions with multiple members so `_annotation_step` can inspect
    them and their metadata."""
    current = annotation
    while True:
        if typing.get_origin(current) is Annotated:
            current = typing.get_args(current)[0]
            continue
        if typing.get_origin(current) in (types.UnionType, typing.Union):
            args = tuple(a for a in typing.get_args(current) if a is not type(None))
        else:
            break
        if len(args) != 1:
            break
        current = args[0]
    return current


def _strip_optional(annotation: Any) -> Any:
    """`annotation` with a `Optional`/`X | None` wrapper removed, every other
    layer (an `Annotated[...]` carrying a union's own `Discriminator`
    metadata) left in place."""
    current = annotation
    while True:
        if typing.get_origin(current) not in (types.UnionType, typing.Union):
            return current
        args = tuple(a for a in typing.get_args(current) if a is not type(None))
        if len(args) != 1:
            return current
        current = args[0]


def _arm_tag(arm: Any) -> str | None:
    """Return an arm's `Tag` metadata, or `None` when its tag comes from a field."""
    if typing.get_origin(arm) is Annotated:
        for meta in typing.get_args(arm)[1:]:
            if isinstance(meta, Tag):
                return meta.tag
    return None


def _state_annotation(state: Any) -> Any:
    """Return the field annotation from `FieldInfo`, or a bare annotation for dict/list members."""
    return state.annotation if isinstance(state, FieldInfo) else state


def _state_discriminator(state: Any) -> Any:
    """Return a field's `FieldInfo` discriminator, or `None` for a bare annotation."""
    return state.discriminator if isinstance(state, FieldInfo) else None


def _literal_values(model: Any, field_name: str) -> tuple[Any, ...]:
    """Return the field's `Literal` values, or `()` when it has no `Literal` annotation."""
    fields = getattr(model, "model_fields", None)
    if not isinstance(fields, dict) or field_name not in fields:
        return ()
    annotation = _unwrap_annotation(fields[field_name].annotation)
    if typing.get_origin(annotation) is not typing.Literal:
        return ()
    return typing.get_args(annotation)


def _discriminated_arms(annotation: Any, hoisted: Any) -> tuple[Any, tuple[Any, ...]] | None:
    """Return a discriminated union's discriminator and arms, or `None`.

    Handles `FieldInfo`-hoisted discriminators and metadata on callable or dict-value
    annotations."""
    raw = _strip_optional(annotation)
    if typing.get_origin(raw) is dict:
        args = typing.get_args(raw)
        if len(args) == 2:
            raw = args[1]
    if typing.get_origin(raw) is Annotated:
        members = typing.get_args(raw)
        metadata = next(
            (m for m in members[1:] if isinstance(m, (Discriminator, FieldInfo))),
            None,
        )
        discriminator = metadata.discriminator if isinstance(metadata, FieldInfo) else metadata
        inner = _unwrap_annotation(members[0])
    else:
        discriminator = hoisted
        inner = _unwrap_annotation(raw)
    if typing.get_origin(inner) not in (types.UnionType, typing.Union):
        return None
    arms = typing.get_args(inner)
    # FastAPI body `FieldInfo` preserves arm `Tag` metadata and drops the enclosing
    # `Discriminator`. Tagged arms therefore identify a discriminated union on their own.
    if not isinstance(discriminator, (str, Discriminator)) and not any(
        _arm_tag(arm) is not None for arm in arms
    ):
        return None
    return discriminator, arms


def _union_tag(state: Any, seg: Any) -> tuple[str, Any] | None:
    """Return the arm selected by injected union tag `seg`, or `None`.

    Pydantic inserts `Literal` or callable `Tag` values into `loc`. Plain unions have no tags."""
    resolved = _discriminated_arms(_state_annotation(state), _state_discriminator(state))
    if resolved is None:
        return None
    discriminator, arms = resolved
    for arm in arms:
        model = _unwrap_annotation(arm)
        if isinstance(discriminator, str):
            if seg in _literal_values(model, discriminator):
                return str(seg), model
            continue
        tag = _arm_tag(arm)
        if tag is not None and tag == seg:
            return tag, model
    return None


def error_field_loc(error: dict[str, Any]) -> list[str]:
    """Return stringified location segments with FastAPI's leading `body` segment removed.

    Used for exact top-level field checks."""
    return [str(p) for p in error.get("loc", ()) if p != "body"]


def pointer_segment(segment: object) -> str:
    """Escape one JSON Pointer segment per RFC 6901 by mapping `~` to `~0` and `/` to `~1`."""
    return str(segment).replace("~", "~0").replace("/", "~1")


def _resolves_at(node: object, seg: object) -> bool:
    """Whether `seg` is a present dict key or an in-range list index on `node`."""
    if isinstance(node, dict):
        return seg in node
    if isinstance(node, list) and isinstance(seg, int):
        return 0 <= seg < len(node)
    return False


@dataclass(frozen=True)
class BodyParams:
    """Initial pointer-walk state for a route's body parameters.

    Embedded or multiple bodies put the parameter name first in `loc`, resolved through
    `by_name`. A single unembedded body starts at its field name and uses `fallback`."""

    by_name: Mapping[str, Any]
    embedded: bool
    fallback: Any = None


def _annotation_step(state: Any, seg: object) -> tuple[bool, Any]:
    """Return `(known, next_state)` after following `seg` through a model, dict, or list.

    `known=False` ends annotation tracking. Model fields retain `FieldInfo`, including hoisted
    discriminators."""
    unwrapped = _unwrap_annotation(_state_annotation(state))
    if isinstance(unwrapped, type) and issubclass(unwrapped, BaseModel):
        fields = unwrapped.model_fields
        if isinstance(seg, str) and seg in fields:
            return True, fields[seg]
        return False, None
    origin = typing.get_origin(unwrapped)
    if origin is dict and isinstance(seg, str):
        args = typing.get_args(unwrapped)
        return (True, args[1]) if len(args) == 2 else (False, None)
    if origin is list and isinstance(seg, int):
        args = typing.get_args(unwrapped)
        return (True, args[0]) if args else (False, None)
    return False, None


def pointer_for_error(
    exc: RequestValidationError,
    error: dict[str, Any],
    *,
    model: type[BaseModel] | Any | None = None,
) -> str | None:
    """Return a JSON Pointer for a body error, or `None` when no faithful pointer exists.

    Pydantic inserts discriminated-union tag values into `loc`, creating segments absent from
    the body. Walk `loc`, the parsed body, and the model annotation together. Keep body keys and
    in-range list indices, plus the terminal segment for a `missing` error. Use union metadata to
    drop injected tags in place when a segment is absent or the next segment resolves against the
    same body node. This avoids mistaking a tag for a sibling key or shifting a later list index.

    Discriminated unions supply tag metadata. Drop unresolved synthetic arm names from plain unions,
    including terminal names on arm mismatches. Preserve a missing field's terminal name when it
    matches a union tag. The error type separates missing-field errors from arm mismatches.

    Without a model, inference uses body shape alone. A tag matching a real key may be followed as
    a key and produce a pointer one level too deep. The app binds the matched route's model. Direct
    callers pass `model`. `BodyParams` selects the parameter annotation for embedded or multiple
    bodies. List bodies are supported. Return `None` for non-body errors and malformed JSON. A
    malformed body is raw text, with no mapping or list keys to traverse."""
    loc = error.get("loc", ())
    if not loc or loc[0] != "body":
        return None
    body = exc.body
    if not isinstance(body, (dict, list)):
        return None

    segments = loc[1:]
    if not segments:
        return None

    bound = model if model is not None else request_body_model.get()
    state: Any = bound
    tracking = bound is not None

    kept: list[str] = []
    current: object = body
    last = len(segments) - 1
    for i, seg in enumerate(segments):
        # Resolve embedded parameter names before checking union tags.
        consumed = False
        if isinstance(state, BodyParams):
            if state.embedded:
                state = state.by_name.get(seg) if isinstance(seg, str) else None
                consumed = state is not None
            else:
                state = state.fallback
            tracking = state is not None
        tag_match = _union_tag(state, seg) if tracking and not consumed else None
        if i != last and tag_match is not None:
            doubled = _resolves_at(current, segments[i + 1])
            if doubled or not _resolves_at(current, seg):
                state = tag_match[1]
                continue
        resolved = _resolves_at(current, seg)
        if resolved:
            current = current[seg]
        if tracking and not consumed:
            known, next_state = _annotation_step(state, seg)
            tracking = known
            state = next_state if known else None
        if resolved or (i == last and error.get("type") == "missing"):
            kept.append(pointer_segment(seg))
    return "/" + "/".join(kept) if kept else None


def validation_problem(
    exc: RequestValidationError, *, model: type[BaseModel] | Any | None = None
) -> ProblemSpec:
    """Map validation errors to `request_invalid` (400) using the first error's pointer and detail.

    A top-level `schema_version` error maps to `unsupported_schema_version` (422) with the
    catalogue's default detail. With no errors, detail is `None`. `model` overrides the
    annotation bound in `request_body_model`."""
    errors = exc.errors()
    pointer: str | None = None
    code = "request_invalid"
    status = 400
    detail: str | None = None
    if errors:
        pointer = pointer_for_error(exc, errors[0], model=model)
        detail = errors[0].get("msg")

    version_error = next(
        (e for e in errors if error_field_loc(e) == ["schema_version"]),
        None,
    )
    if version_error is not None:
        code = "unsupported_schema_version"
        status = 422
        pointer = pointer_for_error(exc, version_error, model=model)
        detail = None

    return ProblemSpec(status=status, code=code, detail=detail, pointer=pointer, retryable=False)
