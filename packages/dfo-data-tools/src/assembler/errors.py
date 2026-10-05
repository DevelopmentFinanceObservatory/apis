"""Assembler refusals, including frame and revision-store errors re-exported from `ports`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from assembler.ports import (
    DraftNotFound,
    EditConflict,
    FrameUnavailable,
    StructuralView,
    WriteSubjectKind,
)

__all__ = [
    "AssemblerRefusal",
    "BindingInvalid",
    "ConstraintViolation",
    "DraftNotFound",
    "EditConflict",
    "FrameUnavailable",
    "ReleaseMismatch",
    "ResourceExhausted",
    "RevisionNotFound",
    "SchemaInvalid",
    "WriteRefused",
    "escape_pointer_segment",
    "pointer_from_path",
]


def escape_pointer_segment(segment: str) -> str:
    """Escapes one RFC 6901 segment, replacing `~` before `/`.

    Dimension names may contain both characters, so callers must escape
    them before building refusal pointers."""
    return segment.replace("~", "~0").replace("/", "~1")


def pointer_from_path(path: Sequence[object]) -> str:
    """One RFC 6901 JSON Pointer joined from a `jsonschema` validator's
    `absolute_path` segments, each escaped via `escape_pointer_segment`;
    `"/"` for an empty path."""
    if not path:
        return "/"
    return "/" + "/".join(escape_pointer_segment(str(segment)) for segment in path)


class AssemblerRefusal(Exception):
    """Base for every refusal below. `pointer` names the offending field of
    the *submitted intent* -- a raise site working against the resolved
    component subtree (a constraint check, say) is responsible for
    translating its own component-relative pointer into an intent-relative
    one before raising. `detail` is the
    human-readable explanation the wire's `detail` field carries through
    unchanged.
    """

    def __init__(self, *, pointer: str, detail: str) -> None:
        self.pointer = pointer
        self.detail = detail
        super().__init__(detail)


class BindingInvalid(AssemblerRefusal):
    """Selector resolution failed against the frame.

    `dimension` is empty for frame-wide refusals. `legal_values` and
    `truncated` describe the offending dimension; an unknown dimension gets
    the frame's dimension names. `structural_view` is set for an unbound facet
    that splits a series. `labels` maps refs to frame labels, with refs as the fallback."""

    def __init__(
        self,
        *,
        pointer: str,
        dimension: str,
        legal_values: tuple[str, ...],
        truncated: bool,
        detail: str,
        structural_view: StructuralView | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(pointer=pointer, detail=detail)
        self.dimension = dimension
        self.legal_values = legal_values
        self.truncated = truncated
        self.structural_view = structural_view
        self.labels = labels if labels is not None else {}


class ResourceExhausted(AssemblerRefusal):
    """A pre-resolution or resolved quantity exceeds a `limits.py` maximum.
    `resolve.py` raises this for the selector-count cap and the series-count
    estimate, checked before any selector is matched against frame rows,
    ahead of the cross-product expansion."""

    def __init__(self, *, pointer: str, limit_name: str, detail: str) -> None:
        super().__init__(pointer=pointer, detail=detail)
        self.limit_name = limit_name


class SchemaInvalid(AssemblerRefusal):
    """The intent fails its catalogue schema. Missing required fields carry
    `minimal_intent` for missing required fields. Other errors carry `legal_fields`.
    A missing or unknown type gets the catalogue's closed type union as `minimal_intent`."""

    def __init__(
        self,
        *,
        pointer: str,
        detail: str,
        minimal_intent: Mapping[str, object] | None = None,
        legal_fields: tuple[str, ...] | None = None,
    ) -> None:
        if (minimal_intent is None) == (legal_fields is None):
            msg = "SchemaInvalid takes exactly one of minimal_intent or legal_fields"
            raise TypeError(msg)
        super().__init__(pointer=pointer, detail=detail)
        self.minimal_intent = minimal_intent
        self.legal_fields = legal_fields


class ConstraintViolation(AssemblerRefusal):
    """A blocking catalogue constraint failed. `pointer` addresses the submitted intent.

    `fix` carries corrective text. `_stat_card_change` raises this at `/change/basis`."""

    def __init__(self, *, constraint_id: str, pointer: str, detail: str, fix: str) -> None:
        super().__init__(pointer=pointer, detail=detail)
        self.constraint_id = constraint_id
        self.fix = fix


class ReleaseMismatch(Exception):
    """A write would move a newer head back to this deployment's older release, or a stored
    payload's shape differs from its release schema.

    `catalog_version` names the head's release on a write and the stored release on a read.
    Details identify the two releases or the stored payload shape."""

    def __init__(self, *, catalog_version: str, detail: str) -> None:
        self.catalog_version = catalog_version
        self.detail = detail
        super().__init__(detail)


class RevisionNotFound(Exception):
    """No revision numbered `revision` exists for `component_id`.

    `get_component` raises `EditConflict` when no current revision exists. The detail names the requested component and revision."""

    def __init__(self, *, component_id: str, revision: int, detail: str) -> None:
        self.component_id = component_id
        self.revision = revision
        self.detail = detail
        super().__init__(detail)


class WriteRefused(Exception):
    """The write token sent with a revise, or with a create into an existing composition, does
    not grant writing `subject_id`. The token was missing or wrong, or the subject predates
    write tokens and so has no stored digest. Maps to `write_refused`.

    `fix` tells the caller to create a new component instead."""

    def __init__(
        self, *, subject_kind: WriteSubjectKind, subject_id: str, detail: str, fix: str
    ) -> None:
        self.subject_kind = subject_kind
        self.subject_id = subject_id
        self.detail = detail
        self.fix = fix
        super().__init__(detail)
