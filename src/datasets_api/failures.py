"""Classify the store, graph, SDMX, artifact and frame failures a Datasets door meets that are not
refusals. A retry may clear them, or they are faults on this side.

Each `_Failure` row names its classes, the code and detail the problem carries, whether a retry
can clear it, and the line it logs, if any. A failure maps by its most specific class, so a row
for a subclass outranks its base's whatever the order. The rows read as the mint words them.
Classifying logs nothing: each problem names its line (`ProblemSpec.log`), which the request
path logs (`dfo_platform.problem.log_problem`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, NamedTuple

from dfo_frame_grammar import FrameError

from artifact_store import (
    ARTIFACT_INVENTORY_WRITE_FAILED_MARKER,
    ArtifactInventoryError,
    ArtifactRecordConflictError,
    ArtifactSigningError,
    ArtifactWriteError,
)
from dc_sdmx import (
    SdmxCapExceededError,
    SdmxContainmentFilterError,
    SdmxHttpError,
    SdmxIncompatibleVariablesError,
    SdmxMissingFilterError,
    SdmxQueryError,
    SdmxResponseError,
    SdmxRoleCeilingError,
    SdmxTransportError,
    SdmxUnsupportedFilterError,
)
from dfo_platform import ProblemSpec
from dfo_platform.codes import registered_status
from materialise.errors import MixedGrainError, UpstreamResponseError, VariableBreadthUnreadError
from materialise.graph import GraphDeadlineExceededError, GraphReadError
from statvar_members import MemberReadDeadline

_WRITE_FAILED = "writing an artifact to storage failed; retry"
_LOOKUP_FAILED = "reading the artifact inventory failed; retry"


class FoldPermitWaitExpired(TimeoutError):
    """A render's deadline passed while its groupby waited for a fold permit
    (`datasets_api.render`)."""


class _Failure(NamedTuple):
    classes: tuple[type[BaseException], ...]
    code: str
    # `{exc}` stands for the exception's message.
    detail: str
    retryable: bool
    log: str | None = None

    def classify(self, exc: BaseException) -> ProblemSpec:
        detail = self.detail.format(exc=exc)
        spec = ProblemSpec(registered_status(self.code), self.code, self.retryable, detail)
        if self.log is None:
            return spec
        return spec._replace(log=f"Datasets door {self.log}: {exc}", log_traceback=True)


_FAILURES = (
    _Failure(
        (GraphDeadlineExceededError,),
        "timeout",
        "the request's deadline expired during the graph facet probe; retry",
        retryable=True,
    ),
    # A walk out of time says nothing about the size of the set.
    _Failure(
        (MemberReadDeadline,),
        "timeout",
        "{exc}; the member count past it is unknown, not zero; retry",
        retryable=True,
    ),
    # The fold the collapse waited for may finish within a retry's deadline.
    _Failure((FoldPermitWaitExpired,), "timeout", "{exc}; retry", retryable=True),
    _Failure(
        (GraphReadError,),
        "upstream_error",
        "the graph read this mint needed failed upstream; retry",
        retryable=True,
        log="graph read failed",
    ),
    # Invalid role bindings and the mixer's three role slots are the caller's to change.
    _Failure((SdmxQueryError, SdmxRoleCeilingError), "request_invalid", "{exc}", retryable=False),
    # A door composes no query of these shapes, so one is a composition bug.
    _Failure(
        (
            SdmxMissingFilterError,
            SdmxUnsupportedFilterError,
            SdmxContainmentFilterError,
            SdmxIncompatibleVariablesError,
        ),
        "internal_error",
        "{exc}",
        retryable=False,
        log="composed an SDMX query the mixer refused as malformed",
    ),
    _Failure(
        (SdmxTransportError, SdmxHttpError, SdmxResponseError),
        "upstream_error",
        "the SDMX read this mint needed failed upstream; retry",
        retryable=True,
        log="SDMX read failed",
    ),
    _Failure(
        (UpstreamResponseError,),
        "upstream_error",
        "the upstream response for this mint was inconsistent with the request; retry",
        retryable=True,
        log="read an upstream response inconsistent with its request",
    ),
    _Failure(
        (FrameError,),
        "internal_error",
        "the frame this request builds does not conform to the frame grammar: {exc}",
        retryable=False,
        log="built a frame the grammar refuses",
    ),
    _Failure(
        (ArtifactSigningError,),
        "service_unavailable",
        "signing an artifact's download url failed; retry",
        retryable=True,
        log="caught an unattributed signing failure",
    ),
    _Failure(
        (ArtifactWriteError,),
        "service_unavailable",
        _WRITE_FAILED,
        retryable=True,
        log="artifact store write failed",
    ),
)


def _sdmx_cap_exceeded(exc: SdmxCapExceededError) -> ProblemSpec:
    # A read past a caller's cap is a size refusal, as the raw-fetch guard's `over_ceiling` is.
    detail = (
        f"the SDMX read was aborted after {exc.observed} {exc.cap_kind}, past its cap of "
        f"{exc.cap} {exc.cap_kind}"
    )
    spec = _Failure((), "over_ceiling", detail, retryable=False).classify(exc)
    return spec._replace(ceiling_kind=exc.cap_kind, count=exc.observed, cap=exc.cap)


def _record_conflict(exc: ArtifactRecordConflictError) -> ProblemSpec:
    # Render attaches the subject and version. Other raise sites leave them unset.
    log = (
        f"Datasets artifact write hit a record conflict for subject "
        f"{getattr(exc, 'subject_id', None)}, content_version "
        f"{getattr(exc, 'content_version', None)}, digest {exc.digest}, tier {exc.tier.value}: "
        "the uploaded object has no inventory row pointing at it and needs manual reconciliation"
    )
    spec = _Failure((), "internal_error", "{exc}", retryable=False).classify(exc)
    return spec._replace(log=log, log_level=logging.WARNING)


def _mixed_grain(exc: MixedGrainError) -> ProblemSpec:
    # A facet stating no observation period dated its rows at two grains. The problem carries
    # the catch-all's code and no detail.
    return ProblemSpec(
        registered_status("internal_error"),
        "internal_error",
        False,
        log=f"Datasets door served rows of two grains: {exc}",
        log_traceback=True,
    )


def _inventory_failed(exc: ArtifactInventoryError) -> ProblemSpec:
    # A failed write orphans the uploaded bytes and logs the reconciliation marker. A failed
    # lookup carries no bucket or object.
    if exc.bucket is None or exc.object_name is None:
        log = "artifact inventory lookup failed"
        return _Failure((), "service_unavailable", _LOOKUP_FAILED, True, log).classify(exc)
    tier = exc.tier.value if exc.tier is not None else None
    spec = _Failure((), "service_unavailable", _WRITE_FAILED, retryable=True).classify(exc)
    return spec._replace(
        log=(
            f"{ARTIFACT_INVENTORY_WRITE_FAILED_MARKER} digest={exc.digest} tier={tier} "
            f"bucket={exc.bucket} object_name={exc.object_name}"
        )
    )


def _variable_breadth_unread(exc: VariableBreadthUnreadError) -> ProblemSpec:
    # A mint stores its election, so it refuses rather than record the facet the unit's name
    # would pick. A retry reads a transient failure again; a defect would fail alike.
    if not exc.transient:
        spec = _Failure((), "internal_error", "{exc}", retryable=False).classify(exc)
        return spec._replace(
            log=f"Datasets mint refused a single-place tie on a defective breadth: {exc}"
        )
    spec = _Failure((), "service_unavailable", "{exc}", retryable=True).classify(exc)
    return spec._replace(
        log=f"Datasets mint refused a single-place tie it could not break: {exc}",
        log_level=logging.WARNING,
    )


# Each function takes the one class it is keyed by. `Any` fits the contravariant slot.
_CLASSIFIERS: dict[type, Callable[[Any], ProblemSpec]] = {
    **{cls: failure.classify for failure in _FAILURES for cls in failure.classes},
    SdmxCapExceededError: _sdmx_cap_exceeded,
    ArtifactRecordConflictError: _record_conflict,
    ArtifactInventoryError: _inventory_failed,
    MixedGrainError: _mixed_grain,
    VariableBreadthUnreadError: _variable_breadth_unread,
}


def classify_failure(exc: BaseException) -> ProblemSpec | None:
    """Classify `exc` by its most specific class with a row, or return `None`."""
    for cls in type(exc).__mro__:
        if cls in _CLASSIFIERS:
            return _CLASSIFIERS[cls](exc)
    return None
