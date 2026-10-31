"""Classify failures from direct writes to `dataset_store`.

`LiveVersionRecordWriter` and `LiveArtifactReferenceWriter` wrap driver
errors as `DatasetsStoreUnavailable`. SQLSTATE class determines whether a
failure is permanent. Transient writes retry safely because version inserts
and artifact-reference upserts are idempotent. Permanent errors map to
non-retryable 500 responses. Other failures map to retryable 503 responses.

A spec registry read or write that fails in its store raises
`SpecRegistryStoreError`, which maps to a retryable 503 response on every door.
"""

from __future__ import annotations

import logging

from dfo_frame_grammar import FrameError

from dfo_platform import ProblemSpec
from dfo_platform.store_errors import store_failure_is_permanent
from materialise import formats
from materialise.formats import Frame
from spec_registry.errors import SpecRegistryStoreError

logger = logging.getLogger(__name__)


class DatasetsStoreUnavailable(Exception):
    """Safe summary of a store failure, with its SQLSTATE class.

    Wrap sites raise after leaving the driver's `except` block, so the
    The original exception and its bound SQL parameters stay out of caller
    and log text. `sqlstate` is `None` when the driver supplied none. The classifier
    maps permanent failures to non-retryable 500 and transient failures to
    retryable 503 responses. `connection_broken` says the connection itself failed
    (`dataset_store.default_is_broken`), so a retry on a fresh one can pass.
    """

    def __init__(
        self, message: str, *, sqlstate: str | None = None, connection_broken: bool = False
    ) -> None:
        self.sqlstate = sqlstate
        self.connection_broken = connection_broken
        super().__init__(message)


class StoredFrameUnreadable(Exception):
    """A stored frame artifact does not read as a frame (`read_stored_frame`), a stored-artifact
    fault. It is classified by no door, so it answers `internal_error` as a stored canonical
    frame that does not decode does, and its message names only the digest."""

    def __init__(self, *, digest: str) -> None:
        self.digest = digest
        super().__init__(f"stored frame artifact_digest {digest!r} does not read as a frame")


def read_stored_frame(data: bytes, *, media_type: str, digest: str) -> Frame:
    """A version's stored frame `data`, at `digest`, as a `Frame`: its frame1 document when
    `media_type` names one, read by `read_frame1`, else its canonical frame. The one way a
    Datasets reader reads a stored frame, so every door refuses the same artifact alike.

    Raises:
        StoredFrameUnreadable: `materialise.formats.read_stored_frame` raised `FrameError`.
    """
    try:
        return formats.read_stored_frame(data, media_type=media_type)
    except FrameError as exc:
        raise StoredFrameUnreadable(digest=digest) from exc


def classify_store_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, DatasetsStoreUnavailable):
        if store_failure_is_permanent(exc.sqlstate):
            logger.error("Datasets store write failed with a permanent cause: %s", exc)
            return ProblemSpec(
                status=500,
                code="internal_error",
                detail="a datasets store write failed with a non-retryable cause",
                retryable=False,
            )
        logger.error("Datasets store write failed: %s", exc)
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="a datasets store write failed; retry",
            retryable=True,
        )
    if isinstance(exc, SpecRegistryStoreError):
        logger.error("Spec registry store call failed: %s", exc)
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="a spec registry store call failed; retry",
            retryable=True,
        )
    return None
