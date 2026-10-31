"""Errors for `GraphClient` reads from `/v2/node`, `/v2/observation` and `/v2/resolve`.

Callers may supply subclasses through `read_error_cls`, `deadline_error_cls` and
`cap_error_cls`. Each must preserve the matching constructor signature. A stuck cursor
raises `GraphNotAdvancingError` itself, and dcids a re-request still omits raise
`read_error_cls` from a `GraphOmittedDcidsError` naming them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol


class ReadErrorFactory(Protocol):
    """Constructor protocol for read and deadline error subclasses."""

    def __call__(self, *args: object, upstream_status: int | None = None) -> Exception: ...


class CapErrorFactory(Protocol):
    """Constructor protocol for cap error subclasses."""

    def __call__(
        self, message: str, *, cap_kind: Literal["bytes"], cap: int, observed: int
    ) -> Exception: ...


class GraphReadError(Exception):
    """Read failure from transport, HTTP, JSON or paging protocol. `upstream_status` is the HTTP
    status for non-2xx responses and `None` when no response arrived.
    """

    def __init__(self, *args: object, upstream_status: int | None = None) -> None:
        super().__init__(*args)
        self.upstream_status = upstream_status


class GraphDeadlineExceededError(GraphReadError):
    """Raised when the absolute monotonic deadline expires before a call or bounds an httpx timeout.
    A timeout under the configured limit remains an ordinary retryable failure while deadline
    budget remains.
    """


class GraphNotAdvancingError(GraphReadError):
    """Raised when `nextToken` repeats or `node_subjects` sees 3 fresh-token pages with no new
    dcids. `dcid` and `property` identify the stuck read.
    """

    def __init__(
        self,
        *args: object,
        dcid: str,
        property: str,
        upstream_status: int | None = None,
    ) -> None:
        super().__init__(*args, upstream_status=upstream_status)
        self.dcid = dcid
        self.property = property


class GraphOmittedDcidsError(GraphReadError):
    """The cause of the read error raised when the graph leaves `dcids` out of a response and
    again out of a re-request scoped to them, so a later read would leave them out too."""

    def __init__(
        self, *args: object, dcids: Sequence[str], upstream_status: int | None = None
    ) -> None:
        super().__init__(*args, upstream_status=upstream_status)
        self.dcids = tuple(dcids)


class GraphCapExceededError(Exception):
    """Raised when the decoded response body exceeds `max_bytes` while streaming, before parsing.
    `cap_kind` is `"bytes"`. `observed` counts bytes consumed before abort and is a lower bound on
    the full body size.
    """

    def __init__(
        self, message: str, *, cap_kind: Literal["bytes"], cap: int, observed: int
    ) -> None:
        super().__init__(message)
        self.cap_kind = cap_kind
        self.cap = cap
        self.observed = observed
