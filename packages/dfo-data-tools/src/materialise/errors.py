"""Materialisation refusals, caller errors, and upstream response failures.
`MaterialisationError` subclasses become typed HTTP problems. Graph and SDMX
infrastructure errors pass through. `UpstreamResponseError` marks inconsistent data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from dfo_platform.problem import Alternative, ProblemIssue

if TYPE_CHECKING:
    from materialise.pipeline import MaterialisationRequest

MaterialisationIssue = ProblemIssue
"""Shared pydantic type for one aggregated refusal issue. Fields use attribute access."""


class MaterialisationError(Exception):
    """Base for deterministic refusals rendered as typed HTTP problems. Raise a named
    subclass so the code and status match the refusal."""

    def __init__(
        self,
        *args: object,
        code: str,
        status: int,
        pointer: str | None = None,
        alternatives: list[Alternative] | None = None,
        # Aggregated axis refusals carry one `issues` entry per unbound or unavailable axis.
        issues: tuple[MaterialisationIssue, ...] | None = None,
        detail: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(*args)
        self.code = code
        self.status = status
        self.pointer = pointer
        self.alternatives = alternatives
        self.issues = issues
        self.detail = detail
        self.retryable = retryable


class CollectionProbeCeilingExceededError(MaterialisationError):
    """422 `collection_probe_ceiling`. Raised before any facet probe when a collection's sum of
    distinct StatVars x entities across subject-only scope groups exceeds the pair cap."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args, code="collection_probe_ceiling", status=422)


class DefaultFacetUndefinedError(MaterialisationError):
    """422 `default_facet_undefined`. Raised when an unbound axis has no default candidate.
    Aggregated member errors give this refusal priority over `facet_not_available`. `request`
    names the member whose selection failed, where one member's selection raised it."""

    def __init__(
        self,
        *args: object,
        pointer: str | None = None,
        issues: tuple[MaterialisationIssue, ...] | None = None,
        request: MaterialisationRequest | None = None,
    ) -> None:
        super().__init__(
            *args, code="default_facet_undefined", status=422, pointer=pointer, issues=issues
        )
        self.request = request


class FacetNotAvailableError(MaterialisationError):
    """422 `facet_not_available`. Raised when a bound ref or declared default has no
    observations, or when member errors contain no `default_facet_undefined`.
    When no member of a group holds a candidate with a facet id, `empty_members` names them
    all, so a caller can tell an empty answer from a read that failed, as `request` lets it
    for `DefaultFacetUndefinedError`. It is empty for every other refusal."""

    def __init__(
        self,
        *args: object,
        pointer: str | None = None,
        alternatives: list[Alternative] | None = None,
        alternatives_truncated: bool = False,
        issues: tuple[MaterialisationIssue, ...] | None = None,
        empty_members: Sequence[MaterialisationRequest] = (),
    ) -> None:
        super().__init__(
            *args,
            code="facet_not_available",
            status=422,
            pointer=pointer,
            alternatives=alternatives,
            issues=issues,
        )
        # `group_facets._as_issue` combines this with its cap check for the wire `truncated` flag.
        self.alternatives_truncated = alternatives_truncated
        self.empty_members = tuple(empty_members)


class OverCeilingError(MaterialisationError):
    """422 `over_ceiling`. Raised by row, byte, member-cardinality, and candidate-window checks.

    `ceiling_kind` identifies the unit. Narrower scope can clear row/byte limits. Identity
    fixes member-cardinality and candidate-window limits. A candidate-window refusal means
    the bounded reverse-arc read cannot establish the full member count.

    `count` and `cap` use that unit. Unmeasured values are `None`. Current checks measure both.
    For truncated reverse-arc reads, `count` is returned candidates and `cap` is the read
    window, with `count <= cap`."""

    def __init__(
        self,
        *args: object,
        detail: str | None = None,
        ceiling_kind: str,
        count: int | None = None,
        cap: int | None = None,
    ) -> None:
        super().__init__(*args, code="over_ceiling", status=422, detail=detail)
        self.ceiling_kind = ceiling_kind
        self.count = count
        self.cap = cap


class MembersNotHeldError(MaterialisationError):
    """500 `members_not_held`. Raised when SDMX rejects one fetch chunk's StatVars as having
    incompatible shapes and a graph probe finds some of them have no node. `stat_vars` holds
    that chunk's absent bare dcids, sorted. StatVars in other chunks are not probed."""

    def __init__(self, *, stat_vars: Sequence[str]) -> None:
        self.stat_vars = tuple(sorted(stat_vars))
        detail = (
            f"{len(self.stat_vars)} StatVar(s) in a rejected SDMX fetch have no node in the "
            f"knowledge graph: {list(self.stat_vars)!r}. Other StatVars this definition "
            "names may also be absent."
        )
        super().__init__(detail, code="members_not_held", status=500, detail=detail)


class DatasetDisagreementError(MaterialisationError):
    """500 `internal_error`. Raised when the facets found for a member of an index-resolved
    family disagree with the dataset elected for that family: the probe finds the member only
    under provenances the code index does not record for it, or the selected provenance
    differs from the elected one. The code index and the series coverage it was joined from
    have drifted apart. `dataset` is the elected provenance and `stat_var` the member."""

    def __init__(self, detail: str, *, dataset: str, stat_var: str) -> None:
        self.dataset = dataset
        self.stat_var = stat_var
        super().__init__(detail, code="internal_error", status=500, detail=detail)


class UpstreamVersionChangedError(MaterialisationError):
    """503 `upstream_version_changed`, retryable. Raised when two generations disagree."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            detail,
            code="upstream_version_changed",
            status=503,
            detail=detail,
            retryable=True,
        )


class MaterialisationUsageError(ValueError):
    """Caller-side input bug, raised before network calls. Collection members need unique
    `spec_id`s, and `stat_var` values include the `dcid:` prefix."""


class UpstreamResponseError(Exception):
    """Marks upstream data that conflicts with a valid request. `frame_build` raises when a
    row's `variableMeasured` differs from the requested StatVar. Generation raises when
    response roles differ from declared roles, which can drift between resolution and fetch.
    This marks upstream data breakage for a valid request."""
