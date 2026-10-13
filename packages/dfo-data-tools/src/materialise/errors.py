"""Materialisation refusals, caller errors, and upstream response failures.
`MaterialisationError` subclasses become typed HTTP problems. Graph and SDMX
infrastructure errors pass through. `UpstreamResponseError` marks inconsistent data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from dfo_platform.problem import Alternative, ProblemIssue, Refusal

if TYPE_CHECKING:
    from materialise.pipeline import MaterialisationRequest

MaterialisationIssue = ProblemIssue
"""Shared pydantic type for one aggregated refusal issue. Fields use attribute access."""


class MaterialisationError(Refusal):
    """Base for deterministic refusals rendered as typed HTTP problems. Each named subclass sets
    `code` and `status`, and `retryable` where a retry can succeed. `status` serves surfaces
    that render the refusal themselves. The Datasets surface reads the code registry's status
    for the endpoint the route serves (`dfo_platform.codes.registered_status`).

    `pointer` names a location in the body a caller posted, as the reshape refusals' `/options`
    paths do. `facet_pointer` locates a facet refusal in the materialisation request
    (`/facets/<axis>`), outside any posted body. Aggregation copies it into an issue, and it
    never becomes the refusal's own pointer."""

    status: int

    def __init__(
        self,
        *args: object,
        pointer: str | None = None,
        facet_pointer: str | None = None,
        alternatives: list[Alternative] | None = None,
        # Aggregated axis refusals carry one `issues` entry per unbound or unavailable axis.
        issues: tuple[MaterialisationIssue, ...] | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(*args)
        self.pointer = pointer
        self.facet_pointer = facet_pointer
        self.alternatives = alternatives
        self.issues = issues
        self.detail = detail


class CollectionProbeCeilingExceededError(MaterialisationError):
    """422 `collection_probe_ceiling`. Raised before any facet probe when a request group's sum of
    distinct StatVars x entities across subject-only scope groups exceeds the pair cap."""

    code = "collection_probe_ceiling"
    status = 422

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class DefaultFacetUndefinedError(MaterialisationError):
    """422 `default_facet_undefined`. Raised when an unbound axis has no default candidate.
    None survives narrowing by the bound axes, or the winning candidate carries no provenance.
    Aggregated member errors give this refusal priority over `facet_not_available`. `request`
    names the member whose selection failed, where one member's selection raised it."""

    code = "default_facet_undefined"
    status = 422

    def __init__(
        self,
        *args: object,
        facet_pointer: str | None = None,
        issues: tuple[MaterialisationIssue, ...] | None = None,
        request: MaterialisationRequest | None = None,
    ) -> None:
        super().__init__(*args, facet_pointer=facet_pointer, issues=issues)
        self.request = request


class FacetNotAvailableError(MaterialisationError):
    """422 `facet_not_available`. Raised when a bound ref or declared default has no
    observations, or when member errors contain no `default_facet_undefined`."""

    code = "facet_not_available"
    status = 422

    def __init__(
        self,
        *args: object,
        facet_pointer: str | None = None,
        alternatives: list[Alternative] | None = None,
        alternatives_truncated: bool = False,
        issues: tuple[MaterialisationIssue, ...] | None = None,
    ) -> None:
        super().__init__(
            *args,
            facet_pointer=facet_pointer,
            alternatives=alternatives,
            issues=issues,
        )
        # `group_facets._as_issue` combines this with its cap check for the wire `truncated` flag.
        self.alternatives_truncated = alternatives_truncated


class FacetBindingUnsupportedError(MaterialisationError):
    """422 `facet_binding_unsupported`. Raised when a recognised facet binding cannot be
    served for the request's shape, as a national-currency unit binding on a group of
    variables or on a role-bearing variable."""

    code = "facet_binding_unsupported"
    status = 422


class OverCeilingError(MaterialisationError):
    """422 `over_ceiling`. Raised by row, byte, member-cardinality, and candidate-window checks.

    `ceiling_kind` identifies the unit. Narrower scope can clear row/byte limits. Identity
    fixes member-cardinality and candidate-window limits. A candidate-window refusal means
    the bounded reverse-arc read cannot establish the full member count.

    `count` and `cap` use that unit. Unmeasured values are `None`. Current checks measure both.
    For truncated reverse-arc reads, `count` is returned candidates and `cap` is the read
    window, with `count <= cap`."""

    code = "over_ceiling"
    status = 422

    def __init__(
        self,
        *args: object,
        detail: str | None = None,
        ceiling_kind: str,
        count: int | None = None,
        cap: int | None = None,
    ) -> None:
        super().__init__(*args, detail=detail)
        self.ceiling_kind = ceiling_kind
        self.count = count
        self.cap = cap


class MembersNotHeldError(MaterialisationError):
    """500 `members_not_held`. Raised when SDMX rejects one fetch chunk's StatVars as having
    incompatible shapes and a graph probe finds some of them have no node. `stat_vars` holds
    that chunk's absent bare dcids, sorted. StatVars in other chunks are not probed."""

    code = "members_not_held"
    status = 500

    def __init__(self, *, stat_vars: Sequence[str]) -> None:
        self.stat_vars = tuple(sorted(stat_vars))
        detail = (
            f"{len(self.stat_vars)} StatVar(s) in a rejected SDMX fetch have no node in the "
            f"knowledge graph: {list(self.stat_vars)!r}. Other StatVars this definition "
            "names may also be absent."
        )
        issues = tuple(ProblemIssue(member=stat_var) for stat_var in self.stat_vars)
        super().__init__(detail, detail=detail, issues=issues)


class DatasetDisagreementError(MaterialisationError):
    """500 `internal_error`. Raised when the facets found for a member of an index-resolved
    family disagree with the dataset elected for that family: the probe finds the member only
    under provenances the code index does not record for it, or the selected provenance
    differs from the elected one. The code index and the series coverage it was joined from
    have drifted apart. `dataset` is the elected provenance and `stat_var` the member."""

    code = "internal_error"
    status = 500

    def __init__(self, detail: str, *, dataset: str, stat_var: str) -> None:
        self.dataset = dataset
        self.stat_var = stat_var
        super().__init__(detail, detail=detail, issues=(ProblemIssue(member=stat_var),))


class MixedGrainError(MaterialisationError):
    """500 `internal_error`. Raised when the rows a materialisation serves hold periods of
    more than one grain, as `2019` beside `2019-01`, which no frame binds
    (`assembler.frame_view`). The election dispatches one cadence
    (`facets.dispatch_cadence`), so a facet stating no observation period dated its rows at
    two grains. `grains` maps each grain to its earliest period, and `stat_vars` names the
    request's members."""

    code = "internal_error"
    status = 500

    def __init__(self, grains: Mapping[str, str], *, stat_vars: Sequence[str]) -> None:
        self.grains = dict(grains)
        self.stat_vars = tuple(stat_vars)
        detail = (
            f"the rows served for {list(self.stat_vars)!r} mix {' and '.join(self.grains)} "
            f"periods, first {', '.join(repr(period) for period in self.grains.values())}; "
            "a frame holds periods at one grain"
        )
        super().__init__(detail, detail=detail)


class UpstreamVersionChangedError(MaterialisationError):
    """503 `upstream_version_changed`, retryable. Raised when two generations disagree."""

    code = "upstream_version_changed"
    status = 503
    retryable = True

    def __init__(self, detail: str) -> None:
        super().__init__(
            detail,
            detail=detail,
        )


class MaterialisationUsageError(ValueError):
    """Caller-side input bug, raised before network calls. `stat_var` values include the
    `dcid:` prefix."""


class UpstreamResponseError(Exception):
    """Marks upstream data that conflicts with a valid request. `frame_build` raises when a
    row's `variableMeasured` differs from the requested StatVar. Generation raises when
    response roles differ from declared roles, which can drift between resolution and fetch.
    This marks upstream data breakage for a valid request."""


class VariableBreadthUnreadError(Exception):
    """A single-place election's facets tie on every key before whole-variable breadth, the
    order the breadth would give them decides the axes or the cadence the election serves, and
    the breadth could not be read (`facets.rank_candidates`, `facets.dispatch_cadence`). Only a
    caller that stores its election refuses on an unread breadth
    (`facets.bound_variable_breadth(refuse_unread=True)`), rather than record another facet.

    `transient` is true when the read failed in a way a later read can answer
    (`facets.BreadthUnread.UNREAD`), so a Datasets mint answers 503 and a retry reads again,
    and false when the variable's own profile raised for another reason
    (`facets.BreadthUnread.DEFECT`), a defect a retry would meet again, so it answers 500. Not
    a `MaterialisationError`, so per-axis aggregation does not fold it into a facet refusal."""

    def __init__(self, candidates: Sequence[str], *, transient: bool = True) -> None:
        self.candidates = tuple(candidates)
        self.transient = transient
        cause = "could not be read; retry" if transient else "could not be built"
        super().__init__(
            f"facets {list(self.candidates)!r} tie at one place and their whole-variable "
            f"breadth {cause}"
        )
