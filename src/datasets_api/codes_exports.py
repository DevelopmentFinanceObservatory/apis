"""Refusal codes the exports door owns, and its refusals.

`provenance_unavailable` and `licence_not_redistributable` are the door's own. Its other refusals
reuse codes other doors own: `over_ceiling` (`codes_materialise`), and the platform's
`not_found` and `allowance_exhausted`. `NewestVersionUnservable` carries whichever registered
code stands for why a bare `spec_id` could not be resolved.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from dfo_platform.problem import ProblemIssue, Refusal

EXPORTS_DOOR_CODE_CATALOGUE: dict[str, tuple[str, str]] = {
    "provenance_unavailable": (
        "Provenance Unavailable",
        "The sources of this derived frame and their licences cannot be named, so it cannot be "
        "exported.",
    ),
    "licence_not_redistributable": (
        "Licence Not Redistributable",
        "A source of this frame is listed as one whose licence does not allow redistribution, "
        "so the frame cannot be exported and its export link is withdrawn.",
    ),
}


class ProvenanceUnavailable(Refusal):
    """A derived frame with no stored recipe, a recipe with no roots, or a root that cannot be
    cited, leaves a source unnamed, so its licence cannot be checked. Maps to
    `provenance_unavailable`, 422."""

    code = "provenance_unavailable"

    def __init__(self, *, spec_id: str, content_version: str) -> None:
        self.spec_id = spec_id
        self.content_version = content_version
        super().__init__(
            f"the sources of {spec_id!r} at content_version {content_version!r} cannot all be "
            "named, so their licences are unknown and it cannot be exported"
        )


class LicenceNotRedistributable(Refusal):
    """A source the export names is on the licence deny-list (`exports.restricted_sources`).
    Maps to `licence_not_redistributable`, 403."""

    code = "licence_not_redistributable"

    def __init__(self, *, sources: Sequence[tuple[str, str]]) -> None:
        self.sources = tuple(sources)
        named = "; ".join(f"{ref} ({reason})" for ref, reason in self.sources)
        super().__init__(f"the licence of these sources does not allow redistribution: {named}")


class ExportOverCeiling(Refusal):
    """A CSV or xlsx export of more rows than `exports.EXPORT_MAX_ROWS`. Maps to `over_ceiling`
    with `ceiling_kind` `rows`. A Parquet export of the same frame has no row ceiling."""

    code = "over_ceiling"
    ceiling_kind = "rows"
    decided_by_request = True

    def __init__(self, *, format: str, count: int, cap: int) -> None:
        self.count = count
        self.cap = cap
        super().__init__(
            f"a {format} export holds at most {cap} rows and this frame has {count}; export it "
            "as parquet"
        )


class ExportFileIncomplete(Refusal):
    """The file a render gave is linked incomplete (`render.incomplete_rendering`): a graph
    metadata read failed, and an export is kept for good, so none is prepared. Maps to
    `service_unavailable`, 503. A retry renders again once the read answers."""

    code = "service_unavailable"
    retryable = True

    def __init__(self, *, id: str, content_version: str) -> None:
        self.id = id
        self.content_version = content_version
        super().__init__(
            f"the file of {id!r} at content_version {content_version!r} rendered without its "
            "sources' documentation; retry"
        )


class ExportPrepareRunning(Refusal):
    """Another prepare of the export still runs when the caller's wait for it ends (`wait_seconds`,
    `materialise_wait_seconds`). Maps to `service_unavailable`, 503, with `Retry-After` of that
    wait, since a prepare that outlasted one wait has no reason to end within a second."""

    code = "service_unavailable"
    retryable = True

    def __init__(self, *, wait_seconds: float) -> None:
        self.headers = MappingProxyType({"Retry-After": str(math.ceil(wait_seconds))})
        super().__init__("another prepare of this export is still running; retry")


class ExportFileRefused(Refusal):
    """The refusal a concurrent prepare's file gave, read back from the prepare's claim
    (`allowance.ExportClaim.refusal`) and rendered as that prepare's was. It carries the code
    that refusal registered."""

    def __init__(
        self,
        *,
        code: str,
        detail: str,
        pointer: str | None = None,
        ceiling_kind: str | None = None,
        count: int | None = None,
        cap: int | None = None,
        issues: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.code = code
        self.pointer = pointer
        self.ceiling_kind = ceiling_kind
        self.count = count
        self.cap = cap
        self.issues = [ProblemIssue.model_validate(issue) for issue in issues] if issues else None
        super().__init__(detail)

    @classmethod
    def facts_of(cls, refusal: Refusal) -> dict[str, Any]:
        """The keyword arguments that rebuild `refusal`, as JSON, with no traceback."""
        return {
            "code": refusal.code,
            "detail": refusal.detail or str(refusal),
            "pointer": refusal.pointer,
            "ceiling_kind": refusal.ceiling_kind,
            "count": refusal.count,
            "cap": refusal.cap,
            "issues": [issue.model_dump(mode="json") for issue in refusal.issues or ()],
        }


class ExportNotFound(Refusal):
    """No export is recorded under the id. Maps to the platform's `not_found`, 404."""

    code = "not_found"

    def __init__(self, *, export_id: str) -> None:
        self.export_id = export_id
        super().__init__(f"no export is recorded under {export_id!r}")


class NewestVersionUnservable(Refusal):
    """A bare `spec_id`'s newest version cannot be served (`exports.resolve_export_frame`).
    It carries a code `datasets.exports` registers: `service_unavailable` (503, retryable)
    while the version builds or for a cause a retry may clear, `internal_error` (500) for a
    server fault, and `snapshot_not_found` (404) for any other cause, whose own code and
    detail its detail names."""

    def __init__(
        self,
        *,
        code: str,
        detail: str,
        retryable: bool = False,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.code = code
        self.retryable = retryable
        self.headers = headers
        super().__init__(detail)
