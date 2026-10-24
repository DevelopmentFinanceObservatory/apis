"""Registry of materialisation refusal and warning codes, using the shared wire vocabulary."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from dfo_platform.codes import CodeCatalogue
from dfo_platform.diagnostics import WarningSeverity


class UnservedMembers(TypedDict):
    """The group members of one kind that served nothing: how many, and the `dcid:` refs of
    the first five in request order."""

    count: int
    refs: list[str]


class MaterialisationWarning(TypedDict):
    """One `{code, severity, message}` warning entry, as carried on
    `MaterialisationResult.warnings` (pipeline.py). `severity` is always one
    of the shared `info`/`warn`/`degraded` severity vocabulary
    (`dfo_platform.diagnostics.WarningSeverity`), always `"warn"` for the
    warnings this library raises.

    A group's `coverage_below_scope` also carries `member_count`, the series the group
    requested, with `off_cadence_members`, those reporting the requested years at another
    frequency than the group serves, and `out_of_window_members`, those holding no observation
    in the requested years, each present only when it names a member.

    This is a `TypedDict`. No shared platform pydantic warning model
    exists yet to alias.
    """

    code: str
    severity: WarningSeverity
    message: str
    member_count: NotRequired[int]
    off_cadence_members: NotRequired[UnservedMembers]
    out_of_window_members: NotRequired[UnservedMembers]


MATERIALISE_CODE_CATALOGUE: CodeCatalogue = {
    "entities_unbound": (
        "Entities Unbound",
        "The request binds no entity dcid to probe for a facet candidate set.",
    ),
    "default_facet_undefined": (
        "Default Facet Undefined",
        "An unbound facet axis has no probed candidate to draw a default from.",
    ),
    "facet_not_available": (
        "Facet Not Available",
        "The requested facet does not resolve to any observations.",
    ),
    "over_ceiling": (
        "Over Ceiling",
        "This request exceeds the sync materialisation size ceiling.",
    ),
    "collection_probe_ceiling": (
        "Collection Probe Ceiling",
        "The collection's facet probe would exceed the cap on StatVar and entity "
        "pairs it may ask the graph for, so no probe is issued and the request "
        "is refused.",
    ),
    "members_not_held": (
        "Members Not Held",
        "One or more StatVars a definition names have no node in the knowledge graph, so "
        "its frame cannot be retrieved. The refusal lists those in the rejected fetch. "
        "Others in the same definition may also be absent.",
    ),
    "upstream_version_changed": (
        "Upstream Version Changed",
        "The upstream source refreshed mid-materialisation and the retry did not converge.",
    ),
    "capability_unavailable": (
        "Capability Unavailable",
        "A well-formed request asks for a conversion class this deployment does not enable.",
    ),
    "coverage_below_scope": (
        "Coverage Below Scope",
        "The served coverage falls short of the requested scope.",
    ),
    "empty_result": (
        "Empty Result",
        "The materialised frame holds zero observations.",
    ),
    "deflator_projected": (
        "Deflator Projected",
        "A converted observation's exchange rate or price deflator at the observation "
        "year or the base year came from a projection facet rather than an actual.",
    ),
    "currency_unmapped_omitted": (
        "Currency Unmapped Omitted",
        "A conversion into each entity's own currency left out the rows of the entities "
        "that no single currency names through nationalCurrencyOf.",
    ),
    "deflator_missing_omitted": (
        "Deflator Missing Omitted",
        "A constant-price conversion left out each entity's rows in the years the deflator "
        "table holds no value for it, and every row of an entity it holds no value for in "
        "the base year.",
    ),
}
