"""Defines the plain-data trace each materialisation returns in its manifest.

`ManifestTrace` extends `dfo_frame_grammar`'s `FrameTrace` and avoids imports
from `facets`, `window`, and `ceiling`. The pipeline converts richer values
through their `as_trace_entry()` methods.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import cast

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedStatVar:
    """One StatVar's ref, role keys, and surviving row count.

    `roles` is `request.declared_roles`. `observation_count` is per member,
    after filtering and duplicate-facet collapse. Group coverage reports the
    combined row count and cannot show which member contributed zero rows.
    """

    ref: str
    roles: tuple[str, ...]
    observation_count: int

    def as_trace_entry(self) -> dict[str, object]:
        """Returns a JSON-safe entry with `roles` as a list."""
        return {
            "ref": self.ref,
            "roles": list(self.roles),
            "observation_count": self.observation_count,
        }


@dataclass(frozen=True)
class MemberFamily:
    """The whole population an identity admits, before antichain reduction.

    `digest` is `statvar_members.member_digest` over the bare dcids of every admitted
    member and `count` their number. `resolved_stat_vars` lists the members served, which
    leave out each admitted member a broader one dominates.
    """

    digest: str
    count: int


@dataclass(frozen=True)
class StatedFamily:
    """What a trace states of the family an identity admits. `members` is the family, `None`
    when member resolution could not state the whole of it, and `stub_candidates` the `dcid:`
    refs the member walk left out of it as stubs (`MemberGroup.stub_candidates`)."""

    members: MemberFamily | None
    stub_candidates: tuple[str, ...] = ()


UNKNOWN_FAMILY = StatedFamily(members=None)
"""The family stated unknown, with a null digest and count and no stub candidates."""


def family_fields(family: StatedFamily) -> dict[str, object]:
    """The trace fields that state `family`: `family_member_digest` and `family_member_count`,
    both null without members, and `stub_candidates`, omitted when there are none."""
    members = family.members
    fields: dict[str, object] = {
        "family_member_digest": members.digest if members is not None else None,
        "family_member_count": members.count if members is not None else None,
    }
    if family.stub_candidates:
        fields["stub_candidates"] = list(family.stub_candidates)
    return fields


def with_family(trace: Mapping[str, object], family: StatedFamily) -> dict[str, object]:
    """`trace` with its family fields (`family_fields`) stating `family` in place of its own."""
    kept = {key: value for key, value in trace.items() if key != "stub_candidates"}
    return {**kept, **family_fields(family)}


def trace_family(trace: Mapping[str, object]) -> StatedFamily:
    """The family a trace's family fields (`family_fields`) state, with no members when the
    digest or count is null or absent, as on a version stored before the fields existed."""
    digest = trace.get("family_member_digest")
    count = trace.get("family_member_count")
    members = (
        MemberFamily(digest=cast(str, digest), count=cast(int, count))
        if digest is not None and count is not None
        else None
    )
    stubs = cast("list[str]", trace.get("stub_candidates", []))
    return StatedFamily(members=members, stub_candidates=tuple(stubs))


@dataclass(frozen=True)
class RecordedFamily:
    """A `StatedFamily` as a store records it: `stated` at the head confirmation
    `confirmed_at`, resolved under `basis` (`datasets_api.adapter.family_basis`)."""

    stated: StatedFamily
    confirmed_at: datetime
    basis: str

    def stated_at(self, confirmed_at: datetime | None, basis: str) -> StatedFamily:
        """The family a response confirmed at `confirmed_at` under the serving `basis` states:
        `stated` when it was recorded at that confirmation under that basis, and
        `UNKNOWN_FAMILY` otherwise, since a later confirmation or another basis may admit
        another family."""
        if self.confirmed_at != confirmed_at or self.basis != basis:
            return UNKNOWN_FAMILY
        return self.stated


def recorded_family_json(recorded: RecordedFamily) -> dict[str, object]:
    """The JSON object a store keeps for `recorded`: its `family_fields`, `confirmed_at` and
    `basis`. `parse_recorded_family` reads it back."""
    return {
        **family_fields(recorded.stated),
        "confirmed_at": recorded.confirmed_at.isoformat(),
        "basis": recorded.basis,
    }


def parse_recorded_family(value: object) -> RecordedFamily | None:
    """The `RecordedFamily` a `recorded_family_json` object holds. `None` for a NULL column and
    for a value this code cannot read, such as one a newer image wrote with other types, so a
    reader states the family unknown and logs the value."""
    if value is None:
        return None
    stored = cast("Mapping[str, object]", value) if isinstance(value, Mapping) else {}
    confirmed_at = _recorded_instant(stored)
    if confirmed_at is None or not _readable_family(stored):
        logger.warning("Stored member family is unreadable; stating it unknown: %r", value)
        return None
    return RecordedFamily(
        stated=trace_family(stored),
        confirmed_at=confirmed_at,
        basis=cast(str, stored["basis"]),
    )


def _recorded_instant(stored: Mapping[str, object]) -> datetime | None:
    """The timezone-aware `confirmed_at` of a stored family, `None` when it holds none."""
    text = stored.get("confirmed_at")
    try:
        instant = datetime.fromisoformat(text) if isinstance(text, str) else None
    except ValueError:
        return None
    return instant if instant is not None and instant.tzinfo is not None else None


def _readable_family(value: Mapping[str, object]) -> bool:
    """Whether every field of a stored family has the type `recorded_family_json` writes."""
    digest = value.get("family_member_digest")
    count = value.get("family_member_count")
    stubs = value.get("stub_candidates", [])
    return (
        (digest is None or isinstance(digest, str))
        and (count is None or (isinstance(count, int) and not isinstance(count, bool)))
        and isinstance(stubs, list)
        and all(isinstance(stub, str) for stub in stubs)
        and isinstance(value.get("basis"), str)
    )


@dataclass(frozen=True)
class ManifestTrace:
    """Plain-data trace record returned with each materialisation.

    Each facet's `selection_reason` is caller-supplied free text.
    """

    applied_window: dict[str, object] | None
    # The graph serving revision the run read (`GenerationStamp.token`), `None` when unknown.
    graph_revision: str | None
    # Each source the run read and its refresh token, sorted by source.
    source_revisions: tuple[dict[str, object], ...]
    resolved_stat_vars: tuple[dict[str, object], ...]
    facets: tuple[dict[str, object], ...]
    normalisations: tuple[dict[str, object], ...]  # Empty because this library applies zero.
    conversion: tuple[dict[str, object], ...] | None
    # Omitted when no conversion runs. Otherwise one entry per recipe and
    # table-version pair, sorted by recipe then table versions: `recipe`,
    # `rate_table_version`, and `deflator_table_version` when a deflator
    # joined, then `members` (each member's
    # bare StatVar ref and native unit, sorted by StatVar), and the members'
    # fallback/projected counts with sorted, deduplicated examples capped at
    # 10, and `left_out` when the conversion left data out: one `{reason, count, of}` per
    # reason and counted kind. One entry per recipe and table-version pair.
    frame_grammar_version: str  # Semver label, e.g. "1.0.0".
    materialiser_build: str
    capability_version: str
    code_index_stamp: dict[str, object] | None = None
    # Index stamp fields, attached by `datasets_api.manifest.assemble_manifest`
    # when available. Omitted from `to_dict()` when `None`.
    origin_catalogue_build_id: str | None = None
    # The origin catalogue build that admitted the manifest's data, attached
    # by `datasets_api.manifest.assemble_manifest`. Omitted from `to_dict()`
    # when `None`, as on versions minted before the catalogue existed.
    family: MemberFamily | None = None
    # The population the definition's identity admits, from the request. `None` when the
    # resolver could not state the whole of it, and on a trace that describes no one family.
    stub_candidates: tuple[str, ...] = ()
    # The `dcid:` refs the member walk left out of the family under a confirmed root because
    # its reader holds their arcs complete and they carry no definition arc
    # (`MemberGroup.stub_candidates`).
    # Omitted from `to_dict()` when empty.
    coverage_gaps: tuple[dict[str, object], ...] = ()
    # The places a national-currency unit binding serves no row at, one entry per role and
    # reason (`national_currency.CoverageGap.as_trace_entry`): `role`, `reason` and the
    # `entities` in dcid order. Omitted from `to_dict()` when empty.
    choices: tuple[dict[str, object], ...] = ()
    # Each axis the deployment's default policy chose a value on over disagreeing candidates
    # (`facets.FacetSelection.choice_entry`): `axis`, `ranking`, and `candidates` in rank
    # order, the chosen value first, each with the `places`, `years` and `observations` the
    # ranking read. Omitted from `to_dict()` when empty.
    continuations: tuple[dict[str, object], ...] = ()
    # Each selection whose series the same source's estimates continue past its last period
    # (`facets.FacetSelection.continued_by`): `axis`, the continuing `method`, and `periods`,
    # the first and last period that method's rows supply in the served frame. Omitted from
    # `to_dict()` when empty.
    origin_gaps: tuple[dict[str, object], ...] = ()
    # The places the origin reads found the answer holds no data at, attached by
    # `datasets_api.serve` (`datasets_api.origin.OriginGap.as_trace_entry`): `code`
    # (`withheld_by_registry` or `un_system_unavailable`), `variable` and `provenances` for a
    # registry withholding, and `entities`. Omitted from `to_dict()` when empty.
    party_totals_left_out: tuple[dict[str, object], ...] = ()
    # The group totals `options.without_party_totals` left out of a version's frame, attached by
    # `datasets_api.serve` (`datasets_api.party_totals.PartyTotalsCut.trace_entries`): one
    # `role`, `provenance` (null for rows with none) and its `entities` in dcid order per open
    # party role and provenance that lost rows. Omitted from `to_dict()` when empty.
    components: tuple[dict[str, object], ...] = ()
    # Each component a composite's frame was composed from
    # (`materialise.composite.ComponentSummary.as_trace_entry`), in declaration order. Empty
    # for a plain definition, and omitted from `to_dict()` then.

    @property
    def family_statement(self) -> StatedFamily:
        """The family this trace states (`family` and `stub_candidates`)."""
        return StatedFamily(members=self.family, stub_candidates=self.stub_candidates)

    def to_dict(self) -> dict[str, object]:
        """Returns the wire shape. Optional fields are omitted when `None`.
        `graph_revision` is always present, null when unknown, `normalisations` is always present,
        including as an empty list, and
        `family_member_digest` and `family_member_count` are always present, null without a
        `family`.
        """
        result: dict[str, object] = {}
        if self.applied_window is not None:
            result["applied_window"] = self.applied_window
        result["graph_revision"] = self.graph_revision
        if self.code_index_stamp is not None:
            result["code_index_stamp"] = self.code_index_stamp
        if self.origin_catalogue_build_id is not None:
            result["origin_catalogue_build_id"] = self.origin_catalogue_build_id
        result["source_revisions"] = list(self.source_revisions)
        result["resolved_stat_vars"] = list(self.resolved_stat_vars)
        result.update(family_fields(self.family_statement))
        result["facets"] = list(self.facets)
        result["normalisations"] = list(self.normalisations)
        if self.conversion is not None:
            result["conversion"] = list(self.conversion)
        if self.coverage_gaps:
            result["coverage_gaps"] = list(self.coverage_gaps)
        if self.choices:
            result["choices"] = list(self.choices)
        if self.continuations:
            result["continuations"] = list(self.continuations)
        if self.origin_gaps:
            result["origin_gaps"] = list(self.origin_gaps)
        if self.party_totals_left_out:
            result["party_totals_left_out"] = list(self.party_totals_left_out)
        if self.components:
            result["components"] = list(self.components)
        result["frame_grammar_version"] = self.frame_grammar_version
        result["materialiser_build"] = self.materialiser_build
        result["capability_version"] = self.capability_version
        return result
