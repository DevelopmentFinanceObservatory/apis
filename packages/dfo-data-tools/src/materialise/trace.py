"""Defines the plain-data trace each materialisation returns in its manifest.

`ManifestTrace` extends `dfo_frame_grammar`'s `FrameTrace` and avoids imports
from `facets`, `window`, and `ceiling`. The pipeline converts richer values
through their `as_trace_entry()` methods.
"""

from __future__ import annotations

from dataclasses import dataclass


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
        result["family_member_digest"] = self.family.digest if self.family is not None else None
        result["family_member_count"] = self.family.count if self.family is not None else None
        if self.stub_candidates:
            result["stub_candidates"] = list(self.stub_candidates)
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
