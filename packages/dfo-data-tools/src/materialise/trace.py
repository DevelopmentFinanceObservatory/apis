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
    graph_revision: str
    source_revisions: tuple[dict[str, object], ...]
    resolved_stat_vars: tuple[dict[str, object], ...]
    facets: tuple[dict[str, object], ...]
    normalisations: tuple[dict[str, object], ...]  # Empty because this library applies zero.
    conversion: tuple[dict[str, object], ...] | None
    # Omitted when no conversion runs. Otherwise one entry per recipe and
    # table-version pair, sorted by recipe then table versions: `recipe`,
    # `rate_table_version`, and `deflator_table_version` when a deflator
    # joined (the grammar's preimage keys), then `members` (each member's
    # bare StatVar ref and native unit, sorted by StatVar), and the members'
    # fallback/projected counts with sorted, deduplicated examples capped at
    # 10. A definition has at most one entry; a collection mixing recipes or
    # table versions has one per pair.
    frame_grammar_version: str  # Semver label, e.g. "1.0.0".
    materialiser_build: str
    capability_version: str
    code_index_stamp: dict[str, object] | None = None
    # Index stamp fields, attached by `datasets_api.manifest.assemble_manifest`
    # when available. Omitted from `to_dict()` when `None`.
    mint_facets: tuple[dict[str, object], ...] | None = None
    # The exact `{axis, ref, basis}` facets used to mint `content_version`,
    # distinct from the wider `facets` trace entries. Conversion requests
    # retain their requested unit here so `mint_preimage_from_trace` can
    # rebuild the preimage without a graph read. `None` means the preimage
    # cannot be safely reconstructed.
    origin_catalogue_build_id: str | None = None
    # The origin catalogue build that admitted the manifest's data, attached
    # by `datasets_api.manifest.assemble_manifest`. Omitted from `to_dict()`
    # when `None`, as on versions minted before the catalogue existed.
    family: MemberFamily | None = None
    # The population the definition's identity admits, from the request. `None` when the
    # resolver could not state the whole of it, and on a trace that describes no one family.

    def to_dict(self) -> dict[str, object]:
        """Returns the wire shape. Optional fields are omitted when `None`.
        `normalisations` is always present, including as an empty list, and
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
        result["facets"] = list(self.facets)
        if self.mint_facets is not None:
            result["mint_facets"] = list(self.mint_facets)
        result["normalisations"] = list(self.normalisations)
        if self.conversion is not None:
            result["conversion"] = list(self.conversion)
        result["frame_grammar_version"] = self.frame_grammar_version
        result["materialiser_build"] = self.materialiser_build
        result["capability_version"] = self.capability_version
        return result
