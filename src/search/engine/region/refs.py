"""Shared graph-ref and provenance helpers for region grounding.

Provenance comes from facet `provenanceId`. Every provenance is named by
`provenance_labels`, which is `materialise.provenance_doc.provenance_names`: the
bound origin catalogue's name for a listed one, with no graph read, else its
parent Dataset's name or its own. A listed provenance also takes its source and
Data Commons from the catalogue, and an unlisted one carries neither. Admission
drops every unlisted provenance in production, so the graph read is empty there.
Other label helpers preserve input order, omit unresolved refs, or raise
`GroundingMiss` as specified. `confirm_dcids_present` batch-checks posted refs
and raises `invalid_role_ref` (422) for the first group with an absent or
unlabelled value.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from code_index.shared import get_index
from dfo_platform.named import named_capped
from dfo_platform.roles import FlowFamily
from materialise.provenance_doc import ProvenanceNameReader, provenance_names
from origin_catalogue import current_catalogue
from search.engine.degraded_reads import note_degraded_read
from search.engine.errors import EngineInputError
from search.engine.ground import _ref_from_label
from search.models import (
    EntityRoleKey,
    GraphRef,
)
from search.role_kinds import RoleKindRefused, TypesRead, check_role_kinds


def _ref_or_miss(dcid: str, labels: dict[str, str]) -> GraphRef:
    """Build a GraphRef from cached labels or raise `GroundingMiss` if absent."""
    return _ref_from_label(dcid, labels.get(dcid))


def _refs_present(dcids: list[str], labels: dict[str, str]) -> list[GraphRef]:
    """Build refs for labelled DCIDs in input order, omitting unresolved values."""
    return [GraphRef(ref=d, label=labels[d]) for d in dcids if d in labels]


def _where_slot_refs(
    dcids: list[str],
    labels: dict[str, str],
    roles: Mapping[str, EntityRoleKey] | list[tuple[str, EntityRoleKey]],
) -> list[GraphRef]:
    """Build where-slot refs in input order, echoing confirmed DCIDs without labels.

    Omit DCIDs absent from `labels` and `roles`. Callers warn about echoed labels.
    """
    role_dcids = roles if isinstance(roles, Mapping) else {dcid for dcid, _role in roles}
    refs = []
    for d in dcids:
        if d in labels:
            refs.append(GraphRef(ref=d, label=labels[d]))
        elif d in role_dcids:
            refs.append(GraphRef(ref=d, label=d))
    return refs


def provenance_labels(
    dcids: Sequence[str], graph: ProvenanceNameReader, *, deadline: float | None
) -> dict[str, str]:
    """The name of each provenance of `dcids` (`materialise.provenance_doc.provenance_names`),
    its graph read bounded by `deadline`. A failed read keeps the catalogue's names and notes
    the lost read (`degraded_reads.note_degraded_read`), so no cache stores the answer. An id
    nothing names is absent."""
    names, read_ok = provenance_names(
        graph, dcids, catalogue=current_catalogue(), deadline=deadline
    )
    if not read_ok:
        note_degraded_read("provenance names")
    return names


def confirm_dcids_present[V](
    dcid_groups: Mapping[str, Sequence[str]],
    *,
    read: Callable[[list[str]], Mapping[str, V]],
    pointer_for: Callable[[str, list[str]], str],
    noun: str = "reference",
) -> dict[str, V]:
    """Batch-read distinct DCIDs once and return the confirmed values.

    The first group with absent or unlabelled DCIDs raises `invalid_role_ref`
    (422), using its sorted missing values and `pointer_for`'s pointer.
    """
    all_dcids = list(dict.fromkeys(dcid for dcids in dcid_groups.values() for dcid in dcids))
    if not all_dcids:
        return {}
    confirmed = dict(read(all_dcids))
    for group, dcids in dcid_groups.items():
        absent = sorted({dcid for dcid in dcids if dcid not in confirmed})
        if absent:
            detail = f"posted entity {noun}(s) not found in the graph: {named_capped(absent)}"
            raise EngineInputError(
                detail,
                code="invalid_role_ref",
                status=422,
                pointer=pointer_for(group, absent),
            )
    return confirmed


def refuse_wrong_kind(
    entities: Mapping[str, Sequence[str]],
    *,
    read_types: TypesRead,
    family: FlowFamily | None,
    pointer_prefix: str,
) -> None:
    """Refuse an entity in `entities`, by role key, whose kind its flow role does not take
    (`search.role_kinds.check_role_kinds`, the rule the Datasets mint refuses by) as
    `invalid_role_ref` (422) at `{pointer_prefix}/entities/{role}`, naming the role in the
    wording of the series' flow `family`."""
    try:
        check_role_kinds(entities, read_types=read_types, code_index=get_index(), family=family)
    except RoleKindRefused as refused:
        raise EngineInputError(
            str(refused),
            code="invalid_role_ref",
            status=422,
            pointer=f"{pointer_prefix}/entities/{refused.role}",
        ) from refused
