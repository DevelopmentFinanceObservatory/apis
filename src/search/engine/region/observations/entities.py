"""Map observation entities to declared roles and flatten subject probes."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from search.engine.errors import RoleEntityUnplaced
from search.engine.region.codes import OBSERVATIONS_PROBE_FAILED
from search.engine.shape import ShapeDraft, entities_by_role, is_role_bearing, role_names_by_axis
from search.models import EntityRoleKey, Warning
from search.observations import as_entity_role_key

if TYPE_CHECKING:
    from search.engine.retrieval.retrieve import Materialised

logger = logging.getLogger(__name__)


def _observation_entities(
    shape: ShapeDraft,
    *,
    destination_dcids: list[str] | None,
    destination_dcid: str | None,
    origin_dcids: list[str] | None = None,
) -> tuple[dict[str, list[str]], Warning | None]:
    """Build the role-to-entity map used by the observation probe.

    Role-bearing shapes use their declared destination and origin axes. Other shapes use
    `observationAbout`. Place lists take precedence over the scalar destination. No entities
    returns an empty map without a warning. An unplaced role returns `OBSERVATIONS_PROBE_FAILED`
    as a warning, leaving the primary resolve intact. The map uses the entities already
    confirmed for coverage.
    """
    subject = destination_dcids or ([destination_dcid] if destination_dcid else [])
    origin = origin_dcids or []
    try:
        if is_role_bearing(shape):
            return (
                entities_by_role(
                    shape=shape,
                    by_axis={
                        "destination": subject,
                        "origin": origin,
                    },
                ),
                None,
            )
        return entities_by_role(shape=shape, subject=subject), None
    except RoleEntityUnplaced:
        logger.warning(
            "observations preview skipped for shape_id=%r: the resolved "
            "entity has no role to bind under",
            shape.shape_id,
        )
        return {}, Warning(
            code=OBSERVATIONS_PROBE_FAILED,
            severity="warn",
            message="observations fetch failed; Spec.resolved.observations was not populated.",
        )


def scope_axis_dcids(
    mat_result: Materialised, member_dcids: list[str] | None
) -> tuple[list[str] | None, list[str] | None]:
    """Route discovered members to `(destination_dcids, origin_dcids)` by scope axis.

    Origin scopes populate the origin list. All others populate the destination list. The
    caller supplies the scalar destination fallback separately.
    """
    if mat_result.discovered_scope is not None and mat_result.discovered_scope.axis == "origin":
        return None, member_dcids
    return member_dcids, None


def open_role_axis(
    shape: ShapeDraft, resolved_entities: Mapping[str, Sequence[str]]
) -> EntityRoleKey | None:
    """Return the open flow axis for discovery, preferring destination over origin.

    Return None for subject-only shapes or when the candidate axis already has an entity.
    """
    if not is_role_bearing(shape):
        return None
    names = role_names_by_axis(shape)
    destination = names.get("destination")
    if destination is not None:
        return as_entity_role_key(destination) if not resolved_entities.get(destination) else None
    origin = names.get("origin")
    if origin is not None:
        return as_entity_role_key(origin) if not resolved_entities.get(origin) else None
    return None


def coverage_probe_role_entities(
    shape: ShapeDraft,
    mat_result: Materialised,
    *,
    destination_dcids: list[str] | None,
    destination_dcid: str | None,
    origin_dcids: list[str] | None,
) -> dict[str, list[str]]:
    """Map each bound role to the entities slot and facet coverage probes read.

    Suppress destination expansion after discovery to avoid probing scope members, leaving
    that axis open. Keep the caller's bound origin list. All coverage probes use this
    derivation. Slot probes read the `_flat_obs_entities` view of it.
    """
    _probe_destination_dcids = destination_dcids if mat_result.discovered_scope is None else None
    _probe_entities, _ = _observation_entities(
        shape,
        destination_dcids=_probe_destination_dcids,
        destination_dcid=destination_dcid,
        origin_dcids=origin_dcids,
    )
    return _probe_entities


def _flat_obs_entities(obs_entities: Mapping[str, Sequence[str]]) -> list[str]:
    """Return the `observationAbout` entities for flat coverage probes.

    Role-bearing maps yield an empty list because a flat probe cannot preserve role pairs.
    """
    return list(obs_entities.get("observationAbout") or [])
