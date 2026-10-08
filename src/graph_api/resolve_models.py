"""Response models for `POST /api/graph/resolve`.

`results` is index-aligned with `texts`. Its six statuses are outcomes in
HTTP 200 responses. `not_found` means nothing matched the text; `not_held` means a real
entity matched but this graph does not hold it. Infrastructure failures raise refusals.

`entity_type` is the open `typeOf` value. `kind` is one of `country`,
`continent`, `subregion`, `continental_union`, `organisation`, `aggregate`,
`place_other`, or `other`. `aggregate` is a DAC stand-in for parties it does not name, not a
place, and `place_other` is a subnational place. Only continents and subregions expand to member
countries. Clients use `entity_type` to open a node and to choose a role,
by matching it against the role's `rangeIncludes`.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from entity_resolution import EntityKind

_MAX_TEXT_LENGTH = 200

_ResolveText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=_MAX_TEXT_LENGTH)
]


class ResolveRequest(BaseModel):
    """`texts`: 1..25 free-text strings resolved as a single batch, so
    `entity_resolution.EntityResolver.resolve` and this door's own
    region-expansion reads
    (`graph_api.reader.GraphNodeReader.child_places_batch`/
    `node_arcs_many`) each cover the whole batch in one round trip. Each
    text is stripped of leading and trailing whitespace before resolution;
    the stripped form must be non-empty and no longer than 200 characters.
    """

    model_config = ConfigDict(extra="forbid", title="GraphResolveRequest")
    texts: list[_ResolveText] = Field(min_length=1, max_length=25)


class EntityRef(BaseModel):
    """One graph-confirmed entity: a `resolved` result's own `entity`, one
    of an `ambiguous` result's own `candidates`, or a `resolved_region`
    result's own `region`. The same shape in all three positions, since all
    three name one entity the input text itself resolved to.
    """

    dcid: str
    name: str
    entity_type: str
    kind: EntityKind


class ResolvedResult(BaseModel):
    """A single graph-confirmed entity."""

    status: Literal["resolved"] = "resolved"
    entity: EntityRef


class RegionMember(BaseModel):
    dcid: str
    name: str


class ResolvedRegionResult(BaseModel):
    """A continent or subregion, expanded to its member countries through
    graph containment (`graph_api.doors.resolve`'s own module docstring).
    Every member in `members` is a graph node by construction -- it came
    from the graph's own containment arc, not from `entity_resolution` --
    so there is no separate "member not held" case.
    """

    status: Literal["resolved_region"] = "resolved_region"
    region: EntityRef
    members: list[RegionMember]


class ResolvedGroupResult(BaseModel):
    """A membership group (`entity_resolution.Group`) -- G7, G20, DAC, the
    EU, or any other spelling whose winning candidate names a group rather
    than a single held place or organisation. `members` is the group's own
    membership, drawn from a closed offline set of 33 curated groups (the
    resolvekit `geo.*` modules plus `dfo_graph`), not from graph
    containment; the largest is the United Nations at 193. This is not
    configurable and no request parameter narrows it -- for example G20
    resolves to its 19 country members, and this door does not add the EU
    or the African Union to that list.

    Named `entity`, not `region` as in the sibling `ResolvedRegionResult`
    above: a region's own node IS the region, so `region` names it
    directly and is never `null`, while a group's `entity` is one of two
    readings this door reports alongside `members`, and is `null` when the
    group has no held node of its own.

    This door has no `role_property` (`entity_resolution.Admission.admits`
    checks a resolved entity's `graph_type` against one, e.g.
    `observationAbout` or `financingProvider`) to admit `entity` against,
    so it reports both readings and lets the client decide
    (`graph_api.doors.resolve`'s own module docstring): `entity` is the
    group's own entity reading when one exists -- the group's own held
    node (the EU) or an organisation survivor of the same text (a group
    whose own node the graph does not hold) -- and `null` when neither
    does. `members` is the group's own graph-confirmed membership, ordered
    by dcid, and is never empty: a group with no confirmed member resolves
    `not_held` instead, regardless of `entity`. Every entry in `members`
    is a full `EntityRef`, not a bare `{ dcid, name }` pair as in
    `resolved_region.members` -- a group's membership comes from
    `entity_resolution`'s own graph confirmation, which already carries
    `entity_type` and `kind` for each member, unlike a region's members,
    which come from raw containment.
    """

    status: Literal["resolved_group"] = "resolved_group"
    entity: EntityRef | None
    members: list[EntityRef]


class AmbiguousResult(BaseModel):
    """Two or more graph-confirmed candidates the input text itself names
    (`entity_resolution.Ambiguous`) -- genuinely more than one entity
    ("Congo"), or a curated alias naming several ("World Bank" naming both
    IBRD and IDA). `candidates` is ordered by confidence descending, then
    by `dcid` -- a caller that auto-picks `candidates[0]` is picking the
    most confident match.
    """

    status: Literal["ambiguous"] = "ambiguous"
    candidates: list[EntityRef]


class NotFoundResult(BaseModel):
    """No candidate the input text itself names survived
    (`entity_resolution.NotFound`). `near_matches` carries the canonical
    names of the candidates the listed-name gate rejected, in confidence
    order, deduplicated, capped at 3. A spelling-corrected match ("Lagos"
    resolving to Laos) lands
    here with the corrected name as its first near match. A caller resends
    the corrected name, or a name from this list.
    """

    status: Literal["not_found"] = "not_found"
    near_matches: list[str]


class NotHeldResult(BaseModel):
    """`entity_resolution` resolved (or narrowed to) exactly one entity,
    but this deployment's graph does not hold it (`entity_resolution.
    NotHeld`). `dcid` is the id `entity_resolution` produced -- from
    resolvekit's own custom pack, or from the region id map -- naming it
    helps a graph maintainer even though this deployment cannot open it.
    That `dcid` does not open on `GET /api/graph/node/{dcid}` either (that
    door reports `404 node_not_held` for it too); a caller's own move here
    is the same as for `not_found` -- report the gap, don't retry.
    """

    status: Literal["not_held"] = "not_held"
    name: str
    dcid: str


ResolveResult = Annotated[
    ResolvedResult
    | ResolvedRegionResult
    | ResolvedGroupResult
    | AmbiguousResult
    | NotFoundResult
    | NotHeldResult,
    Field(discriminator="status"),
]


class ResolveResponse(BaseModel):
    """`results` is index-aligned with the request's own `texts`."""

    model_config = ConfigDict(title="GraphResolveResponse")

    results: list[ResolveResult]
