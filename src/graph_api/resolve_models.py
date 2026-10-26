"""Request and response models for `POST /api/graph/resolve`.

`results` is index-aligned with `texts`, and every result has one shape whatever its status.
The four statuses are outcomes in HTTP 200 responses: `resolved` names one entity, `ambiguous`
lists the candidates the text names, `not_found` means no candidate of the requested kinds
matched, and `not_held` means the resolver named a real entity this graph holds no node for.
Infrastructure failures raise refusals.

A group or a region resolves like any entity, with its kind (`group`, `continent` or
`subregion`), the projection its count follows, `member_count` and a `members` link to the
node route's list. `expand: ["members"]` puts the members inline in place of the link.
"""

from __future__ import annotations

from typing import Annotated, Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
)

from dfo_platform.wire import ResponseModel
from entity_resolution import DEFAULT_KINDS, EntityKind, MatchedAs, Reading
from entity_resolution.membership import Projection
from graph_api.kinds import Kind
from graph_api.refs import NodeRef

_MAX_TEXT_LENGTH = 200
_MAX_TEXTS = 25

MAX_INLINE_MEMBERS = 256
"""The most members `expand: ["members"]` puts inline in one result."""

_ResolveText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=_MAX_TEXT_LENGTH)
]

RESOLVE_KINDS: tuple[EntityKind, ...] = get_args(EntityKind)
"""The kinds a request may name: the resolver's kinds, a subset of `graph_api.kinds.KINDS`."""

DEFAULT_RESOLVE_KINDS: tuple[EntityKind, ...] = tuple(
    k for k in RESOLVE_KINDS if k in DEFAULT_KINDS
)
"""The kinds a request that names none admits: every place and organisation kind and `group`."""

Status = Literal["resolved", "ambiguous", "not_found", "not_held"]


class ResolveRequest(BaseModel):
    """`texts`: 1 to 25 strings resolved as one batch. Each is stripped of surrounding
    whitespace and must then be 1 to 200 characters.

    `kinds` admits only candidates of those kinds. It defaults to `DEFAULT_RESOLVE_KINDS`, so a
    request that omits it equals one naming those eight. `code` searches the code index's labels
    and synonyms, and `code_system` narrows codes to one system's graph id
    (`CRSPurposeCodeEnum`); naming `code_system` without `code` in `kinds` is refused at
    `/code_system`. `expand: ["members"]` inlines a group's or region's members.
    """

    model_config = ConfigDict(extra="forbid", title="GraphResolveRequest")

    texts: list[_ResolveText] = Field(min_length=1, max_length=_MAX_TEXTS)
    kinds: list[EntityKind] = Field(default=list(DEFAULT_RESOLVE_KINDS), min_length=1)
    code_system: str | None = Field(default=None, min_length=1, max_length=_MAX_TEXT_LENGTH)
    expand: list[Literal["members"]] = Field(default=[])

    @field_validator("code_system")
    @classmethod
    def _code_system_needs_code(cls, value: str | None, info: ValidationInfo) -> str | None:
        # `kinds` is absent from `info.data` when it failed its own validation, which already
        # reports the request; this check then stays silent.
        kinds = info.data.get("kinds")
        if value is not None and kinds is not None and "code" not in kinds:
            raise ValueError("code_system narrows the code kind, which kinds does not include")
        return value


class Matched(ResponseModel):
    """The listed text the input matched, and whether it is the entity's name, an alias, or a
    code. A phrase naming a group's members ("EU countries") matches as `alias`."""

    model_config = ConfigDict(
        title="GraphResolveMatched", validate_by_name=True, serialize_by_alias=True
    )

    text: str
    as_: MatchedAs = Field(alias="as")


class EntityRef(ResponseModel):
    """A reference with its kind. `kind` is `null` only for a `not_held` entity, which the
    graph holds no node to classify."""

    model_config = ConfigDict(title="GraphResolveEntity")

    ref: str
    label: str
    kind: Kind | None


class ResolveResult(ResponseModel):
    """One text's outcome. `input` echoes the text; fields a status does not use are `null`.

    - `resolved`: `entity`, `matched` and `reading`. A group or region adds `projection`,
      `member_count` and `members`: the list route's link, or the members inline (at most
      256) when the request expanded them. The three are `null` for a region the membership
      policy does not hold. `near_matches` lists up to 3 other places or organisations of the
      requested kinds that share the name ("Georgia" resolves to the country, with the US
      state `geoId/13` here), else `null`. When `kinds` removed the first match, a node of a
      requested kind sharing the name resolves in its place and `near_matches` lists the
      removed match ("Georgia" with `kinds: ["place_other"]` resolves to `geoId/13`, with
      the country here).
    - `ambiguous`: `candidates`, in the resolver's order. When `kinds` removed the first
      match and up to 3 nodes of a requested kind share the name, `near_matches` lists the
      removed match, else `null`.
    - `not_found`: `near_matches`, hints for a person; a program should pick none by id.
    - `not_held`: `entity`, the id the resolver named, which does not open on the node route.

    `reading` is `members` when the text names a group's members ("EU countries"), else
    `entity`.
    """

    model_config = ConfigDict(title="GraphResolveResult")

    input: str
    status: Status
    entity: EntityRef | None = None
    matched: Matched | None = None
    reading: Reading | None = None
    projection: Projection | None = None
    member_count: int | None = None
    members: str | list[NodeRef] | None = None
    candidates: list[EntityRef] | None = None
    near_matches: list[EntityRef] | None = None


class ResolveResponse(ResponseModel):
    """`results` is index-aligned with the request's `texts`."""

    model_config = ConfigDict(title="GraphResolveResponse")

    results: list[ResolveResult]
