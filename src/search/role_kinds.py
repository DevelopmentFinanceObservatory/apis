"""The kinds of entity each flow role takes, and the check that refuses an entity of another kind.

A flow role takes the kinds of the parties the code index lists on its axis (`CodeIndex.parties`,
`PartyRow.axes`), so no table names them and a party kind the index adds on an axis is taken
there. A `party` row brings the kinds the index's party types give a party
(`CodeIndex.party_types`: `Country` a country, `Organization` an organisation), a `total` or
`residual` row brings `aggregate`, and a total that holds a group (`PartyRow.totals`) brings the
group's kind under the membership policy: a continent, a subregion, or a group. The party types
are declared for the index as a whole, so both axes take a country and an organisation once each
lists a party. An axis the index lists no party on has no kinds to judge by, so the check passes
its entities unread and unjudged: the rule refuses only what the catalogue's parties rule out.

A posted entity takes the kind resolve and the Graph API's node route serve
(`entity_resolution.kinds.classify`), over its graph types. Search's role routes and the
Datasets mint both run `check_role_kinds` on the entities a role-bearing request binds, before
anything is built, and both refuse a `RoleKindRefused` as `invalid_role_ref`. An entity neither
the graph nor the membership policy holds refuses too. A first answer binds a place's own node
on a flow role only where `role_takes` says the role takes it, the same rule, so it never binds
a node its own resubmit refuses (`place_roles.set_choices`).

The membership policy is built from resolvekit's stores, so the `datasets` extra installs
dfo-data-tools's `resolve` extra (`tests/datasets_api/test_independence.py`). `entity_resolution`
is imported where it is used, so importing `datasets_api` loads no resolvekit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from code_index.shared import per_index
from dfo_platform.roles import FlowFamily, RoleAxis, role_key_axis, role_wording

if TYPE_CHECKING:
    from code_index.query import CodeIndex
    from entity_resolution.types import Kind

TYPE_OF = "typeOf"

# How a refusal names an entity of each kind, in the order it lists the kinds a role takes.
_KIND_PHRASES: Mapping[Kind, str] = {
    "country": "a country",
    "organisation": "an organisation",
    "group": "a group of countries",
    "continent": "a continent",
    "subregion": "a subregion",
    "continental_union": "a continental union",
    "aggregate": "an aggregate of other parties",
    "place_other": "a place other than a country or region",
    "stat_var": "a statistical variable",
    "code_system": "a code system",
    "code": "a classification code",
    "unit": "a unit",
    "source": "a source",
    "dataset": "a dataset",
    "provenance": "a provenance",
    "other": "no kind of party",
}


type TypesRead = Callable[[list[str]], Mapping[str, Sequence[str] | None]]
"""The graph types of each of some dcids, `None` for a node the graph does not hold."""


class TypeReader(Protocol):
    """The graph read `graph_types` makes, which both Search's graph client and the Datasets
    graph reader implement."""

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str]
    ) -> Mapping[str, Mapping[str, tuple[str, ...]] | None]: ...


def graph_types(graph: TypeReader) -> TypesRead:
    """The `TypesRead` of one `typeOf` read of `graph` per call."""

    def read(dcids: list[str]) -> Mapping[str, Sequence[str] | None]:
        arcs = graph.node_arcs_many(dcids, properties=[TYPE_OF])
        return {dcid: types_of(arcs.get(dcid)) for dcid in dcids}

    return read


def types_of(node: Mapping[str, tuple[str, ...]] | None) -> Sequence[str] | None:
    """The `typeOf` values of one node `node_arcs_many` answered, `None` for no node."""
    return None if node is None else node.get(TYPE_OF, ())


class RoleKindRefused(Exception):
    """An entity posted under the flow role `role` whose kind the role does not take
    (`role_kinds`). `kind` is `None` for an entity neither the graph nor the membership policy
    holds. `allowed` are the kinds the role takes, in the order the message lists them. The
    message names the role by the wording of the series' flow `family`
    (`dfo_platform.roles.role_wording`): a creditor in a debt family, a provider in aid."""

    def __init__(
        self,
        *,
        role: str,
        axis: RoleAxis,
        dcid: str,
        kind: Kind | None,
        allowed: Sequence[Kind],
        family: FlowFamily | None,
    ) -> None:
        self.role = role
        self.axis = axis
        self.dcid = dcid
        self.kind = kind
        self.allowed = tuple(allowed)
        noun = role_wording(family).noun(axis).singular
        if kind is None:
            message = f"{dcid} is not in the graph."
        else:
            phrases = [_KIND_PHRASES[k] for k in self.allowed]
            message = f"{dcid} is {_KIND_PHRASES[kind]}; a {noun} is {_either(phrases)}."
        super().__init__(message)


def _either(phrases: Sequence[str]) -> str:
    """`phrases` joined as alternatives: "a", "a or b", "a, b, or c"."""
    if len(phrases) <= 2:
        return " or ".join(phrases)
    return f"{', '.join(phrases[:-1])}, or {phrases[-1]}"


@per_index
def role_kinds(code_index: CodeIndex) -> Mapping[RoleAxis, frozenset[Kind]]:
    """The kinds each flow axis takes, by the module's rule over `code_index`'s parties."""
    from entity_resolution.shared import shared_membership_policy
    from entity_resolution.types import entity_kind

    policy = shared_membership_policy()
    party_kinds = frozenset(
        entity_kind((graph_type,))
        for graph_type, kind in code_index.party_types.items()
        if kind == "party"
    )
    kinds: dict[RoleAxis, set[Kind]] = {"origin": set(), "destination": set()}
    for party in code_index.parties.values():
        held = party.kinds_over()
        for axis in party.axes:
            if "party" in held:
                kinds[axis] |= party_kinds
            if held & {"total", "residual"}:
                kinds[axis].add("aggregate")
            if party.totals is not None and (group := policy.kind_of(party.totals)) is not None:
                kinds[axis].add(group)
    return {axis: frozenset(found) for axis, found in kinds.items()}


def check_role_kinds(
    entities: Mapping[str, Sequence[str]],
    *,
    read_types: TypesRead,
    code_index: CodeIndex,
    family: FlowFamily | None,
) -> None:
    """Refuse the first entity in `entities`, by role key, whose kind its flow role does not
    take (`role_kinds`), raising `RoleKindRefused`, which names the role in the wording of
    the series' flow `family` (`None` for the generic wording). A key naming no flow axis, as the subject
    role `observationAbout`, is not checked, nor is a role whose axis `role_kinds` gives no
    kind, so a refusal always lists at least one kind.

    A party the index lists on the role's axis takes one of the axis's kinds, since they are
    derived from it, and a group or region the membership policy holds takes the policy's kind
    whatever its types (`classify`), so neither is read. One call of `read_types` types the
    rest, and none runs when nothing is left."""
    from entity_resolution.shared import shared_membership_policy

    policy = shared_membership_policy()
    allowed_by_axis = role_kinds(code_index)
    checked = [
        (role, axis, dcid)
        for role, dcids in entities.items()
        if (axis := role_key_axis(role)) is not None and allowed_by_axis[axis]
        for dcid in dcids
        if not _listed_on(code_index, dcid, axis)
    ]
    unread = list(
        dict.fromkeys(dcid for _role, _axis, dcid in checked if policy.kind_of(dcid) is None)
    )
    types = read_types(unread) if unread else {}
    for role, axis, dcid in checked:
        node_types = types.get(dcid)
        kind = (
            None
            if dcid in types and node_types is None
            else _kind(dcid, node_types or (), code_index=code_index)
        )
        allowed = allowed_by_axis[axis]
        if kind not in allowed:
            raise RoleKindRefused(
                role=role,
                axis=axis,
                dcid=dcid,
                kind=kind,
                allowed=[k for k in _KIND_PHRASES if k in allowed],
                family=family,
            )


def role_takes(dcid: str, types: Sequence[str], axis: RoleAxis, *, code_index: CodeIndex) -> bool:
    """Whether the flow role on `axis` takes the entity `dcid` whose graph types are `types`,
    by the rule `check_role_kinds` refuses by: the index lists it as a party on `axis`, the
    axis has no kinds to judge by, or the entity's kind is one the axis takes."""
    allowed = role_kinds(code_index)[axis]
    return (
        not allowed
        or _listed_on(code_index, dcid, axis)
        or _kind(dcid, types, code_index=code_index) in allowed
    )


def _kind(dcid: str, types: Sequence[str], *, code_index: CodeIndex) -> Kind:
    """The kind of the entity `dcid` whose graph types are `types` (`classify`), under the
    shared membership policy."""
    from entity_resolution.kinds import classify
    from entity_resolution.shared import shared_membership_policy

    return classify(
        dcid=dcid, types=types, code_index=code_index, membership=shared_membership_policy()
    )


def _listed_on(code_index: CodeIndex, dcid: str, axis: RoleAxis) -> bool:
    """Whether the index lists `dcid` as a party on `axis`."""
    party = code_index.parties.get(dcid)
    return party is not None and axis in party.axes
