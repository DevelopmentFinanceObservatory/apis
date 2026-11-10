"""The flow axis a part reads a place named without a direction on, settled once before the
menu seats (`settle_undirected_axis`, `ResolveState.undirected_axis`).

Every reader of that place reads the settled axis: the shapes' role assignment
(`place_roles._undirected_place_axis`), the unasked-axes and merge rules
(`place_roles.without_unasked_axes`, `place_roles.on_undirected_axis`), the house policies
(`house_policy.context_of`) and the measure basis (`scope.basis.part_shape`), the last two
through `Frame.directions`. While no rule settles it the direction stays open, and the part
offers each reading that holds data (`phase_place_materialise._settle_direction`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from code_index.declarations import TOTAL_GROUPS
from code_index.query import CodeIndex
from code_index.shared import get_index, per_index
from search.engine.region.availability import undirected_series
from search.engine.region.place_roles import directed_axes, operation_axes, undirected_set
from search.engine.region.place_set import PlaceSet
from search.engine.region.state import ResolveState
from search.engine.retrieval.bound_values import shape_part_datasets
from search.engine.shape import ShapeDraft, is_role_bearing, role_names_by_axis
from search.heading.departure_fact import UndirectedReading, UndirectedRule
from search.models import RoleAxis
from search.scope.region import policy_members


def settle_undirected_axis(
    state: ResolveState, shapes: Sequence[ShapeDraft]
) -> UndirectedReading | None:
    """The axis the one place set the part names without a direction is read on, over the
    eligible `shapes`, with the rule that settles it. The first rule that gives an axis
    decides:

    1. `operation`: an operation over one flow axis beside a single place, with no place
       named with a direction, where a role-bearing shape declares both axes, takes the
       other axis, with no data needed: "Korea's top donors" ranks the donors, so Korea is
       the recipient of their aid whatever the coverage holds. A group can scope the ranked
       axis ("top recipients of aid in Africa"), so an operation takes no axis beside one.
    2. `complement`: the part's places named with a direction stand on one flow axis, a
       role-bearing shape declares both, and the coverage read finds a series with the
       undirected place on the other axis (`availability.undirected_series`), so the place
       takes it. A read that finds none, or cannot tell, leaves the axis to the rules
       below. "Low income countries budget support from Germany" reads the group as the
       recipient, as "Germany budget support to low income countries" does. "Aid from
       Germany and France" read with France undirected does not make France a recipient
       that holds no aid from Germany.
    3. `coverage`: only one axis holds a series at the places within the part's window
       (`availability.undirected_series`). "ODA in Ethiopia" reads Ethiopia as the
       recipient, and "Korea total aid in 1980" reads Korea as the recipient.
    4. `declaration`: where both axes hold a series, or neither does, a dataset that
       declares every place of the set a provider (`declared_providers`) and answers a
       shape that declares a provider role makes the set the provider, and the reading
       carries each such dataset (`declaring_datasets`, `UndirectedReading.listed_by`), one
       of which the answer's `direction_inferred` departure cites. A set whose places only
       different datasets declare settles no axis by declaration. "Korea total aid" reads Korea
       as the donor though Korea also receives aid, and "Turkey total aid", whose group's
       total declares no providers (`TotalGroup.declares_providers`), stays open.

    Every rule chooses among the open axes alone (`_open_axes`): those some role-bearing
    shape declares a role on, less any axis a place named with a direction holds and, for
    a single place, any axis an operation runs over, which the operation leaves open for
    the values it works across. "Aid from Germany and France", read with France
    undirected, never seats France beside Germany on the donor side, though DAC2a declares
    France a provider.

    `None` when the part names no such set or several, and when no rule settles the axis.
    """
    place_set = undirected_set(state)
    if place_set is None:
        return None
    open_axes = _open_axes(state, place_set, shapes)
    complement = _complement_axis(state, place_set, shapes)
    if complement is not None and complement[0] not in open_axes:
        complement = None
    if complement is not None and complement[1] == "operation":
        return UndirectedReading(*complement)
    reads = undirected_series(state, shapes)
    if complement is not None and True in reads.get(complement[0], frozenset()):
        return UndirectedReading(*complement)
    held = [axis for axis, found in reads.items() if True in found and axis in open_axes]
    if len(held) == 1:
        return UndirectedReading(held[0], "coverage")
    if "origin" in open_axes and (datasets := declaring_datasets(place_set, shapes)):
        return UndirectedReading("origin", "declaration", listed_by=datasets)
    return None


def _open_axes(
    state: ResolveState, place_set: PlaceSet, shapes: Sequence[ShapeDraft]
) -> frozenset[RoleAxis]:
    """The flow axes a rule may seat `place_set` on: those some role-bearing shape of
    `shapes` declares a role on, less those the part's directed places hold and, where
    `place_set` is a single place, those its operations run over."""
    declared = frozenset(
        axis for shape in shapes if is_role_bearing(shape) for axis in role_names_by_axis(shape)
    )
    ranked = operation_axes(state) if not place_set.members else frozenset()
    return declared - ranked - directed_axes(state)


def _complement_axis(
    state: ResolveState, place_set: PlaceSet, shapes: Sequence[ShapeDraft]
) -> tuple[RoleAxis, UndirectedRule] | None:
    """The flow axis the part leaves `place_set` free, with what took the other: the other
    axis when the part's directed places, and its operations over a flow axis where
    `place_set` is a single place, all take one, and some role-bearing shape of `shapes`
    declares a role on each. `complement` where a directed place takes it, `operation`
    where an operation alone does."""
    directed = directed_axes(state)
    taken = directed | (operation_axes(state) if not place_set.members else frozenset())
    if len(taken) != 1:
        return None
    if not any(is_role_bearing(shape) and len(role_names_by_axis(shape)) == 2 for shape in shapes):
        return None
    return ("destination" if "origin" in taken else "origin"), (
        "complement" if directed else "operation"
    )


def declaring_datasets(place_set: PlaceSet, shapes: Sequence[ShapeDraft]) -> tuple[str, ...]:
    """The datasets, by dcid, that each declare every place of `place_set` a provider
    (`declared_providers`) and answer a role-bearing shape of `shapes` with a provider role
    (`shape_part_datasets`). Empty when no one dataset declares them all."""
    places = place_set.dcids()
    if not places:
        return ()
    index = get_index()
    providers = declared_providers(index)
    answering = frozenset(
        dataset
        for shape in shapes
        if is_role_bearing(shape) and "origin" in role_names_by_axis(shape)
        for dataset in shape_part_datasets(shape, index)
    )
    declaring = answering.intersection(*(providers.get(place, frozenset()) for place in places))
    return tuple(sorted(declaring))


@per_index
def declared_providers(index: CodeIndex) -> Mapping[str, frozenset[str]]:
    """Each place a dataset declares a provider, by dcid, with the datasets that declare it.
    A dataset publishing a total that declares providers
    (`code_index.declarations.TotalGroup.declares_providers`) declares each member the
    membership policy lists for the group the total holds (`PartyRow.totals`,
    `policy_members`): DAC1, DAC2a and DAC2b publish "DAC countries, total"
    (`groups/OECD.DAC`). A dataset publishing a private party declares that party
    (`PartyRow.party_class`, the export's `PRIVATE_PROVIDERS`)."""
    declared: dict[str, set[str]] = {}
    for row in index.parties.values():
        if row.kind == "party" and row.party_class == "private":
            members: frozenset[str] = frozenset((row.dcid,))
        elif (
            (total := TOTAL_GROUPS.get(row.dcid)) is not None
            and total.declares_providers
            and row.totals is not None
        ):
            members = policy_members(row.totals)
        else:
            continue
        for member in members:
            declared.setdefault(member, set()).update(row.datasets)
    return {member: frozenset(datasets) for member, datasets in declared.items()}
