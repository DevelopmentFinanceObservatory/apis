"""What an answer assumed: each value it filled that its request did not ask for.

`assumptions_for` reads a resolved Spec's own records, with no graph or model call:

- a slot whose trace records `declared_default` provenance, a value its code system's or
  shape's declared default fixed, named by the code's full label in the code index
  ("official development assistance" for `CRSTypeOfFlow10`, whose short name is "ODA").
  A part a same-shape merge serves reads the provenance from its own request's Spec;
- a flow role no request word asked for, read across every entity on it with data
  ("every provider", "every creditor"): one the engine's role scope default filled
  (`resolved.scope` on that axis), whatever its member count, or one the heading states
  with no entity of its own. An axis the identity binds in `entities` or lists in
  `open_roles`, that an operation runs over ("top 5 donors"), or that the request's words
  asked every member of ("all donors"), is asked;
- the place scope of a request that named no place (`resolved.scope` on
  `observationAbout`), named by its members' types ("the world as a whole", "every
  country", "every place").

A `dataset_resubmit` identity entry posts every slot bound and may echo the part's list
back. Its part states the echoed items that still hold, matched by kind, binding and value
dcids (`Assumption.refs`) and worded from the rebuilt Spec. An entry with no echo reads as
a fresh request.

The sentence closes with the items as one clause (`clause_tokens`): "Assumed:
official development assistance; every provider."
"""

from __future__ import annotations

from collections.abc import Sequence

from code_index.labels import without_property_suffix
from code_index.shared import get_index
from dfo_platform.roles import role_key_axis, role_wording
from search.models import (
    WHERE_REF,
    Assumption,
    Operation,
    ResolvedFilter,
    RoleAxis,
    SentenceToken,
    Spec,
    asked_open_axes,
    role_ref,
)

from .heading import Heading, heading_for
from .labels import _humanize_ref, _join_labels

CLAUSE_LEAD = "Assumed: "
"""The text that opens the assumptions clause."""
CLAUSE_SEPARATOR = "; "
"""The text between two items of the clause."""

# The members of a place scope that reads the world as one total.
_WORLD = "Earth"


def assumptions_for(
    spec: Spec,
    *,
    echo: Sequence[Assumption] | None = None,
    asked_axes: frozenset[RoleAxis] = frozenset(),
    operations: Sequence[Operation] = (),
    own_spec: Spec | None = None,
) -> list[Assumption]:
    """The values `spec` filled that its request did not ask for, in clause order: declared
    defaults in `trace.slot_filters` order, then open roles, then the place scope.
    ``own_spec`` is the Spec the part's own request resolved to before a same-shape merge
    gave it `spec`, None when `spec` is its own. A merge ranks the group's slot provenance
    (`same_shape._merge_slot_filter_properties`), so a slot one member's request left open
    and another's named reads `text_recognized` on the merged Spec. The declared defaults
    are read from ``own_spec``: each states the values its slot fixed there that the same
    slot of `spec` still binds, worded from `spec`. The open roles are those the heading of
    a definite part answering `spec` states. ``asked_axes`` are the flow axes the request's words
    asked every member of (`RegionResult.every_member_axes`), and ``operations`` the part's
    operations, each asking the flow axis it runs over; neither states an open role.

    ``echo`` is the list a `dataset_resubmit` identity entry posts back, None when it posts
    none. With an echo, the result is the echoed items that still hold for `spec`, each with
    its words read from `spec`: a declared default whose slot still binds exactly the echoed
    `refs`, an open role `spec` still reads open on the echoed axis, and the place scope of a
    `spec` that still names no place. An item the echo lacks is left out, since the posted
    identity or its edit asked for it, so an empty echo states none. With no echo, `spec`
    reads as a fresh request does."""
    heading = heading_for(spec).heading
    echoed = {_echo_key(item) for item in echo or ()}
    # The posted identity of an echo binds every slot, so none carries declared_default
    # provenance: on an echo a how slot holds its default while it binds the echoed values.
    fixed = {
        slot_filter.key.as_ref(): {value.ref for value in slot_filter.refs}
        for slot_filter in (own_spec or spec).resolved.trace.slot_filters
        if slot_filter.refs
        and (
            slot_filter.key.axis == "how" and _slot_key(slot_filter) in echoed
            if echo is not None
            else slot_filter.provenance == "declared_default"
        )
    }
    defaults = [
        _declared_default(slot_filter, refs=fixed[ref])
        for slot_filter in spec.resolved.trace.slot_filters
        if (ref := slot_filter.key.as_ref()) in fixed
        and fixed[ref] <= {value.ref for value in slot_filter.refs}
    ]
    filled = [*_open_roles(spec, heading, asked_axes, operations), *_place_scope(spec)]
    if echo is not None:
        filled = [item for item in filled if _echo_key(item) in echoed]
    return [*defaults, *filled]


def _echo_key(item: Assumption) -> tuple[str, str, frozenset[str]]:
    """What an echoed item shares with one the answer yields while it still holds: its kind,
    its binding and its values. Its words are read from the answer."""
    return item.kind, item.binding_ref, frozenset(item.refs)


def _slot_key(slot_filter: ResolvedFilter) -> tuple[str, str, frozenset[str]]:
    """The `_echo_key` of the declared default `slot_filter` would state."""
    return (
        "declared_default",
        slot_filter.key.as_ref(),
        frozenset(value.ref for value in slot_filter.refs),
    )


def _declared_default(slot_filter: ResolvedFilter, *, refs: set[str]) -> Assumption:
    """The `declared_default` assumption of the values ``refs`` among those `slot_filter`
    binds, in its order, named by their labels (`declared_default_text`)."""
    stated = slot_filter.model_copy(
        update={"refs": [value for value in slot_filter.refs if value.ref in refs]}
    )
    return Assumption(
        kind="declared_default",
        binding_ref=stated.key.as_ref(),
        text=declared_default_text(stated),
        refs=[value.ref for value in stated.refs],
    )


def common_assumptions(lists: Sequence[Sequence[Assumption]]) -> list[Assumption]:
    """The items every one of `lists` holds alike, compared on the whole item, in the first
    one's order: what a candidates part assumed whichever reading is chosen, from each
    shown reading's own list (`CandidatesPart.reading_assumptions`). [] when there are no
    lists."""
    if not lists:
        return []
    first, *rest = lists
    return [item for item in first if all(item in others for others in rest)]


def declared_default_text(slot_filter: ResolvedFilter) -> str:
    """The words for a declared default: each bound value's full label in the code index
    (`CodeRow.label`, "official development assistance" where the short name is "ODA"),
    or, for a value the index holds no row for (a shape's slot default), its graph label
    without the property suffix the graph adds, joined as one list."""
    index = get_index()
    prop = slot_filter.key.property
    property_label = prop.label if prop is not None else None
    labels: list[str] = []
    for value in slot_filter.refs:
        row = index.code_by_dcid(value.ref)
        labels.append(
            row.label
            if row is not None
            else without_property_suffix(_humanize_ref(value.ref, value.label), property_label)
        )
    return _join_labels(labels)


def _open_roles(
    spec: Spec,
    heading: Heading,
    asked_axes: frozenset[RoleAxis],
    operations: Sequence[Operation],
) -> list[Assumption]:
    """An `open_role` assumption for each flow role the heading states that no request word
    asked for, in the heading's role order: the role the role scope default filled
    (`resolved.scope` on a flow axis), named or counted ("to Norway", "to 2 recipients"),
    and a role stated with no entity of its own ("from 164 providers", "owed to all
    creditors"). A role is asked when the identity binds it in `entities`, or when
    `search.models.asked_open_axes` holds it: the identity lists it in `open_roles`, it is
    in ``asked_axes``, or one of ``operations`` runs over it. The noun is the role's in the answer's
    flow family (`role_wording`). A role a spoken how value narrows ("owed to bondholders")
    states no open role, so it assumes none."""
    identity = spec.definition.identity
    asked = {role_key_axis(key) for key in identity.entities} | asked_open_axes(
        {*(identity.open_roles or ()), *asked_axes}, operations
    )
    scope = spec.resolved.scope
    scoped = role_key_axis(scope.axis) if scope is not None else None
    wording = role_wording(heading.family)
    return [
        Assumption(
            kind="open_role",
            binding_ref=role_ref(phrase.axis),
            text=f"every {wording.noun(phrase.axis).singular}",
        )
        for phrase in heading.roles
        if phrase.axis is not None
        and (not phrase.entity_dcids or phrase.axis == scoped)
        and phrase.axis not in asked
    ]


def _place_scope(spec: Spec) -> list[Assumption]:
    """The `place_scope` assumption of an answer read across every place because its
    request named none, named by its members' types (`_place_scope_text`)."""
    scope = spec.resolved.scope
    if scope is None or scope.axis != "observationAbout":
        return []
    return [Assumption(kind="place_scope", binding_ref=WHERE_REF, text=_place_scope_text(spec))]


def _place_scope_text(spec: Spec) -> str:
    """The place scope's words from its members: the world as a whole for Earth alone,
    every country when every member is typed Country, else every place."""
    scope = spec.resolved.scope
    members = scope.members if scope is not None else []
    if [member.ref for member in members] == [_WORLD]:
        return "the world as a whole"
    if members and all(member.entity_type == "Country" for member in members):
        return "every country"
    return "every place"


def clause_tokens(texts: Sequence[str], *, after_text: bool) -> list[SentenceToken]:
    """The assumptions clause naming ``texts`` in order, as assumption-segment text tokens:
    a space when it follows text (``after_text``), "Assumed: ", each text with "; " between
    two, and ".". [] when ``texts`` is empty."""
    if not texts:
        return []
    words: list[str] = [" "] if after_text else []
    words.append(CLAUSE_LEAD)
    for position, text in enumerate(texts):
        if position:
            words.append(CLAUSE_SEPARATOR)
        words.append(text)
    words.append(".")
    return [SentenceToken(kind="text", text=word, segment="assumption") for word in words]
