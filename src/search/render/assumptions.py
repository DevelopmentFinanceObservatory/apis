"""What an answer assumed: each value it filled that its request did not ask for.

`assumptions_for` reads a resolved Spec's own records, with no graph or model call:

- a slot whose trace records `declared_default` provenance, a value its code system's or
  shape's declared default fixed, named in the heading's words for that slot;
- a flow role no request word asked for, read across every entity on it with data
  ("every provider", "every creditor"): one the engine's role scope default filled
  (`resolved.scope` on that axis), whatever its member count, or one the heading states
  with no entity of its own. An axis the identity binds in `entities` or lists in
  `open_roles`, or that the request's words asked every member of ("all donors"), is asked;
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

from dfo_platform.roles import role_key_axis, role_wording
from search.models import (
    WHERE_REF,
    Assumption,
    ResolvedFilter,
    RoleAxis,
    SentenceToken,
    Spec,
    role_ref,
)

from .heading import Heading, heading_for
from .how import _how_value_text
from .labels import _join_labels

CLAUSE_LEAD = "Assumed: "
"""The text that opens the assumptions clause."""
CLAUSE_SEPARATOR = "; "
"""The text between two items of the clause."""

# The members of a place scope that reads the world as one total.
_WORLD = "Earth"


def assumptions_for(
    spec: Spec,
    *,
    keep_how_properties: frozenset[str] = frozenset(),
    echo: Sequence[Assumption] | None = None,
    asked_axes: frozenset[RoleAxis] = frozenset(),
) -> list[Assumption]:
    """The values `spec` filled that its request did not ask for, in clause order: declared
    defaults in `trace.slot_filters` order, then open roles, then the place scope.
    ``keep_how_properties`` are the how properties a candidates part keeps visible, under
    which its heading names each value. ``asked_axes`` are the flow axes the request's words
    asked every member of (`RegionResult.every_member_axes`), which state no open role.

    ``echo`` is the list a `dataset_resubmit` identity entry posts back, None when it posts
    none. With an echo, the result is the echoed items that still hold for `spec`, each with
    its words read from `spec`: a declared default whose slot still binds exactly the echoed
    `refs`, an open role `spec` still reads open on the echoed axis, and the place scope of a
    `spec` that still names no place. An item the echo lacks is left out, since the posted
    identity or its edit asked for it, so an empty echo states none. With no echo, `spec`
    reads as a fresh request does."""
    heading = heading_for(spec, keep_how_properties=keep_how_properties).heading
    echoed = {_echo_key(item) for item in echo or ()}
    # The posted identity of an echo binds every slot, so none carries declared_default
    # provenance: on an echo a how slot holds its default while it binds the echoed values.
    defaults = [
        _declared_default(slot_filter, heading)
        for slot_filter in spec.resolved.trace.slot_filters
        if slot_filter.refs
        and (
            slot_filter.key.axis == "how" and _slot_key(slot_filter) in echoed
            if echo is not None
            else slot_filter.provenance == "declared_default"
        )
    ]
    filled = [*_open_roles(spec, heading, asked_axes), *_place_scope(spec)]
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


def _declared_default(slot_filter: ResolvedFilter, heading: Heading) -> Assumption:
    """The `declared_default` assumption of a slot bound to `slot_filter.refs`, in the
    heading's words for it (`declared_default_text`)."""
    return Assumption(
        kind="declared_default",
        binding_ref=slot_filter.key.as_ref(),
        text=declared_default_text(slot_filter, heading),
        refs=[value.ref for value in slot_filter.refs],
    )


def common_assumptions(
    specs: Sequence[Spec],
    *,
    keep_how_properties: frozenset[str] = frozenset(),
    asked_axes: frozenset[RoleAxis] = frozenset(),
) -> list[Assumption]:
    """The assumptions every one of `specs` yields alike, in the first one's order: what a
    candidates part assumed whichever reading is chosen. [] when there are no specs."""
    if not specs:
        return []
    first, *rest = (
        assumptions_for(spec, keep_how_properties=keep_how_properties, asked_axes=asked_axes)
        for spec in specs
    )
    return [item for item in first if all(item in others for others in rest)]


def declared_default_text(slot_filter: ResolvedFilter, heading: Heading) -> str:
    """The words for a declared default: the text the heading states for its slot, or,
    where the heading leaves the slot silent, its bound values as a how value reads them
    (`how._how_value_text`)."""
    ref = slot_filter.key.as_ref()
    for value in (*heading.modifiers, *heading.how):
        if value.text and value.slot is not None and value.slot.key.as_ref() == ref:
            return value.text
    prop = slot_filter.key.property
    return _join_labels(
        [
            _how_value_text(
                value.ref,
                value.label,
                prop.ref if prop is not None else None,
                prop.label if prop is not None else None,
            )
            for value in slot_filter.refs
        ]
    )


def _open_roles(spec: Spec, heading: Heading, asked_axes: frozenset[RoleAxis]) -> list[Assumption]:
    """An `open_role` assumption for each flow role the heading states that no request word
    asked for, in the heading's role order: the role the role scope default filled
    (`resolved.scope` on a flow axis), named or counted ("to Norway", "to 2 recipients"),
    and a role stated with no entity of its own ("from 164 providers", "owed to all
    creditors"). A role is asked when the identity binds it in `entities` or lists it in
    `open_roles`, or when it is in ``asked_axes``. The noun is the role's in the answer's
    flow family (`role_wording`). A role a spoken how value narrows ("owed to bondholders")
    states no open role, so it assumes none."""
    identity = spec.definition.identity
    asked = (
        {role_key_axis(key) for key in identity.entities}
        | set(identity.open_roles or ())
        | asked_axes
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
