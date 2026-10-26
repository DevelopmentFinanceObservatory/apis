"""What an answer assumed: each value it filled that its request did not ask for.

`assumptions_for` reads a resolved Spec's own records, and the names the request's door
corrected, with no graph or model call:

- an entity whose name corrects the words the request typed for it ("Germny" read as
  Germany), recorded by the raw-text door as a `NameCorrection`, stated while the Spec
  binds the entity, named by the Spec's label for it. Words that are a listed name or alias
  of the entity ("Kenia" for Kenya), and a translation the request writes no words of, are
  no correction (`surface_resolution.corrected_surfaces`);
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
back. Its part states the echoed items that still hold, matched by kind and binding and
worded from the rebuilt Spec. A declared default holds while its slot still binds the
echoed value dcids (`Assumption.refs`). A name correction holds while the Spec still binds
its entity on the echoed role, worded from the Spec with the typed words the echo carries
(`Assumption.asked`), since the resubmit posts them nowhere else. An entry with no echo states the open roles and
place scope the answer reads and no declared default, since it binds every slot.

The sentence closes with the items as one clause (`clause_tokens`), for example "Assumed:
official development assistance; every provider."
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from code_index.labels import without_property_suffix
from code_index.shared import get_index
from dfo_platform.roles import role_key_axis, role_wording
from search.models import (
    MAX_ASSUMPTION_TEXT_CHARS,
    WHERE_REF,
    Assumption,
    Operation,
    ResolvedEntity,
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


@dataclass(frozen=True)
class NameCorrection:
    """An entity name the extraction wrote in place of the request's own words for it, when
    those words list no name or alias of the entity it settles on
    (`surface_resolution.corrected_surfaces`).

    `asked` is the request's words ("Germny") and `ref` the entity's dcid ("country/DEU").
    """

    asked: str
    ref: str


def assumptions_for(
    spec: Spec,
    *,
    echo: Sequence[Assumption] | None = None,
    corrections: Sequence[NameCorrection] = (),
    asked_axes: frozenset[RoleAxis] = frozenset(),
    operations: Sequence[Operation] = (),
    own_spec: Spec | None = None,
) -> list[Assumption]:
    """The values `spec` filled that its request did not ask for, in clause order: name
    corrections in `resolved.entities` order, then declared defaults in
    `trace.slot_filters` order, then open roles, then the place scope. ``corrections`` are
    the names the request's door corrected (`RegionResult.name_corrections`); each states
    a `name_corrected` item while `spec` binds its entity.
    ``own_spec`` is the Spec the part's own request resolved to before a same-shape merge
    gave it `spec`, None when `spec` is its own. A merge ranks the group's slot provenance
    (`same_shape._merge_slot_filter_properties`), so a slot one member's request left open
    and another's named reads `text_recognized` on the merged Spec. The declared defaults
    are read from ``own_spec``. Each states the values its slot fixed there that the same
    slot of `spec` still binds, worded from `spec`. The open roles are those the heading of
    a definite part answering `spec` states. ``asked_axes`` are the flow axes the request's
    words asked every member of (`RegionResult.every_member_axes`), and ``operations`` the
    part's operations, each asking the flow axis it runs over. Neither states an open role.

    ``echo`` is the list a `dataset_resubmit` identity entry posts back, None when it posts
    none. With an echo, the result is the echoed items that still hold for `spec`, each with
    its words read from `spec`. A declared default holds while its slot still binds the
    echoed `refs`, and states those refs. An open role holds while `spec` still reads it open
    on the echoed axis, and the place scope while `spec` still names no place. An item the
    echo lacks is left out, since the posted identity or its edit asked for it, so an empty
    echo states none. A name correction holds while `spec` binds its echoed ref on the
    echoed role, reworded with the typed words it carries (`Assumption.asked`), the one
    record of them.
    With no echo, the result is every item `spec` yields."""
    heading = heading_for(spec).heading
    echoed = {_echo_key(item) for item in echo or ()}
    slots = (own_spec or spec).resolved.trace.slot_filters
    if echo is None:
        fixed = {
            slot_filter.key.as_ref(): _defaulted_values(slot_filter)
            for slot_filter in slots
            if slot_filter.refs and slot_filter.provenance == "declared_default"
        }
    else:
        # The posted identity of an echo binds every slot, so none carries declared_default
        # provenance. On an echo a how slot holds an echoed default while it binds its refs.
        how = {slot_filter.key.as_ref() for slot_filter in slots if slot_filter.key.axis == "how"}
        fixed = {
            item.binding_ref: set(item.refs)
            for item in echo
            if item.kind == "declared_default" and item.refs and item.binding_ref in how
        }
    defaults = [
        _declared_default(slot_filter, refs=fixed[ref])
        for slot_filter in spec.resolved.trace.slot_filters
        if (ref := slot_filter.key.as_ref()) in fixed
        and fixed[ref] <= {value.ref for value in slot_filter.refs}
    ]
    filled = [*_open_roles(spec, heading, asked_axes, operations), *_place_scope(spec)]
    if echo is None:
        named = _name_corrections(spec, corrections)
    else:
        named = _held_name_corrections(spec, echo)
        filled = [item for item in filled if _echo_key(item) in echoed]
    return [*named, *defaults, *filled]


def _echo_key(item: Assumption) -> tuple[str, str, frozenset[str]]:
    """What an echoed item shares with one the answer yields while it still holds: its kind,
    its binding and its values. Its words are read from the answer. A `name_corrected`
    item matches the entity the answer binds under the same key, read off the Spec's
    entities rather than any item the answer yields (`_held_name_corrections`)."""
    return item.kind, item.binding_ref, frozenset(item.refs)


def _name_corrections(spec: Spec, corrections: Sequence[NameCorrection]) -> list[Assumption]:
    """A `name_corrected` assumption for each entity of `spec` one of ``corrections``
    names, in `resolved.entities` order (`_name_corrected`), once for an entity on two
    roles."""
    asked = {correction.ref: correction.asked for correction in corrections}
    items: list[Assumption] = []
    for entity in spec.resolved.entities:
        words = asked.pop(entity.ref, None)
        if words is not None and (item := _name_corrected(entity, asked=words)) is not None:
            items.append(item)
    return items


def _held_name_corrections(spec: Spec, echo: Sequence[Assumption]) -> list[Assumption]:
    """The echoed `name_corrected` items whose entity `spec` still binds on the echoed role,
    in echo order. Each is reworded from `spec` with the typed words it carries
    (`Assumption.asked`), so a label that changed since the echoed text was written is
    restated; an item carrying no words is dropped."""
    entities = {
        ("name_corrected", role_ref(entity.role), frozenset({entity.ref})): entity
        for entity in spec.resolved.entities
    }
    held: list[Assumption] = []
    for item in echo:
        entity = entities.get(_echo_key(item))
        if entity is None or item.asked is None:
            continue
        if (restated := _name_corrected(entity, asked=item.asked)) is not None:
            held.append(restated)
    return held


def _name_corrected(entity: ResolvedEntity, *, asked: str) -> Assumption | None:
    """The `name_corrected` assumption of `entity` read for the typed words ``asked``, bound
    on its role and worded '"Germny" read as Germany' from the entity's label. None when the
    words would pass the bound of `Assumption.text`: they are the request's own, and the
    parsed door refuses an entity text past `request._PARSED_MAX_ENTITY_TEXT_CHARS`, well
    under it."""
    text = f'"{asked}" read as {_entity_label(entity)}'
    if len(text) > MAX_ASSUMPTION_TEXT_CHARS:
        return None
    return Assumption(
        kind="name_corrected",
        binding_ref=role_ref(entity.role),
        text=text,
        refs=[entity.ref],
        asked=asked,
    )


def _entity_label(entity: ResolvedEntity) -> str:
    """The name the answer shows for `entity`: its label, else its dcid's readable tail."""
    return _humanize_ref(entity.ref, entity.label)


def _defaulted_values(slot_filter: ResolvedFilter) -> set[str]:
    """The values a declared-default slot's default fixed: its code system's declared code
    (`CodeSystemDeclaration.default`) where the slot binds it beside the codes a measure
    basis reads it as (ODA's flow 10 beside flow 60 on grant equivalents), else every value
    the slot binds."""
    refs = {value.ref for value in slot_filter.refs}
    prop = slot_filter.key.property
    declaration = get_index().declarations.get(prop.ref) if prop is not None else None
    if declaration is None or declaration.default is None:
        return refs
    return refs & {declaration.default.code} or refs


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
    in ``asked_axes``, or one of ``operations`` runs over it. The noun is the role's in the
    answer's flow family (`role_wording`). A role a spoken how value narrows ("owed to bondholders")
    states no open role, so it assumes none. The destination shares the where slot, so a
    place scope (`_place_scope`) is that slot's one statement and the destination states no
    open role beside it."""
    identity = spec.definition.identity
    asked = {role_key_axis(key) for key in identity.entities} | asked_open_axes(
        {*(identity.open_roles or ()), *asked_axes}, operations
    )
    scope = spec.resolved.scope
    scoped = role_key_axis(scope.axis) if scope is not None else None
    place_scoped = scope is not None and scope.axis == "observationAbout"
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
        and not (place_scoped and phrase.axis == "destination")
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
