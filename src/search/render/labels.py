"""Shared label humanisation and SentenceToken helpers for axis renderers."""

from __future__ import annotations

import re
from collections.abc import Sequence

from search.models import (
    Binding,
    BindingSet,
    BindingValue,
    EntityGroup,
    GraphRef,
    RoleAxis,
    SentenceAxis,
    SentenceToken,
    Slot,
    SlotValue,
    role_ref,
)

_CAMEL_BOUNDARY = re.compile(
    r"(?<=[a-zA-Z])(?=[0-9])|(?<=[0-9])(?=[a-zA-Z])"
    r"|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)


# Full-string match for bare underscore identifiers such as "Kenya_Census".
# Authored prose containing an underscore does not match.
_UNDERSCORE_IDENTIFIER = re.compile(r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+$")


# Full-string match for bare alphanumeric identifiers. Spaced or underscored
# labels bypass camel-case splitting.
_BARE_ALNUM_IDENTIFIER = re.compile(r"^[A-Za-z0-9]+$")


# Split lower/digit-to-upper and acronym-to-word boundaries. Require two
# lowercase letters after the capital so suffixes such as "v2" stay intact.
_LOWER_UPPER_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z]{2})")


def _binding_values(binding: Binding) -> list[SlotValue]:
    """Return bound SlotValues, or [] when unbound.

    Callers inspect refs and literals before reducing values to labels.
    """
    if isinstance(binding, BindingValue):
        return [binding.value]
    if isinstance(binding, BindingSet):
        return binding.values
    return []


def _binding_labels(binding: Binding) -> list[str]:
    """Return bound value labels, or [] when unbound."""
    return _value_labels(_binding_values(binding))


def _entity_labels(binding: Binding) -> list[str]:
    """Return bound entity names in their graph label's case (`_entity_name`), or []."""
    labels: list[str] = []
    for value in _binding_values(binding):
        if value.ref is not None:
            labels.append(_entity_name(value.ref.ref, value.ref.label))
        elif value.literal:
            labels.append(value.literal)
    return labels


def _value_labels(values: list[SlotValue]) -> list[str]:
    labels: list[str] = []
    for v in values:
        if v.ref is not None:
            labels.append(_humanize(v.ref))
        elif v.literal:
            labels.append(v.literal)
    return labels


def _already_named_in(label_text: str, what_text: str) -> bool:
    """Return whether a label occurs in ``what_text`` as a case-insensitive phrase.

    Word boundaries prevent matches inside words, and matching spans the full
    text, including trailing parentheticals.
    """
    if not label_text or not what_text:
        return False
    pattern = r"\b" + re.escape(label_text.lower()) + r"\b"
    return re.search(pattern, what_text.lower()) is not None


def _camel_to_words(text: str) -> str:
    """Split bare camelCase/PascalCase identifiers at word boundaries.

    The source-label helper calls this only on a single alphanumeric segment.
    It splits lower- or digit-to-upper transitions and acronym-to-word
    transitions with at least two lowercase letters (``OECDQuarterly``).
    Acronyms, single words, and suffixes such as ``v2`` stay intact. Case is
    preserved. Authored labels are handled elsewhere and remain verbatim.
    """
    if not _BARE_ALNUM_IDENTIFIER.fullmatch(text):
        return text
    if not _LOWER_UPPER_BOUNDARY.search(text):
        return text
    return _LOWER_UPPER_BOUNDARY.sub(" ", text)


def _humanize_source_label(text: str) -> str:
    """Humanise bare-looking source labels and preserve already-spaced text.

    Split underscore-separated labels into segments, then split camel case
    within each segment. This handles underscore, camel-case, and mixed forms.
    The bound and resolved-source paths share this helper.
    """
    if _UNDERSCORE_IDENTIFIER.fullmatch(text):
        return " ".join(_camel_to_words(segment) for segment in text.split("_"))
    return _camel_to_words(text)


def _humanize_ref(dcid: str, label: str | None) -> str:
    """Return a distinct display label, or humanise the dcid tail as fallback.

    Missing, empty, or dcid-equal labels fall back to splitting underscores,
    camel case, and letter-digit boundaries, then lower-casing. Shared by
    GraphRef and bare-dcid entity rendering.
    """
    if label and label != dcid:
        return label
    tail = dcid.rsplit("/", 1)[-1]
    words = " ".join(_CAMEL_BOUNDARY.sub(" ", segment) for segment in tail.split("_"))
    return words.lower()


def _entity_name(dcid: str, label: str | None) -> str:
    """Return an entity's name in its graph label's case ("Gavi", "Gates Foundation").

    A missing or dcid-equal label falls back to the dcid tail split at underscores and
    camel case, keeping its case: the label of `Gavi` is its dcid, and reads "Gavi".
    """
    if label and label != dcid:
        return label
    tail = dcid.rsplit("/", 1)[-1]
    return " ".join(_CAMEL_BOUNDARY.sub(" ", segment) for segment in tail.split("_"))


def _humanize(ref: GraphRef) -> str:
    """Return a GraphRef's display label or humanised dcid fallback."""
    return _humanize_ref(ref.ref, ref.label)


# Match any trailing parenthetical. Used for label sentence-casing and to
# decide whether the basis phrase joins an existing parenthetical.
_TRAILING_PARENTHETICAL = re.compile(r"\s*\([^()]*\)\s*$")


def _text_token(text: str) -> SentenceToken:
    return SentenceToken(kind="text", text=text)


MEASURE_AXIS_REF_PREFIX = "axis:"
"""The prefix of a ref to a measure axis that no slot carries (`SentenceToken.states`)."""


def axis_ref(prop_dcid: str, slots: Sequence[Slot]) -> str:
    """The ref a token states for how axis ``prop_dcid``: the how slot's key when the
    shape declares one, else ``axis:<prop_dcid>``."""
    for slot in slots:
        if slot.key.axis == "how" and slot.key.property and slot.key.property.ref == prop_dcid:
            return slot.key.as_ref()
    return f"{MEASURE_AXIS_REF_PREFIX}{prop_dcid}"


def token_states(text: str, binding_ref: str | None, also: Sequence[str] = ()) -> list[str]:
    """What a token states (`SentenceToken.states`): nothing when its text is empty,
    else its own ``binding_ref`` and then each ref in ``also``, once each, in order."""
    if not text:
        return []
    refs = [binding_ref] if binding_ref is not None else []
    return list(dict.fromkeys([*refs, *also]))


def _slot_token(
    text: str,
    *,
    axis: SentenceAxis,
    slot: Slot | None,
    role: RoleAxis | None = None,
    states: Sequence[str] = (),
) -> SentenceToken:
    """Build a slot token, retaining its binding key when ``slot`` exists.

    ``binding_ref`` joins the token to SlotOptions. ``role`` marks a where
    fallback resolved from a directional scope. Place defaults leave it unset.
    ``states`` are the refs the token states besides its own (`token_states`).
    """
    if slot is None:
        return SentenceToken(
            kind="slot", text=text, axis=axis, role=role, states=token_states(text, None, states)
        )
    prop_dcid = slot.key.property.ref if slot.key.property else None
    binding_ref = slot.key.as_ref()
    return SentenceToken(
        kind="slot",
        text=text,
        axis=axis,
        property_dcid=prop_dcid,
        binding_ref=binding_ref,
        role=role,
        states=token_states(text, binding_ref, states),
    )


def _entity_token(
    text: str,
    *,
    role: RoleAxis,
    entity_dcids: Sequence[str],
    where_slot: Slot | None = None,
) -> SentenceToken:
    """Build an origin or destination token with entity dcids in display order.

    Destination tokens use the where axis and retain ``where_slot``'s property
    when supplied. Origin tokens have no slot axis. Both carry the role's
    ``binding_ref`` (`role_ref`: "where:" for the destination, "origin:" for the
    origin) and all ``entity_dcids`` as ``entity_refs`` so clients need not parse the
    joined text. A destination backed by ``where_slot`` carries that slot's own key
    instead, which reads "where:" whenever the slot names no property. An empty
    ``entity_dcids`` gives a role-only token: text that names an open role ("162
    recipients", "all creditors") and no entity refs.
    """
    if role == "destination" and where_slot is not None:
        token = _slot_token(text, axis="where", slot=where_slot, role=role)
        return token.model_copy(update={"entity_refs": list(entity_dcids)})
    return SentenceToken(
        kind="slot",
        text=text,
        axis="where" if role == "destination" else None,
        role=role,
        binding_ref=role_ref(role),
        entity_refs=list(entity_dcids),
        states=token_states(text, role_ref(role)),
    )


def join_labels(labels: list[str]) -> str:
    """`labels` as a reader lists them: "a", "a and b", "a, b, and c", empty for none."""
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + ", and " + labels[-1]


# Plural nouns a group label can end in that already name its members. Group labels are
# graph names, so this set follows the labels and stays apart from the rendered role nouns.
_MEMBER_NOUNS = frozenset(
    {"countries", "states", "donors", "providers", "recipients", "creditors", "debtors"}
)


def _is_region(dcid: str | None) -> bool:
    """Whether the group node `dcid` is a continent or subregion by the membership policy
    (`MembershipPolicy.holds_region`), whatever its graph type. The places inside one are said
    to be in it."""
    # Imported here: Datasets imports `search.render` through `search.models` without the
    # engine extra, which carries resolvekit.
    from entity_resolution import shared_membership_policy

    return dcid is not None and shared_membership_policy().holds_region(dcid)


def group_members_text(dcid: str | None) -> str:
    """Refer back to the members of a group just named, whose node is `dcid`: "its members",
    and for a region (`_is_region`) "the countries in it"."""
    if _is_region(dcid):
        return "the countries in it"
    return "its members"


def group_text(count: int, group: EntityGroup) -> str:
    """Name a group of ``count`` entities: "the {n} {label} members" ("the 27 European
    Union members").

    A group whose node is a region (`EntityGroup.ref`, `_is_region`) reads as the countries
    the membership policy holds for it: "the 54 countries in Africa".

    A label whose last word already names the members keeps its own noun ("the 32 DAC
    donors", "the 27 EU member states"). That word is one of `_MEMBER_NOUNS`, compared as
    text. The heading and the `entity_unavailable` and `group_read_as_members` departures
    all read this.

    A side read as its whole population (`EntityGroup.whole_of`) names it by its label
    alone: "all donors including private" holds two published totals, not two members.
    """
    if group.whole_of:
        return group.label
    if _is_region(group.ref):
        return f"the {count} countries in {group.label}"
    last_word = group.label.rsplit(" ", 1)[-1].lower()
    noun = "" if last_word in _MEMBER_NOUNS else " members"
    return f"the {count} {group.label}{noun}"
