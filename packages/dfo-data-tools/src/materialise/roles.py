"""Derives StatVar entity roles from its `observationProperties` arc.

The graph owns the open set of role names. An absent or empty arc defaults
to `observationAbout`. The engine maps known role names to its fixed
`origin` and `destination` axes (`dfo_platform.roles`). This dependency-free module serves both
`datasets_api.shape.resolve_shape` and `graph_api`.
"""

from __future__ import annotations

from typing import Literal

from dfo_platform.roles import AXIS_LABELS as _AXIS_LABELS
from dfo_platform.roles import GENERIC_ROLE_WORDING as _GENERIC_ROLE_WORDING
from dfo_platform.roles import ROLE_AXES as _ROLE_AXES
from dfo_platform.roles import RoleAxis
from dfo_platform.roles import axis_for_role as axis_for_role
from dfo_platform.roles import declared_role_for as declared_role_for

_OBSERVATION_ABOUT = "observationAbout"


def axis_noun(axis: RoleAxis | Literal["observationAbout"]) -> str:
    """Return the singular flow-axis noun of `dfo_platform.roles.GENERIC_ROLE_WORDING`.

    Any key other than `destination` reads as `origin`. Search's role-scope discovery
    supplies origin or destination.
    """
    return _GENERIC_ROLE_WORDING.noun("destination" if axis == "destination" else "origin").singular


class DegenerateObservationProperties(Exception):
    """Raised when a non-empty `observationProperties` arc declares
    `observationAbout`. That default subject role cannot also be explicit,
    since it could let two scope bindings address one observation slot.
    """

    def __init__(self, *, observation_properties: tuple[str, ...]) -> None:
        self.observation_properties = observation_properties
        super().__init__(
            "observationProperties non-emptily declares observationAbout "
            f"(the default subject role belongs to an absent arc): "
            f"{observation_properties!r}"
        )


def declared_roles(observation_properties: tuple[str, ...]) -> tuple[str, ...]:
    """Returns declared roles in arc order. An empty arc defaults to
    `(observationAbout,)`. An explicit `observationAbout` raises
    `DegenerateObservationProperties`.
    """
    if not observation_properties:
        return (_OBSERVATION_ABOUT,)
    if _OBSERVATION_ABOUT in observation_properties:
        raise DegenerateObservationProperties(observation_properties=observation_properties)
    return tuple(observation_properties)


def roles_by_axis(roles: tuple[str, ...]) -> dict[RoleAxis, str]:
    """Maps each represented flow axis to its declared role.

    Raises `ValueError` when two roles map to the same axis.
    """
    by_axis: dict[RoleAxis, str] = {}
    for role in roles:
        axis = axis_for_role(role)
        if axis is None:
            continue
        if axis in by_axis:
            raise ValueError(
                f"two declared roles name the {axis!r} axis: {by_axis[axis]!r} and {role!r}"
            )
        by_axis[axis] = role
    return by_axis


def axis_ambiguous_for(role_key: str, declared_roles: tuple[str, ...]) -> bool:
    """Whether an axis alias maps to multiple declared roles.

    Raw role names return `False`. Refusal messages use this to distinguish
    an unknown role from an ambiguous shape.
    """
    if role_key not in _AXIS_LABELS:
        return False
    count = 0
    for role in declared_roles:
        if axis_for_role(role) == role_key:
            count += 1
            if count > 1:
                return True
    return False


KNOWN_ROLE_NAMES: frozenset[str] = frozenset(_ROLE_AXES) | {_OBSERVATION_ABOUT}
"""Every role name this module places on a flow axis (`_ROLE_AXES`'s own
keys) plus the default subject role, `observationAbout`. Role names
themselves are graph-owned and open (see this module's own docstring): a
StatVar's `observationProperties` arc can declare a role this table never
places on an axis, and that role is still real. This constant is the
practical, deployment-global vocabulary available without a live graph
read, for a caller that needs to tell a genuine role name apart from a
typo without enumerating one frame's own declared roles
(`datasets_api.groupby.refuse_unknown_by_axes`)."""
