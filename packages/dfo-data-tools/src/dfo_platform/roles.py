"""The two closed flow axes for role-bearing observations, the role names placed on them,
and the English wording for each axis per flow family.

Graph-declared role names are open. `ROLE_AXES` maps the known ones to these axes, and
`declared_role_for` resolves a role key against a member's declared roles. `role_wording`
gives the nouns and prepositions prose uses for each axis; graph role names never reach
prose. This standard-library module keeps `search.models` and `statvar_members` free of
materialisation dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

type RoleAxis = Literal["origin", "destination"]

type FlowFamily = Literal["aid", "debt_service", "debt_stock", "debt_lending"]
"""What a role-bearing series counts, which decides how its roles read.

- `aid`: development finance from providers to recipients (OECD CRS flows and commitments,
  ONE imputed multilateral flows).
- `debt_service`: payments a debtor makes on its debt (debt service, interest, principal).
- `debt_stock`: debt a debtor holds at a point in time (outstanding debt, arrears).
- `debt_lending`: lending from creditors to a debtor (disbursements, undisbursed debt).

`search.heading.flow_family.flow_family` computes it from a shape's constraint values.
"""

# Maps known role names to engine-owned flow axes. Search slot axes
# (what/how/where/when) serve a separate purpose.
ROLE_AXES: dict[str, RoleAxis] = {
    "provider": "origin",
    "creditor": "origin",
    "source": "origin",
    "financingProvider": "origin",
    "recipient": "destination",
    "debtor": "destination",
    "target": "destination",
    "financingRecipient": "destination",
}

# The flow-axis labels themselves, `ROLE_AXES`'s own values. A caller's role
# key can be spelled as one of these (`search.engine.spec_id.entity_role_key`
# mints `origin`/`destination`), which `axis_for_role` cannot read back, so
# the resolver checks the labels against declared roles' axes directly.
AXIS_LABELS: frozenset[RoleAxis] = frozenset(ROLE_AXES.values())


def axis_for_role(role_dcid: str) -> RoleAxis | None:
    """The flow axis for `role_dcid`, or `None` if this vocabulary has no
    mapping. Unmapped declared roles remain valid but cannot be filled from
    a directional phrase.
    """
    return ROLE_AXES.get(role_dcid)


def role_key_axis(role: str) -> RoleAxis | None:
    """The flow axis an identity's `entities` key names, either an axis label itself or the
    axis of a known graph role, since keys keep the caller's spelling (`creditor`, `debtor`)."""
    return role if role in AXIS_LABELS else axis_for_role(role)


def declared_role_for(role_key: str, declared_roles: tuple[str, ...]) -> str | None:
    """Resolves a raw declared role or its `origin`/`destination` alias.

    Returns `None` for unknown or ambiguous axis keys. This total resolver
    keeps bound-role resolution consistent between Search and the mint.
    """
    if role_key in declared_roles:
        return role_key
    if role_key in AXIS_LABELS:
        resolved: str | None = None
        for role in declared_roles:
            if axis_for_role(role) == role_key:
                if resolved is not None:
                    # The alias cannot choose between these roles, so resolve neither.
                    return None
                resolved = role
        return resolved
    return None


@dataclass(frozen=True, slots=True)
class RoleNoun:
    """How prose names one flow axis.

    `preposition` introduces an entity in this role after the measure noun ("to Kenya",
    "owed to China"). It is unused for the axis a heading names in possessive form.
    """

    singular: str
    plural: str
    preposition: str


@dataclass(frozen=True, slots=True)
class RoleWording:
    """The nouns and prepositions for both flow axes of one flow family.

    `possessor` is the axis whose entity leads a heading in possessive form, such as the
    provider in "Germany's gross ODA disbursements to Kenya" or the debtor in "Ghana's external
    debt service owed to China". The other axis follows the measure noun with its preposition.
    """

    origin: RoleNoun
    destination: RoleNoun
    possessor: RoleAxis

    def noun(self, axis: RoleAxis) -> RoleNoun:
        """The wording for `axis`."""
        return self.origin if axis == "origin" else self.destination


GENERIC_ROLE_WORDING = RoleWording(
    origin=RoleNoun(singular="provider", plural="providers", preposition="from"),
    destination=RoleNoun(singular="recipient", plural="recipients", preposition="to"),
    possessor="origin",
)
"""The wording when no flow family is known, as for a series outside every family or a
caller that reads the axis alone (a refusal naming the role, a coverage dimension label)."""

_DEBT_OWED = RoleWording(
    origin=RoleNoun(singular="creditor", plural="creditors", preposition="owed to"),
    destination=RoleNoun(singular="debtor", plural="debtors", preposition="of"),
    possessor="destination",
)

ROLE_WORDING: Mapping[FlowFamily, RoleWording] = MappingProxyType(
    {
        "aid": RoleWording(
            origin=RoleNoun(singular="provider", plural="providers", preposition="from"),
            destination=RoleNoun(singular="recipient", plural="recipients", preposition="to"),
            possessor="origin",
        ),
        "debt_service": _DEBT_OWED,
        "debt_stock": _DEBT_OWED,
        "debt_lending": RoleWording(
            origin=RoleNoun(singular="creditor", plural="creditors", preposition="from"),
            destination=RoleNoun(singular="debtor", plural="debtors", preposition="to"),
            possessor="destination",
        ),
    }
)
"""Role wording per flow family, the one declaration every role noun in prose reads."""


def role_wording(family: FlowFamily | None) -> RoleWording:
    """The role wording for `family`, or `GENERIC_ROLE_WORDING` when it is `None`."""
    return GENERIC_ROLE_WORDING if family is None else ROLE_WORDING[family]
