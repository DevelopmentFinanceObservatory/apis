from __future__ import annotations

from pydantic import Field

from .base import StrictModel
from .vocab import AssumptionKind


class Assumption(StrictModel):
    """One value the answer filled that its request did not ask for, stated for people
    in `text` and for code in `kind` and `binding_ref`.

    Assumptions live on the part (`_PartBase.assumptions`). The sentence that carries
    the part closes with them as one clause: "Assumed: official development assistance;
    every provider."
    """

    kind: AssumptionKind = Field(
        description=(
            "what the engine filled: 'declared_default', a slot the request left open "
            "that its code system's or shape's declared default fixed; 'open_role', a "
            "flow role the request neither named an entity on nor asked member by member, "
            "read as the total over every entity on it; 'place_scope', a request that "
            "named no place, read across every place with data"
        )
    )
    binding_ref: str = Field(
        description=(
            "the identity binding whose edit replaces this assumption: 'how:<property "
            "dcid>' for a declared default, 'origin:' or 'where:' for a role or the place "
            "scope, as on the sentence's own tokens"
        )
    )
    text: str = Field(
        min_length=1,
        description=(
            "the words the sentence's assumptions clause states for this fill: the "
            "heading's words for a declared default ('official development assistance'), "
            "'every' and the role's noun in the answer's flow family for an open role "
            "('every provider', 'every creditor'), and 'every country', 'every place' or "
            "'the world as a whole' for the place scope, as the heading counts it"
        ),
    )
