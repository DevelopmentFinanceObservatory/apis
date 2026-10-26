from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .base import StrictModel
from .vocab import AssumptionKind

# The model is also the request type of `ResubmitIdentityEntry.assumptions`, so its strings
# carry the identity's bounds: a value dcid is an identity string (1,024 code points).
_MAX_REF_CHARS = 1024
# A declared default fixes one code (`DeclaredDefault.code`, `ShapeSlotDefault.value_dcid`), and
# a name correction reads one entity.
_MAX_REFS = 1
# A binding ref is "how:" and a property dcid, or a role's "origin:" or "where:". The bound
# is a dcid after the longest role-key prefix, "destination:", a cap above every ref the
# server emits that clients validate against.
_MAX_BINDING_REF_CHARS = len("destination:") + _MAX_REF_CHARS
# The words for a declared default are one code's full label. The longest label in the
# packaged index runs 126 characters, within the bound of a variable text
# (`request._PARSED_MAX_VARIABLE_TEXT_CHARS`). A name correction past it is not stated
# (`render.assumptions`).
MAX_ASSUMPTION_TEXT_CHARS = 200


class Assumption(StrictModel):
    """One value the answer filled that its request did not ask for, stated for people
    in `text` and for code in `kind`, `binding_ref` and `refs`.

    Assumptions live on the part (`_PartBase.assumptions`), and a candidates part holds
    each shown reading's own list as well (`CandidatesPart.reading_assumptions`), the one
    a client echoes on a dataset_resubmit after picking that reading. The sentence that
    carries the part closes with them as one clause: "Assumed: official development
    assistance; every provider."
    """

    kind: AssumptionKind = Field(
        description=(
            "what the engine filled. 'name_corrected' is an entity the request typed in "
            "words that are no listed name or alias of it, such as a misspelling, read as "
            "that entity. 'declared_default' is a slot the request left open "
            "that its code system's or shape's declared default fixed. 'open_role' is a "
            "flow role no request word asked for, read across every entity on it with "
            "data, as one series per member where the role scope default filled it and "
            "the shape declares the role, else as the total over them. 'place_scope' is "
            "a request that named no place, read across every place with data"
        )
    )
    binding_ref: str = Field(
        max_length=_MAX_BINDING_REF_CHARS,
        description=(
            "the identity binding whose edit replaces this assumption: 'how:<property "
            "dcid>' for a declared default, 'origin:' or 'where:' for a role, the place "
            "scope or the role a corrected name binds, as on the sentence's own tokens"
        ),
    )
    text: str = Field(
        min_length=1,
        max_length=MAX_ASSUMPTION_TEXT_CHARS,
        description=(
            "the words the sentence's assumptions clause states for this fill: the typed "
            "words in quotes and the entity's label for a corrected name ('\"Germny\" read "
            "as Germany'), the bound "
            "code's full label for a declared default ('official development assistance' "
            "for CRSTypeOfFlow10, whose short name is 'ODA'), "
            "'every' and the role's noun in the answer's flow family for an open role "
            "('every provider', 'every creditor'), and 'every country', 'every place' or "
            "'the world as a whole' for the place scope, as its members are typed"
        ),
    )
    refs: list[Annotated[str, Field(max_length=_MAX_REF_CHARS)]] = Field(
        default_factory=list,
        max_length=_MAX_REFS,
        description=(
            "the value dcid a declared default bound its slot to, as the slot's "
            "trace.slot_filters[].refs list it (['CRSTypeOfFlow10'] for official "
            "development assistance), or the entity dcid a corrected name reads "
            "(['country/DEU']). An open role and the place scope carry []. A "
            "dataset_resubmit echo matches a declared default to its slot, and a "
            "corrected name to its role's entity, by these values"
        ),
    )
    asked: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ASSUMPTION_TEXT_CHARS,
        description=(
            "the words the request typed for a corrected name ('Germny'), which `text` "
            "quotes; absent on every other kind. A dataset_resubmit echo restates a "
            "corrected name with these words, and drops one that carries none"
        ),
    )
