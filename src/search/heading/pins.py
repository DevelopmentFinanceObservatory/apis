"""The slots an identity's five-tuple pins, which no sentence states.

`CodeIndex.five_tuple_pins` maps a five-tuple to each property every index row of it
carries with one value: `crsTypeOfFlow` and `dac2aAidType` on the ONE imputed
multilateral five-tuple. A pinned slot tells no member of the five-tuple apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from code_index.shared import get_index
from search.models import BaseIdentity
from statvar_members import five_tuple_from_identity


def identity_pinned_properties(identity: BaseIdentity) -> frozenset[str]:
    """The properties `identity`'s five-tuple pins. Empty for an identity that names its one
    StatVar."""
    shape = cast("Mapping[str, object]", identity.core_document(mode="json")["shape"])
    if shape.get("kind") == "stat_var":
        return frozenset()
    return frozenset(get_index().five_tuple_pins.get(five_tuple_from_identity(shape), {}))
