"""The slots an identity's elected dataset pins, which no sentence states.

`CodeIndex.dataset_pins` maps a dataset and a five-tuple to each property every index row
the dataset publishes of it carries with one value: `crsTypeOfFlow` and `dac2aAidType` on
the ONE imputed multilateral table. An identity reads the pins of the dataset elected for
its family (`code_index.members.elected_pins`). A pinned slot tells no member of the
family apart.
"""

from __future__ import annotations

from code_index.members import elected_pins, identity_family_query
from code_index.shared import get_index
from search.models import BaseIdentity, IdentityShapeSchema


def identity_pinned_properties(identity: BaseIdentity) -> frozenset[str]:
    """The properties the dataset elected for `identity`'s family pins. Empty for an
    identity that names its one StatVar."""
    if not isinstance(identity.shape, IdentityShapeSchema):
        return frozenset()
    index = get_index()
    query = identity_family_query(index, identity.core_document(mode="json"))
    return frozenset(elected_pins(index, query))
