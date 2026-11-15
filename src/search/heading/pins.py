"""The slots an identity's elected dataset pins, which no sentence states.

`CodeIndex.dataset_pins` maps a dataset and a five-tuple to each property every index row
the dataset publishes of it carries with one value: `crsTypeOfFlow` and `dac2aAidType` on
the ONE imputed multilateral table. An identity reads the pins of the dataset elected for
its family (`code_index.members.elected_pins`). A pinned slot tells no member of the
family apart.
"""

from __future__ import annotations

from code_index.composites import PopulationUnavailable, table_identity
from code_index.members import elected_pins, identity_family_query
from code_index.shared import get_index
from search.models import BaseIdentity, IdentityShapeSchema


def identity_pinned_properties(identity: BaseIdentity) -> frozenset[str]:
    """The properties the dataset elected for `identity`'s family pins, or for a composite
    those of the identity its option was set on (`code_index.composites.table_identity`).
    Refuses a composite whose population is unavailable as Search's doors refuse it
    (`identity_edit.population_refusal`). Empty for an identity that names its one
    StatVar."""
    if not isinstance(identity.shape, IdentityShapeSchema):
        return frozenset()
    index = get_index()
    try:
        document = table_identity(
            index, identity.model_dump(mode="json", exclude_none=True)
        ).identity
    except PopulationUnavailable as exc:
        # Importing search.engine runs its package init, which imports this module.
        from search.engine.identity_edit import population_refusal

        raise population_refusal(exc, pointer="/input/identity") from exc
    query = identity_family_query(index, document)
    return frozenset(elected_pins(index, query))
