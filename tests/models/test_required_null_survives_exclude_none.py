"""A required, nullable field stays on the wire as null under `exclude_none`.

The API dumps responses with `exclude_none=True`, which would strip the key and leave a
payload that fails validation against the model that produced it. `StrictModel` restores
those fields on every model that has them.
"""

from __future__ import annotations

from pydantic import BaseModel

import search.models
from search.models import (
    ConversionCurrency,
    GraphRef,
    IdentityFacetValueAbsent,
    ResolvedStatVar,
    RoleAxisFact,
    StatVar,
)
from search.models.base import required_nullable_fields

# One instance per exported model with a required, nullable field, each of those fields
# None. A model added to the exports with such a field fails the walk until it is listed.
_NULL_INSTANCES: dict[type[BaseModel], BaseModel] = {
    ConversionCurrency: ConversionCurrency(
        code="SYP",
        noun="Syrian pounds",
        current_years=None,
        constant_years=None,
        base_year_min=None,
        base_year_max=None,
    ),
    IdentityFacetValueAbsent: IdentityFacetValueAbsent(ref=None),
    ResolvedStatVar: ResolvedStatVar(
        ref="Count_Person", slot_values=[], constraint_properties=None
    ),
    RoleAxisFact: RoleAxisFact(role="destination", member_count=None, members_available=False),
    StatVar: StatVar(
        ref=GraphRef(dcid="Count_Person", label="Population"),
        shape_id="shape",
        slot_values=[],
        constraint_properties=None,
    ),
}


def test_every_required_null_field_survives_exclude_none_and_validates_back():
    exported = {
        model
        for name in search.models.__all__
        if isinstance(model := getattr(search.models, name), type)
        and issubclass(model, BaseModel)
        and required_nullable_fields(model)
    }
    assert exported == set(_NULL_INSTANCES)

    for model, instance in _NULL_INSTANCES.items():
        dumped = instance.model_dump(mode="json", exclude_none=True)
        assert {name: dumped.get(name, "absent") for name in required_nullable_fields(model)} == (
            dict.fromkeys(required_nullable_fields(model))
        ), model.__name__
        assert model.model_validate(dumped) == instance, model.__name__
