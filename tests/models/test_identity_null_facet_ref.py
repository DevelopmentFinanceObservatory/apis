"""A null facet ref binds measurementMethod or unit to the facets that declare no value.

`BaseIdentity` mirrors the identity core: the null ref is legal on those two axes only,
survives `exclude_none` dumps, and mints the id the `spec-kit` repository's
`facet-null-ref-measurement-method` vector pins.
"""

from __future__ import annotations

import pytest
from dfo_spec_kit.identity import spec_id_from_value
from pydantic import ValidationError

from search.models import BaseIdentity, IdentityFacetValueAbsent


def _identity(**overrides: object) -> dict:
    document: dict = {
        "identity_version": "spec1",
        "shape": {"kind": "statvar_schema", "measured_property": "developmentFinanceFlow"},
        "slots": [
            {
                "property": "DevelopmentFinancePurpose",
                "binding": {"state": "value", "ref": "Malaria"},
            }
        ],
        "entities": {"origin": ["country/FRA"]},
        "requested_window": {"kind": "all"},
    }
    document.update(overrides)
    return document


@pytest.mark.parametrize("axis", ["measurementMethod", "unit"])
def test_a_null_ref_on_method_or_unit_parses_and_survives_exclude_none(axis):
    identity = BaseIdentity.model_validate(
        _identity(facets={axis: {"state": "value", "ref": None}})
    )

    assert isinstance(identity.facets[axis], IdentityFacetValueAbsent)
    assert identity.model_dump(exclude_none=True)["facets"] == {
        axis: {"state": "value", "ref": None}
    }
    assert '"ref":null' in identity.model_dump_json(exclude_none=True)


def test_a_null_method_ref_mints_the_spec_kit_vector_id():
    identity = BaseIdentity.model_validate(
        _identity(facets={"measurementMethod": {"state": "value", "ref": None}})
    )

    assert (
        spec_id_from_value(identity.model_dump(exclude_none=True))
        == "spec_8972b8b9a034b8fa6754042c2fc02317"
    )


@pytest.mark.parametrize(
    "document",
    [
        _identity(facets={"provenance": {"state": "value", "ref": None}}),
        _identity(facets={"measurementMethod": {"state": "set", "refs": [None, "Estimated"]}}),
        _identity(
            slots=[
                {
                    "property": "DevelopmentFinancePurpose",
                    "binding": {"state": "value", "ref": None},
                }
            ]
        ),
    ],
    ids=["provenance", "set-member", "slot"],
)
def test_a_null_ref_is_refused_outside_a_method_or_unit_value_binding(document):
    with pytest.raises(ValidationError):
        BaseIdentity.model_validate(document)
