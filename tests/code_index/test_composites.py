"""The code index serves the declared composites, refuses malformed ones, and derives each
composite identity's component identities (`code_index.composites.component_identities`)."""

from __future__ import annotations

import dataclasses
import gzip
import json

import pytest

from code_index import IndexFormatError
from code_index.composites import (
    CompositeNotServable,
    CompositeProvenanceRead,
    component_identities,
)
from code_index.declarations import COMPOSITES
from code_index.loader import parse_index
from code_index.members import pinned_refs
from code_index.query import CodeIndex
from code_index.types import YearPeriod
from tests.code_index.aid_identity import (
    AID_TYPE,
    CRS_PROVENANCE,
    IMPUTED_MULTILATERAL,
    IMPUTED_PROVENANCE,
    education_aid_identity,
    imputed_education_aid_identity,
)
from tests.code_index.support import INDEX_FIXTURE

_STACKED, _DAC2A_STACKED, _SPLICED = COMPOSITES


def _payload() -> dict:
    with gzip.open(INDEX_FIXTURE) as handle:
        return json.load(handle)


def _parse(composites=COMPOSITES) -> CodeIndex:
    return parse_index(_payload(), source="fixture", composites=composites)


_INDEX = _parse()


def _with_component(composite, role, **fields):
    return dataclasses.replace(
        composite,
        components=tuple(
            dataclasses.replace(recipe, **fields) if recipe.role == role else recipe
            for recipe in composite.components
        ),
    )


def test_the_fixture_index_serves_every_declared_composite():
    assert _INDEX.composite(_STACKED.provenance) == _STACKED
    assert _INDEX.composite(_DAC2A_STACKED.provenance) == _DAC2A_STACKED
    assert _INDEX.composite(_SPLICED.provenance) == _SPLICED
    assert _INDEX.composite(CRS_PROVENANCE) is None
    # Both component tables read an absent row as zero, so the composite does too.
    assert _INDEX.absent_observation(_STACKED.provenance) == "zero"


@pytest.mark.parametrize(
    ("composites", "message"),
    [
        pytest.param(
            (_with_component(_SPLICED, "headline_ge", periods=YearPeriod(2017, None)),),
            "overlap or leave a year out",
            id="overlapping-spliced-periods",
        ),
        pytest.param(
            (_with_component(_SPLICED, "headline_ge", periods=YearPeriod(2019, None)),),
            "overlap or leave a year out",
            id="spliced-gap",
        ),
        pytest.param(
            (_with_component(_STACKED, "imputed", periods=YearPeriod(2010, None)),),
            "is stacked",
            id="stacked-periods",
        ),
        pytest.param(
            (dataclasses.replace(_STACKED, provenance=CRS_PROVENANCE),),
            "also declared as a dataset",
            id="collides-with-a-dataset",
        ),
        pytest.param(
            (
                dataclasses.replace(
                    _STACKED, components=(_STACKED.components[0], _STACKED.components[0])
                ),
            ),
            "declares components",
            id="duplicate-role",
        ),
        pytest.param(
            (_with_component(_SPLICED, "headline_net", measured_property="amount"),),
            "both a measured property and a basis",
            id="measure-and-basis",
        ),
        pytest.param(
            (_with_component(_SPLICED, "headline_net", zero_absent=True),),
            "reads absence as zero",
            id="spliced-zero-absent",
        ),
    ],
)
def test_the_loader_refuses_a_malformed_composite(composites, message):
    with pytest.raises(IndexFormatError, match=message):
        _parse(composites)


def test_a_composite_whose_component_table_the_index_lacks_is_not_served():
    absent = _with_component(_STACKED, "imputed", dataset="provenance/NotDeclared")

    index = _parse((absent,))

    assert index.composite(_STACKED.provenance) is None
    with pytest.raises(CompositeNotServable, match="serves no component table"):
        component_identities(index, education_aid_identity(source=_STACKED.provenance))


def test_the_stacked_components_are_the_crs_identity_and_its_imputed_counterpart():
    identity = education_aid_identity(source=_STACKED.provenance)

    components = component_identities(_INDEX, identity)

    assert components is not None
    assert [component.role for component in components] == ["bilateral", "imputed"]
    assert components[0].identity == education_aid_identity()
    assert components[1].identity == imputed_education_aid_identity(source=IMPUTED_PROVENANCE)


_DAC2A = "provenance/OECDDAC2A"


def _dac2a_oda_identity(netting: str, *, source: str | None = None) -> dict:
    """DAC2a ODA from the United States to Kenya on the `netting` basis, binding `source`."""
    identity: dict = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "FinancialFlow",
            "measured_property": "amount",
            "stat_type": "measuredValue",
            "measurement_qualifier": "Nominal",
        },
        "slots": [
            {"property": AID_TYPE, "binding": {"state": "value", "ref": "DAC2aAidType206"}},
            {"property": "financialFlowType", "binding": {"state": "value", "ref": "Disbursement"}},
            {"property": "nettingBasis", "binding": {"state": "value", "ref": netting}},
        ],
        "entities": {"origin": ["country/USA"], "destination": ["country/KEN"]},
        "requested_window": {"kind": "all"},
    }
    if source is not None:
        identity["facets"] = {"provenance": {"state": "value", "ref": source}}
    return identity


@pytest.mark.parametrize("netting", ["Gross", "Net"])
def test_dac2a_stacks_its_total_oda_on_its_own_gross_imputed_line_on_either_basis(netting: str):
    """DAC2a publishes its imputed multilateral line gross alone, and the convention adds it
    to total ODA on either basis, so the imputed component reads the gross line from DAC2a
    itself whatever basis the composite asks for."""
    components = component_identities(
        _INDEX, _dac2a_oda_identity(netting, source=_DAC2A_STACKED.provenance)
    )

    assert components is not None
    bilateral, imputed = components
    assert bilateral.identity == _dac2a_oda_identity(netting)
    assert imputed.identity == {
        **_dac2a_oda_identity("Gross", source=_DAC2A),
        "shape": {**_dac2a_oda_identity("Gross")["shape"], "measured_property": "attributedAmount"},
        "slots": [
            {"property": AID_TYPE, "binding": {"state": "value", "ref": IMPUTED_MULTILATERAL}},
            {"property": "financialFlowType", "binding": {"state": "value", "ref": "Disbursement"}},
            {"property": "nettingBasis", "binding": {"state": "value", "ref": "Gross"}},
        ],
    }
    assert not imputed.recipe.zero_absent


def test_an_identity_binding_no_composite_has_no_components():
    assert component_identities(_INDEX, education_aid_identity(source=CRS_PROVENANCE)) is None


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        pytest.param(
            {"kind": "range", "start_year": 2010, "end_year": 2022},
            {
                "headline_net": {"kind": "range", "start_year": 2010, "end_year": 2017},
                "headline_ge": {"kind": "range", "start_year": 2018, "end_year": 2022},
            },
            id="spanning-the-splice",
        ),
        pytest.param(
            {"kind": "range", "start_year": 2019, "end_year": 2022},
            {"headline_ge": {"kind": "range", "start_year": 2019, "end_year": 2022}},
            id="after-the-splice",
        ),
        pytest.param(
            {"kind": "all"},
            {
                "headline_net": {"kind": "range", "end_year": 2017},
                "headline_ge": {"kind": "range", "start_year": 2018},
            },
            id="all-years",
        ),
    ],
)
def test_a_spliced_component_reads_its_own_years_of_the_window(window, expected):
    identity = education_aid_identity(source=_SPLICED.provenance) | {"requested_window": window}

    components = component_identities(_INDEX, identity)

    assert components is not None
    assert {c.role: c.identity["requested_window"] for c in components} == expected


@pytest.mark.parametrize(
    ("edit", "pointer"),
    [
        pytest.param(
            lambda identity: identity["slots"][0].update(
                binding={"state": "value", "ref": "CRSCooperationModalityC01"}
            ),
            "/slots/0",
            id="bound-modality",
        ),
        pytest.param(
            lambda identity: identity.update(
                facets={
                    "provenance": {"state": "set", "refs": [_STACKED.provenance, CRS_PROVENANCE]}
                }
            ),
            "/facets/provenance",
            id="composite-in-a-set",
        ),
        pytest.param(
            lambda identity: identity.update(shape={"kind": "stat_var", "ref": "Amount_X"}),
            "/shape",
            id="pinned-shape",
        ),
    ],
)
def test_a_composite_identity_its_components_cannot_serve_is_refused(edit, pointer):
    identity = education_aid_identity(source=_STACKED.provenance)
    edit(identity)

    with pytest.raises(CompositeNotServable) as refused:
        component_identities(_INDEX, identity)

    assert refused.value.pointer == pointer


def test_a_provenance_pin_read_as_a_dataset_refuses_a_composite():
    with pytest.raises(CompositeProvenanceRead):
        pinned_refs(education_aid_identity(source=_STACKED.provenance)["facets"])
