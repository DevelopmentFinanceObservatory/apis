"""A composite component that binds its destinations to the bilateral side reads DAC2a's
bilateral population (`code_index.composites.destination_population`): an open destination
binds the population's members and is tagged as a population, a named one passes only on the
bilateral side, and an absent one binds the population's total. A destination no side
declaration places, in the code index or in coverage, makes the population unavailable."""

from __future__ import annotations

import gc
import gzip
import json
import weakref

import pytest

import code_index.composites as composites_module
from code_index.composites import (
    CompositeNotServable,
    PopulationUnavailable,
    UnclassifiedDestinations,
    component_identities,
    composite_digest,
    destination_population,
    population_check,
    table_identity,
)
from code_index.declarations import (
    COMPOSITES,
    CRS_INCLUSIVE,
    DAC2A,
    DAC2A_BILATERAL,
    DAC2A_INCLUSIVE,
)
from code_index.loader import parse_index
from code_index.query import CodeIndex
from code_index.scope_conventions import (
    BILATERAL_SOURCES,
    DAC_DEVELOPING_COUNTRIES,
    destination_side,
)
from series_coverage.schema import FacetMetadata, RoleNames
from tests.code_index.aid_identity import AID_TYPE
from tests.code_index.support import INDEX_FIXTURE
from tests.engine._fake_coverage import build_coverage_index

_DAC2A_INCLUSIVE, _BILATERAL = DAC2A_INCLUSIVE, DAC2A_BILATERAL
_KENYA = "country/KEN"
_FRANCE = "country/FRA"


def _parse() -> CodeIndex:
    with gzip.open(INDEX_FIXTURE) as handle:
        return parse_index(json.load(handle), source="fixture", composites=COMPOSITES)


_INDEX = _parse()


def _identity(provenance: str, **entities: object) -> dict:
    """France's net DAC2a total ODA over every year, binding the composite `provenance`, with
    `entities` merged over France as origin."""
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
            {"property": "nettingBasis", "binding": {"state": "value", "ref": "Net"}},
        ],
        "entities": {"origin": [_FRANCE]},
        "requested_window": {"kind": "all"},
        "facets": {"provenance": {"state": "value", "ref": provenance}},
    }
    open_roles = entities.pop("open_roles", None)
    identity["entities"].update(entities)
    if open_roles is not None:
        identity["open_roles"] = open_roles
    return identity


def test_the_population_holds_bilateral_parties_and_residuals_and_names_its_totals():
    population = destination_population(_INDEX, DAC2A)

    assert population.side == "bilateral"
    assert population.total == DAC_DEVELOPING_COUNTRIES
    assert {_KENYA, "DACBilateralUnallocated", "EastAfricanCommunity"} <= set(population.members)
    assert not {"EuropeanCommission", "DACOtherMultilateralRecipient", "DACAllRecipients"} & (
        population.named
    )
    assert not {DAC_DEVELOPING_COUNTRIES, "DACAfrica"} & set(population.members)
    assert {DAC_DEVELOPING_COUNTRIES, "DACAfrica"} <= population.named
    assert set(population.members) <= population.named
    assert list(population.members) == sorted(population.members)


@pytest.mark.parametrize("provenance", [_BILATERAL, _DAC2A_INCLUSIVE])
def test_an_open_destination_binds_the_population_and_tags_its_key(provenance: str):
    population = destination_population(_INDEX, DAC2A)

    components = component_identities(_INDEX, _identity(provenance, open_roles=["destination"]))

    assert components is not None
    for component in components:
        assert component.identity["entities"] == {
            "destination": list(population.members),
            "origin": [_FRANCE],
        }
        assert "open_roles" not in component.identity
        assert component.populations == {"destination": population}


@pytest.mark.parametrize(
    "named", [[_KENYA], [DAC_DEVELOPING_COUNTRIES], [_KENYA, "DACAfrica"], ["country/ZZZ"]]
)
def test_a_named_destination_on_the_bilateral_side_passes_untagged(named: list[str]):
    """A ref DAC2a does not publish as a destination, such as `country/ZZZ`, passes and reads
    no rows, as on the plain table."""
    components = component_identities(_INDEX, _identity(_BILATERAL, destination=named))

    assert components is not None
    (component,) = components
    assert component.identity["entities"]["destination"] == named
    assert component.populations == {}


@pytest.mark.parametrize(
    "named",
    [
        ["EuropeanCommission"],
        ["DACAllRecipients"],
        ["DACAllMultilateralRecipients"],
        [_KENYA, "DACOtherMultilateralRecipient"],
    ],
)
@pytest.mark.parametrize("provenance", [_BILATERAL, _DAC2A_INCLUSIVE])
def test_a_named_destination_off_the_bilateral_side_is_refused(provenance: str, named: list[str]):
    with pytest.raises(CompositeNotServable) as refused:
        component_identities(_INDEX, _identity(provenance, destination=named))

    assert refused.value.ref == provenance
    assert refused.value.pointer == "/entities/destination"
    assert refused.value.refs == tuple(ref for ref in named if ref != _KENYA)
    assert refused.value.key == "destination"


def test_an_absent_destination_binds_the_population_s_total_untagged():
    components = component_identities(_INDEX, _identity(_BILATERAL))

    assert components is not None
    (component,) = components
    assert component.identity["entities"]["destination"] == [DAC_DEVELOPING_COUNTRIES]
    assert component.populations == {}


def test_the_table_identity_carries_the_first_component_s_population():
    population = destination_population(_INDEX, DAC2A)

    read = table_identity(_INDEX, _identity(_DAC2A_INCLUSIVE, open_roles=["destination"]))

    assert read.identity["entities"]["destination"] == list(population.members)
    assert "facets" not in read.identity
    assert read.populations == {"destination": population}


def test_the_table_identity_of_an_identity_binding_no_composite_is_itself():
    identity = _identity(DAC2A, open_roles=["destination"])

    read = table_identity(_INDEX, identity)

    assert read.identity is identity
    assert read.populations == {}


def _placing_kenya_as(monkeypatch: pytest.MonkeyPatch, side: str | None) -> CodeIndex:
    """A fresh index, so the population memo builds again, whose classifier places Kenya on
    `side`."""

    def placed(row, *, datasets):
        return side if row.dcid == _KENYA else destination_side(row, datasets=datasets)

    monkeypatch.setattr(composites_module, "destination_side", placed)
    return _parse()


@pytest.mark.parametrize("provenance", [_BILATERAL, _DAC2A_INCLUSIVE])
def test_a_destination_the_code_index_cannot_place_refuses_the_composite_reading_it(
    monkeypatch: pytest.MonkeyPatch, provenance: str
):
    """The refusal is a `CompositeNotServable`, so every handler of one catches it, and names
    the composite asked for and the drifted destination."""
    index = _placing_kenya_as(monkeypatch, None)
    expected = UnclassifiedDestinations(
        dataset=DAC2A, destinations=(_KENYA,), found_in="code_index"
    )

    with pytest.raises(CompositeNotServable) as drifted:
        component_identities(index, _identity(provenance, open_roles=["destination"]))

    assert isinstance(drifted.value, PopulationUnavailable)
    assert drifted.value.unclassified == expected
    assert (drifted.value.ref, drifted.value.pointer) == (provenance, "/facets/provenance")
    coverage = _coverage([_KENYA])
    assert population_check(index, coverage).refusal(provenance) == expected


def test_the_composite_digest_moves_with_the_population(monkeypatch: pytest.MonkeyPatch):
    composite = _INDEX.composite(_BILATERAL)
    assert composite is not None
    before = composite_digest(composite, index=_INDEX)

    after = composite_digest(composite, index=_placing_kenya_as(monkeypatch, "multilateral"))

    assert after != before


# The variables DAC2a's population reads in the fixture index: the 206 and imputed 106
# totals a component binding its destinations binds.
_POPULATION_VARIABLES = composites_module._population_variables(_INDEX, DAC2A)


def _coverage(recipients: list[str], *, metadata: FacetMetadata | None = None):
    """Coverage holding `recipients` for France under each population variable, in one facet
    whose metadata is `metadata`, DAC2a's by default."""
    return build_coverage_index(
        [
            (variable, "dac2a", _FRANCE, recipient, [2021])
            for variable in _POPULATION_VARIABLES
            for recipient in recipients
        ],
        role_names={
            variable: RoleNames(donor="financingProvider", recipient="financingRecipient")
            for variable in _POPULATION_VARIABLES
        },
        facet_metadata={"dac2a": metadata or FacetMetadata(provenance_id=DAC2A, unit="USD")},
    )


def test_coverage_whose_recipients_the_population_places_leaves_it_available():
    check = population_check(_INDEX, _coverage([_KENYA, "EuropeanCommission", "DACAllRecipients"]))

    assert check.unclassified == {}
    assert check.refusal(_BILATERAL) is None


@pytest.mark.parametrize("provenance", [_BILATERAL, _DAC2A_INCLUSIVE])
def test_a_covered_recipient_the_code_index_does_not_publish_makes_the_population_unavailable(
    provenance: str,
):
    check = population_check(_INDEX, _coverage([_KENYA, "SomeNewAgency"]))

    assert check.refusal(provenance) == UnclassifiedDestinations(
        dataset=DAC2A, destinations=("SomeNewAgency",), found_in="coverage"
    )
    assert check.refusal(CRS_INCLUSIVE) is None


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(FacetMetadata(unit="USD"), id="no_provenance"),
        pytest.param(
            FacetMetadata(provenance_id="provenance/OECDCRS", unit="USD"), id="another_provenance"
        ),
    ],
)
def test_coverage_rows_not_attributed_to_the_dataset_make_the_population_unavailable(
    metadata: FacetMetadata,
):
    """The check fails closed: a destination in rows coverage cannot attribute to the dataset
    could sit off every side unseen. The reason names the variables."""
    check = population_check(_INDEX, _coverage([_KENYA], metadata=metadata))

    assert check.refusal(_BILATERAL) == UnclassifiedDestinations(
        dataset=DAC2A, destinations=(), found_in="coverage", unlisted=_POPULATION_VARIABLES
    )
    assert _POPULATION_VARIABLES[0] in check.unclassified[DAC2A].message()


def test_every_declared_component_binding_destinations_has_a_declared_source():
    """A component binding the destinations of a dataset no `BILATERAL_SOURCES` row declares
    would fail at its first read."""
    bound = {
        recipe.dataset
        for composite in COMPOSITES
        for recipe in composite.components
        if recipe.destinations is not None
    }
    assert bound == {source.dataset for source in BILATERAL_SOURCES} == {DAC2A}


def test_the_population_check_memo_holds_one_index_finalizer_however_many_coverages(
    monkeypatch: pytest.MonkeyPatch,
):
    """A code index outlives many coverage snapshots. Each check is dropped with its own
    coverage, and a snapshot after the first adds no finalizer to the index."""
    index = _parse()
    finalized: list[object] = []
    register = weakref.finalize

    def recording(obj, *args, **kwargs):
        finalized.append(obj)
        return register(obj, *args, **kwargs)

    monkeypatch.setattr(weakref, "finalize", recording)
    coverages = [_coverage([_KENYA]) for _ in range(3)]
    first = population_check(index, coverages[0])
    on_index = sum(obj is index for obj in finalized)

    for coverage in coverages[1:]:
        population_check(index, coverage)

    assert population_check(index, coverages[0]) is first
    assert sum(obj is index for obj in finalized) == on_index
    assert [obj for obj in finalized if obj is not index] == coverages


def test_dropping_a_code_index_frees_its_check_memo_while_a_coverage_index_lives():
    """The finalizer on a coverage index must not hold the code index's memo of checks."""
    index = _parse()
    coverage = _coverage([_KENYA])
    population_check(index, coverage)
    memo = weakref.ref(composites_module._checks(index))

    del index
    gc.collect()

    assert memo() is None
