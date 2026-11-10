"""Search marks the donors that are multilateral organisations and states whether a donor
population can hold one (`search.engine.multilaterals`), so a caller can tell a multilateral's
own outflows, which spend its funders' core contributions, from a donor's total that counts
those contributions.

Each bound party states its kind and class (`multilaterals.party_facts`) beside the marker.

The parts resolve through the parsed door over the fixture graph of `tests.scope.test_basis`
with the International Development Association, Bloomberg Philanthropies, Africa's unallocated
residual and DAC countries' total beside it, and coverage holding CRS health series from the
United States, France, IDA and Bloomberg Philanthropies to Kenya, from the United States to
the residual, and DAC1 series for the United States and DAC countries' total.
"""

from __future__ import annotations

import pytest

from code_index.types import PartyRow
from search.engine.multilaterals import PartyFacts, is_multilateral, party_facts
from search.models import (
    DatasetResubmitInput,
    DefinitePart,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    Resolution,
    ResolveRequest,
    ResubmitIdentityEntry,
    Spec,
)
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from tests.engine._fake_coverage import patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.fixtures import FakeGraph
from tests.scope import test_basis as basis

_USA = "country/USA"
_FRA = "country/FRA"
_KEN = "country/KEN"
_IDA = "InternationalDevelopmentAssociation"
_BLOOMBERG = "BloombergPhilanthropies"
_AFRICA_UNALLOCATED = "DACAfricaUnallocated"
_DAC_COUNTRIES = "DACCountriesTotal"
_YEARS = list(range(2005, 2025))


@pytest.fixture(autouse=True)
def _coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    donor_only = RoleNames(donor="financingProvider", recipient=None)
    flow = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            *(
                (dcid, "dac1", donor, ABSENT_ENTITY, _YEARS)
                for dcid in basis._DAC1_MEMBERS
                for donor in (_USA, _DAC_COUNTRIES)
            ),
            *(
                (dcid, "crs", donor, _KEN, _YEARS)
                for dcid in basis._CRS_MEMBERS
                for donor in (_USA, _FRA, _IDA, _BLOOMBERG)
            ),
            *((dcid, "crs", _USA, _AFRICA_UNALLOCATED, _YEARS) for dcid in basis._CRS_MEMBERS),
        ],
        role_names={
            **dict.fromkeys(basis._DAC1_MEMBERS, donor_only),
            **dict.fromkeys(basis._CRS_MEMBERS, flow),
        },
        facet_metadata={
            "dac1": FacetMetadata(provenance_id=basis._DAC1, unit="USD"),
            "crs": FacetMetadata(provenance_id=basis._CRS, unit="USD"),
        },
    )


def _spec(request: ResolveRequest, variable: str) -> Spec:
    members = (*basis._DAC1_MEMBERS, *basis._CRS_MEMBERS)
    graph = FakeGraph(
        nodes={
            _USA: {"label": "United States", "type": "Country"},
            _KEN: {"label": "Kenya", "type": "Country"},
            _IDA: {"label": "International Development Association", "type": "Organization"},
            _BLOOMBERG: {"label": "Bloomberg Philanthropies", "type": "Organization"},
            _AFRICA_UNALLOCATED: {
                "label": "Africa, unallocated by country (DAC)",
                "type": "Organization",
            },
            _DAC_COUNTRIES: {"label": "DAC countries, total", "type": "Organization"},
            **{dcid: basis._member_node(dcid) for dcid in members},
        },
        detect={variable: {"svs": list(members), "cosine_scores": [0.9] * len(members)}},
        resolve={
            "United States": _USA,
            "Kenya": _KEN,
            "IDA": _IDA,
            "Bloomberg Philanthropies": _BLOOMBERG,
        },
    )
    response = offline_resolve(
        request,
        graph=graph,
        llm=basis._PickingLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    return part.spec


def _health_aid(donor: str, recipient: ParsedEntity | None = None) -> Spec:
    variable = "health aid"
    entities = [
        ParsedEntity(text=donor, role_text="from"),
        recipient or ParsedEntity(text="Kenya", role_text="to"),
    ]
    return _spec(
        ResolveRequest(
            input=ParsedInput(
                parts=[ParsedPart(variable_text=variable, entities=entities, time_text="2019")]
            )
        ),
        variable,
    )


def _facts(spec: Spec) -> dict[tuple[str, str], tuple[bool | None, str | None, str | None]]:
    return {
        (entity.role, entity.ref): (entity.multilateral, entity.party_kind, entity.party_class)
        for entity in spec.resolved.entities
    }


_KENYA = ("destination", _KEN), (None, "party", "official")
_TO_AFRICA_UNALLOCATED = ParsedEntity(
    text="Africa, unallocated", role_text="to", dcid=_AFRICA_UNALLOCATED
)


@pytest.mark.parametrize(
    ("donor", "recipient", "facts"),
    [
        pytest.param(
            "IDA", None, [(("origin", _IDA), (True, "party", "official")), _KENYA], id="ida"
        ),
        pytest.param(
            "United States",
            None,
            [(("origin", _USA), (False, "party", "official")), _KENYA],
            id="united-states",
        ),
        pytest.param(
            "Bloomberg Philanthropies",
            None,
            [(("origin", _BLOOMBERG), (False, "party", "private")), _KENYA],
            id="private-donor",
        ),
        pytest.param(
            "United States",
            _TO_AFRICA_UNALLOCATED,
            [
                (("origin", _USA), (False, "party", "official")),
                (("destination", _AFRICA_UNALLOCATED), (None, "residual", "residual")),
            ],
            id="residual-recipient",
        ),
    ],
)
def test_each_bound_party_states_its_kind_class_and_whether_it_is_multilateral(
    donor: str, recipient: ParsedEntity | None, facts: list
):
    """IDA's outflows spend its funders' contributions, so its origin entity is marked
    multilateral; every other origin party is marked false, and a recipient carries no
    marker. Each party states its kind and class."""
    assert _facts(_health_aid(donor, recipient)) == dict(facts)


def test_a_party_the_datasets_give_different_kinds_states_no_facts():
    """A trust fund that is a party in one dataset and a total in another has no one kind
    across both, so it states no kind, no class and no multilateral marker."""
    row = PartyRow(
        dcid="IMFConcessionalTrustFunds",
        kind="party",
        party_class="official",
        published=(("A", "origin"), ("B", "origin")),
        dataset_kinds=(("B", "total"),),
    )

    assert party_facts(row, datasets=("A", "B")) == PartyFacts(None, None, None)
    assert not is_multilateral(row, datasets=("A", "B"))


def test_a_field_ignoring_consumer_reads_the_entities_as_before():
    """Dropping the kind and class, and a false marker, leaves the shape the entities had
    before they stated them."""
    spec = _health_aid("United States")

    entities = [
        {
            key: value
            for key, value in entity.items()
            if key not in {"party_kind", "party_class"} and (key, value) != ("multilateral", False)
        }
        for entity in spec.model_dump(mode="json")["resolved"]["entities"]
    ]
    assert entities == [
        {
            "ref": _KEN,
            "label": "Kenya",
            "entity_type": "Country",
            "role": "destination",
            "confidence": None,
        },
        {
            "ref": _USA,
            "label": "United States",
            "entity_type": "Country",
            "role": "origin",
            "confidence": None,
        },
    ]


def test_dac_countries_total_holds_no_multilateral():
    """A donor-less DAC1 question binds DAC countries' total, whose members are countries."""
    variable = "aid"
    spec = _spec(
        ResolveRequest(
            input=ParsedInput(parts=[ParsedPart(variable_text=variable, time_text="2019")])
        ),
        variable,
    )

    group = (spec.resolved.entity_groups or {})["origin"]
    assert (group.total, group.holds_multilaterals) == (_DAC_COUNTRIES, False)
    # The total is marked false and states its kind; it holds no one class's whole population.
    assert _facts(spec) == {("origin", _DAC_COUNTRIES): (False, "total", None)}


def test_an_open_crs_donor_role_may_hold_a_multilateral():
    """A resubmit of a CRS part with its donor dropped reads every CRS donor, IDA among
    them, so the donor axis states it holds a multilateral and the recipient axis states
    nothing."""
    part = _health_aid("United States")
    identity = part.definition.identity
    entities = {role: refs for role, refs in identity.entities.items() if role != "origin"}

    whole = _spec(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(
                        variable_text="health aid",
                        identity=identity.model_copy(update={"entities": entities}),
                    )
                ]
            )
        ),
        "health aid",
    )

    facts = whole.resolved.role_axis_facts or []
    assert [(fact.role, fact.holds_multilaterals) for fact in facts] == [("origin", True)]
