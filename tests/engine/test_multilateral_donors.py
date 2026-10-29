"""Search marks the donors that are multilateral organisations and states whether a donor
population can hold one (`search.engine.multilaterals`), so a caller can tell a multilateral's
own outflows, which spend its funders' core contributions, from a donor's total that counts
those contributions.

The parts resolve through the parsed door over the fixture graph of `tests.scope.test_basis`
with the International Development Association and DAC countries' total beside it, and
coverage holding CRS health series from the United States, France and IDA to Kenya and DAC1
series for the United States and DAC countries' total.
"""

from __future__ import annotations

import pytest

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
                for donor in (_USA, _FRA, _IDA)
            ),
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
            _DAC_COUNTRIES: {"label": "DAC countries, total", "type": "Organization"},
            **{dcid: basis._member_node(dcid) for dcid in members},
        },
        detect={variable: {"svs": list(members), "cosine_scores": [0.9] * len(members)}},
        resolve={"United States": _USA, "Kenya": _KEN, "IDA": _IDA},
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


def _health_aid_to_kenya(donor: str) -> Spec:
    variable = "health aid"
    entities = [
        ParsedEntity(text=donor, role_text="from"),
        ParsedEntity(text="Kenya", role_text="to"),
    ]
    return _spec(
        ResolveRequest(
            input=ParsedInput(
                parts=[ParsedPart(variable_text=variable, entities=entities, time_text="2019")]
            )
        ),
        variable,
    )


@pytest.mark.parametrize(
    ("donor", "ref", "multilateral"),
    [
        pytest.param("IDA", _IDA, True, id="ida"),
        pytest.param("United States", _USA, None, id="united-states"),
    ],
)
def test_a_crs_donor_is_marked_multilateral_only_when_it_is_one(
    donor: str, ref: str, multilateral: bool | None
):
    """IDA's outflows spend its funders' contributions, so its origin entity carries the
    marker. The United States and the recipient carry none."""
    spec = _health_aid_to_kenya(donor)

    marks = {(entity.role, entity.ref): entity.multilateral for entity in spec.resolved.entities}
    assert marks == {("origin", ref): multilateral, ("destination", _KEN): None}


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


def test_an_open_crs_donor_role_may_hold_a_multilateral():
    """A resubmit of a CRS part with its donor dropped reads every CRS donor, IDA among
    them, so the donor axis states it holds a multilateral and the recipient axis states
    nothing."""
    part = _health_aid_to_kenya("United States")
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
