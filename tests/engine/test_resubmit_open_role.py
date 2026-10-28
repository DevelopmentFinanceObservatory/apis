"""A flow side a question leaves open: a donor-less DAC1 answer binds the DAC countries'
total, and a dataset_resubmit whose identity drops a flow role reads that side as a fresh
search does.

The harness builds the whole of a part this way: it copies the part's served identity, drops
the donor (origin) or the recipient (destination) and the scope entries on that role's axis,
and posts it beside the part. The answer stays one dataset with the part's scope. On the CRS,
which publishes no total of either side, it reads every party there with the group totals
left out: every donor for a donor whole, and the regional and unallocated rows beside the
countries for a recipient whole. On DAC1 it binds the containing total its declaration names
(`code_index.declarations.CONTAINING_TOTALS`), "DAC countries, total", where coverage holds a
series there, and the group lists the DAC members the membership registry holds, so a caller
can tell whether the part's donor sits inside the total.

The parts resolve through the parsed door over the fixture graph of `tests.scope.test_basis`
(the United States, Kenya, DAC1's donor totals of ODA and CRS health members) with DAC
countries' total beside them, and coverage that adds France as a second donor, Eastern
Africa's unallocated residual as a second recipient, and the group totals DAC1 and the CRS
publish beside them.
"""

from __future__ import annotations

import pytest

from search.engine.departures import served_datasets
from search.engine.region import pipeline
from search.engine.region.whole_totals import with_containing_total
from search.models import (
    BaseIdentity,
    ChoiceOrigin,
    DatasetResubmitInput,
    DefinitePart,
    Entry,
    IdentityBindingValue,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    Resolution,
    ResolveRequest,
    ResubmitCandidate,
    ResubmitCandidatesEntry,
    ResubmitIdentityEntry,
    RoleAxis,
    Spec,
)
from series_coverage.index import CoverageIndex
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from tests.engine._fake_coverage import patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.fixtures import FakeGraph
from tests.scope import test_basis as basis

_USA = "country/USA"
_FRA = "country/FRA"
_KEN = "country/KEN"
# A non-DAC donor DAC1 publishes.
_ARE = "country/ARE"
# A residual: the CRS's "Eastern Africa, unallocated by country".
_EAST_AFRICA_UNALLOCATED = "DACEasternAfricaUnallocated"
# Group totals the code index declares beside their members.
_EAST_AFRICA = "DACEasternAfrica"
_DAC_COUNTRIES = "DACCountriesTotal"
_MULTILATERALS = "DACMultilaterals"
_DAC_GROUP = "groups/OECD.DAC"
_BILATERAL_ODA = "DAC1AidType1015"
_DAC1_BILATERAL_MEMBERS = (
    "Amount_FinancialFlow_Gross_Disbursement_DAC1AidType1015_Nominal",
    "Amount_FinancialFlow_Net_Disbursement_DAC1AidType1015_Nominal",
    "GrantEquivalent_FinancialFlow_Disbursement_DAC1AidType1015_Nominal",
)
_DAC1_MEMBERS = (*basis._DAC1_MEMBERS, *_DAC1_BILATERAL_MEMBERS)
_MEMBERS = (*basis._ALL_MEMBERS, *_DAC1_BILATERAL_MEMBERS)
_YEARS = list(range(2005, 2025))


@pytest.fixture(autouse=True)
def _coverage(monkeypatch: pytest.MonkeyPatch) -> CoverageIndex:
    """DAC1 series of total and bilateral ODA for the United States, France, the United
    Arab Emirates, DAC countries' total and the multilaterals' total; CRS health series from
    the United States to Kenya, to Eastern Africa's unallocated residual and to Eastern
    Africa's total, and from France to Kenya."""
    donor_only = RoleNames(donor="financingProvider", recipient=None)
    flow = RoleNames(donor="financingProvider", recipient="financingRecipient")
    return patch_coverage(
        monkeypatch,
        [
            *(
                (dcid, "dac1", donor, ABSENT_ENTITY, _YEARS)
                for dcid in _DAC1_MEMBERS
                for donor in (_USA, _FRA, _ARE, _DAC_COUNTRIES, _MULTILATERALS)
            ),
            *(
                (dcid, "crs", donor, recipient, _YEARS)
                for dcid in basis._CRS_MEMBERS
                for donor, recipient in (
                    (_USA, _KEN),
                    (_USA, _EAST_AFRICA_UNALLOCATED),
                    (_USA, _EAST_AFRICA),
                    (_FRA, _KEN),
                )
            ),
        ],
        role_names={
            **dict.fromkeys(_DAC1_MEMBERS, donor_only),
            **dict.fromkeys(basis._CRS_MEMBERS, flow),
        },
        facet_metadata={
            "dac1": FacetMetadata(provenance_id=basis._DAC1, unit="USD"),
            "crs": FacetMetadata(provenance_id=basis._CRS, unit="USD"),
        },
    )


def _graph(variable: str, members: tuple[str, ...] = _MEMBERS) -> FakeGraph:
    """`basis._graph` with DAC countries' total and, by default, DAC1's bilateral ODA
    members."""
    return FakeGraph(
        nodes={
            _USA: {"label": "United States", "type": "Country"},
            _KEN: {"label": "Kenya", "type": "Country"},
            _DAC_COUNTRIES: {"label": "DAC countries, total", "type": "Organization"},
            **{dcid: basis._member_node(dcid) for dcid in members},
        },
        detect={variable: {"svs": list(members), "cosine_scores": [0.9] * len(members)}},
        resolve={"United States": _USA, "Kenya": _KEN, "DAC donor countries": _DAC_COUNTRIES},
    )


def _answer(
    request: ResolveRequest, variable: str, members: tuple[str, ...] = _MEMBERS
) -> DefinitePart:
    response = offline_resolve(
        request,
        graph=_graph(variable, members),
        llm=basis._PickingLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    return part


def _resubmit(
    identity: BaseIdentity, entries: list[Entry], variable: str, groups: dict | None = None
) -> Spec:
    return _answer(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(
                        variable_text=variable,
                        identity=identity,
                        entity_groups=groups,
                        scope_entries=entries,
                    )
                ]
            )
        ),
        variable,
    ).spec


def _entries(spec: Spec) -> list[Entry]:
    scope = spec.resolved.scope_choices
    return [] if scope is None else list(scope.entries)


def _without(spec: Spec, role: RoleAxis, variable: str) -> Spec:
    """The Spec a resubmit of `spec`'s identity answers with `role` dropped from its entities
    and the scope entries on `role`'s axis (`region` for the recipient) dropped from its own."""
    identity = spec.definition.identity
    entities = {key: refs for key, refs in identity.entities.items() if key != role}
    entries = [
        entry for entry in _entries(spec) if not (role == "destination" and entry.axis == "region")
    ]
    return _resubmit(identity.model_copy(update={"entities": entities}), entries, variable)


def _choices(spec: Spec) -> list[tuple[str, str]]:
    scope = spec.resolved.scope_choices
    return [] if scope is None else [(choice.axis, choice.active) for choice in scope.choices]


def _aid_type(spec: Spec) -> str | None:
    binding = next(
        (s.binding for s in spec.definition.identity.slots if s.property == "dac1AidType"), None
    )
    return binding.ref if isinstance(binding, IdentityBindingValue) else None


def _assert_dac_countries(spec: Spec) -> None:
    """`spec` binds DAC countries' total on the donor side, named by its registry group with
    the DAC members, the United States among them and no non-DAC donor or multilateral."""
    assert spec.definition.identity.entities["origin"] == [_DAC_COUNTRIES]
    group = (spec.resolved.entity_groups or {})["origin"]
    assert (group.total, group.group_id, group.whole_of) == (_DAC_COUNTRIES, _DAC_GROUP, None)
    assert group.members is not None
    assert _USA in group.members
    assert _ARE not in group.members
    assert all(member.startswith("country/") for member in group.members)


@pytest.mark.parametrize(
    ("cell", "role", "population"),
    [
        pytest.param("sector", "origin", 2, id="crs-donor-whole"),
        pytest.param("sector", "destination", 2, id="crs-recipient-whole"),
    ],
)
def test_a_crs_resubmit_dropping_a_role_reads_every_party_of_its_side(
    cell: str, role: RoleAxis, population: int
):
    """The CRS publishes no total of either side, so the whole answers from the part's
    dataset under the part's scope choices, binds nothing and opens nothing on the dropped
    side, and counts every party the CRS holds there with the group totals left out: the
    United States and France as donors, Kenya and Eastern Africa's unallocated residual as
    recipients."""
    row = basis._CELLS[cell]
    part = basis._ask(row.variable, time=row.time, recipient=row.recipient)

    whole = _without(part, role, row.variable)

    identity = whole.definition.identity
    assert served_datasets(whole) == served_datasets(part) == {row.basis.dataset}
    assert role not in identity.entities
    assert role not in (identity.open_roles or [])
    assert role not in (whole.resolved.entity_groups or {})
    assert _choices(whole) == _choices(part)
    assert [(fact.role, fact.member_count) for fact in whole.resolved.role_axis_facts or []] == [
        (role, population)
    ]


@pytest.mark.parametrize(
    "donors",
    [
        pytest.param(None, id="no-donor"),
        pytest.param("DAC donor countries", id="named-dac-countries"),
    ],
)
def test_aid_in_2019_from_no_donor_or_dac_countries_reads_dac_countries_total(
    donors: str | None,
):
    """A donor-less DAC1 question binds DAC countries' total on the donor side, which a
    series there holds, and so keeps the measure basis and provider choices a donor's total
    carries, as a question naming the DAC countries does. A no-op resubmit of either answer
    answers the same Spec with the same groups, the named one under its own label."""
    variable = "aid"
    entities = None if donors is None else [ParsedEntity(text=donors, role_text="from")]
    fresh = _answer(
        ResolveRequest(
            input=ParsedInput(
                parts=[ParsedPart(variable_text=variable, time_text="2019", entities=entities)]
            )
        ),
        variable,
    ).spec

    _assert_dac_countries(fresh)
    assert served_datasets(fresh) == {basis._DAC1}
    assert _choices(fresh) == [
        ("measure_basis", "measure_basis:grant_equivalent"),
        ("provider", "provider:contributions"),
    ]
    again = _resubmit(
        fresh.definition.identity, _entries(fresh), variable, fresh.resolved.entity_groups
    )
    assert again.spec_id == fresh.spec_id
    assert again.resolved.entity_groups == fresh.resolved.entity_groups


def test_a_donor_less_candidate_reads_dac_countries_total_as_an_identity_entry_does():
    """A candidates entry posting a donor-less DAC1 identity binds DAC countries' total on
    the donor side, as an identity entry and a fresh search do, so it answers their Spec."""
    variable = "aid"
    fresh = _answer(
        ResolveRequest(
            input=ParsedInput(parts=[ParsedPart(variable_text=variable, time_text="2019")])
        ),
        variable,
    ).spec
    identity = fresh.definition.identity
    open_side = identity.model_copy(
        update={"entities": {k: v for k, v in identity.entities.items() if k != "origin"}}
    )

    part = _answer(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitCandidatesEntry(
                        variable_text=variable,
                        ordering="intent_rank",
                        truncated=False,
                        candidates=[ResubmitCandidate(identity=open_side)] * 2,
                    )
                ]
            )
        ),
        variable,
    )

    assert part.spec.definition.identity.entities["origin"] == [_DAC_COUNTRIES]
    assert part.spec.spec_id == _resubmit(open_side, [], variable).spec_id == fresh.spec_id


def test_a_stated_basis_a_donor_less_answer_serves_is_applied_as_on_resubmit():
    """ "net aid in 2019" over DAC1's gross and net members serves net to DAC countries'
    total with no binder settling an axis, so the stated "net" is the entry the basis
    choice applies and is reported dropped nowhere. A no-op resubmit states the same."""
    variable = "net aid"
    fresh = _answer(
        ResolveRequest(
            input=ParsedInput(
                parts=[ParsedPart(variable_text=variable, time_text="2019")],
                user_message="net aid in 2019",
            )
        ),
        variable,
        basis._DAC1_MEMBERS[:2],
    ).spec
    scope = fresh.resolved.scope_choices
    assert scope is not None

    assert basis._basis_choice(fresh) == ("measure_basis:net", "explicit", "net")
    assert scope.dropped_entries == []
    again = _resubmit(
        fresh.definition.identity, _entries(fresh), variable, fresh.resolved.entity_groups
    )
    assert again.spec_id == fresh.spec_id
    assert basis._basis_choice(again) == basis._basis_choice(fresh)


def test_a_dac1_donor_whole_is_dac_countries_total_on_the_parts_aid_type():
    """The whole of a DAC member's DAC1 total binds DAC countries' total under the part's
    scope choices and is the Spec a donor-less question answers. On bilateral ODA, the
    part's aid type carries: the whole is DAC countries' bilateral ODA."""
    row = basis._CELLS["donor-only-2019"]
    part = basis._ask(row.variable, time=row.time, recipient=False)
    scope = part.resolved.scope_choices
    assert scope is not None
    (provider,) = (c for c in scope.choices if c.axis == "provider")
    bilateral = _resubmit(
        part.definition.identity,
        [
            Entry(
                axis="provider",
                option_id="provider:bilateral",
                provenance="explicit",
                origin=ChoiceOrigin(choice_id=provider.id),
            )
        ],
        row.variable,
    )

    whole = _without(part, "origin", row.variable)
    bilateral_whole = _without(bilateral, "origin", row.variable)

    _assert_dac_countries(whole)
    assert _choices(whole) == _choices(part)
    assert _aid_type(whole) == _aid_type(part) == "DAC1AidType1010"
    _assert_dac_countries(bilateral_whole)
    assert _aid_type(bilateral) == _aid_type(bilateral_whole) == _BILATERAL_ODA
    assert _choices(bilateral_whole) == _choices(bilateral)


def test_a_donor_bound_under_a_role_name_keeps_the_containing_total_off_its_side(
    _coverage: CoverageIndex,
):
    """Entity keys keep the caller's spelling, so a DAC1 identity that binds its donor under
    `creditor` binds that side: no containing total joins it under `origin`. The same
    identity with the donor dropped reads DAC countries' total there."""
    row = basis._CELLS["donor-only-2019"]
    identity = basis._ask(row.variable, time=row.time, recipient=False).definition.identity
    donor = identity.entities["origin"]
    others = {key: refs for key, refs in identity.entities.items() if key != "origin"}
    creditor = identity.model_copy(update={"entities": {**others, "creditor": donor}})
    open_side = identity.model_copy(update={"entities": others})

    assert with_containing_total(creditor, coverage=_coverage) is creditor
    assert with_containing_total(open_side, coverage=_coverage).entities == {
        **others,
        "origin": [_DAC_COUNTRIES],
    }


@pytest.mark.parametrize(
    ("members", "dataset"),
    [
        pytest.param(basis._CRS_MEMBERS, basis._CRS, id="crs"),
        pytest.param(basis._DAC2A_MEMBERS, basis._DAC2A, id="dac2a"),
    ],
)
def test_a_part_naming_no_place_answers_as_it_did_before_its_scope_was_read(
    monkeypatch: pytest.MonkeyPatch, members: tuple[str, ...], dataset: str
):
    """A no-place part reads the scope of its served identity only where the measure-basis
    axis applies to it (`basis.basis_dataset`). A CRS or DAC2a part naming no place binds no
    side and no sector, so the conventions read no shape of it: the part answers the same
    Spec, warnings and departures as with that reading switched off, and states no scope
    choices."""
    flow = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [(dcid, "flows", _USA, _KEN, _YEARS) for dcid in members],
        role_names=dict.fromkeys(members, flow),
        facet_metadata={"flows": FacetMetadata(provenance_id=dataset, unit="USD")},
    )
    request = ResolveRequest(
        input=ParsedInput(parts=[ParsedPart(variable_text="aid", time_text="2019")])
    )

    read = _answer(request, "aid", members)
    with monkeypatch.context() as unread:
        unread.setattr(pipeline, "basis_dataset", lambda *_: None)
        before = _answer(request, "aid", members)

    assert served_datasets(read.spec) == {dataset}
    assert read.spec.definition.identity.entities == {}
    assert read.spec.resolved.scope_choices is None
    assert read.model_dump(mode="json", exclude={"timing_ms"}) == before.model_dump(
        mode="json", exclude={"timing_ms"}
    )
