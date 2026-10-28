"""A part's whole: its own identity with the sector slot unbound, posted back as a
dataset_resubmit identity entry with every other choice held by an explicit entry from its
choice, so the whole keeps the part's donor, recipient, basis, prices, window and provider
scope or refuses.

The bilateral parts resolve over `test_basis`'s fixture graph, where OECD DAC1 donor totals
sit beside the CRS health members, so a whole that left the CRS would be elected there. The
inclusive parts resolve over `test_provider`'s, which adds the imputed multilateral table's
health members.
"""

from __future__ import annotations

import pytest

from search.engine.departures import served_datasets
from search.engine.errors import EngineInputError
from search.models import (
    BaseIdentity,
    ChoiceOrigin,
    DatasetResubmitInput,
    DefinitePart,
    Entry,
    IdentityBindingSet,
    PartScope,
    Resolution,
    ResolveRequest,
    ResubmitIdentityEntry,
    SlotBinding,
    Spec,
)
from tests.engine._fake_coverage import patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.scope import test_basis as basis_fixture
from tests.scope import test_provider as provider_fixture

_FRA = "country/FRA"
_HEALTH = ["CRSPurposeCode120", "CRSPurposeCode130"]


def _whole_entry(scope: PartScope) -> ResubmitIdentityEntry:
    """The resubmit asking for the whole of the part `scope` states: `scope.base.identity`
    with the slot its sector options bind set unbound, its entries less the sector's, and an
    explicit entry from each other choice holding the option it holds."""
    (sector,) = (c for c in scope.choices if c.axis == "sector")
    bindings = {o.binding.slot for o in sector.options if isinstance(o.binding, SlotBinding)}
    (slot,) = bindings
    document = scope.base.identity.model_dump(mode="json", exclude_none=True)
    for item in document["slots"]:
        if item["property"] == slot:
            item["binding"] = {"state": "unbound"}
    held = [
        Entry(
            axis=choice.axis,
            option_id=choice.active,
            provenance="explicit",
            concept=next((e.concept for e in scope.entries if e.axis == choice.axis), None),
            origin=ChoiceOrigin(choice_id=choice.id),
        )
        for choice in scope.choices
        if choice.axis != "sector"
    ]
    return ResubmitIdentityEntry(
        variable_text="aid in every sector",
        identity=BaseIdentity.model_validate(document),
        scope_entries=held,
    )


def _resolve_whole(entry: ResubmitIdentityEntry, *, graph, llm) -> Spec:
    response = offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[entry])),
        graph=graph,
        llm=llm,
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    return part.spec


def _scope(spec: Spec) -> PartScope:
    scope = spec.resolved.scope_choices
    assert scope is not None
    return scope


def _actives(spec: Spec) -> dict[str, str]:
    return {choice.axis: choice.active for choice in _scope(spec).choices}


def _without_sector(identity: BaseIdentity) -> dict:
    """`identity` as a document with its purpose slot unbound."""
    document = identity.model_dump(mode="json", exclude_none=True)
    for item in document["slots"]:
        if item["property"] == "crsPurpose":
            item["binding"] = {"state": "unbound"}
    return document


def _bilateral_whole(monkeypatch: pytest.MonkeyPatch) -> tuple[Spec, Spec]:
    basis_fixture._cover(monkeypatch, basis_fixture._DAC1_MEMBERS)
    part = basis_fixture._ask("health aid", time="2023", recipient=False)
    whole = _resolve_whole(
        _whole_entry(_scope(part)),
        graph=basis_fixture._graph("aid in every sector", basis_fixture._ALL_MEMBERS),
        llm=basis_fixture._PickingLLM(),
    )
    return part, whole


def test_a_bilateral_crs_sector_parts_whole_is_the_crs_bilateral_total(monkeypatch):
    """US bilateral health aid in 2023, CRS 120 + 130 gross. Its whole is the CRS's gross
    bilateral total over every purpose code, with the part's donor, window, basis and
    provider scope, under a spec_id of its own."""
    part, whole = _bilateral_whole(monkeypatch)

    purpose = next(s for s in part.definition.identity.slots if s.property == "crsPurpose")
    assert isinstance(purpose.binding, IdentityBindingSet) and purpose.binding.refs == _HEALTH
    assert _actives(part) == {
        "sector": "sector:120+130",
        "measure_basis": "measure_basis:gross",
        "provider": "provider:bilateral",
    }
    assert whole.definition.identity.model_dump(mode="json", exclude_none=True) == (
        _without_sector(part.definition.identity)
    )
    assert served_datasets(whole) == served_datasets(part) == {basis_fixture._CRS}
    assert _actives(whole) == {
        "measure_basis": "measure_basis:gross",
        "provider": "provider:bilateral",
    }
    scope = _scope(whole)
    assert whole.spec_id == scope.compiled.spec_id != part.spec_id


def test_a_whole_posted_again_answers_its_own_spec_id(monkeypatch):
    """The whole's own scope, posted back with its entries, compiles to the whole. A
    later switch of it makes the same resubmit."""
    _, whole = _bilateral_whole(monkeypatch)
    scope = _scope(whole)

    again = _resolve_whole(
        ResubmitIdentityEntry(
            variable_text="aid in every sector",
            identity=scope.base.identity,
            scope_entries=scope.entries,
        ),
        graph=basis_fixture._graph("aid in every sector", basis_fixture._ALL_MEMBERS),
        llm=basis_fixture._PickingLLM(),
    )

    assert again.spec_id == whole.spec_id


def _inclusive_part(monkeypatch: pytest.MonkeyPatch) -> Spec:
    provider_fixture._cover(monkeypatch, provider_fixture._ALL_YEARS)
    return provider_fixture._ask(
        "total health aid from the United States to Kenya, 2015 to 2020", time="2015-2020"
    )


def _resolve_inclusive_whole(entry: ResubmitIdentityEntry) -> Spec:
    return _resolve_whole(
        entry,
        graph=provider_fixture._graph("aid in every sector", provider_fixture._CRS_MEMBERS),
        llm=basis_fixture._PickingLLM(),
    )


def test_an_inclusive_parts_whole_is_the_inclusive_total_over_every_sector(monkeypatch):
    """A named donor's total of health aid counts bilateral plus imputed multilateral flows.
    Its whole is the same composite over every purpose code, served from both tables over the
    part's years."""
    part = _inclusive_part(monkeypatch)
    assert _actives(part)["provider"] == "provider:inclusive"

    whole = _resolve_inclusive_whole(_whole_entry(_scope(part)))

    assert whole.definition.identity.model_dump(mode="json", exclude_none=True) == (
        _without_sector(part.definition.identity)
    )
    assert provider_fixture._source(whole) == provider_fixture._INCLUSIVE_SOURCE
    assert _actives(whole) == {
        "measure_basis": "measure_basis:gross",
        "provider": "provider:inclusive",
    }
    assert [(c.role, c.provenance, c.periods) for c in _scope(whole).compiled.components] == [
        ("bilateral", provider_fixture._CRS, [(2015, 2020)]),
        ("imputed", provider_fixture._IMPUTED, [(2015, 2020)]),
    ]


def test_an_inclusive_whole_the_imputed_table_cannot_serve_refuses(monkeypatch):
    """Where the imputed table carries no row for the donor, the inclusive scope is not
    offered on the whole, and the explicit entry holding it refuses, so the whole cannot fall
    back to bilateral flows."""
    part = _inclusive_part(monkeypatch)
    rows, extra = provider_fixture._coverage_rows(provider_fixture._ALL_YEARS)
    imputed = set(provider_fixture._IMPUTED_MEMBERS)
    patch_coverage(
        monkeypatch,
        [
            (dcid, facet, _FRA if dcid in imputed else donor, recipient, years)
            for dcid, facet, donor, recipient, years in rows
        ],
        **extra,
    )

    with pytest.raises(EngineInputError) as refused:
        _resolve_inclusive_whole(_whole_entry(_scope(part)))

    assert refused.value.code == "requested_option_unavailable"
    assert refused.value.status == 422
    # The held provider entry is the last of the posted entries.
    assert refused.value.pointer == "/input/entries/0/scope_entries/1"


def test_a_part_binding_no_sector_states_no_sector_choice(monkeypatch):
    """A donor's total of all aid binds no purpose code, so its scope offers no sector
    choice, which is how a reader knows it has no whole."""
    basis_fixture._cover(monkeypatch, basis_fixture._DAC1_MEMBERS)

    part = basis_fixture._ask("aid", time="2019", recipient=False)

    assert served_datasets(part) == {basis_fixture._DAC1}
    assert "sector" not in _actives(part)
