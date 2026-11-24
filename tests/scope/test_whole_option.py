"""A part's whole, asked in the `whole` field beside the part's base and entries
(`scope.whole.derive_whole`): the part compiled, then compiled again with every choice it
settled pinned and one ordinary switch on the whole's axis, the pair stated in
`PartScope.derived`, or the request refused once, with the options that derive offered in
its place.

The DAC1 parts resolve over `test_basis`'s fixture graph with `test_provider`'s DAC1 coverage,
and the CRS and DAC2a parts are `test_provider`'s identities over its coverage snapshots.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from datetime import date
from types import SimpleNamespace

import pytest

from code_index.composites import destination_population
from code_index.shared import get_index
from search.engine.entry.dataset_entries import served_posted_identity
from search.engine.region import whole_totals
from search.models import (
    BaseIdentity,
    Entry,
    IdentityWindowAll,
    IdentityWindowRange,
    OptionAvailability,
    PartScope,
    ProviderBinding,
    ScopeAxis,
    SlotBinding,
    WholeRequest,
)
from search.scope import region, side
from search.scope.build import build_part_scope
from search.scope.rules import AXIS_RULES, AxisRule
from search.scope.side import AGENCIES_COUNTED_TWICE
from search.scope.tables import AxisTable, PartFacts, ScopeFacts
from search.scope.whole import WholeRefusal, derive_whole
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.scope import test_provider as provider_fixture
from tests.scope import test_region as region_fixture
from tests.scope._axes import switch
from tests.scope._axes import tables as axes_tables

_TOTAL = provider_fixture._DAC1_TOTAL_ODA
_BILATERAL = provider_fixture._DAC1_BILATERAL_ODA
_MULTILATERAL = provider_fixture._DAC1_MULTILATERAL_ODA
_USA = provider_fixture._USA


def _facts() -> ScopeFacts:
    return provider_fixture._dac1_facts(_TOTAL, _BILATERAL, _MULTILATERAL)


def _total_base(monkeypatch: pytest.MonkeyPatch) -> tuple[BaseIdentity, str]:
    """The identity and spec_id of US total ODA in 2019 on DAC1."""
    spec = provider_fixture._ask_dac1(monkeypatch, "donor-only-2019")
    return provider_fixture._scope(spec).base.identity, spec.spec_id


def _switched(base: BaseIdentity, option_id: str) -> PartScope:
    scope = build_part_scope(base=base, entries=[switch(option_id)], records={}, facts=_facts())
    assert isinstance(scope, PartScope)
    return scope


def _part(
    base: BaseIdentity,
    entries: Sequence[Entry] = (),
    *,
    facts: ScopeFacts,
    rules: Mapping[ScopeAxis, AxisRule] | None = None,
) -> PartScope:
    scope = build_part_scope(base=base, entries=list(entries), records={}, facts=facts, rules=rules)
    assert isinstance(scope, PartScope)
    return scope


def _whole(
    part: PartScope,
    axis: str,
    option_id: str | None = None,
    *,
    facts: ScopeFacts,
    rules: Mapping[ScopeAxis, AxisRule] | None = None,
) -> PartScope | WholeRefusal:
    request = WholeRequest.model_validate({"axis": axis, "option_id": option_id})
    return derive_whole(part, request, records={}, facts=facts, rules=rules)


@pytest.mark.parametrize("code", [_BILATERAL, _MULTILATERAL])
def test_a_dac1_provider_lines_whole_is_total_oda_for_the_same_donor(monkeypatch, code: str):
    """US bilateral (or multilateral) ODA's whole over the provider axis is DAC1 total ODA
    for the US. The whole's spec_id is the total's, the part's is the line's, and the donor
    is untouched."""
    total, total_spec_id = _total_base(monkeypatch)
    option = "provider:bilateral" if code == _BILATERAL else "provider:multilateral"
    part = _switched(total, option)

    scope = _whole(part, "provider", facts=_facts())

    assert isinstance(scope, PartScope) and scope.derived is not None
    assert scope.compiled.spec_id == total_spec_id
    derived = scope.derived
    assert (derived.axis, derived.kind) == ("provider", "option")
    assert derived.part.spec_id == part.compiled.spec_id
    assert derived.whole.spec_id == scope.compiled.spec_id
    assert (derived.part.option_id, derived.whole.option_id) == (option, "provider:contributions")
    assert derived.part.populations == derived.whole.populations == {"origin": [_USA]}


def test_a_donor_outside_the_containing_total_has_no_donor_whole(monkeypatch):
    """DAC1's DAC countries' total holds no multilateral, so a part naming the DAC's
    multilaterals is refused a donor whole naming the parties outside it, with the side
    summed over its members offered in its place."""
    total, _ = _total_base(monkeypatch)
    base = total.model_copy(update={"entities": {"origin": [provider_fixture._DAC_MULTILATERALS]}})

    refused = _whole(_part(base, facts=_facts()), "donor", facts=_facts())

    assert isinstance(refused, WholeRefusal)
    assert (refused.code, refused.cause) == ("requested_option_unavailable", "part_outside_whole")
    assert refused.alternatives == (("donor:members", "every member"),)


def test_total_odas_provider_whole_refuses_as_the_part_itself(monkeypatch):
    """The whole of a whole refuses once."""
    total, _ = _total_base(monkeypatch)

    refused = _whole(_part(total, facts=_facts()), "provider", facts=_facts())

    assert isinstance(refused, WholeRefusal)
    assert (refused.code, refused.cause) == ("requested_option_unavailable", "part_is_whole")


@pytest.mark.parametrize(
    ("covered", "expected"),
    [
        pytest.param(True, ["DACCountriesTotal"], id="covered"),
        pytest.param(False, None, id="uncovered"),
    ],
)
def test_a_dac1_donors_whole_is_the_dac_countries_total_where_covered(
    monkeypatch, covered: bool, expected: list[str] | None
):
    """The donor whole on DAC1 binds the containing total, provider option and years held,
    or refuses where the request's coverage holds no series there."""
    total, total_spec_id = _total_base(monkeypatch)
    facts = (
        _facts()
        if covered
        else provider_fixture._dac1_facts(
            extra=[(m, _USA, range(2005, 2025)) for m in provider_fixture._DAC1_TOTAL_MEMBERS]
        )
    )

    scope = _whole(_part(total, facts=facts), "donor", facts=facts)

    if expected is None:
        assert isinstance(scope, WholeRefusal)
        assert (scope.code, scope.cause) == ("requested_option_unavailable", "option_not_covered")
        return
    assert isinstance(scope, PartScope) and scope.derived is not None
    assert scope.compiled.identity.entities == {"origin": expected}
    assert scope.compiled.identity.requested_window == total.requested_window
    derived = scope.derived
    assert (derived.kind, derived.part.spec_id) == ("published_total", total_spec_id)
    assert derived.part.provider == derived.whole.provider == "provider:contributions"


def test_a_summed_whole_offers_no_total_the_part_falls_outside(monkeypatch):
    """The DAC's multilaterals fall outside the DAC countries' total, so their donor side
    summed over its members offers the total nowhere in its place."""
    total, _ = _total_base(monkeypatch)
    base = total.model_copy(update={"entities": {"origin": [provider_fixture._DAC_MULTILATERALS]}})

    whole = _whole(_part(base, facts=_facts()), "donor", "donor:members", facts=_facts())

    assert isinstance(whole, PartScope) and whole.derived is not None
    assert (whole.derived.kind, whole.derived.alternatives) == ("sum_of_members", [])


def test_a_whole_lists_only_the_alternatives_that_derive(monkeypatch):
    """Beside total ODA the donor side summed over its members would count the agencies'
    outflows twice, so the DAC countries' total whole offers no alternative, and asking for
    the sum refuses with the total offered in its place."""
    total, _ = _total_base(monkeypatch)
    part = _part(total, facts=_facts())

    whole = _whole(part, "donor", facts=_facts())
    summed = _whole(part, "donor", "donor:members", facts=_facts())

    assert isinstance(whole, PartScope) and whole.derived is not None
    assert whole.derived.alternatives == []
    assert isinstance(summed, WholeRefusal)
    assert (summed.code, summed.cause, [ref for ref, _ in summed.alternatives]) == (
        "conflicting_requested_options",
        AGENCIES_COUNTED_TWICE,
        ["donor:DACCountriesTotal"],
    )


def test_a_provider_axis_with_two_wholes_refuses_naming_both(monkeypatch):
    """Where a provider table declares no whole and two options each hold the part's streams,
    the request refuses with both as alternatives, each with its label."""
    total, _ = _total_base(monkeypatch)
    alternatives = ("provider:multilateral", "provider:contributions")

    def ambiguous(base, record, facts) -> AxisTable | None:
        table = AXIS_RULES["provider"](base, record, facts)
        assert table is not None
        return dataclasses.replace(
            table, whole=None, whole_alternatives=alternatives, whole_cause="whole_ambiguous"
        )

    rules: dict[ScopeAxis, AxisRule] = {**AXIS_RULES, "provider": ambiguous}
    part = _part(total, [switch("provider:bilateral")], facts=_facts(), rules=rules)
    table = ambiguous(part.compiled.identity, None, _facts())
    assert table is not None

    refused = _whole(part, "provider", facts=_facts(), rules=rules)

    assert isinstance(refused, WholeRefusal)
    assert (refused.cause, refused.alternatives) == (
        "whole_ambiguous",
        tuple((ref, option.label) for ref in alternatives if (option := table.option(ref))),
    )
    assert len(refused.alternatives) == 2


def test_an_option_the_axis_offers_as_no_whole_refuses(monkeypatch):
    """A whole request names the whole or an option offered in its place; any other option on
    the axis refuses `not_a_whole`, a plain switch being the way to ask for it."""
    total, _ = _total_base(monkeypatch)

    refused = _whole(_part(total, facts=_facts()), "provider", "provider:bilateral", facts=_facts())

    assert isinstance(refused, WholeRefusal)
    assert (refused.code, refused.cause) == ("requested_option_unavailable", "not_a_whole")


def test_the_dac1_provider_table_declares_total_oda_its_whole(monkeypatch):
    """The whole is read off the table, so the fresh answer that offers the switch states
    it."""
    total, _ = _total_base(monkeypatch)
    table = AXIS_RULES["provider"](total, None, _facts())

    assert table is not None and table.whole == "provider:contributions"
    whole = table.option("provider:contributions")
    assert whole is not None and isinstance(whole.binding, ProviderBinding | SlotBinding)


@pytest.mark.parametrize(
    ("code", "covered", "alternatives"),
    [
        pytest.param(_BILATERAL, _MULTILATERAL, ("donor:members",), id="bilateral"),
        pytest.param(_TOTAL, _BILATERAL, (), id="total-oda"),
    ],
)
def test_a_donor_whole_the_snapshot_cannot_read_refuses_with_its_own_cause(
    monkeypatch, code: str, covered: str, alternatives: tuple[str, ...]
):
    """A snapshot holding no series of the whole's variables cannot say whether the DAC
    countries' total serves the part, so the donor whole refuses `coverage_unknown`, where
    reading it as covered would state an unread total. It offers the side summed over its
    members in its place only where that sum holds with the provider option the part holds:
    beside total ODA it would count the agencies' outflows twice."""
    total, _ = _total_base(monkeypatch)
    # The snapshot covers none of the part's own variables at the total.
    facts = provider_fixture._dac1_facts(covered)
    base = total if code == _TOTAL else _switched(total, "provider:bilateral").compiled.identity
    part = _part(base, facts=facts)

    refused = _whole(part, "donor", facts=facts)

    assert isinstance(refused, WholeRefusal)
    assert (refused.code, refused.cause, tuple(ref for ref, _ in refused.alternatives)) == (
        "requested_option_unavailable",
        "coverage_unknown",
        alternatives,
    )


def test_a_side_two_totals_mark_whole_refuses_naming_both(monkeypatch):
    """Where the code index marks two totals whole on a side, the side declares no whole and
    refuses `whole_ambiguous`, offering each total and the members' sum that derive, in
    place of summing the members unasked."""
    marked = (provider_fixture._ALL_RECIPIENTS, provider_fixture._DEVELOPING)
    monkeypatch.setattr(whole_totals, "population_totals", lambda axes: marked)
    monkeypatch.setattr(side, "whole_reading", lambda dcids, **_: object())
    facts = provider_fixture._dac2a_facts(
        provider_fixture._ALL_YEARS, destinations=(provider_fixture._KEN, *marked)
    )
    part = _part(provider_fixture._dac2a_identity("Gross"), facts=facts)

    refused = _whole(part, "recipient", facts=facts)

    assert isinstance(refused, WholeRefusal)
    assert (refused.cause, {ref for ref, _ in refused.alternatives}) == (
        "whole_ambiguous",
        {f"recipient:{dcid}" for dcid in marked} | {"recipient:members"},
    )


def test_a_crs_donor_whole_is_summed_on_bilateral_and_refused_on_inclusive():
    """The CRS names no donor total, so the donor whole sums every reporting donor and
    states them, totals left out. The multilateral agencies report as donors too, so the sum
    refuses beside the inclusive scope, which counts a donor's money through them."""
    facts = provider_fixture._facts()
    bilateral = provider_fixture._bilateral_identity()

    summed = _whole(_part(bilateral, facts=facts), "donor", facts=facts)
    refused = _whole(
        _part(bilateral, [switch("provider:inclusive")], facts=facts), "donor", facts=facts
    )

    assert isinstance(summed, PartScope) and summed.derived is not None
    assert summed.derived.kind == "sum_of_members"
    assert "origin" not in summed.compiled.identity.entities
    members = summed.derived.members
    assert members is not None and _USA in members
    index = get_index()
    assert all(index.parties[dcid].kind_in(provider_fixture._CRS) != "total" for dcid in members)
    assert isinstance(refused, WholeRefusal)
    assert (refused.code, refused.cause) == (
        "conflicting_requested_options",
        AGENCIES_COUNTED_TWICE,
    )


def test_a_summed_donor_side_is_served_open_where_a_containing_total_exists(monkeypatch):
    """DAC1 publishes a containing total of the donor side, which a posted identity leaving
    the side unnamed binds. A whole summing the side states it open, so the identity serves
    as compiled and reads the members `derived.members` names."""
    total, _ = _total_base(monkeypatch)
    facts = _facts()
    part = _part(total, [switch("provider:bilateral")], facts=facts)

    whole = _whole(part, "donor", "donor:members", facts=facts)

    assert isinstance(whole, PartScope) and whole.derived is not None
    identity = whole.compiled.identity
    assert "origin" not in identity.entities and identity.open_roles == ["origin"]
    served, _ = served_posted_identity(identity, whole.entries, coverage=facts.coverage)
    assert served == identity
    assert whole.derived.members


def test_a_summed_recipient_side_reads_the_members_it_states():
    """A recipient whole summed over its members reads the bilateral destinations of the
    population on DAC2a: the composite's component lists the members `derived.members`
    names, the rows it serves."""
    part = _dac2a_part(_AFRICA)

    whole = _whole(part, "recipient", "recipient:members", facts=_dac2a_facts())

    assert isinstance(whole, PartScope) and whole.derived is not None
    assert whole.derived.kind == "sum_of_members"
    (component,) = whole.compiled.components
    assert component.identity.entities["destination"] == whole.derived.members


def test_a_donor_whole_keeps_the_part_s_region_and_a_recipient_whole_drops_it():
    """The whole holds every choice the part settled, less one setting the role the whole
    sets: a donor whole keeps the recipient region's reading, and a recipient whole, which
    replaces the recipients, leaves it to its default."""
    facts = dataclasses.replace(
        provider_fixture._facts(),
        part=PartFacts(
            entity_groups={"destination": region_fixture._AFRICA},
            datasets=(region_fixture._CRS,),
        ),
    )
    destination = sorted([*region_fixture._AFRICA_MEMBERS, *region_fixture._AFRICA_RESIDUALS])
    base = provider_fixture._bilateral_identity().model_copy(
        update={"entities": {"origin": [_USA], "destination": destination}}
    )
    part = _part(base, [switch(region_fixture._MEMBERS)], facts=facts)

    donor = _whole(part, "donor", facts=facts)
    recipient = _whole(part, "recipient", facts=facts)

    def regions(scope: PartScope | WholeRefusal) -> list[tuple[str, str]]:
        assert isinstance(scope, PartScope)
        return [(c.active, c.provenance) for c in scope.choices if c.axis == "region"]

    assert regions(donor) == [(region_fixture._MEMBERS, "explicit")]
    assert regions(recipient) != regions(donor)


def test_the_whole_s_entries_are_the_part_s_own_plus_its_switch(monkeypatch):
    """The pins serve the whole's compile alone: the whole's entries are the part's own and
    the switch on its axis, so a choice the part took by default stays a default, which a
    later switch moves with the provider."""
    total, _ = _total_base(monkeypatch)
    part = _part(total, [switch("provider:bilateral")], facts=_facts())

    whole = _whole(part, "provider", facts=_facts())

    assert isinstance(whole, PartScope)
    assert [(e.axis, e.option_id) for e in whole.entries] == [
        ("provider", "provider:contributions")
    ]
    assert [(c.axis, c.provenance) for c in whole.choices] == [
        ("measure_basis", "default"),
        ("provider", "explicit"),
    ]


def _reposted(whole: PartScope, facts: ScopeFacts) -> PartScope:
    """`whole`'s base and entries posted again with no `whole`, as a same-shape merge
    compiles them."""
    scope = build_part_scope(
        base=whole.base.identity, entries=whole.entries, records={}, facts=facts, refuse=False
    )
    assert isinstance(scope, PartScope)
    return scope


def test_a_rebinding_whole_reposts_as_the_figure_it_serves():
    """France's provider whole at the developing countries rebinds the recipient side. Its
    scope states the identity it serves as its base and holds no side entry, so its base
    and entries posted again without `whole` compile the same Spec with nothing dropped."""
    facts = _dac2a_facts()
    part = _dac2a_part(provider_fixture._DEVELOPING)

    whole = _whole(part, "provider", facts=facts)

    assert isinstance(whole, PartScope) and whole.derived is not None
    assert whole.derived.rebind == f"recipient:{provider_fixture._ALL_RECIPIENTS}"
    assert whole.base.identity == whole.compiled.identity
    assert not {e.axis for e in whole.entries} & {"donor", "recipient"}
    again = _reposted(whole, facts)
    assert (again.compiled.spec_id, again.dropped_entries) == (whole.compiled.spec_id, [])


@pytest.mark.parametrize("option", [None, "donor:members"], ids=["total", "members"])
def test_a_donor_whole_reposts_as_the_figure_it_serves(monkeypatch, option: str | None):
    """US bilateral ODA's donor whole, the DAC countries' total or every donor summed, states
    the identity it serves as its base and no donor entry, so its base and entries posted
    again compile the same Spec with nothing dropped."""
    total, _ = _total_base(monkeypatch)
    facts = _facts()
    part = _part(total, [switch("provider:bilateral")], facts=facts)

    whole = _whole(part, "donor", option, facts=facts)

    assert isinstance(whole, PartScope)
    assert not {e.axis for e in whole.entries} & {"donor", "recipient"}
    again = _reposted(whole, facts)
    assert (again.compiled.spec_id, again.dropped_entries) == (whole.compiled.spec_id, [])


def test_the_whole_carries_the_entries_its_part_dropped(monkeypatch):
    """A part built without refusing reports the entries it could not apply; its whole
    reports the same."""
    total, _ = _total_base(monkeypatch)
    part = build_part_scope(
        base=total,
        entries=[switch("provider:bilateral"), switch("sector:120")],
        records={},
        facts=_facts(),
        refuse=False,
    )
    assert isinstance(part, PartScope) and part.dropped_entries

    whole = _whole(part, "provider", facts=_facts())

    assert isinstance(whole, PartScope)
    assert whole.dropped_entries == part.dropped_entries


def test_a_whole_moving_an_axis_the_part_holds_refuses(monkeypatch):
    """A part holds an option no longer served, kept because no option serves, which the
    whole cannot pin. Where the whole's compile then settles another option on that axis,
    the whole refuses `whole_moves_<axis>` in place of answering a figure the part does not
    count beside."""
    (sector, _region, _basis) = axes_tables()
    served = {"now": False}

    def sector_rule(base, record, facts) -> AxisTable:
        unavailable = OptionAvailability(status="unavailable", code="option_not_covered")
        set_option, one_option = sector.options
        if not served["now"]:
            options = tuple(
                o.model_copy(update={"availability": unavailable}) for o in sector.options
            )
            return dataclasses.replace(sector, options=options)
        return dataclasses.replace(
            sector,
            options=(set_option.model_copy(update={"availability": unavailable}), one_option),
            preference=(one_option.option_id, set_option.option_id),
        )

    rules: dict[ScopeAxis, AxisRule] = {**AXIS_RULES, "sector": sector_rule}
    facts = provider_fixture._facts()
    part = _part(provider_fixture._bilateral_identity(), facts=facts, rules=rules)
    assert [c.active for c in part.choices if c.axis == "sector"] == [sector.held]
    served["now"] = True

    refused = _whole(part, "provider", facts=facts, rules=rules)

    assert isinstance(refused, WholeRefusal)
    assert (refused.code, refused.cause) == (
        "conflicting_requested_options",
        "whole_moves_sector",
    )


def test_a_merge_build_reads_no_whole_from_the_scope_entries(monkeypatch):
    """The whole is asked in its own field, so a build that drops what it cannot apply in
    place of refusing reads `provider:whole` as an option no table offers: dropped, and the
    part answered as itself, bilateral ODA, with no pair."""
    total, _ = _total_base(monkeypatch)
    part = _switched(total, "provider:bilateral")

    scope = build_part_scope(
        base=part.compiled.identity,
        entries=[switch("provider:bilateral"), switch("provider:whole")],
        records={},
        facts=_facts(),
        refuse=False,
    )

    assert isinstance(scope, PartScope)
    assert [d.entry.option_id for d in scope.dropped_entries] == ["provider:whole"]
    assert scope.derived is None
    assert scope.compiled.spec_id == part.compiled.spec_id


def test_no_fresh_scope_offers_a_side_axis(monkeypatch):
    """The side axes are read only for a whole request, so a part's choices list none of
    them."""
    total, _ = _total_base(monkeypatch)

    scope = _part(total, facts=_facts())

    assert not {"donor", "recipient"} & set(AXIS_RULES)
    assert not {"donor", "recipient"} & {choice.axis for choice in scope.choices}


def _dac2a_part(destination: str | None, option: str = "provider:bilateral") -> PartScope:
    """France's DAC2a ODA to `destination`, an open destination where it is None, switched
    to `option`."""
    facts = _dac2a_facts()
    base = provider_fixture._dac2a_identity(
        "Gross", donor=provider_fixture._FRA, destination=destination
    )
    scope = build_part_scope(base=base, entries=[switch(option)], records={}, facts=facts)
    assert isinstance(scope, PartScope)
    return scope


def _dac2a_facts() -> ScopeFacts:
    return provider_fixture._dac2a_facts(
        provider_fixture._ALL_YEARS,
        donor=provider_fixture._FRA,
        destinations=(
            provider_fixture._KEN,
            _AFRICA,
            provider_fixture._DEVELOPING,
            provider_fixture._ALL_RECIPIENTS,
        ),
    )


_AFRICA = "DACAfrica"


@pytest.mark.parametrize(
    "option", ["provider:bilateral", "provider:contributions"], ids=["bilateral", "total-oda"]
)
@pytest.mark.parametrize("destination", [provider_fixture._KEN, _AFRICA], ids=["country", "region"])
def test_a_dac2a_part_naming_a_country_or_region_takes_the_inclusive_scope_as_its_whole(
    destination: str, option: str
):
    """Below the population's total DAC2a's total ODA counts the bilateral flows alone, so
    the provider whole of a part naming a country or region is the inclusive scope, with no
    alternative, and the recipient held."""
    part = _dac2a_part(destination, option)

    scope = _whole(part, "provider", facts=_dac2a_facts())

    assert isinstance(scope, PartScope) and scope.derived is not None
    derived = scope.derived
    assert (derived.whole.option_id, derived.alternatives) == ("provider:inclusive", [])
    assert derived.part.spec_id == part.compiled.spec_id
    assert scope.compiled.identity.entities["destination"] == [destination]


def test_dac2a_total_oda_and_bilateral_read_one_row_for_a_named_country():
    """Over a named country DAC2a's total ODA and its bilateral composite read the same
    series: the composite's one component is the total ODA identity itself."""
    bilateral = _dac2a_part(provider_fixture._KEN)
    total = _dac2a_part(provider_fixture._KEN, "provider:contributions")

    (component,) = bilateral.compiled.components
    assert (total.compiled.identity.facets or {}).get("provenance") is None
    assert component.identity == total.compiled.identity


@pytest.mark.parametrize(
    ("destination", "option"),
    [
        pytest.param(None, "provider:bilateral", id="totalled"),
        pytest.param("listed", "provider:bilateral", id="listed"),
        pytest.param(provider_fixture._DEVELOPING, "provider:bilateral", id="developing"),
        pytest.param(provider_fixture._DEVELOPING, "provider:contributions", id="developing-total"),
        pytest.param(provider_fixture._IDA, "provider:contributions", id="agency-total"),
    ],
)
def test_a_dac2a_part_over_the_whole_population_takes_total_oda_to_all_recipients(
    destination: str | None, option: str
):
    """DAC2a declares that the provider whole of a part over the whole recipient population
    rebinds the recipient side (`ProviderWhole.recipient`): a part totalling or listing
    every recipient, naming the developing countries' total, or naming an agency, on the
    bilateral composite or on total ODA itself, takes total ODA to all recipients, one
    figure, so a listing's every recipient divides by it. `derived` states the rebind, and
    the inclusive scope is offered in its place where it serves the part."""
    facts = _dac2a_facts()
    part = _dac2a_part(None if destination == "listed" else destination, option)
    if destination is None:
        unlisted = part.compiled.identity.model_copy(update={"open_roles": None})
        part = _part(unlisted, facts=facts)

    whole = _whole(part, "provider", facts=facts)

    assert isinstance(whole, PartScope) and whole.derived is not None
    identity = whole.compiled.identity
    assert identity.entities["destination"] == [provider_fixture._ALL_RECIPIENTS]
    assert not identity.open_roles
    assert (identity.facets or {}).get("provenance") is None
    assert (whole.derived.whole.option_id, whole.derived.rebind) == (
        "provider:contributions",
        f"recipient:{provider_fixture._ALL_RECIPIENTS}",
    )
    agency = destination == provider_fixture._IDA
    assert whole.derived.alternatives == ([] if agency else ["provider:inclusive"])
    assert whole.derived.part.spec_id == part.compiled.spec_id


def test_a_listing_s_plain_switch_to_total_oda_keeps_its_listing():
    """The rebind belongs to the whole: a plain switch to total ODA from a part listing every
    recipient keeps the recipients the part names, with no pair."""
    facts = _dac2a_facts()
    part = _dac2a_part(None)

    switched = _part(part.compiled.identity, [switch("provider:contributions")], facts=facts)

    assert switched.derived is None
    assert "destination" not in switched.compiled.identity.entities
    assert switched.compiled.identity.open_roles == ["destination"]


def test_a_part_holding_total_oda_to_all_recipients_is_its_own_whole():
    """A part already reading total ODA to all recipients compiles to its own whole, which
    refuses `part_is_whole`, offering the inclusive scope in its place."""
    part = _dac2a_part(provider_fixture._ALL_RECIPIENTS, "provider:contributions")

    refused = _whole(part, "provider", facts=_dac2a_facts())

    assert isinstance(refused, WholeRefusal)
    assert (refused.cause, [ref for ref, _ in refused.alternatives]) == (
        "part_is_whole",
        ["provider:inclusive"],
    )


def test_the_developing_countries_part_reads_the_total_of_the_population_the_open_part_sums():
    """France's ODA to developing countries names the published total of the destinations
    the composite reads when no destination is named, and a narrower part's recipient whole
    compiles to that named total."""
    population = destination_population(get_index(), provider_fixture._DAC2A)
    named = _dac2a_part(provider_fixture._DEVELOPING)
    unnamed = _dac2a_part(None)
    africa = _dac2a_part(_AFRICA)

    (named_read,) = named.compiled.components
    (open_read,) = unnamed.compiled.components
    whole = _whole(africa, "recipient", facts=_dac2a_facts())

    assert named_read.identity.entities["destination"] == [population.total]
    assert open_read.identity.entities["destination"] == list(population.members)
    assert isinstance(whole, PartScope) and whole.derived is not None
    assert whole.derived.kind == "published_total"
    assert whole.compiled.identity.entities["destination"] == [population.total]
    assert whole.compiled.identity.facets == africa.compiled.identity.facets


def test_a_part_naming_its_side_whole_refuses_as_the_whole_itself(monkeypatch):
    """A part naming the DAC countries' total already is its donor whole, which refuses
    once, as a provider whole of total ODA does."""
    total, _ = _total_base(monkeypatch)
    base = total.model_copy(update={"entities": {"origin": [provider_fixture._DAC_COUNTRIES]}})

    refused = _whole(_part(base, facts=_facts()), "donor", facts=_facts())

    assert isinstance(refused, WholeRefusal)
    assert (refused.code, refused.cause) == ("requested_option_unavailable", "part_is_whole")


def test_a_side_left_open_is_its_own_whole():
    """A part listing every recipient already is its recipient whole, which refuses
    `part_is_whole`."""
    part = _dac2a_part(None)
    assert part.compiled.identity.open_roles == ["destination"]

    refused = _whole(part, "recipient", facts=_dac2a_facts())

    assert isinstance(refused, WholeRefusal)
    assert (refused.code, refused.cause) == ("requested_option_unavailable", "part_is_whole")


def test_a_donor_outside_the_dac_over_the_part_s_years_has_no_dac_total_whole(monkeypatch):
    """A containing total holds a donor only over the years it is a member. With Korea in
    the DAC from 2010, in two rows that meet in 2015, over 2000-2005 Korea's donor whole
    refuses `part_outside_whole`; over 2012-2019 the merged rows hold it; and over every
    year the window reads the years Korea's own series hold (2016-2019), so the total holds
    it. The membership policy dates no DAC membership, so the test dates Korea's."""
    policy = region.shared_membership_policy()
    group = get_index().parties[provider_fixture._DAC_COUNTRIES].totals
    assert group is not None

    class Dated:
        def __contains__(self, ref: str) -> bool:
            return ref in policy

        def get(self, ref: str):
            held = policy.get(ref)
            if ref != group:
                return held
            rows = [row for row in held.rows if row.member != _KOR]
            (korea,) = [row for row in held.rows if row.member == _KOR]
            joined = dataclasses.replace(
                korea, valid_from=date(2010, 1, 1), valid_until=date(2015, 1, 1)
            )
            stayed = dataclasses.replace(korea, valid_from=date(2015, 1, 1))
            return SimpleNamespace(rows=[*rows, joined, stayed])

    monkeypatch.setattr(region, "shared_membership_policy", lambda: Dated())
    total, _ = _total_base(monkeypatch)
    facts = provider_fixture._dac1_facts(
        _TOTAL, extra=[(m, _KOR, range(2016, 2020)) for m in provider_fixture._DAC1_TOTAL_MEMBERS]
    )

    def asked(window: IdentityWindowRange | IdentityWindowAll) -> PartScope | WholeRefusal:
        base = total.model_copy(update={"entities": {"origin": [_KOR]}, "requested_window": window})
        return _whole(_part(base, facts=facts), "donor", facts=facts)

    def outside(scope: PartScope | WholeRefusal) -> bool:
        return isinstance(scope, WholeRefusal) and scope.cause == "part_outside_whole"

    assert outside(asked(IdentityWindowRange(start_year=2000, end_year=2005)))
    assert not outside(asked(IdentityWindowRange(start_year=2012, end_year=2019)))
    assert not outside(asked(IdentityWindowAll()))


_KOR = "country/KOR"


def test_a_whole_serving_fewer_years_than_the_part_refuses():
    """The CRS inclusive scope serving fewer years than the part asks for would compile the
    whole over its own shorter window, so the provider whole refuses with the option's own
    reason."""
    rows, extra = provider_fixture._coverage_rows(range(2018, 2025))
    facts = ScopeFacts(
        index=get_index(),
        coverage=FakeCoverageService(build_coverage_index(rows, **extra)).snapshot(),
    )
    base = provider_fixture._bilateral_identity()
    table = AXIS_RULES["provider"](base, None, facts)
    assert table is not None
    inclusive = table.option("provider:inclusive")
    assert inclusive is not None and inclusive.availability.status == "partial"

    refused = _whole(_part(base, facts=facts), "provider", facts=facts)

    assert isinstance(refused, WholeRefusal)
    assert (refused.code, refused.cause) == (
        "requested_option_unavailable",
        inclusive.availability.code,
    )


def test_an_option_offered_in_place_of_the_whole_is_asked_by_its_id():
    """The inclusive scope a DAC2a whole offers in its place is asked for as the whole's
    `option_id`, which answers the pair with total ODA as the other whole, while a plain
    switch to `provider:inclusive` answers the same figure with no pair."""
    part = _dac2a_part(None)

    whole = _whole(part, "provider", "provider:inclusive", facts=_dac2a_facts())
    switched = _part(part.compiled.identity, [switch("provider:inclusive")], facts=_dac2a_facts())

    assert isinstance(whole, PartScope) and whole.derived is not None
    assert (whole.derived.whole.option_id, whole.derived.alternatives) == (
        "provider:inclusive",
        ["provider:contributions"],
    )
    assert whole.derived.part.spec_id == part.compiled.spec_id
    assert switched.derived is None
    assert switched.compiled.spec_id == whole.compiled.spec_id


def test_a_side_named_by_a_total_whose_dcid_makes_no_option_id_is_its_own_whole(monkeypatch):
    """A total whose dcid cannot form an option id (a `/` in it) still names the side the
    part holds as its whole: the side states it as `<axis>:named`, and a whole request
    refuses `part_is_whole`."""
    facts = _dac2a_facts()
    part = _dac2a_part(provider_fixture._KEN, "provider:contributions")
    monkeypatch.setattr(side, "_marked", lambda *_: (provider_fixture._KEN,))

    refused = _whole(part, "recipient", facts=facts)

    assert isinstance(refused, WholeRefusal)
    assert refused.cause == "part_is_whole"
