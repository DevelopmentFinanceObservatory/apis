"""A part's whole asked as one scope entry, `<axis>:whole`, posted beside the part's base and
entries. Search compiles the whole with every other axis held and states the pair in
`PartScope.derived`, or refuses the entry once.

The DAC1 parts resolve over `test_basis`'s fixture graph with `test_provider`'s DAC1 coverage,
and the CRS and DAC2a parts are `test_provider`'s identities over its coverage snapshots.
"""

from __future__ import annotations

import dataclasses
from datetime import date
from types import SimpleNamespace

import pytest

from code_index.composites import destination_population
from code_index.shared import get_index
from search.engine.region import whole_totals
from search.models import (
    BaseIdentity,
    IdentityWindowRange,
    Option,
    OptionAvailability,
    PartScope,
    ProviderBinding,
    ProviderScope,
    SlotBinding,
)
from search.scope import region, side
from search.scope.build import build_part_scope
from search.scope.compile import Refusal
from search.scope.rules import AXIS_RULES
from search.scope.side import AGENCIES_COUNTED_TWICE
from search.scope.tables import AxisTable, ScopeFacts
from tests.scope import test_provider as provider_fixture
from tests.scope._axes import switch

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


@pytest.mark.parametrize("code", [_BILATERAL, _MULTILATERAL])
def test_a_dac1_provider_lines_whole_is_total_oda_for_the_same_donor(monkeypatch, code: str):
    """US bilateral (or multilateral) ODA's whole over the provider axis is
    DAC1 total ODA for the US, whatever entry the part holds on the axis. The whole's spec_id
    is the total's, the part's is the line's, and the donor is untouched."""
    total, total_spec_id = _total_base(monkeypatch)
    option = "provider:bilateral" if code == _BILATERAL else "provider:multilateral"
    part = _switched(total, option)

    scope = build_part_scope(
        base=part.compiled.identity,
        entries=[switch(option), switch("provider:whole")],
        records={},
        facts=_facts(),
    )

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

    refused = build_part_scope(
        base=base, entries=[switch("donor:whole")], records={}, facts=_facts()
    )

    assert isinstance(refused, Refusal)
    assert (refused.code, refused.cause) == ("requested_option_unavailable", "part_outside_whole")
    assert "donor:members" in refused.alternatives


def test_total_odas_provider_whole_refuses_as_the_part_itself(monkeypatch):
    """A whole posted again asks for the whole of a whole, which refuses once."""
    total, _ = _total_base(monkeypatch)

    refused = build_part_scope(
        base=total, entries=[switch("provider:whole")], records={}, facts=_facts()
    )

    assert isinstance(refused, Refusal)
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
    """The donor whole on DAC1 binds the containing total, provider option and
    years held, or refuses where the request's coverage holds no series there."""
    total, total_spec_id = _total_base(monkeypatch)
    facts = (
        _facts()
        if covered
        else provider_fixture._dac1_facts(
            extra=[(m, _USA, range(2005, 2025)) for m in provider_fixture._DAC1_TOTAL_MEMBERS]
        )
    )

    scope = build_part_scope(base=total, entries=[switch("donor:whole")], records={}, facts=facts)

    if expected is None:
        assert isinstance(scope, Refusal)
        assert (scope.code, scope.cause) == ("requested_option_unavailable", "option_not_covered")
        return
    assert isinstance(scope, PartScope) and scope.derived is not None
    assert scope.compiled.identity.entities == {"origin": expected}
    assert scope.compiled.identity.requested_window == total.requested_window
    derived = scope.derived
    assert (derived.kind, derived.part.spec_id) == ("published_total", total_spec_id)
    assert derived.part.provider == derived.whole.provider == "provider:contributions"


def test_a_provider_axis_with_two_wholes_refuses_naming_both(monkeypatch):
    """Where a provider table declares no whole and two options each hold the part's streams,
    the entry refuses with both as alternatives."""
    total, _ = _total_base(monkeypatch)

    def ambiguous(base, record, facts) -> AxisTable:
        def option(scope: ProviderScope) -> Option:
            return Option(
                option_id=f"provider:{scope}",
                label=scope,
                binding=ProviderBinding(scope=scope),
                availability=OptionAvailability(status="available"),
            )

        return AxisTable(
            axis="provider",
            options=(option("bilateral"), option("inclusive"), option("contributions")),
            preference=("provider:bilateral",),
            held="provider:bilateral",
            rules={"provider:bilateral": "test"},
            whole_alternatives=("provider:inclusive", "provider:contributions"),
            whole_cause="whole_ambiguous",
        )

    refused = build_part_scope(
        base=total,
        entries=[switch("provider:whole")],
        records={},
        facts=_facts(),
        rules={**AXIS_RULES, "provider": ambiguous},
    )

    assert isinstance(refused, Refusal)
    assert (refused.cause, refused.alternatives) == (
        "whole_ambiguous",
        ("provider:inclusive", "provider:contributions"),
    )


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
    base = total if code == _TOTAL else _switched(total, "provider:bilateral").compiled.identity
    # The snapshot covers none of the part's own variables at the total.
    facts = provider_fixture._dac1_facts(covered)

    refused = build_part_scope(base=base, entries=[switch("donor:whole")], records={}, facts=facts)

    assert isinstance(refused, Refusal)
    assert (refused.code, refused.cause, refused.alternatives) == (
        "requested_option_unavailable",
        "coverage_unknown",
        alternatives,
    )


def test_a_side_two_totals_mark_whole_refuses_naming_both(monkeypatch):
    """Where the code index marks two totals whole on a side, the side declares no whole and
    refuses `whole_ambiguous` with each total and the members' sum as alternatives, in place
    of summing the members unasked."""
    marked = (provider_fixture._ALL_RECIPIENTS, provider_fixture._DEVELOPING)
    monkeypatch.setattr(whole_totals, "population_totals", lambda axes: marked)
    monkeypatch.setattr(side, "whole_reading", lambda dcids, **_: object())
    facts = provider_fixture._dac2a_facts(
        provider_fixture._ALL_YEARS, destinations=(provider_fixture._KEN, *marked)
    )
    base = provider_fixture._dac2a_identity("Gross")

    refused = build_part_scope(
        base=base, entries=[switch("recipient:whole")], records={}, facts=facts
    )

    assert isinstance(refused, Refusal)
    assert (refused.cause, set(refused.alternatives)) == (
        "whole_ambiguous",
        {f"recipient:{dcid}" for dcid in marked} | {"recipient:members"},
    )


def test_a_crs_donor_whole_is_summed_on_bilateral_and_refused_on_inclusive():
    """The CRS names no donor total, so the donor whole sums every reporting donor and
    states them, totals left out. The multilateral agencies report as donors too, so the sum
    refuses beside the inclusive scope, which counts a donor's money through them, whether
    the part holds the scope or posts it."""
    facts = provider_fixture._facts()
    bilateral = provider_fixture._bilateral_identity()
    inclusive = build_part_scope(
        base=bilateral, entries=[switch("provider:inclusive")], records={}, facts=facts
    )
    assert isinstance(inclusive, PartScope)

    summed = build_part_scope(
        base=bilateral, entries=[switch("donor:whole")], records={}, facts=facts
    )
    held = build_part_scope(
        base=inclusive.compiled.identity,
        entries=[switch("donor:whole")],
        records={},
        facts=facts,
    )
    posted = build_part_scope(
        base=inclusive.compiled.identity,
        entries=[switch("provider:inclusive"), switch("donor:whole")],
        records={},
        facts=facts,
    )

    assert isinstance(summed, PartScope) and summed.derived is not None
    assert summed.derived.kind == "sum_of_members"
    assert "origin" not in summed.compiled.identity.entities
    members = summed.derived.members
    assert members is not None and _USA in members
    index = get_index()
    assert all(index.parties[dcid].kind_in(provider_fixture._CRS) != "total" for dcid in members)
    for refused in (held, posted):
        assert isinstance(refused, Refusal)
        assert (refused.code, refused.cause) == (
            "conflicting_requested_options",
            AGENCIES_COUNTED_TWICE,
        )


def test_a_refusal_after_the_whole_entry_points_at_the_entry_posted(monkeypatch):
    """The whole compiles without the part's entry on its axis, so a refusal of a later
    entry is pointed at that entry's place among the entries posted."""
    total, _ = _total_base(monkeypatch)
    part = _switched(total, "provider:bilateral")

    refused = build_part_scope(
        base=part.compiled.identity,
        entries=[
            switch("provider:bilateral"),
            switch("provider:whole"),
            switch("sector:120+130"),
        ],
        records={},
        facts=_facts(),
    )

    assert isinstance(refused, Refusal)
    assert (refused.cause, refused.entry) == ("axis_absent", 2)


def test_no_fresh_scope_offers_a_side_axis(monkeypatch):
    """The side axes are read only where an entry posts them, so a part's choices list none
    of them."""
    total, _ = _total_base(monkeypatch)

    scope = build_part_scope(base=total, entries=[], records={}, facts=_facts())

    assert not {"donor", "recipient"} & set(AXIS_RULES)
    assert isinstance(scope, PartScope)
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

    scope = build_part_scope(
        base=part.compiled.identity,
        entries=[switch("provider:whole")],
        records={},
        facts=_dac2a_facts(),
    )

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
    "destination", [None, provider_fixture._DEVELOPING], ids=["open", "developing-countries"]
)
@pytest.mark.parametrize(
    "option", ["provider:bilateral", "provider:inclusive", "provider:contributions"]
)
def test_a_dac2a_part_over_the_whole_population_takes_total_oda_as_its_whole(
    destination: str | None, option: str
):
    """Over the whole recipient population the provider whole is DAC2a's total ODA to all
    recipients, the published total of both sides, with the inclusive scope offered in its
    place where the part does not hold it. A part naming the developing countries' total and
    one leaving the side open take one whole, whatever option the part holds: total ODA to
    the developing countries counts the bilateral flows alone."""
    part = _dac2a_part(destination, option)

    scope = build_part_scope(
        base=part.compiled.identity,
        entries=[switch("provider:whole")],
        records={},
        facts=_dac2a_facts(),
    )

    assert isinstance(scope, PartScope) and scope.derived is not None
    derived = scope.derived
    assert derived.whole.option_id == "provider:contributions"
    assert derived.alternatives == (
        [] if option == "provider:inclusive" else ["provider:inclusive"]
    )
    assert derived.part.spec_id == part.compiled.spec_id
    whole = scope.compiled.identity
    assert whole.entities["destination"] == [provider_fixture._ALL_RECIPIENTS]
    assert not whole.open_roles
    assert (whole.facets or {}).get("provenance") is None
    open_part = _dac2a_part(None, option)
    opened = build_part_scope(
        base=open_part.compiled.identity,
        entries=[switch("provider:whole")],
        records={},
        facts=_dac2a_facts(),
    )
    assert isinstance(opened, PartScope)
    assert opened.compiled.spec_id == scope.compiled.spec_id


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
    whole = build_part_scope(
        base=africa.compiled.identity,
        entries=[switch("recipient:whole")],
        records={},
        facts=_dac2a_facts(),
    )

    assert named_read.identity.entities["destination"] == [population.total]
    assert open_read.identity.entities["destination"] == list(population.members)
    assert isinstance(whole, PartScope) and whole.derived is not None
    assert whole.derived.kind == "published_total"
    assert whole.compiled.identity.entities["destination"] == [population.total]
    assert whole.compiled.identity.facets == africa.compiled.identity.facets


def test_a_part_naming_its_side_whole_refuses_as_the_whole_itself(monkeypatch):
    """A part naming the DAC countries' total already is its donor whole, which refuses
    once, as a provider whole posted again does."""
    total, _ = _total_base(monkeypatch)
    base = total.model_copy(update={"entities": {"origin": [provider_fixture._DAC_COUNTRIES]}})

    refused = build_part_scope(
        base=base, entries=[switch("donor:whole")], records={}, facts=_facts()
    )

    assert isinstance(refused, Refusal)
    assert (refused.code, refused.cause) == ("requested_option_unavailable", "part_is_whole")


def test_a_donor_outside_the_dac_over_the_part_s_years_has_no_dac_total_whole(monkeypatch):
    """A containing total holds a donor only over the years it is a member: with Korea in
    the DAC from 2010, over 2000-2005 the DAC countries' total holds no Korean ODA and
    Korea's donor whole refuses `part_outside_whole`, and over 2015-2019 it holds it. The
    membership policy dates no DAC membership, so the test dates Korea's."""
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
            rows = [
                dataclasses.replace(row, valid_from=date(2010, 1, 1)) if row.member == _KOR else row
                for row in held.rows
            ]
            return SimpleNamespace(rows=rows)

    monkeypatch.setattr(region, "shared_membership_policy", lambda: Dated())
    total, _ = _total_base(monkeypatch)

    def asked(first: int, last: int) -> PartScope | Refusal:
        base = total.model_copy(
            update={
                "entities": {"origin": [_KOR]},
                "requested_window": IdentityWindowRange(start_year=first, end_year=last),
            }
        )
        return build_part_scope(
            base=base, entries=[switch("donor:whole")], records={}, facts=_facts()
        )

    before = asked(2000, 2005)
    after = asked(2015, 2019)

    assert isinstance(before, Refusal) and before.cause == "part_outside_whole"
    assert not (isinstance(after, Refusal) and after.cause == "part_outside_whole")


_KOR = "country/KOR"


def test_two_whole_entries_refuse_as_conflicting():
    """One post asks for one whole: a second whole entry refuses as conflicting with the
    first."""
    part = _dac2a_part(_AFRICA)

    refused = build_part_scope(
        base=part.compiled.identity,
        entries=[switch("provider:whole"), switch("recipient:whole")],
        records={},
        facts=_dac2a_facts(),
    )

    assert isinstance(refused, Refusal)
    assert (refused.code, refused.cause, refused.entry, refused.blocker) == (
        "conflicting_requested_options",
        "two_wholes",
        1,
        0,
    )


@pytest.mark.parametrize("listed", [False, True], ids=["totalled", "listed"])
def test_a_dac2a_whole_over_the_whole_population_compiles_again_from_its_own_scope(listed: bool):
    """Over the whole population the provider whole reads total ODA to all recipients, and
    its scope states the identity it compiled over, so its base and entries posted again
    compile to the same Spec. A switch to total ODA from a part that totals its recipient
    side reads that total too; from a part listing every recipient it keeps the listing."""
    part = _dac2a_part(None)
    if not listed:
        unlisted = part.compiled.identity.model_copy(update={"open_roles": None})
        part = build_part_scope(base=unlisted, entries=[], records={}, facts=_dac2a_facts())
        assert isinstance(part, PartScope)
    facts = _dac2a_facts()

    whole = build_part_scope(
        base=part.compiled.identity, entries=[switch("provider:whole")], records={}, facts=facts
    )
    switched = build_part_scope(
        base=part.compiled.identity,
        entries=[switch("provider:contributions")],
        records={},
        facts=facts,
    )
    assert isinstance(whole, PartScope) and isinstance(switched, PartScope)
    again = build_part_scope(
        base=whole.base.identity, entries=whole.entries, records={}, facts=facts
    )

    assert isinstance(again, PartScope)
    assert again.compiled.spec_id == whole.compiled.spec_id
    destination = switched.compiled.identity.entities.get("destination")
    if listed:
        assert destination is None
        assert switched.compiled.identity.open_roles == ["destination"]
    else:
        assert destination == [provider_fixture._ALL_RECIPIENTS]
        assert switched.compiled.spec_id == whole.compiled.spec_id


def test_total_oda_to_an_agency_takes_total_oda_to_all_recipients_as_its_whole():
    """A multilateral agency is on the side total ODA alone reads, so France's total ODA to
    IDA takes the population's whole, total ODA to all recipients; the inclusive scope
    serves no agency, so it is offered nowhere in its place."""
    base = provider_fixture._dac2a_identity(
        "Gross", donor=provider_fixture._FRA, destination=provider_fixture._IDA
    )

    scope = build_part_scope(
        base=base, entries=[switch("provider:whole")], records={}, facts=_dac2a_facts()
    )

    assert isinstance(scope, PartScope) and scope.derived is not None
    whole = scope.compiled.identity
    assert whole.entities["destination"] == [provider_fixture._ALL_RECIPIENTS]
    assert (scope.derived.whole.option_id, scope.derived.alternatives) == (
        "provider:contributions",
        [],
    )
