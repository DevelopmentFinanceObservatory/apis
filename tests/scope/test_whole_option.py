"""A part's whole asked as one scope entry, `<axis>:whole`, posted beside the part's base and
entries. Search compiles the whole with every other axis held and states the pair in
`PartScope.derived`, or refuses the entry once.

The DAC1 parts resolve over `test_basis`'s fixture graph with `test_provider`'s DAC1 coverage.
"""

from __future__ import annotations

import pytest

from search.models import (
    BaseIdentity,
    Option,
    OptionAvailability,
    PartScope,
    ProviderBinding,
    SlotBinding,
)
from search.scope.build import build_part_scope
from search.scope.compile import Refusal
from search.scope.rules import AXIS_RULES
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
    """B12, C8, C9: US bilateral (or multilateral) ODA's whole over the provider axis is
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
    """B2: DAC1's DAC countries' total holds no multilateral, so a part naming the DAC's
    multilaterals is refused a donor whole naming the parties outside it."""
    total, _ = _total_base(monkeypatch)
    base = total.model_copy(update={"entities": {"origin": [provider_fixture._DAC_MULTILATERALS]}})

    refused = build_part_scope(
        base=base, entries=[switch("donor:whole")], records={}, facts=_facts()
    )

    assert isinstance(refused, Refusal)
    assert (refused.code, refused.cause) == ("requested_option_unavailable", "part_outside_whole")


def test_total_odas_provider_whole_refuses_as_the_part_itself(monkeypatch):
    """C12: a whole posted again asks for the whole of a whole, which refuses once."""
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
    """B1, B2, B6: the donor whole on DAC1 binds the containing total, provider option and
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
    """Kill criterion 2: where two options each hold the part's streams and neither holds
    the other (DAC2a's inclusive scope and total ODA), the entry refuses with both as
    alternatives."""
    total, _ = _total_base(monkeypatch)

    def ambiguous(base, record, facts) -> AxisTable:
        def option(scope: str) -> Option:
            return Option(
                option_id=f"provider:{scope}",
                label=scope,
                binding=ProviderBinding(scope=scope),  # type: ignore[arg-type] -- literal scopes
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
