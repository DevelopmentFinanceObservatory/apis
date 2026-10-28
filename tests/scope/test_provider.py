"""The provider axis: a named donor's total of a CRS sector counts its bilateral flows plus
its imputed share of multilateral spending when both tables cover the years asked for; a
window the imputed table covers in part offers the inclusive scope over the years it covers,
which only a switch applies; a typed "bilateral" applies; a latest window elects the latest
year both tables cover; and a resubmit of a composite with its own entries keeps its spec_id.

The parts resolve through the parsed door over the fixture graph of `test_basis`: CRS health
members under CRS 120 and 130 from the United States to Kenya. The imputed multilateral table
holds two health members of its own, which only the code index and the coverage snapshot
read, as the deployed service reads them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import pytest

from code_index.composites import component_identities
from code_index.scope_conventions import (
    OPTION_LABELS,
    PROVIDER_CONVENTIONS,
    PROVIDER_SCOPE_RULES,
    PUBLISHED_PROVIDER_SCOPES,
)
from code_index.shared import get_index
from search.engine.spec_id import identity_spec_id
from search.models import (
    BaseIdentity,
    ChoiceOrigin,
    DatasetResubmitInput,
    DefinitePart,
    Entry,
    IdentityBindingValue,
    IdentityWindowRange,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    PartScope,
    ProviderBinding,
    ProviderScope,
    Resolution,
    ResolveRequest,
    ResubmitIdentityEntry,
    ScopeChoice,
    SlotBinding,
    Spec,
)
from search.scope import provider
from search.scope.build import build_part_scope
from search.scope.compile import Refusal
from search.scope.edits import ScopeTableError, apply_binding
from search.scope.provider import _settled, inclusive_reach, provider_rule, settle_provider
from search.scope.rules import AXIS_RULES
from search.scope.state import AxisRecord, ScopeState
from search.scope.tables import ScopeFacts
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from tests.engine._fake_coverage import (
    FakeCoverageService,
    build_coverage_index,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.fixtures import FakeGraph
from tests.scope import test_basis as basis
from tests.scope._axes import carried, explicit, switch
from tests.scope.test_basis import _crs_member, _graph, _PickingLLM

_USA = "country/USA"
_KEN = "country/KEN"
_CRS = "provenance/OECDCreditorReportingSystem"
_IMPUTED = "provenance/ONEImputedMultilateralSectors"
_INCLUSIVE_SOURCE = "provenance/ONEBilateralAndImputedMultilateral"
_ODA = "CRSTypeOfFlow10"

_CRS_MEMBERS = tuple(
    _crs_member(
        code,
        measured_property="amount",
        netting="Gross",
        flow=_ODA,
        modality="CRSCooperationModalityA01",
    )
    for code in ("CRSPurposeCode120", "CRSPurposeCode130")
)
_IMPUTED_MEMBERS = (
    "AttributedAmount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_DAC2aAidType106_"
    "CRSPurposeCode12110_Nominal",
    "AttributedAmount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_DAC2aAidType106_"
    "CRSPurposeCode13010_Nominal",
)
_BILATERAL_YEARS = tuple(range(2005, 2025))
_FLOW = RoleNames(donor="financingProvider", recipient="financingRecipient")


def _coverage_rows(imputed_years: Sequence[int]) -> tuple[list, dict]:
    """CRS health series from the United States to Kenya in every year from 2005 to 2024,
    and imputed multilateral health series in `imputed_years`, with their role names and
    facets."""
    rows = [
        *((dcid, "crs", _USA, _KEN, list(_BILATERAL_YEARS)) for dcid in _CRS_MEMBERS),
        *((dcid, "imputed", _USA, _KEN, list(imputed_years)) for dcid in _IMPUTED_MEMBERS),
    ]
    return rows, {
        "role_names": dict.fromkeys((*_CRS_MEMBERS, *_IMPUTED_MEMBERS), _FLOW),
        "facet_metadata": {
            "crs": FacetMetadata(provenance_id=_CRS, unit="USD"),
            "imputed": FacetMetadata(provenance_id=_IMPUTED, unit="USD"),
        },
    }


def _cover(monkeypatch: pytest.MonkeyPatch, imputed_years: Sequence[int]) -> None:
    rows, extra = _coverage_rows(imputed_years)
    patch_coverage(monkeypatch, rows, **extra)


def _resolve(request: ResolveRequest) -> Spec:
    response = offline_resolve(
        request,
        graph=_graph("health aid", _CRS_MEMBERS),
        llm=_PickingLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    return part.spec


def _ask(message: str, *, time: str | None) -> Spec:
    return _resolve(
        ResolveRequest(
            input=ParsedInput(
                parts=[
                    ParsedPart(
                        variable_text="health aid",
                        entities=[
                            ParsedEntity(text="United States", role_text="from"),
                            ParsedEntity(text="Kenya", role_text="to"),
                        ],
                        time_text=time,
                    )
                ],
                user_message=message,
            )
        )
    )


def _resubmit(identity: BaseIdentity, entries: list[Entry]) -> Spec:
    return _resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(
                        variable_text="health aid", identity=identity, scope_entries=entries
                    )
                ]
            )
        )
    )


def _scope(spec: Spec) -> PartScope:
    scope = spec.resolved.scope_choices
    assert scope is not None
    return scope


def _provider(spec: Spec) -> ScopeChoice:
    (choice,) = (c for c in _scope(spec).choices if c.axis == "provider")
    return choice


def _source(spec: Spec) -> str | None:
    binding = (spec.definition.identity.facets or {}).get("provenance")
    return binding.ref if isinstance(binding, IdentityBindingValue) else None


def _switch(option_id: str, choice: ScopeChoice) -> Entry:
    return Entry(
        axis="provider",
        option_id=option_id,
        provenance="explicit",
        origin=ChoiceOrigin(choice_id=choice.id),
    )


def _served_window_texts(spec: Spec) -> dict[str, object]:
    """The years an answer states it serves: its served date range, its when slot's window,
    and the period its sentence names."""
    served = spec.resolved.stat_vars.data_date_range
    (when,) = (s for s in spec.resolved.slots if s.key.axis == "when")
    window = when.binding.value.time_window
    (token,) = (t for t in spec.resolved.sentence_plan or () if t.axis == "when")
    return {
        "data_date_range": (served.start, served.end) if served is not None else None,
        "when_slot": (window.start_year, window.end_year) if window is not None else None,
        "sentence_when": token.text,
    }


_SCOPE_RULE = {f"provider:{row.scope}": row.rule for row in PROVIDER_SCOPE_RULES}
"""The rule each provider option names on a part the CRS serves where no convention settles
it."""
_RULE_SCOPES = {
    **{row.rule: row.scope for row in PROVIDER_SCOPE_RULES},
    **{row.rule: row.scope for row in PROVIDER_CONVENTIONS},
    **{row.rule: row.scope for row in PUBLISHED_PROVIDER_SCOPES},
}
"""The provider scope each declared provider rule describes."""

_ALL_YEARS = tuple(range(2005, 2025))
_FROM_2011 = tuple(range(2011, 2025))


@pytest.mark.parametrize(
    ("message", "time", "imputed", "active", "provenance", "rule", "inclusive"),
    [
        pytest.param(
            "total health aid from the United States to Kenya, 2015 to 2020",
            "2015-2020",
            _ALL_YEARS,
            "provider:inclusive",
            "default",
            "named_donor_total",
            "available",
            id="named-donor-total-covered",
        ),
        pytest.param(
            "total health aid from the United States to Kenya, 2005 to 2024",
            "2005-2024",
            _FROM_2011,
            "provider:bilateral",
            "default",
            "bilateral_flows",
            "partial",
            id="named-donor-total-shortened",
        ),
        pytest.param(
            "health aid from the United States to Kenya, 2015 to 2020",
            "2015-2020",
            _ALL_YEARS,
            "provider:bilateral",
            "default",
            "bilateral_flows",
            "available",
            id="no-total",
        ),
        pytest.param(
            "total bilateral health aid from the United States to Kenya, 2015 to 2020",
            "2015-2020",
            _ALL_YEARS,
            "provider:bilateral",
            "explicit",
            "bilateral_flows",
            "available",
            id="bilateral-typed",
        ),
        # The request asks for no total, so the convention does not match and the stated
        # scope names its own rule.
        pytest.param(
            "health aid from the United States to Kenya including multilateral, 2015 to 2020",
            "2015-2020",
            _ALL_YEARS,
            "provider:inclusive",
            "explicit",
            "bilateral_plus_imputed",
            "available",
            id="inclusive-typed-no-total",
        ),
        pytest.param(
            "total health aid from the United States to Kenya including multilateral, 2015 to 2020",
            "2015-2020",
            _ALL_YEARS,
            "provider:inclusive",
            "explicit",
            "named_donor_total",
            "available",
            id="inclusive-typed-total",
        ),
    ],
)
def test_the_provider_default_rule(
    monkeypatch, message, time, imputed, active, provenance, rule, inclusive
):
    _cover(monkeypatch, imputed)

    spec = _ask(message, time=time)

    choice = _provider(spec)
    assert (choice.active, choice.provenance, choice.rule) == (active, provenance, rule)
    (option,) = (o for o in choice.options if o.option_id == "provider:inclusive")
    assert option.availability.status == inclusive
    expected_source = _INCLUSIVE_SOURCE if active == "provider:inclusive" else None
    assert _source(spec) == expected_source
    entry = next((e for e in _scope(spec).entries if e.axis == "provider"), None)
    if provenance == "explicit":
        assert entry is not None
        assert entry.option_id == active
    else:
        assert entry is None
    # A resubmit matches no convention: it holds the option the search settled, posted as
    # the identity or chosen by a switch, under the option's own rule.
    own = _SCOPE_RULE[active]
    again = _provider(_resubmit(spec.definition.identity, []))
    assert (again.active, again.rule) == (active, own)
    chosen = _provider(_resubmit(_scope(spec).base.identity, [_switch(active, choice)]))
    assert (chosen.active, chosen.rule) == (active, own)


def test_a_covered_total_answers_the_composite_with_its_components(monkeypatch):
    _cover(monkeypatch, _ALL_YEARS)

    spec = _ask("total health aid from the United States to Kenya, 2015 to 2020", time="2015-2020")

    scope = _scope(spec)
    assert spec.spec_id == scope.compiled.spec_id == scope.id
    assert spec.spec_id == identity_spec_id(spec.definition.identity)
    assert spec.resolved.facets["provenance"].ref == _INCLUSIVE_SOURCE
    assert [(c.role, c.provenance, c.periods) for c in scope.compiled.components] == [
        ("bilateral", _CRS, [(2015, 2020)]),
        ("imputed", _IMPUTED, [(2015, 2020)]),
    ]
    # A mint serves every member of each component's family; two CRS and two imputed of
    # them hold a series from the US to Kenya.
    counts = [c.stat_vars for c in scope.compiled.components]
    assert all(isinstance(count, int) and count >= 2 for count in counts)
    assert spec.resolved.stat_vars.member_count == len(_CRS_MEMBERS) + len(_IMPUTED_MEMBERS)
    assert spec.resolved.stat_vars.served_count == sum(counts)
    headline = spec.resolved.headline
    assert headline is not None
    assert (headline.year, headline.series) == (2020, 4)
    assert [year.year for year in headline.recent_years] == [2020, 2019, 2018]
    components = component_identities(
        get_index(), spec.definition.identity.model_dump(mode="json", exclude_none=True)
    )
    assert components is not None
    assert [
        c.identity.model_dump(mode="json", exclude_none=True) for c in scope.compiled.components
    ] == [dict(c.identity) for c in components]


def test_a_composite_answer_cites_both_tables_and_names_its_measure_by_the_declaration(
    monkeypatch,
):
    _cover(monkeypatch, _ALL_YEARS)
    bilateral = _ask("health aid from the United States to Kenya, 2015 to 2020", time="2015-2020")
    declaration = get_index().composite(_INCLUSIVE_SOURCE)
    assert declaration is not None and bilateral.resolved.measure_label is not None

    spec = _ask("total health aid from the United States to Kenya, 2015 to 2020", time="2015-2020")

    sources = spec.resolved.trace.resolved_sources
    assert [source.ref for source in sources] == [_CRS, _IMPUTED]
    assert spec.resolved.measure_label == (
        f"{bilateral.resolved.measure_label} {declaration.measure_noun}"
    )
    sentence = spec.resolved.rendered_sentence or ""
    assert declaration.measure_noun in sentence
    assert all(source.label.split(" (")[0] in sentence for source in sources)


def test_a_window_the_imputed_table_covers_in_part_is_offered_shortened_and_switched_back(
    monkeypatch,
):
    """2005 to 2024 with imputed series from 2011: the inclusive scope is offered over 2011 to
    2024 and the part stays bilateral over 2005 to 2024. A switch compiles a 2011 to 2024
    composite, and the switch back returns the base spec_id."""
    _cover(monkeypatch, _FROM_2011)
    first = _ask("total health aid from the United States to Kenya, 2005 to 2024", time="2005-2024")
    scope = _scope(first)
    choice = _provider(first)
    (inclusive,) = (o for o in choice.options if o.option_id == "provider:inclusive")

    assert inclusive.availability.years == [(2011, 2024)]
    assert inclusive.availability.code == "imputed_coverage"
    assert inclusive.binding.model_dump() == {
        "kind": "provider",
        "scope": "inclusive",
        "window": (2011, 2024),
    }
    assert inclusive.label == "bilateral plus imputed multilateral, 2011 to 2024"
    assert choice.active == "provider:bilateral"
    assert first.definition.identity.requested_window == IdentityWindowRange(
        start_year=2005, end_year=2024
    )

    switched = _resubmit(scope.base.identity, [_switch("provider:inclusive", choice)])
    assert _source(switched) == _INCLUSIVE_SOURCE
    assert switched.definition.identity.requested_window == IdentityWindowRange(
        start_year=2011, end_year=2024
    )
    assert _provider(switched).active == "provider:inclusive"
    assert _scope(switched).base.spec_id == first.spec_id

    back = _resubmit(scope.base.identity, [_switch("provider:bilateral", choice)])
    assert back.spec_id == first.spec_id


def test_a_stated_inclusive_scope_over_a_part_covered_window_is_reported_not_applied(
    monkeypatch,
):
    _cover(monkeypatch, _FROM_2011)

    spec = _ask(
        "health aid from the United States to Kenya including multilateral, 2005 to 2024",
        time="2005-2024",
    )

    assert _provider(spec).active == "provider:bilateral"
    (dropped,) = (d for d in _scope(spec).dropped_entries if d.entry.axis == "provider")
    assert (dropped.entry.option_id, dropped.code, dropped.cause) == (
        "provider:inclusive",
        "stated_choice_not_applied",
        "imputed_coverage",
    )


def test_a_latest_window_elects_the_latest_year_both_tables_cover_and_switches_back(
    monkeypatch,
):
    """CRS series through 2024 and imputed ones through 2023: the inclusive answer reads 2023,
    the base keeps the `latest` window, a switch to bilateral answers the bilateral search's
    own latest year, and the switch back returns the inclusive spec_id."""
    _cover(monkeypatch, tuple(range(2011, 2024)))
    bilateral = _ask("health aid from the United States to Kenya, latest year", time="latest")
    latest_year = IdentityWindowRange(start_year=2023, end_year=2023)

    spec = _ask("total health aid from the United States to Kenya, latest year", time="latest")

    scope = _scope(spec)
    choice = _provider(spec)
    assert choice.active == "provider:inclusive"
    assert _source(spec) == _INCLUSIVE_SOURCE
    assert spec.definition.identity.requested_window == latest_year
    assert scope.base.identity.requested_window.kind == "latest"
    assert [c.periods for c in scope.compiled.components] == [[(2023, 2023)], [(2023, 2023)]]
    assert _served_window_texts(spec) == {
        "data_date_range": ("2023", "2023"),
        "when_slot": (2023, 2023),
        "sentence_when": "in 2023",
    }
    assert spec.resolved.download is None

    to_bilateral = _resubmit(scope.base.identity, [_switch("provider:bilateral", choice)])
    assert _source(to_bilateral) is None
    assert to_bilateral.spec_id == bilateral.spec_id

    back = _resubmit(scope.base.identity, [_switch("provider:inclusive", choice)])
    assert back.spec_id == spec.spec_id
    assert _resubmit(scope.base.identity, []).spec_id == spec.spec_id
    # Posted with no scope entries, the composite still reads the year both tables cover.
    posted = _resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(variable_text="health aid", identity=scope.base.identity)
                ]
            )
        )
    )
    assert posted.definition.identity.requested_window == latest_year
    assert posted.spec_id == spec.spec_id


def test_a_resubmit_of_a_composite_with_its_own_entries_returns_its_spec_id(monkeypatch):
    _cover(monkeypatch, _ALL_YEARS)
    first = _ask("total health aid from the United States to Kenya, 2015 to 2020", time="2015-2020")
    scope = _scope(first)

    again = _resubmit(scope.base.identity, list(scope.entries))

    assert again.spec_id == first.spec_id
    assert _scope(again).compiled.spec_id == first.spec_id
    assert _provider(again).active == "provider:inclusive"


def test_a_resubmit_of_a_composite_identity_answers_the_composite(monkeypatch):
    """A composite identity posted with no scope entries resolves the identity its option
    was set on and answers as the composite, under the composite's spec_id."""
    _cover(monkeypatch, _ALL_YEARS)
    first = _ask("total health aid from the United States to Kenya, 2015 to 2020", time="2015-2020")

    again = _resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(
                        variable_text="health aid", identity=first.definition.identity
                    )
                ]
            )
        )
    )

    assert again.spec_id == first.spec_id
    assert again.resolved.facets["provenance"].ref == _INCLUSIVE_SOURCE


def _bilateral_identity(
    *, origin: Sequence[str] = (_USA,), latest: bool = False, **slots: str
) -> BaseIdentity:
    document = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "FinancialFlow",
            "measured_property": "amount",
            "stat_type": "measuredValue",
            "measurement_qualifier": "Nominal",
        },
        "slots": sorted(
            [
                {
                    "property": "crsPurpose",
                    "binding": {"state": "set", "refs": ["CRSPurposeCode120", "CRSPurposeCode130"]},
                },
                {"property": "crsTypeOfFlow", "binding": {"state": "value", "ref": _ODA}},
                *(
                    {"property": prop, "binding": {"state": "unbound"}}
                    for prop in ("crsCooperationModality", "crsTypeOfFinance")
                    if prop not in slots
                ),
                {
                    "property": "financialFlowType",
                    "binding": {"state": "value", "ref": "Disbursement"},
                },
                {"property": "nettingBasis", "binding": {"state": "value", "ref": "Gross"}},
                *(
                    {"property": prop, "binding": {"state": "value", "ref": code}}
                    for prop, code in slots.items()
                ),
            ],
            key=lambda slot: slot["property"],
        ),
        "entities": {"origin": list(origin), "destination": [_KEN]},
        "requested_window": (
            {"kind": "latest"}
            if latest
            else {"kind": "range", "start_year": 2015, "end_year": 2020}
        ),
    }
    if not origin:
        del document["entities"]["origin"]
    return BaseIdentity.model_validate(document)


def _facts(rows: Sequence[tuple] | None = None) -> ScopeFacts:
    """The fixture code index and a coverage snapshot of `rows`, by default
    `_coverage_rows(_ALL_YEARS)`'s."""
    default, extra = _coverage_rows(_ALL_YEARS)
    index = build_coverage_index(list(rows) if rows is not None else default, **extra)
    return ScopeFacts(index=get_index(), coverage=FakeCoverageService(index).snapshot())


def _total() -> ScopeState:
    return ScopeState(total=True)


@pytest.mark.parametrize(
    ("slots", "why"),
    [
        pytest.param(
            {"crsCooperationModality": "CRSCooperationModalityA01"}, "modality", id="modality"
        ),
        pytest.param(
            {"crsTypeOfFinance": "CRSTypeOfFinance110"}, "finance_type", id="finance-type"
        ),
    ],
)
def test_a_bound_modality_or_instrument_keeps_a_total_bilateral_with_the_reason(slots, why):
    facts = _facts()
    identity = _bilateral_identity(**slots)
    table = provider_rule(identity, None, facts)
    record = settle_provider(identity, _total(), facts)

    assert table is not None
    assert record is not None
    (inclusive,) = (o for o in table.options if o.option_id == "provider:inclusive")
    assert (inclusive.availability.status, inclusive.availability.code) == (
        "unavailable",
        "provider_not_offered",
    )
    assert inclusive.availability.detail == {"why": why}
    assert (record.active, record.provenance, record.rule) == (
        "provider:bilateral",
        "default",
        "bilateral_flows",
    )


def test_the_bilateral_and_inclusive_edits_are_inverses():
    base = _bilateral_identity()
    inclusive = apply_binding(base, ProviderBinding(scope="inclusive"), [])
    back = apply_binding(inclusive, ProviderBinding(scope="bilateral"), [])

    assert (inclusive.facets or {})["provenance"] == IdentityBindingValue(ref=_INCLUSIVE_SOURCE)
    assert identity_spec_id(back) == identity_spec_id(base)
    windowed = apply_binding(base, ProviderBinding(scope="inclusive", window=(2011, 2024)), [])
    assert windowed.requested_window == IdentityWindowRange(start_year=2011, end_year=2024)


_GBR = "country/GBR"
_IDA = "InternationalDevelopmentAssociation"
_EU = "EuropeanUnionInstitutions"
_FRA = "country/FRA"


def _rows_for(donors: Sequence[str], *, imputed: Sequence[str]) -> list[tuple]:
    """CRS health series to Kenya from each of `donors` in every year from 2005 to 2024, and
    imputed ones from each of `imputed`, the providers the imputed table carries rows for."""
    return [
        *(
            (dcid, "crs", donor, _KEN, list(_ALL_YEARS))
            for donor in donors
            for dcid in _CRS_MEMBERS
        ),
        *(
            (dcid, "imputed", donor, _KEN, list(_ALL_YEARS))
            for donor in imputed
            for dcid in _IMPUTED_MEMBERS
        ),
    ]


@pytest.mark.parametrize(
    ("origin", "carried", "active", "inclusive", "why"),
    [
        pytest.param((_USA,), (_USA,), "provider:inclusive", "available", None, id="us-alone"),
        # The imputed table attributes each multilateral's spending to each bilateral
        # provider by that provider's own share of its core resources, so two providers'
        # imputed shares are disjoint and add.
        pytest.param(
            (_USA, _GBR), (_USA, _GBR), "provider:inclusive", "available", None, id="two-donors"
        ),
        # IDA's own outflows are what the imputed rows attribute to the US, so the pool
        # would count them twice. The imputed table carries no rows for IDA.
        pytest.param(
            (_USA, _IDA),
            (_USA,),
            "provider:bilateral",
            "unavailable",
            "pooled_providers",
            id="us-and-ida",
        ),
        # EU institutions are a provider the imputed table carries, and a multilateral
        # whose own outflows its rows attribute to the EU members.
        pytest.param(
            (_FRA, _EU),
            (_FRA, _EU),
            "provider:bilateral",
            "unavailable",
            "pooled_providers",
            id="france-and-eu-institutions",
        ),
        pytest.param(
            (_IDA,), (_USA,), "provider:bilateral", "unavailable", "provider", id="ida-alone"
        ),
        pytest.param(
            (), (_USA,), "provider:bilateral", "unavailable", "all_providers", id="all-providers"
        ),
    ],
)
def test_the_inclusive_scope_is_offered_only_for_providers_the_imputed_table_allocates_to(
    origin, carried, active, inclusive, why
):
    """The inclusive scope adds each named provider's imputed share of multilateral
    spending. It is offered when the imputed table carries rows for every named provider
    and none of them is a multilateral whose outflows its rows redistribute; otherwise it is
    unavailable, `provider_not_offered`, with the reason, and neither the default, a typed
    "including multilateral" nor a switch applies it."""
    facts = _facts(_rows_for((_USA, _GBR, _IDA, _FRA, _EU), imputed=carried))
    identity = _bilateral_identity(origin=origin)

    table = provider_rule(identity, None, facts)
    record = settle_provider(identity, _total(), facts)
    typed = settle_provider(
        identity,
        ScopeState(entries=(explicit("provider:inclusive"),), total=True),
        facts,
    )
    switched = build_part_scope(
        base=identity, entries=[switch("provider:inclusive")], records={}, facts=facts
    )

    assert table is not None and record is not None and typed is not None
    (option,) = (o for o in table.options if o.option_id == "provider:inclusive")
    assert option.availability.status == inclusive
    assert record.active == active
    if why is None:
        assert option.availability.code is None
        assert typed.active == "provider:inclusive"
        assert isinstance(switched, PartScope)
        assert (switched.compiled.identity.facets or {}).get("provenance") == (
            IdentityBindingValue(ref=_INCLUSIVE_SOURCE)
        )
    else:
        assert option.availability.code == "provider_not_offered"
        assert option.availability.detail == {"why": why}
        assert typed.active == "provider:bilateral"
        assert switched == Refusal(
            code="requested_option_unavailable", entry=0, cause="provider_not_offered"
        )


def test_a_latest_window_elects_the_latest_year_both_tables_hold():
    """Bilateral series in 2022 and 2024 and imputed ones in 2022 and 2023: the only year
    both hold is 2022, so the inclusive scope reads 2022."""
    rows = [
        *((dcid, "crs", _USA, _KEN, [2022, 2024]) for dcid in _CRS_MEMBERS),
        *((dcid, "imputed", _USA, _KEN, [2022, 2023]) for dcid in _IMPUTED_MEMBERS),
    ]
    facts = _facts(rows)
    identity = _bilateral_identity(latest=True)

    reach = inclusive_reach(identity, facts)
    table = provider_rule(identity, None, facts)

    assert reach.latest == 2022
    assert table is not None
    (option,) = (o for o in table.options if o.option_id == "provider:inclusive")
    assert option.binding.model_dump()["window"] == (2022, 2022)


def test_the_inclusive_scope_serves_the_years_the_imputed_table_covers_not_those_the_part_holds():
    """The imputed table carries no row to Kenya before 2018 but covers every year (it holds
    Ghana's), so the mint writes zero imputed rows for Kenya there: the inclusive scope
    serves every year the bilateral answer holds, as the mint does."""
    rows = [
        *((dcid, "crs", _USA, _KEN, list(_BILATERAL_YEARS)) for dcid in _CRS_MEMBERS),
        *((dcid, "imputed", _USA, _KEN, list(range(2018, 2025))) for dcid in _IMPUTED_MEMBERS),
        *((dcid, "imputed", _USA, "country/GHA", list(_ALL_YEARS)) for dcid in _IMPUTED_MEMBERS),
    ]
    _, extra = _coverage_rows(_ALL_YEARS)
    facts = ScopeFacts(
        index=get_index(),
        coverage=FakeCoverageService(build_coverage_index(rows, **extra)).snapshot(),
    )
    identity = _bilateral_identity()

    reach = inclusive_reach(identity, facts)
    table = provider_rule(identity, None, facts)

    assert reach.covered == reach.asked == frozenset(range(2015, 2021))
    assert table is not None
    (option,) = (o for o in table.options if o.option_id == "provider:inclusive")
    assert option.availability.status == "available"


# DAC1's published totals: a donor's total ODA counts its core contributions to
# multilaterals, and its bilateral ODA its bilateral flows alone. The parts resolve over
# `test_basis`'s fixture graph, where DAC1's total ODA members sit beside the CRS's.
_DAC1 = basis._DAC1
_HEADLINE = basis._HEADLINE
_DAC1_TOTAL_ODA = "DAC1AidType1010"
_DAC1_BILATERAL_ODA = "DAC1AidType1015"
_DAC_MULTILATERALS = "DACMultilaterals"
_DAC1_TOTAL_MEMBERS = basis._DAC1_MEMBERS
_DAC1_BILATERAL_MEMBERS = (
    "Amount_FinancialFlow_Gross_Disbursement_DAC1AidType1015_Nominal",
    "Amount_FinancialFlow_Net_Disbursement_DAC1AidType1015_Nominal",
    "GrantEquivalent_FinancialFlow_Disbursement_DAC1AidType1015_Nominal",
)
_DAC1_MEMBERS_BY_CODE = {
    _DAC1_TOTAL_ODA: _DAC1_TOTAL_MEMBERS,
    _DAC1_BILATERAL_ODA: _DAC1_BILATERAL_MEMBERS,
}


def _provider_options(spec: Spec) -> dict[str, tuple[str, str | None, dict[str, str]]]:
    """The spec's provider options, by id: each one's status, reason code and detail."""
    return {
        o.option_id: (o.availability.status, o.availability.code, o.availability.detail)
        for o in _provider(spec).options
    }


def _ask_dac1(monkeypatch: pytest.MonkeyPatch, cell: str, *, message: str | None = None) -> Spec:
    """`basis._CELLS[cell]` asked with `message`, over US series of DAC1's total and
    bilateral ODA."""
    basis._cover(monkeypatch, (*_DAC1_TOTAL_MEMBERS, *_DAC1_BILATERAL_MEMBERS))
    row = basis._CELLS[cell]
    return basis._ask(row.variable, time=row.time, recipient=row.recipient, message=message)


def _slot_ref(identity: BaseIdentity, prop: str) -> str | None:
    """The code `identity` binds on `prop` as a value, None when it binds none."""
    binding = next((s.binding for s in identity.slots if s.property == prop), None)
    return binding.ref if isinstance(binding, IdentityBindingValue) else None


@pytest.mark.parametrize("cell", ["donor-only-2019", "donor-only-2016", "donor-only-2010-2024"])
def test_a_dac1_total_holds_the_contributions_it_counts_and_offers_bilateral_oda(
    monkeypatch, cell: str
):
    """DAC1's total ODA counts core contributions to multilaterals, on every basis and on
    the OECD headline composite. It holds the contributions scope, offers bilateral flows as
    DAC1's bilateral ODA, and the inclusive scope as unavailable since it counts the
    contributions in place of an imputed share."""
    spec = _ask_dac1(monkeypatch, cell)

    choice = _provider(spec)
    assert (choice.active, choice.provenance, choice.rule) == (
        "provider:contributions",
        "default",
        "contribution_total",
    )
    assert _provider_options(spec) == {
        "provider:bilateral": ("available", None, {}),
        "provider:inclusive": ("unavailable", "contributions_counted", {"dataset": _DAC1}),
        "provider:contributions": ("available", None, {}),
    }
    (bilateral,) = (o for o in choice.options if o.option_id == "provider:bilateral")
    assert bilateral.binding == SlotBinding(
        slot="dac1AidType", state="value", refs=[_DAC1_BILATERAL_ODA]
    )


@pytest.mark.parametrize("cell", ["donor-only-2019", "donor-only-2010-2024"])
def test_the_contributions_scope_leaves_a_dac1_totals_spec_id_as_it_is(monkeypatch, cell: str):
    """The provider record edits nothing. A DAC1 total compiles to the spec_id its scope
    compiles to with no provider rule, and a switch to the contributions scope it holds
    resubmits to the same spec_id."""
    spec = _ask_dac1(monkeypatch, cell)
    scope = _scope(spec)
    choice = _provider(spec)
    facts = ScopeFacts(index=get_index(), coverage=empty_coverage_snapshot())
    without_provider = {axis: rule for axis, rule in AXIS_RULES.items() if axis != "provider"}

    with_rule = build_part_scope(base=scope.base.identity, entries=[], records={}, facts=facts)
    without_rule = build_part_scope(
        base=scope.base.identity, entries=[], records={}, facts=facts, rules=without_provider
    )
    row = basis._CELLS[cell]
    switched = basis._resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(
                        variable_text=row.variable,
                        identity=scope.base.identity,
                        scope_entries=[_switch("provider:contributions", choice)],
                    )
                ]
            )
        ),
        row.variable,
    )

    assert isinstance(with_rule, PartScope) and isinstance(without_rule, PartScope)
    assert any(c.axis == "provider" for c in with_rule.choices)
    assert with_rule.compiled.spec_id == without_rule.compiled.spec_id == spec.spec_id
    assert switched.spec_id == spec.spec_id


def test_a_switch_to_bilateral_on_a_dac1_total_reads_bilateral_oda_and_switches_back(
    monkeypatch,
):
    """A switch to bilateral flows on DAC1's total ODA compiles the part's identity onto
    bilateral ODA. That identity holds the bilateral scope under its own rule, and a switch
    back to core contributions compiles to the total's spec_id."""
    spec = _ask_dac1(monkeypatch, "donor-only-2019")
    base = _scope(spec).base.identity
    facts = _dac1_facts(_DAC1_TOTAL_ODA, _DAC1_BILATERAL_ODA)

    to_bilateral = build_part_scope(
        base=base, entries=[switch("provider:bilateral")], records={}, facts=facts
    )
    assert isinstance(to_bilateral, PartScope)
    bilateral_oda = to_bilateral.compiled.identity
    assert _slot_ref(base, "dac1AidType") == _DAC1_TOTAL_ODA
    assert _slot_ref(bilateral_oda, "dac1AidType") == _DAC1_BILATERAL_ODA
    (choice,) = (c for c in to_bilateral.choices if c.axis == "provider")
    assert (choice.active, choice.rule) == ("provider:bilateral", "bilateral_total")

    held = build_part_scope(base=bilateral_oda, entries=[], records={}, facts=facts)
    back = build_part_scope(
        base=bilateral_oda, entries=[switch("provider:contributions")], records={}, facts=facts
    )
    assert isinstance(held, PartScope) and isinstance(back, PartScope)
    (choice,) = (c for c in held.choices if c.axis == "provider")
    assert (choice.active, choice.rule) == ("provider:bilateral", "bilateral_total")
    assert back.compiled.spec_id == spec.spec_id


@pytest.mark.parametrize(
    ("message", "option_id", "cause", "switched"),
    [
        pytest.param(
            "bilateral aid from the US in 2019",
            "provider:bilateral",
            "not_served",
            _DAC1_BILATERAL_ODA,
            id="bilateral",
        ),
        pytest.param(
            "aid from the US including multilateral in 2019",
            "provider:inclusive",
            "contributions_counted",
            None,
            id="including-multilateral",
        ),
    ],
)
def test_a_stated_provider_scope_on_a_dac1_total_is_reported_and_left_to_a_switch(
    monkeypatch, message: str, option_id: str, cause: str, switched: str | None
):
    """A fresh search answers the total it resolved. A typed "bilateral" on a donor's DAC1
    total is reported as not served, and a switch to it reads bilateral ODA. A typed
    "including multilateral" is reported as the total counting core contributions in place
    of an imputed share, and a switch to it refuses with that cause."""
    spec = _ask_dac1(monkeypatch, "donor-only-2019", message=message)
    scope = _scope(spec)

    assert _provider(spec).active == "provider:contributions"
    assert _provider_options(spec)["provider:contributions"] == ("available", None, {})
    assert _slot_ref(spec.definition.identity, "dac1AidType") == _DAC1_TOTAL_ODA
    (dropped,) = (d for d in scope.dropped_entries if d.entry.axis == "provider")
    assert (dropped.entry.option_id, dropped.code, dropped.cause) == (
        option_id,
        "stated_choice_not_applied",
        cause,
    )
    compiled = build_part_scope(
        base=scope.base.identity,
        entries=[switch(option_id)],
        records={},
        facts=_dac1_facts(_DAC1_TOTAL_ODA, _DAC1_BILATERAL_ODA),
    )
    if switched is None:
        assert compiled == Refusal(code="requested_option_unavailable", entry=0, cause=cause)
    else:
        assert isinstance(compiled, PartScope)
        assert _slot_ref(compiled.compiled.identity, "dac1AidType") == switched


def _dac1_identity(
    aid_type: str,
    origin: Sequence[str],
    *,
    window: tuple[int, int] | None = (2019, 2019),
    headline: bool = False,
) -> BaseIdentity:
    """A DAC1 grant-equivalent total of `aid_type` from `origin` over `window`, by default
    2019, its latest year when `window` is None, or with `headline` its OECD headline
    composite: net flows to 2017 and grant equivalents from 2018."""
    netting = [{"property": "nettingBasis", "binding": {"state": "value", "ref": "Net"}}]
    return BaseIdentity.model_validate(
        {
            "identity_version": "spec1",
            "shape": {
                "kind": "statvar_schema",
                "population_type": "FinancialFlow",
                "measured_property": "amount" if headline else "grantEquivalent",
                "stat_type": "measuredValue",
                "measurement_qualifier": "Nominal",
            },
            "slots": [
                {"property": "dac1AidType", "binding": {"state": "value", "ref": aid_type}},
                {
                    "property": "financialFlowType",
                    "binding": {"state": "value", "ref": "Disbursement"},
                },
                *(netting if headline else ()),
            ],
            "entities": {"origin": list(origin)} if origin else {},
            "requested_window": (
                {"kind": "latest"}
                if window is None
                else {"kind": "range", "start_year": window[0], "end_year": window[1]}
            ),
            **(
                {"facets": {"provenance": {"state": "value", "ref": _HEADLINE}}} if headline else {}
            ),
        }
    )


_GROUP_OF_SEVEN = "GroupOfSeven"
_DAC_COUNTRIES = "DACCountriesTotal"
_DAC_EU_MEMBERS = "DACEUMembers"
_EU_INSTITUTIONS = "EuropeanUnionInstitutions"
_DEU = "country/DEU"
_DAC1_DONORS = (
    _USA,
    _DEU,
    _DAC_COUNTRIES,
    _GROUP_OF_SEVEN,
    _EU_INSTITUTIONS,
    _DAC_MULTILATERALS,
    _DAC_EU_MEMBERS,
)


def _dac1_facts(*codes: str, extra: Sequence[tuple] = ()) -> ScopeFacts:
    """The fixture code index and a coverage snapshot holding series of every member of
    DAC1's aid types `codes` from every donor the DAC1 tests name, in every year from 2005 to
    2024, and the DAC1 series `extra`, each a member, a donor and its years."""
    rows = [
        *(
            (dcid, "dac1", donor, ABSENT_ENTITY, list(_ALL_YEARS))
            for code in codes
            for dcid in _DAC1_MEMBERS_BY_CODE[code]
            for donor in _DAC1_DONORS
        ),
        *((dcid, "dac1", donor, ABSENT_ENTITY, list(years)) for dcid, donor, years in extra),
    ]
    members = (*_DAC1_TOTAL_MEMBERS, *_DAC1_BILATERAL_MEMBERS)
    index = build_coverage_index(
        rows,
        role_names=dict.fromkeys(members, RoleNames(donor="financingProvider", recipient=None)),
        facet_metadata={"dac1": FacetMetadata(provenance_id=_DAC1, unit="USD")},
    )
    return ScopeFacts(index=get_index(), coverage=FakeCoverageService(index).snapshot())


@pytest.mark.parametrize(
    ("aid_type", "origin", "expected"),
    [
        pytest.param(
            _DAC1_TOTAL_ODA,
            (_USA,),
            ("contributions", "contribution_total", ["bilateral", "contributions"]),
            id="total-oda",
        ),
        pytest.param(
            _DAC1_BILATERAL_ODA,
            (_USA,),
            ("bilateral", "bilateral_total", ["bilateral", "contributions"]),
            id="bilateral-oda",
        ),
        pytest.param(
            _DAC1_TOTAL_ODA,
            (_DAC_COUNTRIES,),
            ("contributions", "contribution_total", ["bilateral", "contributions"]),
            id="group-of-countries",
        ),
        pytest.param(
            _DAC1_TOTAL_ODA,
            (_GROUP_OF_SEVEN,),
            ("contributions", "contribution_total", ["bilateral", "contributions"]),
            id="g7",
        ),
        pytest.param(
            _DAC1_TOTAL_ODA,
            (_EU_INSTITUTIONS,),
            ("contributions", "contribution_total", ["bilateral", "contributions"]),
            id="eu-institutions",
        ),
        # A pool of an EU member and the EU institutions counts the member's core
        # contributions to the institutions twice in total ODA, which a switch from
        # bilateral ODA would reach. A total of a group holding an EU member pools one.
        pytest.param(
            _DAC1_TOTAL_ODA, (_DEU, _EU_INSTITUTIONS), None, id="pool-with-eu-institutions"
        ),
        pytest.param(
            _DAC1_TOTAL_ODA,
            (_GROUP_OF_SEVEN, _EU_INSTITUTIONS),
            None,
            id="pool-of-a-group-with-an-eu-member-and-eu-institutions",
        ),
        # The EU institutions' aid holds no core contribution of a donor outside the EU.
        pytest.param(
            _DAC1_TOTAL_ODA,
            (_USA, _EU_INSTITUTIONS),
            ("contributions", "contribution_total", ["bilateral", "contributions"]),
            id="pool-of-a-non-member-and-eu-institutions",
        ),
        pytest.param(
            _DAC1_BILATERAL_ODA,
            (_DEU, _EU_INSTITUTIONS),
            None,
            id="bilateral-pool-with-eu-institutions",
        ),
        pytest.param(_DAC1_TOTAL_ODA, (_DAC_MULTILATERALS,), None, id="multilaterals"),
        pytest.param(_DAC1_TOTAL_ODA, (_USA, _DAC_MULTILATERALS), None, id="pool-with-them"),
        pytest.param(_DAC1_TOTAL_ODA, (_DAC_EU_MEMBERS,), None, id="total-of-no-declared-group"),
        pytest.param(_DAC1_TOTAL_ODA, (), None, id="no-provider"),
    ],
)
def test_each_dac1_aid_type_holds_its_declared_provider_scope_for_donors(
    aid_type: str,
    origin: tuple[str, ...],
    expected: tuple[ProviderScope, str, list[ProviderScope]] | None,
):
    """Total ODA holds the contributions scope and bilateral ODA the bilateral one, each
    under its row's rule, for a part naming countries, the EU institutions or totals of a
    group of countries alone. A part naming the multilaterals' total, a total of no
    declared group, a pool of the EU institutions and a member, or no provider, has no
    provider record."""
    facts = _dac1_facts(_DAC1_TOTAL_ODA, _DAC1_BILATERAL_ODA)
    identity = _dac1_identity(aid_type, origin)

    table = provider_rule(identity, None, facts)
    record = settle_provider(identity, ScopeState(), facts)

    if expected is None:
        assert table is None and record is None
        return
    assert table is not None and record is not None
    scope, rule, available = expected
    assert (record.active, record.rule, table.rule) == (f"provider:{scope}", rule, rule)
    assert [o.option_id for o in table.options if o.availability.status == "available"] == [
        f"provider:{held}" for held in available
    ]
    assert apply_binding(identity, ProviderBinding(scope=scope), []) == identity


def _members_on(code: str, prefix: str) -> tuple[str, ...]:
    """The fixture's members of DAC1 aid type `code` whose dcid starts with `prefix`."""
    return tuple(m for m in _DAC1_MEMBERS_BY_CODE[code] if m.startswith(prefix))


@pytest.mark.parametrize("row", PUBLISHED_PROVIDER_SCOPES, ids=lambda row: row.code)
@pytest.mark.parametrize(
    ("donor", "prefix", "years", "available"),
    [
        pytest.param(_USA, "GrantEquivalent_", (2019,), True, id="covered"),
        pytest.param(None, "", (), False, id="no-series"),
        pytest.param(_DEU, "GrantEquivalent_", (2019,), False, id="another-donor"),
        pytest.param(_USA, "Amount_", (2019,), False, id="another-basis"),
        pytest.param(_USA, "GrantEquivalent_", (2018, 2020), False, id="another-window"),
    ],
)
def test_a_dac1_total_offers_its_sibling_total_only_where_a_series_covers_the_part(
    row, donor: str | None, prefix: str, years: tuple[int, ...], available: bool
):
    """An option binding the other DAC1 aid type is available where that aid type holds a
    series from the part's donor, on its basis, in its window, and unavailable with
    `option_not_covered` otherwise. The part keeps the total it holds either way. The part
    is a 2019 grant-equivalent total from the United States."""
    (sibling,) = (other for other in PUBLISHED_PROVIDER_SCOPES if other is not row)
    extra = [] if donor is None else [(m, donor, years) for m in _members_on(sibling.code, prefix)]
    facts = _dac1_facts(row.code, extra=extra)
    identity = _dac1_identity(row.code, (_USA,))

    table = provider_rule(identity, None, facts)
    record = settle_provider(identity, ScopeState(), facts)

    assert table is not None and record is not None
    option = table.option(f"provider:{sibling.scope}")
    assert option is not None
    assert (option.availability.status, option.availability.code) == (
        ("available", None) if available else ("unavailable", "option_not_covered")
    )
    assert (record.active, record.rule) == (f"provider:{row.scope}", row.rule)


_GE = "GrantEquivalent_"
_SCOPE_LABELS = {
    "bilateral": OPTION_LABELS.provider_bilateral,
    "contributions": OPTION_LABELS.provider_contributions,
}
_NET = "Amount_FinancialFlow_Net_"
_GROSS = "Amount_FinancialFlow_Gross_"
_COVERED = ("available", None, None)
_NOT_COVERED = ("unavailable", "option_not_covered", None)


def _partial(*years: tuple[int, int]) -> tuple[str, str, list[tuple[int, int]]]:
    return ("partial", "option_not_covered", list(years))


@pytest.mark.parametrize("row", PUBLISHED_PROVIDER_SCOPES, ids=lambda row: row.code)
@pytest.mark.parametrize(
    ("origin", "covered", "expected"),
    [
        pytest.param((_USA,), {_USA: (2015, 2020)}, _COVERED, id="every-year"),
        pytest.param((_USA,), {_USA: (2018, 2020)}, _partial((2018, 2020)), id="later-years"),
        pytest.param(
            (_USA, _DEU),
            {_USA: (2015, 2020), _DEU: (2015, 2020)},
            _COVERED,
            id="pool-every-year",
        ),
        pytest.param(
            (_USA, _DEU),
            {_USA: (2015, 2020), _DEU: (2017, 2020)},
            _partial((2017, 2020)),
            id="pool-later-years",
        ),
        pytest.param((_USA, _DEU), {_USA: (2015, 2020)}, _NOT_COVERED, id="pool-one-donor"),
    ],
)
def test_a_dac1_sibling_option_serves_the_years_every_donor_holds(
    row, origin: tuple[str, ...], covered: dict[str, tuple[int, int]], expected
):
    """A 2015 to 2020 grant-equivalent total offers the other DAC1 aid type over the years
    each donor it names holds a series of it: available where they hold every year the
    total holds, partial over the years they hold where those are fewer, and unavailable
    where a donor of the pool holds none."""
    (sibling,) = (other for other in PUBLISHED_PROVIDER_SCOPES if other is not row)
    extra = [
        (member, donor, range(first, last + 1))
        for donor, (first, last) in covered.items()
        for member in _members_on(sibling.code, _GE)
    ]
    facts = _dac1_facts(row.code, extra=extra)

    table = provider_rule(_dac1_identity(row.code, origin, window=(2015, 2020)), None, facts)

    assert table is not None
    option = table.option(f"provider:{sibling.scope}")
    assert option is not None
    availability = option.availability
    assert (availability.status, availability.code, availability.years) == expected


@pytest.mark.parametrize("row", PUBLISHED_PROVIDER_SCOPES, ids=lambda row: row.code)
@pytest.mark.parametrize(
    ("origin", "sibling_years", "expected", "window"),
    [
        pytest.param((_USA,), {_USA: (2018, 2023)}, _COVERED, 2023, id="same-year"),
        pytest.param(
            (_USA, _DEU),
            {_USA: (2005, 2023), _DEU: (2005, 2023)},
            _COVERED,
            2023,
            id="pool-same-year",
        ),
        pytest.param(
            (_USA,), {_USA: (2018, 2020)}, _partial((2020, 2020)), 2020, id="earlier-year"
        ),
        # The pool's read of the other total elects 2023, which Germany does not hold, so
        # the option serves the newest year both hold.
        pytest.param(
            (_USA, _DEU),
            {_USA: (2005, 2023), _DEU: (2005, 2015)},
            _partial((2015, 2015)),
            2015,
            id="pool-one-donor-ends-earlier",
        ),
        pytest.param((_USA, _DEU), {_USA: (2018, 2023)}, _NOT_COVERED, None, id="pool-one-donor"),
        pytest.param(
            (_USA, _DEU),
            {_USA: (2015, 2016), _DEU: (2019, 2020)},
            _NOT_COVERED,
            None,
            id="pool-no-shared-year",
        ),
    ],
)
def test_a_dac1_sibling_option_on_a_latest_window_serves_the_year_its_pool_shares(
    row,
    origin: tuple[str, ...],
    sibling_years: dict[str, tuple[int, int]],
    expected,
    window: int | None,
):
    """A total each donor holds from 2005 to 2023, asking for the latest year, elects 2023.
    The other DAC1 aid type serves the newest year every donor holds it in, as its read
    elects among those years: available where that year is 2023, partial over that one
    year otherwise, stated in its label, and unavailable where the donors share no year. A
    switch reads that year."""
    (sibling,) = (other for other in PUBLISHED_PROVIDER_SCOPES if other is not row)
    extra = [
        *(
            (member, donor, range(2005, 2024))
            for donor in origin
            for member in _members_on(row.code, _GE)
        ),
        *(
            (member, donor, range(first, last + 1))
            for donor, (first, last) in sibling_years.items()
            for member in _members_on(sibling.code, _GE)
        ),
    ]
    facts = _dac1_facts(extra=extra)
    identity = _dac1_identity(row.code, origin, window=None)

    table = provider_rule(identity, None, facts)
    assert table is not None
    option = table.option(f"provider:{sibling.scope}")
    assert option is not None
    switched = build_part_scope(
        base=identity, entries=[switch(option.option_id)], records={}, facts=facts
    )

    availability = option.availability
    assert (availability.status, availability.code, availability.years) == expected
    label = _SCOPE_LABELS[sibling.scope]
    assert option.label == (label if availability.status != "partial" else f"{label}, {window}")
    if window is None:
        assert not isinstance(switched, PartScope)
        return
    assert isinstance(switched, PartScope)
    assert _slot_ref(switched.compiled.identity, "dac1AidType") == sibling.code
    assert switched.compiled.identity.requested_window == IdentityWindowRange(
        start_year=window, end_year=window
    )


@pytest.mark.parametrize("row", PUBLISHED_PROVIDER_SCOPES, ids=lambda row: row.code)
@pytest.mark.parametrize(
    ("prefixes", "expected"),
    [
        pytest.param((_NET, _GE), _COVERED, id="both-bases"),
        pytest.param((_NET,), _partial((2016, 2017)), id="net-only"),
        pytest.param((_GE,), _partial((2018, 2019)), id="grant-equivalent-only"),
    ],
)
def test_a_dac1_sibling_option_on_the_headline_reads_each_basis_it_splices(
    row, prefixes: tuple[str, ...], expected
):
    """On the OECD headline over 2016 to 2019, the other DAC1 aid type serves the years its
    net series hold before 2018 and its grant-equivalent series hold from 2018."""
    (sibling,) = (other for other in PUBLISHED_PROVIDER_SCOPES if other is not row)
    extra = [
        (member, _USA, _ALL_YEARS)
        for prefix in prefixes
        for member in _members_on(sibling.code, prefix)
    ]
    facts = _dac1_facts(row.code, extra=extra)
    identity = _dac1_identity(row.code, (_USA,), window=(2016, 2019), headline=True)

    table = provider_rule(identity, None, facts)

    assert table is not None
    option = table.option(f"provider:{sibling.scope}")
    assert option is not None
    availability = option.availability
    assert (availability.status, availability.code, availability.years) == expected


def test_a_switch_of_basis_and_provider_together_judges_the_sibling_on_the_switched_basis():
    """The United States' total ODA holds grant-equivalent and gross series and its bilateral
    ODA gross series alone, over 2015 to 2020. On the grant-equivalent total, bilateral ODA
    is not covered. Switching to gross and to bilateral flows in one resubmit judges
    bilateral ODA on gross, so it compiles to the identity and choices the two switches
    compile to one after the other."""
    extra = [
        *(
            (member, _USA, range(2015, 2021))
            for prefix in (_GE, _GROSS)
            for member in _members_on(_DAC1_TOTAL_ODA, prefix)
        ),
        *((member, _USA, range(2015, 2021)) for member in _members_on(_DAC1_BILATERAL_ODA, _GROSS)),
    ]
    facts = _dac1_facts(extra=extra)
    base = _dac1_identity(_DAC1_TOTAL_ODA, (_USA,), window=(2015, 2020))
    to_gross, to_bilateral = switch("measure_basis:gross"), switch("provider:bilateral")

    alone = build_part_scope(base=base, entries=[to_bilateral], records={}, facts=facts)
    gross = build_part_scope(base=base, entries=[to_gross], records={}, facts=facts)
    assert isinstance(gross, PartScope)
    in_turn = build_part_scope(
        base=gross.compiled.identity, entries=[to_bilateral], records={}, facts=facts
    )
    together = build_part_scope(
        base=base, entries=[to_gross, to_bilateral], records={}, facts=facts
    )

    assert alone == Refusal(
        code="requested_option_unavailable", entry=0, cause="option_not_covered"
    )
    assert isinstance(in_turn, PartScope) and isinstance(together, PartScope)
    assert _slot_ref(together.compiled.identity, "dac1AidType") == _DAC1_BILATERAL_ODA
    assert together.compiled.spec_id == in_turn.compiled.spec_id
    (provider_together,) = (c for c in together.choices if c.axis == "provider")
    (provider_in_turn,) = (c for c in in_turn.choices if c.axis == "provider")
    assert provider_together.active == provider_in_turn.active == "provider:bilateral"
    assert provider_together.options == provider_in_turn.options


def test_two_switches_whose_options_conflict_refuse_the_conflict_whatever_each_edits():
    """The imputed table holds gross disbursements alone, so a switch to the inclusive scope
    and one to net flows, each available on the gross part, refuse as conflicting, naming
    the switch to inclusive as the blocker, and neither is judged on the identity the other
    would produce."""
    net = tuple(
        _crs_member(
            code,
            measured_property="amount",
            netting="Net",
            flow=_ODA,
            modality="CRSCooperationModalityA01",
        )
        for code in ("CRSPurposeCode120", "CRSPurposeCode130")
    )
    rows, extra = _coverage_rows(_ALL_YEARS)
    rows += [(dcid, "crs", _USA, _KEN, list(_BILATERAL_YEARS)) for dcid in net]
    extra["role_names"] |= dict.fromkeys(net, _FLOW)
    index = build_coverage_index(rows, **extra)
    facts = ScopeFacts(index=get_index(), coverage=FakeCoverageService(index).snapshot())
    entries = [switch("provider:inclusive"), switch("measure_basis:net")]

    compiled = build_part_scope(
        base=_bilateral_identity(), entries=entries, records={}, facts=facts
    )

    assert compiled == Refusal(
        code="conflicting_requested_options",
        entry=1,
        cause="imputed_gross_disbursements_only",
        blocker=0,
    )


@pytest.mark.parametrize("row", PUBLISHED_PROVIDER_SCOPES, ids=lambda row: row.code)
def test_a_partial_dac1_sibling_option_serves_the_years_the_part_holds_over_their_run(row):
    """A 2015 to 2020 total the United States holds from 2015 to 2017, whose other DAC1 aid
    type it holds from 2016 to 2020: the option serves 2016 and 2017, the years both hold,
    and a switch reads that run."""
    (sibling,) = (other for other in PUBLISHED_PROVIDER_SCOPES if other is not row)
    extra = [
        *((member, _USA, range(2015, 2018)) for member in _members_on(row.code, _GE)),
        *((member, _USA, range(2016, 2021)) for member in _members_on(sibling.code, _GE)),
    ]
    facts = _dac1_facts(extra=extra)
    identity = _dac1_identity(row.code, (_USA,), window=(2015, 2020))

    table = provider_rule(identity, None, facts)
    assert table is not None
    option = table.option(f"provider:{sibling.scope}")
    assert option is not None
    switched = build_part_scope(
        base=identity, entries=[switch(option.option_id)], records={}, facts=facts
    )

    availability = option.availability
    assert (availability.status, availability.code, availability.years) == _partial((2016, 2017))
    assert option.label == f"{_SCOPE_LABELS[sibling.scope]}, 2016 to 2017"
    assert isinstance(switched, PartScope)
    assert _slot_ref(switched.compiled.identity, "dac1AidType") == sibling.code
    assert switched.compiled.identity.requested_window == IdentityWindowRange(
        start_year=2016, end_year=2017
    )


@pytest.mark.parametrize("row", PUBLISHED_PROVIDER_SCOPES, ids=lambda row: row.code)
@pytest.mark.parametrize(
    ("own", "sibling_years", "expected"),
    [
        # A donor holding no year of the part's own total places no demand on the other.
        pytest.param(
            {_USA: (2015, 2020)},
            {_USA: (2015, 2020), _DEU: (2019, 2019)},
            _COVERED,
            id="a-donor-holding-no-year",
        ),
        # Each donor holds the other total over its own run of the part's years.
        pytest.param(
            {_USA: (2015, 2017), _DEU: (2018, 2020)},
            {_USA: (2015, 2017), _DEU: (2018, 2020)},
            _COVERED,
            id="different-runs-each-donor-holds-its-own",
        ),
        pytest.param(
            {_USA: (2015, 2020), _DEU: (2015, 2020)},
            {_USA: (2015, 2020), _DEU: (2018, 2020)},
            _partial((2018, 2020)),
            id="a-donor-holding-part-of-its-run",
        ),
        # With no year of the part's own total held, the other total is available only
        # where every donor holds it in the window.
        pytest.param(
            {},
            {_USA: (2015, 2020), _DEU: (2019, 2019)},
            _COVERED,
            id="no-own-year-every-donor-holds-the-other",
        ),
        pytest.param({}, {_USA: (2015, 2020)}, _NOT_COVERED, id="no-own-year-one-donor-holds"),
        pytest.param({}, {}, _NOT_COVERED, id="no-own-year-no-donor-holds"),
    ],
)
def test_a_dac1_sibling_option_is_measured_against_the_years_any_donor_of_the_pool_holds(
    row, own: dict[str, tuple[int, int]], sibling_years: dict[str, tuple[int, int]], expected
):
    """A pool's answer holds the years any of its donors holds the part's own total in. A
    year is covered when every donor holding the part's total in it holds the other total
    in it, so a donor holding neither places no demand, and a donor holding the other total
    over part of its own run leaves the option partial over the years covered. A partial
    option states its years in its label."""
    (sibling,) = (other for other in PUBLISHED_PROVIDER_SCOPES if other is not row)
    extra = [
        *(
            (member, donor, range(first, last + 1))
            for donor, (first, last) in own.items()
            for member in _members_on(row.code, _GE)
        ),
        *(
            (member, donor, range(first, last + 1))
            for donor, (first, last) in sibling_years.items()
            for member in _members_on(sibling.code, _GE)
        ),
    ]
    facts = _dac1_facts(extra=extra)
    identity = _dac1_identity(row.code, (_USA, _DEU), window=(2015, 2020))

    table = provider_rule(identity, None, facts)

    assert table is not None
    option = table.option(f"provider:{sibling.scope}")
    assert option is not None
    availability = option.availability
    assert (availability.status, availability.code, availability.years) == expected
    label = _SCOPE_LABELS[sibling.scope]
    if availability.years is None:
        assert option.label == label
    else:
        (first, last) = availability.years[-1]
        years = str(first) if first == last else f"{first} to {last}"
        assert option.label == f"{label}, {years}"


@pytest.mark.parametrize("row", PUBLISHED_PROVIDER_SCOPES, ids=lambda row: row.code)
def test_a_latest_dac1_sibling_option_no_donor_demands_is_unavailable(monkeypatch, row):
    """Where the series count elects a year for the part's latest read and the donor scan
    finds no donor holding its own total in it, no donor demands the year, and the other
    DAC1 aid type is unavailable with `option_not_covered`."""
    (sibling,) = (other for other in PUBLISHED_PROVIDER_SCOPES if other is not row)
    facts = _dac1_facts(row.code, sibling.code)
    monkeypatch.setattr(provider, "donor_years", lambda table, *, facts: {})

    table = provider_rule(_dac1_identity(row.code, (_USA,), window=None), None, facts)

    assert table is not None
    option = table.option(f"provider:{sibling.scope}")
    assert option is not None
    availability = option.availability
    assert (availability.status, availability.code, availability.years) == _NOT_COVERED


_GBR = "country/GBR"


@pytest.mark.parametrize(
    ("window", "kept"),
    [
        pytest.param((2010, 2019), False, id="member-throughout"),
        pytest.param((2015, 2022), False, id="member-in-part"),
        pytest.param((2021, 2024), True, id="after-leaving"),
    ],
)
def test_a_pool_of_the_eu_institutions_reads_eu_membership_over_its_window(
    window: tuple[int, int], kept: bool
):
    """The United Kingdom funded the EU institutions until it left the EU in 2020, so a
    pool of the two counts its core contributions twice in total ODA over any window
    holding a year of its membership, and has no provider record there. A window after it
    left keeps the record."""
    extra = [
        (member, _GBR, _ALL_YEARS)
        for code in (_DAC1_TOTAL_ODA, _DAC1_BILATERAL_ODA)
        for member in _DAC1_MEMBERS_BY_CODE[code]
    ]
    facts = _dac1_facts(_DAC1_TOTAL_ODA, _DAC1_BILATERAL_ODA, extra=extra)
    identity = _dac1_identity(_DAC1_TOTAL_ODA, (_GBR, _EU_INSTITUTIONS), window=window)

    record = settle_provider(identity, ScopeState(), facts)

    assert (record is not None) == kept


def test_a_convention_the_pass_cannot_serve_leaves_the_part_on_the_option_it_holds():
    """Where a matched convention's option is not served, the record holds the part's own
    option before the bilateral one, so it names only a total the answer reads. DAC1's total
    ODA keeps the contributions scope though bilateral ODA is available."""
    facts = _dac1_facts(_DAC1_TOTAL_ODA, _DAC1_BILATERAL_ODA)
    table = provider_rule(_dac1_identity(_DAC1_TOTAL_ODA, (_USA,)), None, facts)
    (convention,) = (row for row in PROVIDER_CONVENTIONS if row.scope == "inclusive")

    assert table is not None
    bilateral = table.option("provider:bilateral")
    assert bilateral is not None and bilateral.availability.status == "available"
    record = _settled(table, ScopeState(), convention)

    assert (record.active, record.rule) == ("provider:contributions", "contribution_total")


def test_only_a_published_total_offers_a_provider_option_binding_another_code():
    """A provider option binding a slot reads another published total, so a table of a part
    the CRS serves refuses one, and a published total's table holds one only as published."""
    crs = provider_rule(_bilateral_identity(), None, _facts())
    dac1 = provider_rule(
        _dac1_identity(_DAC1_TOTAL_ODA, (_USA,)),
        None,
        _dac1_facts(_DAC1_TOTAL_ODA, _DAC1_BILATERAL_ODA),
    )
    assert crs is not None and dac1 is not None
    slot_options = tuple(o for o in dac1.options if isinstance(o.binding, SlotBinding))
    assert slot_options

    with pytest.raises(ScopeTableError):
        dataclasses.replace(crs, options=(*crs.options, *slot_options))
    with pytest.raises(ScopeTableError):
        dataclasses.replace(dac1, published=False)


def test_a_stated_provider_scope_on_the_multilaterals_dac1_total_is_reported_absent(monkeypatch):
    """The multilaterals' DAC1 total has no provider axis, so a typed "bilateral" on it is
    reported as not applied with `axis_absent`."""
    members = (*_DAC1_TOTAL_MEMBERS, *_DAC1_BILATERAL_MEMBERS)
    patch_coverage(
        monkeypatch,
        [(dcid, "dac1", _DAC_MULTILATERALS, ABSENT_ENTITY, list(_ALL_YEARS)) for dcid in members],
        role_names=dict.fromkeys(members, RoleNames(donor="financingProvider", recipient=None)),
        facet_metadata={"dac1": FacetMetadata(provenance_id=_DAC1, unit="USD")},
    )
    graph = _graph("aid", _DAC1_TOTAL_MEMBERS)
    label = "multilaterals (DAC)"
    graph = FakeGraph(
        nodes={**graph._nodes, _DAC_MULTILATERALS: {"label": label, "type": "Organization"}},
        detect=graph._detect,
        resolve={label: _DAC_MULTILATERALS},
    )
    response = offline_resolve(
        ResolveRequest(
            input=ParsedInput(
                parts=[
                    ParsedPart(
                        variable_text="aid",
                        entities=[ParsedEntity(text=label, role_text="from")],
                        time_text="2019",
                    )
                ],
                user_message="bilateral aid in 2019",
            )
        ),
        graph=graph,
        llm=_PickingLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    identity = part.spec.definition.identity
    assert identity.entities["origin"] == [_DAC_MULTILATERALS]
    assert _slot_ref(identity, "dac1AidType") == _DAC1_TOTAL_ODA
    scope = _scope(part.spec)
    assert all(choice.axis != "provider" for choice in scope.choices)
    (dropped,) = (d for d in scope.dropped_entries if d.entry.axis == "provider")
    assert (dropped.entry.option_id, dropped.code, dropped.cause) == (
        "provider:bilateral",
        "stated_choice_not_applied",
        "axis_absent",
    )


@pytest.mark.parametrize(
    "origin",
    [pytest.param((), id="no-donor"), pytest.param((_DAC_MULTILATERALS,), id="multilaterals")],
)
def test_a_stated_provider_scope_on_a_dac1_total_with_no_provider_axis_is_dropped_absent(
    origin: tuple[str, ...],
):
    """A DAC1 total naming no donor, or the multilaterals' total, has no provider axis, so
    the scope reports a stated "bilateral" read for it as not applied with `axis_absent`."""
    facts = _dac1_facts(_DAC1_TOTAL_ODA, _DAC1_BILATERAL_ODA)
    identity = _dac1_identity(_DAC1_TOTAL_ODA, origin)
    entry = explicit("provider:bilateral")

    built = build_part_scope(
        base=identity, entries=[], records={}, facts=facts, read=(entry,), refuse=False
    )

    assert settle_provider(identity, ScopeState(entries=(entry,)), facts) is None
    assert isinstance(built, PartScope)
    assert all(choice.axis != "provider" for choice in built.choices)
    (dropped,) = built.dropped_entries
    assert (dropped.entry, dropped.code, dropped.cause) == (
        entry,
        "stated_choice_not_applied",
        "axis_absent",
    )


def test_a_carried_inclusive_scope_on_dac1_bilateral_oda_is_dropped_with_the_reason():
    """DAC1's bilateral ODA is a total of another table than the CRS, so a carried
    inclusive scope is not offered there. The part keeps bilateral ODA under its own rule and
    reports the entry with the reason."""
    facts = ScopeFacts(index=get_index(), coverage=empty_coverage_snapshot())
    identity = _dac1_identity(_DAC1_BILATERAL_ODA, (_USA,))
    entry = carried("provider:inclusive")

    record = settle_provider(identity, ScopeState(entries=(entry,)), facts)
    assert record is not None
    built = build_part_scope(
        base=identity,
        entries=[],
        records={"provider": record},
        facts=facts,
        read=(entry,),
        refuse=False,
    )

    assert (record.active, record.provenance, record.rule) == (
        "provider:bilateral",
        "default",
        "bilateral_total",
    )
    assert isinstance(built, PartScope)
    assert built.compiled.spec_id == identity_spec_id(identity)
    (dropped,) = built.dropped_entries
    assert (dropped.entry, dropped.code, dropped.cause) == (
        entry,
        "carried_choice_not_applied",
        "provider_not_offered",
    )


def _scope_of(option_id: str) -> str:
    """The provider scope option `option_id` holds."""
    return option_id.removeprefix("provider:")


def _held_choice(identity: BaseIdentity, facts: ScopeFacts) -> str:
    """The provider option a resubmit of `identity` with no entries answers."""
    built = build_part_scope(base=identity, entries=[], records={}, facts=facts)
    assert isinstance(built, PartScope)
    (choice,) = (c for c in built.choices if c.axis == "provider")
    return choice.active


def _assert_holds_available(identity: BaseIdentity, facts: ScopeFacts) -> None:
    """A fresh search's record of `identity` and a resubmit of it each hold an option the
    table marks available, where the axis applies. Each record's rule, and the rule each
    option names by default, describes that option's own scope."""
    table = provider_rule(identity, None, facts)
    record = settle_provider(identity, ScopeState(), facts)
    total = settle_provider(identity, _total(), facts)

    assert (table is None) == (record is None) == (total is None)
    if table is None or record is None or total is None:
        return
    for active in (record.active, total.active, _held_choice(identity, facts)):
        option = table.option(active)
        assert option is not None and option.availability.status == "available", active
    for held in (record, total):
        assert _RULE_SCOPES[held.rule] == _scope_of(held.active), held
    for option in table.options:
        assert _RULE_SCOPES[table.rule_of(option.option_id)] == _scope_of(option.option_id)


def test_every_declared_provider_rule_names_one_scope():
    """Each provider rule a table or record can name belongs to one declared row, so the
    scope it describes is defined, and each scope has its own default rule."""
    rules = [
        *(row.rule for row in PROVIDER_SCOPE_RULES),
        *(row.rule for row in PROVIDER_CONVENTIONS),
        *(row.rule for row in PUBLISHED_PROVIDER_SCOPES),
    ]
    assert len(rules) == len(set(rules))
    scopes = [row.scope for row in PROVIDER_SCOPE_RULES]
    assert len(scopes) == len(set(scopes))


@pytest.mark.parametrize("row", PUBLISHED_PROVIDER_SCOPES, ids=lambda row: row.code)
@pytest.mark.parametrize("covered", [True, False], ids=["both-covered", "own-covered"])
@pytest.mark.parametrize(
    "origin",
    [
        (_USA,),
        (_DAC_COUNTRIES,),
        (_EU_INSTITUTIONS,),
        (_USA, _DEU),
        (_DEU, _EU_INSTITUTIONS),
        (_USA, _DAC_MULTILATERALS),
        (_DAC_EU_MEMBERS,),
    ],
    ids=lambda origin: "+".join(origin),
)
def test_a_dac1_record_never_holds_an_option_it_marks_unavailable(row, covered, origin):
    """For every declared DAC1 row and every kind of provider set, with the other aid type's
    series or without them, the record's active option, fresh or carried to a resubmit, is
    available on the table it reads."""
    codes = (_DAC1_TOTAL_ODA, _DAC1_BILATERAL_ODA) if covered else (row.code,)
    _assert_holds_available(_dac1_identity(row.code, origin), _dac1_facts(*codes))


@pytest.mark.parametrize(
    ("inclusive", "latest", "rows"),
    [
        pytest.param(False, False, None, id="bilateral-covered"),
        pytest.param(True, False, None, id="inclusive-covered"),
        pytest.param(True, True, None, id="inclusive-latest-covered"),
        # The tables cover no common year, so a latest read elects none.
        pytest.param(
            True,
            True,
            [
                *((dcid, "crs", _USA, _KEN, [2024]) for dcid in _CRS_MEMBERS),
                *((dcid, "imputed", _USA, _KEN, [2010]) for dcid in _IMPUTED_MEMBERS),
            ],
            id="inclusive-latest-no-common-year",
        ),
        pytest.param(
            True, False, _rows_for((_USA,), imputed=()), id="inclusive-provider-not-carried"
        ),
    ],
)
def test_a_crs_record_never_holds_an_option_it_marks_unavailable(inclusive, latest, rows):
    """On a part the CRS serves, bilateral or bound to the inclusive composite, the record's
    active option, fresh or carried to a resubmit, is available on the table it reads, even
    where the composite the part holds is not."""
    identity = _bilateral_identity(latest=latest)
    if inclusive:
        identity = apply_binding(identity, ProviderBinding(scope="inclusive"), [])
    _assert_holds_available(identity, _facts(rows))


def test_a_crs_record_the_answer_cannot_serve_names_the_rule_of_the_option_it_holds():
    """A record holding the inclusive scope where its option is partial leaves the part on
    the bilateral scope it holds, under the bilateral scope's own rule."""
    facts = _facts(_coverage_rows(_FROM_2011)[0])
    identity = _bilateral_identity(origin=(_USA,)).model_copy(
        update={"requested_window": IdentityWindowRange(start_year=2005, end_year=2024)}
    )
    record = AxisRecord(
        axis="provider", rule="named_donor_total", active="provider:inclusive", provenance="default"
    )

    built = build_part_scope(base=identity, entries=[], records={"provider": record}, facts=facts)

    assert isinstance(built, PartScope)
    (choice,) = (c for c in built.choices if c.axis == "provider")
    assert (choice.active, choice.rule) == ("provider:bilateral", "bilateral_flows")
    assert (built.compiled.identity.facets or {}).get("provenance") is None


def test_a_record_of_the_bilateral_scope_strips_the_composite_the_part_holds():
    """A part bound to the inclusive composite over the latest year, whose tables hold no
    common year, is recorded on the bilateral scope, and its answer reads the bilateral
    flows alone over their own latest year."""
    rows = [
        *((dcid, "crs", _USA, _KEN, [2024]) for dcid in _CRS_MEMBERS),
        *((dcid, "imputed", _USA, _KEN, [2010]) for dcid in _IMPUTED_MEMBERS),
    ]
    facts = _facts(rows)
    bilateral = _bilateral_identity(latest=True)
    identity = apply_binding(bilateral, ProviderBinding(scope="inclusive"), [])

    record = settle_provider(identity, ScopeState(), facts)
    assert record is not None
    built = build_part_scope(base=identity, entries=[], records={"provider": record}, facts=facts)

    assert (record.active, record.rule) == ("provider:bilateral", "bilateral_flows")
    assert isinstance(built, PartScope)
    (choice,) = (c for c in built.choices if c.axis == "provider")
    assert (choice.active, choice.rule) == ("provider:bilateral", "bilateral_flows")
    assert built.compiled.identity == bilateral
    assert built.base.identity == bilateral
