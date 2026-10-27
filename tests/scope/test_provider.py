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

from collections.abc import Sequence

import pytest

from code_index.composites import component_identities
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
    Resolution,
    ResolveRequest,
    ResubmitIdentityEntry,
    ScopeChoice,
    Spec,
)
from search.scope.edits import apply_binding
from search.scope.provider import provider_rule, settle_provider
from search.scope.state import ScopeState
from search.scope.tables import ScopeFacts
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
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
            "named_donor_total",
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
            "named_donor_total",
            "available",
            id="bilateral-typed",
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
        assert getattr(entry.origin, "text", None) == "bilateral"
    else:
        assert entry is None


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


def _bilateral_identity(**slots: str) -> BaseIdentity:
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
        "entities": {"origin": [_USA], "destination": [_KEN]},
        "requested_window": {"kind": "range", "start_year": 2015, "end_year": 2020},
    }
    return BaseIdentity.model_validate(document)


def _facts() -> ScopeFacts:
    rows, extra = _coverage_rows(_ALL_YEARS)
    index = build_coverage_index(rows, **extra)
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
        "named_donor_total",
    )


def test_the_bilateral_and_inclusive_edits_are_inverses():
    base = _bilateral_identity()
    inclusive = apply_binding(base, ProviderBinding(scope="inclusive"), [])
    back = apply_binding(inclusive, ProviderBinding(scope="bilateral"), [])

    assert (inclusive.facets or {})["provenance"] == IdentityBindingValue(ref=_INCLUSIVE_SOURCE)
    assert identity_spec_id(back) == identity_spec_id(base)
    windowed = apply_binding(base, ProviderBinding(scope="inclusive", window=(2011, 2024)), [])
    assert windowed.requested_window == IdentityWindowRange(start_year=2011, end_year=2024)
