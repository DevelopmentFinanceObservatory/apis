"""The sector axis: a sector concept the request names binds its declared grouping and states a
sector choice, a code the request names stays a code binding and is recorded as explicit, and
a switch through dataset_resubmit moves the binding and back.

The parts resolve through the parsed door over a fixture graph holding one CRS gross
disbursements member under education (110), health (120), population and reproductive
health (130), basic health (122) and emergency response (720), each with a series from France
to Kenya. Reconstruction relief (730) and disaster prevention (740) hold none, so the
humanitarian options binding them alone are unavailable.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from code_index.scope_conventions import SECTOR_GROUPINGS
from code_index.shared import get_index
from search.engine.bind import _UniversalBindOutput
from search.engine.qualifier_match import QualifierMatch
from search.engine.quantity_read import QuantityOutput
from search.engine.stated_words_read import StatedWordsOutput
from search.models import (
    BaseIdentity,
    ChoiceOrigin,
    DatasetResubmitInput,
    DefinitePart,
    Entry,
    EntryProvenance,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    PartScope,
    Resolution,
    ResolveRequest,
    ResubmitIdentityEntry,
    Spec,
    TextOrigin,
)
from search.scope.build import build_part_scope
from search.scope.sector import read_sector, sector_rule
from search.scope.state import AxisRecord, ScopeState
from search.scope.tables import ScopeFacts
from series_coverage.schema import FacetMetadata, RoleNames
from tests.code_index.aid_identity import CRS_PROVENANCE
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.engine._quantity import NO_QUANTITY, NO_RATING
from tests.fixtures import FakeGraph

_KEN = "country/KEN"
_FRA = "country/FRA"
_PURPOSE = "crsPurpose"
_EDUCATION_MEMBER = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode11330_Nominal"
)
_HEALTH = ("CRSPurposeCode120", "CRSPurposeCode130")
_HUMANITARIAN = ("CRSPurposeCode720", "CRSPurposeCode730", "CRSPurposeCode740")


def _member_under(code: str) -> str:
    """A member on the education member's five-tuple, flow, flow type and netting basis whose
    purpose is `code` or a code below it."""
    index = get_index()
    template = index.statvar_by_dcid(_EDUCATION_MEMBER)
    assert template is not None
    fixed = ("crsTypeOfFlow", "financialFlowType", "nettingBasis")
    return next(
        row.dcid
        for row in index.statvars
        if (row.population_type, row.measured_property, row.stat_type, row.measurement_qualifier)
        == (
            template.population_type,
            template.measured_property,
            template.stat_type,
            template.measurement_qualifier,
        )
        and all(
            row.constraint_properties.get(p) == template.constraint_properties[p] for p in fixed
        )
        and row.constraint_properties.get(_PURPOSE) in index.descendants(code)
    )


_MEMBERS = (
    _EDUCATION_MEMBER,
    *(
        _member_under(code)
        for code in ("CRSPurposeCode120", "CRSPurposeCode130", "CRSPurposeCode122")
    ),
    _member_under("CRSPurposeCode720"),
)


def _arcs(dcid: str) -> dict:
    row = get_index().statvar_by_dcid(dcid)
    assert row is not None
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": dcid}]},
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "measurementQualifier": {"nodes": [{"dcid": row.measurement_qualifier}]},
        "observationProperties": {"nodes": [{"dcid": p} for p in row.observation_properties]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }


def _graph(variable: str) -> FakeGraph:
    """Kenya and France, the CRS, and every member, which recall returns for `variable`
    with the education member first."""
    return FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            _FRA: {"label": "France", "type": "Country"},
            CRS_PROVENANCE: {"label": "OECD Creditor Reporting System (CRS)"},
            **{
                dcid: {"label": dcid, "type": "StatisticalVariable", "arcs": _arcs(dcid)}
                for dcid in _MEMBERS
            },
        },
        detect={variable: {"svs": [_EDUCATION_MEMBER], "cosine_scores": [0.9]}},
        resolve={"Kenya": _KEN},
    )


class _ElectingLLM:
    """Elects the first shape with no bindings of its own, reads no quantity and rates no
    word, so every sector binding is the text's."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        if schema is StatedWordsOutput:
            return NO_RATING
        assert issubclass(schema, _UniversalBindOutput), schema.__name__
        return elected_reply(0), None


@pytest.fixture(autouse=True)
def _coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [(dcid, "crs", _FRA, _KEN, [2019, 2020, 2021]) for dcid in _MEMBERS],
        role_names=dict.fromkeys(_MEMBERS, names),
        facet_metadata={"crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD")},
    )


def _resolve(request: ResolveRequest, variable: str) -> Spec:
    response = offline_resolve(
        request,
        graph=_graph(variable),
        llm=_ElectingLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    return part.spec


def _ask(variable: str) -> Spec:
    return _resolve(
        ResolveRequest(
            input=ParsedInput(
                parts=[
                    ParsedPart(
                        variable_text=variable,
                        entities=[ParsedEntity(text="Kenya", role_text="to")],
                    )
                ]
            )
        ),
        variable,
    )


def _purpose(spec: Spec) -> tuple[list[str], str]:
    """The refs the spec's purpose slot binds and their provenance."""
    (purpose,) = (
        f
        for f in spec.resolved.trace.slot_filters
        if f.key.property is not None and f.key.property.ref == _PURPOSE
    )
    return [ref.ref for ref in purpose.refs], purpose.provenance


def _choice(spec: Spec) -> tuple | None:
    """The spec's sector choice: its rule, provenance, active option and every option with
    its status; None when the spec states no scope."""
    scope = spec.resolved.scope_choices
    if scope is None:
        return None
    (choice,) = scope.choices
    assert choice.axis == "sector"
    return (
        choice.rule,
        choice.provenance,
        choice.active,
        [(o.option_id, o.availability.status) for o in choice.options],
    )


_HEALTH_OPTIONS = [
    ("sector:120+130", "available"),
    ("sector:120", "available"),
    ("sector:130", "available"),
]


class _Row(NamedTuple):
    refs: list[str]
    provenance: str
    choice: tuple | None
    entry_text: str | None


_MATRIX = {
    "health": _Row(
        list(_HEALTH),
        "declared_grouping",
        ("sector_grouping", "default", "sector:120+130", _HEALTH_OPTIONS),
        None,
    ),
    "health aid": _Row(
        list(_HEALTH),
        "declared_grouping",
        ("sector_grouping", "default", "sector:120+130", _HEALTH_OPTIONS),
        None,
    ),
    "basic health": _Row(
        ["CRSPurposeCode122"],
        "text_recognized",
        ("sector_code", "explicit", "sector:122", [("sector:122", "available")]),
        "basic health",
    ),
    "reproductive health": _Row(
        ["CRSPurposeCode130"],
        "text_recognized",
        ("sector_grouping", "explicit", "sector:130", _HEALTH_OPTIONS),
        "reproductive health",
    ),
    "purpose code 120": _Row(
        ["CRSPurposeCode120"],
        "text_recognized",
        ("sector_grouping", "explicit", "sector:120", _HEALTH_OPTIONS),
        "120",
    ),
    # The matcher reads a bare code only after a word of its own code system's name, and
    # "CRS" names every CRS system, so neither phrase binds a purpose or states a choice.
    "CRS 120": _Row([], "binder", None, None),
    "code 120": _Row([], "binder", None, None),
    "humanitarian aid": _Row(
        list(_HUMANITARIAN),
        "declared_grouping",
        (
            "sector_grouping",
            "default",
            "sector:720+730+740",
            [
                ("sector:720+730+740", "available"),
                ("sector:720", "available"),
                ("sector:730", "unavailable"),
                ("sector:740", "unavailable"),
            ],
        ),
        None,
    ),
}


@pytest.mark.parametrize("variable", list(_MATRIX))
def test_a_sector_phrase_binds_and_states_its_sector_choice(variable: str, search_settings):
    row = _MATRIX[variable]
    if variable == "purpose code 120":
        # The parsed door leaves "purpose code", the words naming the code system, uncovered,
        # so the part refuses on them unless uncovered words are only logged.
        search_settings(qualifier_coverage="log")

    spec = _ask(variable)

    assert _purpose(spec) == (row.refs, row.provenance)
    assert _choice(spec) == row.choice
    scope = spec.resolved.scope_choices
    texts = (
        [e.origin.text for e in scope.entries if isinstance(e.origin, TextOrigin)]
        if scope is not None
        else []
    )
    assert texts == ([] if row.entry_text is None else [row.entry_text])
    if scope is not None:
        assert scope.compiled.spec_id == scope.base.spec_id == spec.spec_id


def test_an_unavailable_option_states_why():
    scope = _ask("humanitarian aid").resolved.scope_choices
    assert scope is not None

    unavailable = [o for o in scope.choices[0].options if o.availability.status == "unavailable"]

    assert {o.availability.code for o in unavailable} == {"option_not_covered"}


def _switch(identity: BaseIdentity, option_id: str, choice_id: str) -> Spec:
    entry = Entry(
        axis="sector",
        option_id=option_id,
        provenance="explicit",
        origin=ChoiceOrigin(choice_id=choice_id),
    )
    return _resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(
                        variable_text="health aid", identity=identity, scope_entries=[entry]
                    )
                ]
            )
        ),
        "health aid",
    )


def test_a_switch_to_one_code_and_back_returns_the_first_spec_id():
    first = _ask("health aid")
    scope = first.resolved.scope_choices
    assert scope is not None
    base, choice_id = scope.base.identity, scope.choices[0].id

    narrowed = _switch(base, "sector:120", choice_id)
    restored = _switch(base, "sector:120+130", choice_id)

    assert narrowed.spec_id != first.spec_id
    assert _purpose(narrowed)[0] == ["CRSPurposeCode120"]
    assert _choice(narrowed) == ("sector_grouping", "explicit", "sector:120", _HEALTH_OPTIONS)
    narrowed_scope = narrowed.resolved.scope_choices
    assert narrowed_scope is not None
    assert narrowed_scope.base.spec_id == first.spec_id
    assert restored.spec_id == first.spec_id
    assert _choice(restored) == ("sector_grouping", "explicit", "sector:120+130", _HEALTH_OPTIONS)


# ---------------------------------------------------------------------------
# The binder's reading and the rule, on their own.
# ---------------------------------------------------------------------------


def _health_match(phrase: str = "health", code: str = "CRSPurposeCode120") -> QualifierMatch:
    return QualifierMatch(
        code_system=_PURPOSE,
        code_dcid=code,
        label=phrase,
        source="label",
        matched_text=phrase,
        start=0,
        end=len(phrase),
    )


def _entry(option_id: str, provenance: EntryProvenance, concept: str | None) -> Entry:
    return Entry(
        axis="sector",
        option_id=option_id,
        provenance=provenance,
        concept=concept,
        origin=ChoiceOrigin(choice_id="earlier:sector"),
    )


@pytest.mark.parametrize(
    ("entries", "codes", "provenance"),
    [
        pytest.param((), _HEALTH, "default", id="no-entry"),
        pytest.param(
            (_entry("sector:120", "carried", "health"),),
            ("CRSPurposeCode120",),
            "carried",
            id="carried",
        ),
        pytest.param(
            (_entry("sector:130", "carried", "health"), _entry("sector:120", "explicit", None)),
            ("CRSPurposeCode120",),
            "explicit",
            id="explicit-over-carried",
        ),
        pytest.param(
            (_entry("sector:120", "carried", "education"),),
            _HEALTH,
            "default",
            id="another-concepts-entry",
        ),
        pytest.param(
            (_entry("sector:110", "carried", "health"),),
            _HEALTH,
            "default",
            id="an-option-the-grouping-does-not-offer",
        ),
    ],
)
def test_an_entry_for_the_concept_wins_over_the_grouping(entries, codes, provenance):
    reading = read_sector(
        codes=["CRSPurposeCode120"],
        matches=[_health_match()],
        variable="health aid",
        scope=ScopeState(entries=entries),
    )

    assert (reading.codes, reading.grouped) == (codes, True)
    assert (reading.record.provenance, reading.record.concept) == (provenance, "health")


def test_a_concept_beside_another_code_binds_the_grouping_beside_it():
    education = QualifierMatch(
        code_system=_PURPOSE,
        code_dcid="CRSPurposeCode110",
        label="education",
        source="label",
        matched_text="education",
        start=11,
        end=20,
    )

    reading = read_sector(
        codes=["CRSPurposeCode120", "CRSPurposeCode110"],
        matches=[_health_match(), education],
        variable="health and education",
        scope=ScopeState(),
    )

    assert reading.codes == ("CRSPurposeCode110", *_HEALTH)
    assert (reading.record.active, reading.record.provenance) == ("sector:110+120+130", "default")


def test_every_declared_grouping_is_read_from_a_word_the_matcher_reads_on_its_system():
    from search.engine.qualifier_match import match_qualifiers

    for grouping in SECTOR_GROUPINGS:
        for word in grouping.words:
            matches = [m for m in match_qualifiers(word, get_index()) if m.phrase == word]
            if not matches:
                # A word whose span ties across systems is read on the elected shape's own
                # vocabulary: "humanitarian aid" names CRS 700 and a DAC2a aid type.
                continue
            assert {m.code_system for m in matches} == {grouping.code_system}


def _facts() -> ScopeFacts:
    return ScopeFacts(index=get_index(), coverage=empty_coverage_snapshot())


def _base(binding: dict | None) -> BaseIdentity:
    slots = [{"property": "nettingBasis", "binding": {"state": "value", "ref": "Gross"}}]
    if binding is not None:
        slots.append({"property": _PURPOSE, "binding": binding})
    return BaseIdentity.model_validate(
        {
            "identity_version": "spec1",
            "shape": {
                "kind": "statvar_schema",
                "population_type": "FinancialFlow",
                "measured_property": "amount",
                "stat_type": "measuredValue",
                "measurement_qualifier": "Nominal",
            },
            "slots": slots,
            "entities": {"destination": [_KEN]},
            "requested_window": {"kind": "all"},
        }
    )


@pytest.mark.parametrize(
    ("binding", "rule", "held", "options"),
    [
        pytest.param(
            {"state": "set", "refs": list(_HEALTH)},
            "sector_grouping",
            "sector:120+130",
            ["sector:120+130", "sector:120", "sector:130"],
            id="grouping",
        ),
        pytest.param(
            {"state": "value", "ref": "CRSPurposeCode130"},
            "sector_grouping",
            "sector:130",
            ["sector:120+130", "sector:120", "sector:130"],
            id="one-code-of-a-grouping",
        ),
        pytest.param(
            {"state": "value", "ref": "CRSPurposeCode122"},
            "sector_code",
            "sector:122",
            ["sector:122"],
            id="a-code-no-grouping-holds",
        ),
        pytest.param({"state": "unbound"}, None, None, None, id="unbound"),
        pytest.param(None, None, None, None, id="no-sector-slot"),
    ],
)
def test_the_rule_reads_the_base_alone_and_prefers_what_it_holds(binding, rule, held, options):
    base = _base(binding)
    record = AxisRecord(axis="sector", rule="any", active="sector:1", provenance="default")

    with_record, without = sector_rule(base, record, _facts()), sector_rule(base, None, _facts())

    assert with_record == without
    if rule is None:
        assert without is None
        return
    assert without is not None
    assert (without.rule, without.held, without.preference[0]) == (rule, held, held)
    assert [o.option_id for o in without.options] == options
    statuses = {o.option_id: o.availability.status for o in without.options}
    # An empty coverage index covers nothing but the option the base resolved.
    assert statuses == {o: "available" if o == held else "unavailable" for o in options}


def test_a_record_entry_joins_the_entries_only_while_the_base_holds_its_option():
    entry = Entry(
        axis="sector",
        option_id="sector:122",
        provenance="explicit",
        origin=TextOrigin(text="basic health", start=0, end=12),
    )
    record = AxisRecord(
        axis="sector", rule="sector_code", active="sector:122", provenance="explicit", entry=entry
    )

    held = build_part_scope(
        base=_base({"state": "value", "ref": "CRSPurposeCode122"}),
        entries=[],
        records={"sector": record},
        facts=_facts(),
    )
    moved = build_part_scope(
        base=_base({"state": "value", "ref": "CRSPurposeCode110"}),
        entries=[],
        records={"sector": record},
        facts=_facts(),
    )

    assert isinstance(held, PartScope)
    assert isinstance(moved, PartScope)
    assert [(c.active, c.provenance) for c in held.choices] == [("sector:122", "explicit")]
    assert held.entries == [entry]
    assert [(c.active, c.provenance) for c in moved.choices] == [("sector:110", "default")]
    assert moved.entries == []
