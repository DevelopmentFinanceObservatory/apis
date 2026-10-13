"""The case format: eight hand-converted cases, then each rule against a minimal case."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import get_args

import pytest
import yaml
from pydantic import ValidationError

from search.eval.cases import (
    CHECK_NAMES,
    CandidatesExpect,
    Case,
    CorpusError,
    Family,
    Gl,
    NoDataExpect,
    StatusKey,
    asserted_checks,
    corpus_schema,
    expectation_hash,
    ineligibility,
    load_case_files,
    load_cases,
    load_families,
    unconfirmed_paths,
    validate_case,
)
from search.models import StatusLiteral

FAMILIES = {
    "crs.yaml": """
crs.disbursement_gross:
  origin: dfo
  dataset: provenance/OECDCreditorReportingSystem
  shape: {population_type: FinancialFlow, measured_property: amount, stat_type: measuredValue, measurement_qualifier: Nominal, measurement_denominator: null}
  fixed: {financialFlowType: Disbursement, nettingBasis: Gross}
crs.commitment:
  origin: dfo
  dataset: provenance/OECDCreditorReportingSystem
  shape: {population_type: FinancialObligation, measured_property: amount, stat_type: measuredValue, measurement_qualifier: Nominal, measurement_denominator: null}
  fixed: {financialObligationType: Commitment}
""",
    "un.yaml": """
un.co2_fuel_combustion:
  origin: un_system
  dataset: {gl: UN System provenance dcid for SDG indicator EN_ATM_CO2}
  stat_vars: [undata/sdg/EN_ATM_CO2]
un.co2_manufacturing:
  origin: un_system
  dataset: {gl: UN System provenance dcid for UNIDO indicator SDG9_EN_ATM_CO2MAN}
  stat_vars: [undata/unido/SDG9_EN_ATM_CO2MAN]
""",
}

EXAMPLES = """
- id: dq-crs-001
  query: aid from Germany to Kenya
  reuse: panel/crs-001
  why: Aid with no netting word reads as gross ODA disbursements from Germany to Kenya.
  tags: {theme: aid, entity: two_roles, direction: both, phrasing: casual}
  expect:
    entities: [{dcid: country/DEU, role: origin}, {dcid: country/KEN, role: destination}]
    window: {kind: all}
    operations: []
    definite:
      measure: {family: crs.disbursement_gross, bound: {crsTypeOfFlow: CRSTypeOfFlow10}}

- id: dq-crs-031
  query: commitments from Germany to Kenya in 2022
  reuse: panel/crs-036
  why: Commitments are a separate CRS measure from disbursements, and the year narrows the window.
  tags: {theme: aid, entity: two_roles, time: year, breakdown: commitment_vs_disbursement}
  expect:
    entities: [{dcid: country/DEU, role: origin}, {dcid: country/KEN, role: destination}]
    window: {kind: range, start: 2022, end: 2022}
    definite:
      measure: {family: crs.commitment}

- id: dq-crs-061
  query: gender equality aid to Afghanistan
  why: Gender marker is dropped; only purpose codes 15170 and 15180 answer, or the qualifier is refused.
  tags: {theme: gender, entity: country, direction: destination_only, breakdown: sector}
  known_failure: {issue: 0, checks: [measure]}
  expect:
    entities: [{dcid: country/AFG, role: destination}]
    window: {kind: all}
    definite:
      measure:
        family: crs.disbursement_gross
        bound: {crsTypeOfFlow: CRSTypeOfFlow10, crsPurpose: [CRSPurpose15170, CRSPurpose15180]}
    no_data: {reason: qualifier_not_bound, category: absent}

- id: synth-population-kenya
  query: population of Kenya
  why: Count_Person is the bare population count, with observations for Kenya from 1960.
  tags: {domain: standard}
  expect:
    entities: [{dcid: country/KEN, role: observationAbout}]
    definite:
      measure:
        shape: {population_type: Person, measured_property: count, stat_type: measuredValue, measurement_qualifier: null, measurement_denominator: null}
        stat_vars: [Count_Person]

- id: co2-emissions-sdg-kenya
  query: CO2 emissions from fuel combustion in Kenya
  why: Only the UN System Data Commons answers, with two indicators that differ by source.
  tags: {domain: sdg}
  expect:
    entities: [{dcid: country/KEN, role: observationAbout}]
    candidates: {include_families: [un.co2_fuel_combustion, un.co2_manufacturing]}

- id: ent-04
  query: ODA from the World Bank
  slice: holdout
  why: World Bank names both IBRD and IDA on the dev graph, so the entity is ambiguous.
  tags: {domain: standard}
  expect:
    no_data: {reason: entity_ambiguous, category: unresolved}

- id: oos-sky-blue
  query: Why is the sky blue?
  why: A general science question has no answer in development finance data.
  tags: {domain: standard}
  expect:
    kind: declined
    declined: {reason: out_of_scope}
    scope_place: null

- id: cur-ids-tza-imf-usd-constant-2024-projected
  query: "dataset_resubmit: IMF debt disbursements to Tanzania, 2015 to 2024, in constant 2024 US dollars"
  door: dataset_resubmit
  include_observations: true
  why: Converting 2024 dollars projects the deflator past its last published year, which warns.
  tags: {domain: standard, topic: currency}
  resubmit:
    variable_text: debt disbursements from the IMF, 2015 to 2024
    shape: {kind: statvar_schema, population_type: FinancialFlow, measured_property: amount, stat_type: measuredValue, measurement_qualifier: Nominal}
    slots:
      - key: {axis: how, property: {ref: financialFlowType, label: financial flow type}, label: financial flow type}
        binding: {kind: value, value: {ref: {ref: Disbursement, label: disbursement}, value_kind: enum_value, time_window: null, literal: null}}
      - key: {axis: how, property: {ref: financingResource, label: financing resource}, label: financing resource}
        binding: {kind: value, value: {ref: {ref: DebtInstrument, label: debt instrument}, value_kind: enum_value, time_window: null, literal: null}}
      - key: {axis: when, property: null, label: when}
        binding: {kind: value, value: {ref: null, value_kind: time_window, time_window: {start_year: 2015, end_year: 2024}, literal: null}}
    entity_dcids: [country/TZA]
    origin_dcids: [InternationalMonetaryFund]
    facets: {unit: {state: value, request: {currency: USD, prices: constant, base_year: 2024}}}
  expect:
    entities: [{dcid: country/TZA, role: destination}, {dcid: InternationalMonetaryFund, role: origin}]
    window: {kind: range, start: 2015, end: 2024}
    unit: {currency: USD, prices: constant, base_year: 2024}
    definite:
      measure:
        shape: {population_type: FinancialFlow, measured_property: amount, stat_type: measuredValue, measurement_qualifier: Nominal, measurement_denominator: null}
        bound: {financialFlowType: Disbursement, financingResource: DebtInstrument}
        stat_vars: [Amount_FinancialFlow_Disbursement_DebtInstrument_Nominal]
      warnings: [deflator_projected]
"""

ALL_CHECKS = frozenset(CHECK_NAMES)


def _write_corpus(root: Path, cases: str = EXAMPLES, families: dict[str, str] = FAMILIES) -> Path:
    (root / "families").mkdir()
    for name, text in families.items():
        (root / "families" / name).write_text(text)
    (root / "examples.yaml").write_text(cases)
    return root


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    return _write_corpus(tmp_path)


@pytest.fixture(scope="module")
def families(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Family]:
    root = _write_corpus(tmp_path_factory.mktemp("corpus"))
    return load_families(root / "families")


@pytest.fixture(scope="module")
def examples(families: dict[str, Family]) -> dict[str, Case]:
    return {raw["id"]: validate_case(raw, families=families) for raw in yaml.safe_load(EXAMPLES)}


@pytest.fixture(scope="module")
def raw_examples() -> dict[str, dict]:
    return {raw["id"]: raw for raw in yaml.safe_load(EXAMPLES)}


def _minimal(**fields) -> dict:
    return {
        "id": "case-1",
        "query": "population of Kenya",
        "why": "Count_Person answers.",
        "expect": {"definite": {}},
    } | fields


def _errors(raw: dict, families: dict[str, Family]) -> set[tuple[str, str]]:
    """The (location, error type) pairs validating ``raw`` raises."""
    with pytest.raises(ValidationError) as info:
        validate_case(raw, families=families)
    return {(".".join(map(str, e["loc"])), e["type"]) for e in info.value.errors()}


# The eight examples


def test_the_eight_examples_load_from_a_corpus_directory(corpus_dir: Path):
    by_file = load_case_files(corpus_dir)
    assert list(by_file) == ["examples"]
    assert [c.id for c in by_file["examples"]] == [
        "dq-crs-001",
        "dq-crs-031",
        "dq-crs-061",
        "synth-population-kenya",
        "co2-emissions-sdg-kenya",
        "ent-04",
        "oos-sky-blue",
        "cur-ids-tza-imf-usd-constant-2024-projected",
    ]


def test_a_family_expands_into_shape_fixed_slots_and_provenance(examples: dict[str, Case]):
    measure = examples["dq-crs-001"].expect.definite.measure
    [form] = measure.forms
    assert form.family == "crs.disbursement_gross"
    assert form.shape["population_type"] == "FinancialFlow"
    assert form.shape["measurement_denominator"] is None
    assert form.bound == {
        "financialFlowType": "Disbursement",
        "nettingBasis": "Gross",
        "crsTypeOfFlow": "CRSTypeOfFlow10",
    }
    assert form.provenance.model_dump() == {
        "origin": "dfo",
        "dataset": "provenance/OECDCreditorReportingSystem",
    }


def test_an_inline_shape_is_one_form_with_no_provenance(examples: dict[str, Case]):
    [form] = examples["synth-population-kenya"].expect.definite.measure.forms
    assert form.family is None
    assert form.stat_vars == ["Count_Person"]
    assert form.provenance is None


def test_fields_under_a_status_key_apply_only_to_that_status(examples: dict[str, Case]):
    expect = examples["dq-crs-061"].expect
    assert expect.accepted_statuses == ("definite", "no_data")
    definite = expect.for_status("definite")
    no_data = expect.for_status("no_data")
    assert definite.measure is not None
    assert isinstance(no_data, NoDataExpect)
    assert no_data.measure is None
    assert no_data.reason == "qualifier_not_bound"
    assert expect.for_status("candidates") is None


def test_shared_fields_apply_to_every_accepted_status(examples: dict[str, Case]):
    expect = examples["dq-crs-061"].expect
    for status in expect.accepted_statuses:
        merged = expect.for_status(status)
        assert [(e.dcid, e.role) for e in merged.entities] == [("country/AFG", "destination")]
        assert "entities" in merged.model_fields_set


def test_candidates_expand_their_families(examples: dict[str, Case]):
    candidates = examples["co2-emissions-sdg-kenya"].expect.for_status("candidates")
    assert isinstance(candidates, CandidatesExpect)
    assert candidates.family_forms["un.co2_fuel_combustion"].stat_vars == ["undata/sdg/EN_ATM_CO2"]


@pytest.mark.parametrize(
    ("case_id", "checks"),
    [
        (
            "dq-crs-001",
            {
                "disposition",
                "measure",
                "provenance",
                "entities",
                "request",
                "operations",
                "materialisation",
            },
        ),
        (
            "dq-crs-031",
            {"disposition", "measure", "provenance", "entities", "request", "materialisation"},
        ),
        (
            "dq-crs-061",
            {
                "disposition",
                "measure",
                "provenance",
                "entities",
                "request",
                "refusal",
                "materialisation",
            },
        ),
        ("synth-population-kenya", {"disposition", "measure", "entities", "materialisation"}),
        ("co2-emissions-sdg-kenya", {"disposition", "entities", "candidates"}),
        ("ent-04", {"disposition", "refusal"}),
        ("oos-sky-blue", {"disposition", "refusal"}),
        (
            "cur-ids-tza-imf-usd-constant-2024-projected",
            {"disposition", "measure", "entities", "request", "warnings", "materialisation"},
        ),
    ],
)
def test_asserted_checks(examples: dict[str, Case], case_id: str, checks: set[str]):
    assert asserted_checks(examples[case_id]) == checks


@pytest.mark.parametrize(
    ("case_id", "reasons"),
    [
        ("dq-crs-001", []),
        ("dq-crs-061", []),
        ("synth-population-kenya", []),
        ("oos-sky-blue", []),
        ("cur-ids-tza-imf-usd-constant-2024-projected", []),
        ("ent-04", ["slice holdout"]),
        (
            "co2-emissions-sdg-kenya",
            [
                "unconfirmed value at expect.candidates.include_families.un.co2_fuel_combustion.provenance.dataset",
                "unconfirmed value at expect.candidates.include_families.un.co2_manufacturing.provenance.dataset",
            ],
        ),
    ],
)
def test_eligibility_is_computed(examples: dict[str, Case], case_id: str, reasons: list[str]):
    assert ineligibility(examples[case_id], implemented_checks=ALL_CHECKS) == reasons


def test_a_case_whose_check_has_no_evaluator_is_ineligible(examples: dict[str, Case]):
    implemented = ALL_CHECKS - {"refusal"}
    assert ineligibility(examples["dq-crs-061"], implemented_checks=implemented) == [
        "no evaluator for check refusal"
    ]


def test_a_pending_activation_is_ineligible(raw_examples, families):
    raw = raw_examples["dq-crs-031"] | {"activation": "pending_data:crs"}
    case = validate_case(raw, families=families)
    assert ineligibility(case, implemented_checks=ALL_CHECKS) == ["activation pending_data:crs"]


# Expectation hash


def test_the_hash_ignores_fields_outside_the_request_and_expectation(raw_examples, families):
    raw = raw_examples["dq-crs-061"]
    edited = raw | {
        "why": "Another reason.",
        "tags": {},
        "slice": "holdout",
        "activation": "pending_data:crs",
        "reuse": "panel/x",
        "known_failure": {"issue": 7, "checks": ["refusal"]},
    }
    assert expectation_hash(validate_case(edited, families=families)) == expectation_hash(
        validate_case(raw, families=families)
    )


@pytest.mark.parametrize(
    "edit",
    [
        lambda raw: raw.update(query="gender aid to Afghanistan"),
        lambda raw: raw.update(include_observations=True),
        lambda raw: raw["expect"]["no_data"].update(category="unresolved"),
        lambda raw: raw["expect"]["definite"]["measure"]["bound"].update(crsPurpose="X"),
    ],
)
def test_the_hash_moves_with_the_request_and_expectation(raw_examples, families, edit):
    raw = copy.deepcopy(raw_examples["dq-crs-061"])
    before = expectation_hash(validate_case(raw, families=families))
    edit(raw)
    assert expectation_hash(validate_case(raw, families=families)) != before


def test_the_hash_moves_with_a_family_the_case_names(raw_examples, families):
    raw = raw_examples["dq-crs-031"]
    before = expectation_hash(validate_case(raw, families=families))
    edited = families | {
        "crs.commitment": families["crs.commitment"].model_copy(update={"fixed": {}})
    }
    assert expectation_hash(validate_case(raw, families=edited)) != before


def test_the_hash_moves_with_the_resubmit_payload(raw_examples, families):
    raw = copy.deepcopy(raw_examples["cur-ids-tza-imf-usd-constant-2024-projected"])
    before = expectation_hash(validate_case(raw, families=families))
    raw["resubmit"]["entity_dcids"] = ["country/KEN"]
    assert expectation_hash(validate_case(raw, families=families)) != before


def test_the_hash_ignores_families_the_case_does_not_name(raw_examples, families):
    raw = raw_examples["dq-crs-031"]
    before = expectation_hash(validate_case(raw, families=families))
    edited = families | {
        "crs.disbursement_gross": families["crs.disbursement_gross"].model_copy(
            update={"fixed": {}}
        )
    }
    assert expectation_hash(validate_case(raw, families=edited)) == before


# Case-level rules


def test_why_is_required(families):
    raw = _minimal()
    del raw["why"]
    assert ("why", "missing") in _errors(raw, families)


def test_why_is_at_most_200_characters(families):
    assert validate_case(_minimal(why="x" * 200), families=families)
    assert ("why", "string_too_long") in _errors(_minimal(why="x" * 201), families)


def test_why_is_one_line(families):
    assert ("why", "value_error") in _errors(_minimal(why="one\ntwo"), families)


@pytest.mark.parametrize("field", ["notes", "decision", "status"])
def test_retired_golden_fields_are_refused(families, field):
    assert (field, "extra_forbidden") in _errors(_minimal(**{field: "x"}), families)


def test_known_failure_names_only_known_checks(families):
    good = _minimal(known_failure={"issue": 0, "checks": ["measure", "task_outcome"]})
    assert validate_case(good, families=families).known_failure.checks == [
        "measure",
        "task_outcome",
    ]
    bad = _minimal(known_failure={"issue": 0, "checks": ["interpretation_match"]})
    assert ("known_failure.checks.0", "literal_error") in _errors(bad, families)


def test_known_failure_refuses_the_derived_answer_check(families):
    bad = _minimal(known_failure={"issue": 0, "checks": ["measure", "answer"]})
    assert ("known_failure.checks", "value_error") in _errors(bad, families)


@pytest.mark.parametrize(
    ("fields", "loc"),
    [
        ({"door": "parsed"}, ""),
        ({"parsed": [{"variable_text": "population"}]}, ""),
        ({"door": "dataset_resubmit"}, ""),
        (
            {"door": "parsed", "parsed": [{"variable_text": "population", "colour": "red"}]},
            "parsed.0.colour",
        ),
    ],
)
def test_the_door_payload_matches_the_door(families, fields, loc):
    errors = _errors(_minimal(**fields), families)
    assert loc in {e[0] for e in errors}


def test_a_parsed_case_validates_its_parts(families):
    case = validate_case(
        _minimal(
            door="parsed",
            parsed=[{"variable_text": "population", "entities": [{"text": "Kenya"}]}],
        ),
        families=families,
    )
    assert case.parsed[0].entities[0].text == "Kenya"


@pytest.mark.parametrize("activation", ["live", "pending_data:weo"])
def test_activation_is_live_or_pending_data(families, activation):
    assert validate_case(_minimal(activation=activation), families=families)


def test_an_unknown_activation_is_refused(families):
    assert ("activation", "string_pattern_mismatch") in _errors(
        _minimal(activation="deferred"), families
    )


# Expect rules


def test_a_resolution_case_needs_a_status_key(families):
    assert ("expect", "value_error") in _errors(_minimal(expect={}), families)


def test_a_status_key_may_not_repeat_a_shared_field(families):
    expect = {
        "entities": [{"dcid": "country/KEN", "role": "observationAbout"}],
        "definite": {"entities": [{"dcid": "country/KEN", "role": "observationAbout"}]},
    }
    assert ("expect", "value_error") in _errors(_minimal(expect=expect), families)


@pytest.mark.parametrize(
    ("status", "field", "value"),
    [
        ("no_data", "skills", {"present": True}),
        ("definite", "reason", "no_observations"),
        ("definite", "include_families", ["crs.commitment"]),
        ("candidates", "code", "unit_request_unsupported"),
    ],
)
def test_a_field_belongs_to_its_status_key(families, status, field, value):
    raw = _minimal(expect={status: {field: value}})
    assert (f"expect.{status}.{field}", "extra_forbidden") in _errors(raw, families)


def test_skills_sit_under_definite(families):
    raw = _minimal(expect={"definite": {"skills": {"present": True, "ids": ["oda-measures"]}}})
    assert validate_case(raw, families=families).expect.definite.skills.ids == ["oda-measures"]


def test_no_data_reason_may_be_a_list(families):
    raw = _minimal(expect={"no_data": {"reason": ["no_observations", "no_matching_slice"]}})
    assert validate_case(raw, families=families).expect.no_data.reason == [
        "no_observations",
        "no_matching_slice",
    ]


def test_an_unknown_no_data_reason_is_refused(families):
    raw = _minimal(expect={"no_data": {"reason": "nothing_here"}})
    assert any(loc.startswith("expect.no_data.reason") for loc, _ in _errors(raw, families))


def test_unsupported_carries_a_code(families):
    raw = _minimal(expect={"unsupported": {"code": "unit_request_unsupported"}})
    case = validate_case(raw, families=families)
    assert case.expect.unsupported.code == "unit_request_unsupported"
    assert asserted_checks(case) == {"disposition", "refusal"}


def test_kind_may_be_a_list(families):
    raw = _minimal(
        expect={
            "kind": ["resolution", "declined"],
            "declined": {"reason": "requires_analysis"},
            "definite": {},
        }
    )
    assert validate_case(raw, families=families).expect.kinds == {"resolution", "declined"}


@pytest.mark.parametrize(
    "expect",
    [
        {"kind": "declined"},
        {"kind": "resolution", "declined": {"reason": "out_of_scope"}, "definite": {}},
        {"kind": "declined", "declined": {"reason": "out_of_scope"}, "definite": {}},
        {"kind": "catalogue", "window": {"kind": "all"}},
        {"definite": {}, "scope_place": None},
        {"kind": "problem"},
        {"definite": {}, "problem": {"code": "facet_binding_unsupported"}},
        {"kind": "problem", "problem": {"code": "x"}, "definite": {}},
    ],
)
def test_kind_governs_declined_scope_place_and_status_keys(families, expect):
    assert ("expect", "value_error") in _errors(_minimal(expect=expect), families)


def test_a_case_may_expect_a_problem_document(families):
    expect = {
        "kind": ["resolution", "problem"],
        "unsupported": {"code": "facet_binding_unsupported"},
        "problem": {"code": "facet_binding_unsupported"},
    }
    case = validate_case(_minimal(expect=expect), families=families)
    assert case.expect.kinds == {"resolution", "problem"}
    assert case.expect.problem.code == "facet_binding_unsupported"
    assert asserted_checks(case) >= {"disposition", "refusal"}


def test_the_hash_covers_the_problem_code(families):
    def hashed(code: str) -> str:
        raw = _minimal(expect={"kind": "problem", "problem": {"code": code}})
        return expectation_hash(validate_case(raw, families=families))

    assert hashed("facet_binding_unsupported") != hashed("unit_request_unsupported")


def test_a_catalogue_case_may_assert_its_scope_place(families):
    raw = _minimal(expect={"kind": "catalogue", "scope_place": "country/ZMB"})
    assert validate_case(raw, families=families).expect.scope_place == "country/ZMB"


def test_parts_are_status_keyed_in_query_order(families):
    expect = {
        "parts": [
            {"definite": {"measure": {"family": "crs.commitment"}}},
            {"no_data": {"reason": "no_observations"}, "window": {"kind": "latest"}},
        ]
    }
    case = validate_case(_minimal(expect=expect), families=families)
    assert [p.accepted_statuses for p in case.expect.parts] == [("definite",), ("no_data",)]
    assert case.expect.parts[1].for_status("no_data").window.kind == "latest"
    assert "parts" in asserted_checks(case)


@pytest.mark.parametrize(
    "expect",
    [
        {"definite": {}, "parts": [{"definite": {}}, {"definite": {}}]},
        {"window": {"kind": "all"}, "parts": [{"definite": {}}, {"definite": {}}]},
    ],
)
def test_parts_are_exclusive_with_top_level_status_keys_and_shared_fields(families, expect):
    assert ("expect", "value_error") in _errors(_minimal(expect=expect), families)


def test_a_part_needs_a_status_key(families):
    expect = {"parts": [{"definite": {}}, {"window": {"kind": "all"}}]}
    assert ("expect.parts.1", "value_error") in _errors(_minimal(expect=expect), families)


def test_mixed_is_a_top_level_status(families):
    raw = _minimal(expect={"mixed": {}})
    assert validate_case(raw, families=families).expect.accepted_statuses == ("mixed",)


def test_status_keys_match_the_wire_statuses():
    assert get_args(StatusKey) == (*get_args(StatusLiteral), "mixed")


# Measure rules


@pytest.mark.parametrize(
    ("measure", "loc", "error"),
    [
        ({"family": "crs.unknown"}, "expect.definite.measure", "value_error"),
        (
            {"family": "crs.commitment", "shape": {"stat_type": "measuredValue"}},
            "expect.definite.measure",
            "value_error",
        ),
        ({}, "expect.definite.measure", "value_error"),
        ({"bound": {"a": "x"}, "open": ["a"]}, "expect.definite.measure", "value_error"),
        (
            {"family": "crs.commitment", "bound": {"financialObligationType": "Pledge"}},
            "expect.definite.measure",
            "value_error",
        ),
        (
            {"family": "crs.commitment", "open": ["financialObligationType"]},
            "expect.definite.measure",
            "value_error",
        ),
        (
            {"shape": {"populationType": "Person"}},
            "expect.definite.measure.shape.populationType.[key]",
            "literal_error",
        ),
    ],
)
def test_measure_rules(families, measure, loc, error):
    assert (loc, error) in _errors(_minimal(expect={"definite": {"measure": measure}}), families)


def test_a_family_list_yields_one_form_per_family(families):
    measure = {"family": ["crs.commitment", "crs.disbursement_gross"], "open": ["crsPurpose"]}
    case = validate_case(_minimal(expect={"definite": {"measure": measure}}), families=families)
    forms = case.expect.definite.measure.forms
    assert [f.family for f in forms] == ["crs.commitment", "crs.disbursement_gross"]
    assert all(f.open == ["crsPurpose"] for f in forms)


def test_a_bound_value_repeating_the_family_value_is_accepted(families):
    measure = {"family": "crs.commitment", "bound": {"financialObligationType": "Commitment"}}
    case = validate_case(_minimal(expect={"definite": {"measure": measure}}), families=families)
    [form] = case.expect.definite.measure.forms
    assert form.bound == {"financialObligationType": "Commitment"}


def test_an_unknown_candidate_family_is_refused(families):
    raw = _minimal(expect={"candidates": {"include_families": ["un.nothing"]}})
    assert ("expect.candidates", "value_error") in _errors(raw, families)


def test_provenance_overrides_the_family(families):
    expect = {
        "definite": {
            "measure": {"family": "crs.commitment"},
            "provenance": {"origin": "base"},
        }
    }
    case = validate_case(_minimal(expect=expect), families=families)
    assert case.expect.definite.provenance.origin == "base"
    assert "provenance" in asserted_checks(case)


def test_provenance_asserts_something(families):
    raw = _minimal(expect={"definite": {"provenance": {}}})
    assert ("expect.definite.provenance", "value_error") in _errors(raw, families)


# Entities, window, unit, operations


def test_entities_take_known_roles_and_form_a_set(families):
    expect = {
        "definite": {"entities": [{"dcid": "africa", "role": "destination", "expands": True}]}
    }
    case = validate_case(_minimal(expect=expect), families=families)
    assert case.expect.definite.entities[0].expands is True

    bad_role = {"definite": {"entities": [{"dcid": "country/KEN", "role": "subject"}]}}
    errors = _errors(_minimal(expect=bad_role), families)
    assert any(loc.startswith("expect.definite.entities.0.role") for loc, _ in errors)

    entity = {"dcid": "country/KEN", "role": "destination"}
    twice = {"definite": {"entities": [entity, entity]}}
    assert ("expect.definite.entities", "value_error") in _errors(_minimal(expect=twice), families)


def test_an_expanding_group_without_a_node_is_named_by_label(families):
    group = {"label": "G20", "role": "observationAbout", "expands": True}
    case = validate_case(_minimal(expect={"definite": {"entities": [group]}}), families=families)
    assert case.expect.definite.entities[0].label == "G20"
    assert case.expect.definite.entities[0].dcid is None


@pytest.mark.parametrize(
    "entity",
    [
        {"dcid": "africa", "label": "Africa", "role": "destination", "expands": True},
        {"role": "destination", "expands": True},
        {"label": "Kenya", "role": "destination"},
        {"dcid": "country/KEN", "label": "Kenya", "role": "destination"},
    ],
)
def test_an_entity_names_itself_once(families, entity):
    raw = _minimal(expect={"definite": {"entities": [entity]}})
    assert ("expect.definite.entities.0", "value_error") in _errors(raw, families)


def test_a_label_differs_from_another_label_on_one_role(families):
    groups = [
        {"label": "G20", "role": "origin", "expands": True},
        {"label": "G7", "role": "origin", "expands": True},
    ]
    validate_case(_minimal(expect={"definite": {"entities": groups}}), families=families)
    twice = {"definite": {"entities": [groups[0], groups[0]]}}
    assert ("expect.definite.entities", "value_error") in _errors(_minimal(expect=twice), families)


def test_the_hash_covers_a_group_label(families):
    def hashed(label: str) -> str:
        group = {"label": label, "role": "observationAbout", "expands": True}
        raw = _minimal(expect={"definite": {"entities": [group]}})
        return expectation_hash(validate_case(raw, families=families))

    assert hashed("G20") != hashed("G7")


@pytest.mark.parametrize(
    "window",
    [
        {"kind": "all"},
        {"kind": "latest"},
        {"kind": "range", "start": 2010},
        {"kind": "range", "end": 2020},
    ],
)
def test_window_kinds(families, window):
    case = validate_case(_minimal(expect={"window": window, "definite": {}}), families=families)
    assert case.expect.window.kind == window["kind"]


@pytest.mark.parametrize(
    "window",
    [
        {"kind": "range"},
        {"kind": "all", "start": 2010},
        {"kind": "range", "start": 2020, "end": 2010},
        {"kind": "decade"},
    ],
)
def test_window_rules(families, window):
    errors = _errors(_minimal(expect={"window": window, "definite": {}}), families)
    assert any(loc.startswith("expect.window") for loc, _ in errors)


def test_unit_is_a_request_or_none(families):
    none = validate_case(_minimal(expect={"unit": "none", "definite": {}}), families=families)
    assert none.expect.unit == "none"
    role = {"currency": {"role": "destination"}, "prices": "current"}
    case = validate_case(_minimal(expect={"unit": role, "definite": {}}), families=families)
    assert case.expect.unit.currency.role == "destination"


def test_an_omitted_unit_field_is_not_asserted(families):
    any_base_year = {"prices": "constant"}
    case = validate_case(
        _minimal(expect={"unit": any_base_year, "definite": {}}), families=families
    )
    assert (case.expect.unit.currency, case.expect.unit.prices, case.expect.unit.base_year) == (
        None,
        "constant",
        None,
    )
    assert case.expect.unit.model_fields_set == {"prices"}
    assert "request" in asserted_checks(case)


@pytest.mark.parametrize(
    "unit",
    [{}, {"prices": "current", "base_year": 2020}, {"currency": "usd"}, {"prices": "real"}],
)
def test_unit_rules(families, unit):
    errors = _errors(_minimal(expect={"unit": unit, "definite": {}}), families)
    assert any(loc.startswith("expect.unit") for loc, _ in errors)


def test_operations_take_known_verbs(families):
    ops = [{"verb": "rank", "over": "destination", "target_year": 2022}]
    case = validate_case(_minimal(expect={"operations": ops, "definite": {}}), families=families)
    assert case.expect.operations[0].over == "destination"
    bad = [{"verb": "average"}]
    errors = _errors(_minimal(expect={"operations": bad, "definite": {}}), families)
    assert ("expect.operations.0.verb", "literal_error") in errors


# Unconfirmed values


def test_a_gl_value_marks_the_case_unconfirmed(families):
    expect = {
        "entities": [{"dcid": {"gl": "the Sahel group dcid"}, "role": "destination"}],
        "definite": {
            "measure": {"family": "crs.commitment", "bound": {"gl": "the sector slots"}},
            "warnings": [{"gl": "the dropped-qualifier code"}],
        },
    }
    case = validate_case(_minimal(expect=expect), families=families)
    assert isinstance(case.expect.entities[0].dcid, Gl)
    assert case.expect.definite.measure.forms[0].unconfirmed == [Gl(gl="the sector slots")]
    assert unconfirmed_paths(case) == [
        "expect.entities[0].dcid",
        "expect.definite.measure.forms[0].unconfirmed[0]",
        "expect.definite.warnings[0]",
    ]


def test_a_gl_in_a_named_family_marks_the_case_unconfirmed(families):
    pending = families | {
        "ids.commitment": Family(
            origin="dfo", dataset="provenance/IDS", fixed=Gl(gl="IDS COM constraint slots")
        )
    }
    case = validate_case(
        _minimal(expect={"definite": {"measure": {"family": "ids.commitment"}}}),
        families=pending,
    )
    assert unconfirmed_paths(case) == ["expect.definite.measure.forms[0].unconfirmed[0]"]


# Loading


def test_a_missing_families_directory_holds_no_families(tmp_path: Path):
    assert load_families(tmp_path / "families") == {}
    (tmp_path / "empty.yaml").write_text("")
    assert load_cases(tmp_path) == []


def test_loading_reports_every_error(tmp_path: Path):
    second = """
- id: dq-crs-001
  query: again
  why: A repeat.
  expect: {definite: {}}
- id: bad-family
  query: x
  why: y
  expect: {definite: {measure: {family: crs.nothing}}}
- id: long-why
  query: x
  why: %s
  expect: {definite: {}}
""" % ("x" * 201)
    root = _write_corpus(tmp_path)
    (root / "second.yaml").write_text(second)
    with pytest.raises(CorpusError) as info:
        load_cases(root)
    errors = info.value.errors
    assert len(errors) == 3
    assert errors[0].startswith("second.yaml: dq-crs-001: duplicate id")
    assert errors[1].startswith("second.yaml: bad-family: expect.definite.measure")
    assert errors[2].startswith("second.yaml: long-why: why")


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("dup.yaml", "- id: a\n  id: b\n  query: q\n  why: w\n  expect: {definite: {}}\n"),
        ("map.yaml", "id: a\n"),
        ("broken.yaml", "- [unclosed\n"),
    ],
)
def test_a_malformed_case_file_is_an_error(tmp_path: Path, name: str, text: str):
    (tmp_path / name).write_text(text)
    with pytest.raises(CorpusError) as info:
        load_cases(tmp_path)
    assert info.value.errors[0].startswith(name)


def test_a_family_id_repeated_across_files_is_an_error(tmp_path: Path):
    root = _write_corpus(tmp_path, families=FAMILIES | {"crs_more.yaml": FAMILIES["crs.yaml"]})
    with pytest.raises(CorpusError) as info:
        load_families(root / "families")
    assert info.value.errors == [
        "families/crs_more.yaml: crs.disbursement_gross: duplicate family id",
        "families/crs_more.yaml: crs.commitment: duplicate family id",
    ]


def test_the_schema_describes_a_list_of_cases():
    schema = corpus_schema()
    assert schema["type"] == "array"
    case = schema["$defs"]["Case"]
    assert set(case["required"]) == {"id", "query", "why", "expect"}
    assert "forms" not in schema["$defs"]["Measure"]["properties"]
