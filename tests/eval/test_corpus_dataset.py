"""The eligible corpus as one Pydantic Evals dataset."""

from __future__ import annotations

import json

import pytest
from dfo_evals.report import input_digest
from pydantic_core import to_jsonable_python
from pydantic_evals import Case as EvalCase
from pydantic_evals import Dataset

from search.eval.cases import (
    CHECK_NAMES,
    Case,
    expectation_hash,
    ineligibility,
    load_case_files,
    validate_case,
)
from search.eval.checks.base import CHECKS
from search.eval.corpus_dataset import (
    SUITE,
    CaseInputs,
    case_inputs,
    corpus_dataset,
    corpus_digest,
)

ALL_CHECKS = frozenset(CHECK_NAMES)


def make_case(**fields) -> Case:
    """A minimal valid case; ``fields`` override it."""
    raw = {
        "id": "case-1",
        "query": "population of Kenya",
        "why": "Count_Person answers.",
        "expect": {"definite": {}},
    } | fields
    return validate_case(raw, families={})


def test_the_dataset_holds_exactly_the_eligible_cases_of_the_real_corpus():
    by_file = load_case_files()

    dataset = corpus_dataset(by_file, implemented_checks=CHECKS)

    eligible = [
        case.id
        for cases in by_file.values()
        for case in cases
        if not ineligibility(case, implemented_checks=CHECKS)
    ]
    assert [eval_case.name for eval_case in dataset.cases] == eligible
    assert dataset.name == SUITE


def test_an_ineligible_case_is_left_out():
    by_file = {
        "a": [make_case(id="live"), make_case(id="held", slice="holdout")],
        "b": [make_case(id="pending", activation="pending_data:crs_2025")],
        "c": [make_case(id="other")],
    }

    dataset = corpus_dataset(by_file, implemented_checks=ALL_CHECKS)
    # Every case asserts disposition, so without its evaluator none is eligible.
    unchecked = corpus_dataset(by_file, implemented_checks=ALL_CHECKS - {"disposition"})

    assert [c.name for c in dataset.cases] == ["live", "other"]
    assert unchecked.cases == []


def test_inputs_are_the_authored_request_and_metadata_the_case_and_its_file():
    case = make_case(id="kenya", include_observations=True)

    (eval_case,) = corpus_dataset({"dac": [case]}, implemented_checks=ALL_CHECKS).cases

    assert eval_case.inputs == CaseInputs(
        case_id="kenya",
        query="population of Kenya",
        door="raw_text",
        parsed=None,
        resubmit=None,
        include_observations=True,
    )
    assert eval_case.expected_output == case.expect
    assert eval_case.metadata is not None
    assert eval_case.metadata.case == case
    assert eval_case.metadata.file == "dac"
    assert eval_case.metadata.source == "corpus/dac.yaml"
    assert eval_case.metadata.expectation_hash == expectation_hash(case)


def test_two_cases_authoring_the_same_request_keep_distinct_inputs():
    first = make_case(id="first")
    second = make_case(id="second")

    dataset = corpus_dataset({"a": [first], "b": [second]}, implemented_checks=ALL_CHECKS)

    assert dataset.cases[0].inputs != dataset.cases[1].inputs
    assert case_inputs(first).model_dump(exclude={"case_id"}) == case_inputs(second).model_dump(
        exclude={"case_id"}
    )


def test_a_case_id_in_two_files_is_refused():
    with pytest.raises(ValueError, match="case-1"):
        corpus_dataset({"a": [make_case()], "b": [make_case()]}, implemented_checks=ALL_CHECKS)


def test_the_digest_ignores_order_and_moves_with_cases_and_expectations():
    a = make_case(id="a")
    b = make_case(id="b")
    edited = make_case(id="b", expect={"no_data": {}})

    def digest(*cases: Case) -> str:
        return corpus_digest(corpus_dataset({"f": list(cases)}, implemented_checks=ALL_CHECKS))

    assert digest(a, b) == digest(b, a)
    assert digest(a, b).startswith("sha256:")
    assert digest(a, b) != digest(a)
    assert digest(a, b) != digest(a, edited)
    assert digest(a, b) == digest(a, b.model_copy(update={"why": "Another reason."}))


@pytest.mark.parametrize("door", ["raw_text", "parsed", "dataset_resubmit"])
def test_a_case_s_inputs_survive_a_json_round_trip_with_their_digest(door: str):
    """A Langfuse item stores the inputs as JSON, and a push checks the stored input's
    digest against the run record's, so every door's inputs must round-trip exactly."""
    case = next(case for cases in load_case_files().values() for case in cases if case.door == door)
    inputs = case_inputs(case)

    stored = json.loads(json.dumps(to_jsonable_python(inputs)))

    assert CaseInputs.model_validate(stored) == inputs
    assert input_digest(stored) == input_digest(inputs)


def test_the_digest_refuses_a_case_without_corpus_metadata():
    dataset = Dataset[CaseInputs, object, object](
        name=SUITE, cases=[EvalCase(name="bare", inputs=case_inputs(make_case(id="bare")))]
    )

    with pytest.raises(ValueError, match="'bare' has no corpus metadata"):
        corpus_digest(dataset)  # type: ignore[arg-type]


def test_the_digest_moves_with_a_case_s_known_failures():
    """``answer`` leaves known-failure checks out, so adding one changes the score a run
    gives an identical answer, and the digest must show the rule changed."""
    plain = make_case(id="a")
    known = make_case(id="a", known_failure={"issue": 1, "checks": ["entities"]})
    other = make_case(id="a", known_failure={"issue": 1, "checks": ["measure"]})

    def digest(case: Case) -> str:
        return corpus_digest(corpus_dataset({"f": [case]}, implemented_checks=ALL_CHECKS))

    assert expectation_hash(plain) == expectation_hash(known)
    assert len({digest(plain), digest(known), digest(other)}) == 3
