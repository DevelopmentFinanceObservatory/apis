"""The check registry's verdicts as one Pydantic Evals evaluator."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from dfo_evals.records import FaultClass, RunManifest
from dfo_evals.records import Verdict as RecordVerdict
from dfo_evals.report import run_from_report
from pydantic_evals import Case as EvalCase
from pydantic_evals import Dataset

from search.eval import evaluator as evaluator_module
from search.eval.cases import Case, expectation_hash
from search.eval.checks import ANSWER, ScoreContext
from search.eval.checks.base import Result, Verdict, answer_result, infra_error, score_with_answer
from search.eval.corpus_dataset import SUITE, CaseInputs, CorpusCase, case_inputs
from search.eval.evaluator import ScoredAnswer, SearchChecks, answer_is_infra, check_label
from tests.eval.checks.conftest import make_case, panel

CRS_032_ENTITIES = [
    {"dcid": "country/DEU", "role": "origin"},
    {"dcid": "country/BRA", "role": "destination"},
]
MANIFEST = RunManifest(
    suite=SUITE,
    run_id="t",
    started_at=datetime(2026, 11, 25, 2, 0, tzinfo=UTC),
    finished_at=None,
    revision="t",
    environment={},
    dataset_digest="sha256:t",
    headline_check=ANSWER,
)

# Recorded responses with expectations that pass, fail, read unreached and read n/a.
SCORED = [
    ("crs-032", {"entities": CRS_032_ENTITIES, "definite": {}}),
    ("crs-032", {"definite": {"measure": {"family": "crs.commitment"}}}),
    ("crs-032", {"definite": {"measure": {"family": "crs.disbursement_gross"}}}),
    ("crs-032", {"no_data": {"reason": "no_observations"}}),
    ("crs-013", {"entities": CRS_032_ENTITIES, "definite": {}, "no_data": {}}),
    ("weo-034", {"definite": {}}),
]


def evaluate(case: Case, answer: ScoredAnswer) -> Any:
    """Evaluate one case whose task returns ``answer``, labelled by ``SearchChecks``."""
    dataset = Dataset[CaseInputs, ScoredAnswer, CorpusCase](
        name=SUITE,
        cases=[
            EvalCase(
                name=case.id,
                inputs=case_inputs(case),
                metadata=CorpusCase(case=case, file="t", expectation_hash=expectation_hash(case)),
            )
        ],
        evaluators=[SearchChecks()],
    )

    def task(inputs: CaseInputs) -> ScoredAnswer:
        return answer

    return dataset.evaluate_sync(task, progress=False)


def scored(case: Case, row_id: str, ctx: ScoreContext) -> ScoredAnswer:
    """A recorded response scored as a run scores its modal sample."""
    return ScoredAnswer(
        results=score_with_answer(case, panel(row_id), ctx), infra=False, trace_id=None
    )


@pytest.mark.parametrize(("row_id", "expect"), SCORED)
def test_the_evaluator_labels_every_check_as_the_registry_scored_it(
    row_id: str, expect: dict[str, Any], ctx: ScoreContext
):
    case = make_case(expect)
    answer = scored(case, row_id, ctx)

    (report_case,) = evaluate(case, answer).cases

    assert {name: (label.value, label.reason) for name, label in report_case.labels.items()} == {
        name: (result.verdict.value, result.detail or None)
        for name, result in answer.results.items()
    }
    assert ANSWER in report_case.labels
    assert report_case.evaluator_failures == []


def test_the_recorded_responses_cover_pass_fail_unreached_and_na(ctx: ScoreContext):
    verdicts = {
        result.verdict
        for row_id, expect in SCORED
        for result in score_with_answer(make_case(expect), panel(row_id), ctx).values()
    }

    assert {Verdict.PASS, Verdict.FAIL, Verdict.UNREACHED, Verdict.NA} <= verdicts


@pytest.mark.parametrize("value", list(Verdict))
def test_every_check_verdict_is_a_run_record_verdict(value: Verdict):
    label = check_label(Result(value, "why"))

    assert RecordVerdict(label.value).value == value.value
    assert label.reason == "why"


def test_a_run_record_holds_each_check_and_the_answer_headline(ctx: ScoreContext):
    case = make_case({"entities": CRS_032_ENTITIES, "definite": {}})
    answer = scored(case, "crs-032", ctx)

    (result,) = run_from_report(report=evaluate(case, answer), manifest=MANIFEST).results

    checks = {check.name: check.verdict.value for check in result.checks}
    assert checks == {name: r.verdict.value for name, r in answer.results.items()}
    assert checks["entities"] == RecordVerdict.PASS
    assert result.fault is None


def test_results_without_the_answer_check_are_a_grader_fault():
    case = make_case({"definite": {}})
    answer = ScoredAnswer(results={"disposition": Result(Verdict.PASS)}, infra=False, trace_id=None)

    (result,) = run_from_report(report=evaluate(case, answer), manifest=MANIFEST).results

    assert [check.name for check in result.checks] == [ANSWER]
    assert result.checks[0].verdict is RecordVerdict.UNVERIFIABLE
    assert result.fault is FaultClass.GRADER


def test_a_labelling_error_is_a_grader_fault_on_the_headline(
    ctx: ScoreContext, monkeypatch: pytest.MonkeyPatch
):
    def broken(result: Result) -> Any:
        raise RuntimeError("labelling broke")

    monkeypatch.setattr(evaluator_module, "check_label", broken)
    case = make_case({"definite": {}})

    (result,) = run_from_report(
        report=evaluate(case, scored(case, "crs-032", ctx)), manifest=MANIFEST
    ).results

    assert result.fault is FaultClass.GRADER


def scored_results(known: frozenset[str] = frozenset(), **checks: Result) -> dict[str, Result]:
    """Check results with ``answer`` derived from them as the registry derives it."""
    return {**checks, ANSWER: answer_result(checks, known=known)}


LOST_READ = infra_error("graph read failed")
PASSED = Result(Verdict.PASS)


@pytest.mark.parametrize(
    ("results", "run_failed", "infra"),
    [
        (scored_results(entities=PASSED), True, True),
        (
            scored_results(entities=Result(Verdict.FAIL, "got KEN"), groundedness=LOST_READ),
            False,
            False,
        ),
        (scored_results(entities=PASSED, groundedness=LOST_READ), False, True),
        (
            scored_results(frozenset({"groundedness"}), entities=PASSED, groundedness=LOST_READ),
            False,
            False,
        ),
        (scored_results(entities=PASSED), False, False),
    ],
    ids=[
        "the run failed",
        "a definite failure beside a lost read",
        "a lost read on a counted check",
        "a lost read on a known-failure check",
        "nothing lost",
    ],
)
def test_a_sample_is_infra_only_when_it_measures_the_environment(results, run_failed, infra):
    assert answer_is_infra(results, run_failed=run_failed) is infra
