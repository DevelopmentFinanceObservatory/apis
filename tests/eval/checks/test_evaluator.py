"""The check registry as one Pydantic Evals evaluator, against the registry itself."""

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
from search.eval.checks.base import Result, Verdict, score_with_answer
from search.eval.corpus_dataset import SUITE, CaseInputs, CorpusCase, case_inputs
from search.eval.evaluator import SearchChecks, check_label
from tests.eval.checks.conftest import make_case, panel

CRS_032_ENTITIES = [
    {"dcid": "country/DEU", "role": "origin"},
    {"dcid": "country/BRA", "role": "destination"},
]
STARTED_AT = datetime(2026, 11, 25, 2, 0, tzinfo=UTC)

# Recorded responses with expectations that pass, fail, read unreached and read n/a.
SCORED = [
    ("crs-032", {"entities": CRS_032_ENTITIES, "definite": {}}),
    ("crs-032", {"definite": {"measure": {"family": "crs.commitment"}}}),
    ("crs-032", {"definite": {"measure": {"family": "crs.disbursement_gross"}}}),
    ("crs-032", {"no_data": {"reason": "no_observations"}}),
    ("crs-013", {"entities": CRS_032_ENTITIES, "definite": {}, "no_data": {}}),
    ("weo-034", {"definite": {}}),
]


def evaluate(case: Case, response: dict[str, Any], ctx: ScoreContext) -> Any:
    """Evaluate one case whose task returns ``response``, scored by ``SearchChecks``."""
    dataset = Dataset[CaseInputs, Any, CorpusCase](
        name=SUITE,
        cases=[
            EvalCase(
                name=case.id,
                inputs=case_inputs(case),
                metadata=CorpusCase(case=case, file="t", expectation_hash=expectation_hash(case)),
            )
        ],
        evaluators=[SearchChecks(score_context=ctx)],
    )

    def task(inputs: CaseInputs) -> dict[str, Any]:
        return response

    return dataset.evaluate_sync(task, progress=False)


@pytest.mark.parametrize(("row_id", "expect"), SCORED)
def test_the_evaluator_labels_every_check_as_the_registry_scores_it(
    row_id: str, expect: dict[str, Any], ctx: ScoreContext
):
    case = make_case(expect)
    response = panel(row_id)

    (report_case,) = evaluate(case, response, ctx).cases

    expected = score_with_answer(case, panel(row_id), ctx)
    assert {name: (label.value, label.reason) for name, label in report_case.labels.items()} == {
        name: (result.verdict.value, result.detail or None) for name, result in expected.items()
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
    report = evaluate(case, panel("crs-032"), ctx)
    manifest = RunManifest(
        suite=SUITE,
        run_id="t",
        started_at=STARTED_AT,
        finished_at=None,
        revision="t",
        environment={},
        dataset_digest="sha256:t",
        headline_check=ANSWER,
    )

    (result,) = run_from_report(report=report, manifest=manifest).results

    expected = score_with_answer(case, panel("crs-032"), ctx)
    checks = {check.name: check.verdict.value for check in result.checks}
    assert checks == {name: r.verdict.value for name, r in expected.items()}
    assert checks["entities"] == RecordVerdict.PASS
    assert result.fault is None


def test_scoring_that_raises_is_a_grader_fault_on_the_headline(
    ctx: ScoreContext, monkeypatch: pytest.MonkeyPatch
):
    def broken(*args: object, **kwargs: object) -> dict[str, Result]:
        raise RuntimeError("scorer broke")

    monkeypatch.setattr(evaluator_module, "score_with_answer", broken)
    report = evaluate(make_case({"definite": {}}), panel("crs-032"), ctx)
    manifest = RunManifest(
        suite=SUITE,
        run_id="t",
        started_at=STARTED_AT,
        finished_at=None,
        revision="t",
        environment={},
        dataset_digest="sha256:t",
        headline_check=ANSWER,
    )

    (result,) = run_from_report(report=report, manifest=manifest).results

    assert [check.name for check in result.checks] == [ANSWER]
    assert result.checks[0].verdict is RecordVerdict.UNVERIFIABLE
    assert result.fault is FaultClass.GRADER
