"""The check registry's verdicts as one Pydantic Evals evaluator.

A single-mode task samples a case adaptively, as the gate's head does, and judges after
each round whether more samples could change a verdict, so the task scores every sample
itself. It returns the case's modal sample as a ``ScoredAnswer``: every check's result,
the derived ``answer`` included, from ``checks.score_with_answer``. ``SearchChecks``
returns those results as labels named after their checks, each with its verdict and
detail, so the report holds the scores the run judged by and no sample is scored twice.

The evaluator's own name is ``answer``, so a run that let an evaluator error stand would
have ``dfo-evals`` record a grader fault against the headline check. Single mode and the
nightly instead treat any evaluator error as a crash (``evaluate.evaluate_or_raise``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from dfo_evals.checks import verdict
from dfo_evals.records import Verdict as RecordVerdict
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from search.eval.checks import ANSWER
from search.eval.checks.base import Result
from search.eval.corpus_dataset import CaseInputs, CorpusCase


@dataclass(frozen=True)
class ScoredAnswer:
    """A case's modal sample as its run scored it.

    Attributes:
        results: Every check's result on the sample, ``answer`` included.
        infra: The run failed on infrastructure, or a check's read of the graph or
            coverage did, so the sample measures the environment rather than Search.
        trace_id: The trace id the sample's response carries, or None.
    """

    results: Mapping[str, Result]
    infra: bool
    trace_id: str | None


def check_label(result: Result) -> EvaluationReason:
    """Return a check result as the label ``dfo-evals`` reads, its verdict and detail."""
    return verdict(RecordVerdict(result.verdict.value), result.detail or None)


@dataclass
class SearchChecks(Evaluator[CaseInputs, ScoredAnswer, CorpusCase]):
    """Label every check's result on a case's modal sample, ``answer`` included."""

    def get_default_evaluation_name(self) -> str:
        """Name the evaluator after the headline check it produces."""
        return ANSWER

    def evaluate(
        self, ctx: EvaluatorContext[CaseInputs, ScoredAnswer, CorpusCase]
    ) -> dict[str, EvaluationReason]:
        """Return the task's scored results as labels.

        Raises:
            ValueError: The results lack the ``answer`` check.
        """
        if ANSWER not in ctx.output.results:
            raise ValueError(f"case {ctx.name!r}: the scored sample has no {ANSWER!r} result")
        return {name: check_label(result) for name, result in ctx.output.results.items()}
