"""The check registry as one Pydantic Evals evaluator.

``SearchChecks`` scores one raw resolve response with ``checks.score_with_answer``, the
scorer the gate and the nightly use, and returns each check's verdict and detail as a
label named after the check, ``answer`` included. One evaluator scores every check
because the checks share one pass over the response and the graph, and ``answer`` derives
from all of them. Its own name is ``answer``, so when scoring raises, ``dfo-evals`` records
a grader fault against the headline check and leaves the trial out of the scores.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dfo_evals.checks import verdict
from dfo_evals.records import Verdict as RecordVerdict
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from search.eval.checks import ANSWER, ScoreContext
from search.eval.checks.base import Result, score_with_answer
from search.eval.corpus_dataset import CaseInputs, CorpusCase


def check_label(result: Result) -> EvaluationReason:
    """Return a check result as the label ``dfo-evals`` reads, its verdict and detail."""
    return verdict(RecordVerdict(result.verdict.value), result.detail or None)


@dataclass
class SearchChecks(Evaluator[CaseInputs, Mapping[str, Any], CorpusCase]):
    """Score a response with every registered check and the derived ``answer`` check.

    Attributes:
        score_context: The responding app's wire schema and the graph, SDMX, coverage and
            members readers the checks score against.
    """

    score_context: ScoreContext

    def get_default_evaluation_name(self) -> str:
        """Name the evaluator after the headline check it produces."""
        return ANSWER

    def evaluate(
        self, ctx: EvaluatorContext[CaseInputs, Mapping[str, Any], CorpusCase]
    ) -> dict[str, EvaluationReason]:
        """Score the task's response against the case its metadata holds.

        Raises:
            ValueError: The case has no corpus metadata.
        """
        if ctx.metadata is None:
            raise ValueError(f"case {ctx.name!r} has no corpus metadata")
        results = score_with_answer(ctx.metadata.case, ctx.output, self.score_context)
        return {name: check_label(result) for name, result in results.items()}
