"""Evaluate the corpus dataset, stopping at the first crash.

Pydantic Evals records a task that raises as one case's failure and runs every other case,
and records an evaluator that raises the same way. For single mode and the nightly an
exception there is a fault in the driver, not a verdict, so running on would spend the
whole run's traffic and keep nothing. ``evaluate_or_raise`` cancels the evaluation at the
first task that raises and raises that exception with its own traceback.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic_evals import Dataset
from pydantic_evals.reporting import EvaluationReport


async def evaluate_or_raise[InputsT, OutputT, MetadataT](
    dataset: Dataset[InputsT, OutputT, MetadataT],
    task: Callable[[InputsT], Awaitable[OutputT]],
    *,
    name: str,
    max_concurrency: int,
) -> EvaluationReport[InputsT, OutputT, MetadataT]:
    """Evaluate ``dataset`` with ``task``, raising at the first case whose task raises.

    Args:
        dataset: The dataset, with its evaluators.
        task: Runs one case.
        name: The evaluation's name.
        max_concurrency: Cases run at once.

    Returns:
        The report, when no task and no evaluator raised.

    Raises:
        Exception: The first exception a task raised; the other cases are cancelled.
        RuntimeError: An evaluator raised. Each failure's traceback is printed first.
    """
    first: list[BaseException] = []
    evaluation: asyncio.Future[Any] | None = None

    async def guarded(inputs: InputsT) -> OutputT:
        """Run ``task``, cancelling the evaluation the first time a task raises."""
        try:
            return await task(inputs)
        except Exception as exc:
            if not first:
                first.append(exc)
                if evaluation is not None:
                    evaluation.cancel()
            raise

    evaluation = asyncio.ensure_future(
        dataset.evaluate(guarded, name=name, max_concurrency=max_concurrency, progress=False)
    )
    try:
        report = await evaluation
    except asyncio.CancelledError:
        if first:
            raise first[0] from None
        raise
    if first:
        raise first[0]
    if report.failures:
        messages = [f"{failure.name}: {failure.error_message}" for failure in report.failures]
        raise RuntimeError(f"{len(messages)} case(s) failed outside their task: {messages}")
    evaluator_failures = [
        (case.name, failure) for case in report.cases for failure in case.evaluator_failures
    ]
    if evaluator_failures:
        for case_name, failure in evaluator_failures:
            print(
                f"evaluator crashed on case {case_name}:\n{failure.error_stacktrace}",
                file=sys.stderr,
            )
        messages = [
            f"{case_name}: {failure.error_message}" for case_name, failure in evaluator_failures
        ]
        raise RuntimeError(f"the evaluator crashed on {len(messages)} case(s): {messages}")
    return report
