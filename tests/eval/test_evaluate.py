"""Evaluating a dataset that stops at the first crash."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from search.eval.evaluate import evaluate_or_raise


def dataset(*names: str) -> Dataset[str, str, None]:
    """A dataset whose cases' inputs are their names."""
    return Dataset[str, str, None](name="t", cases=[Case(name=n, inputs=n) for n in names])


def test_the_first_task_that_raises_cancels_the_others_and_is_raised():
    finished: list[str] = []

    async def task(inputs: str) -> str:
        if inputs == "bad":
            raise KeyError("a bug")
        await asyncio.sleep(5)
        finished.append(inputs)
        return inputs

    async def run() -> None:
        await evaluate_or_raise(
            dataset("slow-1", "bad", "slow-2"), task, name="t", max_concurrency=3
        )

    with pytest.raises(KeyError, match="a bug"):
        asyncio.run(asyncio.wait_for(run(), timeout=2))
    assert finished == []


def test_a_clean_run_returns_the_report():
    async def task(inputs: str) -> str:
        return inputs.upper()

    report = asyncio.run(evaluate_or_raise(dataset("a", "b"), task, name="t", max_concurrency=2))

    assert sorted(case.output for case in report.cases) == ["A", "B"]


@dataclass
class Broken(Evaluator[str, str, None]):
    """An evaluator that always raises."""

    def evaluate(self, ctx: EvaluatorContext[str, str, None]) -> bool:
        """Raise."""
        raise ValueError("evaluator bug")


def test_an_evaluator_that_raises_is_a_crash_with_its_traceback(capsys):
    async def task(inputs: str) -> str:
        return inputs

    data = dataset("a")
    data.add_evaluator(Broken())

    with pytest.raises(RuntimeError, match="the evaluator crashed on 1 case"):
        asyncio.run(evaluate_or_raise(data, task, name="t", max_concurrency=1))
    assert "evaluator bug" in capsys.readouterr().err
