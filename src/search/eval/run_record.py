"""The ``dfo-evals`` run record of an ``apis-search`` run, shared by single mode and the nightly.

A run evaluates the corpus dataset (``search.eval.corpus_dataset``) with a task that returns
each case's modal sample as a ``ScoredAnswer``. Its record holds one trial per case, tagged
with the case file, scored by the ``answer`` check, and carrying the trace id the sample's
response holds. A sample that failed on infrastructure, or whose check could not read the
graph or coverage, records an infra fault, which the results page leaves out of its scores.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

from dfo_evals.records import FaultClass, Run, RunManifest, SafeName
from dfo_evals.report import AnyReportCase, AnyReportCaseFailure, run_from_report
from pydantic import TypeAdapter, ValidationError
from pydantic_evals.reporting import EvaluationReport

from search.eval.checks import ANSWER
from search.eval.corpus_dataset import SUITE, CorpusCase
from search.eval.evaluator import ScoredAnswer

# A run id names the run record's directory, so it follows dfo-evals' safe-name rule.
_RUN_ID: TypeAdapter[str] = TypeAdapter(SafeName)


def parse_run_id(value: str) -> str:
    """Check a ``--run-id`` is a name a run record's directory can take, as arguments parse.

    Raises:
        argparse.ArgumentTypeError: The value does not match ``dfo-evals``' safe-name rule.
    """
    try:
        return _RUN_ID.validate_python(value)
    except ValidationError as exc:
        raise argparse.ArgumentTypeError(
            f"run id {value!r} must start with a letter or digit and hold only letters, "
            "digits, '.', '_' and '-'"
        ) from exc


def record_manifest(
    *,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    revision: str,
    environment: dict[str, str],
    dataset_digest: str,
) -> RunManifest:
    """Return the manifest of an ``apis-search`` run, scored by ``answer``.

    Args:
        run_id: The run's id, unique within the suite.
        started_at: When the run started, timezone-aware.
        finished_at: When the run finished, timezone-aware.
        revision: The revision of the app the run evaluated.
        environment: Settings the run depended on, such as the app URL and sample count.
        dataset_digest: ``corpus_digest`` of the cases the run evaluated.
    """
    return RunManifest(
        suite=SUITE,
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        revision=revision,
        environment=environment,
        dataset_digest=dataset_digest,
        headline_check=ANSWER,
    )


def run_record(
    report: EvaluationReport[Any, ScoredAnswer, CorpusCase], *, manifest: RunManifest
) -> Run:
    """Return the run record of a report of modal samples, one trial per case.

    Args:
        report: The report of the corpus dataset evaluated with ``SearchChecks``.
        manifest: The run's manifest (``record_manifest``).
    """
    return run_from_report(
        report=report,
        manifest=manifest,
        tags_of=_case_file_tag,
        fault_of=_infra_fault,
        trace_of=_trace_id,
    )


def _case_file_tag(case: AnyReportCase | AnyReportCaseFailure) -> tuple[str, ...]:
    """Tag a case with its case file, such as ``corpus/dac.yaml``."""
    if not isinstance(case.metadata, CorpusCase):
        raise TypeError(f"case {case.name!r} has no corpus metadata")
    return (case.metadata.source,)


def _infra_fault(case: AnyReportCase | AnyReportCaseFailure) -> FaultClass | None:
    """Return an infra fault for a case whose modal sample is infra, else no fault."""
    output = getattr(case, "output", None)
    return FaultClass.INFRA if isinstance(output, ScoredAnswer) and output.infra else None


def _trace_id(case: AnyReportCase | AnyReportCaseFailure) -> str | None:
    """Return the trace id the modal sample's response carries."""
    output = getattr(case, "output", None)
    return output.trace_id if isinstance(output, ScoredAnswer) else None
