"""The eligible corpus as one Pydantic Evals dataset.

Each eligible case becomes one ``pydantic_evals.Case`` named by its case id. Its inputs are
the request as the case file authors it (``CaseInputs``), which a task turns into the HTTP
request it posts. The inputs carry the case id because two cases in different files may
author the same request, and a Pydantic Evals task receives only its inputs. The metadata
(``CorpusCase``) carries the validated case, its file and its ``expectation_hash``, which
evaluators and run records read. Eligibility and ``expectation_hash`` are ``search.eval.cases``'s,
unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Final

from pydantic import BaseModel, ConfigDict
from pydantic_evals import Case as EvalCase
from pydantic_evals import Dataset

from search.eval.cases import Case, Resubmit, expectation_hash, ineligibility
from search.models import EntryPath, ParsedPart

SUITE: Final = "apis-search"
_DIGEST_PREFIX: Final = "sha256:"


class CaseInputs(BaseModel):
    """A case's request as its case file authors it, keyed by the case id."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    query: str
    door: EntryPath
    parsed: list[ParsedPart] | None
    resubmit: Resubmit | None
    include_observations: bool


class CorpusCase(BaseModel):
    """What evaluators and run records read about a case beside its inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case: Case
    file: str
    expectation_hash: str

    @property
    def source(self) -> str:
        """The case file's path under the repository root, such as ``corpus/dac.yaml``."""
        return f"corpus/{self.file}.yaml"


type CorpusDataset = Dataset[CaseInputs, Any, CorpusCase]


def case_inputs(case: Case) -> CaseInputs:
    """Return a case's authored request."""
    return CaseInputs(
        case_id=case.id,
        query=case.query,
        door=case.door,
        parsed=case.parsed,
        resubmit=case.resubmit,
        include_observations=case.include_observations,
    )


def corpus_dataset(
    by_file: Mapping[str, Sequence[Case]], *, implemented_checks: Collection[str]
) -> CorpusDataset:
    """Build one dataset of every eligible case, in file then list order.

    Args:
        by_file: Cases keyed by case file stem, as ``load_case_files`` returns them.
        implemented_checks: Names of the checks with an evaluator, which decide eligibility
            (``ineligibility``); callers pass ``search.eval.checks.base.CHECKS``.

    Returns:
        The dataset, named ``apis-search``, with no evaluators. Each case's expected output
        is its ``expect`` block.

    Raises:
        ValueError: A case id appears twice (``Dataset`` refuses duplicate names).
    """
    cases: list[EvalCase[CaseInputs, Any, CorpusCase]] = []
    for stem, file_cases in by_file.items():
        for case in file_cases:
            if ineligibility(case, implemented_checks=implemented_checks):
                continue
            cases.append(
                EvalCase[CaseInputs, Any, CorpusCase](
                    name=case.id,
                    inputs=case_inputs(case),
                    expected_output=case.expect,
                    metadata=CorpusCase(
                        case=case, file=stem, expectation_hash=expectation_hash(case)
                    ),
                )
            )
    return Dataset[CaseInputs, Any, CorpusCase](name=SUITE, cases=cases)


def corpus_digest(dataset: CorpusDataset) -> str:
    """Digest the dataset's cases by id and ``expectation_hash``, for a run's manifest.

    Two runs share a digest exactly when they ran the same cases with the same requests
    and expectations, so a change in rate between them reflects the system.

    Raises:
        ValueError: A case has no metadata.
    """
    pairs = []
    for eval_case in dataset.cases:
        if eval_case.metadata is None:
            raise ValueError(f"case {eval_case.name!r} has no corpus metadata")
        pairs.append([eval_case.inputs.case_id, eval_case.metadata.expectation_hash])
    canonical = json.dumps(sorted(pairs), separators=(",", ":"))
    return _DIGEST_PREFIX + hashlib.sha256(canonical.encode()).hexdigest()
