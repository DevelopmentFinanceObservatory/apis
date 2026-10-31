"""The analysis runner against the client contract's `operation-outputs` vectors.

Each vector names a verb, its arguments in the client's words, the rows of `data` and of
`other`, and either the output rows or the refusal code. The runner computes the step over the
two readings as `POST /api/datasets/analysis` does (`compute_analysis`), so `data` and `other`
fill the verb's slots and `on` names its pairing as the door reads them. Rows compare in any
order, and a row of an operation that writes under its own variable leaves `stat_var` out.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from dfo_frame_grammar import Frame

from datasets_api.analysis import AnalysisRequest, compute_analysis
from dfo_analysis_engine.refusal import AnalysisRefused
from materialise.frame_build import build_frame
from tests.spec_kit_bundle import spec_kit_dir

from ._analysis_doubles import AID, AID_V1, POPULATION, POPULATION_V1, FakeRootSource, dataset

_VECTORS = (
    spec_kit_dir() / "python/client-contract/src/dfo_client_contract/rules/operation-outputs.json"
)
_MEASURES = ("stat_var", "time_period", "value", "unit")
# Columns a frame row carries and the contract's rows leave out, as the output digest does.
_UNCOMPARED = frozenset({"measurementMethod", "provenance", "status"})


def _vectors() -> list[dict[str, Any]]:
    if _VECTORS.is_file():
        return json.loads(_VECTORS.read_text())["vectors"]
    if os.environ.get("DFO_SPEC_KIT_REQUIRED"):
        raise RuntimeError(
            f"DFO_SPEC_KIT_REQUIRED is set but {_VECTORS} is missing -- the spec-kit checkout "
            "is missing, or the client contract's rules have moved"
        )
    pytest.skip(
        "client contract rules not found -- set DFO_SPEC_KIT_DIR or check out dfo-spec-kit "
        "as a sibling of dfo-apis",
        allow_module_level=True,
    )
    return []


def _frame(rows: list[dict[str, Any]]) -> Frame:
    role_keys = list(dict.fromkeys(key for row in rows for key in row if key not in _MEASURES))
    return build_frame(role_keys=role_keys, rows=rows)


def _arguments(verb: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """The client's `by` and `window` as the op-spec parameters they name; `on` and `change`
    pass as they are."""
    out = dict(arguments)
    if "window" in out:
        out["window_start"], out["window_end"] = out.pop("window")
    if "by" in out:
        out["entity_by" if verb == "growth" else "numerator_by"] = out.pop("by")
    return out


def _output(vector: dict[str, Any]) -> list[dict[str, Any]]:
    frames = {(AID, AID_V1): _frame(vector["data"])}
    inputs: dict[str, Any] = {"data": dataset(AID, AID_V1)}
    if "other" in vector:
        frames[(POPULATION, POPULATION_V1)] = _frame(vector["other"])
        inputs["other"] = dataset(POPULATION, POPULATION_V1)
    source = FakeRootSource(frames=frames, newest={AID: AID_V1, POPULATION: POPULATION_V1})
    step = {
        "verb": vector["verb"],
        "arguments": _arguments(vector["verb"], vector["arguments"]),
        "inputs": inputs,
    }
    computed = compute_analysis(AnalysisRequest.model_validate({"tree": step}), source=source)
    frame = computed.evaluation.frame
    return [dict(zip(frame.columns, row, strict=True)) for row in frame.rows]


def _rows(rows: list[dict[str, Any]], *, own_variable: bool) -> list[str]:
    left_out = _UNCOMPARED | ({"stat_var"} if own_variable else set())
    return sorted(
        json.dumps({key: cell for key, cell in row.items() if key not in left_out}, sort_keys=True)
        for row in rows
    )


@pytest.mark.parametrize("vector", _vectors(), ids=lambda vector: vector["name"])
def test_the_runner_answers_each_operation_output_vector(vector: dict[str, Any]) -> None:
    if "error" in vector:
        with pytest.raises(AnalysisRefused) as refused:
            _output(vector)
        assert refused.value.code == vector["error"]
        return
    own_variable = any("stat_var" not in row for row in vector["rows"])
    assert _rows(_output(vector), own_variable=own_variable) == _rows(
        vector["rows"], own_variable=own_variable
    )
