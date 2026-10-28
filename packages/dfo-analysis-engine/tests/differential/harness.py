"""Runs a `cases.generate_cases()` case through a verb engine and serialises
the whole outcome, for `engine_cases.jsonl` and its replay test.

`IMPLEMENTATIONS` maps each verb to the engine functions under test. Every
entry takes the verb's input frames positionally and the case's `params` as
keyword arguments, the signature of today's `run_<verb>` functions. Adding a
second function to a verb's tuple (an Arrow engine beside the row engine)
replays every case of that verb against both.

An outcome is one of:

- `{"result": ...}`: every field of the engine's result dataclass, the output
  frame as its `frame_sha256`, columns and canonical rows, a `UnitRequest`
  as `{"unit_request": {...}}`, and join's `MatchedKeys` as the list of its
  `MatchedKey` rows.
- `{"refusal": {"code", "message", "fix"}}`: an engine exception the verb
  refuses, mapped by `dfo_analysis_engine.refusal.refusal_for_engine_error`
  and read back from the JSON of `Refusal.as_json`. The mapping is tried
  before the `FrameError` check, so an exception the verb refuses is a
  refusal whatever its type.
- `{"frame_error": category}`: a `FrameError` from building the output frame.
  A service turns it into `internal_error`; only its category is compared.
- `{"crash": {"stage", "type"}}`: any other exception, from the engine
  (`"engine"`) or from mapping the engine exception to a refusal and
  rendering it (`"refusal"`, such as a `TypeError` from a `fix` that holds
  no JSON form). Only the type is kept, since a message's operand order can
  follow set iteration order.

Record the fixture again, from `packages/dfo-analysis-engine/tests`, with
`uv run python -m differential.harness`. Do so only when an engine change is
meant to move an outcome, and review the diff.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from dfo_frame_grammar import Frame, FrameError, canonical_frame_bytes, frame_sha256
from dfo_frame_grammar.grammar import UnitRequest
from differential.cases import Case, frame_from_spec, generate_cases
from pydantic import TypeAdapter

from dfo_analysis_engine.engine.aggregate import run_aggregate
from dfo_analysis_engine.engine.compare import run_compare
from dfo_analysis_engine.engine.filter import run_filter
from dfo_analysis_engine.engine.growth import run_growth
from dfo_analysis_engine.engine.join import MatchedKeys, run_join
from dfo_analysis_engine.engine.rank import run_rank
from dfo_analysis_engine.engine.share_of import run_share_of
from dfo_analysis_engine.op_spec import Predicate
from dfo_analysis_engine.refusal import refusal_for_engine_error

FIXTURE_PATH = Path(__file__).with_name("engine_cases.jsonl")

type Engine = Callable[..., object]

IMPLEMENTATIONS: dict[str, tuple[Engine, ...]] = {
    "aggregate": (run_aggregate,),
    "compare": (run_compare,),
    "growth": (run_growth,),
    "rank": (run_rank,),
    "share_of": (run_share_of,),
    "join": (run_join,),
    "filter": (run_filter,),
}

# How many input frames each verb's engine takes positionally. A two-input
# case with one frame spec passes that frame as both sides.
_ARITY = {
    "aggregate": 1,
    "compare": 2,
    "growth": 1,
    "rank": 1,
    "share_of": 2,
    "join": 2,
    "filter": 1,
}

_PREDICATES = TypeAdapter(list[Predicate])


def implementation_label(engine: Engine) -> str:
    """The dotted name a replay test id shows for `engine`."""
    return f"{engine.__module__}.{engine.__qualname__}"


def input_frames(case: Case) -> list[Frame]:
    """The frames `case` passes to its engine, one per positional input."""
    frames = [frame_from_spec(spec) for spec in case["frames"]]
    if _ARITY[case["verb"]] == 2:
        return [frames[0], frames[-1]]
    return frames


def _engine_kwargs(case: Case) -> dict[str, Any]:
    params = dict(case["params"])
    if case["verb"] == "filter":
        params["predicates"] = _PREDICATES.validate_python(params["predicates"])
    return params


def to_json(value: object) -> object:
    """`value` as JSON: see the module docstring for the shapes."""
    if isinstance(value, MatchedKeys):
        value = value.rows()
    if isinstance(value, Frame):
        document = json.loads(canonical_frame_bytes(value))
        return {
            "frame_sha256": frame_sha256(value),
            "columns": document["columns"],
            "rows": document["rows"],
        }
    if isinstance(value, UnitRequest):
        return {"unit_request": value.as_json()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_json(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"no JSON form for {type(value)!r}")


def outcome_of_exception(case: Case, exc: Exception) -> dict[str, object]:
    """The outcome of `exc`, an exception `case`'s engine raised: the refusal
    the verb gives for it, else a `frame_error`, else a `crash`."""
    try:
        refusal = refusal_for_engine_error(case["verb"], exc)
        emitted = None if refusal is None else json.loads(json.dumps(refusal.as_json()))
    except Exception as mapping_exc:
        return {"crash": {"stage": "refusal", "type": type(mapping_exc).__name__}}
    if emitted is not None:
        return {"refusal": {key: emitted[key] for key in ("code", "message", "fix")}}
    if isinstance(exc, FrameError):
        return {"frame_error": exc.category}
    return {"crash": {"stage": "engine", "type": type(exc).__name__}}


def run_case(case: Case, engine: Engine) -> dict[str, object]:
    """`engine`'s outcome on `case`, serialised."""
    frames = input_frames(case)
    try:
        result = engine(*frames, **_engine_kwargs(case))
    except Exception as exc:
        return outcome_of_exception(case, exc)
    return {"result": to_json(result)}


def record_case(case: Case) -> dict[str, object]:
    """`case` plus its input frames' `frame_sha256` and the outcome of the
    first engine `IMPLEMENTATIONS` lists for its verb."""
    engine = IMPLEMENTATIONS[case["verb"]][0]
    return {
        **case,
        "input_frame_sha256": [frame_sha256(frame_from_spec(spec)) for spec in case["frames"]],
        "outcome": run_case(case, engine),
    }


def dumps(value: object) -> str:
    """The one-line JSON form each fixture line holds."""
    return json.dumps(value, separators=(",", ":"))


def load_fixture(path: Path = FIXTURE_PATH) -> list[dict[str, Any]]:
    """Every recorded case, in file order."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_fixture(records: Sequence[Mapping[str, object]], path: Path = FIXTURE_PATH) -> None:
    path.write_text("".join(f"{dumps(record)}\n" for record in records))


def main() -> None:
    records = [record_case(case) for case in generate_cases()]
    write_fixture(records)
    print(f"wrote {len(records)} cases to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
