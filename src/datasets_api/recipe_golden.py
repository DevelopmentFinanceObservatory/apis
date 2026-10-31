"""Recorded recipes with their root and output frames, for a test that the runner reproduces
each output.

A case is one directory:

- `case.json`: `{"origin", "spec_id", "content_version", "recipe", "roots", "expected"}`.
  `origin` names where the case was recorded (`dfo-dev`, or `synthetic` for one built from the
  engine's own fixture frames). `recipe` is the stored recipe body. `roots` lists each root
  record the recipe's tree names, in order, with `file`, its frame's path in the case. `expected`
  holds the stored output frame's `frame_content_version` and `output_digest`
  (`recipe_replay.frame_facts`), its operations' variables named over the recorded roots.
- `roots/<i>.parquet`: each root frame as the runner read it, in the canonical frame encoding.
- `output.parquet`: the stored output frame.

`record_case` writes a case from the stores, read-only, through the same reads `diff-recipes`
makes (`recipe_replay.StoredRootSource`), and skips a recipe whose frames together pass
`max_bytes`. `FixtureRootSource` reads a case's roots back as an `analysis.RootSource`, and
`replay_case` computes its recipe through the runner (`analysis.compute_analysis`).
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dfo_frame_grammar import Frame

from datasets_api.analysis import (
    AnalysisBounds,
    AnalysisRequest,
    DatasetRoot,
    DerivedRoot,
    ResolvedRoot,
    RootRendering,
    SpecRoot,
    compute_analysis,
    plan_roots,
    resolve_roots,
)
from datasets_api.derived import root_rendering
from datasets_api.output_digest import minted_variable_names, stat_vars
from datasets_api.recipe_replay import StoredRootSource, frame_facts
from dfo_analysis_engine.frame_ref import FrameRef
from materialise.formats import read_canonical_frame, write_canonical_frame

CASE_FILE = "case.json"
OUTPUT_FILE = "output.parquet"
DEFAULT_MAX_BYTES = 2_000_000
_BOUNDS = AnalysisBounds()


def _frame_bytes(frame: Frame) -> bytes:
    sink = io.BytesIO()
    write_canonical_frame((frame,), sink=sink)
    return sink.getvalue()


def record_case(
    *,
    spec_id: str,
    content_version: str,
    source: StoredRootSource,
    out: Path,
    origin: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> bool:
    """Write the case of the stored recipe of `{spec_id, content_version}` under `out`, and
    return whether it was written: `False` when the frame holds no recipe, a frame is no longer
    held, or the frames pass `max_bytes`.

    Raises:
        FrameUnavailable: A root cannot be read.
    """
    recipe = source.recipe(spec_id=spec_id, content_version=content_version)
    output = source.stored_output(spec_id=spec_id, content_version=content_version)
    if recipe is None or output is None:
        return False
    request = AnalysisRequest(tree=DerivedRoot(spec_id=spec_id, content_version=content_version))
    _, roots = plan_roots(request, source=source, bounds=_BOUNDS)
    resolved = resolve_roots(roots, source=source, latest=False, trace_id=None)
    frames = [source.read(root) for root in resolved]
    files = [_frame_bytes(frame) for frame in frames]
    output_bytes = _frame_bytes(output)
    if sum(map(len, files)) + len(output_bytes) > max_bytes:
        return False
    out.mkdir(parents=True, exist_ok=True)
    (out / "roots").mkdir(exist_ok=True)
    for index, data in enumerate(files):
        (out / "roots" / f"{index}.parquet").write_bytes(data)
    (out / OUTPUT_FILE).write_bytes(output_bytes)
    case = {
        "origin": origin,
        "spec_id": spec_id,
        "content_version": content_version,
        "recipe": recipe,
        "roots": [
            {**root.record, "file": f"roots/{index}.parquet"} for index, root in enumerate(resolved)
        ],
        "expected": frame_facts(
            output,
            _names(
                recipe, [(root.record, frame) for root, frame in zip(resolved, frames, strict=True)]
            ),
        ),
    }
    (out / CASE_FILE).write_text(json.dumps(case, indent=1, sort_keys=True) + "\n")
    return True


def _names(
    recipe: Mapping[str, Any], roots: Sequence[tuple[Mapping[str, Any], Frame]]
) -> dict[str, str]:
    """The names of the variables `recipe`'s operations wrote, over `roots`, each root record
    with its frame. A tree's root index is its record's place in the recipe's `roots`, which
    the order roots are read in need not follow."""
    variables = {
        index: next(stat_vars(frame) for held, frame in roots if held == record)
        for index, record in enumerate(recipe["roots"])
    }
    return minted_variable_names(recipe["tree"], variables)


@dataclass(kw_only=True)
class FixtureRootSource:
    """`analysis.RootSource` over one recorded case's root frames."""

    case: Path
    body: Mapping[str, Any]
    _seen: list[tuple[DatasetRoot, Mapping[str, Any]]] = field(default_factory=list)

    @classmethod
    def load(cls, case: Path) -> FixtureRootSource:
        return cls(case=case, body=json.loads((case / CASE_FILE).read_text()))

    def recipe(self, *, spec_id: str, content_version: str) -> Mapping[str, Any] | None:
        body = self.body
        if (spec_id, content_version) != (body["spec_id"], body["content_version"]):
            return None
        return body["recipe"]

    def dataset_root(self, record: Mapping[str, Any]) -> DatasetRoot:
        options = record.get("options") or {}
        root = DatasetRoot(
            spec_id=record["dataset"]["spec_id"],
            content_version=record["dataset"]["content_version"],
            rendering=RootRendering.of_options(options),
        )
        held = next(held for held in self.body["roots"] if held["frame"] == record["frame"])
        self._seen.append((root, held))
        return root

    def resolve(
        self, root: SpecRoot | DatasetRoot, *, latest: bool, trace_id: str | None
    ) -> ResolvedRoot:
        held = next(held for seen, held in self._seen if seen == root)
        record = {key: value for key, value in held.items() if key != "file"}
        return ResolvedRoot(
            frame=FrameRef.model_validate(record["frame"]),
            dataset=FrameRef.model_validate(record["dataset"]),
            record=record,
            rendering=root_rendering(record.get("options") or {}),
            source=held["file"],
        )

    def locate(self, root: ResolvedRoot) -> ResolvedRoot:
        """`root` as it is. A case file names its frame as it resolves."""
        return root

    def read(self, root: ResolvedRoot) -> Frame:
        assert isinstance(root.source, str)
        return read_canonical_frame((self.case / root.source).read_bytes())


def recorded_facts(case: Path) -> dict[str, str]:
    """The output facts of `case`'s recorded output frame, its variables named over its
    recorded roots."""
    body = json.loads((case / CASE_FILE).read_text())
    roots = [
        (
            {key: value for key, value in held.items() if key != "file"},
            read_canonical_frame((case / held["file"]).read_bytes()),
        )
        for held in body["roots"]
    ]
    output = read_canonical_frame((case / OUTPUT_FILE).read_bytes())
    return frame_facts(output, _names(body["recipe"], roots))


def replay_case(case: Path) -> tuple[dict[str, str], dict[str, str]]:
    """The recorded output facts of `case` and those the runner computes from its roots."""
    source = FixtureRootSource.load(case)
    body = source.body
    computed = compute_analysis(
        AnalysisRequest(
            tree=DerivedRoot(spec_id=body["spec_id"], content_version=body["content_version"])
        ),
        source=source,
    )
    facts = {
        "frame_content_version": computed.evaluation.summary["frame_content_version"],
        "output_digest": computed.output_digest,
    }
    return dict(body["expected"]), facts


def cases(directory: Path) -> Sequence[Path]:
    """Every case under `directory`, in name order."""
    return sorted(path.parent for path in directory.glob(f"*/{CASE_FILE}"))
