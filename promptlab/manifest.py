"""Typed rung manifests and the loader that validates stage files against `rung.schema.json`.

Each rung has extract and bind manifests. The bind manifest holds the shared run configuration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator

PROMPTLAB_ROOT = Path(__file__).resolve().parent
_SCHEMA_PATH = PROMPTLAB_ROOT / "rung.schema.json"


class ManifestError(ValueError):
    """A rung manifest fails schema validation or names a missing block file."""


@dataclass(frozen=True)
class BlockRef:
    """A stage block. `sep` precedes it except at the start.

    `unit` groups blocks. `core` excludes the block from ablation.
    """

    id: str
    sep: Literal["blank", "line"] = "blank"
    unit: str | None = None
    core: bool = False


@dataclass(frozen=True)
class StageSpec:
    """One stage's ordered block list."""

    stage: Literal["extract", "bind"]
    blocks: tuple[BlockRef, ...]


@dataclass(frozen=True)
class RunConfig:
    """The run configuration shared by both stages of a rung, read from the
    bind stage's manifest file."""

    capture: str
    top_k: int
    seeds: tuple[int, ...]
    model: str
    thinking: Mapping[str, str | None]


@dataclass(frozen=True)
class Rung:
    """A named, fully loaded prompt ladder rung: both stages plus the run
    configuration. `name` and `description` are the bind manifest's own
    fields, since the bind file is the run configuration's single home."""

    name: str
    description: str
    extract: StageSpec
    bind: StageSpec
    run: RunConfig


def block_path(stage: str, block_id: str, root: Path) -> Path:
    """The on-disk path a block id resolves to for the given stage and root."""
    return root / "blocks" / stage / f"{block_id}.md"


def require_block_files(stage: StageSpec, *, root: Path) -> None:
    """Raises `ManifestError` naming the first block id whose file is missing."""
    for block in stage.blocks:
        path = block_path(stage.stage, block.id, root)
        if not path.is_file():
            raise ManifestError(f"unknown block file: {block.id!r} ({path})")


def _validator() -> Validator:
    schema = json.loads(_SCHEMA_PATH.read_text())
    return Draft202012Validator(schema)


def _load_stage_data(path: Path) -> dict:
    data = json.loads(path.read_text())
    errors = sorted(_validator().iter_errors(data), key=lambda e: list(e.path))
    if errors:
        messages = "; ".join(e.message for e in errors)
        raise ManifestError(f"{path}: {messages}")
    return data


def _stage_spec(data: dict) -> StageSpec:
    blocks = tuple(
        BlockRef(
            id=b["id"],
            sep=b.get("sep", "blank"),
            unit=b.get("unit"),
            core=b.get("core", False),
        )
        for b in data["blocks"]
    )
    return StageSpec(stage=data["stage"], blocks=blocks)


def load_rung(extract_path: Path, bind_path: Path, *, root: Path = PROMPTLAB_ROOT) -> Rung:
    """Validates both manifests and their block files.

    Raises `ManifestError` if a block file is missing.
    """
    extract_data = _load_stage_data(extract_path)
    bind_data = _load_stage_data(bind_path)

    if extract_data["stage"] != "extract":
        raise ManifestError(
            f"{extract_path}: stage must be 'extract', got {extract_data['stage']!r}"
        )
    if bind_data["stage"] != "bind":
        raise ManifestError(f"{bind_path}: stage must be 'bind', got {bind_data['stage']!r}")
    if extract_data["name"] != bind_data["name"]:
        raise ManifestError(
            f"{extract_path} names rung {extract_data['name']!r} but {bind_path} names "
            f"{bind_data['name']!r} -- a rung's two stage files must share one name"
        )

    extract_spec = _stage_spec(extract_data)
    bind_spec = _stage_spec(bind_data)
    require_block_files(extract_spec, root=root)
    require_block_files(bind_spec, root=root)

    run_data = bind_data["run"]
    run = RunConfig(
        capture=run_data["capture"],
        top_k=run_data["top_k"],
        seeds=tuple(run_data["seeds"]),
        model=run_data["model"],
        thinking=dict(run_data["thinking"]),
    )

    return Rung(
        name=bind_data["name"],
        description=bind_data["description"],
        extract=extract_spec,
        bind=bind_spec,
        run=run,
    )


def units(stage: StageSpec) -> dict[str, tuple[BlockRef, ...]]:
    """Groups blocks by unit in first-appearance order.

    Blocks without a unit use their id.
    """
    grouped: dict[str, list[BlockRef]] = {}
    for block in stage.blocks:
        key = block.unit or block.id
        grouped.setdefault(key, []).append(block)
    return {key: tuple(refs) for key, refs in grouped.items()}
