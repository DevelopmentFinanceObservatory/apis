"""Generate leave-one-out and build-upward rungs as manifest pairs.

Ablations change one stage. `write_rungs` refuses to overwrite existing pairs.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from promptlab.manifest import BlockRef, RunConfig, Rung, StageSpec, units

# Removing `05b-list-every` alone orphans its list. `06a-header` changes formatting only.
# Unit-level ablations also remove both full units.

_CONFOUNDED_SUB_BLOCKS: frozenset[str] = frozenset({"05b-list-every", "06a-header"})

_ABLATION_PREFIX_RE = re.compile(r"^abl-(?:extract|bind)-(?P<unit>.+)$")


class AblationError(ValueError):
    """A rung cannot be ablated, or a ranking cannot be built into a rung
    sequence, as asked."""


@dataclasses.dataclass(frozen=True)
class GeneratedRung:
    """One leave-one-out or build-upward rung as a manifest pair.

    `description` states the rung's composition in present tense.
    """

    name: str
    description: str
    extract: StageSpec
    bind: StageSpec


# ---------------------------------------------------------------------------
# Leave-one-out ablation
# ---------------------------------------------------------------------------


def _ablatable_units(stage: StageSpec) -> list[tuple[str, tuple[BlockRef, ...]]]:
    """Return non-core units and their blocks in first-appearance order."""
    return [
        (unit_name, refs)
        for unit_name, refs in units(stage).items()
        if not any(block.core for block in refs)
    ]


def _drop_blocks(stage: StageSpec, dropped_ids: set[str]) -> StageSpec:
    return dataclasses.replace(
        stage, blocks=tuple(b for b in stage.blocks if b.id not in dropped_ids)
    )


def unit_leave_one_out(rung: Rung) -> list[GeneratedRung]:
    """Remove each ablatable unit from its stage, leaving the sibling stage unchanged."""
    generated: list[GeneratedRung] = []
    for stage_name, stage, sibling in (
        ("extract", rung.extract, rung.bind),
        ("bind", rung.bind, rung.extract),
    ):
        for unit_name, refs in _ablatable_units(stage):
            mutated = _drop_blocks(stage, {block.id for block in refs})
            generated.append(
                GeneratedRung(
                    name=f"abl-{stage_name}-{unit_name}",
                    description=f"{rung.name}, minus the {stage_name} {unit_name!r} unit.",
                    extract=mutated if stage_name == "extract" else sibling,
                    bind=mutated if stage_name == "bind" else sibling,
                )
            )
    return generated


def sub_block_leave_one_out(rung: Rung) -> list[GeneratedRung]:
    """Remove eligible sub-blocks from multi-block bind units.

    Extract keeps the rung's composition.
    """
    generated: list[GeneratedRung] = []
    for unit_name, refs in units(rung.bind).items():
        if len(refs) < 2 or any(block.core for block in refs):
            continue
        for block in refs:
            if block.id in _CONFOUNDED_SUB_BLOCKS:
                continue
            generated.append(
                GeneratedRung(
                    name=f"abl-bind-{block.id}",
                    description=f"{rung.name}, minus the {block.id!r} sub-block of {unit_name!r}.",
                    extract=rung.extract,
                    bind=_drop_blocks(rung.bind, {block.id}),
                )
            )
    return generated


def ablate(
    rung: Rung, *, granularity: Literal["unit", "block", "both"] = "both"
) -> list[GeneratedRung]:
    generated: list[GeneratedRung] = []
    if granularity in ("unit", "both"):
        generated.extend(unit_leave_one_out(rung))
    if granularity in ("block", "both"):
        generated.extend(sub_block_leave_one_out(rung))
    if not generated:
        raise AblationError(f"rung {rung.name!r} has no ablatable unit or sub-block")
    return generated


# ---------------------------------------------------------------------------
# Build-upward
# ---------------------------------------------------------------------------


def unit_from_rung_name(rung_name: str) -> str | None:
    """Return the unit encoded in an ablation rung name, or `None` for other rungs."""
    match = _ABLATION_PREFIX_RE.match(rung_name)
    return match.group("unit") if match else None


def ranking_from_report_json(data: Mapping[str, Any]) -> dict[str, float]:
    """Map a JSON ladder report to each ablated unit's mean normalized distance
    outside ceiling bands.

    For metrics with a nonzero-width band, contribution is `max(0, lo - value, value - hi) /
    (hi - lo)`. The score averages these distances without direction adjustment, since deviation
    on either side misses the build target of falling inside the band. Metrics without a band or
    with zero-width bands are omitted. Non-ablation cells and cells with no included metrics are
    skipped.
    """
    scores: dict[str, float] = {}
    for cell in data.get("cells", []):
        unit = unit_from_rung_name(cell.get("rung_name", ""))
        if unit is None:
            continue
        deviations: list[float] = []
        for metric in cell.get("metrics", {}).values():
            band = metric.get("ceiling_band")
            if band is None:
                continue
            lo, hi = band
            width = hi - lo
            if width == 0:
                continue
            value = metric["value"]
            deviations.append(max(0.0, lo - value, value - hi) / width)
        if not deviations:
            continue
        scores[unit] = sum(deviations) / len(deviations)
    return scores


def build_upward(rung0: Rung, base: Rung, ranking: Mapping[str, float]) -> list[GeneratedRung]:
    """Start with `rung0`, then append base units in descending score order, one per rung.

    Ties sort by unit id. Each unit's blocks retain base order and separators. Unknown ranking
    keys raise `AblationError`.
    """
    extract_units = dict(_ablatable_units(base.extract))
    bind_units = dict(_ablatable_units(base.bind))
    ordered = sorted(ranking, key=lambda unit: (-ranking[unit], unit))

    unknown = [unit for unit in ordered if unit not in extract_units and unit not in bind_units]
    if unknown:
        raise AblationError(
            f"ranking names unit(s) not ablatable in base rung {base.name!r}: {unknown}"
        )

    generated = [
        GeneratedRung(
            name="build-0",
            description=f"{rung0.name}'s own composition, the build-upward sequence's floor.",
            extract=rung0.extract,
            bind=rung0.bind,
        )
    ]
    extract_blocks = rung0.extract.blocks
    bind_blocks = rung0.bind.blocks
    for i, unit in enumerate(ordered, start=1):
        if unit in extract_units:
            extract_blocks = extract_blocks + extract_units[unit]
        else:
            bind_blocks = bind_blocks + bind_units[unit]
        generated.append(
            GeneratedRung(
                name=f"build-{i}",
                description=(
                    f"{rung0.name} plus the top {i} unit(s) by measured contribution, "
                    f"most recently adding the {unit!r} unit."
                ),
                extract=dataclasses.replace(rung0.extract, blocks=extract_blocks),
                bind=dataclasses.replace(rung0.bind, blocks=bind_blocks),
            )
        )
    return generated


# ---------------------------------------------------------------------------
# Writing generated rungs to disk
# ---------------------------------------------------------------------------


def _dump_block(block: BlockRef) -> dict[str, object]:
    data: dict[str, object] = {"id": block.id}
    if block.sep != "blank":
        data["sep"] = block.sep
    if block.unit is not None:
        data["unit"] = block.unit
    if block.core:
        data["core"] = block.core
    return data


def write_rungs(
    generated: Sequence[GeneratedRung], *, out_dir: Path, run: RunConfig
) -> list[tuple[Path, Path]]:
    """Write extract/bind manifest pairs under `out_dir`.

    The bind file holds `run`. Existing pairs raise `AblationError`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[Path, Path]] = []
    for g in generated:
        extract_path = out_dir / f"{g.name}-extract.json"
        bind_path = out_dir / f"{g.name}-bind.json"
        if extract_path.exists() or bind_path.exists():
            raise AblationError(
                f"{g.name!r}: a rung already exists at {out_dir} -- remove it first"
            )
        extract_data = {
            "name": g.name,
            "description": g.description,
            "stage": "extract",
            "blocks": [_dump_block(b) for b in g.extract.blocks],
        }
        bind_data = {
            "name": g.name,
            "description": g.description,
            "stage": "bind",
            "blocks": [_dump_block(b) for b in g.bind.blocks],
            "run": {
                "capture": run.capture,
                "top_k": run.top_k,
                "seeds": list(run.seeds),
                "model": run.model,
                "thinking": dict(run.thinking),
            },
        }
        extract_path.write_text(json.dumps(extract_data, indent=2) + "\n")
        bind_path.write_text(json.dumps(bind_data, indent=2) + "\n")
        written.append((extract_path, bind_path))
    return written
