"""Every Spec a response carries, read from the raw body.

A definite part carries its spec, a candidates part each candidate, and a no-data part its
``nearest_real`` suggestions, ``flagged_match`` and ``alternate_match``. Each spec may add
``resolved.equivalent_readings``, whose twins carry no twins of their own. This is the set
``search.engine.envelope.iter_response_specs`` walks to build the envelope's skills map.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from search.eval.checks.base import PartView, ScoreContext, join, read_path, response_parts


@dataclass(frozen=True)
class SpecView:
    at: str
    """The schema path of ``body``."""
    pointer: str
    """The JSON pointer of ``body`` in the response, for reports."""
    body: Mapping[str, Any]

    def read(self, path: str, ctx: ScoreContext) -> Any:
        return read_path(self.body, path, ctx, at=self.at)


def part_specs(part: PartView, ctx: ScoreContext, *, twins: bool = True) -> Iterator[SpecView]:
    """The specs ``part`` carries, each followed by its equivalent readings when ``twins``."""
    at, pointer = PartView.path, f"/parts/{part.index}"
    found: list[SpecView | None] = []
    match part.status:
        case "definite":
            found.append(_view(at, pointer, "spec", part.body, ctx))
        case "candidates":
            specs = read_path(part.body, "candidates.specs", ctx, at=at) or []
            found.extend(
                SpecView(join(at, "candidates.specs[]"), f"{pointer}/candidates/specs/{i}", spec)
                for i, spec in enumerate(specs)
            )
        case "no_data":
            nearest = read_path(part.body, "no_data.nearest_real", ctx, at=at) or []
            found.extend(
                SpecView(
                    join(at, "no_data.nearest_real[]"), f"{pointer}/no_data/nearest_real/{i}", s
                )
                for i, s in enumerate(nearest)
            )
            found.extend(
                _view(at, pointer, f"no_data.{name}", part.body, ctx)
                for name in ("flagged_match", "alternate_match")
            )
    for spec in found:
        if spec is None:
            continue
        yield spec
        if twins:
            yield from _twins(spec, ctx)


def response_specs(
    response: Mapping[str, Any], ctx: ScoreContext, *, twins: bool = True
) -> Iterator[SpecView]:
    """Every spec across the response's parts (``part_specs``)."""
    for part in response_parts(response, ctx):
        yield from part_specs(part, ctx, twins=twins)


def _view(
    at: str, pointer: str, path: str, body: Mapping[str, Any], ctx: ScoreContext
) -> SpecView | None:
    """The spec at ``path`` under ``body``, or None when the body omits it."""
    spec = read_path(body, path, ctx, at=at)
    if spec is None:
        return None
    return SpecView(join(at, path), f"{pointer}/{path.replace('.', '/')}", spec)


def _twins(spec: SpecView, ctx: ScoreContext) -> Iterator[SpecView]:
    for i, twin in enumerate(spec.read("resolved.equivalent_readings", ctx) or []):
        yield SpecView(
            join(spec.at, "resolved.equivalent_readings[]"),
            f"{spec.pointer}/resolved/equivalent_readings/{i}",
            twin,
        )
