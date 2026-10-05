"""``groundedness``: every graph reference the response makes names a node in the graph.

Walks every spec (``specs.response_specs``) and reads, under ``resolved``:

- ``shape.{population_type, measured_property, stat_type, measurement_qualifier,
  measurement_denominator}.dcid``;
- ``slots[].binding.value.ref.dcid`` and ``slots[].binding.values[].ref.dcid``;
- ``stat_vars.items[].ref``;
- ``entities[].ref`` and ``entities[].entity_type``;
- ``scope.members[].ref`` and ``scope.members[].entity_type``.

One ``ctx.graph.exists_batch`` call checks them all. A failed read (``base.READ_ERRORS``)
is an infrastructure result (``base.infra_error``). Any other error propagates. A response
naming no reference is n/a.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from search.eval.cases import Case
from search.eval.checks.base import (
    NA,
    PASS,
    READ_ERRORS,
    Result,
    ScoreContext,
    fail,
    infra_error,
    read_path,
    register,
)
from search.eval.checks.specs import SpecView, response_specs

_SHAPE_FIELDS = (
    "population_type",
    "measured_property",
    "stat_type",
    "measurement_qualifier",
    "measurement_denominator",
)


def _entity_refs(
    entities: list[Mapping[str, Any]] | None, at: str, pointer: str, ctx: ScoreContext
) -> Iterator[tuple[str | None, str]]:
    for i, entity in enumerate(entities or []):
        for name in ("ref", "entity_type"):
            yield read_path(entity, name, ctx, at=at), f"{pointer}/{i}/{name}"


def _spec_refs(spec: SpecView, ctx: ScoreContext) -> Iterator[tuple[str | None, str]]:
    """Each graph reference in ``spec`` with its JSON pointer; absent ones read None."""
    resolved_at, resolved = f"{spec.at}.resolved", f"{spec.pointer}/resolved"
    for name in _SHAPE_FIELDS:
        yield spec.read(f"resolved.shape.{name}.dcid", ctx), f"{resolved}/shape/{name}/dcid"

    slot_at = f"{resolved_at}.slots[]"
    for i, slot in enumerate(spec.read("resolved.slots", ctx) or []):
        binding = f"{resolved}/slots/{i}/binding"
        yield (
            read_path(slot, "binding.value.ref.dcid", ctx, at=slot_at),
            f"{binding}/value/ref/dcid",
        )
        values = read_path(slot, "binding.values", ctx, at=slot_at) or []
        for j, value in enumerate(values):
            dcid = read_path(value, "ref.dcid", ctx, at=f"{slot_at}.binding.values[]")
            yield dcid, f"{binding}/values/{j}/ref/dcid"

    items = spec.read("resolved.stat_vars.items", ctx) or []
    for i, item in enumerate(items):
        ref = read_path(item, "ref", ctx, at=f"{resolved_at}.stat_vars.items[]")
        yield ref, f"{resolved}/stat_vars/items/{i}/ref"

    yield from _entity_refs(
        spec.read("resolved.entities", ctx),
        f"{resolved_at}.entities[]",
        f"{resolved}/entities",
        ctx,
    )
    yield from _entity_refs(
        spec.read("resolved.scope.members", ctx),
        f"{resolved_at}.scope.members[]",
        f"{resolved}/scope/members",
        ctx,
    )


@register("groundedness")
def groundedness(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    """Every graph reference in every spec exists in ``ctx.graph``."""
    references = [
        (dcid, pointer)
        for spec in response_specs(response, ctx)
        for dcid, pointer in _spec_refs(spec, ctx)
        if dcid is not None
    ]
    if not references:
        return NA
    if ctx.graph is None:
        return infra_error("groundedness needs a graph client")
    try:
        known = ctx.graph.exists_batch(list(dict.fromkeys(dcid for dcid, _ in references)))
    except READ_ERRORS as exc:
        return infra_error(f"graph exists_batch: {type(exc).__name__}: {exc}")
    fabricated = [f"{pointer}={dcid}" for dcid, pointer in references if dcid not in known]
    if fabricated:
        return fail(
            f"{len(fabricated)} of {len(references)} references not in the graph: {fabricated}"
        )
    return PASS
