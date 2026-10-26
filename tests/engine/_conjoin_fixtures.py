"""Share `RegionResult` and shape fixtures across conjoin tests."""

from __future__ import annotations

from search.engine.conjoin import assemble_region
from search.engine.region import RegionResult
from search.engine.retrieval.bound_values import MeasurePartition, OpenAxisFact
from search.models import Spec, StatusLiteral
from tests.engine._builders import shape_five_tuple as _shape


def _region(
    spec,
    variable_text: str,
    *,
    status: StatusLiteral = "definite",
    earliest_index: int = 0,
    no_data_reason: str | None = None,
    extra_specs: tuple = (),
    gate: str | None = None,
    alternate_match: Spec | None = None,
    measure_partition: MeasurePartition | None = None,
    measure_siblings: tuple[MeasurePartition, ...] = (),
    open_axis_facts: tuple[OpenAxisFact, ...] = (),
) -> RegionResult:
    if status == "definite":
        specs: tuple = (spec,)
    elif status == "candidates":
        specs = (spec, *extra_specs)
    else:
        specs = ()
    return RegionResult(
        variable_text=variable_text,
        status=status,
        specs=specs,
        no_data_reason=no_data_reason,
        warnings=(),
        timing_by_step={},
        earliest_index=earliest_index,
        gate=gate,
        alternate_match=alternate_match,
        measure_partition=measure_partition,
        measure_siblings=measure_siblings,
        open_axis_facts=open_axis_facts,
    )


# Common shapes for reuse across tests
_PERSON_SHAPE = _shape("Person", "count")
_ECON_SHAPE = _shape("EconomicActivity", "amount", mq="Nominal")


def member_departures(region: RegionResult) -> list[tuple[str | None, str | None, str | None]]:
    """The `member_unavailable` departures of the one part `region` assembles into, each as
    its `(binding_ref, asked, shown)`."""
    (part,) = assemble_region(
        region, query="", extra_warnings=[], start_ms=0, engine_build="test", max_candidates=10
    ).parts
    return [
        (d.binding_ref, d.asked, d.shown) for d in part.departures if d.code == "member_unavailable"
    ]
