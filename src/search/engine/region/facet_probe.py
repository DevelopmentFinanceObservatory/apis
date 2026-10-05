"""The inputs a spec's facet options are judged over, shared by the probe and the pin refusal.

`FacetPins` holds the bindings a pin is refused against. `FacetProbe` adds the StatVars, the
role-to-entity map and the years the options read, and travels on `RegionResult` so a
same-shape merge and a trailing relaunch judge the merged spec over the same inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from search.models import IdentityFacetBinding

if TYPE_CHECKING:
    from search.engine.retrieval.elected_dataset import ElectedDatasets
    from search.engine.retrieval.read_plan import PlanYears


@dataclass(frozen=True)
class FacetPins:
    """A spec's facet pins as its pin refusal judges them.

    `bindings` maps each pinned axis to its binding, the identity's facets. `elected_units` maps each StatVar to the unit refs
    it elected for a request-form unit, or is `None` when no election ran, in which case the
    request is tested literally (`facet_refusals._unit_request_admits`).
    """

    bindings: Mapping[str, IdentityFacetBinding] = field(
        default_factory=lambda: MappingProxyType({})
    )
    elected_units: Mapping[str, frozenset[str]] | None = None


NO_PINS = FacetPins()


def facet_pins(
    facets: Mapping[str, IdentityFacetBinding] | None,
    *,
    elected_units: Mapping[str, frozenset[str]] | None = None,
) -> FacetPins:
    """Build the pins for `facets`."""
    return FacetPins(bindings=MappingProxyType(dict(facets or {})), elected_units=elected_units)


@dataclass(frozen=True)
class FacetProbe:
    """What one spec's facet options are judged over.

    `sv_dcids` are its StatVars, `probe_entities` the role-to-entity map the probe reads
    (`observationAbout` for subject StatVars, role names for role-bearing ones), `years` the
    read plan's years a point must fall in (`read_plan.preview_years`), `pins` the spec's
    pins, and `datasets` the elected dataset of each StatVar of an index-resolved family,
    whose facets alone are offered and judged (`retrieval.elected_dataset`).
    """

    sv_dcids: tuple[str, ...]
    probe_entities: Mapping[str, tuple[str, ...]]
    years: PlanYears
    pins: FacetPins = NO_PINS
    datasets: ElectedDatasets = field(default_factory=lambda: MappingProxyType({}))


def facet_probe(
    sv_dcids: Sequence[str],
    probe_entities: Mapping[str, Sequence[str]],
    *,
    years: PlanYears,
    pins: FacetPins = NO_PINS,
    datasets: ElectedDatasets | None = None,
) -> FacetProbe:
    """Build a `FacetProbe`, copying the StatVar list, entity map and elected datasets."""
    return FacetProbe(
        sv_dcids=tuple(sv_dcids),
        probe_entities=MappingProxyType(
            {role: tuple(dcids) for role, dcids in probe_entities.items()}
        ),
        years=years,
        pins=pins,
        datasets=MappingProxyType(dict(datasets or {})),
    )


def merge_facet_probes(probes: Sequence[FacetProbe], *, years: PlanYears) -> FacetProbe:
    """One probe over every member's StatVars and entities, in first-seen order.

    Same-shape members share their identity facets, so the first member's bindings stand
    for all. Elected units merge per StatVar and stay `None` only when no member ran an
    election.
    """
    sv_dcids = list(dict.fromkeys(sv for probe in probes for sv in probe.sv_dcids))
    entities: dict[str, list[str]] = {}
    for probe in probes:
        for role, dcids in probe.probe_entities.items():
            merged = entities.setdefault(role, [])
            merged.extend(d for d in dcids if d not in merged)
    elected: dict[str, frozenset[str]] | None = None
    for probe in probes:
        if probe.pins.elected_units is None:
            continue
        elected = elected if elected is not None else {}
        for sv, units in probe.pins.elected_units.items():
            elected[sv] = elected.get(sv, frozenset()) | units
    pins = FacetPins(bindings=probes[0].pins.bindings, elected_units=elected)
    datasets = {sv: dataset for probe in probes for sv, dataset in probe.datasets.items()}
    return facet_probe(sv_dcids, entities, years=years, pins=pins, datasets=datasets)
