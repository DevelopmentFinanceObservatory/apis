"""Facts the origin registry declares for a UN series shape, which the UN graph does not state.

A UN series shape's population type is its series (`undata/<segment>/SERIES-<code>`), so
the shape reads the registry's declaration of that series (`SeriesDeclaration`). A series
with no declaration, and every other shape, declares nothing.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping

from materialise.roles import RoleAxis
from origin_catalogue.registry import SeriesDeclaration, SeriesSide, load_registry
from search.engine.shape import ShapeDraft

_SIDE_AXES: Mapping[SeriesSide, RoleAxis] = {"provider": "origin", "recipient": "destination"}


@functools.cache
def _declarations() -> Mapping[str, SeriesDeclaration]:
    return load_registry().series_declarations()


def declared_side(shape: ShapeDraft) -> RoleAxis | None:
    """The flow axis a one-way UN series' place sits on: `origin` for finance the place
    provides ("climate-specific financial support provided"), `destination` for finance it
    receives ("gross receipts by developing countries"). `None` when undeclared."""
    declaration = _declarations().get(shape.pop_type_dcid)
    if declaration is None or declaration.side is None:
        return None
    return _SIDE_AXES[declaration.side]


def declared_measurement_qualifier(shape: ShapeDraft) -> str | None:
    """The `measurementQualifier` value of a UN series' values ("RealValue" for one at
    constant prices), or `None` when undeclared. A UN series asserts no
    `measurementQualifier` arc, so the shape's own `meas_qual_dcid` stays empty: setting it
    would put a qualifier no member carries into the shape's identity (`spec_build`)."""
    declaration = _declarations().get(shape.pop_type_dcid)
    return None if declaration is None else declaration.measurement_qualifier
