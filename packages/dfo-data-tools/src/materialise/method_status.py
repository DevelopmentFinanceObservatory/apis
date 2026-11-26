"""The declared status of a facet's values, read by every rule that tells an estimate from a
reported value.

`data/method_status.json` classes each known `measurementMethod`, lists the provenances
whose method-less values are reported, and declares the methods each provenance uses only for
years after its last actual one. One file holds all three, so one content hash covers the
classification. `materialise.statistics` counts a version's estimated and provisional values
from it, `materialise.facets` continues a reported series with an estimated one by it, and
`materialise.held_back` holds projected rows back by it.
"""

from __future__ import annotations

import importlib.resources
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Final, Literal, cast, get_args

StatusClass = Literal["estimated", "provisional", "reported"]
"""What a measurement method says about the values it produced."""

_METHOD_STATUS_FILE: Final = "method_status.json"


@dataclass(frozen=True, slots=True)
class _MethodStatus:
    """`data/method_status.json`: the status class of each classed method, the provenances
    whose method-less values are reported, and the projection methods of each provenance."""

    classes: Mapping[str, StatusClass]
    reported_provenances: frozenset[str]
    projection_methods: Mapping[str, frozenset[str]]


@cache
def _method_status() -> _MethodStatus:
    """The packaged `data/method_status.json`, checked.

    Raises:
        ValueError: a method's entry names no `StatusClass`.
    """
    raw = (importlib.resources.files(__package__) / "data" / _METHOD_STATUS_FILE).read_text("utf-8")
    entries = cast("dict[str, object]", json.loads(raw))
    classes = cast("dict[str, str]", entries["methods"])
    known = get_args(StatusClass)
    for method, status in classes.items():
        if status not in known:
            raise ValueError(
                f"{_METHOD_STATUS_FILE} classes {method!r} as {status!r}, not one of {known}"
            )
    return _MethodStatus(
        classes=MappingProxyType(cast("dict[str, StatusClass]", classes)),
        reported_provenances=frozenset(cast("list[str]", entries["reported_provenances"])),
        projection_methods=MappingProxyType(
            {
                provenance: frozenset(methods)
                for provenance, methods in cast(
                    "dict[str, list[str]]", entries["projection_methods"]
                ).items()
            }
        ),
    )


def method_status_classes() -> Mapping[str, StatusClass]:
    """The packaged status class of each classed `measurementMethod`, by method dcid.

    Each entry classes a method by its node's name and description in the graph. A
    method the file leaves out is unclassed.
    """
    return _method_status().classes


def reported_provenances() -> frozenset[str]:
    """The provenances whose values with no `measurementMethod` count as reported.

    Each entry splits its estimates into a facet of their own: at every place its
    method-less series ends the year before its `Estimated` series starts. IMF WEO's seven
    corpus variables bear this out at all 1,472 place, unit and variable pairs holding both
    series on the dev graph. World Bank IDS holds it by construction: the pipeline leaves the
    method of every year up to the last reported debt-stock year empty and marks every later
    year `dcid:Estimated` (`dfo-pipelines`, `int_ids_observations.sql`, its `last_reported`
    CTE). A method-less value under any other provenance has an unknown status.
    """
    return _method_status().reported_provenances


def projection_methods(provenance: str | None) -> frozenset[str]:
    """The `measurementMethod`s `provenance` uses only for years after its last actual year,
    so every row under one is a projection. Empty for a provenance the file declares none
    for, whose estimates are actual rows.

    Each entry rests on the evidence its `reported_provenances` entry cites: IMF WEO's
    `Estimated` series starts the year after its method-less series ends at all 1,472 pairs
    holding both on the dev graph, and World Bank IDS marks `Estimated` exactly the years
    after its last reported debt-stock year.
    """
    if provenance is None:
        return frozenset()
    return _method_status().projection_methods.get(provenance, frozenset())


def is_projection(provenance: str | None, method: str | None) -> bool:
    """Whether a row under `method` and `provenance` is a projection (`projection_methods`)."""
    return method is not None and method in projection_methods(provenance)


def status_of(method: str | None, provenance: str | None) -> StatusClass | None:
    """The status of the values a facet with `method` under `provenance` holds: the
    method's class, or for no method `reported` under a `reported_provenances` entry.
    `None` when unknown."""
    if method is None:
        return "reported" if provenance in reported_provenances() else None
    return method_status_classes().get(method)
