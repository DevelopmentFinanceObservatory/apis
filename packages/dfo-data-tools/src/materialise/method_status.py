"""The declared status of a facet's values, read by every rule that tells an estimate from a
reported value.

`data/method_status.json` classes each known `measurementMethod` and lists the provenances
whose method-less values are reported. One file holds both, so one content hash covers the
classification. `materialise.statistics` counts a version's estimated and provisional values
from it, and `materialise.facets` continues a reported series with an estimated one by it.
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
    """`data/method_status.json`: the status class of each classed method, and the
    provenances whose method-less values are reported."""

    classes: Mapping[str, StatusClass]
    reported_provenances: frozenset[str]


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
    series on the dev graph. A method-less value under any other provenance has an unknown
    status.
    """
    return _method_status().reported_provenances


def status_of(method: str | None, provenance: str | None) -> StatusClass | None:
    """The status of the values a facet with `method` under `provenance` holds: the
    method's class, or for no method `reported` under a `reported_provenances` entry.
    `None` when unknown."""
    if method is None:
        return "reported" if provenance in reported_provenances() else None
    return method_status_classes().get(method)
