"""The graph rows DFO authors because Spanner lacks them.

Live variables name eleven nodes DFO's Spanner holds no `Node` row for: the base Data Commons
variable `Count_Person`, four measure values and six properties. The mixer answers them from the
base graph. A release copies only DFO's Spanner, so it carries them as `nodes` rows flagged
`authored`, and the four measure values' descriptions as literal nodes joined by `edges` rows
under `AUTHORED_PROVENANCE`: 15 nodes and 4 edges.

The values were recorded once from the dev mixer's `/v2/node` reads; the builder never reads
them live. `graph_release.platform` refuses a build where Spanner gains a row for one of these
nodes, so a change upstream reaches this module by hand.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Final

import pyarrow as pa

AUTHORED_PROVENANCE: Final[str] = "provenance/DFOAuthored"

_LITERAL_PREFIX_CHARS: Final[int] = 16


@dataclass(frozen=True)
class AuthoredNode:
    """One `nodes` row. A literal has a `value`, an empty `name` and no types, as Spanner's
    literal rows do."""

    subject_id: str
    name: str
    types: tuple[str, ...]
    value: str | None = None


@dataclass(frozen=True)
class AuthoredEdge:
    """One `edges` row, under `AUTHORED_PROVENANCE`."""

    subject_id: str
    predicate: str
    object_id: str


@dataclass(frozen=True)
class Authored:
    """The authored `nodes` and `edges` rows."""

    nodes: tuple[AuthoredNode, ...]
    edges: tuple[AuthoredEdge, ...]

    def nodes_table(self) -> pa.Table:
        """The rows in the `nodes` part's columns, `authored` true."""
        return pa.table(
            {
                "subject_id": pa.array([node.subject_id for node in self.nodes], pa.string()),
                "name": pa.array([node.name for node in self.nodes], pa.string()),
                "types": pa.array([list(node.types) for node in self.nodes], pa.list_(pa.string())),
                "value": pa.array([node.value for node in self.nodes], pa.string()),
                "authored": pa.array([True] * len(self.nodes), pa.bool_()),
            }
        )

    def edges_table(self) -> pa.Table:
        """The rows in the `edges` part's columns."""
        return pa.table(
            {
                "subject_id": pa.array([edge.subject_id for edge in self.edges], pa.string()),
                "predicate": pa.array([edge.predicate for edge in self.edges], pa.string()),
                "object_id": pa.array([edge.object_id for edge in self.edges], pa.string()),
                "provenance": pa.array([AUTHORED_PROVENANCE] * len(self.edges), pa.string()),
            }
        )


def literal_id(text: str) -> str:
    """The dcid Spanner gives a literal node: its first 16 characters, a colon, and the base64
    SHA-256 of the whole text."""
    digest = base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode("ascii")
    return f"{text[:_LITERAL_PREFIX_CHARS]}:{digest}"


_ENTITIES: Final[tuple[AuthoredNode, ...]] = (
    AuthoredNode(
        subject_id="Count_Person", name="Total population", types=("StatisticalVariable",)
    ),
    AuthoredNode(
        subject_id="InflationAdjusted", name="InflationAdjusted", types=("InflationAdjustmentEnum",)
    ),
    AuthoredNode(subject_id="Nominal", name="Nominal", types=("EconomicMetricEnum",)),
    AuthoredNode(
        subject_id="PurchasingPowerParity",
        name="PurchasingPowerParity",
        types=("EconomicMetricEnum",),
    ),
    AuthoredNode(subject_id="RealValue", name="RealValue", types=("EconomicMetricEnum",)),
    AuthoredNode(subject_id="amount", name="amount", types=("Property",)),
    AuthoredNode(subject_id="exchangeRate", name="exchangeRate", types=("Property",)),
    AuthoredNode(subject_id="interestRate", name="interestRate", types=("Property",)),
    AuthoredNode(subject_id="maturity", name="maturity", types=("Property",)),
    AuthoredNode(subject_id="meanValue", name="meanValue", types=("Property",)),
    AuthoredNode(subject_id="measuredValue", name="measuredValue", types=("Property",)),
)

_DESCRIPTIONS: Final[dict[str, str]] = {
    "InflationAdjusted": (
        "Indicates adjustment of a time series data to eliminate the effect Inflation or price "
        "changes over a period"
    ),
    "Nominal": "Measurement is evaluated at current market prices, without adjusting for inflation.",
    "PurchasingPowerParity": (
        "Represents prices of different areas using a common good or goods to contrast the real "
        "purchasing power between different currencies."
    ),
    "RealValue": (
        "Measurement is evaluated against the price of goods or services at a different time to "
        "adjust for inflation. This adjustment could be via a chained volume series "
        "(https://en.wikipedia.org/wiki/Chained_volume_series) or a fixed-weight method."
    ),
}

AUTHORED: Final[Authored] = Authored(
    nodes=(
        *_ENTITIES,
        *(
            AuthoredNode(subject_id=literal_id(text), name="", types=(), value=text)
            for text in _DESCRIPTIONS.values()
        ),
    ),
    edges=tuple(
        AuthoredEdge(subject_id=subject, predicate="description", object_id=literal_id(text))
        for subject, text in _DESCRIPTIONS.items()
    ),
)


__all__ = [
    "AUTHORED",
    "AUTHORED_PROVENANCE",
    "Authored",
    "AuthoredEdge",
    "AuthoredNode",
    "literal_id",
]
