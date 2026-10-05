"""The three Data Commons a fact can come from, with their wire ids and labels.

The wire field is `data_commons`, because the Search contract already uses
`origin` as a donor role key. The ids are the only contract. The labels
travel inline beside them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

type OriginId = Literal["dfo", "base", "un_system"]


@dataclass(frozen=True, slots=True)
class DataCommons:
    """One origin: `id` is the wire enum value, `label` its display name."""

    id: OriginId
    label: str


DFO: Final = DataCommons(id="dfo", label="Development Finance Observatory")
BASE: Final = DataCommons(id="base", label="Data Commons by Google")
UN_SYSTEM: Final = DataCommons(id="un_system", label="UN System Data Commons")

ORIGINS: Final[dict[OriginId, DataCommons]] = {
    origin.id: origin for origin in (DFO, BASE, UN_SYSTEM)
}
"""Every origin by id, in wire order: DFO, then base, then the UN."""

UN_ID_PREFIX: Final[str] = "undata/"
"""Every UN System Data Commons variable, topic, provenance and constraint
value starts with this prefix. Base Data Commons and DFO use none."""


__all__ = ["BASE", "DFO", "ORIGINS", "UN_ID_PREFIX", "UN_SYSTEM", "DataCommons", "OriginId"]
