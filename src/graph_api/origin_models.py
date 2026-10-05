"""Wire models naming where a fact comes from, shared by the node and sources doors.

`data_commons` is the wire name for an origin, because the Search contract already uses
`origin` as a donor role key. `origin_catalogue.origins` holds the ids and labels.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from origin_catalogue import DataCommons, SourceRef


class DataCommonsRef(BaseModel):
    """The Data Commons that serves a fact. `id` is the wire enum; `label` its display name."""

    model_config = ConfigDict(title="GraphDataCommonsRef")

    id: Literal["dfo", "base", "un_system"]
    label: str

    @classmethod
    def of(cls, origin: DataCommons) -> DataCommonsRef:
        return cls(id=origin.id, label=origin.label)


class SourceNodeRef(BaseModel):
    """A Data Commons `Source` node (`dc/s/*`). On the UN System Data Commons, where every
    agency sits under one source, it is the agency provenance (`undata/p/*`)."""

    model_config = ConfigDict(title="GraphSourceNodeRef")

    ref: str
    label: str

    @classmethod
    def of(cls, source: SourceRef) -> SourceNodeRef:
        return cls(ref=source.ref, label=source.label)
