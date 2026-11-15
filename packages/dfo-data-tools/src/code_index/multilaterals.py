"""Which parties are multilateral organisations, read from the code index and the entity
resolver's organisation records, so Search (donors) and the mint (DAC2a destinations) classify
a party the same way.

A multilateral is an official party the graph types `Organization`: the organisation records
the entity resolver exports from the graph (`organisation_dcids`), less the private providers,
which the code index classes `private` (`PartyRow.party_class`). The EU institutions count,
since they spend their member states' contributions
(`code_index.scope_conventions.REDISTRIBUTED_PROVIDERS`).
"""

from __future__ import annotations

import functools
import importlib.resources
import importlib.util
import json
from collections.abc import Collection
from typing import Final, NamedTuple

from code_index.types import PartyClass, PartyKind, PartyRow, RoleAxis

COUNTRY_PREFIX: Final[str] = "country/"
"""The dcid prefix of a country node."""

_RECORDS_PACKAGE = "entity_resolution"
_RECORDS = ("data", "graph_entities.json")


@functools.cache
def organisation_dcids() -> frozenset[str]:
    """The dcids of the entity resolver's committed records of kind `organisation`: the nodes
    the graph types `Organization`, multilaterals and private foundations alike. The one
    reader of them, which `entity_resolution.pack` imports.

    The records are read through `importlib.resources` from a module built from the package's
    import spec and never executed, so the read works from a wheel or a zip as from disk, and
    `import code_index` runs no `entity_resolution.__init__` and loads no resolvekit."""
    spec = importlib.util.find_spec(_RECORDS_PACKAGE)
    if spec is None:
        raise RuntimeError(f"the {_RECORDS_PACKAGE} package is not installed")
    package = importlib.util.module_from_spec(spec)
    records = json.loads(
        importlib.resources.files(package).joinpath(*_RECORDS).read_text(encoding="utf-8")
    )
    return frozenset(record["dcid"] for record in records if record["kind"] == "organisation")


class PartyFacts(NamedTuple):
    """What one party is where some datasets publish it on one axis: its kind
    (`PartyRow.kind_over`), its class (`PartyRow.class_of`), and whether it is a multilateral,
    each `None` where those datasets give it different kinds."""

    kind: PartyKind | None
    party_class: PartyClass | None
    multilateral: bool | None


def party_facts(row: PartyRow, *, axis: RoleAxis, datasets: Collection[str] = ()) -> PartyFacts:
    """The facts of `row` where `datasets` publish it on `axis`. A multilateral is an
    organisation that is an official party there; a total, such as the World Bank Group beside
    its members in DAC2a, is none."""
    kind = row.kind_over(axis, datasets)
    if kind is None:
        return PartyFacts(None, None, None)
    party_class = row.class_of(kind)
    multilateral = (
        kind == "party" and party_class == "official" and row.dcid in organisation_dcids()
    )
    return PartyFacts(kind, party_class, multilateral)


def is_multilateral(row: PartyRow | None, *, datasets: Collection[str] = ()) -> bool:
    """Whether `row` is a multilateral where `datasets` publish it as a donor, on the origin
    axis (`party_facts`)."""
    return (
        row is not None and party_facts(row, axis="origin", datasets=datasets).multilateral is True
    )
