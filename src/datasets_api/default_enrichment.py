"""The default enrichment of a dataset version: the enrichment columns a client's frame of it
holds, chosen from the roles the version's frame holds.

A version's roles are its coverage dimensions less the years (`coverage_roles`), since
`materialise.columnar.coverage` counts one dimension per role the frame declares.
`enrichment_columns` gives the type column of each typed role among them (`observationAbout`,
`financingProvider`, `financingRecipient`), and the party kind, class and whole columns when one
is a flow party role. The identity is not read: a flow identity that binds and opens no role, a
total over a role's members, holds its flow roles all the same.

`MintOptions.enrichment = "default"` asks a mint for those columns once the version is known
(`resolve_enrichment`): they are added to the requested `columns`, or are the `columns` when it
names none, and the option is dropped, so the rendering, its key and the frame it names are those
of a request naming the resolved columns. The analysis runner's roots (`analysis_roots`) and the
exports (`exports`) read a version's default enrichment through the same rule.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from datasets_api.manifest_models import CoverageModel
from datasets_api.mint_models import MintOptions
from datasets_api.party_totals import YEARS_DIMENSION
from dfo_analysis_engine.place_type import PARTY_ROLES, TYPED_ROLES, type_column

__all__ = [
    "KIND_COLUMNS",
    "coverage_roles",
    "enrichment_columns",
    "resolve_enrichment",
    "with_default_enrichment",
]

KIND_COLUMNS: tuple[str, ...] = ("party_kind", "party_class", "party_whole")
"""The party cells a frame of a flow party role is enriched with."""


def coverage_roles(coverage: CoverageModel) -> tuple[str, ...]:
    """The role columns of a version's frame: its coverage dimensions, which count each role
    the frame declares (`materialise.columnar.coverage`), less the years."""
    return tuple(
        dimension.label for dimension in coverage.dimensions if dimension.label != YEARS_DIMENSION
    )


def enrichment_columns(roles: Collection[str]) -> tuple[str, ...]:
    """The enrichment columns a frame holding `roles` is read with by default, sorted: the type
    column of each typed role among them, and the party kind, class and whole columns when one
    is a flow party role."""
    typed = {type_column(role) for role in roles if role in TYPED_ROLES}
    kinds = KIND_COLUMNS if not PARTY_ROLES.isdisjoint(roles) else ()
    return tuple(sorted({*typed, *kinds}))


def with_default_enrichment(
    columns: Sequence[str] | None, *, roles: Collection[str]
) -> list[str] | None:
    """`columns` with the default enrichment of `roles` it does not name added after them, or
    that enrichment alone when `columns` is `None`; `None` when both are empty."""
    added = [column for column in enrichment_columns(roles) if column not in (columns or ())]
    if columns is None:
        return added or None
    return [*columns, *added]


def resolve_enrichment(options: MintOptions, *, coverage: CoverageModel) -> MintOptions:
    """`options` with `enrichment = "default"` resolved against a version of `coverage`: its
    columns are `with_default_enrichment` of the version's roles, and the option is dropped.
    Options with no `enrichment` are returned as they are."""
    if options.enrichment is None:
        return options
    columns = with_default_enrichment(options.columns, roles=coverage_roles(coverage))
    return options.model_copy(update={"enrichment": None, "columns": columns})
