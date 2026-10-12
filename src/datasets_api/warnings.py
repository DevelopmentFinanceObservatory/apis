"""Datasets manifest-warning catalogue, keyed by code to title, detail, and severity.

Warnings appear in 200 manifests. The refusal registry handles errors. The capability
version hashes this catalogue, so wording and severity changes affect it.
This package reserves `enrichment_unavailable`, `deflator_projected`, `currency_unmapped_omitted`
and `deflator_missing_omitted` without raising them here. Only the stored manifests of versions
minted before `content_version` hashed the frame alone carry `revision_unavailable`, and a snapshot
read replays them.

`facet_default_selected` is built from the completed materialisation trace.
The `materialise` package keeps its warning catalogue independent, except that
`enrichment_unavailable` takes its title and detail from `materialise.reshape`, which raises it.
`stat_vars_summed` reports the contributing-member count and family reference after a groupby
collapse sums multiple member series.
"""

from __future__ import annotations

from materialise.reshape import RESHAPE_CODE_CATALOGUE

DATASETS_WARNING_CATALOGUE: dict[str, tuple[str, str, str]] = {
    "revision_unavailable": (
        "Revision Unavailable",
        "No live revision endpoint is available; graph_revision and source_revisions "
        "are a fixed sentinel, and a drift check against them cannot detect an "
        "upstream refresh mid-materialisation.",
        "info",
    ),
    "enrichment_unavailable": (*RESHAPE_CODE_CATALOGUE["enrichment_unavailable"], "info"),
    "facet_default_selected": (
        "Facet Default Selected",
        "An unbound facet axis had more than one candidate value, DFO precedence did "
        "not decide between them, and this deployment's default policy chose one by "
        "the bound places or role tuples it covers, then the requested years it "
        "covers, then observation count, then recency, then a deterministic tie-break.",
        "warn",
    ),
    "deflator_projected": (
        "Deflator Projected",
        "A converted observation's exchange rate or price deflator at the observation "
        "year or the base year came from a projection facet rather than an actual.",
        "info",
    ),
    "currency_unmapped_omitted": (
        "Currency Unmapped Omitted",
        "A conversion into each entity's own currency left out the rows of the entities "
        "that no single currency names through nationalCurrencyOf; the warning names them.",
        "warn",
    ),
    "deflator_missing_omitted": (
        "Deflator Missing Omitted",
        "A constant-price conversion left out each entity's rows in the years the deflator "
        "table holds no value for it, and every row of an entity it holds no value for in "
        "the base year; the warning names them.",
        "warn",
    ),
    "stat_vars_summed": (
        "StatVars Summed",
        "The series sums observations from more than one StatVar in the requested "
        "family, and carries the family's reference as its stat_var.",
        "info",
    ),
    "withheld_by_registry": (
        "Withheld By Registry",
        "A series came back empty because every facet for it sits under a provenance "
        "the origin registry does not admit; the warning names those provenances.",
        "warn",
    ),
    "un_system_unavailable": (
        "UN System Unavailable",
        "The UN System Data Commons did not answer for part of the request: either the "
        "resolve, so the answer holds no UN candidates, or reads for named places, so the "
        "answer holds no UN data for the places the warning names.",
        "warn",
    ),
    "origin_catalogue_stale": (
        "Origin Catalogue Stale",
        "The loaded origin catalogue is more than 14 days old.",
        "warn",
    ),
}
