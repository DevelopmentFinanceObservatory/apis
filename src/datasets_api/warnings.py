"""Datasets manifest-warning catalogue, keyed by code to title, detail, and severity.

Warnings appear in 200 manifests. The refusal registry handles errors. The capability
version hashes this catalogue, so wording and severity changes affect it.

A new manifest raises `enrichment_unavailable`, `withheld_by_registry`, `un_system_unavailable`,
`currency_unmapped_omitted` and `deflator_missing_omitted` (the last two from `materialise`), and
a group-by collapse summing rows of several reporting organisations raises `reporters_summed`
(`datasets_api.groupby.ReporterCheck`).
`origin_catalogue_stale` goes to the manifest's `diagnostics`. `facet_default_selected`,
`stat_vars_summed` and `deflator_projected` are steps and caveats of the version's description
record (`datasets_api.description_record`), and nothing raises `revision_unavailable`. Those
four stay catalogued because stored manifests serve them: a snapshot read replays the warnings
a version was minted with.

The `materialise` package keeps its warning catalogue independent, except that
`enrichment_unavailable` takes its title and detail from `materialise.reshape`, which raises it.
"""

from __future__ import annotations

from materialise.reshape import RESHAPE_CODE_CATALOGUE

REPORTERS_SUMMED = "reporters_summed"
"""The warning a group-by collapse raises when a total sums rows that two or more reporting
organisations of a dataset declaring `members_report_overlapping_accounts` filed
(`datasets_api.groupby.ReporterCheck`)."""


def reporters_summed_message(count: int) -> str:
    """The `REPORTERS_SUMMED` message for a total summing the rows of `count` reporting
    organisations, which a reader of the message alone can act on."""
    return (
        f"This figure adds up transactions that {count} reporting organisations each report "
        "in their own accounts. Money one of them passes to another can be counted twice: "
        "once when the first reports disbursing it and again when the second reports "
        "spending it."
    )


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
        "that no single currency names through nationalCurrencyOf; the warning counts them "
        "and carries the refs of the first ten.",
        "warn",
    ),
    "deflator_missing_omitted": (
        "Deflator Missing Omitted",
        "A constant-price conversion left out each entity's rows in the years the deflator "
        "table holds no value for it, and every row of an entity it holds no value for in "
        "the base year; the warning counts them and carries the refs of the first ten.",
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
    REPORTERS_SUMMED: (
        "Reporters Summed",
        "The figure sums transactions that several reporting organisations each report, so "
        "money one organisation passes to another can be counted again when the second "
        "reports spending it.",
        "warn",
    ),
    "origin_catalogue_stale": (
        "Origin Catalogue Stale",
        "The loaded origin catalogue is more than 14 days old.",
        "warn",
    ),
}
