from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import StrictModel
from .grammar import GraphRef, ProvenanceRef, RequestedWindow, SlotKey, TimeWindow
from .vocab import BindingKind, BindingProvenance, DateSource, PipelineStepName


class ResolvedFilter(StrictModel):
    """One applied bind decision, slot by slot."""

    key: SlotKey = Field(description="the slot identity")
    binding_kind: BindingKind = Field(description="which binding state was applied")
    refs: list[GraphRef] = Field(description="the bound graph objects (empty for unbound)")
    provenance: BindingProvenance = Field(
        default="binder",
        description=(
            "whether the bind call's own election produced this value, a code "
            "row recognised in the query's own text forced it, or the "
            "measure-family election forced it on an axis the binder left open"
        ),
    )


class ResolvedQualifier(StrictModel):
    """One narrowing term the request's text named, and what became of it.

    The term's text can come from the variable phrase, from the verbatim
    `query_text`, or from an admitted qualifier tie (see
    `_resolve_query_qualifiers`). `text` is the verbatim span of whichever
    source named it. `named` is every code row -- any system -- that span
    names in the packaged index;
    empty when the text names no code at all (a `no_vocabulary` term). `slot`
    is the constraint property dcid of the `slot_filters` entry whose
    binding consumed the term; `None` when none did. `disposition` is
    `"bound"` when some final binding on a named row's code system covers
    it (that row or a descendant), `"unbound"` when a row was named but no
    binding covers it, and `"no_vocabulary"` when the term names no code at
    all -- an LLM-reported phrase (`bind`'s own `unplaced` output) that no
    taxonomy in this deployment's index can express.
    """

    text: str = Field(
        description=(
            "verbatim span of the source text that named the term: variable_text, "
            "query_text, or a dropped qualifier phrase"
        )
    )
    named: list[GraphRef] = Field(
        description=(
            "every code row the span names in the packaged index, any code "
            "system; empty when none did"
        )
    )
    slot: str | None = Field(
        default=None,
        description=(
            "property dcid of the slot_filters entry whose binding consumed "
            "this term; None when none did"
        ),
    )
    disposition: Literal["bound", "unbound", "no_vocabulary"] = Field(
        description=(
            '"bound" when a final binding covers a named row (or a '
            'descendant of one); "unbound" when a row was named but no '
            'binding covers it; "no_vocabulary" when the term names no code '
            "at all"
        )
    )


class PipelineStep(StrictModel):
    """One step in the resolution pipeline, with whether it ran and its latency."""

    step: PipelineStepName = Field(description="which pipeline step this record describes")
    ran: bool = Field(description="false when this step was skipped for the request")
    ms: int | None = Field(default=None, description="step latency, when measured")


class TraceConversion(StrictModel):
    """A Spec-level record of a unit-conversion recipe. The engine converts
    per series and records each converted series' recipe and tables in
    ObservationSeries.conversion, since one Spec's series can convert under
    different recipes. ResolutionTrace.conversion is modelled here but always
    absent. Capabilities.recipe_classes lists the recipe members this
    deployment applies."""

    recipe: str = Field(description="the conversion recipe class that ran")
    input_tables: dict[str, str] = Field(
        description="the input series/table dcids the recipe combined, keyed by role"
    )


class TraceDefaultReading(StrictModel):
    """The default price-basis reading applied before bind. Price bases rank Nominal,
    RealValue, InflationAdjusted, then PurchasingPowerParity, with any basis the query names
    moved first. A StatVar leaves the menu when a StatVar of a better-ranked basis with the
    same constraints holds data wherever it does over the requested years. The trace records
    whether the step ran and the StatVars it removed."""

    applied: bool = Field(
        description=(
            "false when a phrase the request dropped uses a price-basis word no qualifier "
            "match read, so every price basis stayed on the menu and the alternate guard "
            "applied no default reading"
        )
    )
    removed_stat_vars: list[str] = Field(
        default_factory=list,
        description="the StatVar dcids the default reading removed; empty when it removed none",
    )


class ResolutionTrace(StrictModel):
    """Consolidated record of how the query was understood."""

    resolved_stat_vars: list[GraphRef] = Field(
        description=(
            "the first StatVars behind this spec, in resolved.stat_vars.items order, at "
            "most 200; resolved.stat_vars.items carries the served list"
        )
    )
    resolved_stat_vars_truncated: bool = Field(
        default=False,
        description="true when resolved_stat_vars leaves out StatVars behind this spec",
    )
    resolved_entities: list[GraphRef] = Field(description="every entity")
    resolved_sources: list[ProvenanceRef] = Field(
        description="every dataset/source (a Data Commons provenance)"
    )
    slot_filters: list[ResolvedFilter] = Field(
        description="each applied bind decision, slot by slot"
    )
    query_qualifiers: list[ResolvedQualifier] = Field(
        default_factory=list,
        description=(
            "each narrowing term the request's text named, and what became "
            "of it (see ResolvedQualifier). Defaulted empty: a dataset_resubmit "
            "identity entry carries no query text to re-derive this from, and every "
            "hand-built fixture and same-shape merge that predates this field "
            "still validates."
        ),
    )
    applied_window: TimeWindow | None = Field(
        default=None, description="the date signal actually applied"
    )
    date_source: DateSource | None = Field(default=None, description="why that window was applied")
    pipeline_trace: list[PipelineStep] = Field(
        description="the pipeline steps, in order, with skip and timing"
    )
    n_recalled: int | None = Field(
        default=None,
        description=(
            "count of recall's candidate SVs excluding Topic-typed candidates, before "
            "derive_shapes confirms and groups them into shapes; larger than the confirmed "
            "candidate-shape count"
        ),
    )
    requested_window: RequestedWindow | None = Field(
        default=None,
        description=(
            "the date window originally requested (query intent), independent of "
            "applied_window's coverage narrowing; None when no date was requested. "
            'The literal "latest" means the query explicitly asked for the '
            "most-recent/current value -- distinct from None (no date "
            'signal at all), since a resubmit of a "latest" spec must re-resolve '
            "to whatever year is CURRENTLY latest, not pin to the year this "
            "response happened to resolve. spec_id is anchored on this value, not "
            "on applied_window, so identity stays stable across a resubmit even "
            "when coverage narrows the dates. Preserve the returned "
            "Spec.definition.identity when resubmitting; if editing it, carry this "
            "value at ResubmitIdentityPayload.identity.requested_window to keep that "
            "stability."
        ),
    )
    capability_version: str | None = Field(
        default=None,
        description=(
            "the deployment capability version that produced this Spec, shared "
            "verbatim with GET /api/search/capabilities and every problem-details "
            "body's own capability_version. None only in hand-built fixtures that "
            "predate this field; every engine-produced Spec populates it."
        ),
    )
    unit_lexicon_version: str | None = Field(
        default=None,
        description=(
            "the unit binder's lexicon version (currency codes/names/aliases, "
            "price-basis phrases, year bounds) that produced this Spec's unit "
            "facet, when one was bound; None when no unit request was bound for "
            "this Spec, or in hand-built fixtures that predate this field."
        ),
    )
    declaration_version: str | None = Field(
        default=None,
        description=(
            "the packaged code index's content_hash (code_index.shared.get_index()."
            "stamp), identifying which CodeSystemDeclaration set (hierarchical, "
            "disjoint_from_children, summable, absent_reading) and which "
            "DatasetDeclaration set (opt_in, absent_observation) governed this "
            "Spec's constraint resolution -- including any slot expansion (see "
            "Slot.expansion) and the dataset its members come from; set whenever "
            "the index decides which StatVars the Spec's identity denotes, even "
            "with no code-system slot bound. None when the index neither decides "
            "the membership nor declares a bound slot's code system, or in "
            "hand-built fixtures that predate this field."
        ),
    )
    default_reading: TraceDefaultReading | None = Field(
        default=None,
        description=(
            "what the default price-basis reading did before bind (see "
            "TraceDefaultReading); None when no bind ran for this Spec, as on a resubmit, "
            "or in hand-built fixtures that predate this field"
        ),
    )
    conversion: TraceConversion | None = Field(
        default=None,
        description=(
            "always absent: conversion is recorded per series in "
            "resolved.observations.series[].conversion (see TraceConversion)"
        ),
    )
