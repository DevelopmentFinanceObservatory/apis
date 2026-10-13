from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import StrictModel
from .grammar import GraphRef, ProvenanceRef, RequestedWindow, SlotKey
from .vocab import BindingKind, BindingProvenance, QuantityForm


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
    binding consumed the term, or, on an `"unbound"` value the answer's shape
    has a slot for but leaves unmet, that slot's property; `None` otherwise.
    `disposition` is
    `"bound"` when some final binding on a named row's code system covers
    it (that row or a descendant), `"unbound"` when a row was named but no
    binding covers it, and `"no_vocabulary"` when the term names no code at
    all -- an LLM-reported phrase (`bind`'s own `unplaced` output) that no
    taxonomy in this deployment's index can express.

    `start` and `end` locate the term in the part's variable phrase. `evidence` says what
    read it, and `None` means a code's label or synonym matched the term as the query
    spelled it, the reading every recognised span records:

    - `"label"` and `"synonym"` name a match on a code's label or synonym; no record sets
      them, since a recognised span records `None`.
    - `"model"` marks a phrase the bind model read through a code it chose from the
      phrase's candidate menu, a lexical hit of the menu or a descendant of one
      (`phase_rank_bind._model_records`); `named` holds the chosen codes.
    - `"uncovered"` marks words no reader covers beside a recognised code
      (`phase_rank_bind._coverage_records`), recorded `"unbound"` with that code's rows as
      `named`.
    - `"unavailable"` marks a code term the elected shape cannot apply: one naming only
      codes no shape can bind, or, on a part naming a dataset, one no row of that dataset
      holds (`phase_rank_bind._unavailable_qualifiers`), recorded `"unbound"`.
    - `"unsupported"` marks an exclusion or comparison phrase no binding expresses
      (`phase_rank_bind._unsupported_records`), recorded `"no_vocabulary"`.

    `operation` and `selected` describe how the term's codes combine with
    another term's on one code system, and the codes it asked for before that
    combination simplified.
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
            "this term, or, on an unbound value the answer's shape has a slot for "
            "but leaves unmet, that slot's property; None otherwise"
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
    start: int | None = Field(
        default=None,
        description="character offset where the term starts in the variable phrase",
    )
    end: int | None = Field(
        default=None,
        description="character offset where the term ends in the variable phrase",
    )
    evidence: (
        Literal["label", "synonym", "model", "uncovered", "unavailable", "unsupported"] | None
    ) = Field(
        default=None,
        description=(
            'what read the term: a code label or synonym, the bind model, "uncovered" '
            'for words no reader covers beside a recognised code, "unavailable" for a '
            'code no table holds, "unsupported" for a distinction no binding expresses'
        ),
    )
    operation: Literal["union", "refinement"] | None = Field(
        default=None,
        description=(
            "how the term's codes combine with another term's on one code system: a "
            "union of both, or a refinement to the narrower"
        ),
    )
    selected: list[GraphRef] = Field(
        default_factory=list,
        description="the codes the term asked for before the combination simplified them",
    )


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
    requested_window: RequestedWindow | None = Field(
        default=None,
        description=(
            "the date window originally requested (query intent), independent of "
            "coverage narrowing; None when no date was requested. "
            'The literal "latest" means the query explicitly asked for the '
            "most-recent/current value -- distinct from None (no date "
            'signal at all), since a resubmit of a "latest" spec must re-resolve '
            "to whatever year is CURRENTLY latest, not pin to the year this "
            "response happened to resolve. spec_id is anchored on this value, not "
            "on the served window, so identity stays stable across a resubmit even "
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
    default_reading: TraceDefaultReading | None = Field(
        default=None,
        description=(
            "what the default price-basis reading did before bind (see "
            "TraceDefaultReading); None when no bind ran for this Spec, as on a resubmit, "
            "or in hand-built fixtures that predate this field"
        ),
    )
    stated_form: QuantityForm | None = Field(
        default=None,
        description=(
            "the normalised form the quantity read found in the question when no wording "
            'pattern named one ("debt to GDP" asks for a rate), and which the answer was '
            "then checked against; None when a wording pattern named the form, when the "
            "question asks for none, and when no bind ran for this Spec"
        ),
    )
    conversion: TraceConversion | None = Field(
        default=None,
        description=(
            "always absent: conversion is recorded per series in "
            "resolved.observations.series[].conversion (see TraceConversion)"
        ),
    )
