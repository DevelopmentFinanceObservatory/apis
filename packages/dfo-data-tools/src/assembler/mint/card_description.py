"""Composes a card revision's description from the record of the version it draws.

A card over a frame whose version holds a description record carries a `card_revision`
description. A line chart, bar chart, table or stat card takes it from `describe_card`, over
that record, the card's selection facts (`assembler.mint.selection.selection_facts`) and today's
words; a stat card's change is the step after its selection. A source card takes it from
`describe_source_card`, over the documents of the sources it lists.

A card with a description keeps lineage alone in provenance (`lineage_only`), and names each
source by its dcid, a `description.sources[].ref` (`assembler.mint.source_refs`). A series of a
computed version whose rows name no source was computed from the inputs its variable was
computed from (`description.blocks.drawn_inputs`): the card's description lists the sources
those inputs' records credit (`ResolvedFrame.input_records`) and then those inputs as sources,
its citation line names those sources, and the series names the first of those inputs whose id
the record knows. The version's caveats reach it through its record. The card adds caveats only for facts
about its selection: the series a rolling card leaves out make `coverage_gap` caveats on its
`select` step, at the period they hold no value at, one per set of entities they are about,
naming those entities and the variables left out for them, which the composer words. A caveat
leaves its variables unnamed when they are every variable the card draws. A series known by
facet values alone, such as a unit the frame holds no row in, is named by those values.

A card over a version with no record, a card naming a source with no dcid and no input to
name in its place, a source card listing no source and a note narration carry no description.
Nor does a card whose series name no source where the record knows no input id, as on an
`ops_` version published without steps, or where another series names the source the frame
cites first, which provenance names those series by (`assembler.mint.provenance`); its
provenance lists every source the frame cites.
A source card lists the sources of the frame's rows, so one over a computed version whose rows
name none carries no description.

The words are the `VocabularySource`'s, with the frame's labels filling what they lack: the
labels the frame store resolved for the frame's refs, under a variable's bare dcid, the label
and publisher of each source the frame's manifest cites, and what each source's document says,
its last download and next release, as the frame store read it
(`ResolvedFrame.source_documents`), and the title of each version a computed version was
computed from, worded from the input records the frame store read
(`ResolvedFrame.input_records`). They add no read, and are clipped and blank-filtered as
the Datasets description door's are (`build_vocabulary`). The description's vocabulary
revision hashes the source's code index with the graph generation and
catalogue build the frame store read the frame under (`ResolvedFrame.graph_generation`,
`ResolvedFrame.catalogue_build`), so a frame read at another generation or under another build
moves it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import cast

from dfo_frame_grammar import STAT_VAR_COLUMN
from dfo_frame_grammar.grammar import STAT_VAR_REF_PREFIX

from assembler.bind import BoundSeries
from assembler.frame_view import facet_dimension_names
from assembler.mint.dispatch import MintedComponent
from assembler.mint.provenance import (
    LeftOut,
    _citation_sources_by_ref,
    card_derivation_steps,
    lineage_only,
)
from assembler.mint.selection import selection_facts
from assembler.mint.source_refs import renamed_source_refs, source_dcids, source_refs
from assembler.ports import ResolvedFrame, VocabularySource
from assembler.resolve import ResolvedSeries
from description.blocks import drawn_inputs
from description.compose import (
    SELECT_OP,
    describe_card,
    describe_source_card,
    version_title,
)
from description.facts import CaveatFact, DescriptionRecord, SelectionFacts
from description.steps import VersionRef
from description.vocabulary import SourceDocumentFacts, Vocabulary, build_vocabulary

__all__ = ["DescribedRevision", "with_card_description"]


@dataclass(frozen=True)
class DescribedRevision:
    """The card revision a description describes, and where its words come from.

    `live` marks a render, whose description carries `rendering: "live"` and the frame it drew
    on, and is stored nowhere."""

    vocabulary_source: VocabularySource
    component_id: str
    revision: int
    live: bool = False


def with_card_description(
    *,
    minted: MintedComponent,
    member: str,
    merged_intent: Mapping[str, object],
    resolved_frame: ResolvedFrame | None,
    series: Sequence[ResolvedSeries],
    bound_series: Sequence[BoundSeries],
    described: DescribedRevision,
    left_out: LeftOut | None = None,
) -> MintedComponent:
    """`minted` with its description, its sources named by dcid and lineage-only provenance,
    or `minted` unchanged when the card carries no description.

    `merged_intent` is the intent the card is minted from, its relative periods resolved.
    `left_out` names the series a rolling card leaves out."""
    record = resolved_frame.description_record if resolved_frame is not None else None
    if record is None:
        return minted
    assert resolved_frame is not None  # a record comes from the frame
    selection = (
        selection_facts(
            merged_intent=merged_intent,
            resolved_frame=resolved_frame,
            series=series,
            bound_series=bound_series,
        )
        if member != "source_card" and bound_series
        else None
    )
    names = source_dcids(minted.provenance)
    unattributed = minted.source_ids.get(None)
    if unattributed is not None:
        # A row naming no source shares the id of the first source its frame cites
        # (`assembler.mint.provenance`), and the description names it by the first input it
        # lists.
        computed_from = _computed_from(record, selection) if member != "source_card" else None
        if computed_from is None or any(
            bound.source_ref is not None and minted.source_ids[bound.source_ref] == unattributed
            for bound in bound_series
        ):
            return minted
        names[unattributed] = computed_from
    if any(ref not in names for _, ref in source_refs(minted.component)):
        return minted
    component = cast("Mapping[str, object]", renamed_source_refs(minted.component, names))
    vocabulary = _card_vocabulary(
        vocabulary=described.vocabulary_source.current(), resolved_frame=resolved_frame
    )
    frame_ref = resolved_frame.frame_ref
    frame = (
        VersionRef(id=frame_ref.spec_id, content_version=frame_ref.content_version)
        if described.live
        else None
    )
    if member == "source_card":
        sources = cast("Sequence[str]", component["source_refs"])
        if not sources:
            return minted
        description = describe_source_card(
            record,
            vocabulary,
            sources=sources,
            component_id=described.component_id,
            revision=described.revision,
            frame=frame,
        )
    elif selection is not None:
        drawn_variables = {
            coordinate.value
            for one in selection.series
            for coordinate in one.coordinates
            if coordinate.column == STAT_VAR_COLUMN and coordinate.value is not None
        }
        description = describe_card(
            record,
            selection,
            vocabulary,
            component_id=described.component_id,
            revision=described.revision,
            frame=frame,
            steps=card_derivation_steps(component),
            caveats=_left_out_caveats(left_out, drawn_variables=drawn_variables)
            if left_out is not None
            else [],
            inputs=resolved_frame.input_records,
        )
    else:
        return minted
    return dataclasses.replace(
        minted,
        component=component,
        provenance=lineage_only(minted.provenance),
        description=description,
    )


def _computed_from(record: DescriptionRecord, selection: SelectionFacts | None) -> str | None:
    """The id a series naming no source of its own names: of the first input the series the
    card draws were computed from (`description.blocks.drawn_inputs`), which its description
    lists, else of the version's first, whose id the record knows. `None` for a dataset
    version or where no such input id is known."""
    if record.computed is None:
        return None
    inputs = record.computed.inputs if selection is None else drawn_inputs(record, selection)
    return next((one.id for one in inputs if one.id is not None), None)


def _left_out_caveats(left_out: LeftOut, *, drawn_variables: Set[str]) -> list[CaveatFact]:
    """The `coverage_gap` caveats on the card's selection for the series a rolling card leaves
    out at `left_out.period`: one per set of variables left out, naming them by bare dcid and
    the entities they are left out for, in the order the series first name them.

    A series' entities are the values of its dimensions other than the variable and the facets.
    Entities whose left-out variables are every variable in `drawn_variables`, or whose series
    name no variable, are left out whole, and their caveat names no variable. A series naming
    neither a variable nor an entity, such as a unit value the frame holds no row in, is named
    by its facet values."""
    facets = facet_dimension_names()
    variables_by_entities: dict[tuple[str, ...], list[str | None]] = {}
    for one in left_out.series:
        entities = tuple(
            value
            for name, value in one.coordinates.items()
            if name != STAT_VAR_COLUMN and name not in facets
        )
        variable = one.coordinates.get(STAT_VAR_COLUMN)
        if variable is not None:
            names: list[str | None] = [variable.removeprefix(STAT_VAR_REF_PREFIX)]
        elif entities:
            names = [None]
        else:
            names = [value for name, value in one.coordinates.items() if name in facets]
        variables = variables_by_entities.setdefault(entities, [])
        variables.extend(name for name in names if name not in variables)
    entities_by_variables: dict[frozenset[str], tuple[list[str], list[str]]] = {}
    for entities, variables in variables_by_entities.items():
        named = [variable for variable in variables if variable is not None]
        if len(named) < len(variables) or set(named) >= drawn_variables:
            named = []
        _, grouped = entities_by_variables.setdefault(frozenset(named), (named, []))
        grouped.extend(entity for entity in entities if entity not in grouped)
    return [
        CaveatFact(
            kind="coverage_gap",
            series=series,
            entities=entities,
            periods=[left_out.period],
            step=SELECT_OP,
        )
        for series, entities in entities_by_variables.values()
    ]


def _card_vocabulary(*, vocabulary: Vocabulary, resolved_frame: ResolvedFrame) -> Vocabulary:
    """`vocabulary` with the frame's labels, its variables' descriptions
    (`ResolvedFrame.descriptions`) and its cited sources' document facts added where it names
    none, built through `build_vocabulary` under the graph generation and catalogue build the
    frame was read at. A cited source's facts are the frame's
    (`ResolvedFrame.source_documents`), with the publisher its citation names and the last
    update its cited documentation records filling what those lack. Each input record the
    frame holds (`ResolvedFrame.input_records`) is labelled by its title in those words
    (`version_title`), as the Datasets description door labels it."""
    labels = {
        ref.removeprefix(STAT_VAR_REF_PREFIX): label for ref, label in resolved_frame.labels.items()
    }
    sources: dict[str, SourceDocumentFacts] = {}
    for ref, cited in _citation_sources_by_ref(resolved_frame.manifest).items():
        label = cast("str | None", cited.get("label"))
        if label and label != ref:
            labels[ref] = label
        publisher = cast("Mapping[str, str] | None", cited.get("source"))
        documentation = cast("Mapping[str, object] | None", cited.get("documentation"))
        facts = resolved_frame.source_documents.get(ref, SourceDocumentFacts())
        sources[ref] = SourceDocumentFacts(
            publisher=facts.publisher or (publisher["ref"] if publisher is not None else None),
            last_downloaded=facts.last_downloaded
            or (cast("str | None", documentation.get("last_update")) if documentation else None),
            next_release=facts.next_release,
        )
    labels |= vocabulary.labels
    sources |= vocabulary.sources

    def built(labels: Mapping[str, str]) -> Vocabulary:
        return build_vocabulary(
            index=vocabulary.index,
            measure_index=vocabulary.measure_index,
            labels=labels,
            sources=sources,
            graph_generation=resolved_frame.graph_generation,
            catalogue_build=resolved_frame.catalogue_build,
            definitions=dict(resolved_frame.descriptions) | dict(vocabulary.definitions),
        )

    described = built(labels)
    titles = {
        held.id: title
        for held in resolved_frame.input_records
        if (title := version_title(held, described)) is not None
    }
    return built(labels | titles) if titles else described
