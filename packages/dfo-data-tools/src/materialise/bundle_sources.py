"""Maps a version's stored citation of one source to the bundle's `SourceProvenance`.

Both bundle producers cite sources from the same stored shape: a Datasets version body's
`citation.sources[]` entry, as `datasets_api.manifest_models.CitationSource` writes it in JSON
mode. `render` dumps its parsed citation to that form, and a card export reads it from the
frame manifest, where derived frames carry the same shape. A composite provenance's rows are
cited by its component tables, and `composite_notes` names the composite itself from its
declaration. Nothing here reads the graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from code_index.composites import DECLARED_COMPOSITES
from code_index.scope_conventions import BASIS_MEASURES, BILATERAL_SOURCES
from materialise.bundle import (
    ChangeDoc,
    CompositeNote,
    Licence,
    ReleaseDoc,
    SourceDocumentation,
    SourceProvenance,
    TransformationDoc,
)

if TYPE_CHECKING:
    from code_index.query import CodeIndex
    from code_index.types import ComponentRecipe, CompositeDeclaration


def composite_bundle_label(index: CodeIndex, composite: CompositeDeclaration) -> str:
    """The name a bundle gives `composite`: its label, followed by its `measure_noun` where
    the label is one of its component tables' labels in `index`, so the composite and the
    table read apart in one bundle ("OECD DAC2a (bilateral flows)" beside "OECD DAC2a").
    Citations keep the composite's label."""
    tables = {
        declared.label
        for recipe in composite.components
        if (declared := index.datasets.get(recipe.dataset)) is not None
    }
    if composite.label in tables:
        return f"{composite.label} {composite.measure_noun}"
    return composite.label


def composite_bundle_labels(index: CodeIndex) -> dict[str, str]:
    """Each declared composite's ref and its bundle name (`composite_bundle_label`)."""
    return {
        ref: composite_bundle_label(index, composite)
        for ref, composite in DECLARED_COMPOSITES.items()
    }


def composite_summary(index: CodeIndex, composite: CompositeDeclaration) -> str:
    """One sentence on what `composite` adds or reads, derived from its declaration for a
    bundle reader: a stacked composite of several components adds them for each provider,
    recipient and year, one of one component reads it, and a spliced one joins its
    components. Each component is named by its table's label in `index`, after the labels of
    the codes it binds and the words of the declared basis it sets (`BASIS_MEASURES`), with
    the years its periods hold. A component binding its destinations to the bilateral side
    (`ComponentRecipe.destinations`) states what that reading counts
    (`BILATERAL_SOURCES`) where it binds the source's published total, once for the whole
    composite where its one component does, and beside its own component where there are
    several, so a component that adds an imputed share is not worded as the bilateral total."""
    shared = len(composite.components) == 1 and composite.components[0].destinations is not None
    phrases = [
        _component_phrase(index, recipe, reading=not shared) for recipe in composite.components
    ]
    joined = " and ".join(phrases)
    if composite.kind == "spliced":
        sentence = f"Joins {joined}"
    elif len(phrases) > 1:
        sentence = f"Adds {joined} for each provider, recipient and year"
    else:
        sentence = f"Reads {joined}"
    if shared and (reading := _destination_reading(composite.components[0])):
        sentence += f", counting {reading}"
    return sentence + "."


def _destination_reading(recipe: ComponentRecipe) -> str | None:
    """What a component binding its destinations to the bilateral side counts, as the
    `BILATERAL_SOURCES` row of its dataset words it, where the component binds that row's
    published total (`code` on `code_system`)."""
    if recipe.destinations is None:
        return None
    return next(
        (
            row.label
            for row in BILATERAL_SOURCES
            if row.dataset == recipe.dataset and recipe.bind.get(row.code_system) == row.code
        ),
        None,
    )


def _component_phrase(index: CodeIndex, recipe: ComponentRecipe, *, reading: bool) -> str:
    """`recipe`'s table by its label, after what it binds and its basis, with its years, and
    its destination reading when `reading`."""
    declared = index.datasets.get(recipe.dataset)
    table = declared.label if declared is not None else recipe.dataset
    qualifiers = [
        row.label if (row := index.code_by_dcid(code)) is not None else code
        for code in recipe.bind.values()
    ]
    if recipe.basis is not None:
        basis = next(
            (
                row.words
                for row in BASIS_MEASURES
                if (row.measured_property, row.netting_basis) == recipe.basis
            ),
            None,
        )
        if basis is not None:
            qualifiers.append(basis)
    phrase = f"{' as '.join(qualifiers)} from {table}" if qualifiers else table
    periods = recipe.periods
    if periods is not None and periods.first is not None and periods.last is not None:
        phrase += f" for {periods.first} to {periods.last}"
    elif periods is not None and periods.last is not None:
        phrase += f" for years to {periods.last}"
    elif periods is not None and periods.first is not None:
        phrase += f" for years from {periods.first}"
    if reading and (counted := _destination_reading(recipe)):
        phrase += f", counting {counted}"
    return phrase


def composite_notes(
    provenance_refs: Sequence[str | None], *, index: CodeIndex
) -> tuple[CompositeNote, ...]:
    """Each declared composite (`code_index.composites.DECLARED_COMPOSITES`) among
    `provenance_refs`, the provenances a member's rows carry, once each in first-seen order,
    with its bundle name (`composite_bundle_label`) and what it adds or reads
    (`composite_summary`)."""
    return tuple(
        CompositeNote(
            ref=ref,
            label=composite_bundle_label(index, composite),
            summary=composite_summary(index, composite),
        )
        for ref in dict.fromkeys(provenance_refs)
        if ref is not None and (composite := DECLARED_COMPOSITES.get(ref)) is not None
    )


def source_provenance(ref: str, cited: Mapping[str, object] | None) -> SourceProvenance:
    """One bundle source from the stored citation entry of `ref`, in its JSON form. A ref the
    citation lacks, or an entry without a label, takes the ref as its name. A missing entry has
    no url, licence, documentation or publisher. The cited source node's name is the publisher,
    with or without documentation."""
    if cited is None:
        return SourceProvenance(ref=ref, name=ref)
    source = cast("Mapping[str, object] | None", cited.get("source"))
    documentation = cast("Mapping[str, object] | None", cited.get("documentation"))
    return SourceProvenance(
        ref=ref,
        name=str(cited.get("label") or ref),
        url=_text(cited, "url"),
        licence=Licence(
            type_id=_text(cited, "license_type"),
            label=_text(cited, "license"),
            url=_text(cited, "license_url"),
        ),
        documentation=(_source_documentation(documentation) if documentation is not None else None),
        publisher=_text(source, "label") if source is not None else None,
    )


def _source_documentation(documentation: Mapping[str, object]) -> SourceDocumentation:
    """The bundle's view of a source's documentation snapshot."""
    return SourceDocumentation(
        publisher=_text(documentation, "publisher"),
        description=_text(documentation, "description"),
        source_data_url=_text(documentation, "source_data_url"),
        attribution=_text(documentation, "attribution"),
        update_cadence=_text(documentation, "update_cadence"),
        last_update=_text(documentation, "last_update"),
        years_covered=_text(documentation, "years_covered"),
        granularity=_text(documentation, "granularity"),
        currencies=tuple(cast("Sequence[str]", documentation.get("currencies") or ())),
        transformation_log=tuple(
            TransformationDoc(
                title=cast(str, step["transformation"]), description=cast(str, step["description"])
            )
            for step in _entries(documentation, "transformation_log")
        ),
        version_history=tuple(
            ReleaseDoc(
                date=cast(str, release["date"]),
                title=cast(str, release["title"]),
                changes=tuple(
                    ChangeDoc(kind=cast(str, change["kind"]), note=cast(str, change["note"]))
                    for change in _entries(release, "changes")
                ),
            )
            for release in _entries(documentation, "version_history")
        ),
    )


def _text(entry: Mapping[str, object], key: str) -> str | None:
    return cast("str | None", entry.get(key))


def _entries(entry: Mapping[str, object], key: str) -> Sequence[Mapping[str, object]]:
    return cast("Sequence[Mapping[str, object]]", entry.get(key) or ())
