"""Maps a version's stored citation of one source to the bundle's `SourceProvenance`.

Both bundle producers cite sources from the same stored shape: a Datasets version body's
`citation.sources[]` entry, as `datasets_api.manifest_models.CitationSource` writes it in JSON
mode. `render` dumps its parsed citation to that form, and a card export reads it from the
frame manifest, where derived frames carry the same shape. Nothing here reads the graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from materialise.bundle import (
    ChangeDoc,
    Licence,
    ReleaseDoc,
    SourceDocumentation,
    SourceProvenance,
    TransformationDoc,
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
