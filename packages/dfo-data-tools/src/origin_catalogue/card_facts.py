"""The facts a source card and its documentation page both show, the further
facts the page and a citation's documentation show, and the graph rules that
derive them.

A card describes one entry of `GET /api/graph/sources`: a Dataset, or a
provenance with no parent Dataset (an orphan), which stands for itself. The
catalogue job computes each entry's `CardFacts` at build time, and
`materialise.provenance_doc` computes the same members of the documentation
page with the same functions, so a card agrees with its page for one graph.

A Dataset's children are the admitted provenances that name it anywhere in
`isPartOf` and are typed `Provenance`. An orphan is its own sole child.

- `description` is the sole child's `description`, `None` for several children.
- `license_type` and `update_cadence` come from the value every child
  asserts (`agreed_child_value`). `update_cadence` is the raw
  `sourceReleaseFrequency` token. `license_type` is the label of the agreed
  `licenseType` node: its `name` in the graph, else its dcid
  (`licence_labels_of`). The caller reads the `LICENCE_PROPERTIES` of the
  licence nodes.
- `years_covered` spans the children's observation dates (`years_covered_of`).
- `source_url` is the Dataset's `url`, else its sole child's (`source_url_of`).

`source_documentation_of` derives the rest of the page's graph facts, which
the catalogue job stores per entry as `SourceDocumentation`, and
`licence_of` derives the licence a citation names for one provenance, which
the job stores on its row.

Every function here expects arcs with `UNRESOLVED_NODE` already removed
(`without_unresolved`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dc_graph import UNRESOLVED_NODE, display_name
from origin_catalogue.catalogue import CardFacts, Licence, SourceDocumentation

DATASET_TYPE = "Dataset"
PROVENANCE_TYPE = "Provenance"

type Arcs = Mapping[str, tuple[str, ...]]
"""One node's arcs, keyed by property."""

type ChildArcs = Mapping[str, Arcs]
"""Each Provenance child's arcs, keyed by child dcid."""

CHILD_PROPERTIES: tuple[str, ...] = (
    "licenseType",
    "sourceReleaseFrequency",
    "earliestObservationDate",
    "latestObservationDate",
    "description",
    "url",
)
"""The child arcs `card_facts_of` reads."""

LICENCE_PROPERTIES: tuple[str, ...] = ("name",)
"""The arcs of a `licenseType` node its label reads."""

DOCUMENTATION_CHILD_PROPERTIES: tuple[str, ...] = (
    "lastDataRefreshDate",
    "nextDataRefreshDate",
    "licenseAttribution",
    "license",
    "sourceDataUrl",
)
"""The child arcs `source_documentation_of` reads beyond `CHILD_PROPERTIES`,
from every child."""

DOCUMENT_PROPERTIES: tuple[str, ...] = (
    "dataTransformationDescription",
    "dataTransformationChangelog",
    "sourceMethodologyDescription",
    "dataNotes",
)
"""The MCF-escaped Markdown documents `source_documentation_of` reads from a
sole child."""


def without_unresolved(
    arcs_by_dcid: Mapping[str, Mapping[str, tuple[str, ...]] | None],
) -> dict[str, dict[str, tuple[str, ...]] | None]:
    """Every property tuple with the graph's unresolved-arc sentinel removed,
    so an `isPartOf` or `source` lookup never picks it."""
    return {
        dcid: (
            {
                prop: tuple(v for v in values if v != UNRESOLVED_NODE)
                for prop, values in arcs.items()
            }
            if arcs is not None
            else None
        )
        for dcid, arcs in arcs_by_dcid.items()
    }


def node_label(arcs: Arcs | None, dcid: str) -> str | None:
    """The `display_name` of node `dcid` among its arcs `arcs`, `None` for a node
    the graph does not hold."""
    return None if arcs is None else display_name(arcs.get("name", ()), dcid)


def first_dataset_parent[A: Arcs](
    parents: Sequence[str], parent_arcs: Mapping[str, A | None]
) -> tuple[str | None, A | None]:
    """The first of `parents` whose arcs type it `Dataset`, with those arcs. A
    parent the graph does not hold, or holds as another type (a Source, a
    `ProvisionalNode`), is not a parent Dataset, so a provenance with no
    such parent is an orphan."""
    for parent in parents:
        arcs = parent_arcs.get(parent)
        if arcs is not None and DATASET_TYPE in arcs.get("typeOf", ()):
            return parent, arcs
    return None, None


def _first(values: tuple[str, ...]) -> str | None:
    return values[0] if values else None


def _sole(values: tuple[str, ...]) -> str | None:
    return values[0] if len(values) == 1 else None


def _leading_year(value: str) -> int | None:
    """The leading four-digit year of `YYYY`, `YYYY-MM` or `YYYY-MM-DD`, else `None`."""
    head = value[:4]
    return int(head) if len(head) == 4 and head.isdigit() else None


def _agree(child_values: Mapping[str, tuple[str, ...]]) -> str | None:
    """The common value when every child has at least one value and all
    asserted values agree, else `None`. Two values from one child conflict."""
    if not child_values:
        return None
    all_values: set[str] = set()
    for values in child_values.values():
        if not values:
            return None
        all_values.update(values)
    return next(iter(all_values)) if len(all_values) == 1 else None


def agreed_child_value(children: ChildArcs, prop: str) -> str | None:
    """The one value of `prop` every child asserts, following `_agree`."""
    return _agree({child: arcs.get(prop, ()) for child, arcs in children.items()})


def licence_labels_of(licence_arcs: Mapping[str, Arcs | None]) -> dict[str, str]:
    """Each `licenseType` node's label: its `name` (`node_label`), else its
    dcid when the graph holds no name for it."""
    return {dcid: node_label(arcs, dcid) or dcid for dcid, arcs in licence_arcs.items()}


def years_covered_of(children: ChildArcs) -> str | None:
    """`YYYY` or `YYYY-YYYY` from the minimum `earliestObservationDate` year and
    the maximum `latestObservationDate` year across the children, or `None`
    when either end has no readable year."""
    earliest = [
        year
        for arcs in children.values()
        for value in arcs.get("earliestObservationDate", ())
        if (year := _leading_year(value)) is not None
    ]
    latest = [
        year
        for arcs in children.values()
        for value in arcs.get("latestObservationDate", ())
        if (year := _leading_year(value)) is not None
    ]
    if not earliest or not latest:
        return None
    start, end = min(earliest), max(latest)
    return str(start) if start == end else f"{start}-{end}"


def source_url_of(dataset_arcs: Arcs | None, sole_child_arcs: Arcs | None) -> str | None:
    """The Dataset's `url`, else the sole child's. An orphan has no Dataset
    and is its own sole child."""
    dataset_url = _first(dataset_arcs.get("url", ())) if dataset_arcs is not None else None
    if dataset_url is not None:
        return dataset_url
    return _first(sole_child_arcs.get("url", ())) if sole_child_arcs is not None else None


def card_facts_of(
    dataset_arcs: Arcs | None, children: ChildArcs, licence_labels: Mapping[str, str]
) -> CardFacts:
    """The facts of the entry whose Dataset has `dataset_arcs` (`None` for an
    orphan) and whose children have `children`. `licence_labels` holds the
    label of each `licenseType` node the children name (`licence_labels_of`)."""
    sole_child = next(iter(children.values())) if len(children) == 1 else None
    licence = agreed_child_value(children, "licenseType")
    return CardFacts(
        description=_first(sole_child.get("description", ())) if sole_child is not None else None,
        license_type=licence_labels.get(licence, licence) if licence is not None else None,
        update_cadence=agreed_child_value(children, "sourceReleaseFrequency"),
        years_covered=years_covered_of(children),
        source_url=source_url_of(dataset_arcs, sole_child),
    )


def licence_of(arcs: Arcs | None, licence_labels: Mapping[str, str]) -> Licence:
    """The licence a provenance with `arcs` states: its one `licenseType`,
    labelled from `licence_labels` (`licence_labels_of`), and its one
    `license` terms page."""
    licence = _sole((arcs or {}).get("licenseType", ()))
    return Licence(
        type=licence,
        label=licence_labels.get(licence, licence) if licence is not None else None,
        url=_sole((arcs or {}).get("license", ())),
    )


def publisher_node_of(dataset_arcs: Arcs | None, node_arcs: Arcs) -> str | None:
    """The Source whose `name` is a documentation's publisher: the Dataset's
    first `isPartOf`, or for an orphan (`dataset_arcs` is `None`) its own
    first `source`."""
    if dataset_arcs is not None:
        return _first(dataset_arcs.get("isPartOf", ()))
    return _first(node_arcs.get("source", ()))


def source_documentation_of(children: ChildArcs, publisher: str | None) -> SourceDocumentation:
    """The documentation facts of the entry whose children have `children`,
    which must carry `DOCUMENTATION_CHILD_PROPERTIES` and, for a sole child,
    `DOCUMENT_PROPERTIES`. `provenances` is every child. `publisher` is the
    publisher Source's label (`publisher_node_of`, `node_label`). ISO 8601
    dates sort chronologically as strings."""
    sole_child = next(iter(children.values())) if len(children) == 1 else None
    last_dates = [v for arcs in children.values() for v in arcs.get("lastDataRefreshDate", ())]
    next_dates = [v for arcs in children.values() for v in arcs.get("nextDataRefreshDate", ())]
    return SourceDocumentation(
        provenances=tuple(children),
        publisher=publisher,
        source_data_url=agreed_child_value(children, "sourceDataUrl"),
        citation=agreed_child_value(children, "licenseAttribution"),
        license_url=agreed_child_value(children, "license"),
        last_update=max(last_dates) if last_dates else None,
        next_expected_update=min(next_dates) if next_dates else None,
        documents=(
            {
                prop: tuple(dict.fromkeys(values))
                for prop in DOCUMENT_PROPERTIES
                if (values := sole_child.get(prop, ()))
            }
            if sole_child is not None
            else {}
        ),
    )


__all__ = [
    "CHILD_PROPERTIES",
    "DATASET_TYPE",
    "DOCUMENTATION_CHILD_PROPERTIES",
    "DOCUMENT_PROPERTIES",
    "LICENCE_PROPERTIES",
    "PROVENANCE_TYPE",
    "Arcs",
    "ChildArcs",
    "agreed_child_value",
    "card_facts_of",
    "first_dataset_parent",
    "licence_labels_of",
    "licence_of",
    "node_label",
    "publisher_node_of",
    "source_documentation_of",
    "source_url_of",
    "without_unresolved",
    "years_covered_of",
]
