"""Reads source documentation from a node and its Dataset and Provenance children.

`read_provenance_doc` returns graph-derived details (`details.name_of_source`,
`details.publisher`, `details.description`, `details.source_url`,
`details.source_data_url`, `details.citation`), the origin catalogue's
`details.source` and `details.data_commons` when the caller passes a
catalogue, licence fields (`license.type`, `license.url`,
`license.update_cadence`, `license.last_update`,
`license.next_expected_update`),
`scope_and_units.years_covered`, the four parsed Markdown documents
(`transformation_log`, `version_history`, `methodology.description`,
`data_notes`), any `document_errors`, `scope_and_units.currencies` and
`scope_and_units.granularity` when the caller passes a `coverage_facts_of`
lookup, and this reader's own `gaps` list. It makes at most five bounded reads:

1. the node itself, for its identity arcs (`typeOf`, `name`, `url`,
   `isPartOf`, `source`);
2. its `isPartOf` parents, only when the node is a Provenance that has any.
   The first parent typed Dataset is its parent Dataset. A Provenance whose
   parents are all something else (a Source, a `ProvisionalNode`, a node the
   graph does not hold) is an orphan;
3. the Dataset's own children, via the reverse `isPartOf` arc filtered to
   `{typeOf:Provenance}`, keeping those the caller's catalogue admits;
4. one batched `node_arcs_many` covering the publisher's `name`, every
   child's roll-up properties, and a sole child's `description`, `url` and
   four documents;
5. the `name` of the agreed `licenseType` node, only when every child names
   the same one.

The graph chain is Source → Dataset → Provenance. Datasets link to Sources
through `isPartOf`; Provenance nodes link to Datasets through `isPartOf` and
may also link directly to Sources. An orphan Provenance has no parent Dataset,
acts as its own sole child, and is read in the batch like any other child. The
batch is skipped only when there is neither a publisher nor a child to read.

Roll-up rules apply to a Dataset's Provenance children. With a catalogue,
they are the children it admits, and a Dataset with none raises
`SourceDocumentNotFound`, as the source index lists no entry for it.
`details.description`, `details.source_url`, `license.type`,
`license.update_cadence` and `scope_and_units.years_covered` come from
`origin_catalogue.card_facts.card_facts_of`, the function the catalogue job
computes the source index's card facts with, so a card and its page agree.
The Dataset supplies `name_of_source` and `source_url`, and the publisher
comes from the Source's `name`. A sole child's `url` fills a missing Dataset
URL, and the node's own name fills a missing Dataset name. `description` and
the four documents come from the sole child only. A Dataset with several
children leaves them empty, gapped, and error-free, because the batch requests
them only when there is exactly one child. It requests `name` only when there is a Source.

`last_update` is the maximum child `lastDataRefreshDate`; `next_expected_update`
is the minimum `nextDataRefreshDate`, the pipeline's own next refresh. ISO 8601
dates sort chronologically as strings. `years_covered` uses each date's leading
four-digit year, takes the minimum start and maximum end, and returns `YYYY`,
`YYYY-YYYY`, or `None` if either endpoint has no readable year.

`license.type`, `license.url` (`license`), `update_cadence`, `citation`
(`licenseAttribution`) and `source_data_url` (`sourceDataUrl`) are each
returned only when every child has exactly one value and all values agree. A
missing or conflicting child empties the field. `license.type` is the label of
the agreed `licenseType` node: its `name` in the graph, else its dcid. The
other four are plain text as stored. `sourceDataUrl` and `license` come from
the children only, because Datasets carry neither.

The four document properties (`dataTransformationDescription`,
`dataTransformationChangelog`, `sourceMethodologyDescription`, `dataNotes`)
hold Markdown stored MCF-escaped. Each is decoded with
`dfo_provenance_docs.decode_mcf_text` and parsed against its template by the
matching `dfo_provenance_docs` parser. Identical repeated values collapse to
one. A stored value longer than six characters per byte of
`dfo_provenance_docs.MAX_DOCUMENT_BYTES` cannot decode within that limit, so it
is refused as `document_too_large` before decoding. A value that is refused,
fails to decode, or breaks its template, or a node carrying
two distinct values of one document property (`conflicting_values`), leaves
the member empty and gapped, adds a `DocumentIssue`, and logs a WARNING. The
rest of the document is still served.

Filter `UNRESOLVED_NODE` from every property tuple after each graph read.
This also protects `isPartOf` and `source` lookups, which choose the parent
Dataset and publisher.

The reverse `isPartOf` read is one hop. A transitive query would mix
Provenance children from other Datasets under the same Source.

`scope_and_units.countries` is empty because the graph has no path from a
source to its StatVars and no place-coverage property on Dataset or Provenance.
`details.source` and `details.data_commons` come from the caller's catalogue,
which holds a source and a Data Commons for each provenance. A Provenance dcid
uses its own row, and its `name` becomes `details.name_of_source`. A Dataset
dcid uses its admitted children's rows when they agree. Without a catalogue,
or when a Provenance is not admitted or the rows disagree, that member is
empty and gapped.

`scope_and_units.currencies` and `scope_and_units.granularity` come from the
caller's `coverage_facts_of` lookup, called once with the node's Provenance
dcids (the node itself for an orphan Provenance, else every Provenance child of
its Dataset). Without a lookup, or when the lookup leaves a fact unknown (`None`,
or no currencies), that member is empty and gapped. `with_coverage_facts` sets
both facts, and the gaps they decide, on a document already read.

`read_source_citations` is the citation-grain sibling. It returns the name, url,
licence, source and Data Commons for each cited provenance ref
(`datasets_api.manifest.build_citation`). The licence is the provenance's
`licenseType` dcid, that node's label (its `name`, else its dcid) and the
provenance's `license` terms page. Asked for documentation, it also attaches each
cited ref's document. A ref the origin catalogue lists is cited from its catalogue
row and entry with no graph read. The catalogue job derives the licence and the
entry's documentation facts with the `origin_catalogue.card_facts` functions this
module derives them with, and `_documented` turns either into the same
`ProvenanceDoc`. Any other ref is read from the graph, its document by
`read_provenance_doc`. A ref nothing documents keeps its citation without a
document, and a ref whose document read fails keeps its citation without one and is
marked `documentation_read_failed`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Protocol

from dfo_provenance_docs import (
    MAX_DOCUMENT_BYTES,
    ChangelogRelease,
    DocumentError,
    DocumentErrorReason,
    TransformationStep,
    decode_mcf_text,
    parse_changelog,
    parse_notes,
    parse_prose,
    parse_transformations,
)

from dc_graph import ArcTarget, GraphReadError, display_name
from dfo_platform.roles import RoleAxis
from materialise.cadence import known_cadence_label
from materialise.graph import SubjectPage
from materialise.roles import axis_noun
from origin_catalogue import Catalogue, DataCommons, SourceRef
from origin_catalogue.card_facts import (
    DATASET_TYPE,
    DOCUMENT_PROPERTIES,
    LICENCE_PROPERTIES,
    PROVENANCE_TYPE,
    agreed_child_value,
    card_facts_of,
    first_dataset_parent,
    licence_labels_of,
    licence_of,
    node_label,
    publisher_node_of,
    source_documentation_of,
    without_unresolved,
)
from origin_catalogue.catalogue import (
    CardFacts,
    ProvenanceRow,
    SourceDocumentation,
    entry_ref,
)

if TYPE_CHECKING:
    from series_coverage.index import CoverageIndex

logger = logging.getLogger(__name__)

# The named node's identity arcs. Roll-up properties come from the batch.
_NODE_PROPERTIES: tuple[str, ...] = ("typeOf", "name", "url", "isPartOf", "source")

# A Provenance's Dataset parent supplies its name, URL, and Source parent.
_DATASET_PROPERTIES: tuple[str, ...] = ("name", "url", "isPartOf")
# The parent read adds `typeOf`, since only a parent typed Dataset counts.
_PARENT_DATASET_PROPERTIES: tuple[str, ...] = ("typeOf", *_DATASET_PROPERTIES)

# The Source's name, requested only when there is a Source.
_PUBLISHER_PROPERTIES: tuple[str, ...] = ("name",)

# The roll-up fields read from every child.
_ROLLUP_PROPERTIES: tuple[str, ...] = (
    "licenseType",
    "sourceReleaseFrequency",
    "lastDataRefreshDate",
    "nextDataRefreshDate",
    "earliestObservationDate",
    "latestObservationDate",
    "licenseAttribution",
    "license",
    "sourceDataUrl",
)

# The MCF-escaped Markdown documents, keyed by graph property to the wire member
# they fill. The batch requests them only when there is exactly one child.
_TRANSFORMATIONS, _CHANGELOG, _METHODOLOGY, _NOTES = DOCUMENT_PROPERTIES
# The wire paths of the four documents.
DocumentMember = Literal[
    "transformation_log", "version_history", "methodology.description", "data_notes"
]
_DOCUMENT_MEMBERS: dict[str, DocumentMember] = {
    _TRANSFORMATIONS: "transformation_log",
    _CHANGELOG: "version_history",
    _METHODOLOGY: "methodology.description",
    _NOTES: "data_notes",
}

# The fields read from a sole child only: its description, its URL (which
# fills a missing Dataset URL) and the four documents.
_SOLE_CHILD_PROPERTIES: tuple[str, ...] = ("description", "url", *DOCUMENT_PROPERTIES)

# The longest stored value a `MAX_DOCUMENT_BYTES` document can have. JSON
# escaping spells one decoded byte in at most six characters (`\u00XX`), so a
# longer value decodes past the grammar's limit and is refused before decoding.
_MAX_STORED_DOCUMENT_CHARS = 6 * MAX_DOCUMENT_BYTES

# Cap child reads at 256 Provenance nodes per Dataset.
_MAX_DATASET_CHILDREN = 256

# The reason for a node carrying two distinct values of one document property.
# Every other reason comes from `dfo_provenance_docs.DocumentErrorReason`.
CONFLICTING_VALUES: Literal["conflicting_values"] = "conflicting_values"


class NodeArcsReader(Protocol):
    """The batched node read a citation's identity takes (`_read_unlisted_citations`). Graph
    clients and test doubles satisfy it structurally."""

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        """Read the requested arcs by dcid; missing nodes map to `None`."""
        ...


class ProvenanceNameReader(Protocol):
    """The one graph read `provenance_names` makes. Graph clients and test doubles satisfy it
    structurally."""

    def node_arc_targets_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        """Each dcid's arcs with their targets' names and types; missing nodes map to `None`."""
        ...


class ProvenanceGraphReader(NodeArcsReader, Protocol):
    """Narrow interface for the two graph reads this module uses.
    Graph clients and test doubles satisfy it structurally.
    """

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage:
        """Read one page of subjects; `SubjectPage.truncated` marks a partial result."""
        ...


class SourceDocumentNotFound(Exception):
    """The root `dcid` is not held by this deployment's graph, or its Dataset
    has no Provenance child the caller's catalogue admits."""

    def __init__(self, dcid: str, *, detail: str | None = None) -> None:
        self.dcid = dcid
        super().__init__(detail or f"graph does not hold {dcid!r}")


class NotASourceNode(Exception):
    """The root is present, but its `typeOf` includes neither `Dataset` nor `Provenance`."""

    def __init__(self, dcid: str, type_of: tuple[str, ...]) -> None:
        self.dcid = dcid
        self.type_of = type_of
        found = "/".join(type_of) if type_of else "no recognised type"
        super().__init__(
            f"{dcid!r} is a {found} node, not a Dataset or Provenance; this door "
            "documents a publication, not a publisher"
        )


class TooManyChildProvenances(Exception):
    """The Dataset has more than `_MAX_DATASET_CHILDREN` Provenance children.
    A partial child set could hide a licence or cadence conflict.
    """

    def __init__(self, dataset_dcid: str) -> None:
        self.dataset_dcid = dataset_dcid
        super().__init__(
            f"{dataset_dcid!r} reported more than {_MAX_DATASET_CHILDREN} Provenance "
            "children; refusing to roll up a truncated child set"
        )


@dataclass(frozen=True)
class DocumentIssue:
    """A document property that could not be served.

    `member` is the wire path the document fills. `reason` is a
    `DocumentErrorReason` from the grammar, or `CONFLICTING_VALUES` when the
    node carries two distinct values. `line` is 1-based in the decoded
    document, or `None` when the problem has no line.
    """

    member: DocumentMember
    reason: DocumentErrorReason | Literal["conflicting_values"]
    line: int | None


@dataclass(frozen=True, slots=True)
class CoverageFacts:
    """Facts the coverage index computes for a document's Provenances.

    `currencies` holds the sorted ISO 4217 codes of every currency the
    Provenances carry, empty when none is known. `granularity` is a display
    phrase such as `Annual, by provider and recipient`, `None` when unknown.
    """

    currencies: tuple[str, ...] = ()
    granularity: str | None = None


_NO_COVERAGE_FACTS = CoverageFacts()

# The one axis pair a granularity phrase names, origin noun first.
_FLOW_AXES: frozenset[RoleAxis | None] = frozenset({"origin", "destination"})


def coverage_facts_from_index(
    index: CoverageIndex, provenance_dcids: tuple[str, ...]
) -> CoverageFacts:
    """The `CoverageFacts` the coverage index gives these Provenances, the
    `coverage_facts_of` lookup `read_provenance_doc` takes.

    `currencies` are the sorted ISO codes of every currency facet the
    Provenances carry. A Provenance without currency-valued coverage series
    adds none. `granularity` is `"<period label>, by <origin noun> and
    <destination noun>"` when every Provenance has coverage series, all share
    one observation period that `materialise.cadence` labels, and the union of
    their variables' role axes is the flow pair. The union is the finest
    granularity a source serves, so one-role totals beside by-creditor
    variables still read as a two-axis source. Anything else is `None`.

    Each fact is derived under its own guard, so a failure logs a WARNING and
    empties only that fact, and the function does not raise.
    """
    return CoverageFacts(
        currencies=_guarded_fact("currencies", _currencies_of, (), index, provenance_dcids),
        granularity=_guarded_fact("granularity", _granularity_of, None, index, provenance_dcids),
    )


def _guarded_fact[T](
    fact: str,
    derive: Callable[[CoverageIndex, tuple[str, ...]], T],
    empty: T,
    index: CoverageIndex,
    provenance_dcids: tuple[str, ...],
) -> T:
    """Derive one coverage fact, logging a failure at WARNING and returning `empty`."""
    try:
        return derive(index, provenance_dcids)
    except Exception:
        logger.warning(
            "Could not derive %s for Provenances %s from the coverage index; it is empty",
            fact,
            list(provenance_dcids),
            exc_info=True,
        )
        return empty


def _currencies_of(index: CoverageIndex, provenance_dcids: tuple[str, ...]) -> tuple[str, ...]:
    currencies = index.provenance_currencies
    return tuple(
        sorted({code for provenance in provenance_dcids for code in currencies.get(provenance, ())})
    )


def _granularity_of(index: CoverageIndex, provenance_dcids: tuple[str, ...]) -> str | None:
    shapes_by_provenance = index.provenance_shapes
    if not all(provenance in shapes_by_provenance for provenance in provenance_dcids):
        return None
    shapes = [shapes_by_provenance[provenance] for provenance in provenance_dcids]
    periods = {period for shape in shapes for period in shape.observation_periods}
    axes = frozenset(axis for shape in shapes for axis_set in shape.role_axes for axis in axis_set)
    if len(periods) != 1 or axes != _FLOW_AXES:
        return None
    (period,) = periods
    label = known_cadence_label(period)
    if label is None:
        return None
    return f"{label}, by {axis_noun('origin')} and {axis_noun('destination')}"


@dataclass(frozen=True)
class ProvenanceDoc:
    """Graph-derived source details, parsed documents, and gap paths. `gaps`
    names every empty member in spec order. `license_type` is the licence's
    label, the `name` of its `licenseType` node; `update_cadence` is the raw ISO 8601
    `sourceReleaseFrequency` token. `source_data_url` (`sourceDataUrl`) and
    `license_url` (`license`) are the stored strings.
    """

    dcid: str
    provenances: tuple[str, ...]
    """The Provenance dcids the document covers: the node itself for an
    orphan Provenance, else every Provenance child of its Dataset."""
    name_of_source: str | None
    publisher: str | None
    source: SourceRef | None
    data_commons: DataCommons | None
    description: str | None
    source_url: str | None
    source_data_url: str | None
    citation: str | None
    license_type: str | None
    license_url: str | None
    update_cadence: str | None
    last_update: str | None
    next_expected_update: str | None
    years_covered: str | None
    granularity: str | None
    currencies: tuple[str, ...]
    methodology: str | None
    data_notes: tuple[str, ...]
    transformation_log: tuple[TransformationStep, ...]
    version_history: tuple[ChangelogRelease, ...]
    document_errors: tuple[DocumentIssue, ...]
    gaps: tuple[str, ...]


def _first(values: tuple[str, ...]) -> str | None:
    return values[0] if values else None


def _read_document[T](
    arcs: Mapping[str, tuple[str, ...]],
    prop: str,
    parse: Callable[[str], T],
    *,
    dcid: str,
    issues: list[DocumentIssue],
) -> T | None:
    """Decode and parse one document property of the sole child.

    Return `None` when the property is absent. Two distinct values, a stored
    value longer than `_MAX_STORED_DOCUMENT_CHARS` (refused before decoding),
    an undecodable value, or a template violation also return `None`, after
    appending a `DocumentIssue` and logging a WARNING.
    """
    member = _DOCUMENT_MEMBERS[prop]
    distinct = tuple(dict.fromkeys(arcs.get(prop, ())))
    if not distinct:
        return None
    if len(distinct) > 1:
        issue = DocumentIssue(member=member, reason=CONFLICTING_VALUES, line=None)
        detail = f"{len(distinct)} distinct values"
    elif len(distinct[0]) > _MAX_STORED_DOCUMENT_CHARS:
        issue = DocumentIssue(
            member=member, reason=DocumentErrorReason.DOCUMENT_TOO_LARGE, line=None
        )
        detail = (
            f"the stored value is {len(distinct[0])} characters, past the "
            f"{_MAX_STORED_DOCUMENT_CHARS}-character ceiling of an escaped "
            f"{MAX_DOCUMENT_BYTES}-byte document"
        )
    else:
        try:
            return parse(decode_mcf_text(distinct[0]))
        except DocumentError as exc:
            issue = DocumentIssue(member=member, reason=exc.reason, line=exc.line)
            detail = exc.detail
    issues.append(issue)
    logger.warning(
        "Source document %s: %s unusable (reason=%s, line=%s): %s",
        dcid,
        member,
        issue.reason,
        issue.line,
        detail,
    )
    return None


def _parent_dataset(
    graph: ProvenanceGraphReader, parents: tuple[str, ...], *, deadline: float | None
) -> tuple[str | None, dict[str, tuple[str, ...]] | None]:
    """Return a Provenance's first `isPartOf` parent typed Dataset, with its
    arcs (`first_dataset_parent`), or `(None, None)` for an orphan."""
    if not parents:
        return None, None
    parent_arcs = without_unresolved(
        graph.node_arcs_many(
            list(parents), properties=_PARENT_DATASET_PROPERTIES, deadline=deadline
        )
    )
    return first_dataset_parent(parents, parent_arcs)


def read_provenance_doc(
    graph: ProvenanceGraphReader,
    dcid: str,
    *,
    deadline: float | None = None,
    catalogue: Catalogue | None = None,
    coverage_facts_of: Callable[[tuple[str, ...]], CoverageFacts] | None = None,
) -> ProvenanceDoc:
    """Assemble source documentation for a Dataset or Provenance.
    Raise `SourceDocumentNotFound` for a missing node or, with a catalogue, a
    Dataset none of whose children it admits, `NotASourceNode` for another
    type, or `TooManyChildProvenances` when the child page truncates.
    A direct Dataset uses its own first read; an orphan Provenance is its own
    sole child. Dataset takes precedence if both types appear.

    `deadline` is one absolute `time.monotonic()` value passed unchanged to
    every read. `None` uses each read's configured timeout.

    `catalogue` decides which of a Dataset's children the document covers and
    supplies `source` and `data_commons` for the node's Provenances.
    `coverage_facts_of` receives the node's Provenance dcids and
    returns their `CoverageFacts`. It runs after the graph reads and must not
    raise.
    """
    node_arcs = without_unresolved(
        graph.node_arcs_many([dcid], properties=_NODE_PROPERTIES, deadline=deadline)
    )
    node = node_arcs.get(dcid)
    if node is None:
        raise SourceDocumentNotFound(dcid)

    type_of = node.get("typeOf", ())
    is_dataset = DATASET_TYPE in type_of
    is_provenance = PROVENANCE_TYPE in type_of
    if not is_dataset and not is_provenance:
        raise NotASourceNode(dcid, type_of)

    dataset_dcid: str | None
    dataset_arcs: dict[str, tuple[str, ...]] | None
    if is_dataset:
        dataset_dcid = dcid
        dataset_arcs = node
    else:
        dataset_dcid, dataset_arcs = _parent_dataset(
            graph, node.get("isPartOf", ()), deadline=deadline
        )

    source_dcid = publisher_node_of(dataset_arcs, node)

    if dataset_dcid is not None:
        page = graph.node_subjects(
            dataset_dcid,
            property="isPartOf",
            filters={"typeOf": PROVENANCE_TYPE},
            max_results=_MAX_DATASET_CHILDREN,
            deadline=deadline,
        )
        if page.truncated:
            raise TooManyChildProvenances(dataset_dcid)
        children_dcids = [
            child for child in page.dcids if catalogue is None or catalogue.is_admitted(child)
        ]
        if catalogue is not None and not children_dcids:
            raise SourceDocumentNotFound(
                dcid,
                detail=(
                    f"{dcid!r} documents Dataset {dataset_dcid!r}, which has no Provenance "
                    "child this deployment admits"
                ),
            )
    else:
        children_dcids = [dcid]
    sole_child = children_dcids[0] if len(children_dcids) == 1 else None

    batch_targets = list(
        dict.fromkeys([*([source_dcid] if source_dcid is not None else []), *children_dcids])
    )
    batch_properties = (
        *(_PUBLISHER_PROPERTIES if source_dcid is not None else ()),
        *(_ROLLUP_PROPERTIES if children_dcids else ()),
        *(_SOLE_CHILD_PROPERTIES if sole_child is not None else ()),
    )
    # Skip the batch when no publisher or child exists, so this path makes no read.
    batch = (
        without_unresolved(
            graph.node_arcs_many(batch_targets, properties=batch_properties, deadline=deadline)
        )
        if batch_targets
        else {}
    )
    child_properties: dict[str, dict[str, tuple[str, ...]]] = {
        c: (batch.get(c) or {}) for c in children_dcids
    }

    publisher = node_label(batch.get(source_dcid), source_dcid) if source_dcid is not None else None

    dataset_name = (
        node_label(dataset_arcs, dataset_dcid)
        if dataset_arcs is not None and dataset_dcid is not None
        else None
    )
    own_row = (
        catalogue.provenance(dcid)
        if catalogue is not None
        and is_provenance
        and not is_dataset
        and catalogue.is_admitted(dcid)
        else None
    )
    if own_row is not None:
        name_of_source = own_row.name
    else:
        name_of_source = dataset_name if dataset_name is not None else node_label(node, dcid)

    licence = agreed_child_value(child_properties, "licenseType")
    licence_labels = (
        licence_labels_of(
            without_unresolved(
                graph.node_arcs_many([licence], properties=LICENCE_PROPERTIES, deadline=deadline)
            )
        )
        if licence is not None
        else {}
    )
    source, data_commons = (
        _agreed_origin(catalogue, (dcid,) if is_provenance and not is_dataset else children_dcids)
        if catalogue is not None
        else (None, None)
    )
    return _documented(
        dcid,
        name_of_source=name_of_source,
        source=source,
        data_commons=data_commons,
        card=card_facts_of(dataset_arcs, child_properties, licence_labels),
        documentation=source_documentation_of(child_properties, publisher),
        coverage_facts_of=coverage_facts_of,
    )


def _documented(
    dcid: str,
    *,
    name_of_source: str | None,
    source: SourceRef | None,
    data_commons: DataCommons | None,
    card: CardFacts,
    documentation: SourceDocumentation,
    coverage_facts_of: Callable[[tuple[str, ...]], CoverageFacts] | None,
) -> ProvenanceDoc:
    """The `ProvenanceDoc` for `dcid` from its entry's card facts and documentation facts,
    whether read from the graph (`read_provenance_doc`) or the origin catalogue
    (`_catalogued_documentation`). It parses the four documents and calls
    `coverage_facts_of` with `documentation.provenances`."""
    documents = documentation.documents
    issues: list[DocumentIssue] = []
    methodology = _read_document(documents, _METHODOLOGY, parse_prose, dcid=dcid, issues=issues)
    data_notes = _read_document(documents, _NOTES, parse_notes, dcid=dcid, issues=issues)
    transformation_log = _read_document(
        documents, _TRANSFORMATIONS, parse_transformations, dcid=dcid, issues=issues
    )
    version_history = _read_document(
        documents, _CHANGELOG, parse_changelog, dcid=dcid, issues=issues
    )

    coverage_facts = (
        coverage_facts_of(documentation.provenances)
        if coverage_facts_of is not None
        else _NO_COVERAGE_FACTS
    )

    doc = ProvenanceDoc(
        dcid=dcid,
        provenances=documentation.provenances,
        name_of_source=name_of_source,
        publisher=documentation.publisher,
        source=source,
        data_commons=data_commons,
        description=card.description,
        source_url=card.source_url,
        source_data_url=documentation.source_data_url,
        citation=documentation.citation,
        license_type=card.license_type,
        license_url=documentation.license_url,
        update_cadence=card.update_cadence,
        last_update=documentation.last_update,
        next_expected_update=documentation.next_expected_update,
        years_covered=card.years_covered,
        granularity=None,
        currencies=(),
        methodology=methodology,
        data_notes=data_notes or (),
        transformation_log=transformation_log or (),
        version_history=version_history or (),
        document_errors=tuple(issues),
        gaps=(),
    )
    return with_coverage_facts(doc, coverage_facts)


def with_coverage_facts(doc: ProvenanceDoc, facts: CoverageFacts) -> ProvenanceDoc:
    """`doc` with the currencies and granularity of `facts`, and its `gaps` recomputed to
    match. A caller that keeps a document read without a `coverage_facts_of` lookup composes
    each response's coverage facts with this, so the facts follow the coverage index the
    response reads."""
    updated = replace(doc, currencies=facts.currencies, granularity=facts.granularity)
    return replace(
        updated,
        gaps=_compute_gaps(
            name_of_source=updated.name_of_source,
            publisher=updated.publisher,
            has_source=updated.source is not None,
            has_data_commons=updated.data_commons is not None,
            description=updated.description,
            source_url=updated.source_url,
            source_data_url=updated.source_data_url,
            citation=updated.citation,
            license_type=updated.license_type,
            license_url=updated.license_url,
            update_cadence=updated.update_cadence,
            last_update=updated.last_update,
            next_expected_update=updated.next_expected_update,
            years_covered=updated.years_covered,
            granularity=updated.granularity,
            has_currencies=bool(updated.currencies),
            methodology=updated.methodology,
            has_data_notes=bool(updated.data_notes),
            has_transformation_log=bool(updated.transformation_log),
            has_version_history=bool(updated.version_history),
        ),
    )


def _agreed_origin(
    catalogue: Catalogue, provenances: Sequence[str]
) -> tuple[SourceRef | None, DataCommons | None]:
    """The source and the Data Commons every one of `provenances` shares, each `None`
    when a provenance is not admitted, they disagree, or there are none."""
    rows = [
        catalogue.provenance(provenance) if catalogue.is_admitted(provenance) else None
        for provenance in provenances
    ]
    admitted = [row for row in rows if row is not None]
    if not admitted or len(admitted) < len(rows):
        return None, None
    sources = {row.source for row in admitted}
    origins = {row.data_commons for row in admitted}
    return (
        sources.pop() if len(sources) == 1 else None,
        origins.pop() if len(origins) == 1 else None,
    )


# These spec paths have no source in this reader, so every document names them
# in `gaps`. Capabilities publishes this list directly.
UNCONDITIONAL_GAPS: tuple[str, ...] = (
    "scope_and_units.countries",
    "usage.api_access",
    "usage.python",
)


def _compute_gaps(
    *,
    name_of_source: str | None,
    publisher: str | None,
    has_source: bool,
    has_data_commons: bool,
    description: str | None,
    source_url: str | None,
    source_data_url: str | None,
    citation: str | None,
    license_type: str | None,
    license_url: str | None,
    update_cadence: str | None,
    last_update: str | None,
    next_expected_update: str | None,
    years_covered: str | None,
    granularity: str | None,
    has_currencies: bool,
    methodology: str | None,
    has_data_notes: bool,
    has_transformation_log: bool,
    has_version_history: bool,
) -> tuple[str, ...]:
    """Every empty member path, in the consuming spec's top-to-bottom
    declaration order: each graph-filled member when this source's read left
    it empty, interleaved with `UNCONDITIONAL_GAPS` at each one's position.
    """
    ordered: tuple[tuple[str, bool], ...] = (
        ("details.name_of_source", name_of_source is None),
        ("details.publisher", publisher is None),
        ("details.source", not has_source),
        ("details.data_commons", not has_data_commons),
        ("details.description", description is None),
        ("details.source_url", source_url is None),
        ("details.source_data_url", source_data_url is None),
        ("details.citation", citation is None),
        ("license.type", license_type is None),
        ("license.url", license_url is None),
        ("license.update_cadence", update_cadence is None),
        ("license.last_update", last_update is None),
        ("license.next_expected_update", next_expected_update is None),
        ("scope_and_units.countries", True),
        ("scope_and_units.years_covered", years_covered is None),
        ("scope_and_units.granularity", granularity is None),
        ("scope_and_units.currencies", not has_currencies),
        ("methodology.description", methodology is None),
        ("data_notes", not has_data_notes),
        ("usage.api_access", True),
        ("usage.python", True),
        ("transformation_log", not has_transformation_log),
        ("version_history", not has_version_history),
    )
    # `UNCONDITIONAL_GAPS` also feeds capabilities; keep its paths aligned with this spec-order table.
    return tuple(path for path, empty in ordered if empty)


# `read_source_citations` reads every cited node the catalogue does not list. `typeOf`
# separates a Provenance's `isPartOf` (its Dataset) from a Dataset's (its Source).
_CITATION_NODE_PROPERTIES: tuple[str, ...] = (
    "typeOf",
    "name",
    "url",
    "isPartOf",
    "licenseType",
    "license",
)

# Its second read covers every distinct `isPartOf` parent, typed so the Dataset is found,
# and every `licenseType` node.
_CITATION_LINKED_PROPERTIES: tuple[str, ...] = ("typeOf", "name", "url")


@dataclass(frozen=True)
class SourceCitation:
    """The name, url, licence, source, Data Commons and documentation citing one
    provenance. Each is `None` where neither the origin catalogue nor the graph has it.

    `licence` is the label of the provenance's `licenseType` node, `licence_type` that
    node's dcid, and `licence_url` the provenance's `license` terms page.
    `documentation` is the provenance's documentation, `None` when the caller did not ask
    for it, nothing documents it, or its graph read failed. `documentation_read_failed` is
    `True` only in the last case, which a later read can change. A provenance nothing
    documents stays undocumented until the catalogue or the graph changes.
    """

    dcid: str
    name: str | None
    url: str | None
    licence: str | None
    source: SourceRef | None = None
    data_commons: DataCommons | None = None
    licence_type: str | None = None
    licence_url: str | None = None
    documentation: ProvenanceDoc | None = None
    documentation_read_failed: bool = False


PROVENANCE_NAME_PROPERTIES: tuple[str, ...] = ("name", "typeOf", "isPartOf")
"""The arcs `provenance_names` reads of each provenance the catalogue does not list."""


def provenance_names(
    graph: ProvenanceNameReader,
    dcids: Sequence[str],
    *,
    catalogue: Catalogue,
    deadline: float | None,
    targets: Mapping[str, Mapping[str, tuple[ArcTarget, ...]] | None] | None = None,
) -> tuple[dict[str, str], bool]:
    """The name of each distinct provenance of `dcids`, by dcid, and whether the graph read
    succeeded. This is the name a source citation gives it (`read_source_citations`), since a
    base provenance's own graph `name` is its import's, such as `WHO_GHO`.

    A provenance `catalogue` lists takes its row's name with no read. Every other is read in
    one `node_arc_targets_many` of `PROVENANCE_NAME_PROPERTIES`, bounded by `deadline`, or
    taken from `targets` when the caller read those properties for it already. A node typed
    Dataset is named by its own `name`. Any other is named by its first `isPartOf` target
    typed Dataset (`first_dataset_parent`'s rule), from the name the read carries beside that
    target, else by its own `name`. Each name passes `display_name`'s guard, and a dcid with
    no name is absent. A target carries the first of several names where `display_name`
    takes the greatest; the 680 Dataset parents of the dev graph's 1,277 Provenances each
    carry one.

    A call makes no request when the catalogue lists every dcid, or one otherwise. A failed
    read, `GraphReadError` with the deadline included, is logged and returns the catalogue's
    names with `False`.
    """
    unique = list(dict.fromkeys(dcids))
    names = {dcid: row.name for dcid in unique if (row := catalogue.provenance(dcid)) is not None}
    unlisted = [dcid for dcid in unique if dcid not in names]
    if not unlisted:
        return names, True
    if targets is None:
        try:
            targets = graph.node_arc_targets_many(
                unlisted, properties=PROVENANCE_NAME_PROPERTIES, deadline=deadline
            )
        except GraphReadError:
            logger.warning(
                "Provenance name read failed for %d provenances the catalogue does not list; "
                "naming the listed ones alone",
                len(unlisted),
                exc_info=True,
            )
            return names, False
    for dcid in unlisted:
        node = targets.get(dcid)
        if node is None:
            continue
        parent = (
            None
            if DATASET_TYPE in (target.value for target in node.get("typeOf", ()))
            else next(
                (target for target in node.get("isPartOf", ()) if DATASET_TYPE in target.types),
                None,
            )
        )
        name = (
            display_name(() if parent.name is None else (parent.name,), parent.value)
            if parent is not None
            else None
        ) or display_name(tuple(target.value for target in node.get("name", ())), dcid)
        if name is not None:
            names[dcid] = name
    return names, True


def read_source_citations(
    graph: ProvenanceGraphReader,
    dcids: Sequence[str],
    *,
    catalogue: Catalogue,
    deadline: float | None = None,
    documented: bool = False,
    coverage_facts_of: Callable[[tuple[str, ...]], CoverageFacts] | None = None,
) -> dict[str, SourceCitation]:
    """One `SourceCitation` per distinct dcid the catalogue lists or the graph holds, in
    first-seen order.

    A dcid in `catalogue`'s provenance table is cited from its row alone, with no graph
    read: its name, url, source, Data Commons and licence (`ProvenanceRow.licence`), and,
    when `documented`, its entry's documentation (`_catalogued_documentation`). Any other
    dcid, such as a DFO provenance newer than the loaded catalogue, is read from the graph
    whole and carries no source or Data Commons:

    - `name`: the parent Dataset's `name` when the node is a Provenance with an `isPartOf`
      parent typed Dataset (`first_dataset_parent`), else the node's own, each through
      `node_label`'s self-reference guard. A node whose parents are missing or are not
      Datasets is its own publication, as for `read_provenance_doc`.
    - `url`: the node's `url`, else the publication's. The import's own page resolves where
      its Dataset's may 404.
    - the licence (`licence_of`): the node's `licenseType` when it asserts exactly one,
      labelled with that licence node's `name`, else its dcid, and the node's `license`
      when it asserts exactly one.

    A dcid that neither the catalogue lists nor the graph holds has no entry, and a call
    whose every dcid the catalogue lists reads nothing. The graph reads are one batched node
    read, and a second for the licence nodes and the `isPartOf` parents of unlisted
    Provenances.

    `documented` adds each unlisted dcid's `read_provenance_doc` document, read with
    `catalogue`, `deadline` and `coverage_facts_of`, one dcid after another
    (`_read_citation_documentation`). The citation itself stands whatever that read finds.
    """
    unique = list(dict.fromkeys(dcids))
    rows = {dcid: row for dcid in unique if (row := catalogue.provenance(dcid)) is not None}
    read = _read_unlisted_citations(
        graph, [dcid for dcid in unique if dcid not in rows], deadline=deadline
    )
    citations: dict[str, SourceCitation] = {}
    for dcid in unique:
        row = rows.get(dcid)
        if row is not None:
            citations[dcid] = _catalogued_citation(
                row,
                catalogue=catalogue,
                documented=documented,
                coverage_facts_of=coverage_facts_of,
            )
            continue
        citation = read.get(dcid)
        if citation is None:
            continue
        if documented:
            documentation, failed = _read_citation_documentation(
                graph,
                dcid,
                catalogue=catalogue,
                deadline=deadline,
                coverage_facts_of=coverage_facts_of,
            )
            citation = replace(
                citation, documentation=documentation, documentation_read_failed=failed
            )
        citations[dcid] = citation
    return citations


def _catalogued_citation(
    row: ProvenanceRow,
    *,
    catalogue: Catalogue,
    documented: bool,
    coverage_facts_of: Callable[[tuple[str, ...]], CoverageFacts] | None,
) -> SourceCitation:
    return SourceCitation(
        dcid=row.id,
        name=row.name,
        url=row.url,
        licence=row.licence.label,
        source=row.source,
        data_commons=row.data_commons,
        licence_type=row.licence.type,
        licence_url=row.licence.url,
        documentation=(
            _catalogued_documentation(row, catalogue=catalogue, coverage_facts_of=coverage_facts_of)
            if documented
            else None
        ),
    )


def _catalogued_documentation(
    row: ProvenanceRow,
    *,
    catalogue: Catalogue,
    coverage_facts_of: Callable[[tuple[str, ...]], CoverageFacts] | None,
) -> ProvenanceDoc | None:
    """The document `read_provenance_doc` gives a listed provenance, from the card facts and
    documentation facts of its catalogue entry (`entry_ref`), with the row's name, source and
    Data Commons. `None` when the catalogue holds no card facts or no documentation facts for
    the entry, as for a UN row, or the entry's documentation does not list the provenance."""
    ref = entry_ref(row)
    card = catalogue.dataset_facts(ref)
    documentation = catalogue.source_documentation(ref)
    if card is None or documentation is None or row.id not in documentation.provenances:
        return None
    return _documented(
        row.id,
        name_of_source=row.name,
        source=row.source,
        data_commons=row.data_commons,
        card=card,
        documentation=documentation,
        coverage_facts_of=coverage_facts_of,
    )


def _read_unlisted_citations(
    graph: NodeArcsReader, dcids: Sequence[str], *, deadline: float | None
) -> dict[str, SourceCitation]:
    """The graph's citation of each of `dcids` it holds, from at most two batched reads.
    An empty `dcids` reads nothing."""
    if not dcids:
        return {}
    nodes = without_unresolved(
        graph.node_arcs_many(dcids, properties=_CITATION_NODE_PROPERTIES, deadline=deadline)
    )
    held = {dcid: node for dcid in dcids if (node := nodes.get(dcid)) is not None}
    parent_dcids = [
        parent
        for node in held.values()
        if DATASET_TYPE not in node.get("typeOf", ())
        for parent in node.get("isPartOf", ())
    ]
    licence_dcids = [
        licence for node in held.values() if (licence := _sole(node.get("licenseType", ())))
    ]
    linked_dcids = list(dict.fromkeys([*parent_dcids, *licence_dcids]))
    linked = (
        without_unresolved(
            graph.node_arcs_many(
                linked_dcids, properties=_CITATION_LINKED_PROPERTIES, deadline=deadline
            )
        )
        if linked_dcids
        else {}
    )
    licence_labels = licence_labels_of({licence: linked.get(licence) for licence in licence_dcids})

    citations: dict[str, SourceCitation] = {}
    for dcid, node in held.items():
        licence = licence_of(node, licence_labels)
        parent, parent_arcs = (
            (None, None)
            if DATASET_TYPE in node.get("typeOf", ())
            else first_dataset_parent(node.get("isPartOf", ()), linked)
        )
        publication = parent or dcid
        publication_arcs = parent_arcs or node
        citations[dcid] = SourceCitation(
            dcid=dcid,
            name=node_label(publication_arcs, publication) or node_label(node, dcid),
            url=_first(node.get("url", ())) or _first(publication_arcs.get("url", ())),
            licence=licence.label,
            licence_type=licence.type,
            licence_url=licence.url,
        )
    return citations


def _sole(values: tuple[str, ...]) -> str | None:
    return values[0] if len(values) == 1 else None


def _read_citation_documentation(
    graph: ProvenanceGraphReader,
    dcid: str,
    *,
    catalogue: Catalogue,
    deadline: float | None,
    coverage_facts_of: Callable[[tuple[str, ...]], CoverageFacts] | None,
) -> tuple[ProvenanceDoc | None, bool]:
    """`read_provenance_doc` for one cited dcid, and whether the read failed.

    A dcid the graph documents nothing for (`SourceDocumentNotFound`, `NotASourceNode`,
    `TooManyChildProvenances`) returns `(None, False)`, since the graph gives that answer
    until it changes. A `GraphReadError`, deadline expiry included, returns `(None, True)` after a
    logged WARNING. Every other error, `GraphCapExceededError` included, propagates.
    """
    try:
        documentation = read_provenance_doc(
            graph,
            dcid,
            deadline=deadline,
            catalogue=catalogue,
            coverage_facts_of=coverage_facts_of,
        )
    except (SourceDocumentNotFound, NotASourceNode, TooManyChildProvenances) as exc:
        logger.info("Source %s has no documentation to cite: %s", dcid, exc)
        return None, False
    except GraphReadError:
        logger.warning(
            "Source documentation read failed for %s; citing it without documentation",
            dcid,
            exc_info=True,
        )
        return None, True
    return documentation, False
