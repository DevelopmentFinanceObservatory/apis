"""A part's source scope as the region pipeline reads it.

A part's source mentions compile into a `SourceScope` before the pipeline runs
(`search.engine.source_scope`). This module holds what the pipeline reads from it. That is
the unbound-qualifier records that name the scope's sources in a refusal (`scope_records`),
the mark and warning of an excluding mention that removes nothing (`with_idle_exclusions`,
`idle_exclusion_warnings`) and of an including mention that serves nothing of the answer
(`with_idle_inclusions`, `idle_inclusion_warnings`), and whether the scope allows a
provenance of the UN System Data Commons (`allows_un_system`). It also decides, before a
part's pass, whether the part reads each organisation its query names as its source or as a
party to the flow (`read_organisations`, `choose_readings`), and how the answer discloses
each reading (`read_as_source`, `read_as_party`).
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from code_index.composites import provenance_datasets
from code_index.query import CodeIndex
from code_index.shared import get_index
from materialise.roles import RoleAxis, roles_by_axis
from origin_catalogue import UN_SYSTEM, Catalogue, current_catalogue, scoped_reads
from search.engine.clients.graph import DetectedSV, EngineGraphClient
from search.engine.departures import (
    organisation_read_as_party,
    organisation_read_as_source,
    warning_key,
)
from search.engine.interpret import Detected, detect_measure
from search.engine.region.codes import (
    ORGANISATION_READ_AS_PARTY,
    ORGANISATION_READ_AS_SOURCE,
    QUERY_QUALIFIER_DROPPED,
)
from search.engine.region.flow_governance import governs
from search.engine.region.result import with_departure_facts
from search.engine.retrieval.bound_values import shape_datasets, shape_listed_rows
from search.engine.retrieval.topics import is_topic_candidate
from search.engine.shape import ShapeDraft
from search.engine.source_code import PartCode, SeriesCodeMembers
from search.engine.source_scope import (
    NamedSource,
    SourceCollection,
    SourceRefusal,
    SourceScope,
    normalise_source_text,
    source_label,
)
from search.heading.departure_fact import DirectionWord, PartyReason
from search.models import (
    GraphRef,
    IdentityBindingSet,
    IdentityBindingValue,
    ResolvedQualifier,
    SourceMention,
    Spec,
    Warning,
)

if TYPE_CHECKING:
    from search.engine.region.result import RegionResult
    from search.heading.departure_fact import DepartureFact


def source_record(ref: str) -> ResolvedQualifier:
    """The unbound record of the source `ref`, under its label (`source_label`)."""
    label = source_label(ref)
    return ResolvedQualifier(
        text=label, named=[GraphRef(ref=ref, label=label)], slot=None, disposition="unbound"
    )


def _named_record(named: NamedSource) -> ResolvedQualifier:
    """The unbound record of a reference, or of a collection under its label with its
    members named."""
    members = named.members if isinstance(named, SourceCollection) else (named,)
    return ResolvedQualifier(
        text=named.label,
        named=[GraphRef(ref=member.ref, label=member.label) for member in members],
        slot=None,
        disposition="unbound",
    )


def scope_records(scope: SourceScope) -> tuple[ResolvedQualifier, ...]:
    """The unbound records naming the sources `scope` was compiled from: those its part
    includes when it names any (`SourceScope.named`), else those it excludes, less a
    mention that resolved to no source or is marked idle (`SourceScope.idle`), which
    limited no answer."""
    polarity = "include" if scope.named else "exclude"
    return tuple(
        _named_record(named)
        for position, (mention, named) in enumerate(
            zip(scope.mentions, scope.references, strict=True)
        )
        if mention.polarity == polarity and named is not None and position not in scope.idle
    )


def with_idle_exclusions(
    scope: SourceScope | None, shapes: Sequence[ShapeDraft]
) -> SourceScope | None:
    """`scope` with each excluding mention that removes nothing from a part that recalled
    `shapes` marked idle (`SourceScope.idle`). One mark decides both the warning that the
    exclusion was not applied (`idle_exclusion_warnings`) and that no refusal names it
    (`scope_records`).

    A mention whose words resolved to no source stands for no provenance, so it always
    removes nothing. Any other mention removes nothing when the provenances it stands for
    (`SourceScope.expansions`) may publish none of `shapes` (`_may_publish`): candidacy
    then keeps every shape it would keep without the mention, and no read of them meets a
    facet of those provenances."""
    if scope is None:
        return None
    index = get_index()
    catalogue = current_catalogue()
    idle = frozenset(
        position
        for position, (mention, expansion) in enumerate(
            zip(scope.mentions, scope.expansions, strict=True)
        )
        if mention.polarity == "exclude"
        and not any(_may_publish(shape, expansion, index, catalogue) for shape in shapes)
    )
    return dataclasses.replace(scope, idle=idle)


def idle_exclusion_warnings(scope: SourceScope | None) -> list[Warning]:
    """One `query_qualifier_dropped` warning per excluding mention of `scope` marked idle
    (`with_idle_exclusions`), quoting the mention's words."""
    return [
        _dropped_warning(
            f'The requested exclusion "{words}" was not applied: no data this question '
            "reads comes from it."
        )
        for words in _idle_words(scope, "exclude")
    ]


def with_idle_inclusions(scope: SourceScope, served: frozenset[str]) -> SourceScope:
    """`scope` with each including mention none of whose provenances
    (`SourceScope.expansions`) is among `served`, the provenances that serve the part's
    answer (`answer_provenances`), marked idle (`SourceScope.idle`). Such a mention served nothing
    of the answer, so the part warns that it was not served (`idle_inclusion_warnings`)."""
    idle = frozenset(
        position
        for position, (mention, expansion) in enumerate(
            zip(scope.mentions, scope.expansions, strict=True)
        )
        if mention.polarity == "include" and expansion.isdisjoint(served)
    )
    return dataclasses.replace(scope, idle=scope.idle | idle)


def idle_inclusion_warnings(scope: SourceScope | None) -> list[Warning]:
    """One `query_qualifier_dropped` warning per including mention of `scope` marked idle
    (`with_idle_inclusions`), quoting the mention's words."""
    return [
        _dropped_warning(
            f'The requested source "{words}" was not served: no data in this answer comes from it.'
        )
        for words in _idle_words(scope, "include")
    ]


def answer_provenances(specs: Sequence[Spec]) -> frozenset[str] | None:
    """The provenances that serve `specs`: each provenance their identities bind as their
    source (`facets.provenance`, `spec_build.with_served_provenance`), and for a declared
    composite its component datasets too (`code_index.composites.provenance_datasets`), so
    the OECD headline serves DAC1. `None` when one binds none, so what served it is
    unknown."""
    bound: set[str] = set()
    for spec in specs:
        binding = (spec.definition.identity.facets or {}).get("provenance")
        if isinstance(binding, IdentityBindingValue):
            bound.add(binding.ref)
        elif isinstance(binding, IdentityBindingSet):
            bound.update(binding.refs)
        else:
            return None
    return frozenset(bound).union(*(provenance_datasets(ref) for ref in bound))


def _idle_words(scope: SourceScope | None, polarity: Literal["include", "exclude"]) -> list[str]:
    """The words of each mention of `scope` with `polarity` marked idle, in posted order."""
    if scope is None:
        return []
    return [
        _mention_words(mention, named)
        for position, (mention, named) in enumerate(
            zip(scope.mentions, scope.references, strict=True)
        )
        if position in scope.idle and mention.polarity == polarity
    ]


def _dropped_warning(message: str) -> Warning:
    return Warning(code=QUERY_QUALIFIER_DROPPED, severity="degraded", message=message)


def _may_publish(
    shape: ShapeDraft, provenances: frozenset[str], index: CodeIndex, catalogue: Catalogue
) -> bool:
    """Whether one of `provenances` may publish `shape`, read as candidacy reads a shape's
    publishers (`phase_rank_bind._in_scope`): the datasets publishing its family
    (`shape_datasets`), else the provenances holding the rows the index lists for its
    five-tuple (`shape_listed_rows`), else the provenances the catalogue knows to hold one
    of its members (`Catalogue.variable_provenances`). Where none of these is known, any
    provenance but a dataset the index declares may, since a declared dataset publishes
    only the index's rows."""
    if datasets := shape_datasets(shape, index):
        return not provenances.isdisjoint(datasets)
    if listed := shape_listed_rows(shape, index):
        return any(not provenances.isdisjoint(row.provenances) for row in listed)
    known = frozenset(
        row.id for sv in shape.sv_arc_facts or () for row in catalogue.variable_provenances(sv)
    )
    if not known:
        return not provenances <= index.datasets.keys()
    return not provenances.isdisjoint(known)


def _mention_words(mention: SourceMention, named: NamedSource | None) -> str:
    """The words a mention was posted with, else the label of what its ref names."""
    if mention.text is not None or named is None:
        return mention.text or ""
    return named.label


def allows_un_system(scope: SourceScope | None) -> bool:
    """Whether a part under `scope` may answer from the UN System Data Commons, which holds
    when it has no scope or its scope allows a provenance of that origin."""
    if scope is None:
        return True
    catalogue = current_catalogue()
    return any(
        row is not None and row.data_commons == UN_SYSTEM
        for row in map(catalogue.provenance, scope.allowed)
    )


@dataclass(frozen=True, slots=True)
class ContestedOrganisation:
    """An organisation a part's query names that it may read as its source or as a party to
    the flow (`source_scan.SourceHit.contested`): its words, the dcids the entity pack names
    it (`entity_resolution.pack.organisations_named`), the word naming a party to a flow
    that governs it, "to" where a mention of it follows "to", else "from" where one follows
    "from", else `None` (`source_scan.direction_word`), the position of its mention among
    the part's source mentions in the reading with it as the source (`mention`,
    `source_scan.SourceScan.hit_mentions`; `None` where the part's mentions were full), and
    the entity its party reading names it by (`entity`, `raw_text._entity_key`)."""

    words: str
    parties: frozenset[str]
    direction: DirectionWord | None
    mention: int | None
    entity: str


@dataclass(frozen=True, slots=True)
class RecalledFamily:
    """A family a detection recalls, as one of its rows stands for it: the provenances
    publishing it, the flow axes it declares and the row's measured recall score."""

    datasets: frozenset[str]
    axes: frozenset[RoleAxis]
    score: float | None = None


@dataclass(frozen=True, slots=True)
class OrganisationReading:
    """How a part reads one contested organisation (`choose_reading`): as `"party"` to the
    flow or as its `"source"`, whether the source reading was possible, why a party reading
    was chosen (`reason`, `None` on a source reading), and the words of the other sources the
    part names that list the organisation as a party (`listed_by`)."""

    role: Literal["party", "source"]
    source_possible: bool
    reason: PartyReason | None = None
    listed_by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OrganisationEvidence:
    """What one contested organisation brings to its reading (`choose_readings`): its words,
    the word governing it, its own tables (`organisation_tables`) and the `(dataset, axis)`
    pairs that publish it as a party (`PartyRow.published`)."""

    words: str
    direction: DirectionWord | None
    allowed: frozenset[str]
    published: frozenset[tuple[str, RoleAxis]]


@dataclass(frozen=True, slots=True)
class OrganisationReadings:
    """How a part reads each of its contested organisations (`read_organisations`), parallel
    to them, and the detection the pass of that assignment recalls from (`detected`): the
    party reading's probe detection where every organisation reads as a party, the source
    reading's where every one reads as the source, and `None` where the readings differ, since
    that pass's text and scope are neither probe's."""

    readings: tuple[OrganisationReading, ...]
    detected: Detected | None


def choose_reading(
    *,
    measure: Sequence[RecalledFamily],
    scoped: Sequence[RecalledFamily],
    allowed: frozenset[str],
    named: Mapping[str, frozenset[str]] | None = None,
    published: AbstractSet[tuple[str, RoleAxis]],
    direction: DirectionWord | None,
    refused: bool = False,
) -> OrganisationReading:
    """The reading of one contested organisation, from the families the party
    reading's pass recalls (`measure`) and those the source reading's pass recalls
    (`scoped`), each in recall order, the organisations' own tables (`allowed`,
    `organisation_tables`), the tables of each other source the part names by its words
    (`named`, `other_named_tables`), whether the source reading's scope refused
    (`refused`), the `(dataset, axis)` pairs that publish the organisation as a party
    (`PartyRow.published`), and the word governing it.

    One rule judges both readings: a family governs a recall as the pass reads a flow
    family (`flow_governance.governs`: the first family, or one whose score is at least
    ``SEARCH_UNKNOWN_SCORE_THRESHOLD``). A party reading is possible where a flow family of
    `measure` governs and a family of `measure` is published by a dataset that publishes the
    organisation on an axis the family declares. A debt family recalled at 0.61 under GDP
    families leaves "GDP of Kenya from the IMF" no party reading; DAC1 families leading
    "contributions" let the DAC2a family recalled under them, which publishes the World
    Bank, make one possible. A source reading is possible where a family of `scoped` with a
    publisher among `allowed` governs, so DAC1 families leading the scoped recall of "World
    Bank core contributions" leave a World Bank family recalled at 0.53 no source reading,
    and none is possible where the party reading is and another source the part names
    publishes the organisation as a party: that source says whose data the part reads, so
    "CRS World Bank disbursements to Kenya", the CRS publishing IDA and IBRD as donors,
    reads the World Bank as a party. Where no party reading is possible the listing rules
    out nothing, as `choose_readings` lists one organisation by another: "CRS World Bank
    GDP of Kenya" reads the World Bank as the source, since GDP has no flow it can be a
    party to.

    The part reads the organisation as a party when that is possible and either a
    direction word governs it or the source reading is impossible, else as its source
    when that is possible, else as a party, whose pass refuses on the organisation's own
    words. Two possible readings with no direction word read as the source: a publisher
    named before the indicator, as in "World Bank external debt owed to China".

    A party reading carries why it was chosen (`PartyReason`): the direction word, where the
    source reading was possible; else another named source listing the organisation as a
    party; else the source reading's scope refusing; else none of the organisation's own
    data leading the source reading's recall."""
    party = party_possible(measure, published)
    listed_by = tuple(
        words
        for words, tables in (named or {}).items()
        if party and any(dataset in tables for dataset, _ in published)
    )
    source = not listed_by and any(
        not family.datasets.isdisjoint(allowed) and governs(top=position == 0, score=family.score)
        for position, family in enumerate(scoped)
    )
    if party and (direction is not None or not source):
        role: Literal["party", "source"] = "party"
    else:
        role = "source" if source else "party"
    if role == "source":
        return OrganisationReading(role=role, source_possible=True)
    reason: PartyReason
    if source:
        reason = "direction"
    elif listed_by:
        reason = "named_source"
    elif refused:
        reason = "source_unresolved"
    else:
        reason = "no_own_data"
    return OrganisationReading(
        role=role, source_possible=source, reason=reason, listed_by=listed_by
    )


def party_possible(
    measure: Sequence[RecalledFamily], published: AbstractSet[tuple[str, RoleAxis]]
) -> bool:
    """Whether a party reading of an organisation that the `(dataset, axis)` pairs
    `published` publish as a party is possible against `measure`, the families the party
    reading's pass recalls (`choose_reading`): a flow family of `measure` governs, and a
    family of `measure` is published by a dataset that publishes the organisation on an axis
    the family declares.

    The judgement is a forecast of the pass's. It reads the raw recall in recall order,
    while the pass applies the same rule (`flow_governance.governs`) to the shapes it
    admitted, in rank order (`phase_rank_bind._role_bearing_outranks`). Where the pass
    admits no flow shape that governs, as when the recall's leading flow family is not
    admitted and a subject-only shape ranks first, the party reading refuses
    `entity_role_not_admitted` on the organisation's words, and the answer says it was read
    as a party (`read_as_party`)."""
    return any(
        family.axes and governs(top=position == 0, score=family.score)
        for position, family in enumerate(measure)
    ) and any(
        (dataset, axis) in published
        for family in measure
        for dataset in family.datasets
        for axis in family.axes
    )


def choose_readings(
    organisations: Sequence[OrganisationEvidence],
    *,
    measure: Sequence[RecalledFamily],
    scoped: Sequence[RecalledFamily],
    named: Mapping[str, frozenset[str]] | None = None,
    refused: bool = False,
) -> tuple[OrganisationReading, ...]:
    """The reading of each of a part's contested `organisations`, parallel to them, each
    judged on its own evidence by `choose_reading` against the families both readings'
    passes recall (`measure`, `scoped`), the other sources the part names (`named`) and
    whether the source reading's scope refused (`refused`).

    An organisation read as the source names a source as any other does, so where its
    tables list another organisation whose party reading is possible (`party_possible`),
    the other reads as a party (`OrganisationReading.listed_by`). "World Bank debt service
    from Kenya to the IMF" reads the World Bank, which no word governs, as the source, and
    the IMF, after "to" and an IDS creditor, as a party. "World Bank and IMF GDP of Kenya"
    reads both as sources, since GDP has no flow in which the IMF can be a party.

    Only an organisation read as the source that no other such organisation can list in
    this way names a source for the others, so the listing goes one step: where A lists B and
    B lists C, B reads as a party and C stays a source, and two organisations that list each
    other both stay sources, since the data give no ground to choose which one names the
    data."""
    first = [
        choose_reading(
            measure=measure,
            scoped=scoped,
            allowed=organisation.allowed,
            named=named,
            published=organisation.published,
            direction=organisation.direction,
            refused=refused,
        )
        for organisation in organisations
    ]
    sources = [
        organisation
        for organisation, reading in zip(organisations, first, strict=True)
        if reading.role == "source"
    ]

    def listed_by(
        organisation: OrganisationEvidence, among: Sequence[OrganisationEvidence]
    ) -> list[OrganisationEvidence]:
        """The organisations of `among`, other than `organisation`, whose tables list it,
        where its party reading is possible; none where it is not."""
        if not party_possible(measure, organisation.published):
            return []
        return [
            source
            for source in among
            if source is not organisation and _lists(source, organisation)
        ]

    naming = [source for source in sources if not listed_by(source, sources)]
    readings: list[OrganisationReading] = []
    for organisation, reading in zip(organisations, first, strict=True):
        listing = {source.words: source.allowed for source in listed_by(organisation, naming)}
        if listing:
            reading = choose_reading(
                measure=measure,
                scoped=scoped,
                allowed=organisation.allowed,
                named={**(named or {}), **listing},
                published=organisation.published,
                direction=organisation.direction,
                refused=refused,
            )
        readings.append(reading)
    return tuple(readings)


def _lists(source: OrganisationEvidence, party: OrganisationEvidence) -> bool:
    """Whether one of `source`'s tables publishes `party` as a party."""
    return any(dataset in source.allowed for dataset, _ in party.published)


async def read_organisations(
    contested: Sequence[ContestedOrganisation],
    *,
    party_text: str,
    party_variable: str,
    party_scope: SourceScope | SourceRefusal | None,
    party_code: PartCode | None = None,
    source_text: str,
    source_variable: str,
    source_scope: SourceScope | SourceRefusal | None,
    source_code: PartCode | None = None,
    graph: EngineGraphClient,
) -> OrganisationReadings:
    """The reading of each of a part's `contested` organisations (`choose_readings`), from the
    recall each uniform reading's pass makes: the reading of every organisation as a party
    from `party_text`, `party_variable`, `party_scope` and `party_code`, and of every one as
    the source from `source_text`, `source_variable`, `source_scope` and `source_code`.

    A reading whose series code compiled to StatVars recalls those StatVars and searches
    nothing (`interpret.recall`), so it is judged on their families, each at score 1.0, and
    a reading whose code refused recalls nothing and is judged on none. Any other reading's
    text is detected under its scope, read through the catalogue scoped as that pass's
    recall reads it (`interpret.detect_measure`). The searches run together, and a failed
    search raises. Where every organisation takes one reading, the readings carry that
    reading's detection (`OrganisationReadings.detected`), which its pass takes as its
    recall's (`interpret.recall`), so the pass searches its text once.

    Each organisation's source reading counts only its own tables (`organisation_tables`),
    so another source the part names cannot make it possible: in "CRS World Bank
    disbursements to Kenya" the CRS families leading the scoped recall leave the World Bank
    no source reading. Every organisation's source reading is judged on the one recall
    scoped to all the part's sources, so another organisation's family can lead it. A part
    whose source reading names none of the organisations' tables, its scope refused, runs no
    scoped search and has no source reading."""
    index = get_index()
    tables = [organisation_tables(organisation, source_scope) for organisation in contested]
    others = other_named_tables(contested, source_scope)
    all_tables = frozenset().union(*tables)

    async def recall_of(
        text: str,
        variable: str,
        scope: SourceScope | SourceRefusal | None,
        code: PartCode | None,
    ) -> tuple[list[DetectedSV], Detected | None]:
        """The candidates the reading's pass recalls, and the detection made for them."""
        if isinstance(code, SeriesCodeMembers):
            return [
                DetectedSV(dcid=dcid, score=1.0, type_of="StatisticalVariable", sentence=None)
                for dcid in code.dcids
            ], None
        if code is not None:
            return [], None
        with scoped_reads(scope.allowed if isinstance(scope, SourceScope) else None):
            detected = await detect_measure(graph, text, variable=variable)
        return detected.candidates, detected

    async def scoped_recall() -> tuple[list[DetectedSV], Detected | None]:
        if not all_tables:
            return [], None
        return await recall_of(source_text, source_variable, source_scope, source_code)

    (measure, party_detected), (scoped, source_detected) = await asyncio.gather(
        recall_of(party_text, party_variable, party_scope, party_code), scoped_recall()
    )
    catalogue = current_catalogue()
    readings = choose_readings(
        [
            OrganisationEvidence(
                words=organisation.words,
                direction=organisation.direction,
                allowed=allowed,
                published=frozenset(
                    pair
                    for dcid in organisation.parties
                    if (party := index.parties.get(dcid)) is not None
                    for pair in party.published
                ),
            )
            for organisation, allowed in zip(contested, tables, strict=True)
        ],
        measure=_recalled(measure, index, catalogue, unrecorded=frozenset()),
        # As candidacy reads a shape no one records a publisher for (`_may_publish`), any of
        # the organisations' tables but a declared dataset may publish such a series.
        scoped=_recalled(scoped, index, catalogue, unrecorded=all_tables - index.datasets.keys()),
        named=others,
        refused=isinstance(source_scope, SourceRefusal),
    )
    roles = {reading.role for reading in readings}
    if roles == {"party"}:
        return OrganisationReadings(readings=readings, detected=party_detected)
    if roles == {"source"}:
        return OrganisationReadings(readings=readings, detected=source_detected)
    return OrganisationReadings(readings=readings, detected=None)


def organisation_tables(
    contested: ContestedOrganisation, scope: SourceScope | SourceRefusal | None
) -> frozenset[str]:
    """The provenances `scope` allows for `contested`'s own mention
    (`ContestedOrganisation.mention`): its expansion within `scope.allowed`. None for a
    scope that refused or holds no mention of it."""
    return frozenset().union(
        *(
            tables
            for position, _, tables in _included_tables(scope)
            if position == contested.mention
        )
    )


def other_named_tables(
    contested: Sequence[ContestedOrganisation], scope: SourceScope | SourceRefusal | None
) -> dict[str, frozenset[str]]:
    """The provenances `scope` allows for each source the part names that is none of the
    `contested` organisations, by its words (`_mention_words`): the expansions of its
    included mentions that are no organisation's mention (`ContestedOrganisation.mention`),
    within `scope.allowed`."""
    own = {organisation.mention for organisation in contested}
    named: dict[str, frozenset[str]] = {}
    for position, words, tables in _included_tables(scope):
        if position not in own:
            named[words] = named.get(words, frozenset()) | tables
    return named


def _included_tables(
    scope: SourceScope | SourceRefusal | None,
) -> list[tuple[int, str, frozenset[str]]]:
    """Each included mention of `scope`, its position among the scope's mentions, its words
    (`_mention_words`) and its expansion within `scope.allowed`; none for a scope that
    refused."""
    if not isinstance(scope, SourceScope):
        return []
    references = scope.references or (None,) * len(scope.mentions)
    return [
        (position, _mention_words(mention, reference), expansion & scope.allowed)
        for position, (mention, reference, expansion) in enumerate(
            zip(scope.mentions, references, scope.expansions, strict=True)
        )
        if mention.polarity == "include"
    ]


def _recalled(
    hits: Sequence[DetectedSV],
    index: CodeIndex,
    catalogue: Catalogue,
    *,
    unrecorded: frozenset[str],
) -> list[RecalledFamily]:
    """The family each series of `hits` stands for, in their order, with its hit's score; a
    topic stands for none. An index row's family is published by its family's datasets
    (`CodeIndex.family_by_dcid`), else by the row's provenances, and declares the flow axes
    of the row's roles. Any other series is published by the provenances the catalogue
    knows to hold it, else by `unrecorded`, and declares no axis the index records."""
    recalled: list[RecalledFamily] = []
    for hit in hits:
        if is_topic_candidate(hit.dcid, hit.type_of):
            continue
        row = index.statvar_by_dcid(hit.dcid)
        if row is None:
            provenances = frozenset(p.id for p in catalogue.variable_provenances(hit.dcid))
            recalled.append(
                RecalledFamily(
                    datasets=provenances or unrecorded, axes=frozenset(), score=hit.score
                )
            )
            continue
        family = index.family_by_dcid.get(hit.dcid)
        recalled.append(
            RecalledFamily(
                datasets=frozenset(family.datasets if family is not None else row.provenances),
                axes=frozenset(roles_by_axis(row.observation_properties)),
                score=hit.score,
            )
        )
    return recalled


def read_as_source(region: RegionResult, contested: ContestedOrganisation) -> RegionResult:
    """`region`, the answer of a part reading `contested` as its source, disclosing that
    reading where a direction word governs the organisation and the answer's measure has a
    flow axis (`_answers_a_flow`): an `organisation_read_as_source` warning and its
    departure. The part reads an organisation a direction word governs as its source only
    where no party reading of it is possible (`choose_reading`): no flow family governs the
    recall, or none of its families lists the organisation as a party. The text says that
    the data that best matches the question does not list it as a party, which holds in
    both cases. "GDP of Kenya from the IMF" answers from a GDP series with no flow axis
    and is `region` as it is, as is any part with no direction word."""
    if contested.direction is None or not _answers_a_flow(region):
        return region
    names = contested.words
    warning = Warning(
        code=ORGANISATION_READ_AS_SOURCE,
        severity="warn",
        message=(
            f'{names}, written after "{contested.direction}", was read as the source of the '
            "data, since the data that best matches this question does not list it as a "
            "party to a flow."
        ),
    )
    return _disclosed(
        region, warning, organisation_read_as_source(names, direction=contested.direction)
    )


def _answers_a_flow(region: RegionResult) -> bool:
    """Whether a series `region` answers from, or elected and found no data for
    (`RegionResult.elected_stat_vars`), is an index row declaring a role on a flow axis."""
    index = get_index()
    dcids = [
        *(member.ref.ref for spec in region.specs for member in spec.resolved.members),
        *(ref.ref for ref in region.elected_stat_vars or ()),
    ]
    return any(
        (row := index.statvar_by_dcid(dcid)) is not None
        and bool(roles_by_axis(row.observation_properties))
        for dcid in dcids
    )


def read_as_party(
    region: RegionResult, contested: ContestedOrganisation, reading: OrganisationReading
) -> RegionResult:
    """`region`, the answer of a part reading `contested` as a party to the flow
    (`reading`), disclosing that where the answer reads it as a party: an
    `organisation_read_as_party` warning and its departure, which name the direction word
    where the query wrote one and give the reading's reason (`OrganisationReading.reason`).
    It discloses where a Spec binds one of the organisation's dcids
    (`ContestedOrganisation.parties`) or the part refuses on its own words, as ambiguous,
    unresolved or with no admitted role. Any other answer turns on something else, says
    nothing of a party reading, and is `region` as it is."""
    if not _reads_as_party(region, contested):
        return region
    names = contested.words
    reason = reading.reason or "no_own_data"
    listed_by = _join_words(reading.listed_by) or None
    written = (
        f'{names}, written after "{contested.direction}",'
        if contested.direction is not None
        else names
    )
    because = {
        "direction": "not as the source of the data",
        "named_source": f"since {listed_by} lists it as a party",
        "source_unresolved": "since its words name no single source of data",
        "no_own_data": "since none of the data it publishes measures this",
    }[reason]
    warning = Warning(
        code=ORGANISATION_READ_AS_PARTY,
        severity="info",
        message=f"{written} was read as a party to the flow, {because}.",
    )
    return _disclosed(
        region,
        warning,
        organisation_read_as_party(
            names, direction=contested.direction, reason=reason, listed_by=listed_by
        ),
    )


def _reads_as_party(region: RegionResult, contested: ContestedOrganisation) -> bool:
    """Whether `region` binds one of `contested`'s dcids or refuses on its entity
    (`ContestedOrganisation.entity`), the two names compared under `normalise_source_text`,
    the key the entity is found by, so casing, punctuation or a leading "the" does not part
    them."""
    bound = any(
        dcid in contested.parties
        for spec in region.specs
        for dcids in spec.definition.identity.entities.values()
        for dcid in dcids
    )
    refused_on = {
        normalise_source_text(name)
        for name in (
            *(entity.text for entity in region.ambiguous_entities),
            *region.unresolved_entity_names,
            *(entity.text for entity in region.role_not_admitted_entities),
        )
    }
    return bound or normalise_source_text(contested.entity) in refused_on


def _join_words(words: Sequence[str]) -> str:
    """`words` joined with "and"."""
    return " and ".join(words)


def _disclosed(region: RegionResult, warning: Warning, fact: DepartureFact) -> RegionResult:
    """`region` holding `warning` and, under its key, the departure `fact` it raises."""
    warned = dataclasses.replace(region, warnings=(*region.warnings, warning))
    return with_departure_facts(warned, {warning_key(warning): fact})
