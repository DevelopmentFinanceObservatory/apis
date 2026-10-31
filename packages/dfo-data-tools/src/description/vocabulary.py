"""The words a description is worded from, and the revision that names them.

`build_vocabulary` joins the code index, the measure index, and the labels and source-document
facts a caller read into one `Vocabulary`. It reads no graph, store or file beyond the
packaged `data/publishers.json`. A word the vocabulary lacks reads as `None` and is never an
empty string, so the composer can mark it as a gap or fall back to the label stored as
evidence. A word longer than a served label's 512 characters is clipped to them, ending in "…",
and a definition longer than a served definition's 4096 characters is clipped to those. Code and
measure words are looked up in the indexes, so building a vocabulary costs the size of the
caller's inputs.

A description carries the vocabulary's `revision` in `built_from`. The revision hashes the
inputs that decide the words (`vocabulary_revision`); no snapshot is kept behind it.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import cache
from types import MappingProxyType
from typing import Final

from code_index.labels import HOW_CONNECTIVES, ROLE_CLASS_PROPERTIES, compose_measure_label
from code_index.query import CodeIndex
from code_index.types import MeasureIndex, label_for_pairs
from description.stored import DEFINITION_MAX_LENGTH, clipped
from dfo_platform.roles import GENERIC_ROLE_WORDING, ROLE_WORDING, FlowFamily, RoleWording
from unit_words.currency_nouns import CURRENCY_NOUNS
from unit_words.unit_class import classify_unit

COMPOSER_VERSION: Final = "3"
"""The version of how the composer phrases sentences from the same words. Raising it moves
every `voc_` revision. An edit to the packaged word tables moves the revision on its own
(`vocabulary_revision`)."""

_PUBLISHERS_FILE: Final = "publishers.json"


@dataclass(frozen=True, slots=True)
class SourceDocumentFacts:
    """What a source's document says about the source today, as the caller read it.

    `publisher` is the dcid of the `Source` that publishes it
    (`dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd`); a UN agency's provenance is
    its own source (`undata/p/WHO`). `last_downloaded` is the date the DFO pipeline last
    downloaded the source (`2026-10-08`) and `next_release` the publisher's next release
    (`2026-09`), each as the document writes it. Each is `None` when the document gives none.
    """

    publisher: str | None = None
    last_downloaded: str | None = None
    next_release: str | None = None


@dataclass(frozen=True, slots=True)
class MeasureWords:
    """A definition's measure label, the constraint properties it states, which no other
    phrase repeats (the measure axes, and the codes an authored label's text names:
    "imputed multilateral ODA" names `dac2aAidType`), and the measure's authored definition,
    or `None` where the code index authors none."""

    label: str
    stated: frozenset[str]
    definition: str | None


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """One immutable snapshot of the words a description is worded from.

    `revision` is the `voc_` revision of the inputs (`vocabulary_revision`). `index` and
    `measure_index` give code and measure-value labels, axis nouns and the measure label's
    inputs, read through `label`, `axis_name` and `measure`. The remaining mappings are
    read-only copies of the caller's inputs and hold non-empty strings only. `labels` maps any
    other dcid (a place, an organisation, a variable, a source) to its current label, and
    `definitions` maps any other dcid to its current definition, read through `definition`.
    `role_wording` holds the provider and recipient nouns of each flow family.
    `publishers` maps a `Source` dcid to the short name people cite it by ("OECD"). `sources`
    maps a source's provenance or dataset dcid to what its document says today.
    """

    revision: str
    index: CodeIndex
    measure_index: MeasureIndex
    labels: Mapping[str, str]
    definitions: Mapping[str, str]
    role_wording: Mapping[FlowFamily, RoleWording]
    publishers: Mapping[str, str]
    sources: Mapping[str, SourceDocumentFacts]

    def label(self, dcid: str) -> str | None:
        """The current label of `dcid`: a code's index label, then a measure value's, then the
        caller's, else `None`."""
        if (row := self.index.code_by_dcid(dcid)) is not None:
            return _word(row.label)
        for value in self.measure_index.values:
            if value.dcid == dcid:
                return _word(value.label)
        return self.labels.get(dcid)

    def definition(self, dcid: str) -> str | None:
        """The current definition of `dcid`: a code's index definition, then a measure
        value's, then the caller's, else `None`. A code or measure value whose row carries
        no definition falls through to the caller's."""
        if (row := self.index.code_by_dcid(dcid)) is not None and (
            text := _definition(row.definition)
        ) is not None:
            return text
        for value in self.measure_index.values:
            if value.dcid == dcid and (text := _definition(value.definition)) is not None:
                return text
        return self.definitions.get(dcid)

    def axis_name(self, prop: str) -> str | None:
        """The noun a sentence names a constraint property or measure axis by ("sector" for
        `crsPurpose`, "flow type" for `financialFlowType`), else `None`."""
        if (declaration := self.index.declarations.get(prop)) is not None:
            return _word(declaration.plain_name)
        for axis in self.measure_index.axes:
            if axis.axis == prop:
                return _word(axis.label)
        return None

    def measure(
        self, *, shape: Mapping[str, str], constraints: Mapping[str, str]
    ) -> MeasureWords | None:
        """The measure a definition names, or `None` when no authored label covers it.

        Args:
            shape: the shape's members by graph property (`populationType`,
                `measuredProperty`, `statType`, `measurementQualifier`).
            constraints: the values the definition fixes, by property.

        The measure is the partition label over the measure axes, each read from `shape` or
        `constraints` by its kind as Search keys a partition (`measure_label_for_partition`)
        and worded by `code_index.labels.compose_measure_label` ("gross disbursements"), else
        the measure noun (`MeasureIndex.nouns`) over the shape and constraints ("external
        debt service"), as the Graph service names a variable's measure. Its definition is
        the matched label's or noun's.
        """
        pairs = [
            (axis.axis, (shape if axis.kind == "shape" else constraints).get(axis.axis))
            for axis in self.measure_index.axes
        ]
        if (match := label_for_pairs(pairs, self.measure_index.partition_labels)) is not None:
            return MeasureWords(
                label=compose_measure_label(match, self.measure_index.values),
                definition=_definition(match.definition),
                stated=frozenset(
                    axis.axis
                    for axis in self.measure_index.axes
                    if axis.kind == "constraint" and axis.axis in constraints
                )
                | {prop for prop, value in match.named_values if constraints.get(prop) == value},
            )
        noun = label_for_pairs([*shape.items(), *constraints.items()], self.measure_index.nouns)
        if noun is None:
            return None
        return MeasureWords(
            label=noun.label,
            definition=_definition(noun.definition),
            stated=frozenset(
                prop for prop, value in constraints.items() if (prop, value) not in noun.uncovered
            ),
        )

    def unit_noun(self, unit: str) -> str | None:
        """The plural currency noun of a unit dcid ("US dollars" for `CurrencyUSD`), or `None`
        for a unit that is no currency or a currency with no authored noun."""
        currency = classify_unit(unit=unit, denominator=None, currency_units={}).currency
        return CURRENCY_NOUNS.get(currency) if currency is not None else None

    def roles(self, family: FlowFamily | None) -> RoleWording:
        """The nouns of a flow family's two parties, or the generic provider and recipient
        when the family is unknown."""
        return self.role_wording[family] if family is not None else GENERIC_ROLE_WORDING

    def publisher(self, source: str) -> str | None:
        """The short name of the publisher of a source's provenance or dataset dcid, or `None`
        when its document names no publisher or the publisher has no short name. The
        composer then names the publisher by its label."""
        facts = self.sources.get(source)
        if facts is None or facts.publisher is None:
            return None
        return self.publishers.get(facts.publisher)


def vocabulary_revision(
    *, index_stamp: str, graph_generation: str | None, catalogue_build: str | None
) -> str:
    """`voc_` and the hex SHA-256, cut to 32 characters, of the inputs that decide the words.

    Args:
        index_stamp: the code index's `IndexStamp.content_hash`.
        graph_generation: the graph generation token the labels were read at, or `None` when
            unknown.
        catalogue_build: the origin catalogue build id joined to the bound release
            (`origin_catalogue.context.current_words_build`), or `None` when none is served.

    The digest of the packaged word tables (`packaged_words_digest`) and `COMPOSER_VERSION`
    join them, so the same inputs, tables and composer give the same revision, and an edit to
    any of them moves it.
    """
    canonical = json.dumps(
        [index_stamp, graph_generation, catalogue_build, packaged_words_digest(), COMPOSER_VERSION],
        separators=(",", ":"),
    )
    return "voc_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


@cache
def packaged_words_digest() -> str:
    """The hex SHA-256 of the canonical JSON of the packaged word tables: the publisher short
    names (`publisher_names`), the currency nouns (`CURRENCY_NOUNS`), the role wording of
    each flow family (`ROLE_WORDING`), the connective before each constraint property's
    value (`HOW_CONNECTIVES`) and the properties whose values read as a role's entities
    (`ROLE_CLASS_PROPERTIES`). A code's short name is the code index's, whose stamp joins
    the revision. Computed once per process."""
    tables = {
        "publishers": dict(publisher_names()),
        "currency_nouns": dict(CURRENCY_NOUNS),
        "role_wording": {family: asdict(wording) for family, wording in ROLE_WORDING.items()},
        "how_connectives": dict(HOW_CONNECTIVES),
        "role_class_properties": dict(ROLE_CLASS_PROPERTIES),
    }
    canonical = json.dumps(tables, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@cache
def publisher_names() -> Mapping[str, str]:
    """The packaged short names of publishers, by `Source` dcid.

    Only publishers the platform serves and that one short name identifies are listed. A
    `Source` left out is named by its label.
    """
    raw = (importlib.resources.files(__package__) / "data" / _PUBLISHERS_FILE).read_text("utf-8")
    return MappingProxyType(_words(json.loads(raw)))


def build_vocabulary(
    *,
    index: CodeIndex,
    measure_index: MeasureIndex,
    labels: Mapping[str, str],
    sources: Mapping[str, SourceDocumentFacts],
    graph_generation: str | None,
    catalogue_build: str | None,
    definitions: Mapping[str, str] = MappingProxyType({}),
) -> Vocabulary:
    """Build the vocabulary from in-memory inputs.

    Args:
        index: the code index. Its codes give their labels and its declarations their axis
            nouns, read on lookup, and its stamp joins the revision.
        measure_index: the measure-axis export. Its axes give their nouns and its values
            their labels, read on lookup.
        labels: the labels the caller read for places, organisations, variables and sources,
            by dcid. A code's or measure value's index label takes precedence (`Vocabulary.label`).
        sources: each source's document facts, by provenance or dataset dcid.
        graph_generation: the graph generation token the labels were read at.
        catalogue_build: the origin catalogue build id joined to the bound release
            (`origin_catalogue.context.current_words_build`).
        definitions: the definitions the caller read, by dcid. A code's or measure value's
            index definition takes precedence (`Vocabulary.definition`).

    Empty and blank strings in the inputs count as absent.
    """
    return Vocabulary(
        revision=vocabulary_revision(
            index_stamp=index.stamp.content_hash,
            graph_generation=graph_generation,
            catalogue_build=catalogue_build,
        ),
        index=index,
        measure_index=measure_index,
        labels=MappingProxyType(_words(labels)),
        definitions=MappingProxyType(
            {
                ref: text
                for ref, raw in definitions.items()
                if (text := _definition(raw)) is not None
            }
        ),
        role_wording=ROLE_WORDING,
        publishers=publisher_names(),
        sources=MappingProxyType(
            {
                ref: SourceDocumentFacts(
                    publisher=_word(facts.publisher),
                    last_downloaded=_word(facts.last_downloaded),
                    next_release=_word(facts.next_release),
                )
                for ref, facts in sources.items()
            }
        ),
    )


def _word(text: str | None) -> str | None:
    """`text` clipped to the length a served label holds, `None` when empty or blank."""
    return clipped(text) if text is not None and text.strip() else None


def _definition(text: str | None) -> str | None:
    """`text` clipped to the length a served definition holds, `None` when empty or blank."""
    return clipped(text, limit=DEFINITION_MAX_LENGTH) if text is not None and text.strip() else None


def _words(words: Mapping[str, str]) -> dict[str, str]:
    """`words` without its empty and blank values, each clipped as `_word` clips it."""
    return {key: clipped(text) for key, text in words.items() if text.strip()}
