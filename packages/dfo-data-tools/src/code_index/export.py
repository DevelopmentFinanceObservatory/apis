"""Build the code index's graph blocks from a graph reader.

`export_graph_blocks` reads the graph through `ExportReader` and joins it with the authored
declarations (`code_index.declarations`). `index_bytes` is the one serialisation of an index
document, so the same document gives the same bytes. Nothing here imports `scripts/` or
`dc_graph`: a reader supplies the graph reads, and the two names the export needs from
`dc_graph` are imported where they are used.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from code_index.declarations import (
    CODE_SHORT_NAMES,
    CODE_SYNONYMS,
    CODE_SYSTEMS,
    CONCEPT_GROUPS,
    DATASETS,
    DECLARED_MEASURE_VALUES,
    MEASURE_AXES,
    MEASURE_CLASSES,
    MEASURE_NOUNS,
    MEASURE_PARTITION_LABELS,
    MEASURE_VALUE_TEXT,
    MEASURE_WORD_SYNONYMS_KEPT,
    PARTLY_STATED_VALUES,
    PARTY_TYPE_KINDS,
    SHAPE_SLOT_DEFAULTS,
    STAT_TYPE_WORDS,
    VARIABLE_ALIASES,
    WHOLE_TOTAL_SETS,
    CodeSystemConfig,
    KindOverride,
    MeasureValueText,
    TotalGroup,
    WholePopulationTotal,
    WholeWindow,
)
from code_index.members import key_sets_cover
from code_index.tokens import Token, TokenLookup, singular, tokenize
from code_index.types import (
    DatasetDeclaration,
    DeclaredDefault,
    MeasureAxis,
    MeasureClassDeclaration,
    MeasurePartitionLabel,
    PartlyStatedValue,
    PartyKind,
    PartyWord,
    Sector,
    ShapeSlotDefault,
    StatedWord,
    StatTypeWordDeclaration,
    WholeTotalSet,
    covered_sector,
    invalid_when_named_entry,
    label_for_pairs,
    party_kind,
)
from dfo_platform.roles import RoleAxis

if TYPE_CHECKING:
    from dc_graph import SubjectPage

# Truncated code-system enumeration aborts. Partition reads use truncation as a split signal.
_DEFAULT_MAX_RESULTS = 5000
# Subjects one unsplit read of a listed row's five-tuple may return
# (`_verify_listed_five_tuples`). The dev graph's `FinancialFlow` amount five-tuple holds about
# 14,000 StatVars, CRS, DAC and IDS together, and reads whole in about 45 s.
_UNSPLIT_MAX_RESULTS = 40_000
# Six levels cover every partition axis and bound recursive splitting.
_MAX_PARTITION_DEPTH = 6

_SOURCE_CODE = "sourceIndicatorCode"

_PRIVATE_CODE_BLOCK = range(1601, 1700)
"""The OECD donor codes reserved for private donors."""

_LABEL_SUFFIX_RE = re.compile(r"\s*\((?:CRS|DAC)\s[^()]*\)\s*$")

_CODE_ARC_PROPERTIES = (
    "classificationCode",
    "name",
    "alternateName",
    "shortName",
    "specializationOf",
    "description",
)
# `export_statvars` rejects multiple distinct values on these single-valued fields.
_STATVAR_SHAPE_PROPERTIES = (
    "populationType",
    "measuredProperty",
    "statType",
    "measurementQualifier",
    "measurementDenominator",
)
_STATVAR_PASS1_PROPERTIES = (
    "typeOf",
    "constraintProperties",
    *_STATVAR_SHAPE_PROPERTIES,
    "observationProperties",
)


class ExportRefused(Exception):
    """The graph or the declarations break a rule the code index relies on, so the export
    refuses to build it. The message names the rule and the offending rows."""


@dataclass(frozen=True)
class PartyPublication:
    """The datasets that publish one party, by provenance, and the flow axes they publish it
    on: `origin` as a donor, `destination` as a recipient. `years` gives the first and last
    year, inclusive, in which each dataset publishes it."""

    datasets: frozenset[str]
    axes: frozenset[RoleAxis]
    years: Mapping[str, tuple[int, int]] = field(default_factory=dict)


def _kind_from_types(
    dcid: str, types: Collection[str], type_kinds: Mapping[str, PartyKind | None]
) -> PartyKind:
    """The kind `types` give the party `dcid` under `type_kinds`, or `ExportRefused` naming why they
    give none."""
    if not types:
        raise ExportRefused(f"party {dcid!r} has no typeOf arc -- its kind cannot be declared")
    undeclared = sorted(set(types) - type_kinds.keys())
    if undeclared:
        raise ExportRefused(
            f"party {dcid!r} carries type(s) {undeclared!r} that PARTY_TYPE_KINDS does not "
            "declare -- add each with the kind it gives a party"
        )
    kind = party_kind(types, type_kinds)
    if kind is None:
        raise ExportRefused(
            f"party {dcid!r}'s types {sorted(types)!r} give it no single kind under "
            "PARTY_TYPE_KINDS"
        )
    return kind


def _dac_codes(arcs: Mapping[str, tuple[str, ...]]) -> tuple[int, ...]:
    """The numeric OECD donor codes in a party's `dacCode` arcs. A code that is not an integer is
    not a donor code and is left out."""
    return tuple(int(code) for code in arcs.get("dacCode", ()) if code.isdecimal())


def _check_private_providers(
    codes: Mapping[str, tuple[int, ...]],
    origins: Collection[str],
    private_providers: Mapping[int, str],
    kind_overrides: Mapping[str, KindOverride],
) -> None:
    """Raises `ExportRefused` when `private_providers` and the published parties disagree.

    `codes` is each published party's donor codes and `origins` the parties published as a donor.
    A listed code must be carried by its listed party alone once either is published; an origin
    party carrying a code in `_PRIVATE_CODE_BLOCK` must be listed; a listed party takes no kind
    override, since only a `party` carries a sector."""
    holders: dict[int, list[str]] = defaultdict(list)
    for dcid, party_codes in codes.items():
        for code in party_codes:
            holders[code].append(dcid)
    wrong = sorted(
        (code, dcid, sorted(holders[code]))
        for code, dcid in private_providers.items()
        if (code in holders or dcid in codes) and holders.get(code) != [dcid]
    )
    if wrong:
        raise ExportRefused(
            f"PRIVATE_PROVIDERS lists a code for a different party than the graph gives it: "
            f"{wrong!r} (code, listed party, parties carrying the code)"
        )
    missing = sorted(
        (code, dcid)
        for dcid in origins
        for code in codes.get(dcid, ())
        if code in _PRIVATE_CODE_BLOCK and code not in private_providers
    )
    if missing:
        raise ExportRefused(
            f"published donors carry OECD private donor codes (1601 to 1699) that PRIVATE_PROVIDERS "
            f"does not list: {missing!r} (code, party) -- review each and add it"
        )
    overridden = sorted(set(private_providers.values()) & kind_overrides.keys())
    if overridden:
        raise ExportRefused(
            f"PRIVATE_PROVIDERS and PARTY_KIND_OVERRIDES both name {overridden!r}; a private "
            "provider is a party"
        )


def _check_whole_population_totals(
    whole_totals: Mapping[str, WholePopulationTotal],
    kinds: Mapping[str, PartyKind],
    parties: Mapping[str, PartyPublication],
) -> None:
    """Raises `ExportRefused` for a `whole_totals` row naming a party that is not a published
    total, a total not published on exactly the row's axis, a dataset that does not publish the
    total, a window whose first year lies outside the years the dataset publishes the total, or
    a row with no dataset, no class or no reason. Two rows covering one class on one dataset and
    axis fail when the finished document loads (`code_index.loader._check_whole_totals`), as
    does a row whose windows cover different sectors (`code_index.loader._check_sector`)."""
    for dcid, whole in sorted(whole_totals.items()):
        if kinds.get(dcid) != "total":
            raise ExportRefused(
                f"WHOLE_POPULATION_TOTALS names {dcid!r}, which no declared or listed dataset "
                "publishes as a total"
            )
        if parties[dcid].axes != {whole.axis}:
            raise ExportRefused(
                f"WHOLE_POPULATION_TOTALS gives {dcid!r} the axis {whole.axis!r}, but the datasets "
                f"publish it on {sorted(parties[dcid].axes)!r}"
            )
        if unpublished := sorted(whole.datasets.keys() - parties[dcid].datasets):
            raise ExportRefused(
                f"WHOLE_POPULATION_TOTALS marks {dcid!r} in {unpublished!r}, which do not publish it"
            )
        if (
            not whole.datasets
            or not all(window.covers for window in whole.datasets.values())
            or not whole.reason
        ):
            raise ExportRefused(
                f"WHOLE_POPULATION_TOTALS row {dcid!r} names no dataset, covers no class or "
                "gives no reason"
            )
        outside = []
        for dataset, window in sorted(whole.datasets.items()):
            span = parties[dcid].years.get(dataset)
            if window.first_year is not None and (
                span is None or not span[0] <= window.first_year <= span[1]
            ):
                outside.append((dataset, window.first_year, span))
        if outside:
            raise ExportRefused(
                f"WHOLE_POPULATION_TOTALS starts {dcid!r} outside the years its datasets publish "
                f"it: {outside!r} (dataset, first year, published years)"
            )


def _check_classes_on_one_axis(
    rows: Sequence[Mapping], parties: Mapping[str, PartyPublication]
) -> None:
    """Raises `ExportRefused` for a party of any class but `official`, or a total, published on
    more than one axis.

    A row keeps `axes` and `datasets` as two lists, so a reader cannot tell which dataset holds
    which axis. `datasets_api.output_options` reads the totals a side holds in a dataset from
    `axis in party.axes and dataset in party.datasets`, and search reads the classes a side
    holds the same way. That is exact for a one-axis party and for an `official` party, because
    every side a two-axis party reaches already holds an `official` row. A private party, a
    residual or a total on both axes would count on both axes in every dataset. `rows` are the
    exported `parties` rows, each with its `kind` and `sector`."""
    ambiguous = sorted(
        (row["dcid"], row["kind"], row["sector"], sorted(parties[row["dcid"]].axes))
        for row in rows
        if (row["kind"], row["sector"]) != ("party", "official")
        and len(parties[row["dcid"]].axes) > 1
    )
    if ambiguous:
        raise ExportRefused(
            f"parties other than official parties are published on more than one axis: "
            f"{ambiguous!r} (party, kind, sector, axes) -- the index format cannot tell which "
            "dataset holds which axis"
        )


def export_parties(
    reader: ExportReader,
    parties: Mapping[str, PartyPublication],
    *,
    type_kinds: Mapping[str, PartyKind | None],
    kind_overrides: Mapping[str, KindOverride],
    total_groups: Mapping[str, TotalGroup],
    private_providers: Mapping[int, str],
    whole_totals: Mapping[str, WholePopulationTotal],
) -> list[dict]:
    """Each party's kind, sector, label, the datasets that publish it and the flow axes they
    publish it on (`parties`, each sorted), the group and residuals a total holds, and for a
    whole population total what it covers in each dataset (`whole`), sorted by dcid.

    A party in `kind_overrides` takes its override's kind. Any other takes the kind its graph
    types give under `type_kinds`. A `party` whose OECD donor code is in `private_providers` has
    the sector `private`, and every other `party` `official`, on every axis. A total in
    `whole_totals` has the sector its windows cover (`covered_sector`), and every other total
    and every residual none; the document refuses to load when two windows of a total cover
    different sectors. Its label is its graph name, `None` when the graph gives it
    none. A row carries `whole` only for a total in `whole_totals`, and a marker's `years` only
    for a window with a first year, ending in the last year the dataset publishes the total.

    Raises `ExportRefused` for a `kind_overrides` row naming a party not in `parties`, a party the
    graph holds no node or type for, a party carrying a type `type_kinds` does not name, a party
    whose types give it no single kind, a `total_groups` row naming a party that is not a total
    of `parties`, a `total_groups` residual that is not a residual of `parties`, a
    `private_providers` row the graph contradicts (`_check_private_providers`) or whose party the
    types give a kind other than `party`, a party of a class other than `official` or a total
    published on more than one axis (`_check_classes_on_one_axis`), and a `whole_totals` row the
    parties contradict (`_check_whole_population_totals`).
    """
    published = frozenset(parties)
    unpublished = sorted(kind_overrides.keys() - published)
    if unpublished:
        raise ExportRefused(
            f"PARTY_KIND_OVERRIDES names {unpublished!r}, which no declared or listed dataset publishes"
        )
    arcs = _fetch_arcs_many(reader, sorted(published), properties=("typeOf", "name", "dacCode"))
    codes = {dcid: _dac_codes(arcs.get(dcid) or {}) for dcid in published}
    origins = [dcid for dcid in published if "origin" in parties[dcid].axes]
    _check_private_providers(codes, origins, private_providers, kind_overrides)
    private = frozenset(
        dcid
        for dcid in published
        if any(private_providers.get(code) == dcid for code in codes[dcid])
    )
    rows = []
    kinds: dict[str, PartyKind] = {}
    for dcid in sorted(published):
        party_arcs = arcs.get(dcid) or {}
        if (override := kind_overrides.get(dcid)) is not None:
            kind = override.kind
        else:
            kind = _kind_from_types(dcid, party_arcs.get("typeOf", ()), type_kinds)
        if dcid in private and kind != "party":
            raise ExportRefused(
                f"PRIVATE_PROVIDERS lists {dcid!r}, whose types give it the kind {kind!r}; "
                "only a party can be private"
            )
        kinds[dcid] = kind
        total = total_groups.get(dcid)
        row: dict = {
            "dcid": dcid,
            "kind": kind,
            "sector": _party_sector(kind, private=dcid in private),
            "label": _first_or_none(party_arcs.get("name", ())),
            "totals": None if total is None else total.group,
            "residuals": [] if total is None else sorted(total.residuals),
            "sums_members": True if total is None else total.sums_members,
            "datasets": sorted(parties[dcid].datasets),
            "axes": sorted(parties[dcid].axes),
        }
        rows.append(row)
    stale = sorted(dcid for dcid in total_groups if kinds.get(dcid) != "total")
    if stale:
        raise ExportRefused(
            f"TOTAL_GROUPS names {stale!r}, which no declared or listed dataset publishes as a total"
        )
    not_residuals = sorted(
        residual
        for total in total_groups.values()
        for residual in total.residuals
        if kinds.get(residual) != "residual"
    )
    if not_residuals:
        raise ExportRefused(
            f"TOTAL_GROUPS counts {not_residuals!r} as residuals, which no declared or listed dataset "
            "publishes as a residual"
        )
    _check_classes_on_one_axis(rows, parties)
    _check_whole_population_totals(whole_totals, kinds, parties)
    for row in rows:
        if (whole := whole_totals.get(row["dcid"])) is not None:
            row["sector"] = covered_sector(next(iter(whole.datasets.values())).covers)
            row["whole"] = [
                _whole_marker_to_dict(dataset, window, parties[row["dcid"]])
                for dataset, window in sorted(whole.datasets.items())
            ]
    return rows


def _party_sector(kind: PartyKind, *, private: bool) -> Sector | None:
    """A row's sector before whole markers: `private` or `official` for a `party`, none for a
    total or a residual."""
    if kind != "party":
        return None
    return "private" if private else "official"


def strip_system_suffix(name: str) -> str:
    """Strip a trailing `(CRS ...)` or `(DAC ...)` suffix from a name.

    Preserve other parentheses, including interior groups and `(malaria (severe))`.
    """
    stripped = _LABEL_SUFFIX_RE.sub("", name)
    return stripped if stripped else name


def _single_value(dcid: str, prop: str, values: tuple[str, ...]) -> str:
    if not values:
        raise ExportRefused(f"{dcid} carries no {prop} arc -- cannot export a row missing it")
    return values[0]


def _check_single_valued(dcid: str, prop: str, values: tuple[str, ...]) -> None:
    """Reject multiple distinct values on `prop`.

    `parse_arcs` collapses repeated imports. Callers apply this to shape and constraint fields.
    """
    if len(values) > 1:
        raise ExportRefused(
            f"{dcid}'s {prop!r} arc carries {len(values)} bound values, not one: {values!r} -- "
            "cannot export a row that would silently pick one of several"
        )


def _check_typed_as_statvar(dcid: str, type_of: tuple[str, ...]) -> None:
    """Require `StatisticalVariable` in `typeOf`.

    `ProvisionalNode` alongside a real type is allowed.
    """
    if "StatisticalVariable" not in type_of:
        raise ExportRefused(
            f"{dcid}'s typeOf arc {type_of!r} does not include StatisticalVariable -- cannot "
            "export a row the graph does not hold as a StatisticalVariable"
        )


class ExportReader(Protocol):
    """The graph reads the export makes. `dc_graph.GraphClient` implements them, and so can a
    reader over a release."""

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
    ) -> SubjectPage:
        """Distinct subjects of `property` arcs into `dcid`, narrowed by `filters`. A page past
        `max_results` is `truncated`. See `dc_graph.GraphClient.node_subjects`."""
        ...

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str]
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        """Each dcid's values per property, `None` for an absent node. See
        `dc_graph.GraphClient.node_arcs_many`."""
        ...

    def node_arcs_batch(self, dcids: list[str]) -> Mapping[str, dict | None]:
        """Each dcid's forward arcs as the graph sends them, `None` for an absent node. The
        values carry the provenance that asserted them, which `export_source_codes` needs.
        See `dc_graph.GraphClient.node_arcs_batch`."""
        ...


def _first_or_none(values: tuple[str, ...]) -> str | None:
    return values[0] if values else None


def _fetch_arcs_many(
    reader: ExportReader, dcids: Sequence[str], *, properties: Sequence[str]
) -> dict[str, dict[str, tuple[str, ...]] | None]:
    """Read all dcids in one call.

    GraphClient chunks them.
    """
    return reader.node_arcs_many(dcids, properties=list(properties))


def _build_code_row(dcid: str, code_system: str, arcs: dict[str, tuple[str, ...]]) -> dict:
    code = _single_value(dcid, "classificationCode", arcs.get("classificationCode", ()))
    name = _single_value(dcid, "name", arcs.get("name", ()))
    label = strip_system_suffix(name)
    synonyms = sorted(
        {v for v in arcs.get("alternateName", ()) if v and v != label}
        | set(CODE_SYNONYMS.get(dcid, ()))
    )
    parent = _first_or_none(arcs.get("specializationOf", ()))
    short_name = _first_or_none(arcs.get("shortName", ())) or CODE_SHORT_NAMES.get(dcid)
    return {
        "dcid": dcid,
        "code_system": code_system,
        "code": code,
        "label": label,
        "synonyms": synonyms,
        "short_name": short_name,
        "parent": parent,
        "definition": _first_or_none(arcs.get("description", ())),
    }


def _apply_concept_groups(
    codes: Sequence[dict],
    groups: Sequence[Sequence[tuple[str, str]]],
    exported_systems: Collection[str],
) -> None:
    """Give every code row of each of `groups` the union of the group's labels and synonyms
    as its synonyms, its own label left out. A member of a code system the export does not
    enumerate (`exported_systems`) is skipped.

    Raises:
        ExportRefused: a code is in two groups, a group joins two codes of one code system, or
            a group names a code of an exported system that the export holds no row of.
    """
    by_dcid = {row["dcid"]: row for row in codes}
    repeated = sorted(
        dcid
        for dcid, count in Counter(dcid for group in groups for _system, dcid in group).items()
        if count > 1
    )
    if repeated:
        raise ExportRefused(f"concept groups name codes in more than one group: {repeated}")
    for group in groups:
        systems = [system for system, _dcid in group]
        if len(set(systems)) != len(systems):
            raise ExportRefused(f"concept group {group!r} joins two codes of one code system")
        members = [(system, dcid) for system, dcid in group if system in exported_systems]
        missing = [
            dcid
            for system, dcid in members
            if dcid not in by_dcid or by_dcid[dcid]["code_system"] != system
        ]
        if missing:
            raise ExportRefused(
                f"concept group {group!r} names codes the export holds no row of: {missing}"
            )
        rows = [by_dcid[dcid] for _system, dcid in members]
        texts = {text for row in rows for text in (row["label"], *row["synonyms"])}
        for row in rows:
            row["synonyms"] = sorted(text for text in texts if text != row["label"])


type _MeasureNames = frozenset[tuple[str, str]]


def _measure_texts(measure_values: Sequence[dict]) -> TokenLookup[_MeasureNames]:
    """Every matchable measure-value label and synonym, found as the matcher finds it
    (`code_index.tokens`), to the `(axis, dcid)` pairs it reads. A text names its value's
    whole reading: the canonical dcid and every alias dcid filed under it
    (`same_reading_as`), as the matcher files them."""
    reading: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in measure_values:
        reading[(row["axis"], row["same_reading_as"] or row["dcid"])].add(row["dcid"])
    texts: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in measure_values:
        if not row["matchable"]:
            continue
        canonical = (row["axis"], row["same_reading_as"] or row["dcid"])
        for text in (row["label"], *row["synonyms"]):
            texts[text.strip().casefold()].update(
                (row["axis"], dcid) for dcid in reading[canonical]
            )
    return TokenLookup((text, frozenset(names)) for text, names in texts.items())


def _measure_names(
    span: Sequence[Token] | None, measure_texts: TokenLookup[_MeasureNames]
) -> _MeasureNames:
    """The `(axis, dcid)` pairs every measure text matching `span` reads."""
    if span is None:
        return frozenset()
    return frozenset(pair for names in measure_texts.get(span) for pair in names)


def _measure_runs(
    text: str, measure_texts: TokenLookup[_MeasureNames]
) -> list[tuple[int, int, _MeasureNames]]:
    """The runs of `text`'s words a query holding `text` would lose to it, as `(start, end,
    names)` word ranges, left to right and longest first at each start.

    The matcher (`search.engine.qualifier_match`) reads a query as `code_index.tokens` words
    and separators, and a query holding `text` matches it whatever separators `text` holds
    ("ODA: Total Net"). A matching `text` claims its every word, so it takes each run of its
    words that a measure text matches, exact or less one trailing "s". A `text` that is
    itself a measure text ("PPP") shares its span with the measure value, which keeps both
    readings (`match_qualifier_readings`), and takes nothing.
    """
    tokens = tokenize(text)
    word_at = [position for position, token in enumerate(tokens) if token.kind == "word"]
    if not word_at or _measure_names(tokens, measure_texts):
        return []
    runs: list[tuple[int, int, _MeasureNames]] = []
    start = 0
    while start < len(word_at):
        for end in range(len(word_at), start, -1):
            span = tokens[word_at[start] : word_at[end - 1] + 1]
            names = _measure_names(span, measure_texts) or _measure_names(
                singular(span), measure_texts
            )
            if names:
                runs.append((start, end, names))
                start = end
                break
        else:
            start += 1
    return runs


def _strip_measure_words(
    codes: Sequence[dict],
    statvars: Sequence[dict],
    measure_values: Sequence[dict],
    measure_axes: Sequence[MeasureAxis],
) -> None:
    """Take the measure words out of every code synonym that would take them from a query.

    Measure values are read in every code system together, and the matcher claims the longest
    span first, so a code synonym holding "net" or "gross" would take the word from the netting
    basis in every vocabulary: "gross ODA" would read the ODA code and no basis
    (`_measure_runs`). Such a synonym keeps the rest of its words, so "Gross ODA Loans" becomes
    "ODA Loans", so a query reads the code in the remainder and the measure in the word. A run
    stays when every series carrying any code that holds the text has the measure it names,
    since the text then implies it ("ODA grant equivalents of bilateral loans" on a code whose
    one series is a grant equivalent). The test covers every code holding the text, because
    the text takes the words in every vocabulary, and codes of a concept group must keep
    reading one set of texts. A synonym left with no words is dropped, and a remainder the code
    already reads adds nothing. A label stays, since it is the code's own name, and so does a
    text in `MEASURE_WORD_SYNONYMS_KEPT`.
    """
    measure_texts = _measure_texts(measure_values)
    axes = {axis.axis: axis for axis in measure_axes}
    series_by_code: dict[str, list[dict]] = defaultdict(list)
    for statvar in statvars:
        for value in statvar["constraint_properties"].values():
            series_by_code[value].append(statvar)

    codes_by_text: dict[str, list[str]] = defaultdict(list)
    for row in codes:
        for text in {text.strip().casefold() for text in (row["label"], *row["synonyms"])}:
            codes_by_text[text].append(row["dcid"])

    def implied(text: str, names: _MeasureNames) -> bool:
        series = [
            statvar
            for dcid in codes_by_text[text.strip().casefold()]
            for statvar in series_by_code[dcid]
        ]
        return bool(series) and all(
            any(
                axis in axes and _measure_axis_value(statvar, axes[axis]) == value
                for axis, value in names
            )
            for statvar in series
        )

    for row in codes:
        kept: list[str] = []
        remainders: list[str] = []
        for text in row["synonyms"]:
            stripped = [
                (start, end)
                for start, end, names in _measure_runs(text, measure_texts)
                if not implied(text, names)
            ]
            if not stripped or text.strip().casefold() in MEASURE_WORD_SYNONYMS_KEPT:
                kept.append(text)
                continue
            words = [token.text for token in tokenize(text) if token.kind == "word"]
            remainders.append(
                " ".join(
                    word
                    for position, word in enumerate(words)
                    if not any(start <= position < end for start, end in stripped)
                )
            )
        known = {text.casefold() for text in (row["label"], *kept)}
        for remainder in remainders:
            if remainder and remainder.casefold() not in known:
                kept.append(remainder)
                known.add(remainder.casefold())
        row["synonyms"] = sorted(kept)


def _reject_cross_system_parents(
    config: CodeSystemConfig, enumerated: set[str], rows: Sequence[dict]
) -> None:
    """Reject parents outside this code system's enumeration with `ExportRefused`.

    `specializationOf` must stay within one vocabulary. Parent and child dcids are already in hand,
    so this check needs no graph read.
    """
    for row in rows:
        parent = row["parent"]
        if parent is not None and parent not in enumerated:
            raise ExportRefused(
                f"{row['dcid']} in code system {config.code_system!r} has parent {parent!r}, "
                f"which {config.enum_dcid} does not enumerate -- a code row's parent must be a "
                "code of its own code system"
            )


def _shape_slot_default_to_dict(default: ShapeSlotDefault) -> dict:
    return {
        "population_type": default.population_type,
        "property": default.property_dcid,
        "value": default.value_dcid,
    }


def _partly_stated_value_to_dict(declared: PartlyStatedValue) -> dict:
    return {
        "provenance": declared.provenance,
        "property": declared.property_dcid,
        "value": declared.value_dcid,
    }


def _measure_class_to_dict(declaration: MeasureClassDeclaration) -> dict:
    return {
        "class": declaration.measure_class,
        "pattern": declaration.pattern.pattern,
        "fallback": declaration.fallback,
    }


def _stat_type_word_to_dict(declaration: StatTypeWordDeclaration) -> dict:
    return {"pattern": declaration.pattern.pattern, "stat_type": declaration.stat_type}


def export_code_system(
    reader: ExportReader, config: CodeSystemConfig, *, max_results: int = _DEFAULT_MAX_RESULTS
) -> tuple[dict, list[dict]]:
    """Return a declaration and rows for `config`'s code system.

    Derive `hierarchical` from parent arcs and reject disagreement with the configured value or
    parents outside the enumerated system.
    """
    page = reader.node_subjects(config.enum_dcid, property="typeOf", max_results=max_results)
    if page.truncated:
        raise ExportRefused(
            f"{config.code_system} ({config.enum_dcid}) population truncated at "
            f"{max_results} dcids -- raise max_results, the export must read the full "
            "code system"
        )
    dcids = sorted(set(page.dcids))

    arcs_by_dcid = _fetch_arcs_many(reader, dcids, properties=_CODE_ARC_PROPERTIES)
    rows = []
    for dcid in dcids:
        arcs = arcs_by_dcid.get(dcid)
        if arcs is None:
            raise ExportRefused(
                f"{dcid} was enumerated via <-typeOf on {config.enum_dcid} but the graph "
                "holds no node for it"
            )
        rows.append(_build_code_row(dcid, config.code_system, arcs))
    rows.sort(key=lambda row: row["dcid"])
    _reject_cross_system_parents(config, set(dcids), rows)

    derived_hierarchical = any(row["parent"] is not None for row in rows)
    if derived_hierarchical != config.expected_hierarchical:
        raise ExportRefused(
            f"{config.code_system}: derived hierarchical={derived_hierarchical} disagrees "
            f"with the declared expected value {config.expected_hierarchical}"
        )

    declaration = {
        "code_system": config.code_system,
        "hierarchical": derived_hierarchical,
        "disjoint_from_children": config.disjoint_from_children,
        "summable": config.summable,
        "absent_reading": config.absent_reading,
        "plain_name": config.plain_name,
        "default": _declared_default_to_dict(config.default),
        "enum_dcid": config.enum_dcid,
        "no_vocabulary_terms": list(config.no_vocabulary_terms),
    }
    return declaration, rows


def _declared_default_to_dict(default: DeclaredDefault | None) -> dict | None:
    if default is None:
        return None
    return {
        "code": default.code,
        "when_bound_under": list(default.when_bound_under),
        "when_named": list(default.when_named),
        "when_open": default.when_open,
        "note": default.note,
    }


def _validate_declared_defaults(
    code_systems: Sequence[CodeSystemConfig], code_rows_by_system: dict[str, list[dict]]
) -> None:
    """Validate each default against exported rows, raising `ExportRefused` on invalid entries.

    Its code must belong to its own system. `when_bound_under` codes must belong to other systems.
    `when_named` entries are non-empty lowercase strings. A default that fires on an open slot
    (`when_open`) carries a note.
    """
    dcids_by_system = {
        system: {row["dcid"] for row in rows} for system, rows in code_rows_by_system.items()
    }
    for config in code_systems:
        default = config.default
        if default is None:
            continue
        if default.code not in dcids_by_system.get(config.code_system, set()):
            raise ExportRefused(
                f"{config.code_system}: declared default code {default.code!r} is not a row "
                "of that code system"
            )
        for scoping_dcid in default.when_bound_under:
            owning_system = next(
                (system for system, dcids in dcids_by_system.items() if scoping_dcid in dcids),
                None,
            )
            if owning_system is None:
                raise ExportRefused(
                    f"{config.code_system}: declared default when_bound_under code "
                    f"{scoping_dcid!r} is not a row of any exported code system"
                )
            if owning_system == config.code_system:
                raise ExportRefused(
                    f"{config.code_system}: declared default when_bound_under code "
                    f"{scoping_dcid!r} belongs to its own code system -- a default cannot "
                    "condition on the slot it defaults"
                )
        invalid = invalid_when_named_entry(default.when_named)
        if invalid is not None:
            raise ExportRefused(
                f"{config.code_system}: declared default when_named entry "
                f"{default.when_named[invalid]!r} must be a non-empty lowercase string"
            )
        if default.when_open and not (default.note and default.note.strip()):
            raise ExportRefused(
                f"{config.code_system}: declared default {default.code!r} fires on an open "
                "slot and carries no note"
            )


@dataclass(frozen=True)
class _PartitionAxis:
    """A filter axis and its split values.

    `None` uses the run-wide observed-value pool.
    """

    property: str
    values: tuple[str, ...] | None


def _complete_systems(
    dataset: DatasetDeclaration, code_systems: Sequence[CodeSystemConfig]
) -> list[CodeSystemConfig]:
    """Return the code systems every StatVar of `dataset` declares."""
    return [
        config
        for config in code_systems
        if dataset.provenance in config.declared_by_every_statvar_of
    ]


def _choose_spine(
    dataset: DatasetDeclaration,
    code_systems: Sequence[CodeSystemConfig],
    code_rows_by_system: dict[str, list[dict]],
) -> CodeSystemConfig:
    """Return the code system `dataset`'s StatVars are enumerated through, or raise if none
    qualifies: the largest system every one of its StatVars declares.
    """
    eligible = _complete_systems(dataset, code_systems)
    if not eligible:
        raise ExportRefused(
            f"no code system in the table is declared by every StatVar of dataset "
            f"{dataset.provenance!r} -- cannot choose an enumeration spine"
        )
    return max(eligible, key=lambda config: len(code_rows_by_system[config.code_system]))


def _partition_axes(
    spine: CodeSystemConfig,
    dataset: DatasetDeclaration,
    code_systems: Sequence[CodeSystemConfig],
    code_rows_by_system: dict[str, list[dict]],
) -> tuple[_PartitionAxis, ...]:
    """Return `dataset`'s other complete code systems by descending size, then observed
    `populationType`."""
    code_axes = sorted(
        (
            _PartitionAxis(
                property=config.code_system,
                values=tuple(
                    sorted(row["dcid"] for row in code_rows_by_system[config.code_system])
                ),
            )
            for config in _complete_systems(dataset, code_systems)
            if config.code_system != spine.code_system
        ),
        key=lambda axis: len(axis.values or ()),
        reverse=True,
    )
    return (*code_axes, _PartitionAxis(property="populationType", values=None))


def _observe_axis_values(
    reader: ExportReader, dcids: Sequence[str], property: str
) -> tuple[str, ...]:
    """Return distinct `property` values from the dcids already read."""
    sample = sorted(set(dcids))
    if not sample:
        return ()
    arcs_by_dcid = _fetch_arcs_many(reader, sample, properties=(property,))
    values = {value for arcs in arcs_by_dcid.values() if arcs for value in arcs.get(property, ())}
    return tuple(sorted(values))


class _ObservedAxisPool:
    """Pool observed axis values across one StatVar enumeration.

    Truncated leaf samples are arbitrary prefixes, so a leaf can use values found by earlier
    reads. `used()` records split values for `_verify_axis_coverage` against exported rows.
    """

    def __init__(self) -> None:
        self._observed: dict[str, set[str]] = {}
        self._used: dict[str, set[str]] = {}

    def observe(self, reader: ExportReader, dcids: Sequence[str], property: str) -> None:
        self._observed.setdefault(property, set()).update(
            _observe_axis_values(reader, dcids, property)
        )

    def values_for(self, property: str) -> tuple[str, ...]:
        return tuple(sorted(self._observed.get(property, ())))

    def record_used(self, property: str, values: Sequence[str]) -> None:
        self._used.setdefault(property, set()).update(values)

    def used(self) -> dict[str, tuple[str, ...]]:
        return {property: tuple(sorted(values)) for property, values in self._used.items()}


def _format_filter_path(base_property: str, base_dcid: str, filters: Mapping[str, str]) -> str:
    parts = [f"{base_property}={base_dcid!r}"]
    parts.extend(f"{key}={value!r}" for key, value in sorted(filters.items()))
    return ", ".join(parts)


def _read_partitioned(
    reader: ExportReader,
    base_property: str,
    base_dcid: str,
    filters: dict[str, str],
    remaining_axes: tuple[_PartitionAxis, ...],
    pool: _ObservedAxisPool,
    *,
    max_results: int,
    max_depth: int,
    depth: int = 0,
) -> set[str]:
    """Read reverse subjects with filters and return only complete leaves.

    This graph repeats reverse-arc pages. Truncation and `GraphNotAdvancingError` both trigger
    partitioning. Other `GraphReadError`s propagate.

    Split along remaining axes in order, using declared code values or the run-wide observed
    `populationType` pool. Union completed leaves. `ExportRefused` names the filter path when axes,
    observed values, or `max_depth` are exhausted.
    """
    from dc_graph import GraphNotAdvancingError

    try:
        page = reader.node_subjects(
            base_dcid, property=base_property, filters=filters or None, max_results=max_results
        )
    except GraphNotAdvancingError:
        page = None

    if page is not None and not page.truncated:
        for axis in remaining_axes:
            if axis.values is None:
                pool.observe(reader, page.dcids, axis.property)
        return set(page.dcids)

    filter_path = _format_filter_path(base_property, base_dcid, filters)
    if depth >= max_depth:
        raise ExportRefused(
            f"partition recursion exceeded max_depth={max_depth} reading {filter_path} -- "
            "still truncated"
        )
    if not remaining_axes:
        raise ExportRefused(
            f"exhausted every partition property reading {filter_path} -- still truncated"
        )

    axis, *rest = remaining_axes
    if axis.values is None:
        if page is not None:
            pool.observe(reader, page.dcids, axis.property)
        values = pool.values_for(axis.property)
        if not values:
            raise ExportRefused(
                f"could not observe any {axis.property!r} value while partitioning "
                f"{filter_path} -- cannot split further"
            )
        pool.record_used(axis.property, values)
    else:
        values = axis.values

    dcids: set[str] = set()
    for value in values:
        dcids |= _read_partitioned(
            reader,
            base_property,
            base_dcid,
            {**filters, axis.property: value},
            tuple(rest),
            pool,
            max_results=max_results,
            max_depth=max_depth,
            depth=depth + 1,
        )
    return dcids


def _verify_spine_coverage(
    reader: ExportReader,
    enumerated: set[str],
    spines: Sequence[CodeSystemConfig],
    code_systems: Sequence[CodeSystemConfig],
    code_rows_by_system: dict[str, list[dict]],
    *,
    max_results: int,
) -> int:
    """Check that complete one-page reads from non-spine codes stay within `enumerated`.

    An outside subject raises `ExportRefused`. Truncated or non-advancing reads are skipped, and the
    function returns their count.
    """
    from dc_graph import GraphNotAdvancingError

    spine_systems = {spine.code_system for spine in spines}
    skipped = 0
    total = 0
    for config in code_systems:
        if config.code_system in spine_systems:
            continue
        for row in code_rows_by_system[config.code_system]:
            code_dcid = row["dcid"]
            total += 1
            try:
                page = reader.node_subjects(
                    code_dcid, property=config.code_system, max_results=max_results
                )
            except GraphNotAdvancingError:
                skipped += 1
                continue
            if page.truncated:
                skipped += 1
                continue
            outside = set(page.dcids) - enumerated
            if outside:
                raise ExportRefused(
                    f"{config.code_system}={code_dcid!r} carries {sorted(outside)!r} outside "
                    f"the population enumerated through {sorted(spine_systems)!r} -- the "
                    "declared datasets do not cover the whole StatVar population"
                )
    if skipped:
        print(
            f"spine coverage check skipped {skipped}/{total} non-spine code(s) too large to "
            "read in one response",
            file=sys.stderr,
        )
    return skipped


_AXIS_ROW_FIELD = {"populationType": "population_type"}
# Map each observed partition axis to its StatVar row field for coverage checks.


def _verify_axis_coverage(
    statvars: Sequence[dict], observed_axis_usage: Mapping[str, tuple[str, ...]]
) -> None:
    """Require every exported value on a used observed axis to appear in its partition filters.

    Unexpected values raise `ExportRefused` with the axis and StatVar dcid.
    """
    for axis_property, used_values in observed_axis_usage.items():
        row_field = _AXIS_ROW_FIELD[axis_property]
        used = set(used_values)
        for row in statvars:
            value = row.get(row_field)
            if value is not None and value not in used:
                raise ExportRefused(
                    f"{row['dcid']} carries {axis_property}={value!r}, which was never used to "
                    f"partition the {axis_property} axis (used values: {sorted(used)!r}) -- some "
                    "leaf may have split on an incomplete sample and dropped rows silently"
                )


# Row fields in the order of `_STATVAR_SHAPE_PROPERTIES`.
_ROW_SHAPE_FIELDS = (
    "population_type",
    "measured_property",
    "stat_type",
    "measurement_qualifier",
    "measurement_denominator",
)
# A refusal names at most this many missing StatVars, then the total.
_MISSING_STATVAR_NAME_CAP = 20

type _FiveTuple = tuple[str | None, ...]


def _row_five_tuple(row: Mapping) -> _FiveTuple:
    return tuple(row[field] for field in _ROW_SHAPE_FIELDS)


def _read_five_tuple_subjects(
    reader: ExportReader,
    five_tuple: _FiveTuple,
    axes: tuple[_PartitionAxis, ...],
    *,
    max_results: int,
    max_depth: int,
) -> set[str]:
    """Return every `<-measuredProperty` subject on `five_tuple` that carries a value of each axis.

    Filters on the tuple's other set fields and splits first on every value of the first axis,
    since an unsplit read of a CRS five-tuple does not finish inside a request. A read still
    truncated after the last axis raises `ExportRefused` through `_read_partitioned`.
    """
    shape = dict(zip(_STATVAR_SHAPE_PROPERTIES, five_tuple, strict=True))
    measured_property = shape.pop("measuredProperty")
    if measured_property is None:
        raise ExportRefused(f"five-tuple {five_tuple!r} has no measuredProperty to read from")
    filters = {prop: value for prop, value in shape.items() if value is not None}
    pool = _ObservedAxisPool()
    if not axes:
        return _read_partitioned(
            reader,
            "measuredProperty",
            measured_property,
            filters,
            (),
            pool,
            max_results=max_results,
            max_depth=max_depth,
        )
    first, *rest = axes
    subjects: set[str] = set()
    for value in first.values or ():
        subjects |= _read_partitioned(
            reader,
            "measuredProperty",
            measured_property,
            {**filters, first.property: value},
            tuple(rest),
            pool,
            max_results=max_results,
            max_depth=max_depth,
        )
    return subjects


def _minimal_system_sets(
    key_sets: set[frozenset[str]], code_rows_by_system: Mapping[str, list[dict]]
) -> list[frozenset[str]]:
    """Return the code systems each key set of one five-tuple declares, keeping only the sets
    no other set is a proper subset of.

    A StatVar covering some row's key set carries every system of that set, so a read split on
    a minimal set below it reaches the StatVar. Two datasets on one five-tuple with disjoint
    systems, such as the CRS and DAC1, each get a read of their own.
    """
    system_sets = {
        frozenset(system for system in keys if system in code_rows_by_system) for keys in key_sets
    }
    return sorted(
        (systems for systems in system_sets if not any(other < systems for other in system_sets)),
        key=sorted,
    )


def _verify_five_tuple_completeness(
    reader: ExportReader,
    statvars: Sequence[dict],
    code_rows_by_system: Mapping[str, list[dict]],
    spine_systems: frozenset[str],
    *,
    max_results: int = _DEFAULT_MAX_RESULTS,
    max_depth: int = _MAX_PARTITION_DEPTH,
) -> None:
    """Refuse an export that misses a StatVar the index would decide.

    Search takes a StatVar's family from the index when its constraint properties hold every
    constraint property of some exported row of its five-tuple (`index_covers_slot_keys` in
    `code_index.members`, which this check shares through `key_sets_cover`), so such a StatVar
    must be an exported row. One the index would not decide, such as an IDS debt series with
    no CRS keys, passes. It is a listed row (`_verify_listed_five_tuples`) or the graph walk
    resolves it.

    A decided StatVar carries every code system of the exported row it covers. Each five-tuple
    gets one reverse read per minimal set of code systems its rows declare
    (`_minimal_system_sets`), split on those systems' enum codes, smallest non-spine system first
    and spine systems last, so the reads reach StatVars the spine enumeration did not. Values
    come from the enum, since exported rows would hide a value no row shows. The forward read of
    each unexported subject confirms its five-tuple, including the denominator, and its key set.
    Truncated or exhausted reads refuse, as does a subject the forward read finds no node for.

    A decided StatVar whose codes lie outside the typed enums escapes both the spine and this
    check, and nothing else catches it. The check proves completeness at export time. The live
    drift test (`tests/code_index/test_export_drift.py`) re-exports through
    `export_graph_blocks` and catches rows changed or removed since the committed index.
    """
    key_sets: dict[_FiveTuple, set[frozenset[str]]] = {}
    for row in statvars:
        key_sets.setdefault(_row_five_tuple(row), set()).add(
            frozenset(row["constraint_properties"])
        )
    exported = {row["dcid"] for row in statvars}

    unexported: set[str] = set()
    for five_tuple, keys in key_sets.items():
        for systems in _minimal_system_sets(keys, code_rows_by_system):
            ordered = sorted(
                systems,
                key=lambda system: (
                    system in spine_systems,
                    len(code_rows_by_system[system]),
                    system,
                ),
            )
            axes = tuple(
                _PartitionAxis(
                    property=system,
                    values=tuple(sorted(row["dcid"] for row in code_rows_by_system[system])),
                )
                for system in ordered
            )
            subjects = _read_five_tuple_subjects(
                reader, five_tuple, axes, max_results=max_results, max_depth=max_depth
            )
            unexported |= subjects - exported
    if not unexported:
        return

    candidates = sorted(unexported)
    arcs_by_dcid = _fetch_arcs_many(
        reader, candidates, properties=("constraintProperties", *_STATVAR_SHAPE_PROPERTIES)
    )
    missing = []
    for dcid in candidates:
        arcs = arcs_by_dcid.get(dcid)
        if arcs is None:
            raise ExportRefused(
                f"{dcid} is a <-measuredProperty subject of an exported five-tuple but the "
                "graph holds no node for it"
            )
        five_tuple = tuple(_first_or_none(arcs.get(prop, ())) for prop in _STATVAR_SHAPE_PROPERTIES)
        declared = frozenset(arcs.get("constraintProperties", ()))
        if key_sets_cover(key_sets, five_tuple, declared):
            missing.append(dcid)
    if missing:
        named = missing[:_MISSING_STATVAR_NAME_CAP]
        more = f" and {len(missing) - len(named)} more" if len(missing) > len(named) else ""
        raise ExportRefused(
            f"{len(missing)} StatVar(s) on an exported five-tuple carry the constraint "
            f"properties of an exported row but are not exported rows: {named!r}{more} -- the "
            "index would decide their family without them"
        )


def listed_dcids(
    provenances: Mapping[str, Collection[str]], datasets: Collection[str]
) -> tuple[str, ...]:
    """Return the sorted StatVars `provenances` assigns to any of `datasets`.

    `provenances` maps a StatVar to the datasets publishing it: a coverage build's
    (`CoverageJoin.provenances`), or the committed index's listed rows for the drift test. A
    dataset of `datasets` with no StatVar raises `ExportRefused`, since the export could not list
    its rows."""
    found = {dcid: set(datasets).intersection(sets) for dcid, sets in provenances.items()}
    empty = sorted(set(datasets).difference(*found.values()))
    if empty:
        raise ExportRefused(
            f"listed dataset(s) {empty!r} hold no StatVar in the coverage build -- the export "
            "cannot list their rows"
        )
    return tuple(sorted(dcid for dcid, datasets_of in found.items() if datasets_of))


def export_source_codes(reader: ExportReader, dcids: Sequence[str]) -> list[dict]:
    """Build the `source_codes` rows from the `sourceIndicatorCode` arcs of `dcids`: one row
    per arc provenance and upper-cased code, naming its StatVars, sorted.

    The rows key on the provenance each arc value carries. The coverage build gives the same
    StatVar its provenances separately, and `parse_index` refuses a code row whose provenance
    its StatVar's row lacks. A StatVar of `dcids` with no code, and a code value with no
    provenance or no text, raise `ExportRefused`. Every listed dataset gives each of its
    StatVars a series code (World Bank IDS is the only one), so a listed dataset without codes
    must lift this rule before it is listed."""
    arcs = reader.node_arcs_batch(list(dcids))
    members: dict[tuple[str, str], set[str]] = defaultdict(set)
    uncoded: list[str] = []
    for dcid in dcids:
        values = ((arcs.get(dcid) or {}).get(_SOURCE_CODE) or {}).get("nodes", ())
        if not values:
            uncoded.append(dcid)
            continue
        for value in values:
            provenance, code = value.get("provenanceId"), (value.get("value") or "").strip()
            if not provenance or not code:
                raise ExportRefused(
                    f"{dcid}'s {_SOURCE_CODE} arc carries {value!r}, which names no provenance "
                    "or no code -- cannot key a series code on it"
                )
            members[provenance, code.upper()].add(dcid)
    if uncoded:
        named = uncoded[:_MISSING_STATVAR_NAME_CAP]
        more = f" and {len(uncoded) - len(named)} more" if len(uncoded) > len(named) else ""
        raise ExportRefused(
            f"{len(uncoded)} listed StatVar(s) carry no {_SOURCE_CODE}: {named!r}{more} -- "
            "search could not answer their series code"
        )
    return [
        {"provenance": provenance, "code": code, "dcids": sorted(found)}
        for (provenance, code), found in sorted(members.items())
    ]


def export_value_parents(reader: ExportReader, statvars: Sequence[dict]) -> list[dict]:
    """Build the `value_parents` rows: the `specializationOf` chain above each constraint
    value a row of `statvars` holds, one row per value with a parent, the parents' own
    parents included, sorted by dcid.

    A value with two parents, and a chain that returns to a value it holds, raise
    `ExportRefused`, since `CodeIndex.value_ancestors` walks one finite chain."""
    parent_of: dict[str, str] = {}
    read: set[str] = set()
    frontier = sorted(
        {value for row in statvars for value in row["constraint_properties"].values()}
    )
    while frontier:
        read.update(frontier)
        arcs = _fetch_arcs_many(reader, frontier, properties=("specializationOf",))
        for dcid in frontier:
            parents = (arcs.get(dcid) or {}).get("specializationOf", ())
            _check_single_valued(dcid, "specializationOf", parents)
            if parents:
                parent_of[dcid] = parents[0]
        frontier = sorted({parent_of[d] for d in frontier if d in parent_of} - read)
    for dcid in parent_of:
        chain = [dcid]
        while (node := parent_of.get(chain[-1])) is not None:
            if node in chain:
                raise ExportRefused(
                    f"the specializationOf chain of {dcid} returns to {node}: "
                    f"{' -> '.join([*chain, node])}"
                )
            chain.append(node)
    return [{"dcid": dcid, "parent": parent} for dcid, parent in sorted(parent_of.items())]


def _verify_listed_five_tuples(
    reader: ExportReader,
    statvars: Sequence[dict],
    listed: Sequence[dict],
    *,
    max_results: int = _UNSPLIT_MAX_RESULTS,
    max_depth: int = _MAX_PARTITION_DEPTH,
) -> None:
    """Refuse an export when a five-tuple a listed row has holds a StatVar that is neither an
    exported nor a listed row, or when a listed StatVar is also an exported row.

    Search and the mint elect a family that holds a listed row from the index's rows of its
    five-tuple, in place of a graph walk over it (`code_index.members.listed_group`), so those
    rows must be every StatVar of the five-tuple. Listed rows declare no code system to split
    on, so each five-tuple gets one unsplit read of up to `max_results` subjects, and a read
    still truncated refuses."""
    exported = {row["dcid"] for row in statvars}
    both = sorted(exported.intersection(row["dcid"] for row in listed))
    if both:
        raise ExportRefused(
            f"{len(both)} listed StatVar(s) are also exported rows: "
            f"{both[:_MISSING_STATVAR_NAME_CAP]!r}"
        )
    known = exported.union(row["dcid"] for row in listed)
    unknown: set[str] = set()
    for five_tuple in sorted({_row_five_tuple(row) for row in listed}, key=repr):
        unknown |= (
            _read_five_tuple_subjects(
                reader, five_tuple, (), max_results=max_results, max_depth=max_depth
            )
            - known
        )
    if unknown:
        named = sorted(unknown)[:_MISSING_STATVAR_NAME_CAP]
        more = f" and {len(unknown) - len(named)} more" if len(unknown) > len(named) else ""
        raise ExportRefused(
            f"{len(unknown)} StatVar(s) on a listed row's five-tuple are neither exported nor "
            f"listed rows: {named!r}{more} -- members elected from the index would miss them"
        )


def enumerate_statvar_dcids(
    reader: ExportReader,
    code_rows_by_system: dict[str, list[dict]],
    code_systems: Sequence[CodeSystemConfig],
    datasets: Sequence[DatasetDeclaration],
    *,
    max_results: int = _DEFAULT_MAX_RESULTS,
    max_partition_depth: int = _MAX_PARTITION_DEPTH,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Return sorted StatVar dcids and observed-axis values used for partitioning.

    Enumerate each dataset through its own spine (`_choose_spine`), split incomplete reads on the
    systems every StatVar of that dataset declares, and share one `_ObservedAxisPool` across all
    reads. Datasets with the same spine and axes share one pass. Check non-spine coverage of the
    union afterward.
    """
    pool = _ObservedAxisPool()
    dcids: set[str] = set()
    spines: dict[str, CodeSystemConfig] = {}
    read_plans: set[tuple[str, tuple[_PartitionAxis, ...]]] = set()
    for dataset in datasets:
        spine = _choose_spine(dataset, code_systems, code_rows_by_system)
        axes = _partition_axes(spine, dataset, code_systems, code_rows_by_system)
        if (spine.code_system, axes) in read_plans:
            continue
        read_plans.add((spine.code_system, axes))
        spines[spine.code_system] = spine
        for row in code_rows_by_system[spine.code_system]:
            dcids |= _read_partitioned(
                reader,
                spine.code_system,
                row["dcid"],
                {},
                axes,
                pool,
                max_results=max_results,
                max_depth=max_partition_depth,
            )

    _verify_spine_coverage(
        reader,
        dcids,
        tuple(spines.values()),
        code_systems,
        code_rows_by_system,
        max_results=max_results,
    )
    return tuple(sorted(dcids)), pool.used()


def _reject_provisional_constraint_values(
    reader: ExportReader, rows: Sequence[tuple[str, dict[str, str]]]
) -> None:
    """Require each constraint value to identify a resolved graph node.

    One batched read covers all distinct values. Missing nodes, missing `typeOf`, or nodes typed
    only `ProvisionalNode` raise `ExportRefused`; a provisional type alongside a real type passes.
    """
    candidate_values = sorted(
        {value for _, constraint_properties in rows for value in constraint_properties.values()}
    )
    if not candidate_values:
        return
    type_arcs = _fetch_arcs_many(reader, candidate_values, properties=("typeOf",))
    for dcid, constraint_properties in rows:
        for prop, value in constraint_properties.items():
            held = type_arcs.get(value)
            if held is None:
                raise ExportRefused(
                    f"{dcid} binds {prop!r} to {value!r}, which the graph holds no node for "
                    "at all -- cannot export a dangling reference as a plain dcid "
                    "indistinguishable from a resolved code"
                )
            types = set(held.get("typeOf", ()))
            if not types:
                raise ExportRefused(
                    f"{dcid} binds {prop!r} to {value!r}, whose node carries no typeOf arc "
                    "at all -- cannot export a dangling reference as a plain dcid "
                    "indistinguishable from a resolved code"
                )
            if types == {"ProvisionalNode"}:
                raise ExportRefused(
                    f"{dcid} binds {prop!r} to {value!r}, which the graph types exactly "
                    "ProvisionalNode -- cannot export a dangling reference as a plain dcid "
                    "indistinguishable from a resolved code"
                )


def export_statvars(reader: ExportReader, statvar_dcids: Sequence[str]) -> list[dict]:
    """Build StatVar rows with two forward reads.

    Pass 1 reads types, shape fields, and each node's declared `constraintProperties`. Pass 2 reads
    the union of those properties and fills each row's mapping.
    Constraint and observation properties remain list-valued.

    Missing declared values, unresolved dcids, non-StatisticalVariables, multi-valued single-value
    fields, and unverified targets raise `ExportRefused`. The target check rejects missing nodes,
    missing `typeOf`, and nodes typed only `ProvisionalNode`.
    """
    from dc_graph import UNRESOLVED_NODE

    pass1 = _fetch_arcs_many(reader, statvar_dcids, properties=_STATVAR_PASS1_PROPERTIES)

    partial: list[tuple[str, dict[str, tuple[str, ...]], tuple[str, ...]]] = []
    constraint_property_names: set[str] = set()
    for dcid in statvar_dcids:
        arcs = pass1.get(dcid)
        if arcs is None:
            raise ExportRefused(
                f"the graph holds no node for {dcid} -- it is either a StatVar this "
                "export just enumerated and the graph dropped mid-read, or a row the "
                "committed index still carries and the graph no longer holds"
            )
        _check_typed_as_statvar(dcid, arcs.get("typeOf", ()))
        for prop in _STATVAR_SHAPE_PROPERTIES:
            _check_single_valued(dcid, prop, arcs.get(prop, ()))
        declared_properties = arcs.get("constraintProperties", ())
        constraint_property_names.update(declared_properties)
        partial.append((dcid, arcs, declared_properties))

    pass2_properties = sorted(constraint_property_names)
    pass2: dict[str, dict[str, tuple[str, ...]] | None] = {}
    if pass2_properties:
        pass2 = _fetch_arcs_many(reader, statvar_dcids, properties=pass2_properties)

    tentative: list[tuple[str, dict[str, tuple[str, ...]], dict[str, str]]] = []
    for dcid, arcs, declared_properties in partial:
        values_by_property = pass2.get(dcid) or {}
        constraint_properties = {}
        for prop in declared_properties:
            values = values_by_property.get(prop, ())
            if not values:
                raise ExportRefused(
                    f"{dcid} declares {prop!r} in its constraintProperties but the pass-2 "
                    "forward read returned no value for it -- cannot export a row that "
                    "would silently drop a declared constraint"
                )
            _check_single_valued(dcid, prop, values)
            value = values[0]
            if value == UNRESOLVED_NODE:
                raise ExportRefused(
                    f"{dcid} declares {prop!r} in its constraintProperties but the graph "
                    "could not resolve its value to a dcid -- cannot export a constraint "
                    "value the index would carry as if it were resolved"
                )
            constraint_properties[prop] = value
        tentative.append((dcid, arcs, constraint_properties))

    _reject_provisional_constraint_values(
        reader, [(dcid, constraint_properties) for dcid, _, constraint_properties in tentative]
    )

    rows = [
        {
            "dcid": dcid,
            "population_type": _first_or_none(arcs.get("populationType", ())),
            "measured_property": _single_value(
                dcid, "measuredProperty", arcs.get("measuredProperty", ())
            ),
            "stat_type": _first_or_none(arcs.get("statType", ())),
            "measurement_qualifier": _first_or_none(arcs.get("measurementQualifier", ())),
            "measurement_denominator": _first_or_none(arcs.get("measurementDenominator", ())),
            "observation_properties": sorted(arcs.get("observationProperties", ())),
            "constraint_properties": dict(sorted(constraint_properties.items())),
        }
        for dcid, arcs, constraint_properties in sorted(tentative, key=lambda entry: entry[0])
    ]
    return rows


_MEASURE_AXIS_SHAPE_FIELDS = {
    "populationType": "population_type",
    "measuredProperty": "measured_property",
    "measurementQualifier": "measurement_qualifier",
}
_MEASURE_VALUE_ARC_PROPERTIES = ("name", "alternateName", "shortName", "description")


def _measure_axis_value(row: dict, axis: MeasureAxis) -> str | None:
    if axis.kind == "shape":
        return row[_MEASURE_AXIS_SHAPE_FIELDS[axis.axis]]
    return row["constraint_properties"].get(axis.axis)


def _observed_measure_values(
    statvars: Sequence[dict],
    axes: Sequence[MeasureAxis],
    declared: Mapping[str, Sequence[str]],
) -> dict[str, set[str]]:
    """Return distinct axis values carried by exported StatVar rows, plus `declared` values.

    Filtering to observed values excludes unpopulated `GrossValue` and `NettingBasisEnum.Gross`,
    whose name collides with `GrossValue`. A declared value is exported only on an axis in `axes`.
    """
    observed: dict[str, set[str]] = {axis.axis: set(declared.get(axis.axis, ())) for axis in axes}
    for row in statvars:
        for axis in axes:
            value = _measure_axis_value(row, axis)
            if value is not None:
                observed[axis.axis].add(value)
    return observed


def export_measure_values(
    reader: ExportReader,
    statvars: Sequence[dict],
    axes: Sequence[MeasureAxis],
    declared: Mapping[str, Sequence[str]] = DECLARED_MEASURE_VALUES,
) -> list[dict]:
    """Build rows for observed and declared `(axis, dcid)` pairs with graph labels and authored
    text overrides. A row's `definition` is the value's graph `description`.

    Authored labels replace graph names. Synonyms merge with `alternateName`, and `matchable`
    and `same_reading_as` pass through. A missing `name` raises `ExportRefused` because a
    dcid fallback would enter text matching as a query term.
    """
    observed = _observed_measure_values(statvars, axes, declared)
    all_dcids = sorted({dcid for values in observed.values() for dcid in values})
    arcs_by_dcid = _fetch_arcs_many(reader, all_dcids, properties=_MEASURE_VALUE_ARC_PROPERTIES)

    rows = []
    for axis in axes:
        for dcid in sorted(observed[axis.axis]):
            arcs = arcs_by_dcid.get(dcid)
            if arcs is None:
                raise ExportRefused(
                    f"{dcid} is a {axis.axis!r} measure value an exported StatVar row carries "
                    "or the export declares, but the graph holds no node for it"
                )
            name = _single_value(dcid, "name", arcs.get("name", ()))
            overrides = MEASURE_VALUE_TEXT.get(dcid, MeasureValueText())
            label = overrides.label if overrides.label is not None else strip_system_suffix(name)
            synonyms = sorted(
                {v for v in arcs.get("alternateName", ()) if v and v != label}
                | set(overrides.synonyms)
            )
            rows.append(
                {
                    "dcid": dcid,
                    "axis": axis.axis,
                    "label": label,
                    "synonyms": synonyms,
                    "short_name": _first_or_none(arcs.get("shortName", ())),
                    "matchable": overrides.matchable,
                    "same_reading_as": overrides.same_reading_as,
                    "definition": _first_or_none(arcs.get("description", ())),
                }
            )
    rows.sort(key=lambda row: (row["axis"], row["dcid"]))
    return rows


def _observed_measure_partitions(
    statvars: Sequence[dict], axes: Sequence[MeasureAxis]
) -> set[tuple[tuple[str, str | None], ...]]:
    """Return distinct measure-axis tuples exhibited by exported StatVar rows."""
    return {tuple((axis.axis, _measure_axis_value(row, axis)) for axis in axes) for row in statvars}


def _partition_sort_key(pairs: tuple[tuple[str, str | None], ...]) -> tuple:
    return tuple((axis, value is None, value or "") for axis, value in pairs)


def _verify_measure_partition_labels(
    statvars: Sequence[dict],
    axes: Sequence[MeasureAxis],
    labels: Sequence[MeasurePartitionLabel],
) -> None:
    """Require each observed partition to match exactly one label with `label_for_pairs`.

    This is the runtime rule. Empty axes make validation a no-op.
    """
    if not axes:
        return
    offending: list[tuple[tuple[str, str | None], ...]] = []
    for pairs in sorted(_observed_measure_partitions(statvars, axes), key=_partition_sort_key):
        try:
            match = label_for_pairs(pairs, labels)
        except ValueError:
            offending.append(pairs)
            continue
        if match is None:
            offending.append(pairs)
    if offending:
        raise ExportRefused(
            f"{len(offending)} measure partition(s) carried by the exported StatVar rows "
            f"resolve to no single authored label: {offending!r} -- add or fix entries in "
            "MEASURE_PARTITION_LABELS to cover each"
        )


def _verify_measure_nouns(statvars: Sequence[dict], nouns: Sequence[MeasurePartitionLabel]) -> None:
    """Require that no exported StatVar row matches a measure noun: every row belongs to a
    measure partition, whose label names it, and a noun beside it would put two names on
    one measure."""
    matched = sorted(
        row["dcid"]
        for row in statvars
        if label_for_pairs(
            [
                ("populationType", row["population_type"]),
                ("measuredProperty", row["measured_property"]),
                ("statType", row["stat_type"]),
                *row["constraint_properties"].items(),
            ],
            nouns,
        )
        is not None
    )
    if matched:
        raise ExportRefused(
            f"{len(matched)} exported StatVar row(s) match a measure noun as well as their "
            f"partition label, first {matched[:5]!r} -- narrow the entries in MEASURE_NOUNS"
        )


def _verify_measure_label_named_values(
    statvars: Sequence[dict],
    codes: Sequence[dict],
    axes: Sequence[MeasureAxis],
    labels: Sequence[MeasurePartitionLabel],
) -> None:
    """Require each label's `named_values` pair to be an exported code of its property's
    code system, carried by at least one StatVar row under the label's partition. A pair
    that fails names nothing and silences nothing in a heading. Empty axes make validation
    a no-op, as in `_verify_measure_partition_labels`."""
    if not axes:
        return
    known = {(row["code_system"], row["dcid"]) for row in codes}
    axes_by_name = {axis.axis: axis for axis in axes}
    offending: list[tuple[str, tuple[str, str]]] = []
    for label in labels:
        rows = [
            row
            for row in statvars
            if all(
                axis in axes_by_name and _measure_axis_value(row, axes_by_name[axis]) == value
                for axis, value in label.axis_values
            )
        ]
        offending.extend(
            (label.label, (prop, code))
            for prop, code in label.named_values
            if (prop, code) not in known
            or not any(row["constraint_properties"].get(prop) == code for row in rows)
        )
    if offending:
        raise ExportRefused(
            f"{len(offending)} measure label named value(s) are no exported code carried by "
            f"a StatVar row under the label's partition: {offending!r} -- fix the entries' "
            "named_values in MEASURE_PARTITION_LABELS"
        )


def _measure_axis_to_dict(axis: MeasureAxis) -> dict:
    return {
        "axis": axis.axis,
        "kind": axis.kind,
        "role": axis.role,
        "prefer": list(axis.prefer),
        "label": axis.label,
    }


def _measure_partition_label_to_dict(partition_label: MeasurePartitionLabel) -> dict:
    return {
        "axis_values": [list(pair) for pair in partition_label.axis_values],
        "label": partition_label.label,
        "named_values": [list(pair) for pair in partition_label.named_values],
        "definition": partition_label.definition,
    }


def _stated_word_to_dict(stated: StatedWord) -> dict:
    return {
        "word": stated.word,
        "measured_property": stated.measured_property,
        **({"where": dict(sorted(stated.where.items()))} if stated.where else {}),
    }


def _whole_marker_to_dict(dataset: str, window: WholeWindow, publication: PartyPublication) -> dict:
    """A `PartyRow.whole` entry, its `years` running from the window's first year to the last
    year `publication` gives `dataset`. `years` is left out when the window has no first year."""
    marker: dict = {"dataset": dataset, "covers": sorted(window.covers)}
    if window.first_year is not None:
        marker["years"] = [window.first_year, publication.years[dataset][1]]
    return marker


def _whole_total_set_to_dict(whole_set: WholeTotalSet) -> dict:
    return {"phrase": whole_set.phrase, "axis": whole_set.axis, "members": list(whole_set.members)}


def _party_word_to_dict(named: PartyWord) -> dict:
    return {
        "word": named.word,
        "measured_property": named.measured_property,
        "axis": named.axis,
        "party": named.party,
        **({"where": dict(sorted(named.where.items()))} if named.where else {}),
    }


def _dataset_to_dict(dataset: DatasetDeclaration) -> dict:
    return {
        "provenance": dataset.provenance,
        "label": dataset.label,
        "description": dataset.description,
        "opt_in": dataset.opt_in,
        "absent_observation": dataset.absent_observation,
        "account": dataset.account,
        "rank": dataset.rank,
        "invariants": dict(sorted(dataset.invariants.items())),
        **(
            {"stated_words": [_stated_word_to_dict(w) for w in dataset.stated_words]}
            if dataset.stated_words
            else {}
        ),
        **(
            {"party_words": [_party_word_to_dict(w) for w in dataset.party_words]}
            if dataset.party_words
            else {}
        ),
    }


def verify_complete_systems(
    statvars: Sequence[dict], code_systems: Sequence[CodeSystemConfig]
) -> None:
    """Require every row of a dataset to declare each code system listing that dataset in
    `declared_by_every_statvar_of`. A row that does not raises `ExportRefused`, since the enumeration
    split that dataset's reads on the system and can have dropped rows like it."""
    for config in code_systems:
        complete_in = set(config.declared_by_every_statvar_of)
        for row in statvars:
            if (
                complete_in.intersection(row["provenances"])
                and config.code_system not in row["constraint_properties"]
            ):
                raise ExportRefused(
                    f"{row['dcid']} belongs to {row['provenances']!r} and does not declare "
                    f"{config.code_system!r}, which is configured as declared by every StatVar "
                    f"of {sorted(complete_in)!r}"
                )


def _statvar_shape_counts(statvars: list[dict], code_systems: Sequence[str]) -> dict[str, int]:
    """Count, by code system, the StatVar rows that name it in their
    `constraint_properties`.
    """
    return {
        code_system: sum(1 for row in statvars if code_system in row["constraint_properties"])
        for code_system in code_systems
    }


def export_graph_blocks(
    reader: ExportReader,
    *,
    code_systems: Sequence[CodeSystemConfig] = CODE_SYSTEMS,
    measure_axes: Sequence[MeasureAxis] = MEASURE_AXES,
    measure_partition_labels: Sequence[MeasurePartitionLabel] = MEASURE_PARTITION_LABELS,
    measure_nouns: Sequence[MeasurePartitionLabel] = MEASURE_NOUNS,
    datasets: Sequence[DatasetDeclaration] = DATASETS,
    party_type_kinds: Mapping[str, PartyKind | None] = PARTY_TYPE_KINDS,
    listed: Collection[str] = (),
    shape_slot_defaults: Sequence[ShapeSlotDefault] = SHAPE_SLOT_DEFAULTS,
    partly_stated_values: Sequence[PartlyStatedValue] = PARTLY_STATED_VALUES,
    measure_classes: Sequence[MeasureClassDeclaration] = MEASURE_CLASSES,
    variable_aliases: Mapping[str, str] = VARIABLE_ALIASES,
    stat_type_words: Sequence[StatTypeWordDeclaration] = STAT_TYPE_WORDS,
    whole_total_sets: Sequence[WholeTotalSet] = WHOLE_TOTAL_SETS,
) -> dict:
    """Build every document block the graph and the authored config determine, sorted.

    `listed` names the StatVars of the datasets no code system enumerates (`listed_dcids`),
    read into `listed_statvars` and checked by `_verify_listed_five_tuples`, their series
    codes into `source_codes` (`export_source_codes`), and the parents of their constraint
    values into `value_parents` (`export_value_parents`).

    StatVar rows carry no `provenances` and the result has no stamp, so it does not load as an
    index. `export_index` completes it from a coverage build. The drift test compares it with the
    committed export, listing the same StatVars the committed index lists, so it needs graph
    credentials alone.
    """
    declarations = []
    codes = []
    code_rows_by_system: dict[str, list[dict]] = {}
    for config in code_systems:
        declaration, rows = export_code_system(reader, config)
        declarations.append(declaration)
        codes.extend(rows)
        code_rows_by_system[config.code_system] = rows
    _apply_concept_groups(codes, CONCEPT_GROUPS, code_rows_by_system.keys())
    declarations.sort(key=lambda d: d["code_system"])
    codes.sort(key=lambda row: row["dcid"])
    _validate_declared_defaults(code_systems, code_rows_by_system)

    statvar_dcids, observed_axis_usage = enumerate_statvar_dcids(
        reader, code_rows_by_system, code_systems, datasets
    )
    statvars = export_statvars(reader, statvar_dcids)
    _verify_axis_coverage(statvars, observed_axis_usage)
    _verify_five_tuple_completeness(
        reader,
        statvars,
        code_rows_by_system,
        frozenset(
            _choose_spine(dataset, code_systems, code_rows_by_system).code_system
            for dataset in datasets
        ),
    )
    listed_statvars = export_statvars(reader, sorted(listed)) if listed else []
    _verify_listed_five_tuples(reader, statvars, listed_statvars)
    source_codes = export_source_codes(reader, sorted(listed)) if listed else []
    value_parents = export_value_parents(reader, listed_statvars)
    _verify_measure_partition_labels(statvars, measure_axes, measure_partition_labels)
    _verify_measure_nouns(statvars, measure_nouns)
    _verify_measure_label_named_values(statvars, codes, measure_axes, measure_partition_labels)
    measure_values = export_measure_values(reader, statvars, measure_axes)
    _strip_measure_words(codes, statvars, measure_values, measure_axes)

    return {
        "declarations": declarations,
        "datasets": sorted(
            (_dataset_to_dict(dataset) for dataset in datasets), key=lambda row: row["provenance"]
        ),
        "codes": codes,
        "statvars": statvars,
        "listed_statvars": listed_statvars,
        "source_codes": source_codes,
        "value_parents": value_parents,
        "measure_axes": [_measure_axis_to_dict(axis) for axis in measure_axes],
        "measure_values": measure_values,
        "measure_partition_labels": [
            _measure_partition_label_to_dict(label) for label in measure_partition_labels
        ],
        "measure_nouns": [_measure_partition_label_to_dict(noun) for noun in measure_nouns],
        "party_types": [
            {"type": type_, "kind": kind} for type_, kind in sorted(party_type_kinds.items())
        ],
        "shape_slot_defaults": [_shape_slot_default_to_dict(d) for d in shape_slot_defaults],
        "partly_stated_values": [_partly_stated_value_to_dict(v) for v in partly_stated_values],
        "measure_classes": [_measure_class_to_dict(d) for d in measure_classes],
        "variable_aliases": dict(variable_aliases),
        "stat_type_words": [_stat_type_word_to_dict(d) for d in stat_type_words],
        "whole_total_sets": [_whole_total_set_to_dict(d) for d in whole_total_sets],
    }


def stamp_counts(blocks: Mapping[str, list[dict]]) -> dict[str, dict[str, int]]:
    """Return the stamp's `code_counts` and `statvar_shape_counts` for `blocks`, keyed by each
    declared code system."""
    code_systems = [declaration["code_system"] for declaration in blocks["declarations"]]
    return {
        "code_counts": {
            code_system: sum(1 for row in blocks["codes"] if row["code_system"] == code_system)
            for code_system in code_systems
        },
        "statvar_shape_counts": _statvar_shape_counts(blocks["statvars"], code_systems),
    }


def content_hash(blocks: Mapping[str, list[dict]]) -> str:
    """SHA-256 of canonical JSON for the document's data blocks, which excludes `stamp`."""
    encoded = json.dumps(blocks, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def index_bytes(document: Mapping[str, object]) -> bytes:
    """Serialise `document` as gzipped JSON.

    The loader detects gzip by its magic number. `mtime=0` keeps identical documents byte-identical.
    """
    payload = (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return gzip.compress(payload, compresslevel=9, mtime=0)
