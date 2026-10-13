"""Builds provenance sources, citations, and warnings for minted components.

Data components cite bound-series sources. `source_card` cites all frame sources.
`execution_ref` names the replay key of a derived frame's recipe, and is `None` for a dataset
frame and a note narration. A component with a description, and a note narration, keep lineage
alone (`lineage_only`)."""

from __future__ import annotations

import datetime
import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from assembler.bind import BoundSeries
from assembler.limits import LABEL_MAX
from assembler.ports import NamedSeries, ResolvedFrame
from description.citation import citation_line, figures_processed
from description.compose import change_step
from description.steps import Step

DESCRIPTION_PROVENANCE_MEMBERS = ("sources", "freshness", "warnings", "citation")
"""The provenance members a lineage-only provenance leaves out, since a description carries
what they hold."""

_LICENCE_MAX = 200
_REF_MAX = 200  # the envelope's `sources[].ref` maximum
_CITATION_MAX = 500
_WARNING_DETAIL_MAX = 500
_WARNINGS_MAX = 50  # the envelope's `warnings` maximum
_SOURCE_ID_RE = re.compile(r"^src_[a-z0-9_]{1,60}$")
_UNATTRIBUTED_SOURCE_ID = "src_unattributed"
_UNATTRIBUTED_SOURCE_NAME = "Unattributed source"
_PROVENANCE_COLUMN = "provenance"  # frame-grammar's FACET_IDENTITY_COLUMNS names this axis
# The envelope requires a non-empty `licence`. This states that the graph
# declares no licence for the source; it names no licence of its own.
_LICENCE_NOT_STATED = "Not stated"

# Maps materialiser warning codes to the envelope vocabulary.
# Unknown codes raise. Parity tests cover both catalogues.
_WARNING_KIND_BY_CODE: Mapping[str, str] = {
    # No live revision endpoint confirms the frame's upstream state.
    "revision_unavailable": "provisional",
    # The deployment selected one of several graph facet values.
    "facet_default_selected": "comparability_break",
    # Some rows lack a graph value for the requested enrichment column.
    "enrichment_unavailable": "coverage_gap",
    # The source declares no processing tier, so the value's transformations are unknown.
    "processing_level_undeclared": "provisional",
    # The frame covers less than the requested scope.
    "coverage_below_scope": "coverage_gap",
    # Filtering left no observations.
    "empty_result": "coverage_gap",
    # The conversion uses a projected exchange rate or price deflator.
    "deflator_projected": "estimated",
    # Rows of entities no single currency names were left out of the conversion.
    "currency_unmapped_omitted": "coverage_gap",
    # Rows in years the deflator table lacks were left out of a constant-price conversion.
    "deflator_missing_omitted": "coverage_gap",
    # The groupby sums multiple StatVars into one family total.
    "stat_vars_summed": "comparability_break",
    # Every facet for a series sits under a provenance the origin registry does not admit.
    "withheld_by_registry": "coverage_gap",
    # The origin catalogue that admitted the frame's sources is past its staleness limit.
    "origin_catalogue_stale": "provisional",
    # The UN System Data Commons refused reads for some requested places.
    "un_system_unavailable": "coverage_gap",
}


def _derive_source_id(*, raw_ref: str) -> str:
    """Returns a deterministic `src_` id.

    Refs already shaped as legal source ids keep their ids."""
    if _SOURCE_ID_RE.match(raw_ref):
        return raw_ref
    slug = re.sub(r"[^a-z0-9]+", "_", raw_ref.lower()).strip("_")[:40] or "source"
    digest = hashlib.sha256(raw_ref.encode()).hexdigest()[:8]
    return f"src_{slug}_{digest}"[:60]


def _citation_sources_by_ref(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Keys the manifest's `citation.sources[]` by `ref`.

    The manifest keeps one entry per `(ref, processing_level)` pair. The fields read here do not
    vary by level, so the first entry for a ref stands for all of them."""
    citation = cast(Mapping[str, object], manifest.get("citation") or {})
    entries = cast(Sequence[Mapping[str, object]], citation.get("sources") or ())
    by_ref: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        by_ref.setdefault(cast(str, entry["ref"]), entry)
    return by_ref


def _build_sources_for_refs(
    *, raw_refs: Sequence[str | None], resolved_frame: ResolvedFrame
) -> tuple[tuple[dict[str, object], ...], dict[str | None, str]]:
    """Builds `sources[]` and maps each raw ref to its minted source id.

    `name`, `url` and `licence` come from the manifest's citation entry for
    the ref (`_citation_sources_by_ref`). Without one, `name` falls back to
    the raw ref, `url` to `""` and `licence` to `_LICENCE_NOT_STATED`, since
    the envelope requires all three. `retrieved` is the frame's `retrieved_on`.
    `ref` is the raw ref, the provenance dcid a renderer looks the source up by. It is left
    out for the unattributed source and for a raw ref over `_REF_MAX`, since a truncated
    dcid would name another node.
    `source` and `data_commons` carry the citation entry's members of the
    same names as `{id, name}`, and are left out where the entry has none, as
    on a dataset version minted before the origin catalogue.
    """
    citation_by_ref = _citation_sources_by_ref(resolved_frame.manifest)
    retrieved = resolved_frame.retrieved_on.isoformat()

    id_by_raw_ref: dict[str | None, str] = {}
    sources: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for raw_ref in raw_refs:
        source_id = (
            _derive_source_id(raw_ref=raw_ref) if raw_ref is not None else _UNATTRIBUTED_SOURCE_ID
        )
        id_by_raw_ref[raw_ref] = source_id
        if source_id in seen_ids:
            continue
        seen_ids.add(source_id)
        cited = citation_by_ref.get(raw_ref, {}) if raw_ref is not None else {}
        source: dict[str, object] = {
            "id": source_id,
            "name": str(cited.get("label") or raw_ref or _UNATTRIBUTED_SOURCE_NAME)[:LABEL_MAX],
            "url": str(cited.get("url") or ""),
            "licence": str(cited.get("license") or _LICENCE_NOT_STATED)[:_LICENCE_MAX],
            "retrieved": retrieved,
        }
        if raw_ref is not None and len(raw_ref) <= _REF_MAX:
            source["ref"] = raw_ref
        for member, key in (("source", "ref"), ("data_commons", "id")):
            named = cast("Mapping[str, str] | None", cited.get(member))
            if named is not None:
                source[member] = {
                    "id": named[key][:LABEL_MAX],
                    "name": named["label"][:LABEL_MAX],
                }
        sources.append(source)
    return tuple(sources), id_by_raw_ref


def _build_sources(
    *, bound_series: Sequence[BoundSeries], resolved_frame: ResolvedFrame
) -> tuple[tuple[dict[str, object], ...], dict[str | None, str]]:
    """`sources[]` for the raw refs a data component's own bound series
    reference, deduplicated in first-seen order."""
    raw_refs = list(dict.fromkeys(one.source_ref for one in bound_series))
    return _build_sources_for_refs(raw_refs=raw_refs, resolved_frame=resolved_frame)


def _build_all_sources(
    *, resolved_frame: ResolvedFrame
) -> tuple[tuple[dict[str, object], ...], dict[str | None, str]]:
    """Builds `source_card` sources from distinct row refs in first-seen order.

    A ref with no citation entry is handled by `_build_sources_for_refs`."""
    columns = cast("Sequence[str]", resolved_frame.frame["columns"])
    if _PROVENANCE_COLUMN not in columns:
        return (), {}
    index = columns.index(_PROVENANCE_COLUMN)
    rows = cast("Sequence[Sequence[object]]", resolved_frame.frame["rows"])
    raw_refs = list(dict.fromkeys(cast("str | None", row[index]) for row in rows))
    return _build_sources_for_refs(raw_refs=raw_refs, resolved_frame=resolved_frame)


def _build_citation(
    *, sources: Sequence[Mapping[str, object]], retrieved: datetime.date, processed: bool
) -> str:
    """The envelope's citation: `description.citation.citation_line` over the sources' names,
    held to `_CITATION_MAX` characters."""
    names = [cast(str, one["name"]) for one in sources]
    return citation_line(retrieved=retrieved, sources=names, processed=processed)[:_CITATION_MAX]


def _translate_warnings(warnings: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Maps manifest warnings to the envelope's closed `{kind, detail}` vocabulary.

    Drops `severity` and raises `AssertionError` for an unmapped code."""
    translated: list[dict[str, object]] = []
    for warning in warnings:
        code = cast(str, warning["code"])
        if code not in _WARNING_KIND_BY_CODE:
            raise AssertionError(
                f"manifest warning code {code!r} has no entry in mint's own "
                "envelope-kind translation table (_WARNING_KIND_BY_CODE)"
            )
        detail = str(warning["message"])[:_WARNING_DETAIL_MAX]
        translated.append({"kind": _WARNING_KIND_BY_CODE[code], "detail": detail})
    return translated


@dataclass(frozen=True)
class LeftOut:
    """The series a rolling component leaves out, one per display label, and the period they
    hold no value at."""

    period: str
    series: tuple[NamedSeries, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """The display labels of the series, in their order."""
        return tuple(one.label for one in self.series)


def _left_out_warning(*, period: str, names: Sequence[str]) -> dict[str, object]:
    """The `coverage_gap` warning naming the series a rolling component leaves out at
    `period`: the count, the period, then the names in order, joined by `", "`.

    When every name would take the detail past `_WARNING_DETAIL_MAX`, the names that do not
    fit are replaced by `" and N more"`."""
    count = len(names)
    lead = f"{count} series {'has' if count == 1 else 'have'} no {period} value: "
    labels = [name[:LABEL_MAX] for name in names]
    for shown in range(count, 0, -1):
        hidden = count - shown
        detail = lead + ", ".join(labels[:shown]) + (f" and {hidden} more" if hidden else "")
        if len(detail) <= _WARNING_DETAIL_MAX:
            break
    return {"kind": "coverage_gap", "detail": detail[:_WARNING_DETAIL_MAX]}


def _with_left_out_warning(
    *, provenance: Mapping[str, object], period: str, names: Sequence[str]
) -> dict[str, object]:
    """`provenance` with the left-out series warning first in `warnings`, ahead of the
    frame manifest's own in their order, and the list held to `_WARNINGS_MAX` entries."""
    warnings = cast("Sequence[Mapping[str, object]]", provenance["warnings"])
    return {
        **provenance,
        "warnings": [_left_out_warning(period=period, names=names), *warnings][:_WARNINGS_MAX],
    }


def _data_through(manifest: Mapping[str, object]) -> str | None:
    """The last year of the applied window, or `None` when the manifest records no window."""
    coverage = cast(Mapping[str, object], manifest.get("coverage") or {})
    window = cast(Mapping[str, object], coverage.get("window") or {})
    end_year = window.get("end_year")
    return str(end_year) if end_year is not None else None


def card_derivation_steps(component: Mapping[str, object]) -> list[Step]:
    """The steps a card takes after drawing its selection: a stat card's `change` between its
    two basis periods (`description.compose.change_step`), else none."""
    change = cast("Mapping[str, object] | None", component.get("change"))
    if change is None:
        return []
    return [
        change_step(
            derivation=cast(str, change["derivation"]),
            basis=cast("Sequence[str]", change["basis"]),
        )
    ]


def _build_provenance(
    *,
    resolved_frame: ResolvedFrame,
    sources: tuple[dict[str, object], ...],
    card_steps: Sequence[Step],
    clock: Callable[[], datetime.datetime],
) -> dict[str, object]:
    """The provenance of a card drawn from `resolved_frame`, citing `sources`. The citation
    routes through the DFO as the steps known to have made the card's figures decide
    (`description.citation.figures_processed`): the version's, then the card's own
    `card_steps`."""
    manifest = resolved_frame.manifest
    record = resolved_frame.description_record
    warnings = cast(Sequence[Mapping[str, object]], manifest.get("warnings") or ())
    return {
        "inputs": [
            {
                "spec_id": resolved_frame.frame_ref.spec_id,
                "content_version": resolved_frame.frame_ref.content_version,
            }
        ],
        "execution_ref": resolved_frame.replay_key,
        "sources": list(sources),
        "freshness": {
            "data_through": _data_through(manifest),
            "last_refreshed": None,
            "next_expected": None,
            "partial_period": False,
        },
        "warnings": _translate_warnings(warnings),
        "generated_at": clock().astimezone(datetime.UTC).isoformat().replace("+00:00", "Z"),
        "citation": _build_citation(
            sources=sources,
            retrieved=resolved_frame.retrieved_on,
            processed=figures_processed(
                version_id=resolved_frame.frame_ref.spec_id,
                steps=[*(() if record is None else record.known_steps), *card_steps],
            ),
        ),
        "refreshed_from": None,
    }


def _refreshed_provenance(
    *, provenance: Mapping[str, object], refreshed_from: int
) -> dict[str, object]:
    """`provenance` with `refreshed_from` naming the stored revision whose intent was rendered
    on the frame in `inputs`. Create and revise leave `refreshed_from` at `None`."""
    return {**provenance, "refreshed_from": refreshed_from}


def lineage_only(provenance: Mapping[str, object]) -> dict[str, object]:
    """`provenance` without `DESCRIPTION_PROVENANCE_MEMBERS`."""
    return {
        member: value
        for member, value in provenance.items()
        if member not in DESCRIPTION_PROVENANCE_MEMBERS
    }


def _build_frameless_provenance(*, clock: Callable[[], datetime.datetime]) -> dict[str, object]:
    """Builds `note_narration` provenance: lineage alone, with no inputs, and the timestamp."""
    return {
        "inputs": [],
        "execution_ref": None,
        "generated_at": clock().astimezone(datetime.UTC).isoformat().replace("+00:00", "Z"),
        "refreshed_from": None,
    }
