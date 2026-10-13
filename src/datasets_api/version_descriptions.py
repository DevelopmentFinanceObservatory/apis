"""The `VersionDescriber` each version-writing site hands its version write.

A describer builds the version's description record (`datasets_api.description_record`) inside
the transaction that writes the version row, from the version as stored, and settles a record
the version already holds. It reads nothing there. Its site reads the labels of every dcid the
record names before the transaction, in one batched graph read (`read_labels`), so a record
stores each label it could read, dated by the day of the read. A failed read stores none, and
the composer words those refs from today's vocabulary or leaves a gap.

- `mint_describers`: a `spec_` mint's record, with the title its labels compose, and the record
  of a collapse at mint, derived from the raw version's record.
- `collapse_describer`: a collapsed head's record, derived from its base version's record.
- `publish_describer`: an `ops_` publish's record, with the posted step records. On a version
  that already holds a record without steps it fills them once, together with what derives
  from them, replacing the inputs its lineage listed with those its steps retrieve, the
  series facts with those counted under the continuations those inputs carry, and the version
  row's continuations with them, and it refuses steps whose facts differ from the stored
  ones (`step_facts`).

A defect while building a record, here or inside the transaction, never fails the version
write: the failure is logged once at error level (`versions.log_description_build_failure`),
and the version is stored without a record, which the `backfill-descriptions` task fills once
the defect is fixed. A mint or collapse factory then gives no describer. A publish factory gives
one that describes nothing and still refuses posted steps that differ from stored ones, since
comparing them needs no record built; it fills no late steps.
"""

from __future__ import annotations

import datetime
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

from dfo_frame_grammar import Frame
from pydantic import BaseModel, JsonValue
from pydantic.experimental.missing_sentinel import MISSING

from code_index.query import CodeIndex
from code_index.shared import get_measure_index
from dataset_store.clock import utc_now
from dataset_store.snapshot import FrameLineage
from datasets_api.description_record import (
    collapsed_record,
    mint_record,
    named_refs,
    publish_record,
    with_labels,
)
from datasets_api.versions import (
    DescriptionDiverges,
    LateFields,
    VersionRecord,
    log_description_build_failure,
)
from dc_graph import GraphCapExceededError, GraphReadError
from description.compose import version_title
from description.facts import DescriptionRecord
from description.steps import Step
from description.vocabulary import build_vocabulary
from materialise import EnrichmentLookup
from materialise.statistics import SeriesFigures

logger = logging.getLogger(__name__)

# Members that word a fact without stating one, which a republish may change.
_WORDING_MEMBERS = frozenset({"label", "at_mint"})

# A publish's label read has the description route's bound (`doors.description`).
_PUBLISH_LABEL_READ_SECONDS: Final = 10.0


def read_labels(
    enrichment: EnrichmentLookup, refs: Iterable[str], *, deadline: float | None
) -> dict[str, str]:
    """The graph label of each of `refs` that has one, in one batched read bounded by the
    absolute monotonic `deadline`. A failed read, including one past its deadline or its size
    cap, gives no label."""
    requested = sorted(set(refs))
    if not requested:
        return {}
    try:
        fields = enrichment.lookup(requested, deadline=deadline)
    except (GraphReadError, GraphCapExceededError):
        logger.warning(
            "description record label read failed refs=%d", len(requested), exc_info=True
        )
        return {}
    return {ref: label for ref, read in fields.items() if (label := read.get("label"))}


def mint_title(
    record: DescriptionRecord, *, labels: Mapping[str, str], code_index: CodeIndex
) -> str | None:
    """The title the composer gives `record`, which holds none of its own, in the words
    `labels` and the code index hold, or `None` when they cannot word it. It composes the title
    alone, by the rule `describe_version` titles a version by (`version_title`). Only the title
    leaves, and the record stores its labels dated by their read, so no graph generation or
    catalogue build joins the vocabulary's revision."""
    assert record.evidence.title is None, "a record composing its mint title holds none"
    vocabulary = build_vocabulary(
        index=code_index,
        measure_index=get_measure_index(),
        labels=labels,
        sources={},
        currency_units={},
        graph_generation=None,
        catalogue_build=None,
    )
    return version_title(record, vocabulary)


@dataclass(frozen=True)
class MintDescriber:
    """A `spec_` version's record, built from its stored facts and this mint's figures."""

    figures: SeriesFigures
    code_index: CodeIndex
    title: str | None
    labels: Mapping[str, str]
    read: datetime.date

    def describe(
        self, version: VersionRecord, *, base: DescriptionRecord | None
    ) -> DescriptionRecord:
        record = mint_record(
            version, figures=self.figures, code_index=self.code_index, title=self.title
        )
        return with_labels(record, self.labels, read=self.read)

    def settle(self, stored: DescriptionRecord, *, version: VersionRecord) -> LateFields | None:
        return None


@dataclass(frozen=True)
class CollapseDescriber:
    """A multi-contributor collapse's record, derived from its base version's stored record.
    A base that holds no record leaves the collapse without one."""

    figures: SeriesFigures
    labels: Mapping[str, str]
    read: datetime.date

    def describe(
        self, version: VersionRecord, *, base: DescriptionRecord | None
    ) -> DescriptionRecord | None:
        if base is None:
            logger.warning(
                "description record not built: the base holds no record id=%s "
                "content_version=%s base_content_version=%s",
                version.subject_id,
                version.content_version,
                version.base_content_version,
            )
            return None
        record = collapsed_record(base, version, figures=self.figures)
        return with_labels(record, self.labels, read=self.read)

    def settle(self, stored: DescriptionRecord, *, version: VersionRecord) -> LateFields | None:
        return None


@dataclass(frozen=True)
class MintDescribers:
    """The describers of one mint: its version's and, for a collapse, the collapsed one's.
    Both are `None` when building them failed."""

    version: MintDescriber | None
    collapsed: CollapseDescriber | None


def mint_describers(
    version: VersionRecord,
    *,
    figures: SeriesFigures,
    collapsed: tuple[VersionRecord, SeriesFigures] | None,
    code_index: CodeIndex,
    enrichment: EnrichmentLookup,
    deadline: float | None,
) -> MintDescribers:
    """The describers of a mint of `version`, and of the collapse it renders, with the labels
    both records name read in one batch, and the title they compose. A build that fails gives
    neither."""

    def build() -> MintDescribers:
        return _mint_describers(
            version,
            figures=figures,
            collapsed=collapsed,
            code_index=code_index,
            enrichment=enrichment,
            deadline=deadline,
        )

    built = _guarded(version, build)
    return built if built is not None else MintDescribers(version=None, collapsed=None)


def _mint_describers(
    version: VersionRecord,
    *,
    figures: SeriesFigures,
    collapsed: tuple[VersionRecord, SeriesFigures] | None,
    code_index: CodeIndex,
    enrichment: EnrichmentLookup,
    deadline: float | None,
) -> MintDescribers:
    provisional = mint_record(version, figures=figures, code_index=code_index, title=None)
    refs = named_refs(provisional)
    if collapsed is not None:
        collapsed_version, collapsed_figures = collapsed
        refs += named_refs(
            collapsed_record(provisional, collapsed_version, figures=collapsed_figures)
        )
    labels = read_labels(enrichment, refs, deadline=deadline)
    read = utc_now().date()
    title = mint_title(
        with_labels(provisional, labels, read=read), labels=labels, code_index=code_index
    )
    return MintDescribers(
        version=MintDescriber(
            figures=figures, code_index=code_index, title=title, labels=labels, read=read
        ),
        collapsed=None
        if collapsed is None
        else CollapseDescriber(figures=collapsed[1], labels=labels, read=read),
    )


def collapse_describer(
    version: VersionRecord,
    *,
    figures: SeriesFigures,
    code_index: CodeIndex,
    enrichment: EnrichmentLookup,
    deadline: float | None,
) -> CollapseDescriber | None:
    """The describer of a collapsed head's `version`, with the labels of what it sums, or
    `None` when building it failed. The version's facts hold its base's trace, which stands in
    for the base's record here."""

    def build() -> CollapseDescriber:
        base = mint_record(version, figures=figures, code_index=code_index, title=None)
        refs = named_refs(collapsed_record(base, version, figures=figures))
        return CollapseDescriber(
            figures=figures,
            labels=read_labels(enrichment, refs, deadline=deadline),
            read=utc_now().date(),
        )

    return _guarded(version, build)


@dataclass(frozen=True)
class PublishDescriber:
    """An `ops_` version's record, built from this publish's frame and posted steps.

    It describes and settles only a stored version holding this publish's frame and lineage,
    which the door verifies apart. A version without a record gets one, unless `describes` is
    false because building it failed. A record holding `steps: null` takes the posted steps
    once, with the caveats, label evidence and inputs a first publish with them records, the
    inputs replacing those the lineage listed, and with `continuations`, those the publish
    derived from the versions the steps retrieve, which replace the version row's and the
    record's series facts counted under the earlier ones; a
    record holding steps refuses posted steps whose facts differ. A publish without steps
    settles nothing. `built` is the posted lineage and the record built from it before the
    write, which a stored version holding that lineage takes as it is; one holding another
    lineage, as a stored version holding composed variables does (the door posts none), gets
    its own.
    """

    steps: tuple[Step, ...] | None
    frame: Frame
    frame_sha256: str
    input_content_versions: tuple[str, ...]
    computed_at: datetime.datetime
    labels: Mapping[str, str]
    describes: bool = True
    built: tuple[FrameLineage, DescriptionRecord] | None = None
    continuations: tuple[Mapping[str, object], ...] = ()

    def describe(
        self, version: VersionRecord, *, base: DescriptionRecord | None
    ) -> DescriptionRecord | None:
        if not self.describes or not self._holds_this_frame(version):
            return None
        if self.built is not None and self.built[0] == version.facts:
            record = self.built[1]
        else:
            record = publish_record(
                version, steps=self.steps, frame=self.frame, computed_at=self.computed_at
            )
        return with_labels(record, self.labels, read=self.computed_at.date())

    def settle(self, stored: DescriptionRecord, *, version: VersionRecord) -> LateFields | None:
        if self.steps is None or not self._holds_this_frame(version):
            return None
        if stored.steps is None:
            lineage = version.facts
            assert isinstance(lineage, FrameLineage)
            described = self.describe(
                replace(version, facts=replace(lineage, continuations=list(self.continuations))),
                base=None,
            )
            if described is None:
                return None
            return _late_steps(described, continuations=self.continuations)
        if step_facts(stored.steps) != step_facts(self.steps):
            raise DescriptionDiverges(
                f"id {stored.id!r} content_version {stored.content_version!r} holds step "
                "records whose inputs, operations, policies, parameters, units or counts differ "
                "from this publish call's steps",
                pointer="/steps",
            )
        return None

    def _holds_this_frame(self, version: VersionRecord) -> bool:
        lineage = version.facts
        return (
            isinstance(lineage, FrameLineage)
            and tuple(lineage.input_content_versions) == self.input_content_versions
            and version.frame_sha256 == self.frame_sha256
        )


def publish_describer(
    version: VersionRecord,
    *,
    steps: Sequence[Step] | None,
    frame: Frame,
    enrichment: EnrichmentLookup | None,
) -> PublishDescriber:
    """The describer of a publish of `version`, computed now, with the labels its record
    names, read within `_PUBLISH_LABEL_READ_SECONDS`. `version` holds this publish's lineage
    and frame hash. `enrichment` `None` reads no label. When building the record fails, the
    describer describes nothing and still settles a stored record."""
    assert version.frame_sha256 is not None
    lineage = version.facts
    assert isinstance(lineage, FrameLineage)
    computed_at = utc_now()
    posted = None if steps is None else tuple(steps)
    describer = PublishDescriber(
        steps=posted,
        frame=frame,
        frame_sha256=version.frame_sha256,
        input_content_versions=tuple(lineage.input_content_versions),
        computed_at=computed_at,
        labels={},
        continuations=tuple(lineage.continuations or ()),
    )
    if enrichment is None:
        return describer

    def labelled() -> PublishDescriber:
        record = publish_record(version, steps=posted, frame=frame, computed_at=computed_at)
        deadline = time.monotonic() + _PUBLISH_LABEL_READ_SECONDS
        labels = read_labels(enrichment, named_refs(record), deadline=deadline)
        return replace(describer, labels=labels, built=(lineage, record))

    built = _guarded(version, labelled)
    return built if built is not None else replace(describer, describes=False)


def _late_steps(
    record: DescriptionRecord, *, continuations: Sequence[Mapping[str, object]]
) -> LateFields:
    """The members a late `steps` fills from `record`, a publish's record with steps: the
    steps, the caveats and label evidence that derive from them, and the inputs they retrieve,
    which replace the lineage's inputs with unknown ids in the same write, as the series facts
    counted under `continuations`, those the retrieved inputs carry, replace the record's and
    `continuations` the version row's."""
    assert record.computed is not None, "a published record holds what it was computed from"
    return LateFields(
        values={
            "steps": _json(record.steps),
            "caveats": _json(record.caveats),
            "evidence.labels": _json(record.evidence.labels),
        },
        derived={
            "computed.inputs": _json(record.computed.inputs),
            "series": record.series.model_dump(mode="json"),
        },
        basis="publish",
        continuations=list(continuations),
    )


def _json(items: Sequence[BaseModel] | None) -> JsonValue:
    return None if items is None else [item.model_dump(mode="json") for item in items]


def _guarded[T](version: VersionRecord, build: Callable[[], T]) -> T | None:
    """`build()`, or `None` after logging its failure: a defect in a record's builder leaves
    the version without a record and never fails its write."""
    try:
        return build()
    except Exception:
        log_description_build_failure(version)
        return None


def step_facts(steps: Sequence[Step]) -> list[object]:
    """The facts `steps` state, without the labels that word them, for comparing two lists."""
    return [_facts(step) for step in steps]


def _facts(value: object) -> object:
    if isinstance(value, BaseModel):
        return {
            name: _facts(member)
            for name in type(value).model_fields
            if name not in _WORDING_MEMBERS and (member := getattr(value, name)) is not MISSING
        }
    if isinstance(value, list):
        return [_facts(item) for item in value]
    if isinstance(value, dict):
        return {key: _facts(item) for key, item in value.items()}
    return value
