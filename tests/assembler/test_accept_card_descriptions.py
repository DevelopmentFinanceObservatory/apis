"""Acceptance cases for a card's description, written and rendered by the assembler.

A card's description is composed in `_finish_candidate` from the record of the version the
card draws, which the frame store attaches as `ResolvedFrame.description_record`, the card's
selection facts and today's vocabulary, which `write_component` and `render_component` read
from an injected `VocabularySource`. `stamp_envelope` places it in the envelope.

- Create and revise store the description and its vocabulary revision in the card payload.
  An issued revision keeps its payload byte for byte, whatever the vocabulary or the version's
  record does later.
- A revise is a no-op when its envelope equals the head's in everything a reader sees. One
  that gains or rewords a description, or changes a source, citation or warning, writes a
  revision.
- A create over a version with no record issues the card as before descriptions, and the
  card gains one at its next revision or refresh.
- A live-share render composes per request, stores nothing, and marks its description
  `rendering: "live"` with the frame it rendered on.
- The envelope size bound counts `description`.
- The description matches what the card draws. A stat card's change is a card step. A
  source card takes its source's document. A note narration carries none.
- Under lineage-only `provenance` (U2, P26) a new card's `provenance` holds no `sources`,
  `freshness`, `warnings` or `citation`, and each `source_ref` names a
  `description.sources[].ref`.

These cases run against the catalogue bundle of the kit checkout `DFO_SPEC_KIT_DIR` names.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from collections.abc import Iterator, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any, cast

import pytest

from assembler.errors import ResourceExhausted
from assembler.pipeline import get_component, render_component, write_component
from assembler.ports import FrameRef, ResolvedFrame, RevisionRecord
from assembler.reply import WriteReply
from assembler.testing import build_resolved_frame, malaria_ethiopia_frame_ref
from assembler.write_token import COMPONENT_WRITE_TOKEN_PREFIX, mint_write_token, write_token_digest
from code_index.shared import get_index, get_measure_index
from description.facts import (
    Builds,
    CodeIndexStamp,
    ComputedFacts,
    DescriptionRecord,
    DimensionFacts,
    DimensionValue,
    Evidence,
    FixedValue,
    InputVersion,
    SeriesFacts,
    SourceEvidence,
    SourceUnitCount,
    Span,
    StatusCounts,
    VariableFacts,
)
from description.steps import Step, VersionRef
from description.stored import clipped
from description.vocabulary import (
    SourceDocumentFacts,
    Vocabulary,
    build_vocabulary,
    vocabulary_revision,
)

from ._pipeline_support import _malaria_intent

_CRS = "provenance/OECDCreditorReportingSystem"
_CRS_NAME = "OECD Creditor Reporting System (CRS)"
_RECIPIENTS = ("country/ETH", "country/GHA", "country/KEN")
_PERIODS = ("2020", "2021", "2022")
_CLOCK = datetime.datetime(2026, 10, 11, 12, 0, tzinfo=datetime.UTC)
_LEGACY = Path(__file__).parent / "fixtures" / "legacy_line_chart_revision.json"
_DOWNLOADED = "2026-10-08"
_NEXT_RELEASE = "2026-12"
_MOVED_PROVENANCE = frozenset({"sources", "freshness", "warnings", "citation"})


def _clock() -> datetime.datetime:
    return _CLOCK


# -- inputs: a frame, its version's record, the vocabulary -----------------------------------


def _frame(
    *,
    frame_ref: FrameRef | None = None,
    recipients: Sequence[str] = _RECIPIENTS,
    moved_by: float = 0.0,
    provenance: str | None = _CRS,
) -> ResolvedFrame:
    """Malaria grant disbursements to three recipients from 2020 to 2022, from the CRS, or
    from no source the rows name when `provenance` is `None`."""
    return build_resolved_frame(
        frame_ref=frame_ref or malaria_ethiopia_frame_ref(),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": recipient,
                "time_period": period,
                "value": 100.0 * (n + 1) + 10.0 * p + moved_by,
                "unit": "CurrencyUSD",
                "measurementMethod": None,
                "provenance": provenance,
            }
            for n, recipient in enumerate(recipients)
            for p, period in enumerate(_PERIODS)
        ],
        labels={"country/ETH": "Ethiopia", "country/GHA": "Ghana", "country/KEN": "Kenya"},
    )


def _record(
    frame: ResolvedFrame, *, source: str = _CRS, series: int = len(_RECIPIENTS)
) -> DescriptionRecord:
    """The record Datasets wrote for `frame`'s version: one variable, its series, one source.
    An `ops_` version's record says it was computed from the malaria version."""
    computed = frame.frame_ref.spec_id.startswith("ops_")
    malaria = malaria_ethiopia_frame_ref()
    return DescriptionRecord(
        id=frame.frame_ref.spec_id,
        content_version=frame.frame_ref.content_version,
        schema_version=2,
        basis="mint",
        index_stamp=None,
        variables=None
        if computed
        else VariableFacts(
            shape=[],
            variable="MalariaGrantDisbursements",
            fixed=[FixedValue(property="financialFlowType", value="Disbursement")],
            dimensions=None,
            admitted_at_mint=1,
            admitted_from="resolved_variables",
            with_data=1,
        ),
        computed=ComputedFacts(
            inputs=[InputVersion(id=malaria.spec_id, content_version=malaria.content_version)],
            composed=None,
        )
        if computed
        else None,
        series=SeriesFacts(
            series=series,
            rows=None,
            periods=None,
            span=Span(start="2015", end="2024"),
            grain="year",
            latest_period="2024",
            series_reaching_latest=None,
            entities_by_role=None,
            by_source_unit=[SourceUnitCount(source=source, unit=None, series=series)],
            status=StatusCounts(estimated=None, provisional=None),
        ),
        steps=[],
        caveats=[],
        evidence=Evidence(
            prepared=datetime.date(2026, 9, 28),
            title="Malaria grant disbursements",
            sources=[
                SourceEvidence(
                    ref=source, name=source, licence=None, downloaded=None, pipeline_steps=None
                )
            ],
            source_release=None,
            labels=[],
            membership=None,
            builds=Builds(
                origin_catalogue=None, code_index=None, coverage=None, method_status=None
            ),
        ),
        late_fields=[],
    )


@cache
def _vocabulary(ethiopia: str = "Ethiopia", crs: str = _CRS_NAME) -> Vocabulary:
    """Today's words, as the vocabulary source gives them."""
    return build_vocabulary(
        index=get_index(),
        measure_index=get_measure_index(),
        labels={
            _CRS: crs,
            "src_oecd_dac": "OECD DAC",
            "country/ETH": ethiopia,
            "country/GHA": "Ghana",
            "country/KEN": "Kenya",
        },
        sources={
            _CRS: SourceDocumentFacts(
                publisher=None, last_downloaded=_DOWNLOADED, next_release=_NEXT_RELEASE
            )
        },
        currency_units={"CurrencyUSD": "USD"},
        graph_generation=None,
        catalogue_build=None,
    )


def _vocabulary_port() -> Any:
    """The fake of the vocabulary port cards read their words through. Every test reaches it
    before building a described frame, so a tree without the port fails on this import."""
    from assembler.fakes import FakeVocabularySource

    return FakeVocabularySource


def _vocabulary_source(vocabulary: Vocabulary) -> Any:
    return _vocabulary_port()(vocabulary=vocabulary)


def _described(
    frame: ResolvedFrame,
    record: DescriptionRecord | None = None,
    *,
    graph_generation: str = "gen-1",
    catalogue_build: str = "build-1",
) -> ResolvedFrame:
    """`frame` as the frame store loads it once its version's record exists, its labels read
    at `graph_generation` under catalogue build `catalogue_build`."""
    _vocabulary_port()
    return dataclasses.replace(
        frame,
        description_record=record or _record(frame),
        graph_generation=graph_generation,
        catalogue_build=catalogue_build,
    )


def _revision(frame: ResolvedFrame) -> str:
    """The vocabulary revision of a card drawn from `frame`."""
    return vocabulary_revision(
        index_stamp=get_index().stamp.content_hash,
        graph_generation=frame.graph_generation,
        catalogue_build=frame.catalogue_build,
    )


def _intent(frame: ResolvedFrame, member: str, **fields: object) -> dict[str, object]:
    ref = frame.frame_ref
    intent: dict[str, object] = {
        "type": member,
        "frame_ref": {"spec_id": ref.spec_id, "content_version": ref.content_version},
    }
    if member not in ("source_card",):
        intent["select"] = {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": "country/ETH",
        }
    return intent | fields


def _write(
    intent: Mapping[str, object],
    *,
    vocabulary: Vocabulary,
    frame_store,
    revision_store,
    release_registry,
    component_id: str | None = None,
    write_token: str | None = None,
) -> WriteReply:
    return write_component(
        intent=intent,
        component_id=component_id,
        write_token=write_token,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        vocabulary_source=_vocabulary_source(vocabulary),
        clock=_clock,
    )


def _bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _stored(revision_store, component_id: str) -> list[bytes]:
    """Every stored revision's payload, oldest first."""
    head = revision_store.read_current(component_id=component_id)
    return [
        _bytes(revision_store.read_revision(component_id=component_id, revision=n).resolved_payload)
        for n in range(1, head.revision + 1)
    ]


def _description(payload: Mapping[str, object]) -> dict[str, Any]:
    return cast("dict[str, Any]", payload["description"])


def _source_refs(component: object) -> Iterator[str]:
    """Every `source_ref` and `source_refs` member at any depth of a component."""
    if isinstance(component, Mapping):
        for key, value in component.items():
            if key == "source_ref":
                yield cast(str, value)
            elif key == "source_refs":
                yield from cast("list[str]", value)
            else:
                yield from _source_refs(value)
    elif isinstance(component, list):
        for item in component:
            yield from _source_refs(item)


# -- case 2: an issued revision keeps its payload ---------------------------------------------


@pytest.mark.parametrize(
    ("words", "revision"),
    [
        pytest.param(
            {"ethiopia": "Federal Democratic Republic of Ethiopia"}, 1, id="no-served-word-moves"
        ),
        pytest.param({"crs": "OECD CRS"}, 2, id="the-source-is-reworded"),
    ],
)
def test_an_issued_revision_keeps_its_payload_through_a_vocabulary_change(
    frame_store, revision_store, release_registry, words, revision
):
    """Today's words and the frame's read move after the card is issued. The stored revision
    and a read of it keep the bytes it was issued with; a live render takes the new words.

    A same-intent revise compares what a reader sees. Where no word the card serves moves, it
    is a no-op, though the vocabulary revision moved with the graph generation. Where the
    source the card names is reworded, it writes a revision with the new words."""
    today, later = _vocabulary(), _vocabulary(**words)
    frame = _described(_frame())
    frame_store.put(resolved_frame=frame)
    created = _write(
        _intent(frame, "line_chart", title="Malaria grants to Ethiopia"),
        vocabulary=today,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    issued = _bytes(created.resolved_payload)

    reread = _described(_frame(), graph_generation="gen-2")
    assert _revision(reread) != _revision(frame)
    frame_store.put(resolved_frame=reread)
    revised = _write(
        {"title": "Malaria grants to Ethiopia"},
        vocabulary=later,
        component_id=created.component_id,
        write_token=created.write_token,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    rendered = render_component(
        component_id=created.component_id,
        revision=1,
        target_frame=frame.frame_ref,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        vocabulary_source=_vocabulary_source(later),
        clock=_clock,
    )

    assert revised.revision == revision
    assert _stored(revision_store, created.component_id)[0] == issued
    if revision == 1:
        assert _bytes(revised.resolved_payload) == issued
        assert _stored(revision_store, created.component_id) == [issued]
    else:
        sources = _description(revised.resolved_payload)["sources"]
        assert [source["label"] for source in sources] == ["OECD CRS"]
    read = get_component(
        component_id=created.component_id, revision=1, revision_store=revision_store
    )
    assert _bytes(cast("Mapping[str, object]", read["resolved_payload"])) == issued
    assert _description(created.resolved_payload)["built_from"]["vocabulary"] == _revision(frame)
    assert _description(rendered.resolved_payload)["built_from"]["vocabulary"] == _revision(reread)


def test_a_card_created_before_its_versions_record_gains_a_description_at_its_next_revision(
    frame_store, revision_store, release_registry
):
    """A create over a version with no record issues the card as before descriptions: the
    legacy provenance members and no `description`. Once the record exists, the next
    content-changing revision carries a description and lineage-only provenance, and a
    refresh renders one."""
    source = _vocabulary_source(_vocabulary())
    bare = _frame()
    frame_store.put(resolved_frame=bare)
    created = write_component(
        intent=_intent(bare, "line_chart", title="Malaria grants to Ethiopia"),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        vocabulary_source=source,
        clock=_clock,
    )

    assert "description" not in created.resolved_payload
    assert _MOVED_PROVENANCE.issubset(cast("dict", created.resolved_payload["provenance"]))

    frame_store.put(resolved_frame=_described(bare))
    revised = write_component(
        intent={"title": "Malaria grants to Ethiopia, 2020 to 2022"},
        component_id=created.component_id,
        write_token=created.write_token,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        vocabulary_source=source,
        clock=_clock,
    )
    rendered = render_component(
        component_id=created.component_id,
        revision=1,
        target_frame=bare.frame_ref,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        vocabulary_source=source,
        clock=_clock,
    )

    assert revised.revision == 2
    assert _description(revised.resolved_payload)["describes"]["revision"] == 2
    assert not _MOVED_PROVENANCE & set(cast("dict", revised.resolved_payload["provenance"]))
    assert _description(rendered.resolved_payload)["rendering"] == "live"


def test_an_issued_revision_keeps_its_payload_when_its_versions_record_arrives_later(
    frame_store, revision_store, release_registry
):
    """The Datasets backfill writes a record for a version first read without one. A card
    issued before it keeps its stored bytes and a read returns them. A revise with the intent
    it was issued with writes a revision that carries the description, and a live render
    describes the card from the record."""
    vocabulary = _vocabulary()
    source = _vocabulary_source(vocabulary)
    bare = _frame()
    frame_store.put(resolved_frame=bare)
    created = write_component(
        intent=_intent(bare, "line_chart", title="Malaria grants to Ethiopia"),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        vocabulary_source=source,
        clock=_clock,
    )
    issued = _bytes(created.resolved_payload)

    frame_store.put(resolved_frame=_described(bare))
    revised = _write(
        {"title": "Malaria grants to Ethiopia"},
        vocabulary=vocabulary,
        component_id=created.component_id,
        write_token=created.write_token,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    rendered = render_component(
        component_id=created.component_id,
        revision=1,
        target_frame=bare.frame_ref,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        vocabulary_source=source,
        clock=_clock,
    )

    assert revised.revision == 2
    assert _description(revised.resolved_payload)["describes"]["revision"] == 2
    assert _stored(revision_store, created.component_id)[0] == issued
    read = get_component(
        component_id=created.component_id, revision=1, revision_store=revision_store
    )
    assert _bytes(cast("Mapping[str, object]", read["resolved_payload"])) == issued
    assert _description(rendered.resolved_payload)["rendering"] == "live"


# -- P11 and the size bound ---------------------------------------------------------------------


_OPS_FRAME = FrameRef(spec_id="ops_" + "c" * 32, content_version="ocv_" + "d" * 32)


_DATASET_CITATION = "src_oecd_dac (2026), via the Development Finance Observatory"
_OPS_CITATION = "src_oecd_dac (2026), processed by the Development Finance Observatory"


@pytest.mark.parametrize(
    ("ops_frame", "head_execution_ref", "replay_key", "recorded", "citation", "revision"),
    [
        pytest.param(None, None, None, False, _DATASET_CITATION, 1, id="dataset-frame"),
        pytest.param(
            _OPS_FRAME,
            None,
            "rpk_" + "a" * 32,
            False,
            _OPS_CITATION,
            1,
            id="ops-frame-head-without-key",
        ),
        pytest.param(
            _OPS_FRAME,
            "rpk_" + "b" * 32,
            "rpk_" + "a" * 32,
            False,
            _OPS_CITATION,
            2,
            id="ops-frame-key-moved",
        ),
        pytest.param(
            None,
            None,
            None,
            True,
            _DATASET_CITATION,
            2,
            id="dataset-frame-whose-version-now-has-a-record",
        ),
        pytest.param(None, None, None, False, None, 2, id="dataset-frame-earlier-citation-line"),
    ],
)
def test_a_revise_of_a_head_issued_before_descriptions(
    frame_store,
    revision_store,
    release_registry,
    release,
    ops_frame: FrameRef | None,
    head_execution_ref: str | None,
    replay_key: str | None,
    recorded: bool,
    citation: str | None,
    revision: int,
):
    """A line chart issued before descriptions, as the assembler stored it then, revised with
    the same intent. On its frame, whose version has no record, the revise writes no revision
    and keeps the head's bytes. Once the version has a record, the revise writes a revision
    that carries the description.

    Over an `ops_` frame the candidate names the frame's replay key as `execution_ref`. A head
    that stores null there still equals it; a head that names another key does not.

    The head's provenance citation is a line a reader sees, so the rows above store the line
    the mint writes (`citation`), and a head keeping the line it was issued with
    (`None`) gains a revision."""
    legacy_text = _LEGACY.read_text()
    frame, _ = _malaria_intent()
    if ops_frame is not None:
        legacy_text = legacy_text.replace(frame.frame_ref.spec_id, ops_frame.spec_id).replace(
            frame.frame_ref.content_version, ops_frame.content_version
        )
        frame = dataclasses.replace(frame, frame_ref=ops_frame, replay_key=replay_key)
    legacy = json.loads(legacy_text)
    legacy["resolved_payload"]["provenance"]["execution_ref"] = head_execution_ref
    if citation is not None:
        legacy["resolved_payload"]["provenance"]["citation"] = citation
    frame_store.put(
        resolved_frame=_described(frame, _record(frame, source="src_oecd_dac", series=1))
        if recorded
        else frame
    )
    component_id = revision_store.allocate_component_id()
    token = mint_write_token(prefix=COMPONENT_WRITE_TOKEN_PREFIX)
    payload = legacy["resolved_payload"] | {
        "component_id": component_id,
        "catalog_version": release.catalog_version,
        "schema_version": release.schema_version,
    }
    with revision_store.begin_write(
        component_id=component_id,
        is_create=True,
        composition_id=None,
        write_token_sha256=write_token_digest(token),
    ) as ctx:
        ctx.append(
            record=RevisionRecord(
                component_id=component_id,
                revision=1,
                stored_intent=legacy["stored_intent"],
                resolved_payload=payload,
                catalog_version=release.catalog_version,
                schema_version=release.schema_version,
                formatter_profile=release.formatter_profile,
                series_coordinates=legacy["series_coordinates"],
            ),
            composition_entry=None,
        )

    revised = _write(
        {"title": legacy["stored_intent"]["title"]},
        vocabulary=_vocabulary(),
        component_id=component_id,
        write_token=token,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    assert revised.revision == revision
    assert ("description" in revised.resolved_payload) is recorded
    if revision == 1:
        assert revised.changed == ()
        assert _stored(revision_store, component_id) == [_bytes(payload)]


_HEALED_CITATION = {
    "sources": [
        {"ref": _CRS, "label": _CRS_NAME, "url": "https://www.oecd.org/crs", "license": "CC BY 4.0"}
    ]
}
_COVERAGE_WARNING = {"code": "coverage_below_scope", "severity": "warn", "message": "Less asked."}


@pytest.mark.parametrize(
    ("manifest", "revision"),
    [
        pytest.param({}, 1, id="control-the-same-manifest"),
        pytest.param({"citation": _HEALED_CITATION}, 2, id="the-citation-heals"),
        pytest.param({"warnings": [_COVERAGE_WARNING]}, 2, id="a-warning-arrives"),
    ],
)
def test_a_revise_over_a_version_without_a_record_writes_a_changed_provenance(
    frame_store, revision_store, release_registry, manifest, revision
):
    """A card issued over a version with no record, while its citation's documentation read
    had failed, names its source by dcid with no url or licence. A same-intent revise on the
    frame read again writes a revision when the citation has healed or the manifest names a
    warning, and is a no-op on the same manifest."""
    bare = _frame()
    frame_store.put(resolved_frame=bare)
    created = _write(
        _intent(bare, "line_chart", title="Malaria grants to Ethiopia"),
        vocabulary=_vocabulary(),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    frame_store.put(
        resolved_frame=dataclasses.replace(bare, manifest={"schema_version": 1, **manifest})
    )

    revised = _write(
        {"title": "Malaria grants to Ethiopia"},
        vocabulary=_vocabulary(),
        component_id=created.component_id,
        write_token=created.write_token,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    assert revised.revision == revision
    provenance = cast("dict[str, Any]", revised.resolved_payload["provenance"])
    assert [source["name"] for source in provenance["sources"]] == [
        _CRS_NAME if "citation" in manifest else _CRS
    ]
    assert [warning["detail"] for warning in provenance["warnings"]] == (
        ["Less asked."] if "warnings" in manifest else []
    )


def _measured(payload: Mapping[str, object]) -> int:
    """The bytes the size bound measures: the component, its provenance and its description,
    in the encoding a response sends them in."""
    return len(
        json.dumps(
            {key: payload[key] for key in ("component", "provenance", "description")},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


@pytest.mark.parametrize(
    ("headroom", "refused"),
    [
        pytest.param(-1, True, id="one-byte-under-the-card-with-its-description-refuses"),
        pytest.param(0, False, id="control-exactly-the-card-with-its-description-writes"),
    ],
)
@pytest.mark.parametrize(
    "computed",
    [
        pytest.param(False, id="dataset-version"),
        pytest.param(True, id="computed-version-whose-rows-name-no-source"),
    ],
)
def test_the_envelope_size_bound_counts_the_description(
    frame_store, revision_store, release_registry, monkeypatch, headroom, refused, computed
):
    if computed:
        bare = _computed_frame()
        frame = _described(bare, _computed_record(bare))
    else:
        frame = _described(_frame())
    frame_store.put(resolved_frame=frame)
    intent = _intent(frame, "line_chart", title="Malaria grants to Ethiopia")
    first = _write(
        intent,
        vocabulary=_vocabulary(),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    monkeypatch.setattr(
        "assembler.pipeline.candidate.MAX_ENVELOPE_BYTES",
        _measured(first.resolved_payload) + headroom,
    )

    def again() -> WriteReply:
        return _write(
            intent,
            vocabulary=_vocabulary(),
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )

    if refused:
        with pytest.raises(ResourceExhausted) as exhausted:
            again()
        assert exhausted.value.limit_name == "MAX_ENVELOPE_BYTES"
    else:
        assert _measured(again().resolved_payload) == _measured(first.resolved_payload)


_OVERSIZE_DIMENSIONS = tuple(f"dim{n:02d}" for n in range(32))
_OVERSIZE_SERIES = 24


def _oversize_code(prop: str, series: int) -> str:
    return f"code/{prop}/{series:02d}"


def _oversize_variable(series: int) -> str:
    return f"OversizeVariable{series:02d}"


@pytest.mark.parametrize(
    ("label_length", "defined", "listed"),
    [
        pytest.param(20, [30, 31], list(range(32)), id="definitions-given-up"),
        pytest.param(500, [], list(range(12, 32)), id="values-given-up"),
    ],
)
def test_a_card_whose_drawn_definitions_overrun_the_envelope_degrades_and_mints(
    frame_store, revision_store, release_registry, label_length, defined, listed
):
    """24 variables drawn across 32 dimensions, each value defined at the schema's 4,096
    characters, would encode to about 3.2 MB, over the 2 MiB envelope bound. The card mints
    under it. Its lists give up their definitions first, keeping their labels, one at a time
    and the largest first, the earlier of equal ones first, so short labels leave the last
    two lists defined. Long labels leave no definition and the lists then give up their
    values in the same order, leaving counts and a gap naming each list given up."""
    from code_index import CodeIndex
    from code_index.types import StatVarRow
    from description.blocks import DRAWN_VALUES_BUDGET_BYTES

    variables = [_oversize_variable(n) for n in range(_OVERSIZE_SERIES)]
    rows = [
        {
            "stat_var": f"dcid:{variable}",
            "recipient": "country/ETH",
            "time_period": period,
            "value": 100.0 * n + p,
            "unit": "CurrencyUSD",
            "measurementMethod": None,
            "provenance": _CRS,
        }
        for n, variable in enumerate(variables)
        for p, period in enumerate(_PERIODS)
    ]
    frame = build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(), role_keys=("recipient",), rows=rows
    )
    record = _record(frame, series=_OVERSIZE_SERIES)
    assert record.variables is not None
    record = record.model_copy(
        update={
            "index_stamp": CodeIndexStamp(
                content_hash="index-hash", exported_at="2026-10-11T00:00:00Z", graph_endpoint="dev"
            ),
            "variables": record.variables.model_copy(
                update={
                    "dimensions": [
                        DimensionFacts(
                            property=prop,
                            within=None,
                            values=[
                                DimensionValue(value=_oversize_code(prop, n), variables=1)
                                for n in range(_OVERSIZE_SERIES)
                            ],
                            undeclared_variables=0,
                            partial=False,
                        )
                        for prop in _OVERSIZE_DIMENSIONS
                    ]
                }
            ),
        }
    )
    codes = [_oversize_code(prop, n) for prop in _OVERSIZE_DIMENSIONS for n in range(24)]
    base = _vocabulary()
    index = CodeIndex(
        stamp=base.index.stamp,
        declarations=base.index.declarations.values(),
        codes=base.index.codes,
        statvars=base.index.statvars,
        listed_statvars=[
            StatVarRow(
                dcid=variable,
                population_type=None,
                measured_property="amount",
                stat_type=None,
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=(),
                constraint_properties={
                    prop: _oversize_code(prop, n) for prop in _OVERSIZE_DIMENSIONS
                },
            )
            for n, variable in enumerate(variables)
        ],
    )
    vocabulary = build_vocabulary(
        index=index,
        measure_index=base.measure_index,
        labels={**base.labels, **{code: code.ljust(label_length, "x") for code in codes}},
        sources=base.sources,
        currency_units=base.currency_units,
        graph_generation=None,
        catalogue_build=None,
        definitions=dict.fromkeys(codes, "d" * 4096),
    )
    frame_store.put(resolved_frame=_described(frame, record))

    created = _write(
        _intent(frame, "table", select={"stat_var": [f"dcid:{v}" for v in variables]}),
        vocabulary=vocabulary,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    description = _description(created.resolved_payload)
    dimensions = description["about"]["drawn"]["dimensions"]
    values = [dimension["values"] for dimension in dimensions if "values" in dimension]
    assert _measured(created.resolved_payload) < 2 * 1024 * 1024
    assert len(json.dumps(values, separators=(",", ":")).encode()) <= DRAWN_VALUES_BUDGET_BYTES
    assert [dimension["count"] for dimension in dimensions] == [_OVERSIZE_SERIES] * 32
    assert [n for n, dimension in enumerate(dimensions) if "values" in dimension] == listed
    assert [
        n
        for n, dimension in enumerate(dimensions)
        if any("definition" in value for value in dimension.get("values", ()))
    ] == defined
    assert [gap for gap in description["gaps"] if gap.startswith("about.drawn.dimensions")] == [
        *(f"about.drawn.dimensions.{n}.property.label" for n in range(32)),
        *(f"about.drawn.dimensions.{n}.values" for n in range(32) if n not in listed),
    ]


# -- where a description is stored and where it is composed per request ------------------------


def test_create_and_revise_store_a_description_with_its_vocabulary_revision(
    frame_store, revision_store, release_registry
):
    vocabulary = _vocabulary()
    frame = _described(_frame())
    frame_store.put(resolved_frame=frame)
    created = _write(
        _intent(frame, "line_chart", title="Malaria grants to Ethiopia"),
        vocabulary=vocabulary,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    revised = _write(
        {"title": "Malaria grants to Ethiopia, 2020 to 2022"},
        vocabulary=vocabulary,
        component_id=created.component_id,
        write_token=created.write_token,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    for reply, revision in ((created, 1), (revised, 2)):
        stored = revision_store.read_revision(component_id=created.component_id, revision=revision)
        assert _bytes(stored.resolved_payload) == _bytes(reply.resolved_payload)
        description = _description(stored.resolved_payload)
        assert description["describes"] == {
            "kind": "card_revision",
            "component_id": created.component_id,
            "revision": revision,
        }
        assert description["built_from"]["vocabulary"] == _revision(frame)
        assert "rendering" not in description
        assert "frame" not in description


def test_a_live_render_composes_its_description_and_stores_nothing(
    frame_store, revision_store, release_registry
):
    """A refresh renders revision 1 on a later frame of the same dataset. Its description
    carries `rendering: "live"`, that frame, and the vocabulary revision of that frame's read,
    and the revision store holds exactly what it held before."""
    issued_with = _vocabulary()
    frame = _described(_frame())
    frame_store.put(resolved_frame=frame)
    created = _write(
        _intent(frame, "line_chart", title="Malaria grants to Ethiopia"),
        vocabulary=issued_with,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    before = _stored(revision_store, created.component_id)
    target = FrameRef(
        spec_id=frame.frame_ref.spec_id, content_version="cv_0123456789abcdef0123456789abcdef"
    )
    target_frame = _described(_frame(frame_ref=target, moved_by=5.0), graph_generation="gen-2")
    frame_store.put(resolved_frame=target_frame)

    rendered = render_component(
        component_id=created.component_id,
        revision=1,
        target_frame=target,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        vocabulary_source=_vocabulary_source(issued_with),
        clock=_clock,
    )

    description = _description(rendered.resolved_payload)
    assert description["rendering"] == "live"
    assert description["frame"] == {"id": target.spec_id, "content_version": target.content_version}
    assert description["built_from"]["vocabulary"] == _revision(target_frame)
    assert description["describes"]["revision"] == 1
    assert _stored(revision_store, created.component_id) == before


@pytest.mark.parametrize(
    ("graph_generation", "catalogue_build", "moves"),
    [
        pytest.param("gen-1", "build-1", False, id="neither-moves"),
        pytest.param("gen-2", "build-1", True, id="graph-generation-moves"),
        pytest.param("gen-1", "build-2", True, id="catalogue-build-moves"),
    ],
)
def test_a_cards_vocabulary_revision_moves_with_its_frames_graph_generation_and_catalogue_build(
    frame_store, revision_store, release_registry, graph_generation, catalogue_build, moves
):
    """A render over the card's frame read again names that read's generation and build."""
    frame = _described(_frame())
    frame_store.put(resolved_frame=frame)
    created = _write(
        _intent(frame, "line_chart", title="Malaria grants to Ethiopia"),
        vocabulary=_vocabulary(),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    frame_store.put(
        resolved_frame=_described(
            _frame(), graph_generation=graph_generation, catalogue_build=catalogue_build
        )
    )

    rendered = render_component(
        component_id=created.component_id,
        revision=1,
        target_frame=frame.frame_ref,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        vocabulary_source=_vocabulary_source(_vocabulary()),
        clock=_clock,
    )

    issued = _description(created.resolved_payload)["built_from"]["vocabulary"]
    live = _description(rendered.resolved_payload)["built_from"]["vocabulary"]
    assert live.startswith("voc_")
    assert (live != issued) is moves


# -- journey 1: a card's description matches what it draws -------------------------------------


def _create(member: str, frame_store, revision_store, release_registry, **fields: object):
    frame = _described(_frame())
    frame_store.put(resolved_frame=frame)
    return _write(
        _intent(frame, member, **fields),
        vocabulary=_vocabulary(),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )


def _step(description: Mapping[str, Any], op: str) -> tuple[int, dict[str, Any]]:
    steps = cast("list[dict[str, Any]]", description["steps"])
    (found,) = [(n, step) for n, step in enumerate(steps) if step["op"] == op]
    return found


@pytest.mark.parametrize(
    ("member", "select", "drawn"),
    [
        pytest.param(
            "line_chart",
            {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/ETH"},
            ["country/ETH"],
            id="line-chart-of-one-recipient",
        ),
        pytest.param(
            "table",
            {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": list(_RECIPIENTS)},
            sorted(_RECIPIENTS),
            id="table-of-three-recipients",
        ),
    ],
)
def test_a_cards_description_counts_the_series_it_draws(
    frame_store, revision_store, release_registry, member, select, drawn
):
    """The frame holds three recipients. The description's selection names the recipients
    the card draws, its series count is theirs, and its sources count only them."""
    created = _create(
        member,
        frame_store,
        revision_store,
        release_registry,
        select=select,
        window={"start": "2020", "end": "2022"},
    )
    description = _description(created.resolved_payload)

    _, select_step = _step(description, "select")
    assert select_step["params"]["coordinates"]["recipient"] == drawn
    assert select_step["params"]["window"] == {"start": "2020", "end": "2022"}
    assert select_step["kept"] == {"series": len(drawn)}
    assert description["about"]["drawn"]["series"] == len(drawn)
    assert [(source["ref"], source["series"]) for source in description["sources"]] == [
        (_CRS, len(drawn))
    ]


_MALARIA_CONTROL = "CRSPurposeCode12262"


def test_a_cards_description_names_its_versions_measure_and_fixed_values_and_what_it_draws(
    frame_store, revision_store, release_registry
):
    """The card names its version's measure, here its one variable, and the version's fixed
    values, each with the fixture code index's definition, and the recipients it draws in the
    order it draws them. A variable's name need not state its flow type, so the flow type
    stays among the fixed values."""
    frame = _frame()
    record = _record(frame)
    assert record.variables is not None
    record = record.model_copy(
        update={
            "index_stamp": CodeIndexStamp(
                content_hash="index-hash", exported_at="2026-10-11T00:00:00Z", graph_endpoint="dev"
            ),
            "variables": record.variables.model_copy(
                update={
                    "fixed": [
                        FixedValue(property="financialFlowType", value="Disbursement"),
                        FixedValue(property="crsPurpose", value=_MALARIA_CONTROL),
                    ],
                    "dimensions": [
                        DimensionFacts(
                            property="recipient",
                            within=None,
                            values=[
                                DimensionValue(value=recipient, variables=1)
                                for recipient in _RECIPIENTS
                            ],
                            undeclared_variables=0,
                            partial=False,
                        )
                    ],
                }
            ),
        }
    )
    frame_store.put(resolved_frame=_described(frame, record))
    created = _write(
        _intent(
            frame,
            "table",
            select={
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": ["country/KEN", "country/ETH"],
            },
        ),
        vocabulary=_vocabulary(),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    about = _description(created.resolved_payload)["about"]
    malaria = get_index().code_by_dcid(_MALARIA_CONTROL)
    assert malaria is not None and malaria.definition
    disbursement = _vocabulary().definition("Disbursement")
    assert disbursement is not None

    assert about["measure"] == {
        "ref": "MalariaGrantDisbursements",
        "label": "MalariaGrantDisbursements",
        "definition": None,
    }
    assert about["fixed"] == [
        {
            "property": {"ref": "financialFlowType", "label": "flow type"},
            "value": {"ref": "Disbursement", "label": "disbursement", "definition": disbursement},
        },
        {
            "property": {"ref": "crsPurpose", "label": "sector"},
            "value": {
                "ref": _MALARIA_CONTROL,
                "label": malaria.label,
                "definition": malaria.definition,
            },
        },
    ]
    (recipients,) = about["drawn"]["dimensions"]
    assert recipients["values"] == [
        {"ref": "country/ETH", "label": "Ethiopia"},
        {"ref": "country/KEN", "label": "Kenya"},
    ]


_WDI_GDP = "worldBank/NY_GDP_MKTP_CD"
_GDP_DEFINITION = (
    "GDP at purchaser's prices is the sum of gross value added by all resident producers in the "
    "economy."
)


@pytest.mark.parametrize(
    ("descriptions", "definition"),
    [
        pytest.param(
            {_WDI_GDP: f"{_GDP_DEFINITION} (WorldBank Series Code: NY.GDP.MKTP.CD)"},
            _GDP_DEFINITION,
            id="frame-read-the-variables-description",
        ),
        pytest.param({}, None, id="control-no-description-read"),
    ],
)
def test_a_card_drawing_one_world_bank_variable_is_defined_by_the_frames_description_read(
    frame_store, revision_store, release_registry, descriptions, definition
):
    """The frame store reads a World Development Indicators variable's description into
    `ResolvedFrame.descriptions`, and a card that draws that variable alone takes it, without
    its series code, as its measure's definition."""
    frame = build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": f"dcid:{_WDI_GDP}",
                "recipient": "country/ETH",
                "time_period": period,
                "value": 1.0e9 + p,
                "unit": "CurrencyUSD",
                "measurementMethod": None,
                "provenance": _CRS,
            }
            for p, period in enumerate(_PERIODS)
        ],
        labels={"country/ETH": "Ethiopia", f"dcid:{_WDI_GDP}": "GDP (current US$)"},
    )
    record = _record(frame)
    assert record.variables is not None
    record = record.model_copy(
        update={
            "variables": record.variables.model_copy(update={"variable": _WDI_GDP, "fixed": []})
        }
    )
    frame_store.put(
        resolved_frame=dataclasses.replace(_described(frame, record), descriptions=descriptions)
    )
    created = _write(
        _intent(
            frame, "line_chart", select={"stat_var": f"dcid:{_WDI_GDP}", "recipient": "country/ETH"}
        ),
        vocabulary=_vocabulary(),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    description = _description(created.resolved_payload)

    assert description["about"]["measure"] == {
        "ref": _WDI_GDP,
        "label": "GDP (current US$)",
        "definition": definition,
    }
    assert ("about.measure.definition" in description["gaps"]) is (definition is None)


def test_a_cards_words_from_its_frame_are_clipped_and_blank_ones_dropped(
    frame_store, revision_store, release_registry
):
    """A source the vocabulary does not name takes the frame's label and document facts, as
    the Datasets description door words them: a label past the served length is clipped, and
    a blank publisher is a gap."""
    other = "provenance/Other"
    bare = _frame(provenance=other)
    frame = dataclasses.replace(
        _described(bare, _record(bare, source=other)),
        labels={**bare.labels, other: "Long " * 200},
        source_documents={other: SourceDocumentFacts(publisher="  ", last_downloaded=" ")},
    )
    frame_store.put(resolved_frame=frame)
    created = _write(
        _intent(frame, "line_chart", title="Malaria grants to Ethiopia"),
        vocabulary=_vocabulary(),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    description = _description(created.resolved_payload)

    (source,) = description["sources"]
    assert source["label"] == clipped("Long " * 200)
    assert source["publisher"] is None
    assert "sources.0.publisher" in description["gaps"]


def test_a_stat_cards_change_is_a_step_after_its_selection(
    frame_store, revision_store, release_registry
):
    created = _create(
        "stat_card",
        frame_store,
        revision_store,
        release_registry,
        change={"basis": ["2020", "2022"], "derivation": "percent_change"},
    )
    change = cast("dict[str, Any]", created.resolved_payload["component"])["change"]
    description = _description(created.resolved_payload)

    at, _ = _step(description, "select")
    following = description["steps"][at + 1]
    assert following["op"] == "change"
    assert following["params"]["derivation"] == change["derivation"]
    assert following["params"]["basis"] == change["basis"]
    assert following["text"].strip()


def test_a_source_card_takes_its_sources_document_as_its_description(
    frame_store, revision_store, release_registry
):
    """A card-revision description composed from the sources' documents: `sources` holds each
    source's document fields, `about` names the sources, `freshness` gives each source's latest
    download and next release, `steps` is empty and `cite` names the card and the sources."""
    created = _create("source_card", frame_store, revision_store, release_registry)
    description = _description(created.resolved_payload)
    component = cast("dict[str, Any]", created.resolved_payload["component"])

    assert description["describes"] == {
        "kind": "card_revision",
        "component_id": created.component_id,
        "revision": 1,
    }
    assert [source["ref"] for source in description["sources"]] == [_CRS]
    assert description["sources"][0]["label"] == _CRS_NAME
    assert description["sources"][0]["documentation"] == f"/api/graph/documentation/{_CRS}"
    assert _CRS_NAME in json.dumps(description["about"])
    freshness = json.dumps(description["freshness"])
    assert _DOWNLOADED in freshness
    assert _NEXT_RELEASE in freshness
    assert description["steps"] == []
    assert description["cite"]["component_id"] == created.component_id
    assert _CRS_NAME in description["cite"]["text"]
    assert set(component["source_refs"]) <= {source["ref"] for source in description["sources"]}


def test_a_live_source_cards_freshness_comes_from_the_frame_stores_citation_read(
    revision_store, release_registry
):
    """Production words: the process-wide `LiveVocabularySource`, which holds no source
    documents, and a frame the `LiveFrameStore` loaded, whose citation read documents its
    source. The source card's freshness gives that source's latest download and next
    release."""
    from presentation_api.vocabulary import LiveVocabularySource
    from tests.presentation_store.test_frame_store import (
        _documenting_graph,
        _store,
        _thin_spec_fixture,
    )

    snapshots, artifacts, frame_ref, _ = _thin_spec_fixture()
    record = _record(_frame(frame_ref=frame_ref), source="src_oecd_dac")
    frame_store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        graph=_documenting_graph(),
        description_reader=_RecordReader(record),
    )

    created = write_component(
        intent=_intent(_frame(frame_ref=frame_ref), "source_card"),
        component_id=None,
        write_token=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        vocabulary_source=LiveVocabularySource(),
        clock=_clock,
    )

    freshness = json.dumps(_description(created.resolved_payload)["freshness"])
    assert _DOWNLOADED in freshness
    assert _NEXT_RELEASE in freshness


class _RecordReader:
    def __init__(self, record: DescriptionRecord) -> None:
        self.record = record

    def read_description(self, *, id: str, content_version: str) -> DescriptionRecord | None:
        return self.record


def test_a_note_narration_carries_no_description(frame_store, revision_store, release_registry):
    created = _write(
        {"type": "note_narration", "text": "Grants rose after 2020."},
        vocabulary=_vocabulary(),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    assert "description" not in created.resolved_payload
    assert not _MOVED_PROVENANCE & set(cast("dict", created.resolved_payload["provenance"]))


@pytest.mark.parametrize(
    ("member", "fields", "names_sources"),
    [
        pytest.param("line_chart", {"title": "Malaria grants to Ethiopia"}, True, id="line-chart"),
        pytest.param("bar_chart", {"title": "Malaria grants to Ethiopia"}, True, id="bar-chart"),
        pytest.param("table", {"title": "Malaria grants to Ethiopia"}, True, id="table"),
        pytest.param("stat_card", {}, False, id="stat-card-names-no-source"),
        pytest.param("source_card", {}, True, id="source-card"),
    ],
)
def test_a_new_cards_provenance_is_lineage_only_and_its_source_refs_name_description_sources(
    frame_store, revision_store, release_registry, member, fields, names_sources
):
    """A described card holds lineage alone in `provenance`. A stat card's component names no
    source, as its schema has no `source_ref`; every other card names each source by a
    `description.sources[].ref`."""
    created = _create(member, frame_store, revision_store, release_registry, **fields)
    payload = created.resolved_payload

    assert not _MOVED_PROVENANCE & set(cast("dict", payload["provenance"]))
    named = {source["ref"] for source in _description(payload)["sources"]}
    refs = list(_source_refs(payload["component"]))
    assert bool(refs) is names_sources
    assert set(refs) <= named


def test_a_rolling_cards_left_out_series_are_a_caveat_on_its_selection(
    frame_store, revision_store, release_registry
):
    """A rolling line chart of Ethiopia and Kenya leaves Kenya out at 2022, its latest period.
    The description words that as a `coverage_gap` caveat on the card's `select` step naming
    Kenya. The frame's manifest warning is the version's, so the card does not copy it: the
    version's caveats reach the card through its record, which holds none here."""
    frame = _frame()
    columns = cast("list[str]", frame.frame["columns"])
    kept = [
        {column: cell for column, cell in zip(columns, row, strict=True) if column != "status"}
        for row in cast("list[list[object]]", frame.frame["rows"])
        if row[columns.index("recipient")] != "country/KEN"
        or row[columns.index("time_period")] != "2022"
    ]
    bare = build_resolved_frame(
        frame_ref=frame.frame_ref,
        role_keys=("recipient",),
        rows=kept,
        labels=frame.labels,
        manifest={
            "schema_version": 1,
            "warnings": [
                {"code": "coverage_below_scope", "severity": "warn", "message": "Less asked."}
            ],
        },
    )
    frame_store.put(resolved_frame=_described(bare))

    created = _write(
        _intent(
            bare,
            "line_chart",
            select={
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": ["country/ETH", "country/KEN"],
            },
            window={
                "start": {"anchor": "latest", "offset": -2},
                "end": {"anchor": "latest", "offset": 0},
            },
        ),
        vocabulary=_vocabulary(),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    description = _description(created.resolved_payload)
    assert description["caveats"] == [
        {
            "kind": "coverage_gap",
            "entities": [{"ref": "country/KEN", "label": "Kenya"}],
            "periods": ["2022"],
            "step": _selection_step_id(description),
            "text": "Kenya left out of the selection in 2022.",
        }
    ]


def _selection_step_id(description: dict[str, Any]) -> str:
    (selection,) = [step["id"] for step in description["steps"] if step["op"] == "select"]
    return selection


_MALARIA = "dcid:MalariaGrantDisbursements"
_TB = "dcid:TuberculosisGrantDisbursements"


@pytest.mark.parametrize(
    ("unheld", "stat_vars", "units", "caveats", "texts"),
    [
        pytest.param(
            {(_TB, "country/KEN")},
            [_MALARIA, _TB],
            None,
            [(["TuberculosisGrantDisbursements"], ["country/KEN"])],
            ["Tuberculosis grants for Kenya left out of the selection in 2022."],
            id="one-variable-at-a-place",
        ),
        pytest.param(
            {(_MALARIA, "country/KEN"), (_TB, "country/KEN")},
            [_MALARIA, _TB],
            None,
            [],
            [],
            id="every-variable-at-a-place",
        ),
        pytest.param(
            set(),
            [_MALARIA, "dcid:PolioGrantDisbursements"],
            None,
            [(["PolioGrantDisbursements"], [])],
            ["Polio grants left out of the selection in 2022."],
            id="a-variable-the-frame-lacks",
        ),
        pytest.param(
            set(),
            _MALARIA,
            ["CurrencyUSD", "CurrencyEUR"],
            [(["CurrencyEUR"], [])],
            ["Euros left out of the selection in 2022."],
            id="a-unit-the-frame-lacks",
        ),
    ],
)
def test_a_rolling_cards_left_out_caveat_names_the_variables_left_out(
    frame_store, revision_store, release_registry, unheld, stat_vars, units, caveats, texts
):
    """A rolling line chart of variables at Kenya, over a frame that also holds Ethiopia,
    with no 2022 row for each variable and place in `unheld`. The caveat names the variables
    left out at Kenya. When every variable lacks 2022 the card's window ends at 2021, so it
    leaves nothing out. A selected variable the frame holds no row for is left out everywhere,
    so its caveat names no place, and a selected unit the frame holds no row in is named by
    its value."""
    frame = build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": stat_var,
                "recipient": recipient,
                "time_period": period,
                "value": 1.0,
                "unit": "CurrencyUSD",
                "measurementMethod": None,
                "provenance": _CRS,
            }
            for stat_var in (_MALARIA, _TB)
            for recipient in ("country/ETH", "country/KEN")
            for period in _PERIODS
            if period != "2022" or (stat_var, recipient) not in unheld
        ],
        labels={
            "country/ETH": "Ethiopia",
            "country/KEN": "Kenya",
            _TB: "Tuberculosis grants",
            "dcid:PolioGrantDisbursements": "Polio grants",
            "CurrencyEUR": "Euros",
        },
    )
    frame_store.put(resolved_frame=_described(frame))

    created = _write(
        _intent(
            frame,
            "line_chart",
            select={"stat_var": stat_vars, "recipient": "country/KEN"}
            | ({"unit": units} if units is not None else {}),
            window={
                "start": {"anchor": "latest", "offset": -2},
                "end": {"anchor": "latest", "offset": 0},
            },
        ),
        vocabulary=_vocabulary(),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    description = _description(created.resolved_payload)
    served = description["caveats"]
    assert [
        (
            [one["ref"] for one in caveat.get("series", [])],
            [one["ref"] for one in caveat.get("entities", [])],
        )
        for caveat in served
    ] == caveats
    selection = _selection_step_id(description)
    assert all(caveat["periods"] == ["2022"] and caveat["step"] == selection for caveat in served)
    assert [caveat["text"] for caveat in served] == texts


# -- a card over a computed version whose rows name no source --------------------------------


_GNI = FrameRef(spec_id="spec_" + "e" * 32, content_version="cv_" + "f" * 32)


def _computed_frame() -> ResolvedFrame:
    """A quotient computed by an analysis: its rows name no source of their own."""
    return _frame(frame_ref=_OPS_FRAME, provenance=None)


def _computed_record(frame: ResolvedFrame) -> DescriptionRecord:
    """The record Datasets writes when the `ops_` version is published with its steps: the two
    versions its `retrieve` steps name, and no source evidence of its own."""
    malaria = malaria_ethiopia_frame_ref()
    record = _record(frame)
    assert record.computed is not None
    return record.model_copy(
        update={
            "computed": record.computed.model_copy(
                update={
                    "inputs": [
                        InputVersion(id=malaria.spec_id, content_version=malaria.content_version),
                        InputVersion(id=_GNI.spec_id, content_version=_GNI.content_version),
                    ]
                }
            ),
            "series": record.series.model_copy(
                update={"by_source_unit": [SourceUnitCount(source=None, unit=None, series=3)]}
            ),
            "evidence": record.evidence.model_copy(update={"sources": []}),
        }
    )


@pytest.mark.parametrize(
    "member",
    [
        pytest.param("line_chart", id="line-chart"),
        pytest.param("bar_chart", id="bar-chart"),
        pytest.param("table", id="table"),
    ],
)
def test_a_card_over_a_computed_version_names_its_inputs_as_sources(
    frame_store, revision_store, release_registry, member
):
    """A card drawn from an `ops_` version whose rows name no source lists the version's inputs
    as its sources, each with its content version and the link to its description, and each
    series names the first input. Its provenance holds lineage alone."""
    frame = _computed_frame()
    frame_store.put(resolved_frame=_described(frame, _computed_record(frame)))
    created = _write(
        _intent(frame, member, title="Malaria grants per dollar of GNI"),
        vocabulary=_vocabulary(),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    payload = created.resolved_payload
    malaria = malaria_ethiopia_frame_ref()

    sources = _description(payload)["sources"]
    assert [(one["ref"], one["content_version"]) for one in sources] == [
        (malaria.spec_id, malaria.content_version),
        (_GNI.spec_id, _GNI.content_version),
    ]
    assert sources[0]["description"] == (
        f"/api/datasets/{malaria.spec_id}/{malaria.content_version}/description"
    )
    assert set(_source_refs(payload["component"])) == {malaria.spec_id}
    assert not _MOVED_PROVENANCE & set(cast("dict", payload["provenance"]))
    assert not {"measure", "fixed"} & set(_description(payload)["about"])


def _quotient_record(frame: ResolvedFrame) -> DescriptionRecord:
    """`_computed_record` with the steps its publish stored: the malaria grants retrieved, GNI
    retrieved, and the first divided by the second."""
    malaria = malaria_ethiopia_frame_ref()
    return _computed_record(frame).model_copy(
        update={
            "steps": [
                Step(
                    id="s1",
                    op="retrieve",
                    input=VersionRef(id=malaria.spec_id, content_version=malaria.content_version),
                    options={},
                ),
                Step(
                    id="s2",
                    op="retrieve",
                    input=VersionRef(id=_GNI.spec_id, content_version=_GNI.content_version),
                    options={},
                ),
                Step(id="s3", op="quotient", inputs={"numerator": "s1", "denominator": "s2"}),
            ]
        }
    )


def _input_records() -> tuple[DescriptionRecord, DescriptionRecord]:
    """The records of the malaria and GNI versions the quotient was computed from."""
    gni = _record(_frame(frame_ref=_GNI))
    assert gni.variables is not None
    return _record(_frame()), gni.model_copy(
        update={"variables": gni.variables.model_copy(update={"variable": "GNI", "fixed": []})}
    )


@pytest.mark.parametrize(
    ("inputs_read", "title", "labels", "step_labels"),
    [
        pytest.param(
            True,
            "Malaria grant disbursements for disbursement per gross national income, 2020 to 2022",
            ["Malaria grant disbursements for disbursement", "Gross national income"],
            ["Malaria grant disbursements for disbursement", "Gross national income"],
            id="inputs-read",
        ),
        pytest.param(
            False,
            "Malaria grant disbursements, 2020 to 2022",
            [malaria_ethiopia_frame_ref().spec_id, _GNI.spec_id],
            [None, None],
            id="input-read-failed",
        ),
    ],
)
def test_a_card_over_a_computed_version_names_its_inputs_by_their_titles(
    frame_store, revision_store, release_registry, inputs_read, title, labels, step_labels
):
    """A card over an `ops_` version whose frame holds its inputs' records titles itself with
    the name its steps derive and its window, and labels its sources and retrieve steps by the
    inputs' titles, worded from the labels the frame store read for their title refs. Where
    the input read failed, its sources keep the ids, its retrieve steps name no input, and its
    title is the one composed at mint."""
    frame = _computed_frame()
    frame_store.put(
        resolved_frame=dataclasses.replace(
            _described(frame, _quotient_record(frame)),
            input_records=_input_records() if inputs_read else (),
            labels={
                **frame.labels,
                "MalariaGrantDisbursements": "Malaria grant disbursements",
                "GNI": "Gross national income",
            },
        )
    )
    created = _write(
        _intent(frame, "line_chart", title="Malaria grants per dollar of GNI"),
        vocabulary=_vocabulary(),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    description = _description(created.resolved_payload)

    assert description["title"] == title
    assert [source["label"] for source in description["sources"]] == labels
    retrieved = [step for step in description["steps"] if step["op"] == "retrieve"]
    assert [step["input"].get("label") for step in retrieved] == step_labels


@pytest.mark.parametrize(
    ("member", "record"),
    [
        pytest.param("line_chart", False, id="line-chart-over-a-version-without-a-record"),
        pytest.param("source_card", True, id="source-card-listing-no-source"),
    ],
)
def test_a_card_over_a_computed_version_without_a_record_or_a_source_stays_legacy(
    frame_store, revision_store, release_registry, member, record
):
    """A card over an `ops_` version with no record, and a source card over one whose rows
    name no source, carry no description and keep the legacy provenance members."""
    frame = _computed_frame()
    frame_store.put(resolved_frame=_described(frame, _computed_record(frame)) if record else frame)
    fields = {} if member == "source_card" else {"title": "Malaria grants per dollar of GNI"}
    created = _write(
        _intent(frame, member, **fields),
        vocabulary=_vocabulary(),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    assert "description" not in created.resolved_payload
    assert _MOVED_PROVENANCE.issubset(cast("dict", created.resolved_payload["provenance"]))


@pytest.mark.parametrize(
    ("described_at_create", "revision"),
    [
        pytest.param(False, 2, id="head-issued-before-the-versions-record"),
        pytest.param(True, 1, id="head-issued-with-a-description"),
    ],
)
def test_a_revise_with_a_heads_intent_over_a_computed_version(
    frame_store, revision_store, release_registry, described_at_create, revision
):
    """A line chart over an `ops_` version whose rows name no source, revised with the intent
    its head stores once the version has a record. A head issued with a description, which
    names the source by the input its description lists, keeps its bytes. A head issued before
    the record gains the description in a new revision."""
    frame = _computed_frame()
    described = _described(frame, _computed_record(frame))
    frame_store.put(resolved_frame=described if described_at_create else frame)
    created = _write(
        _intent(frame, "line_chart", title="Malaria grants per dollar of GNI"),
        vocabulary=_vocabulary(),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert ("description" in created.resolved_payload) is described_at_create
    frame_store.put(resolved_frame=described)

    revised = _write(
        {"title": "Malaria grants per dollar of GNI"},
        vocabulary=_vocabulary(),
        component_id=created.component_id,
        write_token=created.write_token,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    assert revised.revision == revision
    assert "description" in revised.resolved_payload
    assert _stored(revision_store, created.component_id)[0] == _bytes(created.resolved_payload)


_WDI = "dc/base/WorldDevelopmentIndicators"
_ROOTS_CITATION = {
    "text": f"{_CRS_NAME}.\n\nWorld Development Indicators.",
    "sources": [
        {"ref": _CRS, "label": _CRS_NAME, "license": None, "url": ""},
        {"ref": _WDI, "label": "World Development Indicators", "license": None, "url": ""},
    ],
}
"""The citation the frame store gives a quotient of a CRS root over a WDI root
(`datasets_api.derived.recipe_citation`)."""


@pytest.mark.parametrize("member", ["line_chart", "table"])
def test_a_card_over_a_computed_version_published_without_steps_cites_its_roots(
    frame_store, revision_store, release_registry, member
):
    """A card over an `ops_` version whose rows name no source and whose record, written by a
    publish without steps, knows no input id, carries no description: its rows name the first
    source the roots cite, which the record does not list. Its provenance lists every source
    the roots cite, and each `source_ref` names one of them."""
    frame = dataclasses.replace(
        _computed_frame(), manifest={"schema_version": 1, "citation": _ROOTS_CITATION}
    )
    record = _computed_record(frame)
    assert record.computed is not None
    unknown = [
        InputVersion(id=None, content_version=one.content_version) for one in record.computed.inputs
    ]
    frame_store.put(
        resolved_frame=_described(
            frame,
            record.model_copy(
                update={"computed": record.computed.model_copy(update={"inputs": unknown})}
            ),
        )
    )
    created = _write(
        _intent(frame, member, title="Malaria grants per dollar of GNI"),
        vocabulary=_vocabulary(),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    payload = created.resolved_payload

    assert "description" not in payload
    sources = cast("list[dict[str, Any]]", cast("dict", payload["provenance"])["sources"])
    assert [(one["name"], one["ref"]) for one in sources] == [
        (_CRS_NAME, _CRS),
        ("World Development Indicators", _WDI),
    ]
    assert set(_source_refs(payload["component"])) == {sources[0]["id"]}


def test_a_card_whose_series_name_the_cited_source_and_none_carries_no_description(
    frame_store, revision_store, release_registry
):
    """A line chart over an `ops_` version where Ethiopia's rows name the CRS and the other
    rows name no source, both named by the CRS's provenance id, carries no description, since
    the description would name Ethiopia's series by the version's first input."""
    frame = build_resolved_frame(
        frame_ref=_OPS_FRAME,
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": recipient,
                "time_period": period,
                "value": 100.0,
                "unit": "CurrencyUSD",
                "measurementMethod": None,
                "provenance": _CRS if recipient == "country/ETH" else None,
            }
            for recipient in _RECIPIENTS
            for period in _PERIODS
        ],
        manifest={"schema_version": 1, "citation": _ROOTS_CITATION},
    )
    frame_store.put(resolved_frame=_described(frame, _computed_record(frame)))
    intent = _intent(frame, "line_chart", title="Malaria grants per dollar of GNI")
    intent["select"] = {"stat_var": "dcid:MalariaGrantDisbursements"}
    created = _write(
        intent,
        vocabulary=_vocabulary(),
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    assert "description" not in created.resolved_payload
