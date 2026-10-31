"""A stored version's facts assemble the wire its stored body rendered.

`version_facts_golden.json` holds six stored bodies (a definition, a collapsed head, a version
with warnings, a converted version, a body stored before the optional fields existed, and two
published frames) and what the body readers rendered from each: the snapshot manifest, the
output profile, conversion records and default scope a preview reads, the manifest a re-mint
returns, and the document a Presentation frame carries. Each body is projected onto its facts
as migration 0031 backfills them, and the read-time assembler must render the same JSON. The
writers must store the same facts the body held, and the `immutable_body` each version write
derives from its facts (`version_body`) must be the body without its `series_conversions`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from code_index.query import CodeIndex
from code_index.types import IndexStamp
from dataset_store.snapshot import FrameLineage, ManifestFacts, StoredVersion
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.publish import frame_lineage
from datasets_api.manifest import (
    assemble_manifest,
    collapsed_head_manifest,
    definition_version_record,
    persisted_manifest,
    record_collapsed_head_version,
    reshape_warning_models,
    stored_manifest,
    stored_output_profile,
    stored_series_conversions,
    version_body,
    version_document,
)
from datasets_api.manifest_models import ArtifactEntryModel, Manifest, WarningModel
from datasets_api.render import DerivedVersion
from datasets_api.versions import VersionRecord
from materialise.series_conversion import encode_series_conversions
from tests.datasets_api._legacy_conversion_trace import MIGRATED_CONVERSION

from ._version_facts import facts_from_body, golden_cases
from .test_manifest_assembly import (
    _PREPARED_AT,
    _SERIES_CONVERSIONS,
    _fixture_stored_definition,
    _statistics_fixture_frame,
    _statistics_fixture_result,
)
from .test_version_record_writer import RecordingVersionRecordWriter

_CASES = golden_cases()
_MANIFEST_CASES = [name for name, case in _CASES.items() if "manifest" in case["expected"]]
_FRAME_CASES = [name for name in _CASES if name not in _MANIFEST_CASES]

_ARTIFACT = ArtifactEntryModel(
    artifact_digest="sha256:" + "ab" * 32,
    rendering={"format": "csv"},
    bytes=12,
    media_type="text/csv",
    created_at="2026-10-04T12:00:00Z",
    tier="working",
    url="https://example.test/a.csv",
)
_STATS = _ARTIFACT.model_copy(
    update={"artifact_digest": "sha256:" + "cd" * 32, "rendering": {"statistics": "series"}}
)
# The stamp the golden bodies carry in `trace.code_index_stamp`. The cases mint against an index
# that holds it, so a re-export of the fixture code index cannot move a body.
_GOLDEN_CODE_INDEX = CodeIndex(
    stamp=IndexStamp(
        graph_endpoint="https://dfo-dev-dc-datacommons-service-zms5lj4zhq-nn.a.run.app",
        exported_at="2026-10-09T13:55:13Z",
        code_counts={},
        statvar_shape_counts={},
        content_hash="6db7833665fc20424b76038fc7d741079f0a9d055373532a59ce62a6267a7226",
    ),
    declarations=[],
    codes=[],
    statvars=[],
)
_CALL_WARNINGS = [
    WarningModel(code="enrichment_unavailable", severity="info", message="call warning"),
]


def _as_json(value: object) -> object:
    """`value` as storage returns it: JSON arrays read back as lists."""
    return json.loads(json.dumps(value, default=str))


def _stored(name: str) -> StoredVersion:
    case = _CASES[name]
    return StoredVersion(
        subject_id=case["subject_id"],
        subject_kind="spec",
        content_version=case["content_version"],
        schema_version=1,
        facts=facts_from_body(case["body"]),
        base_content_version=case["base_content_version"],
    )


@pytest.mark.parametrize("name", _MANIFEST_CASES)
def test_a_dataset_version_assembles_the_wire_its_body_rendered(name: str) -> None:
    expected = _CASES[name]["expected"]
    stored = _stored(name)
    facts = stored.facts
    assert isinstance(facts, ManifestFacts)

    manifest = stored_manifest(stored)
    assert manifest.model_dump(mode="json") == expected["manifest"]
    profile = stored_output_profile(facts)
    assert (
        _as_json(profile.to_json() if profile is not None else None) == (expected["output_profile"])
    )
    assert (
        _as_json(encode_series_conversions(stored_series_conversions(facts)))
        == expected["series_conversions"]
    )
    assert facts.default_scope_plural == expected["default_scope_plural"]

    served = manifest.model_copy(
        update={"artifacts": [_ARTIFACT, _STATS], "series_coverage_stamp": "sha256:served"}
    )
    persisted = persisted_manifest(stored=stored, served=served, reshape_warnings=_CALL_WARNINGS)
    assert persisted.model_dump(mode="json") == expected["persisted"]

    # Presentation reads a dataset version's citation, coverage and warnings.
    document = version_document(stored)
    for key in ("citation", "coverage", "warnings"):
        assert document[key] == expected["presentation_manifest"][key]


@pytest.mark.parametrize("name", _FRAME_CASES)
def test_a_published_frame_carries_its_lineage(name: str) -> None:
    body = _CASES[name]["body"]
    stored = _stored(name)
    assert isinstance(stored.facts, FrameLineage)
    assert stored.facts.composed_stat_vars == body.get("composed_stat_vars")
    # A body stored before versions stored continuations leaves them unknown.
    assert version_document(stored) == {
        **{key: value for key, value in body.items() if key != "composed_stat_vars"},
        "continuations": None,
    }


def _recorded(record_call, **kwargs) -> VersionRecord:
    writer = RecordingVersionRecordWriter()
    record_call(version_writer=writer, **kwargs)
    (record,) = writer.records
    return record


def _manifest(result, *, coverage_stamp: str | None = "sha256:coverage") -> Manifest:
    """A mint's manifest, with the citation text the golden bodies stored. They were stored
    before `citation_line` wrote the text, and a stored body keeps its text."""
    minted = assemble_manifest(
        stored=_fixture_stored_definition(spec_id="spec_golden", identity_digest="sha256:golden"),
        materialisation=result,
        artifacts=[],
        provenance_refs=["OECD_DAC2A"],
        source_citations={},
        prepared_at=_PREPARED_AT,
        capability_version="dset_test",
        materialiser_build="test-build",
        code_index=_GOLDEN_CODE_INDEX,
        settings=DatasetsBuildSettings(),
        coverage_stamp=coverage_stamp,
    )
    return minted.model_copy(
        update={"citation": minted.citation.model_copy(update={"text": "OECD_DAC2A"})}
    )


def _definition_record(result, manifest: Manifest) -> VersionRecord:
    return definition_version_record(
        manifest=manifest,
        stored=_fixture_stored_definition(spec_id="spec_golden", identity_digest="sha256:golden"),
        materialisation=result,
        artifacts=[],
    )


def _written_records() -> dict[str, VersionRecord]:
    """The records today's writers store for the golden versions, built as the golden bodies
    were."""
    frame = _statistics_fixture_frame()
    result = replace(
        _statistics_fixture_result(frame), default_scope=("country/KEN", "country/GHA")
    )
    manifest = _manifest(result)
    collapsed = collapsed_head_manifest(
        manifest=manifest,
        derived=DerivedVersion(
            frame_chunks=lambda: iter((frame,)),
            role_columns=(),
            rows=frame.num_rows,
            distinct={},
            frame_sha256="",
            content_version="cv_golden_collapsed",
            summed=True,
            party_totals_left_out=(),
        ),
        artifact=_ARTIFACT,
        statistics_artifact=_STATS,
        series_count=1,
        output_options=manifest.output_options,
        extra_warnings=reshape_warning_models(
            [{"code": "enrichment_unavailable", "column": "recipient_name", "count": 2}]
        ),
    )
    collapsed = collapsed.model_copy(update={"citation": manifest.citation})
    warned_result = replace(
        _statistics_fixture_result(frame), content_version="cv_golden_warned", default_scope=()
    )
    warned = _manifest(warned_result, coverage_stamp=None).model_copy(
        update={
            "warnings": [
                # Written as a mint wrote it before collapses reported their sum as a step.
                WarningModel.model_validate(_CASES["warnings"]["body"]["warnings"][0]),
                WarningModel(
                    code="origin_catalogue_stale", severity="warning", message="stored stale"
                ),
                WarningModel(
                    code="facet_default_selected", severity="info", message="facet axis 'unit'"
                ),
                *_CALL_WARNINGS,
            ]
        }
    )
    converted_result = replace(
        _statistics_fixture_result(frame),
        content_version="cv_golden_converted",
        series_conversions=_SERIES_CONVERSIONS,
        default_scope=("Earth",),
    )
    converted = _manifest(converted_result)
    converted = converted.model_copy(
        update={"trace": {**converted.trace, "conversion": MIGRATED_CONVERSION}}
    )
    return {
        "definition": _definition_record(result, manifest),
        "collapsed": _recorded(
            record_collapsed_head_version,
            manifest=collapsed,
            subject_id="spec_golden",
            schema_version=1,
            base_content_version=result.content_version,
            output_profile=result.output_profile,
            frame_sha256=result.frame_sha256,
            frame_summed=True,
        ),
        "warnings": _definition_record(warned_result, warned),
        "conversions": _definition_record(converted_result, converted),
    }


def _as_written(body: dict) -> dict:
    """`body` as today's writers store it: a citation's `processing_level_undeclared` is a
    manifest diagnostic, which a version does not store."""
    return {
        **body,
        "warnings": [
            warning
            for warning in body["warnings"]
            if warning["code"] != "processing_level_undeclared"
        ],
    }


@pytest.mark.parametrize("name", ["definition", "collapsed", "warnings", "conversions"])
def test_a_writer_stores_the_facts_its_body_held(name: str) -> None:
    record = _written_records()[name]
    case = _CASES[name]
    assert (record.subject_id, record.content_version, record.base_content_version) == (
        case["subject_id"],
        case["content_version"],
        case["base_content_version"],
    )
    expected = facts_from_body(_as_written(case["body"]))
    assert isinstance(expected, ManifestFacts)
    assert _as_json(asdict(record.facts)) == _as_json(asdict(expected))


@pytest.mark.parametrize("name", _FRAME_CASES)
def test_a_publish_stores_the_lineage_its_body_held(name: str) -> None:
    body = _CASES[name]["body"]
    assert frame_lineage(input_content_versions=body["input_content_versions"]) == replace(
        facts_from_body(body), composed_stat_vars=None, continuations=[]
    )


def _published_record(name: str) -> VersionRecord:
    case = _CASES[name]
    body = case["body"]
    return VersionRecord(
        subject_id=case["subject_id"],
        subject_kind="spec",
        content_version=case["content_version"],
        schema_version=1,
        facts=FrameLineage(
            input_content_versions=body["input_content_versions"],
            composed_stat_vars=body.get("composed_stat_vars") or None,
        ),
    )


@pytest.mark.parametrize(
    "name", ["definition", "collapsed", "warnings", "conversions", *_FRAME_CASES]
)
def test_a_version_body_is_the_stored_body_without_its_conversion_records(name: str) -> None:
    """The body a write derives from the facts is the JSON a case stored for the same version,
    less the warnings the writers do not store (`_as_written`) and the per-series conversion
    records, which only the `series_conversions` column holds."""
    if name in _FRAME_CASES:
        assert _as_json(version_body(_published_record(name))) == _CASES[name]["body"]
    else:
        body = _as_json(version_body(_written_records()[name]))
        stored = _as_written(_CASES[name]["body"])
        assert body == {key: value for key, value in stored.items() if key != "series_conversions"}
