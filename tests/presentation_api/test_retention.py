"""The retention a card states is that of the frame its load reads."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

from assembler.ports import FrameRef
from dataset_store.snapshot import Revocation
from datasets_api.render import CANONICAL_FRAME_RENDERING
from presentation_api.retention import stated_retention
from tests.datasets_api._snapshot_doubles import FakeSnapshotReader, _location, _version

_BASE = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "0" * 32)
_COLLAPSE = FrameRef(spec_id=_BASE.spec_id, content_version="cv_" + "1" * 32)
# Long past the 30-day working tier, whatever day the suite runs.
_STORED_AT = datetime(2025, 10, 1, tzinfo=UTC)


def _reader() -> FakeSnapshotReader:
    """A collapse whose own frame aged out of the working tier, over a pinned raw base."""
    reader = FakeSnapshotReader()
    for frame, base, tier in ((_BASE, None, "pinned"), (_COLLAPSE, _BASE, "working")):
        key = (frame.spec_id, frame.content_version)
        reader.versions[key] = dataclasses.replace(
            _version(
                content_version=frame.content_version,
                base_content_version=base.content_version if base is not None else None,
            ),
            subject_id=frame.spec_id,
        )
        reader.locations[key] = [
            _location(
                digest=f"sha256:{frame.content_version}",
                tier=tier,
                stored_at=_STORED_AT,
                rendering=CANONICAL_FRAME_RENDERING,
            )
        ]
    return reader


def test_a_collapse_states_its_own_frames_retention_not_its_pinned_bases():
    reader = _reader()

    assert stated_retention(_COLLAPSE, snapshot_reader=reader) == {
        "pinned": False,
        "held_until": "2025-10-31T00:00:00Z",
    }
    assert stated_retention(_BASE, snapshot_reader=reader) == {"pinned": True}


def test_a_revoked_base_leaves_its_collapse_with_no_retention():
    reader = _reader()
    reader.revocations[(_BASE.spec_id, _BASE.content_version)] = Revocation(
        reason="takedown", revoked_at=_STORED_AT
    )

    assert stated_retention(_COLLAPSE, snapshot_reader=reader) is None
