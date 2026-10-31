"""A stored frame1 artifact that does not read as a frame1 document is a stored-artifact fault."""

from __future__ import annotations

import pytest

from datasets_api.app import classify_datasets_exception
from datasets_api.store_errors import StoredFrameUnreadable, read_stored_frame1_table


def test_a_stored_frame1_the_rule_refuses_names_its_digest_and_not_the_parser():
    with pytest.raises(StoredFrameUnreadable) as excinfo:
        read_stored_frame1_table(b'{"frame_version": "frame0"}', digest="sha256:exemplar")

    assert "sha256:exemplar" in str(excinfo.value)
    assert "frame0" not in str(excinfo.value)


def test_no_datasets_door_classifies_an_unreadable_stored_frame():
    """It answers the platform's `internal_error`, as a stored canonical frame that does not
    decode does, so its detail carries nothing of the artifact."""
    exc = StoredFrameUnreadable(digest="sha256:exemplar")

    assert classify_datasets_exception(exc) is None
