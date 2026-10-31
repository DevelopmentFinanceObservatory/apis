"""A stored frame1 artifact that does not read as a frame1 document is a stored-artifact fault."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

import pytest

from assembler.testing import build_frame_doc, malaria_ethiopia_rows
from datasets_api.app import classify_datasets_exception
from datasets_api.store_errors import (
    StoredFrameUnreadable,
    read_stored_frame1_table,
    validated_stored_frame1,
)


def _malaria_document() -> tuple[dict[str, object], list[list[object]]]:
    """The malaria frame1 document and its rows."""
    document = build_frame_doc(role_keys=("recipient",), rows=malaria_ethiopia_rows())
    return document, cast("list[list[object]]", document["rows"])


def _null_role(rows: list[list[object]]) -> None:
    rows[0][1] = None


def _repeated_grain(rows: list[list[object]]) -> None:
    rows.insert(1, list(rows[1]))


@pytest.mark.parametrize("read", [read_stored_frame1_table, validated_stored_frame1])
def test_a_stored_frame1_the_rule_refuses_names_its_digest_and_not_the_parser(
    read: Callable[..., object],
):
    with pytest.raises(StoredFrameUnreadable) as excinfo:
        read(b'{"frame_version": "frame0"}', digest="sha256:exemplar")

    assert "sha256:exemplar" in str(excinfo.value)
    assert "frame0" not in str(excinfo.value)


@pytest.mark.parametrize("damage", [_null_role, _repeated_grain])
def test_a_stored_frame1_that_is_no_grammar_frame_refuses_a_validated_read(
    damage: Callable[[list[list[object]]], None],
):
    """The rows read as a table, and a reader needing a grammar-valid frame refuses them."""
    document, rows = _malaria_document()
    damage(rows)
    data = json.dumps(document).encode()

    read_stored_frame1_table(data, digest="sha256:exemplar")
    with pytest.raises(StoredFrameUnreadable):
        validated_stored_frame1(data, digest="sha256:exemplar")


def test_a_validated_read_sorts_a_stored_frame1s_rows():
    document, rows = _malaria_document()
    data = json.dumps(document).encode()
    rows.reverse()

    reversed_frame = validated_stored_frame1(json.dumps(document).encode(), digest="d")

    assert reversed_frame == validated_stored_frame1(data, digest="d")


def test_no_datasets_door_classifies_an_unreadable_stored_frame():
    """It answers the platform's `internal_error`, as a stored canonical frame that does not
    decode does, so its detail carries nothing of the artifact."""
    exc = StoredFrameUnreadable(digest="sha256:exemplar")

    assert classify_datasets_exception(exc) is None
