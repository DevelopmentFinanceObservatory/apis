"""Tests that the wire Literals in `graph_api.source_models` track the grammar's enums."""

from __future__ import annotations

from typing import get_args

from dfo_provenance_docs import ChangeKind, DocumentErrorReason

from graph_api.source_models import (
    ChangeKindWire,
    DocumentErrorReasonWire,
    DocumentMember,
    SourceDocumentation,
)
from materialise.provenance_doc import _DOCUMENT_MEMBERS, CONFLICTING_VALUES


def test_document_error_reasons_are_the_grammar_s_plus_conflicting_values():
    assert set(get_args(DocumentErrorReasonWire)) == {
        *(reason.value for reason in DocumentErrorReason),
        CONFLICTING_VALUES,
    }


def test_change_kinds_are_the_grammar_s():
    assert set(get_args(ChangeKindWire)) == {kind.value for kind in ChangeKind}


def test_document_members_are_the_reader_s():
    assert set(get_args(DocumentMember)) == set(_DOCUMENT_MEMBERS.values())


def test_version_history_entries_are_date_title_and_changes():
    defs = SourceDocumentation.model_json_schema()["$defs"]
    assert set(defs["VersionHistoryEntry"]["properties"]) == {"date", "title", "changes"}
    assert set(defs["ChangeEntry"]["properties"]) == {"kind", "note"}
