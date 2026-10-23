"""`index_provenance` reads whichever index `get_index` serves, and a swapped index leaves
the fixture index's answer intact."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from code_index.shared import get_index
from tests.fixtures import provenance


class _SyntheticIndex:
    def __init__(self, dcid: str, provenance: str) -> None:
        self.statvars = [SimpleNamespace(dcid=dcid, provenances=(provenance,))]


def test_swapped_index_does_not_leak_into_the_fixture_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = next(row for row in get_index().statvars if len(row.provenances) == 1)
    packaged = provenance.index_provenance(row.dcid)
    assert packaged == row.provenances[0]

    synthetic = _SyntheticIndex(row.dcid, "provenance/Synthetic")
    with monkeypatch.context() as patch:
        patch.setattr(provenance, "get_index", lambda: synthetic)
        assert provenance.index_provenance(row.dcid) == "provenance/Synthetic"

    assert provenance.index_provenance(row.dcid) == packaged
