"""The committed fixture index (`INDEX_FIXTURE`) holds the blocks the export writes from the
declarations (`code_index.declarations`), so a declaration edit fails here until the fixture
is rewritten with the export's serialisers and `export.index_bytes`.

The fixture predates the climate finance tables and holds no row of their code system, so it
holds the measure partition labels whose `codes` lie in code systems its own `declarations`
block declares (`_HELD_LABELS`); a label keyed on another system would fail the export's
guard against the fixture's rows."""

from __future__ import annotations

import gzip
import json

import pytest

from code_index.declarations import (
    CODE_ALSO_CALLED,
    MEASURE_AXES,
    MEASURE_NOUNS,
    MEASURE_PARTITION_LABELS,
)
from code_index.export import _measure_axis_to_dict, _measure_partition_label_to_dict
from tests.code_index.support import INDEX_FIXTURE

_DOCUMENT = json.loads(gzip.decompress(INDEX_FIXTURE.read_bytes()))
_FIXTURE_CODE_SYSTEMS = frozenset(entry["code_system"] for entry in _DOCUMENT["declarations"])
_HELD_LABELS = [
    label
    for label in MEASURE_PARTITION_LABELS
    if {prop for prop, _ in label.codes} <= _FIXTURE_CODE_SYSTEMS
]


@pytest.mark.parametrize(
    ("block", "declared"),
    [
        ("measure_axes", [_measure_axis_to_dict(axis) for axis in MEASURE_AXES]),
        (
            "measure_partition_labels",
            [_measure_partition_label_to_dict(label) for label in _HELD_LABELS],
        ),
        ("measure_nouns", [_measure_partition_label_to_dict(noun) for noun in MEASURE_NOUNS]),
    ],
)
def test_the_fixture_measure_blocks_are_the_declarations(block: str, declared: list[dict]):
    assert _DOCUMENT[block] == declared


def test_each_fixture_code_row_holds_its_declared_also_called_names():
    assert {row["dcid"]: row["also_called"] for row in _DOCUMENT["codes"]} == {
        row["dcid"]: list(CODE_ALSO_CALLED.get(row["dcid"], ())) for row in _DOCUMENT["codes"]
    }
