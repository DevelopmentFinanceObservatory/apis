"""Tests the labels `presentation_store.composed_stat_vars` words from a frame's entries."""

from __future__ import annotations

import pytest

from assembler.limits import LABEL_MAX
from assembler.ports import FrameRef
from presentation_store.composed_stat_vars import (
    ComposedStatVar,
    composed_stat_var_labels,
    read_composed_stat_vars,
)

_LABELS = {"dcid:GDP": "Nominal GDP", "dcid:Count_Person": "Total population", "dcid:ODA": "Aid"}


def _entry(relation, first, second, *, stat_var="dcid:ops_a"):
    return ComposedStatVar(
        stat_var=stat_var, relation=relation, first=tuple(first), second=tuple(second)
    )


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        pytest.param(
            [_entry("quotient", ["dcid:GDP"], ["dcid:Count_Person"])],
            {"dcid:ops_a": "Nominal GDP per Total population"},
            id="quotient",
        ),
        pytest.param(
            [_entry("sum", ["dcid:GDP"], ["dcid:ODA"])],
            {"dcid:ops_a": "Nominal GDP plus Aid"},
            id="sum",
        ),
        pytest.param(
            [_entry("quotient", ["dcid:GDP"], ["dcid:GDP"])],
            {"dcid:ops_a": "Ratio of Nominal GDP"},
            id="quotient-of-equal-sides",
        ),
        pytest.param(
            [_entry("share", ["dcid:ODA"], ["dcid:GDP"])],
            {"dcid:ops_a": "Aid as a share of Nominal GDP"},
            id="share",
        ),
        pytest.param(
            [_entry("share", ["dcid:GDP"], ["dcid:GDP"])],
            {"dcid:ops_a": "Share of Nominal GDP"},
            id="share-of-equal-sides",
        ),
        pytest.param(
            [_entry("sum", ["dcid:GDP"], ["dcid:GDP"])],
            {"dcid:ops_a": "Nominal GDP"},
            id="sum-of-equal-sides",
        ),
        pytest.param(
            [_entry("quotient", ["dcid:GDP", "dcid:ODA"], ["dcid:Count_Person"])],
            {"dcid:ops_a": "Nominal GDP and Aid per Total population"},
            id="side-with-two-refs",
        ),
        pytest.param(
            [_entry("quotient", ["dcid:GDP"], ["dcid:Count_Person_Female"])],
            {"dcid:ops_a": "Nominal GDP per Count_Person_Female"},
            id="unnamed-ref-reads-as-its-bare-dcid",
        ),
        pytest.param(
            [
                _entry("sum", ["dcid:GDP"], ["dcid:ODA"]),
                _entry("quotient", ["dcid:ops_a"], ["dcid:Count_Person"], stat_var="dcid:ops_b"),
            ],
            {
                "dcid:ops_a": "Nominal GDP plus Aid",
                "dcid:ops_b": "Nominal GDP plus Aid per Total population",
            },
            id="nested-entry-reads-the-earlier-label",
        ),
        pytest.param(
            [_entry("quotient", ["dcid:" + "A" * 150], ["dcid:" + "B" * 150])],
            {"dcid:ops_a": "A" * 97 + " per " + "B" * 98},
            id="two-long-sides-take-half-each",
        ),
        pytest.param(
            [_entry("quotient", ["dcid:GDP"], ["dcid:" + "B" * 250])],
            {"dcid:ops_a": "Nominal GDP per " + "B" * 184},
            id="a-short-side-keeps-its-text",
        ),
    ],
)
def test_a_composed_stat_var_is_worded_from_its_sides(entries, expected):
    labels = composed_stat_var_labels(entries=entries, labels=_LABELS)

    assert labels == expected
    assert all(len(label) <= LABEL_MAX for label in labels.values())


def test_a_stored_share_entry_is_read():
    stored = [
        {
            "stat_var": "dcid:ops_a",
            "relation": "share",
            "first": ["dcid:ODA"],
            "second": ["dcid:GDP"],
        }
    ]

    entries = read_composed_stat_vars(
        stored, frame_ref=FrameRef(spec_id="ops_" + "a" * 32, content_version="ocv_" + "b" * 32)
    )

    assert entries == (_entry("share", ["dcid:ODA"], ["dcid:GDP"]),)
