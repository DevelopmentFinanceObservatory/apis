"""Tests content-version minting and recomputation against grammar vectors."""

from __future__ import annotations

import importlib.resources
import json

import pytest
from dfo_frame_grammar import Frame, FrameError, canonical_frame_bytes, widen_frame_table
from dfo_frame_grammar import stream_frame_sha256 as grammar_stream_frame_sha256

from materialise import content_version
from materialise.content_version import (
    mint_version,
    mint_version_from_chunks,
    recompute_version,
    stream_frame_sha256,
)
from materialise.frame_build import build_frame


def _load_vector(name: str) -> dict:
    path = importlib.resources.files("dfo_frame_grammar").joinpath(f"vectors/frame/{name}")
    return json.loads(path.read_text())


def test_mint_version_and_recompute_match_installed_golden_vector():
    vector = _load_vector("frame-basic-single-row.json")
    frame = build_frame(role_keys=vector["role_keys"], rows=vector["rows"])

    assert mint_version(frame) == vector["expected_content_version"]
    assert recompute_version(canonical_frame_bytes(frame)) == vector["expected_content_version"]


def test_mint_version_from_chunks_matches_mint_version_over_the_concatenated_frame():
    """Hashing sorted, validated chunks matches the whole frame across a chunk boundary."""
    row_a = {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/AAA",
        "time_period": "2015",
        "value": 1000.0,
    }
    row_b = {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/BBB",
        "time_period": "2015",
        "value": 2000.0,
    }
    whole_frame = build_frame(role_keys=("recipient",), rows=[row_b, row_a])
    chunk_a = build_frame(role_keys=("recipient",), rows=[row_a])
    chunk_b = build_frame(role_keys=("recipient",), rows=[row_b])

    assert mint_version_from_chunks((chunk_a, chunk_b)) == mint_version(whole_frame)


def _accepted_vectors() -> list[dict]:
    root = importlib.resources.files("dfo_frame_grammar").joinpath("vectors/frame")
    vectors = [
        json.loads(path.read_text()) for path in root.iterdir() if path.name.endswith(".json")
    ]
    return [v for v in vectors if v.get("kind") == "frame" and v["classification"] == "positive"]


@pytest.mark.parametrize("vector", _accepted_vectors(), ids=lambda vector: vector["name"])
def test_stream_frame_sha256_matches_every_accepted_grammar_vector(vector):
    frame = build_frame(role_keys=vector["role_keys"], rows=vector["rows"])

    assert stream_frame_sha256([frame]) == vector["expected_frame_sha256"]


_UNITS = [
    "unit/USD",
    {"currency": "EUR", "prices": "constant", "base_year": 2023},
    {"currency": "EUR", "prices": "constant", "base_year": 2023, "recipe": "recipe/eur-2023"},
    {"numerator": "unit/USD", "denominator": {"stat_var": "dcid:Count_Person"}},
    None,
]
_TEXT = ["country/TGO", 'quote " and \\ backslash', "line\nbreak\ttab\u0001", "é ü 日本 😀", "z"]
_VALUES = [
    0.1,
    -5.5,
    12.0,
    1e21,
    1e-7,
    123456789012345680000.0,
    5e-324,
    1.7976931348623157e308,
    0.0,
]


def _mixed_rows(count: int) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        row: dict[str, object] = {
            "stat_var": f"dcid:Var{index % 3}",
            "recipient": f"{_TEXT[index % len(_TEXT)]}/{index}",
            "time_period": str(2000 + index % 7),
            "value": _VALUES[index % len(_VALUES)],
        }
        if (unit := _UNITS[index % len(_UNITS)]) is not None:
            row["unit"] = unit
        if index % 4:
            row["measurementMethod"] = _TEXT[(index + 1) % len(_TEXT)]
        rows.append(row)
    return rows


def _chunks_of(frame: Frame, sizes: list[int]) -> list[Frame]:
    """`frame` cut into chunks of `sizes` rows, every other chunk in the recipe unit layout and the
    rest in the narrowest layout their cells allow."""
    chunks, start = [], 0
    for index, size in enumerate(sizes):
        table = widen_frame_table(frame.table.slice(start, size), recipe=index % 2 == 1)
        chunks.append(Frame(columns=frame.columns, table=table))
        start += size
    assert start == frame.table.num_rows
    return chunks


@pytest.mark.parametrize("slice_rows", [1, 2, 7, 65_536])
def test_stream_frame_sha256_matches_the_grammar_across_chunks_and_slices(monkeypatch, slice_rows):
    """Escapes, non-ASCII text, sparse facets, every unit variant, float spellings, unit layouts
    that differ by chunk, and slices that cross chunk boundaries hash as the grammar hashes."""
    monkeypatch.setattr(content_version, "_HASH_SLICE_ROWS", slice_rows)
    frame = build_frame(role_keys=("recipient",), rows=_mixed_rows(40))
    chunks = _chunks_of(frame, [1, 12, 0, 20, 7])

    assert stream_frame_sha256(chunks) == grammar_stream_frame_sha256(chunks)
    assert stream_frame_sha256([frame]) == grammar_stream_frame_sha256([frame])


def test_stream_frame_sha256_serialises_each_distinct_cell_once(monkeypatch):
    """Cell bytes come from one `rfc8785.dumps` per distinct cell of each column, so a row adds
    no call of its own."""
    rows = [
        {
            "stat_var": f"dcid:Var{index % 2}",
            "recipient": f"country/R{index // 2:04d}",
            "time_period": "2020",
            "value": float(index),
            "unit": "unit/USD",
        }
        for index in range(2_000)
    ]
    frame = build_frame(role_keys=("recipient",), rows=rows)
    calls = 0
    real_dumps = content_version.rfc8785.dumps

    def counting_dumps(value):
        nonlocal calls
        calls += 1
        return real_dumps(value)

    monkeypatch.setattr(content_version.rfc8785, "dumps", counting_dumps)
    digest = stream_frame_sha256([frame])
    monkeypatch.undo()

    assert digest == grammar_stream_frame_sha256([frame])
    # prefix + stat_var 2 + recipient 1,000 + time_period 1 + value 2,000 + unit 1
    # + measurementMethod, provenance and status 1 each
    assert calls == 1 + 2 + 1_000 + 1 + 2_000 + 1 + 3


def _two_row_chunks(first: str, second: str) -> list[Frame]:
    def row(recipient: str) -> dict[str, object]:
        return {"stat_var": "dcid:V", "recipient": recipient, "time_period": "2020", "value": 1.0}

    return [build_frame(role_keys=("recipient",), rows=[row(r)]) for r in (first, second)]


@pytest.mark.parametrize(
    ("chunks", "code"),
    [
        pytest.param(
            _two_row_chunks("country/B", "country/A"), "row_key_not_increasing", id="decreasing"
        ),
        pytest.param(
            _two_row_chunks("country/A", "country/A"), "row_key_not_increasing", id="equal"
        ),
        pytest.param(
            [
                _two_row_chunks("country/A", "country/B")[0],
                build_frame(
                    role_keys=("provider",),
                    rows=[
                        {"stat_var": "dcid:V", "provider": "p", "time_period": "2020", "value": 1.0}
                    ],
                ),
            ],
            "column_mismatch",
            id="columns",
        ),
        pytest.param([], "schema_violation", id="no-chunks"),
    ],
)
def test_stream_frame_sha256_refuses_as_the_grammar_does(chunks, code):
    with pytest.raises(FrameError) as grammar_error:
        grammar_stream_frame_sha256(chunks)
    with pytest.raises(FrameError) as ours:
        stream_frame_sha256(chunks)

    assert ours.value.category == grammar_error.value.category == code
