"""Tests for `dfo_analysis_engine.recipe`: the normal form and the two hashes.

The tests of `build_recipe`, which reads a materialisation store, stay with that
function.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from dfo_analysis_engine.recipe import normal_form, normal_form_hash, replay_key

_IDENTITY = "0.3.0|1.0.0|0.2.0"
_OPTIONS: dict[str, Any] = {"format": "parquet", "bundle": False, "layout": "frame"}


_ROOT_NODE = {
    "frame": {"spec_id": "spec_" + "a" * 32, "content_version": "cv_" + "1" * 32},
    "root": 0,
}
_ROOTS = [{"dataset": {"spec_id": "spec_" + "a" * 32}, "options": _OPTIONS}]
_LATEST = {"anchor": "latest", "offset": 0, "rule": "half1"}
_TEN_BACK = {"anchor": "latest", "offset": -10, "rule": "half1"}


def _node(verb: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """A hand-built derived node over one root."""
    frame_ref = _ROOT_NODE["frame"]
    return {
        "frame": {"spec_id": "ops_" + "0" * 32, "content_version": "ocv_" + "0" * 32},
        "op_spec": {
            "identity_version": "ops1",
            "verb": verb,
            "frame_ref": frame_ref,
            "parameters": parameters,
        },
        "inputs": {"frame_ref": _ROOT_NODE},
    }


def _growth(start: str, end: str, rolling: dict[str, Any] | None) -> dict[str, Any]:
    parameters: dict[str, Any] = {"window_start": start, "window_end": end, "entity_by": ["r"]}
    return _node("growth", parameters if rolling is None else parameters | {"rolling": rolling})


def _threshold(value: str, rolling: dict[str, Any] | None) -> dict[str, Any]:
    predicate = {"kind": "ordered_threshold", "column": "time_period", "op": "gte", "value": value}
    return _node(
        "filter",
        {"predicates": [predicate if rolling is None else predicate | {"rolling": rolling}]},
    )


_ANCHOR_ROOT = {
    "frame": {"spec_id": "spec_" + "b" * 32, "content_version": "cv_" + "2" * 32},
    "root": 1,
}
_ANCHORED_ROOTS = [*_ROOTS, {"dataset": {"spec_id": "spec_" + "b" * 32}, "options": _OPTIONS}]


@pytest.mark.parametrize(
    ("anchors", "expected"),
    [
        pytest.param({}, {}, id="no-anchor-roots"),
        pytest.param(
            {"anchor_roots": [_ROOT_NODE, _ANCHOR_ROOT]},
            {
                "anchor_roots": [
                    {"spec_id": "spec_" + "a" * 32, "rendering": _OPTIONS},
                    {"spec_id": "spec_" + "b" * 32, "rendering": _OPTIONS},
                ]
            },
            id="anchor-roots-as-their-roots",
        ),
    ],
)
def test_the_normal_form_names_no_frame_and_no_root_version(
    anchors: dict[str, Any], expected: dict[str, Any]
) -> None:
    """A node's anchor roots, which its tree's latest period is taken over, stay in the
    normal form as roots do."""
    assert normal_form(_growth("2013", "2023", None) | anchors, _ANCHORED_ROOTS) == {
        "identity_version": "ops1",
        "verb": "growth",
        "parameters": {"window_start": "2013", "window_end": "2023", "entity_by": ["r"]},
        "inputs": {"frame_ref": {"spec_id": "spec_" + "a" * 32, "rendering": _OPTIONS}},
        **expected,
    }


def test_the_normal_form_drops_each_literal_a_rolling_overlay_governs() -> None:
    both = {"window_start": _TEN_BACK, "window_end": _LATEST}

    assert normal_form(_growth("2013", "2023", both), _ROOTS)["parameters"] == {
        "entity_by": ["r"],
        "rolling": both,
    }
    # Only the governed end is dropped, and a fixed window keeps both.
    assert normal_form(_growth("2013", "2023", {"window_end": _LATEST}), _ROOTS)["parameters"] == {
        "window_start": "2013",
        "entity_by": ["r"],
        "rolling": {"window_end": _LATEST},
    }
    assert normal_form(_threshold("2023", _LATEST), _ROOTS)["parameters"] == {
        "predicates": [
            {"kind": "ordered_threshold", "column": "time_period", "op": "gte", "rolling": _LATEST}
        ]
    }


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        (
            _growth("2013", "2023", {"window_start": _TEN_BACK, "window_end": _LATEST}),
            _growth("2014", "2024", {"window_start": _TEN_BACK, "window_end": _LATEST}),
            True,
        ),
        (_growth("2013", "2023", None), _growth("2014", "2024", None), False),
        (
            _growth("2013", "2023", {"window_end": _LATEST}),
            _growth("2014", "2024", {"window_end": _LATEST}),
            False,
        ),
        (
            _growth("2013", "2023", {"window_start": _TEN_BACK, "window_end": _LATEST}),
            _growth("2013", "2023", {"window_start": _LATEST, "window_end": _LATEST}),
            False,
        ),
        (_threshold("2023", _LATEST), _threshold("2024", _LATEST), True),
        (_threshold("2023", None), _threshold("2024", None), False),
        (_threshold("2023", _LATEST), _threshold("2023", None), False),
    ],
    ids=[
        "rolling-window-moved",
        "fixed-window-moved",
        "half-rolling-window-moved-at-its-fixed-end",
        "rolling-offset-changed",
        "rolling-threshold-moved",
        "fixed-threshold-moved",
        "rolling-against-fixed-threshold",
    ],
)
def test_only_a_rolling_literal_can_move_without_moving_the_normal_form_hash(
    first: dict[str, Any], second: dict[str, Any], same: bool
) -> None:
    hashes = [normal_form_hash(normal_form(tree, _ROOTS)) for tree in (first, second)]

    assert (hashes[0] == hashes[1]) is same


def test_the_two_hashes_are_sha256_of_their_documented_preimages() -> None:
    """Pinned to bytes written out here: apis indexes both values, so a change to
    either preimage orphans every stored recipe."""
    normal = {"spec_id": "spec_" + "a" * 32, "rendering": {"layout": "frame"}}
    normal_bytes = (
        b'{"identity_version":"rnf1","recipe":{"rendering":{"layout":"frame"},'
        b'"spec_id":"spec_' + b"a" * 32 + b'"}}'
    )
    hashed = normal_form_hash(normal)
    roots = [
        {"spec_id": "spec_" + "a" * 32, "content_version": "cv_" + "1" * 32},
        {"spec_id": "spec_" + "b" * 32, "content_version": "cv_" + "2" * 32},
    ]
    replay_bytes = (
        b'{"execution_identities":["0.2.0|1.0.0|0.2.0","0.3.0|1.0.0|0.2.0"],'
        b'"identity_version":"rpk1","normal_form_hash":"' + hashed.encode() + b'","roots":['
        b'{"content_version":"cv_' + b"1" * 32 + b'","spec_id":"spec_' + b"a" * 32 + b'"},'
        b'{"content_version":"cv_' + b"2" * 32 + b'","spec_id":"spec_' + b"b" * 32 + b'"}]}'
    )

    assert hashed == f"rnf_{hashlib.sha256(normal_bytes).hexdigest()[:32]}"
    assert (
        replay_key(
            normal_form_hash=hashed,
            root_datasets=roots,
            execution_identities=[_IDENTITY, "0.2.0|1.0.0|0.2.0", _IDENTITY],
        )
        == f"rpk_{hashlib.sha256(replay_bytes).hexdigest()[:32]}"
    )
    # Root order is part of the key: it says which root holds which version.
    assert replay_key(
        normal_form_hash=hashed, root_datasets=roots[::-1], execution_identities=[_IDENTITY]
    ) != replay_key(normal_form_hash=hashed, root_datasets=roots, execution_identities=[_IDENTITY])
