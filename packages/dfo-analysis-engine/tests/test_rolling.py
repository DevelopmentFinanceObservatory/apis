"""Tests for `dfo_analysis_engine.rolling`: the canonical populations, the anchor key, and the
populations of a recipe tree.

dfo-ai's `services/analysis-mcp/tests/test_rolling_tool.py` checks
`anchor_populations` against the populations real verb calls hash.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from dfo_analysis_engine.refusal import AnalysisRefused, RefusalCode
from dfo_analysis_engine.rolling import (
    PairingKeysUnstatedError,
    PopulationEmptyError,
    PopulationLevelColumnMissingError,
    anchor_key,
    anchor_populations,
    canonical_populations,
    rolling_roots,
)

_AID = "spec_" + "a" * 32
_GNI = "spec_" + "b" * 32
_COLUMNS = ["stat_var", "reporter", "recipient", "time_period", "value", "unit", "status"]


def _root(spec_id: str) -> dict[str, Any]:
    return {"dataset": {"spec_id": spec_id, "content_version": "cv_" + "1" * 32}}


def _node(verb: str, parameters: dict[str, Any], **inputs: dict[str, Any]) -> dict[str, Any]:
    return {"op_spec": {"verb": verb, "parameters": parameters}, "inputs": inputs}


def _rolled(below: dict[str, Any]) -> dict[str, Any]:
    """A rolling step over `below`, which binds nothing itself."""
    window_end = {"anchor": "latest", "offset": 0, "rule": "half1"}
    return _node("growth", {"rolling": {"window_end": window_end}}, frame_ref=below)


def _match(column: str, *values: str) -> dict[str, Any]:
    return {"kind": "category_match", "column": column, "values": list(values)}


_RIGHT = {"spec_id": "x", "content_version": "y"}
# A join keyed on the column both sides call `reporter`, as the compiler writes it.
_JOIN_ON_REPORTER = {"right_frame_ref": _RIGHT, "join_keys": ["reporter"]}


def test_populations_have_one_spelling_however_they_are_listed() -> None:
    one = [
        {"spec_id": _GNI},
        {"spec_id": _AID, "bindings": {"reporter": ["dcid:USA", "dcid:FRA", "dcid:USA"]}},
        {"spec_id": _AID, "bindings": {"reporter": ["dcid:FRA", "dcid:USA"]}},
    ]
    other = [
        {"spec_id": _AID, "bindings": {"reporter": ["dcid:FRA", "dcid:USA"]}},
        {"spec_id": _GNI, "bindings": {}},
    ]

    assert canonical_populations(one) == canonical_populations(other)
    assert canonical_populations(one) == [
        {"spec_id": _AID, "bindings": {"reporter": ["dcid:FRA", "dcid:USA"]}},
        {"spec_id": _GNI, "bindings": {}},
    ]


def test_the_anchor_key_hashes_the_canonical_populations_the_sorted_roots_and_the_rule() -> None:
    populations = [{"spec_id": _AID, "bindings": {"reporter": ["dcid:USA", "dcid:FRA"]}}]
    first = {"spec_id": _AID, "content_version": "cv_" + "1" * 32}
    second = {"spec_id": _AID, "content_version": "cv_" + "2" * 32}
    # Every string is ASCII, so sorted compact JSON is the RFC 8785 form.
    preimage = json.dumps(
        {
            "populations": [{"spec_id": _AID, "bindings": {"reporter": ["dcid:FRA", "dcid:USA"]}}],
            "roots": [first, second],
            "rule": "half1",
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    key = anchor_key(populations=populations, roots=[second, first, second], rule="half1")

    assert key == "rak_" + hashlib.sha256(preimage.encode()).hexdigest()[:32]
    assert key != anchor_key(populations=populations, roots=[first], rule="half1")
    assert key != anchor_key(populations=populations, roots=[first, second], rule="half2")
    assert key != anchor_key(populations=[{"spec_id": _AID}], roots=[first, second], rule="half1")


def test_a_path_is_bound_by_its_filters_its_side_of_a_two_sided_verb_and_the_selection() -> None:
    # share_of(filter(aid), gni): the filter narrows the numerator's root only, and
    # each side's entity binding its own root.
    tree = _node(
        "share_of",
        {
            "denominator_frame_ref": {"spec_id": "x", "content_version": "y"},
            "numerator_entity": {"recipient": "dcid:GHA"},
            "denominator_entity": {"reporter": "dcid:FRA"},
        },
        frame_ref=_node(
            "filter",
            {
                "predicates": [
                    _match("reporter", "dcid:USA", "dcid:FRA", "dcid:DEU"),
                    # A period threshold, a value threshold and a column the root
                    # lacks bind no series.
                    {"kind": "ordered_threshold", "column": "time_period", "op": "gte"},
                    {"kind": "ordered_threshold", "column": "value", "op": "gt", "value": 0},
                    _match("sector", "dcid:Health"),
                ]
            },
            frame_ref={"root": 0},
        ),
        denominator_frame_ref={"root": 1},
    )

    populations = anchor_populations(
        _rolled(tree),
        [_root(_AID), _root(_GNI)],
        root_columns=[_COLUMNS, ["stat_var", "reporter", "time_period", "value", "unit"]],
        selection={"reporter": ["dcid:FRA", "dcid:USA"], "time_period": ["2023"]},
    )

    assert populations == [
        {
            "spec_id": _AID,
            "bindings": {"recipient": ["dcid:GHA"], "reporter": ["dcid:FRA", "dcid:USA"]},
        },
        {"spec_id": _GNI, "bindings": {"reporter": ["dcid:FRA"]}},
    ]


def test_a_verb_reading_one_frame_twice_gives_a_population_for_each_side() -> None:
    tree = _node(
        "compare",
        {
            "right_frame_ref": None,
            "left_entity": {"reporter": "dcid:USA"},
            "right_entity": {"reporter": "dcid:FRA"},
        },
        frame_ref={"root": 0},
    )

    assert anchor_populations(_rolled(tree), [_root(_AID)], root_columns=[_COLUMNS]) == [
        {"spec_id": _AID, "bindings": {"reporter": ["dcid:FRA"]}},
        {"spec_id": _AID, "bindings": {"reporter": ["dcid:USA"]}},
    ]


def test_a_tree_with_no_binding_keeps_every_series_of_each_root() -> None:
    tree = _node(
        "sum_of",
        {"right_frame_ref": {"spec_id": "x", "content_version": "y"}},
        frame_ref={"root": 0},
        right_frame_ref={"root": 0},
    )

    assert anchor_populations(_rolled(tree), [_root(_AID)], root_columns=[_COLUMNS]) == [
        {"spec_id": _AID, "bindings": {}}
    ]


@pytest.mark.parametrize(
    "predicate", ["category_match", None, ["reporter"]], ids=["string", "none", "list"]
)
def test_a_predicate_that_is_no_object_refuses_op_spec_invalid(predicate: object) -> None:
    tree = _node("filter", {"predicates": [predicate]}, frame_ref={"root": 0})

    with pytest.raises(AnalysisRefused) as raised:
        anchor_populations(_rolled(tree), [_root(_AID)], root_columns=[_COLUMNS])

    assert raised.value.refusal.code is RefusalCode.OP_SPEC_INVALID


def test_bindings_that_share_no_value_raise() -> None:
    tree = _node("filter", {"predicates": [_match("reporter", "dcid:USA")]}, frame_ref={"root": 0})

    with pytest.raises(PopulationEmptyError) as raised:
        anchor_populations(
            _rolled(tree),
            [_root(_AID)],
            root_columns=[_COLUMNS],
            selection={"reporter": ["dcid:FRA"]},
        )

    assert (raised.value.spec_id, raised.value.column) == (_AID, "reporter")


_TYPED = [*_COLUMNS, "observationAbout", "observationAbout_type"]


def _level(level: str) -> dict[str, Any]:
    return {"kind": "place_level", "column": "observationAbout_type", "level": level}


def test_a_level_filter_holds_its_roots_population_to_that_level() -> None:
    tree = _node(
        "rank",
        {},
        frame_ref=_node(
            "filter",
            {"predicates": [_level("Country"), _match("reporter", "dcid:USA")]},
            frame_ref={"root": 0},
        ),
    )

    assert anchor_populations(_rolled(tree), [_root(_AID)], root_columns=[_TYPED]) == [
        {
            "spec_id": _AID,
            "bindings": {"reporter": ["dcid:USA"]},
            "levels": {"observationAbout_type": "Country"},
        }
    ]


def test_a_population_with_no_level_is_spelled_and_keyed_as_before_levels() -> None:
    plain = {"spec_id": _AID, "bindings": {"reporter": ["dcid:USA"]}}
    root = {"spec_id": _AID, "content_version": "cv_" + "1" * 32}
    levelled = {**plain, "levels": {"observationAbout_type": "Country"}}

    assert canonical_populations([{**plain, "levels": {}}]) == [plain]
    assert anchor_key(populations=[{**plain, "levels": {}}], roots=[root], rule="half1") == (
        anchor_key(populations=[plain], roots=[root], rule="half1")
    )
    # A level is part of the population, and so of the key.
    preimage = json.dumps(
        {"populations": [levelled], "roots": [root], "rule": "half1"},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert anchor_key(populations=[levelled], roots=[root], rule="half1") == (
        "rak_" + hashlib.sha256(preimage.encode()).hexdigest()[:32]
    )
    assert canonical_populations([plain, levelled, plain]) == [levelled, plain]


def test_a_strict_level_is_canonical_and_changes_the_population_key() -> None:
    plain = {"spec_id": _AID, "bindings": {}, "levels": {"observationAbout_type": "Country"}}
    strict = {**plain, "strict_levels": ["observationAbout_type"]}
    root = {"spec_id": _AID, "content_version": "cv_" + "1" * 32}

    assert canonical_populations([strict]) == [strict]
    assert anchor_key(populations=[plain], roots=[root], rule="half1") != anchor_key(
        populations=[strict], roots=[root], rule="half1"
    )


def test_a_level_filter_over_a_root_without_the_type_column_raises() -> None:
    tree = _node("filter", {"predicates": [_level("Country")]}, frame_ref={"root": 0})

    with pytest.raises(PopulationLevelColumnMissingError) as raised:
        anchor_populations(_rolled(tree), [_root(_AID)], root_columns=[_COLUMNS])

    assert (raised.value.spec_id, raised.value.column) == (_AID, "observationAbout_type")


def test_two_levels_on_one_path_keep_no_series() -> None:
    tree = _node(
        "filter",
        {"predicates": [_level("Country")]},
        frame_ref=_node("filter", {"predicates": [_level("Continent")]}, frame_ref={"root": 0}),
    )

    with pytest.raises(PopulationEmptyError):
        anchor_populations(_rolled(tree), [_root(_AID)], root_columns=[_TYPED])


def test_only_a_path_through_a_rolling_step_gives_a_population_and_a_root() -> None:
    # join(growth over aid, gni): no rolling step reads gni, whatever is bound on it.
    beside = _node(
        "join",
        _JOIN_ON_REPORTER,
        frame_ref=_rolled(
            _node("filter", {"predicates": [_match("reporter", "dcid:USA")]}, frame_ref={"root": 0})
        ),
        right_frame_ref=_node(
            "filter", {"predicates": [_match("reporter", "dcid:FRA")]}, frame_ref={"root": 1}
        ),
    )
    roots = [_root(_AID), _root(_GNI)]
    columns = [_COLUMNS, _COLUMNS]

    assert anchor_populations(beside, roots, root_columns=columns) == [
        {"spec_id": _AID, "bindings": {"reporter": ["dcid:USA"]}}
    ]
    assert rolling_roots(beside) == [{"root": 0}]
    # A rolling step above the join reads both, and one on each side does too.
    assert [node["root"] for node in rolling_roots(_rolled(beside))] == [0, 1]
    both = _node(
        "sum_of",
        {"right_frame_ref": {"spec_id": "x", "content_version": "y"}},
        frame_ref=_rolled({"root": 1}),
        right_frame_ref=_rolled({"root": 0}),
    )
    assert [node["root"] for node in rolling_roots(both)] == [0, 1]
    assert anchor_populations(both, roots, root_columns=columns) == [
        {"spec_id": _AID, "bindings": {}},
        {"spec_id": _GNI, "bindings": {}},
    ]


def test_a_tree_with_no_rolling_step_has_no_population_and_no_root() -> None:
    fixed = _node("filter", {"predicates": [_match("reporter", "dcid:USA")]}, frame_ref={"root": 0})

    assert anchor_populations(fixed, [_root(_AID)], root_columns=[_COLUMNS]) == []
    assert rolling_roots(fixed) == []


def _quotient(**parameters: Any) -> dict[str, Any]:
    """quotient_of(aid, gni) with `parameters` beside its frame slots."""
    return _node(
        "quotient_of",
        {"denominator_frame_ref": _RIGHT, **parameters},
        frame_ref={"root": 0},
        denominator_frame_ref={"root": 1},
    )


_AID_GNI = [_root(_AID), _root(_GNI)]
_GNI_COLUMNS = ["stat_var", "observationAbout", "time_period", "value", "unit"]


@pytest.mark.parametrize(
    ("right_columns", "right"),
    [
        pytest.param(
            [*_GNI_COLUMNS, "observationAbout_type"],
            {
                "spec_id": _GNI,
                "bindings": {"observationAbout": ["dcid:GHA"]},
                "levels": {"observationAbout_type": "Country"},
            },
            id="second-side-holds-the-column",
        ),
        pytest.param(
            _GNI_COLUMNS,
            {"spec_id": _GNI, "bindings": {"observationAbout": ["dcid:GHA"]}},
            id="second-side-lacks-the-column",
        ),
    ],
)
def test_a_narrowing_above_a_keyed_pairing_crosses_its_key_map(
    right_columns: list[str], right: dict[str, Any]
) -> None:
    """The slots are left out of the parameters, as the runner writes a tree."""
    quotient = _quotient(
        numerator_by=["recipient"], denominator_by={"recipient": "observationAbout"}
    )
    del quotient["op_spec"]["parameters"]["denominator_frame_ref"]
    tree = _node(
        "filter",
        {
            "predicates": [
                {"kind": "place_level", "column": "recipient_type", "level": "Country"},
                _match("recipient", "dcid:GHA"),
            ]
        },
        frame_ref=quotient,
    )

    populations = anchor_populations(
        _rolled(tree), _AID_GNI, root_columns=[[*_COLUMNS, "recipient_type"], right_columns]
    )

    assert populations == [
        {
            "pair": [
                [
                    {
                        "spec_id": _AID,
                        "bindings": {"recipient": ["dcid:GHA"]},
                        "levels": {"recipient_type": "Country"},
                    }
                ],
                [right],
            ],
            "keys": {"recipient": "observationAbout"},
        }
    ]


_FLOW_COLUMNS = [
    "stat_var",
    "financingProvider",
    "financingRecipient",
    "time_period",
    "value",
    "unit",
]
_SHAPES = pytest.mark.parametrize(
    "stored", [True, False], ids=["stored-shape", "runner-shape-slots-left-out"]
)


def _one_frame(verb: str, slot: str, *, stored: bool, **parameters: Any) -> dict[str, Any]:
    """`verb` reading root 0 on both sides, with the null second slot the stored shape keeps."""
    return _node(verb, {**parameters, **({slot: None} if stored else {})}, frame_ref={"root": 0})


def _two_frames(verb: str, slot: str, *, stored: bool, **parameters: Any) -> dict[str, Any]:
    """`verb` over roots 0 and 1, with the second slot's pair the stored shape keeps."""
    return _node(
        verb,
        {**parameters, **({slot: _RIGHT} if stored else {})},
        frame_ref={"root": 0},
        **{slot: {"root": 1}},
    )


@_SHAPES
@pytest.mark.parametrize(
    ("verb", "slot", "parameters"),
    [
        pytest.param(
            "compare",
            "right_frame_ref",
            {
                "left_entity": {"recipient": "dcid:GHA"},
                "right_entity": {"recipient": "dcid:KEN"},
            },
            id="compare",
        ),
        pytest.param(
            "share_of",
            "denominator_frame_ref",
            {
                "numerator_entity": {"recipient": "dcid:GHA"},
                "denominator_entity": {"recipient": "dcid:KEN"},
            },
            id="share_of",
        ),
    ],
)
def test_a_one_frame_two_sided_verb_counts_both_sides_in_either_tree_shape(
    stored: bool, verb: str, slot: str, parameters: dict[str, Any]
) -> None:
    tree = _one_frame(verb, slot, stored=stored, **parameters)

    assert anchor_populations(_rolled(tree), [_root(_AID)], root_columns=[_COLUMNS]) == [
        {"spec_id": _AID, "bindings": {"recipient": ["dcid:GHA"]}},
        {"spec_id": _AID, "bindings": {"recipient": ["dcid:KEN"]}},
    ]


@_SHAPES
def test_a_column_outside_the_key_map_stays_on_the_output_s_side(stored: bool) -> None:
    keyed = _two_frames(
        "share_of",
        "denominator_frame_ref",
        stored=stored,
        numerator_by=["financingRecipient"],
        denominator_by={"financingRecipient": "financingRecipient"},
    )
    level = {"kind": "place_level", "column": "financingProvider_type", "level": "Country"}
    tree = _node(
        "filter",
        {"predicates": [_match("financingProvider", "dcid:USA"), level]},
        frame_ref=keyed,
    )
    columns = [*_FLOW_COLUMNS, "financingProvider_type"]

    populations = anchor_populations(_rolled(tree), _AID_GNI, root_columns=[columns, columns])

    assert populations == [
        {
            "pair": [
                [
                    {
                        "spec_id": _AID,
                        "bindings": {"financingProvider": ["dcid:USA"]},
                        "levels": {"financingProvider_type": "Country"},
                    }
                ],
                [{"spec_id": _GNI, "bindings": {}}],
            ],
            "keys": {"financingRecipient": "financingRecipient"},
        }
    ]


@_SHAPES
def test_a_pooled_denominator_takes_no_narrowing_from_above(stored: bool) -> None:
    pooled = _two_frames("share_of", "denominator_frame_ref", stored=stored)
    tree = _node(
        "filter", {"predicates": [_match("financingProvider", "dcid:USA")]}, frame_ref=pooled
    )

    populations = anchor_populations(
        _rolled(tree), _AID_GNI, root_columns=[_FLOW_COLUMNS, _FLOW_COLUMNS]
    )

    assert populations == [
        {"spec_id": _AID, "bindings": {"financingProvider": ["dcid:USA"]}},
        {"spec_id": _GNI, "bindings": {}},
    ]


@_SHAPES
def test_a_key_column_crosses_under_its_mapped_name_and_its_namesake_stays(stored: bool) -> None:
    """The key maps recipients to the denominator's `financingProvider`, so the numerator's
    providers bind nothing there, and its recipients bind that column."""
    keyed = _two_frames(
        "share_of",
        "denominator_frame_ref",
        stored=stored,
        numerator_by=["financingRecipient"],
        denominator_by={"financingRecipient": "financingProvider"},
    )
    tree = _node(
        "filter",
        {
            "predicates": [
                _match("financingProvider", "dcid:USA"),
                _match("financingRecipient", "dcid:GHA"),
            ]
        },
        frame_ref=keyed,
    )

    populations = anchor_populations(
        _rolled(tree), _AID_GNI, root_columns=[_FLOW_COLUMNS, _FLOW_COLUMNS]
    )

    assert populations == [
        {
            "pair": [
                [
                    {
                        "spec_id": _AID,
                        "bindings": {
                            "financingProvider": ["dcid:USA"],
                            "financingRecipient": ["dcid:GHA"],
                        },
                    }
                ],
                [{"spec_id": _GNI, "bindings": {"financingProvider": ["dcid:GHA"]}}],
            ],
            "keys": {"financingRecipient": "financingProvider"},
        }
    ]


@_SHAPES
def test_a_join_that_states_no_keys_crosses_a_narrowing_under_its_own_names(stored: bool) -> None:
    joined = _two_frames("join", "right_frame_ref", stored=stored)
    level = {"kind": "place_level", "column": "financingRecipient_type", "level": "Country"}
    tree = _node(
        "filter",
        {"predicates": [_match("financingRecipient", "dcid:KEN"), level]},
        frame_ref=joined,
    )
    typed = [*_FLOW_COLUMNS, "financingRecipient_type"]

    populations = anchor_populations(_rolled(tree), _AID_GNI, root_columns=[typed, _FLOW_COLUMNS])

    assert populations == [
        {
            "spec_id": _AID,
            "bindings": {"financingRecipient": ["dcid:KEN"]},
            "levels": {"financingRecipient_type": "Country"},
        },
        {"spec_id": _GNI, "bindings": {"financingRecipient": ["dcid:KEN"]}},
    ]


def test_a_compare_takes_its_output_columns_from_its_right_side() -> None:
    compare = {
        "op_spec": {
            "verb": "compare",
            "parameters": {"on": {"observationAbout": "recipient"}},
        },
        "inputs": {"frame_ref": {"root": 0}, "right_frame_ref": {"root": 1}},
    }
    tree = _node("filter", {"predicates": [_level("Country")]}, frame_ref=compare)

    populations = anchor_populations(
        _rolled(tree), _AID_GNI, root_columns=[_COLUMNS, [*_GNI_COLUMNS, "observationAbout_type"]]
    )

    assert populations == [
        {
            "pair": [
                [{"spec_id": _AID, "bindings": {}}],
                [
                    {
                        "spec_id": _GNI,
                        "bindings": {},
                        "levels": {"observationAbout_type": "Country"},
                    }
                ],
            ],
            "keys": {"recipient": "observationAbout"},
        }
    ]


def test_a_level_above_a_verb_of_no_keys_holds_only_where_a_side_carries_it() -> None:
    sums = {
        "op_spec": {"verb": "sum_of", "parameters": {}},
        "inputs": {"frame_ref": {"root": 0}, "right_frame_ref": {"root": 1}},
    }
    tree = _node("filter", {"predicates": [_level("Country")]}, frame_ref=sums)

    populations = anchor_populations(_rolled(tree), _AID_GNI, root_columns=[_TYPED, _COLUMNS])

    assert populations == [
        {"spec_id": _AID, "bindings": {}, "levels": {"observationAbout_type": "Country"}},
        {"spec_id": _GNI, "bindings": {}},
    ]
    # The side the output takes its columns from must carry what is filtered on.
    with pytest.raises(PopulationLevelColumnMissingError):
        anchor_populations(_rolled(tree), _AID_GNI, root_columns=[_COLUMNS, _TYPED])


@pytest.mark.parametrize(
    "slots_named", [True, False], ids=["slots-named", "slots-left-out-as-the-runner-writes"]
)
def test_a_keyed_pairing_whose_sides_both_count_gives_one_pairing_population(
    slots_named: bool,
) -> None:
    tree = _quotient(
        numerator_by=["recipient"],
        denominator_by={"recipient": "observationAbout"},
        numerator_entity={"reporter": "dcid:USA"},
    )
    if not slots_named:
        del tree["op_spec"]["parameters"]["denominator_frame_ref"]

    assert [node["root"] for node in rolling_roots(_rolled(tree))] == [0, 1]
    assert anchor_populations(_rolled(tree), _AID_GNI, root_columns=[_COLUMNS, _GNI_COLUMNS]) == [
        {
            "pair": [
                [{"spec_id": _AID, "bindings": {"reporter": ["dcid:USA"]}}],
                [{"spec_id": _GNI, "bindings": {}}],
            ],
            "keys": {"recipient": "observationAbout"},
        }
    ]


@pytest.mark.parametrize(
    ("verb", "on"),
    [
        ("sum_of", {"recipient": "observationAbout"}),
        # compare's `data` is its right side, so `on` runs from right to left.
        ("compare", {"observationAbout": "recipient"}),
    ],
)
def test_a_slice_pair_paired_by_on_gives_one_pairing_population(
    verb: str, on: dict[str, str]
) -> None:
    tree = _node(
        verb,
        {"right_frame_ref": _RIGHT, "on": on},
        frame_ref={"root": 0},
        right_frame_ref={"root": 1},
    )

    populations = anchor_populations(_rolled(tree), _AID_GNI, root_columns=[_COLUMNS, _GNI_COLUMNS])

    assert populations == [
        {
            "pair": [
                [{"spec_id": _AID, "bindings": {}}],
                [{"spec_id": _GNI, "bindings": {}}],
            ],
            "keys": {"recipient": "observationAbout"},
        }
    ]


@pytest.mark.parametrize(
    "tree",
    [
        # Every numerator key divides by one pooled denominator, so no key is matched.
        _quotient(numerator_by=["recipient"]),
        _node(
            "compare",
            {"right_frame_ref": _RIGHT, "left_entity": {"reporter": "dcid:USA"}},
            frame_ref={"root": 0},
            right_frame_ref={"root": 1},
        ),
    ],
    ids=["pooled-quotient_of", "compare"],
)
def test_an_unkeyed_pairing_passes_each_side_through_flat(tree: dict[str, Any]) -> None:
    populations = anchor_populations(_rolled(tree), _AID_GNI, root_columns=[_COLUMNS, _GNI_COLUMNS])

    assert [population["spec_id"] for population in populations] == [_AID, _GNI]


def test_a_keyed_pairing_with_one_side_counted_passes_that_side_through_flat() -> None:
    tree = _node(
        "quotient_of",
        {
            "denominator_frame_ref": _RIGHT,
            "numerator_by": ["recipient"],
            "denominator_by": {"recipient": "observationAbout"},
        },
        frame_ref=_rolled({"root": 0}),
        denominator_frame_ref={"root": 1},
    )

    assert anchor_populations(tree, _AID_GNI, root_columns=[_COLUMNS, _GNI_COLUMNS]) == [
        {"spec_id": _AID, "bindings": {}}
    ]


def test_a_join_pairs_its_keys_through_right_keys_and_nests_a_pairing_on_a_side() -> None:
    # growth(join(quotient_of(aid, gni) keyed, gni)): the join's left side is itself
    # a pairing, and its right side is gni again under another name.
    tree = _node(
        "join",
        {
            "right_frame_ref": _RIGHT,
            "join_keys": ["recipient"],
            "right_keys": {"recipient": "observationAbout"},
        },
        frame_ref=_quotient(
            numerator_by=["recipient"], denominator_by={"recipient": "observationAbout"}
        ),
        right_frame_ref={"root": 1},
    )

    inner = {
        "pair": [[{"spec_id": _AID, "bindings": {}}], [{"spec_id": _GNI, "bindings": {}}]],
        "keys": {"recipient": "observationAbout"},
    }
    assert anchor_populations(_rolled(tree), _AID_GNI, root_columns=[_COLUMNS, _GNI_COLUMNS]) == [
        {
            "pair": [[inner], [{"spec_id": _GNI, "bindings": {}}]],
            "keys": {"recipient": "observationAbout"},
        }
    ]
    assert [node["root"] for node in rolling_roots(_rolled(tree))] == [0, 1]


def test_a_join_that_leaves_its_keys_to_the_engine_counts_each_side_flat() -> None:
    tree = {
        "op_spec": {"verb": "join", "parameters": {"right_frame_ref": _RIGHT}},
        "inputs": {"frame_ref": {"root": 0}, "right_frame_ref": {"root": 1}},
    }

    assert anchor_populations(_rolled(tree), _AID_GNI, root_columns=[_COLUMNS, _COLUMNS]) == [
        {"spec_id": _AID, "bindings": {}},
        {"spec_id": _GNI, "bindings": {}},
    ]


def test_an_on_with_no_verb_raises_when_both_sides_count() -> None:
    # A plan tree's op-spec carries parameters alone, and `on` runs one way
    # for sum_of and the other for compare.
    tree = {
        "op_spec": {
            "parameters": {"right_frame_ref": _RIGHT, "on": {"recipient": "observationAbout"}}
        },
        "inputs": {"frame_ref": {"root": 0}, "right_frame_ref": {"root": 1}},
    }

    with pytest.raises(PairingKeysUnstatedError):
        anchor_populations(_rolled(tree), _AID_GNI, root_columns=[_COLUMNS, _COLUMNS])
    # The roots a period is taken over are known whatever the keys.
    assert [node["root"] for node in rolling_roots(_rolled(tree))] == [0, 1]


def test_a_pairing_has_one_spelling_and_sorts_after_every_flat_population() -> None:
    flat = {"spec_id": _GNI, "bindings": {}}
    pairing = {
        "pair": [
            [{"spec_id": _GNI}, {"spec_id": _AID, "bindings": {"reporter": ["dcid:USA"]}}],
            [{"spec_id": _GNI, "bindings": {}}, {"spec_id": _GNI}],
        ],
        "keys": {"recipient": "observationAbout", "reporter": "reporter"},
    }

    assert canonical_populations([pairing, flat, pairing]) == [
        flat,
        {
            "pair": [
                [{"spec_id": _AID, "bindings": {"reporter": ["dcid:USA"]}}, flat],
                [flat],
            ],
            "keys": {"recipient": "observationAbout", "reporter": "reporter"},
        },
    ]
    # Which side is which is part of the pairing.
    swapped = {"pair": pairing["pair"][::-1], "keys": pairing["keys"]}
    root = {"spec_id": _AID, "content_version": "cv_" + "1" * 32}
    assert anchor_key(populations=[pairing], roots=[root], rule="half1") != anchor_key(
        populations=[swapped], roots=[root], rule="half1"
    )
    with pytest.raises(ValueError, match="keys"):
        canonical_populations([{**pairing, "keys": {}}])
