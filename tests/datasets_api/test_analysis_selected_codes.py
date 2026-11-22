"""A sum the planner infers for a root reads the codes the filters over it keep: a
`category_match` on a kept `stat_var_<axis>` column, below the nearest step that is no
`filter`, becomes the groupby's `codes` selection, so the sum reads only the StatVars those
filters keep. The answer equals the one read with no selection, row for row."""

from __future__ import annotations

from typing import Any

import pytest

from .test_analysis_summed_roots import _aggregate, _post, _rig

_TEST_AXIS = {"by": ["testAxis", "time_period"], "op": "sum"}
_OTHER_AXIS = {"by": ["otherAxis", "time_period"], "op": "sum"}


def _keep(data: dict, values: list[str], *, column: str = "stat_var_testAxis") -> dict:
    """A filter keeping the rows whose `column` cell is one of `values`."""
    return {
        "verb": "filter",
        "arguments": {
            "predicates": [{"kind": "category_match", "column": column, "values": values}]
        },
        "inputs": {"data": data},
    }


def _stated(root: dict, groupby: dict) -> dict:
    return {**root, "rendering": {**root["rendering"], "groupby": groupby}}


def _groupbys(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [read["rendering"]["groupby"] for read in body["ref"]["members"][0]["roots"]]


def _answer(body: dict[str, Any]) -> dict[str, Any]:
    """The fields of a 200 body that hold the rows and values, every id left out."""
    return {
        "frame_content_version": body["frame_content_version"],
        "output_digest": body["output_digest"],
        "rows": body["rows"],
        "role_keys": body["role_keys"],
        "stat_var_count": body["stat_var_count"],
        "exclusions": body["exclusions"],
        "time": body["time"],
        "result": body["summary"]["result"],
        "stat_vars_summed": [one["count"] for one in body["stat_vars_summed"]],
    }


@pytest.mark.parametrize(
    ("column", "values", "groupby", "codes"),
    [
        pytest.param(
            "stat_var_testAxis",
            ["TestAxisA"],
            _TEST_AXIS,
            {"testAxis": ["TestAxisA"]},
            id="kept-column",
        ),
        pytest.param(
            "testAxis", ["Axis value A"], _TEST_AXIS, {"testAxis": ["TestAxisA"]}, id="axis-label"
        ),
        pytest.param(
            "stat_var_otherAxis", ["OtherX"], _OTHER_AXIS, {"otherAxis": ["OtherX"]}, id="summed"
        ),
    ],
)
@pytest.mark.parametrize("kind", ["dataset", "spec"])
def test_a_filter_on_a_kept_axis_selects_the_codes_its_sum_reads(
    kind: str, column: str, values: list[str], groupby: dict, codes: dict
) -> None:
    """The filter's codes, as dcids, join the inferred groupby and so its record, and the
    answer equals that of the same tree over a root stating the groupby with no codes."""
    client, _, root = _rig(two_axes=True)
    if kind == "spec":
        root = {key: value for key, value in root.items() if key != "content_version"}

    pushed = _post(client, _aggregate(_keep(root, values, column=column), group_by=["recipient"]))
    unpushed = _post(
        client,
        _aggregate(_keep(_stated(root, groupby), values, column=column), group_by=["recipient"]),
    )

    assert pushed.status_code == unpushed.status_code == 200, (pushed.text, unpushed.text)
    assert _groupbys(pushed.json()) == [{**groupby, "codes": codes}]
    assert _groupbys(unpushed.json()) == [groupby]
    assert _answer(pushed.json()) == _answer(unpushed.json())


def test_two_filters_on_one_axis_select_the_codes_both_keep() -> None:
    client, _, root = _rig(two_axes=True)
    tree = _keep(_keep(root, ["TestAxisA", "TestAxisB"]), ["TestAxisA"])

    pushed = _post(client, _aggregate(tree, group_by=["recipient"]))

    assert pushed.status_code == 200, pushed.text
    assert _groupbys(pushed.json()) == [{**_TEST_AXIS, "codes": {"testAxis": ["TestAxisA"]}}]
    assert pushed.json()["summary"]["result"]["group_summaries"][0]["value"] == 1000.0


def _over_a_stated_groupby(root: dict) -> dict:
    return _aggregate(_keep(_stated(root, _TEST_AXIS), ["TestAxisA"]), group_by=["recipient"])


def _shared(root: dict) -> dict:
    """One root read by two branches that keep different codes."""
    return {
        "verb": "share_of",
        "arguments": {"on": ["recipient"]},
        "inputs": {
            "data": _aggregate(_keep(root, ["TestAxisA"]), group_by=["recipient"]),
            "other": _aggregate(_keep(root, ["TestAxisB"]), group_by=["recipient"]),
        },
    }


def _unmatched(root: dict) -> dict:
    return _aggregate(_keep(root, ["no such code", "TestAxisA"]), group_by=["recipient"])


def _after_a_reduce(root: dict) -> dict:
    """A filter above an aggregate that keeps the axis."""
    inner = _aggregate(root, group_by=["recipient", "testAxis"])
    return _aggregate(_keep(inner, ["TestAxisA"]), group_by=["recipient"])


@pytest.mark.parametrize(
    "tree",
    [
        pytest.param(_over_a_stated_groupby, id="stated"),
        pytest.param(_shared, id="two-branches"),
        pytest.param(_unmatched, id="unmatched-value"),
        pytest.param(_after_a_reduce, id="after-a-reduce"),
    ],
)
def test_a_root_is_read_with_no_selection_where_a_filter_does_not_select(tree: Any) -> None:
    """A stated groupby, a root two branches filter apart, a filter value naming no code and a
    filter above a step that reduces the rows select nothing: each root reads the groupby
    with no codes, and the answer is the one over a root stating it."""
    client, _, root = _rig(two_axes=True)

    response = _post(client, tree(root))
    stated = _post(client, tree(_stated(root, _TEST_AXIS)))

    assert response.status_code == stated.status_code == 200, (response.text, stated.text)
    assert _groupbys(response.json()) == _groupbys(stated.json())
    assert all("codes" not in groupby for groupby in _groupbys(response.json()))
    assert _answer(response.json()) == _answer(stated.json())


def test_a_value_naming_no_code_refuses_as_with_no_selection() -> None:
    """A filter keeping no row refuses the aggregate over it as it does over a root stating
    the groupby."""
    client, _, root = _rig(two_axes=True)

    def tree(data: dict) -> dict:
        return _aggregate(_keep(data, ["no such code"]), group_by=["recipient"])

    response = _post(client, tree(root))
    stated = _post(client, tree(_stated(root, _TEST_AXIS)))

    assert response.status_code == stated.status_code == 422, response.text
    assert {key: value for key, value in response.json().items() if key != "trace_id"} == {
        key: value for key, value in stated.json().items() if key != "trace_id"
    }


def test_a_derived_root_read_with_a_selection_replays_to_its_own_answer() -> None:
    """The stored recipe's root record states the selection, so the derived frame computed
    again from it reads the same rendering and keeps its ids and its rows."""
    client, _, root = _rig(two_axes=True)
    first = _post(client, _aggregate(_keep(root, ["TestAxisA"]), group_by=["recipient"]))
    assert first.status_code == 200, first.text
    (member,) = first.json()["ref"]["members"]

    again = _post(
        client, {"spec_id": member["spec"]["spec_id"], "content_version": member["content_version"]}
    )

    assert again.status_code == 200, again.text
    (replayed,) = again.json()["ref"]["members"]
    assert (replayed["spec"], replayed["content_version"]) == (
        member["spec"],
        member["content_version"],
    )
    assert _groupbys(again.json()) == _groupbys(first.json())
    assert _answer(again.json()) == _answer(first.json())
