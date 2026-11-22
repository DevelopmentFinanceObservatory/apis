"""A sum the planner infers for a root reads the codes the filters over it keep: a
`category_match` on a kept `stat_var_<axis>` column, below the nearest step that is no
`filter`, becomes the groupby's `codes` selection, so the sum reads only the StatVars those
filters keep. The answer equals the one read with no selection, row for row, and counts only
the StatVars the selection keeps as summed."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from code_index import CodeIndex
from dfo_analysis_engine.engine import filter as engine_filter

from ._description_doubles import read_record
from .test_analysis_summed_roots import _aggregate, _post, _rig, _two_axis_code_index
from .test_serve_groupby_collapse import _MEMBER_A, _MEMBER_B

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
    """The fields of a 200 body that hold the rows and values, every id and the summed count
    (`_summed`) left out."""
    return {
        "frame_content_version": body["frame_content_version"],
        "output_digest": body["output_digest"],
        "rows": body["rows"],
        "role_keys": body["role_keys"],
        "stat_var_count": body["stat_var_count"],
        "exclusions": body["exclusions"],
        "time": body["time"],
        "result": body["summary"]["result"],
    }


def _summed(body: dict[str, Any]) -> list[int]:
    """The count of StatVars summed into each root of a 200 body that sums more than one."""
    return [one["count"] for one in body["stat_vars_summed"]]


@pytest.mark.parametrize(
    ("column", "values", "groupby", "codes", "summed"),
    [
        pytest.param(
            "stat_var_testAxis",
            ["TestAxisA"],
            _TEST_AXIS,
            {"testAxis": ["TestAxisA"]},
            [],
            id="kept-column",
        ),
        pytest.param(
            "testAxis",
            ["Axis value A"],
            _TEST_AXIS,
            {"testAxis": ["TestAxisA"]},
            [],
            id="axis-label",
        ),
        pytest.param(
            "stat_var_otherAxis",
            ["OtherX"],
            _OTHER_AXIS,
            {"otherAxis": ["OtherX"]},
            [2],
            id="summed",
        ),
    ],
)
@pytest.mark.parametrize("kind", ["dataset", "spec"])
def test_a_filter_on_a_kept_axis_selects_the_codes_its_sum_reads(
    kind: str, column: str, values: list[str], groupby: dict, codes: dict, summed: list[int]
) -> None:
    """The filter's codes, as dcids, join the inferred groupby and so its record, and the
    answer equals that of the same tree over a root stating the groupby with no codes. Each
    counts the StatVars its own sum adds: a sum keeping the axis adds none either way."""
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
    assert _summed(pushed.json()) == _summed(unpushed.json()) == summed


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
    assert _summed(response.json()) == _summed(stated.json())


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


def _with_unheld_code() -> CodeIndex:
    """`_two_axis_code_index` with `TestAxisC`, a `testAxis` code no StatVar holds."""
    base = _two_axis_code_index()
    (held, *_) = (row for row in base.codes if row.code_system == "testAxis")
    return CodeIndex(
        stamp=base.stamp,
        declarations=tuple(base.declarations.values()),
        codes=(*base.codes, replace(held, dcid="TestAxisC", code="C", label="Axis value C")),
        statvars=base.statvars,
    )


def test_a_filter_keeping_only_codes_the_version_lacks_reads_no_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`TestAxisC` is a code of the axis, but no StatVar with data holds it. A selection of it
    would sum no row, so the root reads no selection and the filter lists the values its
    column holds, as over a root stating the groupby. The aggregate over the empty filter
    refuses either way, so the filter's report is read where the engine makes it."""
    reports: list[tuple[Any, ...]] = []
    unmatched = engine_filter._unmatched_breakdown_values

    def recorded(*args: Any) -> tuple[Any, ...]:
        reports.append(unmatched(*args))
        return reports[-1]

    monkeypatch.setattr(engine_filter, "_unmatched_breakdown_values", recorded)
    client, _, root = _rig(two_axes=True, code_index=_with_unheld_code())

    def tree(data: dict) -> dict:
        return _aggregate(_keep(data, ["TestAxisC"]), group_by=["recipient"])

    response = _post(client, tree(root))
    stated = _post(client, tree(_stated(root, _TEST_AXIS)))

    assert response.status_code == stated.status_code == 422, response.text
    assert {key: value for key, value in response.json().items() if key != "trace_id"} == {
        key: value for key, value in stated.json().items() if key != "trace_id"
    }
    pushed, over_stated = reports
    assert pushed == over_stated
    (report,) = pushed
    assert report.column == "stat_var_testAxis"
    assert report.labels == ("TestAxisA", "TestAxisB")


@pytest.mark.parametrize("route", ["analysis", "anchor"])
def test_a_selected_root_names_no_dataset_version_as_its_base(route: str) -> None:
    """A selection can leave a period or an entity out, as a party-totals cut can, so the
    answer and the anchor list no base for a root summed under one, and one for the same sum
    with no selection (conformance C121f)."""
    client, _, root = _rig(two_axes=True)
    selected = _stated(root, {**_TEST_AXIS, "codes": {"testAxis": ["TestAxisA"]}})
    unselected = _stated(root, _TEST_AXIS)

    def bases(read: dict) -> list[str | None]:
        if route == "anchor":
            populations = [{"spec_id": root["spec_id"], "bindings": {}}]
            response = client.post(
                "/api/datasets/analysis/anchor", json={"roots": [read], "populations": populations}
            )
            assert response.status_code == 200, response.text
            return [entry.get("base_content_version") for entry in response.json()["roots"]]
        response = _post(client, _aggregate(read, group_by=["recipient"]))
        assert response.status_code == 200, response.text
        return [entry["base_content_version"] for entry in response.json()["derived_roots"]]

    assert bases(unselected) == [root["content_version"]]
    assert bases(selected) == ([None] if route == "anchor" else [])


def test_a_pushed_root_names_no_dataset_version_as_its_base() -> None:
    client, _, root = _rig(two_axes=True)

    response = _post(client, _aggregate(_keep(root, ["TestAxisA"]), group_by=["recipient"]))

    assert response.status_code == 200, response.text
    assert _groupbys(response.json()) == [{**_TEST_AXIS, "codes": {"testAxis": ["TestAxisA"]}}]
    assert response.json()["derived_roots"] == []


def test_a_selected_level_sum_counts_and_names_only_the_stat_vars_it_selects() -> None:
    """A level roll-up adds its StatVars, so a sum of both members counts two and its derived
    record's `sum_variables` step names both. Selecting one code keeps one StatVar: the answer
    counts none, as fewer than two, and the step names that StatVar alone."""
    client, store, root = _rig()
    level = {"by": ["testAxis", "time_period"], "op": "sum", "levels": {"testAxis": 0}}

    every = _post(client, _aggregate(_stated(root, level), group_by=["recipient"]))
    selected = _post(
        client,
        _aggregate(
            _stated(root, {**level, "codes": {"testAxis": ["TestAxisA"]}}), group_by=["recipient"]
        ),
    )

    assert every.status_code == selected.status_code == 200, (every.text, selected.text)
    assert _summed(every.json()) == [2]
    assert _summed(selected.json()) == []

    def summed_step(body: dict[str, Any]) -> dict[str, Any]:
        (read,) = body["ref"]["members"][0]["roots"]
        record = read_record(
            store, subject_id=root["spec_id"], content_version=read["content_version"]
        )
        assert record is not None and record.steps is not None
        (step,) = (step for step in record.steps if step.op == "sum_variables")
        return step.model_dump(mode="json", exclude_unset=True)["summed"]

    assert summed_step(every.json()) == {
        "count": 2,
        "variables": [{"ref": _MEMBER_A}, {"ref": _MEMBER_B}],
    }
    assert summed_step(selected.json()) == {"count": 1, "variables": [{"ref": _MEMBER_A}]}
