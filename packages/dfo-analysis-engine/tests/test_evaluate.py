"""`evaluate` against the verbs called one at a time, and its rolling and refusal rules."""

from __future__ import annotations

import json
from typing import Any

import pytest
from dfo_frame_grammar import Frame, frame_sha256
from frame_fixtures import (
    join_compatible_frame_pair,
    multi_unit_frame,
    plain_single_unit_frame,
    time_series_with_gap_year_frame,
)

from dfo_analysis_engine.calls import derive_frame_content_version
from dfo_analysis_engine.engine.aggregate import run_aggregate
from dfo_analysis_engine.engine.paired import run_paired_compare
from dfo_analysis_engine.engine.rank import run_rank
from dfo_analysis_engine.evaluate import Policy, RootFrame, evaluate
from dfo_analysis_engine.execution_identity import current_execution_identity
from dfo_analysis_engine.frame_ref import FrameRef
from dfo_analysis_engine.identity import (
    canonicalize_op_spec,
    mint_toolkit_content_version,
    op_spec_hash_for,
)
from dfo_analysis_engine.op_spec import OpSpecAdapter
from dfo_analysis_engine.refusal import AnalysisRefused, RefusalCode

_ROOT = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "1" * 32)
_OTHER_ROOT = FrameRef(spec_id="spec_" + "b" * 32, content_version="cv_" + "2" * 32)
_LATEST = {"anchor": "latest", "offset": 0, "rule": "half1"}


def _pair(ref: FrameRef) -> dict[str, str]:
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


def _node(verb: str, inputs: dict[str, Any], **parameters: Any) -> dict[str, Any]:
    """A derived node of `verb` over `inputs`, its op-spec in canonical form. The
    frame slots name placeholder pairs, which `evaluate` replaces."""
    slots = {slot: _pair(_ROOT) for slot in inputs}
    op_spec = OpSpecAdapter.validate_python({"verb": verb, **parameters, **slots})
    return {"op_spec": json.loads(canonicalize_op_spec(op_spec)), "inputs": inputs}


def _root(index: int) -> dict[str, int]:
    return {"root": index}


def _expected_ref(arguments: dict[str, Any], input_content_versions: list[str]) -> FrameRef:
    """The pair the verb called alone mints for `arguments`."""
    spec_id = op_spec_hash_for(OpSpecAdapter.validate_python(arguments))
    return FrameRef(
        spec_id=spec_id,
        content_version=mint_toolkit_content_version(
            op_spec_hash=spec_id,
            input_content_versions=input_content_versions,
            execution_identity=current_execution_identity(),
        ),
    )


def test_a_chain_gives_the_frames_and_ids_of_its_verbs_called_one_at_a_time() -> None:
    frame = plain_single_unit_frame()
    tree = _node(
        "rank",
        {
            "frame_ref": _node(
                "aggregate", {"frame_ref": _root(0)}, aggregation="sum", group_by=["reporter"]
            )
        },
        rank_by=["reporter"],
        aggregation="sum",
        ascending=False,
    )

    evaluation = evaluate(tree, {0: RootFrame(ref=_ROOT, frame=frame)})

    summed = run_aggregate(frame, group_by=["reporter"], aggregation="sum").frame
    summed_ref = _expected_ref(
        {
            "verb": "aggregate",
            "frame_ref": _pair(_ROOT),
            "aggregation": "sum",
            "group_by": ["reporter"],
        },
        [_ROOT.content_version],
    )
    ranked = run_rank(
        summed,
        rank_by=["reporter"],
        aggregation="sum",
        window_start=None,
        window_end=None,
        ascending=False,
    ).frame
    ranked_ref = _expected_ref(
        {
            "verb": "rank",
            "frame_ref": _pair(summed_ref),
            "rank_by": ["reporter"],
            "aggregation": "sum",
            "ascending": False,
        },
        [derive_frame_content_version(summed)],
    )
    assert evaluation.ref == ranked_ref
    assert frame_sha256(evaluation.frame) == frame_sha256(ranked)
    assert evaluation.tree["frame"] == _pair(ranked_ref)
    assert evaluation.tree["inputs"]["frame_ref"]["frame"] == _pair(summed_ref)
    assert evaluation.tree["inputs"]["frame_ref"]["inputs"]["frame_ref"] == {
        "frame": _pair(_ROOT),
        "root": 0,
    }
    assert evaluation.resolved_periods is None


def test_a_summary_states_the_verbs_counts_as_json() -> None:
    oda, gni = join_compatible_frame_pair()
    tree = _node("join", {"frame_ref": _root(0), "right_frame_ref": _root(1)})

    evaluation = evaluate(
        tree, {0: RootFrame(ref=_ROOT, frame=oda), 1: RootFrame(ref=_OTHER_ROOT, frame=gni)}
    )

    summary = json.loads(json.dumps(evaluation.summary))
    assert summary["verb"] == "join"
    assert summary["spec_id"] == evaluation.ref.spec_id
    assert summary["rows"] == evaluation.frame.num_rows
    assert summary["role_keys"] == ["reporter"]
    assert summary["dropped_key_count"] == 0
    assert summary["right_dropped_key_count"] == 0
    assert summary["time"] == {"kind": "observation"}
    assert "frame" not in summary["result"]
    assert summary["execution_identity"]["engine_version"] == "0.9.0"


def _reporter_years(cells: list[tuple[str, str, float]]) -> Frame:
    """A frame of `(reporter, year, value)` cells of one ODA variable."""
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:Oda",
                "reporter": reporter,
                "time_period": year,
                "value": value,
                "unit": "dcid:USDollar",
            }
            for reporter, year, value in cells
        ],
    )


def test_each_step_of_a_chain_names_the_series_it_dropped_for_a_data_gap() -> None:
    """The `quotient_of` below drops FRA, which its denominator lacks, and GBR's 2021
    denominator, which its numerator lacks; the top `growth` then drops GBR, which lacks its
    end year. Each step names its own drops by node."""
    aid = _reporter_years(
        [
            (reporter, year, 10.0)
            for reporter in ("dcid:USA", "dcid:GBR", "dcid:FRA")
            for year in ("2020", "2021")
            if (reporter, year) != ("dcid:GBR", "2021")
        ]
    )
    income = _reporter_years(
        [
            (reporter, year, 100.0)
            for reporter in ("dcid:USA", "dcid:GBR")
            for year in ("2020", "2021")
        ]
    )
    tree = _node(
        "growth",
        {
            "frame_ref": _node(
                "quotient_of",
                {"frame_ref": _root(0), "denominator_frame_ref": _root(1)},
                numerator_by=["reporter"],
                denominator_by={"reporter": "reporter"},
            )
        },
        window_start="2020",
        window_end="2021",
        entity_by=["reporter"],
        change="absolute",
    )

    evaluation = evaluate(
        tree, {0: RootFrame(ref=_ROOT, frame=aid), 1: RootFrame(ref=_OTHER_ROOT, frame=income)}
    )

    assert [(step.node, step.verb) for step in evaluation.steps] == [
        (0, "growth"),
        (1, "quotient_of"),
    ]
    growth, quotient = evaluation.steps
    assert [
        (dict(s.coordinates)["reporter"], s.reason, s.side, s.rows) for s in growth.dropped
    ] == [("dcid:GBR", "endpoint_missing", None, 1)]
    assert [(dict(s.coordinates), s.reason, s.side, s.rows) for s in quotient.dropped] == [
        ({"reporter": "dcid:FRA"}, "unmatched_key", "numerator", 2),
        ({"reporter": "dcid:GBR"}, "period_unmatched", "denominator", 1),
    ]
    assert "series" not in json.dumps(evaluation.summary)


def _flows(role: str, cells: list[tuple[str, str, str, float]]) -> Frame:
    """A frame of `(member, origin, year, value)` cells, its members under `role`."""
    return Frame.build(
        role_keys=[role, "origin"],
        rows=[
            {
                "stat_var": f"dcid:Flow_{role}",
                role: member,
                "origin": origin,
                "time_period": year,
                "value": value,
                "unit": "dcid:USDollar",
            }
            for member, origin, year, value in cells
        ],
    )


def test_a_paired_compare_over_a_derived_side_gives_data_minus_other_per_member() -> None:
    """compare's `data` (its right side) is an aggregate of CRS flows by
    recipient, paired with net ODA by country through `on`."""
    crs = _flows(
        "destination",
        [
            ("country/KEN", "country/DEU", "2022", 200.0),
            ("country/KEN", "country/FRA", "2022", 100.0),
            ("country/GHA", "country/DEU", "2022", 90.0),
        ],
    )
    wdi = Frame.build(
        role_keys=["observationAbout"],
        rows=[
            {
                "stat_var": "dcid:NetOda",
                "observationAbout": place,
                "time_period": "2022",
                "value": value,
                "unit": "dcid:USDollar",
            }
            for place, value in (
                ("country/KEN", 280.0),
                ("country/NGA", 50.0),
                ("country/SOM", 7.0),
            )
        ],
    )
    on = {"destination": "observationAbout"}
    tree = _node(
        "compare",
        {
            "frame_ref": _root(1),
            "right_frame_ref": _node(
                "aggregate", {"frame_ref": _root(0)}, aggregation="sum", group_by=["destination"]
            ),
        },
        on=on,
    )

    evaluation = evaluate(
        tree, {0: RootFrame(ref=_ROOT, frame=crs), 1: RootFrame(ref=_OTHER_ROOT, frame=wdi)}
    )

    summed = run_aggregate(crs, group_by=["destination"], aggregation="sum").frame
    summed_ref = _expected_ref(
        {
            "verb": "aggregate",
            "frame_ref": _pair(_ROOT),
            "aggregation": "sum",
            "group_by": ["destination"],
        },
        [_ROOT.content_version],
    )
    expected_ref = _expected_ref(
        {
            "verb": "compare",
            "frame_ref": _pair(_OTHER_ROOT),
            "right_frame_ref": _pair(summed_ref),
            "on": on,
        },
        [_OTHER_ROOT.content_version, derive_frame_content_version(summed)],
    )
    expected = run_paired_compare(
        wdi,
        summed,
        stat_var=f"dcid:{expected_ref.spec_id}",
        on=on,
        left_entity={},
        right_entity={},
    )
    assert evaluation.ref == expected_ref
    assert frame_sha256(evaluation.frame) == frame_sha256(expected.frame)
    columns = evaluation.frame.table.to_pydict()
    assert (columns["destination"], columns["value"]) == (["country/KEN"], [20.0])
    summary = json.loads(json.dumps(evaluation.summary))
    assert summary["sides"] == {"data": "right", "other": "left"}
    assert summary["parameters"]["on"] == on
    assert summary["role_keys"] == ["destination"]
    assert (summary["dropped_key_count"], summary["right_dropped_key_count"]) == (2, 1)
    assert summary["result"]["right_unmatched_keys"] == [[["destination", "country/GHA"]]]


def _rolling_growth(spec_id: str, *, window_start: str, window_end: str) -> dict[str, Any]:
    return _node(
        "growth",
        {"frame_ref": _root(0 if spec_id == _ROOT.spec_id else 1)},
        window_start=window_start,
        window_end=window_end,
        entity_by=["reporter"],
        rolling={
            "window_start": {**_LATEST, "offset": -2},
            "window_end": _LATEST,
            "population": [{"spec_id": spec_id, "bindings": {}}],
        },
    )


_SERIES = {0: RootFrame(ref=_ROOT, frame=time_series_with_gap_year_frame())}


def test_a_rolling_tree_counts_from_the_latest_period_of_its_roots() -> None:
    evaluation = evaluate(
        _rolling_growth(_ROOT.spec_id, window_start="2022", window_end="2024"), _SERIES
    )

    assert evaluation.resolved_periods is not None
    assert evaluation.resolved_periods.as_json() == {
        "period": "2024",
        "rule": "half1",
        "left_out": [],
    }
    assert evaluation.summary["parameters"]["window_start"] == "2022"


def test_an_exact_rolling_literal_off_the_latest_period_is_refused() -> None:
    tree = _rolling_growth(_ROOT.spec_id, window_start="2021", window_end="2023")

    with pytest.raises(AnalysisRefused) as refused:
        evaluate(tree, _SERIES)

    assert refused.value.code is RefusalCode.ROLLING_ANCHOR_UNRESOLVED
    assert refused.value.refusal.fix["detail"] == "literal_differs"


def test_latest_rewrites_each_rolling_literal_to_the_latest_period() -> None:
    tree = _rolling_growth(_ROOT.spec_id, window_start="2021", window_end="2023")

    rewritten = evaluate(tree, _SERIES, policy=Policy(latest=True))
    exact = evaluate(
        _rolling_growth(_ROOT.spec_id, window_start="2022", window_end="2024"), _SERIES
    )

    assert rewritten.ref == exact.ref
    assert rewritten.tree["op_spec"] == exact.tree["op_spec"]


def test_rolling_branches_over_different_populations_are_refused_in_the_same_year() -> None:
    series = time_series_with_gap_year_frame()
    tree = _node(
        "join",
        {
            "frame_ref": _rolling_growth(_ROOT.spec_id, window_start="2022", window_end="2024"),
            "right_frame_ref": _rolling_growth(
                _OTHER_ROOT.spec_id, window_start="2022", window_end="2024"
            ),
        },
    )
    roots = {0: RootFrame(ref=_ROOT, frame=series), 1: RootFrame(ref=_OTHER_ROOT, frame=series)}

    with pytest.raises(AnalysisRefused) as refused:
        evaluate(tree, roots)

    assert refused.value.code is RefusalCode.ROLLING_POPULATIONS_DIFFER


@pytest.mark.parametrize(
    ("tree", "frame", "code"),
    [
        pytest.param(
            _node("aggregate", {"frame_ref": _root(0)}, aggregation="sum"),
            multi_unit_frame(),
            RefusalCode.UNIT_HETEROGENEOUS,
            id="engine-exception",
        ),
        pytest.param(
            _node(
                "rank",
                {
                    "frame_ref": _node(
                        "rank",
                        {"frame_ref": _root(0)},
                        rank_by=["reporter"],
                        per_period=True,
                        ascending=False,
                    )
                },
                rank_by=["reporter"],
                ascending=False,
            ),
            plain_single_unit_frame(),
            RefusalCode.DERIVED_INPUT_ORDINAL,
            id="rank-over-rank-positions",
        ),
        pytest.param(
            _node(
                "quotient_of",
                {
                    "frame_ref": _node("share_of", {"frame_ref": _root(0)}),
                    "denominator_frame_ref": _root(0),
                },
            ),
            plain_single_unit_frame(),
            RefusalCode.QUOTIENT_OF_SIDE_UNIT_UNSUPPORTED,
            id="quotient-over-a-share",
        ),
    ],
)
def test_a_refused_node_refuses_the_tree(
    tree: dict[str, Any], frame: Frame, code: RefusalCode
) -> None:
    with pytest.raises(AnalysisRefused) as refused:
        evaluate(tree, {0: RootFrame(ref=_ROOT, frame=frame)})

    assert refused.value.code is code
