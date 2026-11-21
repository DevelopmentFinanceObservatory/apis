"""A request root of several StatVars read summed when no step over it names `stat_var`: the
groupby the planner implies from the words its steps name, a summable axis word naming the
`stat_var_<axis>` column a groupby keeps, the count of StatVars summed in the answer and the
derived record, and the refusal that names the summed read when no sum is read."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from code_index import CodeIndex, CodeRow
from datasets_api import analysis_roots
from datasets_api.analysis import (
    AnalysisBounds,
    AnalysisRequest,
    DatasetRoot,
    PlannedRoot,
    SpecRoot,
    plan_roots,
)
from datasets_api.mint_models import MintOptions
from datasets_api.operations.submit import OperationSubmitted
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.version_store import LiveVersionLister
from dfo_platform import ProblemIssue, ProblemSpec
from spec_registry.identity import recompute_identity

from ._analysis_doubles import FakeRootSource
from ._description_doubles import read_record
from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door import _client
from .test_definition_door_groupby_collapse import _hierarchical_code_index
from .test_meta_doors import _operation
from .test_serve_groupby_collapse import (
    _IDENTITY,
    _MEMBER_A,
    _MEMBER_ARCS,
    _MEMBER_B,
    _RECIPIENT,
    _code_index,
    _code_index_no_summable_axis,
    _coverage_service,
    _graph,
    _mint,
    _sdmx_transport,
    _sole_member_sdmx_transport,
    _spec_id_and_sidecar,
    _spec_id_and_sidecar_sole_member,
)

_BARE = {"format": "parquet", "bundle": False, "layout": "frame"}
_SUMMED = {"by": ["time_period"], "op": "sum"}


def _two_axis_code_index() -> CodeIndex:
    """`_code_index` with a second summable axis, `otherAxis`, on which both members hold
    `OtherX`, so a sum keeping `testAxis` collapses `otherAxis` into a frame of its own."""
    base = _code_index()
    (declaration,) = base.declarations.values()
    return CodeIndex(
        stamp=base.stamp,
        declarations=(
            declaration,
            replace(declaration, code_system="otherAxis", plain_name="other axis"),
        ),
        codes=(
            *base.codes,
            CodeRow(
                dcid="OtherX",
                code_system="otherAxis",
                code="X",
                label="Other value X",
                synonyms=(),
                parent=None,
            ),
        ),
        statvars=tuple(
            replace(row, constraint_properties={**row.constraint_properties, "otherAxis": "OtherX"})
            for row in base.statvars
        ),
    )


def _two_axis_graph() -> FakeMintGraphReader:
    arcs = {**_MEMBER_ARCS, "constraintProperties": ("testAxis", "otherAxis")}
    return FakeMintGraphReader(
        arcs={
            _MEMBER_A: {**arcs, "testAxis": ("TestAxisA",), "otherAxis": ("OtherX",)},
            _MEMBER_B: {**arcs, "testAxis": ("TestAxisB",), "otherAxis": ("OtherX",)},
        },
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
    )


def _two_axis_spec_id_and_sidecar() -> tuple[str, dict]:
    other = {"property": "otherAxis", "binding": {"state": "set", "refs": ["OtherX"]}}
    identity = {**_IDENTITY, "slots": [*_IDENTITY["slots"], other]}
    recomputed = recompute_identity(identity)
    return recomputed.spec_id, {
        "schema_version": 1,
        "spec_id": recomputed.spec_id,
        "identity": identity,
    }


def _rig(*, code_index=None, sole: bool = False, two_axes: bool = False):
    """A client over a minted dataset of two StatVars (one with `sole`), each 2020 value of one
    recipient, its version store, and the root naming that version with no enrichment. With
    `two_axes` its StatVars decode a second summable axis (`_two_axis_code_index`)."""
    store = FakeVersionRecordDatasetStore()
    if two_axes:
        code_index = _two_axis_code_index()
    client = _client(
        graph=_two_axis_graph() if two_axes else _graph(),
        sdmx_transport=_sole_member_sdmx_transport() if sole else _sdmx_transport(),
        version_store=store,
        code_index=code_index or _hierarchical_code_index(),
        coverage=_coverage_service(stat_vars=(_MEMBER_A,) if sole else (_MEMBER_A, _MEMBER_B)),
        recipe_store=LiveRecipeStore(store=store),
        version_lister=LiveVersionLister(store=store),
    )
    if two_axes:
        spec_id, sidecar = _two_axis_spec_id_and_sidecar()
    elif sole:
        spec_id, sidecar = _spec_id_and_sidecar_sole_member()
    else:
        spec_id, sidecar = _spec_id_and_sidecar()
    raw = _mint(client, spec_id=spec_id, sidecar=sidecar, options=_BARE)["content_version"]
    root = {"spec_id": spec_id, "content_version": raw, "rendering": {"columns": []}}
    return client, store, root


def _aggregate(data: dict, *, group_by: list[str]) -> dict:
    return {
        "verb": "aggregate",
        "arguments": {"aggregation": "sum", "group_by": group_by},
        "inputs": {"data": data},
    }


def _post(client, tree: dict) -> Any:
    return client.post("/api/datasets/analysis", json={"tree": tree})


@pytest.mark.parametrize("kind", ["dataset", "spec"])
def test_an_aggregate_over_a_root_of_several_stat_vars_sums_them_when_it_is_read(
    kind: str,
) -> None:
    """A root naming no groupby, under steps that name no `stat_var`, is read summed over its
    StatVars: the ref and the recipe state the groupby, the answer and the derived record count
    the StatVars it summed, and the frame is the one a root stating that groupby computes."""
    client, store, root = _rig()
    if kind == "spec":
        root = {key: value for key, value in root.items() if key != "content_version"}

    response = _post(client, _aggregate(root, group_by=["recipient"]))
    stated = _post(
        client,
        _aggregate(
            {**root, "rendering": {"columns": [], "groupby": _SUMMED}}, group_by=["recipient"]
        ),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    (member,) = body["ref"]["members"]
    (read,) = member["roots"]
    assert read["rendering"] == {**_BARE, "groupby": _SUMMED}
    assert body["frame_content_version"] == stated.json()["frame_content_version"]
    assert stated.json()["stat_vars_summed"] == body["stat_vars_summed"]
    assert body["stat_vars_summed"] == [
        {"spec_id": root["spec_id"], "content_version": read["content_version"], "count": 2}
    ]
    assert body["rows"] == 1
    derived = client.get(
        f"/api/datasets/derived/{member['spec']['spec_id']}/{member['content_version']}"
    ).json()
    (record,) = derived["recipe"]["roots"]
    assert record["options"] == {**_BARE, "groupby": _SUMMED}
    description = read_record(
        store, subject_id=member["spec"]["spec_id"], content_version=member["content_version"]
    )
    assert description is not None and description.steps is not None
    retrieve = description.steps[0].model_dump(mode="json", exclude_unset=True)
    assert retrieve["op"] == "retrieve"
    assert retrieve["summed"]["count"] == 2
    assert sorted(one["ref"] for one in retrieve["summed"]["variables"]) == [_MEMBER_A, _MEMBER_B]


def test_a_summable_axis_word_keeps_the_axis_and_names_its_column() -> None:
    """A step naming the summable axis `testAxis` keeps it in the implied groupby, and the word
    names the `stat_var_testAxis` column the groupby keeps, so the sum splits by it."""
    client, _, root = _rig(two_axes=True)

    response = _post(client, _aggregate(root, group_by=["recipient", "testAxis"]))

    assert response.status_code == 200, response.text
    body = response.json()
    (read,) = body["ref"]["members"][0]["roots"]
    assert read["rendering"]["groupby"] == {"by": ["testAxis"], "op": "sum"}
    assert body["rows"] == 2
    assert "stat_var_testAxis" in body["role_keys"]


def test_a_root_under_a_step_naming_stat_var_is_read_unsummed_and_refused_with_the_sum() -> None:
    """A filter naming `stat_var` keeps the root unsummed, so the aggregate over it refuses
    `stat_var_unresolved`, and the refusal names the summed read."""
    client, _, root = _rig()
    filtered = {
        "verb": "filter",
        "arguments": {
            "predicates": [
                {
                    "kind": "category_match",
                    "column": "stat_var",
                    "values": [f"dcid:{_MEMBER_A}", f"dcid:{_MEMBER_B}"],
                }
            ]
        },
        "inputs": {"data": root},
    }

    response = _post(client, _aggregate(filtered, group_by=["recipient"]))

    assert response.status_code == 422, response.text
    problem = response.json()
    assert problem["code"] == "stat_var_unresolved"
    assert problem["fix"]["groupby"] == _SUMMED
    assert "groupby_refused" not in problem["fix"]


def test_a_root_whose_sum_the_door_refuses_is_read_unsummed_and_names_the_refusal() -> None:
    """With no summable axis declared the datasets door refuses the implied sum
    `rollup_not_summable`, so the root is read unsummed and the engine refuses
    `stat_var_unresolved`, naming the summed read and the door's precondition."""
    client, _, root = _rig(code_index=_code_index_no_summable_axis())

    response = _post(client, _aggregate(root, group_by=["recipient"]))

    assert response.status_code == 422, response.text
    problem = response.json()
    assert problem["code"] == "stat_var_unresolved"
    assert problem["fix"]["groupby"] == _SUMMED
    assert problem["fix"]["groupby_refused"] == [
        {
            "spec_id": root["spec_id"],
            "content_version": root["content_version"],
            "axis": "stat_var",
            "precondition": "no_summable_axis",
        }
    ]


_ROLLUP_DETAIL = "groupby axis 'stat_var' refused: no summable axis"


@pytest.mark.parametrize(
    ("refusal_code", "refusal", "status", "precondition"),
    [
        pytest.param(
            "rollup_not_summable",
            ProblemSpec(
                status=422,
                code="rollup_not_summable",
                retryable=False,
                detail=_ROLLUP_DETAIL,
                issues=[ProblemIssue(axis="stat_var")],
            ),
            422,
            {"axis": "stat_var", "precondition": _ROLLUP_DETAIL},
            id="recorded",
        ),
        pytest.param(
            "rollup_not_summable",
            None,
            422,
            {"axis": "unrecorded", "precondition": "unrecorded"},
            id="code-alone",
        ),
        pytest.param("over_ceiling", None, 500, None, id="another-failure-refuses"),
    ],
)
def test_a_spec_root_whose_queued_sum_fails_not_summable_is_read_unsummed(
    monkeypatch: pytest.MonkeyPatch,
    refusal_code: str,
    refusal: ProblemSpec | None,
    status: int,
    precondition: dict[str, str] | None,
) -> None:
    """A Spec root's summed rebuild that ran as a queued operation and failed
    `rollup_not_summable` reads the root unsummed, as a sum refused inline does, so the engine
    refuses `stat_var_unresolved` naming the refused sum. Any other failure still refuses."""
    client, _, root = _rig()
    spec_root = {"spec_id": root["spec_id"], "rendering": root["rendering"]}
    serve = analysis_roots.serve_spec

    def queued_sum(*, options: MintOptions, **kwargs: Any) -> Any:
        if options.groupby is None:
            return serve(options=options, **kwargs)
        failed = _operation(state="failed", refusal_code=refusal_code, refusal=refusal)
        raise OperationSubmitted(operation=failed)

    monkeypatch.setattr(analysis_roots, "serve_spec", queued_sum)

    response = _post(client, _aggregate(spec_root, group_by=["recipient"]))

    assert response.status_code == status, response.text
    problem = response.json()
    if precondition is None:
        assert problem["code"] == "internal_error"
        return
    assert problem["code"] == "stat_var_unresolved"
    (refused,) = problem["fix"]["groupby_refused"]
    assert refused["spec_id"] == root["spec_id"]
    assert {key: refused[key] for key in ("axis", "precondition")} == precondition


def test_a_root_of_one_stat_var_keeps_its_recipe_ids() -> None:
    """A root of one StatVar is read as it was before the planner implied sums: under two sets
    of path words it is one root, its record states no groupby, and the frame keeps the ids a
    run of this tree had then."""
    client, _, root = _rig(sole=True)
    tree = {
        "verb": "share_of",
        "arguments": {"on": ["recipient"]},
        "inputs": {
            "data": _aggregate(root, group_by=["recipient", "time_period"]),
            "other": root,
        },
    }

    response = _post(client, tree)

    assert response.status_code == 200, response.text
    (member,) = response.json()["ref"]["members"]
    assert [read["rendering"] for read in member["roots"]] == [_BARE]
    assert response.json()["stat_vars_summed"] == []
    assert (member["spec"], member["content_version"]) == _SOLE_IDS


# The ids this tree's frame had before the planner implied sums.
_SOLE_IDS = (
    {
        "spec_id": "ops_f810d2b04f345cbe4aa80ce276ad820c",
        "recipe": {
            "normal_form_hash": "rnf_c7a657e19e68385b52e938965d1deee6",
            "replay_key": "rpk_ab785f6e0fdcc2f3f880d95bd189e94a",
        },
    },
    "ocv_555fd80d73b51ec8fc84ad8add102891",
)


@pytest.mark.parametrize(
    ("verb", "arguments", "data", "other"),
    [
        pytest.param(
            "share_of",
            {"on": ["destination"]},
            {"destination", "crsPurpose"},
            {"destination"},
            id="share_of-sides-apart",
        ),
        pytest.param(
            "compare",
            {"on": {"destination": "recipient"}},
            {"destination", "crsPurpose"},
            {"recipient", "crsPurpose"},
            id="compare-inherits-the-words-above",
        ),
    ],
)
def test_each_side_of_a_two_reading_step_gets_its_own_path_words(
    verb: str, arguments: dict, data: set[str], other: set[str]
) -> None:
    """The data side of a two-reading step reads the words above it and its own, the other side
    its own, and under `compare` and `sum_of` the words above it too."""
    root = DatasetRoot(spec_id="spec_" + "a" * 32, content_version="cv_" + "1" * 32)
    two_sided = {
        "verb": verb,
        "arguments": arguments,
        "inputs": {
            "data": root.model_dump(),
            "other": SpecRoot(spec_id="spec_" + "a" * 32).model_dump(),
        },
    }
    tree = {
        "verb": "aggregate",
        "arguments": {"aggregation": "sum", "group_by": ["crsPurpose"]},
        "inputs": {"data": two_sided},
    }
    source = FakeRootSource(frames={}, newest={})

    _, roots = plan_roots(
        AnalysisRequest.model_validate({"tree": tree}), source=source, bounds=AnalysisBounds()
    )

    assert [planned.path_words for planned in roots] == [frozenset(data), frozenset(other)]
    assert all(isinstance(planned, PlannedRoot) for planned in roots)
