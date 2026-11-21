"""A request root of several StatVars read summed when no step over it names `stat_var`: the
groupby the planner implies from the words its steps name, a summable axis word naming the
`stat_var_<axis>` column a groupby keeps, the count of StatVars summed in the answer and the
derived record, and the refusal that names the summed read when no sum is read."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import timedelta
from typing import Any

import httpx
import pytest

from code_index import CodeIndex, CodeRow
from datasets_api import analysis_roots, current
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
from dfo_platform.cloud_logging import JSON_FIELDS_ATTR
from materialise.trace import ResolvedStatVar
from spec_registry.identity import recompute_identity

from ._analysis_doubles import FakeRootSource
from ._description_doubles import _FakeHeadStore, read_record
from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door import _client
from .test_definition_door_groupby_collapse import _hierarchical_code_index
from .test_meta_doors import _operation
from .test_serve_groupby_collapse import (
    _CSV_BODY,
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


def _rig(*, code_index=None, sole: bool = False, two_axes: bool = False, sdmx_transport=None):
    """A client over a minted dataset of two StatVars (one with `sole`), each 2020 value of one
    recipient, its version store, and the root naming that version with no enrichment. With
    `two_axes` its StatVars decode a second summable axis (`_two_axis_code_index`)."""
    store = FakeVersionRecordDatasetStore()
    if two_axes:
        code_index = _two_axis_code_index()
    client = _client(
        graph=_two_axis_graph() if two_axes else _graph(),
        sdmx_transport=sdmx_transport
        or (_sole_member_sdmx_transport() if sole else _sdmx_transport()),
        version_store=store,
        head_writer=_FakeHeadStore(),
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


def _filter(data: dict) -> dict:
    """A filter keeping every row from 2000 on."""
    return {
        "verb": "filter",
        "arguments": {
            "predicates": [
                {"kind": "ordered_threshold", "column": "time_period", "op": "gte", "value": "2000"}
            ]
        },
        "inputs": {"data": data},
    }


def _post(client, tree: dict) -> Any:
    return client.post("/api/datasets/analysis", json={"tree": tree})


@pytest.mark.parametrize("kind", ["dataset", "spec"])
def test_an_aggregate_over_a_root_of_several_stat_vars_sums_them_when_it_is_read(
    kind: str,
) -> None:
    """A root naming no groupby, under steps that name no `stat_var`, is read summed over its
    StatVars. The ref and the recipe state the groupby, the answer and the derived record count
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
    assert read["rendering"]["groupby"] == {"by": ["testAxis", "time_period"], "op": "sum"}
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
    `stat_var_unresolved`, naming the door's precondition and no summed read, which would
    refuse again."""
    client, _, root = _rig(code_index=_code_index_no_summable_axis())

    response = _post(client, _aggregate(root, group_by=["recipient"]))

    assert response.status_code == 422, response.text
    problem = response.json()
    assert problem["code"] == "stat_var_unresolved"
    assert "groupby" not in problem["fix"]
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
    # A filter alone reads the root as stated, which serves the dataset and writes its head.
    assert _post(client, _filter(spec_root)).status_code == 200
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
    """A root of one StatVar is read as stated, with no implied sum: under two sets of path
    words it is one root, its record states no groupby, and the frame has the ids in
    `_SOLE_IDS`."""
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


# The ids of this tree's frame, minted from a read that states no groupby.
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
        pytest.param(
            "join",
            {"join_keys": ["destination"]},
            {"destination", "crsPurpose"},
            {"destination"},
            id="join-keys-name-both-sides",
        ),
        pytest.param(
            "join",
            {"join_keys": ["destination"], "right_keys": {"destination": "recipient"}},
            {"destination", "crsPurpose"},
            {"recipient"},
            id="right-keys-name-the-other-side",
        ),
    ],
)
def test_each_side_of_a_two_reading_step_gets_its_own_path_words(
    verb: str, arguments: dict, data: set[str], other: set[str]
) -> None:
    """The data side of a two-reading step reads the words above it and its own, the other side
    its own, and under `compare` and `sum_of` the words above it too. A join's `join_keys` key
    both sides unless `right_keys` names the other side's columns."""
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


def test_a_root_under_filters_alone_keeps_every_stat_var() -> None:
    """A filter selects rows and reads a frame of several StatVars, so a root under filters
    alone is read as stated: each StatVar keeps its row and the read states no groupby."""
    client, _, root = _rig()

    response = _post(client, _filter(_filter(root)))

    assert response.status_code == 200, response.text
    body = response.json()
    (read,) = body["ref"]["members"][0]["roots"]
    assert "groupby" not in read["rendering"]
    assert body["rows"] == 2
    assert body["stat_vars_summed"] == []


def test_an_inferred_sum_keeping_an_axis_is_the_stated_one() -> None:
    """The groupby the planner infers for a step naming `testAxis` is spelled as a caller
    following the refusal's summed read states it, so both read one rendering."""
    client, _, root = _rig(two_axes=True)
    stated = {
        **root,
        "rendering": {"columns": [], "groupby": {"by": ["time_period", "testAxis"], "op": "sum"}},
    }

    inferred = _post(client, _aggregate(root, group_by=["recipient", "testAxis"]))
    named = _post(client, _aggregate(stated, group_by=["recipient", "testAxis"]))

    assert inferred.status_code == named.status_code == 200, inferred.text
    (inferred_read,) = inferred.json()["ref"]["members"][0]["roots"]
    (named_read,) = named.json()["ref"]["members"][0]["roots"]
    assert inferred_read == named_read
    assert inferred.json()["ref"]["members"][0]["spec"] == named.json()["ref"]["members"][0]["spec"]


def test_a_refused_sum_logs_the_root_it_reads_unsummed(caplog: pytest.LogCaptureFixture) -> None:
    client, _, root = _rig(code_index=_code_index_no_summable_axis())

    with caplog.at_level(logging.INFO, logger="datasets_api.analysis_roots"):
        _post(client, _aggregate(root, group_by=["recipient"]))

    (fields,) = [
        getattr(record, JSON_FIELDS_ATTR)
        for record in caplog.records
        if getattr(record, JSON_FIELDS_ATTR, {}).get("event")
        == "datasets_analysis_root_sum_refused"
    ]
    assert fields == {
        "event": "datasets_analysis_root_sum_refused",
        "spec_id": root["spec_id"],
        "content_version": root["content_version"],
        "by": ["time_period"],
        "axis": "stat_var",
        "precondition": "no_summable_axis",
        "operation_id": None,
    }


def test_two_request_roots_reading_one_version_stay_two_roots() -> None:
    """A Spec root and a dataset root naming the version it reads resolve to one record, and
    stay two roots of the recipe, as two roots the request names apart."""
    client, _, root = _rig(sole=True)
    spec_root = {"spec_id": root["spec_id"], "rendering": root["rendering"]}
    tree = {
        "verb": "share_of",
        "arguments": {"on": ["recipient"]},
        "inputs": {
            "data": _aggregate(spec_root, group_by=["recipient", "time_period"]),
            "other": root,
        },
    }

    response = _post(client, tree)

    assert response.status_code == 200, response.text
    (member,) = response.json()["ref"]["members"]
    derived = client.get(
        f"/api/datasets/derived/{member['spec']['spec_id']}/{member['content_version']}"
    ).json()
    assert len(derived["recipe"]["roots"]) == 2


def test_a_root_merged_under_two_word_sets_keeps_the_refused_sum() -> None:
    """One root read unsummed under `stat_var` words and again after the door refused its sum
    is one root, and the refusal over it names the refused sum."""
    client, _, root = _rig(code_index=_code_index_no_summable_axis())
    tree = {
        "verb": "share_of",
        "arguments": {"on": ["recipient"]},
        "inputs": {
            "data": _aggregate(root, group_by=["recipient", "stat_var"]),
            "other": _aggregate(root, group_by=["recipient"]),
        },
    }

    response = _post(client, tree)

    assert response.status_code == 422, response.text
    problem = response.json()
    assert problem["code"] == "stat_var_unresolved"
    assert [one["axis"] for one in problem["fix"]["groupby_refused"]] == ["stat_var"]


def test_a_spec_root_under_two_word_sets_serves_its_head_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Spec root reached under two sets of words is served once, in the summed rendering the
    stored head's trace implies, and its other reading renders from the version that serve
    read, so one request reads one version of the dataset."""
    client, _, root = _rig(two_axes=True)
    spec_root = {"spec_id": root["spec_id"], "rendering": root["rendering"]}
    assert _post(client, _filter(spec_root)).status_code == 200
    serve = analysis_roots.serve_spec
    served: list[object] = []

    def counted(*, options: MintOptions, **kwargs: Any) -> Any:
        served.append(None if options.groupby is None else options.groupby.model_dump(mode="json"))
        return serve(options=options, **kwargs)

    monkeypatch.setattr(analysis_roots, "serve_spec", counted)
    tree = {
        "verb": "share_of",
        "arguments": {"on": ["recipient"]},
        "inputs": {
            "data": _aggregate(spec_root, group_by=["recipient", "testAxis"]),
            "other": _aggregate(spec_root, group_by=["recipient"]),
        },
    }

    response = _post(client, tree)

    assert response.status_code == 200, response.text
    assert len(served) == 1
    assert served[0]["by"] == ["testAxis", "time_period"]
    reads = response.json()["ref"]["members"][0]["roots"]
    assert len({read["content_version"] for read in reads}) == 2


@pytest.mark.parametrize("stated", [False, True], ids=["inferred", "stated"])
def test_a_version_whose_trace_counts_no_observations_is_summed_by_its_frame(
    monkeypatch: pytest.MonkeyPatch, stated: bool
) -> None:
    """A version minted before traces counted each StatVar's observations is read summed when
    its stored frame holds several StatVars, and the answer counts the StatVars the sum
    added, inferred or stated."""
    monkeypatch.setattr(
        ResolvedStatVar,
        "as_trace_entry",
        lambda self: {"ref": self.ref, "roles": list(self.roles)},
    )
    client, _, root = _rig()
    if stated:
        root = {**root, "rendering": {"columns": [], "groupby": _SUMMED}}

    response = _post(client, _aggregate(root, group_by=["recipient"]))

    assert response.status_code == 200, response.text
    body = response.json()
    assert [summed["count"] for summed in body["stat_vars_summed"]] == [2]
    assert body["rows"] == 1


@pytest.mark.parametrize("route", ["analysis", "anchor"])
def test_a_summed_root_names_the_dataset_version_its_rendering_derives_from(route: str) -> None:
    """A summed rendering answers a version of its own. The answer and the anchor name, beside
    it, the dataset version it derives from, so a caller holding that version, as an anchor
    records it, matches the rendering to it."""
    client, _, root = _rig()
    summed = {**root, "rendering": {"columns": [], "groupby": _SUMMED}}
    if route == "anchor":
        populations = [{"spec_id": root["spec_id"], "bindings": {}}]
        response = client.post(
            "/api/datasets/analysis/anchor", json={"roots": [summed], "populations": populations}
        )
    else:
        response = _post(client, _aggregate(root, group_by=["recipient"]))

    assert response.status_code == 200, response.text
    (read,) = response.json()["roots" if route == "anchor" else "derived_roots"]
    assert read["spec_id"] == root["spec_id"]
    assert read["content_version"] != root["content_version"]
    assert read["base_content_version"] == root["content_version"]


def _above(data: dict, threshold: float) -> dict:
    """A filter keeping the rows whose value is above `threshold`."""
    return {
        "verb": "filter",
        "arguments": {
            "predicates": [
                {"kind": "ordered_threshold", "column": "value", "op": "gt", "value": threshold}
            ]
        },
        "inputs": {"data": data},
    }


def test_a_value_filter_under_a_sum_reads_as_a_saved_filter_does() -> None:
    """The two StatVars hold 1,000 and 500. A value threshold keeps observations, so it runs
    on the StatVars' own rows inline as it does saved: the sum over 750 is 1,000 both ways."""
    client, _, root = _rig()

    inline = _post(client, _aggregate(_above(root, 750.0), group_by=["recipient"]))
    saved = _post(client, _above(root, 750.0))
    assert saved.status_code == 200, saved.text
    over_saved = _post(client, _aggregate({"ref": saved.json()["ref"]}, group_by=["recipient"]))

    assert inline.status_code == over_saved.status_code == 200, (inline.text, over_saved.text)
    assert inline.json()["frame_content_version"] == over_saved.json()["frame_content_version"]
    assert inline.json()["stat_vars_summed"] == []


@pytest.mark.parametrize("word", ["testAxis", "stat_var_testAxis"])
def test_grouping_by_the_only_summable_axis_reads_one_row_per_axis_value(word: str) -> None:
    """The frame's one summable axis is kept, so the sum adds nothing: each axis value holds
    one StatVar. The read is still the single-measure frame keyed by that axis, so an
    aggregate by it answers one row per value."""
    client, _, root = _rig()

    response = _post(client, _aggregate(root, group_by=[word]))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rows"] == 2
    assert "stat_var_testAxis" in body["role_keys"]


def test_a_revoked_stale_head_reads_the_version_its_rebuild_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The head names a version since revoked and past its age, and the source now holds
    other values. A Spec root reads the rebuilt version, summed, as a root of a dataset with
    no head does; a root pinned to the revoked version still refuses."""
    body = {"text": _CSV_BODY}
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=body["text"],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )
    client, store, root = _rig(sdmx_transport=transport)
    spec_root = {"spec_id": root["spec_id"], "rendering": root["rendering"]}
    assert _post(client, _filter(spec_root)).status_code == 200
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_revocation "
            "(subject_id, content_version, reason, revoked_at) VALUES (%s, %s, %s, %s)",
            (root["spec_id"], root["content_version"], "takedown", "2026-01-01T00:00:00+00:00"),
        )
    body["text"] = _CSV_BODY.replace(",2020,1000,", ",2020,600,")
    monkeypatch.setattr(current, "_head_is_fresh", lambda head, *, now, deps: False)
    monkeypatch.setattr(current, "_max_age", lambda head, *, deps: timedelta(0))

    fresh = _post(client, _aggregate(spec_root, group_by=["recipient"]))
    pinned = _post(client, _aggregate(root, group_by=["recipient"]))

    assert fresh.status_code == 200, fresh.text
    (read,) = fresh.json()["ref"]["members"][0]["roots"]
    assert read["rendering"] == {**_BARE, "groupby": _SUMMED}
    assert [summed["count"] for summed in fresh.json()["stat_vars_summed"]] == [2]
    assert pinned.status_code == 410, pinned.text
    assert pinned.json()["code"] == "snapshot_revoked"
