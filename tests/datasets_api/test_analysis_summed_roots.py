"""A request root of several StatVars read summed when no step over it names `stat_var`: the
groupby the planner implies from the words its steps name, a summable axis word naming the
`stat_var_<axis>` column a groupby keeps, the count of StatVars summed in the answer and the
derived record, and the refusal that names the summed read when no sum is read."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from artifact_store.errors import ArtifactNotFoundError
from code_index import CodeIndex, CodeRow
from dataset_store import snapshot
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api import analysis, analysis_roots, current, store_errors
from datasets_api import render as render_module
from datasets_api import version_store as version_store_module
from datasets_api import versions as versions_module
from datasets_api.analysis import (
    AnalysisBounds,
    AnalysisRequest,
    DatasetRoot,
    PlannedRoot,
    SpecRoot,
    plan_roots,
)
from datasets_api.config import DatasetsBuildSettings
from datasets_api.mint_models import MintOptions
from datasets_api.operations.submit import OperationSubmitted
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.version_store import LiveVersionLister
from dfo_platform import ProblemIssue, ProblemSpec
from dfo_platform.cloud_logging import JSON_FIELDS_ATTR
from materialise.trace import ResolvedStatVar
from spec_registry.identity import recompute_identity

from . import test_serve_party_totals as party_totals
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
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

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


def _rig(
    *,
    code_index=None,
    sole: bool = False,
    two_axes: bool = False,
    sdmx_transport=None,
    settings=None,
):
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
        settings=settings,
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


def test_a_spec_root_under_two_word_sets_reads_one_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Spec root reached under two sets of words is served in each summed rendering the
    stored head's trace implies, both at the version the first serve read, so one request
    reads one version of the dataset."""
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
    assert served == [
        {"by": ["testAxis", "time_period"], "op": "sum", "levels": None},
        {"by": ["time_period"], "op": "sum", "levels": None},
    ]
    reads = response.json()["ref"]["members"][0]["roots"]
    assert len({read["content_version"] for read in reads}) == 2
    derived = response.json()["derived_roots"]
    assert len(derived) == 2
    assert len({one["base_content_version"] for one in derived}) == 1


@pytest.mark.parametrize("stated", [False, True], ids=["inferred", "stated"])
def test_a_version_whose_trace_counts_no_observations_is_summed_by_its_frame(
    monkeypatch: pytest.MonkeyPatch, stated: bool
) -> None:
    """A version minted before traces counted each StatVar's observations is read summed when
    its stored frame holds several StatVars, and the answer counts the StatVars the sum
    added, inferred or stated. The frame's `stat_var` column is read once, and alone."""
    monkeypatch.setattr(
        ResolvedStatVar,
        "as_trace_entry",
        lambda self: {"ref": self.ref, "roles": list(self.roles)},
    )
    client, _, root = _rig()
    if stated:
        root = {**root, "rendering": {"columns": [], "groupby": _SUMMED}}
    projected: list[object] = []
    read = store_errors.formats.read_canonical_frame_table

    def counted(data: bytes, *, columns=None):
        projected.append(columns)
        return read(data, columns=columns)

    monkeypatch.setattr(store_errors.formats, "read_canonical_frame_table", counted)

    response = _post(client, _aggregate(root, group_by=["recipient"]))

    assert response.status_code == 200, response.text
    body = response.json()
    assert [summed["count"] for summed in body["stat_vars_summed"]] == [2]
    assert body["rows"] == 1
    assert projected == [["stat_var"]]


def test_a_version_whose_trace_counts_no_observations_and_whose_frame_is_gone_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the stored frame gone, the StatVars of an old trace are its resolved list, and the
    sum that needs the frame refuses as an expired root does, never a server error."""
    monkeypatch.setattr(
        ResolvedStatVar,
        "as_trace_entry",
        lambda self: {"ref": self.ref, "roles": list(self.roles)},
    )
    client, _, root = _rig()

    def gone(self, *, artifact_digest: str, tier: object) -> bytes:
        raise ArtifactNotFoundError(digest=artifact_digest, tier=tier)

    monkeypatch.setattr(_InventoryRecordingArtifactStore, "get", gone)

    response = _post(client, _aggregate(root, group_by=["recipient"]))

    assert response.status_code == 410, response.text
    assert response.json()["code"] == "artifact_expired"


def test_a_version_whose_trace_counts_no_observations_past_the_fold_bound_decodes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Counting the StatVars of an old trace from a stored frame past `groupby_fold_max_rows`,
    the bound of the sum that would read it, refuses on the frame's Parquet row count before
    any of its cells is decoded."""
    monkeypatch.setattr(
        ResolvedStatVar,
        "as_trace_entry",
        lambda self: {"ref": self.ref, "roles": list(self.roles)},
    )
    client, _, root = _rig(
        settings=DatasetsBuildSettings.from_env().model_copy(update={"groupby_fold_max_rows": 1})
    )
    decoded: list[object] = []
    read = store_errors.formats._checked_canonical_table

    def counted(data: bytes, *, columns, dictionary: bool):
        decoded.append(columns)
        return read(data, columns=columns, dictionary=dictionary)

    monkeypatch.setattr(store_errors.formats, "_checked_canonical_table", counted)

    response = _post(client, _aggregate(root, group_by=["recipient"]))

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "over_ceiling"
    assert decoded == []


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
    aggregate by it answers one row per value. A re-key sums nothing, so the answer counts no
    StatVars summed and the version's record holds no `sum_variables` step."""
    client, store, root = _rig()

    response = _post(client, _aggregate(root, group_by=[word]))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rows"] == 2
    assert "stat_var_testAxis" in body["role_keys"]
    assert body["stat_vars_summed"] == []
    (derived,) = body["derived_roots"]
    record = read_record(
        store, subject_id=derived["spec_id"], content_version=derived["content_version"]
    )
    assert record is not None and record.steps is not None
    assert "sum_variables" not in [step.op for step in record.steps]


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


def test_a_later_newest_root_of_a_dataset_whose_frame_expired_is_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two readings of one Spec root, with the head's stored frame expired. The first root's
    serve rebuilds what it reads, and the second root is served at the version that serve
    read, rebuilding what it needs as the first did, so neither root refuses as expired."""
    client, _, root = _rig(two_axes=True)
    spec_root = {"spec_id": root["spec_id"], "rendering": root["rendering"]}
    assert _post(client, _filter(spec_root)).status_code == 200
    cutoff = datetime.now(UTC)
    monkeypatch.setattr(
        snapshot, "working_tier_expired", lambda stored_at, *, now: stored_at <= cutoff
    )
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


@pytest.mark.parametrize(
    ("predicate", "summed"),
    [
        pytest.param(
            {"kind": "ordered_threshold", "column": "value", "op": "gt", "value": 750.0},
            False,
            id="value",
        ),
        pytest.param(
            {"kind": "category_match", "column": "status", "values": ["E"]}, False, id="status"
        ),
        pytest.param(
            {"kind": "category_match", "column": "measurementMethod", "values": ["M"]},
            False,
            id="measurement-method",
        ),
        pytest.param(
            {"kind": "category_match", "column": "recipient", "values": ["country/KEN"]},
            True,
            id="a-role",
        ),
    ],
)
def test_a_filter_on_an_observation_column_under_a_sum_reads_the_root_as_stated(
    predicate: dict, summed: bool
) -> None:
    """A predicate on a column each observation holds apart, its value, status or facets,
    keeps other rows before a sum than after it, so the root under it is read as stated. A
    predicate on a role keeps the same rows either way, and the sum stays."""
    root = SpecRoot(spec_id="spec_" + "a" * 32).model_dump()
    tree = _aggregate(
        {"verb": "filter", "arguments": {"predicates": [predicate]}, "inputs": {"data": root}},
        group_by=["recipient"],
    )

    _, (planned,) = plan_roots(
        AnalysisRequest.model_validate({"tree": tree}),
        source=FakeRootSource(frames={}, newest={}),
        bounds=AnalysisBounds(),
    )

    assert (planned.path_words is not None) is summed


@pytest.mark.parametrize(
    ("predicate", "names_stat_var"),
    [
        pytest.param(
            {
                "kind": "category_match",
                "column": "stat_var",
                "values": [f"dcid:{_MEMBER_A}", f"dcid:{_MEMBER_B}"],
            },
            True,
            id="words-name-stat-var",
        ),
        pytest.param(
            {"kind": "ordered_threshold", "column": "value", "op": "gt", "value": 1.0},
            False,
            id="a-value-filter",
        ),
    ],
)
def test_the_summed_read_advice_names_the_cause_that_left_the_root_unsummed(
    predicate: dict, names_stat_var: bool
) -> None:
    """A root left unsummed over two StatVars is told to read them summed. Only a root whose
    steps name `stat_var` is told to stop naming it; any other is given the groupby alone."""
    client, _, root = _rig()
    tree = _aggregate(
        {"verb": "filter", "arguments": {"predicates": [predicate]}, "inputs": {"data": root}},
        group_by=["recipient"],
    )

    response = _post(client, tree)

    assert response.status_code == 422, response.text
    problem = response.json()
    assert problem["code"] == "stat_var_unresolved", problem
    assert problem["fix"].get("groupby") == _SUMMED, problem
    assert ("name no 'stat_var'" in problem["detail"]) is names_stat_var


@pytest.mark.parametrize("route", ["analysis", "anchor"])
def test_a_party_totals_cut_root_names_no_dataset_version_as_its_base(route: str) -> None:
    """A cut that leaves totals out answers a version of its own, and a cut can leave a
    period out, so it stands in for no dataset version: the answer and the anchor list no
    base for it, and a caller matching an anchor of the raw version refuses it."""
    store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=party_totals._graph(),
        sdmx_transport=party_totals._sdmx_transport(),
        version_store=store,
        code_index=party_totals._code_index(),
        coverage=party_totals._coverage_service(),
        head_writer=_FakeHeadStore(),
        recipe_store=LiveRecipeStore(store=store),
        version_lister=LiveVersionLister(store=store),
    )
    spec_id, sidecar = party_totals._spec_id_and_sidecar()
    raw = party_totals._mint(client, spec_id=spec_id, sidecar=sidecar, options=_BARE)
    root = {
        "spec_id": spec_id,
        "content_version": raw["content_version"],
        "rendering": {"columns": [], "without_party_totals": True},
    }

    if route == "anchor":
        populations = [{"spec_id": spec_id, "bindings": {}}]
        response = client.post(
            "/api/datasets/analysis/anchor", json={"roots": [root], "populations": populations}
        )
        assert response.status_code == 200, response.text
        (read,) = response.json()["roots"]
        assert read["content_version"] != raw["content_version"]
        assert "base_content_version" not in read
        return
    response = _post(client, _aggregate(root, group_by=["time_period"]))

    assert response.status_code == 200, response.text
    assert response.json()["derived_roots"] == []


def _count_work(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """A counter of the renders of a stored frame, which a root's sum is one of, and of the
    runs that read root frames and evaluate a tree, from the call on."""
    work: Counter[str] = Counter()

    def counted(name: str, call: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            work[name] += 1
            return call(*args, **kwargs)

        return wrapped

    for module, name in ((current, "_render_stored_rendering"), (analysis, "_evaluated")):
        monkeypatch.setattr(module, name, counted(name, getattr(module, name)))
    return work


@pytest.mark.parametrize("kind", ["dataset", "spec"])
@pytest.mark.parametrize(
    "rendering",
    [{"columns": []}, {"columns": [], "without_party_totals": True}, {}],
    ids=["stored_frame", "cut", "default_columns"],
)
def test_a_repeated_request_over_a_summed_root_renders_and_evaluates_nothing(
    monkeypatch: pytest.MonkeyPatch, kind: str, rendering: dict
) -> None:
    """The sum a root is read with is a version of its own, found by its base version and
    minting once a request has rendered it, so the same request again reads its stored
    answer: no sum is rendered, no root frame read and no tree evaluated."""
    client, _, root = _rig()
    root = {**root, "rendering": rendering}
    if kind == "spec":
        root = {key: value for key, value in root.items() if key != "content_version"}
    tree = _aggregate(_filter(root), group_by=["recipient"])
    first = _post(client, tree)
    assert first.status_code == 200, first.text
    assert [summed["count"] for summed in first.json()["stat_vars_summed"]] == [2]
    # A cut can leave a period or an entity out, so a summed root that cuts names no base
    # (conformance C121f).
    assert len(first.json()["derived_roots"]) == (0 if rendering.get("without_party_totals") else 1)

    work = _count_work(monkeypatch)
    again = _post(client, tree)

    assert again.status_code == 200, again.text
    assert again.json() == first.json()
    assert work == Counter()


def test_a_repeated_spec_root_request_reads_the_version_a_moved_head_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Spec root's repeat is answered from its stored answer while its head stands. Once the
    head is past its age and the source holds other values, the repeat rebuilds and answers
    the new version's sum."""
    body = {"text": _CSV_BODY}
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=body["text"],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )
    client, _, root = _rig(sdmx_transport=transport)
    tree = _aggregate(
        {"spec_id": root["spec_id"], "rendering": root["rendering"]}, group_by=["recipient"]
    )
    first = _post(client, tree)
    assert first.status_code == 200, first.text
    body["text"] = _CSV_BODY.replace(",2020,1000,", ",2020,600,")

    standing = _post(client, tree)
    monkeypatch.setattr(current, "_head_is_fresh", lambda head, *, now, deps: False)
    monkeypatch.setattr(current, "_max_age", lambda head, *, deps: timedelta(0))
    moved = _post(client, tree)

    assert standing.json() == first.json()
    assert moved.status_code == 200, moved.text
    (before,) = first.json()["derived_roots"]
    (after,) = moved.json()["derived_roots"]
    assert after["base_content_version"] != before["base_content_version"]
    assert moved.json()["frame_content_version"] != first.json()["frame_content_version"]


def _derived_version(store: FakeVersionRecordDatasetStore) -> dict[str, Any]:
    """The one derived version `store` records: its version, base and record row."""
    with store.transaction() as tx:
        (row,) = tx.fetch_all(
            "SELECT * FROM datasets.version_record WHERE base_content_version IS NOT NULL"
        )
    return row


def _revoke(store: FakeVersionRecordDatasetStore, spec_id: str, content_version: str) -> None:
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_revocation "
            "(subject_id, content_version, reason, revoked_at) VALUES (%s, %s, %s, %s)",
            (spec_id, content_version, "takedown", "2026-01-01T00:00:00+00:00"),
        )


def _swapping_transport(body: dict[str, str]) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=body["text"],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )


# The two members' 2020 values swapped: another dataset version whose sum is the same.
_SWAPPED_BODY = (
    _CSV_BODY.replace(",2020,1000,", ",2020,FIRST,")
    .replace(",2020,500,", ",2020,1000,")
    .replace(",2020,FIRST,", ",2020,500,")
)


def test_two_mintings_of_one_dataset_version_each_read_their_own_recorded_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two sums of one dataset version, keeping different axes, are two recorded versions. In
    either order, each repeat reads the version its own sum recorded, with no render."""
    client, _, root = _rig(two_axes=True)
    by_recipient = _aggregate(root, group_by=["recipient"])
    by_axis = _aggregate(root, group_by=["recipient", "testAxis"])
    first = {"recipient": _post(client, by_recipient), "axis": _post(client, by_axis)}
    assert first["recipient"].status_code == first["axis"].status_code == 200
    assert (
        first["recipient"].json()["frame_content_version"]
        != (first["axis"].json()["frame_content_version"])
    )

    work = _count_work(monkeypatch)
    repeats = [
        ("axis", _post(client, by_axis)),
        ("recipient", _post(client, by_recipient)),
        ("recipient", _post(client, by_recipient)),
        ("axis", _post(client, by_axis)),
    ]

    for name, repeat in repeats:
        assert repeat.json() == first[name].json(), name
    assert work["_render_stored_rendering"] == 0


@pytest.mark.parametrize("moved", ["code_index", "render_format"])
def test_a_recorded_sum_is_read_again_only_under_the_stamp_it_was_rendered_under(
    monkeypatch: pytest.MonkeyPatch, moved: str
) -> None:
    """A sum's levels, cut and summable axes follow the code index, and its cells the render
    format, so a recorded sum answers a repeat only under the stamp of both it was rendered
    under (`versions.rendering_stamp`): under another index or format the root renders once,
    which records the new stamp, and the next repeat renders nothing."""
    index = _hierarchical_code_index()
    client, _, root = _rig(code_index=index)
    tree = _aggregate(_filter(root), group_by=["recipient"])
    first = _post(client, tree)
    assert first.status_code == 200, first.text

    if moved == "code_index":
        monkeypatch.setattr(index, "stamp", replace(index.stamp, content_hash="sha256:another"))
    else:
        monkeypatch.setattr(versions_module, "RENDER_FORMAT", "another_format")
    work = _count_work(monkeypatch)
    moved = _post(client, tree)
    renders_under_another_index = work["_render_stored_rendering"]
    work.clear()
    again = _post(client, tree)

    assert moved.status_code == again.status_code == 200, moved.text
    assert again.json()["frame_content_version"] == first.json()["frame_content_version"]
    assert renders_under_another_index == 1
    assert work["_render_stored_rendering"] == 0


def test_a_newest_recorded_sum_that_cannot_serve_renders_in_place_of_an_older_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the newest version recorded for a dataset version and minting is read. When its
    stored frame is not servable the root renders, and an older recorded version does not
    answer in its place."""
    client, store, root = _rig()
    tree = _aggregate(root, group_by=["recipient"])
    first = _post(client, tree)
    assert first.status_code == 200, first.text
    recorded = _derived_version(store)
    newer = {
        **recorded,
        "content_version": "cv_newer",
        "recorded_at": recorded["recorded_at"] + timedelta(seconds=1),
    }
    # The newer version links the sum's file with the same head and index, and no stored frame.
    with store.transaction() as tx:
        rows = {
            table: [
                {**row, "content_version": "cv_newer"}
                for row in tx.fetch_all(
                    f"SELECT * FROM datasets.{table} WHERE content_version = %s",
                    (recorded["content_version"],),
                )
            ]
            for table in ("version_artifact", "version_artifact_head")
        }
        sums = {
            row["rendering_key"]
            for row in rows["version_artifact"]
            if "groupby" in row["rendering"]
        }
        for table, row in [
            ("version_record", newer),
            *(("version_artifact", row) for row in rows["version_artifact"]),
            *(("version_artifact_head", row) for row in rows["version_artifact_head"]),
        ]:
            if table != "version_record" and row["rendering_key"] not in sums:
                continue
            tx.execute(
                f"INSERT INTO datasets.{table} ({', '.join(row)}) "
                f"VALUES ({', '.join('%s' for _ in row)})",
                tuple(
                    json.dumps(value) if isinstance(value, list) else value
                    for value in row.values()
                ),
            )

    work = _count_work(monkeypatch)
    again = _post(client, tree)

    assert again.status_code == 200, again.text
    assert again.json()["frame_content_version"] == recorded["content_version"]
    assert work["_render_stored_rendering"] == 1


@pytest.mark.parametrize("expired", ["file", "every_file"], ids=["its_file", "its_frame_too"])
def test_a_recorded_sum_whose_files_expired_is_rendered_again(
    monkeypatch: pytest.MonkeyPatch, expired: str
) -> None:
    """A recorded sum past its working tier, its dataset version still servable, is not read:
    the root renders. With only its file expired the render links a new one. With its stored
    frame expired too, the render makes the same version and stores its frame again, which
    renews the frame's location (`<digest>#r<n>`), so the answer stands either way."""
    client, store, root = _rig()
    tree = _aggregate(root, group_by=["recipient"])
    first = _post(client, tree)
    assert first.status_code == 200, first.text
    derived = _derived_version(store)["content_version"]
    frame_kept = 'AND rendering NOT LIKE \'%"format": "frame"%\'' if expired == "file" else ""
    with store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_location SET stored_at = %s WHERE artifact_digest IN ("
            "SELECT artifact_digest FROM datasets.version_artifact WHERE content_version = %s "
            f"AND rendering LIKE '%groupby%' {frame_kept})"
            if expired == "file"
            else "UPDATE datasets.artifact_location SET stored_at = %s WHERE artifact_digest IN ("
            "SELECT artifact_digest FROM datasets.version_artifact WHERE content_version = %s)",
            ("2000-01-01T00:00:00+00:00", derived),
        )

    work = _count_work(monkeypatch)
    again = _post(client, tree)

    assert again.status_code == 200, again.text
    assert again.json() == first.json()
    assert work["_render_stored_rendering"] == 1
    if expired == "every_file":
        with store.transaction() as tx:
            renewals = {
                row["renewal"]
                for row in tx.fetch_all(
                    "SELECT renewal FROM datasets.artifact_location WHERE artifact_digest IN ("
                    "SELECT artifact_digest FROM datasets.version_artifact "
                    'WHERE content_version = %s AND rendering LIKE \'%"format": "frame"%\')',
                    (derived,),
                )
            }
        assert renewals == {1}


def test_a_revoked_recorded_sum_is_passed_over_and_the_root_renders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A revoked recorded sum, its dataset version not revoked, is passed over and the root
    renders it from the dataset version's stored frame (conformance C121a). The render makes
    the same version, so the answer stands: revoking the dataset version withdraws it."""
    client, store, root = _rig()
    tree = _aggregate(root, group_by=["recipient"])
    first = _post(client, tree)
    assert first.status_code == 200, first.text
    _revoke(store, root["spec_id"], _derived_version(store)["content_version"])

    work = _count_work(monkeypatch)
    again = _post(client, tree)

    assert again.status_code == 200, again.text
    assert again.json() == first.json()
    assert work["_render_stored_rendering"] == 1


def _moved_head_of_the_same_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, FakeVersionRecordDatasetStore, dict, str, str]:
    """A Spec root's client after its head moved to a version whose sum is the first head's:
    the render from the new head links its file under the version the first head recorded,
    whose record keeps the first base. Returns the client, its store, the Spec root, that
    recorded version and the new head."""
    body = {"text": _CSV_BODY}
    client, store, root = _rig(sdmx_transport=_swapping_transport(body))
    spec_root = {"spec_id": root["spec_id"], "rendering": root["rendering"]}
    first = _post(client, _aggregate(spec_root, group_by=["recipient"]))
    assert first.status_code == 200, first.text
    body["text"] = _SWAPPED_BODY
    with monkeypatch.context() as stale:
        stale.setattr(current, "_head_is_fresh", lambda head, *, now, deps: False)
        stale.setattr(current, "_max_age", lambda head, *, deps: timedelta(0))
        moved = _post(client, _aggregate(spec_root, group_by=["recipient"]))
    assert moved.status_code == 200, moved.text
    recorded = _derived_version(store)
    assert moved.json()["frame_content_version"] == recorded["content_version"]
    assert recorded["base_content_version"] == root["content_version"]
    with store.transaction() as tx:
        (new_head,) = {
            row["head_content_version"]
            for row in tx.fetch_all(
                "SELECT head_content_version FROM datasets.version_artifact_head "
                "WHERE head_content_version != %s",
                (root["content_version"],),
            )
        }
    return client, store, spec_root, recorded["content_version"], new_head


def test_a_repeated_request_over_a_moved_head_of_the_same_sum_names_the_new_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored answer is keyed on the dataset version each root was read from, so the same
    tree over a moved head whose sum is unchanged names the new head as its base, and its
    repeat reads that answer with no render and no evaluation."""
    client, _, spec_root, recorded, new_head = _moved_head_of_the_same_sum(monkeypatch)

    work = _count_work(monkeypatch)
    again = _post(client, _aggregate(spec_root, group_by=["recipient"]))

    assert again.status_code == 200, again.text
    assert work == Counter()
    (read,) = again.json()["derived_roots"]
    assert (read["content_version"], read["base_content_version"]) == (recorded, new_head)


@pytest.mark.parametrize("tree", ["filtered", "repeated"])
def test_a_spec_root_read_from_a_recorded_sum_names_the_head_it_was_read_through(
    monkeypatch: pytest.MonkeyPatch, tree: str
) -> None:
    """A Spec root read from a recorded version through a head other than the base its record
    keeps names that head as the dataset version it sums. The tree the first head answered,
    repeated, answers under another key, since the key holds each root's base version, so no
    stored body names the first head."""
    client, _, spec_root, recorded, new_head = _moved_head_of_the_same_sum(monkeypatch)
    data = _filter(spec_root) if tree == "filtered" else spec_root

    standing = _post(client, _aggregate(data, group_by=["recipient"]))

    assert standing.status_code == 200, standing.text
    assert standing.json()["frame_content_version"] == recorded
    (read,) = standing.json()["derived_roots"]
    assert read["base_content_version"] == new_head


def test_a_head_that_moves_between_two_roots_of_one_derived_version_keeps_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two roots of one Spec that both read one recorded sum, the head moving between their
    serves to another dataset version of the same sum: the second serve reads the same derived
    version through the new head, so the second root is read from the version the first serve
    read, and both name that version as their base."""
    heads: list[tuple[Any, Any]] = []
    read_head = _FakeHeadStore.read_head

    def recorded(store: Any, subject_id: str) -> Any:
        head = read_head(store, subject_id)
        heads.append((store, head))
        return head

    monkeypatch.setattr(_FakeHeadStore, "read_head", recorded)
    client, _, spec_root, _, new_head = _moved_head_of_the_same_sum(monkeypatch)
    store = heads[-1][0]
    moved = store._heads[spec_root["spec_id"]]
    assert moved.content_version == new_head
    first = next(head for _, head in heads if head is not None and head.content_version != new_head)
    store._heads[spec_root["spec_id"]] = first
    serve = analysis_roots.serve_spec
    serves: list[None] = []

    def moving(**kwargs: Any) -> Any:
        serves.append(None)
        if len(serves) == 2:
            store._heads[spec_root["spec_id"]] = moved
        return serve(**kwargs)

    monkeypatch.setattr(analysis_roots, "serve_spec", moving)
    tree = {
        "verb": "share_of",
        "arguments": {"on": ["recipient"]},
        "inputs": {
            "data": _aggregate(spec_root, group_by=["recipient"]),
            # Another rendering, so the two roots stay two, of the same derived version.
            "other": _aggregate({**spec_root, "rendering": {}}, group_by=["recipient"]),
        },
    }

    response = _post(client, tree)

    assert response.status_code == 200, response.text
    assert len(serves) == 2
    derived = response.json()["derived_roots"]
    assert {one["base_content_version"] for one in derived} == {first.content_version}


def test_a_recorded_sum_is_not_read_through_a_revoked_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recorded version whose record names another base is not read through a revoked head:
    a root stating the sum refuses `snapshot_revoked` as a render from that head does."""
    client, store, spec_root, _, new_head = _moved_head_of_the_same_sum(monkeypatch)
    _revoke(store, spec_root["spec_id"], new_head)
    # A stated sum, since a revoked head implies none (`LiveRootSource._head_version`).
    stated = {**spec_root, "rendering": {**spec_root["rendering"], "groupby": _SUMMED}}

    revoked = _post(client, _aggregate(stated, group_by=["recipient"]))

    assert revoked.status_code == 410, revoked.text
    assert revoked.json()["code"] == "snapshot_revoked"


def test_a_repeated_request_reads_its_recorded_sum_in_one_store_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lookup reads the recorded version, its revocation and its locations in one read
    (`read_version_state`), and no second read of its locations alone."""
    client, store, root = _rig()
    tree = _aggregate(root, group_by=["recipient"])
    assert _post(client, tree).status_code == 200
    derived = _derived_version(store)["content_version"]
    reads: Counter[str] = Counter()

    def counted(name: str) -> Callable[..., Any]:
        call = getattr(LiveSnapshotReader, name)

        def wrapped(reader: Any, subject_id: str, content_version: str, **kwargs: Any) -> Any:
            if content_version == derived:
                reads[name] += 1
            return call(reader, subject_id, content_version, **kwargs)

        return wrapped

    for name in ("read_artifact_locations", "read_version_state"):
        monkeypatch.setattr(LiveSnapshotReader, name, counted(name))
    again = _post(client, tree)

    assert again.status_code == 200, again.text
    assert reads == Counter({"read_version_state": 1})


@pytest.mark.parametrize(
    "stamped",
    [versions_module, version_store_module],
    ids=["recorded-sum-stamp", "frame-file-key"],
)
def test_a_sum_recorded_under_an_earlier_render_format_renders_once_with_code_cells(
    monkeypatch: pytest.MonkeyPatch, stamped: Any
) -> None:
    """A sum recorded while `stat_var_<axis>` cells held labels, under an earlier answer format
    and an earlier render format in its head rows' stamp (`versions.rendering_stamp`) or in its
    frame-layout file's link key (`version_store.rendering_key`), is not read again: either
    alone makes the root render once, the answer is the one a current build gives, and the next
    repeat renders nothing."""

    def posted(client: Any, root: dict) -> Any:
        return _post(client, _aggregate(_filter(root), group_by=["recipient", "testAxis"]))

    control_client, _, control_root = _rig()
    control = posted(control_client, control_root)
    assert control.status_code == 200, control.text

    client, store, root = _rig()
    with monkeypatch.context() as earlier:
        build = render_module.build_decode_plan

        def label_cells(table: Any, *, index: CodeIndex, extra_refs: Any = ()) -> Any:
            plan = build(table, index=index, extra_refs=extra_refs)
            if plan is None:
                return None
            return replace(
                plan,
                lookup={
                    ref: {
                        system: (code.label if (code := index.code_by_dcid(cell)) else cell)
                        for system, cell in cells.items()
                    }
                    for ref, cells in plan.lookup.items()
                },
            )

        earlier.setattr(render_module, "build_decode_plan", label_cells)
        earlier.setattr(stamped, "RENDER_FORMAT", "axis_label")
        earlier.setattr(analysis, "ANSWER_FORMAT", "analysis_answer/3")
        old = posted(client, root)
    assert old.status_code == 200, old.text
    assert old.json()["frame_content_version"] != control.json()["frame_content_version"]

    work = _count_work(monkeypatch)
    renewed = posted(client, root)
    renders = work["_render_stored_rendering"]
    work.clear()
    again = posted(client, root)

    assert renewed.status_code == again.status_code == 200, renewed.text
    assert renders == 1
    assert renewed.json()["frame_content_version"] == control.json()["frame_content_version"]
    assert again.json() == renewed.json()
    assert work == Counter()
    with store.transaction() as tx:
        derived = tx.fetch_all(
            "SELECT content_version FROM datasets.version_record "
            "WHERE base_content_version IS NOT NULL"
        )
    assert len(derived) == 2
