"""A rank partitioned by another role and the sparse-breakdown flag through the analysis route.
The words `partition_by` names read the root summed by them, a partition naming a ranked column
refuses, and an aggregate or rank that sets neither mints the ids an op-spec without the two
fields mints."""

from __future__ import annotations

from typing import Any

from datasets_api.analysis import AnalysisBounds, AnalysisRequest, DatasetRoot, plan_roots

from ._analysis_doubles import FakeRootSource
from .test_analysis_summed_roots import _aggregate, _post, _rig


def _rank(data: dict, **arguments: Any) -> dict:
    return {
        "verb": "rank",
        "arguments": {"aggregation": "sum", **arguments},
        "inputs": {"data": data},
    }


def test_partition_by_words_are_path_words() -> None:
    """The planner reads a root summed by the summable axes a rank names in `partition_by`."""
    root = DatasetRoot(spec_id="spec_" + "a" * 32, content_version="cv_" + "1" * 32)
    tree = _rank(root.model_dump(), rank_by=["origin"], partition_by=["crsPurpose"])
    source = FakeRootSource(frames={}, newest={})

    _, roots = plan_roots(
        AnalysisRequest.model_validate({"tree": tree}), source=source, bounds=AnalysisBounds()
    )

    assert [planned.path_words for planned in roots] == [frozenset({"origin", "crsPurpose"})]


def test_a_partition_naming_a_ranked_column_refuses() -> None:
    client, _, root = _rig(sole=True)

    response = _post(client, _rank(root, rank_by=["recipient"], partition_by=["recipient"]))

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "op_spec_invalid"


def test_an_aggregate_and_rank_setting_no_new_parameter_keep_their_recipe_ids() -> None:
    """A rank over an aggregate that sets neither `partition_by` nor `allow_sparse_breakdowns`
    leaves both out of its op-spec preimages (`HASHED_WHEN_SET`), so its ids are `_RANK_IDS`."""
    client, _, root = _rig(sole=True)
    tree = _rank(_aggregate(root, group_by=["recipient", "time_period"]), rank_by=["recipient"])

    response = _post(client, tree)

    assert response.status_code == 200, response.text
    (member,) = response.json()["ref"]["members"]
    assert (member["spec"], member["content_version"]) == _RANK_IDS


# The recipe ids and content version of a rank over an aggregate of the one-StatVar rig root,
# minted over op-specs that hold no `partition_by` or `allow_sparse_breakdowns` key.
_RANK_IDS = (
    {
        "spec_id": "ops_77858a33592f5e1e3650e3652ccff52b",
        "recipe": {
            "normal_form_hash": "rnf_6631d4431b29d528ac56a82cfe662fce",
            "replay_key": "rpk_c3609a033a8951c9190b8b16f3261327",
        },
    },
    "ocv_3e5deb2b8586c589d845b486e81aed8b",
)
