"""A filter value on a breakdown column (`<axis>` or `stat_var_<axis>`) given as a code dcid, a
publisher code or a label naming one code of the axis, resolved before the request key
(`analysis.resolved_breakdown_values`); an ambiguous label refused with its candidates; and the
codes an unmatched filter names beside the dcids its column holds; and a stored recipe's label
value matched to a frame whose cells hold dcids (`analysis._Engine.matched`)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from code_index import CodeIndex
from datasets_api import analysis
from datasets_api.analysis import StepNode, resolved_breakdown_values
from datasets_api.config import DatasetsBuildSettings
from tests.code_index.support import INDEX_FIXTURE, load_index

from .test_analysis_summed_roots import _aggregate, _count_work, _post, _rig
from .test_definition_door_groupby_collapse import _hierarchical_code_index

# The purpose labels fragility-finance filters `stat_var_crsPurpose` by (the website's
# fragility-finance `_data/live.json` `purposes`), and hiv-financing's loans filter.
_FRAGILITY_PURPOSES = (
    "emergency food assistance",
    "immediate post-emergency reconstruction and rehabilitation",
    "material relief assistance and service",
    "multi-hazard response preparedness",
    "relief co-ordination and support service",
    "anti-corruption organisations and institutions",
    "child soldier prevention and demobilisation",
    "civilian peace-building, conflict prevention and resolution",
    "decentralisation and support to subnational government",
    "facilitation of orderly, safe, regular and responsible migration and mobility",
    "human rights institutions and mechanisms",
    "legal and judicial development",
    "legislature and parliament strengthening",
    "participation in international peacekeeping operations",
    "public finance management",
    "public sector policy and administrative management",
    "reintegration and small arms and light weapons control",
    "removal of land mines and explosive remnants of war",
    "security system management and reform",
    "support for democratic participation and civil society",
    "support for media and free flow of information",
    "support to end violence against women and girls",
    "women's rights organisations, movements, and institutions",
)


def _keep(data: dict, column: str, values: list[str]) -> dict:
    return {
        "verb": "filter",
        "arguments": {
            "predicates": [{"kind": "category_match", "column": column, "values": values}]
        },
        "inputs": {"data": data},
    }


def _step(column: str, values: list[str]) -> StepNode:
    root = {"spec_id": "spec_" + "a" * 32, "content_version": "cv_" + "1" * 32}
    return StepNode.model_validate(_keep(root, column, values))


def _sent(step: StepNode) -> list[str]:
    (predicate,) = step.arguments["predicates"]  # ty: ignore[not-iterable]
    return predicate["values"]  # ty: ignore[invalid-argument-type, not-subscriptable]


def _sharing(index: CodeIndex, *, synonym: str | None = None, label: str | None = None):
    """`index` with both test-axis codes given `synonym` or the one `label`."""
    codes = tuple(
        replace(
            row,
            synonyms=(*row.synonyms, synonym) if synonym else row.synonyms,
            label=label or row.label,
        )
        for row in index.codes
    )
    return CodeIndex(
        stamp=index.stamp,
        declarations=tuple(index.declarations.values()),
        codes=codes,
        statvars=index.statvars,
    )


@pytest.mark.parametrize("column", ["testAxis", "stat_var_testAxis"])
def test_a_dcid_a_publisher_code_and_a_label_key_one_answer(
    monkeypatch: pytest.MonkeyPatch, column: str
) -> None:
    """Each spelling of the code `TestAxisA` reads the rows of that code alone, and the
    spellings after the first answer from the first one's stored answer, evaluating nothing."""
    client, _, root = _rig()

    first = _post(client, _aggregate(_keep(root, column, ["TestAxisA"]), group_by=["recipient"]))
    assert first.status_code == 200, first.text
    work = _count_work(monkeypatch)
    again = [
        _post(client, _aggregate(_keep(root, column, [value]), group_by=["recipient"]))
        for value in ("A", "axis value a", "Axis value A")
    ]

    assert first.json()["rows"] == 1
    assert [one.json() for one in again] == [first.json()] * 3
    assert work["_evaluated"] == 0
    everything = _post(client, _aggregate(root, group_by=["recipient"]))
    assert everything.json()["frame_content_version"] != first.json()["frame_content_version"]


def test_an_ambiguous_label_refuses_with_its_candidates() -> None:
    """A text two codes of the axis hold refuses before any frame is read, naming both."""
    client, _, root = _rig(code_index=_sharing(_hierarchical_code_index(), synonym="shared"))

    response = _post(
        client, _aggregate(_keep(root, "testAxis", ["Shared"]), group_by=["recipient"])
    )

    assert response.status_code == 422, response.text
    problem = response.json()
    assert problem["code"] == "breakdown_label_ambiguous"
    assert problem["retry_mode"] == "correct_and_retry"
    assert problem["fix"]["column"] == "testAxis"
    assert problem["fix"]["candidates"] == [
        {"dcid": "TestAxisA", "label": "Axis value A"},
        {"dcid": "TestAxisB", "label": "Axis value B"},
    ]


def test_a_verification_records_an_ambiguous_label_as_its_refusal() -> None:
    """A verification whose filter label names two codes answers 200, the refusal recorded."""
    settings = DatasetsBuildSettings.from_env().model_copy(update={"verify_principal": "probe"})
    client, _, root = _rig(
        code_index=_sharing(_hierarchical_code_index(), synonym="shared"), settings=settings
    )
    tree = _aggregate(_keep(root, "testAxis", ["Shared"]), group_by=["recipient"])

    response = client.post(
        "/api/datasets/analysis",
        json={"tree": tree, "verify": True},
        headers={"x-dfo-principal": "probe"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["outcome"], body["refusal_code"]) == ("not_stored", "breakdown_label_ambiguous")


def test_a_label_matched_exactly_wins_over_a_synonym() -> None:
    """A text that is one code's label and another code's synonym names the labelled code."""
    index = _hierarchical_code_index()
    codes = tuple(
        replace(row, synonyms=("Axis value A",)) if row.dcid == "TestAxisB" else row
        for row in index.codes
    )
    index = CodeIndex(
        stamp=index.stamp,
        declarations=tuple(index.declarations.values()),
        codes=codes,
        statvars=index.statvars,
    )

    assert _sent(resolved_breakdown_values(_step("testAxis", ["axis value a"]), index)) == [
        "TestAxisA",
        "Axis value A",
    ]


@pytest.mark.parametrize(
    ("values", "sent"),
    [
        # The label goes beside the dcid, so a frame whose cells held labels still matches.
        (["TestAxisA"], ["TestAxisA", "Axis value A"]),
        (["B", "TestAxisB"], ["TestAxisB", "Axis value B"]),
        # A value naming no code of the axis stays as given.
        (["Z", "TestAxisA"], ["Z", "TestAxisA", "Axis value A"]),
    ],
)
def test_a_value_naming_one_code_goes_as_its_dcid_and_label(
    values: list[str], sent: list[str]
) -> None:
    step = resolved_breakdown_values(_step("stat_var_testAxis", values), _hierarchical_code_index())

    assert _sent(step) == sent


def test_a_code_whose_label_another_code_shares_goes_as_its_dcid_alone() -> None:
    """Both codes hold one label, so the label would also match the other code's old cells."""
    index = _sharing(_hierarchical_code_index(), label="One label")

    assert _sent(resolved_breakdown_values(_step("testAxis", ["A"]), index)) == ["TestAxisA"]


@pytest.mark.parametrize("column", ["recipient", "stat_var", "time_period"])
def test_a_value_on_another_column_stays_as_given(column: str) -> None:
    step = _step(column, ["TestAxisA", "A"])

    assert _sent(resolved_breakdown_values(step, _hierarchical_code_index())) == ["TestAxisA", "A"]


def test_an_unmatched_breakdown_filter_names_the_codes_its_column_holds() -> None:
    """A value naming no code keeps no row, and the result names the dcids the column holds
    with their labels beside them."""
    client, _, root = _rig()
    summed = {
        **root,
        "rendering": {"columns": [], "groupby": {"by": ["testAxis", "time_period"], "op": "sum"}},
    }

    response = _post(client, _keep(summed, "testAxis", ["Z"]))

    assert response.status_code == 200, response.text
    (unmatched,) = response.json()["summary"]["result"]["unmatched_breakdown_values"]
    assert unmatched["labels"] == ["TestAxisA", "TestAxisB"]
    assert unmatched["codes"] == [
        {"dcid": "TestAxisA", "label": "Axis value A"},
        {"dcid": "TestAxisB", "label": "Axis value B"},
    ]


@pytest.mark.parametrize(
    ("column", "label"),
    [
        *(("stat_var_crsPurpose", label) for label in _FRAGILITY_PURPOSES),
        ("crsTypeOfFinance", "standard loan"),
    ],
)
def test_the_website_filter_labels_each_name_one_code_of_the_recorded_index(
    column: str, label: str
) -> None:
    """Each label the website's pages filter by names one code of the code index recorded from
    dev, the index the suite's release serves, so it resolves to that code's dcid and its own
    label."""
    step = resolved_breakdown_values(_step(column, [label]), load_index(INDEX_FIXTURE))

    dcid, sent_label = _sent(step)
    assert dcid.startswith(("CRSPurposeCode", "CRSTypeOfFinance"))
    assert sent_label.casefold() == label.casefold()


@pytest.mark.parametrize(
    ("stored", "same_frame"),
    [
        # A label stored before the cells held dcids names the code the frame now holds.
        (["Axis value A"], False),
        # Values the frame still holds stay as stored, so the derived root keeps its ids.
        (["TestAxisA", "Axis value A"], True),
    ],
    ids=["label", "resolved"],
)
def test_a_stored_recipe_filter_value_keeps_the_rows_of_its_code(
    monkeypatch: pytest.MonkeyPatch, stored: list[str], same_frame: bool
) -> None:
    """A derived root over a recipe whose filter holds `stored` reads the rows of `TestAxisA`
    from a frame whose `testAxis` cells hold dcids, and names its stored pair while the frame
    holds every stored value's code as stored."""
    client, _, root = _rig()
    summed = {
        **root,
        "rendering": {"columns": [], "groupby": {"by": ["testAxis", "time_period"], "op": "sum"}},
    }
    with monkeypatch.context() as as_stored:
        as_stored.setattr(analysis, "resolved_breakdown_values", lambda step, _codes: step)
        first = _post(client, _keep(summed, "testAxis", stored))
    assert first.status_code == 200, first.text
    (member,) = first.json()["ref"]["members"]
    derived = {"spec_id": member["spec"]["spec_id"], "content_version": member["content_version"]}

    alone = _post(client, derived)
    response = _post(client, _keep(derived, "testAxis", ["TestAxisA"]))

    assert alone.status_code == response.status_code == 200, response.text
    assert [alone.json()["rows"], response.json()["rows"]] == [1, 1]
    assert (alone.json()["ref"] == first.json()["ref"]) is same_frame
    assert (first.json()["rows"] == 1) is same_frame
