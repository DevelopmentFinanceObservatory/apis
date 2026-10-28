"""The analysis runner over a double root source and live store SQL: a two-sided tree, latest
and exact roots on one path, a stored rerun, derived roots, rolling periods, pending roots, and
the bounds."""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from importlib.resources import files

import jsonschema
import pytest
import rfc8785

from datasets_api import analysis
from datasets_api.analysis import (
    AnalysisBounds,
    AnalysisPending,
    AnalysisRequest,
    AnalysisStoreDiverges,
    AnalysisTooLarge,
    RootRecipeMissing,
    SeveralMembers,
    column_for,
    run_analysis,
)
from datasets_api.doors.analysis import classify_analysis_exception
from datasets_api.output_digest import output_digest
from dfo_analysis_engine.evaluate import RootFrame, evaluate
from dfo_analysis_engine.frame_ref import FrameRef
from dfo_analysis_engine.refusal import AnalysisRefused, RefusalCode

from ._analysis_doubles import (
    AID,
    AID_V1,
    AID_V2,
    POPULATION,
    POPULATION_V1,
    Stores,
    aid_frame,
    dataset,
    population_frame,
    ref_of,
    source,
    step,
)
from ._description_doubles import read_record

_REF_SCHEMA = json.loads(
    (files("dfo_client_contract") / "dataset-ref.schema.json").read_text(encoding="utf-8")
)


def _run(stores: Stores, tree: dict, *, latest: bool = False, **kwargs: object):
    request = AnalysisRequest.model_validate({"tree": tree, "latest": latest})
    return run_analysis(
        request,
        source=kwargs.pop("root_source", None) or source(stores),  # ty: ignore[invalid-argument-type]
        stores=stores.analysis(),
        **kwargs,  # ty: ignore[invalid-argument-type]
    )


_PER_PERSON = step(
    "quotient_of",
    dataset(AID, AID_V1),
    dataset(POPULATION, POPULATION_V1),
    on={"destination": "place"},
)


def test_a_two_sided_tree_pairs_roles_by_column_and_stores_its_frame() -> None:
    stores = Stores()

    result = _run(stores, _PER_PERSON)

    expected = evaluate(
        {
            "op_spec": {
                "verb": "quotient_of",
                "parameters": {
                    "numerator_by": ["financingRecipient"],
                    "denominator_by": {"financingRecipient": "observationAbout"},
                },
            },
            "inputs": {"frame_ref": {"root": 0}, "denominator_frame_ref": {"root": 1}},
        },
        {
            0: RootFrame(ref=FrameRef(spec_id=AID, content_version=AID_V1), frame=aid_frame()),
            1: RootFrame(
                ref=FrameRef(spec_id=POPULATION, content_version=POPULATION_V1),
                frame=population_frame(),
            ),
        },
    )
    assert result.evaluation.ref == expected.ref
    assert result.output_digest == output_digest(expected.frame)
    assert result.created
    jsonschema.validate(result.ref(), _REF_SCHEMA)
    member = result.ref()["members"][0]
    assert member["spec"]["spec_id"] == expected.ref.spec_id
    assert member["content_version"] == expected.ref.content_version
    assert member["output_digest"] == result.output_digest
    assert [root["spec_id"] for root in member["roots"]] == [AID, POPULATION]
    assert member["resolved_periods"] == []
    stored = stores.recipe_store.read_recipe(
        spec_id=expected.ref.spec_id, content_version=expected.ref.content_version
    )
    assert stored is not None
    assert stored.normal_form_hash == member["spec"]["recipe"]["normal_form_hash"]
    assert stored.replay_key == member["spec"]["recipe"]["replay_key"]
    assert result.body()["summary"]["verb"] == "quotient_of"


def test_a_derived_record_names_each_step_and_the_inputs_its_roots_retrieve() -> None:
    """The record of a computed frame holds a step per operation, its inputs naming earlier
    steps, and a `retrieve` step per root, so its inputs are named versions."""
    stores = Stores()
    summed = step(
        "aggregate",
        dataset(AID, AID_V1),
        aggregation="sum",
        group_by=["destination", "time_period"],
    )
    tree = step(
        "quotient_of",
        summed,
        dataset(POPULATION, POPULATION_V1),
        on={"destination": "place"},
    )

    result = _run(stores, tree)

    record = read_record(
        stores.store,
        subject_id=result.evaluation.ref.spec_id,
        content_version=result.evaluation.ref.content_version,
    )
    assert record is not None and record.steps is not None and record.computed is not None
    steps = [
        step.model_dump(mode="json", include={"id", "op", "input", "inputs"})
        for step in record.steps
    ]
    assert steps == [
        {"id": "s1", "op": "retrieve", "input": {"id": AID, "content_version": AID_V1}},
        {"id": "s2", "op": "aggregate", "input": "s1"},
        {
            "id": "s3",
            "op": "retrieve",
            "input": {"id": POPULATION, "content_version": POPULATION_V1},
        },
        {"id": "s4", "op": "quotient", "inputs": {"numerator": "s2", "denominator": "s3"}},
    ]
    assert record.steps[-1].stat_var == result.evaluation.ref.spec_id
    assert [(one.id, one.content_version) for one in record.computed.inputs] == [
        (AID, AID_V1),
        (POPULATION, POPULATION_V1),
    ]


def _stored_steps(stores: Stores, result) -> list[dict] | None:
    record = read_record(
        stores.store,
        subject_id=result.evaluation.ref.spec_id,
        content_version=result.evaluation.ref.content_version,
    )
    assert record is not None
    if record.steps is None:
        return None
    return [step.model_dump(mode="json", exclude_unset=True) for step in record.steps]


def test_a_step_list_over_the_byte_bound_publishes_with_its_coverage_shed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    whole_stores = Stores()
    whole = _stored_steps(whole_stores, _run(whole_stores, _PER_PERSON))
    assert whole is not None and "period_coverage" in whole[-1]
    bound = len(rfc8785.dumps(whole)) - 1
    monkeypatch.setattr(analysis, "MAX_STEPS_BYTES", bound)
    stores = Stores()

    steps = _stored_steps(stores, _run(stores, _PER_PERSON))

    shed = {name: value for name, value in whole[-1].items() if name != "period_coverage"}
    assert steps == [*whole[:-1], shed]
    assert len(rfc8785.dumps(steps)) <= bound


def test_compare_gives_data_minus_other() -> None:
    stores = Stores()

    result = _run(
        stores,
        step(
            "compare",
            dataset(AID, AID_V2),
            dataset(AID, AID_V1),
            on=["destination", "origin"],
        ),
    )

    values = sorted(result.evaluation.frame.table.column("value").to_pylist())
    assert values == [10.0, 12.0, 20.0, 30.0]
    assert result.evaluation.summary["sides"] == {"data": "right", "other": "left"}


def test_latest_and_exact_roots_run_one_path() -> None:
    stores = Stores()
    tree = step("aggregate", dataset(AID, AID_V1), aggregation="sum", group_by=["destination"])
    newest = step("aggregate", dataset(AID, AID_V2), aggregation="sum", group_by=["destination"])
    roots = source(stores)

    latest = _run(stores, tree, latest=True, root_source=roots)
    exact = _run(stores, newest)
    older = _run(stores, tree)

    assert roots.resolved == [(AID, AID_V1, True)]
    assert latest.evaluation.ref == exact.evaluation.ref
    assert latest.ref()["members"][0]["roots"][0]["content_version"] == AID_V2
    assert older.evaluation.ref != exact.evaluation.ref


def test_a_rerun_stores_nothing_new() -> None:
    stores = Stores()
    first = _run(stores, _PER_PERSON)
    versions = stores.version_count()

    again = _run(stores, _PER_PERSON)

    assert not again.created
    assert again.evaluation.ref == first.evaluation.ref
    assert again.artifact_digest == first.artifact_digest
    assert again.ref() == first.ref()
    assert stores.version_count() == versions


def test_columns_listed_in_another_order_name_the_same_frame_and_recipe() -> None:
    """The frame pair sorts a root's columns, so its record and normal form must as well, or the
    second run's recipe diverges from the one stored under the same pair."""
    stores = Stores()

    def summed(columns: list[str]) -> dict:
        root = {**dataset(AID, AID_V1), "rendering": {"columns": columns}}
        return step("aggregate", root, aggregation="sum", group_by=["destination"])

    first = _run(stores, summed(["party_kind", "financingRecipient_type", "party_kind"]))
    again = _run(stores, summed(["financingRecipient_type", "party_kind"]))

    assert not again.created
    assert again.ref() == first.ref()
    (root,) = first.ref()["members"][0]["roots"]
    assert root["rendering"]["columns"] == ["financingRecipient_type", "party_kind"]


def test_a_pair_holding_another_recipe_answers_500_logged_with_both_identities(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stores = Stores()
    ref = _run(stores, _PER_PERSON).evaluation.ref
    with stores.store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.frame_recipe SET normal_form_hash = %s "
            "WHERE subject_id = %s AND content_version = %s",
            ("rnf_" + "0" * 32, ref.spec_id, ref.content_version),
        )

    with caplog.at_level(logging.ERROR), pytest.raises(AnalysisStoreDiverges) as raised:
        _run(stores, _PER_PERSON)

    spec = classify_analysis_exception(raised.value)
    assert spec is not None
    assert (spec.status, spec.code) == (500, "internal_error")
    (record,) = [r for r in caplog.records if r.name == "datasets_api.analysis"]
    assert "stored_normal_form_hash=rnf_" + "0" * 32 in record.getMessage()
    assert "run_normal_form_hash=rnf_" in record.getMessage()


def test_the_artifact_digest_does_not_depend_on_the_time_of_publish() -> None:
    """The frame file holds no publish time, so one tree published on another day names the
    same file."""
    first = _run(Stores(), _PER_PERSON)
    later = Stores()
    later.now = later.now + timedelta(days=400)

    again = _run(later, _PER_PERSON)

    assert again.artifact_digest == first.artifact_digest


@pytest.mark.parametrize(
    "rendering",
    [{}, {"columns": ["financingRecipient_party_kind"], "without_party_totals": True}],
    ids=["default", "columns"],
)
def test_a_derived_root_runs_as_the_chain_it_was_computed_by(rendering: dict) -> None:
    """A derived root expands into its recipe's roots read in their recorded rendering, so it
    answers the ref of the chain it was computed by."""
    stores = Stores()
    root = {**dataset(AID, AID_V1), "rendering": rendering}
    summed = step("aggregate", root, aggregation="sum", group_by=["destination"])
    stored = _run(stores, summed).evaluation.ref

    chained = _run(stores, step("rank", summed, rank_by=["destination"], limit=1))
    from_derived = _run(stores, step("rank", stored.model_dump(), rank_by=["destination"], limit=1))

    assert from_derived.evaluation.ref == chained.evaluation.ref
    assert from_derived.ref() == chained.ref()


def test_a_derived_root_without_a_stored_recipe_refuses() -> None:
    stores = Stores()
    missing = {"spec_id": "ops_" + "c" * 32, "content_version": "ocv_" + "d" * 32}

    with pytest.raises(RootRecipeMissing) as raised:
        _run(stores, step("rank", missing, rank_by=["destination"]))
    assert raised.value.fix["latest"] is True


def test_rolling_window_bounds_take_the_latest_period_and_their_populations() -> None:
    stores = Stores()
    tree = step(
        "growth",
        dataset(AID, AID_V1),
        entity_by=["destination"],
        window_start={"anchor": "latest", "offset": -1},
        window_end={"anchor": "latest", "offset": 0},
    )

    result = _run(stores, tree)

    parameters = result.evaluation.tree["op_spec"]["parameters"]
    assert (parameters["window_start"], parameters["window_end"]) == ("2021", "2022")
    assert parameters["rolling"]["population"] == [{"spec_id": AID, "bindings": {}}]
    assert result.resolved_periods == (
        {"anchor": "latest", "offset": -1, "rule": "half1", "period": "2021"},
        {"anchor": "latest", "offset": 0, "rule": "half1", "period": "2022"},
    )
    assert result.ref()["members"][0]["resolved_periods"] == list(result.resolved_periods)
    jsonschema.validate(result.ref(), _REF_SCHEMA)


def test_anchor_roots_take_the_latest_period_over_roots_the_tree_does_not_read() -> None:
    """A rank over the aid data whose period the anchor route took over aid and population
    answers from the year both hold, 2021, and a replay of its stored recipe takes the period
    over the same roots. Without the anchor roots the tree reads no population root."""
    stores = Stores()
    roots = source(
        stores,
        frames={
            (AID, AID_V1): aid_frame(),
            (POPULATION, POPULATION_V1): population_frame(years=("2021",)),
        },
    )
    latest = {"anchor": "latest", "offset": 0}
    tree = step(
        "rank",
        dataset(AID, AID_V1),
        rank_by=["destination"],
        window_start=latest,
        window_end=latest,
        rolling={
            "population": [
                {"spec_id": AID, "bindings": {}},
                {"spec_id": POPULATION, "bindings": {}},
            ]
        },
    )
    anchor_roots = [dataset(AID, AID_V1), dataset(POPULATION, POPULATION_V1)]

    with pytest.raises(AnalysisRefused) as raised:
        _run(stores, tree, root_source=roots)
    result = run_analysis(
        AnalysisRequest.model_validate({"tree": tree, "anchor_roots": anchor_roots}),
        source=roots,
        stores=stores.analysis(),
    )
    replayed = _run(stores, result.evaluation.ref.model_dump(), root_source=roots)

    assert raised.value.code is RefusalCode.ROLLING_ANCHOR_UNRESOLVED
    assert raised.value.refusal.fix["detail"] == "population_without_root"
    parameters = result.evaluation.tree["op_spec"]["parameters"]
    assert (parameters["window_start"], parameters["window_end"]) == ("2021", "2021")
    assert [root["spec_id"] for root in result.ref()["members"][0]["roots"]] == [AID, POPULATION]
    assert replayed.evaluation.ref == result.evaluation.ref
    assert replayed.ref() == result.ref()


def test_rolling_steps_over_different_populations_refuse_before_any_verb() -> None:
    stores = Stores()
    first = {"spec_id": AID, "bindings": {}}
    second = {"spec_id": AID, "bindings": {"origin": ["country/DEU"]}}
    inner = step(
        "growth",
        dataset(AID, AID_V1),
        entity_by=["destination"],
        window_start="2021",
        window_end="2022",
        rolling={
            "window_end": {"anchor": "latest", "offset": 0, "rule": "half1"},
            "population": [first],
        },
    )
    tree = step(
        "rank",
        inner,
        rank_by=["destination"],
        window_start="2021/2022",
        window_end="2021/2022",
        rolling={
            "window_end": {"anchor": "latest", "offset": 0, "rule": "half1"},
            "population": [second],
        },
    )

    with pytest.raises(AnalysisRefused) as raised:
        _run(stores, tree)
    assert raised.value.code is RefusalCode.ROLLING_POPULATIONS_DIFFER


def test_an_input_holding_several_readings_refuses_naming_the_call_that_takes_one() -> None:
    stores = Stores()
    tree = step(
        "quotient_of",
        {"ref": ref_of((AID, AID_V1), (POPULATION, POPULATION_V1))},
        dataset(POPULATION, POPULATION_V1),
    )

    with pytest.raises(SeveralMembers) as raised:
        _run(stores, tree)
    assert raised.value.fix == {
        "parameter": "data",
        "call": "fetch",
        "spec_ids": [AID, POPULATION],
    }


def test_a_ref_of_one_member_reads_as_that_member() -> None:
    stores = Stores()
    from_ref = _run(
        stores,
        step(
            "aggregate", {"ref": ref_of((AID, AID_V1))}, aggregation="sum", group_by=["destination"]
        ),
    )
    direct = _run(
        stores, step("aggregate", dataset(AID, AID_V1), aggregation="sum", group_by=["destination"])
    )

    assert from_ref.evaluation.ref == direct.evaluation.ref


def test_every_pending_root_answers_at_once_before_any_frame_is_read() -> None:
    stores = Stores()
    roots = source(stores, pending={AID: "op_aid", POPULATION: "op_population"})
    tree = step(
        "quotient_of",
        {"spec_id": AID},
        {"spec_id": POPULATION},
        on={"destination": "place"},
    )

    with pytest.raises(AnalysisPending) as raised:
        _run(stores, tree, root_source=roots)
    assert raised.value.operations == ("op_aid", "op_population")
    assert roots.reads == 0
    assert stores.version_count() == 0


@pytest.mark.parametrize(
    ("bounds", "ceiling_kind"),
    [
        pytest.param(AnalysisBounds(max_nodes=2), "analysis_nodes", id="nodes"),
        pytest.param(AnalysisBounds(max_root_rows=7), "analysis_root_rows", id="root-rows"),
    ],
)
def test_a_tree_past_a_bound_refuses_before_it_computes(
    bounds: AnalysisBounds, ceiling_kind: str
) -> None:
    stores = Stores()

    with pytest.raises(AnalysisTooLarge) as raised:
        _run(stores, _PER_PERSON, bounds=bounds)
    assert raised.value.ceiling_kind == ceiling_kind
    assert stores.version_count() == 0


@pytest.mark.parametrize(
    ("recorded", "cap", "reads"),
    [
        pytest.param(True, 7, 0, id="recorded-counts-refuse-before-any-read"),
        pytest.param(False, 3, 1, id="unrecorded-counts-stop-at-the-first-frame-past"),
    ],
)
def test_root_rows_past_the_bound_refuse_before_reading_past_it(
    recorded: bool, cap: int, reads: int
) -> None:
    """Each root holds 4 rows."""
    stores = Stores()
    roots = source(stores, recorded_rows=recorded)

    with pytest.raises(AnalysisTooLarge) as raised:
        _run(stores, _PER_PERSON, root_source=roots, bounds=AnalysisBounds(max_root_rows=cap))

    assert raised.value.ceiling_kind == "analysis_root_rows"
    assert roots.reads == reads


def test_a_frame_past_the_publish_rows_refuses_before_it_is_stored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = Stores()
    monkeypatch.setattr(analysis, "PUBLISH_MAX_ROWS", 1)

    with pytest.raises(AnalysisTooLarge) as raised:
        _run(stores, _PER_PERSON)

    assert (raised.value.ceiling_kind, raised.value.cap) == ("analysis_output_rows", 1)
    assert stores.version_count() == 0


def test_a_join_past_its_row_bound_refuses() -> None:
    stores = Stores()
    tree = step(
        "join",
        dataset(AID, AID_V1),
        dataset(POPULATION, POPULATION_V1),
        on={"destination": "place"},
    )

    with pytest.raises(AnalysisRefused) as raised:
        _run(stores, tree, bounds=AnalysisBounds(max_join_rows=1))
    assert raised.value.code is RefusalCode.FRAME_TOO_LARGE


@pytest.mark.parametrize(
    ("word", "columns", "column"),
    [
        pytest.param("place", ["financingRecipient"], "observationAbout", id="place"),
        pytest.param(
            "destination", ["financingRecipient", "origin"], "financingRecipient", id="flow-axis"
        ),
        pytest.param("origin", ["financingRecipient", "origin"], "origin", id="column-itself"),
        pytest.param("sector", ["financingRecipient"], "sector", id="other-words-pass"),
    ],
)
def test_a_role_word_names_its_column(word: str, columns: list[str], column: str) -> None:
    assert column_for(word, columns, parameter="group_by") == column


def test_a_role_two_columns_fill_refuses_role_ambiguous() -> None:
    with pytest.raises(AnalysisRefused) as raised:
        column_for("destination", ["financingRecipient", "recipient"], parameter="group_by")
    assert raised.value.code is RefusalCode.ROLE_AMBIGUOUS
