"""`maintenance normalise-recipes`: a recipe stored with its root columns or `groupby.by` out of
order gets the
recipe and hashes the analysis route stores for the same tree, and the card revisions drawing
its frame move to the new `normal_form_hash`, over the SQLite dataset store double and the
Presentation store fake."""

from __future__ import annotations

import json
from typing import Any

import pytest

from datasets_api.analysis import AnalysisRequest, run_analysis
from datasets_api.maintenance import main, run_normalise_recipes
from dfo_analysis_engine.recipe import normal_form, normal_form_hash, replay_key, root_identity
from tests.presentation_store.conftest import FakePresentationStore

from ._analysis_doubles import AID, AID_V1, Stores, dataset, source, step

_COLUMNS = ["party_kind", "financingRecipient_type"]
_BY = ["time_period", "origin"]


def _summed(rendering: dict[str, Any]) -> dict[str, Any]:
    return step(
        "aggregate",
        {**dataset(AID, AID_V1), "rendering": rendering},
        aggregation="sum",
        group_by=["destination"],
    )


_SUMMED = _summed({"columns": _COLUMNS})


def _stored(stores: Stores) -> list[dict[str, Any]]:
    with stores.store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT subject_id, content_version, normal_form_hash, replay_key, recipe "
            "FROM datasets.frame_recipe ORDER BY subject_id, content_version"
        )
    return [dict(row) for row in rows]


def _hashes(recipe: dict[str, Any]) -> tuple[str, str]:
    """The hashes of `recipe`, computed independently of the task under test."""
    normal = normal_form_hash(normal_form(recipe["tree"], recipe["roots"]))
    roots = [
        root_identity(r["dataset"], enrichment_stamp=r.get("enrichment_stamp"))
        for r in recipe["roots"]
    ]
    return normal, replay_key(
        normal_form_hash=normal,
        root_datasets=roots,
        execution_identities=[recipe["execution_identity"]],
    )


def _unsorted(recipe: dict[str, Any]) -> dict[str, Any]:
    """`recipe` as data-mcp stored it: its root columns and any `groupby.by` in the order a
    client sent them."""
    (root,) = recipe["roots"]
    assert root["options"]["columns"] == sorted(_COLUMNS)
    options: dict[str, Any] = {**root["options"], "columns": _COLUMNS}
    if "groupby" in options:
        assert options["groupby"]["by"] == sorted(_BY)
        options = {**options, "groupby": {**options["groupby"], "by": _BY}}
    old = {**recipe, "roots": [{**root, "options": options}]}
    normal, key = _hashes(old)
    return {**old, "normal_form_hash": normal, "replay_key": key}


def _store_recipe(stores: Stores, spec_id: str, content_version: str, recipe: dict) -> None:
    """Hold `recipe` under the pair in place of any recipe stored there."""
    with stores.store.transaction() as tx:
        tx.execute(
            "DELETE FROM datasets.frame_recipe WHERE subject_id = %s AND content_version = %s",
            (spec_id, content_version),
        )
        tx.execute(
            "INSERT INTO datasets.frame_recipe (subject_id, content_version, normal_form_hash, "
            "replay_key, recipe, recorded_at) VALUES (%s, %s, %s, %s, %s, %s)",
            (
                spec_id,
                content_version,
                recipe["normal_form_hash"],
                recipe["replay_key"],
                recipe,
                stores.now,
            ),
        )


def _analysed(stores: Stores, tree: dict[str, Any] = _SUMMED) -> dict[str, Any]:
    """The recipe row the analysis route stores for `tree`."""
    run_analysis(
        AnalysisRequest.model_validate({"tree": tree}),
        source=source(stores),
        stores=stores.analysis(),
    )
    (row,) = _stored(stores)
    return row


def _revision(spec_id: str, content_version: str, normal_hash: str, revision: int) -> dict:
    return {
        "revision": revision,
        "stored_intent": {"frame_ref": {"spec_id": spec_id, "content_version": content_version}},
        "frame_rendering": {"normal_form_hash": normal_hash},
    }


def _summary(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    (summary,) = [line for line in lines if line["event"] == "datasets_recipes_normalised"]
    return {
        key: summary[key]
        for key in ("normalised", "revisions", "replay_key_held", "unreproducible")
    }


@pytest.mark.parametrize(
    "rendering",
    [{"columns": _COLUMNS}, {"columns": _COLUMNS, "groupby": {"by": _BY, "op": "sum"}}],
    ids=["columns", "groupby"],
)
def test_a_recipe_stored_unsorted_gets_the_analysis_routes_recipe_and_its_cards_follow(
    capsys: pytest.CaptureFixture[str], rendering: dict[str, Any]
):
    stores = Stores()
    analysed = _analysed(stores, _summed(rendering))
    spec_id, content_version = analysed["subject_id"], analysed["content_version"]
    old = _unsorted(analysed["recipe"])
    _store_recipe(stores, spec_id, content_version, old)
    presentation = FakePresentationStore()
    presentation.state.component_revisions["cmp_drawn"] = [
        _revision(spec_id, content_version, old["normal_form_hash"], 1),
        _revision(spec_id, content_version, old["normal_form_hash"], 2),
    ]
    # Another frame's card under the same hash moves only when that frame's recipe does.
    presentation.state.component_revisions["cmp_other"] = [
        _revision("ops_other", "ocv_other", old["normal_form_hash"], 1)
    ]

    def revisions() -> dict[str, list[dict]]:
        # A fake transaction replaces the state's mappings when it commits.
        return presentation.state.component_revisions

    before = (_stored(stores), json.dumps(revisions(), sort_keys=True))

    dry = run_normalise_recipes(datasets=stores.store, presentation=presentation, dry_run=True)

    assert (dry, _summary(capsys)) == (
        0,
        {"normalised": 1, "revisions": 2, "replay_key_held": 0, "unreproducible": 0},
    )
    assert (_stored(stores), json.dumps(revisions(), sort_keys=True)) == before

    skipped = run_normalise_recipes(datasets=stores.store, presentation=presentation)

    assert (skipped, _summary(capsys)) == (
        0,
        {"normalised": 1, "revisions": 2, "replay_key_held": 0, "unreproducible": 0},
    )
    (row,) = _stored(stores)
    assert {key: row[key] for key in ("normal_form_hash", "replay_key", "recipe")} == {
        key: analysed[key] for key in ("normal_form_hash", "replay_key", "recipe")
    }
    assert [r["frame_rendering"] for r in revisions()["cmp_drawn"]] == [
        {"normal_form_hash": analysed["normal_form_hash"]}
    ] * 2
    assert revisions()["cmp_other"][0]["frame_rendering"] == {
        "normal_form_hash": old["normal_form_hash"]
    }

    after = (_stored(stores), json.dumps(revisions(), sort_keys=True))
    again = run_normalise_recipes(datasets=stores.store, presentation=presentation)

    assert (again, _summary(capsys)) == (
        0,
        {"normalised": 0, "revisions": 0, "replay_key_held": 0, "unreproducible": 0},
    )
    assert (_stored(stores), json.dumps(revisions(), sort_keys=True)) == after


@pytest.mark.parametrize("reason", ["replay_key_held", "unreproducible"])
def test_a_recipe_it_cannot_normalise_safely_is_reported_and_left_as_it_is(
    capsys: pytest.CaptureFixture[str], reason: str
):
    stores = Stores()
    analysed = _analysed(stores)
    old = _unsorted(analysed["recipe"])
    if reason == "unreproducible":
        # A key computed over identities the stored recipe does not name.
        old = {**old, "replay_key": "rpk_" + "0" * 32}
    # The same tree stored unsorted under an earlier frame of its own.
    _store_recipe(stores, "ops_earlier", "ocv_earlier", old)
    if reason == "unreproducible":
        with stores.store.transaction() as tx:
            tx.execute(
                "DELETE FROM datasets.frame_recipe WHERE subject_id = %s",
                (analysed["subject_id"],),
            )
    presentation = FakePresentationStore()
    presentation.state.component_revisions["cmp_earlier"] = [
        _revision("ops_earlier", "ocv_earlier", old["normal_form_hash"], 1)
    ]
    before = (_stored(stores), json.dumps(presentation.state.component_revisions))

    skipped = run_normalise_recipes(datasets=stores.store, presentation=presentation)

    assert skipped == 1
    assert _summary(capsys) == {
        "normalised": 0,
        "revisions": 0,
        "replay_key_held": int(reason == "replay_key_held"),
        "unreproducible": int(reason == "unreproducible"),
    }
    assert (_stored(stores), json.dumps(presentation.state.component_revisions)) == before


def test_normalise_recipes_exits_non_zero_when_it_skips_a_recipe(monkeypatch):
    ran: list[bool] = []

    def run(*, datasets: object, presentation: object, dry_run: bool) -> int:
        ran.append(dry_run)
        return 1

    monkeypatch.setattr("datasets_api.maintenance.run_normalise_recipes", run)
    monkeypatch.setattr("datasets_api.maintenance.shared_pool", lambda: None)
    monkeypatch.setattr("datasets_api.maintenance.close_shared_pool", lambda: None)
    monkeypatch.setattr("presentation_store.provider.shared_pool", lambda: None)
    monkeypatch.setattr("presentation_store.provider.close_shared_pool", lambda: None)

    assert main(["normalise-recipes", "--dry-run"]) == 1
    assert ran == [True]
