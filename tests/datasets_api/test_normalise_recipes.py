"""`maintenance normalise-recipes`: a recipe stored with its root columns or `groupby.by` out of
order gets the
recipe and hashes the analysis route stores for the same tree, and the card revisions drawing
its frame move to the new `normal_form_hash`, over the SQLite dataset store double and the
Presentation store fake."""

from __future__ import annotations

import json
from typing import Any, get_args

import pytest

from datasets_api import maintenance
from datasets_api.analysis import AnalysisRequest, run_analysis
from datasets_api.maintenance import main, run_normalise_recipes
from dfo_analysis_engine.op_spec import OpSpec, PinnedUnit
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


_OUTCOMES = ("normalised", "unchanged", "replay_key_held", "unreproducible", "basis_undetermined")
_COUNTS = ("normalised", "revisions", "replay_key_held", "unreproducible", "basis_undetermined")


def _summary(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    """The summary's counts, once its outcome counts are checked to add up to its recipes."""
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    (summary,) = [line for line in lines if line["event"] == "datasets_recipes_normalised"]
    assert sum(summary[key] for key in _OUTCOMES) == summary["recipes"], summary
    return {key: summary[key] for key in _COUNTS}


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
        {
            "normalised": 1,
            "revisions": 2,
            "replay_key_held": 0,
            "unreproducible": 0,
            "basis_undetermined": 0,
        },
    )
    assert (_stored(stores), json.dumps(revisions(), sort_keys=True)) == before

    skipped = run_normalise_recipes(datasets=stores.store, presentation=presentation)

    assert (skipped, _summary(capsys)) == (
        0,
        {
            "normalised": 1,
            "revisions": 2,
            "replay_key_held": 0,
            "unreproducible": 0,
            "basis_undetermined": 0,
        },
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
        {
            "normalised": 0,
            "revisions": 0,
            "replay_key_held": 0,
            "unreproducible": 0,
            "basis_undetermined": 0,
        },
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
        "basis_undetermined": 0,
    }
    assert (_stored(stores), json.dumps(presentation.state.component_revisions)) == before


def test_a_rewrite_that_keeps_its_own_replay_key_is_normalised(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    stores = Stores()
    analysed = _analysed(stores)
    old = _unsorted(analysed["recipe"])
    _store_recipe(stores, "ops_earlier", "ocv_earlier", old)
    with stores.store.transaction() as tx:
        tx.execute(
            "DELETE FROM datasets.frame_recipe WHERE subject_id = %s", (analysed["subject_id"],)
        )
    # A rewrite the normal form ignores: the recipe's own row holds the key it computes.
    monkeypatch.setattr(
        maintenance, "_recipe_hashes", lambda _: (old["normal_form_hash"], old["replay_key"])
    )

    skipped = run_normalise_recipes(datasets=stores.store, presentation=FakePresentationStore())

    assert skipped == 0
    assert _summary(capsys) == {
        "normalised": 1,
        "revisions": 0,
        "replay_key_held": 0,
        "unreproducible": 0,
        "basis_undetermined": 0,
    }
    (row,) = _stored(stores)
    assert row["recipe"]["roots"][0]["options"]["columns"] == sorted(_COLUMNS)


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


# Pins of a constant-price unit without its deflation basis.

_DAC = "spec_" + "d" * 32
_IDS = "spec_" + "e" * 32
_POP = "spec_" + "f" * 32
_UNRECORDED = "spec_" + "0" * 32
_UNKNOWN_RECIPE = "spec_" + "1" * 32
_NO_DEFLATOR_ROLE = "spec_" + "2" * 32
# A root whose version the store does not hold.
_UNHELD = "spec_" + "3" * 32
_CV = "cv_" + "9" * 32
_CONSTANT = {"currency": "NGN", "prices": "constant", "base_year": 2024}
# What each root's mint converted: the recipe and the declared roles of each series.
_ROOT_SERIES: dict[str, tuple[str | None, list[tuple[str, str]]]] = {
    _DAC: ("ConversionRecipeDAC", [("financingProvider", "country/DEU")]),
    _IDS: ("ConversionRecipeIDS", [("financingRecipient", "country/NGA")]),
    _POP: (None, []),
    _UNRECORDED: ("", []),
    _UNKNOWN_RECIPE: ("ConversionRecipeRetired", [("financingProvider", "country/DEU")]),
    # The IDS recipe deflates on the recipient alone.
    _NO_DEFLATOR_ROLE: ("ConversionRecipeIDS", [("financingProvider", "country/DEU")]),
}


def _store_root(stores: Stores, spec_id: str) -> None:
    """Record the version `_CV` of the root `spec_id` as its mint stores it: a constant-price
    unit request and one conversion record per series (`_ROOT_SERIES`), `_POP` asking no unit
    and `_UNRECORDED` storing no series record."""
    recipe_id, roles = _ROOT_SERIES[spec_id]
    unit = {} if recipe_id is None else {"unit": {"request": _CONSTANT}}
    series = None
    if recipe_id:
        series = [
            {
                "stat_var": "dcid:Amount",
                "roles": [list(pair) for pair in roles],
                "recipe_id": recipe_id,
                "table_provenance_id": "dcid:Tables",
                "exchange_rate_table_version": "rates",
                "deflator_table_version": "deflators",
                "fallback_count": 0,
                "fallback_examples": [],
                "projected_count": 0,
                "projected_examples": [],
            }
        ]
    with stores.store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record (subject_id, subject_kind, content_version, "
            "schema_version, immutable_body, recorded_at, prepared_at, definition, coverage, "
            "citation, trace, warnings, series_conversions) "
            "VALUES (%s, 'spec', %s, 1, '{}', %s, %s, %s, '{}', '{}', '{}', '[]', %s)",
            (
                spec_id,
                _CV,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                json.dumps({"identity": {"facets": unit}}),
                None if series is None else json.dumps(series),
            ),
        )


def _root(index: int) -> dict[str, Any]:
    return {"frame": {"spec_id": "spec_frame", "content_version": "cv_frame"}, "root": index}


def _op(verb: str, inputs: dict[str, Any], **parameters: Any) -> dict[str, Any]:
    """A derived node of `verb` over `inputs`, its slots also named in its parameters."""
    slots = {slot: {"spec_id": "ops_in", "content_version": "ocv_in"} for slot in inputs}
    return {
        "frame": {"spec_id": "ops_node", "content_version": "ocv_node"},
        "op_spec": {
            "identity_version": 1,
            "verb": verb,
            "frame_ref": slots.pop("frame_ref"),
            "parameters": {**slots, **parameters},
        },
        "inputs": inputs,
    }


def _pinned(
    stores: Stores,
    tree: dict[str, Any],
    roots: list[str],
    options: dict[str, Any] | None = None,
    *,
    replay_key: str | None = None,
) -> dict[str, Any]:
    """Store, under `ops_pinned`, the recipe of `tree` over the `_CV` versions of `roots`,
    each root with `options`, recording each root, and return it. `replay_key` replaces the
    key its hashes give."""
    for spec_id in roots:
        if spec_id != _UNHELD:
            _store_root(stores, spec_id)
    recipe = {
        "tree": tree,
        "roots": [
            {"dataset": {"spec_id": spec_id, "content_version": _CV}, "options": options or {}}
            for spec_id in roots
        ],
        "execution_identity": "engine-1",
    }
    normal, key = _hashes(recipe)
    recipe = {**recipe, "normal_form_hash": normal, "replay_key": replay_key or key}
    _store_recipe(stores, "ops_pinned", "ocv_pinned", recipe)
    return recipe


def _with(unit: dict[str, Any], basis: str) -> dict[str, Any]:
    return {**unit, "deflation_basis": basis}


_COMPOSED = {"numerator": _CONSTANT, "denominator": {"stat_var": "dcid:Count_Person"}}


@pytest.mark.parametrize(
    ("tree", "roots", "migrated"),
    [
        (
            _op(
                "compare",
                {"frame_ref": _root(0), "right_frame_ref": _root(1)},
                left_unit=_CONSTANT,
                right_unit=_CONSTANT,
            ),
            [_DAC, _IDS],
            {"left_unit": _with(_CONSTANT, "donor"), "right_unit": _with(_CONSTANT, "recipient")},
        ),
        (
            _op(
                "share_of",
                {"frame_ref": _root(0), "denominator_frame_ref": _root(1)},
                numerator_unit=_CONSTANT,
                denominator_unit=_CONSTANT,
            ),
            [_IDS, _DAC],
            {
                "numerator_unit": _with(_CONSTANT, "recipient"),
                "denominator_unit": _with(_CONSTANT, "donor"),
            },
        ),
        (
            # The other side is a second slice of `frame_ref`.
            _op("compare", {"frame_ref": _root(0)}, right_unit=_CONSTANT),
            [_IDS],
            {"right_unit": _with(_CONSTANT, "recipient")},
        ),
        (
            # A composed unit's converted side takes the basis, its StatVar side stays.
            _op(
                "aggregate",
                {
                    "frame_ref": _op(
                        "quotient_of", {"frame_ref": _root(0), "denominator_frame_ref": _root(1)}
                    )
                },
                unit=_COMPOSED,
            ),
            [_DAC, _POP],
            {"unit": {**_COMPOSED, "numerator": _with(_CONSTANT, "donor")}},
        ),
    ],
    ids=["compare-sides", "share-of-sides", "second-slice", "composed"],
)
def test_a_pin_without_its_basis_gets_the_basis_of_the_roots_feeding_it(
    capsys: pytest.CaptureFixture[str],
    tree: dict[str, Any],
    roots: list[str],
    migrated: dict[str, Any],
):
    stores = Stores()
    old = _pinned(stores, tree, roots)
    presentation = FakePresentationStore()
    presentation.state.component_revisions["cmp_drawn"] = [
        _revision("ops_pinned", "ocv_pinned", old["normal_form_hash"], 1)
    ]

    def state() -> tuple[list[dict[str, Any]], str]:
        # A fake transaction replaces the state's mappings when it commits.
        return _stored(stores), json.dumps(presentation.state.component_revisions, sort_keys=True)

    before = state()

    dry = run_normalise_recipes(datasets=stores.store, presentation=presentation, dry_run=True)

    assert (dry, _summary(capsys)) == (
        0,
        {
            "normalised": 1,
            "revisions": 1,
            "replay_key_held": 0,
            "unreproducible": 0,
            "basis_undetermined": 0,
        },
    )
    assert state() == before

    assert run_normalise_recipes(datasets=stores.store, presentation=presentation) == 0
    (row,) = _stored(stores)
    parameters = row["recipe"]["tree"]["op_spec"]["parameters"]
    assert {name: parameters[name] for name in migrated} == migrated
    expected = {**old, "tree": row["recipe"]["tree"]}
    del expected["normal_form_hash"], expected["replay_key"]
    normal, key = _hashes(expected)
    assert (row["normal_form_hash"], row["replay_key"]) == (normal, key)
    assert (row["recipe"]["normal_form_hash"], row["recipe"]["replay_key"]) == (normal, key)
    assert normal != old["normal_form_hash"]
    assert presentation.state.component_revisions["cmp_drawn"][0]["frame_rendering"] == {
        "normal_form_hash": normal
    }
    capsys.readouterr()

    after = state()

    assert run_normalise_recipes(datasets=stores.store, presentation=presentation) == 0
    assert _summary(capsys)["normalised"] == 0
    assert state() == after


@pytest.mark.parametrize(
    ("roots", "inputs", "cause", "named"),
    [
        (
            [_DAC, _IDS],
            {"frame_ref": _root(0), "right_frame_ref": _root(1)},
            "several_bases",
            [_DAC, _IDS],
        ),
        ([_UNRECORDED], {"frame_ref": _root(0)}, "no_series_record", [_UNRECORDED]),
        ([_POP], {"frame_ref": _root(0)}, "no_basis", [_POP]),
        # A root whose version cannot tell is named alone.
        (
            [_DAC, _UNHELD],
            {"frame_ref": _root(0), "right_frame_ref": _root(1)},
            "root_unread",
            [_UNHELD],
        ),
        ([_UNKNOWN_RECIPE], {"frame_ref": _root(0)}, "unknown_recipe", [_UNKNOWN_RECIPE]),
        (
            [_NO_DEFLATOR_ROLE],
            {"frame_ref": _root(0)},
            "no_single_deflator_role",
            [_NO_DEFLATOR_ROLE],
        ),
    ],
    ids=[
        "two-bases",
        "no-series-record",
        "no-basis",
        "root-unread",
        "unknown-recipe",
        "no-single-deflator-role",
    ],
)
def test_a_pin_whose_roots_give_no_single_basis_is_reported_and_left_as_it_is(
    capsys: pytest.CaptureFixture[str],
    roots: list[str],
    inputs: dict[str, Any],
    cause: str,
    named: list[str],
):
    stores = Stores()
    # `unit` reads every root under its op-spec.
    old = _pinned(
        stores, _op("aggregate", {"frame_ref": _op("compare", inputs)}, unit=_CONSTANT), roots
    )
    presentation = FakePresentationStore()
    presentation.state.component_revisions["cmp_drawn"] = [
        _revision("ops_pinned", "ocv_pinned", old["normal_form_hash"], 1)
    ]
    before = (_stored(stores), json.dumps(presentation.state.component_revisions))

    skipped = run_normalise_recipes(datasets=stores.store, presentation=presentation)

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    (summary,) = [line for line in lines if line["event"] == "datasets_recipes_normalised"]
    (skip,) = [line for line in lines if line["event"] == "datasets_recipe_normalise_skipped"]
    assert (skipped, {key: summary[key] for key in _COUNTS}) == (
        1,
        {
            "normalised": 0,
            "revisions": 0,
            "replay_key_held": 0,
            "unreproducible": 0,
            "basis_undetermined": 1,
        },
    )
    assert {key: skip[key] for key in ("reason", "parameter", "roots", "cause")} == {
        "reason": "basis_undetermined",
        "parameter": "unit",
        "roots": [f"{spec_id}@{_CV}" for spec_id in named],
        "cause": cause,
    }
    assert (_stored(stores), json.dumps(presentation.state.component_revisions)) == before


@pytest.mark.parametrize("sort", ["written", "unreproducible"])
def test_a_recipe_whose_pin_basis_is_undetermined_takes_one_outcome(
    capsys: pytest.CaptureFixture[str], sort: str
):
    """The sort of a root's options and the pin's basis are separate rewrites, and a recipe
    takes one outcome. With unsorted columns and a pin its roots give two bases, it is sorted
    and counted `basis_undetermined` under the replay key the run writes. A sort it cannot
    take leaves it whole and counted under that reason alone, its line naming the pin."""
    stores = Stores()
    tree = _op(
        "aggregate",
        {"frame_ref": _op("compare", {"frame_ref": _root(0), "right_frame_ref": _root(1)})},
        unit=_CONSTANT,
    )
    # A key computed over identities the stored recipe does not name.
    stored_key = "rpk_" + "0" * 32 if sort == "unreproducible" else None
    old = _pinned(stores, tree, [_DAC, _IDS], options={"columns": _COLUMNS}, replay_key=stored_key)
    presentation = FakePresentationStore()
    presentation.state.component_revisions["cmp_drawn"] = [
        _revision("ops_pinned", "ocv_pinned", old["normal_form_hash"], 1)
    ]
    before = _stored(stores)

    skipped = run_normalise_recipes(datasets=stores.store, presentation=presentation)

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    (summary,) = [line for line in lines if line["event"] == "datasets_recipes_normalised"]
    (skip,) = [line for line in lines if line["event"] == "datasets_recipe_normalise_skipped"]
    assert sum(summary[key] for key in _OUTCOMES) == summary["recipes"] == 1
    assert skipped == 1
    assert {key: summary[key] for key in _COUNTS} == {
        "normalised": 0,
        "revisions": int(sort == "written"),
        "replay_key_held": 0,
        "unreproducible": int(sort == "unreproducible"),
        "basis_undetermined": int(sort == "written"),
    }
    sorted_roots = [{**root, "options": {"columns": sorted(_COLUMNS)}} for root in old["roots"]]
    _normal, key = _hashes({**old, "roots": sorted_roots})
    assert {k: skip[k] for k in ("reason", "replay_key", "parameter", "cause")} == {
        "reason": "basis_undetermined" if sort == "written" else "unreproducible",
        "replay_key": key,
        "parameter": "unit",
        "cause": "several_bases",
    }
    if sort == "unreproducible":
        assert _stored(stores) == before
        return
    (row,) = _stored(stores)
    assert [root["options"] for root in row["recipe"]["roots"]] == [
        {"columns": sorted(_COLUMNS)}
    ] * 2
    assert row["recipe"]["tree"] == tree
    assert row["replay_key"] == key
    assert presentation.state.component_revisions["cmp_drawn"][0]["frame_rendering"] == {
        "normal_form_hash": row["normal_form_hash"]
    }


# The `PinnedUnit` fields each op-spec model declares.
_PINS = {
    "AggregateOpSpec": ("unit",),
    "CompareOpSpec": ("left_unit", "right_unit"),
    "GrowthOpSpec": ("unit",),
    "RankOpSpec": ("unit",),
    "ShareOfOpSpec": ("numerator_unit", "denominator_unit"),
    "JoinOpSpec": ("left_unit", "right_unit"),
    "SumOfOpSpec": ("left_unit", "right_unit"),
    "QuotientOfOpSpec": ("numerator_unit", "denominator_unit"),
}


def test_every_unit_pin_of_an_op_spec_names_the_slot_feeding_it():
    """The migration rewrites the pins `_PINS` lists, and each names a frame slot the models
    that declare it hold, so a new `PinnedUnit` field fails here until it has one."""
    models = {model.__name__: model for model in get_args(get_args(OpSpec)[0])}
    declared = {
        name: tuple(
            n for n, field in model.model_fields.items() if field.annotation == PinnedUnit | None
        )
        for name, model in models.items()
    }

    assert {name: pins for name, pins in declared.items() if pins} == _PINS
    assert set(maintenance._PIN_SLOTS) == {pin for pins in _PINS.values() for pin in pins}
    for name, pins in _PINS.items():
        for pin in pins:
            slot = maintenance._PIN_SLOTS[pin]
            assert slot is None or slot in models[name].model_fields, (name, pin, slot)
