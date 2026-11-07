"""`maintenance count-pinned-units`: the derived frames whose recipes pin a constant-price unit
without its deflation basis, and the cards drawing them, over the SQLite dataset store double
and the Presentation store fake."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from datasets_api import maintenance
from datasets_api.maintenance import main, run_count_pinned_units
from tests.presentation_store.conftest import FakePresentationStore

from ._version_record_store import FakeVersionRecordDatasetStore


def _node(
    parameters: Mapping[str, object], child: Mapping[str, object], *, verb: str = "aggregate"
) -> dict[str, object]:
    return {
        "frame": {"spec_id": "ops_node", "content_version": "ocv_1"},
        "op_spec": {"verb": verb, "parameters": dict(parameters)},
        "inputs": {"frame_ref": child},
    }


def _recipe(
    datasets: FakeVersionRecordDatasetStore,
    spec_id: str,
    roots: list[str],
    *,
    outer: Mapping[str, object] | None = None,
    inner: Mapping[str, object] | None = None,
    outer_verb: str = "aggregate",
    superseded: bool = False,
) -> None:
    """Store a derived frame's recipe whose roots are the `cv_1` versions of `roots`: an
    `outer_verb` call over an aggregate of the first root, with the `outer` and `inner`
    parameters."""
    root = {"frame": {"spec_id": roots[0], "content_version": "cv_1"}, "root": 0}
    recipe = {
        "tree": _node(outer or {}, _node(inner or {}, root), verb=outer_verb),
        "roots": [{"dataset": {"spec_id": root, "content_version": "cv_1"}} for root in roots],
    }
    with datasets.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.frame_recipe (subject_id, content_version, normal_form_hash, "
            "replay_key, recipe, recorded_at, superseded_at) "
            "VALUES (%s, 'ocv_1', %s, %s, %s, '2026-11-01T00:00:00+00:00', %s)",
            (
                spec_id,
                f"rnf_{spec_id}",
                f"rpk_{spec_id}",
                json.dumps(recipe),
                "2026-11-02T00:00:00+00:00" if superseded else None,
            ),
        )


def _revision(revision: int, spec_id: str, select: Mapping[str, object]) -> dict[str, Any]:
    content_version = "ocv_1" if spec_id.startswith("ops_") else "cv_1"
    return {
        "revision": revision,
        "stored_intent": {
            "frame_ref": {"spec_id": spec_id, "content_version": content_version},
            "select": select,
        },
    }


def _share(
    presentation: FakePresentationStore,
    share_id: str,
    cards: list[tuple[str, int]],
    *,
    mode: str = "live",
    deleted: bool = False,
) -> None:
    state = presentation.state
    state.shares[share_id] = {"mode": mode, "deleted_at": "2026-11-02" if deleted else None}
    state.share_revisions[share_id] = [{"revision": 1}, {"revision": 2}]
    state.share_slots[(share_id, 2)] = [
        {"slot": slot, "component_id": component_id, "component_revision": revision}
        for slot, (component_id, revision) in enumerate(cards, start=1)
    ]


def test_counts_the_recipes_pinning_a_unit_without_its_basis_and_the_cards_drawing_them(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """A live recipe pinning a constant-price unit with no basis, on a composed side or deeper
    in the tree, is named, and a card drawing a frame of such a recipe counts its head and the
    live slots showing any of its revisions, superseded recipe or not, a source card included.
    A pin naming its basis, current prices, a unit ref, a derived frame comparing two roots, a
    card on a dataset frame binding a key with no basis, a snapshot, a taken-down share and a
    revision that is neither a head nor live count nothing."""
    datasets = FakeVersionRecordDatasetStore()
    # Two roots deflated apart refuse `cross_frame_unit_mismatch` only on their rows.
    _recipe(datasets, "ops_dac_and_ids", ["spec_dac", "spec_ids"])
    usd_2024 = {"currency": "USD", "prices": "constant", "base_year": 2024}
    per_person = {"stat_var": "dcid:Count_Person"}
    _recipe(datasets, "ops_pinned", ["spec_dac"], outer={"unit": usd_2024})
    _recipe(
        datasets,
        "ops_pinned_inner",
        ["spec_dac"],
        inner={"unit": {"numerator": per_person, "denominator": usd_2024}},
    )
    _recipe(
        datasets,
        "ops_pinned_numerator",
        ["spec_dac"],
        outer={"numerator_unit": usd_2024},
        outer_verb="quotient_of",
    )
    _recipe(datasets, "ops_pinned_gone", ["spec_dac"], outer={"unit": usd_2024}, superseded=True)
    _recipe(
        datasets,
        "ops_pin_based",
        ["spec_dac"],
        outer={"unit": {**usd_2024, "deflation_basis": "donor"}},
    )
    _recipe(
        datasets,
        "ops_pin_current",
        ["spec_dac"],
        outer={"unit": {"currency": "USD", "prices": "current"}},
    )
    _recipe(datasets, "ops_pin_ref", ["spec_dac"], outer={"unit": "USDollar"})
    presentation = FakePresentationStore()
    revisions = presentation.state.component_revisions
    revisions["cmp_pinned"] = [_revision(1, "ops_pinned", {}), _revision(2, "ops_pinned", {})]
    # A refresh replays the recipe of the pair the card holds, superseded or not.
    revisions["cmp_pinned_gone"] = [_revision(1, "ops_pinned_gone", {})]
    revisions["cmp_pinned_source"] = [
        {
            "revision": 1,
            "stored_intent": {
                "type": "source_card",
                "frame_ref": {"spec_id": "ops_pinned", "content_version": "ocv_1"},
            },
        }
    ]
    revisions["cmp_pin_based"] = [_revision(1, "ops_pin_based", {})]
    revisions["cmp_derived_apart"] = [_revision(1, "ops_dac_and_ids", {})]
    revisions["cmp_derived_no_recipe"] = [_revision(1, "ops_unrecorded", {})]
    # On a rebuilt frame of one unit the binding follows it (`StoredSelect` rule 4).
    revisions["cmp_bound"] = [_revision(1, "spec_dac", {"unit": "USD/constant/2024"})]
    revisions["cmp_note"] = [{"revision": 1, "stored_intent": {"select": {}}}]
    _share(presentation, "live0001", [("cmp_pinned", 1), ("cmp_bound", 1)])
    _share(presentation, "snap0001", [("cmp_pin_based", 1)], mode="snapshot")
    _share(presentation, "gone0001", [("cmp_pinned_gone", 1)], deleted=True)

    # Seven live recipes, read two at a time.
    monkeypatch.setattr(maintenance, "RECIPE_SCAN_PAGE", 2)

    run_count_pinned_units(presentation=presentation, datasets=datasets)

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    (counted,) = [line for line in lines if line["event"] == "datasets_pinned_units_counted"]
    assert {key: value for key, value in counted.items() if key != "event"} == {
        "severity": "WARNING",
        "pinned_heads": 3,
        "pinned_live_slots": 1,
        "pinned_components": ["cmp_pinned", "cmp_pinned_gone", "cmp_pinned_source"],
        "pinned_recipe_count": 3,
        "pinned_recipes": [
            "ops_pinned@ocv_1",
            "ops_pinned_inner@ocv_1",
            "ops_pinned_numerator@ocv_1",
        ],
    }


def test_the_task_is_registered(monkeypatch: pytest.MonkeyPatch):
    ran: list[str] = []
    monkeypatch.setattr(
        "datasets_api.maintenance._count_pinned_units", lambda: ran.append("ran") or 0
    )
    assert main(["count-pinned-units"]) == 0
    assert ran == ["ran"]


def test_the_task_exits_0_when_it_counts_affected_cards(monkeypatch: pytest.MonkeyPatch):
    """The deploy runs the task and reads its counts from the logs, so a count fails no
    deploy."""
    from presentation_store import provider as presentation_provider

    for module in (maintenance, presentation_provider):
        monkeypatch.setattr(module, "shared_pool", lambda: None)
        monkeypatch.setattr(module, "close_shared_pool", lambda: None)
    monkeypatch.setattr(maintenance, "LiveDatasetStore", lambda *, connections: connections)
    monkeypatch.setattr(
        maintenance,
        "run_count_pinned_units",
        lambda **_stores: maintenance.emit(
            "datasets_pinned_units_counted", severity="WARNING", pinned_recipe_count=1
        ),
    )

    assert main(["count-pinned-units"]) == 0
