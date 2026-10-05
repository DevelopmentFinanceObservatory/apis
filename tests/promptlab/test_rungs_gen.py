"""Checks rung ablation, report ranking, and build-upward generation."""

from __future__ import annotations

import dataclasses

import pytest

from promptlab import rungs_gen
from promptlab.manifest import (
    PROMPTLAB_ROOT,
    BlockRef,
    RunConfig,
    Rung,
    StageSpec,
    load_rung,
    units,
)


def _load_production_rung() -> Rung:
    return load_rung(
        PROMPTLAB_ROOT / "rungs" / "production-extract.json",
        PROMPTLAB_ROOT / "rungs" / "production-bind.json",
    )


def _load_rung0() -> Rung:
    return load_rung(
        PROMPTLAB_ROOT / "rungs" / "rung0-extract.json",
        PROMPTLAB_ROOT / "rungs" / "rung0-bind.json",
    )


# ---------------------------------------------------------------------------
# Family sizes, derived from the manifests themselves -- the strong
# regression test against manifest drift.
# ---------------------------------------------------------------------------


def test_ablate_totals_24_for_the_landed_production_manifests():
    rung = _load_production_rung()

    expected_unit_count = sum(
        1
        for stage in (rung.extract, rung.bind)
        for _unit_name, refs in units(stage).items()
        if not any(block.core for block in refs)
    )
    expected_sub_block_count = sum(
        1
        for _unit_name, refs in units(rung.bind).items()
        if len(refs) >= 2 and not any(block.core for block in refs)
        for block in refs
        if block.id not in rungs_gen._CONFOUNDED_SUB_BLOCKS
    )

    generated = rungs_gen.ablate(rung, granularity="both")

    assert len(generated) == expected_unit_count + expected_sub_block_count
    assert len(generated) == 24  # pins the 17 unit + 7 sub-block arithmetic
    assert len(generated) == len({g.name for g in generated})  # every name is unique


def test_ablate_granularity_selects_the_family():
    rung = _load_production_rung()
    assert len(rungs_gen.ablate(rung, granularity="unit")) == 17
    assert len(rungs_gen.ablate(rung, granularity="block")) == 7
    assert len(rungs_gen.ablate(rung, granularity="both")) == 24


def test_ablate_raises_when_nothing_is_ablatable():
    core_only = Rung(
        name="all-core",
        description="d",
        extract=StageSpec(stage="extract", blocks=(BlockRef(id="r0-task", core=True),)),
        bind=StageSpec(stage="bind", blocks=(BlockRef(id="r0-role", core=True),)),
        run=RunConfig(capture="c", top_k=3, seeds=(1,), model="m", thinking={}),
    )
    with pytest.raises(rungs_gen.AblationError):
        rungs_gen.ablate(core_only, granularity="both")


# ---------------------------------------------------------------------------
# Unit / sub-block leave-one-out: exact composition
# ---------------------------------------------------------------------------


def test_unit_leave_one_out_drops_exactly_the_units_own_blocks():
    rung = _load_production_rung()
    by_name = {g.name: g for g in rungs_gen.unit_leave_one_out(rung)}

    also_plausible = by_name["abl-bind-also-plausible"]
    assert also_plausible.extract == rung.extract  # untouched sibling stage
    dropped = {b.id for b in rung.bind.blocks} - {b.id for b in also_plausible.bind.blocks}
    assert dropped == {
        "05a-alternate-definition",
        "05b-list-every",
        "05c-not-alternates",
        "05d-both-directions",
    }

    scope_words = by_name["abl-extract-04-scope-words"]
    assert scope_words.bind == rung.bind  # untouched sibling stage
    dropped = {b.id for b in rung.extract.blocks} - {b.id for b in scope_words.extract.blocks}
    assert dropped == {"04-scope-words"}


def test_unit_leave_one_out_never_drops_a_core_block():
    rung = _load_production_rung()
    core_ids = {b.id for b in rung.extract.blocks if b.core} | {
        b.id for b in rung.bind.blocks if b.core
    }
    for g in rungs_gen.unit_leave_one_out(rung):
        remaining = {b.id for b in g.extract.blocks} | {b.id for b in g.bind.blocks}
        assert core_ids <= remaining


def test_sub_block_leave_one_out_excludes_the_confounded_blocks():
    rung = _load_production_rung()
    generated = rungs_gen.sub_block_leave_one_out(rung)
    names = {g.name for g in generated}

    assert "abl-bind-05b-list-every" not in names
    assert "abl-bind-06a-header" not in names
    assert len(generated) == 7
    for g in generated:
        assert g.extract == rung.extract  # sub-block ablation never touches extract


# ---------------------------------------------------------------------------
# Writing generated rungs to disk
# ---------------------------------------------------------------------------


def test_write_rungs_produces_loadable_pairs_and_refuses_overwrite(tmp_path):
    rung = _load_production_rung()
    generated = rungs_gen.unit_leave_one_out(rung)[:2]
    run = dataclasses.replace(rung.run, capture="test-capture")

    written = rungs_gen.write_rungs(generated, out_dir=tmp_path / "rungs", run=run)
    assert len(written) == 2

    for g in generated:
        loaded = load_rung(
            tmp_path / "rungs" / f"{g.name}-extract.json",
            tmp_path / "rungs" / f"{g.name}-bind.json",
        )
        assert loaded.name == g.name
        assert loaded.run.capture == "test-capture"
        assert loaded.run.top_k == rung.run.top_k
        assert loaded.run.seeds == rung.run.seeds

    with pytest.raises(rungs_gen.AblationError, match="already exists"):
        rungs_gen.write_rungs(generated, out_dir=tmp_path / "rungs", run=run)


# ---------------------------------------------------------------------------
# ranking_from_report_json / unit_from_rung_name
# ---------------------------------------------------------------------------


def test_unit_from_rung_name():
    assert rungs_gen.unit_from_rung_name("abl-bind-also-plausible") == "also-plausible"
    assert rungs_gen.unit_from_rung_name("abl-extract-04-scope-words") == "04-scope-words"
    assert rungs_gen.unit_from_rung_name("production") is None
    assert rungs_gen.unit_from_rung_name("ceiling-e2e-123") is None


def test_ranking_from_report_json_band_normalised_two_sided_deviation():
    """Each metric contributes its distance outside the band divided by band width."""
    data = {
        "cells": [
            {
                "cell_id": "c1",
                "rung_name": "abl-bind-also-plausible",
                "n_seeds": 3,
                "metrics": {
                    "over_nomination_count_mean": {
                        "value": 0.2757,
                        "ceiling_band": [0.4938, 0.5679],
                    },
                    "expected_alternate_recall_mean": {
                        "value": 0.2583,
                        "ceiling_band": [0.3000, 0.3250],
                    },
                    "shape_election_match_rate": {
                        "value": 0.8333,
                        "ceiling_band": [0.8523, 0.8636],
                    },
                },
            },
        ]
    }

    ranking = rungs_gen.ranking_from_report_json(data)

    expected = (0.2181 / 0.0741 + 0.0417 / 0.0250 + 0.0190 / 0.0113) / 3
    assert ranking == pytest.approx({"also-plausible": expected})


def test_ranking_from_report_json_excludes_a_zero_width_band_metric():
    """Zero-width bands are excluded because their normalized distance is undefined."""
    data = {
        "cells": [
            {
                "cell_id": "c1",
                "rung_name": "abl-bind-also-plausible",
                "n_seeds": 3,
                "metrics": {
                    "shape_election_match_rate": {
                        "value": 0.5,
                        "ceiling_band": [0.9, 0.9],  # zero width
                    },
                    "over_nomination_count_mean": {
                        "value": 1.2,
                        "ceiling_band": [0.5, 0.9],
                    },
                },
            },
        ]
    }

    ranking = rungs_gen.ranking_from_report_json(data)

    assert ranking == pytest.approx({"also-plausible": 0.3 / 0.4})  # only the non-zero band counts


def test_ranking_from_report_json_scores_zero_inside_every_band():
    data = {
        "cells": [
            {
                "cell_id": "c1",
                "rung_name": "abl-bind-also-plausible",
                "n_seeds": 3,
                "metrics": {
                    "shape_election_match_rate": {"value": 0.92, "ceiling_band": [0.9, 0.95]},
                    "over_nomination_count_mean": {"value": 0.6, "ceiling_band": [0.5, 0.9]},
                },
            },
        ]
    }

    ranking = rungs_gen.ranking_from_report_json(data)

    assert ranking == pytest.approx({"also-plausible": 0.0})


def test_ranking_from_report_json_excludes_rows_without_a_band():
    """Rows without a ceiling band do not contribute to the ranking."""
    data = {
        "cells": [
            {
                "cell_id": "c1",
                "rung_name": "abl-bind-also-plausible",
                "n_seeds": 3,
                "metrics": {
                    "shape_election_match_rate": {"value": 0.5, "ceiling_band": [0.9, 0.95]},
                    "expected_alternate_recall_scored": {"value": 60.0},  # no ceiling_band
                },
            },
        ]
    }

    ranking = rungs_gen.ranking_from_report_json(data)

    assert ranking == pytest.approx({"also-plausible": 0.4 / 0.05})


# ---------------------------------------------------------------------------
# build_upward
# ---------------------------------------------------------------------------


def test_build_upward_orders_by_descending_contribution_and_produces_n_plus_1_rungs():
    base = _load_production_rung()
    rung0 = _load_rung0()
    ranking = {"also-plausible": 0.4, "04-scope-words": 0.05}

    generated = rungs_gen.build_upward(rung0, base, ranking)

    assert [g.name for g in generated] == ["build-0", "build-1", "build-2"]
    assert generated[0].bind == rung0.bind
    assert generated[0].extract == rung0.extract

    added_step1 = {b.id for b in generated[1].bind.blocks} - {b.id for b in rung0.bind.blocks}
    assert added_step1 == {
        "05a-alternate-definition",
        "05b-list-every",
        "05c-not-alternates",
        "05d-both-directions",
    }
    assert generated[1].extract == rung0.extract

    added_step2 = {b.id for b in generated[2].extract.blocks} - {b.id for b in rung0.extract.blocks}
    assert added_step2 == {"04-scope-words"}
    assert generated[2].bind == generated[1].bind  # bind unchanged from step 1


def test_build_upward_raises_on_a_ranking_naming_an_unknown_unit():
    base = _load_production_rung()
    rung0 = _load_rung0()
    with pytest.raises(rungs_gen.AblationError, match="not-a-real-unit"):
        rungs_gen.build_upward(rung0, base, {"not-a-real-unit": 1.0})


def test_build_upward_breaks_equal_scores_by_ascending_unit_id():
    """Equal scores sort by ascending unit ID."""
    base = _load_production_rung()
    rung0 = _load_rung0()
    ranking = {"also-plausible": 0.0, "04-scope-words": 0.0}

    generated = rungs_gen.build_upward(rung0, base, ranking)

    added_step1 = {b.id for b in generated[1].extract.blocks} - {b.id for b in rung0.extract.blocks}
    assert added_step1 == {"04-scope-words"}  # "0" sorts before "a"
    added_step2 = {b.id for b in generated[2].bind.blocks} - {
        b.id for b in generated[1].bind.blocks
    }
    assert added_step2 == {
        "05a-alternate-definition",
        "05b-list-every",
        "05c-not-alternates",
        "05d-both-directions",
    }
