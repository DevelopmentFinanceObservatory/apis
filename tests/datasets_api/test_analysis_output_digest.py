"""The output digest against the client contract's `output-digest` vectors, each frame built
through the frame grammar as the runner holds it, and the names it gives the variables the
runner's operations write."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

import pytest
from dfo_frame_grammar import Frame

from datasets_api.analysis import AnalysisRequest, compute_analysis
from datasets_api.output_digest import digest_document, minted_variable_names, output_digest
from dfo_analysis_engine import execution_identity
from materialise.frame_build import build_frame

from ._analysis_doubles import (
    AID,
    AID_V1,
    POPULATION,
    POPULATION_V1,
    Stores,
    aid_frame,
    dataset,
    population_frame,
    source,
    step,
)

_VECTORS: list[dict[str, Any]] = json.loads(
    (files("dfo_client_contract") / "rules" / "output-digest.json").read_text(encoding="utf-8")
)["vectors"]
_FIXED = frozenset(
    {"stat_var", "time_period", "value", "unit", "measurementMethod", "provenance", "status"}
)


@pytest.mark.parametrize("vector", _VECTORS, ids=[vector["name"] for vector in _VECTORS])
def test_each_vector_frame_digests_to_its_document_and_digest(vector: dict[str, Any]) -> None:
    columns = vector["frame"]["columns"]
    roles = [column for column in columns if column not in _FIXED]
    frame = build_frame(
        role_keys=roles,
        rows=[
            {column: cell for column, cell in zip(columns, row, strict=True) if column != "status"}
            for row in vector["frame"]["rows"]
        ],
    )

    names = vector.get("variables", {})

    assert output_digest(frame, names) == vector["digest"]
    picked = ["stat_var", *roles, "time_period", "value", "unit"]
    rows = [[row[columns.index(name)] for name in picked] for row in vector["frame"]["rows"]]
    for row in rows:
        row[0] = names.get(row[0].removeprefix("dcid:"), row[0])
    assert digest_document(role_columns=roles, rows=rows).decode() == vector["document"]


def _digest(tree: dict, *, frames: dict | None = None) -> str:
    stores = Stores()
    roots = source(stores, **({"frames": frames} if frames is not None else {}))
    request = AnalysisRequest.model_validate({"tree": tree})
    return compute_analysis(request, source=roots).output_digest


def _per_person(numerator: object) -> dict:
    return step(
        "quotient_of", numerator, dataset(POPULATION, POPULATION_V1), on={"destination": "place"}
    )


_EVERY_ROW = {
    "predicates": [{"kind": "ordered_threshold", "column": "value", "op": "gte", "value": 0}]
}


def _filtered(data: object) -> dict:
    return step("filter", data, **_EVERY_ROW)


def test_a_quotient_and_a_filter_then_the_quotient_keeping_its_rows_share_a_digest() -> None:
    one_step = _digest(_per_person(dataset(AID, AID_V1)))

    assert _digest(_per_person(_filtered(dataset(AID, AID_V1)))) == one_step


def test_a_quotient_with_its_sides_swapped_holds_another_digest() -> None:
    # Both sides hold the same values under two variables, so only the names tell them apart.
    frames = {
        (AID, AID_V1): aid_frame(),
        (POPULATION, POPULATION_V1): aid_frame(stat_var="dcid:Amount_OOF"),
    }
    on = {"destination": "destination"}
    numerator, denominator = dataset(AID, AID_V1), dataset(POPULATION, POPULATION_V1)

    forward = _digest(step("quotient_of", numerator, denominator, on=on), frames=frames)
    swapped = _digest(step("quotient_of", denominator, numerator, on=on), frames=frames)

    assert forward != swapped


@pytest.mark.parametrize("change", ["engine_release", "root_rebuilt"])
def test_an_identity_change_that_moves_no_number_keeps_the_digest(
    change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = _per_person(_filtered(dataset(AID, AID_V1)))
    before = _digest(tree)
    if change == "engine_release":
        monkeypatch.setattr(execution_identity, "ENGINE_VERSION", "99.0.0")
        after = _digest(tree)
    else:
        rebuilt = "cv_" + "9" * 32
        frames = {(AID, rebuilt): aid_frame(), (POPULATION, POPULATION_V1): population_frame()}
        after = _digest(_per_person(_filtered(dataset(AID, rebuilt))), frames=frames)

    assert after == before


def test_a_join_of_two_operations_with_values_under_swapped_variables_differs() -> None:
    def minted(spec_id: str, verb: str, second_slot: str) -> dict:
        inputs = {"frame_ref": {"root": 0}, second_slot: {"root": 1}}
        return {"frame": {"spec_id": spec_id}, "op_spec": {"verb": verb}, "inputs": inputs}

    tree = {
        "frame": {"spec_id": "ops_" + "c" * 32},
        "op_spec": {"verb": "join"},
        "inputs": {
            "frame_ref": minted("ops_" + "a" * 32, "quotient_of", "denominator_frame_ref"),
            "right_frame_ref": minted("ops_" + "b" * 32, "sum_of", "right_frame_ref"),
        },
    }
    names = minted_variable_names(tree, {0: ["dcid:Amount_ODA"], 1: ["dcid:Count_Person"]})

    def joined(first: float, second: float) -> Frame:
        rows = [
            {
                "stat_var": f"dcid:ops_{letter * 32}",
                "destination": "country/KEN",
                "time_period": "2022",
                "value": value,
                "unit": "USDollar",
            }
            for letter, value in (("a", first), ("b", second))
        ]
        return build_frame(role_keys=["destination"], rows=rows)

    assert output_digest(joined(1.0, 2.0), names) != output_digest(joined(2.0, 1.0), names)
