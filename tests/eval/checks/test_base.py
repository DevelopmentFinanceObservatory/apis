"""Tests for the wire schema, path reads, verdict combination and the check registry."""

from __future__ import annotations

import copy

import pytest

from search.eval.cases import CHECK_NAMES
from search.eval.checks import (
    CHECKS,
    PART_CHECKS,
    Result,
    ScoreContext,
    Unverifiable,
    Verdict,
    WireSchema,
    read_path,
    register,
    score,
    score_results,
)
from search.eval.checks.base import all_of, answer_result, any_of, infra_error
from tests.eval.checks.conftest import make_case, panel, response_schema

P, F, U, NA = (Result(v) for v in (Verdict.PASS, Verdict.FAIL, Verdict.UNVERIFIABLE, Verdict.NA))


@pytest.mark.parametrize(
    ("path", "declared"),
    [
        ("kind", True),
        ("status", True),
        ("diagnostics.warnings[].code", True),
        ("parts[].no_data.reason", True),
        ("parts[].spec.definition.identity.requested_window.start_year", True),
        # A facets key is any axis name.
        ("parts[].spec.resolved.facets.provenance.data_commons.id", True),
        ("parts[].spec.definition.identity.facets.unit.request.currency", True),
        # Unsupported.definition is a free-form object.
        ("parts[].unsupported.definition.identity.slots[].binding.state", True),
        ("parts[].spec.definition.identity.requested_period", False),
        ("parts[].no_data.reasons", False),
        ("scope.place.dcid", True),
    ],
)
def test_wire_schema_declares_paths(path, declared):
    schema = WireSchema.from_document({"response": response_schema()})
    assert schema.has(path) is declared


def test_read_path_reads_an_omitted_field_as_none(ctx: ScoreContext):
    response = panel("crs-001")
    part = response["parts"][0]
    assert read_path(part, "spec.definition.identity.facets", ctx, at="parts[]") is None
    assert (
        read_path(part, "spec.definition.identity.requested_window.kind", ctx, at="parts[]")
        == "all"
    )


def test_read_path_raises_unverifiable_for_a_path_outside_the_schema(ctx: ScoreContext):
    with pytest.raises(Unverifiable) as raised:
        read_path({}, "no_data.reasons", ctx, at="parts[]")
    assert raised.value.path == "parts[].no_data.reasons"


@pytest.mark.parametrize(
    ("results", "all_verdict", "any_verdict"),
    [
        ([P, F, U], Verdict.FAIL, Verdict.PASS),
        ([P, U], Verdict.UNVERIFIABLE, Verdict.PASS),
        ([F, U], Verdict.FAIL, Verdict.UNVERIFIABLE),
        ([P, NA], Verdict.PASS, Verdict.PASS),
        ([NA, NA], Verdict.NA, Verdict.NA),
        ([], Verdict.NA, Verdict.NA),
    ],
)
def test_all_of_and_any_of(results, all_verdict, any_verdict):
    assert all_of(results).verdict == all_verdict
    assert any_of(results).verdict == any_verdict


@pytest.mark.parametrize(
    ("results", "known", "expected"),
    [
        ({"measure": P, "entities": F, "disposition": F}, (), Result(Verdict.FAIL, "entities")),
        ({"measure": U, "entities": F}, ("entities",), Result(Verdict.UNVERIFIABLE, "measure")),
        ({"measure": P, "skills": NA}, (), Result(Verdict.PASS)),
        (
            {"measure": F, "groundedness": infra_error("down")},
            (),
            Result(Verdict.FAIL, "measure", infra=True),
        ),
    ],
    ids=["first failure", "known left out", "n/a passes", "infra carries over"],
)
def test_answer_combines_the_checks_outside_the_known_failures(results, known, expected):
    assert answer_result(results, known=known) == expected


def test_every_check_name_is_registered():
    group_a = {
        "disposition",
        "measure",
        "entities",
        "provenance",
        "request",
        "operations",
        "refusal",
        "warnings",
        "candidates",
        "parts",
    }
    assert set(CHECKS) == set(CHECK_NAMES)
    assert set(PART_CHECKS) == group_a - {"disposition", "parts"} | {"skills"}


def test_a_name_registers_once():
    with pytest.raises(ValueError, match="already registered"):
        register("measure")(lambda case, response, ctx: P)


def test_score_reports_every_registered_check_in_check_name_order(ctx: ScoreContext):
    case = make_case({"definite": {}})
    verdicts = score(case, panel("crs-001"), ctx)
    assert list(verdicts) == [name for name in CHECK_NAMES if name in CHECKS]
    assert verdicts["disposition"] == Verdict.PASS
    assert verdicts["measure"] == Verdict.NA


def test_a_renamed_field_scores_unverifiable_and_leaves_other_checks_alone():
    """Base renamed `requested_window`: the window cannot be verified there, though the
    body still carries the old key, and the entities check still passes."""
    renamed = copy.deepcopy(response_schema())
    identity = renamed["$defs"]["BaseIdentity"]["properties"]
    identity["requested_period"] = identity.pop("requested_window")
    ctx = ScoreContext(schema=WireSchema.from_document({"response": renamed}), side="base")
    case = make_case(
        {
            "window": {"kind": "all"},
            "entities": [
                {"dcid": "country/DEU", "role": "origin"},
                {"dcid": "country/KEN", "role": "destination"},
            ],
            "definite": {},
        }
    )
    results = score_results(case, panel("crs-001"), ctx)
    assert results["request"].verdict == Verdict.UNVERIFIABLE
    assert results["entities"].verdict == Verdict.PASS
