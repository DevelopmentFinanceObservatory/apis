"""Check group B: skills, entry_path, structural, groundedness and task_outcome pass, fail
and stand aside (n/a) on recorded and minimal responses, and the registry carries every
check name. ``test_materialisation.py`` covers materialisation."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from search.eval.cases import CHECK_NAMES
from search.eval.checks import (
    CHECKS,
    MODAL_ONLY_CHECKS,
    ScoreContext,
    Verdict,
    WireSchema,
    score,
    score_results,
)
from tests.conftest import base_response, minimal_spec
from tests.eval.checks.conftest import make_case, panel, response_schema
from tests.eval.conftest import RaisingGraphClient, make_worked_example_response

PASS, FAIL, NA, U = Verdict.PASS, Verdict.FAIL, Verdict.NA, Verdict.UNVERIFIABLE


class KnownGraph:
    """Knows every dcid except ``missing``."""

    def __init__(self, missing: frozenset[str] = frozenset()):
        self.missing = missing
        self.calls: list[list[str]] = []

    def exists_batch(self, dcids: list[str]) -> set[str]:
        self.calls.append(dcids)
        return set(dcids) - self.missing


def ctx_with(**fields: Any) -> ScoreContext:
    return ScoreContext(schema=WireSchema.from_document({"response": response_schema()}), **fields)


def definite(
    skills: list[str] | None,
    *,
    warning_codes: tuple[str, ...] = (),
    skills_map: list[str] | None = None,
) -> dict[str, Any]:
    """A definite response; the envelope skills map defaults to the ids the spec names."""
    spec = minimal_spec()
    if skills is not None:
        spec["resolved"]["skills"] = skills
    response = base_response(status="definite", parts=[{"status": "definite", "spec": spec}])
    response["diagnostics"]["warnings"] = [
        {"code": code, "severity": "warn", "message": code} for code in warning_codes
    ]
    response["skills"] = {
        skill_id: {"id": skill_id}
        for skill_id in (skills or [] if skills_map is None else skills_map)
    }
    return response


def no_data_naming_a_skill() -> dict[str, Any]:
    """A definite part with no skills beside a no-data part whose flagged match names one."""
    response = definite([])
    flagged = minimal_spec("s2")
    flagged["resolved"]["skills"] = ["oda-concepts"]
    response["status"] = "mixed"
    response["parts"].append(
        {"status": "no_data", "no_data": {"reason": "no_observations", "flagged_match": flagged}}
    )
    return response


def skills_case(**skills: Any):
    return make_case({"definite": {"skills": skills}})


SKILLS_ROWS = [
    pytest.param(skills_case(present=True), definite(["oda-measures"]), PASS, id="present"),
    pytest.param(skills_case(present=True), definite([]), FAIL, id="present-none-found"),
    pytest.param(skills_case(present=False), definite([]), PASS, id="absent"),
    pytest.param(skills_case(present=False), definite(["oda-measures"]), FAIL, id="absent-found"),
    pytest.param(
        skills_case(present=True, ids=["oda-concepts"]),
        definite(["oda-concepts", "oda-measures"]),
        PASS,
        id="ids-held-extra-passes",
    ),
    pytest.param(
        skills_case(present=True, ids=["oda-concepts"]),
        definite(["oda-measures"]),
        FAIL,
        id="ids-missing",
    ),
    pytest.param(skills_case(present=True), definite(None), FAIL, id="skills-omitted"),
    pytest.param(
        skills_case(present=True),
        definite(["oda-measures"], warning_codes=("skills_index_unavailable",)),
        FAIL,
        id="index-warning",
    ),
    pytest.param(
        skills_case(present=True),
        definite(["oda-measures", "oda-concepts"], skills_map=["oda-measures"]),
        FAIL,
        id="orphan-in-the-map",
    ),
    pytest.param(
        make_case({"parts": [{"definite": {"skills": {"present": False}}}, {"no_data": {}}]}),
        no_data_naming_a_skill(),
        FAIL,
        id="orphan-on-a-flagged-match",
    ),
    pytest.param(make_case({"definite": {}}), definite(["oda-measures"]), NA, id="not-asserted"),
    pytest.param(skills_case(present=True), panel("crs-005"), Verdict.UNREACHED, id="no-data-part"),
]


@pytest.mark.parametrize(("case", "response", "verdict"), SKILLS_ROWS)
def test_skills(case, response, verdict, ctx: ScoreContext):
    result = CHECKS["skills"](case, response, ctx)
    assert result.verdict == verdict, result.detail


def test_skills_runs_inside_parts_under_a_per_part_expectation(ctx: ScoreContext):
    """Only the first part asserts skills, and it found none."""
    response = definite([])
    second = copy.deepcopy(response["parts"][0])
    second["spec"]["resolved"]["skills"] = ["oda-measures"]
    response["parts"].append(second)
    response["skills"] = {"oda-measures": {"id": "oda-measures"}}
    case = make_case({"parts": [{"definite": {"skills": {"present": True}}}, {"definite": {}}]})
    results = score_results(case, response, ctx)
    assert (results["skills"].verdict, results["parts"].verdict) == (FAIL, FAIL)
    assert results["skills"].detail.startswith("parts[0]")


def with_echo(response: dict[str, Any], *, skipped: bool) -> dict:
    response = copy.deepcopy(response)
    response["query_echo"]["extract_skipped"] = skipped
    return response


PARSED = make_case({"definite": {}}, door="parsed")
UNSUPPORTED_FACET = {
    "type": "/codes/facet_binding_unsupported",
    "title": "Facet binding unsupported",
    "status": 501,
    "detail": "the unit facet cannot bind XDR",
    "code": "facet_binding_unsupported",
    "trace_id": "0" * 32,
    "capability_version": "1",
    "retryable": False,
}

ENTRY_PATH_ROWS = [
    pytest.param(PARSED, with_echo(definite([]), skipped=True), PASS, id="parsed-skips"),
    pytest.param(PARSED, with_echo(definite([]), skipped=False), FAIL, id="echo-not-skipped"),
    pytest.param(make_case({"definite": {}}), definite([]), NA, id="raw-text"),
    pytest.param(PARSED, UNSUPPORTED_FACET, NA, id="problem-document"),
]


@pytest.mark.parametrize(("case", "response", "verdict"), ENTRY_PATH_ROWS)
def test_entry_path(case, response, verdict, ctx: ScoreContext):
    result = CHECKS["entry_path"](case, response, ctx)
    assert result.verdict == verdict, result.detail


def bad_schema_version() -> dict[str, Any]:
    response = make_worked_example_response()
    response["schema_version"] = 0
    return response


def bad_body() -> dict[str, Any]:
    response = make_worked_example_response()
    del response["parts"][0]["spec"]["spec_id"]
    return response


@pytest.mark.parametrize(
    ("response", "side", "verdict"),
    [
        pytest.param(make_worked_example_response(), "head", PASS, id="valid"),
        pytest.param(bad_schema_version(), "head", FAIL, id="schema-version"),
        pytest.param(bad_body(), "head", FAIL, id="invalid-body"),
        pytest.param(bad_body(), "base", NA, id="base-side"),
        pytest.param(UNSUPPORTED_FACET, "head", PASS, id="problem-document"),
        pytest.param(
            {k: v for k, v in UNSUPPORTED_FACET.items() if k != "title"},
            "head",
            FAIL,
            id="invalid-problem-document",
        ),
        pytest.param(UNSUPPORTED_FACET, "base", NA, id="problem-document-base-side"),
    ],
)
def test_structural(response, side, verdict):
    result = CHECKS["structural"](make_case({"definite": {}}), response, ctx_with(side=side))
    assert result.verdict == verdict, result.detail


def test_groundedness_passes_when_every_reference_exists():
    graph = KnownGraph()
    result = CHECKS["groundedness"](
        make_case({"definite": {}}), make_worked_example_response(), ctx_with(graph=graph)
    )
    assert result.verdict == PASS, result.detail
    (asked,) = graph.calls
    assert {"country/ETH", "ODAGrants", "DAC/Health"} <= set(asked)


def first_value_slot(resolved: dict[str, Any]) -> tuple[int, str]:
    return next(
        (i, s["binding"]["value"]["ref"]["ref"])
        for i, s in enumerate(resolved["slots"])
        if s["binding"]["kind"] == "value"
    )


@pytest.mark.parametrize("where", ["shape", "slot", "entity"])
def test_groundedness_names_the_path_of_a_fabricated_reference(where: str):
    response = panel("crs-001")
    resolved = response["parts"][0]["spec"]["resolved"]
    at = "/parts/0/spec/resolved"
    slot, slot_dcid = first_value_slot(resolved)
    fabricated, path = {
        "shape": (resolved["shape"]["population_type"]["ref"], f"{at}/shape/population_type/ref"),
        "slot": (slot_dcid, f"{at}/slots/{slot}/binding/value/ref/ref"),
        "entity": (resolved["entities"][0]["ref"], f"{at}/entities/0/ref"),
    }[where]
    result = CHECKS["groundedness"](
        make_case({"definite": {}}), response, ctx_with(graph=KnownGraph(frozenset({fabricated})))
    )
    assert result.verdict == FAIL
    assert f"{path}={fabricated}" in result.detail


def test_groundedness_walks_candidates_and_equivalent_readings():
    response = panel("crs-021")
    specs = response["parts"][0]["candidates"]["specs"]
    twin = copy.deepcopy(specs[0])
    twin["resolved"].pop("equivalent_readings", None)
    twin["resolved"]["shape"]["population_type"]["ref"] = "TwinPop"
    specs[-1]["resolved"]["equivalent_readings"] = [twin]
    graph = KnownGraph(frozenset({"TwinPop"}))
    result = CHECKS["groundedness"](make_case({"candidates": {}}), response, ctx_with(graph=graph))
    assert result.verdict == FAIL
    last = len(specs) - 1
    assert f"/parts/0/candidates/specs/{last}/resolved/equivalent_readings/0/" in result.detail
    assert specs[0]["resolved"]["entities"][0]["ref"] in graph.calls[0]


@pytest.mark.parametrize(
    ("graph", "response", "verdict", "infra"),
    [
        pytest.param(RaisingGraphClient(), panel("crs-001"), U, True, id="graph-error"),
        pytest.param(None, panel("crs-001"), U, True, id="no-graph-client"),
        pytest.param(
            KnownGraph(), {"kind": "declined", "reason": "out_of_scope"}, NA, False, id="no-refs"
        ),
    ],
)
def test_groundedness_infra_and_nothing_to_check(graph, response, verdict, infra: bool):
    result = CHECKS["groundedness"](make_case({"definite": {}}), response, ctx_with(graph=graph))
    assert (result.verdict, result.infra) == (verdict, infra)


def test_groundedness_lets_an_error_that_is_no_read_failure_propagate():
    """A bug in the check or its client stops the run; it is no infrastructure result."""
    graph = RaisingGraphClient(ValueError("a bug"))
    with pytest.raises(ValueError, match="a bug"):
        CHECKS["groundedness"](make_case({"definite": {}}), panel("crs-001"), ctx_with(graph=graph))


def test_groundedness_reads_a_renamed_field_as_unverifiable():
    """Base renamed `entity_type`, so the references under it cannot be read there."""
    renamed = copy.deepcopy(response_schema())
    entity = renamed["$defs"]["ResolvedEntity"]["properties"]
    entity["entity_kind"] = entity.pop("entity_type")
    ctx = ScoreContext(
        schema=WireSchema.from_document({"response": renamed}), graph=KnownGraph(), side="base"
    )
    result = score_results(make_case({"definite": {}}), panel("crs-001"), ctx)["groundedness"]
    assert (result.verdict, result.infra) == (U, False)


@pytest.mark.parametrize(
    ("response", "verdict"),
    [
        pytest.param(panel("crs-001"), PASS, id="answer"),
        pytest.param({"task_error": "ReadTimeout: timed out"}, FAIL, id="task-error"),
    ],
)
def test_task_outcome(response, verdict, ctx: ScoreContext):
    assert CHECKS["task_outcome"](make_case({"definite": {}}), response, ctx).verdict == verdict


def test_score_skips_the_modal_only_checks_on_other_samples():
    case = make_case({"definite": {}})
    verdicts = score(case, panel("crs-001"), ctx_with(graph=KnownGraph()), skip=MODAL_ONLY_CHECKS)
    assert list(verdicts) == [name for name in CHECK_NAMES if name != "materialisation"]
