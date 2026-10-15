"""The HTTP eval runner: request building, retries, degraded and rejected samples, the vote."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from search.eval import runner
from search.eval.apis_auth import ApisAuth, ApisAuthError, apis_auth
from search.eval.cases import Case, validate_case
from search.eval.runner import (
    DEGRADED_LEG_HEADER,
    CaseRun,
    Sample,
    build_request,
    extend_run,
    fetch_schema,
    members_door_reader,
    modal_sample,
    run_cases,
)

_RESUBMIT = {
    "variable_text": "debt disbursements from the IMF, 2015 to 2024",
    "shape": {
        "kind": "statvar_schema",
        "population_type": "FinancialFlow",
        "measured_property": "amount",
        "stat_type": "measuredValue",
        "measurement_qualifier": "Nominal",
    },
    "slots": [
        {
            "key": {
                "axis": "how",
                "property": {"ref": "financialFlowType", "label": "financial flow type"},
                "label": "financial flow type",
            },
            "binding": {
                "kind": "value",
                "value": {
                    "ref": {"ref": "Disbursement", "label": "disbursement"},
                    "value_kind": "enum_value",
                    "time_window": None,
                    "literal": None,
                },
            },
        }
    ],
    "entity_dcids": ["country/TZA"],
    "origin_dcids": ["InternationalMonetaryFund"],
}


def _case(case_id: str = "case-1", **fields) -> Case:
    raw = {
        "id": case_id,
        "query": "population of Kenya",
        "why": "Count_Person answers.",
        "expect": {"definite": {}},
    } | fields
    return validate_case(raw, families={})


def _definite(
    *,
    spec_id: str = "s1",
    served_count: int = 1,
    member_digest: str | None = None,
    entities: tuple[tuple[str, str], ...] = (("country/KEN", "observationAbout"),),
    operations: list[dict] | None = None,
    twins: list[str] | None = None,
    timed_out: bool = False,
) -> dict:
    resolved: dict = {
        "shape": {"member_digest": member_digest},
        "stat_vars": {"served_count": served_count},
        "entities": [{"ref": ref, "role": role} for ref, role in entities],
    }
    if twins is not None:
        resolved["equivalent_readings"] = [{"spec_id": twin} for twin in twins]
    part: dict = {"status": "definite", "spec": {"spec_id": spec_id, "resolved": resolved}}
    if operations is not None:
        part["operations"] = operations
    return {"kind": "resolution", "diagnostics": {"timed_out": timed_out}, "parts": [part]}


def _sample(body: dict, status: int = 200) -> Sample:
    return Sample(
        body=body, status=status, degraded=False, request_rejected=False, attempts=1, elapsed_s=0
    )


class _Script:
    """Replay one scripted reply per post and record each posted body."""

    def __init__(self, *replies: httpx.Response | Exception) -> None:
        self.replies = list(replies)
        self.posted: list[dict] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.posted.append(json.loads(request.content))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _run(cases: list[Case], handler, *, samples: int = 1, concurrency: int = 4):
    async def go() -> dict[str, CaseRun]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await run_cases(
                cases,
                url="http://app/",
                samples=samples,
                concurrency=concurrency,
                client=client,
                backoff_s=0,
            )

    return asyncio.run(go())


# Requests


@pytest.mark.parametrize(
    ("fields", "expected_input", "expected_options"),
    [
        (
            {},
            {"kind": "raw_text", "query": "population of Kenya"},
            {"include_skills": True},
        ),
        (
            {"include_observations": True},
            {"kind": "raw_text", "query": "population of Kenya"},
            {"include_skills": True, "include_observations": True},
        ),
        (
            {
                "door": "parsed",
                "parsed": [{"variable_text": "population", "entities": [{"text": "Kenya"}]}],
            },
            {
                "kind": "parsed",
                "parts": [{"variable_text": "population", "entities": [{"text": "Kenya"}]}],
                "raw_query": "population of Kenya",
            },
            {"include_skills": True},
        ),
    ],
    ids=["raw_text", "observations", "parsed"],
)
def test_the_request_holds_only_the_fields_the_case_sets(fields, expected_input, expected_options):
    assert build_request(_case(**fields)) == {
        "input": expected_input,
        "response_options": expected_options,
    }


def test_a_resubmit_request_carries_one_identity_entry_without_null_fields():
    request = build_request(_case(door="dataset_resubmit", resubmit=_RESUBMIT))

    (entry,) = request["input"]["entries"]
    assert request["input"]["kind"] == "dataset_resubmit"
    assert entry["entry_kind"] == "identity"
    assert entry["variable_text"] == _RESUBMIT["variable_text"]
    assert set(entry) == {"entry_kind", "variable_text", "identity"}
    assert None not in entry.values()
    assert "schema_version" not in request


def test_the_runner_posts_the_built_request_to_the_resolve_route():
    case = _case()
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_definite())

    _run([case], handler)

    assert [str(r.url) for r in seen] == ["http://app/api/search/resolve"]
    assert json.loads(seen[0].content) == build_request(case)


# Retries and sample flags


@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(503, json={"code": "service_unavailable"}),
        httpx.Response(500, json={"code": "internal_error"}),
        httpx.Response(504, text="<html>gateway timeout</html>"),
        httpx.Response(504, json={"code": "upstream_timeout"}),
        httpx.Response(200, text="<html>proxy</html>"),
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
    ],
    ids=["503", "500", "504-gateway", "504-other-code", "non-json", "connect", "timeout"],
)
def test_an_infra_failure_is_retried_and_the_answer_scored(failure):
    script = _Script(failure, httpx.Response(200, json=_definite()))

    run = _run([_case()], script)["case-1"]

    assert run.infra_error is None
    assert run.modal is not None
    assert (run.modal.attempts, run.modal.degraded, run.modal.status) == (2, False, 200)
    assert len(script.posted) == 2


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        (httpx.Response(503, json={"code": "service_unavailable"}), "HTTP 503"),
        (httpx.Response(504, text="<html>gateway timeout</html>"), "HTTP 504"),
        (httpx.Response(504, json={"code": "upstream_timeout"}), "HTTP 504"),
    ],
    ids=["503", "504-gateway", "504-other-code"],
)
def test_a_sample_failing_three_times_marks_the_case_infra(failure, error):
    script = _Script(*[failure] * 3)

    run = _run([_case()], script)["case-1"]

    assert run.infra_error == error
    assert run.modal is None
    assert run.samples == ()
    assert len(script.posted) == 3


def test_the_apps_own_request_timeout_is_infra_and_is_not_retried():
    script = _Script(
        httpx.Response(504, json={"code": "timeout", "status": 504}),
        httpx.Response(200, json=_definite()),
    )

    run = _run([_case()], script)["case-1"]

    assert run.infra_error == "HTTP 504"
    assert (run.modal, run.samples) == (None, ())
    assert len(script.posted) == 1


def test_an_app_error_after_its_retries_is_kept_as_the_sample():
    """A 500 is the app's own crash, so the checks score it; 502, 503 and 504 are infra."""
    script = _Script(*[httpx.Response(500, json={"code": "internal_error", "status": 500})] * 3)

    run = _run([_case()], script)["case-1"]

    assert run.infra_error is None
    assert run.modal is not None
    assert (run.modal.status, run.modal.attempts, run.modal.degraded) == (500, 3, False)
    assert run.modal.body["code"] == "internal_error"


def test_an_infra_failure_after_an_app_error_marks_the_case_infra():
    script = _Script(
        httpx.Response(500, json={"code": "internal_error", "status": 500}),
        httpx.Response(500, json={"code": "internal_error", "status": 500}),
        httpx.Response(503, json={"code": "service_unavailable"}),
    )

    run = _run([_case()], script)["case-1"]

    assert (run.infra_error, run.modal) == ("HTTP 503", None)


@pytest.mark.parametrize(
    ("body", "headers"),
    [
        (_definite(timed_out=True), {}),
        (_definite(), {DEGRADED_LEG_HEADER: "1"}),
    ],
    ids=["timed_out", "header"],
)
def test_a_degraded_answer_is_retried_twice_then_kept_and_marked(body, headers, caplog):
    last = {**body, "marker": "third"}
    script = _Script(
        httpx.Response(200, json=body, headers=headers),
        httpx.Response(200, json=body, headers=headers),
        httpx.Response(200, json=last, headers=headers),
    )

    with caplog.at_level("WARNING", logger="search.eval.runner"):
        run = _run([_case()], script)["case-1"]

    assert run.modal is not None
    assert (run.modal.degraded, run.modal.attempts) == (True, 3)
    assert run.modal.body["marker"] == "third"
    assert len(script.posted) == 3
    cause = "timed_out" if headers == {} else f"{DEGRADED_LEG_HEADER} header"
    retries = [r.getMessage() for r in caplog.records if "eval retry" in r.getMessage()]
    assert retries == [f"eval retry: case=case-1 attempt={n} cause={cause}" for n in (1, 2, 3)]


def test_a_degraded_answer_followed_by_a_clean_one_is_not_degraded():
    script = _Script(
        httpx.Response(200, json=_definite(), headers={DEGRADED_LEG_HEADER: "1"}),
        httpx.Response(200, json=_definite()),
    )

    run = _run([_case()], script)["case-1"]

    assert run.modal is not None
    assert (run.modal.degraded, run.modal.attempts) == (False, 2)


@pytest.mark.parametrize(
    ("status", "code", "rejected"),
    [
        (400, "request_invalid", True),
        (422, "unsupported_schema_version", True),
        (422, "query_too_long", False),
        (422, "facet_not_available", False),
        (501, "door_not_implemented", False),
    ],
)
def test_a_problem_answer_is_scored_once_and_flags_a_rejected_request(status, code, rejected):
    script = _Script(httpx.Response(status, json={"code": code, "status": status}))

    run = _run([_case()], script)["case-1"]

    assert run.modal is not None
    assert (run.modal.status, run.modal.request_rejected) == (status, rejected)
    assert run.modal.body["code"] == code
    assert len(script.posted) == 1


def test_posts_in_flight_never_exceed_the_concurrency_bound():
    in_flight = 0
    peak = 0
    posts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak, posts
        in_flight += 1
        posts += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, json=_definite())

    cases = [_case(f"case-{i}") for i in range(5)]
    runs = _run(cases, handler, samples=3, concurrency=2)

    assert (peak, posts) == (2, 15)
    assert list(runs) == [case.id for case in cases]
    assert all(len(run.samples) == 3 and run.n_agreeing == 3 for run in runs.values())


def test_samples_below_one_are_refused():
    with pytest.raises(ValueError, match="samples"):
        _run([_case()], _Script(), samples=0)


# The modal vote


def test_extending_a_run_votes_again_over_every_sample():
    first = _run([_case()], _Script(httpx.Response(200, json=_definite())))["case-1"]
    nodata = {"kind": "resolution", "parts": [{"status": "no_data"}]}
    more = _run(
        [_case()],
        _Script(httpx.Response(200, json=nodata), httpx.Response(200, json=nodata)),
        samples=2,
    )["case-1"]

    run = extend_run(first, more)

    assert len(run.samples) == 3
    assert (run.modal, run.n_agreeing) == (run.samples[1], 2)


def test_the_majority_answer_wins_over_an_earlier_minority():
    samples = [_sample(_definite(spec_id=spec_id)) for spec_id in ("a", "b", "b")]
    assert modal_sample(samples) == (1, 2)


def test_a_three_way_tie_picks_the_first_sample():
    samples = [_sample(_definite(spec_id=s)) for s in ("a", "b", "c")]
    assert modal_sample(samples) == (0, 1)


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        (_definite(), _definite(entities=(("country/KEN", "destination"),)), False),
        (_definite(served_count=2), _definite(served_count=3), False),
        (_definite(member_digest="a"), _definite(member_digest="b"), False),
        (
            _definite(operations=[{"verb": "rank", "over": "entity", "phrase": "top 5"}]),
            _definite(
                operations=[
                    {
                        "verb": "rank",
                        "over": "entity",
                        "phrase": "top 5 donors",
                        "target_year": None,
                    }
                ]
            ),
            True,
        ),
        (
            _definite(operations=[{"verb": "rank", "over": "entity"}]),
            _definite(operations=[{"verb": "rank", "over": "time"}]),
            False,
        ),
        (
            {"kind": "declined", "reason": "out_of_scope"},
            {"kind": "declined", "reason": "requires_analysis"},
            False,
        ),
        (
            {"kind": "catalogue", "scope": {"place": {"dcid": "country/KEN"}}},
            {"kind": "catalogue", "scope": {"place": {"dcid": "country/KEN"}}},
            True,
        ),
        (_definite(twins=["X"]), _definite(twins=["Y"]), True),
    ],
    ids=[
        "entity-role",
        "served-count",
        "member-digest",
        "operation-phrase",
        "operation-axis",
        "declined-reason",
        "catalogue-place",
        "twins-outside-key",
    ],
)
def test_the_vote_groups_answers_by_what_a_client_reads(first, second, same):
    _, n_agreeing = modal_sample([_sample(first), _sample(second)])
    assert n_agreeing == (2 if same else 1)


def test_the_http_status_separates_otherwise_equal_bodies():
    problem = {"code": "facet_not_available"}
    assert modal_sample([_sample(problem, 422), _sample(problem, 400)]) == (0, 1)


def test_the_modal_twin_set_picks_the_representative_within_the_winning_group():
    samples = [
        _sample(_definite(twins=["X"])),
        _sample(_definite(twins=["Y"])),
        _sample(_definite(twins=["Y"])),
    ]
    assert modal_sample(samples) == (1, 3)


def test_the_vote_refuses_no_samples():
    with pytest.raises(ValueError, match="at least one"):
        modal_sample([])


# Schema


def test_fetch_schema_returns_the_app_document():
    document = {"schema_version": 1, "request": {}, "response": {"type": "object"}}

    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://app/api/search/schema"
        return httpx.Response(200, json=document)

    async def go() -> dict:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_schema("http://app/", client=client)

    assert asyncio.run(go()) == document


def test_fetch_schema_raises_when_the_app_fails():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async def go() -> dict:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_schema("http://app", client=client)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(go())


def test_the_runner_names_the_eval_and_sends_the_apis_token():
    """Request logs and analytics tell eval traffic apart by its User-Agent, and every
    client the runner owns shares the process's one ID token source."""

    async def identity() -> tuple[str, httpx.Auth]:
        async with runner._client_scope(None) as client:
            return client.headers["user-agent"], client.auth

    assert asyncio.run(identity()) == ("dfo-search-eval", apis_auth())


@pytest.mark.parametrize("status", [401, 403])
def test_a_service_refusing_the_token_fails_the_run_rather_than_counting_as_infra(status):
    """Cloud Run's refusal is an HTML page, which the runner would otherwise score as an
    infrastructure failure. It has to stop the run."""

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="<html>Forbidden</html>")

    async def go() -> None:
        auth = ApisAuth(static_token="token")
        transport = httpx.MockTransport(refuse)
        async with httpx.AsyncClient(transport=transport, auth=auth) as client:
            await run_cases(
                [_case()],
                url="https://apis.example",
                samples=1,
                concurrency=1,
                client=client,
                backoff_s=0,
            )

    with pytest.raises(ExceptionGroup) as raised:
        asyncio.run(go())
    assert raised.group_contains(ApisAuthError)


# Members door


def test_the_members_door_reader_posts_each_definition_once():
    """A Spec's served StatVars come from the app's members door, read once per spec id
    with the Spec's definition, as bare dcids."""
    posts = []

    def post(url, *, json, headers, timeout, auth):
        assert auth is apis_auth()
        posts.append((url, json))
        body = {"members": [{"ref": "dcid:Count_Person"}, {"ref": "dcid:Count_Person_Female"}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    read = members_door_reader("http://app/")
    definition = {"schema_version": 1, "spec_id": "spec_1", "identity": {}}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner.httpx, "post", post)
        first, again = read("spec_1", definition), read("spec_1", definition)

    assert first == again == ["Count_Person", "Count_Person_Female"]
    assert posts == [("http://app/api/datasets/spec_1/members", {"definition": definition})]


def test_a_refused_members_read_raises_for_the_check_to_score_as_infrastructure():
    def post(url, *, json, headers, timeout, auth):
        return httpx.Response(404, json={}, request=httpx.Request("POST", url))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner.httpx, "post", post)
        with pytest.raises(httpx.HTTPStatusError):
            members_door_reader("http://app")("spec_1", {})
