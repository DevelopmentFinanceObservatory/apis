"""Tests for the dev parser replay harness without any network or credentials."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import httpx
import pytest

import scripts.replay_parser_experiment as m
from search.engine import parser_replay


def _expected(case: dict[str, object]) -> dict[str, Any]:
    expected = case["expected"]
    assert isinstance(expected, dict)
    return m._json_object(expected) or {}


def test_synthetic_cases_are_labelled_unique_and_balanced():
    cases = m.synthetic_cases()

    assert len(cases) == 400
    assert len({case["id"] for case in cases}) == 400
    assert len({case["query"] for case in cases}) == 400
    categories = [_expected(case)["category"] for case in cases]
    assert {category: categories.count(category) for category in set(categories)} == {
        "place_measure": 50,
        "donor_recipient": 50,
        "two_places": 50,
        "group_scope": 50,
        "date_range": 50,
        "forecast": 50,
        "multilingual": 50,
        "no_place": 50,
    }
    donor_case = next(case for case in cases if _expected(case)["category"] == "donor_recipient")
    roles = _expected(donor_case)["roles"]
    assert all(name in str(donor_case["query"]) for name in roles)


def test_load_cases_accepts_screened_json_array(tmp_path):
    source = tmp_path / "cases.json"
    source.write_text(json.dumps([{"id": 1, "query": "GDP in Kenya"}]))

    assert m._load_cases(source) == [m.Case(case_id="1", query="GDP in Kenya")]


def test_run_arm_records_only_the_allowed_interpretation_fields():
    body = (
        'event: interpretation\ndata: {"kind":"interpretation","variables":["GDP"],'
        '"entities":["Kenya"],"dates":[],"expected_results":1,"query_echo":"secret"}\n\n'
        'event: response\ndata: {"kind":"response","response":{"status":"ok",'
        '"parts":[{"spec":{"spec_id":"spec-1"}}]}}\n\n'
        'event: done\ndata: {"kind":"done","elapsed_ms":7,"trace_id":"trace"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers[m.PARSER_ARM_HEADER] == "ministral"
        assert request.headers[m.PARSER_CACHE_HEADER] == "bypass"
        assert request.headers[m.PARSER_REPLAY_HEADER].startswith("run-1-")
        return httpx.Response(200, headers={m.PARSER_ARM_HEADER: "ministral"}, text=body)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await m._run_arm(
                client,
                url="http://localhost:8080",
                case=m.Case(case_id="case-1", query="private input"),
                arm="ministral",
                order=2,
                replay_id="run-1",
                replay_key="key",
                stream_probe=False,
            )

    result = asyncio.run(run())

    assert result.error is None
    assert result.interpretation == {
        "variables": ["GDP"],
        "entities": ["Kenya"],
        "dates": [],
        "expected_results": 1,
    }
    assert result.response_status == "ok"
    assert result.response_ms is not None
    assert result.spec_ids == ["spec-1"]
    assert "private input" not in json.dumps(result.__dict__)
    assert result.request_id.startswith("run-1-")


def test_run_arm_rejects_a_mislabelled_server_response():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, headers={m.PARSER_ARM_HEADER: "flash-lite"})
            )
        ) as client:
            return await m._run_arm(
                client,
                url="http://localhost:8080",
                case=m.Case(case_id="case-1", query="private input"),
                arm="ministral",
                order=1,
                replay_id="run-1",
                replay_key="key",
                stream_probe=False,
            )

    result = asyncio.run(run())

    assert result.error is not None
    assert result.status_code == 200
    assert result.assigned_arm == "flash-lite"


def test_run_arm_records_a_gate_refusal_status_and_problem_code():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(403, json={"code": "parser_replay_refused"})
            )
        ) as client:
            return await m._run_arm(
                client,
                url="http://localhost:8080",
                case=m.Case(case_id="case-1", query="private input"),
                arm="ministral",
                order=1,
                replay_id="run-1",
                replay_key="key",
                stream_probe=False,
            )

    result = asyncio.run(run())

    assert result.error is not None
    assert result.status_code == 403
    assert result.problem_code == "parser_replay_refused"
    assert result.assigned_arm is None


def test_run_arm_treats_sse_error_and_missing_done_as_failures():
    async def run(body: str):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, headers={m.PARSER_ARM_HEADER: "ministral"}, text=body)
            )
        ) as client:
            return await m._run_arm(
                client,
                url="http://localhost:8080",
                case=m.Case(case_id="case-1", query="private input"),
                arm="ministral",
                order=1,
                replay_id="run-1",
                replay_key="key",
                stream_probe=False,
            )

    error_event = asyncio.run(run('event: error\ndata: {"kind":"error"}\n\n'))
    no_events = asyncio.run(run(""))

    assert error_event.error is not None
    assert error_event.outcome == "error"
    assert no_events.error is not None
    assert no_events.outcome is None


def _result(case_id: str, arm: m.ARM, order: int, replay_id: str) -> m.Result:
    return m.Result(
        case_id=case_id,
        replay_id=replay_id,
        request_id=m._request_id(replay_id, case_id, arm),
        arm=arm,
        order=order,
        status_code=200,
        problem_code=None,
        assigned_arm=arm,
        first_byte_ms=1,
        interpretation_ms=1,
        response_ms=1,
        done_ms=1,
        wall_ms=1,
        event_kinds=["done"],
        outcome="done",
        response_status="ok",
        interpretation=None,
        spec_ids=[],
        error=None,
    )


def _replay(cases, *, concurrency, **kwargs):
    return asyncio.run(
        m.replay(
            [m.Case(case_id=case, query=case) for case in cases],
            url="http://localhost:8080",
            replay_key="key",
            replay_id="run-1",
            concurrency=concurrency,
            pace_ms=0,
            stream_probe=False,
            **kwargs,
        )
    )


@pytest.mark.parametrize(("concurrency", "overlap"), [(1, False), (2, True)])
def test_replay_pairs_hold_one_slot_for_both_arms(monkeypatch, concurrency, overlap):
    events: list[tuple[str, str]] = []

    async def fake_run_arm(_client, *, case, arm, order, replay_id, **_kwargs):
        events.append(("started", case.case_id))
        await asyncio.sleep(0)
        events.append(("finished", case.case_id))
        return _result(case.case_id, arm, order, replay_id)

    monkeypatch.setattr(m, "_run_arm", fake_run_arm)
    monkeypatch.setattr(m, "apis_auth", lambda: None)

    results = _replay(["one", "two", "three"], concurrency=concurrency)

    assert len(results) == 3 * len(m.ARMS)
    # Another case starting between a pair's first start and last finish means overlap.
    interleaved = False
    for case_id in ("one", "two", "three"):
        indexes = [i for i, (_, c) in enumerate(events) if c == case_id]
        window = events[indexes[0] : indexes[-1] + 1]
        interleaved |= any(c != case_id for _, c in window)
    assert interleaved is overlap


def test_replay_runs_each_arm_once_per_case_in_an_order_that_varies(monkeypatch):
    started: list[tuple[str, str]] = []

    async def fake_run_arm(_client, *, case, arm, order, replay_id, **_kwargs):
        started.append((case.case_id, arm))
        return _result(case.case_id, arm, order, replay_id)

    monkeypatch.setattr(m, "_run_arm", fake_run_arm)
    monkeypatch.setattr(m, "apis_auth", lambda: None)
    case_ids = [f"case-{n}" for n in range(12)]

    results = _replay(case_ids, concurrency=1)

    orders = {c: tuple(a for cc, a in started if cc == c) for c in case_ids}
    assert all(sorted(order) == sorted(m.ARMS) for order in orders.values())
    assert len(results) == 3 * len(case_ids)
    assert len(set(orders.values())) > 1
    assert len({order[0] for order in orders.values()}) > 1
    assert orders == {
        c: tuple(r.arm for r in sorted(results, key=lambda r: r.order) if r.case_id == c)
        for c in case_ids
    }


def test_replay_stops_sending_after_consecutive_transport_errors(monkeypatch):
    calls: list[str] = []
    streamed: list[m.Result] = []

    async def fake_run_arm(_client, *, case, arm, order, replay_id, **_kwargs):
        calls.append(case.case_id)
        result = _result(case.case_id, arm, order, replay_id)
        if case.case_id == "ok":
            return result
        return replace(result, status_code=None, error="ConnectError: refused")

    monkeypatch.setattr(m, "_run_arm", fake_run_arm)
    monkeypatch.setattr(m, "apis_auth", lambda: None)

    with pytest.raises(m.ReplayStopped) as stopped:
        _replay(["ok", "a", "b", "c", "d"], concurrency=1, on_result=streamed.append)

    assert len(calls) == 3 + m.TRANSPORT_FAILURE_LIMIT
    assert len(stopped.value.results) == len(calls)
    assert streamed == stopped.value.results
    assert stopped.value.cause == "ConnectError: refused"


def test_an_http_status_failure_resets_the_transport_error_count(monkeypatch):
    async def fake_run_arm(_client, *, case, arm, order, replay_id, **_kwargs):
        result = _result(case.case_id, arm, order, replay_id)
        if order == 2:
            return replace(result, error="HTTP 503")
        return replace(result, status_code=None, error="ConnectError: refused")

    monkeypatch.setattr(m, "_run_arm", fake_run_arm)
    monkeypatch.setattr(m, "apis_auth", lambda: None)

    assert len(_replay(["a", "b", "c", "d"], concurrency=1)) == 12


def test_main_exits_non_zero_when_the_run_stops(monkeypatch, tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(json.dumps({"id": f"c{n}", "query": "q"}) + "\n" for n in range(4)))
    output = tmp_path / "out.jsonl"

    async def fake_run_arm(_client, *, case, arm, order, replay_id, **_kwargs):
        result = _result(case.case_id, arm, order, replay_id)
        return replace(result, status_code=None, error="ConnectError: refused")

    monkeypatch.setattr(m, "_run_arm", fake_run_arm)
    monkeypatch.setattr(m, "apis_auth", lambda: None)
    monkeypatch.setenv("SEARCH_PARSER_REPLAY_KEY", "key")

    code = m.main(
        ["--url", "http://127.0.0.1:8080", "--cases", str(cases), "--output", str(output)]
    )

    assert code == 1
    assert len(output.read_text().splitlines()) == m.TRANSPORT_FAILURE_LIMIT


def test_replay_records_a_pair_exception_and_keeps_the_other_pairs(monkeypatch):
    streamed: list[m.Result] = []

    async def fake_run_arm(_client, *, case, arm, order, replay_id, **_kwargs):
        if case.case_id == "bad":
            raise ValueError("unexpected")
        return _result(case.case_id, arm, order, replay_id)

    monkeypatch.setattr(m, "_run_arm", fake_run_arm)
    monkeypatch.setattr(m, "apis_auth", lambda: None)

    results = _replay(["bad", "good"], concurrency=2, on_result=streamed.append)

    by_case = {case: [r for r in results if r.case_id == case] for case in ("bad", "good")}
    assert all(r.error is not None for r in by_case["bad"])
    assert all(r.error is None for r in by_case["good"])
    assert len(streamed) == 2 * len(m.ARMS)


def test_main_writes_a_private_output_file(monkeypatch, tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "one", "query": "private"}) + "\n")
    output = tmp_path / "out.jsonl"

    async def fake_run_arm(_client, *, case, arm, order, replay_id, **_kwargs):
        return _result(case.case_id, arm, order, replay_id)

    monkeypatch.setattr(m, "_run_arm", fake_run_arm)
    monkeypatch.setattr(m, "apis_auth", lambda: None)
    monkeypatch.setenv("SEARCH_PARSER_REPLAY_KEY", "key")

    code = m.main(
        ["--url", "http://127.0.0.1:8080", "--cases", str(cases), "--output", str(output)]
    )

    assert code == 0
    assert output.stat().st_mode & 0o777 == 0o600
    assert [json.loads(line)["case_id"] for line in output.read_text().splitlines()] == [
        "one"
    ] * len(m.ARMS)


def test_main_refuses_an_existing_output_file(monkeypatch, tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "one", "query": "private"}) + "\n")
    output = tmp_path / "out.jsonl"
    output.write_text("earlier result\n")
    monkeypatch.setenv("SEARCH_PARSER_REPLAY_KEY", "key")

    with pytest.raises(SystemExit, match="exists"):
        m.main(["--url", "http://127.0.0.1:8080", "--cases", str(cases), "--output", str(output)])

    assert output.read_text() == "earlier result\n"


@pytest.mark.parametrize(
    "name",
    [
        "PARSER_REPLAY_HEADER",
        "PARSER_ARM_HEADER",
        "PARSER_KEY_HEADER",
        "PARSER_CACHE_HEADER",
        "PARSER_STREAM_HEADER",
    ],
)
def test_the_runner_sends_the_headers_the_engine_reads(name):
    assert getattr(m, name) == getattr(parser_replay, name)


@pytest.mark.parametrize(
    ("url", "valid"),
    [
        ("http://localhost:8080", True),
        ("http://127.0.0.1:8080", True),
        ("https://localhost:8080", False),
        ("http://localhost:8080/unexpected", False),
        ("https://parser-replay---dfo-dev-apis-zms5lj4zhq-nn.a.run.app", False),
        ("https://dfo-prod-apis.run.app", False),
        ("https://example.com", False),
    ],
)
def test_validate_url_limits_the_runner_to_http_loopback(url, valid):
    if valid:
        assert m._validate_url(url) == url
    else:
        with pytest.raises(ValueError):
            m._validate_url(url)
