"""Tests for the dev parser replay harness without any network or credentials."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import scripts.replay_parser_experiment as m


def test_synthetic_cases_are_labelled_unique_and_balanced():
    cases = m.synthetic_cases()

    assert len(cases) == 400
    assert len({case["id"] for case in cases}) == 400
    assert len({case["query"] for case in cases}) == 400
    categories = [case["expected"]["category"] for case in cases]
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
    donor_case = next(case for case in cases if case["expected"]["category"] == "donor_recipient")
    roles = donor_case["expected"]["roles"]
    assert all(name in donor_case["query"] for name in roles)


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
        assert request.headers[m.ARM_HEADER] == "ministral"
        assert request.headers[m.CACHE_HEADER] == "bypass"
        assert request.headers[m.REPLAY_HEADER].startswith("run-1-")
        return httpx.Response(200, headers={m.ARM_HEADER: "ministral"}, text=body)

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
    assert result.spec_ids == ["spec-1"]
    assert "private input" not in json.dumps(result.__dict__)
    assert result.request_id.startswith("run-1-")


def test_run_arm_rejects_a_mislabelled_server_response():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, headers={m.ARM_HEADER: "flash-lite"})
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
    assert result.assigned_arm == "flash-lite"


def test_run_arm_treats_sse_error_and_missing_done_as_failures():
    async def run(body: str):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, headers={m.ARM_HEADER: "ministral"}, text=body)
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

    assert "SSE error" in asyncio.run(run('event: error\ndata: {"kind":"error"}\n\n')).error
    assert "without a terminal" in asyncio.run(run("")).error


def test_replay_runs_each_pair_in_order_while_other_pairs_can_progress(monkeypatch):
    events: list[tuple[str, str, str]] = []

    async def fake_run_arm(_client, *, case, arm, order, replay_id, **_kwargs):
        events.append(("started", case.case_id, arm))
        await asyncio.sleep(0)
        events.append(("finished", case.case_id, arm))
        return m.Result(
            case_id=case.case_id,
            replay_id=replay_id,
            request_id=m._request_id(replay_id, case.case_id, arm),
            arm=arm,
            order=order,
            status_code=200,
            assigned_arm=arm,
            first_byte_ms=1,
            interpretation_ms=1,
            done_ms=1,
            wall_ms=1,
            event_kinds=["done"],
            outcome="done",
            response_status="ok",
            interpretation=None,
            spec_ids=[],
            error=None,
        )

    monkeypatch.setattr(m, "_run_arm", fake_run_arm)
    monkeypatch.setattr(m, "apis_auth", lambda: None)

    results = asyncio.run(
        m.replay(
            [m.Case(case_id="one", query="one"), m.Case(case_id="two", query="two")],
            url="http://localhost:8080",
            replay_key="key",
            replay_id="run-1",
            concurrency=2,
            pace_ms=0,
            stream_probe=False,
        )
    )

    assert len(results) == 4
    for case_id in ("one", "two"):
        starts = [index for index, value in enumerate(events) if value[:2] == ("started", case_id)]
        finishes = [
            index for index, value in enumerate(events) if value[:2] == ("finished", case_id)
        ]
        assert finishes[0] < starts[1]


@pytest.mark.parametrize(
    ("url", "valid"),
    [
        ("https://parser-replay---dfo-dev-apis-zms5lj4zhq-nn.a.run.app", True),
        ("http://localhost:8080", True),
        ("https://dfo-dev-apis-abc.run.app", False),
        ("https://parser-replay---dfo-dev-apis-zms5lj4zhq-nn.a.run.app:444", False),
        ("http://localhost:8080/unexpected", False),
        ("https://dfo-dev-apis-evil.example.com", False),
        ("https://dfo-prod-apis.run.app", False),
        ("https://example.com", False),
    ],
)
def test_validate_url_limits_the_runner_to_dev_or_loopback(url, valid):
    if valid:
        assert m._validate_url(url) == url
    else:
        with pytest.raises(ValueError):
            m._validate_url(url)
