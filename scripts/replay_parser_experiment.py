#!/usr/bin/env python3
"""Replay labelled parser cases against a tagged DFO dev revision.

The runner sends each input once to each parser arm, back to back, through a loopback
``gcloud run services proxy``, which keeps the Cloud Run credentials fresh. The replay key
enables the dev-only parser switch. The result omits the original query field and retains a
structured interpretation for local scoring. Treat both input and result as private data.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TextIO
from urllib.parse import urlparse

import httpx

from search.engine.parser_replay import (
    PARSER_ARM_HEADER,
    PARSER_CACHE_HEADER,
    PARSER_KEY_HEADER,
    PARSER_REPLAY_HEADER,
    PARSER_STREAM_HEADER,
)
from search.eval.apis_auth import apis_auth

ARM = Literal["flash-lite", "ministral"]
ARMS: tuple[ARM, ARM] = ("flash-lite", "ministral")
_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
_SYNTHETIC_COUNTRIES = (
    "Kenya",
    "Ethiopia",
    "Ghana",
    "Uganda",
    "Nepal",
    "Haiti",
    "Chad",
    "Rwanda",
    "Peru",
    "Viet Nam",
)
_SYNTHETIC_DONORS = ("United States", "Germany", "Japan", "France", "Canada")


@dataclass(frozen=True)
class Case:
    """One replay input. ``expected`` remains in the source data for later blind review."""

    case_id: str
    query: str
    expected: dict[str, Any] | None = None


@dataclass(frozen=True)
class Result:
    """One arm's outcome, timings, and private interpretation summary."""

    case_id: str
    replay_id: str
    request_id: str
    arm: ARM
    order: int
    status_code: int | None
    problem_code: str | None
    assigned_arm: str | None
    first_byte_ms: float | None
    interpretation_ms: float | None
    response_ms: float | None
    done_ms: float | None
    wall_ms: float
    event_kinds: list[str]
    outcome: str | None
    response_status: str | None
    interpretation: dict[str, object] | None
    spec_ids: list[str]
    error: str | None


def _validate_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("URL must be a bare replay origin")
    if parsed.scheme == "http" and parsed.hostname in _LOOPBACK:
        return value.rstrip("/")
    raise ValueError("URL must be an http loopback origin")


def _json_object(value: object) -> dict[str, object] | None:
    """Normalise an untyped JSON value to a string-keyed object."""
    if not isinstance(value, dict):
        return None
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        result[key] = item
    return result


def _load_cases(path: Path) -> list[Case]:
    content = path.read_text()
    try:
        decoded = json.loads(content)
    except ValueError:
        decoded = None
    if isinstance(decoded, list):
        rows = enumerate(decoded, start=1)
    else:
        rows = (
            (line_no, json.loads(line))
            for line_no, line in enumerate(content.splitlines(), start=1)
            if line.strip()
        )
    cases: list[Case] = []
    seen: set[str] = set()
    for line_no, item in rows:
        row = _json_object(item)
        if row is None:
            raise ValueError(f"{path}:{line_no}: expected a JSON object")
        try:
            case_id = str(row["id"])
            query = str(row["query"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_no}: expected JSON object with id and query") from exc
        if not case_id or not query or case_id in seen:
            raise ValueError(f"{path}:{line_no}: ids and queries must be non-empty and ids unique")
        seen.add(case_id)
        expected = _json_object(row.get("expected"))
        cases.append(
            Case(
                case_id=case_id,
                query=query,
                expected=expected,
            )
        )
    if not cases:
        raise ValueError(f"{path}: no cases")
    return cases


def _interpretation_summary(payload: dict[str, Any]) -> dict[str, object]:
    """Keep only structured interpretation data suitable for blinded local scoring."""
    dates = payload.get("dates")
    return {
        "variables": payload.get("variables") if isinstance(payload.get("variables"), list) else [],
        "entities": payload.get("entities") if isinstance(payload.get("entities"), list) else [],
        "dates": dates if isinstance(dates, list) else [],
        "expected_results": payload.get("expected_results"),
    }


def _spec_ids(payload: dict[str, Any]) -> list[str]:
    """Extract only identifiers, never response text, from an SSE payload."""
    found: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            spec_id = value.get("spec_id")
            if isinstance(spec_id, str):
                found.add(spec_id)
            spec_ids = value.get("spec_ids")
            if isinstance(spec_ids, list):
                found.update(item for item in spec_ids if isinstance(item, str))
            for key in ("response", "parts", "spec", "definition", "candidates", "specs"):
                children = value.get(key)
                if isinstance(children, dict):
                    visit(children)
                elif isinstance(children, list):
                    for child in children:
                        visit(child)

    visit(payload)
    return sorted(found)


def _problem_code(body: bytes) -> str | None:
    """Read the machine-readable ``code`` of a problem response body."""
    try:
        decoded = _json_object(json.loads(body))
    except ValueError:
        return None
    code = decoded.get("code") if decoded is not None else None
    return code if isinstance(code, str) else None


def _error_result(
    case: Case, *, arm: ARM, order: int, replay_id: str, exc: BaseException
) -> Result:
    """Record an unexpected failure as that arm's error."""
    return Result(
        case_id=case.case_id,
        replay_id=replay_id,
        request_id=_request_id(replay_id, case.case_id, arm),
        arm=arm,
        order=order,
        status_code=None,
        problem_code=None,
        assigned_arm=None,
        first_byte_ms=None,
        interpretation_ms=None,
        response_ms=None,
        done_ms=None,
        wall_ms=0.0,
        event_kinds=[],
        outcome=None,
        response_status=None,
        interpretation=None,
        spec_ids=[],
        error=f"{type(exc).__name__}: {exc}",
    )


def _request_id(replay_id: str, case_id: str, arm: ARM) -> str:
    """Make a Cloud Logging join key within the replay header's 80-character bound."""
    digest = hashlib.sha256(f"{replay_id}:{case_id}:{arm}".encode()).hexdigest()[:20]
    return f"{replay_id[:58]}-{digest}"


def synthetic_cases(*, count: int = 400) -> list[dict[str, object]]:
    """Generate 8 balanced, labelled classes of 50 cases each by default."""
    if count % 8:
        raise ValueError("synthetic count must divide into eight categories")
    categories = (
        "place_measure",
        "donor_recipient",
        "two_places",
        "group_scope",
        "date_range",
        "forecast",
        "multilingual",
        "no_place",
    )
    measures = (
        "official development assistance",
        "GDP per capita",
        "maternal mortality",
        "debt service",
        "HIV incidence",
    )
    rows: list[dict[str, object]] = []
    for category in categories:
        for index in range(count // len(categories)):
            country = _SYNTHETIC_COUNTRIES[index % len(_SYNTHETIC_COUNTRIES)]
            other = _SYNTHETIC_COUNTRIES[(index + 3) % len(_SYNTHETIC_COUNTRIES)]
            donor = _SYNTHETIC_DONORS[index % len(_SYNTHETIC_DONORS)]
            measure = measures[index % len(measures)]
            year = 1970 + index
            match category:
                case "place_measure":
                    query = f"{measure} in {country} in {year}"
                    expected = {"entities": [country], "dates": [str(year)]}
                case "donor_recipient":
                    query = f"{measure} from {donor} to {country} in {year}"
                    expected = {
                        "entities": [donor, country],
                        "roles": {donor: "source", country: "recipient"},
                        "dates": [str(year)],
                    }
                case "two_places":
                    query = f"compare {measure} in {country} and {other} in {year}"
                    expected = {"entities": [country, other], "dates": [str(year)]}
                case "group_scope":
                    query = f"{measure} in African countries in {year}"
                    expected = {
                        "entities": ["Africa"],
                        "members": ["Africa"],
                        "dates": [str(year)],
                    }
                case "date_range":
                    query = f"{measure} in {country} from {year} to {year + 1}"
                    expected = {"entities": [country], "dates": [str(year), str(year + 1)]}
                case "forecast":
                    forecast_year = year + 60
                    query = f"forecast {measure} in {country} for {forecast_year}"
                    expected = {"entities": [country], "dates": [str(forecast_year)]}
                case "multilingual":
                    query = f"{measure} en {country} en {year}"
                    expected = {"entities": [country], "dates": [str(year)]}
                case "no_place":
                    query = f"global {measure} since {year}"
                    expected = {"entities": [], "dates": [str(year)]}
                case _:
                    raise AssertionError(f"unhandled synthetic category: {category}")
            rows.append(
                {
                    "id": f"synthetic-{category}-{index + 1:02d}",
                    "query": query,
                    "expected": {"category": category, **expected},
                }
            )
    return rows


def _sse_events(buffer: str) -> Iterable[tuple[str, dict[str, Any]]]:
    for frame in buffer.split("\n\n"):
        event = None
        data = None
        for line in frame.splitlines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = line.removeprefix("data:").strip()
        if event and data:
            try:
                value = json.loads(data)
            except ValueError:
                continue
            if isinstance(value, dict):
                yield event, value


async def _run_arm(
    client: httpx.AsyncClient,
    *,
    url: str,
    case: Case,
    arm: ARM,
    order: int,
    replay_id: str,
    replay_key: str,
    stream_probe: bool,
) -> Result:
    request_id = _request_id(replay_id, case.case_id, arm)
    headers = {
        "Accept": "text/event-stream",
        PARSER_REPLAY_HEADER: request_id,
        PARSER_ARM_HEADER: arm,
        PARSER_KEY_HEADER: replay_key,
        PARSER_CACHE_HEADER: "bypass",
    }
    if stream_probe:
        headers[PARSER_STREAM_HEADER] = "1"
    started = time.perf_counter()
    first_byte_ms = interpretation_ms = response_ms = done_ms = None
    events: list[str] = []
    outcome = error = None
    response_status = None
    interpretation = None
    spec_ids: set[str] = set()
    assigned_arm = None
    status_code = None
    problem_code = None
    pending = ""
    terminal_event = None
    try:
        async with client.stream(
            "POST",
            f"{url}/api/search/resolve",
            headers=headers,
            json={"input": {"kind": "raw_text", "query": case.query}},
        ) as response:
            status_code = response.status_code
            assigned_arm = response.headers.get(PARSER_ARM_HEADER)
            if status_code != 200:
                problem_code = _problem_code(await response.aread())
                raise RuntimeError(f"HTTP {status_code} problem code {problem_code}")
            if assigned_arm != arm:
                raise RuntimeError(f"server assigned {assigned_arm!r}, expected {arm!r}")
            async for chunk in response.aiter_text():
                now_ms = (time.perf_counter() - started) * 1000
                if first_byte_ms is None:
                    first_byte_ms = now_ms
                pending += chunk
                frames, _, pending = pending.rpartition("\n\n")
                for event, payload in _sse_events(frames):
                    events.append(event)
                    spec_ids.update(_spec_ids(payload))
                    if event == "interpretation" and interpretation_ms is None:
                        interpretation_ms = now_ms
                        interpretation = _interpretation_summary(payload)
                    if event == "response":
                        response_ms = now_ms
                        reply = _json_object(payload.get("response"))
                        status = reply.get("status") if reply is not None else None
                        response_status = status if isinstance(status, str) else None
                    if event in {"done", "error"}:
                        terminal_event = event
                        done_ms = now_ms
                        outcome = str(payload.get("status") or payload.get("kind") or event)
            if terminal_event != "done":
                error = (
                    "SSE error event"
                    if terminal_event == "error"
                    else "SSE stream ended without a terminal done event"
                )
    except (httpx.HTTPError, RuntimeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return Result(
        case_id=case.case_id,
        replay_id=replay_id,
        request_id=request_id,
        arm=arm,
        order=order,
        status_code=status_code,
        problem_code=problem_code,
        assigned_arm=assigned_arm,
        first_byte_ms=first_byte_ms,
        interpretation_ms=interpretation_ms,
        response_ms=response_ms,
        done_ms=done_ms,
        wall_ms=round((time.perf_counter() - started) * 1000, 3),
        event_kinds=events,
        outcome=outcome,
        response_status=response_status,
        interpretation=interpretation,
        spec_ids=sorted(spec_ids),
        error=error,
    )


async def replay(
    cases: Sequence[Case],
    *,
    url: str,
    replay_key: str,
    replay_id: str,
    concurrency: int,
    pace_ms: int,
    stream_probe: bool,
    on_result: Callable[[Result], None] = lambda _result: None,
) -> list[Result]:
    """Run each case's two arms back to back; ``concurrency`` bounds pairs in flight.

    ``on_result`` receives each result as its pair completes.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def run_pair(case: Case) -> list[Result]:
        arms = (
            ARMS
            if hashlib.sha256(f"{replay_id}:{case.case_id}".encode()).digest()[0] % 2 == 0
            else ARMS[::-1]
        )
        async with semaphore:
            if pace_ms:
                await asyncio.sleep(pace_ms / 1000)
            results: list[Result] = []
            for order, arm in enumerate(arms, start=1):
                try:
                    result = await _run_arm(
                        client,
                        url=url,
                        case=case,
                        arm=arm,
                        order=order,
                        replay_id=replay_id,
                        replay_key=replay_key,
                        stream_probe=stream_probe,
                    )
                except Exception as exc:
                    result = _error_result(case, arm=arm, order=order, replay_id=replay_id, exc=exc)
                results.append(result)
        for result in results:
            on_result(result)
        return results

    async with httpx.AsyncClient(timeout=90, follow_redirects=False, auth=apis_auth()) as client:
        pairs = await asyncio.gather(*(run_pair(case) for case in cases))
    return [result for pair in pairs for result in pair]


def _write_private_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as output:
            temporary = Path(output.name)
            os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
            output.writelines(lines)
        temporary.replace(path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _open_private_output(path: Path) -> TextIO:
    """Open ``path`` for writing with mode 0600, line-buffered so each result lands."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Loopback http URL of a gcloud run services proxy")
    parser.add_argument("--cases", type=Path, help="Private JSONL with id and query fields")
    parser.add_argument("--output", type=Path, help="Private JSONL result path")
    parser.add_argument("--replay-id", default=f"parser-{secrets.token_hex(8)}")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--pace-ms", type=int, default=0)
    parser.add_argument("--stream-probe", action="store_true")
    parser.add_argument(
        "--write-synthetic", type=Path, help="Write 400 labelled synthetic cases and exit"
    )
    parser.add_argument("--synthetic-count", type=int, default=400)
    options = parser.parse_args(argv)
    if options.write_synthetic is None and (
        not options.url or not options.cases or not options.output
    ):
        parser.error("--url, --cases, and --output are required unless --write-synthetic is used")
    if (
        not 1 <= options.concurrency <= 20
        or options.pace_ms < 0
        or not _RUN_ID.fullmatch(options.replay_id)
    ):
        parser.error("invalid concurrency, pace, or replay id")
    return options


def main(argv: Sequence[str] | None = None) -> int:
    options = _parse_args(argv)
    if options.write_synthetic is not None:
        rows = synthetic_cases(count=options.synthetic_count)
        _write_private_lines(
            options.write_synthetic, (json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
        print(f"wrote {len(rows)} labelled synthetic cases to {options.write_synthetic}")
        return 0
    replay_key = os.environ.get("SEARCH_PARSER_REPLAY_KEY")
    if not replay_key:
        raise SystemExit("SEARCH_PARSER_REPLAY_KEY is required")
    url = _validate_url(options.url)
    if options.cases.resolve() == options.output.resolve():
        raise SystemExit("--output must not replace the private --cases input file")
    cases = _load_cases(options.cases)
    with _open_private_output(options.output) as output:

        def write_result(result: Result) -> None:
            output.write(json.dumps(asdict(result), sort_keys=True) + "\n")

        results = asyncio.run(
            replay(
                cases,
                url=url,
                replay_key=replay_key,
                replay_id=options.replay_id,
                concurrency=options.concurrency,
                pace_ms=options.pace_ms,
                stream_probe=options.stream_probe,
                on_result=write_result,
            )
        )
    failures = sum(result.error is not None for result in results)
    print(f"wrote {len(results)} paired arm results with {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
