"""Run eval cases against a Search app over HTTP.

``run_cases`` posts each case's request to ``<url>/api/search/resolve`` once per sample,
retries infrastructure failures, app errors and degraded answers, and picks the modal
sample by a vote over the raw response bodies. The app's own request timeout, a 504 with
problem code ``timeout``, is an infrastructure failure that is not retried: a retry
waits out the same deadline. ``extend_run`` adds samples to a run and
votes again. ``fetch_schema`` reads the app's ``GET /api/search/schema``.
The runner reads responses as plain dicts, so it can score an app built from another
revision of the response models.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from collections.abc import AsyncIterator, Hashable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from search.eval.cases import Case
from search.eval.dataset import resubmit_identity
from search.models import ResolveRequest

logger = logging.getLogger(__name__)

RESOLVE_PATH = "/api/search/resolve"
SCHEMA_PATH = "/api/search/schema"
DEGRADED_LEG_HEADER = "X-Search-Degraded-Leg"
# Every request the eval sends names it, so request logs and analytics can leave it out.
EVAL_HEADERS = {"User-Agent": "dfo-search-eval"}

# Retries one sample gets after an infrastructure failure or a degraded answer.
_RETRIES = 2

# Statuses a retry can change and the app does not own: the app maps its upstream failures
# to 502 and 503, and 504 is a gateway timeout.
_INFRA_STATUSES = frozenset({502, 503, 504})

# The app answers its own request deadline with this status and problem code. A retry waits
# out the same deadline, so the runner does not retry it.
_APP_TIMEOUT_STATUS = 504
_APP_TIMEOUT_CODE = "timeout"

# The app's own unhandled error. It is retried, and a 500 that survives the retries is
# scored as the answer. 501 refuses an unimplemented door or capability every time, so it
# is scored like any other refusal.
_APP_ERROR_STATUS = 500

# Problem codes the app returns when its own request model refuses the posted request.
_REQUEST_REJECTED_CODES = frozenset({"request_invalid", "unsupported_schema_version"})

# Above the engine's 45-second request deadline, so the app answers before the client gives up.
_DEFAULT_TIMEOUT_S = 90.0


@dataclass(frozen=True, slots=True)
class Sample:
    """One scored answer to a case's request.

    ``body`` is the JSON body as the app returned it: a response for 2xx, a problem document
    otherwise, including a 500 that survived its retries. ``degraded`` marks an answer that
    stayed degraded after every retry. ``request_rejected`` marks a problem document refusing the request's shape.
    ``attempts`` counts posts for this sample, retries included; ``elapsed_s`` times the
    last one.
    """

    body: dict
    status: int
    degraded: bool
    request_rejected: bool
    attempts: int
    elapsed_s: float


@dataclass(frozen=True, slots=True)
class CaseRun:
    """Every sample of one case, plus the modal one.

    ``modal`` is the sample the vote picked and ``n_agreeing`` the size of its answer
    group. ``infra_error`` is set, and ``modal`` is None, when a sample failed on
    infrastructure after its retries. ``samples`` then holds only the samples that
    completed.
    """

    case_id: str
    request: dict
    samples: tuple[Sample, ...]
    modal: Sample | None
    n_agreeing: int
    infra_error: str | None


class _InfraFailure(Exception):
    """A post that is not a scorable answer: transport error, timeout, 502, 503 or 504, or
    an unreadable body.

    ``retryable`` is False for the app's own request timeout, which a retry would only
    repeat, and True for every other failure.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


def build_request(case: Case) -> dict:
    """Build the minimal resolve request body for a case.

    The body holds only the fields the case sets, so an app built from an older revision
    of the request models accepts it. Every request asks for skills, and a case that opts
    into observations asks for them. Raises ``ValueError`` when a resubmit case names no
    entity or StatVar to anchor an identity.
    """
    match case.door:
        case "raw_text":
            request_input: dict = {"kind": "raw_text", "query": case.query}
        case "parsed":
            assert case.parsed is not None
            request_input = {
                "kind": "parsed",
                "parts": [part.model_dump(mode="json", exclude_unset=True) for part in case.parsed],
                "raw_query": case.query,
            }
        case "dataset_resubmit":
            assert case.resubmit is not None
            base_spec_id, identity = resubmit_identity(case.resubmit)
            identity_dump = identity.model_dump(mode="json", exclude_unset=True)
            entry: dict = {
                "entry_kind": "identity",
                "variable_text": case.resubmit.variable_text,
                "identity": identity_dump,
                "base_identity": identity_dump,
            }
            if base_spec_id is not None:
                entry["base_spec_id"] = base_spec_id
            if case.resubmit.stat_var_dcids is not None:
                entry["resolved_stat_var_dcids"] = case.resubmit.stat_var_dcids
            request_input = {"kind": "dataset_resubmit", "entries": [entry]}
    options: dict = {"include_skills": True}
    if case.include_observations:
        options["include_observations"] = True
    request = ResolveRequest.model_validate({"input": request_input, "response_options": options})
    return request.model_dump(mode="json", exclude_unset=True)


async def fetch_schema(url: str, *, client: httpx.AsyncClient | None = None) -> dict:
    """Return the app's schema document from ``GET <url>/api/search/schema``.

    Raises ``httpx.HTTPError`` when the app does not answer with 2xx.
    """
    async with _client_scope(client) as http:
        response = await http.get(f"{url.rstrip('/')}{SCHEMA_PATH}")
        response.raise_for_status()
        return response.json()


async def run_cases(
    cases: Sequence[Case],
    *,
    url: str,
    samples: int,
    concurrency: int,
    client: httpx.AsyncClient | None = None,
    backoff_s: float = 2.0,
) -> dict[str, CaseRun]:
    """Post every case ``samples`` times to ``<url>/api/search/resolve``.

    At most ``concurrency`` posts are in flight at once. A post that fails on transport,
    times out, returns 502, 503 or 504, or returns an unreadable body is retried, as is a
    500 and an answer that reports a degraded leg or a timeout; each sample gets two
    retries, ``backoff_s`` apart. The app's own request timeout, a 504 with problem code
    ``timeout``, is not retried: the sample is infrastructure at once. A sample degraded
    after the retries is kept and marked degraded, and a 500 after them is kept as the
    sample.
    Requests are built before any post, so a case that cannot build one raises before
    traffic starts. Returns one ``CaseRun`` per case id, in case order. ``client`` defaults
    to a fresh client closed on return.
    """
    if samples < 1:
        raise ValueError(f"samples must be at least 1, got {samples}")
    endpoint = f"{url.rstrip('/')}{RESOLVE_PATH}"
    requests = {case.id: build_request(case) for case in cases}
    gate = asyncio.Semaphore(concurrency)
    async with _client_scope(client) as http, asyncio.TaskGroup() as tasks:
        runs = {
            case_id: tasks.create_task(
                _run_case(
                    http,
                    endpoint,
                    case_id,
                    request,
                    samples=samples,
                    gate=gate,
                    backoff_s=backoff_s,
                )
            )
            for case_id, request in requests.items()
        }
    return {case_id: run.result() for case_id, run in runs.items()}


def modal_sample(samples: Sequence[Sample]) -> tuple[int, int]:
    """Return the modal sample's index and the size of its answer group.

    Samples group by HTTP status and answer: per part, the status with the spec id,
    StatVars and role-keyed entities of a definite part, the spec ids of candidates, the
    no-data reason or unsupported code, and the operations in order. Within the winning
    group a second vote picks the modal set of equivalent readings. Both votes break ties
    by sample order. Raises ``ValueError`` on an empty list.
    """
    if not samples:
        raise ValueError("modal_sample needs at least one sample")
    answers = [(sample.status, _answer_key(sample.body)) for sample in samples]
    winner = _first_modal(answers)
    group = [i for i, answer in enumerate(answers) if answer == answers[winner]]
    twins = [_twin_key(samples[i].body) for i in group]
    return group[_first_modal(twins)], len(group)


async def _run_case(
    http: httpx.AsyncClient,
    endpoint: str,
    case_id: str,
    request: dict,
    *,
    samples: int,
    gate: asyncio.Semaphore,
    backoff_s: float,
) -> CaseRun:
    results = await asyncio.gather(
        *(
            _run_sample(http, endpoint, case_id, request, gate=gate, backoff_s=backoff_s)
            for _ in range(samples)
        )
    )
    done = tuple(r for r in results if isinstance(r, Sample))
    infra_error = next((r for r in results if isinstance(r, str)), None)
    return _voted(case_id, request, done, infra_error)


def extend_run(run: CaseRun, more: CaseRun) -> CaseRun:
    """``run`` with the samples of ``more``, a later run of the same case, and the modal
    sample voted again over all of them. An infrastructure error in either is kept."""
    if more.case_id != run.case_id:
        raise ValueError(f"cannot extend {run.case_id} with {more.case_id}")
    return _voted(
        run.case_id,
        run.request,
        run.samples + more.samples,
        run.infra_error or more.infra_error,
    )


def _voted(
    case_id: str, request: dict, done: tuple[Sample, ...], infra_error: str | None
) -> CaseRun:
    if infra_error is not None:
        return CaseRun(case_id, request, done, None, 0, infra_error)
    index, n_agreeing = modal_sample(done)
    if n_agreeing < len(done):
        logger.warning(
            "eval samples disagree: case=%s agreeing=%d/%d", case_id, n_agreeing, len(done)
        )
    return CaseRun(case_id, request, done, done[index], n_agreeing, None)


async def _run_sample(
    http: httpx.AsyncClient,
    endpoint: str,
    case_id: str,
    request: dict,
    *,
    gate: asyncio.Semaphore,
    backoff_s: float,
) -> Sample | str:
    """Return the sample, or the last infrastructure error once retries run out.

    A degraded answer or a 500 on the last attempt is returned as the sample. An
    infrastructure failure on the last attempt returns the error, even after an earlier
    degraded answer or 500. A failure that is not retryable returns at once.
    """
    last: Sample | str = ""
    for attempt in range(1, _RETRIES + 2):
        if attempt > 1:
            await asyncio.sleep(backoff_s)
        try:
            async with gate:
                last = await _post(http, endpoint, request, attempts=attempt)
        except _InfraFailure as exc:
            last = str(exc)
            if not exc.retryable:
                return last
            cause = last
        else:
            cause = _retry_cause(last)
            if cause is None:
                return last
        logger.warning("eval retry: case=%s attempt=%d cause=%s", case_id, attempt, cause)
    return last


def _retry_cause(sample: Sample) -> str | None:
    """Why a retry might change ``sample``, or None when it is final."""
    if sample.status == _APP_ERROR_STATUS:
        return f"HTTP {_APP_ERROR_STATUS}"
    if not sample.degraded:
        return None
    diagnostics = sample.body.get("diagnostics")
    timed_out = isinstance(diagnostics, dict) and diagnostics.get("timed_out") is True
    return "timed_out" if timed_out else f"{DEGRADED_LEG_HEADER} header"


async def _post(http: httpx.AsyncClient, endpoint: str, request: dict, *, attempts: int) -> Sample:
    started = time.perf_counter()
    try:
        response = await http.post(endpoint, json=request)
    except httpx.TransportError as exc:
        raise _InfraFailure(f"{type(exc).__name__}: {exc}") from exc
    elapsed_s = time.perf_counter() - started
    if response.status_code in _INFRA_STATUSES:
        raise _InfraFailure(f"HTTP {response.status_code}", retryable=not _is_app_timeout(response))
    try:
        body = response.json()
    except ValueError as exc:
        raise _InfraFailure(f"HTTP {response.status_code} with a non-JSON body") from exc
    if not isinstance(body, dict):
        raise _InfraFailure(f"HTTP {response.status_code} with a non-object body")
    diagnostics = body.get("diagnostics")
    timed_out = isinstance(diagnostics, dict) and diagnostics.get("timed_out") is True
    return Sample(
        body=body,
        status=response.status_code,
        degraded=timed_out or response.headers.get(DEGRADED_LEG_HEADER) == "1",
        request_rejected=(response.is_client_error and body.get("code") in _REQUEST_REJECTED_CODES),
        attempts=attempts,
        elapsed_s=elapsed_s,
    )


def _is_app_timeout(response: httpx.Response) -> bool:
    """Whether ``response`` is the app answering its own request deadline."""
    if response.status_code != _APP_TIMEOUT_STATUS:
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and body.get("code") == _APP_TIMEOUT_CODE


@asynccontextmanager
async def _client_scope(client: httpx.AsyncClient | None) -> AsyncIterator[httpx.AsyncClient]:
    """Yield the caller's client unchanged, or a fresh one closed on exit."""
    if client is not None:
        yield client
        return
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_S, headers=EVAL_HEADERS) as owned:
        yield owned


def _first_modal(values: Sequence[Hashable]) -> int:
    """Index of the earliest value among the most frequent ones."""
    counts = Counter(values)
    return max(range(len(values)), key=lambda i: counts[values[i]])


def _answer_key(body: Mapping) -> tuple:
    if body.get("kind") != "resolution":
        return (
            body.get("kind"),
            body.get("reason"),
            _get(body, "scope", "place", "dcid"),
            body.get("code"),
        )
    return tuple(_part_key(part) for part in _items(body.get("parts")))


def _part_key(part: Mapping) -> tuple:
    status = part.get("status")
    match status:
        case "definite":
            resolved = _get(part, "spec", "resolved")
            detail: tuple = (
                _get(part, "spec", "spec_id"),
                _stat_var_refs(resolved),
                _sorted(
                    (entity.get("ref"), entity.get("role"))
                    for entity in _items(_get(resolved, "entities"))
                ),
            )
        case "candidates":
            specs = _items(_get(part, "candidates", "specs"))
            detail = (_sorted(spec.get("spec_id") for spec in specs),)
        case "no_data":
            detail = (_get(part, "no_data", "reason"),)
        case "unsupported":
            detail = (_get(part, "unsupported", "code"),)
        case _:
            detail = ()
    # An operation's phrase is wording; verb, axis and target year are the computation.
    operations = tuple(
        (op.get("verb"), op.get("over"), op.get("target_year"))
        for op in _items(part.get("operations"))
    )
    return (status, *detail, operations)


def _twin_key(body: Mapping) -> tuple:
    """Per definite part, the StatVar sets of its equivalent readings, in no fixed order."""
    if body.get("kind") != "resolution":
        return ()
    return tuple(
        _sorted(
            _stat_var_refs(_get(twin, "resolved"))
            for twin in _items(_get(part, "spec", "resolved", "equivalent_readings"))
        )
        if part.get("status") == "definite"
        else ()
        for part in _items(body.get("parts"))
    )


def _stat_var_refs(resolved: object) -> tuple:
    return _sorted(item.get("ref") for item in _items(_get(resolved, "stat_vars", "items")))


def _get(value: object, *path: str) -> object:
    """Follow ``path`` through nested mappings; a missing or non-mapping step reads None."""
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _items(value: object) -> Iterable[Mapping]:
    """The mappings in a list; anything else reads as empty."""
    if not isinstance(value, list):
        return ()
    return (item for item in value if isinstance(item, Mapping))


def _sorted(values: Iterable[object]) -> tuple:
    # Ordering by repr sorts mixed None and str values, which plain sorting refuses.
    return tuple(sorted(values, key=repr))
