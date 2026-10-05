"""Harvest missing LLM fixtures into `tests/fixtures/llm_responses.json`.

The script replays the engine against frozen graph fixtures and calls the live LLM for missing
prompts. Live recording hashes prompts built from live menus. Offline replay uses frozen menus,
so the fixture keys differ.

Three collectors find keys: `sweep_corpus.py`, the offline pytest suite, and extraction over the
35 distinct role-direction queries. The first two run in subprocesses because pytest blocks
outbound HTTP. The extraction pass uses FakeLLM and runs in-process.

Collection repeats to a fixed point because answering an Extraction prompt can change the next
Bind prompt. Each new key gets one live call. A pass merges after every call succeeds. Call,
parse, schema, or `--max-passes` failures leave that pass unwritten.

The sweep summary must show every attempted golden completed or stopped on a captured
`MissingFixtureError`. Pytest passes only on exit 0, or exit 1 when every failure is a
`MissingFixtureError` for a key in the harvest file or followed a miss logged during that test
(an app test turns a miss into a 500). Other outcomes stop collection before any live LLM call.

The collectors run offline: `_collector_env` removes credentials, points `COVERAGE_ARTIFACT_URI`
at `tests/fixtures/coverage_slice` and `ORIGIN_CATALOGUE_URI` at the sample catalogue unless the
environment sets them. Only the harvest process needs `GEMINI_API_KEY`.

After convergence, pruning keeps every key requested on the final pass, hit or miss. `FakeLLM`
records both outcomes. A miss-only set is empty at convergence. A narrowed `--pytest-args` run
skips pruning because its key set is incomplete. The prune backs up the bank,
prints proposed drops, and refuses an empty reached set against a non-empty bank, a drop of a key
resolved during this run, or a drop over 90%. The bound always applies. Measured valid drops were
54.6%, 65.4%, and about 66%.

Usage:
    uv run python scripts/harvest_llm_fixtures.py
    uv run python scripts/harvest_llm_fixtures.py --max-passes 5
    uv run python scripts/harvest_llm_fixtures.py --pytest-args "-k engine"
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ENGINE_ROOT / "src"))
sys.path.insert(0, str(_ENGINE_ROOT))

from scripts.sweep_corpus import (  # noqa: E402
    OUTCOME_PATH_ENV as SWEEP_OUTCOME_PATH_ENV,
)
from scripts.sweep_corpus import (  # noqa: E402
    OUTCOME_SCHEMA_VERSION as SWEEP_OUTCOME_SCHEMA_VERSION,
)
from tests.fixtures import HARVEST_MISS_PATH_ENV, MissingFixtureError  # noqa: E402
from tests.fixtures._harvest_outcome_plugin import (  # noqa: E402
    OUTCOME_PATH_ENV,
)
from tests.fixtures._harvest_outcome_plugin import (  # noqa: E402
    SCHEMA_VERSION as OUTCOME_SCHEMA_VERSION,
)

if TYPE_CHECKING:
    from search.engine.clients.llm import SupportsLLM

_FIXTURES_DIR = _ENGINE_ROOT / "tests" / "fixtures"
_BANK_PATH = _FIXTURES_DIR / "llm_responses.json"
_DIRECTION_CASES_YAML = _ENGINE_ROOT / "corpus" / "direction" / "role_direction_cases.yaml"


class HarvestError(RuntimeError):
    """The harvest failed after zero or more committed passes.

    The failing pass leaves the bank and fixture file unchanged.
    """


@dataclass(frozen=True)
class Miss:
    """One FakeLLM miss, as captured by tests.fixtures._append_harvest_line."""

    key: str
    schema_name: str
    system: str
    prompt: str


@dataclass(frozen=True)
class PassObservations:
    """Everything one collection pass learned, parsed from its harvest file.

    `misses` (only "miss"-outcome lines) is what the resolve loop consumes.
    `requested` (every line, hit or miss) is what the prune step consumes --
    see harvest()'s own `final_pass_keys` and the module docstring's account
    of why a miss-only set is the wrong "reachable" definition."""

    misses: dict[str, Miss]
    requested: frozenset[str]


def _harvest_line(path: Path, line_no: int, line: str) -> dict:
    """Parse a harvest line as an object, raising `HarvestError` with its file and line if invalid.

    FakeLLM writes each JSON record in one call, so malformed data makes the pass outcome unknown.
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise HarvestError(
            f"{path} line {line_no} is not valid JSON ({exc.msg}); the harvest file's "
            "own record of this pass is unreadable, so the pass is not read as converged"
        ) from exc
    if not isinstance(obj, dict):
        raise HarvestError(
            f"{path} line {line_no} is a {type(obj).__name__}, not a JSON object; the "
            "harvest file's own record of this pass is unreadable, so the pass is not "
            "read as converged"
        )
    return obj


def _read_harvest_lines(harvest_file: Path) -> list[str]:
    """Read the harvest file or raise `HarvestError` if it is inaccessible."""
    try:
        return harvest_file.read_text().splitlines()
    except OSError as exc:
        raise HarvestError(
            f"harvest file {harvest_file} is unreadable ({exc}); the pass's own record "
            "of what it collected cannot be read, so the pass is not read as converged"
        ) from exc


def _parse_miss_lines(harvest_file: Path, lines: Iterable[str]) -> dict[str, Miss]:
    """Return misses deduplicated by fixture key. Hit lines carry no prompt.

    Skip blank lines. `_harvest_line` names malformed input in errors.
    """
    misses: dict[str, Miss] = {}
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        obj = _harvest_line(harvest_file, line_no, line)
        if obj.get("outcome") != "miss":
            continue
        misses[obj["key"]] = Miss(
            key=obj["key"],
            schema_name=obj["schema_name"],
            system=obj["system"],
            prompt=obj["prompt"],
        )
    return misses


def _parse_requested_keys(harvest_file: Path, lines: Iterable[str]) -> frozenset[str]:
    """Return every requested key, including hits.

    Missing keys raise `HarvestError`.
    """
    keys: set[str] = set()
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            keys.add(_harvest_line(harvest_file, line_no, line)["key"])
        except KeyError as exc:
            raise HarvestError(
                f"{harvest_file} line {line_no} carries no {exc.args[0]!r} field; the "
                "harvest file's own record of this pass is unreadable, so the pass is "
                "not read as converged"
            ) from exc
    return frozenset(keys)


def _schema_and_stage(schema_name: str) -> tuple[type, str]:
    """Map `Extraction`, `MentionReadings` and `_UniversalBindOutput` to a schema and stage.

    Unknown names raise `HarvestError`.
    """
    if schema_name == "Extraction":
        from search.engine.extract import Extraction

        return Extraction, "extract"
    if schema_name == "_UniversalBindOutput":
        from search.engine.bind import _UniversalBindOutput

        return _UniversalBindOutput, "bind"
    if schema_name == "MentionReadings":
        from search.engine.dataset_mention import MentionReadings

        return MentionReadings, "extract"
    raise HarvestError(
        f"no schema registered for schema_name={schema_name!r}; add it to "
        f"_schema_and_stage in {Path(__file__).name} before harvesting"
    )


@dataclass(frozen=True)
class HarvestResult:
    bank: dict
    added_per_pass: list[int]
    passes_run: int
    final_pass_keys: frozenset[str]
    """Every key `collect_pass()` observed REQUESTED (hit or miss) on the
    convergent final pass -- see harvest()'s own return and
    PassObservations.requested. This is the "still live" set _prune_bank
    uses to drop keys nothing reaches anymore; it is deliberately not the
    final pass's miss set, which is empty by definition on convergence."""
    newly_recorded: frozenset[str]
    """Every key THIS run resolved via a live call, across every pass --
    not `bank`'s full contents, which also holds keys `existing` already
    had. `_prune_with_safeguards` refuses if the drop set ever intersects
    this: a key just paid for must have been requested on the pass that
    resolved it, so dropping one means the reachable-set accounting is
    broken, not that the key is dead."""

    @property
    def total_added(self) -> int:
        return sum(self.added_per_pass)


async def harvest(
    *,
    existing: Mapping[str, object],
    llm: SupportsLLM,
    collect_pass: Callable[[], PassObservations],
    write_bank: Callable[[dict], None],
    max_passes: int = 10,
) -> HarvestResult:
    """Collect and resolve misses until a pass finds no key absent from `bank`.

    `collect_pass` returns deduplicated misses and every requested key. The final requested set
    supports pruning. Convergence succeeds even when this run adds no keys.

    Resolve each new key once with its stage's thinking-level pin. Merge and write only after all
    calls succeed, leaving the prior bank intact on failure. New keys on `max_passes` raise
    `HarvestError` before those keys are sent to the LLM.
    """
    from search.engine.clients.llm import for_stage

    bank = dict(existing)
    added_per_pass: list[int] = []
    newly_recorded: set[str] = set()
    for pass_num in range(1, max_passes + 1):
        # The direction collector calls `asyncio.run`, which needs a thread with no running loop.
        obs = await asyncio.to_thread(collect_pass)
        misses = obs.misses
        new = {key: miss for key, miss in misses.items() if key not in bank}
        if not new:
            added_per_pass.append(0)
            return HarvestResult(
                bank=bank,
                added_per_pass=added_per_pass,
                passes_run=pass_num,
                final_pass_keys=obs.requested,
                newly_recorded=frozenset(newly_recorded),
            )
        if pass_num == max_passes:
            raise HarvestError(
                f"still finding new keys ({len(new)}) on pass {max_passes}, the "
                "configured --max-passes bound; not calling the LLM for them. The bank "
                "is left at its last fully-resolved state. Re-run with a higher "
                "--max-passes if the corpus genuinely needs more passes, or investigate "
                "why collection keeps finding new keys (e.g. a non-deterministic prompt)."
            )
        resolved: dict[str, object] = {}
        for key, miss in new.items():
            schema, stage = _schema_and_stage(miss.schema_name)
            parsed, _usage = await for_stage(llm, stage).generate_structured(
                prompt=miss.prompt, system=miss.system, schema=schema
            )
            resolved[key] = parsed.model_dump(mode="json")
        bank.update(resolved)
        newly_recorded.update(resolved)
        write_bank(bank)
        added_per_pass.append(len(new))
    # Unreachable: the pass_num == max_passes branch above always returns or
    # raises before the loop would otherwise exhaust.
    raise AssertionError("unreachable")


def _run_pass_subprocess(
    *, cmd: Sequence[str], env: Mapping[str, str], ok_returncodes: Sequence[int]
) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=_ENGINE_ROOT, env=dict(env))
    if result.returncode not in ok_returncodes:
        raise HarvestError(
            f"`{' '.join(cmd)}` exited {result.returncode} (expected one of "
            f"{list(ok_returncodes)}); aborting harvest without writing anything"
        )
    return result


def _direction_corpus_queries() -> list[str]:
    """Return unique query strings in case-file order."""
    cases = yaml.safe_load(_DIRECTION_CASES_YAML.read_text())
    seen: dict[str, None] = {}
    for case in cases:
        seen.setdefault(case["query"], None)
    return list(seen)


def _direction_corpus_collect(*, harvest_path: Path) -> None:
    """Run extraction over direction queries and append every requested key to the shared file.

    Extraction reads no graph, so this collector runs in-process. A fresh `FakeLLM` per query
    reloads fixtures written earlier in the same pass. Pin `extract.date` to `PINNED_DATE` because
    the system prompt includes the date in each fixture key.
    """
    from datetime import date
    from unittest.mock import patch

    from search.engine.extract import extract as _extract
    from tests.fixtures import PINNED_DATE, FakeLLM

    old_env = os.environ.get(HARVEST_MISS_PATH_ENV)
    os.environ[HARVEST_MISS_PATH_ENV] = str(harvest_path)
    try:
        with patch("search.engine.extract.date") as mock_date:
            mock_date.today.return_value = PINNED_DATE
            mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
            for query in _direction_corpus_queries():
                # FakeLLM logs the miss before raising.
                with contextlib.suppress(MissingFixtureError):
                    asyncio.run(_extract(query, llm=FakeLLM()))
    finally:
        if old_env is None:
            os.environ.pop(HARVEST_MISS_PATH_ENV, None)
        else:
            os.environ[HARVEST_MISS_PATH_ENV] = old_env


def _read_outcome_summary(outcome_path: Path) -> dict:
    """Read the pytest outcome summary or raise `HarvestError` if it is unknown or invalid."""
    try:
        summary = json.loads(outcome_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HarvestError(
            f"pytest outcome summary at {outcome_path} is missing or unreadable ({exc}); the "
            "pass's outcome cannot be established, so it is not read as converged"
        ) from exc
    if not isinstance(summary, dict):
        raise HarvestError(
            f"pytest outcome summary at {outcome_path} is a {type(summary).__name__}, not a "
            "JSON object; refusing to read a summary this code cannot interpret"
        )
    if summary.get("schema_version") != OUTCOME_SCHEMA_VERSION:
        raise HarvestError(
            f"pytest outcome summary at {outcome_path} does not carry "
            f"schema_version {OUTCOME_SCHEMA_VERSION} (got {summary.get('schema_version')!r}); "
            "refusing to read a summary this code cannot interpret"
        )
    return summary


def _check_pytest_outcome(
    *, returncode: int, outcome_path: Path, harvest_file: Path, where: str
) -> None:
    """Accept exit 0 with no failures, or exit 1 when every failure traces to a collected miss.

    A failure traces to a miss when it is a `MissingFixtureError` whose key appears in
    `harvest_file`, or when the plugin saw a miss logged while that test ran
    (`misses_during`). Other exits, malformed summaries, unrelated failures, and missing keys
    raise `HarvestError` before any LLM call.
    """
    # Validate the summary for both accepted exit codes.
    summary = _read_outcome_summary(outcome_path)
    if returncode not in (0, 1):
        raise HarvestError(
            f"{where}: pytest exited {returncode}, which is neither 0 (all passed) nor 1 "
            "(test failures); the pass did not run to a completed suite, so it is not read "
            "as converged"
        )
    failures = summary.get("failures")
    if not isinstance(failures, list):
        raise HarvestError(
            f"{where}: pytest outcome summary at {outcome_path} carries no `failures` list; "
            "the pass's outcome cannot be established"
        )
    if returncode == 0:
        if failures:
            raise HarvestError(
                f"{where}: pytest exited 0 but its outcome summary records "
                f"{len(failures)} failing test(s); the two disagree, so the pass is not read "
                "as converged"
            )
        return
    if not failures:
        raise HarvestError(
            f"{where}: pytest exited 1 but its outcome summary records no failing test; the "
            "outcome cannot be established, so the pass is not read as converged"
        )
    for failure in failures:
        exception_type = failure.get("exception_type")
        if exception_type == MissingFixtureError.__name__:
            continue
        if failure.get("misses_during", 0) > 0:
            # Collateral: the test hit a missing fixture that the code under test turned into
            # another error (an app test sees a 500). The miss is in the harvest file.
            continue
        raise HarvestError(
            f"{where}: pytest failed {failure.get('nodeid')!r} with {exception_type!r}, "
            "which is neither a missing FakeLLM fixture nor a failure that followed one. The "
            "suite is broken for a reason this pass cannot collect as a miss, so the pass is "
            "not read as converged and no fixture is written."
        )
    collected_keys = _parse_requested_keys(harvest_file, _read_harvest_lines(harvest_file))
    missing_keys = [
        failure.get("fixture_key")
        for failure in failures
        if failure.get("exception_type") == MissingFixtureError.__name__
        and (not failure.get("fixture_key") or failure["fixture_key"] not in collected_keys)
    ]
    if missing_keys:
        raise HarvestError(
            f"{where}: pytest failed on missing fixture(s) {missing_keys!r} that the harvest "
            f"file {harvest_file} does not carry; the pass did not capture those misses, so it "
            "is not read as converged"
        )


def _read_sweep_outcome_summary(outcome_path: Path) -> dict:
    """Read the sweep outcome summary or raise `HarvestError` if it is unknown or invalid."""
    try:
        summary = json.loads(outcome_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HarvestError(
            f"sweep outcome summary at {outcome_path} is missing or unreadable ({exc}); the "
            "sweep pass's outcome cannot be established, so it is not read as converged"
        ) from exc
    if not isinstance(summary, dict):
        raise HarvestError(
            f"sweep outcome summary at {outcome_path} is a {type(summary).__name__}, not a "
            "JSON object; refusing to read a summary this code cannot interpret"
        )
    if summary.get("schema_version") != SWEEP_OUTCOME_SCHEMA_VERSION:
        raise HarvestError(
            f"sweep outcome summary at {outcome_path} does not carry schema_version "
            f"{SWEEP_OUTCOME_SCHEMA_VERSION} (got {summary.get('schema_version')!r}); refusing "
            "to read a summary this code cannot interpret"
        )
    return summary


def _check_sweep_outcome(*, outcome_path: Path, harvest_file: Path) -> None:
    """Require every attempted golden to complete or to stop on a collected fixture miss.

    The sweep command exits 0 when its per-golden handler skips failures. A golden that raised
    `MissingFixtureError` is a miss this pass exists to collect, and `harvest_file` must hold at
    least one miss line. Any other skipped golden, or an incomplete summary, raises
    `HarvestError` before any LLM call.
    """
    summary = _read_sweep_outcome_summary(outcome_path)
    failures = summary.get("failures")
    attempted = summary.get("attempted")
    completed = summary.get("completed")
    if (
        not isinstance(failures, list)
        or not isinstance(attempted, int)
        or not isinstance(completed, int)
    ):
        raise HarvestError(
            f"sweep outcome summary at {outcome_path} does not carry the expected "
            "attempted/completed ints and failures list; the sweep pass's outcome cannot be "
            "established"
        )
    missed = [f for f in failures if f.get("error_class") == MissingFixtureError.__name__]
    skipped = [f for f in failures if f.get("error_class") != MissingFixtureError.__name__]
    if skipped:
        raise HarvestError(
            f"sweep collector: {len(skipped)} golden(s) raised and were skipped "
            f"({skipped!r}); the pass's reached set omits whatever those goldens' remaining "
            "FakeLLM calls would have requested, so it is not read as converged"
        )
    if completed + len(missed) < attempted:
        raise HarvestError(
            f"sweep collector: completed {completed} of {attempted} attempted golden(s); the "
            "pass's reached set is incomplete, so it is not read as converged"
        )
    if missed and not _parse_miss_lines(harvest_file, _read_harvest_lines(harvest_file)):
        raise HarvestError(
            f"sweep collector: {len(missed)} golden(s) stopped on a missing fixture but "
            f"{harvest_file} holds no miss line; the misses were not captured, so the pass is "
            "not read as converged"
        )


_XDIST_OPTIONS = ("-n", "--numprocesses", "--dist")
"""Every pytest option that turns a run into an xdist run."""


def _refuse_xdist(pytest_args: Sequence[str]) -> None:
    """Reject xdist arguments because the outcome plugin needs per-test exceptions.

    Xdist's controller reports omit `excinfo`, hiding fixture misses from the summary.
    """
    for arg in pytest_args:
        name = arg.split("=", 1)[0]
        if name in _XDIST_OPTIONS or (name.startswith("-n") and name != "-n"):
            raise HarvestError(
                f"`--pytest-args` contains {arg!r}, which asks for pytest-xdist; the "
                "pytest collector runs with `-p no:xdist` because xdist's controller "
                "summary records no failing test, which would abort the harvest on a "
                "run that missed fixtures. Drop the flag (the pass is serial)."
            )


_PYTEST_COLLECTION_ENV_VARS = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")
"""Environment variables pytest reads on every invocation, independent of
any CLI argument, that can narrow or otherwise change which tests are
collected. `main()`'s decision to skip pruning trusts `--pytest-args` alone
to know whether a run was narrowed (see the prune gate at the bottom of
`main()`), so an inherited value for any of these would let a run look
unfiltered while the collector actually ran a subset -- the same hazard
`_refuse_xdist` guards against for an explicit flag. Stripped from the
collector subprocess's environment in `_live_collect_pass`, not from this
process's own `os.environ`, so the harvest keeps behaving the same
regardless of what the invoking shell set.
"""


def _strip_pytest_collection_env(env: dict[str, str]) -> None:
    """Remove pytest collection overrides from `env` and report each removed value."""
    for var in _PYTEST_COLLECTION_ENV_VARS:
        value = env.pop(var, None)
        if value is not None:
            print(
                f"Ignoring inherited {var}={value!r}: the harvest's pytest collector "
                "excludes it from the child environment, since --pytest-args is the "
                "only narrowing the prune gate checks for."
            )


_CHILD_CREDENTIAL_VARS = ("GEMINI_API_KEY", "SEARCH_GRAPH_TOKEN", "SEARCH_GRAPH_BASE")
"""Credentials the harvest process needs for live LLM calls and the collectors must not see. A
credential in a collector's environment un-skips live tests, which can hang or reach the network."""


def _collector_env(*, harvest_path: Path) -> dict[str, str]:
    """Build the environment the sweep and pytest collectors run under.

    The collectors replay recorded fixtures, so they run with the offline suite's configuration.
    Credentials and `LANGFUSE_*` variables are removed. `COVERAGE_ARTIFACT_URI` defaults to
    `tests/fixtures/coverage_slice` with a zero recheck interval, as `tests/conftest.py` sets
    it. `ORIGIN_CATALOGUE_URI` defaults to the sample catalogue the origin tests publish, written
    beside the harvest file. A value already in the environment wins. `RESOLVEKIT_CACHE_DIR` is
    not read by the collectors; `tests/fixtures/record.py` needs it.
    """
    from tests.origin_catalogue.support import publish_sample

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _CHILD_CREDENTIAL_VARS and not k.startswith("LANGFUSE_")
    }
    env[HARVEST_MISS_PATH_ENV] = str(harvest_path)
    env.setdefault("COVERAGE_ARTIFACT_URI", f"file://{_FIXTURES_DIR / 'coverage_slice'}")
    env.setdefault("COVERAGE_RECHECK_SECONDS", "0")
    if "ORIGIN_CATALOGUE_URI" not in env:
        catalogue_dir = harvest_path.parent / "origin-catalogue"
        # Every pass builds this environment, and a published build directory is write-once.
        if not catalogue_dir.exists():
            publish_sample(f"file://{catalogue_dir}")
        uri = f"file://{catalogue_dir}"
        env["ORIGIN_CATALOGUE_URI"] = uri
    return env


def _live_collect_pass(*, harvest_path: Path, pytest_args: Sequence[str]) -> PassObservations:
    """Run sweep, pytest, and direction-query collectors into a fresh harvest file.

    Collectors run under `_collector_env`. Validate sweep and pytest outcome summaries before any
    LLM call. Pytest runs serially with
    the outcome plugin. Explicit xdist args are refused.
    Inherited collection overrides are stripped.

    Return deduplicated misses and every requested key from the shared file.
    """
    _refuse_xdist(pytest_args)
    harvest_path.write_text("")
    env = _collector_env(harvest_path=harvest_path)
    _strip_pytest_collection_env(env)
    sweep_outcome_path = harvest_path.with_name("sweep_outcome.json")
    sweep_outcome_path.unlink(missing_ok=True)
    _run_pass_subprocess(
        cmd=["uv", "run", "python", "scripts/sweep_corpus.py"],
        env={**env, SWEEP_OUTCOME_PATH_ENV: str(sweep_outcome_path)},
        ok_returncodes=(0,),
    )
    _check_sweep_outcome(outcome_path=sweep_outcome_path, harvest_file=harvest_path)
    outcome_path = harvest_path.with_name("pytest_outcome.json")
    outcome_path.unlink(missing_ok=True)
    pytest_run = _run_pass_subprocess(
        # `python -m pytest` puts the repo root on `sys.path` before plugin arguments resolve.
        # Disable xdist so the outcome plugin records failures in the test process.
        cmd=[
            "uv",
            "run",
            "python",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:xdist",
            "-p",
            "tests.fixtures._harvest_outcome_plugin",
            *pytest_args,
        ],
        env={**env, OUTCOME_PATH_ENV: str(outcome_path)},
        ok_returncodes=(0, 1),
    )
    _check_pytest_outcome(
        returncode=pytest_run.returncode,
        outcome_path=outcome_path,
        harvest_file=harvest_path,
        where="pytest collector",
    )
    _direction_corpus_collect(harvest_path=harvest_path)
    lines = _read_harvest_lines(harvest_path)
    return PassObservations(
        misses=_parse_miss_lines(harvest_path, lines),
        requested=_parse_requested_keys(harvest_path, lines),
    )


def _write_bank(bank: dict, *, path: Path = _BANK_PATH) -> None:
    from tests.fixtures.record import _write_json

    _write_json(path, bank)


_PRUNE_MAX_DROP_FRACTION = 0.90
"""A high, unconditional backstop -- see the module docstring's safeguard
(5). A narrowed `--pytest-args` run is handled upstream by main() skipping
the prune outright, not by this bound; what this guards is a near-total,
zero-new-call drop on an UNFILTERED run whose `reached` set is wrongly small
for some other reason (a collector silently under-reporting, say), and it
sits above every legitimate drop measured so far. No override flag: a drop
this large is always worth a human look, not a flag to skip past."""


class PruneRefused(RuntimeError):
    """A prune failed a safety check; no pruned bank was written."""


def _sidecar_path(bank_path: Path) -> Path:
    return bank_path.with_name(bank_path.name + ".preprune")


def _prune_bank(bank: dict, *, reached: frozenset[str]) -> tuple[dict, dict[str, int]]:
    """Remove fixtures outside `reached` and return the bank plus counts by schema.

    `reached` includes hit and miss keys from the converged pass. Keys without `:` are metadata
    and pass through unchanged.
    """
    dropped_by_schema: dict[str, int] = {}
    pruned: dict = {}
    for key, value in bank.items():
        if ":" not in key or key in reached:
            pruned[key] = value
            continue
        schema_name = key.split(":", 1)[0].split("@", 1)[0]
        dropped_by_schema[schema_name] = dropped_by_schema.get(schema_name, 0) + 1
    return pruned, dropped_by_schema


def _prune_with_safeguards(
    bank: dict,
    *,
    reached: frozenset[str],
    newly_recorded: frozenset[str] = frozenset(),
    bank_path: Path = _BANK_PATH,
) -> tuple[dict, dict[str, int]]:
    """Back up the bank before checking the reached set.

    Empty reach raises `PruneRefused`. Report proposed drops, then check newly recorded keys and the
    maximum drop fraction.
    """
    sidecar = _sidecar_path(bank_path)
    _write_bank(bank, path=sidecar)
    print(f"Pre-prune bank backed up to {sidecar}")

    if bank and not reached:
        raise PruneRefused(
            f"reachable set is empty against a {len(bank)}-key bank -- this is exactly the "
            "signature of a converged pass reporting no misses, read as reaching nothing "
            "(see the module docstring). Refusing rather than emptying the bank. Pre-prune "
            f"bank is intact at {bank_path} and backed up at {sidecar}."
        )

    pruned, dropped_by_schema = _prune_bank(bank, reached=reached)
    dropped_keys = frozenset(bank) - frozenset(pruned)
    total_dropped = len(dropped_keys)
    if total_dropped:
        breakdown = ", ".join(
            f"{count} {schema}" for schema, count in sorted(dropped_by_schema.items())
        )
        print(f"Prune would drop {total_dropped}/{len(bank)} key(s): {breakdown}")
    else:
        print("Prune would drop 0 key(s).")

    paid_for_and_dropped = dropped_keys & newly_recorded
    if paid_for_and_dropped:
        raise PruneRefused(
            f"prune would drop {len(paid_for_and_dropped)} key(s) this very run just resolved "
            "and paid for -- a logical impossibility if the reachable-set accounting is "
            "correct, since a key just answered must have been requested on the pass that "
            f"resolved it. Refusing. Sample: {sorted(paid_for_and_dropped)[:5]}. Pre-prune "
            f"bank is intact at {bank_path} and backed up at {sidecar}."
        )

    fraction = total_dropped / len(bank) if bank else 0.0
    if fraction > _PRUNE_MAX_DROP_FRACTION:
        raise PruneRefused(
            f"prune would drop {total_dropped}/{len(bank)} key(s) ({fraction:.0%}), over the "
            f"{_PRUNE_MAX_DROP_FRACTION:.0%} sanity bound -- see _PRUNE_MAX_DROP_FRACTION's own "
            f"docstring for the narrow gap this catches. Refusing. Pre-prune bank is intact at "
            f"{bank_path} and backed up at {sidecar}."
        )
    return pruned, dropped_by_schema


def _report(
    result: HarvestResult,
    *,
    existing_count: int,
    dropped_by_schema: Mapping[str, int],
    final_count: int,
) -> None:
    print(f"passes run: {result.passes_run}")
    for i, added in enumerate(result.added_per_pass, start=1):
        print(f"  pass {i}: {added} new key(s)")
    if result.total_added == 0:
        print(f"No new keys found across {result.passes_run} pass(es).")
    else:
        print(f"Added {result.total_added} key(s) across {result.passes_run} pass(es).")
    total_dropped = sum(dropped_by_schema.values())
    if total_dropped == 0:
        print("Prune: 0 orphaned key(s) dropped.")
    else:
        breakdown = ", ".join(
            f"{count} {schema}" for schema, count in sorted(dropped_by_schema.items())
        )
        print(f"Prune: {total_dropped} orphaned key(s) dropped ({breakdown}).")
    print(f"Bank: {existing_count} entries before, {final_count} after.")


def main() -> None:
    import argparse

    from search.engine.clients.llm import LLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-passes",
        type=int,
        default=10,
        help="Abort (without writing) if the fixed point isn't reached within this many passes.",
    )
    parser.add_argument(
        "--pytest-args",
        default="",
        help=(
            "Extra arguments forwarded to the pytest collector, e.g. '-k engine'. "
            "Any non-empty value narrows collection, so this run still harvests "
            "but skips pruning -- run without --pytest-args to prune."
        ),
    )
    args = parser.parse_args()

    existing = json.loads(_BANK_PATH.read_text())
    live_llm = LLM()
    # Preserve quoted arguments as single argv entries for pytest and `_refuse_xdist`.
    pytest_args = shlex.split(args.pytest_args) if args.pytest_args else []

    print(
        f"LLM fixture harvest (existing bank: {len(existing)} entries; "
        f"max_passes={args.max_passes}; requires GEMINI_API_KEY)"
    )

    with tempfile.TemporaryDirectory(prefix="harvest_llm_fixtures_") as tmp_dir:
        harvest_path = Path(tmp_dir) / "misses.jsonl"

        def collect_pass() -> PassObservations:
            return _live_collect_pass(harvest_path=harvest_path, pytest_args=pytest_args)

        try:
            result = asyncio.run(
                harvest(
                    existing=existing,
                    llm=live_llm,
                    collect_pass=collect_pass,
                    write_bank=_write_bank,
                    max_passes=args.max_passes,
                )
            )
        except HarvestError as exc:
            print(f"HARVEST FAILED: {exc}", file=sys.stderr)
            sys.exit(1)

    if pytest_args:
        print(
            "Pruning skipped: --pytest-args narrows the pytest collector, so this "
            "run's reached set is not a full harvest's -- run without --pytest-args "
            "to prune."
        )
        pruned_bank, dropped_by_schema = result.bank, {}
    else:
        try:
            pruned_bank, dropped_by_schema = _prune_with_safeguards(
                result.bank, reached=result.final_pass_keys, newly_recorded=result.newly_recorded
            )
        except PruneRefused as exc:
            print(f"PRUNE REFUSED: {exc}", file=sys.stderr)
            sys.exit(1)
    if pruned_bank != result.bank:
        _write_bank(pruned_bank)
    _report(
        result,
        existing_count=len(existing),
        dropped_by_schema=dropped_by_schema,
        final_count=len(pruned_bank),
    )


if __name__ == "__main__":
    main()
