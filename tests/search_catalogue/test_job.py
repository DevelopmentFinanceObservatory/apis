"""Pins the search catalogue job with a fake engine over the origin catalogue
sample build and the coverage fixture."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import graph_release.service as release_service
from artifact_publication.jobs import LEASE_FILE, Lease, acquire_lease, new_build_id
from graph_release.context import ReleaseNotBound, current_release
from graph_release.views import ViewRefused
from search.catalogue import job
from search.catalogue.artifact import (
    FORMAT_ROOT,
    REFUSAL_FILE,
    BuildInputs,
    build_key,
    load_published,
    read_refusal,
)
from search.catalogue.check import ROLE_KEYS
from search.catalogue.examples import load_examples
from search.catalogue.job import (
    FLOOR,
    LEASE_TTL_SECONDS,
    LEASE_WAIT_SECONDS,
    REFUSAL_RETRY_SECONDS,
    CatalogueEngineError,
    CatalogueFloorError,
    JobSettings,
    nothing_to_build,
    run,
)
from search.engine.errors import EngineInputError
from series_coverage.index import CoverageIndex
from series_coverage.schema import CoverageManifest
from series_coverage.shared import CoverageSnapshot
from tests.graph_release.support import build_fixture_index_release, build_view_release
from tests.origin_catalogue.support import CENSUS_POVERTY, CRS_VARIABLE, GDP, POPULATION
from tests.search_catalogue.support import (
    Elected,
    FakeEngine,
    definite,
    omitted,
    origin_snapshot,
    publish_catalogue,
    sample_catalogue,
    timed_out,
)
from tests.series_coverage.support import V1, V2, V3, build_test_index

EXAMPLES = f"""
[[example]]
template = "Population of {{place}}"
stat_var = "{POPULATION}"
place = "country/KEN"

[[example]]
template = "GDP of {{place}}"
stat_var = "{GDP}"
place = "country/USA"

[[example]]
template = "Health aid to {{place}}"
stat_var = "{V1}"
role = "recipient"
place = "country/KEN"

[[example]]
template = "Health aid from {{place}}"
stat_var = "{V1}"
role = "donor"
place = "country/FRA"

[[example]]
template = "Real health aid to {{place}}"
stat_var = "{V2}"
role = "recipient"
place = "country/ZWE"

[[example]]
template = "Education aid to {{place}}"
stat_var = "{V3}"
role = "recipient"
place = "country/ETH"

[[example]]
template = "Official aid to {{place}}"
stat_var = "{CRS_VARIABLE}"
place = "country/KEN"

[[example]]
template = "Poverty in {{place}}"
stat_var = "{CENSUS_POVERTY}"
place = "country/USA"
""".encode()

NAMES = {
    "country/DEU": "Germany",
    "country/ETH": "Ethiopia",
    "country/FRA": "France",
    "country/KEN": "Kenya",
    "country/USA": "United States",
    "country/ZWE": "Zimbabwe",
}

_NOW = 1_790_000_000.0


@dataclass
class Setup:
    settings: JobSettings
    catalogue: Any
    coverage: Any
    out: Path


@dataclass
class _StaticCoverage:
    """Serves `index` as the coverage of the release `release_id`, as `ReleaseCoverage`
    serves the bound release's."""

    index: CoverageIndex
    release_id: str = "release-0"

    def snapshot(self) -> CoverageSnapshot:
        manifest = CoverageManifest(
            stamp=self.release_id,
            build_id=self.release_id,
            built_at="2026-09-30T00:00:00Z",
            horizon=self.index.horizon,
        )
        return CoverageSnapshot(index=self.index, manifest=manifest, stamp=self.release_id)


@pytest.fixture
def setup(tmp_path) -> Setup:
    out = tmp_path / "out"
    return Setup(
        settings=JobSettings(
            artifact_uri=out.as_uri(), origin_catalogue_uri=(tmp_path / "origin").as_uri()
        ),
        catalogue=origin_snapshot(tmp_path / "origin"),
        coverage=_StaticCoverage(build_test_index()),
        out=out,
    )


def _engine(answers: dict[str, Any] | None = None) -> FakeEngine:
    """Elects each example's StatVar at its default place, unless `answers`
    says otherwise."""
    examples = load_examples(EXAMPLES).examples
    return FakeEngine(
        elected={
            example.text(NAMES[example.place]): Elected(
                stat_var=example.stat_var, place=example.place, role=ROLE_KEYS[example.role]
            )
            for example in examples
        },
        answers=answers or {},
    )


def _run(
    setup: Setup,
    engine: FakeEngine,
    *,
    commit: str = "c1",
    coverage: Any = None,
    now: float = _NOW,
) -> None:
    asyncio.run(
        run(
            setup.settings,
            examples=load_examples(EXAMPLES),
            catalogue=setup.catalogue,
            coverage=coverage or setup.coverage,
            resolve=engine,
            place_names=lambda dcids: {dcid: NAMES[dcid] for dcid in dcids if dcid in NAMES},
            commit=commit,
            version=new_build_id(),
            started_at=datetime.fromtimestamp(now, UTC),
            lease=Lease(build_id="run", expires_at=now + LEASE_TTL_SECONDS, generation="1"),
            clock=lambda: now,
        )
    )


def _events(capsys) -> list[dict[str, Any]]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def _named(events: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [event for event in events if event["event"] == name]


def test_publishes_the_passing_lines_with_counts_and_places(setup, capsys):
    engine = _engine()
    _run(setup, engine)
    events = _events(capsys)

    [published] = _named(events, "search_catalogue_job_published")
    assert events[-1]["event"] == "search_catalogue_job_heartbeat"
    assert events[-1]["catalogue_version"] == published["catalogue_version"]

    artifact = load_published(setup.out.as_uri())
    assert artifact.catalogue_version == published["catalogue_version"]
    assert [(example.id, example.text) for example in artifact.examples] == [
        (0, "Population of Kenya"),
        (1, "GDP of United States"),
        (2, "Health aid to Kenya"),
        (3, "Health aid from France"),
        (4, "Real health aid to Zimbabwe"),
        (5, "Education aid to Ethiopia"),
    ]
    assert artifact.counts.publishers == 4
    assert artifact.counts.countries == len(artifact.places) == 6
    # The fixture's facets belong to a dataset the packaged code index declares.
    assert artifact.counts.aid.donors == 3
    assert artifact.inputs.build_commit == "c1"
    # Only the default places were sent: the fixture holds neither test place.
    assert sorted(engine.queries) == sorted(example.text for example in artifact.examples)


def test_a_template_is_offered_only_where_its_stat_var_has_data(setup, capsys):
    _run(setup, _engine())
    places = load_published(setup.out.as_uri()).places
    assert {country: place.examples for country, place in places.items()} == {
        "country/DEU": [3],
        "country/ETH": [2, 5],
        "country/FRA": [3],
        "country/KEN": [0, 1, 2, 4, 5],
        "country/USA": [0, 1],
        "country/ZWE": [4],
    }
    assert places["country/KEN"].name == "Kenya"


def test_a_line_without_a_presence_source_drops_and_alerts(setup, capsys):
    _run(setup, _engine())
    dropped = _named(_events(capsys), "search_catalogue_line_dropped")
    assert [(event["example_id"], event["reason"], event["severity"]) for event in dropped] == [
        (6, "no_presence_source", "WARNING"),
        (7, "no_presence_source", "WARNING"),
    ]


def test_the_pin_check_drops_a_line_resolving_to_the_wrong_stat_var(setup, capsys):
    engine = _engine(answers={"Population of Kenya": definite("Count_Person_Female")})
    _run(setup, engine)
    [dropped] = [
        event
        for event in _named(_events(capsys), "search_catalogue_line_dropped")
        if event["reason"] == "check_failed"
    ]
    assert dropped["example_id"] == 0
    assert dropped["failures"] == [
        {
            "place": "country/KEN",
            "text": "Population of Kenya",
            "reason": "stat_var_not_elected",
            "stat_vars": ["Count_Person_Female"],
            "detail": None,
        }
    ]
    artifact = load_published(setup.out.as_uri())
    assert 0 not in [example.id for example in artifact.examples]
    assert artifact.places["country/KEN"].examples == [1, 2, 4, 5]


def test_the_rows_check_drops_a_line_with_an_empty_preview(setup, capsys):
    engine = _engine(answers={"GDP of United States": definite(GDP, place="country/USA", points=0)})
    _run(setup, engine)
    [dropped] = [
        event
        for event in _named(_events(capsys), "search_catalogue_line_dropped")
        if event["reason"] == "check_failed"
    ]
    assert dropped["example_id"] == 1
    assert dropped["failures"][0]["reason"] == "preview_empty"
    assert [example.id for example in load_published(setup.out.as_uri()).examples] == [
        0,
        2,
        3,
        4,
        5,
    ]


def test_an_engine_failure_after_its_retry_fails_the_job_without_a_refusal(setup, capsys):
    engine = _engine(answers={"GDP of United States": RuntimeError("upstream 503")})
    with pytest.raises(CatalogueEngineError):
        _run(setup, engine)
    events = _events(capsys)
    [failed] = _named(events, "search_catalogue_job_failed")
    assert failed["severity"] == "ERROR"
    assert [(f["example_id"], f["reason"]) for f in failed["failures"]] == [(1, "engine_error")]
    assert not _named(events, "search_catalogue_job_refused")
    assert not _named(events, "search_catalogue_job_heartbeat")
    assert engine.queries.count("GDP of United States") == 2
    assert not (setup.out / FORMAT_ROOT / "current.json").exists()
    assert read_refusal(setup.out.as_uri()) is None


def test_a_failed_preview_read_fails_the_job_without_dropping_the_line(setup, capsys):
    engine = _engine(
        answers={
            "GDP of United States": omitted(GDP, "observations_probe_failed", place="country/USA")
        }
    )
    with pytest.raises(CatalogueEngineError):
        _run(setup, engine)
    events = _events(capsys)
    [failed] = _named(events, "search_catalogue_job_failed")
    assert [(f["example_id"], f["reason"]) for f in failed["failures"]] == [(1, "preview_failed")]
    assert not [
        event
        for event in _named(events, "search_catalogue_line_dropped")
        if event["reason"] == "check_failed"
    ]
    assert engine.queries.count("GDP of United States") == 2
    assert not (setup.out / FORMAT_ROOT / "current.json").exists()
    assert read_refusal(setup.out.as_uri()) is None


def test_preview_reads_timing_out_on_every_line_record_no_refusal(setup, capsys):
    engine = _engine()
    engine.answers = {
        query: omitted(
            elected.stat_var, "observations_timed_out", place=elected.place, role=elected.role
        )
        for query, elected in engine.elected.items()
    }
    with pytest.raises(CatalogueEngineError):
        _run(setup, engine)
    events = _events(capsys)
    [failed] = _named(events, "search_catalogue_job_failed")
    assert {f["reason"] for f in failed["failures"]} == {"preview_failed"}
    assert not _named(events, "search_catalogue_job_refused")
    assert read_refusal(setup.out.as_uri()) is None


def test_an_input_refusal_drops_its_line_and_the_rest_publish(setup, capsys):
    refusal = EngineInputError("query refused", code="request_invalid")
    engine = _engine(answers={"GDP of United States": refusal})
    _run(setup, engine)
    events = _events(capsys)
    [dropped] = [
        event
        for event in _named(events, "search_catalogue_line_dropped")
        if event["reason"] == "check_failed"
    ]
    assert dropped["example_id"] == 1
    assert dropped["failures"][0]["reason"] == "engine_refused"
    assert not _named(events, "search_catalogue_job_failed")
    assert engine.queries.count("GDP of United States") == 1
    assert [example.id for example in load_published(setup.out.as_uri()).examples] == [
        0,
        2,
        3,
        4,
        5,
    ]


def test_input_refusals_count_toward_the_floor(setup, capsys):
    refusal = EngineInputError("query refused", code="request_invalid")
    engine = _engine(answers={"Population of Kenya": refusal, "GDP of United States": refusal})
    with pytest.raises(CatalogueFloorError):
        _run(setup, engine)
    [refused] = _named(_events(capsys), "search_catalogue_job_refused")
    assert refused["passed"] == [2, 3, 4, 5]


def test_the_next_run_after_an_engine_failure_builds_again(setup, capsys):
    with pytest.raises(CatalogueEngineError):
        _run(
            setup,
            _engine(
                answers={"GDP of United States": timed_out(definite(GDP, place="country/USA"))}
            ),
        )
    capsys.readouterr()
    _run(setup, _engine(), now=_NOW + 900)
    assert _named(_events(capsys), "search_catalogue_job_published")


def test_a_run_the_engine_fails_once_keeps_its_line(setup, capsys):
    engine = _engine()
    engine.first["GDP of United States"] = RuntimeError("upstream 429")
    _run(setup, engine)
    assert _named(_events(capsys), "search_catalogue_job_published")
    assert 1 in [example.id for example in load_published(setup.out.as_uri()).examples]


def test_a_run_that_loses_the_publish_race_keeps_the_winner(setup, capsys):
    engine = _engine()
    winner: list[str] = []

    async def racing(request):
        # Another run publishes while this one checks its first example.
        if not winner:
            winner.append(publish_catalogue(setup.out.as_uri()).catalogue_version)
        return await engine(request)

    _run(setup, racing)
    events = _events(capsys)
    [lost] = _named(events, "search_catalogue_job_lost_publish_race")
    assert not _named(events, "search_catalogue_job_published")
    assert lost["catalogue_version"] != winner[0]
    assert events[-1]["event"] == "search_catalogue_job_heartbeat"
    assert events[-1]["catalogue_version"] == winner[0]
    assert load_published(setup.out.as_uri()).catalogue_version == winner[0]


def _refusing_engine() -> FakeEngine:
    """Fails two of the six lines with presence, leaving four, under `FLOOR`."""
    return _engine(
        answers={
            "Population of Kenya": definite("Count_Person_Female"),
            "GDP of United States": definite(GDP, place="country/USA", points=0),
        }
    )


def test_under_the_floor_the_run_publishes_nothing(setup, capsys):
    with pytest.raises(CatalogueFloorError):
        _run(setup, _refusing_engine())
    events = _events(capsys)
    [refused] = _named(events, "search_catalogue_job_refused")
    assert refused["severity"] == "ERROR"
    assert refused["passed"] == [2, 3, 4, 5]
    assert refused["floor"] == FLOOR
    assert not _named(events, "search_catalogue_job_heartbeat")
    assert not (setup.out / FORMAT_ROOT / "current.json").exists()
    refusal = read_refusal(setup.out.as_uri())
    assert refusal is not None
    assert (refusal.key, refusal.passed) == (refused["key"], [2, 3, 4, 5])
    assert refusal.refused_at.timestamp() == _NOW


def test_a_refused_key_makes_no_engine_calls_until_an_input_changes(setup, capsys):
    with pytest.raises(CatalogueFloorError):
        _run(setup, _refusing_engine())
    [refused] = _named(_events(capsys), "search_catalogue_job_refused")

    engine = _engine()
    _run(setup, engine, now=_NOW + 900)
    events = _events(capsys)
    assert [event["event"] for event in events] == [
        "search_catalogue_job_still_refused",
        "search_catalogue_job_heartbeat",
    ]
    assert events[0]["severity"] == "WARNING"
    assert events[0]["key"] == refused["key"]
    assert events[0]["passed"] == [2, 3, 4, 5]
    assert engine.queries == []

    _run(setup, engine, commit="c2", now=_NOW + 1800)
    assert _named(_events(capsys), "search_catalogue_job_published")
    assert load_published(setup.out.as_uri()).inputs.build_commit == "c2"


def test_a_refused_key_builds_again_once_the_retry_window_passes(setup, capsys):
    with pytest.raises(CatalogueFloorError):
        _run(setup, _refusing_engine())
    capsys.readouterr()

    engine = _engine()
    _run(setup, engine, now=_NOW + REFUSAL_RETRY_SECONDS)
    assert _named(_events(capsys), "search_catalogue_job_published")
    assert engine.queries


def test_a_malformed_refusal_is_ignored(setup, capsys):
    (setup.out / FORMAT_ROOT).mkdir(parents=True)
    (setup.out / FORMAT_ROOT / REFUSAL_FILE).write_text("{")
    assert read_refusal(setup.out.as_uri()) is None
    _run(setup, _engine())
    assert _named(_events(capsys), "search_catalogue_job_published")


def test_an_unchanged_key_exits_without_building(setup, capsys):
    _run(setup, _engine())
    first = load_published(setup.out.as_uri())
    capsys.readouterr()

    engine = _engine()
    _run(setup, engine)
    events = _events(capsys)
    assert [event["event"] for event in events] == [
        "search_catalogue_job_noop",
        "search_catalogue_job_heartbeat",
    ]
    assert events[0]["catalogue_version"] == first.catalogue_version
    assert engine.queries == []


def test_a_new_build_commit_rebuilds(setup, capsys):
    _run(setup, _engine())
    first = load_published(setup.out.as_uri())
    _run(setup, _engine(), commit="c2")
    second = load_published(setup.out.as_uri())
    assert second.key != first.key
    assert second.inputs.build_commit == "c2"


def test_nothing_to_build_answers_from_the_pointers_alone(setup, capsys):
    inputs = BuildInputs(
        origin_catalogue_build_id=setup.catalogue.build_id,
        coverage_build_id=setup.coverage.release_id,
        build_commit="c1",
        examples_digest=load_examples(EXAMPLES).digest,
    )
    assert not nothing_to_build(setup.settings, inputs=inputs)
    _run(setup, _engine())
    capsys.readouterr()

    assert nothing_to_build(setup.settings, inputs=inputs, clock=lambda: _NOW)
    assert [event["event"] for event in _events(capsys)] == [
        "search_catalogue_job_noop",
        "search_catalogue_job_heartbeat",
    ]
    moved = inputs.model_copy(update={"coverage_build_id": "another-build"})
    assert not nothing_to_build(setup.settings, inputs=moved)


def test_nothing_to_build_answers_a_refused_key_from_the_pointers_alone(setup, capsys):
    inputs = BuildInputs(
        origin_catalogue_build_id=setup.catalogue.build_id,
        coverage_build_id=setup.coverage.release_id,
        build_commit="c1",
        examples_digest=load_examples(EXAMPLES).digest,
    )
    with pytest.raises(CatalogueFloorError):
        _run(setup, _refusing_engine())
    capsys.readouterr()

    assert nothing_to_build(setup.settings, inputs=inputs, clock=lambda: _NOW)
    assert [event["event"] for event in _events(capsys)] == [
        "search_catalogue_job_still_refused",
        "search_catalogue_job_heartbeat",
    ]
    late = _NOW + REFUSAL_RETRY_SECONDS
    assert not nothing_to_build(setup.settings, inputs=inputs, clock=lambda: late)


def _main_with(monkeypatch, setup, *, catalogue_loads: bool = False) -> int:
    """Runs `main` against `setup`'s settings and examples, with no origin catalogue to load
    unless `catalogue_loads`, when the load answers `setup`'s."""

    def unloadable(_settings):
        if catalogue_loads:
            return setup.catalogue
        raise AssertionError("the origin catalogue was loaded")

    monkeypatch.setattr(job, "JobSettings", lambda: setup.settings)
    monkeypatch.setattr(job, "build_commit", lambda: "c1")
    monkeypatch.setattr(job, "load_examples", lambda: load_examples(EXAMPLES))
    monkeypatch.setattr(job, "_load_origin_catalogue", unloadable)
    return job.main([])


def test_the_key_names_the_release_the_pointer_names(setup, monkeypatch, capsys, tmp_path):
    root = (tmp_path / "releases").as_uri()
    pointed = build_fixture_index_release(root).release_id
    assert pointed != current_release().release_id
    monkeypatch.setenv("GRAPH_RELEASE_URI", root)
    _run(setup, _engine(), coverage=replace(setup.coverage, release_id=pointed))
    capsys.readouterr()

    def unopened():
        raise AssertionError("the release was opened")

    # An unchanged run reads the release pointer alone.
    monkeypatch.setattr(release_service, "get_release_service", unopened)
    assert _main_with(monkeypatch, setup) == 0
    assert "search_catalogue_job_noop" in [event["event"] for event in _events(capsys)]

    # A new release changes the key, so the run goes on to load the origin catalogue.
    build_view_release(root)
    with pytest.raises(AssertionError, match="origin catalogue was loaded"):
        _main_with(monkeypatch, setup)


@pytest.mark.parametrize(
    ("published", "reason"),
    [(None, "release_uri_missing"), (False, "no_release")],
    ids=["uri-unset", "nothing-published"],
)
def test_a_run_with_no_release_fails_before_the_catalogue_loads(
    setup, monkeypatch, capsys, tmp_path, published, reason
):
    if published is None:
        monkeypatch.delenv("GRAPH_RELEASE_URI", raising=False)
    else:
        monkeypatch.setenv("GRAPH_RELEASE_URI", (tmp_path / "releases").as_uri())

    assert _main_with(monkeypatch, setup) == 1
    [failed] = _named(_events(capsys), "search_catalogue_job_failed")
    assert (failed["severity"], failed["reason"]) == ("ERROR", reason)


def _raising(error: Exception):
    def raise_(*_args, **_kwargs):
        raise error

    return raise_


@contextmanager
def _no_engine():
    yield _engine(), lambda _dcids: {}


@pytest.mark.parametrize(
    ("opened", "ran", "reason", "error"),
    [
        (_raising(FileNotFoundError("no pointer")), None, "no_release", "no pointer"),
        (_raising(ViewRefused("unbuildable")), None, "views_refused", "unbuildable"),
        (None, _raising(ReleaseNotBound("unbound")), None, "unbound"),
    ],
    ids=["unpublished-at-open", "views-refused", "not-bound"],
)
def test_a_release_failure_is_reported_as_a_failed_job(
    setup, monkeypatch, capsys, tmp_path, opened, ran, reason, error
):
    """A release that fails as it opens, or a read of it during the run, ends in the job's
    failure event like any other failure. `opened` replaces `get_release_service` when set;
    unset, the run binds the process's served release."""
    root = (tmp_path / "releases").as_uri()
    build_fixture_index_release(root)
    monkeypatch.setenv("GRAPH_RELEASE_URI", root)
    if opened is not None:
        monkeypatch.setattr(release_service, "get_release_service", opened)
    monkeypatch.setattr(job, "_live_engine", _no_engine)
    if ran is not None:
        monkeypatch.setattr(job, "run", ran)

    assert _main_with(monkeypatch, setup, catalogue_loads=True) == 1
    [failed] = _named(_events(capsys), "search_catalogue_job_failed")
    assert (failed["severity"], failed.get("reason"), failed["error"]) == ("ERROR", reason, error)


def _hold_lease(setup: Setup, *, ttl_seconds: float) -> None:
    root = (setup.out / FORMAT_ROOT).as_uri()
    assert acquire_lease(root, build_id="holder", ttl_seconds=ttl_seconds, now=lambda: _NOW)


def _inputs(setup: Setup, *, commit: str = "c1", release_id: str | None = None) -> BuildInputs:
    return BuildInputs(
        origin_catalogue_build_id=setup.catalogue.build_id,
        coverage_build_id=release_id or setup.coverage.release_id,
        build_commit=commit,
        examples_digest=load_examples(EXAMPLES).digest,
    )


def _holder_publishes(
    setup: Setup, inputs: BuildInputs, *, started: float = _NOW - 60
) -> Callable[[], None]:
    """On its first call, publishes a build of `inputs` by a run that started at `started`
    and frees the lease, as a holder that finishes does."""
    done: list[bool] = []

    def finish() -> None:
        if done:
            return
        done.append(True)
        artifact = sample_catalogue().model_copy(
            update={
                "key": build_key(inputs),
                "inputs": inputs,
                "started_at": datetime.fromtimestamp(started, UTC),
            }
        )
        publish_catalogue(setup.out.as_uri(), artifact)
        (setup.out / FORMAT_ROOT / LEASE_FILE).unlink()

    return finish


def _main_building(
    monkeypatch,
    setup: Setup,
    *,
    released: list[str] | None = None,
    on_sleep: Callable[[], None] = lambda: None,
    build_seconds: float = 0.0,
) -> tuple[int, float, FakeEngine]:
    """Runs `main` from `_NOW` through a build with the fake engine and commit `c1`.
    `released` holds the release ids the release pointer named, the last current. A sleep
    moves the clock on by its length and then calls `on_sleep`; the first check run moves it
    on by `build_seconds`. Returns the exit code, the clock at the end and the engine."""
    released = released if released is not None else [setup.coverage.release_id]
    clock = [_NOW]
    engine = _engine()

    def sleep(seconds: float) -> None:
        clock[0] += seconds
        on_sleep()

    @dataclass
    class _Service:
        def snapshot(self):
            return current_release()

    async def resolve(request):
        if not engine.queries:
            clock[0] += build_seconds
        return await engine(request)

    @contextmanager
    def live_engine():
        yield resolve, lambda dcids: {dcid: NAMES[dcid] for dcid in dcids if dcid in NAMES}

    monkeypatch.setattr(job, "JobSettings", lambda: setup.settings)
    monkeypatch.setattr(job, "build_commit", lambda: "c1")
    monkeypatch.setattr(job, "load_examples", lambda: load_examples(EXAMPLES))
    monkeypatch.setattr(job, "_load_origin_catalogue", lambda _settings: setup.catalogue)
    monkeypatch.setattr(job, "_pointed_release_id", lambda: released[-1])
    monkeypatch.setattr(job, "_release_service", _Service)
    monkeypatch.setattr(
        job, "get_coverage_service", lambda: replace(setup.coverage, release_id=released[-1])
    )
    monkeypatch.setattr(job, "_live_engine", live_engine)
    code = job.main([], clock=lambda: clock[0], sleep=sleep)
    return code, clock[0], engine


def test_a_held_lease_is_waited_for_until_the_bound_without_building(setup, monkeypatch, capsys):
    _hold_lease(setup, ttl_seconds=2 * LEASE_WAIT_SECONDS)
    code, ended, engine = _main_building(monkeypatch, setup)
    events = _events(capsys)
    assert code == 0
    assert [event["event"] for event in events] == [
        "search_catalogue_job_waiting",
        "search_catalogue_job_skipped",
        "search_catalogue_job_heartbeat",
    ]
    assert (events[1]["reason"], events[1]["severity"]) == ("lease_wait_exhausted", "WARNING")
    assert ended - _NOW == LEASE_WAIT_SECONDS
    assert engine.queries == []


def test_the_published_event_carries_the_seconds_the_lease_was_held(setup, monkeypatch, capsys):
    """Loading and the build both run under the lease; `leased_seconds` covers the span from
    taking it to publishing."""
    code, _, _ = _main_building(monkeypatch, setup, build_seconds=95)
    assert code == 0
    [published] = _named(_events(capsys), "search_catalogue_job_published")
    assert published["leased_seconds"] == 95


def test_a_lapsed_lease_is_taken_over_and_the_run_builds(setup, monkeypatch, capsys):
    _hold_lease(setup, ttl_seconds=60)
    code, ended, _ = _main_building(monkeypatch, setup)
    assert code == 0
    assert _named(_events(capsys), "search_catalogue_job_published")
    assert ended - _NOW < LEASE_WAIT_SECONDS
    assert load_published(setup.out.as_uri()).key == build_key(_inputs(setup))
    assert not (setup.out / FORMAT_ROOT / LEASE_FILE).exists()


@pytest.mark.parametrize(
    "build_seconds", [LEASE_TTL_SECONDS, LEASE_TTL_SECONDS - 20], ids=["lapsed", "within-margin"]
)
def test_a_build_past_its_lease_publishes_nothing(setup, monkeypatch, capsys, build_seconds):
    """A build that overran its lease, or has too little of it left for the pointer write to
    land before another run can reclaim it, publishes nothing."""
    code, _, engine = _main_building(monkeypatch, setup, build_seconds=build_seconds)
    events = _events(capsys)
    assert code == 0
    assert engine.queries
    [skipped] = _named(events, "search_catalogue_job_skipped")
    assert (skipped["reason"], skipped["severity"]) == ("lease_lost", "WARNING")
    assert skipped["leased_seconds"] == build_seconds
    assert not _named(events, "search_catalogue_job_published")
    assert not (setup.out / FORMAT_ROOT / "current.json").exists()
    assert not (setup.out / FORMAT_ROOT / LEASE_FILE).exists()


@pytest.mark.parametrize(
    ("holder", "moves"),
    [({"release_id": "release-older"}, False), ({"commit": "c0"}, False), ({}, True)],
    ids=["older-release", "previous-image", "pointer-moved-during-the-wait"],
)
def test_a_run_started_during_a_build_publishes_its_inputs_once_the_lease_frees(
    setup, monkeypatch, capsys, holder, moves
):
    """The holder started before the waiting run. A pointer that moves while the run waits
    gives the run its newest inputs, since the run loads only once it holds the lease."""
    released = [setup.coverage.release_id]
    finish = _holder_publishes(setup, _inputs(setup, **holder))

    def on_sleep() -> None:
        if moves and len(released) == 1:
            released.append("release-1")
        finish()

    _hold_lease(setup, ttl_seconds=LEASE_WAIT_SECONDS)
    code, _, _ = _main_building(monkeypatch, setup, released=released, on_sleep=on_sleep)
    events = _events(capsys)
    assert code == 0
    assert not _named(events, "search_catalogue_job_skipped")
    expected = build_key(_inputs(setup, release_id=released[-1]))
    [published] = _named(events, "search_catalogue_job_published")
    assert published["key"] == expected
    assert load_published(setup.out.as_uri()).key == expected
    assert events[-1]["event"] == "search_catalogue_job_heartbeat"


@pytest.mark.parametrize(
    ("holder", "started", "reason"),
    [
        ({}, _NOW - 60, "settled_while_waiting"),
        ({"commit": "c0"}, _NOW + 5, "superseded"),
        ({"release_id": "release-older"}, _NOW + 5, "superseded"),
    ],
    ids=["holder-built-these-inputs", "later-start-other-commit", "later-start-older-release"],
)
def test_a_run_that_waited_builds_nothing_its_holder_settled(
    setup, monkeypatch, capsys, holder, started, reason
):
    built = _inputs(setup, **holder)
    _hold_lease(setup, ttl_seconds=LEASE_WAIT_SECONDS)
    code, _, engine = _main_building(
        monkeypatch, setup, on_sleep=_holder_publishes(setup, built, started=started)
    )
    events = _events(capsys)
    assert code == 0
    [skipped] = _named(events, "search_catalogue_job_skipped")
    assert skipped["reason"] == reason
    assert engine.queries == []
    assert load_published(setup.out.as_uri()).key == build_key(built)
    assert events[-1]["event"] == "search_catalogue_job_heartbeat"
