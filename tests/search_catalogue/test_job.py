"""Pins the search catalogue job with a fake engine over the origin catalogue
sample build and the coverage fixture."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

from artifact_publication.jobs import acquire_lease
from search.catalogue import job
from search.catalogue.artifact import (
    FORMAT_ROOT,
    REFUSAL_FILE,
    BuildInputs,
    load_published,
    read_refusal,
)
from search.catalogue.check import ROLE_KEYS
from search.catalogue.examples import load_examples
from search.catalogue.job import (
    FLOOR,
    REFUSAL_RETRY_SECONDS,
    CatalogueEngineError,
    CatalogueFloorError,
    JobSettings,
    nothing_to_build,
    run,
)
from search.engine.errors import EngineInputError
from series_coverage.shared import (
    CoverageIncompatibleError,
    CoverageIndexCheckValueError,
    CoverageService,
    CoverageSnapshot,
)
from tests.origin_catalogue.support import CENSUS_POVERTY, CRS_VARIABLE, GDP, POPULATION
from tests.search_catalogue.support import (
    Elected,
    FakeEngine,
    definite,
    omitted,
    origin_snapshot,
    publish_catalogue,
    timed_out,
)
from tests.series_coverage.support import V1, V2, V3, build_test_artifact, index_for

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


@pytest.fixture
def setup(tmp_path) -> Setup:
    out = tmp_path / "out"
    coverage_uri = build_test_artifact(tmp_path / "coverage")
    return Setup(
        settings=JobSettings(
            artifact_uri=out.as_uri(),
            origin_catalogue_uri=(tmp_path / "origin").as_uri(),
            coverage_uri=coverage_uri,
        ),
        catalogue=origin_snapshot(tmp_path / "origin"),
        coverage=CoverageService(
            artifact_uri=coverage_uri, index=index_for(coverage_uri), recheck_seconds=0
        ),
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
    assert acquire_lease((setup.out / FORMAT_ROOT).as_uri(), build_id="next", ttl_seconds=60)


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


def test_a_held_lease_skips_the_run_without_engine_calls(setup, capsys):
    root = (setup.out / FORMAT_ROOT).as_uri()
    assert acquire_lease(root, build_id="other", ttl_seconds=60, now=lambda: _NOW)
    engine = _engine()
    _run(setup, engine)
    events = _events(capsys)
    assert [event["event"] for event in events] == [
        "search_catalogue_job_skipped",
        "search_catalogue_job_heartbeat",
    ]
    assert events[0]["reason"] == "lease_held"
    assert engine.queries == []
    assert not (setup.out / FORMAT_ROOT / "current.json").exists()


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


def test_under_the_floor_the_run_publishes_nothing_and_frees_the_lease(setup, capsys):
    with pytest.raises(CatalogueFloorError):
        _run(setup, _refusing_engine())
    events = _events(capsys)
    [refused] = _named(events, "search_catalogue_job_refused")
    assert refused["severity"] == "ERROR"
    assert refused["passed"] == [2, 3, 4, 5]
    assert refused["floor"] == FLOOR
    assert not _named(events, "search_catalogue_job_heartbeat")
    assert not (setup.out / FORMAT_ROOT / "current.json").exists()
    assert acquire_lease((setup.out / FORMAT_ROOT).as_uri(), build_id="next", ttl_seconds=60)
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


@dataclass
class _MovingCoverage:
    """Serves the fixture's snapshot, then one naming another build."""

    snapshots: list[CoverageSnapshot]

    def snapshot(self) -> CoverageSnapshot:
        return self.snapshots.pop(0) if len(self.snapshots) > 1 else self.snapshots[0]


def test_a_build_is_discarded_when_the_coverage_index_moves(setup, capsys):
    loaded = setup.coverage.snapshot()
    moved = replace(loaded, manifest=replace(loaded.manifest, build_id="another-build"))
    _run(setup, _engine(), coverage=_MovingCoverage([loaded, moved]))
    [discarded] = _named(_events(capsys), "search_catalogue_job_discarded")
    assert discarded["reason"] == "coverage_moved_during_build"
    assert not (setup.out / FORMAT_ROOT / "current.json").exists()


def test_nothing_to_build_answers_from_the_pointers_alone(setup, capsys):
    loaded = setup.coverage.snapshot()
    inputs = BuildInputs(
        origin_catalogue_build_id=setup.catalogue.build_id,
        coverage_build_id=loaded.manifest.build_id,
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
    loaded = setup.coverage.snapshot()
    inputs = BuildInputs(
        origin_catalogue_build_id=setup.catalogue.build_id,
        coverage_build_id=loaded.manifest.build_id,
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


def _main_with(monkeypatch, setup, *, coverage) -> int:
    """Runs `main` against `setup`'s settings and examples, with `coverage` as
    the process-wide coverage service and no origin catalogue to load."""

    def unloadable(_settings):
        raise AssertionError("the origin catalogue was loaded")

    monkeypatch.setattr(job, "JobSettings", lambda: setup.settings)
    monkeypatch.setattr(job, "build_commit", lambda: "c1")
    monkeypatch.setattr(job, "load_examples", lambda: load_examples(EXAMPLES))
    monkeypatch.setattr(job, "get_coverage_service", coverage)
    monkeypatch.setattr(job, "_load_origin_catalogue", unloadable)
    return job.main([])


def test_a_run_while_a_fallback_build_is_served_settles_as_nothing_to_build(
    setup, monkeypatch, capsys
):
    _run(setup, _engine())
    served = setup.coverage.manifest.build_id
    # `current.json` now names a build the service rejected and does not serve.
    build_test_artifact(Path(urlparse(setup.settings.coverage_uri).path), build_id="rejected-build")
    assert setup.coverage.manifest.build_id == served
    capsys.readouterr()

    assert _main_with(monkeypatch, setup, coverage=lambda: setup.coverage) == 0
    assert "search_catalogue_job_noop" in [event["event"] for event in _events(capsys)]


@pytest.mark.parametrize(
    "error",
    [
        CoverageIncompatibleError("no build matches the index"),
        CoverageIndexCheckValueError("COVERAGE_INDEX_CHECK must be off"),
    ],
)
def test_a_coverage_service_that_refuses_is_reported_as_a_failed_job(
    setup, monkeypatch, capsys, error
):
    def refuse():
        raise error

    assert _main_with(monkeypatch, setup, coverage=refuse) == 1
    [failed] = _named(_events(capsys), "search_catalogue_job_failed")
    assert failed["severity"] == "ERROR"
    assert failed["error"] == str(error)
