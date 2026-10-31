"""`warning_backfill` rewrites a stored member sentence from a probe of the members holding no
observation, against a fake graph and a store that runs the task's two statements in Python."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from code_index import CodeIndex, IndexStamp
from code_index.types import DatasetDeclaration
from datasets_api import maintenance
from datasets_api.warning_backfill import Readers, run_backfill_warnings, select_rows
from dc_sdmx import SdmxHttpError
from materialise.graph import Facet, GraphReadError
from origin_catalogue import record_un_read_failure
from tests.materialise._scope_availability import ScopeAvailability
from tests.materialise.conftest import FakeGraphReader

_KEN = "country/KEN"
_PROVENANCE = "PROV1"
_DETAIL = "The data covers less than the definition asked for."
_SERVED, _IN, _OUT, _GONE, _MOVED, _UN = (
    "Var_Served",
    "Var_In",
    "Var_Out",
    "Var_Gone",
    "Var_Moved",
    "undata/Var_Un",
)
_NOW = datetime(2026, 10, 31, tzinfo=UTC)


def _facet(facet_id: str, year: str, *, provenance: str = _PROVENANCE) -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id=provenance,
        measurement_method="M",
        unit="USD",
        obs_count=1,
        earliest_date=year,
        latest_date=year,
        dates=(year,),
        values=(1.0,),
    )


# Var_In holds the selected source in the requested years, Var_Out only before them, Var_Moved
# only under another source, and the graph no longer holds Var_Gone.
_FACETS = {
    (_SERVED, _KEN): [_facet("f-served", "2015")],
    (_IN, _KEN): [_facet("f-in", "2015")],
    (_OUT, _KEN): [_facet("f-out", "1990")],
    (_MOVED, _KEN): [_facet("f-moved", "2015", provenance="PROV2")],
}


def _code_index(*, absent_reads_zero: bool = False) -> CodeIndex:
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="https://fixture.example.org",
            exported_at="2026-10-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="sha256:fixture",
        ),
        declarations=(),
        codes=(),
        statvars=(),
        datasets=(
            DatasetDeclaration(
                provenance=_PROVENANCE,
                label="Fixture table",
                description="Amounts by place.",
                opt_in=False,
                rank=0,
                absent_observation="zero" if absent_reads_zero else "unknown",
                account="provider",
            ),
        ),
    )


def _identity(*, places: Sequence[str] = (_KEN,)) -> dict[str, object]:
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "DevelopmentFinance",
            "measured_property": "amount",
        },
        "slots": [],
        "entities": {"observationAbout": list(places)} if places else {},
        "requested_window": {"kind": "range", "start_year": 2010, "end_year": 2020},
    }


def _trace(zero: Sequence[str]) -> dict[str, object]:
    members = sorted([_SERVED, *zero])
    return {
        "resolved_stat_vars": [
            {
                "ref": f"dcid:{member}",
                "roles": ["observationAbout"],
                "observation_count": 0 if member in zero else 3,
            }
            for member in members
        ],
        "facets": [
            {
                "axis": "provenance",
                "ref": _PROVENANCE,
                "basis": "deployment_fallback",
                "default_policy": None,
                "default_policy_version": None,
                "selection_reason": "most observations",
                "dispatch_facet_id": "f-served",
            }
        ],
    }


def _old_sentence(named: Sequence[str], more: int = 0) -> str:
    names = ", ".join(f"dcid:{member}" for member in named)
    rest = f" and {more} more" if more else ""
    return (
        f" No observations were served for {names}{rest}, which hold none at the observation "
        "period the other members serve in the requested years."
    )


def _warning(named: Sequence[str], more: int = 0) -> dict[str, object]:
    return {
        "code": "coverage_below_scope",
        "severity": "warn",
        "message": f"{_DETAIL}{_old_sentence(named, more)} No place was named.",
    }


class _Store:
    """Version rows in memory. `fetch_all` answers the task's selection and `fetch_one` its
    update, holding the row to the column's and the body's warnings where the statement names
    each, and writing the body's where it sets it."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = {(row["subject_id"], row["content_version"]): dict(row) for row in rows}
        self.updates = 0

    @contextmanager
    def transaction(self) -> Iterator[_Store]:
        yield self

    def fetch_all(self, sql: str, params: Sequence[object] = ()) -> list[Mapping[str, object]]:
        code, pattern, control = params
        assert "jsonb_array_elements(warnings)" in sql, sql

        def matches(warning: Mapping[str, Any]) -> bool:
            if warning["code"] != code:
                return False
            if re.search(str(pattern), warning["message"]):
                return True
            return bool(control) and (
                "off_cadence_members" in warning or "out_of_window_members" in warning
            )

        return [
            {
                **row,
                "warnings": json.loads(json.dumps(row["warnings"])),
                "body_warnings": json.loads(json.dumps(row["immutable_body"].get("warnings"))),
            }
            for row in self.rows.values()
            if any(matches(warning) for warning in row["warnings"])
        ]

    def fetch_one(self, sql: str, params: Sequence[object] = ()) -> Mapping[str, object] | None:
        assert sql.startswith("UPDATE datasets.version_record SET warnings"), sql
        written, written_body, subject_id, content_version, read, read_body = params
        row = self.rows[(subject_id, content_version)]
        body = row["immutable_body"]
        if "AND warnings = %s" in sql and json.loads(str(read)) != row["warnings"]:
            return None
        if "AND immutable_body -> 'warnings' = %s" in sql and json.loads(str(read_body)) != (
            body.get("warnings")
        ):
            return None
        row["warnings"] = json.loads(str(written))
        if "immutable_body = jsonb_set(immutable_body, '{warnings}', %s::jsonb)" in sql:
            row["immutable_body"] = {**body, "warnings": json.loads(str(written_body))}
        self.updates += 1
        return {"subject_id": subject_id}


def _row(
    warning: Mapping[str, object],
    *,
    zero: Sequence[str],
    places: Sequence[str] = (_KEN,),
    subject_id: str = "spec_a",
    content_version: str = "cv_a",
) -> dict[str, Any]:
    return {
        "subject_id": subject_id,
        "content_version": content_version,
        "definition": {
            "schema_version": 1,
            "spec_id": subject_id,
            "identity": _identity(places=places),
        },
        "trace": _trace(zero),
        "warnings": [warning],
        "immutable_body": {"prepared_at": "2026-10-01T00:00:00Z", "warnings": [warning]},
    }


def _run(
    store: _Store,
    *,
    graph: FakeGraphReader | None = None,
    availability: ScopeAvailability | None = None,
    absent_reads_zero: bool = False,
    dry_run: bool = False,
    readers: Readers | None = None,
) -> int:
    readers = readers or Readers(
        graph=graph or FakeGraphReader(facets=_FACETS),
        availability=availability or ScopeAvailability(data=()),
        code_index=_code_index(absent_reads_zero=absent_reads_zero),
        coverage=None,
    )
    return run_backfill_warnings(
        select_rows(store, control=dry_run),
        store=store,
        readers=readers,
        dry_run=dry_run,
        deadline=_NOW + timedelta(minutes=45),
        now=lambda: _NOW,
    )


def _lines(capsys: pytest.CaptureFixture[str], event: str) -> list[dict[str, Any]]:
    return [
        line
        for line in map(json.loads, capsys.readouterr().out.splitlines())
        if line["event"] == event
    ]


def _stored(store: _Store) -> dict[str, Any]:
    """The row's one warning, which its body holds too."""
    (row,) = store.rows.values()
    (warning,) = row["warnings"]
    assert row["immutable_body"]["warnings"] == [warning]
    return warning


@pytest.mark.parametrize(
    ("member", "field", "sentence"),
    [
        pytest.param(
            _IN,
            "off_cadence_members",
            "1 of 5 series report at a different frequency and are left out of the figures.",
            id="in-the-window",
        ),
        pytest.param(
            _OUT,
            "out_of_window_members",
            "1 of 5 series hold no observations in the requested years.",
            id="out-of-the-window",
        ),
        pytest.param(
            _GONE,
            "out_of_window_members",
            "1 of 5 series hold no observations in the requested years.",
            id="gone",
        ),
        pytest.param(
            _MOVED,
            "unserved_members",
            "1 of 5 series hold no observations under the source, unit and method the figures "
            "use and are left out.",
            id="moved",
        ),
    ],
)
def test_a_named_member_is_split_as_a_mint_splits_it(member, field, sentence, capsys):
    store = _Store([_row(_warning([member]), zero=[_IN, _OUT, _GONE, _MOVED])])

    assert _run(store) == 0

    assert _stored(store) == {
        "code": "coverage_below_scope",
        "severity": "warn",
        "message": f"{_DETAIL} {sentence} No place was named.",
        "member_count": 5,
        field: {"count": 1, "refs": [f"dcid:{member}"]},
    }
    (summary,) = _lines(capsys, "datasets_warnings_backfilled")
    assert (summary["rewritten"], summary["gone"]) == (1, int(member == _GONE))


def test_the_count_past_the_named_members_is_filled_matched_members_first():
    # One named and one more: a member with a facet of the selected source fills the place
    # before Var_Gone and Var_Moved, which come first in the trace and hold none.
    store = _Store([_row(_warning([_OUT], more=1), zero=[_GONE, _IN, _MOVED, _OUT])])

    assert _run(store) == 0

    stored = _stored(store)
    assert stored["off_cadence_members"] == {"count": 1, "refs": ["dcid:Var_In"]}
    assert stored["out_of_window_members"] == {"count": 1, "refs": ["dcid:Var_Out"]}
    assert "unserved_members" not in stored


def test_a_provenance_reading_absence_as_zero_drops_the_out_of_window_members(capsys):
    store = _Store([_row(_warning([_OUT, _GONE]), zero=[_OUT, _GONE])])

    assert _run(store, absent_reads_zero=True) == 0

    assert _stored(store) == {
        "code": "coverage_below_scope",
        "severity": "warn",
        "message": f"{_DETAIL} No place was named.",
    }
    (summary,) = _lines(capsys, "datasets_warnings_backfilled")
    assert (summary["emptied"], summary["gone"]) == (1, 1)


class _FailingGraph(FakeGraphReader):
    def observation_facets_batch_many(self, **kwargs):
        raise GraphReadError("graph read failed")


class _UnFailingBesideOursGraph(FakeGraphReader):
    """A batch naming a UN variable beside ours whose UN part fails, which the origin client
    records and serves as empty."""

    def observation_facets_batch_many(self, *, stat_vars, entities, **kwargs):
        un = [stat_var for stat_var in stat_vars if stat_var.startswith("undata/")]
        assert record_un_read_failure(
            GraphReadError("UN read failed"), variables=un, places=entities, alongside_ours=True
        )
        ours = [stat_var for stat_var in stat_vars if stat_var not in un]
        return super().observation_facets_batch_many(stat_vars=ours, entities=entities, **kwargs)


class _UnsettledAvailability(ScopeAvailability):
    """Availability answering a gateway status, so a member binding no place is scoped at the
    countries with data, unsettled."""

    def availability_values(self, **kwargs):
        raise SdmxHttpError("bad gateway", upstream_status=502)


@pytest.mark.parametrize(
    ("named", "zero", "places", "graph", "availability"),
    [
        pytest.param([_IN], [_IN], (_KEN,), _FailingGraph(facets=_FACETS), None, id="failed-read"),
        pytest.param(
            [_UN],
            [_IN, _UN],
            (_KEN,),
            _UnFailingBesideOursGraph(facets=_FACETS),
            None,
            id="un-failure-beside-ours",
        ),
        pytest.param(
            [_IN],
            [_IN],
            (),
            FakeGraphReader(facets=_FACETS, child_places={("Earth", "Country"): [_KEN]}),
            _UnsettledAvailability(data=[(_IN, _KEN)]),
            id="unsettled-scope",
        ),
    ],
)
def test_an_incomplete_read_leaves_the_row_for_the_next_run(
    named, zero, places, graph, availability, capsys
):
    warning = _warning(named)
    store = _Store([_row(warning, zero=zero, places=places)])

    assert _run(store, graph=graph, availability=availability) == 1

    assert _stored(store) == warning
    (failed,) = _lines(capsys, "datasets_warning_backfill_failed")
    assert failed["transient"] is True


def test_more_members_named_than_hold_no_observation_is_refused_for_good(capsys):
    warning = _warning([_IN], more=3)
    store = _Store([_row(warning, zero=[_IN, _OUT])])

    assert _run(store) == 1

    assert _stored(store) == warning
    (failed,) = _lines(capsys, "datasets_warning_backfill_failed")
    assert failed["transient"] is False


def test_a_second_run_finds_no_row():
    store = _Store([_row(_warning([_IN, _OUT]), zero=[_IN, _OUT])])
    assert _run(store) == 0
    rewritten = _stored(store)

    assert select_rows(store, control=False) == []
    assert _run(store) == 0
    assert (_stored(store), store.updates) == (rewritten, 1)


@pytest.mark.parametrize("where", ["warnings", "immutable_body"])
def test_a_row_changed_after_it_was_read_keeps_the_change(where, capsys):
    store = _Store([_row(_warning([_IN]), zero=[_IN])])
    rows = select_rows(store, control=False)
    changed = {**_warning([_IN]), "message": "changed meanwhile"}
    row = store.rows[("spec_a", "cv_a")]
    before = json.loads(json.dumps(row))
    if where == "warnings":
        row["warnings"] = [changed]
    else:
        row["immutable_body"] = {**row["immutable_body"], "warnings": [changed]}
    after = json.loads(json.dumps(row))

    unfinished = run_backfill_warnings(
        rows,
        store=store,
        readers=Readers(
            graph=FakeGraphReader(facets=_FACETS),
            availability=ScopeAvailability(data=()),
            code_index=_code_index(),
            coverage=None,
        ),
        dry_run=False,
        deadline=_NOW + timedelta(minutes=45),
        now=lambda: _NOW,
    )

    assert unfinished == 1
    assert store.rows[("spec_a", "cv_a")] == after != before
    (summary,) = _lines(capsys, "datasets_warnings_backfilled")
    assert summary["conflicts"] == 1


def test_a_row_whose_body_and_column_disagree_is_refused_for_good(capsys):
    warning = _warning([_IN])
    row = _row(warning, zero=[_IN])
    row["immutable_body"] = {**row["immutable_body"], "warnings": [{**warning, "message": "x"}]}
    store = _Store([row])
    before = json.loads(json.dumps(row))

    assert _run(store) == 1

    assert store.rows[("spec_a", "cv_a")] == before
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    (failed,) = [line for line in lines if line["event"] == "datasets_warning_backfill_failed"]
    assert (failed["transient"], failed["severity"]) == (False, "ERROR")
    (summary,) = [line for line in lines if line["event"] == "datasets_warnings_backfilled"]
    assert (summary["permanent"], summary["mismatched"]) == (1, 1)


def test_a_control_row_that_fails_warns_and_leaves_the_exit_code(capsys):
    # The stored split names a member the trace says served observations.
    split = {
        "code": "coverage_below_scope",
        "severity": "warn",
        "message": f"{_DETAIL} 1 of 2 series hold no observations in the requested years.",
        "member_count": 2,
        "out_of_window_members": {"count": 1, "refs": ["dcid:Var_Served"]},
    }
    store = _Store([_row(split, zero=[_OUT], subject_id="spec_b", content_version="cv_b")])

    assert _run(store, dry_run=True) == 0

    (failed,) = _lines(capsys, "datasets_warning_backfill_failed")
    assert (failed["severity"], failed["subject_id"], failed["content_version"]) == (
        "WARNING",
        "spec_b",
        "cv_b",
    )


def test_a_dry_run_writes_nothing_and_compares_a_mint_split(capsys):
    old = _warning([_IN])
    split = {
        "code": "coverage_below_scope",
        "severity": "warn",
        "message": f"{_DETAIL} 1 of 2 series hold no observations in the requested years.",
        "member_count": 2,
        "out_of_window_members": {"count": 1, "refs": ["dcid:Var_Out"]},
    }
    store = _Store(
        [
            _row(old, zero=[_IN]),
            _row(split, zero=[_OUT], subject_id="spec_b", content_version="cv_b"),
        ]
    )
    graph = FakeGraphReader(facets=_FACETS)

    assert _run(store, graph=graph, dry_run=True) == 0

    assert store.updates == 0
    rows = {line["subject_id"]: line for line in _lines(capsys, "datasets_warning_backfill_row")}
    assert (rows["spec_a"]["off_cadence"], rows["spec_a"]["left_out"]) == (1, 1)
    assert rows["spec_b"]["agrees"] is True


def test_the_versions_of_one_subject_share_one_probe(capsys):
    store = _Store(
        [
            _row(_warning([_IN]), zero=[_IN], content_version="cv_a"),
            _row(_warning([_IN]), zero=[_IN], content_version="cv_b"),
        ]
    )
    graph = FakeGraphReader(facets=_FACETS)

    assert _run(store, graph=graph) == 0

    assert graph.calls["observation_facets_batch_many"] == 1
    (summary,) = _lines(capsys, "datasets_warnings_backfilled")
    assert (summary["rewritten"], summary["probes"]) == (2, 1)


def test_the_task_with_no_row_opens_no_release(monkeypatch, capsys):
    monkeypatch.setattr(maintenance, "shared_pool", lambda: None)
    monkeypatch.setattr(maintenance, "close_shared_pool", lambda: None)
    monkeypatch.setattr(maintenance, "select_rows", lambda store, *, control: [])

    def no_release():
        raise AssertionError("the release was opened")

    monkeypatch.setattr(maintenance, "get_release_service", no_release)

    assert maintenance.main(["backfill-warnings", "--dry-run"]) == 0
    (summary,) = _lines(capsys, "datasets_warnings_backfilled")
    assert (summary["dry_run"], summary["rewritten"]) == (True, 0)
