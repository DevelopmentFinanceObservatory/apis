"""`warning_backfill` rewrites a stored member sentence from a probe of the members holding no
observation, against a fake graph and a store that runs the task's two statements in Python."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from code_index import CodeIndex, IndexStamp
from code_index.types import DatasetDeclaration, StatVarRow
from dataset_store.errors import PoolTimeoutError
from datasets_api import maintenance
from datasets_api.warning_backfill import (
    Readers,
    WarningRow,
    run_backfill_warnings,
    select_rows,
)
from dc_graph import GraphNotAdvancingError, GraphOmittedDcidsError
from dc_sdmx import SdmxHttpError, SdmxUnsupportedFilterError
from materialise import facets as facets_module
from materialise.codes import MATERIALISE_CODE_CATALOGUE
from materialise.finish import COUNTRIES_FALLBACK_SENTENCE, member_sentences
from materialise.graph import Facet, GraphReadError
from origin_catalogue import record_un_read_failure
from tests.materialise._scope_availability import ScopeAvailability
from tests.materialise.conftest import FakeGraphReader

_KEN = "country/KEN"
_PROVENANCE = "PROV1"
_DETAIL = MATERIALISE_CODE_CATALOGUE["coverage_below_scope"][1]
_SERVED, _IN, _OUT, _GONE, _MOVED, _UN = (
    "Var_Served",
    "Var_In",
    "Var_Out",
    "Var_Gone",
    "Var_Moved",
    "undata/Var_Un",
)
_NOW = datetime(2026, 10, 31, tzinfo=UTC)
_RECORDED_AT = datetime(2025, 6, 1, tzinfo=UTC)


def _facet(facet_id: str, year: str, *, provenance: str = _PROVENANCE, method: str = "M") -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id=provenance,
        measurement_method=method,
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


def _code_index(*, absent_reads_zero: bool = False, indexed: bool = False) -> CodeIndex:
    """The fixture table's index. `indexed` holds a row for each member, recorded under both
    sources, so the index resolves the definition and a mint elects the fixture table."""
    rows = tuple(
        StatVarRow(
            dcid=member,
            population_type="DevelopmentFinance",
            measured_property="amount",
            stat_type=None,
            measurement_qualifier=None,
            measurement_denominator=None,
            observation_properties=(),
            constraint_properties={},
            provenances=(_PROVENANCE, "PROV2"),
        )
        for member in (_SERVED, _IN, _OUT, _GONE, _MOVED)
    )
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
        statvars=rows if indexed else (),
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


def _identity(*, places: Sequence[str] = (_KEN,), end_year: int | None = 2020) -> dict[str, object]:
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "DevelopmentFinance",
            "measured_property": "amount",
        },
        "slots": [],
        "entities": {"observationAbout": list(places)} if places else {},
        "requested_window": {"kind": "range", "start_year": 2010}
        | ({"end_year": end_year} if end_year is not None else {}),
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
        code, pattern, new_format = params
        assert "jsonb_array_elements(warnings)" in sql, sql

        def matches(warning: Mapping[str, Any]) -> bool:
            if warning["code"] != code:
                return False
            if re.search(str(pattern), warning["message"]):
                return True
            return bool(new_format) and (
                "off_cadence_members" in warning or "out_of_window_members" in warning
            )

        assert "ORDER BY subject_id, recorded_at, content_version" in sql, sql
        return [
            {
                **row,
                "warnings": json.loads(json.dumps(row["warnings"])),
                "body_warnings": json.loads(json.dumps(row["immutable_body"].get("warnings"))),
            }
            for row in sorted(
                self.rows.values(),
                key=lambda row: (row["subject_id"], row["recorded_at"], row["content_version"]),
            )
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
    end_year: int | None = 2020,
) -> dict[str, Any]:
    return {
        "subject_id": subject_id,
        "content_version": content_version,
        "recorded_at": _RECORDED_AT,
        "definition": {
            "schema_version": 1,
            "spec_id": subject_id,
            "identity": _identity(places=places, end_year=end_year),
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
    indexed: bool = False,
    dry_run: bool = False,
    rows: Sequence[WarningRow] | None = None,
    readers: Readers | None = None,
) -> int:
    return run_backfill_warnings(
        select_rows(store, new_format=dry_run) if rows is None else rows,
        store=store,
        readers=readers
        or Readers(
            graph=graph or FakeGraphReader(facets=_FACETS),
            availability=availability or ScopeAvailability(data=()),
            code_index=_code_index(absent_reads_zero=absent_reads_zero, indexed=indexed),
            coverage=None,
        ),
        dry_run=dry_run,
        deadline=_NOW + timedelta(minutes=45),
        now=lambda: _NOW,
        clears_on_retry=maintenance._clears_on_retry,
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
    ("member", "kind"),
    [
        pytest.param(_IN, "off_cadence", id="in-the-window"),
        pytest.param(_OUT, "out_of_window", id="out-of-the-window"),
        pytest.param(_GONE, "out_of_window", id="gone"),
        pytest.param(_MOVED, "unserved_members", id="moved"),
    ],
)
def test_a_named_member_is_split_as_a_mint_splits_it(member, kind, capsys):
    store = _Store([_row(_warning([member]), zero=sorted({member, _IN, _OUT, _GONE}))])
    member_count = 5 if member == _MOVED else 4
    kinds = {"off_cadence": (), "out_of_window": (), "unserved_members": ()}
    (sentence,) = member_sentences(
        **{**kinds, kind: (f"dcid:{member}",)}, member_count=member_count
    )

    assert _run(store) == 0

    field = kind if kind == "unserved_members" else f"{kind}_members"
    assert _stored(store) == {
        "code": "coverage_below_scope",
        "severity": "warn",
        "message": f"{_DETAIL} {sentence} No place was named.",
        "member_count": member_count,
        field: {"count": 1, "refs": [f"dcid:{member}"]},
    }
    (summary,) = _lines(capsys, "datasets_warnings_backfilled")
    assert (summary["rewritten"], summary["gone"]) == (1, int(member == _GONE))


def test_the_count_past_the_named_members_is_filled_after_the_last_named_one():
    # Var_Gone comes before the named Var_In in dcid order, so Var_Out fills the one place.
    store = _Store([_row(_warning([_IN], more=1), zero=[_GONE, _IN, _OUT])])

    assert _run(store) == 0

    stored = _stored(store)
    assert stored["off_cadence_members"] == {"count": 1, "refs": ["dcid:Var_In"]}
    assert stored["out_of_window_members"] == {"count": 1, "refs": ["dcid:Var_Out"]}


def test_too_few_members_after_the_last_named_one_is_refused_for_good(capsys):
    warning = _warning([_OUT], more=1)
    store = _Store([_row(warning, zero=[_GONE, _IN, _OUT])])

    assert _run(store) == 0

    assert _stored(store) == warning
    (failed,) = _lines(capsys, "datasets_warning_backfill_failed")
    assert failed["transient"] is False


def test_every_zero_count_member_matching_no_selection_is_named_unserved():
    store = _Store([_row(_warning([_IN]), zero=[_IN, _MOVED])])

    assert _run(store) == 0

    stored = _stored(store)
    assert stored["off_cadence_members"] == {"count": 1, "refs": ["dcid:Var_In"]}
    assert stored["unserved_members"] == {"count": 1, "refs": ["dcid:Var_Moved"]}


def test_a_member_holding_only_another_dataset_is_unpopulated_under_the_elected_one():
    # The index resolves the definition, so the mint elected the fixture table and Var_Moved,
    # holding only PROV2, was unpopulated: never unserved, and named by the old sentence as
    # holding nothing in the requested years.
    store = _Store([_row(_warning([_MOVED]), zero=[_MOVED])])

    assert _run(store, indexed=True) == 0

    stored = _stored(store)
    assert stored["out_of_window_members"] == {"count": 1, "refs": ["dcid:Var_Moved"]}
    assert "unserved_members" not in stored


def test_an_open_range_closes_at_the_year_the_row_was_recorded():
    # Var_Late holds 2026, past the recorded year 2025, so at the mint it held nothing in
    # the window; under a window closed at today's year it would be off cadence.
    late = "Var_Late"
    facets = {**_FACETS, (late, _KEN): [_facet("f-late", "2026")]}
    store = _Store([_row(_warning([late]), zero=[late], end_year=None)])

    assert _run(store, graph=FakeGraphReader(facets=facets)) == 0

    assert _stored(store)["out_of_window_members"] == {"count": 1, "refs": ["dcid:Var_Late"]}


_METHOD_SELECTION = {
    "axis": "measurementMethod",
    "ref": "M",
    "basis": "deployment_fallback",
    "default_policy": None,
    "default_policy_version": None,
    "selection_reason": "most observations",
    "dispatch_facet_id": "f-served",
}


@pytest.mark.parametrize(
    ("method", "stated", "served_until", "field"),
    [
        pytest.param("Estimated", True, "2015", "off_cadence_members", id="stated"),
        pytest.param("Estimated", False, "2015", None, id="unstated"),
        # A method that is not estimated never continues a series, so the trace states all.
        pytest.param("M2", False, "2015", "unserved_members", id="not-estimated"),
        # A served series reaching the window's end continues with nothing.
        pytest.param("Estimated", False, "2020", "unserved_members", id="served-to-the-end"),
    ],
)
def test_a_continuation_the_trace_cannot_state_is_refused(
    method, stated, served_until, field, capsys
):
    # Var_Other holds the selected source only under `method`, past the served series' end,
    # which a continuation of the selected method admits when `method` is estimated.
    other = "Var_Other"
    facets = {
        **_FACETS,
        (_SERVED, _KEN): [_facet("f-served", served_until)],
        (other, _KEN): [_facet("f-other", "2018", method=method)],
    }
    row = _row(_warning([other]), zero=[other])
    row["trace"]["facets"].append(_METHOD_SELECTION)
    if stated:
        row["trace"]["continuations"] = [
            {"axis": "measurementMethod", "method": method, "periods": ["2016", "2020"]}
        ]
    store = _Store([row])

    assert _run(store, graph=FakeGraphReader(facets=facets)) == 0

    if field is not None:
        assert _stored(store)[field] == {"count": 1, "refs": ["dcid:Var_Other"]}
    else:
        assert _stored(store) == _warning([other])
        (failed,) = _lines(capsys, "datasets_warning_backfill_failed")
        assert failed["transient"] is False


@pytest.mark.parametrize(
    ("clamped", "suffix"),
    [
        pytest.param(False, "", id="detail-alone"),
        pytest.param(True, "", id="clamped"),
        pytest.param(False, f" {COUNTRIES_FALLBACK_SENTENCE}", id="countries-fallback"),
    ],
)
def test_a_warning_left_with_the_catalogue_detail_alone_is_dropped(clamped, suffix, capsys):
    warning = {**_warning([_OUT]), "message": f"{_DETAIL}{_old_sentence([_OUT])}{suffix}"}
    row = _row(warning, zero=[_OUT])
    if clamped:
        row["trace"]["applied_window"] = {
            "kind": "range",
            "start_year": 2010,
            "end_year": 2018,
            "origin": "coverage_clamp",
        }
    store = _Store([row])

    assert _run(store, absent_reads_zero=True) == 0

    (stored,) = store.rows.values()
    written = clamped or bool(suffix)
    kept = [{**warning, "message": f"{_DETAIL}{suffix}"}] if written else []
    assert stored["warnings"] == stored["immutable_body"]["warnings"] == kept
    (summary,) = _lines(capsys, "datasets_warnings_backfilled")
    assert summary["dropped"] == int(not written)


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
        self._count("observation_facets_batch_many")
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

    # A refusal follows from the stored row, so every run reports it and none fails on it.
    assert _run(store) == 0

    assert _stored(store) == warning
    (failed,) = _lines(capsys, "datasets_warning_backfill_failed")
    assert failed["transient"] is False


def test_a_second_run_finds_no_row():
    store = _Store([_row(_warning([_IN, _OUT]), zero=[_IN, _OUT])])
    assert _run(store) == 0
    rewritten = _stored(store)

    assert select_rows(store, new_format=False) == []
    assert _run(store) == 0
    assert (_stored(store), store.updates) == (rewritten, 1)


@pytest.mark.parametrize("where", ["warnings", "immutable_body"])
def test_a_row_changed_after_it_was_read_keeps_the_change(where, capsys):
    store = _Store([_row(_warning([_IN]), zero=[_IN])])
    rows = select_rows(store, new_format=False)
    changed = {**_warning([_IN]), "message": "changed meanwhile"}
    row = store.rows[("spec_a", "cv_a")]
    before = json.loads(json.dumps(row, default=str))
    if where == "warnings":
        row["warnings"] = [changed]
    else:
        row["immutable_body"] = {**row["immutable_body"], "warnings": [changed]}
    after = json.loads(json.dumps(row, default=str))

    assert _run(store, rows=rows) == 1
    assert json.loads(json.dumps(store.rows[("spec_a", "cv_a")], default=str)) == after != before
    (summary,) = _lines(capsys, "datasets_warnings_backfilled")
    assert summary["conflicts"] == 1


def test_a_row_whose_body_and_column_disagree_is_refused_for_good(capsys):
    warning = _warning([_IN])
    row = _row(warning, zero=[_IN])
    row["immutable_body"] = {**row["immutable_body"], "warnings": [{**warning, "message": "x"}]}
    store = _Store([row])
    before = json.loads(json.dumps(row, default=str))

    assert _run(store) == 0

    assert json.loads(json.dumps(store.rows[("spec_a", "cv_a")], default=str)) == before
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    (failed,) = [line for line in lines if line["event"] == "datasets_warning_backfill_failed"]
    assert (failed["transient"], failed["severity"]) == (False, "WARNING")
    (summary,) = [line for line in lines if line["event"] == "datasets_warnings_backfilled"]
    assert (summary["permanent"], summary["mismatched"], summary["severity"]) == (1, 1, "WARNING")


def test_a_new_format_row_that_fails_warns_and_leaves_the_exit_code(capsys):
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


def test_a_dry_run_after_a_real_one_counts_the_rewritten_rows_in_the_new_format(capsys):
    store = _Store(
        [
            _row(_warning([_IN]), zero=[_IN], subject_id="spec_a"),
            _row(_warning([_OUT]), zero=[_OUT], subject_id="spec_b"),
        ]
    )

    assert _run(store) == 0
    assert _run(store, dry_run=True) == _run(store, dry_run=True) == 0

    real, *dry = _lines(capsys, "datasets_warnings_backfilled")
    assert real["rewritten"] == 2
    assert dry[0] == dry[1]
    assert (dry[0]["rewritten"], dry[0]["new_format_rows"], dry[0]["new_format_agreed"]) == (
        0,
        2,
        2,
    )


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
    monkeypatch.setenv("DATASETS_GRAPH_TOKEN", "token")
    monkeypatch.setattr(maintenance, "shared_pool", lambda: None)
    monkeypatch.setattr(maintenance, "close_shared_pool", lambda: None)
    monkeypatch.setattr(maintenance, "select_rows", lambda store, *, new_format: [])

    def no_release():
        raise AssertionError("the release was opened")

    monkeypatch.setattr(maintenance, "get_release_service", no_release)

    assert maintenance.main(["backfill-warnings", "--dry-run"]) == 0
    (summary,) = _lines(capsys, "datasets_warnings_backfilled")
    assert (summary["dry_run"], summary["rewritten"]) == (True, 0)


def test_the_task_without_a_graph_token_refuses_to_start(monkeypatch, capsys):
    monkeypatch.delenv("DATASETS_GRAPH_TOKEN", raising=False)

    def no_store():
        raise AssertionError("the store was opened")

    monkeypatch.setattr(maintenance, "shared_pool", no_store)

    assert maintenance.main(["backfill-warnings"]) == 1
    (failed,) = _lines(capsys, "datasets_maintenance_failed")
    assert (failed["task"], failed["severity"]) == ("backfill-warnings", "ERROR")


class _OutageStore(_Store):
    """A store whose updates fail as an unreachable pool does."""

    def fetch_one(self, sql: str, params: Sequence[object] = ()) -> Mapping[str, object] | None:
        raise PoolTimeoutError("no connection came free")


def test_a_store_outage_stops_the_run(capsys):
    store = _OutageStore(
        [
            _row(_warning([_IN]), zero=[_IN], subject_id="spec_a"),
            _row(_warning([_IN]), zero=[_IN], subject_id="spec_b"),
        ]
    )
    graph = FakeGraphReader(facets=_FACETS)

    assert _run(store, graph=graph) == 2

    assert graph.calls["observation_facets_batch_many"] == 1
    (summary,) = _lines(capsys, "datasets_warnings_backfilled")
    assert (summary["remaining"], summary["permanent"]) == (2, 0)


class _BrokenGraph(FakeGraphReader):
    def observation_facets_batch_many(self, **kwargs):
        self._count("observation_facets_batch_many")
        raise ValueError("a defect, not a failed read")


@pytest.mark.parametrize("fault", ["ceiling", "malformed-identity"])
def test_a_deterministic_failure_is_refused_for_good(fault, monkeypatch, capsys):
    row = _row(_warning([_IN]), zero=[_IN])
    if fault == "ceiling":
        monkeypatch.setattr(facets_module, "_MAX_COLLECTION_PROBE_PAIRS", 0)
    else:
        row["definition"]["identity"]["facets"] = {"unit": {"ref": "USD", "request": {}}}
    store = _Store([row])

    assert _run(store) == 0

    (failed,) = _lines(capsys, "datasets_warning_backfill_failed")
    assert (failed["transient"], failed["refused"], failed["severity"]) == (
        False,
        True,
        "WARNING",
    )


def test_a_failed_probe_is_kept_for_the_run(capsys):
    store = _Store(
        [
            _row(_warning([_IN]), zero=[_IN], content_version="cv_a"),
            _row(_warning([_IN]), zero=[_IN], content_version="cv_b"),
        ]
    )
    graph = _FailingGraph(facets=_FACETS)

    assert _run(store, graph=graph) == 2

    assert graph.calls["observation_facets_batch_many"] == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    first, second = [line for line in lines if line["event"] == "datasets_warning_backfill_failed"]
    # The second row raises an error of its own, chained to the first row's.
    assert (first["transient"], second["transient"]) == (True, True)
    assert second["error"].startswith("_Unread: an earlier row's probe")
    (summary,) = [line for line in lines if line["event"] == "datasets_warnings_backfilled"]
    assert (summary["transient"], summary["probes"]) == (2, 1)


def test_a_row_without_a_definition_is_refused_and_the_run_goes_on(capsys):
    bare = _row(_warning([_IN]), zero=[_IN], subject_id="spec_bare")
    bare["definition"] = None
    store = _Store([bare, _row(_warning([_IN]), zero=[_IN], subject_id="spec_b")])

    assert _run(store) == 0

    assert store.rows[("spec_b", "cv_a")]["warnings"][0]["off_cadence_members"]
    (failed,) = _lines(capsys, "datasets_warning_backfill_failed")
    assert (failed["subject_id"], failed["transient"]) == ("spec_bare", False)


class _StatusGraph(FakeGraphReader):
    def __init__(
        self, *, status: int | None = None, error: Exception | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._error = error or GraphReadError("graph read failed", upstream_status=status)

    def observation_facets_batch_many(self, **kwargs):
        self._count("observation_facets_batch_many")
        raise self._error


def _omitted(dcids: Sequence[str]) -> GraphReadError:
    """The read error a graph client raises over dcids its re-request omits too."""
    omitted = GraphOmittedDcidsError("omitted again", dcids=dcids)
    error = GraphReadError(str(omitted))
    error.__cause__ = omitted
    return error


class _StatusAvailability(ScopeAvailability):
    def __init__(self, *, error: Exception, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._error = error

    def availability_values(self, **kwargs):
        raise self._error


_SCOPED_GRAPH = FakeGraphReader(facets=_FACETS, child_places={("Earth", "Country"): [_KEN]})


@pytest.mark.parametrize(
    ("graph", "availability", "transient"),
    [
        pytest.param(_StatusGraph(status=404, facets=_FACETS), None, False, id="graph-404"),
        pytest.param(_StatusGraph(status=503, facets=_FACETS), None, True, id="graph-503"),
        pytest.param(_StatusGraph(status=429, facets=_FACETS), None, True, id="graph-429"),
        pytest.param(_StatusGraph(status=401, facets=_FACETS), None, True, id="graph-401"),
        # A 2xx body the client could not read, and a redirect, are a gateway's fault.
        pytest.param(_StatusGraph(status=200, facets=_FACETS), None, True, id="graph-200-unread"),
        pytest.param(_StatusGraph(status=302, facets=_FACETS), None, True, id="graph-302"),
        pytest.param(
            _StatusGraph(
                error=GraphNotAdvancingError("repeated", dcid=_IN, property="nextToken"),
                facets=_FACETS,
            ),
            None,
            False,
            id="graph-not-advancing",
        ),
        pytest.param(
            _StatusGraph(error=_omitted([_IN]), facets=_FACETS),
            None,
            False,
            id="graph-omits-a-dcid-again",
        ),
        pytest.param(
            _SCOPED_GRAPH,
            _StatusAvailability(
                error=SdmxHttpError("refused", upstream_status=404), data=[(_IN, _KEN)]
            ),
            False,
            id="availability-404",
        ),
        pytest.param(
            _SCOPED_GRAPH,
            _StatusAvailability(
                error=SdmxHttpError("refused", upstream_status=403), data=[(_IN, _KEN)]
            ),
            True,
            id="availability-403",
        ),
        pytest.param(
            _SCOPED_GRAPH,
            _StatusAvailability(
                error=SdmxUnsupportedFilterError("TIME_PERIOD", upstream_status=None),
                data=[(_IN, _KEN)],
            ),
            False,
            id="availability-refused-before-sending",
        ),
    ],
)
def test_a_client_status_is_refused_for_good_and_a_server_status_is_retried(
    graph, availability, transient, capsys
):
    places = () if availability is not None else (_KEN,)
    store = _Store([_row(_warning([_IN]), zero=[_IN], places=places)])

    assert _run(store, graph=graph, availability=availability) == int(transient)

    (failed,) = _lines(capsys, "datasets_warning_backfill_failed")
    assert failed["transient"] is transient


def test_a_probe_timeout_is_retried_and_the_run_goes_on(monkeypatch, capsys):
    from datasets_api import warning_backfill

    monkeypatch.setattr(warning_backfill, "PROBE_DEADLINE_SECONDS", -1.0)
    store = _Store(
        [
            _row(_warning([_IN]), zero=[_IN], places=(), subject_id="spec_a"),
            _row(_warning([_IN]), zero=[_IN], subject_id="spec_b"),
        ]
    )

    assert _run(store, graph=_SCOPED_GRAPH, availability=ScopeAvailability(data=[(_IN, _KEN)])) == 1

    assert store.rows[("spec_b", "cv_a")]["warnings"][0]["off_cadence_members"]
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    (failed,) = [line for line in lines if line["event"] == "datasets_warning_backfill_failed"]
    assert (failed["subject_id"], failed["transient"]) == ("spec_a", True)
    (summary,) = [line for line in lines if line["event"] == "datasets_warnings_backfilled"]
    assert (summary["transient"], summary["rewritten"]) == (1, 1)


def test_a_defect_fails_the_run_and_its_probe_is_kept(capsys):
    store = _Store(
        [
            _row(_warning([_IN]), zero=[_IN], content_version="cv_a"),
            _row(_warning([_IN]), zero=[_IN], content_version="cv_b"),
        ]
    )
    graph = _BrokenGraph(facets=_FACETS)

    assert _run(store, graph=graph) == 2

    assert graph.calls["observation_facets_batch_many"] == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    failed = [line for line in lines if line["event"] == "datasets_warning_backfill_failed"]
    assert [(line["refused"], line["severity"]) for line in failed] == [(False, "ERROR")] * 2
    (summary,) = [line for line in lines if line["event"] == "datasets_warnings_backfilled"]
    assert (summary["failed"], summary["permanent"], summary["probes"]) == (2, 0, 1)


def test_versions_dispatching_different_sources_probe_apart():
    # cv_a selected PROV2, which holds Var_Moved; cv_b selected PROV1, which does not.
    a = _row(_warning([_MOVED]), zero=[_MOVED], content_version="cv_a")
    a["trace"]["facets"][0]["ref"] = "PROV2"
    b = _row(_warning([_MOVED]), zero=[_MOVED], content_version="cv_b")
    together = _Store([a, b])
    alone = _Store([_row(_warning([_MOVED]), zero=[_MOVED], content_version="cv_b")])

    assert _run(together, indexed=True) == _run(alone, indexed=True) == 0

    assert (
        together.rows[("spec_a", "cv_b")]["warnings"] == alone.rows[("spec_a", "cv_b")]["warnings"]
    )
    assert (
        together.rows[("spec_a", "cv_a")]["warnings"] != alone.rows[("spec_a", "cv_b")]["warnings"]
    )


def test_a_run_holds_the_probes_of_one_subject_only():
    store = _Store(
        [
            _row(_warning([_IN]), zero=[_IN], subject_id="spec_b"),
            _row(_warning([_IN]), zero=[_IN], subject_id="spec_a", content_version="cv_b"),
            _row(_warning([_IN]), zero=[_IN], subject_id="spec_a"),
        ]
    )
    readers = Readers(
        graph=FakeGraphReader(facets=_FACETS),
        availability=ScopeAvailability(data=()),
        code_index=_code_index(),
        coverage=None,
    )

    assert _run(store, readers=readers) == 0

    assert {key[0] for key in readers.probes} == {"spec_b"}
    assert all(row["warnings"][0]["off_cadence_members"] for row in store.rows.values())


def test_the_filler_prefers_members_a_mint_counted():
    # After the last named member, Var_Z0 holds no observation at all and Var_Z1 holds the
    # selected source in the requested years, as the member the old mint counted did.
    named = [f"Var_A{index}" for index in range(5)]
    facets = {
        **_FACETS,
        **{(member, _KEN): [_facet(f"f-{member}", "2015")] for member in named},
        ("Var_Z1", _KEN): [_facet("f-z1", "2015")],
    }
    store = _Store([_row(_warning(named, more=1), zero=[*named, "Var_Z0", "Var_Z1"])])

    assert _run(store, graph=FakeGraphReader(facets=facets)) == 0

    stored = _stored(store)
    assert stored["off_cadence_members"]["count"] == 6
    assert "unserved_members" not in stored and "out_of_window_members" not in stored


def test_a_new_format_row_names_its_members_in_dcid_order_across_kinds(capsys):
    # Six off-cadence members, five referenced, and one out of the window that sorts after
    # the sixth: the old sentence named the first five in dcid order across both kinds.
    off = [f"Var_B{index}" for index in range(6)]
    facets = {
        **_FACETS,
        **{(member, _KEN): [_facet(f"f-{member}", "2015")] for member in off},
        ("Var_C0", _KEN): [_facet("f-c0", "1990")],
    }
    split = {
        "code": "coverage_below_scope",
        "severity": "warn",
        "message": f"{_DETAIL} "
        + " ".join(
            member_sentences(
                off_cadence=tuple(f"dcid:{member}" for member in off),
                out_of_window=("dcid:Var_C0",),
                unserved_members=(),
                member_count=8,
            )
        ),
        "member_count": 8,
        "off_cadence_members": {"count": 6, "refs": [f"dcid:{member}" for member in off[:5]]},
        "out_of_window_members": {"count": 1, "refs": ["dcid:Var_C0"]},
    }
    store = _Store([_row(split, zero=[*off, "Var_C0"])])

    assert _run(store, graph=FakeGraphReader(facets=facets), dry_run=True) == 0

    (row,) = _lines(capsys, "datasets_warning_backfill_row")
    assert row["agrees"] is True


def test_a_new_format_row_reading_absence_as_zero_fills_only_what_the_mint_counts(capsys):
    # The mint counted Var_A..Var_E, Var_G and Var_H off cadence and dropped Var_F, out of the
    # window, since the group reads an absent member as zero.
    off = ["Var_A", "Var_B", "Var_C", "Var_D", "Var_E", "Var_G", "Var_H"]
    facets = {
        **_FACETS,
        **{(member, _KEN): [_facet(f"f-{member}", "2015")] for member in off},
        ("Var_F", _KEN): [_facet("f-f", "1990")],
    }
    refs = tuple(f"dcid:{member}" for member in off)
    split = {
        "code": "coverage_below_scope",
        "severity": "warn",
        "message": " ".join(
            (
                _DETAIL,
                *member_sentences(
                    off_cadence=refs, out_of_window=(), unserved_members=(), member_count=9
                ),
            )
        ),
        "member_count": 9,
        "off_cadence_members": {"count": 7, "refs": list(refs[:5])},
    }
    store = _Store([_row(split, zero=sorted([*off, "Var_F"]))])

    assert (
        _run(store, graph=FakeGraphReader(facets=facets), absent_reads_zero=True, dry_run=True) == 0
    )

    (row,) = _lines(capsys, "datasets_warning_backfill_row")
    assert (row["agrees"], row["severity"]) == (True, "INFO")


def test_a_new_format_row_that_disagrees_warns(capsys):
    # The stored split calls Var_In out of the window; the graph holds it in the window.
    split = {
        "code": "coverage_below_scope",
        "severity": "warn",
        "message": "x",
        "member_count": 2,
        "out_of_window_members": {"count": 1, "refs": ["dcid:Var_In"]},
    }
    store = _Store([_row(split, zero=[_IN])])

    assert _run(store, dry_run=True) == 0

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    (row,) = [line for line in lines if line["event"] == "datasets_warning_backfill_row"]
    assert (row["agrees"], row["severity"]) == (False, "WARNING")
    (summary,) = [line for line in lines if line["event"] == "datasets_warnings_backfilled"]
    assert (summary["new_format_agreed"], summary["severity"]) == (0, "WARNING")


def test_a_new_format_row_left_short_by_the_graph_disagrees(capsys):
    # The mint counted Var_A..Var_E and Var_F off cadence; the graph no longer holds Var_F.
    off = ["Var_A", "Var_B", "Var_C", "Var_D", "Var_E"]
    facets = {**_FACETS, **{(member, _KEN): [_facet(f"f-{member}", "2015")] for member in off}}
    split = {
        "code": "coverage_below_scope",
        "severity": "warn",
        "message": "x",
        "member_count": 7,
        "off_cadence_members": {"count": 6, "refs": [f"dcid:{member}" for member in off]},
    }
    store = _Store([_row(split, zero=[*off, "Var_F"])])

    assert _run(store, graph=FakeGraphReader(facets=facets), dry_run=True) == 0

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    (row,) = [line for line in lines if line["event"] == "datasets_warning_backfill_row"]
    assert (row["agrees"], row["found"]["off_cadence_members"]["count"]) == (False, 5)
    (summary,) = [line for line in lines if line["event"] == "datasets_warnings_backfilled"]
    assert (summary["new_format_failed"], summary["new_format_agreed"]) == (0, 0)


@pytest.mark.parametrize(
    ("method", "field"),
    [
        pytest.param("M2", "unserved_members", id="nothing-estimated"),
        pytest.param("Estimated", None, id="an-estimated-facet"),
    ],
)
def test_a_selection_the_graph_no_longer_holds_is_refused_only_where_it_could_continue(
    method, field, capsys
):
    # The graph no longer holds Var_Served, the one member that held the selected facets.
    other = "Var_Other"
    facets = {
        (other, _KEN): [_facet("f-other", "2018", method=method)],
    }
    row = _row(_warning([other]), zero=[other])
    row["trace"]["facets"].append(_METHOD_SELECTION)
    store = _Store([row])

    assert _run(store, graph=FakeGraphReader(facets=facets)) == 0

    if field is not None:
        assert _stored(store)[field] == {"count": 1, "refs": ["dcid:Var_Other"]}
    else:
        assert _stored(store) == _warning([other])
        (failed,) = _lines(capsys, "datasets_warning_backfill_failed")
        assert failed["refused"] is True


def test_a_continuation_is_judged_at_the_cadence_most_members_hold(capsys):
    # Var_AMonthly, first in the trace, dates the selected facet by month; Var_Served and
    # Var_Served2 date it by year, so the group dispatches yearly, and Var_Other's estimated
    # facet continues the yearly series, which the trace does not name.
    monthly, served2, other = "Var_AMonthly", "Var_Served2", "Var_Other"
    facets = {
        **_FACETS,
        (monthly, _KEN): [_facet("f-served", "2015-01")],
        (served2, _KEN): [_facet("f-served", "2015")],
        (other, _KEN): [_facet("f-other", "2018", method="Estimated")],
    }
    row = _row(_warning([other]), zero=[other])
    row["trace"]["resolved_stat_vars"] = sorted(
        [
            *row["trace"]["resolved_stat_vars"],
            *(
                {"ref": f"dcid:{member}", "roles": ["observationAbout"], "observation_count": 3}
                for member in (monthly, served2)
            ),
        ],
        key=lambda entry: entry["ref"],
    )
    row["trace"]["facets"].append(_METHOD_SELECTION)
    store = _Store([row])

    assert _run(store, graph=FakeGraphReader(facets=facets)) == 0

    assert _stored(store) == _warning([other])
    (failed,) = _lines(capsys, "datasets_warning_backfill_failed")
    assert "Estimated" in failed["error"]


def test_a_cadence_tie_breaks_on_the_series_each_facet_holds(capsys):
    # Var_AMonthly holds the selected source by month at two places, Var_Served by year at
    # one over more of the window's years. One member holds each cadence; the monthly facet's
    # two series rank it first, as the mint ranked it, ahead of the years the yearly facet
    # spans, so the yearly estimate of Var_Other continues nothing.
    uga = "country/UGA"
    monthly, other = "Var_AMonthly", "Var_Other"
    facets = {
        (monthly, _KEN): [_facet("f-monthly", "2015-01")],
        (monthly, uga): [_facet("f-monthly", "2015-01")],
        (_SERVED, _KEN): [
            replace(
                _facet("f-served", "2016"),
                earliest_date="2012",
                dates=("2012", "2016"),
                values=(1.0, 1.0),
            )
        ],
        (other, _KEN): [_facet("f-other", "2018", method="Estimated")],
    }
    row = _row(_warning([other]), zero=[other], places=(_KEN, uga))
    row["trace"]["resolved_stat_vars"] = sorted(
        [
            *row["trace"]["resolved_stat_vars"],
            {"ref": f"dcid:{monthly}", "roles": ["observationAbout"], "observation_count": 3},
        ],
        key=lambda entry: entry["ref"],
    )
    row["trace"]["facets"].append(_METHOD_SELECTION)
    store = _Store([row])

    assert _run(store, graph=FakeGraphReader(facets=facets)) == 0

    assert _stored(store)["unserved_members"] == {"count": 1, "refs": ["dcid:Var_Other"]}


def test_a_repeated_axis_in_the_trace_reads_its_first_entry():
    row = _row(_warning([_MOVED]), zero=[_MOVED])
    row["trace"]["facets"].append({**row["trace"]["facets"][0], "ref": "PROV2"})
    store = _Store([row])
    first = _Store([_row(_warning([_MOVED]), zero=[_MOVED])])

    assert _run(store, indexed=True) == _run(first, indexed=True) == 0

    assert _stored(store) == _stored(first)
