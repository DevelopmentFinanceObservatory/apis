"""Golden coverage for open ODA role-scope discovery and its preview dispatch.
Regenerate with ROLE_SCOPE_GOLDEN_WRITE=1 uv run pytest <this file>."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from materialise.roles import RoleAxis
from search.engine.clients.sdmx import RoleFacetCandidate
from search.engine.extract import DateRequest
from search.engine.region.observations.bulk_role_series import (
    _bulk_role_series,
    _plan_role_bulk_fetch,
)
from search.engine.retrieval.facet_election import rank_role_candidates, window_of
from search.engine.retrieval.read_plan import BulkPlan
from search.engine.retrieval.retrieve import Materialised
from search.engine.retrieval.role_scope import discover_role_scope
from search.models import TimeWindow
from search.observations import observation_settings
from series_coverage.loader import load_coverage
from series_coverage.shared import CoverageSnapshot
from tests.engine._fake_sdmx import CountingSdmxClient, plan_observations
from tests.fixtures import FakeGraph

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
_SLICE_URI = f"file://{_FIXTURES / 'coverage_slice'}"
_GOLDEN = _FIXTURES / "role_scope_open_oda.json"

_DATE_REQUESTS: dict[str, DateRequest | None] = {
    "dateless": None,
    "window_2015_2020": DateRequest(
        window=TimeWindow(start_year=2015, end_year=2020), latest=False
    ),
    "latest": DateRequest(window=None, latest=True),
}
_AXES: dict[RoleAxis, tuple[str, str]] = {
    "destination": ("financingRecipient", "financingProvider"),
    "origin": ("financingProvider", "financingRecipient"),
}
_CASES = [(date_name, axis) for date_name in _DATE_REQUESTS for axis in _AXES]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _candidate_projection(candidate: RoleFacetCandidate) -> dict[str, Any]:
    pairs = [
        role_tuple
        for role_tuple, _n_obs in candidate.pairs.head(
            candidate.pair_count, first_year=None, last_year=None
        )
    ]
    # The member sets and pair count must describe the same role tuples the
    # accessor reads back.
    members: dict[str, set[str]] = {}
    for role_tuple in pairs:
        for role, dcid in role_tuple:
            members.setdefault(role, set()).add(dcid)
    assert len(pairs) == candidate.pair_count
    assert members == {role: set(dcids) for role, dcids in candidate.members_by_role.items()}
    return {
        "facet_id": candidate.facet_id,
        "latest_date": candidate.latest_date,
        "provenance_id": candidate.provenance_id,
        "measurement_method": candidate.measurement_method,
        "unit": candidate.unit,
        "scaling_factor": candidate.scaling_factor,
        "pair_count": candidate.pair_count,
        "members_by_role": {
            role: sorted(dcids) for role, dcids in sorted(candidate.members_by_role.items())
        },
        "pairs_sha256": _digest([[list(entry) for entry in rt] for rt in pairs]),
    }


def _bulk_preview(result: Materialised, date_name: str) -> dict[str, Any]:
    """The observation preview's dispatch over this discovery's read plan: the facet it
    serves, the pairs it fetches, the roles its read filters on with each role's value
    count, and the coverage points it counts for the rest."""
    plan = result.read_plan
    assert isinstance(plan, BulkPlan) and plan.elected is not None
    ranked = rank_role_candidates(
        plan.candidates, stat_vars=list(plan.members), window=window_of(plan.read_years)
    )
    # Discovery plans the facet the preview serves, the ranking's top.
    assert ranked[0].facet_id == plan.facet_id
    fetch = _plan_role_bulk_fetch(plan)
    to_fetch = [[sv, [list(entry) for entry in rt]] for sv, rt in fetch.to_fetch]
    return {
        "time_period": list(plan.read_years) if plan.read_years is not None else None,
        "facet_id": plan.facet_id,
        "total_population": fetch.total_population,
        # A ranked read fetches the whole population, so the golden holds its count and digest.
        "to_fetch": (
            to_fetch
            if fetch.order == "fixed"
            else {"pairs": len(to_fetch), "sha256": _digest(to_fetch)}
        ),
        # The filter's values travel in the SDMX URL, so the count bounds its size.
        "entities": {role: len(dcids) for role, dcids in fetch.entities_for_series.items()},
        "unshipped_obs": fetch.population_n_obs - fetch.preview_n_obs,
        "order": fetch.order,
        "ranking": [[c.facet_id, c.pair_count] for c in ranked],
    }


def _projection(result: Materialised) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for f in dataclasses.fields(result):
        value = getattr(result, f.name)
        if f.name == "role_index_by_sv":
            projected[f.name] = {
                sv: [_candidate_projection(c) for c in candidates]
                for sv, candidates in value.items()
            }
        elif f.name == "read_plan" and isinstance(value, BulkPlan):
            # The plan's candidates merge the role index projected above; name them by id.
            projected[f.name] = {
                "members": list(value.members),
                "years": _jsonable(value.years),
                "read_years": value.read_years,
                "facet_id": value.facet_id,
                "candidate_ids": sorted(value.candidates),
                "population": value.population,
                "roles": _jsonable(value.roles),
            }
        else:
            projected[f.name] = _jsonable(value)
    return projected


def _summary(projected: dict[str, Any]) -> dict[str, Any]:
    role_index = projected["role_index_by_sv"]
    candidates = [c for cs in role_index.values() for c in cs]
    discovered = dict(projected["discovered_scope"])
    member_dcids = discovered.pop("member_dcids")
    return {
        "sv_count": len(projected["sv_dcids"]),
        "coverage": projected["coverage"],
        "headline": projected["headline"],
        "discovered_scope": {**discovered, "member_count": len(member_dcids)},
        "candidate_count": len(candidates),
        "pair_count": sum(c["pair_count"] for c in candidates),
        "role_index_sha256": _digest(role_index),
    }


@pytest.fixture(scope="module")
def _snapshot() -> CoverageSnapshot:
    loaded = load_coverage(_SLICE_URI)
    return CoverageSnapshot(
        index=loaded.index, manifest=loaded.manifest, stamp=loaded.manifest.stamp
    )


def _discover(snapshot: CoverageSnapshot, date_name: str, axis: RoleAxis) -> Materialised:
    role_name, opposite_role_name = _AXES[axis]
    oda = sorted(v for v in snapshot.index.role_names if "CRSTypeOfFlow10" in v)
    result = discover_role_scope(
        surviving_svs=oda,
        admitted_dcids=oda,
        member_facts={},
        member_population_exact=True,
        open_flow_axis=axis,
        role_name=role_name,
        opposite_role_name=opposite_role_name,
        opposite_dcids=(),
        facet_filters={},
        date_request=_DATE_REQUESTS[date_name],
        coverage_snapshot=snapshot,
    )
    assert isinstance(result, Materialised)
    return result


def test_golden_regeneration_is_off() -> None:
    assert not os.environ.get("ROLE_SCOPE_GOLDEN_WRITE"), "unset ROLE_SCOPE_GOLDEN_WRITE"


def test_regenerate_golden(_snapshot: CoverageSnapshot) -> None:
    if not os.environ.get("ROLE_SCOPE_GOLDEN_WRITE"):
        pytest.skip("golden regeneration is opt-in")
    golden = {}
    for date_name, axis in _CASES:
        result = _discover(_snapshot, date_name, axis)
        projected = _projection(result)
        golden[f"{date_name}/{axis}"] = {
            "summary": _summary(projected),
            "sha256": _digest(projected),
            "bulk_preview": _bulk_preview(result, date_name),
        }
    _GOLDEN.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize(("date_name", "axis"), _CASES)
def test_open_oda_discovery_matches_the_golden(
    _snapshot: CoverageSnapshot, date_name: str, axis: RoleAxis
) -> None:
    expected = json.loads(_GOLDEN.read_text())[f"{date_name}/{axis}"]
    result = _discover(_snapshot, date_name, axis)
    projected = _projection(result)

    assert _summary(projected) == expected["summary"]
    assert _digest(projected) == expected["sha256"]
    assert _bulk_preview(result, date_name) == expected["bulk_preview"]


def test_a_latest_bulk_read_of_open_oda_names_years_only_for_members_ending_after_theirs(
    _snapshot: CoverageSnapshot,
) -> None:
    """The members elect 48 distinct years. The read asks `LATEST` of every member, 26,994
    rows, one a series, then reads again the 83 members a series of which ends after the
    year its member elected, one read for each of the 9 years they elect, naming each
    member its own year alone. Those reads return 356 rows, fewer than the answer keeps,
    the count the `ranked` gate admitted."""
    plan = _discover(_snapshot, "latest", "destination").read_plan
    assert isinstance(plan, BulkPlan) and plan.elected_years is not None
    fetch = _plan_role_bulk_fetch(plan)
    sdmx = CountingSdmxClient(observations=plan_observations(plan))

    _bulk_role_series(plan, list(plan.members), sdmx=sdmx, graph=FakeGraph(), facet_filter=None)

    newest, *named = sdmx.series_calls
    assert (newest["variables"], newest["time_period"]) == (fetch.facet_svs, "LATEST")
    assert len({plan.elected_years[sv] for sv in fetch.facet_svs}) == 48
    read_again = [sv for call in named for sv in call["variables"]]
    assert (newest["rows"], len(named), len(read_again)) == (26_994, 9, 83)
    assert set(read_again) < set(fetch.facet_svs)
    for call in named:
        assert all(call["time_period"] == [str(plan.elected_years[sv])] for sv in call["variables"])
    rows_again = sum(call["rows"] for call in named)
    assert fetch.order == "ranked"
    assert rows_again == 356
    assert rows_again < fetch.population_n_obs <= observation_settings().role_bulk_rank_max_rows
