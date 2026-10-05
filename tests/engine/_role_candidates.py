"""Build role-facet candidates from in-memory role tuples."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from search.engine.clients.sdmx import RoleFacetCandidate, RoleTuple


@dataclass(frozen=True)
class ListedRolePairs:
    """Role-pair counts have no year data, so window arguments have no effect."""

    n_obs_by_role_tuple: Mapping[RoleTuple, int]

    def head(
        self,
        limit: int,
        *,
        first_year: int | None,
        last_year: int | None,
        with_data: bool = False,
    ) -> list[tuple[RoleTuple, int]]:
        return sorted(
            (role_tuple, n_obs)
            for role_tuple, n_obs in self.n_obs_by_role_tuple.items()
            if n_obs or not with_data
        )[:limit]

    def n_obs(self, *, first_year: int | None, last_year: int | None) -> int:
        return sum(self.n_obs_by_role_tuple.values())

    def n_with_data(self, *, first_year: int | None, last_year: int | None) -> int:
        return sum(1 for n_obs in self.n_obs_by_role_tuple.values() if n_obs)

    def max_n_obs(self, *, first_year: int | None, last_year: int | None) -> int:
        return max(self.n_obs_by_role_tuple.values(), default=0)

    def year_span(self, *, first_year: int | None, last_year: int | None) -> tuple[int, int] | None:
        """Role-pair counts carry no years, so the span is unknown."""
        return None

    def members_with_data(
        self, *, first_year: int | None, last_year: int | None, role: str | None = None
    ) -> frozenset[str]:
        """Every dcid of the tuples with data on `role`, or with none on whichever role the
        test names first."""
        return frozenset(
            dcid
            for role_tuple, n_obs in self.n_obs_by_role_tuple.items()
            if n_obs
            for name, dcid in (role_tuple if role is not None else role_tuple[:1])
            if role is None or name == role
        )

    def excluding(self, *, role: str, dcids: frozenset[str]) -> ListedRolePairs:
        return ListedRolePairs(
            {
                role_tuple: n_obs
                for role_tuple, n_obs in self.n_obs_by_role_tuple.items()
                if not any(name == role and dcid in dcids for name, dcid in role_tuple)
            }
        )


def role_candidate(
    *,
    facet_id: str,
    role_tuples: Iterable[RoleTuple] | Mapping[RoleTuple, int],
    latest_date: str | None,
    provenance_id: str | None = None,
    measurement_method: str | None = None,
    unit: str | None = None,
    scaling_factor: str | None = None,
) -> RoleFacetCandidate:
    """Build a candidate, counting each tuple once unless counts are supplied, whose latest
    point falls on `latest_date`."""
    n_obs_by_role_tuple = (
        dict(role_tuples) if isinstance(role_tuples, Mapping) else dict.fromkeys(role_tuples, 1)
    )
    members_by_role: dict[str, set[str]] = {}
    for role_tuple in n_obs_by_role_tuple:
        for role, dcid in role_tuple:
            members_by_role.setdefault(role, set()).add(dcid)
    return RoleFacetCandidate(
        facet_id=facet_id,
        members_by_role={role: frozenset(dcids) for role, dcids in members_by_role.items()},
        pair_count=len(n_obs_by_role_tuple),
        latest_date=latest_date,
        pairs=ListedRolePairs(n_obs_by_role_tuple),
        provenance_id=provenance_id,
        measurement_method=measurement_method,
        unit=unit,
        scaling_factor=scaling_factor,
    )
