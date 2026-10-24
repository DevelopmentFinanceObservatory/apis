"""The origin catalogue sample build, resolve responses, a fake engine and a
sample search catalogue build for the search catalogue tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from origin_catalogue.artifact import load_catalogue
from origin_catalogue.catalogue import CatalogueSnapshot
from search.catalogue.artifact import (
    AidCounts,
    BuildInputs,
    CatalogueArtifact,
    CatalogueExample,
    CataloguePlace,
    Counts,
    PlaceCounts,
    YearRange,
    build_key,
    publish,
    read_pointer,
)
from search.models import (
    RESOLVE_RESPONSE_ADAPTER,
    DefinitePart,
    EntityRoleKey,
    GraphRef,
    ObservationsOmissionReason,
    Resolution,
    ResolveRequest,
    ResolveResponse,
    StatVar,
)
from tests.conftest import base_response, minimal_spec
from tests.origin_catalogue.support import publish_sample


def origin_snapshot(base: Path) -> CatalogueSnapshot:
    """`tests.origin_catalogue.support`'s sample build, published under `base`."""
    uri = base.as_uri()
    publish_sample(uri)
    return load_catalogue(uri)


def definite(
    *stat_vars: str,
    place: str = "country/KEN",
    role: EntityRoleKey = "observationAbout",
    points: int = 3,
) -> ResolveResponse:
    """A resolution with one definite part resolving `stat_vars` and `place`
    in `role`, whose preview carries `points` points of the first at
    `place` in `role`."""
    spec = minimal_spec()
    spec["resolved"]["entities"] = [{"ref": place, "role": role}]
    spec["resolved"]["stat_vars"] = {"member_count": len(stat_vars)}
    spec["resolved"]["observations"] = {
        "series": [
            {
                "stat_var": {"ref": stat_vars[0]},
                "entities": [{"role": role, "ref": place}],
                "points": [{"date": str(2000 + year), "value": 1.0} for year in range(points)],
                "total_obs": points,
                "source_refs": [],
            }
        ],
        "total_obs": points,
        "truncated": False,
    }
    part = {"status": "definite", "variable_text": "v", "warnings": [], "spec": spec}
    return serving(
        RESOLVE_RESPONSE_ADAPTER.validate_python(base_response(parts=[part])), *stat_vars
    )


def serving(response: ResolveResponse, *stat_vars: str) -> ResolveResponse:
    """`response` with each definite part's Spec serving `stat_vars`, as the engine states
    them off the wire (`ResolvedSpec.members`)."""
    assert isinstance(response, Resolution)
    members = [
        StatVar(
            ref=GraphRef(ref=dcid, label=dcid),
            shape_id="",
            slot_values=[],
            constraint_properties=[],
        )
        for dcid in stat_vars
    ]
    parts = [
        part.model_copy(
            update={
                "spec": part.spec.model_copy(
                    update={"resolved": part.spec.resolved.stating_members(members)}
                )
            }
        )
        if isinstance(part, DefinitePart)
        else part
        for part in response.parts
    ]
    return response.model_copy(update={"parts": parts})


def omitted(
    stat_var: str,
    reason: ObservationsOmissionReason,
    *,
    place: str = "country/KEN",
    role: EntityRoleKey = "observationAbout",
) -> ResolveResponse:
    """A resolution with one definite part resolving `stat_var` and `place`
    in `role`, whose preview the engine omitted for `reason`."""
    response = definite(stat_var, place=place, role=role).model_dump(mode="json")
    response["parts"][0]["spec"]["resolved"]["observations"] = {
        "series": [],
        "total_obs": 0,
        "truncated": False,
        "omitted": {"reason": reason, "message": reason},
    }
    return serving(RESOLVE_RESPONSE_ADAPTER.validate_python(response), stat_var)


def no_data() -> ResolveResponse:
    part = {
        "status": "no_data",
        "variable_text": "v",
        "warnings": [],
        "no_data": {"reason": "no_observations"},
    }
    return RESOLVE_RESPONSE_ADAPTER.validate_python(base_response(parts=[part]))


def catalogue_answer() -> ResolveResponse:
    return RESOLVE_RESPONSE_ADAPTER.validate_python(base_response(kind="catalogue", scope={}))


def timed_out(response: ResolveResponse) -> ResolveResponse:
    """`response` as a partial the soft deadline cut short."""
    diagnostics = response.diagnostics.model_copy(update={"timed_out": True})
    return response.model_copy(update={"diagnostics": diagnostics})


type Answer = ResolveResponse | Exception
"""A response the fake engine returns, or an exception it raises."""


@dataclass(frozen=True, slots=True)
class Elected:
    """What the fake engine resolves a query to by default."""

    stat_var: str
    place: str
    role: EntityRoleKey = "observationAbout"


@dataclass
class FakeEngine:
    """Answers a query from `first` the first time it is sent, then from
    `answers`, else with a definite part electing `elected[query]`, and
    records every query it was sent. An `Exception` answer is raised."""

    elected: dict[str, Elected]
    answers: dict[str, Answer] = field(default_factory=dict)
    first: dict[str, Answer] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)

    async def __call__(self, request: ResolveRequest) -> ResolveResponse:
        assert request.input.kind == "raw_text"
        assert request.response_options is not None
        assert request.response_options.include_observations
        query = request.input.query
        self.queries.append(query)
        answer = self.first.pop(query, None) or self.answers.get(query)
        if isinstance(answer, Exception):
            raise answer
        if answer is not None:
            return answer
        elected = self.elected[query]
        return definite(elected.stat_var, place=elected.place, role=elected.role)


_INPUTS = BuildInputs(
    origin_catalogue_build_id="origin-1",
    coverage_build_id="coverage-1",
    build_commit="abc123",
    examples_digest="digest",
)


def sample_catalogue(version: str = "20260930T000000Z-00000001") -> CatalogueArtifact:
    """A build with two examples, Kenya offering one and Tuvalu none."""
    aid = AidCounts(donors=3, recipients=5, years=YearRange(first=1990, last=2024))
    return CatalogueArtifact(
        catalogue_version=version,
        key=build_key(_INPUTS),
        started_at=datetime(2026, 9, 30, tzinfo=UTC),
        built_at=datetime(2026, 9, 30, tzinfo=UTC),
        inputs=_INPUTS,
        counts=Counts(indicators=9, datasets=5, publishers=4, countries=2, aid=aid),
        examples=[
            CatalogueExample(
                id=0,
                text="Population of Ethiopia",
                template="Population of {place}",
                stat_var="Count_Person",
                place="country/ETH",
            ),
            CatalogueExample(
                id=2,
                text="Health aid to Ghana",
                template="Health aid to {place}",
                stat_var="Amount_Health",
                place="country/GHA",
            ),
        ],
        places={
            "country/KEN": CataloguePlace(
                name="Kenya",
                counts=PlaceCounts(
                    publishers=2,
                    variables=7,
                    aid=AidCounts(donors=2, recipients=0, years=YearRange(first=2000, last=2024)),
                ),
                examples=[0],
            ),
            "country/TUV": CataloguePlace(
                name=None,
                counts=PlaceCounts(
                    publishers=1, variables=1, aid=AidCounts(donors=0, recipients=0, years=None)
                ),
                examples=[],
            ),
        },
    )


def publish_catalogue(uri: str, artifact: CatalogueArtifact | None = None) -> CatalogueArtifact:
    """Publishes `artifact` (the sample by default) over whatever `uri` holds."""
    artifact = artifact or sample_catalogue()
    _, generation = read_pointer(uri)
    assert publish(uri, artifact, expected_generation=generation)
    return artifact
