"""The Datasets and Presentation surfaces composed over one set of in-memory doubles, with a
dataset whose data a test can move.

`compose()` builds `dfo_platform.app.create_app` over both surfaces. The live stores
(`LiveVersionRecordWriter`, `LiveSnapshotReader`, `LiveHeadStore`, `LiveRecipeStore`,
`LiveDescriptionStore`, `LiveVersionLister`, `LiveArtifactInventory`, `LiveArtifactReferenceWriter`,
`LiveOperationStore`, `LiveArtifactStore`, `LiveFrameStore`) run their own SQL and storage calls
against SQLite and a fake GCS client. The graph, coverage index, and SDMX endpoint are doubles
that read one `Observations` table, so a change to that table is what a new upstream release is
to the service. Nothing reaches the network or a cloud service.

The dataset is `DATASET`: two StatVars of one family (`MEMBER_A`, `MEMBER_B`) over the
recipients in `RECIPIENTS`. It starts with `MEMBER_A` alone holding observations, for
`INITIAL_YEARS`. Mint it with `ComposedApp.mint_body(options)`.

Serve it over HTTP from the repository root, with no environment variable set:

    uv run --frozen --extra engine --extra eval --extra datasets --extra graph \
        python -m tests.live_refresh.composed_app --port 8093

It prints the base URL, the two principal values, and the dataset's `spec_id`. Signed artifact
URLs point back at the server (`/__test__/artifacts/...`), so a client that fetches one needs no
storage credential. Routes under `/__test__` exist only in this helper:

- `GET /__test__/state`: the dataset, its mint body, the principals, the observations, the
  dataset's head, and the queued operations.
- `POST /__test__/move`: move the data. Body, every member optional:
  `{"add_year": 2025, "recipients": ["country/KEN"], "stat_var": "LiveRefreshGrants",
  "set_value": {"recipient": "country/KEN", "year": 2024, "value": 5.0},
  "add_stat_var": true, "age_head_seconds": 1200}`. `recipients` narrows `add_year` and
  defaults to all four. `stat_var` narrows `add_year`, which defaults to every member holding
  observations, and names the series of `set_value`, which defaults to `MEMBER_A`. Every move
  is a new upstream release, so the source's `lastDataRefreshDate` moves with it, and the head
  is aged past the recheck interval so the next refresh compares its fingerprint.
- `POST /__test__/head`: `{"age_seconds": 1200}` ages the dataset's head and nothing else.
- `POST /__test__/budget`: `{"inline": false}` puts every rebuild over the inline budget, so a
  refresh that must rebuild answers `pending` and a mint answers 202 with an operation.
  `{"inline": true}` restores the default.
- `POST /__test__/worker/run`: runs every queued operation, as the operations worker would,
  and answers `{"ran": [<operation id>, ...]}`. The worker advances the head.
- `GET /__test__/artifacts/{tier}/{digest}`: the stored bytes a signed URL names.

    curl -s localhost:8093/__test__/state | jq .mint_body > mint.json
    curl -s -X POST localhost:8093/api/datasets -H 'content-type: application/json' -d @mint.json
    curl -s -X POST localhost:8093/__test__/move -H 'content-type: application/json' \
        -d '{"add_year": 2025}'
    curl -s -X POST localhost:8093/api/datasets/frames/refresh \
        -H 'content-type: application/json' -H 'x-dfo-principal: live-refresh-test-principal' \
        -d '{"spec_id": "<spec_id>", "content_version": "<content_version>"}'
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import shutil
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import httpx
import rfc8785
from fastapi import APIRouter, FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict

import origin_catalogue.service as catalogue_service_module
from artifact_store import ArtifactNotFoundError, LiveArtifactInventory, LiveArtifactStore, Tier
from assembler.catalogue import BakedReleaseRegistry
from assembler.fakes import FakeRevisionStore
from assembler.ports import ReleaseRegistry
from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, DatasetDeclaration, IndexStamp
from code_index.types import StatVarRow
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.app import datasets_surface
from datasets_api.config import DatasetsBuildSettings
from datasets_api.description_store import LiveDescriptionStore
from datasets_api.doors.definition import DefinitionDeps
from datasets_api.live_refresh import LiveRefreshPrincipal
from datasets_api.operation_store import LiveOperationStore
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.operations.worker import run_operation
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.reference_store import LiveArtifactReferenceWriter
from datasets_api.version_store import LiveHeadStore, LiveVersionLister, LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from dfo_platform.app import create_app
from graph_release.context import URI_ENV_VAR, bound_served_release
from graph_release.service import get_release_service, reset_release_service
from materialise import EnrichmentLookup
from materialise.graph import Facet
from materialise.rate_tables import RateTableCache
from origin_catalogue import bound_catalogue
from presentation_api.app import presentation_surface
from presentation_store import LiveFrameStore
from series_coverage.index import CoverageIndex
from series_coverage.schema import CoverageManifest, FacetMetadata, RoleNames
from series_coverage.shared import CoverageSnapshot
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from statvar_members import MemberEnumerator
from tests.artifact_store.conftest import FakeGCSClient
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.datasets_api._version_record_store import FakeVersionRecordDatasetStore
from tests.graph_release.support import build_session_release
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, fake_catalogue_service
from tests.series_coverage.support import coverage_table
from tests.spec_registry.conftest import FakeDatasetStore

LIVE_REFRESH_PRINCIPAL = "live-refresh-test-principal"
PUBLISH_PRINCIPAL = "datasets-publish-test-principal"
PRINCIPAL_HEADER = "x-dfo-principal"
LIVE = {PRINCIPAL_HEADER: LIVE_REFRESH_PRINCIPAL}
PUBLISH = {PRINCIPAL_HEADER: PUBLISH_PRINCIPAL}

MEMBER_A = "LiveRefreshGrants"
MEMBER_B = "LiveRefreshLoans"
MEMBERS = (MEMBER_A, MEMBER_B)
RECIPIENTS = ("country/GHA", "country/KEN", "country/TZA", "country/UGA")
INITIAL_YEARS = (2022, 2023, 2024)
GROUPBY = {"by": ["recipient"], "op": "sum"}
# The options the `ai` repository's data-mcp mints a frame with.
FRAME_OPTIONS = {"format": "parquet", "bundle": False, "layout": "frame"}

_AXIS = "liveRefreshFlow"
_AXIS_VALUES = {MEMBER_A: "LiveRefreshFlowGrants", MEMBER_B: "LiveRefreshFlowLoans"}
_FACET_ID = "facet1"
_PROVENANCE = "OECD_DAC2A"
_MEASUREMENT_METHOD = "GrantEquivalent"
_UNIT = "USD"
_SDMX_BASE_URL = "https://datacommons.one.org/core/api"
_LABELS = {
    "country/GHA": "Ghana",
    "country/KEN": "Kenya",
    "country/TZA": "Tanzania",
    "country/UGA": "Uganda",
    MEMBER_A: "Grants",
    MEMBER_B: "Loans",
    _PROVENANCE: "OECD DAC2a",
    _MEASUREMENT_METHOD: "Grant equivalent",
}
_IDENTITY = {
    "identity_version": "spec1",
    "shape": {
        "kind": "statvar_schema",
        "population_type": "DevelopmentFinance",
        "measured_property": "amount",
    },
    "slots": [
        {"property": _AXIS, "binding": {"state": "set", "refs": sorted(_AXIS_VALUES.values())}}
    ],
    "entities": {"recipient": list(RECIPIENTS)},
    "requested_window": {"kind": "all"},
}
DATASET = recompute_identity(_IDENTITY).spec_id

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
# Past `head_recheck_seconds` (900), so the next read compares the head's fingerprint.
_PAST_RECHECK = timedelta(minutes=20)
# The source's `lastDataRefreshDate` before the first move. Each move advances it a day.
_FIRST_REFRESH = datetime(2026, 9, 1, tzinfo=UTC)
_OPERATION_LEASE_SECONDS = 60
_EXECUTION_IDENTITY = "engine-test|grammar-1.0.0|reduction-test"


def _hash32(value: object) -> str:
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()[:32]


def derived_publish_body(
    *,
    root: Mapping[str, str],
    rows: Sequence[Mapping[str, object]],
    verb: str = "growth",
    anchor: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """A `POST /api/datasets/frames/publish` body for a frame derived by `verb` from `root`,
    a frame of `DATASET` minted with `FRAME_OPTIONS`, carrying the recipe the `ai` repository's
    presentation-mcp sends with a publish.

    The derived frame's `spec_id` hashes its op-spec, which names `root`, and its
    `content_version` hashes `rows`. `normal_form_hash` covers the verb and the root's dataset
    and rendering, so the same verb over another version of `DATASET` shares it. `replay_key`
    adds the root's pair, so each such frame has its own.
    """
    root_pair = {"spec_id": root["spec_id"], "content_version": root["content_version"]}
    parameters = {"periods": 1}
    op_spec = {
        "identity_version": "ops1",
        "verb": verb,
        "frame_ref": root_pair,
        "parameters": parameters,
    }
    frame = {
        "spec_id": f"ops_{_hash32(op_spec)}",
        "content_version": f"ocv_{_hash32(list(rows))}",
    }
    normal_form = {
        "identity_version": "ops1",
        "verb": verb,
        "parameters": parameters,
        "inputs": {
            "frame_ref": {"spec_id": root_pair["spec_id"], "rendering": dict(FRAME_OPTIONS)}
        },
    }
    normal_form_hash = f"rnf_{_hash32({'identity_version': 'rnf1', 'recipe': normal_form})}"
    replay_key = "rpk_" + _hash32(
        {
            "identity_version": "rpk1",
            "normal_form_hash": normal_form_hash,
            "roots": [root_pair],
            "execution_identities": [_EXECUTION_IDENTITY],
        }
    )
    return {
        "schema_version": 1,
        **frame,
        "role_keys": ["recipient"],
        "rows": [dict(row) for row in rows],
        "input_content_versions": [root_pair["content_version"]],
        "recipe": {
            "schema_version": 1,
            "normal_form_hash": normal_form_hash,
            "replay_key": replay_key,
            "execution_identity": _EXECUTION_IDENTITY,
            "tree": {
                "frame": frame,
                "op_spec": op_spec,
                "inputs": {"frame_ref": {"frame": root_pair, "root": 0}},
            },
            "roots": [
                {
                    "schema_version": 1,
                    "frame": root_pair,
                    "dataset": root_pair,
                    "options": dict(FRAME_OPTIONS),
                    "scope": {},
                }
            ],
            "anchor": None if anchor is None else dict(anchor),
        },
    }


def initial_value(*, stat_var: str, recipient: str, year: int) -> float:
    """The value a series holds at `year` until a test sets another: distinct per series and
    rising by 10 a year, so a chart of it is easy to check by eye."""
    base = 100.0 * (RECIPIENTS.index(recipient) + 1) + (1000.0 if stat_var == MEMBER_B else 0.0)
    return base + 10.0 * (year - INITIAL_YEARS[0])


def _member(stat_var: str) -> str:
    """The bare member name of `stat_var`, which may carry the `dcid:` prefix."""
    bare = stat_var.removeprefix("dcid:")
    if bare not in MEMBERS:
        raise ValueError(f"not a StatVar of the dataset: {stat_var!r}")
    return bare


class Observations:
    """The upstream's observations, by `(stat_var, recipient)` then year. The graph facets, the
    coverage index, and the SDMX rows are all read from it. `release` counts the moves."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._series: dict[tuple[str, str], dict[int, float]] = {
            (MEMBER_A, recipient): {
                year: initial_value(stat_var=MEMBER_A, recipient=recipient, year=year)
                for year in INITIAL_YEARS
            }
            for recipient in RECIPIENTS
        }
        self.release = 0

    def series(self) -> dict[tuple[str, str], dict[int, float]]:
        with self._lock:
            return {key: dict(points) for key, points in self._series.items()}

    def members(self) -> tuple[str, ...]:
        """The StatVars holding at least one observation."""
        held = {stat_var for stat_var, _recipient in self.series()}
        return tuple(member for member in MEMBERS if member in held)

    def add_year(
        self,
        year: int,
        *,
        recipients: Sequence[str] = RECIPIENTS,
        stat_var: str | None = None,
    ) -> None:
        """Give each of `recipients` an observation at `year`, for `stat_var` or for every
        member holding observations."""
        stat_vars = (_member(stat_var),) if stat_var is not None else self.members()
        unknown = sorted(set(recipients) - set(RECIPIENTS))
        if unknown:
            raise ValueError(f"not a recipient of the dataset: {unknown}")
        with self._lock:
            for member in stat_vars:
                for recipient in recipients:
                    self._series.setdefault((member, recipient), {})[year] = initial_value(
                        stat_var=member, recipient=recipient, year=year
                    )
            self.release += 1

    def set_value(
        self, *, recipient: str, year: int, value: float, stat_var: str = MEMBER_A
    ) -> None:
        """Replace one observation the series already holds."""
        with self._lock:
            points = self._series.get((_member(stat_var), recipient), {})
            if year not in points:
                raise ValueError(f"{stat_var} holds no {year} observation for {recipient}")
            points[year] = value
            self.release += 1

    def growth_rows(self, recipients: Sequence[str]) -> list[dict[str, object]]:
        """Frame rows holding the year-on-year growth, in percent, of `MEMBER_A` for each of
        `recipients`: what an analysis verb would compute from a plain frame of the dataset."""
        rows: list[dict[str, object]] = []
        for (stat_var, recipient), points in sorted(self.series().items()):
            if stat_var != MEMBER_A or recipient not in recipients:
                continue
            years = sorted(points)
            rows.extend(
                {
                    "stat_var": f"dcid:{stat_var}",
                    "recipient": recipient,
                    "time_period": str(year),
                    "value": round(100.0 * (points[year] / points[previous] - 1.0), 6),
                    "unit": "Percent",
                    "provenance": _PROVENANCE,
                }
                for previous, year in itertools.pairwise(years)
            )
        return rows

    def add_stat_var(self) -> None:
        """Give `MEMBER_B` an observation wherever `MEMBER_A` holds one, so a sum over the
        family has a second contributor."""
        with self._lock:
            for (member, recipient), points in list(self._series.items()):
                if member != MEMBER_A:
                    continue
                self._series[(MEMBER_B, recipient)] = {
                    year: initial_value(stat_var=MEMBER_B, recipient=recipient, year=year)
                    for year in points
                }
            self.release += 1

    def drop_stat_var(self) -> None:
        """Take every observation from `MEMBER_B`, so a sum over the family has one
        contributor again."""
        with self._lock:
            for key in [key for key in self._series if key[0] == MEMBER_B]:
                del self._series[key]
            self.release += 1


class _Facets(Mapping[tuple[str, str], list[Facet]]):
    """The graph's facets, read from the observations on every lookup."""

    def __init__(self, observations: Observations) -> None:
        self._observations = observations

    def __getitem__(self, key: tuple[str, str]) -> list[Facet]:
        years = sorted(self._observations.series()[key])
        return [
            Facet(
                facet_id=_FACET_ID,
                provenance_id=_PROVENANCE,
                measurement_method=_MEASUREMENT_METHOD,
                unit=_UNIT,
                obs_count=len(years),
                earliest_date=str(years[0]),
                latest_date=str(years[-1]),
            )
        ]

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self._observations.series())

    def __len__(self) -> int:
        return len(self._observations.series())


class _Arcs(Mapping[str, Mapping[str, tuple[str, ...]]]):
    """The graph's node arcs: `static`, and the source provenance dated by the observations'
    release, so every move refreshes it."""

    def __init__(
        self, observations: Observations, static: Mapping[str, Mapping[str, tuple[str, ...]]]
    ) -> None:
        self._observations = observations
        self._static = static

    def __getitem__(self, dcid: str) -> Mapping[str, tuple[str, ...]]:
        if dcid == _PROVENANCE:
            refreshed = _FIRST_REFRESH + timedelta(days=self._observations.release)
            return {"lastDataRefreshDate": (refreshed.date().isoformat(),)}
        return self._static[dcid]

    def __iter__(self) -> Iterator[str]:
        return iter((*self._static, _PROVENANCE))

    def __len__(self) -> int:
        return len(self._static) + 1


class _Coverage:
    """The coverage service the mint, the currency check, and the worker read, rebuilt from
    the observations on every read. Its stamp names the release, as a coverage build follows
    each upstream release."""

    def __init__(self, observations: Observations) -> None:
        self._observations = observations

    @property
    def stamp(self) -> str:
        return f"live-refresh-coverage-{self._observations.release}"

    def current(self) -> CoverageIndex:
        rows = [
            (stat_var, _FACET_ID, "country/USA", recipient, sorted(points))
            for (stat_var, recipient), points in sorted(self._observations.series().items())
        ]
        table, horizon = coverage_table(rows)
        return CoverageIndex(
            table,
            {member: RoleNames(donor="provider", recipient="recipient") for member in MEMBERS},
            facet_metadata={
                _FACET_ID: FacetMetadata(
                    provenance_id=_PROVENANCE,
                    measurement_method=_MEASUREMENT_METHOD,
                    unit=_UNIT,
                )
            },
            horizon=horizon,
        )

    def snapshot(self) -> CoverageSnapshot:
        index = self.current()
        stamp = self.stamp
        return CoverageSnapshot(
            index=index,
            manifest=CoverageManifest(
                stamp=stamp,
                build_id=f"{stamp}-build",
                built_at="1970-01-01T00:00:00Z",
                source="fake",
                row_counts={},
                files={},
                horizon=index.horizon,
            ),
            stamp=stamp,
        )


def _sdmx_client(observations: Observations) -> SdmxClient:
    """An SDMX client whose endpoint answers with the observations of the StatVars a request
    names."""

    def handler(request: httpx.Request) -> httpx.Response:
        requested = request.url.params.get("c[variableMeasured]", "").split(",")
        rows = "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{recipient},{_UNIT},{_MEASUREMENT_METHOD},"
            f"NotApplicable,{_PROVENANCE},{year},{value:g},,{_FACET_ID}\r\n"
            for (stat_var, recipient), points in sorted(observations.series().items())
            if stat_var in requested
            for year, value in sorted(points.items())
        )
        return httpx.Response(
            200,
            text=_CSV_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(base_url=_SDMX_BASE_URL, transport=httpx.MockTransport(handler))


def _graph(observations: Observations) -> FakeMintGraphReader:
    member_arcs = {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("DevelopmentFinance",),
        "measuredProperty": ("amount",),
        "observationProperties": ("recipient",),
        "constraintProperties": (_AXIS,),
    }
    return FakeMintGraphReader(
        arcs=_Arcs(
            observations,
            {
                **{
                    member: {
                        **member_arcs,
                        _AXIS: (_AXIS_VALUES[member],),
                        "name": (_LABELS[member],),
                    }
                    for member in MEMBERS
                },
                **{
                    recipient: {"typeOf": ("Place",), "name": (_LABELS[recipient],)}
                    for recipient in RECIPIENTS
                },
                _MEASUREMENT_METHOD: {"name": (_LABELS[_MEASUREMENT_METHOD],)},
            },
        ),
        facets=_Facets(observations),
        labels=_LABELS,
    )


def _code_index() -> CodeIndex:
    """A code index declaring the family's one summable axis. Its dataset reads a missing
    observation as zero, so a sum over the family stands while one member has no rows."""
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="https://fixture.example.org",
            exported_at="2026-09-15T00:00:00Z",
            code_counts={_AXIS: len(MEMBERS)},
            statvar_shape_counts={},
            content_hash="sha256:live-refresh-fixture",
        ),
        declarations=(
            CodeSystemDeclaration(
                code_system=_AXIS,
                hierarchical=False,
                disjoint_from_children=False,
                summable=True,
                absent_reading="not_declared",
                plain_name="flow",
            ),
        ),
        codes=tuple(
            CodeRow(
                dcid=_AXIS_VALUES[member],
                code_system=_AXIS,
                code=_LABELS[member],
                label=_LABELS[member],
                synonyms=(),
                parent=None,
            )
            for member in MEMBERS
        ),
        statvars=tuple(
            StatVarRow(
                dcid=member,
                population_type="DevelopmentFinance",
                measured_property="amount",
                stat_type=None,
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=("recipient",),
                constraint_properties={_AXIS: _AXIS_VALUES[member]},
                provenances=(_PROVENANCE,),
            )
            for member in MEMBERS
        ),
        datasets=(
            DatasetDeclaration(
                provenance=_PROVENANCE,
                label=_LABELS[_PROVENANCE],
                description="Aid disbursements by recipient.",
                opt_in=False,
                rank=0,
                absent_observation="zero",
                account="provider",
            ),
        ),
    )


class _SelfServedArtifactStore(LiveArtifactStore):
    """A `LiveArtifactStore` over a fake GCS client whose signed URLs name a route of the
    composed app, which serves the stored bytes (`/__test__/artifacts/{tier}/{digest}`)."""

    def __init__(self, *, inventory: LiveArtifactInventory, base_url: str) -> None:
        super().__init__(
            buckets={Tier.working: "working-bucket", Tier.pinned: "pinned-bucket"},
            inventory=inventory,
            client=FakeGCSClient(),
        )
        self._inventory_reads = inventory
        self.base_url = base_url

    def signed_location_url(
        self,
        digest: str,
        *,
        tier: Tier,
        bucket: str,
        object_name: str,
        ttl: timedelta,
        response_type: str,
        response_disposition: str,
    ) -> str:
        query = httpx.QueryParams(
            {"response_type": response_type, "response_disposition": response_disposition}
        )
        return f"{self.base_url}/__test__/artifacts/{tier.value}/{digest}?{query}"


class _SetValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipient: str
    year: int
    value: float


class _Move(BaseModel):
    model_config = ConfigDict(extra="forbid")

    add_year: int | None = None
    recipients: list[str] | None = None
    stat_var: str | None = None
    set_value: _SetValue | None = None
    add_stat_var: bool = False
    age_head_seconds: float | None = None


class _Head(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age_seconds: float


class _Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inline: bool


@dataclass
class ComposedApp:
    """The composed app and the doubles behind it. The methods are the levers a test pulls,
    and the `/__test__` routes call the same methods."""

    app: FastAPI
    observations: Observations
    version_store: FakeVersionRecordDatasetStore
    artifact_store: _SelfServedArtifactStore
    head_store: LiveHeadStore
    operation_store: LiveOperationStore
    task_client: FakeTaskClient
    settings: DatasetsBuildSettings
    _worker_deps: DefinitionDeps
    _default_budget_rows: int
    _operations_run: set[tuple[str, int]] = field(default_factory=set)

    @staticmethod
    def mint_body(options: Mapping[str, object] = FRAME_OPTIONS) -> dict[str, object]:
        """The `POST /api/datasets` body that mints `DATASET` under `options`."""
        return {
            "schema_version": 1,
            "input": {
                "kind": "spec_id",
                "spec_id": DATASET,
                "definition": {"schema_version": 1, "spec_id": DATASET, "identity": _IDENTITY},
            },
            "options": dict(options),
        }

    def move(
        self,
        *,
        add_year: int | None = None,
        recipients: Sequence[str] = RECIPIENTS,
        stat_var: str | None = None,
        set_value: tuple[str, int, float] | None = None,
        add_stat_var: bool = False,
        age_head: timedelta | None = None,
    ) -> None:
        """Move the dataset's data, which moves its source's refresh date, then age its head so
        the next refresh sees the move.

        `add_year` gives `recipients` an observation at that year, for `stat_var` or for every
        member holding observations. `set_value` is `(recipient, year, value)` and replaces an
        observation of `stat_var`, `MEMBER_A` when unstated. `add_stat_var` gives `MEMBER_B`
        observations. `age_head` defaults to just past the recheck interval.
        """
        if add_year is not None:
            self.observations.add_year(add_year, recipients=recipients, stat_var=stat_var)
        if set_value is not None:
            recipient, year, value = set_value
            self.observations.set_value(
                recipient=recipient,
                year=year,
                value=value,
                stat_var=stat_var if stat_var is not None else MEMBER_A,
            )
        if add_stat_var:
            self.observations.add_stat_var()
        self.age_head(age_head if age_head is not None else _PAST_RECHECK)

    def age_head(self, age: timedelta = _PAST_RECHECK) -> None:
        """Set the dataset head's confirmation and materialisation times `age` into the past.
        A dataset with no head is left alone."""
        aged = datetime.now(UTC) - age
        with self.version_store.transaction() as tx:
            tx.execute(
                "UPDATE datasets.subject_head SET confirmed_at = %s, materialised_at = %s "
                "WHERE subject_id = %s",
                (aged, aged, DATASET),
            )

    def head(self) -> str | None:
        """The content version the dataset's head names."""
        head = self.head_store.read_head(DATASET)
        return None if head is None else head.content_version

    def set_inline_budget(self, *, inline: bool) -> None:
        """With `inline` false, every rebuild is over the inline budget and is queued."""
        self.settings.sync_budget_rows = self._default_budget_rows if inline else 1

    def run_worker(self) -> list[str]:
        """Run every operation queued and not yet run, as the operations worker does, and
        return their ids."""
        ran: list[str] = []
        for operation_id, submission_count in list(self.task_client.enqueued):
            if (operation_id, submission_count) in self._operations_run:
                continue
            self._operations_run.add((operation_id, submission_count))
            claimed = self.operation_store.claim(
                operation_id,
                submission_count=submission_count,
                lease_seconds=_OPERATION_LEASE_SECONDS,
                final_delivery=False,
            )
            if claimed is None:
                continue
            with bound_catalogue(FakeCatalogue()):
                run_operation(
                    operation=claimed,
                    deps=self._worker_deps,
                    operation_store=self.operation_store,
                )
            ran.append(operation_id)
        return ran

    def pins(self) -> list[tuple[str, str, str]]:
        """Each live presentation pin as `(component principal, spec_id, content_version)`."""
        with self.version_store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT principal, subject_id, content_version FROM datasets.artifact_reference "
                "WHERE removed_at IS NULL ORDER BY created_at",
                (),
            )
        return [
            (
                cast(str, row["principal"]),
                cast(str, row["subject_id"]),
                cast(str, row["content_version"]),
            )
            for row in rows
        ]

    def state(self) -> dict[str, object]:
        return {
            "dataset": {
                "spec_id": DATASET,
                "stat_vars": [f"dcid:{member}" for member in MEMBERS],
                "stat_vars_with_observations": [
                    f"dcid:{member}" for member in self.observations.members()
                ],
                "recipients": list(RECIPIENTS),
                "groupby": GROUPBY,
                "head": self.head(),
            },
            "mint_body": self.mint_body(),
            "principals": {
                "header": PRINCIPAL_HEADER,
                "live_refresh": LIVE_REFRESH_PRINCIPAL,
                "publish": PUBLISH_PRINCIPAL,
            },
            "release": self.observations.release,
            "observations": [
                {
                    "stat_var": f"dcid:{stat_var}",
                    "recipient": recipient,
                    "values": {str(year): value for year, value in sorted(points.items())},
                }
                for (stat_var, recipient), points in sorted(self.observations.series().items())
            ],
            "inline_budget": self.settings.sync_budget_rows == self._default_budget_rows,
            "operations_queued": [
                operation_id
                for operation_id, submission_count in self.task_client.enqueued
                if (operation_id, submission_count) not in self._operations_run
            ],
        }


def _test_router(composed: ComposedApp) -> APIRouter:
    router = APIRouter(prefix="/__test__")

    @router.get("/state")
    def state() -> dict[str, object]:
        return composed.state()

    @router.post("/move")
    def move(body: _Move) -> dict[str, object]:
        set_value = body.set_value
        try:
            composed.move(
                add_year=body.add_year,
                recipients=RECIPIENTS if body.recipients is None else body.recipients,
                stat_var=body.stat_var,
                set_value=(
                    None
                    if set_value is None
                    else (set_value.recipient, set_value.year, set_value.value)
                ),
                add_stat_var=body.add_stat_var,
                age_head=(
                    None
                    if body.age_head_seconds is None
                    else timedelta(seconds=body.age_head_seconds)
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return composed.state()

    @router.post("/head")
    def head(body: _Head) -> dict[str, object]:
        composed.age_head(timedelta(seconds=body.age_seconds))
        return composed.state()

    @router.post("/budget")
    def budget(body: _Budget) -> dict[str, object]:
        composed.set_inline_budget(inline=body.inline)
        return composed.state()

    @router.post("/worker/run")
    def run_worker() -> dict[str, object]:
        return {"ran": composed.run_worker()}

    @router.get("/artifacts/{tier}/{digest}")
    def artifact(
        tier: Tier,
        digest: str,
        response_type: str = "application/octet-stream",
        response_disposition: str = "attachment",
    ) -> Response:
        try:
            data = composed.artifact_store.get(artifact_digest=digest, tier=tier)
        except ArtifactNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(
            content=data,
            media_type=response_type,
            headers={"content-disposition": response_disposition},
        )

    return router


def compose(
    *,
    base_url: str = "http://testserver",
    release_registry: ReleaseRegistry | None = None,
) -> ComposedApp:
    """Build the composed app over fresh doubles.

    `base_url` is the origin signed artifact URLs name: the default is the one
    `fastapi.testclient.TestClient` serves, and a served app passes its own address.
    `release_registry` defaults to the catalogue release baked into the installed
    `dfo-spec-kit-components`, the release a deployment pins.

    The origin catalogue service must already be resolvable, as it is under the test suite's
    `tests/conftest.py` and inside `offline_services()`.
    """
    observations = Observations()
    version_store = FakeVersionRecordDatasetStore()
    spec_store = FakeDatasetStore()
    inventory = LiveArtifactInventory(store=version_store)
    artifact_store = _SelfServedArtifactStore(inventory=inventory, base_url=base_url)
    snapshot_reader = LiveSnapshotReader(store=version_store)
    version_writer = LiveVersionRecordWriter(store=version_store)
    head_store = LiveHeadStore(store=version_store)
    recipe_store = LiveRecipeStore(store=version_store)
    description_store = LiveDescriptionStore(store=version_store)
    version_lister = LiveVersionLister(store=version_store)
    reference_writer = LiveArtifactReferenceWriter(store=version_store)
    operation_store = LiveOperationStore(store=version_store)
    task_client = FakeTaskClient()
    spec_reader = LiveSpecReader(store=spec_store)
    graph = _graph(observations)
    coverage = _Coverage(observations)
    code_index = _code_index()
    rate_cache = RateTableCache()
    settings = DatasetsBuildSettings(
        live_refresh_principal=LIVE_REFRESH_PRINCIPAL,
        first_party_principal=PUBLISH_PRINCIPAL,
    )
    revision_store = FakeRevisionStore()
    frame_store = LiveFrameStore(
        snapshots=snapshot_reader,
        artifacts=artifact_store,
        graph=graph,
        reference_writer=reference_writer,
        spec_reader=spec_reader,
        recipe_reader=recipe_store,
        code_index=code_index,
    )

    app = create_app(
        [
            datasets_surface(
                settings=settings,
                registry=LiveSpecRegistry(store=spec_store),
                spec_reader=spec_reader,
                graph=graph,
                sdmx=_sdmx_client(observations),
                artifact_store=artifact_store,
                version_writer=version_writer,
                head_writer=head_store,
                snapshot_reader=snapshot_reader,
                reference_writer=reference_writer,
                recipe_store=recipe_store,
                description_store=description_store,
                version_lister=version_lister,
                code_index=code_index,
                rate_cache=rate_cache,
                coverage=coverage,
                operation_store=operation_store,
                task_client=task_client,
                catalogue=fake_catalogue_service(),
            ),
            presentation_surface(
                revision_store=revision_store,
                frame_store=frame_store,
                release_registry=(
                    release_registry if release_registry is not None else BakedReleaseRegistry()
                ),
                snapshot_reader=snapshot_reader,
                recipe_reader=recipe_store,
                live_refresh=LiveRefreshPrincipal.resolve(settings=settings),
            ),
        ]
    )
    composed = ComposedApp(
        app=app,
        observations=observations,
        version_store=version_store,
        artifact_store=artifact_store,
        head_store=head_store,
        operation_store=operation_store,
        task_client=task_client,
        settings=settings,
        _worker_deps=DefinitionDeps(
            registry=LiveSpecRegistry(store=spec_store),
            spec_reader=spec_reader,
            graph=graph,
            sdmx=_sdmx_client(observations),
            artifact_store=artifact_store,
            materialiser_build="live-refresh-test-build",
            capability_version="dset_live_refresh_test",
            ceiling=1_000_000,
            member_concurrency=4,
            enumerator=MemberEnumerator(),
            enrichment=EnrichmentLookup(graph, parties=lambda: code_index.parties),
            code_index=code_index,
            rate_cache=rate_cache,
            coverage=coverage,
            version_writer=version_writer,
            snapshot_reader=snapshot_reader,
            head_recheck_interval=timedelta(seconds=settings.head_recheck_seconds),
            head_max_age=timedelta(seconds=settings.head_max_age_seconds),
            settings=settings,
        ),
        _default_budget_rows=settings.sync_budget_rows,
    )
    app.include_router(_test_router(composed))
    return composed


# What the surfaces read at construction and startup. The doubles stand in for every service
# these name. `offline_services()` sets them, and a test that enters the app's lifespan sets
# them too.
STARTUP_ENV = {
    "DFO_PUBLIC_SITE_URL": "https://site.example",
    "DATASETS_GRAPH_BASE_URL": _SDMX_BASE_URL,
    "DATASETS_SDMX_BASE_URL": _SDMX_BASE_URL,
    "DATASETS_ARTIFACT_BUCKET_WORKING": "working-bucket",
    "DATASETS_ARTIFACT_BUCKET_PINNED": "pinned-bucket",
}


@contextmanager
def offline_services() -> Iterator[None]:
    """Do for a process outside pytest what `tests/conftest.py` does for the test suite: set
    `STARTUP_ENV` where unset, install the fake origin catalogue service, publish the suite's
    graph release (`build_session_release`) to a temporary `GRAPH_RELEASE_URI`, and bind the
    release the process's release service serves, which every surface reads."""
    for name, value in STARTUP_ENV.items():
        os.environ.setdefault(name, value)
    release_dir = Path(tempfile.mkdtemp(prefix="live-refresh-graph-release-"))
    os.environ[URI_ENV_VAR] = (release_dir / "releases").as_uri()
    try:
        build_session_release(os.environ[URI_ENV_VAR])
        with (
            catalogue_service_module._singleton.poisoned_for_test(fake_catalogue_service()),
            bound_served_release(get_release_service()),
        ):
            yield
    finally:
        reset_release_service()
        shutil.rmtree(release_dir, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8093)
    args = parser.parse_args(argv)
    base_url = f"http://{args.host}:{args.port}"

    with offline_services():
        composed = compose(base_url=base_url)
        print(f"base URL: {base_url}")
        print(f"LIVE_REFRESH_PRINCIPAL={LIVE_REFRESH_PRINCIPAL}")
        print(f"DATASETS_FIRST_PARTY_PRINCIPAL={PUBLISH_PRINCIPAL}")
        print(f"principal header: {PRINCIPAL_HEADER}")
        print(f"dataset: {DATASET} (mint body at {base_url}/__test__/state)", flush=True)
        uvicorn.run(composed.app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
