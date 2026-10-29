"""The analysis runner: a recipe tree in, a stored derived frame and its format 2 ref out.

A request (`AnalysisRequest`) holds a `tree` and a `latest` flag. A step node names a `verb`, its
`arguments` and its `inputs`: `data`, and for a two-sided verb `other`. A root node is one of:

- a Spec (`{"spec_id": "spec_..."}`), read at its newest version;
- a dataset version (`{"spec_id", "content_version": "cv_..."}`);
- a derived frame (`{"spec_id": "ops_...", "content_version": "ocv_..."}`);
- a ref (`{"ref": <DatasetRef>}`), which must hold one member and reads as that member.

A Spec or dataset root carries a `rendering`, the mint options data-mcp records for a fetched
frame: `without_party_totals`; `columns`, the enrichment columns its frame is read with; and
`groupby`, the sum its frame was read with (`mint_models.GroupbyOptions`), such as a breakdown
or a sum over StatVars. Left out, `columns` is the place-type and party-kind columns the
dataset's roles take (`analysis_roots.enrichment_columns`), which the verbs need to keep group
totals out of a sum or rank. The rendering is part of the root's identity: its mint options
are the root record's `options`, which the recipe's normal form names.

Arguments are the engine's op-spec parameters (`dfo_analysis_engine.op_spec`) in the caller's
vocabulary. `data` and `other` fill the verb's frame slots: `frame_ref` and `right_frame_ref` for
`sum_of` and `join`, `frame_ref` and `denominator_frame_ref` for `share_of` and `quotient_of`, and
for `compare`, which subtracts its left side from its right, `data` the right and `other` the left,
so a call gives `data` minus `other`. `on` maps a column of `data` to the column of `other` holding
the same entities, or lists columns both name alike. `compare` and `sum_of` take it as it is. For
`share_of` and `quotient_of` it names `numerator_by` and `denominator_by`, and for `join` its
`join_keys` and, where a pair names two columns, `right_keys`. A column may be named by its role.
`place` reads as `observationAbout`, and `origin` or `destination` as the one column of that flow
axis the side's frame carries
(`dfo_platform.roles.role_key_axis`), so `{"destination": "place"}` pairs a recipient with a
place. A side's columns are those of the root its `data` chain reaches, since every verb names
its output's keys as `data` names them. A window bound or a threshold may be a rolling reference
(`{"anchor": "latest", "offset": -4}`) in place of a period, and a rolling overlay may leave out
its `population`: the runner fills both, the populations from the tree
(`dfo_analysis_engine.rolling.anchor_populations`) and each period from the one latest period of
the roots.

A request may name `anchor_roots`, the roots the latest period is taken over in place of the
tree's own: the roots `POST /api/datasets/analysis/anchor` was asked about, such as both sides of
a pairing when the tree grows one side. The runner reads them as it reads the tree's roots, at
the versions `latest` gives, and takes the period over them by the anchor route's function
(`dfo_analysis_engine.anchor.latest_period`), so a rolling literal the anchor route gave is the
period the run checks it against. They are the top node's `anchor_roots` in the recipe's tree,
so the recipe's normal form names them, its `roots` hold them after the tree's own, and a replay
of the stored recipe takes its period over them again.

A derived root is expanded into its stored recipe, so the runner computes the whole chain from
its root datasets. A request whose whole `tree` is a derived root computes that recipe again
under its own ids, which is how a client reads a derived frame on its datasets' newest versions.
With `latest` false every root is read at the version it names. With `latest` true a Spec or dataset root is read at its dataset's current version of the same rendering, and
each rolling literal follows the latest period of that data. Both read their roots through one
`RootSource`.

A run (`run_analysis`) resolves every root first, so a root still building answers
`AnalysisPending` with every rebuild it waits on submitted. It then holds one materialise permit
while it reads the root frames, evaluates the tree (`dfo_analysis_engine.evaluate`), takes the
output digest (`datasets_api.output_digest`) and stores the result: the frame through
`publish_built_frame`, as its canonical frame and its bare frame-layout Parquet file with the
step list its description record names (`described_steps`), and its recipe through the recipe
store, unless the frame file is stored and servable already, in which
case it writes the recipe alone, which the store keeps once. A rerun of one tree on the same
roots therefore stores nothing new and answers the same ref. The ref's `artifact.digest` names
the frame file (`render.FRAME_FILE_OPTIONS`), as a dataset ref names the file it was read as. The
recipe carries `left_out`, every series a step of the tree dropped for a data gap (`left_out_of`), which the derived reads serve
(`datasets_api.derived`). `compute_analysis` computes a tree as a run does and stores nothing.

A run is prepared (`prepare_analysis`) before any frame is read: its roots resolved and its
request key (`request_key`) taken, a hash of the working tree as sent, each resolved root's
record, `latest`, the engine's execution identity and the answer format (`ANSWER_FORMAT`). The
run (`run_prepared`) stores its 200 body under that key (`RecipeStore.write_answer`) once its
frame is stored, so a later request with the same key answers the stored body
(`stored_answer`) while the frame file it names is still served, reading no frame.

Bounds: a tree holds at most `max_nodes` nodes after expansion, its root frames at most
`max_root_rows` rows together, and a `join` writes at most `max_join_rows`. Every other verb
writes no more rows than its `data` side holds, so these bound every intermediate frame. The root
rows are checked against the counts the roots' versions record before any frame is read, and
against the rows read so far after each read. The computed frame holds at most the rows the
publish door stores (`PUBLISH_MAX_ROWS`), checked as soon as the tree's top step has computed
it, before it is digested or stored. Each refuses `over_ceiling`. The door bounds the run's
time.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Protocol, Self

import rfc8785
from dfo_descriptions import MAX_STEPS_BYTES, StepsError, check_steps, steps_bytes
from dfo_frame_grammar import Frame, FrameError, RollingError, shift_period
from dfo_frame_grammar.grammar import TIME_PERIOD_COLUMN
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    JsonValue,
    Tag,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from artifact_store import ArtifactStore
from dataset_store.clock import Clock
from dataset_store.snapshot import SnapshotReader, select_servable_locations
from datasets_api.doors.publish import (
    PUBLISH_MAX_ROWS,
    publish_built_frame,
    write_frame_recipe,
)
from datasets_api.mint_models import GroupbyOptions
from datasets_api.output_digest import minted_variable_names, output_digest, stat_vars
from datasets_api.publish_models import (
    MAX_RECIPE_BYTES,
    MAX_RECIPE_LEFT_OUT_SERIES,
    MAX_STEPS,
    Recipe,
    RecipeAnchor,
    RecipeDroppedSeries,
    RecipeLeftOut,
    RecipeStepRef,
)
from datasets_api.recipe_store import AnswerRecord, RecipeStore
from datasets_api.render import frame_file_location
from datasets_api.versions import VersionRecordWriter
from description.steps import Step
from dfo_analysis_engine.anchor import ResolvedPeriods, latest_period
from dfo_analysis_engine.call_steps import shed_coverage
from dfo_analysis_engine.calls import admit
from dfo_analysis_engine.evaluate import (
    Evaluation,
    Policy,
    RootFrame,
    evaluate,
    latest_period_of,
)
from dfo_analysis_engine.execution_identity import current_execution_identity
from dfo_analysis_engine.frame_ref import FrameRef
from dfo_analysis_engine.op_spec import Populations, Verb
from dfo_analysis_engine.recipe import (
    RECIPE_SCHEMA_VERSION,
    derived_nodes,
    normal_form,
    normal_form_hash,
    replay_key,
    rolling_literals,
    root_identity,
)
from dfo_analysis_engine.refusal import AnalysisRefused
from dfo_analysis_engine.refusal import Refusal as EngineRefusal
from dfo_analysis_engine.rolling import (
    PairingKeysUnstatedError,
    PopulationEmptyError,
    PopulationLevelColumnMissingError,
    anchor_populations,
    canonical_populations,
)
from dfo_platform.config import PublicSiteSettings
from dfo_platform.problem import Refusal
from dfo_platform.roles import role_key_axis
from materialise import EnrichmentLookup

logger = logging.getLogger(__name__)

PLACE_ROLE_WORD = "place"
PLACE_COLUMN = "observationAbout"

# The frame slots of each verb that `data` and `other` fill, in that order.
_SLOTS: Mapping[str, tuple[str, str | None]] = {
    Verb.AGGREGATE: ("frame_ref", None),
    Verb.GROWTH: ("frame_ref", None),
    Verb.RANK: ("frame_ref", None),
    Verb.FILTER: ("frame_ref", None),
    Verb.SUM_OF: ("frame_ref", "right_frame_ref"),
    Verb.JOIN: ("frame_ref", "right_frame_ref"),
    Verb.SHARE_OF: ("frame_ref", "denominator_frame_ref"),
    Verb.QUOTIENT_OF: ("frame_ref", "denominator_frame_ref"),
    Verb.COMPARE: ("right_frame_ref", "frame_ref"),
}

_PRIMARY_SLOT = "frame_ref"
_ROLLING = "rolling"
_ANCHOR_ROOTS = "anchor_roots"
_WINDOW_BOUNDS = ("window_start", "window_end")
_DEFAULT_RULE = "half1"

DEFAULT_MAX_NODES = 64
DEFAULT_MAX_ROOT_ROWS = 2_000_000
DEFAULT_MAX_JOIN_ROWS = 250_000


# --- Request -------------------------------------------------------------------------------------


# The mint options a root is read with: the bare frame-layout Parquet, and the rendering.
ROOT_MINT_OPTIONS = frozenset(
    {"format", "bundle", "layout", "columns", "without_party_totals", "groupby"}
)


class RootRendering(BaseModel):
    """How a dataset root's frame is read. `columns` left out takes the enrichment its
    dataset's roles need (`analysis_roots.enrichment_columns`); `[]` reads the stored frame
    alone. `groupby` reads the frame summed as a mint with that `options.groupby` sums it."""

    model_config = ConfigDict(extra="forbid")

    without_party_totals: bool = False
    columns: tuple[Annotated[str, Field(min_length=1, max_length=200)], ...] | None = Field(
        default=None, max_length=32
    )
    groupby: GroupbyOptions | None = None

    @classmethod
    def of_options(cls, options: Mapping[str, Any]) -> RootRendering:
        """The rendering a root record's mint `options` state, `columns` as listed."""
        return cls(
            without_party_totals=options.get("without_party_totals") is True,
            columns=tuple(options.get("columns") or ()),
            groupby=options.get("groupby"),
        )


class SpecRoot(BaseModel):
    """A Spec, read at its dataset's newest version."""

    model_config = ConfigDict(extra="forbid")

    spec_id: str = Field(pattern=r"^spec_[0-9a-f]{32}$")
    rendering: RootRendering = Field(default_factory=RootRendering)


class DatasetRoot(BaseModel):
    """A dataset version, as a ref member names it."""

    model_config = ConfigDict(extra="forbid")

    spec_id: str = Field(pattern=r"^spec_[0-9a-f]{32}$")
    content_version: str = Field(pattern=r"^cv_[0-9a-f]{32}$")
    rendering: RootRendering = Field(default_factory=RootRendering)


class DerivedRoot(BaseModel):
    """A frame an earlier analysis stored."""

    model_config = ConfigDict(extra="forbid")

    spec_id: str = Field(pattern=r"^ops_[0-9a-f]{32}$")
    content_version: str = Field(pattern=r"^ocv_[0-9a-f]{32}$")


class _RefSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    spec_id: str = Field(pattern=r"^(?:spec|ops)_[0-9a-f]{32}$")


class _RefRendering(BaseModel):
    model_config = ConfigDict(extra="ignore")

    without_party_totals: bool = False
    groupby: GroupbyOptions | None = None


class _RefArtifact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rendering: _RefRendering = Field(default_factory=_RefRendering)


class _RefMember(BaseModel):
    """A ref member, held to the pair a root of its kind takes (`DatasetRoot`, `DerivedRoot`),
    so a malformed member refuses as the request does."""

    model_config = ConfigDict(extra="ignore")

    spec: _RefSpec
    content_version: str = Field(pattern=r"^(?:cv|ocv)_[0-9a-f]{32}$")
    artifact: _RefArtifact | None = None

    @model_validator(mode="after")
    def _pair_of_one_kind(self) -> Self:
        if self.spec.spec_id.startswith("ops_") != self.content_version.startswith("ocv_"):
            raise ValueError(
                "a member pairs an `ops_` id with an `ocv_` version, and a `spec_` id with a "
                "`cv_` version"
            )
        return self


class _Ref(BaseModel):
    model_config = ConfigDict(extra="ignore")

    format_version: Literal[1, 2]
    members: list[_RefMember] = Field(min_length=1, max_length=256)


class RefRoot(BaseModel):
    """A DatasetRef, read as its one member."""

    model_config = ConfigDict(extra="forbid")

    ref: _Ref


def _node_kind(value: object) -> str:
    """The tag of a node: `step`, `ref`, `derived`, `dataset` or `spec`, read from a parsed
    node's class or a JSON node's keys."""
    if isinstance(value, BaseModel):
        return _MODEL_KINDS[type(value).__name__]
    if not isinstance(value, Mapping):
        return "spec"
    if "verb" in value:
        return "step"
    if "ref" in value:
        return "ref"
    if str(value.get("spec_id", "")).startswith("ops_"):
        return "derived"
    return "dataset" if "content_version" in value else "spec"


_MODEL_KINDS: Mapping[str, str] = {
    "StepNode": "step",
    "RefRoot": "ref",
    "DerivedRoot": "derived",
    "DatasetRoot": "dataset",
    "SpecRoot": "spec",
}

Node = Annotated[
    Annotated["StepNode", Tag("step")]
    | Annotated[RefRoot, Tag("ref")]
    | Annotated[DerivedRoot, Tag("derived")]
    | Annotated[DatasetRoot, Tag("dataset")]
    | Annotated[SpecRoot, Tag("spec")],
    Discriminator(_node_kind),
]


class StepInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: Node
    other: Node | None = None


class StepNode(BaseModel):
    """One verb over its inputs. `arguments` are the verb's op-spec parameters in the caller's
    vocabulary (see the module docstring)."""

    model_config = ConfigDict(extra="forbid")

    verb: Verb
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    inputs: StepInputs


StepInputs.model_rebuild()


class AnalysisRequest(BaseModel):
    """`POST /api/datasets/analysis`: the tree to compute, and whether its roots follow their
    datasets' newest versions. A `tree` that is a derived root alone computes that frame's
    stored recipe again, so `latest` reads it on its datasets' newest versions under the
    recipe's own ids."""

    model_config = ConfigDict(extra="forbid")

    tree: Annotated[
        Annotated[StepNode, Tag("step")] | Annotated[DerivedRoot, Tag("derived")],
        Discriminator(_node_kind),
    ]
    latest: bool = False
    anchor_roots: (
        list[
            Annotated[
                Annotated[SpecRoot, Tag("spec")] | Annotated[DatasetRoot, Tag("dataset")],
                Discriminator(_node_kind),
            ]
        ]
        | None
    ) = Field(default=None, min_length=1, max_length=DEFAULT_MAX_NODES)


# --- Refusals and outcomes -----------------------------------------------------------------------


class SeveralMembers(Refusal):
    """A step input names a ref holding several readings. Maps to `several_members` (422)."""

    code = "several_members"

    def __init__(self, *, verb: str, side: str, spec_ids: Sequence[str]) -> None:
        self.verb = verb
        self.side = side
        self.spec_ids = tuple(spec_ids)
        super().__init__(
            f"{verb} takes one reading per side; the Dataset at {side} holds "
            f"{len(spec_ids)}. Fetch one of its specs on its own, such as "
            f"fetch({spec_ids[0]!r}), and pass that Dataset"
        )

    @property
    def fix(self) -> dict[str, Any]:
        return {"parameter": self.side, "call": "fetch", "spec_ids": list(self.spec_ids)}


class FrameUnavailable(Refusal):
    """A root the runner cannot read. Each subclass names why under a Datasets code."""

    def __init__(self, *, spec_id: str, content_version: str, reason: str) -> None:
        self.spec_id = spec_id
        self.content_version = content_version
        self.reason = reason
        super().__init__(
            f"the frame {spec_id}, {content_version} cannot be read: {reason}. Run the "
            "analysis with latest=True to read its dataset's current version"
        )

    @property
    def fix(self) -> dict[str, Any]:
        return {"spec_id": self.spec_id, "content_version": self.content_version, "latest": True}


class RootNotStored(FrameUnavailable):
    """No version is stored at the root's pair. Maps to `snapshot_not_found`."""

    code = "snapshot_not_found"


class RootExpired(FrameUnavailable):
    """The root's stored frame or file has aged out. Maps to `artifact_expired`."""

    code = "artifact_expired"


class RootRecipeMissing(FrameUnavailable):
    """A derived root holds no stored recipe to compute it from. Maps to `recipe_not_found`."""

    code = "recipe_not_found"


class RootRenderingUnreadable(FrameUnavailable):
    """A stored recipe reads a root with options the runner does not read, such as a pivot or
    a scope. Maps to `rendering_not_recorded`."""

    code = "rendering_not_recorded"


def recorded_dataset_root(record: Mapping[str, Any]) -> DatasetRoot:
    """The dataset root a stored recipe's root record `record` names: its `dataset`, read with
    the rendering its mint `options` state.

    Raises:
        RootRenderingUnreadable: The record has a scope, an option outside
            `ROOT_MINT_OPTIONS`, or a rendering this build does not parse.
    """
    options = record.get("options") or {}
    spec_id, content_version = record["dataset"]["spec_id"], record["dataset"]["content_version"]
    unread = sorted(set(options) - ROOT_MINT_OPTIONS) + (["scope"] if record.get("scope") else [])
    if unread:
        raise RootRenderingUnreadable(
            spec_id=spec_id,
            content_version=content_version,
            reason=f"its recipe reads it with options the runner does not read: {unread}",
        )
    try:
        rendering = RootRendering.of_options(options)
    except ValidationError:
        raise RootRenderingUnreadable(
            spec_id=spec_id,
            content_version=content_version,
            reason="its recipe reads it with a rendering this runner does not parse",
        ) from None
    return DatasetRoot(spec_id=spec_id, content_version=content_version, rendering=rendering)


class AnalysisTooLarge(Refusal):
    """A tree, its root frames or its output pass a bound. Maps to `over_ceiling` (422)."""

    code = "over_ceiling"

    def __init__(self, *, ceiling_kind: str, count: int, cap: int) -> None:
        self.ceiling_kind = ceiling_kind
        self.count = count
        self.cap = cap
        noun = _CEILING_NOUNS[ceiling_kind]
        super().__init__(f"the analysis holds {count} {noun}, over the ceiling of {cap}")


_CEILING_NOUNS = {
    "analysis_nodes": "nodes",
    "analysis_root_rows": "root rows",
    "analysis_output_rows": "output rows",
    "analysis_recipe_bytes": "bytes of recipe",
}


class AnalysisStoreDiverges(Refusal):
    """The pair a run computed already holds a recipe, a replay key or frame bytes that differ
    from the run's own. The pair is minted from the run's inputs, so this is a fault of the
    runner or the stores. Maps to non-retryable `internal_error` (500),
    logged with both identities."""

    code = "internal_error"

    def __init__(self, *, spec_id: str, content_version: str) -> None:
        self.spec_id = spec_id
        self.content_version = content_version
        super().__init__(
            f"spec_id {spec_id!r} content_version {content_version!r} already holds a stored "
            "frame or recipe that differs from this run's own"
        )


class AnalysisPending(Exception):
    """A root is still building. `operations` are the queued rebuilds, and the caller retries
    after `retry_after_s`."""

    def __init__(self, *, operations: Sequence[str], retry_after_s: float) -> None:
        self.operations = tuple(operations)
        self.retry_after_s = retry_after_s
        super().__init__(f"the analysis waits on {len(operations)} rebuild(s)")


# --- Roots ---------------------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ResolvedRoot:
    """A root at the version it is read at.

    `frame` is the pair the op-specs name the frame by: the dataset pair for the stored frame,
    else the pair `analysis_roots.frame_ref_for` derives from its columns. `record` is the mint
    record a recipe's `roots` holds for it (`dataset`, `options`, and the rest).
    `rendering` is the ref format 2 root's rendering. `rows` is the frame's row count as its
    version records it, `None` where the reader has none. `source` is the reader's own handle."""

    frame: FrameRef
    dataset: FrameRef
    record: Mapping[str, Any]
    rendering: Mapping[str, Any]
    rows: int | None = None
    source: object = field(default=None, compare=False)


class RootSource(Protocol):
    """Where the runner reads its roots from (`analysis_roots.LiveRootSource`)."""

    def resolve(
        self, root: SpecRoot | DatasetRoot, *, latest: bool, trace_id: str | None
    ) -> ResolvedRoot:
        """The version `root` is read at.

        Raises:
            AnalysisPending: The root's dataset is still building.
            FrameUnavailable: The version is not held.
        """
        ...

    def read(self, root: ResolvedRoot) -> Frame:
        """The frame of `root`. Runs under the run's materialise permit.

        Raises:
            FrameUnavailable: The frame is no longer held.
        """
        ...

    def recipe(self, *, spec_id: str, content_version: str) -> Mapping[str, Any] | None:
        """The stored recipe of a derived frame, or `None`."""
        ...

    def dataset_root(self, record: Mapping[str, Any]) -> DatasetRoot:
        """The dataset root a stored recipe's root record `record` names.

        Raises:
            FrameUnavailable: The record names a rendering this runner cannot read.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class AnalysisStores:
    """What a run writes to and reads stored frames from."""

    snapshot_reader: SnapshotReader
    version_writer: VersionRecordWriter
    artifact_store: ArtifactStore
    recipe_store: RecipeStore
    enrichment: EnrichmentLookup
    site: PublicSiteSettings
    now: Clock


@dataclass(frozen=True, kw_only=True)
class AnalysisBounds:
    max_nodes: int = DEFAULT_MAX_NODES
    max_root_rows: int = DEFAULT_MAX_ROOT_ROWS
    max_join_rows: int = DEFAULT_MAX_JOIN_ROWS


@dataclass(frozen=True, kw_only=True)
class AnalysisResult:
    """A run's answer: the stored frame's pair and digests, its top step's summary, and the
    format 2 ref naming it."""

    evaluation: Evaluation
    frame_content_version: str
    output_digest: str
    artifact_digest: str
    roots: tuple[ResolvedRoot, ...]
    resolved_periods: tuple[dict[str, Any], ...]
    created: bool

    def ref(self) -> dict[str, Any]:
        """The format 2 ref naming the stored frame."""
        evaluation = self.evaluation
        recipe = _recipe_hashes(evaluation, self.roots)
        return {
            "format_version": 2,
            "members": [
                {
                    "parts": [],
                    "sentence": _sentence(evaluation),
                    "spec": {"spec_id": evaluation.ref.spec_id, "recipe": recipe},
                    "content_version": evaluation.ref.content_version,
                    "output_digest": self.output_digest,
                    "roots": _unique(
                        {
                            "spec_id": root.dataset.spec_id,
                            "content_version": root.dataset.content_version,
                            "rendering": dict(root.rendering),
                        }
                        for root in self.roots
                    ),
                    "execution_identity": evaluation.execution_identity.canonical_string(),
                    "resolved_periods": list(self.resolved_periods),
                    "artifact": {
                        "digest": self.artifact_digest,
                        "rendering": {"format": "parquet", "bundle": False, "layout": "frame"},
                    },
                }
            ],
        }

    def body(self) -> dict[str, Any]:
        """The 200 body of `POST /api/datasets/analysis` (`doors.analysis.AnalysisReady`)."""
        summary = self.evaluation.summary
        return {
            "status": "ready",
            "ref": self.ref(),
            "frame_content_version": self.frame_content_version,
            "output_digest": self.output_digest,
            "summary": summary,
            "rows": summary["rows"],
            "role_keys": summary["role_keys"],
            "stat_var_count": summary["stat_var_count"],
            "exclusions": summary["exclusions"],
            "time": summary["time"],
            "execution_identity": summary["execution_identity"],
            "resolved_periods": list(self.resolved_periods),
        }


def _unique(items: Iterator[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in items:
        if item not in kept:
            kept.append(item)
    return kept


def _sentence(evaluation: Evaluation) -> str:
    verbs = [str(node["op_spec"]["verb"]) for node in derived_nodes(evaluation.tree)]
    return f"Computed by DFO: {', then '.join(reversed(verbs))}."


# --- The plan ------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class _Plan:
    """The tree with every root numbered by its first reach, derived roots expanded. Nodes are
    counted as the walk reaches them, so a tree past `max_nodes` refuses at the first node over
    it, before the rest is expanded or any further recipe is read."""

    max_nodes: int
    nodes: int = 0
    indices: dict[str, int] = field(default_factory=dict)
    roots: list[SpecRoot | DatasetRoot] = field(default_factory=list)

    def count(self) -> None:
        """Count one node.

        Raises:
            AnalysisTooLarge: The node is past `max_nodes`.
        """
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise AnalysisTooLarge(
                ceiling_kind="analysis_nodes", count=self.nodes, cap=self.max_nodes
            )

    def root(self, root: SpecRoot | DatasetRoot) -> dict[str, Any]:
        """Count `root` and number it, once per distinct root."""
        self.count()
        key = f"{type(root).__name__}:{root.model_dump_json()}"
        if key not in self.indices:
            self.indices[key] = len(self.roots)
            self.roots.append(root)
        return {"root": self.indices[key]}


def _member_root(member: _RefMember) -> DerivedRoot | DatasetRoot:
    if member.spec.spec_id.startswith("ops_"):
        return DerivedRoot(spec_id=member.spec.spec_id, content_version=member.content_version)
    rendering = member.artifact.rendering if member.artifact is not None else _RefRendering()
    return DatasetRoot(
        spec_id=member.spec.spec_id,
        content_version=member.content_version,
        rendering=RootRendering(
            without_party_totals=rendering.without_party_totals, groupby=rendering.groupby
        ),
    )


def _plan_node(node: Node, plan: _Plan, *, source: RootSource, verb: str, side: str) -> Any:
    """`node` as the runner's working tree: a step as `{"step", "inputs"}`, a stored recipe's
    node as `{"op_spec", "inputs"}`, a root as `{"root": i}`. A ref or a derived root counts as
    the nodes it stands for."""
    match node:
        case StepNode():
            plan.count()
            inputs = {
                "data": _plan_node(
                    node.inputs.data, plan, source=source, verb=node.verb, side="data"
                )
            }
            if node.inputs.other is not None:
                inputs["other"] = _plan_node(
                    node.inputs.other, plan, source=source, verb=node.verb, side="other"
                )
            return {"step": node, "inputs": inputs}
        case RefRoot():
            members = node.ref.members
            if len(members) > 1:
                raise SeveralMembers(
                    verb=str(verb), side=side, spec_ids=[m.spec.spec_id for m in members]
                )
            return _plan_node(
                _member_root(members[0]),
                plan,
                source=source,
                verb=verb,
                side=side,
            )
        case DerivedRoot():
            stored = source.recipe(spec_id=node.spec_id, content_version=node.content_version)
            if stored is None:
                raise RootRecipeMissing(
                    spec_id=node.spec_id,
                    content_version=node.content_version,
                    reason="no recipe is stored for it",
                )
            records = stored["roots"]
            return _expand(stored["tree"], records, plan, source=source)
        case SpecRoot() | DatasetRoot():
            return plan.root(node)


def _expand(
    tree: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    plan: _Plan,
    *,
    source: RootSource,
) -> dict[str, Any]:
    """A stored recipe's `tree` as working nodes, its roots read again (`dataset_root`)."""
    if "root" in tree:
        return plan.root(source.dataset_root(records[tree["root"]]))
    plan.count()
    op_spec = tree["op_spec"]
    expanded: dict[str, Any] = {
        "op_spec": {"verb": op_spec["verb"], "parameters": copy.deepcopy(op_spec["parameters"])},
        "inputs": {
            slot: _expand(child, records, plan, source=source)
            for slot, child in tree["inputs"].items()
        },
    }
    if anchors := tree.get(_ANCHOR_ROOTS):
        expanded[_ANCHOR_ROOTS] = [_expand(root, records, plan, source=source) for root in anchors]
    return expanded


# --- Caller vocabulary to op-specs ---------------------------------------------------------------


def role_columns(frame: Frame) -> tuple[str, ...]:
    """The role columns of `frame`, between `stat_var` and `time_period`."""
    return frame.columns[1 : frame.columns.index(TIME_PERIOD_COLUMN)]


def column_for(word: str, columns: Sequence[str], *, parameter: str) -> str:
    """The column of `columns` that `word` names: itself when a column is so named, else
    `observationAbout` for `place`, else the one column on the flow axis `origin` or
    `destination` names. Any other word passes through for the engine to judge.

    Raises:
        AnalysisRefused: `role_ambiguous`, when several columns sit on the axis.
    """
    if word in columns:
        return word
    if word == PLACE_ROLE_WORD:
        return PLACE_COLUMN
    on_axis = [column for column in columns if role_key_axis(column) == word]
    if len(on_axis) > 1:
        raise AnalysisRefused(
            EngineRefusal.role_ambiguous(
                parameter=parameter, columns=[word], available=list(columns)
            )
        )
    return on_axis[0] if on_axis else word


@dataclass(frozen=True, kw_only=True)
class _Sides:
    """The role columns each frame slot of one step reads."""

    by_slot: Mapping[str, Sequence[str]]

    def name(self, slot: str, word: object, *, parameter: str) -> object:
        if not isinstance(word, str):
            return word
        return column_for(
            word, self.by_slot.get(slot, self.by_slot[_PRIMARY_SLOT]), parameter=parameter
        )

    def names(self, slot: str, words: object, *, parameter: str) -> object:
        if isinstance(words, list):
            return [self.name(slot, word, parameter=parameter) for word in words]
        return self.name(slot, words, parameter=parameter)

    def keys(self, slot: str, binding: object, *, parameter: str) -> object:
        if not isinstance(binding, Mapping):
            return binding
        return {self.name(slot, key, parameter=parameter): value for key, value in binding.items()}

    def pairing(self, left: str, right: str, pairs: object, *, parameter: str) -> object:
        if isinstance(pairs, list):
            pairs = {word: word for word in pairs if isinstance(word, str)}
        if not isinstance(pairs, Mapping):
            return pairs
        return {
            self.name(left, key, parameter=parameter): self.name(right, value, parameter=parameter)
            for key, value in pairs.items()
        }


def _translated(verb: Verb, arguments: Mapping[str, Any], sides: _Sides) -> dict[str, Any]:
    """`arguments` with every column named in the caller's vocabulary named by its column."""
    data_slot, other_slot = _SLOTS[verb]
    other = other_slot or data_slot
    out = copy.deepcopy(dict(arguments))
    for name in ("group_by", "entity_by", "rank_by", "numerator_by", "join_keys"):
        if name in out:
            out[name] = sides.names(_PRIMARY_SLOT, out[name], parameter=name)
    for name, slot in (
        ("left_entity", _PRIMARY_SLOT),
        ("right_entity", "right_frame_ref"),
        ("numerator_entity", _PRIMARY_SLOT),
        ("denominator_entity", "denominator_frame_ref"),
    ):
        if name in out:
            out[name] = sides.keys(slot, out[name], parameter=name)
    if "on" in out:
        out["on"] = sides.pairing(data_slot, other, out["on"], parameter="on")
        if verb in (Verb.SHARE_OF, Verb.QUOTIENT_OF, Verb.JOIN):
            _keyed(verb, out)
    for name in ("denominator_by", "right_keys"):
        if name in out:
            out[name] = sides.pairing(_PRIMARY_SLOT, other, out[name], parameter=name)
    for predicate in out.get("predicates") or ():
        if isinstance(predicate, dict) and "column" in predicate:
            predicate["column"] = sides.name(
                _PRIMARY_SLOT, predicate["column"], parameter="predicates"
            )
    return out


def _keyed(verb: Verb, arguments: dict[str, Any]) -> None:
    """`on` as the keys of a keyed verb: `numerator_by` and `denominator_by` for `share_of` and
    `quotient_of`, `join_keys` and, where a pair names two columns, `right_keys` for `join`."""
    on = arguments.pop("on")
    if not isinstance(on, Mapping):
        raise AnalysisRefused(EngineRefusal.op_spec_invalid(detail="on: a mapping or a list"))
    if verb is Verb.JOIN:
        arguments["join_keys"] = sorted({*arguments.get("join_keys", ()), *on})
        if any(key != value for key, value in on.items()):
            arguments["right_keys"] = {**arguments.get("right_keys", {}), **on}
        return
    arguments["numerator_by"] = sorted({*arguments.get("numerator_by", ()), *on})
    arguments["denominator_by"] = {**arguments.get("denominator_by", {}), **on}


def _is_rolling_ref(value: object) -> bool:
    return isinstance(value, Mapping) and value.get("anchor") == "latest"


def _offset(reference: Mapping[Any, Any]) -> int:
    """The whole-number offset of a rolling reference.

    Raises:
        AnalysisRefused: `op_spec_invalid`, for an offset that is not a whole number.
    """
    offset = reference.get("offset", 0)
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise AnalysisRefused(
            EngineRefusal.op_spec_invalid(detail=f"a rolling offset is an integer, not {offset!r}")
        )
    return offset


def _rolling_ref(value: object) -> dict[str, Any]:
    """A rolling reference given in place of a period, with its rule defaulted."""
    assert isinstance(value, Mapping)
    return {
        "anchor": "latest",
        "offset": _offset(value),
        "rule": value.get("rule", _DEFAULT_RULE),
    }


def _overlay(parameters: dict[str, Any]) -> dict[str, Any]:
    """The rolling overlay of `parameters`, set to `{}` where it is absent or null.

    Raises:
        AnalysisRefused: `op_spec_invalid`, for an overlay that is not an object.
    """
    overlay = parameters.get(_ROLLING)
    if overlay is None:
        overlay = parameters[_ROLLING] = {}
    if not isinstance(overlay, dict):
        raise AnalysisRefused(EngineRefusal.op_spec_invalid(detail="rolling: an object"))
    return overlay


def _lift_rolling(parameters: dict[str, Any]) -> bool:
    """Move each rolling reference given in place of a period into the rolling overlay, its
    period left to fill (`None`). Return whether one was.

    Raises:
        AnalysisRefused: `op_spec_invalid`, for `predicates` that are not a list of objects,
            which the rolling population walk reads before the engine checks the op-spec.
    """
    predicates = parameters.get("predicates")
    if predicates is not None and (
        not isinstance(predicates, list) or not all(isinstance(p, dict) for p in predicates)
    ):
        raise AnalysisRefused(EngineRefusal.op_spec_invalid(detail="predicates: a list of objects"))
    lifted = False
    for bound in _WINDOW_BOUNDS:
        if _is_rolling_ref(reference := parameters.get(bound)):
            _overlay(parameters)[bound] = _rolling_ref(reference)
            parameters[bound] = None
            lifted = True
    for predicate in parameters.get("predicates") or ():
        if isinstance(predicate, dict) and _is_rolling_ref(predicate.get("value")):
            predicate[_ROLLING] = _rolling_ref(predicate["value"])
            predicate["value"] = None
            _overlay(parameters)
            lifted = True
    return lifted


@dataclass(kw_only=True)
class _Engine:
    """The working tree turned to the engine's tree over the read root frames."""

    frames: Sequence[Frame]
    resolved: Sequence[ResolvedRoot]
    lifted: bool = False

    def columns(self, node: Mapping[str, Any]) -> Sequence[str]:
        """The role columns of the root `node`'s `data` chain reaches."""
        while "root" not in node:
            inputs = node["inputs"]
            if "step" in node:
                node = inputs["data"]
            elif node["op_spec"]["verb"] == Verb.COMPARE and "right_frame_ref" in inputs:
                node = inputs["right_frame_ref"]
            else:
                node = inputs[_PRIMARY_SLOT]
        return role_columns(self.frames[node["root"]])

    def tree(self, node: Mapping[str, Any]) -> dict[str, Any]:
        if "root" in node:
            return {"root": node["root"]}
        anchors = {_ANCHOR_ROOTS: node[_ANCHOR_ROOTS]} if _ANCHOR_ROOTS in node else {}
        if "op_spec" in node:
            return {
                "op_spec": node["op_spec"],
                "inputs": {slot: self.tree(child) for slot, child in node["inputs"].items()},
                **anchors,
            }
        step: StepNode = node["step"]
        data_slot, other_slot = _SLOTS[step.verb]
        inputs = node["inputs"]
        if "other" in inputs and other_slot is None:
            raise AnalysisRefused(
                EngineRefusal.op_spec_invalid(detail=f"{step.verb} takes no other input")
            )
        slots = {data_slot if "other" in inputs else _PRIMARY_SLOT: inputs["data"]}
        if "other" in inputs:
            assert other_slot is not None
            slots[other_slot] = inputs["other"]
        sides = _Sides(by_slot={slot: self.columns(child) for slot, child in slots.items()})
        parameters = _translated(step.verb, step.arguments, sides)
        for slot in ("frame_ref", "right_frame_ref", "denominator_frame_ref"):
            parameters.pop(slot, None)
        self.lifted |= _lift_rolling(parameters)
        return {
            "op_spec": {"verb": str(step.verb), "parameters": parameters},
            "inputs": {slot: self.tree(child) for slot, child in slots.items()},
            **anchors,
        }


def _renumbered(tree: dict[str, Any], count: int) -> tuple[dict[str, Any], list[int]]:
    """`tree` with its roots numbered in the order a depth-first walk first reaches them,
    slots in `recipe.input_slots` order and then a node's anchor roots, and the old index of
    each new one."""
    order: list[int] = []

    def walk(node: dict[str, Any]) -> dict[str, Any]:
        if "root" in node:
            if node["root"] not in order:
                order.append(node["root"])
            return {"root": order.index(node["root"])}
        inputs = node["inputs"]
        ordered = [_PRIMARY_SLOT, *sorted(slot for slot in inputs if slot != _PRIMARY_SLOT)]
        walked = {slot: walk(inputs[slot]) for slot in ordered if slot in inputs}
        renumbered = {"op_spec": node["op_spec"], "inputs": walked}
        if _ANCHOR_ROOTS in node:
            renumbered[_ANCHOR_ROOTS] = [walk(root) for root in node[_ANCHOR_ROOTS]]
        return renumbered

    renumbered = walk(tree)
    assert len(order) <= count
    return renumbered, order


def _fill_populations(
    tree: dict[str, Any], records: Sequence[Mapping[str, Any]], frames: Sequence[Frame]
) -> None:
    """Set the populations of every rolling overlay that states none, from the tree."""
    missing = [
        node["op_spec"]["parameters"][_ROLLING]
        for node in derived_nodes(tree)
        if isinstance(node["op_spec"]["parameters"].get(_ROLLING), dict)
        and "population" not in node["op_spec"]["parameters"][_ROLLING]
    ]
    if not missing:
        return
    try:
        populations = anchor_populations(
            tree, records, root_columns=[frame.columns for frame in frames]
        )
    except (
        PopulationEmptyError,
        PopulationLevelColumnMissingError,
        PairingKeysUnstatedError,
    ) as exc:
        raise AnalysisRefused(
            EngineRefusal.op_spec_invalid(detail=f"rolling.population: {exc}")
        ) from exc
    for overlay in missing:
        overlay["population"] = populations


_POPULATIONS: TypeAdapter[Populations] = TypeAdapter(Populations)


def _fill_literals(tree: Mapping[str, Any], period: str) -> None:
    """Set each period left to fill to the latest `period` at its reference's offset. A null
    period no rolling reference states is left for the engine to refuse."""

    def shifted(reference: Mapping[str, Any]) -> str:
        offset = _offset(reference)
        try:
            return shift_period(period, offset)
        except (RollingError, FrameError) as exc:
            raise AnalysisRefused(
                EngineRefusal.op_spec_invalid(
                    detail=f"offset {offset} from {period} leaves the calendar"
                )
            ) from exc

    for node in derived_nodes(tree):
        parameters = node["op_spec"]["parameters"]
        overlay = parameters.get(_ROLLING)
        for bound in _WINDOW_BOUNDS:
            if (
                parameters.get(bound, "") is None
                and isinstance(overlay, Mapping)
                and _is_rolling_ref(overlay.get(bound))
            ):
                parameters[bound] = shifted(overlay[bound])
        for predicate in parameters.get("predicates") or ():
            if (
                isinstance(predicate, Mapping)
                and predicate.get("value", "") is None
                and _is_rolling_ref(predicate.get(_ROLLING))
            ):
                predicate["value"] = shifted(predicate[_ROLLING])


def resolved_periods(tree: Mapping[str, Any], *, rule: str | None) -> tuple[dict[str, Any], ...]:
    """Each period the computed recipe `tree` states relative to the latest period, once, by
    offset, under the anchor `rule`; none when the recipe has no rolling anchor (`rule`
    `None`)."""
    if rule is None:
        return ()
    periods = {
        int(reference["offset"]): {
            "anchor": "latest",
            "offset": int(reference["offset"]),
            "rule": rule,
            "period": literal,
        }
        for node in derived_nodes(tree)
        for literal, reference in rolling_literals(node["op_spec"]["parameters"])
    }
    return tuple(periods[offset] for offset in sorted(periods))


# --- Recipe and persistence ----------------------------------------------------------------------


def _recipe_hashes(evaluation: Evaluation, roots: Sequence[ResolvedRoot]) -> dict[str, str]:
    records = [root.record for root in roots]
    normal_hash = normal_form_hash(normal_form(evaluation.tree, records))
    return {
        "normal_form_hash": normal_hash,
        "replay_key": replay_key(
            normal_form_hash=normal_hash,
            root_datasets=[
                root_identity(record["dataset"], enrichment_stamp=record.get("enrichment_stamp"))
                for record in records
            ],
            execution_identities=[evaluation.execution_identity.canonical_string()],
        ),
    }


# The call's word for each engine side a two-input verb names, where its summary states no
# `sides`: `join` and the keyed verbs read `data` on the left or the numerator.
_CALLER_SIDES: Mapping[str, Literal["data", "other"]] = {
    "left": "data",
    "right": "other",
    "numerator": "data",
    "denominator": "other",
}


def _caller_side(
    engine_side: str | None, sides: Mapping[str, str] | None
) -> Literal["data", "other"]:
    """The step input, `data` or `other`, that `engine_side` names. A single-input verb's
    rows come from `data`; `compare` and `sum_of` state which side each input fills."""
    if engine_side is None:
        return "data"
    if sides is not None:
        return "data" if sides.get("data") == engine_side else "other"
    return _CALLER_SIDES[engine_side]


def _compact_bytes(value: object) -> int:
    """The UTF-8 bytes of `value`'s compact JSON, as `Recipe` measures a recipe."""
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def left_out_of(evaluation: Evaluation, *, max_bytes: int | None = None) -> RecipeLeftOut:
    """Every series a step of the evaluated tree dropped for a data gap, step by step in
    `derived_nodes` order, each naming its step, the step input it came from and why, at
    most `MAX_RECIPE_LEFT_OUT_SERIES` of them, and with `max_bytes` set only those whose
    compact JSON list fits in it."""
    series = [
        RecipeDroppedSeries(
            coordinates=dict(dropped.coordinates),
            step=RecipeStepRef(verb=step.verb, node=step.node),
            side=_caller_side(dropped.side, step.sides),
            reason=dropped.reason,
            rows=dropped.rows,
        )
        for step in evaluation.steps
        for dropped in step.dropped
    ]
    kept = series[:MAX_RECIPE_LEFT_OUT_SERIES]
    if max_bytes is not None:
        room = max_bytes
        for count, one in enumerate(kept):
            # Each series after the first adds its separating comma.
            room -= _compact_bytes(one.model_dump(mode="json")) + (1 if count else 0)
            if room < 0:
                kept = kept[:count]
                break
    return RecipeLeftOut(series=kept, complete=len(kept) == len(series))


def _recipe(evaluation: Evaluation, roots: Sequence[ResolvedRoot]) -> Recipe:
    """The recipe of `evaluation`, its left-out list cut to the room `MAX_RECIPE_BYTES` leaves
    beside the rest of the recipe.

    Raises:
        AnalysisTooLarge: The recipe passes `MAX_RECIPE_BYTES` with no series listed.
    """
    hashes = _recipe_hashes(evaluation, roots)
    fields: dict[str, Any] = {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "normal_form_hash": hashes["normal_form_hash"],
        "replay_key": hashes["replay_key"],
        "execution_identity": evaluation.execution_identity.canonical_string(),
        "tree": evaluation.tree,
        "roots": [dict(root.record) for root in roots],
        "anchor": None
        if evaluation.resolved_periods is None
        else RecipeAnchor.model_validate(evaluation.resolved_periods.as_json()),
    }
    # `complete: false` is the longer flag, so the bare size is never under the stored one.
    bare = Recipe.model_construct(**fields, left_out=RecipeLeftOut(series=[], complete=False))
    size = _compact_bytes(bare.stored_body())
    if size > MAX_RECIPE_BYTES:
        raise AnalysisTooLarge(
            ceiling_kind="analysis_recipe_bytes", count=size, cap=MAX_RECIPE_BYTES
        )
    return Recipe(**fields, left_out=left_out_of(evaluation, max_bytes=MAX_RECIPE_BYTES - size))


_STEPS: TypeAdapter[list[Step]] = TypeAdapter(list[Step])


def described_steps(evaluation: Evaluation, roots: Sequence[ResolvedRoot]) -> list[Step] | None:
    """The step list of `evaluation`'s frame, as `common.steps.assemble_steps` in dfo-ai joins
    the steps of a frame the MCP published: each root as a `retrieve` step whose input is its
    dataset pair and whose `options` are its mint options, and each derived node as its call's
    step (`EvaluatedStep.step`), each once, after the steps of the frames it reads, in the
    order a depth-first walk of the inputs first finishes them. Steps are numbered `s1`,
    `s2`, ..., and each input names the step of the frame in that role. A list over the kit's
    byte bound (`MAX_STEPS_BYTES`) sheds coverage members until it fits
    (`dfo_analysis_engine.call_steps.shed_coverage`). A list over `MAX_STEPS`, or still over
    the kit's bounds with no coverage left, or holding a step the engine could not build
    (`EvaluatedStep.step` `None`), is logged and gives `None`, which publishes the frame without
    steps."""
    ref = evaluation.ref
    if any(call.step is None for call in evaluation.steps):
        logger.warning(
            "A step of %s %s could not be built; publishing it without steps",
            ref.spec_id,
            ref.content_version,
        )
        return None
    calls = iter(evaluation.steps)
    ids: dict[tuple[str, str], str] = {}
    steps: list[dict[str, Any]] = []

    def emit(pair: Mapping[str, str], step: dict[str, Any]) -> str:
        key = (pair["spec_id"], pair["content_version"])
        if key not in ids:
            ids[key] = step["id"] = f"s{len(steps) + 1}"
            steps.append(step)
        return ids[key]

    def walk(node: Mapping[str, Any]) -> str:
        if "root" in node:
            root = roots[node["root"]]
            return emit(
                node["frame"],
                {
                    "op": "retrieve",
                    "input": {
                        "id": root.dataset.spec_id,
                        "content_version": root.dataset.content_version,
                    },
                    "options": dict(root.record.get("options") or {}),
                },
            )
        built = next(calls).step
        # Every step was built: a `None` step returned above.
        assert built is not None
        step = built.model_dump(mode="json")
        read = {slot: walk(child) for slot, child in node["inputs"].items()}
        by_pair = {
            (child["frame"]["spec_id"], child["frame"]["content_version"]): read[slot]
            for slot, child in node["inputs"].items()
        }

        def named(ref: Mapping[str, str]) -> str:
            return by_pair[ref["id"], ref["content_version"]]

        if "input" in step:
            step["input"] = named(step["input"])
        else:
            step["inputs"] = {role: named(ref) for role, ref in step["inputs"].items()}
        return emit(node["frame"], step)

    walk(evaluation.tree)
    try:
        if len(steps) > MAX_STEPS:
            raise ValueError(f"{len(steps)} steps, over MAX_STEPS={MAX_STEPS}")
        size = steps_bytes(steps)
        if size > MAX_STEPS_BYTES:
            steps, shed = shed_coverage(steps, size=size, max_bytes=MAX_STEPS_BYTES)
            logger.info(
                "The steps of %s %s over the %d-byte bound shed %d coverage member(s)",
                ref.spec_id,
                ref.content_version,
                MAX_STEPS_BYTES,
                shed,
            )
        check_steps(steps)
    except (StepsError, ValueError):
        logger.warning(
            "The steps of %s %s pass the kit's bounds; publishing it without steps",
            ref.spec_id,
            ref.content_version,
            exc_info=True,
        )
        return None
    return _STEPS.validate_python(steps)


def _stored_file_digest(
    stores: AnalysisStores, *, spec_id: str, content_version: str
) -> str | None:
    """The digest of the servable frame file (`render.frame_file_location`) of a recorded pair,
    or `None`."""
    if stores.snapshot_reader.read_version(spec_id, content_version) is None:
        return None
    locations = stores.snapshot_reader.read_artifact_locations(spec_id, content_version)
    location = frame_file_location(
        select_servable_locations(locations, now=stores.now()), site=stores.site
    )
    return None if location is None else location.artifact_digest


def _store(
    evaluation: Evaluation,
    roots: Sequence[ResolvedRoot],
    stores: AnalysisStores,
    *,
    trace_id: str | None,
) -> tuple[str, bool]:
    """Store the evaluated frame and its recipe, and return the digest of the stored frame
    file, the file a ref names, and whether this call published the frame.

    Raises:
        AnalysisStoreDiverges: The pair holds a recipe or frame bytes that differ from these.
    """
    ref = evaluation.ref
    recipe = _recipe(evaluation, roots)
    try:
        return _stored(evaluation, roots, recipe, stores, trace_id=trace_id)
    except Refusal as refused:
        if refused.code != _CONFLICT:
            raise
        _log_divergence(
            refused, recipe, stores, spec_id=ref.spec_id, content_version=ref.content_version
        )
        raise AnalysisStoreDiverges(
            spec_id=ref.spec_id, content_version=ref.content_version
        ) from refused


_CONFLICT = "publish_conflict"


def _log_divergence(
    refused: Refusal,
    recipe: Recipe,
    stores: AnalysisStores,
    *,
    spec_id: str,
    content_version: str,
) -> None:
    """Log a store's refusal of a run's frame or recipe with the run's identity and the stored
    one."""
    stored = stores.recipe_store.read_recipe(spec_id=spec_id, content_version=content_version)
    logger.error(
        "an analysis run diverges from the frame stored under its pair spec_id=%s "
        "content_version=%s constraint_id=%s run_normal_form_hash=%s run_replay_key=%s "
        "stored_normal_form_hash=%s stored_replay_key=%s detail=%s",
        spec_id,
        content_version,
        getattr(refused, "constraint_id", None),
        recipe.normal_form_hash,
        recipe.replay_key,
        None if stored is None else stored.normal_form_hash,
        None if stored is None else stored.replay_key,
        refused,
    )


def _stored(
    evaluation: Evaluation,
    roots: Sequence[ResolvedRoot],
    recipe: Recipe,
    stores: AnalysisStores,
    *,
    trace_id: str | None,
) -> tuple[str, bool]:
    ref = evaluation.ref
    held = _stored_file_digest(stores, spec_id=ref.spec_id, content_version=ref.content_version)
    if held is not None:
        write_frame_recipe(
            recipe_store=stores.recipe_store,
            spec_id=ref.spec_id,
            content_version=ref.content_version,
            recipe=recipe,
        )
        return held, False
    published = publish_built_frame(
        frame=evaluation.frame,
        spec_id=ref.spec_id,
        content_version=ref.content_version,
        input_content_versions=list(dict.fromkeys(root.dataset.content_version for root in roots)),
        artifact_store=stores.artifact_store,
        snapshot_reader=stores.snapshot_reader,
        version_writer=stores.version_writer,
        trace_id=trace_id,
        recipe=recipe,
        recipe_store=stores.recipe_store,
        steps=described_steps(evaluation, roots),
        enrichment=stores.enrichment,
        file_enrichment=stores.enrichment,
        site=stores.site,
        prepared_at=stores.now(),
    )
    return published.artifact_digest, published.created


# --- The run -------------------------------------------------------------------------------------


_DEFAULT_BOUNDS = AnalysisBounds()


def plan_roots(
    request: AnalysisRequest, *, source: RootSource, bounds: AnalysisBounds
) -> tuple[dict[str, Any], list[SpecRoot | DatasetRoot]]:
    """The working tree of `request` and its roots, derived roots expanded.

    Raises:
        SeveralMembers: A ref input holds several members.
        FrameUnavailable: A derived root's recipe is not stored.
        AnalysisTooLarge: The tree holds more than `bounds.max_nodes` nodes.
    """
    plan = _Plan(max_nodes=bounds.max_nodes)
    top = request.tree
    verb = top.verb if isinstance(top, StepNode) else ""
    tree = _plan_node(top, plan, source=source, verb=verb, side="data")
    if request.anchor_roots is not None:
        tree[_ANCHOR_ROOTS] = [plan.root(root) for root in request.anchor_roots]
    return tree, plan.roots


def resolve_roots(
    roots: Sequence[SpecRoot | DatasetRoot],
    *,
    source: RootSource,
    latest: bool,
    trace_id: str | None,
) -> list[ResolvedRoot]:
    """Every root at the version it is read at. Every root is resolved before any pending one
    answers, so each rebuild a run waits on is submitted at once.

    Raises:
        AnalysisPending: A root's dataset is still building.
    """
    resolved: list[ResolvedRoot] = []
    pending: list[AnalysisPending] = []
    for root in roots:
        try:
            resolved.append(source.resolve(root, latest=latest, trace_id=trace_id))
        except AnalysisPending as waiting:
            pending.append(waiting)
    if pending:
        raise AnalysisPending(
            operations=[operation for waiting in pending for operation in waiting.operations],
            retry_after_s=max(waiting.retry_after_s for waiting in pending),
        )
    return resolved


@dataclass(frozen=True, kw_only=True)
class Computed:
    """A tree computed and not stored: the engine's evaluation, the roots it read in the
    recipe's order, the `stat_var` values of each root's frame by its index in the tree, and
    the output digest of its frame."""

    evaluation: Evaluation
    roots: tuple[ResolvedRoot, ...]
    root_variables: Mapping[int, frozenset[str]]
    output_digest: str


def _read_roots(resolved: Sequence[ResolvedRoot], *, source: RootSource, cap: int) -> list[Frame]:
    """The frames of `resolved`, read in turn. The rows the roots' versions record are summed
    before the first read, and the rows read so far after each, so a run over `cap` root rows
    refuses before it reads past it.

    Raises:
        AnalysisTooLarge: The roots hold more than `cap` rows.
    """
    recorded = sum(root.rows or 0 for root in resolved)
    if recorded > cap:
        raise AnalysisTooLarge(ceiling_kind="analysis_root_rows", count=recorded, cap=cap)
    frames: list[Frame] = []
    read = 0
    for root in resolved:
        frame = source.read(root)
        read += frame.num_rows
        if read > cap:
            raise AnalysisTooLarge(ceiling_kind="analysis_root_rows", count=read, cap=cap)
        frames.append(frame)
    return frames


def _evaluated(
    working: dict[str, Any],
    resolved: Sequence[ResolvedRoot],
    *,
    source: RootSource,
    bounds: AnalysisBounds,
    latest: bool,
) -> Computed:
    """Read the resolved roots and evaluate the working tree over them. The caller holds the
    run's materialise permit.

    Raises:
        AnalysisTooLarge: The roots pass `bounds.max_root_rows`, or the frame passes the rows
            the publish door stores (`PUBLISH_MAX_ROWS`), refused before it is digested.
    """
    frames = _read_roots(resolved, source=source, cap=bounds.max_root_rows)
    engine = _Engine(frames=frames, resolved=resolved)
    tree, order = _renumbered(engine.tree(working), len(resolved))
    ordered = [resolved[index] for index in order]
    frames = [frames[index] for index in order]
    _fill_populations(tree, [root.record for root in ordered], frames)
    inputs = {
        index: RootFrame(ref=root.frame, frame=frame)
        for index, (root, frame) in enumerate(zip(ordered, frames, strict=True))
    }
    if engine.lifted:
        anchor = latest_period_of(tree, inputs)
        # A lifted reference sets a rolling overlay, so the tree has a latest period.
        assert anchor is not None
        _fill_literals(tree, anchor.period)
    evaluation = evaluate(
        tree, inputs, policy=Policy(latest=latest, max_frame_rows=bounds.max_join_rows)
    )
    rows = evaluation.frame.num_rows
    if rows > PUBLISH_MAX_ROWS:
        raise AnalysisTooLarge(
            ceiling_kind="analysis_output_rows", count=rows, cap=PUBLISH_MAX_ROWS
        )
    root_variables = {index: stat_vars(root.frame) for index, root in inputs.items()}
    names = minted_variable_names(evaluation.tree, root_variables)
    return Computed(
        evaluation=evaluation,
        roots=tuple(ordered),
        root_variables=root_variables,
        output_digest=output_digest(evaluation.frame, names),
    )


def compute_analysis(
    request: AnalysisRequest,
    *,
    source: RootSource,
    bounds: AnalysisBounds = _DEFAULT_BOUNDS,
    permit: Callable[[], AbstractContextManager[object]] = nullcontext,
    trace_id: str | None = None,
) -> Computed:
    """Compute `request`'s tree as `run_analysis` does, storing nothing. `maintenance
    diff-recipes` compares what this computes with a stored frame.

    Raises:
        As `run_analysis`.
    """
    working, roots = plan_roots(request, source=source, bounds=bounds)
    resolved = resolve_roots(roots, source=source, latest=request.latest, trace_id=trace_id)
    with permit():
        return _evaluated(working, resolved, source=source, bounds=bounds, latest=request.latest)


# The format of a stored answer's body (`AnalysisResult.body`), part of every request key, so
# only a body of this format is answered. Raise it whenever the body changes.
ANSWER_FORMAT = "analysis_answer/1"


def _keyed_tree(node: object) -> object:
    """The working tree with each step node as its verb and arguments, the JSON a key hashes."""
    if isinstance(node, StepNode):
        return node.model_dump(mode="json", exclude={"inputs"})
    if isinstance(node, Mapping):
        return {key: _keyed_tree(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_keyed_tree(value) for value in node]
    return node


def request_key(
    working: Mapping[str, Any], resolved: Sequence[ResolvedRoot], *, latest: bool
) -> str:
    """The key a run's answer is stored under: the sha256 of the canonical JSON of the working
    tree, each resolved root's record, `latest`, the execution identity and `ANSWER_FORMAT`.
    Two requests under one key compute one frame and answer one body."""
    preimage = {
        "answer_format": ANSWER_FORMAT,
        "tree": _keyed_tree(working),
        "roots": [dict(root.record) for root in resolved],
        "latest": latest,
        "execution_identity": current_execution_identity().canonical_string(),
    }
    return hashlib.sha256(rfc8785.dumps(preimage)).hexdigest()


@dataclass(frozen=True, kw_only=True)
class PreparedAnalysis:
    """A request planned and resolved, no frame read: its working tree, its roots at the
    versions they are read at, and its request key."""

    working: dict[str, Any]
    resolved: tuple[ResolvedRoot, ...]
    latest: bool
    key: str


def prepare_analysis(
    request: AnalysisRequest,
    *,
    source: RootSource,
    bounds: AnalysisBounds = _DEFAULT_BOUNDS,
    trace_id: str | None = None,
) -> PreparedAnalysis:
    """Plan `request`'s tree, resolve its roots and take its request key, reading no frame.

    Raises:
        AnalysisPending: A root is still building.
        SeveralMembers, FrameUnavailable, AnalysisTooLarge: As their docstrings say.
    """
    working, roots = plan_roots(request, source=source, bounds=bounds)
    resolved = resolve_roots(roots, source=source, latest=request.latest, trace_id=trace_id)
    return PreparedAnalysis(
        working=working,
        resolved=tuple(resolved),
        latest=request.latest,
        key=request_key(working, resolved, latest=request.latest),
    )


def stored_answer(prepared: PreparedAnalysis, *, stores: AnalysisStores) -> dict[str, Any] | None:
    """The body stored under `prepared`'s key while its frame file is still served at the
    digest the body's ref names, else `None`."""
    stored = stores.recipe_store.read_answer(request_key=prepared.key)
    if stored is None:
        return None
    served = _stored_file_digest(
        stores, spec_id=stored.spec_id, content_version=stored.content_version
    )
    body: dict[str, Any] = dict(stored.body)
    if served is None or served != body["ref"]["members"][0]["artifact"]["digest"]:
        return None
    return body


def run_prepared(
    prepared: PreparedAnalysis,
    *,
    source: RootSource,
    stores: AnalysisStores,
    bounds: AnalysisBounds = _DEFAULT_BOUNDS,
    permit: Callable[[], AbstractContextManager[object]] = nullcontext,
    trace_id: str | None = None,
) -> AnalysisResult:
    """Compute and store `prepared`'s tree, then store its body under its request key.

    Raises:
        AnalysisRefused: The engine refuses the tree.
        AnalysisTooLarge, AnalysisStoreDiverges: As their docstrings say.
    """
    with permit():
        computed = _evaluated(
            prepared.working,
            prepared.resolved,
            source=source,
            bounds=bounds,
            latest=prepared.latest,
        )
        evaluation = computed.evaluation
        artifact_digest, created = _store(evaluation, computed.roots, stores, trace_id=trace_id)
    result = AnalysisResult(
        evaluation=evaluation,
        frame_content_version=evaluation.summary["frame_content_version"],
        output_digest=computed.output_digest,
        artifact_digest=artifact_digest,
        roots=computed.roots,
        resolved_periods=resolved_periods(
            evaluation.tree,
            rule=None if evaluation.resolved_periods is None else evaluation.resolved_periods.rule,
        ),
        created=created,
    )
    stores.recipe_store.write_answer(
        AnswerRecord(
            request_key=prepared.key,
            spec_id=evaluation.ref.spec_id,
            content_version=evaluation.ref.content_version,
            body=result.body(),
        )
    )
    return result


def run_analysis(
    request: AnalysisRequest,
    *,
    source: RootSource,
    stores: AnalysisStores,
    bounds: AnalysisBounds = _DEFAULT_BOUNDS,
    permit: Callable[[], AbstractContextManager[object]] = nullcontext,
    trace_id: str | None = None,
) -> AnalysisResult:
    """Prepare, compute and store `request`'s tree. See the module docstring.

    Raises:
        AnalysisRefused: The engine refuses the tree.
        AnalysisPending: A root is still building.
        SeveralMembers, FrameUnavailable, AnalysisTooLarge: As their docstrings say.
    """
    prepared = prepare_analysis(request, source=source, bounds=bounds, trace_id=trace_id)
    return run_prepared(
        prepared, source=source, stores=stores, bounds=bounds, permit=permit, trace_id=trace_id
    )


# --- The anchor ----------------------------------------------------------------------------------


class AnchorRequest(BaseModel):
    """`POST /api/datasets/analysis/anchor`: the roots and populations a latest period is
    taken over."""

    model_config = ConfigDict(extra="forbid")

    roots: list[
        Annotated[
            Annotated[SpecRoot, Tag("spec")] | Annotated[DatasetRoot, Tag("dataset")],
            Discriminator(_node_kind),
        ]
    ] = Field(min_length=1, max_length=DEFAULT_MAX_NODES)
    populations: list[dict[str, JsonValue]] = Field(min_length=1)
    latest: bool = False


def run_anchor(
    request: AnchorRequest,
    *,
    source: RootSource,
    bounds: AnalysisBounds = _DEFAULT_BOUNDS,
    permit: Callable[[], AbstractContextManager[object]] = nullcontext,
    trace_id: str | None = None,
) -> tuple[ResolvedPeriods, list[ResolvedRoot]]:
    """The latest period of `request`'s roots over its populations, storing nothing. The roots
    are read under the root-row bound a run reads its roots under (`bounds.max_root_rows`).

    Raises:
        AnalysisRefused: `rolling_anchor_unresolved`, `rolling_grain_unsupported`, or
            `op_spec_invalid` for populations that do not parse.
        AnalysisPending: A root is still building.
        AnalysisTooLarge: The roots hold more than `bounds.max_root_rows` rows.
    """
    try:
        populations = _POPULATIONS.validate_python(canonical_populations(request.populations))
    except (ValidationError, ValueError, KeyError, TypeError) as exc:
        raise AnalysisRefused(EngineRefusal.op_spec_invalid(detail=f"populations: {exc}")) from exc
    resolved = resolve_roots(request.roots, source=source, latest=request.latest, trace_id=trace_id)
    with permit():
        frames = _read_roots(resolved, source=source, cap=bounds.max_root_rows)
        anchor = latest_period(
            populations,
            [
                (root.frame.spec_id, admit(frame, content_version=root.frame.content_version))
                for root, frame in zip(resolved, frames, strict=True)
            ],
        )
    return anchor, resolved


__all__ = [
    "ANSWER_FORMAT",
    "ROOT_MINT_OPTIONS",
    "AnalysisBounds",
    "AnalysisPending",
    "AnalysisRequest",
    "AnalysisResult",
    "AnalysisStoreDiverges",
    "AnalysisStores",
    "AnalysisTooLarge",
    "AnchorRequest",
    "Computed",
    "DatasetRoot",
    "DerivedRoot",
    "FrameUnavailable",
    "PreparedAnalysis",
    "RefRoot",
    "ResolvedRoot",
    "RootExpired",
    "RootNotStored",
    "RootRecipeMissing",
    "RootRendering",
    "RootRenderingUnreadable",
    "RootSource",
    "SeveralMembers",
    "SpecRoot",
    "StepNode",
    "column_for",
    "compute_analysis",
    "described_steps",
    "left_out_of",
    "plan_roots",
    "prepare_analysis",
    "recorded_dataset_root",
    "request_key",
    "resolve_roots",
    "resolved_periods",
    "role_columns",
    "run_analysis",
    "run_anchor",
    "run_prepared",
    "stored_answer",
]
