"""Stored recipes computed again on their own root versions, storing nothing.

`StoredRootSource` is an `analysis.RootSource` that reads only what the stores already hold: a
root read with no `columns` is its version's stored frame, and a root read with `columns` is
the frame-layout file of those columns its version links, the newest one. A version with no
record answers `RootNotStored`, and a frame or file the version no longer holds `RootExpired`.
It renders nothing and refreshes nothing, so a run through it writes to no store. It serves
expanded recipe roots alone, the dataset roots `dataset_root` gives; a Spec root, which the
runner would serve by building it, raises `RootRenderingUnreadable`.

`replay_recipe` computes one stored recipe again through the runner's own path
(`analysis.compute_analysis` on the derived root alone, `latest` false) and compares the
result's frame `cv_` and output digest with the stored frame's. The stored frame's digest names
the variables its operations wrote by the stored tree over the roots the run read, each stored
root matched to the root the run read under the same record
(`output_digest.minted_variable_names`), so a run that reads no root or computes nothing
reports the stored frame's `cv_` alone. Its `Replay.outcome` is one of:

- `equal`: both match;
- `differs`: either differs, or the runner refuses or fails on a recipe whose roots and stored
  frame it read;
- `roots_unreadable`: a root version or its frame is no longer held, or the recipe reads a
  root with options the runner does not read;
- `frame_unreadable`: the stored frame itself is no longer held.

`recipe_family` sorts a recipe as `compare` when any of its steps is a `compare`, else `sum_of`
when any is a `sum_of`, else `other`, since those two verbs pair their sides by role and their
counts are read apart.
"""

from __future__ import annotations

import functools
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from dfo_frame_grammar import Frame

from artifact_store import ArtifactStore, Tier
from artifact_store.errors import ArtifactNotFoundError
from dataset_store.clock import Clock
from dataset_store.snapshot import (
    SnapshotReader,
    StoredArtifactLocation,
    select_servable_locations,
)
from datasets_api.analysis import (
    AnalysisRequest,
    Computed,
    DatasetRoot,
    DerivedRoot,
    FrameUnavailable,
    ResolvedRoot,
    RootExpired,
    RootNotStored,
    RootRenderingUnreadable,
    SpecRoot,
    compute_analysis,
    recorded_dataset_root,
)
from datasets_api.analysis_roots import frame_from_frame_layout
from datasets_api.derived import root_rendering
from datasets_api.mint_models import MintOptions
from datasets_api.output_digest import minted_variable_names, output_digest
from datasets_api.recipe_store import RecipeReader
from datasets_api.render import (
    artifact_rendering,
    complete_rendering,
    is_statistics_rendering,
    stored_frame_location,
)
from datasets_api.store_errors import read_stored_frame
from dfo_analysis_engine.calls import derive_frame_content_version
from dfo_analysis_engine.frame_ref import FrameRef
from dfo_analysis_engine.recipe import derived_nodes
from dfo_analysis_engine.refusal import AnalysisRefused
from dfo_platform.config import PublicSiteSettings

# A frame-layout file's rendering holds no site setting, which only a bundle's names.
_NO_SITE = PublicSiteSettings(public_site_url="https://dfo.invalid")


@dataclass(frozen=True, kw_only=True)
class _Located:
    """What `StoredRootSource.read` reads: a linked frame-layout file, or the stored frame."""

    location: StoredArtifactLocation
    frame_layout: bool


@dataclass(kw_only=True)
class StoredRootSource:
    """`analysis.RootSource` over what the stores hold, for one replay, since it keeps the
    root records of the recipe it expands. See the module docstring."""

    snapshot_reader: SnapshotReader
    artifact_store: ArtifactStore
    recipes: RecipeReader
    now: Clock
    _records: list[tuple[DatasetRoot, Mapping[str, Any]]] = field(default_factory=list)

    def recipe(self, *, spec_id: str, content_version: str) -> Mapping[str, Any] | None:
        record = self.recipes.read_recipe(spec_id=spec_id, content_version=content_version)
        return None if record is None else record.body

    def dataset_root(self, record: Mapping[str, Any]) -> DatasetRoot:
        dataset = record["dataset"]
        options = record.get("options") or {}
        if not _frame_layout(options):
            raise RootRenderingUnreadable(
                spec_id=dataset["spec_id"],
                content_version=dataset["content_version"],
                reason="its recipe reads it with options other than a frame-layout file",
            )
        root = recorded_dataset_root(record)
        self._records.append((root, record))
        return root

    def resolve(
        self, root: SpecRoot | DatasetRoot, *, latest: bool, trace_id: str | None
    ) -> ResolvedRoot:
        record = next((held for seen, held in self._records if seen == root), None)
        if record is None or latest:
            raise RootRenderingUnreadable(
                spec_id=root.spec_id,
                content_version=getattr(root, "content_version", ""),
                reason="only a stored recipe's own root versions are read",
            )
        dataset = FrameRef.model_validate(record["dataset"])
        spec_id, content_version = dataset.spec_id, dataset.content_version
        state = self.snapshot_reader.read_version_state(spec_id, content_version)
        if state is None:
            raise RootNotStored(
                spec_id=spec_id, content_version=content_version, reason="no such version is stored"
            )
        servable = select_servable_locations(state.locations, now=self.now())
        options = record.get("options") or {}
        columns = options.get("columns") or ()
        if columns:
            wanted = artifact_rendering(MintOptions.model_validate(options), site=_NO_SITE)
            files = [
                location
                for location in servable
                if not is_statistics_rendering(location.rendering)
                and complete_rendering(location.rendering) == wanted
            ]
            location = max(files, key=lambda held: held.created_at, default=None)
        else:
            location = stored_frame_location(servable)
        if location is None:
            raise RootExpired(
                spec_id=spec_id,
                content_version=content_version,
                reason="no frame of its rendering is held",
            )
        return ResolvedRoot(
            frame=FrameRef.model_validate(record["frame"]),
            dataset=dataset,
            record=record,
            rendering=root_rendering(options),
            source=_Located(location=location, frame_layout=bool(columns)),
        )

    def read(self, root: ResolvedRoot) -> Frame:
        located = root.source
        assert isinstance(located, _Located)
        data = self._bytes(
            located.location,
            spec_id=root.dataset.spec_id,
            content_version=root.dataset.content_version,
        )
        if located.frame_layout:
            return frame_from_frame_layout(data)
        return read_stored_frame(
            data,
            media_type=located.location.media_type,
            digest=located.location.artifact_digest,
        )

    def _bytes(
        self, location: StoredArtifactLocation, *, spec_id: str, content_version: str
    ) -> bytes:
        try:
            return self.artifact_store.get(
                artifact_digest=location.artifact_digest, tier=Tier(location.tier)
            )
        except ArtifactNotFoundError:
            raise RootExpired(
                spec_id=spec_id, content_version=content_version, reason="its bytes are gone"
            ) from None

    def stored_output(self, *, spec_id: str, content_version: str) -> Frame | None:
        """The stored frame of the derived version, or `None` when it is no longer held."""
        state = self.snapshot_reader.read_version_state(spec_id, content_version)
        if state is None:
            return None
        location = stored_frame_location(select_servable_locations(state.locations, now=self.now()))
        if location is None:
            return None
        try:
            data = self._bytes(location, spec_id=spec_id, content_version=content_version)
        except RootExpired:
            return None
        return read_stored_frame(
            data, media_type=location.media_type, digest=location.artifact_digest
        )


def _frame_layout(options: Mapping[str, Any]) -> bool:
    return (
        options.get("format") == "parquet"
        and options.get("bundle") is False
        and options.get("layout") == "frame"
    )


class ReplayOutcome(StrEnum):
    EQUAL = "equal"
    DIFFERS = "differs"
    ROOTS_UNREADABLE = "roots_unreadable"
    FRAME_UNREADABLE = "frame_unreadable"


def recipe_family(tree: Mapping[str, Any]) -> str:
    """`compare`, `sum_of` or `other`. See the module docstring."""
    verbs = {str(node["op_spec"]["verb"]) for node in derived_nodes(tree)}
    if "compare" in verbs:
        return "compare"
    if "sum_of" in verbs:
        return "sum_of"
    return "other"


@dataclass(frozen=True, kw_only=True)
class Replay:
    """One stored recipe computed again. `stored` and `computed` hold each frame's
    `frame_content_version` and `output_digest` where it was read or computed; `detail`
    names why a run differs or could not read."""

    spec_id: str
    content_version: str
    family: str
    outcome: ReplayOutcome
    stored: Mapping[str, str] | None = None
    computed: Mapping[str, str] | None = None
    detail: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "content_version": self.content_version,
            "family": self.family,
            "outcome": self.outcome.value,
            "stored": None if self.stored is None else dict(self.stored),
            "computed": None if self.computed is None else dict(self.computed),
            "detail": self.detail,
        }


def frame_facts(frame: Frame, names: Mapping[str, str]) -> dict[str, str]:
    """A frame's `cv_` and output digest, the two facts a replay compares, with `names` the
    names of the variables its operations wrote (`output_digest.output_digest`)."""
    return {
        "frame_content_version": derive_frame_content_version(frame),
        "output_digest": output_digest(frame, names),
    }


def _stored_root_variables(
    records: Sequence[Mapping[str, Any]], computed: Computed
) -> dict[int, frozenset[str]] | None:
    """The `stat_var` values of each root of the stored recipe, by its index in the stored
    tree, taken from the root the run read under the same record, or `None` when a record has
    no such root. The run numbers its roots in its own order (`analysis._renumbered`), which
    need not be the stored one."""
    variables: dict[int, frozenset[str]] = {}
    for index, record in enumerate(records):
        read = next((at for at, root in enumerate(computed.roots) if root.record == record), None)
        if read is None:
            return None
        variables[index] = computed.root_variables[read]
    return variables


def replay_recipe(
    *, spec_id: str, content_version: str, recipe: Mapping[str, Any], source: StoredRootSource
) -> Replay:
    """Compute the stored recipe of `{spec_id, content_version}` again and compare. See the
    module docstring."""
    family = recipe_family(recipe["tree"])
    replay = functools.partial(
        Replay, spec_id=spec_id, content_version=content_version, family=family
    )
    stored_frame = source.stored_output(spec_id=spec_id, content_version=content_version)
    if stored_frame is None:
        return replay(outcome=ReplayOutcome.FRAME_UNREADABLE)
    stored = {"frame_content_version": derive_frame_content_version(stored_frame)}
    request = AnalysisRequest(tree=DerivedRoot(spec_id=spec_id, content_version=content_version))
    try:
        computed = compute_analysis(request, source=source)
    except FrameUnavailable as unreadable:
        return replay(outcome=ReplayOutcome.ROOTS_UNREADABLE, stored=stored, detail=str(unreadable))
    except AnalysisRefused as refused:
        return replay(
            outcome=ReplayOutcome.DIFFERS,
            stored=stored,
            detail=f"refused {refused.refusal.code.value}: {refused.refusal.message}",
        )
    except Exception as failed:
        return replay(
            outcome=ReplayOutcome.DIFFERS,
            stored=stored,
            detail="".join(traceback.format_exception_only(failed)).strip(),
        )
    variables = _stored_root_variables(recipe["roots"], computed)
    if variables is None:
        return replay(
            outcome=ReplayOutcome.DIFFERS,
            stored=stored,
            detail="the run read roots other than those the recipe records",
        )
    stored = frame_facts(stored_frame, minted_variable_names(recipe["tree"], variables))
    facts = {
        "frame_content_version": computed.evaluation.summary["frame_content_version"],
        "output_digest": computed.output_digest,
    }
    return replay(
        outcome=ReplayOutcome.EQUAL if facts == stored else ReplayOutcome.DIFFERS,
        stored=stored,
        computed=facts,
    )
