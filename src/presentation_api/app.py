"""Compose the meta, components, frames, and release doors into one `Surface`.
Injected stores are shared across doors. Missing stores are built in each door's lifespan.

Construction resolves the pinned catalogue to compute `capability_version` from packaged data,
and loads the origin catalogue, whose snapshot each request binds. Live stores and credentials
load in door lifespans. The fronting Data access proxy authenticates requests. A `cmp_` or
`cps_` id grants read of a draft, and the write token its create returned grants write, so the
doors take no caller or session. The two render routes of `presentation_api.doors.render` are
the exception. They admit the live-refresh principal the composition passes in, and refuse
every caller when it passes none.

The surface maps refusals to RFC 9457 problem details through the platform `Envelope`.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI

from assembler.catalogue import BakedReleaseRegistry
from assembler.ports import FrameStore, ReleaseRegistry, RevisionStore, VocabularySource
from dataset_store.snapshot import SnapshotReader
from datasets_api.live_refresh import LiveRefreshPrincipal
from datasets_api.recipe_store import RecipeReader
from dfo_platform.app import Surface
from dfo_platform.doors import DoorContribution, combine_classifiers, flatten
from origin_catalogue import bound_catalogue
from origin_catalogue.service import get_catalogue_service
from presentation_api.codes import PRESENTATION_CODE_CATALOGUE
from presentation_api.doors.components import build_components_door
from presentation_api.doors.frames import build_frames_door
from presentation_api.doors.meta import build_meta_door
from presentation_api.doors.release import build_release_door
from presentation_api.refusals import request_validation_spec
from presentation_api.version import compute_presentation_capability_version

__all__ = ["presentation_surface"]

logger = logging.getLogger(__name__)


def presentation_surface(
    *,
    revision_store: RevisionStore | None = None,
    frame_store: FrameStore | None = None,
    release_registry: ReleaseRegistry | None = None,
    snapshot_reader: SnapshotReader | None = None,
    recipe_reader: RecipeReader | None = None,
    vocabulary_source: VocabularySource | None = None,
    live_refresh: LiveRefreshPrincipal | None = None,
) -> Surface:
    """Build the Presentation `Surface`. Unset stores use live dependencies. Injected stores
    are shared across doors. The pinned release supplies the surface capability version.

    `live_refresh` is the caller the render routes admit. `None`, the default, refuses every
    call to them, as an empty principal does. `snapshot_reader` and `recipe_reader` are the
    Datasets records those routes accept a target frame from: version records for a dataset
    frame and recipes for a derived one. With no `frame_store` injected the components door
    builds both over the datasets pool. With one injected, a reader left out accepts no frame
    by its rule, so only the frame a revision is stored on renders. `vocabulary_source` gives
    the words cards' descriptions are composed from, and defaults to a `LiveVocabularySource`
    over the code index.
    """
    effective_release_registry: ReleaseRegistry = (
        release_registry if release_registry is not None else BakedReleaseRegistry()
    )
    release = effective_release_registry.pinned()
    catalogue = get_catalogue_service()
    capability_version = compute_presentation_capability_version(release=release)
    logger.info(
        "Presentation surface resolved catalog_version=%s capability_version=%s",
        release.catalog_version,
        capability_version,
    )

    doors: tuple[DoorContribution, ...] = (
        build_meta_door(
            capability_version=capability_version, catalog_version=release.catalog_version
        ),
        build_components_door(
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=effective_release_registry,
            snapshot_reader=snapshot_reader,
            recipe_reader=recipe_reader,
            vocabulary_source=vocabulary_source,
            live_refresh=live_refresh,
        ),
        build_frames_door(frame_store=frame_store),
        build_release_door(release_registry=effective_release_registry),
    )

    router = flatten(doors)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            for door in doors:
                if door.lifespan is not None:
                    await stack.enter_async_context(door.lifespan(app))
            yield

    return Surface(
        name="presentation",
        prefix="/api/presentation",
        router=router,
        capability_version=capability_version,
        codes=PRESENTATION_CODE_CATALOGUE,
        # Request validation runs before route handlers, so the surface classifies it.
        classify=combine_classifiers(*(door.classify for door in doors), request_validation_spec),
        exception_types=tuple(exc_type for door in doors for exc_type in door.exception_types),
        exception_exemplars=tuple(
            exemplar for door in doors for exemplar in door.exception_exemplars
        ),
        lifespan=lifespan,
        request_scope=lambda: bound_catalogue(catalogue.snapshot()),
    )
