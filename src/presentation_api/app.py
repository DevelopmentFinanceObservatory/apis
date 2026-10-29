"""Compose the meta, components, frames, release, and shares doors into one `Surface`.
Injected stores are shared across doors. Missing stores are built in each door's lifespan.
The shares door mounts when a share store is injected or the surface builds its stores live
(no `revision_store` injected), so a surface composed over test doubles without one serves no
share route rather than reaching for a live database.

Construction resolves the pinned catalogue to compute `capability_version` from packaged data,
and loads the origin catalogue, whose snapshot each request binds. With `GRAPH_RELEASE_URI` set,
each request also binds the release service's served release, and an unreadable release fails
construction. Live stores and credentials load in door lifespans. The website's proxy answers
`429` once an IP passes its allowance, 120 calls a minute on `/api/*` by default, counted in a
fixed window that every proxy instance shares in Postgres. A call whose count fails goes
through. The proxy passes `Authorization` upstream unread. A `cmp_`
or `cps_` id grants read of a draft, and the write token its create returned grants write, so the
doors take no session. The two render routes of `presentation_api.doors.render` admit the
live-refresh principal the composition passes in, and refuse every caller when it passes none.
A card create or revise names its caller for the card allowance, and the share routes
(`presentation_api.doors.shares`) for the share allowance. A replace or delete of a share takes
its write token. The share import route admits the principal the composition's `principals`
grants `share_import`, and refuses every caller when it passes none.

The surface maps refusals to RFC 9457 problem details through the platform `Envelope`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager

from fastapi import FastAPI

from assembler.catalogue import BakedReleaseRegistry
from assembler.ports import FrameStore, ReleaseRegistry, RevisionStore, VocabularySource
from dataset_store.snapshot import SnapshotReader
from datasets_api.allowance import SharedAllowanceLedger
from datasets_api.live_refresh import LiveRefreshPrincipal
from datasets_api.principals import Principals
from datasets_api.recipe_store import RecipeReader
from dfo_platform.app import Surface
from dfo_platform.codes import merge_catalogues
from dfo_platform.config import CallerSettings
from dfo_platform.doors import DoorContribution, classify_by_door, combine_classifiers, flatten
from graph_release.context import bound_served_release
from graph_release.service import get_release_service
from origin_catalogue import bound_catalogue
from origin_catalogue.service import get_catalogue_service
from presentation_api.codes import PRESENTATION_CODE_CATALOGUE, SHARE_CODE_CATALOGUE
from presentation_api.doors.components import build_components_door
from presentation_api.doors.frames import build_frames_door
from presentation_api.doors.meta import build_meta_door
from presentation_api.doors.release import build_release_door
from presentation_api.doors.shares import build_shares_door
from presentation_api.newest_frames import NewestFrames
from presentation_api.refusals import request_validation_spec
from presentation_api.version import compute_presentation_capability_version
from presentation_store import ShareStore

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
    share_store: ShareStore | None = None,
    allowances: SharedAllowanceLedger | None = None,
    caller_settings: CallerSettings | None = None,
    principals: Principals | None = None,
    newest_frames: NewestFrames | None = None,
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

    `share_store`, `allowances` and `caller_settings` serve the share routes: the share
    records, the ledger of each caller's daily allowances, and how a caller is named. The
    components door spends the same ledger's card allowance when `allowances` is given or no
    `revision_store` is, except for a caller whose principal `principals` grants `publish`
    (`DatasetsBuildSettings`' by default). The share import route admits the caller
    `principals` grants `share_import`, and with no `principals` refuses every caller. The
    shares door mounts with a share store given or with no `revision_store` given, and builds
    whichever of the three is left out. `newest_frames` answers a live share slot's check for
    newer data (`presentation_api.newest_frames`); with none, every live slot serves its card's
    own frame and states it stale.

    Each request binds the origin catalogue's loaded snapshot and the release service's served
    graph release, so an unset `GRAPH_RELEASE_URI` or an unreadable release fails startup.
    """
    effective_release_registry: ReleaseRegistry = (
        release_registry if release_registry is not None else BakedReleaseRegistry()
    )
    release = effective_release_registry.pinned()
    catalogue = get_catalogue_service()
    release_service = get_release_service()
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
            allowances=allowances,
            caller=caller_settings,
            principals=principals,
        ),
        build_frames_door(frame_store=frame_store),
        build_release_door(release_registry=effective_release_registry),
    )
    if share_store is not None or revision_store is None:
        doors = (
            *doors,
            build_shares_door(
                release_registry=effective_release_registry,
                share_store=share_store,
                revision_store=revision_store,
                frame_store=frame_store,
                snapshot_reader=snapshot_reader,
                newest=newest_frames,
                vocabulary_source=vocabulary_source,
                allowances=allowances,
                caller=caller_settings,
                principals=principals,
            ),
        )

    router = flatten(doors)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            for door in doors:
                if door.lifespan is not None:
                    await stack.enter_async_context(door.lifespan(app))
            yield

    @contextmanager
    def request_scope() -> Iterator[None]:
        with bound_catalogue(catalogue.snapshot()), bound_served_release(release_service):
            yield

    return Surface(
        name="presentation",
        description=(
            "Create, revise and render presentation components, publish them as shares, and "
            "read frames and the release."
        ),
        prefix="/api/presentation",
        router=router,
        capability_version=capability_version,
        codes=merge_catalogues(PRESENTATION_CODE_CATALOGUE, SHARE_CODE_CATALOGUE),
        # The door whose route raised classifies first, so the shares door answers an I/O
        # failure its own way. Request validation runs before route handlers, so the surface
        # classifies it.
        classify=combine_classifiers(classify_by_door(doors), request_validation_spec),
        exception_types=tuple(exc_type for door in doors for exc_type in door.exception_types),
        exception_exemplars=tuple(
            exemplar for door in doors for exemplar in door.exception_exemplars
        ),
        lifespan=lifespan,
        registry_endpoints={
            path: endpoint for door in doors for path, endpoint in door.registry_endpoints.items()
        },
        request_scope=request_scope,
        public_paths=frozenset(path for door in doors for path in door.public_paths),
    )
