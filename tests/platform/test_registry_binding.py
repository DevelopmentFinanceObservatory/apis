"""Checks served code sets against active registry mappings for Search, Datasets, Presentation,
Graph, Skills, and the platform. Served codes come from each contract's doors and classifiers,
including warning and no-data channels. Platform catch-all codes stay under `platform`. Request
validation codes stay with the contract whose request model determines them.

The per-endpoint check compares `(code, status)` on every Datasets and Presentation endpoint a
door binds: each exemplar of a type a door declares for the endpoint, classified with the
endpoint bound as a request binds it, against the endpoint's registry mappings, so a code served
at an endpoint or status the registry does not map fails, and so does a mapping no exemplar
serves. Every declared type has an exemplar of exactly that type. Shared Search observation
warnings must list every resolve door.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import get_args

import pytest
from dfo_spec_kit.registry import load_registry

import datasets_api.codes as datasets_codes
import datasets_api.warnings as datasets_warnings
import graph_api.codes as graph_codes
import materialise.bundle as materialise_bundle
import materialise.codes as materialise_codes
import materialise.reshape as materialise_reshape
import presentation_api.codes as presentation_codes
import search.engine.codes as search_codes
import search.observations as search_observations
import skills_api.codes as skills_codes
from assembler.fakes import FakeReleaseRegistry
from datasets_api.app import classify_datasets_exception, datasets_doors
from datasets_api.config import DatasetsBuildSettings
from datasets_api.deps import DepsCell
from datasets_api.door_executor import build_door_executor
from datasets_api.doors.definition import ROUTE_EXCEPTION_TYPES
from dfo_platform import ProblemSpec
from dfo_platform.codes import bound_registry_endpoint
from dfo_platform.doors import DoorContribution
from dfo_platform.problem import Refusal
from presentation_api.app import presentation_surface
from presentation_api.doors.components import build_components_door
from presentation_api.doors.frames import build_frames_door
from presentation_api.doors.shares import build_shares_door
from search.engine.config import SEARCH_RESOLVE_DOORS
from search.models import NoDataReason
from tests.code_registry import registered_endpoints, registered_status_codes

_REGISTRY = load_registry()

# Platform-generic codes `dfo_platform.refusal.classify_default` (plus the
# `GET /codes/{code}` miss case) can produce regardless of which contract's
# app the request landed on -- see this module's own docstring for why
# `request_invalid`/`unsupported_schema_version` sit with their calling
# contract instead.
_PLATFORM_DEFAULT_CODES = frozenset(
    {"not_found", "method_not_allowed", "internal_error", "timeout", "code_not_found"}
)

_SERVED: dict[str, frozenset[str]] = {
    "search": frozenset(search_codes.SEARCH_CODE_CATALOGUE)
    | search_codes.SHARED_PLATFORM_CODES
    | search_codes.WARNING_CODES
    | frozenset(get_args(NoDataReason)),
    "datasets": frozenset(datasets_codes.DATASETS_CODE_CATALOGUE)
    | datasets_codes.SHARED_PLATFORM_CODES
    | frozenset(datasets_warnings.DATASETS_WARNING_CATALOGUE)
    | frozenset(materialise_codes.MATERIALISE_CODE_CATALOGUE)
    | frozenset(materialise_bundle.BUNDLE_CODE_CATALOGUE)
    | frozenset(materialise_reshape.RESHAPE_CODE_CATALOGUE)
    | search_observations.PREVIEW_OMISSION_REASONS,
    "presentation": presentation_codes.PRESENTATION_SERVED_CODES,
    "graph": frozenset(graph_codes.GRAPH_CODE_CATALOGUE) | graph_codes.SHARED_PLATFORM_CODES,
    "skills": frozenset(skills_codes.SKILLS_CODE_CATALOGUE)
    | frozenset(skills_codes.SKILLS_WARNING_CATALOGUE)
    | skills_codes.SHARED_PLATFORM_CODES,
    "platform": _PLATFORM_DEFAULT_CODES,
}


def _registered(contract: str) -> frozenset[str]:
    """Every active registry entry with at least one mapping whose endpoint
    prefix (the part before the first `.`) is `contract`."""
    return frozenset(
        entry.code
        for entry in _REGISTRY.entries
        if entry.lifecycle == "active"
        and any(mapping.endpoint.split(".", 1)[0] == contract for mapping in entry.mappings)
    )


def _assert_bijection(*, contract: str, served: frozenset[str], registered: frozenset[str]) -> None:
    unregistered = sorted(served - registered)
    unserved = sorted(registered - served)
    assert not unregistered and not unserved, (
        f"{contract}: served/registered mismatch -- "
        f"served but not registered onto {contract}.*: {unregistered}; "
        f"registered onto {contract}.* but never served: {unserved}"
    )


# Named explicitly so a contract dropped from `_SERVED` fails with a KeyError rather than
# disappearing from the parametrization.
@pytest.mark.parametrize(
    "contract", ["datasets", "graph", "platform", "presentation", "search", "skills"]
)
def test_served_codes_match_registry(contract: str):
    _assert_bijection(contract=contract, served=_SERVED[contract], registered=_registered(contract))


# The warning channels a `search.resolve` registry mapping can carry.
_RESOLVE_WARNING_CHANNELS = frozenset({"diagnostics_warning", "done_event_warning"})

# The three observation warnings the shared fetch machinery mints on every
# door that reaches it: `region/observations/fetch.py`'s `_fetch_observations`
# and `bulk_scope.py`'s `_bulk_scope_series` are the same code on all four
# resolve doors, since each door's per-part path funnels into them
# (`dataset_resubmit` through `region.resubmit_promote.resolve_spec_resubmit`).
# A registry row that omits a door understates where a client can see the
# code.
_SHARED_OBSERVATIONS_WARNING_CODES: tuple[str, ...] = (
    "observations_bulk_fetch_failed",
    "observations_scope_partial",
    "observations_no_matching_source",
)

# The control row and warning codes reach every resolve door.
#
# The control door list is pinned as a literal from `observations_no_dispatchable_facet`.
_OBSERVATIONS_DOOR_LIST_CONTROL_CODE = "observations_no_dispatchable_facet"
_OBSERVATIONS_DOOR_LIST_CONTROL_DOORS: frozenset[str] = frozenset(
    {"raw_text", "parsed", "dataset_resubmit"}
)

# Resolve dispatches `raw_text`, `parsed`, and `dataset_resubmit`.


def _resolve_warning_doors() -> frozenset[str]:
    """Returns configured resolve doors independently of registry declarations."""
    return frozenset(SEARCH_RESOLVE_DOORS)


def _registered_resolve_warning_doors(code: str) -> frozenset[str]:
    """Returns the active warning mapping's doors; an unscoped mapping covers every resolve door."""
    entry = _REGISTRY[code]
    assert entry.lifecycle == "active", f"{code} is not an active registry entry"
    doors: set[str] = set()
    for mapping in entry.mappings:
        if mapping.endpoint != "search.resolve" or mapping.channel not in _RESOLVE_WARNING_CHANNELS:
            continue
        doors.update(_resolve_warning_doors() if mapping.doors is None else mapping.doors)
    return frozenset(doors)


def test_observations_door_list_control_row_names_every_resolve_door():
    """Pins the candidate universe the widened-row assertions below compare
    against: this row's doors are exactly the doors that reach the shared
    observations machinery, so `_resolve_warning_doors` returning anything
    else (a door added to `SEARCH_RESOLVE_DOORS` that never runs this path,
    or a door dropped from it) fails here rather than letting the assertions
    below pass against a universe that no longer matches. Both sides are
    literals: a spec-kit revision that adds or drops this row's doors fails
    here until the literals above are re-read from the new pin, and the
    `dispatched` sub-assertion keeps a served door from going unlisted."""
    registered = _registered_resolve_warning_doors(_OBSERVATIONS_DOOR_LIST_CONTROL_CODE)
    assert registered == _OBSERVATIONS_DOOR_LIST_CONTROL_DOORS, (
        f"{_OBSERVATIONS_DOOR_LIST_CONTROL_CODE}: registry doors {sorted(registered)} != "
        f"the pinned row's own doors {sorted(_OBSERVATIONS_DOOR_LIST_CONTROL_DOORS)} -- "
        "re-read the row from the new pin and update the literal"
    )
    dispatched = [door for door in _resolve_warning_doors() if door not in registered]
    assert not dispatched, (
        f"{_OBSERVATIONS_DOOR_LIST_CONTROL_CODE}: doors {dispatched} reach this code through "
        f"`SEARCH_RESOLVE_DOORS` but the row names {sorted(registered)}"
    )


def test_shared_observations_warnings_name_every_resolve_door():
    """Each of these three codes is minted by machinery every resolve door
    reaches, so its registry row must name every door in
    `SEARCH_RESOLVE_DOORS` -- a door omitted from a row is a door a client
    cannot discover the code on."""
    universe = _resolve_warning_doors()
    assert universe, "SEARCH_RESOLVE_DOORS is empty -- the assertion below would pass vacuously"
    for code in _SHARED_OBSERVATIONS_WARNING_CODES:
        registered = _registered_resolve_warning_doors(code)
        missing = sorted(universe - registered)
        assert not missing, (
            f"{code}: registry row omits door(s) {missing} that reach the shared "
            f"observations machinery (row names {sorted(registered)})"
        )


# Codes the platform answers on a route outside every door's classifier: request validation
# (`request_invalid`, `unsupported_schema_version`), a request past its deadline (`timeout`) and an
# exception no classifier recognises (`internal_error`). An endpoint registers them without a door
# exemplar.
_PLATFORM_ANSWERED: frozenset[tuple[str, int]] = frozenset(
    {
        ("request_invalid", 400),
        ("unsupported_schema_version", 422),
        ("timeout", 504),
        ("internal_error", 500),
    }
)

# Mappings the registry holds that no route of the endpoint raises. Nothing in Datasets raises
# `entities_unbound` or `resolution_ambiguous`, and the definition and options routes admit a
# posted definition with no scope, so no scope role refuses there.
_REGISTERED_NOT_RAISED: dict[str, frozenset[tuple[str, int]]] = {
    "datasets.definition": frozenset(
        {
            ("entities_unbound", 422),
            ("invalid_role_ref", 422),
            ("resolution_ambiguous", 422),
            ("role_incompatible", 422),
        }
    ),
    "datasets.options": frozenset(
        {("entities_unbound", 422), ("resolution_ambiguous", 422), ("role_incompatible", 422)}
    ),
    "datasets.mint": frozenset({("entities_unbound", 422), ("resolution_ambiguous", 422)}),
}


# A declared type a door never raises bare, so it has no exemplar of its own: the mint declares
# `Refusal`, the base of every refusal, so the surface handles each subclass, and each subclass
# the mint raises has its exemplar.
_DECLARED_WITHOUT_EXEMPLAR: frozenset[tuple[str, type[BaseException]]] = frozenset(
    {("datasets.mint", Refusal)}
)


def _declared_types(door: DoorContribution) -> dict[str, tuple[type[BaseException], ...]]:
    """The exception types each endpoint `door` binds can raise: the door's own, or for a door
    that binds several endpoints, the per-route table it builds them from."""
    endpoints = set(door.registry_endpoints.values())
    if len(endpoints) == 1:
        return dict.fromkeys(endpoints, door.exception_types)
    assert endpoints <= set(ROUTE_EXCEPTION_TYPES), (
        f"door binding {sorted(endpoints)} declares no exception types per endpoint"
    )
    return {endpoint: ROUTE_EXCEPTION_TYPES[endpoint] for endpoint in endpoints}


@pytest.fixture(scope="module")
def datasets_built() -> tuple[DoorContribution, ...]:
    """Every Datasets door."""
    return datasets_doors(deps=DepsCell(), settings=DatasetsBuildSettings.from_env())


@pytest.fixture(scope="module")
def datasets_exemplars(datasets_built) -> dict[str, tuple[BaseException, ...]]:
    """Per Datasets endpoint, the exemplars every door that binds it declares."""
    by_endpoint: dict[str, tuple[BaseException, ...]] = {}
    for door in datasets_built:
        for endpoint, exemplars in door.endpoint_exemplars.items():
            by_endpoint[endpoint] = (*by_endpoint.get(endpoint, ()), *exemplars)
    return by_endpoint


@pytest.fixture(scope="module")
def presentation_built() -> tuple[
    tuple[DoorContribution, ...], Callable[[BaseException], ProblemSpec | None]
]:
    """The Presentation doors that bind an endpoint, and the surface's classifier."""
    release_registry = FakeReleaseRegistry()
    reads = build_door_executor(1, "test-registry-binding", surface="presentation")
    doors = (
        build_components_door(release_registry=release_registry, bare_reads=reads),
        build_frames_door(bare_reads=reads),
        build_shares_door(release_registry=release_registry),
    )
    return doors, presentation_surface(release_registry=release_registry).classify


@pytest.fixture(scope="module")
def presentation_exemplars(
    presentation_built,
) -> tuple[dict[str, tuple[BaseException, ...]], Callable[[BaseException], ProblemSpec | None]]:
    """Per Presentation endpoint, the exemplars its door declares, and the surface's classifier."""
    doors, classify = presentation_built
    exemplars = {
        endpoint: tuple(declared)
        for door in doors
        for endpoint, declared in door.endpoint_exemplars.items()
    }
    return exemplars, classify


def test_every_declared_type_has_an_exemplar_of_exactly_that_type(
    datasets_built, presentation_built
):
    """A declared type with no exemplar of its own leaves the bijection below nothing to classify
    for it, so each has one, among the exemplars the endpoint's doors declare, apart from
    `_DECLARED_WITHOUT_EXEMPLAR`."""
    presentation_doors, _classify = presentation_built
    declared = [
        (endpoint, exc_type, door.endpoint_exemplars[endpoint])
        for door in datasets_built
        for endpoint, types in _declared_types(door).items()
        for exc_type in types
    ] + [
        # A Presentation door declares its types for all its endpoints together.
        (
            ",".join(sorted(door.endpoint_exemplars)),
            exc_type,
            tuple(exemplar for group in door.endpoint_exemplars.values() for exemplar in group),
        )
        for door in presentation_doors
        for exc_type in door.exception_types
    ]
    assert declared
    missing = {
        (endpoint, exc_type)
        for endpoint, exc_type, exemplars in declared
        if not any(type(exemplar) is exc_type for exemplar in exemplars)
    }
    assert sorted((e, t.__qualname__) for e, t in missing) == sorted(
        (e, t.__qualname__) for e, t in _DECLARED_WITHOUT_EXEMPLAR
    )


def _served_status_codes(
    exemplars: tuple[BaseException, ...],
    *,
    endpoint: str,
    classify: Callable[[BaseException], ProblemSpec | None],
) -> frozenset[tuple[str, int]]:
    """Classifier `(code, status)` pairs under `endpoint`, bound as the route binds it. An
    exemplar the classifier leaves alone, or whose code `endpoint` does not map, fails."""
    outcomes: set[tuple[str, int]] = set()
    for exemplar in exemplars:
        with bound_registry_endpoint(endpoint):
            try:
                spec = classify(exemplar)
            except LookupError as exc:
                pytest.fail(f"{endpoint}: {exemplar!r} has no registered status: {exc}")
        assert spec is not None, f"{endpoint}: nothing classifies {exemplar!r}"
        outcomes.add((spec.code, spec.status))
    return frozenset(outcomes)


def _assert_endpoint_bijection(*, endpoint: str, served: frozenset[tuple[str, int]]) -> None:
    registered = registered_status_codes(endpoint)
    unregistered = sorted(served - registered)
    unserved = sorted(registered - served - _PLATFORM_ANSWERED)
    assert not unregistered, f"{endpoint}: served but not registered: {unregistered}"
    assert unserved == sorted(_REGISTERED_NOT_RAISED.get(endpoint, ())), (
        f"{endpoint}: registered but never served: {unserved}"
    )


def test_every_datasets_endpoint_binds_a_door(datasets_exemplars):
    assert set(datasets_exemplars) == registered_endpoints("datasets.")


def test_every_presentation_endpoint_binds_a_door(presentation_exemplars):
    exemplars, _classify = presentation_exemplars
    assert set(exemplars) == registered_endpoints("presentation.")


@pytest.mark.parametrize("endpoint", sorted(registered_endpoints("datasets.")))
def test_datasets_served_codes_match_registry_by_status(endpoint: str, datasets_exemplars):
    served = _served_status_codes(
        datasets_exemplars[endpoint], endpoint=endpoint, classify=classify_datasets_exception
    )
    _assert_endpoint_bijection(endpoint=endpoint, served=served)


@pytest.mark.parametrize("endpoint", sorted(registered_endpoints("presentation.")))
def test_presentation_served_codes_match_registry_by_status(endpoint: str, presentation_exemplars):
    exemplars, classify = presentation_exemplars
    served = _served_status_codes(exemplars[endpoint], endpoint=endpoint, classify=classify)
    _assert_endpoint_bijection(endpoint=endpoint, served=served)


def test_no_served_code_resolves_to_a_retired_registry_entry():
    """Every served code resolves to an active registry entry."""
    all_served = frozenset.union(*_SERVED.values())
    retired = sorted(
        code for code in all_served if code in _REGISTRY and _REGISTRY[code].lifecycle == "retired"
    )
    assert not retired, f"served code(s) resolve to a retired registry entry: {retired}"
