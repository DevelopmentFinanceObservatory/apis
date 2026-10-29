"""Checks served code sets against active registry mappings for Search, Datasets, Presentation,
Graph, Skills, and the platform. Served codes come from each contract's doors and classifiers,
including warning and no-data channels. Platform catch-all codes stay under `platform`. Request
validation codes stay with the contract whose request model determines them.

The contract-wide comparison misses a code served at the wrong endpoint or status, as when the
publish door emitted `identity_collision` (409) and `pin_not_permitted` (403), which other Datasets
endpoints already registered. The per-endpoint check compares `(code, status)` for
`datasets.publish`, `datasets.operations`, `datasets.exports` and `datasets.export_link`, the
Datasets doors with reproducible exception exemplars. Other endpoints also serve validation codes or codes raised outside their door
classifiers, so comparing them would report false gaps. Shared Search observation warnings must
list every resolve door. A mismatch fails in either direction.
"""

from __future__ import annotations

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
from datasets_api.app import classify_datasets_exception
from datasets_api.config import DatasetsBuildSettings
from datasets_api.deps import DepsCell
from datasets_api.doors.exports import build_export_link_door, build_exports_door
from datasets_api.doors.meta import build_meta_door
from datasets_api.doors.publish import build_publish_door
from dfo_platform.doors import DoorContribution
from search.engine.config import SEARCH_RESOLVE_DOORS
from search.models import NoDataReason

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


# One `DoorContribution` per Datasets endpoint this module can verify
# byte-for-byte against the registry's own per-(code, http_status) mapping
# -- see this module's own docstring for why only these, and why every
# other Datasets endpoint (and every other contract) stays out.
_ENDPOINT_DOOR_EXEMPLARS: dict[str, DoorContribution] = {
    "datasets.publish": build_publish_door(
        deps=DepsCell(), settings=DatasetsBuildSettings.from_env()
    ),
    "datasets.operations": build_meta_door(
        deps=DepsCell(), settings=DatasetsBuildSettings.from_env()
    ),
    "datasets.exports": build_exports_door(
        deps=DepsCell(), settings=DatasetsBuildSettings.from_env()
    ),
    "datasets.export_link": build_export_link_door(deps=DepsCell()),
}


def _served_status_codes(contribution: DoorContribution) -> frozenset[tuple[str, int]]:
    """Returns classifier `(code, status)` pairs; an unrecognized exemplar signals a test mismatch."""
    outcomes: set[tuple[str, int]] = set()
    for exemplar in contribution.exception_exemplars:
        spec = classify_datasets_exception(exemplar)
        assert spec is not None, (
            f"{exemplar!r} (type {type(exemplar).__name__}) was declared as this door's own "
            "exception_exemplars but classify_datasets_exception returned None for it"
        )
        outcomes.add((spec.code, spec.status))
    return frozenset(outcomes)


def _registered_status_codes(endpoint: str) -> frozenset[tuple[str, int]]:
    """Returns active `problem_details` code-status pairs for `endpoint`.

    Manifest warnings use a separate channel without HTTP statuses."""
    pairs: set[tuple[str, int]] = set()
    for entry in _REGISTRY.entries:
        if entry.lifecycle != "active":
            continue
        for mapping in entry.mappings:
            if mapping.endpoint != endpoint or mapping.channel != "problem_details":
                continue
            assert mapping.http_status is not None
            pairs.add((entry.code, mapping.http_status))
    return frozenset(pairs)


def _assert_endpoint_bijection(*, endpoint: str, contribution: DoorContribution) -> None:
    served = frozenset(_served_status_codes(contribution))
    registered = _registered_status_codes(endpoint)
    unregistered = sorted(served - registered)
    unserved = sorted(registered - served)
    assert not unregistered and not unserved, (
        f"{endpoint}: served/registered (code, http_status) mismatch -- "
        f"served but not registered onto {endpoint}: {unregistered}; "
        f"registered onto {endpoint} but never served: {unserved}"
    )


def test_datasets_publish_served_codes_match_registry_by_status():
    _assert_endpoint_bijection(
        endpoint="datasets.publish", contribution=_ENDPOINT_DOOR_EXEMPLARS["datasets.publish"]
    )


def test_datasets_operations_served_codes_match_registry_by_status():
    _assert_endpoint_bijection(
        endpoint="datasets.operations", contribution=_ENDPOINT_DOOR_EXEMPLARS["datasets.operations"]
    )


@pytest.mark.parametrize("endpoint", ["datasets.exports", "datasets.export_link"])
def test_datasets_exports_served_codes_match_registry_by_status(endpoint: str):
    _assert_endpoint_bijection(endpoint=endpoint, contribution=_ENDPOINT_DOOR_EXEMPLARS[endpoint])


def test_no_served_code_resolves_to_a_retired_registry_entry():
    """Every served code resolves to an active registry entry."""
    all_served = frozenset.union(*_SERVED.values())
    retired = sorted(
        code for code in all_served if code in _REGISTRY and _REGISTRY[code].lifecycle == "retired"
    )
    assert not retired, f"served code(s) resolve to a retired registry entry: {retired}"
