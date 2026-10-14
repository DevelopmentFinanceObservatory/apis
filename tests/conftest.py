"""Suite-wide fixtures: response contracts, offline clients, coverage, catalogue, resolvekit."""

import importlib
import os
import sys
import time
from collections.abc import Callable, Iterator
from concurrent import futures
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import httpx
import pytest

import origin_catalogue.service as catalogue_service_module
from materialise import place_scope
from origin_catalogue import Catalogue, bound_catalogue, unbind_catalogue
from origin_catalogue import bind_catalogue as _bind
from statvar_members import shared_enumerator
from tests.file_shards import (  # noqa: F401  (hooks pytest collects here)
    pytest_addoption,
    pytest_collect_file,
    pytest_sessionstart,
)
from tests.llm_miss_guard import (  # noqa: F401  (hooks pytest collects here)
    pytest_runtest_makereport,
    pytest_runtest_setup,
)
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, fake_catalogue_service
from tests.series_coverage.support import session_coverage_service

if TYPE_CHECKING:
    from entity_resolution import SharedResolver
    from search.engine.config import SearchSettings

# Set defaults at import time because tests build surfaces during collection.
# A zero recheck interval keeps the recorded coverage index stable during tests.
_COVERAGE_SLICE_DIR = Path(__file__).resolve().parent / "fixtures" / "coverage_slice"
os.environ.setdefault("COVERAGE_ARTIFACT_URI", f"file://{_COVERAGE_SLICE_DIR}")
os.environ.setdefault("COVERAGE_RECHECK_SECONDS", "0")
# Every bundle render links to the public website origin, which the surfaces require.
os.environ.setdefault("DFO_PUBLIC_SITE_URL", "https://site.example")

# Surfaces acquire the catalogue and coverage services at construction, some during collection,
# so both are installed for the whole session. Tests of either service isolate its singleton.
_session_services = ExitStack()


def pytest_configure(config: pytest.Config) -> None:
    _session_services.enter_context(
        catalogue_service_module._singleton.poisoned_for_test(fake_catalogue_service())
    )
    _session_services.enter_context(session_coverage_service())


def pytest_unconfigure(config: pytest.Config) -> None:
    _session_services.close()


class OfflineContractViolation(RuntimeError):
    """A test reached real network transport or a live client constructor.

    Not an AssertionError, so a guard hit is not mistaken for a failed expectation.
    """


def _refuse_sync_request(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
    raise OfflineContractViolation(
        f"outbound request to {request.url} reached httpx.HTTPTransport.handle_request. "
        "Offline tests must not make real HTTP calls; use httpx.MockTransport, an ASGI "
        "TestClient, or an injected client double instead."
    )


async def _refuse_async_request(
    self: httpx.AsyncHTTPTransport, request: httpx.Request
) -> httpx.Response:
    raise OfflineContractViolation(
        f"outbound request to {request.url} reached httpx.AsyncHTTPTransport."
        "handle_async_request. Offline tests must not make real HTTP calls; use "
        "httpx.MockTransport, an ASGI TestClient, or an injected client double instead."
    )


@pytest.fixture(autouse=True)
def _guard_outbound_http(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Block real HTTP transport during each non-live test across the suite.

    This catches unknown clients and bare HTTP calls. The construction guard catches un-injected
    live clients. MockTransport and ASGITransport remain available. Tests marked `live` are exempt.
    """
    if request.node.get_closest_marker("live"):
        yield
        return
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _refuse_sync_request)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _refuse_async_request)
    yield


# Modules whose fallback client constructors are patched.
_GUARDED_MODULES: tuple[str, ...] = ("search.engine.core", "search.engine.app")


# Patch targets and injection hints for clients that search builds by default.
_LIVE_CLIENT_TARGETS: list[tuple[str, str]] = [
    (
        "search.engine.core.LiveSdmxClient",
        "sdmx=FakeSdmxClient(observations={}, availability={})",
    ),
    ("search.engine.core.LiveGraphClient", "graph=FakeGraph()"),
    ("search.engine.core.LLM", "llm=FakeLLM()"),
    (
        "search.engine.app.LiveSdmxClient",
        "sdmx=FakeSdmxClient(observations={}, availability={})",
    ),
    ("search.engine.app.LiveGraphClient", "graph=FakeGraph()"),
    ("search.engine.app.LLM", "llm=FakeLLM()"),
    ("search.engine.app._llm_mod.warm", "llm=FakeLLM()"),
]


def _unimportable_guarded_modules() -> tuple[str, ...]:
    """Return guarded modules unavailable in this install.

    Missing modules have no constructors to patch. Stale attributes still fail during patching.
    """
    absent = []
    for module in _GUARDED_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            absent.append(module)
    return tuple(absent)


# `search.engine.config` builds its settings at import and requires SEARCH_GRAPH_BASE, so the
# guarded modules import against an unroutable placeholder when the caller sets none. The
# placeholder is removed again so the live tests' SEARCH_GRAPH_BASE gate, read at their import,
# sees only the caller's environment. `_offline_graph_base` sets it again for each offline test.
_OFFLINE_GRAPH_BASE = "https://graph.offline.invalid"
_placeholder_graph_base = "SEARCH_GRAPH_BASE" not in os.environ
if _placeholder_graph_base:
    os.environ["SEARCH_GRAPH_BASE"] = _OFFLINE_GRAPH_BASE
try:
    _UNIMPORTABLE_GUARDED_MODULES = _unimportable_guarded_modules()
finally:
    if _placeholder_graph_base:
        del os.environ["SEARCH_GRAPH_BASE"]
_UNGUARDABLE_PREFIXES = tuple(f"{module}." for module in _UNIMPORTABLE_GUARDED_MODULES)


def _refuse(target: str, fix: str):
    """Return a callable that names `target` and its injection fix in the error.

    Attach the same callable as `from_search_settings` for factory-based construction.
    """

    def _raise(*_args, **_kwargs):
        raise OfflineContractViolation(
            f"{target} was reached with no client injected. Pass {fix} instead."
        )

    _raise.from_search_settings = _raise
    return _raise


@pytest.fixture(autouse=True)
def _offline_graph_base(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each offline test the placeholder SEARCH_GRAPH_BASE when the caller sets none.

    A fresh `SearchSettings()` and a subprocess importing the engine both need the value.
    """
    if request.node.get_closest_marker("live") or "SEARCH_GRAPH_BASE" in os.environ:
        return
    monkeypatch.setenv("SEARCH_GRAPH_BASE", _OFFLINE_GRAPH_BASE)


@pytest.fixture(autouse=True)
def _guard_live_client_construction():
    """Make omitted live-client injections fail at construction in every test.

    Missing modules in reduced installs are skipped.
    Nested patches must use `new=` or `return_value=`, not `autospec=True`.
    The patched attribute is this guard, not the original class.
    """
    with ExitStack() as stack:
        for target, fix in _LIVE_CLIENT_TARGETS:
            if target.startswith(_UNGUARDABLE_PREFIXES):
                continue
            stack.enter_context(patch(target, new=_refuse(target, fix)))
        yield


@pytest.fixture(autouse=True)
def _close_datasets_clients() -> Iterator[None]:
    """Exit the Datasets test clients `make_client` entered, which stops each surface's
    lifespan. A run that never imported the factory module opened none."""
    yield
    factories = sys.modules.get("tests.datasets_api._app_factories")
    if factories is not None:
        factories.close_open_clients()


@pytest.fixture(autouse=True)
def _fresh_family_memo() -> Iterator[None]:
    """Start each test with no index family memoised (`bound_values._resolved_family`), so a
    test that replaces `index_family` reads its own families. A run that never imported the
    module holds no memo, and installs without the engine extras cannot import it."""
    bound_values = sys.modules.get("search.engine.retrieval.bound_values")
    if bound_values is not None:
        bound_values.clear_family_memo()
    yield


@pytest.fixture(autouse=True)
def _fresh_member_groups() -> Iterator[None]:
    """Start each test with no member group cached in the process's shared enumerator
    (`statvar_members.shared_enumerator`), so no test reads a group another test's graph
    double resolved under the same stamp key."""
    shared_enumerator().clear()
    yield


@pytest.fixture(autouse=True)
def _fresh_default_scope_memo() -> Iterator[None]:
    """Start and end each test with no default place scope memoised and no availability read
    in flight (`materialise.place_scope`), so no test reads a scope another test settled."""
    _settle_default_scopes()
    yield
    _settle_default_scopes()


def _settle_default_scopes() -> None:
    """Wait up to 10 s for every availability read in flight, those a read queued included, then
    clear the memo. A flight leaves `place_scope._flights` before it resolves, so a finished
    flight still listed is a leak and fails the test."""
    deadline = time.monotonic() + 10
    while flights := list(place_scope._flights.values()):
        pending = [flight for flight in flights if not flight.done()]
        assert pending, f"finished flights left in place_scope._flights: {flights}"
        assert time.monotonic() < deadline, "availability reads in flight never ended"
        futures.wait(pending, timeout=1)
    place_scope.clear_memo()


@pytest.fixture(autouse=True)
def _bound_fake_catalogue() -> Iterator[None]:
    """Bind a `FakeCatalogue`, which admits everything, for every test.

    Code that reads the request's origin catalogue then behaves as it did before admission.
    """
    with bound_catalogue(FakeCatalogue()):
        yield


@pytest.fixture
def bind_catalogue() -> Iterator[Callable[[Catalogue], None]]:
    """Return a function that binds a catalogue for the rest of the test, over the fake one."""
    tokens = []

    def _install(catalogue: Catalogue) -> None:
        tokens.append(_bind(catalogue))

    yield _install
    for token in reversed(tokens):
        unbind_catalogue(token)


@pytest.fixture
def published_catalogue_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Publish the sample origin catalogue and point `ORIGIN_CATALOGUE_URI` at it.

    For code that builds its own catalogue service, such as a subprocess.
    """
    from tests.origin_catalogue.support import publish_sample

    uri = f"file://{tmp_path / 'origin-catalogue'}"
    publish_sample(uri)
    monkeypatch.setenv("ORIGIN_CATALOGUE_URI", uri)
    return uri


@pytest.fixture
def search_settings(monkeypatch: pytest.MonkeyPatch) -> "Callable[..., SearchSettings]":
    """Return a factory for installing validated settings with keyword overrides.

    Each call layers onto current settings. `model_validate` recomputes derived fields.
    Overrides of the observation preview caps (`ObservationSettings` fields) install a new
    `search.observations` settings object, ahead of the search settings so the capability
    version sees them. `monkeypatch` restores settings at teardown. Lazy import lets reduced
    installs collect this fixture without the engine extra.
    """

    def _install(**overrides: object) -> "SearchSettings":
        from search import observations
        from search.engine import config
        from search.engine.config import SearchSettings
        from search.observations import ObservationSettings

        obs_overrides = {
            k: v for k, v in overrides.items() if k in ObservationSettings.model_fields
        }
        if obs_overrides:
            obs_base = observations.observation_settings().model_dump()
            monkeypatch.setattr(
                observations,
                "_SETTINGS",
                ObservationSettings.model_validate({**obs_base, **obs_overrides}),
            )
        search_overrides = {k: v for k, v in overrides.items() if k not in obs_overrides}
        base = config.settings().model_dump(
            exclude={"capability_version_override", "max_entity_occurrences"}
        )
        updated = SearchSettings.model_validate({**base, **search_overrides})
        monkeypatch.setattr(config, "_SETTINGS", updated)
        return updated

    return _install


@pytest.fixture(scope="session", autouse=True)
def _shared_resolvekit_pack(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator["SharedResolver | None"]:
    """Build the shared resolver once per session with a temporary offline cache, then release it.

    Yield None when resolvekit is absent in a reduced install. Other import errors propagate.
    """
    try:
        from entity_resolution import acquire_shared, release_shared
    except ModuleNotFoundError as exc:
        if exc.name != "resolvekit" and not (exc.name or "").startswith("resolvekit."):
            raise
        yield None
        return
    cache_dir = tmp_path_factory.mktemp("entity_resolution_shared")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("RESOLVEKIT_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("RESOLVEKIT_OFFLINE", "1")
    shared = acquire_shared(cache_dir=cache_dir)
    yield shared
    release_shared(shared)
    monkeypatch.undo()


def minimal_spec(spec_id: str = "s1") -> dict:
    """A contract-valid minimal Spec dict, shared across the test suite."""
    return {
        "spec_id": spec_id,
        "definition": {
            "schema_version": 1,
            # The canonical definition carries the same spec ID as its enclosing row.
            "spec_id": spec_id,
            "identity": {
                "identity_version": "spec1",
                "shape": {"kind": "statvar_schema", "population_type": "sh1"},
                "slots": [],
                "entities": {},
                "requested_window": {"kind": "all"},
            },
        },
        "resolved": {
            "shape": {
                "shape_id": "sh1",
                "label": "test shape",
                "population_type": {"ref": "Pop", "label": "Pop"},
                "measured_property": {"ref": "Prop", "label": "Prop"},
                "stat_type": {"ref": "measuredValue", "label": "measured value"},
                "slot_keys": [],
                "member_count": 1,
            },
            "slots": [],
            "stat_vars": {"member_count": 0},
            "entities": [],
            "coverage": {"kind": "bare", "has_data": True},
            "trace": {
                "resolved_entities": [],
                "resolved_sources": [],
                "slot_filters": [],
            },
        },
    }


def base_response(**extra) -> dict:
    """A minimal valid resolution envelope; pass status + body via kwargs."""
    return {
        "schema_version": 1,
        "kind": "resolution",
        "query_echo": {
            "entry_path": "raw_text",
            "extract_skipped": False,
        },
        "diagnostics": {"engine_build": "test-build", "trace_id": "test-trace", "warnings": []},
        **extra,
    }


_ETH_REF = {"ref": "country/ETH", "label": "Ethiopia"}
_SCHEME_PROP = {"ref": "DevelopmentFinanceScheme", "label": "flow type"}
_PURPOSE_PROP = {"ref": "DevelopmentFinancePurpose", "label": "purpose"}
_RECIPIENT_PROP = {"ref": "DevelopmentFinanceRecipient", "label": "recipient"}
_ODA_GRANTS_REF = {"ref": "ODAGrants", "label": "ODA grants"}
_DAC_HEALTH_REF = {"ref": "DAC/Health", "label": "health (total)"}


def _minimal_resolution() -> dict:
    """A contract-valid ResolutionTrace with all required fields populated."""
    return {
        "resolved_entities": [_ETH_REF],
        "resolved_sources": [],
        "slot_filters": [
            {
                "key": {"axis": "what", "property": _SCHEME_PROP, "label": "flow type"},
                "binding_kind": "value",
                "refs": [_ODA_GRANTS_REF],
            },
            {
                "key": {"axis": "how", "property": _PURPOSE_PROP, "label": "purpose"},
                "binding_kind": "value",
                "refs": [_DAC_HEALTH_REF],
            },
            {
                "key": {"axis": "where", "property": _RECIPIENT_PROP, "label": "recipient"},
                "binding_kind": "value",
                "refs": [_ETH_REF],
            },
            {
                "key": {"axis": "when", "label": "period"},
                "binding_kind": "value",
                "refs": [],
            },
            {
                "key": {"axis": "source", "label": "source"},
                "binding_kind": "unbound",
                "refs": [],
            },
        ],
    }


def _build_worked_example_response() -> dict:
    """The contract's worked example as a valid, one-part resolution dict."""
    return {
        "schema_version": 1,
        "kind": "resolution",
        "status": "definite",
        "query_echo": {
            "entry_path": "raw_text",
            "raw_query": "health ODA grants to Ethiopia since 2015",
            "normalized_query": "health ODA grants Ethiopia 2015",
            "extract_skipped": False,
        },
        "diagnostics": {
            "engine_build": "search-2026.06.18",
            "trace_id": "0" * 32,
            "warnings": [],
        },
        "parts": [
            {
                "status": "definite",
                "variable_text": "health ODA grants",
                "spec": {
                    "spec_id": "df-flow:ODAGrants:DAC/Health:country/ETH:2015:",
                    "definition": {
                        "schema_version": 1,
                        "spec_id": "df-flow:ODAGrants:DAC/Health:country/ETH:2015:",
                        "identity": {
                            "identity_version": "spec1",
                            "shape": {
                                "kind": "statvar_schema",
                                "population_type": "DevelopmentFinance",
                                "measured_property": "DevelopmentFinanceFlow",
                                "stat_type": "measuredValue",
                            },
                            "slots": [
                                {
                                    "property": "DevelopmentFinanceScheme",
                                    "binding": {"state": "value", "ref": "ODAGrants"},
                                },
                                {
                                    "property": "DevelopmentFinancePurpose",
                                    "binding": {"state": "value", "ref": "DAC/Health"},
                                },
                            ],
                            "entities": {"destination": ["country/ETH"]},
                            "requested_window": {"kind": "range", "start_year": 2015},
                        },
                    },
                    "resolved": {
                        "shape": {
                            "shape_id": "df-flow",
                            "label": "development finance flows",
                            "population_type": {
                                "ref": "DevelopmentFinance",
                                "label": "development finance",
                            },
                            "measured_property": {
                                "ref": "DevelopmentFinanceFlow",
                                "label": "development finance flow",
                            },
                            "stat_type": {"ref": "measuredValue", "label": "measured value"},
                            "slot_keys": [
                                {"axis": "what", "property": _SCHEME_PROP, "label": "flow type"},
                                {"axis": "how", "property": _PURPOSE_PROP, "label": "purpose"},
                                {
                                    "axis": "where",
                                    "property": _RECIPIENT_PROP,
                                    "label": "recipient",
                                },
                                {"axis": "when", "label": "period"},
                                {"axis": "source", "label": "source"},
                            ],
                            "member_count": 27,
                        },
                        "slots": [
                            {
                                "key": {
                                    "axis": "what",
                                    "property": _SCHEME_PROP,
                                    "label": "flow type",
                                },
                                "binding": {
                                    "kind": "value",
                                    "value": {"ref": _ODA_GRANTS_REF, "value_kind": "enum_value"},
                                },
                            },
                            {
                                "key": {
                                    "axis": "how",
                                    "property": _PURPOSE_PROP,
                                    "label": "purpose",
                                },
                                "binding": {
                                    "kind": "value",
                                    "value": {"ref": _DAC_HEALTH_REF, "value_kind": "enum_value"},
                                },
                            },
                            {
                                "key": {
                                    "axis": "where",
                                    "property": _RECIPIENT_PROP,
                                    "label": "recipient",
                                },
                                "binding": {
                                    "kind": "value",
                                    "value": {"ref": _ETH_REF, "value_kind": "entity"},
                                },
                            },
                            {
                                "key": {"axis": "when", "label": "period"},
                                "binding": {
                                    "kind": "value",
                                    "value": {
                                        "value_kind": "time_window",
                                        "time_window": {"start_year": 2015},
                                    },
                                },
                            },
                            {
                                "key": {"axis": "source", "label": "source"},
                                "binding": {"kind": "unbound"},
                            },
                        ],
                        "stat_vars": {"member_count": 1, "served_count": 1},
                        "entities": [
                            {
                                "ref": _ETH_REF["ref"],
                                "label": _ETH_REF["label"],
                                "entity_type": "Country",
                                "role": "destination",
                            }
                        ],
                        "coverage": {
                            "kind": "breadth",
                            "has_data": True,
                            "dimensions": [
                                {"label": "donors", "count": 32},
                                {"label": "years", "count": 9},
                            ],
                            "window": {"start_year": 2015, "end_year": 2023},
                        },
                        "trace": _minimal_resolution(),
                    },
                },
            }
        ],
    }


@pytest.fixture
def worked_example_response() -> dict:
    """Return the v1 worked example as a valid resolution dict."""
    return _build_worked_example_response()
