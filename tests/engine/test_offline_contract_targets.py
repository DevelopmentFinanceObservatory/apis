"""Checks that `_LIVE_CLIENT_TARGETS` covers every live fallback in core and app."""

from __future__ import annotations

import inspect
from types import ModuleType

import search.engine.app as app_module
import search.engine.core as core_module
from tests.conftest import (
    _GUARDED_MODULES,
    _LIVE_CLIENT_TARGETS,
    _UNIMPORTABLE_GUARDED_MODULES,
)


def _live_fallback_symbols(module: ModuleType) -> set[str]:
    found: set[str] = set()
    for name, value in vars(module).items():
        if inspect.isclass(value) and (name.startswith("Live") or name == "LLM"):
            found.add(f"{module.__name__}.{name}")
        elif isinstance(value, ModuleType) and inspect.isfunction(getattr(value, "warm", None)):
            found.add(f"{module.__name__}.{name}.warm")
    return found


# Computed at import time, before `_guard_live_client_construction` (an
# autouse fixture) has patched any of these symbols away for a running test
# -- computing this inside the test body would see the guard's own `_refuse`
# stand-ins instead of the real classes it is patching.
_EXPECTED_LIVE_FALLBACK_SYMBOLS = _live_fallback_symbols(core_module) | _live_fallback_symbols(
    app_module
)


def test_live_client_targets_covers_every_fallback_symbol_in_core_and_app():
    actual = {target for target, _fix in _LIVE_CLIENT_TARGETS}
    assert actual == _EXPECTED_LIVE_FALLBACK_SYMBOLS


def test_every_target_lives_in_a_guarded_module():
    """Targets outside `_GUARDED_MODULES` break the guard in extras-partial installs."""
    prefixes = tuple(f"{module}." for module in _GUARDED_MODULES)
    for target, _fix in _LIVE_CLIENT_TARGETS:
        assert target.startswith(prefixes), target


def test_the_guard_is_fully_armed_wherever_the_engine_tests_run():
    """Both modules import here, so the guard must skip none of its targets."""
    assert _UNIMPORTABLE_GUARDED_MODULES == ()
