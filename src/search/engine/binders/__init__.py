"""The binders the engine reads a part's quantity and elects its shape through
(`base.Binder`), and the one `Settings.binder` names."""

from __future__ import annotations

from collections.abc import Callable

from search.engine.binders.base import ApplyPolicies, Binder, BindRequest, BindResult
from search.engine.binders.gemini import GeminiBinder
from search.engine.config import settings

_BINDERS: dict[str, Callable[[], Binder]] = {"gemini": GeminiBinder}


def configured_binder() -> Binder:
    """The binder `Settings.binder` names (``SEARCH_BINDER``)."""
    return _BINDERS[settings().binder]()


__all__ = [
    "ApplyPolicies",
    "BindRequest",
    "BindResult",
    "Binder",
    "GeminiBinder",
    "configured_binder",
]
