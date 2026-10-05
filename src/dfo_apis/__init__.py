"""Compose Search, Datasets, Graph, Presentation, and Skills onto `dfo_platform`.

The platform imports no API surfaces, so each API can depend on it without
pulling in a sibling surface.
"""

from __future__ import annotations

from dfo_apis.app import create_app

__all__ = ["create_app"]
