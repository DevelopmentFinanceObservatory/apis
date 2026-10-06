"""Wrap every binding of `option_coverage._maybe_build_facet_options`, so a latency test can
tell the requests a Spec's facet options make from the rest of its answer's.

Facet options are built for every definite Spec and candidate whose coverage is not
streamed. Their facets read and label batch are a cost the latency tests grant, and only
those requests are granted.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import pytest

from search.engine.region import option_coverage

type Build = Callable[..., Any]


def wrap_facet_option_builds(
    monkeypatch: pytest.MonkeyPatch, wrap: Callable[[Build], Build]
) -> None:
    """Replace `_maybe_build_facet_options` with `wrap(it)` in every engine module that binds
    it, the module that defines it included."""
    original = option_coverage._maybe_build_facet_options
    wrapped = wrap(original)
    modules = [
        module
        for name, module in list(sys.modules.items())
        if name.startswith("search.engine")
        and getattr(module, "_maybe_build_facet_options", None) is original
    ]
    assert modules, "no engine module binds _maybe_build_facet_options"
    for module in modules:
        monkeypatch.setattr(module, "_maybe_build_facet_options", wrapped)
