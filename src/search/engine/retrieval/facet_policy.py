"""Refuse a StatVar whose graph declares a default facet, before coverage is built.

Materialise serves a declared default before it ranks facets, while search coverage, the
headline and the preview route rank alone (`retrieval.coverage`). Resolve refuses such a
StatVar so that no sentence, coverage window or headline names a facet the preview would
not serve.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from materialise.facets import FACET_AXES
from search.engine.errors import DeclaredDefaultUnsupportedError

if TYPE_CHECKING:
    from search.engine.clients.graph import EngineGraphClient

# Materialise asks for policies by the `dcid:`-prefixed ref its requests carry.
_STAT_VAR_REF_PREFIX = "dcid:"


def refuse_declared_defaults(graph: EngineGraphClient, stat_vars: Iterable[str]) -> None:
    """Raise `DeclaredDefaultUnsupportedError` when the graph declares a default facet on
    any axis of any of `stat_vars`."""
    for stat_var in stat_vars:
        for axis in FACET_AXES:
            policy = graph.default_facet_policy(
                stat_var=f"{_STAT_VAR_REF_PREFIX}{stat_var}", axis=axis
            )
            if policy is not None:
                raise DeclaredDefaultUnsupportedError(
                    f"{stat_var!r} declares default {policy.resolved_ref!r} on facet axis "
                    f"{axis!r}; search coverage elects facets by ranking and models no default"
                )
