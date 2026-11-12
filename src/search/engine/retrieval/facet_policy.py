"""Read the default facet each admitted StatVar declares in the graph.

A mint serves a StatVar's declared default on an unbound axis ahead of the deployment
ranking (`materialise.facets.select_facet`, `materialise.group_facets.elect_group`). Search
reads the same declarations once for every admitted StatVar, and every read plan elects under
them from its first election (`read_plan.plan_pairs`, `read_plan.role_election`), so a
preview, the answer's facets, an option's coverage and a download serve one method.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from search.engine.errors import GraphInfraError

if TYPE_CHECKING:
    from materialise.graph import FacetDefaultPolicy
    from search.engine.clients.graph import EngineGraphClient

logger = logging.getLogger(__name__)

type DeclaredPolicies = Mapping[str, Mapping[str, FacetDefaultPolicy]]
"""The defaults each StatVar declares, by StatVar and then by facet axis. A StatVar or axis
declaring nothing is absent."""

NO_DECLARED_POLICIES: DeclaredPolicies = MappingProxyType({})
"""The declarations of StatVars that declare nothing."""


def read_declared_policies(
    graph: EngineGraphClient, stat_vars: Iterable[str], *, deadline: float | None
) -> DeclaredPolicies:
    """The default each of `stat_vars` declares on each facet axis, the declaration a mint of
    each reads (`GraphReader.default_facet_policy`), in one read
    (`EngineGraphClient.default_facet_policies`). The origin client answers a variable the
    catalogue holds from its arc rows and reads the rest in one batch, by `deadline` and the
    request deadline bound to the context. A graph read error declares nothing for any of
    them and logs one warning, and the request goes on under the deployment ranking."""
    dcids = list(dict.fromkeys(stat_vars))
    if not dcids:
        return NO_DECLARED_POLICIES
    try:
        return graph.default_facet_policies(dcids, deadline=deadline)
    except GraphInfraError as exc:
        logger.warning(
            "declared default facets unread for %d StatVars, so none applies: %s",
            len(dcids),
            exc,
        )
        return NO_DECLARED_POLICIES
