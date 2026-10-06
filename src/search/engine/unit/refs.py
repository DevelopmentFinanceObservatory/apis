"""Check unit requests against graph unit dcids and the request grammar.

`satisfies` and `admits` test a native facet: the graph maps unit dcids to ISO 4217
codes, and a native unit ref covers current prices with no base year. Unknown dcids
and failed reads return False, and a role-shaped currency never matches a concrete
graph code. Every graph read takes the caller's absolute monotonic `deadline`, and an
expired deadline raises instead of naming no currency. `expressible` asks whether the engine can serve a request at all, which
conversion widens to every ISO code and role shape at either price basis. Whether
each StatVar can be converted is decided later, per member, by
`materialise.member_conversion.decide_member_conversion`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import dc_graph
from materialise.facets import native_currency_iso
from search.currency_codes import ISO_4217_CODES
from search.engine.clients.graph import EngineGraphClient
from search.engine.degraded_reads import note_degraded_read
from search.engine.errors import GraphInfraError
from search.heading.unit_class import is_known_non_currency
from search.models import IdentityCurrencyRole, IdentityUnitRequest

logger = logging.getLogger(__name__)


def _graph_currency(
    graph: EngineGraphClient, unit_dcid: str, *, deadline: float | None
) -> str | None:
    """The ISO 4217 code the graph's currency units map `unit_dcid` to, or None. A percent
    or PPP unit (`heading.unit_class.is_known_non_currency`) names none with no read. An
    expired `deadline` raises; any other failed read names no currency, logs a warning
    naming `unit_dcid` and notes the lost read (`degraded_reads.note_degraded_read`)."""
    if is_known_non_currency(unit_dcid):
        return None
    try:
        return graph.currency_units(deadline=deadline).get(unit_dcid)
    except dc_graph.GraphDeadlineExceededError:
        raise
    except GraphInfraError as exc:
        logger.warning("currency units read failed; unit %s names no currency: %s", unit_dcid, exc)
        note_degraded_read("currency units")
        return None


def unit_currency(
    graph: EngineGraphClient, unit_dcid: str, *, deadline: float | None
) -> str | None:
    """The ISO 4217 code `unit_dcid` is denominated in, or None when it is no currency.

    A `Currency<ISO>` ref naming an ISO 4217 code, or a `CurrencyUnitOfMeasure` ref, names
    its code (`materialise.facets.native_currency_iso`) with no graph read. Any other ref
    is read from the graph's currency units, which hold the base Data Commons `Currency`
    members (`INR`), within the absolute monotonic `deadline`. An expired deadline raises;
    any other failed read names no currency.
    """
    native = native_currency_iso(unit_dcid)
    if native in ISO_4217_CODES:
        return native
    return _graph_currency(graph, unit_dcid, deadline=deadline)


def _is_current_no_base_year(request: IdentityUnitRequest) -> bool:
    """Whether `request` asks for current prices with no base year -- the
    only prices/base_year shape a native unit ref can ever satisfy."""
    return request.prices == "current" and request.base_year is None


def satisfies(
    graph: EngineGraphClient,
    request: IdentityUnitRequest,
    unit_dcid: str,
    *,
    deadline: float | None,
) -> bool:
    """True when the request has current prices, no base year, and the mapped currency.
    The graph read honours the absolute monotonic `deadline`, and expiry raises."""
    if not _is_current_no_base_year(request):
        return False
    return _graph_currency(graph, unit_dcid, deadline=deadline) == request.currency


def admits(
    graph: EngineGraphClient, request: IdentityUnitRequest | None, *, deadline: float | None
) -> Callable[[str], bool] | None:
    """Return a unit-dcid predicate bounded by `deadline`, or None when no request is bound."""
    if request is None:
        return None
    return lambda u: satisfies(graph, request, u, deadline=deadline)


def expressible(request: IdentityUnitRequest) -> bool:
    """Whether the engine can serve `request`, regardless of a Spec's facets.

    A role shape or a registered ISO 4217 code is expressible at either price basis.
    A pattern-valid code outside ISO 4217 is not. A false result maps to
    ``facet_binding_unsupported``. An expressible request that no member can be
    converted to refuses later, per member, with ``recipe_not_declared`` or
    ``rate_coverage_gap``.
    """
    return isinstance(request.currency, IdentityCurrencyRole) or request.currency in ISO_4217_CODES
