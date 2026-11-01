"""Check unit requests against graph unit dcids and the request grammar.

`satisfies` and `admits` test a native facet. The unit registry
(`dfo_spec_kit.units.parse_unit`) gives a money unit's currency, and a priced one's prices
and base year; the graph maps any other unit dcid to an ISO 4217 code, and a unit whose
prices the registry leaves unstated covers current prices with no base year. Unknown dcids
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

from dfo_spec_kit.units import Money, UnpricedMoney, parse_unit

import dc_graph
from search.engine.clients.graph import EngineGraphClient
from search.engine.degraded_reads import note_degraded_read
from search.engine.errors import GraphInfraError
from search.models import IdentityCurrencyRole, IdentityUnitRequest
from unit_words.currency_codes import ISO_4217_CODES
from unit_words.unit_class import is_known_non_currency

logger = logging.getLogger(__name__)


def _graph_currency(
    graph: EngineGraphClient, unit_dcid: str, *, deadline: float | None
) -> str | None:
    """The ISO 4217 code `unit_dcid` is denominated in, or None. A unit the registry reads
    as money names its currency, and any other unit it types, or a PPP unit
    (`unit_class.is_known_non_currency`), names none, both with no read. The graph's
    currency units map the rest. An expired `deadline` raises; any other failed read names
    no currency, logs a warning naming `unit_dcid` and notes the lost read
    (`degraded_reads.note_degraded_read`)."""
    value = parse_unit(unit_dcid)
    if isinstance(value, Money | UnpricedMoney):
        return value.currency
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

    A ref the unit registry reads as money (`Currency<ISO>`, `USDollar`, a bare `INR`,
    `SDG_CU_USD`) names its code with no graph read. Any other ref the registry leaves
    untyped is read from the graph's currency units within the absolute monotonic
    `deadline`. An expired deadline raises; any other failed read names no currency.
    """
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
    """True when the unit is in the request's currency at its prices and base year. A unit
    the registry prices (`Money`) states its own, and a chained one satisfies no request. Any other unit covers current prices with no base year. The graph
    read honours the absolute monotonic `deadline`, and expiry raises."""
    value = parse_unit(unit_dcid)
    if isinstance(value, Money):
        return (
            value.currency == request.currency
            and value.prices == request.prices
            and value.base_year == request.base_year
            and not value.chained
        )
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
