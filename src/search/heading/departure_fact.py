"""The engine's record of one way a part's answer differs from its request, without text.

`search.engine.departures` raises these facts. `search.render.departures` writes each
one's text. The structured values a text states that `asked` and `shown` do not carry ride
on fields named for the codes that set them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from search.models import DEPARTURE_ORDER, DepartureCode, OperationVerb, RoleAxis


@dataclass(frozen=True)
class DepartureFact:
    """One way a part's answer differs from its request, for code.

    `asked` is what the request named, in its own wording. `shown` is what the answer
    shows instead, when the engine holds it as text. Either is `None` when there is
    nothing to state.

    The remaining fields are set by the codes named on each, and are `None` (or `False`)
    on every other code:

    - `verb` (`operation_not_applied`): the operation the request named.
    - `over` (`breakdown_not_applied`, `direction_discarded`): the flow axis the request
      broke the answer down by, or named the place on.
    - `entities` (`entity_unavailable`): the names of the entities the data holds no
      values for, in the answer's entity order.
    - `group` and `group_size` (`entity_unavailable`, `group_read_as_members`): the label
      and member count of the group the request named for the role, when it named one;
      `asked` is then the request's own words for the group, `None` on a door that holds
      none.
    - `sources` (`entity_unavailable`): the names of sources that hold values for the
      entities but that Search does not read, because the origin registry withholds them.
    - `latest_years` (`entity_unavailable`): on a `latest` request, the first and last
      years of the latest period the answer serves (`ResolvedSpec.coverage.window`), in
      which the data holds no values for the entities.
    - `readings` (`closest_match`): the part shows several readings in place of the
      measure asked, each a measure the data holds.
    - `requested_years` (`year_unavailable`): the requested first and last years, either
      open.
    - `data_years` (`year_unavailable`): the first and last years the data holds.
    - `converted_years` and `limited_by` (`year_unavailable`): on an answer a unit request
      converts over fewer years than the data holds, the first and last years the
      conversion covers, and the table that leaves the rest out (exchange rates or
      deflators). `data_years` is then the years the data serves before the conversion.
    - `omitted_years` (`year_unavailable`): on an answer a constant-price unit request
      converts, each entity whose rows a missing deflator leaves out, by name, with the
      years it leaves out and the years it keeps, in the order the text names them.
      `omitted_unnamed` counts the entities it leaves out that no held name covers, and
      `omission_undecided` marks an answer whose conversion may leave out rows when the
      loaded tables cannot say whose; it names no entity.
    - `partial` (`modelled_values`): only some of the served series are modelled.
    - `limits` (`scope_limited`): the values a no-data check covered that the request did
      not name, each as the code index names it.
    - `values` (`measure_not_applied`): on a door that holds no words (a resubmit, `asked`
      is `None`), the names of the posted values the answer does not apply.
    """

    code: DepartureCode
    binding_ref: str | None
    asked: str | None = None
    shown: str | None = None
    verb: OperationVerb | None = None
    over: RoleAxis | None = None
    entities: tuple[str, ...] = ()
    group: str | None = None
    group_size: int | None = None
    sources: tuple[str, ...] = ()
    latest_years: tuple[int, int] | None = None
    readings: bool = False
    requested_years: tuple[int | None, int | None] | None = None
    data_years: tuple[int, int] | None = None
    converted_years: tuple[int, int] | None = None
    limited_by: Literal["exchange_rates", "deflators"] | None = None
    omitted_years: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] = ()
    omitted_unnamed: int = 0
    omission_undecided: bool = False
    partial: bool = False
    limits: tuple[str, ...] = ()
    values: tuple[str, ...] = ()


def ordered_facts(facts: Iterable[DepartureFact]) -> tuple[DepartureFact, ...]:
    """`facts` without repeats, in `DEPARTURE_ORDER`, raise order kept within a code."""
    distinct = dict.fromkeys(facts)
    return tuple(sorted(distinct, key=lambda fact: DEPARTURE_ORDER.index(fact.code)))
