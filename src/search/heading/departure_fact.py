"""The engine's record of one way a part's answer differs from its request, without text.

`search.engine.departures` raises these facts. `search.render.departures` writes each
one's text. The structured values a text states that `asked` and `shown` do not carry ride
on fields named for the codes that set them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from search.models import DEPARTURE_ORDER, DepartureCode, OperationVerb, PartyClass, RoleAxis


@dataclass(frozen=True)
class UnservedMembers:
    """The members of one set the request named that the data holds no values for, on a
    role that binds every member of the set beside other places.

    `label`, `dcid` and `size` are the set's label, its node's dcid (`EntityGroup.ref`) and
    its member count, `asked` the request's own words for it, and `entities` the names of
    its members without values, in the answer's entity order.
    """

    label: str
    dcid: str | None
    size: int
    asked: str
    entities: tuple[str, ...]


@dataclass(frozen=True)
class DepartureFact:
    """One way a part's answer differs from its request, for code.

    `asked` is what the request named, in its own wording. `shown` is what the answer
    shows instead, when the engine holds it as text. Either is `None` when there is
    nothing to state.

    The remaining fields are set by the codes named on each, and are `None` (or `False`)
    on every other code:

    - `verb` (`operation_not_applied`): the operation the request named.
    - `over` (`breakdown_not_applied`, `direction_discarded`, `direction_inferred`,
      `whole_total_read`): the flow axis the request broke the answer down by, named the
      place on, the axis the answer reads a place named without a direction on, or the side
      the answer reads as its whole population.
    - `entities` (`entity_unavailable`, `whole_total_read`): the names of the entities the
      data holds no values for, in the answer's entity order; or, for a whole population
      read as a union of totals, the names of those totals.
    - `sets` (`entity_unavailable`): on a role that binds the members of sets the request
      named beside other places, each set with the members the data holds no values for.
      `entities` then names only the places the request named one by one.
    - `group`, `group_size` and `group_dcid` (`entity_unavailable`,
      `group_read_as_members`): the label, the member count and the node's dcid
      (`EntityGroup.ref`) of the group the request named for the role, when
      it named one; `asked` is then the request's own words for the group, `None` on a
      door that holds none.
    - `group` (`whole_total_read`): the label of the whole population the side reads
      (`EntityGroup.label`); `asked` is the request's own words for it, `None` when the
      request named nothing on the side or its door holds no words.
    - `whole_of` (`whole_total_read`): the classes of party the population covers
      (`EntityGroup.whole_of`).
    - `side_classes` (`whole_total_read`): the classes of party, totals aside, the answer's
      dataset holds on the side (`EntityGroup.side_classes`).
    - `inferred` (`whole_total_read`): the request named nothing on the side, so the answer
      read it whole. False on a door that holds no words of the request (a resubmit).
    - `members_in_words` (`whole_total_read`): the part's door asks for a side's members in
      the question's words ("for each donor"), as the raw-text door does, where the other doors
      post `open_roles`.
    - `shown`, `over` and `whole_of` (`whole_total_unpublished`): the name of the dataset
      that publishes no total of the whole population the request named, the side it
      leaves unnamed, and the classes of party the dataset holds there; `asked` is the
      request's words for the population.
    - `excluded` (`group_read_as_members`): the labels of the residuals the total of the
      group counts beside its members ("Africa, unallocated by country (DAC)"), which the
      members leave out, in the code index's order. Empty for a group no total holds.
    - `included` (`group_read_as_members`): the labels of the residuals the role binds
      beside the members, where the part sums the group with them (`scope.region`), in the
      answer's entity order.
    - `sum_differs` (`group_read_as_members`): a dataset the answer serves publishes, on the
      role's flow axis, its own figure for the group, which can differ from one built from its
      members (`PartyRow.sums_members` false).
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
    - `values` (`measure_not_applied`): when `asked` is `None`, what the answer does not
      apply: on a door that holds no words (a resubmit) the names of the posted values, and
      for a dropped value no name exists for its slot's words ("a requested gender
      value").
    """

    code: DepartureCode
    binding_ref: str | None
    asked: str | None = None
    shown: str | None = None
    verb: OperationVerb | None = None
    over: RoleAxis | None = None
    entities: tuple[str, ...] = ()
    sets: tuple[UnservedMembers, ...] = ()
    group: str | None = None
    group_size: int | None = None
    group_dcid: str | None = None
    excluded: tuple[str, ...] = ()
    included: tuple[str, ...] = ()
    sum_differs: bool = False
    whole_of: tuple[PartyClass, ...] = ()
    side_classes: tuple[PartyClass, ...] = ()
    inferred: bool = False
    members_in_words: bool = False
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
