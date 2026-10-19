"""The interface the engine reads the stated quantity and elects a shape through.

A `Binder` answers the two model questions of a part: which form and base the variable
phrase states (`Binder.read_quantity`), and which menu shape the part means with its slot
bindings (`Binder.bind`). The engine builds one `BindRequest` per bind call and reads the
binder's `BindResult`; it never reaches into a binder's model calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from search.engine.bind import CandidateMenu, _UniversalBindOutput
    from search.engine.binders.house_policy import PolicyChoice, PolicyPool
    from search.engine.clients.llm import SupportsLLM
    from search.engine.frame import Frame
    from search.engine.interpretation import Interpretation
    from search.engine.qualifier_match import QualifierMatch, QualifierTie
    from search.engine.quantity_read import QuantityOutput
    from search.engine.shape import ShapeDraft
    from search.models import GraphRef


class ApplyPolicies(Protocol):
    """The engine's house-policy step (`house_policy.apply_house_policies`): from the pool a
    binder elects among, its pick first in `PolicyPool.measuring`, the binder's rating of
    each pool shape when it rates them, and whether each pool shape holds the constraints
    the binder recognised on its pick (`holds`), the shape the part answers with. A binder
    calls it once on a reply that elects a menu shape and never on one that does not."""

    def __call__(
        self,
        pool: PolicyPool,
        *,
        ratings: Mapping[int, float] | None = None,
        holds: Mapping[int, bool] | None = None,
    ) -> PolicyChoice: ...


class PolicyContractError(Exception):
    """A binder broke the house-policy contract: the step ran other than once on a reply
    that elects a shape, or the election moved to a shape that does not hold a constraint
    the binder recognised on its pick. The part refuses (`phase_rank_bind._phase_bind`)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BindRequest:
    """One bind call: the menu and its context as `bind.bind_universal` takes them, the
    query's own text, and the engine's house-policy step.

    `shapes` is the seated menu in rank order, and `admitted_qualifiers`, `admitted_ties`,
    `candidate_menus` and `availability` are each shape's own, in that order. `query_text`
    is the verbatim query of a single-variable raw-text request, else the query the part's
    roles were read from.
    """

    variable: str
    shapes: list[ShapeDraft]
    admitted_qualifiers: Sequence[tuple[QualifierMatch, ...]] = ()
    admitted_ties: Sequence[tuple[QualifierTie, ...]] = ()
    withheld_measure_axes: frozenset[str] = frozenset()
    dropped_qualifier_phrases: tuple[str, ...] = ()
    candidate_menus: Sequence[tuple[CandidateMenu, ...]] = ()
    frame: Frame | None = None
    refused: Sequence[Interpretation] = ()
    availability: Sequence[bool | None] = ()
    query_text: str | None
    apply_policies: ApplyPolicies


@dataclass(frozen=True, slots=True, kw_only=True)
class BindResult:
    """A binder's answer to one `BindRequest`, with what the engine reads beside it.

    `output` is the binder's reply in the bind schema, with the engine's rules applied to
    its drafts (`bind.ground_bind_output`), before the drafts on slots the elected shape
    lacks leave it (`bind.split_stray_bindings`), and with the house policies' election in
    place of the binder's. `policy_pool` is the pool the house policies chose from and
    `policy_choice` what they chose, both None on a reply that elects no shape.
    `also_plausible_support` is the binder's support for each `also_plausible` index, and
    None for a binder that states none. `binder` names the binder that answered.
    """

    output: _UniversalBindOutput
    binder: str
    policy_pool: PolicyPool | None = None
    policy_choice: PolicyChoice | None = None
    also_plausible_support: dict[int, float] | None = None


class Binder(Protocol):
    """A model that reads a part's stated quantity and elects and binds its shape."""

    @property
    def name(self) -> str:
        """The binder's name, as `Settings.binder` and the response cache key state it."""
        ...

    @property
    def model(self) -> str:
        """The model the binder answers with, which the response cache key carries."""
        ...

    async def read_quantity(
        self, variable: str, denominators: Sequence[GraphRef], *, llm: SupportsLLM
    ) -> tuple[QuantityOutput, dict | None]:
        """The form and the base among `denominators` the variable phrase states, and the
        call's token usage (`quantity_read.read_stated_quantity`)."""
        ...

    async def bind(
        self, request: BindRequest, *, llm: SupportsLLM
    ) -> tuple[BindResult, dict | None]:
        """The shape `request`'s menu elects with its slot bindings, and the call's token
        usage. Raises `LLMOutputRefused` for a reply the bind schema refuses."""
        ...
