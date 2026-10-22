"""The interface the engine reads the stated quantity and elects a shape through.

A `Binder` answers the model questions of a part: which form and base the variable
phrase states (`Binder.read_quantity`), which menu shape the part means with its slot
bindings (`Binder.bind`), and how likely the elected shape's series states each word of
the variable the coverage readers leave unread (`Binder.rate_words_stated`). The engine
builds one `BindRequest` per bind call and reads the binder's `BindResult`. The quantity
read takes one `QuantityRequest`, and the rating one `StatedWordsRequest`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from search.engine.bind import CandidateMenu, _UniversalBindOutput
    from search.engine.binders.house_policy import PolicyChoice, PolicyName, PolicyPool
    from search.engine.clients.llm import SupportsLLM
    from search.engine.frame import Frame
    from search.engine.interpretation import Interpretation
    from search.engine.qualifier_match import QualifierMatch, QualifierTie
    from search.engine.quantity_read import QuantityOutput
    from search.engine.shape import Pool, ShapeDraft
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


class PolicyChoosesOn(Protocol):
    """Whether a house policy, run over the menu shapes at `indices` alone, would choose
    between their values on the shape-kind measure axis `axis` (`house_policy.chooses_on`).
    `policies` limits the policies asked. It runs no policy step."""

    def __call__(
        self, indices: Sequence[int], axis: str, *, policies: Sequence[PolicyName] = ...
    ) -> bool: ...


class PolicyContractError(Exception):
    """A binder broke the house-policy contract. The step ran other than once on a reply
    that elects a shape, or the election moved to a shape that does not hold a constraint
    the binder recognised on its pick. The part refuses (`phase_rank_bind._phase_bind`)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class QuantityRequest:
    """One quantity read: the variable phrase less the words its places own (`variable`),
    the numbered StatVar denominators the admissible shapes divide by, the phrase the bind
    call reads (`bind_variable`) and the query's own text, as `BindRequest.query_text`
    holds it. The Gemini binder reads `variable` and `denominators` alone. `part` and
    `pool` are the part's index and pass, as the ``bind_decision`` line states them, and
    join a binder's log lines to it; None outside a resolve."""

    variable: str
    denominators: Sequence[GraphRef]
    bind_variable: str
    query_text: str | None
    part: int | None = None
    pool: Pool | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BindRequest:
    """One bind call: the menu and its context as `bind.bind_universal` takes them, the
    query's own text, and the engine's house-policy step.

    `shapes` is the seated menu in rank order, and `admitted_qualifiers`, `admitted_ties`,
    `candidate_menus` and `availability` are each shape's own, in that order. `query_text`
    is the verbatim query of a single-variable raw-text request, else the query the part's
    roles were read from. `policy_chooses_on` asks whether a house policy would choose on
    a measure axis among menu shapes, None when no policy reads the menu. `part` and `pool`
    join a binder's log lines to the part's, as on `QuantityRequest`.

    `nominated_words` holds, for each shape in menu order, the casefolded words of the
    variable that the readers reading no binding leave unread on it
    (`phase_rank_bind._nominated_words`). Every word `Binder.rate_words_stated` is later
    asked to rate on the elected shape is among its shape's, so a binder may answer the
    rating from questions it asks in the bind.
    """

    variable: str
    shapes: list[ShapeDraft]
    admitted_qualifiers: Sequence[tuple[QualifierMatch, ...]] = ()
    admitted_ties: Sequence[tuple[QualifierTie, ...]] = ()
    withheld_measure_axes: frozenset[str] = frozenset()
    dropped_qualifier_phrases: tuple[str, ...] = ()
    candidate_menus: Sequence[tuple[CandidateMenu, ...]] = ()
    nominated_words: Sequence[tuple[str, ...]] = ()
    frame: Frame | None = None
    refused: Sequence[Interpretation] = ()
    availability: Sequence[bool | None] = ()
    query_text: str | None
    apply_policies: ApplyPolicies
    policy_chooses_on: PolicyChoosesOn | None = None
    part: int | None = None
    pool: Pool | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BindResult:
    """A binder's answer to one `BindRequest`, with what the engine reads beside it.

    `output` is the binder's reply in the bind schema, with the engine's rules applied to
    its drafts (`bind.ground_bind_output`), before the drafts on slots the elected shape
    lacks leave it (`bind.split_stray_bindings`), and with the house policies' election in
    place of the binder's. `policy_pool` is the pool the house policies chose from and
    `policy_choice` what they chose, both None on a reply that elects no shape. `binder`
    names the binder that answered. `scores` holds the scores the binder's decisions read,
    each a per-shape list in menu order or one number for the part, as the
    ``bind_decision`` line logs them; empty when the binder states none. `none_first` says
    the binder elected a shape though it rated "none of the shapes" above every one, so the
    UN System Data Commons' pass may replace the answer (`region.pipeline._fill_gap`); a
    binder that offers no such rating leaves it False. `stated_words` holds the
    probability, by menu index and nominated word (`BindRequest.nominated_words`), that
    the shape's series states what the word names, for a binder that asks it in the bind;
    empty for one that does not.
    """

    output: _UniversalBindOutput
    binder: str
    policy_pool: PolicyPool | None = None
    policy_choice: PolicyChoice | None = None
    scores: Mapping[str, Sequence[float] | float] = field(default_factory=dict)
    none_first: bool = False
    stated_words: Mapping[tuple[int, str], float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class StatedWordsRequest:
    """One rating of the words of the variable the coverage readers leave unread on the
    elected shape over the final bindings (`phase_rank_bind._unrated_words`): the variable
    phrase, the elected shape and its index in the bind menu (None for a refused reading
    standing in for a declined menu), the casefolded words, each among the shape's
    `BindRequest.nominated_words`, and the bind call's result. `part` and `pool` join a
    binder's log lines to the part's, as on `QuantityRequest`."""

    variable: str
    shape: ShapeDraft
    shape_index: int | None
    words: tuple[str, ...]
    decision: BindResult | None
    part: int | None = None
    pool: Pool | None = None


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

    @property
    def reads_query_text(self) -> bool:
        """Whether the binder's model calls read the query's own text
        (`BindRequest.query_text`, `QuantityRequest.query_text`), so two queries with one
        extraction can bind differently."""
        ...

    async def read_quantity(
        self, request: QuantityRequest, *, llm: SupportsLLM
    ) -> tuple[QuantityOutput, dict | None]:
        """The form and the base among `request.denominators` the variable phrase states,
        and the call's token usage (`quantity_read.read_stated_quantity`)."""
        ...

    async def bind(
        self, request: BindRequest, *, llm: SupportsLLM
    ) -> tuple[BindResult, dict | None]:
        """The shape `request`'s menu elects with its slot bindings, and the call's token
        usage. Raises `LLMOutputRefused` for a reply the bind schema refuses."""
        ...

    async def rate_words_stated(
        self, request: StatedWordsRequest, *, llm: SupportsLLM
    ) -> tuple[Mapping[str, float], dict | None]:
        """For each word of `request.words` the binder rates, the probability that the
        elected shape's series states what the word names: near 1 when the series counts
        only what the word names, so the word is read, near 0 when it counts more, so the
        word is unread. A word left out is unrated, which the coverage rule counts as
        stated. Also the call's token usage. Raises `LLMInfraError` for a read that
        failed."""
        ...
