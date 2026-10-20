"""House policies: the conventions that choose among menu shapes a phrase leaves open.

A bare "GDP", "aid" or "ODA" measures several menu shapes that differ by a house convention
alone. The policies choose among the shapes a binder declares as measuring the phrase (its
`PolicyPool`), whichever binder elected, so the answer follows the convention.
Every fact the module reads comes from shape data, the frame and declared dataset
defaults.

The six policies run in order, each over the survivors of the one before:

1. `nominal_vs_va`: nominal output (EconomicActivity `amount`, Nominal where only some
   output shapes are) over value added, when a value-added shape is among the candidates.
2. `amount_vs_ge`: on FinancialFlow shapes, grant equivalent, or `amount` when the window
   starts before grant-equivalent reporting does (`GE_STARTS`) or names no start year, so
   the years before it stay in the answer.
3. `net_vs_gross`: a shape whose `nettingBasis` slot offers Gross.
4. `default_dataset`: a shape whose family the part's house default dataset
   (`Context.wanted_dataset`) publishes. An opt-in table is its own dataset, so the
   imputed multilateral sectors family fits no house default.
5. `commit_vs_disb`: a FinancialFlow shape whose `financialFlowType` slot offers
   Disbursement.
6. `data_at_places`: a shape with data at the named places.

Three policies declare the measure-axis values they choose between (`CHOICES`). A binder
that narrows its pool to a value the phrase names leaves the choice between two such values
to the policy unless the phrase names the value outright (`arbitrates`).

The policies choose only among the pool shapes that hold every constraint the binder
recognised on its pick (`apply_house_policies`'s `holds`), so a policy never trades a value
the phrase names for a convention. The pick holds its own.

A policy narrows only when some, not all, candidates fit it, and does nothing with one
candidate left.

Several shapes may survive. The tie-break picks one:

- with election probabilities (`PolicyPool.election`), the most probable survivor;
- else the binder's pick if it survives, else the highest-rated survivor.

Equal probabilities or ratings go to the lower menu index, and a survivor the binder gives no
probability or rating reads as 0.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, get_args

from code_index.query import CodeIndex
from code_index.types import HouseDefaultCase
from search.engine.bind import stated_reading
from search.engine.frame import Frame
from search.engine.qualifier_match import QualifierMatch
from search.engine.retrieval.bound_values import shape_part_datasets
from search.engine.shape import ShapeDraft

type PolicyName = Literal[
    "nominal_vs_va",
    "amount_vs_ge",
    "net_vs_gross",
    "default_dataset",
    "commit_vs_disb",
    "data_at_places",
]

POLICY_ORDER: tuple[PolicyName, ...] = (
    "nominal_vs_va",
    "amount_vs_ge",
    "net_vs_gross",
    "default_dataset",
    "commit_vs_disb",
    "data_at_places",
)

# The first year donors report official development assistance as grant equivalent.
GE_STARTS = 2018

_OUTPUT = "EconomicActivity"
_FLOW = "FinancialFlow"
_OBLIGATION = "FinancialObligation"
_AMOUNT = "amount"
_VALUE_ADDED = "valueAdded"
_GRANT_EQUIVALENT = "grantEquivalent"
_NOMINAL = "Nominal"
_REAL = "RealValue"

# The values each policy chooses between, by measure axis, in their canonical readings:
# nominal output over value added and over real output, amount or grant equivalent by the
# window, and a flow's disbursements over an obligation's commitments.
CHOICES: Mapping[PolicyName, Mapping[str, frozenset[str]]] = {
    "nominal_vs_va": {
        "measuredProperty": frozenset({_AMOUNT, _VALUE_ADDED}),
        "measurementQualifier": frozenset({_NOMINAL, _REAL}),
    },
    "amount_vs_ge": {"measuredProperty": frozenset({_AMOUNT, _GRANT_EQUIVALENT})},
    "commit_vs_disb": {"populationType": frozenset({_FLOW, _OBLIGATION})},
}


@dataclass(frozen=True, slots=True)
class ShapeFacts:
    """What the policies read of one menu shape.

    `netting` and `flow_types` are the shape's `nettingBasis` and `financialFlowType`
    taxonomies, `None` when it has no such slot. `datasets` are the provenances the part
    may answer the shape from (`bound_values.shape_part_datasets`), in serving order, and
    empty for a shape no declared dataset publishes. A shape fits a wanted dataset when the
    wanted provenance is among them. `available` says whether the shape
    has data at the named places, `None` when unknown or no place is named.
    """

    index: int
    population_type: str | None
    measured_property: str | None
    stat_type: str | None
    qualifier: str | None
    netting: tuple[str, ...] | None
    flow_types: tuple[str, ...] | None
    datasets: tuple[str, ...]
    available: bool | None


@dataclass(frozen=True, slots=True)
class Context:
    """What the part asks, as the policies read it.

    `pre_ge` says the window starts before `GE_STARTS` or names no start year. A window
    open at its start, and a part that names no years, ask for every year the data holds
    up to their end, those before `GE_STARTS` included, so both are pre-GE. A part that
    asks for the latest year is not.
    `dataset_case` is the kind of question the part is for the default-dataset policy, and
    `wanted_dataset` that kind's declared house default. Both are `None` when the part names
    no place and admits no code that declares a kind.
    """

    pre_ge: bool
    dataset_case: HouseDefaultCase | None
    wanted_dataset: str | None


@dataclass(frozen=True, slots=True)
class PolicyPool:
    """The shapes a binder declares as measuring the phrase.

    `measuring` holds their menu indices in the binder's order, its pick included.
    `election` is a per-index probability when the binder computes one.
    """

    measuring: tuple[int, ...]
    election: Mapping[int, float] | None = None


@dataclass(frozen=True, slots=True)
class PolicyChoice:
    """The policy step's outcome: the elected menu index, the policies that narrowed in the
    order they ran, the pool shapes they set aside, and the facts of every pool shape."""

    survivor: int
    fired: tuple[PolicyName, ...]
    dropped: frozenset[int]
    facts: tuple[ShapeFacts, ...]


def arbitrates(axis: str, value: str, others: Collection[str]) -> bool:
    """Whether a house policy chooses between `value` and one of `others` on measure axis
    `axis`, all in their canonical readings (`CHOICES`)."""
    return any(
        value in values and not values.isdisjoint(set(others) - {value})
        for choice in CHOICES.values()
        if (values := choice.get(axis))
    )


def shape_facts(
    shapes: Sequence[ShapeDraft],
    availability: Sequence[bool | None],
    indices: Sequence[int],
    *,
    code_index: CodeIndex,
) -> tuple[ShapeFacts, ...]:
    """The facts of the menu shapes at `indices`, in that order.

    `availability` is the menu's data-where-asked list (`ResolveState.menu_availability`),
    empty when the part names no place.
    """
    return tuple(
        _facts(shapes[i], i, availability[i] if availability else None, code_index) for i in indices
    )


def _facts(
    shape: ShapeDraft, index: int, available: bool | None, code_index: CodeIndex
) -> ShapeFacts:
    stated = stated_reading(shape)
    taxonomy = shape.slot_taxonomy or {}
    return ShapeFacts(
        index=index,
        population_type=stated.population_type,
        measured_property=stated.measured_property,
        stat_type=stated.stat_type,
        qualifier=stated.qualifier,
        netting=_slot_values(taxonomy, "nettingBasis"),
        flow_types=_slot_values(taxonomy, "financialFlowType"),
        datasets=shape_part_datasets(shape, code_index),
        available=available,
    )


def _slot_values(taxonomy: Mapping[str, list[str]], prop: str) -> tuple[str, ...] | None:
    return next((tuple(v) for k, v in taxonomy.items() if k.endswith(f":{prop}")), None)


def context_of(
    frame: Frame,
    admitted_qualifiers: Sequence[Sequence[QualifierMatch]],
    pool: Sequence[int],
    *,
    wanted_datasets: Mapping[HouseDefaultCase, str],
    case_systems: Mapping[str, HouseDefaultCase],
) -> Context:
    """The part's context over the pool before any policy narrows it.

    `admitted_qualifiers` holds each menu shape's admitted qualifiers, by menu index.
    `wanted_datasets` maps each kind of question to its house default dataset
    (`CodeIndex.house_defaults`), and `case_systems` each code system that declares a kind
    to that kind (`CodeIndex.house_case_systems`). A code of such a system admitted on a
    pool shape makes the part that kind of question (`sector` for a sector or purpose).
    Otherwise a place on the destination axis, on both axes or on none makes it a
    `recipient` question, and places that are all donors make it a `donor` question.
    """
    window = frame.window.value
    if window == "all":
        pre_ge = True
    elif window == "latest":
        pre_ge = False
    else:
        pre_ge = window.start_year is None or window.start_year < GE_STARTS
    directions = [place.direction for place in frame.places]
    admitted = {
        case_systems[match.code_system]
        for i in pool
        for match in (admitted_qualifiers[i] if admitted_qualifiers else ())
        if match.code_system in case_systems
    }
    case: HouseDefaultCase | None
    if admitted:
        case = next(c for c in get_args(HouseDefaultCase) if c in admitted)
    elif any(d in ("destination", "both", "undirected") for d in directions):
        case = "recipient"
    elif directions:
        case = "donor"
    else:
        case = None
    return Context(
        pre_ge=pre_ge,
        dataset_case=case,
        wanted_dataset=wanted_datasets.get(case) if case else None,
    )


def _nominal_output(shapes: Sequence[ShapeFacts], _ctx: Context) -> list[int]:
    if not any(s.measured_property == _VALUE_ADDED for s in shapes):
        return []
    output = [s for s in shapes if s.population_type == _OUTPUT and s.measured_property == _AMOUNT]
    nominal = [s for s in output if s.qualifier == _NOMINAL]
    return [s.index for s in (nominal if 0 < len(nominal) < len(output) else output)]


def _amount_or_ge(shapes: Sequence[ShapeFacts], ctx: Context) -> list[int]:
    measure = _AMOUNT if ctx.pre_ge else _GRANT_EQUIVALENT
    return [
        s.index for s in shapes if s.population_type == _FLOW and s.measured_property == measure
    ]


def _gross(shapes: Sequence[ShapeFacts], _ctx: Context) -> list[int]:
    return [s.index for s in shapes if s.netting is not None and "Gross" in s.netting]


def _default_dataset(shapes: Sequence[ShapeFacts], ctx: Context) -> list[int]:
    if ctx.wanted_dataset is None:
        return []
    return [s.index for s in shapes if ctx.wanted_dataset in s.datasets]


def _disbursed(shapes: Sequence[ShapeFacts], _ctx: Context) -> list[int]:
    return [
        s.index
        for s in shapes
        if s.population_type == _FLOW and "Disbursement" in (s.flow_types or ())
    ]


def _with_data(shapes: Sequence[ShapeFacts], _ctx: Context) -> list[int]:
    return [s.index for s in shapes if s.available is True]


_FITS: Mapping[PolicyName, Callable[[Sequence[ShapeFacts], Context], list[int]]] = {
    "nominal_vs_va": _nominal_output,
    "amount_vs_ge": _amount_or_ge,
    "net_vs_gross": _gross,
    "default_dataset": _default_dataset,
    "commit_vs_disb": _disbursed,
    "data_at_places": _with_data,
}


def apply_policy(
    policy: PolicyName, candidates: Sequence[ShapeFacts], ctx: Context
) -> list[ShapeFacts]:
    """The candidates `policy` keeps. These are the ones that fit when some, not all, fit, and
    all of them otherwise."""
    if len(candidates) < 2:
        return list(candidates)
    fitting = set(_FITS[policy](candidates, ctx))
    if 0 < len(fitting) < len(candidates):
        return [s for s in candidates if s.index in fitting]
    return list(candidates)


def apply_house_policies(
    facts: Sequence[ShapeFacts],
    pool: PolicyPool,
    context: Context,
    *,
    chosen: int,
    ratings: Mapping[int, float] | None = None,
    holds: Mapping[int, bool] | None = None,
) -> PolicyChoice:
    """The pool shape the house policies and the tie-break elect.

    `facts` covers every pool shape (`shape_facts`), `chosen` is the binder's pick, and
    `ratings` the binder's per-index rating of the pool when it gives no election
    probability. `holds` says, per pool shape, whether it holds every constraint the binder
    recognised on `chosen`. The policies run over `chosen` and the shapes that hold, and
    set the rest aside. A pool shape `holds` leaves out does not hold, and None holds them
    all. Raises `ValueError` when `chosen` or a pool shape's facts are missing.
    """
    if chosen not in pool.measuring:
        raise ValueError(f"chosen shape {chosen} is not in the pool {pool.measuring}")
    by_index = {f.index: f for f in facts}
    missing = [i for i in pool.measuring if i not in by_index]
    if missing:
        raise ValueError(f"no facts for pool shapes {missing}")
    pooled = tuple(by_index[i] for i in pool.measuring)

    candidates: Sequence[ShapeFacts] = [
        f for f in pooled if holds is None or f.index == chosen or holds.get(f.index, False)
    ]
    fired: list[PolicyName] = []
    for policy in POLICY_ORDER:
        kept = apply_policy(policy, candidates, context)
        if len(kept) < len(candidates):
            fired.append(policy)
        candidates = kept

    survivors = [s.index for s in candidates]
    return PolicyChoice(
        survivor=_tie_break(survivors, pool, chosen, ratings),
        fired=tuple(fired),
        dropped=frozenset(pool.measuring) - frozenset(survivors),
        facts=pooled,
    )


def _tie_break(
    survivors: Sequence[int],
    pool: PolicyPool,
    chosen: int,
    ratings: Mapping[int, float] | None,
) -> int:
    if pool.election is not None:
        election = pool.election
        return min(survivors, key=lambda i: (-election.get(i, 0.0), i))
    if chosen in survivors:
        return chosen
    rating = ratings or {}
    return min(survivors, key=lambda i: (-rating.get(i, 0.0), i))
