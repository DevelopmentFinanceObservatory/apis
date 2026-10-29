"""House policies: the conventions that choose among menu shapes a phrase leaves open.

A bare "GDP", "aid" or "ODA" measures several menu shapes that differ by a house convention
alone. The policies choose among the shapes a binder declares as measuring the phrase (its
`PolicyPool`), whichever binder elected, so the answer follows the convention.
Every fact the module reads comes from shape data, the frame, the part's measure-basis
entries and the declared conventions.

A part that names a place, a sector or a restriction is counted on the measure basis the
declared conventions give its shape and window (`scope.basis.elect_basis`), or the one an
explicit or carried entry asks for: `Context.basis`. That basis decides the three policies
it bears on, whatever words name the measure. A part the conventions do not read
falls back to the window and to gross.

The six policies run in order, each over the survivors of the one before:

1. `nominal_vs_va`: among output (EconomicActivity `amount`) and value-added shapes,
   nominal output (Nominal where only some output shapes are) over value added, when a
   value-added shape is among the candidates.
2. `amount_vs_ge`: among FinancialFlow shapes measuring `amount` or `grantEquivalent`, those
   measuring the basis's measured property. With no basis, grant equivalent, or `amount`
   when the window starts before grant-equivalent reporting does (`GE_STARTS`) or names no
   start year, so the years before it stay in the answer.
3. `net_vs_gross`: among shapes with a `nettingBasis` slot, one that offers the basis's
   netting basis, Gross for a basis that carries none and with no basis.
4. `commit_vs_disb`: among FinancialObligation shapes and FinancialFlow shapes with a
   `financialFlowType` slot, the obligations on a commitments basis, else a flow whose slot
   offers Disbursement.
5. `default_dataset`: a shape whose family the part's wanted dataset
   (`Context.wanted_dataset`) publishes: the dataset the basis convention reads, else the
   house default for the kind of question. An opt-in table is its own dataset, so the
   imputed multilateral sectors family fits no house default. A basis the wanted dataset
   does not publish leaves none of its shapes standing, so the policy keeps the shapes of
   the dataset that does. A recipient's grant equivalents are the CRS's.
6. `data_at_places`: a shape with data at the named places.

The four convention policies (`CONVENTION_POLICIES`) run first, so a dataset default
never sets aside a shape a convention would keep: a donor's loans go to the disbursements
of one dataset, not the commitments of the house default.

Where a policy run over a binder's pool alone would choose on a measure axis (`chooses_on`),
the binder narrows the pool on that axis only to a value the phrase names outright.

The policies choose only among the pool shapes that hold every constraint the binder
recognised on its pick (`apply_house_policies`'s `holds`), so a policy never trades a value
the phrase names for a convention. The pick holds its own.

A policy judges only the candidates in its domain, the shapes its convention compares, and
keeps every other one. It narrows only when some, not all, of its domain fit it, and does
nothing with one candidate left.

Several shapes may survive. The tie-break picks one:

- with election probabilities (`PolicyPool.election`), the most probable survivor;
- else the binder's pick if it survives, else the highest-rated survivor.

Equal probabilities or ratings go to the lower menu index, and a survivor the binder gives no
probability or rating reads as 0.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from operator import attrgetter
from typing import Literal, get_args

from code_index.query import CodeIndex
from code_index.types import HouseDefaultCase
from search.engine.bind import stated_reading
from search.engine.frame import Frame
from search.engine.qualifier_match import QualifierMatch
from search.engine.retrieval.bound_values import shape_part_datasets
from search.engine.shape import ShapeDraft
from search.models import Entry
from search.scope.basis import (
    GRANT_EQUIVALENT_HEADLINE_FROM,
    BasisElection,
    elect_part_basis,
)

type PolicyName = Literal[
    "nominal_vs_va",
    "amount_vs_ge",
    "net_vs_gross",
    "commit_vs_disb",
    "default_dataset",
    "data_at_places",
]

POLICY_ORDER: tuple[PolicyName, ...] = (
    "nominal_vs_va",
    "amount_vs_ge",
    "net_vs_gross",
    "commit_vs_disb",
    "default_dataset",
    "data_at_places",
)

# The policies that choose between measure values by a house convention. `default_dataset`
# and `data_at_places` choose by the part's situation, its house default and the data at
# its places, so they set no reading of the words.
CONVENTION_POLICIES: tuple[PolicyName, ...] = (
    "nominal_vs_va",
    "amount_vs_ge",
    "net_vs_gross",
    "commit_vs_disb",
)

# The first year donors report official development assistance as grant equivalent.
GE_STARTS = GRANT_EQUIVALENT_HEADLINE_FROM

# The `ShapeFacts` field each shape-kind measure axis reads.
_AXIS_FACTS: Mapping[str, str] = {
    "populationType": "population_type",
    "measuredProperty": "measured_property",
    "measurementQualifier": "qualifier",
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
    `dataset_case` is the kind of question the part is for the default-dataset policy.
    `basis` is the measure basis the part is counted on, None when the conventions read no
    shape of it (`scope.basis.part_shape`). `wanted_dataset` is the dataset the basis
    convention reads, else the kind's declared house default. `dataset_case` and
    `wanted_dataset` are `None` when the part names no place and admits no code that
    declares a kind.
    """

    pre_ge: bool
    dataset_case: HouseDefaultCase | None
    wanted_dataset: str | None
    basis: BasisElection | None = None


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


def chooses_on(
    facts: Sequence[ShapeFacts],
    context: Context,
    axis: str,
    *,
    policies: Sequence[PolicyName] = POLICY_ORDER,
) -> bool:
    """Whether one of `policies`, run over `facts` alone (`apply_policy`), keeps fewer of
    their values on the shape-kind measure axis `axis` than they hold, so the policy rather
    than the phrase decides between them there."""
    read = attrgetter(_AXIS_FACTS[axis])
    held = {read(f) for f in facts}
    return len(held) > 1 and any(
        {read(f) for f in apply_policy(policy, facts, context)} != held for policy in policies
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
        netting=slot_values(taxonomy, "nettingBasis"),
        flow_types=slot_values(taxonomy, "financialFlowType"),
        datasets=shape_part_datasets(shape, code_index),
        available=available,
    )


def slot_values(taxonomy: Mapping[str, list[str]], prop: str) -> tuple[str, ...] | None:
    """The values a shape's slot taxonomy offers on `prop`, None when it has no such slot."""
    return next((tuple(v) for k, v in taxonomy.items() if k.endswith(f":{prop}")), None)


def context_of(
    frame: Frame,
    admitted_qualifiers: Sequence[Sequence[QualifierMatch]],
    pool: Sequence[int],
    *,
    wanted_datasets: Mapping[HouseDefaultCase, str],
    case_systems: Mapping[str, HouseDefaultCase],
    pool_datasets: Sequence[str],
    basis_entries: Sequence[Entry] = (),
) -> Context:
    """The part's context over the pool before any policy narrows it.

    `admitted_qualifiers` holds each menu shape's admitted qualifiers, by menu index.
    `wanted_datasets` maps each kind of question to its house default dataset
    (`CodeIndex.house_defaults`), and `case_systems` each code system that declares a kind
    to that kind (`CodeIndex.house_case_systems`). A code of such a system admitted on a
    pool shape makes the part that kind of question (`sector` for a sector or purpose).
    Otherwise a place on the destination axis, on both axes or on none makes it a
    `recipient` question, and places that are all donors make it a `donor` question.
    The part's shape (read from the same places and admitted codes) and its window elect
    its basis (`scope.basis.elect_part_basis`), as the part's basis record reads it, from the
    datasets the pool's shapes are answered from (`pool_datasets`, in pool order): the
    part's own convention where they include its dataset, else the convention of the first
    of them a convention reads. `basis_entries`, the part's measure-basis entries with explicit first
    (`ScopeState.entries_on`), win over the convention.
    """
    window = frame.window.value
    if window == "all":
        pre_ge = True
    elif window == "latest":
        pre_ge = False
    else:
        pre_ge = window.start_year is None or window.start_year < GE_STARTS
    directions = [place.direction for place in frame.places]
    systems = {
        match.code_system
        for i in pool
        for match in (admitted_qualifiers[i] if admitted_qualifiers else ())
    }
    admitted = {case_systems[system] for system in systems if system in case_systems}
    case: HouseDefaultCase | None
    if admitted:
        case = next(c for c in get_args(HouseDefaultCase) if c in admitted)
    elif any(d in ("destination", "both", "undirected") for d in directions):
        case = "recipient"
    elif directions:
        case = "donor"
    else:
        case = None
    basis = elect_part_basis(
        datasets=pool_datasets,
        directions=directions,
        admitted_systems=systems,
        sector_systems=[system for system, kind in case_systems.items() if kind == "sector"],
        window=window,
        entries=basis_entries,
    )
    wanted = basis.convention.dataset if basis is not None else None
    return Context(
        pre_ge=pre_ge,
        dataset_case=case,
        wanted_dataset=wanted or (wanted_datasets.get(case) if case else None),
        basis=basis,
    )


def _nominal_output(shapes: Sequence[ShapeFacts], _ctx: Context) -> list[int]:
    if not any(s.measured_property == "valueAdded" for s in shapes):
        return []
    output = [
        s
        for s in shapes
        if s.population_type == "EconomicActivity" and s.measured_property == "amount"
    ]
    nominal = [s for s in output if s.qualifier == "Nominal"]
    return [s.index for s in (nominal if 0 < len(nominal) < len(output) else output)]


def _amount_or_ge(shapes: Sequence[ShapeFacts], ctx: Context) -> list[int]:
    if ctx.basis is not None:
        measure = ctx.basis.measure.measured_property
    else:
        measure = "amount" if ctx.pre_ge else "grantEquivalent"
    return [
        s.index
        for s in shapes
        if s.population_type == "FinancialFlow" and s.measured_property == measure
    ]


def _netting(shapes: Sequence[ShapeFacts], ctx: Context) -> list[int]:
    wanted = (ctx.basis.measure.netting_basis if ctx.basis is not None else None) or "Gross"
    return [s.index for s in shapes if s.netting is not None and wanted in s.netting]


def _default_dataset(shapes: Sequence[ShapeFacts], ctx: Context) -> list[int]:
    if ctx.wanted_dataset is None:
        return []
    return [s.index for s in shapes if ctx.wanted_dataset in s.datasets]


def _disbursed(shapes: Sequence[ShapeFacts], ctx: Context) -> list[int]:
    if ctx.basis is not None and ctx.basis.basis == "commitments":
        return [s.index for s in shapes if s.population_type == "FinancialObligation"]
    return [
        s.index
        for s in shapes
        if s.population_type == "FinancialFlow" and "Disbursement" in (s.flow_types or ())
    ]


def _with_data(shapes: Sequence[ShapeFacts], _ctx: Context) -> list[int]:
    return [s.index for s in shapes if s.available is True]


def _output_or_value_added(s: ShapeFacts) -> bool:
    return (s.population_type, s.measured_property) == ("EconomicActivity", "amount") or (
        s.measured_property == "valueAdded"
    )


def _flow_amount_or_ge(s: ShapeFacts) -> bool:
    return s.population_type == "FinancialFlow" and s.measured_property in (
        "amount",
        "grantEquivalent",
    )


def _netted(s: ShapeFacts) -> bool:
    return s.netting is not None


def _flow_or_obligation(s: ShapeFacts) -> bool:
    return (s.population_type == "FinancialFlow" and s.flow_types is not None) or (
        s.population_type == "FinancialObligation"
    )


def _any_shape(_s: ShapeFacts) -> bool:
    return True


# Each policy's domain, the shapes it compares, and the domain shapes it keeps.
_POLICIES: Mapping[
    PolicyName,
    tuple[Callable[[ShapeFacts], bool], Callable[[Sequence[ShapeFacts], Context], list[int]]],
] = {
    "nominal_vs_va": (_output_or_value_added, _nominal_output),
    "amount_vs_ge": (_flow_amount_or_ge, _amount_or_ge),
    "net_vs_gross": (_netted, _netting),
    "commit_vs_disb": (_flow_or_obligation, _disbursed),
    "default_dataset": (_any_shape, _default_dataset),
    "data_at_places": (_any_shape, _with_data),
}


def apply_policy(
    policy: PolicyName, candidates: Sequence[ShapeFacts], ctx: Context
) -> list[ShapeFacts]:
    """The candidates `policy` keeps: every candidate outside its domain, and of those in
    it the ones that fit when some, not all, fit, else all of them. A policy compares only
    the shapes its convention is about, so "imputed multilateral" outflows stay beside
    the amount and grant-equivalent shapes `amount_vs_ge` chooses between."""
    if len(candidates) < 2:
        return list(candidates)
    in_domain, fits = _POLICIES[policy]
    domain = [s for s in candidates if in_domain(s)]
    fitting = set(fits(domain, ctx))
    if 0 < len(fitting) < len(domain):
        return [s for s in candidates if not in_domain(s) or s.index in fitting]
    return list(candidates)


def apply_house_policies(
    facts: Sequence[ShapeFacts],
    pool: PolicyPool,
    context: Context,
    *,
    chosen: int,
    ratings: Mapping[int, float] | None = None,
    holds: Mapping[int, bool] | None = None,
    reported: Collection[int] | None = None,
) -> PolicyChoice:
    """The pool shape the house policies and the tie-break elect.

    `facts` covers every pool shape (`shape_facts`), `chosen` is the binder's pick, and
    `ratings` the binder's per-index rating of the pool when it gives no election
    probability. `holds` says, per pool shape, whether it holds every constraint the binder
    recognised on `chosen`. The policies run over `chosen` and the shapes that hold, and
    set the rest aside. A pool shape `holds` leaves out does not hold, and None holds them
    all. The choice states the facts of the pool shapes among `reported`, and as fired the
    policies that set one of them aside; every pool shape when `reported` is None. Raises
    `ValueError` when `chosen` or a pool shape's facts are missing.
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
    context = _servable(context, candidates)
    fired: list[PolicyName] = []
    for policy in POLICY_ORDER:
        kept = apply_policy(policy, candidates, context)
        set_aside = {s.index for s in candidates} - {s.index for s in kept}
        if set_aside if reported is None else not set_aside.isdisjoint(reported):
            fired.append(policy)
        candidates = kept

    survivors = [s.index for s in candidates]
    return PolicyChoice(
        survivor=_tie_break(survivors, pool, chosen, ratings),
        fired=tuple(fired),
        dropped=frozenset(pool.measuring) - frozenset(survivors),
        facts=tuple(f for f in pooled if reported is None or f.index in reported),
    )


def _servable(context: Context, candidates: Sequence[ShapeFacts]) -> Context:
    """`context` counting on the convention's basis where the basis an entry asks for is
    measured by none of `candidates`, so the policies fall back to the convention, not to
    the window and gross."""
    basis = context.basis
    if basis is None or basis.entry is None:
        return context
    if any(
        basis.measured_by(f.population_type, f.measured_property, f.netting) for f in candidates
    ):
        return context
    return replace(context, basis=basis.by_convention())


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
