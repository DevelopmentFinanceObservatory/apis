"""Jev's answers turned into the bind output and the quantity read.

Jev writes no text and no dcid: every decision here is a threshold (`Thresholds`) on a
probability Jev returned for a question of `jev_request`, or a comparison of the words its
span answers name, so the functions are pure and a threshold change replays from stored
answers. A noul Jev did not answer reads as a probability of 0.0, and a span question it
did not answer as no words.

`assemble` elects a shape, binds its listed slots and lists its alternates; `bind_result`
applies the rules the engine applies to every binder's drafts (`bind.ground_bind_output`).
`assemble_quantity` reads the stated form and base from the ``q|`` answers.

One stretch of the phrase pays for one slot value. A bind runs in two rounds: `span_round`
reads the first round's answers and returns the ``span|`` questions they call for, and
`assemble` reads the merged answers of both. Where two values of one shape, or a value and
the measure, name overlapping words, the shape keeps one of them (`_span_owned`), so
"public and publicly guaranteed" binds the guarantee status alone and no public-sector
debtor beside it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import get_args

from code_index.shared import get_index, get_measure_index
from search.engine.bind import (
    DraftBindingEvidence,
    DraftBindingKind,
    SlotBindingDraft,
    _UniversalBindOutput,
    ground_bind_output,
    stated_reading,
)
from search.engine.binders.base import BindResult
from search.engine.binders.gemini import (
    holds_drafted_values,
    holds_requested_slots,
    mentioned_drafts,
    nominated_near_best,
)
from search.engine.binders.house_policy import PolicyChoice, PolicyPool
from search.engine.binders.jev_client import JevQuestion
from search.engine.binders.jev_menu import (
    AXIS_FIELDS,
    JevMenu,
    Prop,
    asked_values,
    prop_id,
)
from search.engine.binders.jev_request import (
    NONE,
    NONE_OF_THESE,
    SPAN_MEASURE,
    span_id,
    span_positions,
    span_questions,
)
from search.engine.frame import Frame
from search.engine.quantity_read import QuantityOutput
from search.engine.retrieval.bound_values import bound_values_from_bindings
from search.engine.shape import ShapeDraft
from search.models import GraphRef, QuantityForm

ASK_TEXT = "No candidate shape measures what the variable phrase names (Jev V2)."

_FORM_OF: dict[str, QuantityForm] = {form: form for form in get_args(QuantityForm)}

# The ``q|gate`` probability at which the quantity read states anything.
QUANTITY_GATE = 0.5

# A probability of 0 or 1 has no finite log-odds.
_LOG_ODDS_CLAMP = 1e-6

type Population = tuple[Mapping[str, str], ...]
"""The constraints each member of a shape's complete member population asserts, by
property (`jev.member_population`)."""


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Where each probability turns into a decision.

    `w_m`, `w_mq`, `w_elect` and `w_electq` weigh the two means. A shape's measure score
    above `ask` keeps it electable, and the part asks when none is, as a Gemini reply
    rating its pick under `gemini.DECLINE_FLOOR` declines; above `measures` it
    counts as measuring the phrase for alternates, nominal output and the nearest refused
    reading. Without an election answer the pool keeps shapes within `near` of the best
    measure score; with one, the products of election and measure score at least
    ``1 - near_elect`` of the best. An alternate is "elected alike" when its election is at
    least ``1 - near_alt`` of the top and its measure score above `alt_floor`. A nominal
    output shape joins the pool within `widen` of the best measure score. ``mv`` above
    `measure_value` names a measure-axis value, and above `policy_value` one of two a house
    policy chooses between (`_names_value`); ``named`` above `named` names the dataset, ``n``
    above `value` a slot value. ``part`` at or under `part` asks for no particular value,
    and a value then needs ``n`` above `part_n`. ``several`` above `several` binds a set,
    and ``all`` above `requested_all` an explicit "every value".
    """

    w_m: float = 1.0
    w_mq: float = 1.0
    w_elect: float = 1.0
    w_electq: float = 1.0
    measures: float = 0.5
    # On recorded answers, five wrong picks of the eval corpus score 0.135 or less, and
    # every pick that answers it correctly 0.305 or more.
    ask: float = 0.15
    near: float = 0.15
    near_elect: float = 0.8
    near_alt: float = 0.95
    alt_floor: float = 0.3
    widen: float = 0.2
    part: float = 0.15
    part_n: float = 0.8
    named: float = 0.5
    measure_value: float = 0.4
    # On the recorded answers of the eval corpus for amount against grant equivalent,
    # nominal against real and flow against obligation, Jev scores 61 of 64 at 0.76 or
    # less, grant equivalent for "education aid" and an obligation for "ODA loans" among
    # them, and the three a phrase names outright (debt is an obligation) at 0.94 or more.
    policy_value: float = 0.8
    value: float = 0.6
    several: float = 0.5
    requested_all: float = 0.5


THRESHOLDS = Thresholds()


@dataclass(frozen=True, slots=True)
class JevAnswers:
    """Jev's answers by question id: each noul's probability of yes, and each choice's
    probability per option (`jev_client.JevResponse`)."""

    nouls: Mapping[str, float]
    choices: Mapping[str, Mapping[str, float]]

    def noul(self, qid: str) -> float:
        """The noul's probability, 0.0 when Jev did not answer it."""
        return float(self.nouls.get(qid, 0.0))

    def choice(self, qid: str) -> str | None:
        """The most probable option of a choice, the first listed on a tie, or None when
        Jev did not answer it."""
        probabilities = self.choices.get(qid)
        if not probabilities:
            return None
        return max(probabilities, key=probabilities.__getitem__)

    def merged(self, other: JevAnswers) -> JevAnswers:
        """These answers with `other`'s beside them, `other`'s winning on a shared id."""
        return JevAnswers(
            nouls={**self.nouls, **other.nouls}, choices={**self.choices, **other.choices}
        )


@dataclass(slots=True)
class Confidence:
    """What each decision read: `measures`, `election` and `named` per shape, `refused` per
    reading a pinned part refused, `alternates` the rule that listed each alternate,
    `slots` the probability behind each bound slot, `inferred` the values each slot drafted
    that the span rule left unbound (`_span_owned`), `rules` the election rules that
    applied in order, and the house policies' `pool` and `choice`, None when the part
    asks."""

    measures: dict[int, float] = field(default_factory=dict)
    election: dict[int, float] = field(default_factory=dict)
    named: dict[int, float] = field(default_factory=dict)
    refused: dict[int, float] = field(default_factory=dict)
    alternates: dict[int, str] = field(default_factory=dict)
    slots: dict[str, float] = field(default_factory=dict)
    inferred: dict[str, tuple[str, ...]] = field(default_factory=dict)
    rules: list[str] = field(default_factory=list)
    pool: PolicyPool | None = None
    choice: PolicyChoice | None = None

    @property
    def policy_dropped(self) -> frozenset[int]:
        """The pool shapes the house policies set aside."""
        return self.choice.dropped if self.choice is not None else frozenset()


def frame_stated_form(frame: Frame | None) -> str | None:
    """The form the frame states the phrase asks for, None when it states only the
    default."""
    norm = frame.normalisation if frame is not None else None
    if norm is None or norm.form.stated_by == "default" or not norm.form.value:
        return None
    return norm.form.value[0]


def assemble_quantity(
    answers: JevAnswers, denominators: Sequence[GraphRef], *, gate: float = QUANTITY_GATE
) -> QuantityOutput:
    """The quantity read the ``q|`` answers give, before the engine grounds its quote.

    Under `gate` the phrase states nothing. Else the form is the ``q|form`` choice, and
    the base the ``q|base`` span with the ``q|denominator`` entry, or the index past the
    last entry for "none of these" or when no entry was asked.

    Raises:
        ValueError: Jev did not answer ``q|gate``.
    """
    if "q|gate" not in answers.nouls:
        raise ValueError("the quantity read has no q|gate answer")
    if answers.noul("q|gate") < gate:
        return QuantityOutput()
    form = answers.choice("q|form")
    stated_form = _FORM_OF.get(form) if form is not None else None
    base = answers.choice("q|base")
    if base in (None, NONE):
        return QuantityOutput(stated_form=stated_form)
    denominator = answers.choice("q|denominator")
    if denominator is None or denominator == NONE_OF_THESE:
        index = len(denominators)
    else:
        index = int(denominator.split(":", 1)[0])
    return QuantityOutput(base_words=base, denominator_index=index, stated_form=stated_form)


@dataclass(frozen=True, slots=True)
class SpanRound:
    """The span questions a bind's first answers call for (`span_round`), and the menu
    shapes whose drafts their answers settle, whose member populations `assemble` reads.
    Both are empty when no pool shape drafts named values on two slots."""

    questions: dict[str, JevQuestion]
    shapes: tuple[int, ...]


def span_round(
    menu: JevMenu,
    answers: JevAnswers,
    *,
    stated_form: str | None,
    thresholds: Thresholds = THRESHOLDS,
) -> SpanRound:
    """The second round of a bind over `menu`, from the first round's `answers`.

    The pool is the one the election leaves before the house policies run (`_pool`). Each
    pool shape whose slots draft values the phrase names (`gemini.mentioned_drafts`) on two
    or more properties is contended, and the round asks which words name each such value,
    keyed by slot and value so the ids do not depend on the shape (`jev_request.span_id`),
    and which name the measure of the pool's pick. A bind with no contended shape, and one
    that asks, needs no second round. Runs no house policy.
    """
    t = thresholds
    conf = _scores(menu, answers, t)
    pool = _pool(menu, answers, conf, t, stated_form) if menu.request.shapes else None
    if pool is None:
        return SpanRound({}, ())
    contended: list[int] = []
    values: dict[tuple[Prop, str], None] = {}
    for s in pool:
        named = mentioned_drafts([slot.draft for slot in _slot_drafts(menu, answers, s, conf, t)])
        if len({d.property_dcid for d in named}) < 2:
            continue
        contended.append(s)
        values.update(
            dict.fromkeys(
                ((d.axis, d.property_dcid), v)
                for d in named
                if d.property_dcid is not None
                for v in d.value_dcids
            )
        )
    if not contended:
        return SpanRound({}, ())
    questions = span_questions(menu, list(values), measure=_pick(pool, conf))
    return SpanRound(questions, tuple(contended))


def assemble(
    menu: JevMenu,
    answers: JevAnswers,
    *,
    stated_form: str | None,
    thresholds: Thresholds = THRESHOLDS,
    populations: Mapping[int, Population] | None = None,
) -> tuple[_UniversalBindOutput, Confidence]:
    """The bind output Jev's `answers` give over `menu` (`jev_menu.jev_menu` of the bind
    request, the menu the questions were asked of), before the engine's rules apply to its
    drafts, and what each decision read.

    `answers` holds both rounds' (`span_round`). `populations` holds the complete member
    population of each menu shape that has one (`jev.member_population`); the span rule
    reads it (`_span_owned`). `stated_form` is the form the phrase states (the grounded
    quantity read), None for a plain amount. The request's house-policy step
    (`BindRequest.apply_policies`) runs once when a shape is elected, over the pool the
    election leaves, and never on an ask.
    """
    t = thresholds
    shapes = menu.request.shapes
    populations = populations or {}
    conf = _scores(menu, answers, t)

    e = _elect(menu, answers, conf, t, stated_form, populations) if shapes else None
    if e is None:
        measuring = {r: p for r, p in conf.refused.items() if p > t.measures}
        nearest = max(measuring, key=measuring.__getitem__) if measuring else None
        output = _UniversalBindOutput(
            chosen_shape_index=0, ask=ASK_TEXT, nearest_refused_index=nearest
        )
        return output, conf
    bindings = _shape_drafts(menu, answers, e, conf, t, populations.get(e))
    also = _alternates(menu, e, conf, t, answers, stated_form)
    output = _UniversalBindOutput(
        chosen_shape_index=e, bindings=bindings, also_plausible=also or None
    )
    # An alternate stands only near the best measure score, the bar a Gemini nomination
    # clears on its rating.
    return nominated_near_best(output, conf.measures), conf


def bind_result(
    menu: JevMenu,
    answers: JevAnswers,
    *,
    stated_form: str | None,
    thresholds: Thresholds = THRESHOLDS,
    populations: Mapping[int, Population] | None = None,
) -> tuple[BindResult, Confidence]:
    """`assemble` over `menu`, with the engine's rules applied to the drafts
    (`bind.ground_bind_output`)."""
    output, conf = assemble(
        menu, answers, stated_form=stated_form, thresholds=thresholds, populations=populations
    )
    result = BindResult(
        output=ground_bind_output(output, menu.request.shapes),
        binder="jev",
        policy_pool=conf.pool,
        policy_choice=conf.choice,
    )
    return result, conf


def _scores(menu: JevMenu, answers: JevAnswers, t: Thresholds) -> Confidence:
    """A new record of the per-shape scores the election reads."""
    n = len(menu.request.shapes)
    conf = Confidence()
    conf.measures = _measure_scores(answers, n, t)
    conf.election = _election(answers, t)
    conf.named = {i: answers.noul(f"named|{i}") for i in range(n)}
    conf.refused = {r: answers.noul(f"mr|{r}") for r in menu.refused}
    return conf


def _election(answers: JevAnswers, t: Thresholds) -> dict[int, float]:
    """Each shape's election probability: the weighted mean of its ``elect`` and
    ``electq`` probabilities, over those answered. The ``none`` option counts for no
    shape."""
    parts = [
        (w, answers.choices[q])
        for w, q in ((t.w_elect, "elect"), (t.w_electq, "electq"))
        if q in answers.choices
    ]
    total = sum(w for w, _p in parts)
    out: dict[int, float] = {}
    for w, probabilities in parts:
        for key, p in probabilities.items():
            if key.startswith("Shape "):
                i = int(key.removeprefix("Shape "))
                out[i] = out.get(i, 0.0) + w * float(p) / total
    return out


def _measure_scores(answers: JevAnswers, n: int, t: Thresholds) -> dict[int, float]:
    """Each shape's measure score: the weighted mean of its ``m|i`` and, when answered,
    ``mq|i``."""
    out = {}
    for i in range(n):
        parts = [(t.w_m, answers.noul(f"m|{i}"))]
        if f"mq|{i}" in answers.nouls:
            parts.append((t.w_mq, answers.noul(f"mq|{i}")))
        out[i] = sum(w * p for w, p in parts) / sum(w for w, _p in parts)
    return out


def _with_nominal_output(
    shapes: Sequence[ShapeDraft], pool: list[int], conf: Confidence, t: Thresholds
) -> list[int]:
    """`pool` plus each nominal output shape (EconomicActivity amount, Nominal) stating the
    denominator of a value-added shape in it (EconomicProduction valueAdded) and measuring
    the phrase within `widen` of the best. The election favours the series labelled
    plainly "Gross domestic product", the value-added one, so the house policy preferring
    nominal output needs the output series among its candidates."""
    best = max(conf.measures.values())
    added = {
        stated_reading(shapes[i]).denominator
        for i in pool
        if shapes[i].pop_type_dcid == "EconomicProduction"
        and shapes[i].meas_prop_dcid == "valueAdded"
    }
    return [
        i
        for i, s in enumerate(shapes)
        if i in pool
        or (
            s.pop_type_dcid == "EconomicActivity"
            and s.meas_prop_dcid == "amount"
            and s.meas_qual_dcid == "Nominal"
            and stated_reading(s).denominator in added
            and conf.measures[i] > t.measures
            and conf.measures[i] >= best - t.widen
        )
    ]


def _elect(
    menu: JevMenu,
    answers: JevAnswers,
    conf: Confidence,
    t: Thresholds,
    form: str | None,
    populations: Mapping[int, Population],
) -> int | None:
    """The elected menu index, None to ask: the house policies' survivor of the pool
    (`_pool`)."""
    pool = _pool(menu, answers, conf, t, form)
    if pool is None:
        return None
    # The house policies decide what the phrase leaves open, among the shapes that hold
    # what Jev recognised on its pick; the election among what they keep, and menu rank
    # between equal elections.
    pick = _pick(pool, conf)
    conf.pool = PolicyPool(
        measuring=(pick, *(i for i in pool if i != pick)),
        election=dict(conf.election),
    )
    holds = _holds(menu, answers, pick, conf.pool.measuring, t, populations)
    choice = menu.request.apply_policies(conf.pool, holds=holds)
    conf.choice = choice
    conf.rules += [f"house: {p}" for p in choice.fired]
    return choice.survivor


def _pick(pool: Sequence[int], conf: Confidence) -> int:
    """The pool shape the election rates highest, the first in the pool on a tie."""
    return max(pool, key=lambda i: conf.election.get(i, 0.0))


def _pool(
    menu: JevMenu,
    answers: JevAnswers,
    conf: Confidence,
    t: Thresholds,
    form: str | None,
) -> list[int] | None:
    """The shapes the election leaves for the house policies, None to ask. Each step
    narrows the pool and records its rule in `conf.rules` when it does."""
    shapes = menu.request.shapes
    best = max(conf.measures.values(), default=0.0)
    if best <= t.ask:
        conf.rules.append("ask: no shape measures the phrase")
        return None
    # Only a shape measuring the phrase above `ask` is electable, on every path.
    electable = [i for i in range(len(shapes)) if conf.measures[i] > t.ask]
    if conf.election:
        score = {i: conf.election.get(i, 0.0) * conf.measures[i] for i in electable}
        top = max(score.values())
        pool = [i for i in electable if score[i] >= (1 - t.near_elect) * top]
        pool = _with_nominal_output(shapes, pool, conf, t)
    else:
        pool = [i for i in electable if conf.measures[i] >= best - t.near]
    # A phrase that states no form asks for the plain measure, not a share of a base.
    if form is None:
        plain = [i for i in pool if stated_reading(shapes[i]).denominator is None]
        if plain and len(plain) < len(pool):
            conf.rules.append("form: phrase states none, shapes with a denominator set aside")
            pool = plain
    # A measure value or a dataset the phrase names keeps the shapes that hold it.
    for axis, field_name in AXIS_FIELDS.items():
        owned = _policy_chooses(menu, pool, axis)
        holding = [
            i for i in pool if _names_value(answers, t, getattr(shapes[i], field_name), owned=owned)
        ]
        if holding and len(holding) < len(pool):
            conf.rules.append(f"named: {axis}")
            pool = holding
    named = [i for i in pool if conf.named[i] > t.named]
    if named and len(named) < len(pool):
        conf.rules.append("named: dataset")
        pool = named
    return pool


def _policy_chooses(menu: JevMenu, indices: Sequence[int], axis: str) -> bool:
    """Whether a house policy would choose on measure axis `axis` among the menu shapes at
    `indices` (`BindRequest.policy_chooses_on`), False when no policy reads the menu."""
    chooses = menu.request.policy_chooses_on
    return chooses is not None and chooses(indices, axis)


def _names_value(answers: JevAnswers, t: Thresholds, value: str | None, *, owned: bool) -> bool:
    """Whether Jev's ``mv`` answer says the phrase names `value`, in any spelling of its
    reading: above `measure_value`, or above `policy_value` on an axis a house policy would
    choose on (`owned`), so a convention is traded only for a value the phrase names
    outright."""
    if not value:
        return False
    reading = get_measure_index().canonical_reading(value)
    bar = t.policy_value if owned else t.measure_value
    return answers.noul(f"mv|{reading}") > bar


def _holds(
    menu: JevMenu,
    answers: JevAnswers,
    pick: int,
    pool: Sequence[int],
    t: Thresholds,
    populations: Mapping[int, Population],
) -> dict[int, bool]:
    """Whether each `pool` shape holds the constraints Jev recognised on `pick`, as
    `pick`'s slots bind them at the thresholds once the span rule has run
    (`_shape_drafts`), the drafts `assemble` binds: the value and set drafts the phrase
    names (`gemini.mentioned_drafts`), and the slots it asks for in full.

    A shape holds a named value when it holds it on its own slot as `pick` does
    (`gemini.holds_drafted_values`), or in exchange for a value Jev recognises on a slot of
    that shape that `pick` lacks: one dataset's taxonomy then states the constraint
    another's states on `pick`, as DAC2a's type of finance states the CRS's loans. Each
    such value stands for one constraint of `pick`, so a shape holds the named values only
    when it states at least as many of them in its own taxonomy as it lacks on `pick`'s
    slots, and "health loans" keeps the CRS sector DAC2a has no slot for. A slot asked for
    in full it holds as the Gemini binder reads one (`gemini.holds_requested_slots`), so
    "across all sectors" keeps every sector."""
    shapes = menu.request.shapes

    def drafts(s: int) -> list[SlotBindingDraft]:
        return _shape_drafts(menu, answers, s, Confidence(), t, populations.get(s))

    on_pick = drafts(pick)
    named = mentioned_drafts(on_pick)
    pick_slots = {slot.property_dcid for slot in shapes[pick].slot_keys}

    def holds_named(k: int) -> bool:
        lacked = sum(not holds_drafted_values(shapes[pick], shapes[k], [d]) for d in named)
        stated = sum(d.property_dcid not in pick_slots for d in mentioned_drafts(drafts(k)))
        return lacked <= stated

    return {
        k: holds_named(k) and holds_requested_slots(shapes[pick], shapes[k], on_pick) for k in pool
    }


def _alternates(
    menu: JevMenu,
    e: int,
    conf: Confidence,
    t: Thresholds,
    answers: JevAnswers,
    form: str | None,
) -> list[int]:
    """The shapes measuring the phrase's quantity whose five-tuple states a genuine
    alternate reading of `e`, by measure score, highest first."""
    shapes = menu.request.shapes
    index = get_measure_index()
    family = next((a.prefer for a in index.axes if a.axis == "populationType"), ())

    def named(k: int, axis: str, a: str | None, b: str | None) -> bool:
        """Whether the phrase names either of the values `e` and shape `k` hold on `axis`."""
        owned = _policy_chooses(menu, (e, k), axis)
        return any(_names_value(answers, t, v, owned=owned) for v in (a, b))

    elected = stated_reading(shapes[e])
    out: list[tuple[float, int]] = []
    top = max(conf.election.values(), default=0.0)
    for k in range(len(shapes)):
        # A house policy settled the reading a set-aside shape offers.
        if k in conf.policy_dropped:
            continue
        # A shape the election rates nearly as high as the elected one, and that measures
        # the phrase, is a reading the words do not choose between.
        if (
            k != e
            and top
            and conf.measures[k] > t.alt_floor
            and conf.election.get(k, 0.0) >= (1 - t.near_alt) * top
        ):
            conf.alternates[k] = "elected alike"
            out.append((conf.measures[k], k))
            continue
        if k == e or conf.measures[k] <= t.measures:
            continue
        other = stated_reading(shapes[k])
        if other.denominator != elected.denominator:
            # A denominator variant of the identical measure, when the phrase states no
            # form, and never a per-area one.
            if (
                form is not None
                or "per area" in (other.denominator, elected.denominator)
                or (other.population_type, other.measured_property, other.qualifier)
                != (elected.population_type, elected.measured_property, elected.qualifier)
            ):
                continue
            conf.alternates[k] = "denominator variant"
        elif (
            elected.population_type in family
            and other.population_type in family
            and (other.population_type, other.measured_property, other.qualifier)
            != (elected.population_type, elected.measured_property, elected.qualifier)
        ):
            # The measure family's declared axes are its interpretations, unless the phrase
            # names a value on an axis where the two differ.
            differing = [
                (axis, a, b)
                for axis, a, b in (
                    ("populationType", elected.population_type, other.population_type),
                    ("measuredProperty", elected.measured_property, other.measured_property),
                    ("measurementQualifier", elected.qualifier, other.qualifier),
                )
                if a != b
            ]
            if any(named(k, axis, a, b) for axis, a, b in differing):
                continue
            conf.alternates[k] = "measure family variant"
        elif (other.population_type, other.measured_property) != (
            elected.population_type,
            elected.measured_property,
        ):
            continue
        elif other.qualifier != elected.qualifier:
            if named(k, "measurementQualifier", elected.qualifier, other.qualifier):
                continue
            conf.alternates[k] = "qualifier variant"
        else:
            # Another dataset's copy of the elected reading is settled by data and rank.
            continue
        out.append((conf.measures[k], k))
    return [k for _p, k in sorted(out, key=lambda pk: -pk[0])]


def _draft(
    prop: Prop,
    kind: DraftBindingKind,
    values: Sequence[str] = (),
    *,
    evidence: DraftBindingEvidence = "mentioned",
    requested_all_phrase: str | None = None,
) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis=prop[0],
        property_dcid=prop[1],
        kind=kind,
        value_dcids=list(values),
        evidence=evidence,
        requested_all_phrase=requested_all_phrase,
    )


@dataclass(frozen=True, slots=True)
class _Slot:
    """One slot's draft (`_bind_slot`). A `seed` is a value the menu set: a fixed value,
    a tie's chosen row or the member of a one-member taxonomy. No other value's words
    unseat it, and every reading the span rule weighs holds it when the phrase names it."""

    draft: SlotBindingDraft
    seed: bool = False


def _shape_drafts(
    menu: JevMenu,
    answers: JevAnswers,
    s: int,
    conf: Confidence,
    t: Thresholds,
    population: Population | None,
) -> list[SlotBindingDraft]:
    """The drafts of shape `s`'s listed slots: each slot's own (`_bind_slot`), then the
    span rule over them (`_span_owned`)."""
    return _span_owned(menu, answers, _slot_drafts(menu, answers, s, conf, t), conf, population)


def _slot_drafts(
    menu: JevMenu, answers: JevAnswers, s: int, conf: Confidence, t: Thresholds
) -> list[_Slot]:
    return [_bind_slot(menu, answers, s, prop, conf, t) for prop in menu.listed[s]]


def _span_owned(
    menu: JevMenu,
    answers: JevAnswers,
    slots: Sequence[_Slot],
    conf: Confidence,
    population: Population | None,
) -> list[SlotBindingDraft]:
    """The drafts of `slots` with each stretch of the phrase paying for one value.

    The values the phrase names (`gemini.mentioned_drafts`) on two or more properties
    contend; with fewer the drafts stand as drafted. Seeds (`_Slot.seed`) contend in no
    span contest and take part in every realisability check. ``sole_option`` drafts take
    part in neither: the phrase names none of their values, and the engine unbinds one the
    members do not all share before any requirement is judged.

    Every other named draft is read through the argmax of its ``span|`` answers
    (`_read_span`), against the argmax of ``span|measure``, and is one of:

    - worded: Jev gave it words that do not overlap the measure's;
    - measure-worded: its words overlap the measure's, as "debt stocks" implies an amount
      outstanding and "loans" a loan;
    - wordless: Jev answered "none".

    Each weighs w = logit(p_n x p_span), the log-odds that Jev names the value with the
    words it gave. The rule, in order:

    1. Two worded drafts conflict when their words overlap, pair by pair. A worded draft in
       no conflict stays. Of the conflicting ones, the maximal subset with no conflicting
       pair and the largest sum of w stays. A draft is dropped only because a kept one
       claims its words, so a conflict keeps one value however weak both claims are. A
       losing draft stays when it narrows nothing beyond the kept worded drafts and the
       seeds (`narrows_nothing`), as "net flows" names one concept on two properties.
       Realisability is not weighed here, so a typed value stays against a weaker one that
       fits the data.
    2. A measure-worded draft whose words overlap a kept worded draft is dropped.
    3. Of the rest, the subset with no overlapping pair stays that a member holds together
       with the kept worded drafts and the seeds, then the larger subset, then the larger
       sum of w. When no member holds the kept worded drafts and the seeds, none stays, so
       the engine refuses on the typed values and names no value the user never wrote.
       With no complete population every one stays, which gives the engine's loud refusal
       over a silent broadening.
    4. A wordless draft stays only when it narrows nothing beyond everything kept.

    Ties fall to the subset whose sorted (property, values) list sorts first, and subsets
    are enumerated exactly. A dropped draft is left unbound and listed in `conf.inferred`.

    A member holds a draft when it asserts one of the draft's values, widened as the
    engine widens a bound value (`bound_values.bound_values_from_bindings`), and a member
    silent on the property holds none. `population` is None when the shape has no complete
    member population (`jev.member_population`). No draft then narrows nothing, since one
    dropped over a sample could be held by a member the sample lacks.
    """
    drafts = [slot.draft for slot in slots]
    named = [i for i, slot in enumerate(slots) if mentioned_drafts([slot.draft])]
    if len({drafts[i].property_dcid for i in named}) < 2:
        return drafts
    variable = menu.request.variable
    seeds = [i for i in named if slots[i].seed]
    # A measure Jev finds no words for marks no draft measure-worded.
    measure = _read_span(answers, [SPAN_MEASURE], variable)[0] or frozenset()

    words: dict[int, frozenset[int]] = {}
    weight: dict[int, float] = {}
    worded: list[int] = []
    measured: list[int] = []
    wordless: list[int] = []
    for i in named:
        if slots[i].seed:
            continue
        d = drafts[i]
        prop = (d.axis, d.property_dcid or "")
        read, p_span = _read_span(answers, [span_id(prop, v) for v in d.value_dcids], variable)
        weight[i] = _logit(conf.slots.get(prop_id(prop), 0.5) * p_span)
        if read is None:
            wordless.append(i)
            continue
        words[i] = read
        (measured if read & measure else worded).append(i)

    widened = {
        i: frozenset(bound_values_from_bindings([drafts[i]]).get(prop, drafts[i].value_dcids))
        for i in named
        if (prop := drafts[i].property_dcid) is not None
    }

    def holds(member: Mapping[str, str], i: int) -> bool:
        return member.get(drafts[i].property_dcid or "") in widened[i]

    def realisable(kept: Collection[int]) -> bool:
        return population is not None and any(all(holds(m, i) for i in kept) for m in population)

    def narrows_nothing(i: int, kept: Collection[int]) -> bool:
        if population is None:
            return False
        holding = [m for m in population if all(holds(m, k) for k in kept)]
        return bool(holding) and all(holds(m, i) for m in holding)

    def overlap(a: int, b: int) -> bool:
        return not words[a].isdisjoint(words[b])

    def apart(subset: Sequence[int]) -> bool:
        return not any(overlap(a, b) for a, b in combinations(subset, 2))

    def key(subset: Sequence[int]) -> list[tuple[str, ...]]:
        return sorted((drafts[i].property_dcid or "", *drafts[i].value_dcids) for i in subset)

    # 1. Worded drafts.
    contested = [i for i in worded if any(overlap(i, j) for j in worded if j != i)]

    def maximal(subset: Sequence[int]) -> bool:
        return apart(subset) and all(
            any(overlap(i, k) for k in subset) for i in contested if i not in subset
        )

    won = _best_subset(contested, maximal, lambda c: (sum(weight[i] for i in c),), key)
    kept = [i for i in worded if i not in contested or i in won]
    base = [*kept, *seeds]
    kept += [i for i in contested if i not in won and narrows_nothing(i, base)]
    anchor = [*kept, *seeds]

    # 2 and 3. Measure-worded drafts.
    open_ = [i for i in measured if not any(overlap(i, k) for k in kept)]
    if population is None:
        kept += open_
    elif realisable(anchor):
        kept += _best_subset(
            open_,
            apart,
            lambda c: (realisable([*anchor, *c]), len(c), sum(weight[i] for i in c)),
            key,
        )

    # 4. Wordless drafts.
    kept += [i for i in wordless if narrows_nothing(i, [*kept, *seeds])]

    for i in named:
        if slots[i].seed or i in kept:
            continue
        d = drafts[i]
        prop = (d.axis, d.property_dcid or "")
        conf.inferred[prop_id(prop)] = tuple(d.value_dcids)
        conf.slots.pop(prop_id(prop), None)
        drafts[i] = _draft(prop, "unbound")
    return drafts


def _read_span(
    answers: JevAnswers, qids: Sequence[str], variable: str
) -> tuple[frozenset[int] | None, float]:
    """The words of `variable` the argmax answers to `qids` name and the least probability
    among those argmaxes. A question Jev did not answer names no words at probability 1.0,
    so its draft conflicts with nothing. The words are None when every answered argmax is
    `NONE`."""
    named: list[frozenset[int]] = []
    p = 1.0
    wordless = False
    for qid in qids:
        span = answers.choice(qid)
        if span is None:
            continue
        p = min(p, float(answers.choices[qid][span]))
        if span == NONE:
            wordless = True
        else:
            named.append(span_positions(variable, span))
    if wordless and not named:
        return None, p
    return frozenset[int]().union(*named), p


def _logit(p: float) -> float:
    p = min(max(p, _LOG_ODDS_CLAMP), 1 - _LOG_ODDS_CLAMP)
    return math.log(p / (1 - p))


def _best_subset(
    items: Sequence[int],
    allowed: Callable[[Sequence[int]], bool],
    score: Callable[[Sequence[int]], tuple[float, ...]],
    key: Callable[[Sequence[int]], list[tuple[str, ...]]],
) -> list[int]:
    """The subset of `items` that `allowed` admits with the highest `score`, the one whose
    `key` sorts first on a tie, found by enumerating every subset."""
    subsets = [list(c) for r in range(len(items) + 1) for c in combinations(items, r) if allowed(c)]
    return min(subsets, key=lambda c: (tuple(-x for x in score(c)), key(c)))


def _bind_slot(
    menu: JevMenu,
    answers: JevAnswers,
    s: int,
    prop: Prop,
    conf: Confidence,
    t: Thresholds,
) -> _Slot:
    """The draft of slot `prop` on menu shape `s`."""
    shape, qmenu = menu.request.shapes[s], menu.menus[s]
    axis, property_dcid = prop
    pid = prop_id(prop)
    # The engine alone sets a source.
    if axis == "source":
        return _Slot(_draft(prop, "unbound"))
    named = {
        v: p
        for j, v in enumerate(asked_values(menu, prop))
        if (p := answers.noul(f"n|{pid}|{j}")) > t.value
    }
    # Fixed values bind as given, and a tie group binds the option Jev chose.
    fixed = [m.code_dcid for m in qmenu.fixed if m.code_system == property_dcid]
    # A tie can span code systems ("budget support" is a cooperation modality and a CRS
    # purpose), so the chosen row's own system names the slot it binds.
    for g, rows in enumerate(qmenu.groups):
        qid = f"tie|{s}|{g}"
        if (choice := answers.choice(qid)) and any(
            r.code_dcid == choice and r.code_system == property_dcid for r in rows
        ):
            fixed.append(choice)
            conf.slots[pid] = float(answers.choices[qid][choice])
    forced = list(dict.fromkeys(fixed))
    taxonomy = (shape.slot_taxonomy or {}).get(f"{axis}:{property_dcid}", [])
    # A one-member taxonomy binds that member: mentioned when the phrase names it.
    if len(taxonomy) == 1 and forced in ([], taxonomy):
        evidence: DraftBindingEvidence = "mentioned" if taxonomy[0] in named else "sole_option"
        return _Slot(_draft(prop, "value", taxonomy, evidence=evidence), seed=True)
    if forced:
        return _Slot(_draft(prop, "value" if len(forced) == 1 else "set", forced), seed=True)
    # A named value under another named value adds nothing to it, and the slot's declared
    # default is the engine's to bind.
    index = get_index()
    offered = [v for v in named if v in menu.offered[s][prop]]
    covered = {d for v in offered for d in index.descendants(v) if d != v}
    offered = [v for v in offered if v not in covered]
    declaration = index.declarations.get(property_dcid)
    if declaration is not None and declaration.default is not None:
        offered = [v for v in offered if v != declaration.default.code]
    # When the phrase asks for no particular value, a value it only implies restates the
    # measure; one it names outright stays.
    if f"part|{pid}" in answers.nouls and answers.noul(f"part|{pid}") <= t.part:
        offered = [v for v in offered if named[v] > t.part_n]
    if len(offered) >= 2 and answers.noul(f"several|{pid}") > t.several:
        conf.slots[pid] = answers.noul(f"several|{pid}")
        return _Slot(_draft(prop, "set", offered))
    if offered:
        best = max(offered, key=named.__getitem__)
        conf.slots[pid] = named[best]
        return _Slot(_draft(prop, "value", [best]))
    for k, phrase in enumerate(menu.request.dropped_qualifier_phrases):
        if (p := answers.noul(f"all|{pid}|{k}")) > t.requested_all:
            conf.slots[pid] = p
            return _Slot(
                _draft(prop, "unbound", evidence="requested_all", requested_all_phrase=phrase)
            )
    return _Slot(_draft(prop, "unbound"))
