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
`assemble` reads the merged answers of both. Where two values of one shape name
overlapping words, the shape keeps one of them (`_span_owned`), so "public and publicly
guaranteed" binds the guarantee status alone and no public-sector debtor beside it. A value
whose words lie inside the measure's, or partly across them, yields to the values the
phrase names and stays only where a member holds it beside them; one whose words equal or
contain the measure's is the user's choice and contends as any named value does.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import get_args

from code_index.shared import get_index, get_measure_index
from search.engine.bind import (
    DraftBindingEvidence,
    DraftBindingKind,
    SlotBindingDraft,
    _UniversalBindOutput,
    ground_bind_output,
    stated_reading,
    validate_requested_all_bindings,
)
from search.engine.binders.base import BindResult
from search.engine.binders.gemini import (
    RATED_POOL_FRACTION,
    holds_drafted_values,
    holds_requested_slots,
    mentioned_drafts,
    near_best,
    nominated_near_best,
)
from search.engine.binders.house_policy import (
    CONVENTION_POLICIES,
    POLICY_ORDER,
    PolicyChoice,
    PolicyName,
    PolicyPool,
)
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
    span_occurrences,
    span_questions,
    word_positions,
)
from search.engine.frame import Frame
from search.engine.qualifier_match import QualifierMatch
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

# The nodes the span rule's searches may visit for one shape (`_Search`): each step of a
# subset walk, each fit test and each placement a test tries. The largest bind of the
# golden and span fixtures visits 79, and ten values of each of two slots claiming one
# word visit 1,195. Random menus of 24 and 40 claims on three slots spend 0.5 to 3.4
# microseconds a node, so the budget holds one shape's search under about 70 ms.
SPAN_SEARCH_BUDGET = 20_000

type Population = tuple[Mapping[str, str], ...]
"""The constraints each member of a shape's complete member population asserts, by
property (`jev.member_population`)."""


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Where each probability turns into a decision.

    `w_m`, `w_mq`, `w_elect` and `w_electq` weigh the two means. A shape's measure score
    above `ask` keeps it electable, and the part asks when none is, as a Gemini reply
    rating its pick under `gemini.DECLINE_FLOOR` declines. It also asks when ``electq``
    gives ``none`` more than any shape and no measure score is above `decline`. Above
    `measures` a shape counts as measuring the phrase for nominal output and the nearest
    refused reading. A shape is plausible when its election is at least ``1 - near_alt``
    of the top among electable shapes, or, without an election answer, when its measure
    score is within `near` of the best. The pool keeps the shapes stating the denominator
    of the plausible shape the election leads with: those whose product of election and
    measure score is at least ``1 - near_elect`` of the best product, each convention
    variant of the lead within `near` of the best measure score (`_convention_variant`),
    and each nominal output shape within `widen` of it (`_with_nominal_output`). A
    plausible shape is an alternate when its measure score is above `alt_floor` and its
    election at least 0.9 of the elected shape's, the bar Gemini's rated pool uses
    (`gemini.near_best` with `gemini.RATED_POOL_FRACTION`). ``mv`` above `measure_value`
    names a measure-axis value, and above `policy_value` one of two a house policy chooses
    between (`_names_value`); ``named`` above `named` names the
    dataset, ``n`` above `value` a slot value. ``part`` at or under `part` asks for no
    particular value, and a value then needs ``n`` above `part_n`. ``several`` above
    `several` binds a set, and ``all`` above `requested_all` an explicit "every value".
    """

    w_m: float = 1.0
    w_mq: float = 1.0
    w_elect: float = 1.0
    w_electq: float = 1.0
    measures: float = 0.5
    # On recorded answers, five wrong picks of the eval corpus score 0.135 or less, and
    # every pick that answers it correctly 0.305 or more.
    ask: float = 0.15
    # Over the 1,405 stored dev answers and their labels, ``none`` above every shape with
    # no measure score over 0.3 asks on 57 parts the `ask` bar answers: 44 the labels mark
    # as asks, 8 with a correct definite pick, 2 with a wrong one and 3 candidates. At 0.4
    # it asks on 77, 56 of them asks and 14 correct definite picks. The lone shape of
    # a labelled lone-shape part measures 0.31 to 0.375 across draws and answers; an FDI
    # part at 0.155 declines.
    decline: float = 0.3
    # On the same answers, with the decline off, the pool this bar admits convention
    # variants to changes 12 picks against dev: 7 to a better pick, 3 to a worse one and 2
    # on parts whose labels accept no shape. 0.1, 0.2 and 0.3 change the same picks.
    near: float = 0.15
    near_elect: float = 0.8
    # The election is the mean of two distributions, so two readings Jev cannot tell
    # apart split it about evenly. This bar sets the plausible shapes the lead and the form
    # rule choose from; at 0.5, 1,126 of the 1,405 labelled parts pick right. Alternates
    # need 0.9 of the elected shape's election besides: against nominating every plausible
    # shape, parts whose status matches the label rise from 1,228 to 1,250, candidates
    # listed on parts labelled definite fall from 35 to 11, candidates listed on parts
    # labelled candidates from 3 to 1, and parts listing every shape the label requires
    # from 1,197 to 1,191. One bar for both moves 3 picks.
    near_alt: float = 0.5
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
    """What each decision read: `measures`, `election` and `named` per shape, the shapes the
    election finds `plausible`, `refused` per reading a pinned part refused, `alternates`
    the rule that listed each alternate, ``elected alike`` for every one (`_alternates`),
    `slots` the probability behind each bound slot, `inferred` the values each slot drafted
    that the span rule left unbound (`_span_owned`), `rules` the election rules that
    applied in order, and the house policies' `pool` and `choice`, None when the part
    asks."""

    measures: dict[int, float] = field(default_factory=dict)
    election: dict[int, float] = field(default_factory=dict)
    named: dict[int, float] = field(default_factory=dict)
    plausible: list[int] = field(default_factory=list)
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
    also = _alternates(shapes, e, conf, t, stated_form)
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
    (`bind.ground_bind_output`), and each shape's measure and election scores and the
    ``none`` share of ``electq`` as the result's scores."""
    output, conf = assemble(
        menu, answers, stated_form=stated_form, thresholds=thresholds, populations=populations
    )
    shapes = range(len(menu.request.shapes))
    result = BindResult(
        output=ground_bind_output(output, menu.request.shapes),
        binder="jev",
        policy_pool=conf.pool,
        policy_choice=conf.choice,
        scores={
            "measure": [round(conf.measures[i], 3) for i in shapes],
            "election": [round(conf.election.get(i, 0.0), 3) for i in shapes],
            "none": round(float(answers.choices.get("electq", {}).get(NONE, 0.0)), 3),
        },
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


def _declines(answers: JevAnswers, best: float, t: Thresholds) -> bool:
    """Whether Jev answers that no menu measure gives the user what the query asks for:
    ``electq`` gives ``none`` more than any shape, and no shape's measure score is above
    `decline`."""
    probabilities = answers.choices.get("electq", {})
    shares = [p for key, p in probabilities.items() if key != NONE]
    return best <= t.decline and probabilities.get(NONE, 0.0) > max(shares, default=0.0)


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


def _convention_variant(menu: JevMenu, lead: int, k: int) -> bool:
    """Whether shape `k` states the reading of `lead` but for values a house convention
    chooses between: the same population and statistic type, and on each measure axis
    where the two differ, and on one at least, a convention policy
    (`house_policy.CONVENTION_POLICIES`) run over the pair would choose, as on amount
    against grant equivalent of one flow. Data at the places and the house default
    dataset choose by the part's situation, so a measure the election set aside never
    replaces the lead for having data, and an imputed multilateral flow the election chose
    is never traded for the house default's amount."""
    a, b = menu.request.shapes[lead], menu.request.shapes[k]
    if (a.pop_type_dcid, a.stat_type_dcid) != (b.pop_type_dcid, b.stat_type_dcid):
        return False
    differing = [axis for axis, f in AXIS_FIELDS.items() if getattr(a, f) != getattr(b, f)]
    return bool(differing) and all(
        _policy_chooses(menu, (lead, k), axis, policies=CONVENTION_POLICIES) for axis in differing
    )


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
    narrows the pool and records its rule in `conf.rules` when it does.

    The election says which readings are plausible (`conf.plausible`) and which one it
    leads with, and so fixes the denominator, since the measure questions ignore form and
    base. The pool the house policies see holds the shapes stating that denominator that
    the election puts near its lead, and each variant of the lead on values a house policy
    chooses between that measures the phrase near the best, so the policies settle a
    convention the phrase leaves open."""
    shapes = menu.request.shapes
    best = max(conf.measures.values(), default=0.0)
    if best <= t.ask:
        conf.rules.append("ask: no shape measures the phrase")
        return None
    if _declines(answers, best, t):
        conf.rules.append("ask: none of the shapes gives what the query asks for")
        return None
    # Only a shape measuring the phrase above `ask` is electable, on every path.
    electable = [i for i in range(len(shapes)) if conf.measures[i] > t.ask]
    if conf.election:
        top = max(conf.election.get(i, 0.0) for i in electable)
        conf.plausible = [
            i for i in electable if conf.election.get(i, 0.0) >= (1 - t.near_alt) * top
        ]
        lead = max(conf.plausible, key=lambda i: conf.election.get(i, 0.0))
    else:
        conf.plausible = [i for i in electable if conf.measures[i] >= best - t.near]
        lead = max(conf.plausible, key=conf.measures.__getitem__)
    # A phrase that states no form asks for the plain measure, not a share of a base.
    if form is None and stated_reading(shapes[lead]).denominator is not None:
        plain = [i for i in conf.plausible if stated_reading(shapes[i]).denominator is None]
        if plain:
            conf.rules.append("form: phrase states none, shapes with a denominator set aside")
            lead = max(plain, key=lambda i: (conf.election.get(i, 0.0), conf.measures[i]))
    denominator = stated_reading(shapes[lead]).denominator
    stating = [i for i in electable if stated_reading(shapes[i]).denominator == denominator]
    best_stating = max(conf.measures[i] for i in stating)
    near = [i for i in stating if conf.measures[i] >= best_stating - t.near]
    if conf.election:
        score = {i: conf.election.get(i, 0.0) * conf.measures[i] for i in stating}
        top = max(score.values())
        pool = [
            i
            for i in stating
            if i == lead
            or score[i] >= (1 - t.near_elect) * top
            or (i in near and _convention_variant(menu, lead, i))
        ]
        pool = _with_nominal_output(shapes, pool, conf, t)
    else:
        pool = near
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


def _policy_chooses(
    menu: JevMenu,
    indices: Sequence[int],
    axis: str,
    *,
    policies: Sequence[PolicyName] = POLICY_ORDER,
) -> bool:
    """Whether one of `policies` would choose on measure axis `axis` among the menu shapes
    at `indices` (`BindRequest.policy_chooses_on`), False when no policy reads the menu."""
    chooses = menu.request.policy_chooses_on
    return chooses is not None and chooses(indices, axis, policies=policies)


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
    names (`gemini.mentioned_drafts`), and the slots it validly asks for in full.

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
    shapes: Sequence[ShapeDraft], e: int, conf: Confidence, t: Thresholds, form: str | None
) -> list[int]:
    """The plausible shapes (`conf.plausible`) the election rates at least 0.9 of `e`, the
    bar Gemini's rated pool uses (`gemini.near_best`), that measure the phrase above
    `alt_floor`, by measure score, highest first: readings the words do not choose between.
    A shape a house policy set aside offers a reading it settled, and where the phrase
    states a form, a shape dividing by another base than `e` offers another quantity."""
    denominator = stated_reading(shapes[e]).denominator
    out = [
        k
        for k in conf.plausible
        if k != e
        and k not in conf.policy_dropped
        and conf.measures[k] > t.alt_floor
        and near_best(conf.election.get(k, 0.0), conf.election.get(e, 0.0), RATED_POOL_FRACTION)
        and (form is None or stated_reading(shapes[k]).denominator == denominator)
    ]
    for k in out:
        conf.alternates[k] = "elected alike"
    return sorted(out, key=lambda k: -conf.measures[k])


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
    """One slot's draft (`_bind_slot`), with what the span rule reads beside it.

    `anchors` are the qualifier matches that set its values, the menu's fixed values and a
    tie's chosen row, empty for a slot Jev drafted. `named` is Jev's ``n|`` probability for
    each value it drafted, and `one_member` marks a value bound because the slot's taxonomy
    holds no other."""

    draft: SlotBindingDraft
    anchors: tuple[QualifierMatch, ...] = ()
    named: Mapping[str, float] = field(default_factory=dict)
    one_member: bool = False


def _shape_drafts(
    menu: JevMenu,
    answers: JevAnswers,
    s: int,
    conf: Confidence,
    t: Thresholds,
    population: Population | None,
) -> list[SlotBindingDraft]:
    """The drafts of shape `s`'s listed slots: each slot's own (`_bind_slot`), then the
    span rule over them (`_span_owned`), with each ``requested_all`` claim validated on `s`
    (`bind.validate_requested_all_bindings`), so a "total" that names no slot claims none."""
    drafts = _span_owned(menu, answers, _slot_drafts(menu, answers, s, conf, t), conf, population)
    slots = {slot.property_dcid for slot in menu.request.shapes[s].slot_keys}
    validated, _applied = validate_requested_all_bindings(
        drafts, menu.request.dropped_qualifier_phrases, slot_properties=slots
    )
    return validated


def _slot_drafts(
    menu: JevMenu, answers: JevAnswers, s: int, conf: Confidence, t: Thresholds
) -> list[_Slot]:
    return [_bind_slot(menu, answers, s, prop, conf, t) for prop in menu.listed[s]]


@dataclass(frozen=True, slots=True)
class _Claim:
    """One value a slot drafted, as the span rule weighs it: the slot's index among the
    shape's drafts and its property, the value, w = logit(p_n x p_span), and each
    occurrence of the words its ``span|`` argmax names (`jev_request.span_occurrences`).
    `occurrences` is None for a value Jev found no words for, and empty for one whose words
    were not read, which conflicts with nothing."""

    slot: int
    property_dcid: str
    value: str
    weight: float
    occurrences: tuple[frozenset[int], ...] | None


class SpanSearchExhausted(Exception):
    """The span rule's searches for one shape passed `SPAN_SEARCH_BUDGET` nodes."""


@dataclass(slots=True)
class _Search:
    """The nodes the span rule's searches for one shape have visited, against `budget`
    (`SPAN_SEARCH_BUDGET`)."""

    budget: int
    nodes: int = 0

    def visit(self) -> None:
        """Counts one node.

        Raises:
            SpanSearchExhausted: The search passed its budget.
        """
        self.nodes += 1
        if self.nodes > self.budget:
            raise SpanSearchExhausted(f"span search passed {self.budget} nodes")


def _fits(claims: Sequence[_Claim], reserved: frozenset[int], search: _Search) -> bool:
    """Whether `claims` are word-consistent: one occurrence of each claim's words can be
    chosen, clear of `reserved` and of the other claims' chosen words, two values of one
    slot excepted. Each test and each placement it tries is a node of `search`."""
    search.visit()
    placing = [
        (c.slot, [o for o in c.occurrences if not o & reserved]) for c in claims if c.occurrences
    ]
    if any(not usable for _slot, usable in placing):
        return False
    # Words written once leave nothing to choose: the claims fit when no two of different
    # slots share a word.
    if all(len(usable) == 1 for _slot, usable in placing):
        return not any(
            a != b and o[0] & q[0] for i, (a, o) in enumerate(placing) for b, q in placing[i + 1 :]
        )
    placed: list[tuple[int, frozenset[int]]] = []

    def place(k: int) -> bool:
        search.visit()
        if k == len(placing):
            return True
        slot, usable = placing[k]
        for o in usable:
            if any(other != slot and o & q for other, q in placed):
                continue
            placed.append((slot, o))
            if place(k + 1):
                return True
            placed.pop()
        return False

    return place(0)


def _fitting_subsets(
    claims: Sequence[_Claim],
    reserved: frozenset[int],
    search: _Search,
    *,
    beside: Sequence[_Claim] = (),
    maximal: bool = False,
) -> Iterator[list[_Claim]]:
    """Each subset of `claims` word-consistent with `beside` (`_fits`), in `claims` order,
    built one claim at a time: a branch whose claims stop fitting ends there, so no subset
    holding it is built. With `maximal`, only the subsets no other claim of `claims` fits
    beside. A branch leaving out a claim then ends when the claim fits beside the claims
    chosen and every later claim that fits beside them, since every subset the branch
    reaches lies within those and so leaves room for it."""
    chosen: list[_Claim] = []

    def fits(*more: _Claim) -> bool:
        return _fits([*beside, *chosen, *more], reserved, search)

    def walk(k: int) -> Iterator[list[_Claim]]:
        search.visit()
        if k == len(claims):
            if not maximal or not any(fits(c) for c in claims if c not in chosen):
                yield list(chosen)
            return
        if fits(claims[k]):
            chosen.append(claims[k])
            yield from walk(k + 1)
            chosen.pop()
            if maximal and fits(claims[k], *(c for c in claims[k + 1 :] if fits(c))):
                return
        yield from walk(k + 1)

    return walk(0)


def _sort_key(claims: Sequence[_Claim]) -> list[tuple[str, str]]:
    return sorted((c.property_dcid, c.value) for c in claims)


def _heaviest(claims: Sequence[_Claim]) -> tuple[float, list[tuple[str, str]]]:
    """Step 1's order: the larger sum of w first, then the first sorted list."""
    return -sum(c.weight for c in claims), _sort_key(claims)


def _owners(worded: Sequence[_Claim], reserved: frozenset[int], search: _Search) -> list[_Claim]:
    """Step 1 of the span rule (`_span_owned`): the maximal word-consistent subset of
    `worded` with the largest sum of w, ties to the first sorted (property, value) list.

    Two claims of different slots contend when an occurrence of one's words shares a word
    with an occurrence of the other's, neither touching `reserved`. A claim that contends
    with none on one of its occurrences fits beside any subset, so it is in every maximal
    one. The rest split into groups that contend only within themselves, and fit,
    maximality and the sum of w each hold group by group, so each group is searched alone
    (`_fitting_subsets`). The tie-break composes too: of two maximal subsets of a group
    neither holds the other, since a subset of claims that fit fits, so the one whose
    sorted list sorts first still does with any other group's claims merged into both."""

    def usable(c: _Claim) -> list[frozenset[int]]:
        return [o for o in c.occurrences or () if not o & reserved]

    def contend(a: _Claim, b: _Claim) -> bool:
        return a.slot != b.slot and any(o & q for o in usable(a) for q in usable(b))

    def free(c: _Claim) -> bool:
        return c.occurrences == () or any(
            not any(x.slot != c.slot and o & q for x in worded for q in usable(x))
            for o in usable(c)
        )

    won = [c for c in worded if free(c)]
    groups: list[list[_Claim]] = []
    for c in (c for c in worded if not free(c)):
        joined = [g for g in groups if any(contend(c, x) for x in g)]
        merged = [x for g in joined for x in g]
        groups = [g for g in groups if g not in joined]
        groups.append([*merged, c])
    for group in groups:
        group.sort(key=worded.index)
        won += min(_fitting_subsets(group, reserved, search, maximal=True), key=_heaviest)
    return won


def _span_owned(
    menu: JevMenu,
    answers: JevAnswers,
    slots: Sequence[_Slot],
    conf: Confidence,
    population: Population | None,
) -> list[SlotBindingDraft]:
    """The drafts of `slots` with each stretch of the phrase paying for one value.

    The values the phrase names (`gemini.mentioned_drafts`) on two or more properties
    contend; with fewer the drafts stand as drafted. ``sole_option`` drafts take no part:
    the phrase names none of their values, and the engine unbinds one the members do not
    all share before any requirement is judged.

    A slot the qualifier matcher set (`_Slot.anchors`: the menu's fixed values and a tie's
    chosen row) is an anchor. Its words are the matches' character offsets, it never loses,
    and every realisability check holds it. Every other named value is a claim
    (`_Claim`), one per value of a set, weighed by its own ``n|`` probability and its span
    answer's. A claim is wordless when Jev answered "none"; measure-worded when every
    occurrence of its words overlaps an occurrence of the measure's words
    (``span|measure``) without containing it, so the measure noun implies it, as
    "Outstanding external debt" implies a loan through "debt"; else worded. A value whose
    words equal or contain the measure's is worded: the user chose those words.

    A set of claims is word-consistent when one occurrence of each claim's words can be
    chosen, clear of every anchor's words and of the other claims' chosen words, two values
    of one slot excepted (``several|`` already read them together). The test is joint, so
    three claims on a word written twice do not fit, and every ambiguity of a repeated word
    resolves toward keeping. The rule, in order:

    1. Of the worded claims, the maximal word-consistent subset with the largest sum of w
       stays: a claim is dropped only because a kept claim or an anchor holds its words, so
       a conflict keeps one value however weak both claims are. A losing claim stays when
       it narrows nothing beyond the kept claims and the anchors (`narrows_nothing`), as
       "net flows" names one concept on two properties. Realisability plays no part, so a
       typed value is never traded for a weaker one that fits the data.
    2. A measure-worded claim that is not word-consistent with the claims step 1 chose
       drops.
    3. Of the rest, the word-consistent subset stays that a member holds together with the
       kept claims and the anchors, then the larger subset, then the larger sum of w. When
       no member holds the kept claims and the anchors, none stays: the engine refuses on
       the typed values and should name no value the user never wrote. With no complete
       population every one stays, the engine's loud refusal over a silent broadening.
    4. A wordless claim stays only when it narrows nothing beyond everything kept.

    Ties fall to the subset whose sorted (property, value) list sorts first. Steps 1 and 3
    search exactly (`_owners`, `_fitting_subsets`), within `SPAN_SEARCH_BUDGET` nodes for
    the shape. Each slot is rebuilt from its surviving values: two or more
    bind a set, one a value, none leaves it unbound. A one-member slot whose value loses
    keeps it as ``sole_option``, since the slot offers nothing else, and the engine then
    unbinds it silently if it conflicts. `conf.inferred` lists each value taken from the
    phrase.

    A member holds a slot when it asserts one of its kept values, widened as the engine
    widens a bound value (`bound_values.bound_values_from_bindings`); a member silent on
    the property holds none. `population` is None when the shape has no complete member
    population (`jev.member_population`): no claim then narrows nothing, since one dropped
    over a sample could be held by a member the sample lacks.

    Raises:
        SpanSearchExhausted: The searches passed `SPAN_SEARCH_BUDGET` nodes.
    """
    drafts = [slot.draft for slot in slots]
    named = [i for i, slot in enumerate(slots) if mentioned_drafts([slot.draft])]
    if len({drafts[i].property_dcid for i in named}) < 2:
        return drafts
    variable = menu.request.variable
    anchored = [i for i in named if slots[i].anchors]
    reserved = frozenset[int]().union(
        *(
            word_positions(variable, m.start, m.end)
            for i in anchored
            for m in slots[i].anchors
            if m.start is not None and m.end is not None
        )
    )
    measure = _read_occurrences(answers, SPAN_MEASURE, variable)[0] or ()

    worded: list[_Claim] = []
    measured: list[_Claim] = []
    wordless: list[_Claim] = []
    for i in named:
        if slots[i].anchors:
            continue
        d = drafts[i]
        prop = (d.axis, d.property_dcid or "")
        for value in d.value_dcids:
            occurrences, p_span = _read_occurrences(answers, span_id(prop, value), variable)
            weight = _logit(slots[i].named.get(value, 0.5) * p_span)
            claim = _Claim(i, prop[1], value, weight, occurrences)
            if occurrences is None:
                wordless.append(claim)
            elif occurrences and all(
                any(o & m and not o >= m for m in measure) for o in occurrences
            ):
                measured.append(claim)
            else:
                worded.append(claim)

    widened: dict[tuple[int, frozenset[str]], frozenset[str]] = {}

    def holds(member: Mapping[str, str], claims: Sequence[_Claim]) -> bool:
        values: dict[int, set[str]] = {i: set(drafts[i].value_dcids) for i in anchored}
        for c in claims:
            values.setdefault(c.slot, set()).add(c.value)
        for i, held in values.items():
            key = (i, frozenset(held))
            if key not in widened:
                prop = drafts[i].property_dcid or ""
                bound = bound_values_from_bindings(
                    [_draft((drafts[i].axis, prop), "set", sorted(held))]
                )
                widened[key] = frozenset(bound.get(prop, held))
            if member.get(drafts[i].property_dcid or "") not in widened[key]:
                return False
        return True

    def realisable(claims: Sequence[_Claim]) -> bool:
        return population is not None and any(holds(m, claims) for m in population)

    def narrows_nothing(c: _Claim, kept: Sequence[_Claim]) -> bool:
        if population is None:
            return False
        holding = [m for m in population if holds(m, kept)]
        return bool(holding) and all(holds(m, [c]) for m in holding)

    # 1. Worded claims.
    search = _Search(SPAN_SEARCH_BUDGET)
    won = _owners(worded, reserved, search)
    kept = [*won, *(x for x in worded if x not in won and narrows_nothing(x, won))]

    # 2 and 3. Measure-worded claims. Words are checked against the claims that won; one
    # kept for narrowing nothing shares its words with them.
    def best_held(c: Sequence[_Claim]) -> tuple[int, int, float, list[tuple[str, str]]]:
        return -realisable([*kept, *c]), -len(c), -sum(x.weight for x in c), _sort_key(c)

    open_ = [x for x in measured if _fits([*won, x], reserved, search)]
    if population is None:
        kept += open_
    elif realisable(kept):
        kept += min(_fitting_subsets(open_, reserved, search, beside=won), key=best_held)

    # 4. Wordless claims.
    kept += [x for x in wordless if narrows_nothing(x, kept)]

    survivors = {(c.slot, c.value) for c in kept}
    for i in named:
        if slots[i].anchors:
            continue
        d = drafts[i]
        prop = (d.axis, d.property_dcid or "")
        pid = prop_id(prop)
        staying = [v for v in d.value_dcids if (i, v) in survivors]
        if staying == d.value_dcids:
            continue
        conf.inferred[pid] = tuple(v for v in d.value_dcids if v not in staying)
        if slots[i].one_member:
            drafts[i] = _draft(prop, "value", d.value_dcids, evidence="sole_option")
            conf.slots.pop(pid, None)
        elif len(staying) >= 2:
            drafts[i] = _draft(prop, "set", staying)
        elif staying:
            drafts[i] = _draft(prop, "value", staying)
            conf.slots[pid] = slots[i].named.get(staying[0], conf.slots.get(pid, 0.5))
        else:
            drafts[i] = _draft(prop, "unbound")
            conf.slots.pop(pid, None)
    return drafts


def _read_occurrences(
    answers: JevAnswers, qid: str, variable: str
) -> tuple[tuple[frozenset[int], ...] | None, float]:
    """Each occurrence in `variable` of the words the argmax answer to `qid` names, and its
    probability. None when the argmax is `NONE`. A question Jev did not answer, and an
    answer naming no span of `variable`, give no occurrence at probability 1.0."""
    span = answers.choice(qid)
    if span is None:
        return (), 1.0
    p = float(answers.choices[qid][span])
    if span == NONE:
        return None, p
    return span_occurrences(variable, span), p


def _logit(p: float) -> float:
    p = min(max(p, _LOG_ODDS_CLAMP), 1 - _LOG_ODDS_CLAMP)
    return math.log(p / (1 - p))


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
    anchors = [m for m in qmenu.fixed if m.code_system == property_dcid]
    # A tie can span code systems ("budget support" is a cooperation modality and a CRS
    # purpose), so the chosen row's own system names the slot it binds.
    for g, rows in enumerate(qmenu.groups):
        qid = f"tie|{s}|{g}"
        if (choice := answers.choice(qid)) and (
            chosen := [r for r in rows if r.code_dcid == choice and r.code_system == property_dcid]
        ):
            anchors.append(chosen[0])
            conf.slots[pid] = float(answers.choices[qid][choice])
    forced = list(dict.fromkeys(m.code_dcid for m in anchors))
    taxonomy = (shape.slot_taxonomy or {}).get(f"{axis}:{property_dcid}", [])
    # A one-member taxonomy binds that member: mentioned when the phrase names it or the
    # qualifier matcher set it.
    if len(taxonomy) == 1 and forced in ([], taxonomy):
        sole = taxonomy[0]
        evidence: DraftBindingEvidence = "mentioned" if sole in named or anchors else "sole_option"
        return _Slot(
            _draft(prop, "value", taxonomy, evidence=evidence),
            anchors=tuple(anchors),
            named={sole: named[sole]} if sole in named else {},
            one_member=True,
        )
    if forced:
        kind: DraftBindingKind = "value" if len(forced) == 1 else "set"
        return _Slot(_draft(prop, kind, forced), anchors=tuple(anchors))
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
        return _Slot(_draft(prop, "set", offered), named={v: named[v] for v in offered})
    if offered:
        best = max(offered, key=named.__getitem__)
        conf.slots[pid] = named[best]
        return _Slot(_draft(prop, "value", [best]), named={best: named[best]})
    for k, phrase in enumerate(menu.request.dropped_qualifier_phrases):
        if (p := answers.noul(f"all|{pid}|{k}")) > t.requested_all:
            conf.slots[pid] = p
            return _Slot(
                _draft(prop, "unbound", evidence="requested_all", requested_all_phrase=phrase)
            )
    return _Slot(_draft(prop, "unbound"))
