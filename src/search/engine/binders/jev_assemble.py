"""Jev's answers turned into the bind output and the quantity read.

Jev writes no text and no dcid: every decision here is a threshold (`Thresholds`) on a
probability Jev returned for a question of `jev_request`, so the functions are pure and a
threshold change replays from stored answers. A question Jev did not answer reads as a
probability of 0.0.

`assemble` elects a shape, binds its listed slots and lists its alternates; `bind_result`
applies the rules the engine applies to every binder's drafts (`bind.ground_bind_output`)
and carries each alternate's measure score as `BindResult.also_plausible_support`.
`assemble_quantity` reads the stated form and base from the ``q|`` answers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
)
from search.engine.binders.base import BindResult
from search.engine.binders.gemini import holds_drafted_values, mentioned_drafts
from search.engine.binders.house_policy import PolicyChoice, PolicyPool
from search.engine.binders.jev_menu import (
    AXIS_FIELDS,
    JevMenu,
    Prop,
    asked_values,
    prop_id,
)
from search.engine.binders.jev_request import NONE, NONE_OF_THESE
from search.engine.frame import Frame
from search.engine.quantity_read import QuantityOutput
from search.engine.shape import ShapeDraft
from search.models import GraphRef, QuantityForm

ASK_TEXT = "No candidate shape measures what the variable phrase names (Jev V2)."

_FORM_OF: dict[str, QuantityForm] = {form: form for form in get_args(QuantityForm)}

# The ``q|gate`` probability at which the quantity read states anything.
QUANTITY_GATE = 0.5


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Where each probability turns into a decision.

    `w_m`, `w_mq`, `w_elect` and `w_electq` weigh the two means. A shape's measure score
    above `ask` keeps it electable, and the part asks when none is; above `measures` it
    counts as measuring the phrase for alternates, nominal output and the nearest refused
    reading. Without an election answer the pool keeps shapes within `near` of the best
    measure score; with one, the products of election and measure score at least
    ``1 - near_elect`` of the best. An alternate is "elected alike" when its election is at
    least ``1 - near_alt`` of the top and its measure score above `alt_floor`. A nominal
    output shape joins the pool within `widen` of the best measure score. ``mv`` above
    `measure_value` names a measure-axis value, ``named`` above `named` the dataset, ``n``
    above `value` a slot value. ``part`` at or under `part` asks for no particular value,
    and a value then needs ``n`` above `part_n`. ``several`` above `several` binds a set,
    and ``all`` above `requested_all` an explicit "every value".
    """

    w_m: float = 1.0
    w_mq: float = 1.0
    w_elect: float = 1.0
    w_electq: float = 1.0
    measures: float = 0.5
    ask: float = 0.1
    near: float = 0.15
    near_elect: float = 0.8
    near_alt: float = 0.95
    alt_floor: float = 0.3
    widen: float = 0.2
    part: float = 0.15
    part_n: float = 0.8
    named: float = 0.5
    measure_value: float = 0.4
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


@dataclass(slots=True)
class Confidence:
    """What each decision read: `measures`, `election` and `named` per shape, `refused` per
    reading a pinned part refused, `alternates` the rule that listed each alternate,
    `slots` the probability behind each bound slot, `rules` the election rules that
    applied in order, and the house policies' `pool` and `choice`, None when the part
    asks."""

    measures: dict[int, float] = field(default_factory=dict)
    election: dict[int, float] = field(default_factory=dict)
    named: dict[int, float] = field(default_factory=dict)
    refused: dict[int, float] = field(default_factory=dict)
    alternates: dict[int, str] = field(default_factory=dict)
    slots: dict[str, float] = field(default_factory=dict)
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


def assemble(
    menu: JevMenu,
    answers: JevAnswers,
    *,
    stated_form: str | None,
    thresholds: Thresholds = THRESHOLDS,
) -> tuple[_UniversalBindOutput, Confidence]:
    """The bind output Jev's `answers` give over `menu` (`jev_menu.jev_menu` of the bind
    request, the menu the questions were asked of), before the engine's rules apply to its
    drafts, and what each decision read.

    `stated_form` is the form the phrase states (the grounded quantity read), None for a
    plain amount. The request's house-policy step (`BindRequest.apply_policies`) runs once
    when a shape is elected, over the pool the election leaves, and never on an ask.
    """
    t = thresholds
    shapes = menu.request.shapes
    conf = Confidence()
    conf.measures = _measure_scores(answers, len(shapes), t)
    conf.election = _election(answers, t)
    conf.named = {i: answers.noul(f"named|{i}") for i in range(len(shapes))}
    conf.refused = {r: answers.noul(f"mr|{r}") for r in menu.refused}

    e = _elect(menu, answers, conf, t, stated_form) if shapes else None
    if e is None:
        measuring = {r: p for r, p in conf.refused.items() if p > t.measures}
        nearest = max(measuring, key=measuring.__getitem__) if measuring else None
        output = _UniversalBindOutput(
            chosen_shape_index=0, ask=ASK_TEXT, nearest_refused_index=nearest
        )
        return output, conf
    bindings = [_bind_slot(menu, answers, e, prop, conf, t) for prop in menu.listed[e]]
    also = _alternates(menu, e, conf, t, answers, stated_form)
    return _UniversalBindOutput(
        chosen_shape_index=e, bindings=bindings, also_plausible=also or None
    ), conf


def bind_result(
    menu: JevMenu,
    answers: JevAnswers,
    *,
    stated_form: str | None,
    thresholds: Thresholds = THRESHOLDS,
) -> tuple[BindResult, Confidence]:
    """`assemble` over `menu`, with the engine's rules applied to the drafts
    (`bind.ground_bind_output`) and each alternate's measure score as its support."""
    output, conf = assemble(menu, answers, stated_form=stated_form, thresholds=thresholds)
    support = {k: conf.measures[k] for k in output.also_plausible or ()}
    result = BindResult(
        output=ground_bind_output(output, menu.request.shapes),
        binder="jev",
        policy_pool=conf.pool,
        policy_choice=conf.choice,
        also_plausible_support=support or None,
    )
    return result, conf


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
) -> int | None:
    """The elected menu index, None to ask. Each step narrows the pool and records its
    rule in `conf.rules` when it does."""
    shapes = menu.request.shapes
    best = max(conf.measures.values(), default=0.0)
    if best <= t.ask:
        conf.rules.append("ask: no shape measures the phrase")
        return None
    if conf.election:
        score = {i: conf.election.get(i, 0.0) * conf.measures[i] for i in range(len(shapes))}
        top = max(score.values())
        pool = [
            i
            for i in range(len(shapes))
            if conf.measures[i] > t.ask and score[i] >= (1 - t.near_elect) * top
        ] or [max(score, key=score.__getitem__)]
        pool = _with_nominal_output(shapes, pool, conf, t)
    else:
        pool = [i for i in range(len(shapes)) if conf.measures[i] >= best - t.near]
    # A phrase that states no form asks for the plain measure, not a share of a base.
    if form is None:
        plain = [i for i in pool if stated_reading(shapes[i]).denominator is None]
        if plain and len(plain) < len(pool):
            conf.rules.append("form: phrase states none, shapes with a denominator set aside")
            pool = plain
    # A measure value or a dataset the phrase names keeps the shapes that hold it.
    index = get_measure_index()
    for axis, field_name in AXIS_FIELDS.items():
        holding = [
            i
            for i in pool
            if (v := getattr(shapes[i], field_name))
            and answers.noul(f"mv|{index.canonical_reading(v)}") > t.measure_value
        ]
        if holding and len(holding) < len(pool):
            conf.rules.append(f"named: {axis}")
            pool = holding
    named = [i for i in pool if conf.named[i] > t.named]
    if named and len(named) < len(pool):
        conf.rules.append("named: dataset")
        pool = named
    # The house policies decide what the phrase leaves open, among the shapes that hold
    # what Jev recognised on its pick; the election among what they keep, and menu rank
    # between equal elections.
    pick = max(pool, key=lambda i: conf.election.get(i, 0.0))
    conf.pool = PolicyPool(
        measuring=(pick, *(i for i in pool if i != pick)),
        election=dict(conf.election),
        rated=True,
    )
    holds = _holds(menu, answers, pick, conf.pool.measuring, t)
    choice = menu.request.apply_policies(conf.pool, holds=holds)
    conf.choice = choice
    conf.rules += [f"house: {p}" for p in choice.fired]
    return choice.survivor


def _holds(
    menu: JevMenu, answers: JevAnswers, pick: int, pool: Sequence[int], t: Thresholds
) -> dict[int, bool]:
    """Whether each `pool` shape holds the constraints Jev recognised on `pick`: the value
    and set drafts the phrase names (`gemini.mentioned_drafts`) that `pick`'s slots bind at
    the thresholds (`_bind_slot`).

    A shape holds them when it holds each value on its own slot as `pick` does
    (`gemini.holds_drafted_values`), or when Jev recognises a value on a slot of that shape
    that `pick` lacks: one dataset's taxonomy then states the constraint another's states
    on `pick`, as DAC2a's type of finance states the CRS's loans. The second test does not
    pair constraints one to one, so a shape stating one of `pick`'s several constraints in
    its own taxonomy holds them all."""
    shapes = menu.request.shapes

    def recognised(s: int) -> list[SlotBindingDraft]:
        drafts = [_bind_slot(menu, answers, s, prop, Confidence(), t) for prop in menu.listed[s]]
        return mentioned_drafts(drafts)

    on_pick = recognised(pick)
    pick_slots = {slot.property_dcid for slot in shapes[pick].slot_keys}
    return {
        k: holds_drafted_values(shapes[pick], shapes[k], on_pick)
        or any(d.property_dcid not in pick_slots for d in recognised(k))
        for k in pool
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

    def named(value: str | None) -> bool:
        return bool(value) and answers.noul(f"mv|{index.canonical_reading(value)}") > (
            t.measure_value
        )

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
                (a, b)
                for a, b in (
                    (elected.population_type, other.population_type),
                    (elected.measured_property, other.measured_property),
                    (elected.qualifier, other.qualifier),
                )
                if a != b
            ]
            if any(named(a) or named(b) for a, b in differing):
                continue
            conf.alternates[k] = "measure family variant"
        elif (other.population_type, other.measured_property) != (
            elected.population_type,
            elected.measured_property,
        ):
            continue
        elif other.qualifier != elected.qualifier:
            if named(other.qualifier) or named(elected.qualifier):
                continue
            conf.alternates[k] = "qualifier variant"
        elif other.label != elected.label:
            conf.alternates[k] = "dataset variant"
        else:
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


def _bind_slot(
    menu: JevMenu,
    answers: JevAnswers,
    s: int,
    prop: Prop,
    conf: Confidence,
    t: Thresholds,
) -> SlotBindingDraft:
    """The draft of slot `prop` on the elected shape `s`."""
    shape, qmenu = menu.request.shapes[s], menu.menus[s]
    axis, property_dcid = prop
    pid = prop_id(prop)
    # The engine alone sets a source.
    if axis == "source":
        return _draft(prop, "unbound")
    named = {
        v: p
        for j, v in enumerate(asked_values(menu, prop))
        if (p := answers.noul(f"n|{pid}|{j}")) > t.value
    }
    # Fixed values bind as given, and a tie group binds the option Jev chose.
    fixed = [m.code_dcid for m in qmenu.fixed if m.code_system == property_dcid]
    for g, rows in enumerate(qmenu.groups):
        qid = f"tie|{s}|{g}"
        if rows[0].code_system == property_dcid and (choice := answers.choice(qid)):
            fixed.append(choice)
            conf.slots[pid] = float(answers.choices[qid][choice])
    forced = list(dict.fromkeys(fixed))
    taxonomy = (shape.slot_taxonomy or {}).get(f"{axis}:{property_dcid}", [])
    # A one-member taxonomy binds that member: mentioned when the phrase names it.
    if len(taxonomy) == 1 and forced in ([], taxonomy):
        evidence: DraftBindingEvidence = "mentioned" if taxonomy[0] in named else "sole_option"
        return _draft(prop, "value", taxonomy, evidence=evidence)
    if forced:
        return _draft(prop, "value" if len(forced) == 1 else "set", forced)
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
        return _draft(prop, "set", offered)
    if offered:
        best = max(offered, key=named.__getitem__)
        conf.slots[pid] = named[best]
        return _draft(prop, "value", [best])
    for k, phrase in enumerate(menu.request.dropped_qualifier_phrases):
        if (p := answers.noul(f"all|{pid}|{k}")) > t.requested_all:
            conf.slots[pid] = p
            return _draft(prop, "unbound", evidence="requested_all", requested_all_phrase=phrase)
    return _draft(prop, "unbound")
