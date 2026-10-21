"""Jev's answers turned into the bind output and the quantity read.

Jev writes no text and no dcid: every decision here is a threshold (`Thresholds`) on a
probability Jev returned for a question of `jev_request`, so the functions are pure and a
threshold change replays from stored answers. A question Jev did not answer reads as a
probability of 0.0.

`assemble` elects a shape, binds its listed slots and lists its alternates; `bind_result`
applies the rules the engine applies to every binder's drafts (`bind.ground_bind_output`).
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
    validate_requested_all_bindings,
)
from search.engine.binders.base import BindResult
from search.engine.binders.gemini import (
    holds_drafted_values,
    holds_requested_slots,
    mentioned_drafts,
    nominated_near_best,
)
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
from search.engine.retrieval.bound_values import shape_part_datasets
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
    plausible shape is an alternate when its measure score is above `alt_floor`. ``mv``
    above `measure_value` names a measure-axis value, and above `policy_value` one of two a
    house policy chooses between (`_names_value`); ``named`` above `named` names the
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
    # no measure score over 0.4 asks on 77 parts the `ask` bar answers: 56 the labels mark
    # as asks, 14 with a correct definite pick, 2 with a wrong one and 5 candidates. With no
    # measure bound it asks on 238, 128 of them correct definite picks.
    decline: float = 0.4
    # On the same answers the convention variants it admits change 13 picks at 0.1 and at
    # 0.15, and 17 at 0.3.
    near: float = 0.15
    near_elect: float = 0.8
    # The election is the mean of two distributions, so two readings Jev cannot tell
    # apart split it about evenly. At 0.5, 40 of the 1,405 parts list alternates, 3 of the
    # 60 the labels mark as candidates among them; at 0.05 of the top, 285 did, 12 of the
    # 60 among them.
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


@dataclass(slots=True)
class Confidence:
    """What each decision read: `measures`, `election` and `named` per shape, the shapes the
    election finds `plausible`, `refused` per reading a pinned part refused, `alternates`
    the rule that listed each alternate,
    `slots` the probability behind each bound slot, `rules` the election rules that
    applied in order, and the house policies' `pool` and `choice`, None when the part
    asks."""

    measures: dict[int, float] = field(default_factory=dict)
    election: dict[int, float] = field(default_factory=dict)
    named: dict[int, float] = field(default_factory=dict)
    plausible: list[int] = field(default_factory=list)
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
    bindings = _shape_drafts(menu, answers, e, conf, t)
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
) -> tuple[BindResult, Confidence]:
    """`assemble` over `menu`, with the engine's rules applied to the drafts
    (`bind.ground_bind_output`), and each shape's measure and election scores and the
    ``none`` share of ``electq`` as the result's scores."""
    output, conf = assemble(menu, answers, stated_form=stated_form, thresholds=thresholds)
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
    """Whether shape `k` states the reading of `lead` but for values a house policy
    chooses between: the same population and statistic type and a dataset in common, and on
    each measure axis where the two differ, and on one at least, a policy run over the
    pair would choose (`_policy_chooses`), as on amount against grant equivalent of one
    flow. Across datasets the election's reading of the words stands, so
    `default_dataset` never trades an imputed multilateral flow the election chose for
    the house default's amount."""
    a, b = menu.request.shapes[lead], menu.request.shapes[k]
    if (a.pop_type_dcid, a.stat_type_dcid) != (b.pop_type_dcid, b.stat_type_dcid):
        return False
    index = get_index()
    if not set(shape_part_datasets(a, index)) & set(shape_part_datasets(b, index)):
        return False
    differing = [axis for axis, f in AXIS_FIELDS.items() if getattr(a, f) != getattr(b, f)]
    return bool(differing) and all(_policy_chooses(menu, (lead, k), axis) for axis in differing)


def _elect(
    menu: JevMenu,
    answers: JevAnswers,
    conf: Confidence,
    t: Thresholds,
    form: str | None,
) -> int | None:
    """The elected menu index, None to ask. Each step narrows the pool and records its
    rule in `conf.rules` when it does.

    The election says which readings are plausible (`conf.plausible`) and which one it
    leads with, and so fixes the denominator, since the measure questions ignore form and
    base. The pool the house policies see holds the shapes stating that denominator that
    the election puts near its lead, and each variant of the lead on values a house policy
    chooses between that measures the phrase near the best, so the policies and not the
    election settle a convention the phrase leaves open."""
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
    # The house policies decide what the phrase leaves open, among the shapes that hold
    # what Jev recognised on its pick; the election among what they keep, and menu rank
    # between equal elections.
    pick = max(pool, key=lambda i: conf.election.get(i, 0.0))
    conf.pool = PolicyPool(
        measuring=(pick, *(i for i in pool if i != pick)),
        election=dict(conf.election),
    )
    holds = _holds(menu, answers, pick, conf.pool.measuring, t)
    choice = menu.request.apply_policies(conf.pool, holds=holds)
    conf.choice = choice
    conf.rules += [f"house: {p}" for p in choice.fired]
    return choice.survivor


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
    menu: JevMenu, answers: JevAnswers, pick: int, pool: Sequence[int], t: Thresholds
) -> dict[int, bool]:
    """Whether each `pool` shape holds the constraints Jev recognised on `pick`, as
    `pick`'s slots bind them at the thresholds (`_bind_slot`): the value and set drafts the
    phrase names (`gemini.mentioned_drafts`), and the slots it validly asks for in full
    (`_shape_drafts`).

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
        return _shape_drafts(menu, answers, s, Confidence(), t)

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
    """The shapes the election finds as plausible as `e` (`conf.plausible`) that measure
    the phrase above `alt_floor`, by measure score, highest first: readings the words do
    not choose between. A shape a house policy set aside offers a reading it settled, and
    where the phrase states a form, a shape dividing by another base than `e` offers
    another quantity."""
    denominator = stated_reading(shapes[e]).denominator
    out = [
        k
        for k in conf.plausible
        if k != e
        and k not in conf.policy_dropped
        and conf.measures[k] > t.alt_floor
        and (form is None or stated_reading(shapes[k]).denominator == denominator)
    ]
    for k in out:
        conf.alternates[k] = "elected alike"
    return sorted(out, key=lambda k: -conf.measures[k])


def _shape_drafts(
    menu: JevMenu, answers: JevAnswers, s: int, conf: Confidence, t: Thresholds
) -> list[SlotBindingDraft]:
    """The drafts of shape `s`'s listed slots (`_bind_slot`), with each ``requested_all``
    claim validated on `s` (`bind.validate_requested_all_bindings`), so a "total" that names
    no slot claims none."""
    drafts = [_bind_slot(menu, answers, s, prop, conf, t) for prop in menu.listed[s]]
    slots = {slot.property_dcid for slot in menu.request.shapes[s].slot_keys}
    validated, _applied = validate_requested_all_bindings(
        drafts, menu.request.dropped_qualifier_phrases, slot_properties=slots
    )
    return validated


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
