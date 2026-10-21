"""Recorded Jev binds replayed through the pure assembly: the span rule on real answers.

Each file under tests/fixtures/jev/span is one bind of one query, recorded against dev's
graph and catalogue with `jev-1.13.0`: the variable phrase, the elected shape's listed
slots with the values each asks about in question order, both rounds' answers of each run
(``span|`` answers included, the elected shape renumbered to menu shape 0), and the complete
member population the bind read for the shape (`jev.member_population`), None where the
bind asked no span round. A slot's ``offered`` is the elected shape's own taxonomy where
it differs from the union (a sole option's one value, or the union less a value the shape
does not offer), and ``fixed`` the value the menu fixed (`bind._QualifierMenu.fixed`).
``recorded_drafts`` are the drafts the recording's binder returned. They follow the rule
under test for the typed pair's three runs, DPPG 2020's fourth and PCBK's fourth, and
another span rule for the other runs, so the replays assert the drafts in `_SPANNED`.

The menu is rebuilt as the one shape the recordings elect, so the replay reads the same
question ids, positions and taxonomy the recorded bind did.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from search.engine.bind import SlotBindingDraft, _QualifierMenu
from search.engine.binders.base import BindRequest
from search.engine.binders.jev_assemble import (
    Confidence,
    JevAnswers,
    assemble,
    span_round,
)
from search.engine.binders.jev_menu import JevMenu, Prop
from search.engine.qualifier_match import QualifierMatch
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.models import Axis
from tests.engine._binders import keep_the_pick

SPAN = Path(__file__).parents[1] / "fixtures" / "jev" / "span"


def _record(name: str) -> dict[str, Any]:
    return json.loads((SPAN / f"{name}.json").read_text())


def _prop(slot: str) -> Prop:
    axis, prop = slot.split(":")
    return cast(Axis, axis), prop


def recorded_menu(record: dict[str, Any]) -> JevMenu:
    """The menu of the recorded bind: its elected shape alone, with each slot's values in
    the order the recorded questions asked them."""
    slots = [(_prop(s["slot"]), s) for s in record["slots"]]
    props = [p for p, _s in slots]
    union = {p: [v["dcid"] for v in s["values"]] for p, s in slots}
    offered = {p: s.get("offered", union[p]) for p, s in slots}
    labels = {v["dcid"]: v["label"] for s in record["slots"] for v in s["values"]}
    shape = ShapeDraft(
        shape_id="recorded",
        label=record["variable"],
        pop_type_dcid="Thing",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=tuple(
            SlotKeyDraft(axis=axis, property_dcid=prop, label=s["label"])
            for (axis, prop), s in slots
        ),
        slot_taxonomy={f"{a}:{p}": values for (a, p), values in offered.items()},
        value_labels=labels,
    )
    fixed = tuple(
        QualifierMatch(
            code_system=prop, code_dcid=dcid, label=dcid, source="code", matched_text=dcid
        )
        for (_axis, prop), s in slots
        for dcid in s.get("fixed", ())
    )
    request = BindRequest(
        variable=record["variable"],
        shapes=[shape],
        query_text=record["query"],
        apply_policies=keep_the_pick,
    )
    return JevMenu(
        request,
        (_QualifierMenu(fixed=fixed, groups=()),),
        (tuple(props),),
        (offered,),
        {p: (s["label"], union[p]) for p, s in slots},
        labels,
        (),
    )


def recorded_answers(run: dict[str, Any]) -> JevAnswers:
    return JevAnswers(nouls=run["nouls"], choices=run["choices"])


def replay(record: dict[str, Any], run: dict[str, Any]) -> tuple[dict[str, list[str]], Confidence]:
    """The values each slot's named draft binds (`gemini.mentioned_drafts`'s evidence) once
    the run's answers are assembled over the recorded population."""
    population = record["population"]
    output, conf = assemble(
        recorded_menu(record),
        recorded_answers(run),
        stated_form=None,
        populations={0: tuple(population)} if population is not None else None,
    )
    return _named(output.bindings), conf


def _named(drafts: list[SlotBindingDraft] | list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for d in drafts:
        draft = d if isinstance(d, SlotBindingDraft) else SlotBindingDraft.model_validate(d)
        if draft.kind in ("value", "set") and draft.evidence == "mentioned":
            out[draft.property_dcid or ""] = sorted(draft.value_dcids)
    return out


_AO = {"financialPositionType": ["AmountOutstanding"]}
_PPG = {"guaranteeStatus": ["PublicOrPubliclyGuaranteed"]}

_SPANNED = {
    # "debt stocks" is the measure's, so amount outstanding is measure-worded and stays because
    # members hold it with PPG. The public-sector debtor's "public" loses to PPG's words.
    "kenya_words": {**_AO, **_PPG},
    "dppg_kenya": {**_AO, **_PPG},
    "dppg_kenya_2020": {**_AO, **_PPG},
    # "private sector" names the debtor (w 1.8) and the private creditors (w -0.1).
    "dpng_zambia": {
        **_AO,
        "guaranteeStatus": ["NotPubliclyGuaranteed"],
        "idsDebtorType": ["IDSDebtorTypePrivateSector"],
    },
    # The loan's "debt" is the measure's, and no member holds a loan beside commercial banks.
    "pcbk_bangladesh": {**_AO, "idsCreditorType": ["IDSCreditorTypePCBK"]},
    # "loans" is the measure's word. The finance type stays because members pair a health
    # purpose with a debt instrument.
    "health_loans": {
        "crsPurpose": ["CRSPurposeCode120"],
        "crsTypeOfFinance": ["CRSTypeOfFinance420"],
    },
    # Two typed values in words of their own stay though no member holds them together.
    # No measure-worded value joins them, so the engine refuses on what the user wrote.
    "public_sector_ppg": {"idsDebtorType": ["IDSDebtorTypePS"], **_PPG},
}

# Runs whose round-one drafts differ from the query's other runs. DPPG 2020's round-three
# run scores "publicly guaranteed" at 0.66 above `part` with "several" at 0.81, so the
# guarantee status is drafted as a set; the public-sector debtor still loses to its words.
_RUN_DRAFTS = {
    ("dppg_kenya_2020", 3): {
        **_AO,
        "guaranteeStatus": ["PublicOrPubliclyGuaranteed", "PubliclyGuaranteedClaim"],
    },
}

_RUNS = [
    pytest.param(name, k, id=f"{name}-run{k}")
    for name in _SPANNED
    for k in range(len(_record(name)["runs"]))
]


@pytest.mark.parametrize(("name", "k"), _RUNS)
def test_a_recorded_bind_keeps_the_values_the_phrase_pays_for(name: str, k: int):
    record = _record(name)

    named, _conf = replay(record, record["runs"][k])

    assert named == _RUN_DRAFTS.get((name, k), _SPANNED[name])


@pytest.mark.parametrize("k", range(3))
def test_the_typed_pair_drops_only_values_it_never_wrote(k: int):
    record = _record("public_sector_ppg")

    _named, conf = replay(record, record["runs"][k])

    assert conf.inferred == {
        "how:financialPositionType": ("AmountOutstanding",),
        "how:financingResource": ("DebtInstrument",),
    }


@pytest.mark.parametrize(
    "name", ["net_oda", "female_literacy_p0", "female_literacy_p1", "neonatal"]
)
def test_a_bind_naming_one_slot_asks_no_span_round_and_keeps_its_drafts(name: str):
    record = _record(name)
    (run,) = record["runs"]

    spans = span_round(recorded_menu(record), recorded_answers(run), stated_form=None)
    named, _conf = replay(record, run)

    assert spans.questions == {}
    assert named == _named(run["recorded_drafts"])


@pytest.mark.parametrize("name", list(_SPANNED))
def test_the_recorded_span_round_asks_what_the_rebuilt_menu_asks(name: str):
    """The rebuilt menu reads the recorded answers as the recorded bind did: its span round
    asks exactly the span questions each run answered."""
    record = _record(name)

    for run in record["runs"]:
        spans = span_round(recorded_menu(record), recorded_answers(run), stated_form=None)
        assert sorted(spans.questions) == sorted(q for q in run["choices"] if q.startswith("span|"))
