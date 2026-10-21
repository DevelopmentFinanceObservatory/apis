"""Golden replay of the Jev binder: stored Jev answers give each item's expected output.

Each item (`_jev_golden`) holds the answers Jev gave on one dev item and the bind output,
quantity read and election rules the assembler builds from them under the house policies in
`binders/house_policy.py`. Each expected output is the assembler's own, judged against the
item's agreed label of the evaluation set, so a change to an expectation is a judged change
of behaviour. The requests ask exactly the questions those answers answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from search.engine.binders.jev_assemble import JevAnswers
from tests.engine._jev_golden import (
    GOLDEN,
    asked_ids,
    golden_paths,
    load_item,
    replay,
    stored_ids,
)

_PATHS = golden_paths()


def test_the_golden_set_is_present():
    assert len(_PATHS) == 36


@pytest.mark.parametrize("path", _PATHS, ids=lambda p: p.stem)
def test_stored_answers_give_the_stored_output(path: Path):
    item = load_item(path)

    out, _conf = replay(item)

    assert out == item.expected


@pytest.mark.parametrize("path", _PATHS, ids=lambda p: p.stem)
def test_the_two_requests_ask_the_stored_questions(path: Path):
    """The bind and quantity requests together ask every stored question except ``topic``
    and ``within|i``, and only the quantity request asks the ``q|`` questions."""
    item = load_item(path)

    bind, quantity = asked_ids(item)

    assert sorted(bind + quantity) == stored_ids(item)
    assert not [q for q in bind if q.startswith("q|")]
    assert quantity and all(q.startswith("q|") for q in quantity)


@pytest.mark.parametrize("path", _PATHS, ids=lambda p: p.stem)
def test_the_house_policies_run_once_on_an_election_and_never_on_an_ask(path: Path):
    item = load_item(path)

    out, _conf = replay(item)

    assert item.policies.calls == (0 if out["bind"]["ask"] else 1)


def test_a_house_policy_moves_the_loan_to_the_taxonomy_that_states_it():
    """ "World Bank loans" to Tanzania, with Jev's support for the CRS purpose "action
    relating to debt" at 0.01 in place of the stored 0.67, and no ``none`` share in
    ``electq`` in place of the stored 0.95, under which the part asks (the golden item):
    the loan is then the one constraint Jev recognised on its CRS pick, DAC2a, the
    recipient default, states it as non-grant finance in its own taxonomy, and the policy
    moves the part there."""
    item = load_item(GOLDEN / "synthetic-74b382efc5_p0.json")
    stored = item.answers
    electq = {k: p for k, p in stored.choices["electq"].items() if k != "none"}
    answers = JevAnswers(
        nouls={**stored.nouls, "n|how:crsPurpose|8": 0.01},
        choices={**stored.choices, "electq": electq},
    )

    out, _conf = replay(item, answers=answers)

    assert out["bind"]["shape"] == 1
    assert out["rules"] == ["house: default_dataset"]
    assert ["how", "dacTypeOfFinance", "value", ["DACTypeOfFinanceNonGrants"], "mentioned"] in out[
        "bind"
    ]["bindings"]
