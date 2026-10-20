"""Golden replay of the Jev binder: stored Jev answers reproduce each item's expected output.

Each item (`_jev_golden`) holds the answers Jev gave on one dev item and the bind output,
quantity read and election rules the assembler builds from them under the house policies in
`binders/house_policy.py`. The port's requests ask exactly the questions those answers
answer, and its assembly gives that output.
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


def test_a_house_policy_keeps_the_loan_jev_recognised_on_its_crs_pick():
    """ "World Bank loans" to Tanzania, with Jev's support for DAC2a's non-grant finance at
    0.01 in place of the stored 0.74: DAC2a, the recipient default, then states none of the
    constraints Jev recognised on its CRS pick, so the CRS shape stays with its loan. With
    the stored answers DAC2a states the loan in its own taxonomy and the policy moves the
    part there (the golden item)."""
    item = load_item(GOLDEN / "synthetic-74b382efc5_p0.json")
    stored = item.answers
    answers = JevAnswers(
        nouls={**stored.nouls, "n|how:dacTypeOfFinance|1": 0.01}, choices=stored.choices
    )

    out, _conf = replay(item, answers=answers)

    assert out["bind"]["shape"] == 0
    assert ["how", "crsTypeOfFinance", "value", ["CRSTypeOfFinance420"], "mentioned"] in out[
        "bind"
    ]["bindings"]
