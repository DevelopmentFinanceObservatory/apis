"""Golden replay of the Jev binder: stored Jev answers reproduce the study's stored output.

Each item (`_jev_golden`) holds the answers Jev gave on one dev item and the output the
study's frozen assembler built from them. The port's requests ask exactly the questions
those answers answer, and its assembly gives the same bind output, quantity read and
election rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.engine._jev_golden import asked_ids, golden_paths, load_item, replay, stored_ids

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
    """The bind and quantity requests together ask every stored question but the ``topic``
    and ``within|i`` the release no longer asks, and only the quantity request asks the
    ``q|`` questions."""
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
