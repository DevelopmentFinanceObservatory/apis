"""Pin ordered flattening and deduplication of value groups."""

from __future__ import annotations

from code_index import dedupe_first_seen


def test_flattens_preserving_first_seen_order():
    assert dedupe_first_seen([["a", "b"], ["c"]]) == ("a", "b", "c")


def test_drops_a_repeat_wherever_it_recurs():
    # `b` appears in two groups and stays at its first global position.
    assert dedupe_first_seen([["a", "b"], ["b", "c"], ["a"]]) == ("a", "b", "c")


def test_empty_input_is_an_empty_tuple():
    assert dedupe_first_seen([]) == ()


def test_a_group_that_is_itself_empty_contributes_nothing():
    assert dedupe_first_seen([[], ["a"], []]) == ("a",)
