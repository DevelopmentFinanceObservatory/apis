"""Tests the capped list of values a refusal message names."""

from __future__ import annotations

import pytest

from dfo_platform.named import MAX_NAMED, MAX_VALUE_CHARS, cut, named_capped


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param([], "", id="none"),
        pytest.param(["a"], "a", id="one"),
        pytest.param(list("abcde"), "a, b, c, d, e", id="at-the-cap"),
        pytest.param(list("abcdefg"), "a, b, c, d, e and 2 more", id="past-the-cap"),
        pytest.param(iter("abcdef"), "a, b, c, d, e and 1 more", id="iterator"),
    ],
)
def test_names_the_first_values_and_counts_the_rest(values, expected: str) -> None:
    assert named_capped(values) == expected


def test_the_cap_is_five() -> None:
    assert MAX_NAMED == 5


def test_a_value_is_cut_only_past_max_chars() -> None:
    assert named_capped(["abcd", "abcde", "abcdef"], max_chars=5) == "abcd, abcde, abcde..."


def test_a_value_is_cut_at_the_shared_bound_by_default() -> None:
    at_the_bound, past_it = "a" * MAX_VALUE_CHARS, "b" * (MAX_VALUE_CHARS + 1)

    assert cut(at_the_bound) == at_the_bound
    assert cut(past_it) == f"{'b' * MAX_VALUE_CHARS}..."
    assert named_capped([at_the_bound, past_it]) == f"{at_the_bound}, {cut(past_it)}"


def _encodes(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("\ud83d", id="lone-surrogate"),
        pytest.param("a" * (MAX_VALUE_CHARS - 2) + "\ud83d\ude00", id="pair-split-by-the-bound"),
        pytest.param("\ud83d" * 1000, id="oversized"),
    ],
)
def test_cut_returns_text_that_encodes_as_utf8(value: str) -> None:
    assert _encodes(cut(value))
    assert _encodes(named_capped([value]))


def test_a_lone_surrogate_prints_as_its_escape() -> None:
    assert cut("\ud83d") == "\\ud83d"


def test_a_character_outside_the_bmp_is_kept_whole_at_the_bound() -> None:
    value = "a" * (MAX_VALUE_CHARS - 1) + "\U0001f600" + "b"

    assert cut(value) == "a" * (MAX_VALUE_CHARS - 1) + "\U0001f600..."
