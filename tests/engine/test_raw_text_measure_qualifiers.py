"""A word narrowing one measure is judged on its own part, and applied when the part's answer
binds a value it names."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from search.engine.entry.raw_text import _binds_value_named, _held_by_one, _narrowing_words
from search.models import GraphRef


def _ref(dcid: str, label: str | None = None) -> GraphRef:
    """A bound value whose graph label is `label`, else its dcid, which names no word."""
    return GraphRef(ref=dcid, label=label or dcid)


def _spec(prop: str, *refs: GraphRef, origin: tuple[str, ...] = ()) -> SimpleNamespace:
    """The part of a Spec `_binds_value_named` reads: the values its trace binds per slot,
    and the places its identity binds on each role."""
    slot = SimpleNamespace(key=SimpleNamespace(property=_ref(prop)), refs=list(refs))
    identity = SimpleNamespace(entities={"origin": list(origin)} if origin else {})
    return SimpleNamespace(
        resolved=SimpleNamespace(trace=SimpleNamespace(slot_filters=[slot])),
        definition=SimpleNamespace(identity=identity),
    )


@pytest.mark.parametrize(
    ("phrase", "prop", "refs", "applied"),
    [
        ("bilateral", "dac1AidType", [_ref("DAC1AidType1015")], True),
        (
            "Bilateral",
            "dac2aAidType",
            [_ref("DAC2aAidType206"), _ref("DAC1AidType1015")],
            True,
        ),
        ("imputed multilateral", "dac2aAidType", [_ref("DAC2aAidType106")], True),
        (
            "bilateral",
            "idsCreditorType",
            [_ref("IDSCreditorTypeBLAT", "Bilateral")],
            True,
        ),
        ("bilateral", "dac2aAidType", [_ref("DAC2aAidType206")], False),
        # "Official non-concessional flow" is the opposite of what was asked.
        ("concessional", "dac2bAidType", [_ref("DAC2bAidType296")], False),
        ("multilateral", "dac1AidType", [_ref("DAC1AidType1015")], False),
        ("bilateral", "dac2aAidType", [], False),
    ],
)
def test_a_narrowing_word_is_applied_by_a_bound_value_that_names_it(phrase, prop, refs, applied):
    """The words must run whole in one name of a bound value, its graph label or a code's
    label or synonym: DAC2a "official development assistance" names no "bilateral", DAC1
    "bilateral ODA" no "multilateral", and DAC2b "official non-concessional flow" no
    "concessional"."""
    spec = _spec(prop, *refs)
    # A stand-in for the Spec, which carries far more than the slot filters read here.
    assert _binds_value_named(phrase, spec) is applied  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("phrase", "origin", "applied"),
    [
        ("multilateral", ("DACMultilaterals",), True),
        ("multilateral", ("country/FRA",), False),
        ("bilateral", ("DACMultilaterals",), False),
    ],
)
def test_a_narrowing_word_is_applied_by_the_party_a_dataset_declares_it_names(
    phrase, origin, applied
):
    """DAC2a declares that "multilateral" names `DACMultilaterals` as provider, so an answer
    from that provider applies the word."""
    spec = _spec("dac2aAidType", _ref("DAC2aAidType206"), origin=origin)
    # A stand-in for the Spec, which carries far more than the fields read here.
    assert _binds_value_named(phrase, spec) is applied  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("variable", "words"),
    [
        ("bilateral aid", ("bilateral",)),
        ("imputed multilateral aid", ("multilateral",)),
        ("IDA concessional flows", ("concessional",)),
        # A hyphenated word is one token, so it holds no "concessional".
        ("non-concessional flows", ()),
        ("aid", ()),
    ],
)
def test_a_variable_holds_its_narrowing_words_as_whole_words(variable, words):
    assert _narrowing_words(variable) == words


@pytest.mark.parametrize(
    ("phrase", "narrowing", "held"),
    [
        ("bilateral", [("bilateral",), ()], True),
        ("Bilateral", [(), ("bilateral",)], True),
        # Two variables hold it, so it stays with the request.
        ("multilateral", [("multilateral",), ("multilateral",)], False),
        ("across all flows", [("bilateral",), ()], False),
    ],
)
def test_a_dropped_narrowing_word_moves_to_the_one_variable_holding_it(phrase, narrowing, held):
    assert _held_by_one(phrase, narrowing) is held
