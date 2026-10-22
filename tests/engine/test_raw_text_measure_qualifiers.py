"""A word narrowing one measure is applied when the part's answer binds a value it names."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from search.engine.entry.raw_text import _binds_value_named
from search.models import GraphRef


def _ref(dcid: str, label: str | None = None) -> GraphRef:
    """A bound value whose graph label is `label`, else its dcid, which names no word."""
    return GraphRef(ref=dcid, label=label or dcid)


def _spec(prop: str, *refs: GraphRef) -> SimpleNamespace:
    """The part of a Spec `_binds_value_named` reads: the values its trace binds per slot."""
    slot = SimpleNamespace(key=SimpleNamespace(property=_ref(prop)), refs=list(refs))
    return SimpleNamespace(resolved=SimpleNamespace(trace=SimpleNamespace(slot_filters=[slot])))


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
        ("multilateral", "dac1AidType", [_ref("DAC1AidType1015")], False),
        ("bilateral", "dac2aAidType", [], False),
    ],
)
def test_a_narrowing_word_is_applied_by_a_bound_value_that_names_it(phrase, prop, refs, applied):
    """Every word must sit in one name of a bound value, its graph label or a code's label
    or synonym: DAC2a "official development assistance" names no "bilateral", and DAC1
    "bilateral ODA" no "multilateral"."""
    spec = _spec(prop, *refs)
    # A stand-in for the Spec, which carries far more than the slot filters read here.
    assert _binds_value_named(phrase, spec) is applied  # ty: ignore[invalid-argument-type]
