"""Check installed template select keys through `write_component` on a real frame.

This uses the installed catalogue release, not the sibling checkout fixture. `prices` is a unit
request field, and `health_oda` is absent from this frame's `stat_var` values. Those cases refuse
on their offending dimensions; assertions distinguish them from an unrecognised select key.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest

from assembler.catalogue import load_release
from assembler.errors import BindingInvalid
from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore
from assembler.pipeline import write_component
from assembler.testing import malaria_ethiopia_resolved_frame

_SELECT_MEMBERS = ("line_chart", "bar_chart", "table", "stat_card")
_NO_SELECT_MEMBERS = ("note_narration", "source_card")
_MEMBERS = (*_SELECT_MEMBERS, *_NO_SELECT_MEMBERS)

# Each tuple is (offending dimension, unrecognised frame key); None means success.
_WORKED_EXPECTATIONS: dict[str, tuple[str, bool] | None] = {
    "line_chart": ("prices", True),
    "bar_chart": ("recipient", False),
    "table": ("recipient", False),
    "stat_card": ("prices", True),
    "note_narration": None,
    "source_card": None,
}

_MINIMAL_EXPECTATIONS: dict[str, tuple[str, bool] | None] = {
    "line_chart": ("stat_var", False),
    "bar_chart": ("stat_var", False),
    "table": ("stat_var", False),
    "stat_card": ("stat_var", False),
    "note_narration": None,
    "source_card": None,
}

_UNRECOGNIZED_KEY_DETAIL = "is not a dimension this frame declares"


def _installed_catalogue_dir() -> Path:
    return Path(str(resources.files("dfo_spec_kit_components") / "catalogue"))


def _installed_worked_intent(member: str) -> dict[str, object]:
    path = _installed_catalogue_dir() / "fixtures" / member / "intent.json"
    return json.loads(path.read_text())


def _write(intent: dict[str, object]):
    frame_store = FakeFrameStore()
    frame_store.put(resolved_frame=malaria_ethiopia_resolved_frame())
    return write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=FakeRevisionStore(),
        release_registry=FakeReleaseRegistry(release=load_release()),
    )


def _assert_matches_expectation(
    intent: dict[str, object], expectation: tuple[str, bool] | None
) -> None:
    if expectation is None:
        _write(intent)
        return

    expected_dimension, expects_unrecognized_key = expectation
    with pytest.raises(BindingInvalid) as excinfo:
        _write(intent)
    refusal = excinfo.value
    assert refusal.dimension == expected_dimension
    is_unrecognized_key = _UNRECOGNIZED_KEY_DETAIL in str(refusal)
    assert is_unrecognized_key == expects_unrecognized_key


@pytest.mark.parametrize("member", _MEMBERS)
def test_worked_fixture_intent_binds_against_a_real_frame(member: str) -> None:
    intent = _installed_worked_intent(member)
    _assert_matches_expectation(intent, _WORKED_EXPECTATIONS[member])


@pytest.mark.parametrize("member", _MEMBERS)
def test_minimal_intent_binds_against_a_real_frame(member: str) -> None:
    release = load_release()
    intent = dict(release.minimal_intents[member])
    _assert_matches_expectation(intent, _MINIMAL_EXPECTATIONS[member])
