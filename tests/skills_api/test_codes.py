"""Pins platform-code ownership and warning catalogue agreement with `skills_index`."""

from __future__ import annotations

import importlib
import sys
from typing import get_args

import pytest

from dfo_platform.diagnostics import WarningSeverity
from skills_api.codes import (
    SHARED_PLATFORM_CODES,
    SKILLS_CODE_CATALOGUE,
    SKILLS_WARNING_CATALOGUE,
)
from skills_index import SKILL_ISSUE_SEVERITY, SkillIssueCode

_WIRE_SEVERITY = {"rejection": "degraded", "warning": "warn"}


def test_shared_platform_codes_are_service_unavailable_and_request_invalid():
    assert frozenset({"service_unavailable", "request_invalid"}) == SHARED_PLATFORM_CODES


def test_import_time_assertion_fires_when_a_code_is_not_platform_owned(monkeypatch):
    """Reloads `skills_api.codes` with `dfo_platform.codes
    .PLATFORM_CODE_CATALOGUE` monkeypatched to a catalogue that carries
    neither `service_unavailable` nor `request_invalid`, and asserts the
    module's own import-time check actually raises rather than merely
    replicating the check's logic in the test.
    """
    import dfo_platform.codes as platform_codes

    monkeypatch.setattr(platform_codes, "PLATFORM_CODE_CATALOGUE", {})
    sys.modules.pop("skills_api.codes", None)
    try:
        with pytest.raises(AssertionError):
            importlib.import_module("skills_api.codes")
    finally:
        # Undo the monkeypatch before reimporting -- otherwise this
        # cleanup reimport would hit the very same assertion and its
        # exception would mask the one this test asserts on.
        monkeypatch.undo()
        sys.modules.pop("skills_api.codes", None)
        importlib.import_module("skills_api.codes")


def test_warning_catalogue_key_set_matches_skill_issue_code_exactly():
    """A code added to `skills_index.SkillIssueCode` and not to this
    catalogue (or vice versa) must fail here -- asserted against the
    `Literal`'s own `get_args`, never a hand-typed list that could drift
    from it independently.
    """
    assert frozenset(SKILLS_WARNING_CATALOGUE) == frozenset(get_args(SkillIssueCode))


@pytest.mark.parametrize("code", sorted(SKILLS_WARNING_CATALOGUE))
def test_every_warning_severity_is_a_legal_warning_severity(code: str):
    _title, _detail, severity = SKILLS_WARNING_CATALOGUE[code]
    assert severity in get_args(WarningSeverity)


def test_refusal_catalogue_carries_exactly_the_two_404s():
    assert frozenset(SKILLS_CODE_CATALOGUE) == frozenset(
        {"skill_not_found", "skill_resource_not_found"}
    )


@pytest.mark.parametrize("code", sorted(SKILLS_WARNING_CATALOGUE))
def test_warning_severity_matches_the_index_severity_mapping(code: str):
    """`SKILLS_WARNING_CATALOGUE`'s severity is read from `skills_index
    .SKILL_ISSUE_SEVERITY` -- the same mapping `SkillIssue.severity`
    derives from at every raise site in `skills_index.build`/
    `skills_index.expand` -- translated onto the wire vocabulary. This
    compares against that mapping directly rather than a hand-typed table
    that could drift from it independently.
    """
    _title, _detail, severity = SKILLS_WARNING_CATALOGUE[code]
    assert severity == _WIRE_SEVERITY[SKILL_ISSUE_SEVERITY[code]]
