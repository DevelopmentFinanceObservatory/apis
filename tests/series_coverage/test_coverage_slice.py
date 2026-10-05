"""Checks non-stub SDMX observations against slice counts and year bounds.
LATEST-only stubs are excluded.

Every recorded `sdmx_obs.json` entry is currently a stub, so the test passes vacuously until a
full-history entry is recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from series_coverage.loader import load_coverage

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_SLICE_DIR = _FIXTURES_DIR / "coverage_slice"
_SDMX_OBS_PATH = _FIXTURES_DIR / "sdmx_obs.json"


def _parse_key(key: str) -> tuple[str, dict[str, str]]:
    """`"variable|role1=dcid1,role2=dcid2"` -> `(variable, {role: dcid})`."""
    variable, _, roles_part = key.partition("|")
    roles: dict[str, str] = {}
    if roles_part:
        for segment in roles_part.split(","):
            role, _, dcid = segment.partition("=")
            roles[role] = dcid
    return variable, roles


def _non_stub_full_history_entries() -> list[tuple[str, dict[str, str], dict]]:
    payload = json.loads(_SDMX_OBS_PATH.read_text(encoding="utf-8"))
    entries: list[tuple[str, dict[str, str], dict]] = []
    for key, records in payload.items():
        if key == "_comment":
            continue
        variable, roles = _parse_key(key)
        entries.extend(
            (variable, roles, record) for record in records if record.get("stub") is not True
        )
    return entries


@pytest.mark.parametrize("variable,roles,record", _non_stub_full_history_entries())
def test_slice_agrees_with_full_history_sdmx_obs_entry(variable, roles, record):
    loaded = load_coverage(f"file://{_SLICE_DIR}")
    names = loaded.index.role_names.get(variable)
    assert names is not None, f"{variable!r} has an sdmx_obs.json entry but no coverage_slice rows"
    donor_dcid = roles.get(names.donor)
    recipient_dcid = roles.get(names.recipient)
    assert donor_dcid and recipient_dcid, f"key roles {roles!r} don't cover {names!r}"

    matches = loaded.index.filter_candidates(
        variables=[variable],
        bound={"donor": frozenset({donor_dcid}), "recipient": frozenset({recipient_dcid})},
    )
    assert matches, (
        f"coverage_slice has no row for {variable!r} donor={donor_dcid!r} recipient={recipient_dcid!r}, "
        f"but sdmx_obs.json has a non-stub entry for it"
    )
    coverage = matches[0]
    assert coverage.n_obs == record["obsCount"]
    assert coverage.first_year == int(record["earliestDate"][:4])
    assert coverage.last_year == int(record["latestDate"][:4])


def test_every_sdmx_obs_entry_is_currently_a_stub():
    """Documents the current state (see the module docstring): once this
    starts failing, a non-stub entry has landed and
    `test_slice_agrees_with_full_history_sdmx_obs_entry` is now doing real
    work, not passing vacuously over an empty parametrize list.
    """
    assert _non_stub_full_history_entries() == []
