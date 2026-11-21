"""An open role's members split into countable parties and residual members
(`materialise.enrichment.party_split`), and every surface that counts the role states the same
split: the role axis fact, the `role_scope_defaulted` warning, coverage's open-role dimension,
a composite's scope and the heading. DAC2a's recipients of IDA hold 81 countries and 10
regional "unallocated" lines beside the totals over them, so each counts 81 recipients and
10 residual members, and the frame holds the rows of all 91."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from code_index.types import PartyRow, RoleAxis
from dfo_platform.roles import role_wording
from search.engine.region import grounding
from search.engine.retrieval import coverage
from search.engine.spec_build import RoleAxisFactInput, _role_axis_facts
from search.render.heading import _open_role_reading
from search.scope.composite import CompositeMint

_DAC2A = "provenance/OECDDAC2A"
_SV = "TESTFIXTURE/OpenRoleCounts/Flow"
_RECIPIENT: tuple[tuple[str, RoleAxis], ...] = ((_DAC2A, "destination"),)
_COUNTRIES = tuple(f"country/C{n:02d}" for n in range(81))
_RESIDUALS = tuple(f"DACRegion{n:02d}Unallocated" for n in range(10))
_TOTAL = "DACDevelopingCountries"
_PARTIES = {
    **{dcid: PartyRow(dcid=dcid, kind="party", published=_RECIPIENT) for dcid in _COUNTRIES},
    **{
        dcid: PartyRow(dcid=dcid, kind="residual", party_class="residual", published=_RECIPIENT)
        for dcid in _RESIDUALS
    },
    _TOTAL: PartyRow(dcid=_TOTAL, kind="total", published=_RECIPIENT),
}
_MEMBERS = frozenset({*_COUNTRIES, *_RESIDUALS, _TOTAL})


@pytest.fixture(autouse=True)
def _indexed(monkeypatch: pytest.MonkeyPatch) -> None:
    index = SimpleNamespace(parties=_PARTIES, statvar_provenances=lambda _svs: (_DAC2A,))
    for module in (grounding, coverage):
        monkeypatch.setattr(module, "get_index", lambda: index)


def test_every_surface_counts_81_recipients_and_10_residual_members() -> None:
    (fact,) = _role_axis_facts(
        [RoleAxisFactInput(role="destination", member_dcids=_MEMBERS)],
        parties=_PARTIES,
        datasets=(_DAC2A,),
    ) or [None]
    assert fact is not None
    warning_count = grounding._scope_party_count(
        SimpleNamespace(sv_dcids=[_SV]), _MEMBERS, role="destination"
    )
    coverage_count = coverage._scope_party_count(_MEMBERS, role="recipient", sv_dcids=[_SV])
    snapshot: Any = SimpleNamespace(
        _held_parties={"recipient": _MEMBERS, "donor": frozenset()},
        facts=SimpleNamespace(index=SimpleNamespace(parties=_PARTIES)),
        _datasets=[_DAC2A],
    )
    snapshot.party_split = lambda axis: CompositeMint.party_split(snapshot, axis)
    listed = CompositeMint.parties(snapshot, "destination")

    assert (fact.member_count, fact.residual_count) == (81, 10)
    split = snapshot.party_split("destination")
    assert (split.member_count, split.residual_count) == (81, 10)
    assert warning_count == coverage_count == 81
    # The scope lists what the frame holds: every member but the total.
    assert listed == [*sorted(_COUNTRIES), *sorted(_RESIDUALS)]
    assert len(listed or ()) == fact.member_count + fact.residual_count
    assert _open_role_reading(
        "destination",
        role_wording(None).noun("destination"),
        role_axis_facts=[fact],
        scope=None,
        coverage=None,
        states_all=False,
    ) == ("81 recipients and 10 unallocated amounts", [])


@pytest.mark.parametrize(
    ("members", "parties", "residuals", "reading"),
    [
        pytest.param(_COUNTRIES[:2], 2, None, "2 recipients", id="parties-alone"),
        pytest.param(
            (_COUNTRIES[0], *_RESIDUALS[:2]),
            1,
            2,
            "1 recipient and 2 unallocated amounts",
            id="one-party-beside-residuals",
        ),
        pytest.param(_RESIDUALS[:1], 0, 1, "1 unallocated amount", id="residual-alone"),
        # A side of totals alone counts them all, as the download keeps them.
        pytest.param((_TOTAL,), 1, None, None, id="totals-alone"),
    ],
)
def test_the_heading_names_the_residual_members_its_fact_counts(
    members: tuple[str, ...], parties: int, residuals: int | None, reading: str | None
) -> None:
    (fact,) = _role_axis_facts(
        [RoleAxisFactInput(role="destination", member_dcids=frozenset(members))],
        parties=_PARTIES,
        datasets=(_DAC2A,),
    ) or [None]
    assert fact is not None

    stated = _open_role_reading(
        "destination",
        role_wording(None).noun("destination"),
        role_axis_facts=[fact],
        scope=None,
        coverage=None,
        states_all=False,
    )

    assert (fact.member_count, fact.residual_count) == (parties, residuals)
    assert stated == (None if reading is None else (reading, []))
    # The wire leaves residual_count out where the side holds no residual member.
    assert ("residual_count" in fact.model_dump(mode="json")) == (residuals is not None)
