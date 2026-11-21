"""An open role's members split into countable parties and residual members
(`materialise.enrichment.party_split`), and every surface that counts the role states the same
split: the role axis fact, the `role_scope_defaulted` warning, coverage's open-role dimension,
a composite's scope and the heading. DAC2a's recipients of IDA hold 81 countries and 10
regional "unallocated" lines beside the totals over them, so the frame holds 91 members, of
which 10 are residual, and the warning and the heading name 81 recipients and 10 unallocated
amounts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from code_index.types import PartyRow, RoleAxis
from dfo_platform.roles import GENERIC_ROLE_WORDING, ROLE_WORDING
from search.engine.region import grounding
from search.engine.retrieval import coverage
from search.engine.spec_build import RoleAxisFactInput, _role_axis_facts
from search.models import (
    BreadthDim,
    CoverageBreadth,
    GraphRef,
    ResolvedEntity,
    ResolvedScope,
    RoleAxisFact,
    SentenceFacts,
)
from search.render import heading
from search.render.heading import _open_role_reading
from search.scope.composite import CompositeMint

_DAC2A = "provenance/OECDDAC2A"
_IDS = "provenance/WorldBankIDS"
_SV = "TESTFIXTURE/OpenRoleCounts/Flow"
_RECIPIENT: tuple[tuple[str, RoleAxis], ...] = ((_DAC2A, "destination"),)
_CREDITOR: tuple[tuple[str, RoleAxis], ...] = ((_IDS, "origin"),)
_COUNTRIES = tuple(f"country/C{n:02d}" for n in range(81))
_RESIDUALS = tuple(f"DACRegion{n:02d}Unallocated" for n in range(10))
_TOTAL = "DACDevelopingCountries"
_CREDITORS = tuple(f"country/K{n:02d}" for n in range(40))
_UNIDENTIFIED = "IDSUnidentifiedCreditors"
_PARTIES = {
    **{dcid: PartyRow(dcid=dcid, kind="party", published=_RECIPIENT) for dcid in _COUNTRIES},
    **{
        dcid: PartyRow(dcid=dcid, kind="residual", party_class="residual", published=_RECIPIENT)
        for dcid in _RESIDUALS
    },
    _TOTAL: PartyRow(dcid=_TOTAL, kind="total", published=_RECIPIENT),
    **{dcid: PartyRow(dcid=dcid, kind="party", published=_CREDITOR) for dcid in _CREDITORS},
    _UNIDENTIFIED: PartyRow(
        dcid=_UNIDENTIFIED, kind="residual", party_class="residual", published=_CREDITOR
    ),
}
_MEMBERS = frozenset({*_COUNTRIES, *_RESIDUALS, _TOTAL})
_AID = ROLE_WORDING["aid"]
_DEBT = ROLE_WORDING["debt_lending"]


@pytest.fixture(autouse=True)
def _indexed(monkeypatch: pytest.MonkeyPatch) -> None:
    index = SimpleNamespace(parties=_PARTIES, statvar_provenances=lambda _svs: (_DAC2A, _IDS))
    for module in (grounding, coverage, heading):
        monkeypatch.setattr(module, "get_index", lambda: index)


def _fact(axis: RoleAxis, members: tuple[str, ...] | frozenset[str]) -> RoleAxisFact:
    (fact,) = _role_axis_facts(
        [RoleAxisFactInput(role=axis, member_dcids=frozenset(members))],
        parties=_PARTIES,
        datasets=(_DAC2A, _IDS),
    ) or [None]
    assert fact is not None
    return fact


def _warning(axis: RoleAxis, members: tuple[str, ...] | frozenset[str]) -> str:
    mat: Any = SimpleNamespace(
        sv_dcids=[_SV],
        discovered_scope=SimpleNamespace(axis=axis),
        scope_member_dcids=sorted(members),
        also_defaulted_axis=None,
        open_origin_dcids=None,
    )
    (warning,) = grounding.role_scope_defaulted_warnings(
        mat, boundary_roles={}, label_of=lambda dcids: list(dcids)
    )
    return warning.message


def test_every_surface_counts_91_members_of_which_10_are_residual() -> None:
    fact = _fact("destination", _MEMBERS)
    dim = coverage._open_role_dim("recipients", _MEMBERS, role="recipient", sv_dcids=[_SV])
    snapshot: Any = SimpleNamespace(
        _held_parties={"recipient": _MEMBERS, "donor": frozenset()},
        facts=SimpleNamespace(index=SimpleNamespace(parties=_PARTIES)),
        _datasets=[_DAC2A],
    )
    snapshot.party_split = lambda axis: CompositeMint.party_split(snapshot, axis)
    listed = CompositeMint.parties(snapshot, "destination")
    split = snapshot.party_split("destination")

    assert (fact.member_count, fact.residual_count) == (91, 10)
    assert (split.member_count, split.party_count, split.residual_count) == (91, 81, 10)
    assert (dim.count, dim.residual_count) == (91, 10)
    # The scope lists what the frame holds, every member but the total, sorted by ref.
    assert listed == sorted((*_COUNTRIES, *_RESIDUALS))
    assert len(listed or ()) == fact.member_count
    assert "covers all 81 recipients and 10 unallocated amounts with data" in _warning(
        "destination", _MEMBERS
    )
    assert _open_role_reading(
        "destination",
        _AID.noun("destination"),
        role_axis_facts=[fact],
        scope=None,
        coverage=None,
        states_all=False,
    ) == ("81 recipients and 10 unallocated amounts", [])


@pytest.mark.parametrize(
    ("axis", "members", "counts", "reading", "warned"),
    [
        pytest.param(
            "destination",
            _COUNTRIES[:2],
            (2, None),
            "2 recipients",
            "both recipients",
            id="recipients-alone",
        ),
        pytest.param(
            "destination",
            (_COUNTRIES[0], *_RESIDUALS[:2]),
            (3, 2),
            "1 recipient and 2 unallocated amounts",
            "the one recipient and 2 unallocated amounts",
            id="one-recipient-beside-residuals",
        ),
        pytest.param(
            "destination",
            _RESIDUALS[:1],
            (1, 1),
            "1 unallocated amount",
            "1 unallocated amount",
            id="recipient-residual-alone",
        ),
        pytest.param(
            "destination",
            (_TOTAL, *_RESIDUALS[:2]),
            (2, 2),
            "2 unallocated amounts",
            "2 unallocated amounts",
            id="recipient-residuals-beside-a-total",
        ),
        # A side of totals alone counts them all, as the download keeps them.
        pytest.param(
            "destination", (_TOTAL,), (1, None), None, "the one recipient", id="totals-alone"
        ),
        pytest.param(
            "origin",
            (*_CREDITORS, _UNIDENTIFIED),
            (41, 1),
            "40 creditors and 1 unspecified creditor",
            "all 40 providers and 1 unspecified provider",
            id="creditors-beside-a-residual",
        ),
        pytest.param(
            "origin",
            (_UNIDENTIFIED,),
            (1, 1),
            "1 unspecified creditor",
            "1 unspecified provider",
            id="creditor-residual-alone",
        ),
    ],
)
def test_the_heading_and_warning_name_residual_members_by_side(
    axis: RoleAxis,
    members: tuple[str, ...],
    counts: tuple[int, int | None],
    reading: str | None,
    warned: str,
) -> None:
    fact = _fact(axis, members)

    stated = _open_role_reading(
        axis,
        _DEBT.noun(axis) if axis == "origin" else _AID.noun(axis),
        role_axis_facts=[fact],
        scope=None,
        coverage=None,
        states_all=False,
    )

    assert (fact.member_count, fact.residual_count) == counts
    assert stated == (None if reading is None else (reading, []))
    assert f"covers {warned} with data" in _warning(axis, members)
    # The wire leaves residual_count out where the side holds no residual member.
    assert ("residual_count" in fact.model_dump(mode="json")) == (counts[1] is not None)


def test_a_heading_with_no_fact_splits_its_scope_as_the_fact_does() -> None:
    """The scope counts its members through the same split, so its total is no party and
    its residual members are named apart."""
    scope = ResolvedScope(
        axis="destination",
        members=[
            ResolvedEntity(ref=dcid, label=dcid, role="destination")
            for dcid in (*_COUNTRIES[:3], *_RESIDUALS[:2], _TOTAL)
        ],
    )

    stated = _open_role_reading(
        "destination",
        GENERIC_ROLE_WORDING.noun("destination"),
        role_axis_facts=[],
        scope=scope,
        coverage=None,
        states_all=False,
        datasets=[_DAC2A],
    )

    assert stated == ("3 recipients and 2 unallocated amounts", [])


def _scope(*dcids: str) -> ResolvedScope:
    return ResolvedScope(
        axis="destination",
        members=[ResolvedEntity(ref=dcid, label=dcid, role="destination") for dcid in dcids],
    )


@pytest.mark.parametrize("with_fact", [False, True], ids=["scope-alone", "beside-its-fact"])
def test_a_heading_names_the_one_party_its_scope_holds_beside_a_total(with_fact: bool) -> None:
    members = (_COUNTRIES[0], _TOTAL)

    stated = _open_role_reading(
        "destination",
        _AID.noun("destination"),
        role_axis_facts=[_fact("destination", members)] if with_fact else [],
        scope=_scope(*members),
        coverage=None,
        states_all=False,
        datasets=[_DAC2A],
    )

    assert stated is not None
    assert [entity.dcid for entity in stated[1]] == [_COUNTRIES[0]]


def test_a_heading_counted_by_coverage_names_its_residual_members_apart() -> None:
    dim = coverage._open_role_dim(
        "recipients", (*_COUNTRIES[:3], *_RESIDUALS[:2], _TOTAL), role="recipient", sv_dcids=[_SV]
    )

    stated = _open_role_reading(
        "destination",
        _AID.noun("destination"),
        role_axis_facts=[],
        scope=None,
        coverage=CoverageBreadth(has_data=True, dimensions=[dim]),
        states_all=False,
    )

    assert dim == BreadthDim(label="recipients", count=5, residual_count=2)
    assert stated == ("3 recipients and 2 unallocated amounts", [])


def test_a_heading_names_the_one_party_its_scope_holds_beside_residual_members() -> None:
    members = (_COUNTRIES[0], *_RESIDUALS[:2], _TOTAL)

    stated = _open_role_reading(
        "destination",
        _AID.noun("destination"),
        role_axis_facts=[_fact("destination", members)],
        scope=_scope(*members),
        coverage=None,
        states_all=False,
        datasets=[_DAC2A],
    )

    assert stated is not None
    assert stated[0].endswith(" and 2 unallocated amounts")
    assert [entity.dcid for entity in stated[1]] == [_COUNTRIES[0]]


def test_a_heading_splits_its_scope_in_the_datasets_the_answer_serves() -> None:
    """A party that is a total only in a dataset the answer does not serve is a party."""
    elsewhere = "TESTFIXTURE/ElsewhereTotal"
    parties = {
        **_PARTIES,
        elsewhere: PartyRow(
            dcid=elsewhere,
            kind="party",
            published=((_DAC2A, "destination"), (_IDS, "destination")),
            dataset_kinds=((_IDS, "total"),),
        ),
    }
    index = SimpleNamespace(parties=parties)
    members = (*_COUNTRIES[:2], elsewhere)

    def reading(datasets: list[str]) -> tuple[str, list[Any]] | None:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(heading, "get_index", lambda: index)
            return _open_role_reading(
                "destination",
                _AID.noun("destination"),
                role_axis_facts=[],
                scope=_scope(*members),
                coverage=None,
                states_all=False,
                datasets=datasets,
            )

    assert reading([_DAC2A]) == ("3 recipients", [])
    assert reading([_IDS]) == ("2 recipients", [])


def test_a_heading_names_no_lone_party_its_role_axis_fact_does_not_count() -> None:
    """The scope names one party only where the role's fact, split in the served StatVars'
    datasets, counts one party too; otherwise the fact counts the role."""
    stated = _open_role_reading(
        "destination",
        _AID.noun("destination"),
        role_axis_facts=[_fact("destination", (*_COUNTRIES[:2], _TOTAL))],
        scope=_scope(_COUNTRIES[0], _TOTAL),
        coverage=None,
        states_all=False,
        datasets=[_DAC2A],
    )

    assert stated == ("2 recipients", [])


def test_a_heading_splits_its_scope_in_the_datasets_of_the_served_stat_vars() -> None:
    """The heading reads the datasets the role axis facts are split in, the provenances of
    the served StatVars, whatever sources the byline cites."""
    facts = SentenceFacts(members=[GraphRef(ref=_SV, label="Flow")], constraint_properties=[])

    assert heading._served_datasets(facts, ()) == (_DAC2A, _IDS)
    assert heading._served_datasets(None, [_SV]) == (_DAC2A, _IDS)
