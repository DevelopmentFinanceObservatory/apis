"""Tests same-cluster matches and rejection of unrelated, partial, or identical readings."""

from __future__ import annotations

import pytest

from search.engine.region.equivalent_series import shared_cluster

_ELECTRICITY_ACCESS = (
    "worldBank/EG_ELC_ACCS_ZS",
    "undata/sdg/EG_ACS_ELEC",
)

_UNDER5_MORTALITY = (
    "MortalityRate_Person_Upto4Years_AsFractionOf_Count_BirthEvent_LiveBirth",
    "undata/sdg/SH_DYN_MORT.AGE--Y0T4",
    "worldBank/SH_DYN_MORT",
)

_REMITTANCES_SHARE_OF_GDP = (
    "Amount_Remittance_InwardRemittance_AsFractionOf_"
    "Amount_EconomicActivity_GrossDomesticProduction_Nominal",
    "undata/sdg/BX_TRF_PWKR",
)

_GDP_CURRENT_PRICES = (
    "Amount_EconomicActivity_GrossDomesticProduction_Nominal",
    "ValueAdded_EconomicProduction_IncludingProductTaxes_GrossOfFixedCapitalConsumption_Nominal",
)

_UNKNOWN_DCID = "worldBank/NOT_A_REGISTERED_TWIN"


@pytest.mark.parametrize("pair", [_ELECTRICITY_ACCESS, _GDP_CURRENT_PRICES], ids=["elec", "gdp"])
def test_same_cluster_pair_matches(pair: tuple[str, str]) -> None:
    a, b = pair
    match = shared_cluster([a], [b])
    assert match is not None
    assert match.members == set(pair)


@pytest.mark.parametrize("pair", [_ELECTRICITY_ACCESS, _GDP_CURRENT_PRICES], ids=["elec", "gdp"])
def test_same_cluster_pair_matches_reversed(pair: tuple[str, str]) -> None:
    a, b = pair
    match = shared_cluster([b], [a])
    assert match is not None
    assert match.members == set(pair)


def test_under5_mortality_triplet_every_pairing_matches() -> None:
    for i, elected in enumerate(_UNDER5_MORTALITY):
        for j, candidate in enumerate(_UNDER5_MORTALITY):
            if i == j:
                continue
            match = shared_cluster([elected], [candidate])
            assert match is not None, f"{elected!r} vs {candidate!r} should match"
            assert match.members == set(_UNDER5_MORTALITY)


def test_dcid_absent_from_every_cluster_does_not_match() -> None:
    assert shared_cluster([_UNKNOWN_DCID], [_ELECTRICITY_ACCESS[0]]) is None
    assert shared_cluster([_ELECTRICITY_ACCESS[0]], [_UNKNOWN_DCID]) is None
    assert shared_cluster([_UNKNOWN_DCID], [_UNKNOWN_DCID]) is None


@pytest.mark.parametrize(
    "dcids",
    [
        pytest.param((_ELECTRICITY_ACCESS[0],), id="single-dcid"),
        pytest.param(_UNDER5_MORTALITY[:2], id="multiple-dcids"),
    ],
)
def test_identical_dcid_sides_do_not_match(dcids: tuple[str, ...]) -> None:
    assert shared_cluster(list(dcids), list(dcids)) is None


def test_pair_from_two_different_clusters_does_not_match() -> None:
    assert shared_cluster([_ELECTRICITY_ACCESS[0]], [_UNDER5_MORTALITY[0]]) is None
    assert shared_cluster([_REMITTANCES_SHARE_OF_GDP[0]], [_ELECTRICITY_ACCESS[1]]) is None


def test_mixed_membership_shape_does_not_match() -> None:
    # A mixed side names a superset of the cluster reading and cannot match it.
    elected_dcids = [_UNDER5_MORTALITY[0], _UNKNOWN_DCID]
    candidate_dcids = [_UNDER5_MORTALITY[1]]
    match = shared_cluster(elected_dcids, candidate_dcids)
    assert match is None


def test_multi_dcid_side_matches_when_fully_contained() -> None:
    # A side naming several dcids still matches as long as every one of
    # them is a member of the same cluster, whatever the side's dcid count.
    elected_dcids = [_UNDER5_MORTALITY[0], _UNDER5_MORTALITY[1]]
    candidate_dcids = [_UNDER5_MORTALITY[2]]
    match = shared_cluster(elected_dcids, candidate_dcids)
    assert match is not None
    assert match.members == set(_UNDER5_MORTALITY)


def test_note_is_present_and_nonempty_for_every_cluster() -> None:
    for pair in (
        _ELECTRICITY_ACCESS,
        _UNDER5_MORTALITY,
        _REMITTANCES_SHARE_OF_GDP,
        _GDP_CURRENT_PRICES,
    ):
        match = shared_cluster([pair[0]], [pair[1]])
        assert match is not None
        assert match.note
