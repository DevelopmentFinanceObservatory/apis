"""`destination_side` places a DAC2a destination on the bilateral or multilateral side of a
donor's aid from its graph types, with the classifier Search uses for donors
(`code_index.multilaterals`)."""

from __future__ import annotations

import pytest

from code_index.declarations import DAC2A, DAC_RECIPIENT_LIST_DATASETS
from code_index.query import CodeIndex
from code_index.scope_conventions import (
    DestinationSide,
    bilateral_side_total,
    destination_side,
    lists_bilateral_destinations,
    reads_dac_recipient_list,
)
from code_index.types import IndexStamp, PartyKind, PartyRow
from tests.code_index.support import INDEX_FIXTURE, load_index

_DATASETS = (DAC2A,)


def _row(
    dcid: str, kind: PartyKind, types: tuple[str, ...] = (), residuals: tuple[str, ...] = ()
) -> PartyRow:
    return PartyRow(
        dcid=dcid,
        kind=kind,
        party_class={"party": "official", "residual": "residual", "total": None}[kind],
        residuals=residuals,
        published=((DAC2A, "destination"),),
        types=types,
    )


# The totals that list residuals: Asia counts the Mekong Delta Project, which the graph types
# as an unspecified party, beside its unallocated line; a new region counts a new project the
# graph types alike; an agency family typed as an organisation counts its own project.
_TOTALS = (
    _row(
        "DACAsia",
        "total",
        ("DACGroup",),
        ("DACAsiaUnallocated", "DACMekongDeltaProject"),
    ),
    _row("DACSomeNewRegion", "total", ("DACGroup",), ("DACSomeNewProject",)),
    _row("SomeAgencyGroup", "total", ("Organization",), ("DACSomeAgencyProject",)),
)
_INDEX = CodeIndex(
    stamp=IndexStamp(
        graph_endpoint="https://fake.example",
        exported_at="2026-09-15T00:00:00Z",
        code_counts={},
        statvar_shape_counts={},
        content_hash="",
    ),
    declarations=(),
    codes=(),
    statvars=(),
    parties=_TOTALS,
)


def _side(row: PartyRow) -> DestinationSide | None:
    return destination_side(row, datasets=_DATASETS, index=_INDEX)


@pytest.mark.parametrize(
    ("dcid", "kind", "types", "side"),
    [
        ("EuropeanCommission", "party", ("Organization",), "multilateral"),
        ("InternationalDevelopmentAssociation", "party", ("Organization",), "multilateral"),
        ("DACOtherMultilateralRecipient", "residual", ("DACUnspecifiedParty",), "multilateral"),
        ("DACOtherUnitedNations", "residual", ("DACUnspecifiedParty",), "multilateral"),
        ("DACOtherRegionalDevelopmentBanks", "residual", ("DACUnspecifiedParty",), "multilateral"),
        ("country/SEN", "party", ("Country",), "bilateral"),
        ("DACAfricaUnallocated", "residual", ("DACRecipientUnallocatedByCountry",), "bilateral"),
        (
            "DACBilateralUnallocated",
            "residual",
            ("DACRecipientUnallocatedByCountry",),
            "bilateral",
        ),
        ("DACMekongDeltaProject", "residual", ("DACUnspecifiedParty",), "bilateral"),
        ("EastAfricanCommunity", "party", ("Organization",), "bilateral"),
        ("IndusBasinDevelopmentFund", "party", ("Organization",), "bilateral"),
        ("DACAllRecipients", "total", ("DACGroup",), "spanning"),
        ("DACAllMultilateralRecipients", "total", ("DACGroup",), "multilateral"),
        ("DACDevelopingCountries", "total", ("DACGroup",), "bilateral"),
        ("DACAfrica", "total", ("DACGroup",), "bilateral"),
        pytest.param(
            "SomeNewTerritory",
            "party",
            ("AdministrativeArea1", "Country"),
            "bilateral",
            id="a-country-whose-dcid-has-no-country-prefix",
        ),
        pytest.param(
            "SomeNewAgency",
            "party",
            ("Organization",),
            "multilateral",
            id="an-organisation-typed-party",
        ),
        pytest.param(
            "DACSomeNewGroup", "total", ("DACGroup",), "bilateral", id="an-unlisted-dac-group"
        ),
        pytest.param(
            "SomeNewAgencyGroup",
            "total",
            ("Organization",),
            "multilateral",
            id="an-agency-family-total",
        ),
        pytest.param(
            "DACSomeOtherAgencies",
            "residual",
            ("DACUnspecifiedParty",),
            "multilateral",
            id="an-unlisted-unspecified-party",
        ),
        pytest.param(
            "DACSomeRegionUnallocated",
            "residual",
            ("DACRecipientUnallocatedByCountry",),
            "bilateral",
            id="an-unlisted-unallocated-line",
        ),
        pytest.param(
            "DACSomeNewProject",
            "residual",
            ("DACUnspecifiedParty",),
            "bilateral",
            id="an-unspecified-party-a-bilateral-total-lists",
        ),
        pytest.param(
            "DACSomeAgencyProject",
            "residual",
            ("DACUnspecifiedParty",),
            "multilateral",
            id="an-unspecified-party-a-multilateral-total-lists",
        ),
    ],
)
def test_a_dac2a_destination_is_placed_on_its_side(
    dcid: str, kind: PartyKind, types: tuple[str, ...], side: DestinationSide
):
    assert _side(_row(dcid, kind, types)) == side


def test_an_unknown_party_that_is_no_country_is_unclassified():
    """A party the graph types neither a country nor an organisation is a drift the mint
    refuses."""
    assert _side(_row("SomeNewAgency", "party")) is None


def test_a_residual_of_no_known_type_is_unclassified():
    """A residual no bilateral total lists and whose graph type names no side could count
    multilateral agencies, so the classifier leaves it unplaced and the mint refuses it."""
    assert _side(_row("DACSomeNewUnallocated", "residual")) is None


def test_a_party_no_dataset_publishes_as_a_recipient_is_unclassified():
    row = PartyRow(dcid="country/SEN", kind="party", published=((DAC2A, "origin"),))

    assert _side(row) is None


def test_every_dac2a_destination_of_the_recorded_index_is_classified():
    """The recorded residuals keep their sides: the five unspecified parties that stand for the
    multilaterals read multilateral, and the Mekong Delta Project, an unspecified party Asia
    counts, reads bilateral through the totals that list it."""
    index = load_index(INDEX_FIXTURE)
    destinations = [
        row for row in index.parties.values() if row.publishes("destination", _DATASETS)
    ]
    sides = {
        row.dcid: destination_side(row, datasets=_DATASETS, index=index) for row in destinations
    }

    assert [dcid for dcid, side in sides.items() if side is None] == []
    assert len(destinations) == 279
    assert sorted(
        row.dcid
        for row in destinations
        if row.kind_over("destination", _DATASETS) == "residual"
        and sides[row.dcid] == "multilateral"
    ) == [
        "DACOtherEUInstitutions",
        "DACOtherMultilateralRecipient",
        "DACOtherRegionalDevelopmentBanks",
        "DACOtherUnitedNations",
        "DACOtherWorldBank",
    ]
    assert sides["DACMekongDeltaProject"] == "bilateral"


_CRS = "provenance/OECDCreditorReportingSystem"
_IMPUTED = "provenance/ONEImputedMultilateralSectors"
_DAC1 = "provenance/OECDDAC1"
_IDS = "provenance/WorldBankInternationalDebtStatistics"
_BTR = "provenance/UNFCCCBiennialTransparencyReports"


@pytest.mark.parametrize(
    ("datasets", "reads", "lists"),
    [
        pytest.param((_CRS,), True, True, id="crs"),
        pytest.param((_IMPUTED,), True, True, id="imputed"),
        pytest.param((DAC2A,), False, True, id="dac2a-bilateral-composite"),
        pytest.param((_CRS, DAC2A), False, True, id="crs-and-dac2a"),
        pytest.param((_DAC1,), False, False, id="dac1"),
        pytest.param((_IDS,), False, False, id="ids"),
        pytest.param((_BTR,), False, False, id="unfccc-reports"),
        pytest.param((), False, False, id="no-dataset"),
    ],
)
def test_a_dataset_reads_and_lists_the_dac_recipient_list_where_declared(
    datasets: tuple[str, ...], reads: bool, lists: bool
):
    """The CRS and the imputed table hold the DAC's recipient list; DAC2a lists the agencies
    too, so it reads the developing countries only through its bilateral composite. DAC1
    names the one recipient it publishes, the World Bank's debt statistics its debtors and
    the UNFCCC reports the Convention's developing country Parties, so none reads or lists
    the DAC's developing countries, though every recipient each holds is a country."""
    assert reads_dac_recipient_list(datasets) is reads
    assert lists_bilateral_destinations(datasets, total_kept=False) is lists
    # Total ODA kept on DAC2a lists the agencies; the recipient list holds either way.
    assert lists_bilateral_destinations(datasets, total_kept=True) is (lists and reads)


def test_every_dataset_declared_to_hold_the_dac_recipient_list_holds_bilateral_recipients():
    """A dataset `DAC_RECIPIENT_LIST_DATASETS` names lists no recipient off the bilateral
    side in the recorded index."""
    index = load_index(INDEX_FIXTURE)
    off_side = {
        (dataset, row.dcid)
        for row in index.parties.values()
        for dataset in DAC_RECIPIENT_LIST_DATASETS
        if row.publishes("destination", (dataset,))
        and row.kind_over("destination", (dataset,)) != "total"
        and destination_side(row, datasets=(dataset,), index=index) != "bilateral"
    }
    held = [
        dataset
        for dataset in DAC_RECIPIENT_LIST_DATASETS
        if any(row.publishes("destination", (dataset,)) for row in index.parties.values())
    ]

    assert off_side == set()
    assert held


def test_the_developing_countries_total_is_the_one_bilateral_side_total():
    """`DACDevelopingCountries` totals the bilateral destinations; the total of both sides
    and a region's total do not."""
    index = load_index(INDEX_FIXTURE)

    assert [dcid for dcid in index.parties if bilateral_side_total(dcid)] == [
        "DACDevelopingCountries"
    ]
