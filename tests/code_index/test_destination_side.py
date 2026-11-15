"""`destination_side` places a DAC2a destination on the bilateral or multilateral side of a
donor's aid, with the classifier Search uses for donors (`code_index.multilaterals`)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from code_index.declarations import DAC2A
from code_index.multilaterals import organisation_dcids
from code_index.scope_conventions import DestinationSide, destination_side
from code_index.types import PartyKind, PartyRow
from entity_resolution import pack
from tests.code_index.support import INDEX_FIXTURE, load_index

_DATASETS = (DAC2A,)


def _row(dcid: str, kind: PartyKind) -> PartyRow:
    return PartyRow(
        dcid=dcid,
        kind=kind,
        party_class={"party": "official", "residual": "residual", "total": None}[kind],
        published=((DAC2A, "destination"),),
    )


@pytest.mark.parametrize(
    ("dcid", "kind", "side"),
    [
        ("EuropeanCommission", "party", "multilateral"),
        ("InternationalDevelopmentAssociation", "party", "multilateral"),
        ("DACOtherMultilateralRecipient", "residual", "multilateral"),
        ("DACOtherUnitedNations", "residual", "multilateral"),
        ("DACOtherRegionalDevelopmentBanks", "residual", "multilateral"),
        ("country/SEN", "party", "bilateral"),
        ("DACAfricaUnallocated", "residual", "bilateral"),
        ("DACBilateralUnallocated", "residual", "bilateral"),
        ("DACMekongDeltaProject", "residual", "bilateral"),
        ("EastAfricanCommunity", "party", "bilateral"),
        ("IndusBasinDevelopmentFund", "party", "bilateral"),
        ("DACAllRecipients", "total", "spanning"),
        ("DACAllMultilateralRecipients", "total", "multilateral"),
        ("DACDevelopingCountries", "total", "bilateral"),
        ("DACAfrica", "total", "bilateral"),
    ],
)
def test_a_dac2a_destination_is_placed_on_its_side(
    dcid: str, kind: PartyKind, side: DestinationSide
):
    assert destination_side(_row(dcid, kind), datasets=_DATASETS) == side


def test_an_unknown_party_that_is_no_country_is_unclassified():
    """A party no organisation record or country prefix names is a drift the mint refuses,
    not a bilateral recipient."""
    assert destination_side(_row("SomeNewAgency", "party"), datasets=_DATASETS) is None


@pytest.mark.parametrize(
    ("dcid", "kind"),
    [
        pytest.param("DACSomeNewGroup", "total", id="an-undeclared-total"),
        pytest.param("DACSomeNewUnallocated", "residual", id="an-undeclared-residual"),
    ],
)
def test_a_total_or_residual_no_declaration_places_is_unclassified(dcid: str, kind: PartyKind):
    """A total or residual could count multilateral agencies, so the classifier places one only
    where it is declared on a side, and the mint refuses any other."""
    assert destination_side(_row(dcid, kind), datasets=_DATASETS) is None


def test_a_party_no_dataset_publishes_as_a_recipient_is_unclassified():
    row = PartyRow(dcid="country/SEN", kind="party", published=((DAC2A, "origin"),))

    assert destination_side(row, datasets=_DATASETS) is None


def test_every_dac2a_destination_of_the_recorded_index_is_classified():
    parties = load_index(INDEX_FIXTURE).parties.values()
    destinations = [row for row in parties if row.publishes("destination", _DATASETS)]

    assert destinations
    unclassified = [
        row.dcid for row in destinations if destination_side(row, datasets=_DATASETS) is None
    ]
    assert unclassified == []
    assert len(destinations) == 279


def test_the_organisation_records_read_from_a_zipped_package_without_running_its_init(
    tmp_path: Path,
):
    """`organisation_dcids` reads the entity resolver's records through `importlib.resources`,
    so the read works when the package is served from a zip, and never runs the package's
    `__init__`, which loads resolvekit. The zipped `__init__` raises if it runs."""
    records = Path(importlib.util.find_spec("entity_resolution").origin).parent / "data"
    archive = tmp_path / "packages.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("entity_resolution/__init__.py", "raise ImportError('init ran')\n")
        zipped.write(records / "graph_entities.json", "entity_resolution/data/graph_entities.json")
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(archive)!r})\n"
        "from code_index.multilaterals import organisation_dcids\n"
        "print(len(organisation_dcids()), 'entity_resolution' in sys.modules,"
        " any(name.startswith('resolvekit') for name in sys.modules))\n"
    )

    ran = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, timeout=60
    )

    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.split() == [str(len(organisation_dcids())), "False", "False"]


def test_the_entity_resolver_pack_reads_the_organisations_through_the_one_reader():
    assert pack.organisation_dcids is organisation_dcids
