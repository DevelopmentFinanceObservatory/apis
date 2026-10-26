"""The declared conventions a part's scope choices default to.

`SECTOR_GROUPINGS` names the sets of CRS purpose codes a sector concept word reads as, and
`MEASURE_BASIS_CONVENTIONS` names the measure basis each part shape and window defaults to.
`search.scope` reads both, and `graph_release.declarations` hashes them with the other
declarations, so a change to either moves the declarations hash and the scope rule version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MeasureBasis = Literal["gross", "net", "grant_equivalent", "commitments", "oecd_headline"]
"""Every measure basis a part can be counted on. `oecd_headline` is the OECD's headline
basis per year: net flows to 2017 and grant equivalents from 2018, as two period
components."""

PartShape = Literal["donor_only", "recipient_all_sectors", "sector_or_restriction"]
"""The part shapes the basis conventions distinguish. `donor_only` names a donor and no
recipient or sector. `recipient_all_sectors` binds a recipient and no sector, from any
donors. `sector_or_restriction` binds a sector, or restricts a channel, an instrument or a
modality."""

WindowClass = Literal["from_2018", "before_2018", "crossing_2018", "any"]
"""Where a part's window lies against 2018, the first year the OECD publishes grant
equivalents as its headline: every year from 2018, every year before it, years on both
sides, or any window."""


@dataclass(frozen=True)
class SectorGrouping:
    """One sector concept and the CRS purpose codes it reads as.

    `concept` is the lowercase concept word, `code_system` the code-system property the
    codes bind, and `codes` the code dcids in ascending order. `words` are the phrases,
    casefolded, a qualifier match on `code_system` reads the concept from: a match whose
    phrase is one of them binds `codes` in place of the code it names. `label` names the
    grouping for a reader."""

    concept: str
    code_system: str
    codes: tuple[str, ...]
    words: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class BasisConvention:
    """The measure basis a part of `shape` whose window is `window` is counted on.

    `dataset` names the table the convention reads, by its provenance dcid, and `note` the
    OECD figure it matches."""

    shape: PartShape
    window: WindowClass
    basis: MeasureBasis
    dataset: str
    note: str


_CRS_PURPOSE = "crsPurpose"
_CRS = "provenance/OECDCreditorReportingSystem"
_DAC1 = "provenance/OECDDAC1"
_DAC2A = "provenance/OECDDAC2A"

# ONE's analytical groupings of CRS sectors. Health joins Health (120) and Population
# policies and programmes and reproductive health (130); it is not an OECD definition.
# Humanitarian joins emergency response (720), reconstruction relief and rehabilitation
# (730) and disaster prevention and preparedness (740). The words are the phrases the
# qualifier matcher reads each concept from: "health" is CRS 120's label, so "health aid"
# reads it, and "humanitarian aid" is the label of CRS 700, the sector the three codes
# make up. "basic health" (122) and "reproductive health" (130) are phrases of their own
# and stay code bindings.
SECTOR_GROUPINGS: tuple[SectorGrouping, ...] = (
    SectorGrouping(
        concept="health",
        code_system=_CRS_PURPOSE,
        codes=("CRSPurposeCode120", "CRSPurposeCode130"),
        words=("health",),
        label="health, including population and reproductive health (CRS 120 + 130)",
    ),
    SectorGrouping(
        concept="humanitarian",
        code_system=_CRS_PURPOSE,
        codes=("CRSPurposeCode720", "CRSPurposeCode730", "CRSPurposeCode740"),
        words=("humanitarian aid",),
        label="humanitarian aid (CRS 720 + 730 + 740)",
    ),
)

# The measure basis each part shape and window is counted on, after the OECD's own
# reporting: the DAC1 headline for a donor's total, DAC2a receipts for a recipient's,
# gross CRS disbursements wherever a sector, channel, instrument or modality is bound.
MEASURE_BASIS_CONVENTIONS: tuple[BasisConvention, ...] = (
    BasisConvention(
        shape="donor_only",
        window="from_2018",
        basis="grant_equivalent",
        dataset=_DAC1,
        note="OECD headline ODA, grant equivalents (DAC1 measure 11010)",
    ),
    BasisConvention(
        shape="donor_only",
        window="before_2018",
        basis="net",
        dataset=_DAC1,
        note="OECD headline ODA before 2018, net flows",
    ),
    BasisConvention(
        shape="donor_only",
        window="crossing_2018",
        basis="oecd_headline",
        dataset=_DAC1,
        note="OECD headline ODA per year: net flows to 2017, grant equivalents from 2018",
    ),
    BasisConvention(
        shape="recipient_all_sectors",
        window="any",
        basis="net",
        dataset=_DAC2A,
        note="OECD receipts, net disbursements (DAC2a measure 206)",
    ),
    BasisConvention(
        shape="sector_or_restriction",
        window="any",
        basis="gross",
        dataset=_CRS,
        note="CRS gross disbursements",
    ),
)
