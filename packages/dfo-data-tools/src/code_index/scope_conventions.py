"""The declared conventions a part's scope choices default to.

`SECTOR_GROUPINGS` names the sets of CRS purpose codes a sector concept word reads as.
`MEASURE_BASIS_CONVENTIONS` names the measure basis each part shape and window defaults to,
`BASIS_MEASURES` what each basis binds in an identity, and `BASIS_RESTRICTIONS` the code
systems besides the sector whose admitted code makes a part a restricted one.
`OPTION_LABELS` words the options the rules build from code and group labels. `search.scope`
reads them all, and `graph_release.declarations` hashes the tables beside the other
declarations, so a change to one moves the declarations hash and the scope rule version.
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
class BasisMeasure:
    """What an identity counted on `basis` binds.

    `population_type` and `measured_property` are its shape's, and `netting_basis` the
    value its `nettingBasis` slot binds, None for a measure that carries no netting.
    `flow_type` is the constraint property and value that say a flow is disbursed or
    committed. `oda_flows` are the CRS types of flow an ODA flow binding (`ODA_FLOW`) reads as
    on this basis. `words` name the basis for a reader."""

    basis: MeasureBasis
    population_type: str
    measured_property: str
    netting_basis: str | None
    flow_type: tuple[str, str]
    oda_flows: tuple[str, ...]
    words: str


@dataclass(frozen=True)
class SectorGrouping:
    """One sector concept and the CRS purpose codes it reads as.

    `concept` is the lowercase concept word, `code_system` the code-system property the
    codes bind, and `codes` the code dcids in ascending order. `words` are the phrases,
    casefolded, a qualifier match on `code_system` reads the concept from: a match whose
    phrase is one of them binds `codes` in place of the code it names. `label` names the
    grouping for a reader as a standalone noun phrase, the option's only reader copy."""

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
        label="health incl. population and reproductive health (CRS 120 + 130)",
    ),
    SectorGrouping(
        concept="humanitarian",
        code_system=_CRS_PURPOSE,
        codes=("CRSPurposeCode720", "CRSPurposeCode730", "CRSPurposeCode740"),
        words=("humanitarian aid",),
        label="humanitarian aid (CRS 720 + 730 + 740)",
    ),
)


@dataclass(frozen=True)
class OptionLabels:
    """The wording of the scope options whose label is built from the part: each a
    standalone noun phrase naming its subject, since a reader sees the label alone ("Counted
    as: health, CRS 120 alone · Africa's countries only").

    `sector_code` names one CRS purpose code from its `label` and `code` number, and
    `sector_code_alone` one code of a grouping offered without the others.
    `region_with_residuals` and `region_members_only` name a region's two readings from its
    `region` name and its possessive `region_of`. `measure_basis` names a measure basis from
    its `words` (`BasisMeasure.words`) and the `prices` of the identity's price basis
    ("current", "constant"), and `measure_basis_unpriced` from its words alone, for an
    identity whose measurement qualifier states no price basis."""

    sector_code: str
    sector_code_alone: str
    region_with_residuals: str
    region_members_only: str
    measure_basis: str
    measure_basis_unpriced: str


OPTION_LABELS = OptionLabels(
    sector_code="{label} (CRS {code})",
    sector_code_alone="{label}, CRS {code} alone",
    region_with_residuals="{region} incl. regional and unspecified amounts",
    region_members_only="{region_of} countries only",
    measure_basis="{words}, {prices} USD",
    measure_basis_unpriced="{words}",
)

REGION_NAMES: tuple[tuple[str, str], ...] = (
    ("DACAfrica", "Africa"),
    ("DACAmerica", "America"),
    ("DACAsia", "Asia"),
    ("DACCaribbean", "the Caribbean"),
    ("DACCaribbeanAndCentralAmerica", "the Caribbean and Central America"),
    ("DACCentralAmerica", "Central America"),
    ("DACEasternAfrica", "Eastern Africa"),
    ("DACEurope", "Europe"),
    ("DACFarEastAsia", "Far East Asia"),
    ("DACMelanesia", "Melanesia"),
    ("DACMicronesia", "Micronesia"),
    ("DACMiddleAfrica", "Middle Africa"),
    ("DACMiddleEast", "the Middle East"),
    ("DACNorthOfSahara", "North of Sahara"),
    ("DACOceania", "Oceania"),
    ("DACPolynesia", "Polynesia"),
    ("DACSouthAmerica", "South America"),
    ("DACSouthAndCentralAsia", "South and Central Asia"),
    ("DACSouthOfSahara", "South of Sahara"),
    ("DACSouthernAfrica", "Southern Africa"),
    ("DACWesternAfrica", "Western Africa"),
)
"""The reader's name of each regional total whose region a scope option names, by the
total's dcid: "Africa" for `DACAfrica`, whose graph label is "Africa (DAC)". The region
option labels read it on every door alike, so a search and its switch name the region the
same way. A total named here by no row is named by its graph label (`PartyRow.label`)."""

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

ODA_FLOW_SYSTEM = "crsTypeOfFlow"
"""The code system an ODA flow binding binds."""
ODA_FLOW = "CRSTypeOfFlow10"
"""The CRS type of flow an ODA question binds (`ODA_FLOW_SYSTEM`)."""

# What each basis binds. Every basis but commitments is a disbursed flow. Grant equivalents
# count the private sector instruments the CRS records as flow 60 beside ODA's flow 10, as
# the OECD's headline does, so an ODA flow binding reads as both on that basis.
_DISBURSED = ("financialFlowType", "Disbursement")
BASIS_MEASURES: tuple[BasisMeasure, ...] = (
    BasisMeasure(
        basis="gross",
        population_type="FinancialFlow",
        measured_property="amount",
        netting_basis="Gross",
        flow_type=_DISBURSED,
        oda_flows=(ODA_FLOW,),
        words="gross disbursements",
    ),
    BasisMeasure(
        basis="net",
        population_type="FinancialFlow",
        measured_property="amount",
        netting_basis="Net",
        flow_type=_DISBURSED,
        oda_flows=(ODA_FLOW,),
        words="net disbursements",
    ),
    BasisMeasure(
        basis="grant_equivalent",
        population_type="FinancialFlow",
        measured_property="grantEquivalent",
        netting_basis=None,
        flow_type=_DISBURSED,
        oda_flows=(ODA_FLOW, "CRSTypeOfFlow60"),
        words="grant equivalents (OECD headline)",
    ),
    BasisMeasure(
        basis="commitments",
        population_type="FinancialObligation",
        measured_property="amount",
        netting_basis=None,
        flow_type=("financialObligationType", "Commitment"),
        oda_flows=(ODA_FLOW,),
        words="commitments",
    ),
)

# The code systems besides the sector whose admitted code restricts a part to the CRS: the
# instrument and the modality. A part restricted on one is counted as a sector part is. The
# code index holds no channel system, so no channel code restricts a part.
BASIS_RESTRICTIONS: tuple[str, ...] = ("crsTypeOfFinance", "crsCooperationModality")
