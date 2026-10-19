"""The authored declarations the code index export reads.

Code systems, datasets, party kinds and total groups, measure axes, labels and nouns,
synonyms, concept groups, shape slot defaults, implied values, measure classes and variable
aliases. The export
(`code_index.export`) reads them beside the graph. Nothing here reads the graph or imports a
script, so a build that has no graph client can load them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from code_index.types import (
    DatasetDeclaration,
    DeclaredDefault,
    ImpliedValue,
    MeasureAxis,
    MeasureClassDeclaration,
    MeasurePartitionLabel,
    PartlyStatedValue,
    PartyKind,
    PartyWord,
    ShapeSlotDefault,
    StatedWord,
    StatTypeWordDeclaration,
)


@dataclass(frozen=True)
class CodeSystemConfig:
    """One code system's graph enum and semantic declarations.

    `enum_dcid` is the enumeration the system's codes are `typeOf`. The declaration carries it
    as the system's graph id. `hierarchical` is derived from graph rows and checked against the expected value.

    `declared_by_every_statvar_of` lists the provenances of the datasets whose every StatVar
    declares this property. CRS StatVars all declare `crsPurpose`, `crsTypeOfFinance`, and
    `crsTypeOfFlow`. Imputed multilateral StatVars all declare `crsPurpose`, `crsTypeOfFlow`, and
    `dac2aAidType`, and none declares `crsTypeOfFinance`. DAC1 StatVars all declare `dac1AidType`
    and no CRS code system, and DAC2a and DAC2b StatVars all declare their own table's aid type
    (`dac2aAidType`, `dac2bAidType`) and no CRS code system. `crsCooperationModality` is
    optional for older CRS StatVars. A dataset is enumerated only through systems that list it,
    and the export checks every exported row against the listing.

    `plain_name` is the singular lowercase noun used after "any". A default code belongs to this
    system, scope codes belong to other systems, and `when_named` entries are lowercase and
    non-empty. Export validates these conditions before writing. `no_vocabulary_terms` lists
    the lowercase words naming a distinction of this axis no code expresses
    (`CodeSystemDeclaration.no_vocabulary_terms`).
    """

    code_system: str
    enum_dcid: str
    expected_hierarchical: bool
    disjoint_from_children: bool
    summable: bool
    absent_reading: str
    declared_by_every_statvar_of: tuple[str, ...]
    plain_name: str
    default: DeclaredDefault | None = None
    no_vocabulary_terms: tuple[str, ...] = ()


_CRS = "provenance/OECDCreditorReportingSystem"
_IMPUTED_MULTILATERAL_SECTORS = "provenance/ONEImputedMultilateralSectors"
_DAC1 = "provenance/OECDDAC1"
_DAC2A = "provenance/OECDDAC2A"
_DAC2B = "provenance/OECDDAC2B"
_IDS = "provenance/WorldBankInternationalDebtStatistics"

# Datasets whose StatVars declare no code system, so no code read reaches them. The export lists
# their StatVars from the coverage build into `listed_statvars` (`listed_dcids`).
LISTED_DATASETS: tuple[str, ...] = (_IDS,)

# Exported declarations set CRS behaviour.
# `crsTypeOfFlow` is flat, with nine root codes.
CODE_SYSTEMS: tuple[CodeSystemConfig, ...] = (
    CodeSystemConfig(
        code_system="crsPurpose",
        enum_dcid="CRSPurposeCodeEnum",
        expected_hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_CRS, _IMPUTED_MULTILATERAL_SECTORS),
        plain_name="sector",
    ),
    CodeSystemConfig(
        code_system="crsTypeOfFinance",
        enum_dcid="CRSTypeOfFinanceEnum",
        expected_hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_CRS,),
        plain_name="instrument",
    ),
    CodeSystemConfig(
        code_system="crsCooperationModality",
        enum_dcid="CRSCooperationModalityEnum",
        expected_hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(),
        plain_name="modality",
    ),
    CodeSystemConfig(
        code_system="crsTypeOfFlow",
        enum_dcid="CRSTypeOfFlowEnum",
        expected_hierarchical=False,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_CRS, _IMPUTED_MULTILATERAL_SECTORS),
        plain_name="flow",
        # A grant bound under CRSTypeOfFinance100 defaults to ODA when no flow is named.
        # `when_named=("aid",)` also covers a variable phrase naming only "aid". The trigger
        # requires an open flow slot and no qualifier match or tie.
        default=DeclaredDefault(
            code="CRSTypeOfFlow10",
            when_bound_under=("CRSTypeOfFinance100",),
            when_named=("aid",),
        ),
        # No CRS flow type separates non-concessional flows from concessional ones, so a
        # question naming them on a CRS shape refuses the qualifier.
        no_vocabulary_terms=("non-concessional",),
    ),
    # Imputed multilateral and DAC2a StatVars carry `dac2aAidType`, and CRS StatVars do not. DAC2a
    # aid types nest totals over their parts ("ODA grants" under "ODA"), and a net total holds
    # gross parts and the repayments netted off them, so siblings neither sum nor exclude the
    # parent. Every imputed row holds the one root code its table's invariant names.
    CodeSystemConfig(
        code_system="dac2aAidType",
        enum_dcid="DAC2aAidTypeEnum",
        expected_hierarchical=True,
        disjoint_from_children=False,
        summable=False,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_IMPUTED_MULTILATERAL_SECTORS, _DAC2A),
        plain_name="aid type",
        # A DAC2a question that names no aid type asks about total ODA. The imputed table's
        # slot holds its invariant code alone, so the default binds nothing there.
        default=DeclaredDefault(
            code="DAC2aAidType206",
            when_bound_under=(),
            when_open=True,
            note="No aid type was named, so the answer covers official development assistance.",
        ),
    ),
    # DAC1 aid types nest totals over their parts ("bilateral ODA" under "ODA"), and memo items
    # repeat amounts already counted elsewhere, so siblings neither sum nor exclude the parent.
    CodeSystemConfig(
        code_system="dac1AidType",
        enum_dcid="DAC1AidTypeEnum",
        expected_hierarchical=True,
        disjoint_from_children=False,
        summable=False,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_DAC1,),
        plain_name="aid type",
        # A DAC1 question that names no aid type asks about total ODA.
        default=DeclaredDefault(
            code="DAC1AidType1010",
            when_bound_under=(),
            when_open=True,
            note="No aid type was named, so the answer covers official development assistance.",
        ),
    ),
    # DAC2b flow types nest totals over their parts (OOF under official non-concessional
    # flows), and a net total holds gross parts and the repayments netted off them, so siblings
    # neither sum nor exclude the parent.
    CodeSystemConfig(
        code_system="dac2bAidType",
        enum_dcid="DAC2bAidTypeEnum",
        expected_hierarchical=True,
        disjoint_from_children=False,
        summable=False,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_DAC2B,),
        plain_name="official flow type",
        # A DAC2b question that names no flow type asks about the net total, the one root
        # holding every other code but interest received. The table publishes it net alone.
        default=DeclaredDefault(
            code="DAC2bAidType296",
            when_bound_under=(),
            when_open=True,
            note=(
                "No official flow type was named, so the answer covers net official"
                " non-concessional flows."
            ),
        ),
    ),
)


# One declaration per published table. CRS commitments and some disbursement purpose codes carry
# explicit zero rows, so a missing row for those cells reads as zero, and the imputed sectors table
# reads the same way. DAC1, DAC2a and DAC2b read a missing row as unknown, the reading for a table
# with no zero-row evidence. A question answers from the imputed table only when it names the
# table. Of two tables that can serve one family, the lower rank serves it. DAC1, DAC2a and DAC2b
# share the CRS five-tuples under their own codes (`dac1AidType`, `dac2aAidType`,
# `dac2bAidType`), and DAC2a's one `attributedAmount` row declares fewer properties than the
# imputed rows, so each forms families of its own (`CodeIndex.family_of`) and its rank orders it
# against no other table. Every imputed row is an ODA flow of the imputed multilateral aid type,
# declared as invariants the loader checks against every row. Each of the five is a table a
# provider reports, so each gives the provider's account of a flow. Every row of the CRS is a
# flow from one provider to one recipient, which the OECD calls bilateral whatever the provider
# is, because the CRS import leaves out the rows that record a core contribution to a multilateral
# organisation. So the CRS states "bilateral" of both measured properties it publishes
# (`DatasetDeclaration.stated_words`). A DAC2a or DAC2b `amount` row is bilateral only where
# its recipient is a country, since a flow to a multilateral organisation is a core
# contribution, so those tables state it where the destination is a country. DAC2a's
# `attributedAmount` row, imputed multilateral aid, and DAC1, whose aid types split bilateral
# from multilateral, state nothing. "Multilateral" names a party of DAC2a and DAC2b
# (`DatasetDeclaration.party_words`): their provider `DACMultilaterals` is the DAC's published
# total of the multilateral organisations' own outflows, so a DAC2a or DAC2b `amount` row to a
# country is multilateral where its provider is that total. DAC1, whose aid type 2000 codes the
# word as donors' contributions, the CRS and the imputed table name no party.
_MULTILATERAL_PROVIDER = PartyWord(
    word="multilateral",
    measured_property="amount",
    axis="origin",
    party="DACMultilaterals",
    where={"destination": "country"},
)
DATASETS: tuple[DatasetDeclaration, ...] = (
    DatasetDeclaration(
        provenance=_CRS,
        label="OECD Creditor Reporting System (CRS)",
        description=(
            "Development finance that providers report to the OECD activity by activity, by"
            " sector, type of flow, type of finance and co-operation modality."
        ),
        opt_in=False,
        absent_observation="zero",
        account="provider",
        rank=0,
        stated_words=(
            StatedWord(word="bilateral", measured_property="amount"),
            StatedWord(word="bilateral", measured_property="grantEquivalent"),
        ),
    ),
    DatasetDeclaration(
        provenance=_IMPUTED_MULTILATERAL_SECTORS,
        label="ONE Imputed Multilateral Sectors",
        description=(
            "The share of each multilateral organisation's spending attributed back to a"
            " bilateral provider, in proportion to that provider's share of the organisation's"
            " core resources."
        ),
        opt_in=True,
        absent_observation="zero",
        account="provider",
        rank=10,
        invariants={"crsTypeOfFlow": "CRSTypeOfFlow10", "dac2aAidType": "DAC2aAidType106"},
    ),
    DatasetDeclaration(
        provenance=_DAC1,
        label="OECD DAC1",
        description=(
            "Donors' aggregate official and private flows reported to the OECD, as totals by"
            " donor without a breakdown by recipient."
        ),
        opt_in=False,
        absent_observation="unknown",
        account="provider",
        rank=20,
    ),
    DatasetDeclaration(
        provenance=_DAC2A,
        label="OECD DAC2a",
        description=(
            "Donors' official development assistance disbursements reported to the OECD, by"
            " recipient and type of aid."
        ),
        opt_in=False,
        absent_observation="unknown",
        account="provider",
        rank=30,
        stated_words=(
            StatedWord(
                word="bilateral", measured_property="amount", where={"destination": "country"}
            ),
        ),
        party_words=(_MULTILATERAL_PROVIDER,),
    ),
    DatasetDeclaration(
        provenance=_DAC2B,
        label="OECD DAC2b",
        description=(
            "Other official flows and officially supported export credits that providers report"
            " to the OECD, by recipient and type of flow."
        ),
        opt_in=False,
        absent_observation="unknown",
        account="provider",
        rank=40,
        stated_words=(
            StatedWord(
                word="bilateral", measured_property="amount", where={"destination": "country"}
            ),
        ),
        party_words=(_MULTILATERAL_PROVIDER,),
    ),
)


# The kind each graph type gives a party of a declared dataset (`code_index.types.party_kind`).
# Every type a donor or recipient of a declared or listed dataset carries is named here. The export refuses
# a party with a type this table does not name, or whose types give it no single kind. A type
# mapped to `None` names no kind of its own and sits beside a type that does: `Place` beside
# `Country` (India), `ProvisionalNode` beside `Organization` (IDA), `LendingEntityEnum` beside
# `Organization` (EUROFIMA), the administrative types beside `Country` (Hong Kong, Mayotte). IDS
# publishes World Bank regions ("Sub-Saharan Africa, excluding high income") and lending groups
# ("IDA total") as debtors beside the countries they hold, and its unidentified creditors
# ("bondholders", "multiple lenders") as creditors that hold no other creditor's figures.
PARTY_TYPE_KINDS: Mapping[str, PartyKind | None] = {
    "Country": "party",
    "Organization": "party",
    "DACGroup": "total",
    "Grouping": "total",
    "UNCountryDevelopmentGroup": "total",
    "WorldBankIncomeGroup": "total",
    "WorldBankLendingGroup": "total",
    "WorldBankRegion": "total",
    "DACRecipientUnallocatedByCountry": "residual",
    "DACUnspecifiedParty": "residual",
    "IDSUnidentifiedCreditor": "residual",
    "AdministrativeArea": None,
    "AdministrativeArea1": None,
    "AdministrativeArea2": None,
    "City": None,
    "LendingEntityEnum": None,
    "Place": None,
    "ProvisionalNode": None,
    "SpecialAdministrativeRegion": None,
}


@dataclass(frozen=True)
class KindOverride:
    """The kind one party takes in place of its graph types' kind, and why."""

    kind: PartyKind
    reason: str


# The parties whose kind differs from what their graph types give. An organisation is typed
# `Organization` and so reads as a `party`, yet DAC2A and DAC2B publish a group of organisations
# beside its members, so a sum across donors counts the members twice. A row stands when, in every
# year the group reports (1960 to 2024), its gross and net disbursements equal the sum of its
# members' within one dollar per member: DAC2A's ODA (aid type 206) and DAC2B's total (aid type
# 972), at DACAllRecipients and, wherever the group reports there, at Bangladesh, Ethiopia, India
# and Kenya. A search over every subset of the other donors' gross disbursements at
# DACAllRecipients, for each of the 122 organisation donors of the two datasets, finds a sum for
# these four and for the IMF only. The IMF equals its Concessional Trust Funds from 1986 (DAC2A)
# and its Resilience and Sustainability Trust (DAC2B), but reports alone in 1977 to 1985, so no sum
# covers every year and it holds no row. Neither trust fund holds a row either: in DAC2A and DAC2B
# each equals the IMF in every year and series it reports, but the CRS publishes both and never the
# IMF, and a kind holds in every dataset, so a `total` would drop the IMF's CRS figures from a sum.
PARTY_KIND_OVERRIDES: Mapping[str, KindOverride] = {
    "AfricanDevelopmentBankGroup": KindOverride(
        "total", "AfricanDevelopmentBank plus AfricanDevelopmentFund"
    ),
    "InterAmericanDevelopmentBankGroup": KindOverride(
        "total", "InterAmericanDevelopmentBank plus IDBInvest"
    ),
    "PostcodeLotteryGroup": KindOverride(
        "total", "the Dutch, German, Norwegian, People's and Swedish postcode lotteries"
    ),
    "WorldBankGroup": KindOverride(
        "total",
        "InternationalDevelopmentAssociation, InternationalBankForReconstructionAndDevelopment "
        "and InternationalFinanceCorporation",
    ),
}


@dataclass(frozen=True)
class TotalGroup:
    """The group one total holds, as search names the group, and how it holds it.

    With `sums_members` true the total equals the sum of the group's members plus
    `residuals`, the residuals it counts beside them, sorted. With `sums_members` false the
    total is the figure the source publishes for the group, which can differ from that sum,
    and it counts no residuals."""

    group: str
    residuals: tuple[str, ...] = ()
    sums_members: bool = True


def _unallocated(*regions: str) -> tuple[str, ...]:
    """The DAC residuals "<region>, unallocated by country" of `regions`, sorted."""
    return tuple(sorted(f"DAC{region}Unallocated" for region in regions))


# The group each total holds, as search names the group: a graph place, or a resolver group's
# dcid (`entity_resolution.regions.identity_to_dcid`). The graph links no total to its group, so
# this table is reviewed by hand. A row holds its group in one of two ways. A summing row
# (`sums_members` true) stands when, in every year from 2000 to 2023, the total equals the sum of
# the group's members plus the residuals the row counts beside them: France's DAC2A ODA
# disbursements for the recipient totals, DAC1 ODA by donor for the donor ones. A published row
# (`sums_members` false) names the figure the source publishes for a group whose members' sum
# differs from it, and counts no residuals: DAC countries, total, which DAC1, DAC2A and DAC2B
# publish as a donor, differs from the sum over the DAC's members where members report late.
# A continent's residuals include those of its subregions ("South of Sahara, unallocated by
# country" within Africa). The DAC regions whose members differ from the graph's places (Asia,
# Europe, North and South of Sahara, among others) hold resolvekit's DAC region groups, which
# list the DAC's own members. A region already keyed to a graph place stays there, since a total
# holds one group. The East African Community and the Indus Basin count inside DAC's Eastern
# Africa and South and Central Asia, but they are parties, not residuals, and their amounts stop
# before 1994, so no row counts them. Totals that hold no group: DAC's lower and upper middle
# income and small island states totals differ from the sums over the resolver's members, and
# Northern America has no amounts after 1996.
TOTAL_GROUPS: Mapping[str, TotalGroup] = {
    "DACAfrica": TotalGroup(
        "africa",
        _unallocated(
            "Africa",
            "EasternAfrica",
            "MiddleAfrica",
            "NorthOfSahara",
            "SouthOfSahara",
            "SouthernAfrica",
            "WesternAfrica",
        ),
    ),
    "DACAmerica": TotalGroup(
        "undata-geo/G00134000",
        _unallocated(
            "America", "Caribbean", "CaribbeanAndCentralAmerica", "CentralAmerica", "SouthAmerica"
        ),
    ),
    "DACAsia": TotalGroup(
        "groups/OECD.DAC.Region.Asia",
        tuple(
            sorted(
                (
                    *_unallocated(
                        "Asia",
                        "CentralAsia",
                        "FarEastAsia",
                        "MiddleEast",
                        "SouthAndCentralAsia",
                        "SouthAsia",
                    ),
                    "DACMekongDeltaProject",
                )
            )
        ),
    ),
    "DACCaribbean": TotalGroup("Caribbean", _unallocated("Caribbean")),
    "DACCaribbeanAndCentralAmerica": TotalGroup(
        "groups/OECD.DAC.Region.CaribbeanAndCentralAmerica",
        _unallocated("Caribbean", "CaribbeanAndCentralAmerica", "CentralAmerica"),
    ),
    "DACCentralAmerica": TotalGroup("CentralAmerica", _unallocated("CentralAmerica")),
    "DACCountriesTotal": TotalGroup("groups/OECD.DAC", sums_members=False),
    "DACEasternAfrica": TotalGroup(
        "groups/OECD.DAC.Region.EasternAfrica", _unallocated("EasternAfrica")
    ),
    "DACEurope": TotalGroup(
        "groups/OECD.DAC.Region.Europe", _unallocated("Europe", "ExYugoslavia")
    ),
    "DACFarEastAsia": TotalGroup(
        "groups/OECD.DAC.Region.FarEastAsia",
        tuple(sorted((*_unallocated("FarEastAsia"), "DACMekongDeltaProject"))),
    ),
    "DACMelanesia": TotalGroup("Melanesia", _unallocated("Melanesia")),
    "DACMicronesia": TotalGroup("Micronesia", _unallocated("Micronesia")),
    "DACMiddleAfrica": TotalGroup(
        "groups/OECD.DAC.Region.MiddleAfrica", _unallocated("MiddleAfrica")
    ),
    "DACMiddleEast": TotalGroup("groups/OECD.DAC.Region.MiddleEast", _unallocated("MiddleEast")),
    "DACNonDACCountries": TotalGroup("groups/OECD.NonDAC"),
    "DACNorthOfSahara": TotalGroup(
        "groups/OECD.DAC.Region.NorthOfSahara", _unallocated("NorthOfSahara")
    ),
    "DACOceania": TotalGroup(
        "oceania", _unallocated("Melanesia", "Micronesia", "Oceania", "Polynesia")
    ),
    "DACPolynesia": TotalGroup("Polynesia", _unallocated("Polynesia")),
    "DACSouthAmerica": TotalGroup(
        "groups/OECD.DAC.Region.SouthAmerica", _unallocated("SouthAmerica")
    ),
    "DACSouthAndCentralAsia": TotalGroup(
        "groups/OECD.DAC.Region.SouthAndCentralAsia",
        _unallocated("CentralAsia", "SouthAndCentralAsia", "SouthAsia"),
    ),
    "DACSouthOfSahara": TotalGroup(
        "groups/OECD.DAC.Region.SouthOfSahara",
        _unallocated(
            "EasternAfrica", "MiddleAfrica", "SouthOfSahara", "SouthernAfrica", "WesternAfrica"
        ),
    ),
    "DACSouthernAfrica": TotalGroup("SouthernAfrica", _unallocated("SouthernAfrica")),
    "DACWesternAfrica": TotalGroup(
        "groups/OECD.DAC.Region.WesternAfrica", _unallocated("WesternAfrica")
    ),
    "GroupOfSeven": TotalGroup("GroupOfSeven"),
    "UNLandlockedDevelopingCountries": TotalGroup("groups/UN.LLDC"),
    "UNLeastDevelopedCountries": TotalGroup("groups/UN.LDC"),
    "WorldBankHighIncome": TotalGroup("groups/WB.IncomeGroup.High"),
    "WorldBankLowIncome": TotalGroup("groups/WB.IncomeGroup.Low"),
}

# Axis order controls partition election.
# `measuredProperty` is the basis axis, and either basis value generalises.
# `label` names an axis the graph gives no plain name for, as a description's dimension.
MEASURE_AXES: tuple[MeasureAxis, ...] = (
    MeasureAxis(
        axis="populationType",
        kind="shape",
        role="free",
        prefer=("FinancialFlow", "FinancialObligation"),
    ),
    MeasureAxis(
        axis="measuredProperty",
        kind="shape",
        role="basis",
        prefer=("amount", "grantEquivalent"),
    ),
    MeasureAxis(
        axis="measurementQualifier",
        kind="shape",
        role="free",
        prefer=("Nominal", "RealValue", "PurchasingPowerParity"),
    ),
    MeasureAxis(
        axis="financialFlowType",
        kind="constraint",
        role="free",
        prefer=("Disbursement",),
        label="flow type",
    ),
    MeasureAxis(
        axis="financialObligationType",
        kind="constraint",
        role="free",
        prefer=("Commitment",),
    ),
    MeasureAxis(
        axis="nettingBasis",
        kind="constraint",
        role="free",
        prefer=("Gross", "Net"),
        label="netting basis",
    ),
)


@dataclass(frozen=True, slots=True)
class MeasureValueText:
    """Authored text overrides for one measure-value dcid.

    `label` replaces the graph name. `synonyms` add to `alternateName`. `matchable` controls text
    lookup. `amount` keeps a display label but skips text indexing because queries use it as a
    determiner. `same_reading_as` names the canonical value of an alias dcid. Query text naming
    the alias resolves to the canonical value, so authored synonyms belong on the canonical one.
    """

    label: str | None = None
    synonyms: tuple[str, ...] = ()
    matchable: bool = True
    same_reading_as: str | None = None


# Text overrides merge into measure rows.
# Unmapped dcids keep graph names and aliases, and stay matchable.
MEASURE_VALUE_TEXT: Mapping[str, MeasureValueText] = {
    "Nominal": MeasureValueText(
        label="nominal",
        synonyms=("current prices", "at current prices", "current US dollars"),
    ),
    # "constant" alone covers "constant <year> prices" and "constant <year> euros", since
    # matching reads exact text and a year is not a fixed token.
    "RealValue": MeasureValueText(
        label="real terms",
        synonyms=(
            "real",
            "in real terms",
            "at constant prices",
            "constant",
            "constant prices",
            "constant US dollars",
            "inflation-adjusted",
            "adjusted for inflation",
        ),
    ),
    "InflationAdjusted": MeasureValueText(label="inflation adjusted", same_reading_as="RealValue"),
    "PurchasingPowerParity": MeasureValueText(label="purchasing power parity", synonyms=("PPP",)),
    "amount": MeasureValueText(matchable=False),
    # These phrases name the imputed tables' measure as "grant equivalent" names CRS's. The
    # bare verb would read in any question using "attributed to" ("deaths attributed to air
    # pollution") and warn that it was not applied.
    "attributedAmount": MeasureValueText(synonyms=("attributed aid", "attributed to donors")),
}

# Values exported whether or not a CRS StatVar row carries them. The price bases below are
# carried by World Bank, IMF and DC-native series outside the index, and a query names them.
DECLARED_MEASURE_VALUES: Mapping[str, tuple[str, ...]] = {
    "measurementQualifier": ("InflationAdjusted", "PurchasingPowerParity", "RealValue"),
}

# Extra aliases supplement graph `alternateName`. Unmapped dcids keep graph names and aliases.
# `development assistance` and `other official flows` fill terms the graph aliases miss.
# Qualifier matching uses text alone, so `aid` would bind ODA for any phrase containing it.
# `when_named=("aid",)` handles `aid` as the extracted variable when the flow slot is open and
# qualifier matching leaves the slot unmatched or ambiguous.
# "Imputed" names DAC2a's imputed multilateral line when the query holds no longer label of it
# ("imputed aid from Norway"), so DAC1's donor totals, which read nothing, do not cover it.
CODE_SYNONYMS: Mapping[str, tuple[str, ...]] = {
    # DFO convention: "reproductive health" means the whole sector 130 (population policies and
    # reproductive health), broader than 13020 "reproductive health care".
    "CRSPurposeCode130": ("reproductive health",),
    "CRSTypeOfFlow10": ("development assistance",),
    "CRSTypeOfFlow21": ("other official flows",),
    "DAC1AidType1820": ("refugee costs in donor countries",),
    "DAC2aAidType106": ("imputed",),
}

# Code synonyms `_strip_measure_words` keeps whole although a measure value's words sit inside
# them, because there the word names no measure. Each is written casefolded. Without "social
# safety net" on CRSPurposeCode16010, "net" in "social safety nets in Kenya" would read as the
# netting basis. DAC1AidType1900's and 2110's "negative commitments" are recovered amounts,
# not the commitment measure.
MEASURE_WORD_SYNONYMS_KEPT: frozenset[str] = frozenset(
    {
        "social safety net",
        "recoveries on bilateral oda grants and negative commitments",
        "recoveries on bilateral oda grants / negative commitments",
        "recoveries on multilateral oda grants and capital subscriptions and negative commitments",
        "recoveries on multilateral oda grants and capital subscriptions / negative commitments",
    }
)

# Codes of different tables that name one concept, at one scope. The export gives every code
# of a group the union of the group's labels and synonyms as its synonyms
# (`_apply_concept_groups`), less the measure words they hold (`_strip_measure_words`). The
# five-tuple family election compares the query phrases each table's codes read
# (`search.engine.retrieval.bound_values.elect_five_tuple_family`), so a table whose code
# misses a text another table's code of the same concept reads loses every question naming it.
# Each code is named with its code system. A group joins codes of different code systems only;
# a parent and its child name different scopes and belong to no group together.
CONCEPT_GROUPS: tuple[tuple[tuple[str, str], ...], ...] = (
    (
        ("crsTypeOfFlow", "CRSTypeOfFlow10"),
        ("dac1AidType", "DAC1AidType1010"),
        ("dac2aAidType", "DAC2aAidType206"),
    ),
    (
        ("crsPurpose", "CRSPurposeCode520"),
        ("dac2aAidType", "DAC2aAidType213"),
    ),
    (
        ("crsPurpose", "CRSPurposeCode700"),
        ("dac2aAidType", "DAC2aAidType216"),
    ),
    (
        ("crsPurpose", "CRSPurposeCode930"),
        ("dac1AidType", "DAC1AidType1820"),
    ),
    (
        ("crsTypeOfFlow", "CRSTypeOfFlow21"),
        ("dac1AidType", "DAC1AidType200"),
        ("dac2bAidType", "DAC2bAidType972"),
    ),
    (
        ("crsTypeOfFlow", "CRSTypeOfFlow22"),
        ("dac1AidType", "DAC1AidType3000"),
        ("dac2bAidType", "DAC2bAidType292"),
    ),
    (
        ("crsTypeOfFlow", "CRSTypeOfFlow36"),
        ("dac1AidType", "DAC1AidType340"),
    ),
    (
        ("crsTypeOfFinance", "CRSTypeOfFinance310"),
        ("dac2aAidType", "DAC2aAidType210"),
    ),
    (
        ("crsTypeOfFinance", "CRSTypeOfFinance311"),
        ("dac2aAidType", "DAC2aAidType211"),
    ),
    (
        ("crsPurpose", "CRSPurposeCode910"),
        ("dac1AidType", "DAC1AidType1700"),
    ),
    (
        ("crsPurpose", "CRSPurposeCode99820"),
        ("dac1AidType", "DAC1AidType1810"),
    ),
    (
        ("dac1AidType", "DAC1AidType1902"),
        ("dac2aAidType", "DAC2aAidType207"),
    ),
    (
        ("dac1AidType", "DAC1AidType1600"),
        ("dac2aAidType", "DAC2aAidType255"),
    ),
    (
        ("dac1AidType", "DAC1AidType3102"),
        ("dac2bAidType", "DAC2bAidType295"),
    ),
    (
        ("dac1AidType", "DAC1AidType1020"),
        ("dac2bAidType", "DAC2bAidType298"),
    ),
)

# The graph holds no `shortName` arc on the CRS flow-type or finance-type codes. A flow type's
# name reads inside a possessive heading ("Germany's gross ODA disbursements"), a finance
# type's after "as" ("as grants"). A graph `shortName` arc, where one exists, wins over the
# name authored here.
CODE_SHORT_NAMES: Mapping[str, str] = {
    "CRSTypeOfFlow10": "ODA",
    "CRSTypeOfFlow21": "OOF",
    "CRSTypeOfFlow22": "export credits",
    "CRSTypeOfFlow30": "private philanthropy",
    "CRSTypeOfFlow36": "private FDI",
    "CRSTypeOfFlow37": "other private flows",
    "CRSTypeOfFlow40": "non-flow indicators",
    "CRSTypeOfFlow50": "other flows",
    "CRSTypeOfFlow60": "private sector instruments",
    # Finance types read after "as" in a heading ("to Nigeria as grants").
    "CRSTypeOfFinance0": "non-flow items",
    "CRSTypeOfFinance1": "gross national income",
    "CRSTypeOfFinance2": "ODA as a share of GNI",
    "CRSTypeOfFinance3": "total flows as a share of GNI",
    "CRSTypeOfFinance4": "population",
    "CRSTypeOfFinance100": "grants",
    "CRSTypeOfFinance110": "standard grants",
    "CRSTypeOfFinance210": "interest subsidies",
    "CRSTypeOfFinance310": "capital subscriptions on deposit basis",
    "CRSTypeOfFinance311": "capital subscriptions on encashment basis",
    "CRSTypeOfFinance420": "debt instruments",
    "CRSTypeOfFinance421": "loans",
    "CRSTypeOfFinance422": "reimbursable grants",
    "CRSTypeOfFinance4221": "loan-type reimbursable grants",
    "CRSTypeOfFinance4222": "reflow-based reimbursable grants",
    "CRSTypeOfFinance423": "bonds",
    "CRSTypeOfFinance424": "asset-backed securities",
    "CRSTypeOfFinance425": "other debt securities",
    "CRSTypeOfFinance430": "mezzanine finance instruments",
    "CRSTypeOfFinance431": "subordinated loans",
    "CRSTypeOfFinance432": "preferred equity",
    "CRSTypeOfFinance433": "other hybrid instruments",
    "CRSTypeOfFinance434": "multilateral hybrid capital",
    "CRSTypeOfFinance500": "equity",
    "CRSTypeOfFinance510": "common equity",
    "CRSTypeOfFinance520": "shares in collective investment vehicles",
    "CRSTypeOfFinance530": "reinvested earnings",
    "CRSTypeOfFinance600": "debt relief",
    "CRSTypeOfFinance610": "debt forgiveness of ODA claims on principal",
    "CRSTypeOfFinance611": "debt forgiveness of ODA claims on interest",
    "CRSTypeOfFinance612": "debt forgiveness of OOF claims on principal",
    "CRSTypeOfFinance613": "debt forgiveness of OOF claims on interest",
    "CRSTypeOfFinance614": "debt forgiveness of private claims on principal",
    "CRSTypeOfFinance615": "debt forgiveness of private claims on interest",
    "CRSTypeOfFinance616": "debt forgiveness of OOF claims under debt service reduction",
    "CRSTypeOfFinance617": "debt forgiveness of private claims under debt service reduction",
    "CRSTypeOfFinance618": "other debt forgiveness",
    "CRSTypeOfFinance620": "debt rescheduling of ODA claims on principal",
    "CRSTypeOfFinance621": "debt rescheduling of ODA claims on interest",
    "CRSTypeOfFinance622": "debt rescheduling of OOF claims on principal",
    "CRSTypeOfFinance623": "debt rescheduling of OOF claims on interest",
    "CRSTypeOfFinance624": "debt rescheduling of private claims on principal",
    "CRSTypeOfFinance625": "debt rescheduling of private claims on interest",
    "CRSTypeOfFinance626": "debt rescheduling of OOF claims under debt service reduction",
    "CRSTypeOfFinance627": "debt rescheduling of private claims under debt service reduction",
    "CRSTypeOfFinance630": "debt rescheduling of OOF claims under debt service reduction on original loan principal",
    "CRSTypeOfFinance631": "debt rescheduling of OOF claims under debt service reduction on original loan interest",
    "CRSTypeOfFinance632": "debt rescheduling of private claims under debt service reduction on original loan principal",
    "CRSTypeOfFinance633": "debt forgiveness of export credit claims on principal",
    "CRSTypeOfFinance634": "debt forgiveness of export credit claims on interest",
    "CRSTypeOfFinance635": "debt forgiveness of export credit claims under debt service reduction",
    "CRSTypeOfFinance636": "debt rescheduling of export credit claims on principal",
    "CRSTypeOfFinance637": "debt rescheduling of export credit claims on interest",
    "CRSTypeOfFinance638": "debt rescheduling of export credit claims under debt service reduction",
    "CRSTypeOfFinance639": "debt rescheduling of export credit claims under debt service reduction on original loan principal",
    "CRSTypeOfFinance1000": "guarantees and other unfunded contingent liabilities",
    "CRSTypeOfFinance1100": "guarantees and insurance",
    "CRSTypeOfFinance1101": "individual loan guarantees",
    "CRSTypeOfFinance1102": "loan portfolio guarantees",
    "CRSTypeOfFinance1103": "individual mezzanine finance guarantees",
    "CRSTypeOfFinance1104": "mezzanine finance portfolio guarantees",
    "CRSTypeOfFinance1105": "individual equity finance guarantees",
    "CRSTypeOfFinance1106": "equity portfolio guarantees",
    "CRSTypeOfFinance1107": "first-loss guarantees",
    "CRSTypeOfFinance1108": "other guarantees",
}

# A label keys a measure partition, or a family whose rows share fewer measure axes than a
# partition. `label_for_pairs` picks the most specific label, so a family-level label never
# names a full partition. `measurementQualifier` stays outside the key, because no measure
# label states the price basis (`code_index.labels.compose_measure_label`).
#
# Each label's `definition` restates the OECD's own words: the DAC Glossary for commitments,
# disbursements, imputed multilateral flows and the grant element
# (https://www.oecd.org/dac/dac-glossary.htm), the ODA definition for grant equivalents
# (https://www.oecd.org/dac/financing-sustainable-development/development-finance-standards/officialdevelopmentassistancedefinitionandcoverage.htm),
# and the mobilisation standard for private finance mobilised
# (https://www.oecd.org/dac/financing-sustainable-development/development-finance-standards/mobilisation.htm).
_MOBILISED = (
    "Finance from private sources that official development finance interventions, such as "
    "guarantees, syndicated loans and direct investment, mobilise for a project"
)
MEASURE_PARTITION_LABELS: tuple[MeasurePartitionLabel, ...] = (
    MeasurePartitionLabel(
        axis_values=(
            ("financialObligationType", "Commitment"),
            ("measuredProperty", "amount"),
            ("populationType", "FinancialObligation"),
        ),
        label="commitments",
        definition=(
            "Firm written obligations, backed by the necessary funds, to give specified "
            "resources to a recipient country or multilateral organisation."
        ),
    ),
    # A family spanning both netting bases (CRS and DAC1 disbursements) reads as this
    # label. A gross or net partition matches its own, more specific label.
    MeasurePartitionLabel(
        axis_values=(
            ("financialFlowType", "Disbursement"),
            ("measuredProperty", "amount"),
            ("populationType", "FinancialFlow"),
        ),
        label="disbursements",
        definition=(
            "Funds released to a recipient, or goods and services bought for it, in a period: "
            "the actual transfer of resources, recorded gross or net of repayments."
        ),
    ),
    MeasurePartitionLabel(
        axis_values=(
            ("financialFlowType", "Disbursement"),
            ("measuredProperty", "amount"),
            ("nettingBasis", "Gross"),
            ("populationType", "FinancialFlow"),
        ),
        label="gross disbursements",
        definition=(
            "The total amount released to recipients, or spent on goods and services for them, "
            "in a period, before any repayments are subtracted."
        ),
    ),
    MeasurePartitionLabel(
        axis_values=(
            ("financialFlowType", "Disbursement"),
            ("measuredProperty", "amount"),
            ("nettingBasis", "Net"),
            ("populationType", "FinancialFlow"),
        ),
        label="net disbursements",
        definition=(
            "The amount released to recipients in a period, less any loan principal repaid or "
            "grants recovered in the same period."
        ),
    ),
    MeasurePartitionLabel(
        axis_values=(
            ("financialFlowType", "Disbursement"),
            ("measuredProperty", "attributedAmount"),
            ("nettingBasis", "Gross"),
            ("populationType", "FinancialFlow"),
        ),
        # ONE imputed multilateral StatVars and DAC2a's imputed multilateral row hold this
        # partition, all gross disbursements of one aid type, so the aid type is the
        # measure's name.
        label="imputed multilateral ODA",
        named_values=(
            ("crsTypeOfFlow", "CRSTypeOfFlow10"),
            ("dac2aAidType", "DAC2aAidType106"),
        ),
        definition=(
            "A provider's core contributions to multilateral agencies, shared out among "
            "recipients in proportion to where those agencies disbursed their funds that year."
        ),
    ),
    MeasurePartitionLabel(
        axis_values=(
            ("financialFlowType", "Disbursement"),
            ("measuredProperty", "grantEquivalent"),
            ("populationType", "FinancialFlow"),
        ),
        label="grant equivalents",
        definition=(
            "A measure of a provider's effort that counts each grant, loan or other flow at its "
            "face value times its grant element, so a grant counts in full and a loan counts only "
            "for the share given through terms softer than the market's."
        ),
    ),
    # Only DAC1 StatVars hold `amountMobilised`, the private finance that official flows mobilise.
    MeasurePartitionLabel(
        axis_values=(
            ("financialFlowType", "Disbursement"),
            ("measuredProperty", "amountMobilised"),
            ("populationType", "FinancialFlow"),
        ),
        label="private finance mobilised by disbursements",
        definition=f"{_MOBILISED}, recorded as disbursements.",
    ),
    MeasurePartitionLabel(
        axis_values=(
            ("financialFlowType", "Disbursement"),
            ("measuredProperty", "amountMobilised"),
            ("nettingBasis", "Gross"),
            ("populationType", "FinancialFlow"),
        ),
        label="private finance mobilised by gross disbursements",
        definition=f"{_MOBILISED}, recorded as disbursements before any repayments.",
    ),
    MeasurePartitionLabel(
        axis_values=(
            ("financialFlowType", "Disbursement"),
            ("measuredProperty", "amountMobilised"),
            ("nettingBasis", "Net"),
            ("populationType", "FinancialFlow"),
        ),
        label="private finance mobilised by net disbursements",
        definition=f"{_MOBILISED}, recorded as disbursements less repayments.",
    ),
    MeasurePartitionLabel(
        axis_values=(
            ("financialObligationType", "Commitment"),
            ("measuredProperty", "amountMobilised"),
            ("populationType", "FinancialObligation"),
        ),
        label="private finance mobilised by commitments",
        definition=f"{_MOBILISED}, recorded as commitments.",
    ),
)


# Nouns for IDS and WEO measures, which no measure partition covers. A heading would
# otherwise read the graph name, which carries the unit inside it: "Debt service on
# external debt (current US$)". Each entry is keyed on exact five-tuple and constraint
# values, read on the dev graph, and matched with `label_for_pairs`. A noun never names
# the price basis (`measurementQualifier`) or the denominator, which belong to the unit.
# No entry keys on `financialFlowType = Disbursement` alone, since CRS disbursements carry
# that value and take their partition's label (`_verify_measure_nouns`).
#
# A noun names every measure whose pairs contain its key, and a card shows the constraints
# beyond the key (`maturityBand = LongTermMaturity`, an `economicSector`) beside the
# definition. So each noun's `definition` states only what its key states, and holds for every
# measure the noun names. It follows its publisher's note on the series named beside it, less
# any clause on maturity, IMF credit, guarantee, sector or price basis: the World Bank's
# indicator notes for IDS and World Development Indicators series
# (https://api.worldbank.org/v2/indicator/<code>), and the IMF's WEO indicator descriptions
# (https://api.imf.org/external/sdmx/3.0/structure/codelist/IMF.RES/CL_WEO_INDICATOR/+).
_POPULATION = "populationType"
_PROPERTY = "measuredProperty"
_STAT_TYPE = "statType"

_GDP_BASIS = (
    ("productTaxBasis", "IncludingProductTaxes"),
    ("fixedCapitalConsumptionBasis", "GrossOfFixedCapitalConsumption"),
)
# The IMF's WEO descriptor of NGDPD and PPPGDP, and the World Bank's note on NY.GDP.MKTP.CD:
# GDP "is the total income earned through the production of goods and services in an economic
# territory during an accounting period", measured by the expenditure, income or production
# approach. It names no sector, so it holds for the sector series the WDI noun also names.
_GDP_DEFINITION = (
    "The total income earned through the production of goods and services in an economy "
    "during a period, measured by what is spent, earned or produced."
)


def _noun(label: str, *pairs: tuple[str, str], definition: str) -> MeasurePartitionLabel:
    return MeasurePartitionLabel(axis_values=pairs, label=label, definition=definition)


def _ids(
    label: str, population: str, axis: str, value: str, *, definition: str
) -> MeasurePartitionLabel:
    return _noun(
        label,
        (_POPULATION, population),
        (_PROPERTY, "amount"),
        (_STAT_TYPE, "measuredValue"),
        (axis, value),
        definition=definition,
    )


_IDS_FLOW = (
    (_POPULATION, "FinancialFlow"),
    (_PROPERTY, "amount"),
    (_STAT_TYPE, "measuredValue"),
)
_WITH_IMF_CREDIT = (
    ("maturityBand", "LongTermMaturity"),
    ("includesIMFCredit", "WithIMFCredit"),
)

MEASURE_NOUNS: tuple[MeasurePartitionLabel, ...] = (
    # World Bank International Debt Statistics.
    _ids(
        "external debt service",
        "FinancialFlow",
        "financialFlowType",
        "ServicingPayment",
        # DT.TDS.DLXF.CD: "the sum of principal repayments and interest payments actually made"
        definition=(
            "Principal and interest actually paid by borrowers, in currency, goods or services, "
            "on external debt."
        ),
    ),
    _ids(
        "interest payments on external debt",
        "FinancialFlow",
        "financialFlowType",
        "InterestPayment",
        # DT.INT.DECT.CD: "actual amounts of interest paid by the borrower in currency, goods, or
        # services"
        definition=(
            "Interest actually paid by borrowers, in currency, goods or services, on external debt."
        ),
    ),
    _ids(
        "principal repayments on external debt",
        "FinancialFlow",
        "financialFlowType",
        "PrincipalRepayment",
        # DT.AMT.DLXF.CD
        definition=(
            "Principal actually repaid by borrowers, in currency, goods or services, on "
            "external debt."
        ),
    ),
    # An IDS StatVar declaring `netOf` nets one of two things off disbursements: principal
    # repayments, or servicing payments (principal and interest). Each value has its noun.
    _noun(
        "net flows on external debt",
        *_IDS_FLOW,
        ("financialFlowType", "Disbursement"),
        ("netOf", "PrincipalRepayment"),
        ("nettingBasis", "Net"),
        # DT.NFL.DLXF.CD: "Net flows ... received by the borrower during the year are
        # disbursements minus principal repayments."
        definition=(
            "Disbursements on external debt received by borrowers in a year, less the principal "
            "they repaid in it."
        ),
    ),
    _noun(
        "net transfers on external debt",
        *_IDS_FLOW,
        ("financialFlowType", "Disbursement"),
        ("netOf", "ServicingPayment"),
        ("nettingBasis", "Net"),
        # DT.NTR.DLXF.CD: "Net transfers are net flows minus interest payments during the year"
        definition=(
            "Disbursements on external debt received by borrowers in a year, less the principal "
            "and interest they paid in it."
        ),
    ),
    # The four IDS StatVars declaring `includesIMFCredit = WithIMFCredit` are long-term
    # disbursements and principal repayments. The shared entry names a pair of them.
    _noun(
        "flows on long-term external debt with IMF credit",
        *_IDS_FLOW,
        *_WITH_IMF_CREDIT,
        # DT.DIS.DLTF.CD and DT.AMT.DLTF.CD
        definition=(
            "Disbursements or principal repayments on long-term external debt, counted "
            "together with purchases from or repurchases to the IMF."
        ),
    ),
    _noun(
        "disbursements on long-term external debt with IMF credit",
        *_IDS_FLOW,
        *_WITH_IMF_CREDIT,
        ("financialFlowType", "Disbursement"),
        # DT.DIS.DLTF.CD
        definition=(
            "Drawings by borrowers on long-term external loan commitments, together with "
            "purchases from the IMF outside the reserve tranche."
        ),
    ),
    _noun(
        "principal repayments on long-term external debt with IMF credit",
        *_IDS_FLOW,
        *_WITH_IMF_CREDIT,
        ("financialFlowType", "PrincipalRepayment"),
        # DT.AMT.DLTF.CD
        definition=(
            "Principal actually repaid on long-term external debt, together with repurchases "
            "of IMF credit outside the reserve tranche."
        ),
    ),
    _ids(
        "outstanding external debt",
        "FinancialPosition",
        "financialPositionType",
        "AmountOutstanding",
        # DT.DOD.DECT.CD: "debt owed to nonresidents repayable in currency, goods, or services"
        definition="Debt owed to non-residents and repayable in currency, goods or services.",
    ),
    _ids(
        "principal arrears on external debt",
        "FinancialPosition",
        "financialPositionType",
        "PrincipalArrears",
        # DT.AXA.DPPG.CD: "principal repayment due but not paid, on a cumulative basis"
        definition=(
            "Principal on external debt that has fallen due and not been paid, added up over "
            "all past years."
        ),
    ),
    _ids(
        "interest arrears on external debt",
        "FinancialPosition",
        "financialPositionType",
        "InterestArrears",
        # DT.IXA.DPPG.CD: "interest payment due but not paid, on a cumulative basis"
        definition=(
            "Interest on external debt that has fallen due and not been paid, added up over "
            "all past years."
        ),
    ),
    _ids(
        "undisbursed external debt",
        "FinancialPosition",
        "financialPositionType",
        "AmountUndisbursed",
        # DT.UND.DPPG.CD: "debt undrawn at year end"
        definition="External loans committed but not yet drawn at the end of the year.",
    ),
    # IMF World Economic Outlook.
    _noun(
        "GDP",
        (_POPULATION, "EconomicProduction"),
        (_PROPERTY, "valueAdded"),
        (_STAT_TYPE, "measuredValue"),
        *_GDP_BASIS,
        # NGDPD
        definition=_GDP_DEFINITION,
    ),
    _noun(
        "GDP",
        (_POPULATION, "EconomicProduction"),
        (_PROPERTY, "valueAdded"),
        (_STAT_TYPE, "measuredValue"),
        *_GDP_BASIS,
        ("currencyConversionBasis", "PurchasingPowerParityRate"),
        # PPPGDP
        definition=_GDP_DEFINITION,
    ),
    _noun(
        "output gap",
        (_POPULATION, "EconomicProduction"),
        (_PROPERTY, "outputGap"),
        (_STAT_TYPE, "measuredValue"),
        *_GDP_BASIS,
        # NGAP_NPGDP
        definition="The difference between an economy's actual output and its potential output.",
    ),
    _noun(
        "gross national savings",
        (_POPULATION, "EconomicUnit"),
        (_PROPERTY, "saving"),
        (_STAT_TYPE, "measuredValue"),
        ("fixedCapitalConsumptionBasis", "GrossOfFixedCapitalConsumption"),
        # NGSD_NGDP
        definition=(
            "The income that an economy's residents, including households, corporations and "
            "government, earn in a period and do not consume."
        ),
    ),
    _noun(
        "gross capital formation",
        (_POPULATION, "FinalExpenditure"),
        (_PROPERTY, "amount"),
        (_STAT_TYPE, "measuredValue"),
        ("finalExpenditureType", "CapitalFormation"),
        ("fixedCapitalConsumptionBasis", "GrossOfFixedCapitalConsumption"),
        # NID_NGDP
        definition=(
            "Acquisitions less disposals of produced assets for fixed capital formation, "
            "inventories or valuables."
        ),
    ),
    # Base Data Commons series that WEO questions elect (World Development Indicators).
    _noun(
        "GDP",
        (_POPULATION, "EconomicActivity"),
        (_PROPERTY, "amount"),
        (_STAT_TYPE, "measuredValue"),
        ("activitySource", "GrossDomesticProduction"),
        # NY.GDP.MKTP.CD
        definition=_GDP_DEFINITION,
    ),
    _noun(
        "GDP growth",
        (_POPULATION, "EconomicActivity"),
        (_PROPERTY, "amount"),
        (_STAT_TYPE, "growthRate"),
        ("activitySource", "GrossDomesticProduction"),
        # NY.GDP.MKTP.KD.ZG: "the percentage change over each previous year"
        definition="The change in gross domestic product from the year before, as a percentage.",
    ),
    _noun(
        "health expenditure",
        (_POPULATION, "EconomicActivity"),
        (_PROPERTY, "amount"),
        (_STAT_TYPE, "measuredValue"),
        ("activitySource", "ExpenditureActivity"),
        ("expenditureType", "HealthcareExpenditure"),
        # SH.XPD.CHEX.PC.CD
        definition="Current spending on healthcare goods and services consumed during the year.",
    ),
    _noun(
        "government debt",
        (_POPULATION, "Debt"),
        (_PROPERTY, "amount"),
        (_STAT_TYPE, "measuredValue"),
        ("debtor", "Government"),
        # GC.DOD.TOTL.CN
        definition=(
            "The stock of a central government's direct fixed-term contractual obligations to "
            "others outstanding on a date, including currency and deposits, securities other "
            "than shares, and loans."
        ),
    ),
)
# A birth in demographic statistics is a live birth, and the crude birth rate is live births
# per person. A births question that names no status reads live births.
SHAPE_SLOT_DEFAULTS: tuple[ShapeSlotDefault, ...] = (
    ShapeSlotDefault(
        population_type="BirthEvent", property_dcid="medicalStatus", value_dcid="LiveBirth"
    ),
)

# Life expectancy is life expectancy at birth, and gross domestic product is the product of
# all economic activities. The headline series of each states nothing on the property, and
# the series stating the value covers fewer years (`LifeExpectancy_Person_0Years` runs 1980
# to 2020). The export refuses a declaration no StatVar of the measure states
# (`code_index.export.verify_implied_values`).
IMPLIED_VALUES: tuple[ImpliedValue, ...] = (
    ImpliedValue(
        population_type="Person",
        measured_property="lifeExpectancy",
        property_dcid="age",
        value_dcid="Years0",
    ),
    ImpliedValue(
        population_type="EconomicActivity",
        measured_property="amount",
        property_dcid="economicSector",
        value_dcid="AllEconomicActivities",
    ),
)

# The lender-named IDS debt series state financingResource=DebtInstrument and the debt totals
# state no financingResource, though both hold debt instruments. The export refuses a
# declaration with no five-tuple where the dataset has a row stating the value and a row
# stating nothing on the property (`code_index.loader._check_partly_stated_values`).
PARTLY_STATED_VALUES: tuple[PartlyStatedValue, ...] = (
    PartlyStatedValue(
        provenance=_IDS,
        property_dcid="financingResource",
        value_dcid="DebtInstrument",
    ),
)

# The words naming each measure class, read whole-word from a lowercase phrase or label.
# Incidence also reads from "new" followed within three tokens by "infection(s)" or
# "case(s)", as in "Number of new HIV infections per 1,000 uninfected population" and
# "Number Of New Reported Cases Of ...". A bare "cases" names no class: most "Number of X
# Cases" series are notified or cumulative counts. "Living with" names a population ("people
# living with HIV"), which is a prevalence only when no measure word names another class:
# "Deaths among people living with HIV" is a mortality.
MEASURE_CLASSES: tuple[MeasureClassDeclaration, ...] = (
    MeasureClassDeclaration("prevalence", re.compile(r"\bprevalence\b")),
    MeasureClassDeclaration(
        "incidence",
        re.compile(r"\bincidence\b|\bnew(?:\s+\S+){0,3}?\s+(?:infections?|cases?)\b"),
    ),
    MeasureClassDeclaration(
        "mortality", re.compile(r"\b(?:mortality|deaths|death\s+rates?|dying)\b")
    ),
    MeasureClassDeclaration("prevalence", re.compile(r"\bliving\s+with\b"), fallback=True),
)

# The words naming a statType other than measuredValue, read whole-word from a lowercase
# variable phrase: "GDP growth" asks for the growth rate.
STAT_TYPE_WORDS: tuple[StatTypeWordDeclaration, ...] = (
    StatTypeWordDeclaration(re.compile(r"\bgrowth\b"), "growthRate"),
)

# Acronyms recall confuses, expanded in the variable query before recall. GNI scores 0.6342 for
# GiniIndex_EconomicActivity. Its full term finds the aggregate. U5MR and IMR hit unrelated
# variables, while their full terms find the mortality variables. CO2's full term scores higher
# than the acronym. GDP recalls correctly without expansion. PPG recalls no variable for "debt
# service for PPG debt"; its full term finds the public and publicly guaranteed debt service
# series.
VARIABLE_ALIASES: Mapping[str, str] = {
    "gni": "gross national income",
    "co2": "carbon dioxide emissions",
    "u5mr": "under five mortality rate",
    "imr": "infant mortality rate",
    "ppg": "public and publicly guaranteed",
}
