"""Regenerate `code_index/data/index.json.gz` from the live graph.

Four CRS code systems, `dac2aAidType`, `dac1AidType` and `dac2bAidType` use `<-typeOf`
enumeration. `crsPurpose` has 289 codes. Each declared dataset's StatVars are enumerated through
a system every StatVar of that dataset declares, and the union is checked against complete
non-spine reads. This graph repeats reverse-arc pages, so truncated reads split by code axes and
observed `populationType`, to depth 6. A second read of each exported five-tuple refuses the export when the graph holds a
StatVar the index would decide that is not an exported row.

A dataset whose StatVars declare no code system, World Bank IDS, cannot be enumerated that way.
Its StatVars come from the coverage build into a separate `listed_statvars` block, which no
family, measure axis or shape reads; Search and the mint elect members from it in place of a
graph walk (`code_index.members.listed_group`). An unsplit read of each five-tuple a listed row
has refuses the export when the graph holds a StatVar there that is neither an exported nor a
listed row.

Each listed StatVar's `sourceIndicatorCode` arcs give the `source_codes` block: the StatVars each
publisher's series code names under the provenance the arc carries, so search can answer a code
such as `DT.TDS.DEGG.CD` with its variables. A listed StatVar with no code refuses the export, and
so does a code whose arc provenance the coverage build does not give its StatVar, since the
finished document must load (`code_index.loader._check_source_codes`).

The `specializationOf` arc of each constraint value a listed StatVar holds gives the
`value_parents` block, so the sentence renderer knows a general government debtor
(`IDSDebtorTypeGG`) is a public-sector one (`IDSDebtorTypePS`) without a graph read. A value with
two parents refuses the export.

Pass 1 reads each StatVar's type, shape, and declared constraints. Pass 2 reads their values.
Export requires held, resolved targets with `typeOf` and one value per single-value field. Nodes
typed only `ProvisionalNode` and non-StatisticalVariables raise. Constraint and observation
properties are list-valued. Export validates rows before the mint door trusts them.

Authored config supplies system declarations, defaults, measure axes and labels, the IDS and WEO
measure nouns, synonyms, dataset declarations, the shape slot defaults, the constraint values a
dataset states on only some of its series, the measure class patterns, the variable aliases
recall expands, and the words naming a stat type. Graph reads supply codes, StatVars, and measure
values on those rows. Every observed measure partition needs one authored label.

The published series coverage build supplies each StatVar's provenances: `coverage.parquet` maps a
variable to its facet ids and `facets.json` maps a facet id to its provenance. The stamp records
the build id and the number of StatVars the build holds no series for. A declared dataset with no
StatVar row aborts the export.

The same build names every donor and recipient of a declared or listed dataset. Each party's graph
types give it a kind (party, total or residual) under the authored `PARTY_TYPE_KINDS`, except a
party the authored `PARTY_KIND_OVERRIDES` names. Its graph name is its label, and its row lists the
datasets that publish it and the flow axes they publish it on. The authored `TOTAL_GROUPS` names the group each total holds, whether
the total equals the sum of the group's members plus the residuals it counts beside them, and those
residuals. A party with an undeclared type, or whose types
give no single kind, aborts the export, as does a `TOTAL_GROUPS` or `PARTY_KIND_OVERRIDES` row
naming a party no declared or listed dataset publishes (a `TOTAL_GROUPS` row also needs that party to be a
total, and each residual it names to be a published residual).

Requires `SEARCH_GRAPH_BASE` and `COVERAGE_ARTIFACT_URI`, with read access to the coverage bucket
for a `gs://` URI. Sorted JSON is stable. `content_hash` covers data blocks and excludes the stamp.

Usage:
    COVERAGE_ARTIFACT_URI=gs://<coverage bucket>/ uv run --env-file .env \\
        python scripts/export_code_index.py \\
        --out packages/dfo-data-tools/src/code_index/data/index.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from code_index.loader import IndexFormatError, parse_index, parse_measure_axes
from code_index.members import key_sets_cover
from code_index.tokens import Token, TokenLookup, singular, tokenize
from code_index.types import (
    DatasetDeclaration,
    DeclaredDefault,
    MeasureAxis,
    MeasureClassDeclaration,
    MeasurePartitionLabel,
    PartlyStatedValue,
    PartyKind,
    ShapeSlotDefault,
    StatedWord,
    StatTypeWordDeclaration,
    invalid_when_named_entry,
    label_for_pairs,
    party_kind,
)
from dc_graph import GraphClient
from dfo_platform.roles import RoleAxis
from materialise.graph import (
    UNRESOLVED_NODE,
    GraphNotAdvancingError,
    GraphReader,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = (
    _REPO_ROOT / "packages" / "dfo-data-tools" / "src" / "code_index" / "data" / "index.json.gz"
)

# Truncated code-system enumeration aborts. Partition reads use truncation as a split signal.
_DEFAULT_MAX_RESULTS = 5000
# Subjects one unsplit read of a listed row's five-tuple may return
# (`_verify_listed_five_tuples`). The dev graph's `FinancialFlow` amount five-tuple holds about
# 14,000 StatVars, CRS, DAC and IDS together, and reads whole in about 45 s.
_UNSPLIT_MAX_RESULTS = 40_000
# Twenty-dcid `/v2/node` requests avoid mixer paging loops. GraphClient reads chunks concurrently.
_ARC_CHUNK_SIZE = 20
# Six levels cover every partition axis and bound recursive splitting.
_MAX_PARTITION_DEPTH = 6

_SOURCE_CODE = "sourceIndicatorCode"

_LABEL_SUFFIX_RE = re.compile(r"\s*\((?:CRS|DAC)\s[^()]*\)\s*$")

_CODE_ARC_PROPERTIES = (
    "classificationCode",
    "name",
    "alternateName",
    "shortName",
    "specializationOf",
    "description",
)
# `export_statvars` rejects multiple distinct values on these single-valued fields.
_STATVAR_SHAPE_PROPERTIES = (
    "populationType",
    "measuredProperty",
    "statType",
    "measurementQualifier",
    "measurementDenominator",
)
_STATVAR_PASS1_PROPERTIES = (
    "typeOf",
    "constraintProperties",
    *_STATVAR_SHAPE_PROPERTIES,
    "observationProperties",
)


@dataclass(frozen=True)
class CodeSystemConfig:
    """One code system's graph enum and semantic declarations.

    `enum_dcid` is the enumeration the system's codes are `typeOf`; the declaration carries it
    as the system's graph id. `hierarchical` is derived from graph rows and checked against the expected value.

    `declared_by_every_statvar_of` lists the provenances of the datasets whose every StatVar
    declares this property. CRS StatVars all declare `crsPurpose`, `crsTypeOfFinance`, and
    `crsTypeOfFlow`; imputed multilateral StatVars all declare `crsPurpose`, `crsTypeOfFlow`, and
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
    # Imputed multilateral and DAC2a StatVars carry `dac2aAidType`; CRS StatVars do not. DAC2a
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
# is: the CRS import leaves out the rows that record a core contribution to a multilateral
# organisation. So the CRS states "bilateral" of both measured properties it publishes
# (`DatasetDeclaration.stated_words`). A DAC2a or DAC2b `amount` row is bilateral only where
# its recipient is a country, since a flow to a multilateral organisation is a core
# contribution, so those tables state it where the destination is a country. DAC2a's
# `attributedAmount` row, imputed multilateral aid, and DAC1, whose aid types split bilateral
# from multilateral, state nothing.
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
    ),
)


# The kind each graph type gives a party of a declared dataset (`code_index.types.party_kind`).
# Every type a donor or recipient of a declared or listed dataset carries is named here; the export refuses
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


@dataclass(frozen=True)
class PartyPublication:
    """The datasets that publish one party, by provenance, and the flow axes they publish it
    on: `origin` as a donor, `destination` as a recipient."""

    datasets: frozenset[str]
    axes: frozenset[RoleAxis]


@dataclass(frozen=True)
class CoverageJoin:
    """What the export takes from one series coverage build: its build id, each covered
    variable's sorted provenances, and the donors and recipients of the declared and listed
    datasets."""

    build_id: str
    provenances: Mapping[str, tuple[str, ...]]
    parties: Mapping[str, PartyPublication] = field(default_factory=dict)


def load_coverage_join(base_uri: str, *, datasets: Collection[str]) -> CoverageJoin:
    """Load the published coverage build under `base_uri` and read its variable provenances and
    the parties on the facets of the `datasets` provenances.

    `series_coverage` needs the `coverage` extra, which the graph-only callers of this module
    run without, so the import stays inside this function.
    """
    from series_coverage.loader import load_coverage

    loaded = load_coverage(base_uri)
    publishers: dict[str, set[str]] = defaultdict(set)
    axes: dict[str, set[RoleAxis]] = defaultdict(set)
    for dataset in datasets:
        for party, published_on in loaded.index.dataset_parties({dataset}).items():
            publishers[party].add(dataset)
            axes[party].update(published_on)
    return CoverageJoin(
        build_id=loaded.manifest.build_id,
        provenances=loaded.index.variable_provenances(),
        parties={
            party: PartyPublication(datasets=frozenset(found), axes=frozenset(axes[party]))
            for party, found in publishers.items()
        },
    )


def _kind_from_types(
    dcid: str, types: Collection[str], type_kinds: Mapping[str, PartyKind | None]
) -> PartyKind:
    """The kind `types` give the party `dcid` under `type_kinds`, or `SystemExit` naming why they
    give none."""
    if not types:
        raise SystemExit(f"party {dcid!r} has no typeOf arc -- its kind cannot be declared")
    undeclared = sorted(set(types) - type_kinds.keys())
    if undeclared:
        raise SystemExit(
            f"party {dcid!r} carries type(s) {undeclared!r} that PARTY_TYPE_KINDS does not "
            "declare -- add each with the kind it gives a party"
        )
    kind = party_kind(types, type_kinds)
    if kind is None:
        raise SystemExit(
            f"party {dcid!r}'s types {sorted(types)!r} give it no single kind under "
            "PARTY_TYPE_KINDS"
        )
    return kind


def export_parties(
    reader: GraphReader,
    parties: Mapping[str, PartyPublication],
    *,
    type_kinds: Mapping[str, PartyKind | None],
    kind_overrides: Mapping[str, KindOverride],
    total_groups: Mapping[str, TotalGroup],
) -> list[dict]:
    """Each party's kind, label, the datasets that publish it and the flow axes they publish it
    on (`parties`, each sorted), and the group and residuals a total holds, sorted by dcid.

    A party in `kind_overrides` takes its override's kind; any other takes the kind its graph
    types give under `type_kinds`. Its label is its graph name, `None` when the graph gives it
    none.

    Raises `SystemExit` for a `kind_overrides` row naming a party not in `parties`, a party the
    graph holds no node or type for, a party carrying a type `type_kinds` does not name, a party
    whose types give it no single kind, a `total_groups` row naming a party that is not a total
    of `parties`, and a `total_groups` residual that is not a residual of `parties`.
    """
    published = frozenset(parties)
    unpublished = sorted(kind_overrides.keys() - published)
    if unpublished:
        raise SystemExit(
            f"PARTY_KIND_OVERRIDES names {unpublished!r}, which no declared or listed dataset publishes"
        )
    arcs = _fetch_arcs_many(reader, sorted(published), properties=("typeOf", "name"))
    rows = []
    for dcid in sorted(published):
        party_arcs = arcs.get(dcid) or {}
        if (override := kind_overrides.get(dcid)) is not None:
            kind = override.kind
        else:
            kind = _kind_from_types(dcid, party_arcs.get("typeOf", ()), type_kinds)
        total = total_groups.get(dcid)
        rows.append(
            {
                "dcid": dcid,
                "kind": kind,
                "label": _first_or_none(party_arcs.get("name", ())),
                "totals": None if total is None else total.group,
                "residuals": [] if total is None else sorted(total.residuals),
                "sums_members": True if total is None else total.sums_members,
                "datasets": sorted(parties[dcid].datasets),
                "axes": sorted(parties[dcid].axes),
            }
        )
    kinds = {row["dcid"]: row["kind"] for row in rows}
    stale = sorted(dcid for dcid in total_groups if kinds.get(dcid) != "total")
    if stale:
        raise SystemExit(
            f"TOTAL_GROUPS names {stale!r}, which no declared or listed dataset publishes as a total"
        )
    not_residuals = sorted(
        residual
        for total in total_groups.values()
        for residual in total.residuals
        if kinds.get(residual) != "residual"
    )
    if not_residuals:
        raise SystemExit(
            f"TOTAL_GROUPS counts {not_residuals!r} as residuals, which no declared or listed dataset "
            "publishes as a residual"
        )
    return rows


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
class _MeasureValueText:
    """Authored text overrides for one measure-value dcid.

    `label` replaces the graph name. `synonyms` add to `alternateName`. `matchable` controls text
    lookup. `amount` keeps a display label but skips text indexing because queries use it as a
    determiner. `same_reading_as` names the canonical value of an alias dcid; query text naming
    the alias resolves to the canonical value, so authored synonyms belong on the canonical one.
    """

    label: str | None = None
    synonyms: tuple[str, ...] = ()
    matchable: bool = True
    same_reading_as: str | None = None


# Text overrides merge into measure rows.
# Unmapped dcids keep graph names and aliases, and stay matchable.
_MEASURE_VALUE_TEXT: Mapping[str, _MeasureValueText] = {
    "Nominal": _MeasureValueText(
        label="nominal",
        synonyms=("current prices", "at current prices", "current US dollars"),
    ),
    # "constant" alone covers "constant <year> prices" and "constant <year> euros", since
    # matching reads exact text and a year is not a fixed token.
    "RealValue": _MeasureValueText(
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
    "InflationAdjusted": _MeasureValueText(label="inflation adjusted", same_reading_as="RealValue"),
    "PurchasingPowerParity": _MeasureValueText(label="purchasing power parity", synonyms=("PPP",)),
    "amount": _MeasureValueText(matchable=False),
    # These phrases name the imputed tables' measure as "grant equivalent" names CRS's. The
    # bare verb would read in any question using "attributed to" ("deaths attributed to air
    # pollution") and warn that it was not applied.
    "attributedAmount": _MeasureValueText(synonyms=("attributed aid", "attributed to donors")),
}

# Values exported whether or not a CRS StatVar row carries them. The price bases below are
# carried by World Bank, IMF and DC-native series outside the index, and a query names them.
_DECLARED_MEASURE_VALUES: Mapping[str, tuple[str, ...]] = {
    "measurementQualifier": ("InflationAdjusted", "PurchasingPowerParity", "RealValue"),
}

# Extra aliases supplement graph `alternateName`. Unmapped dcids keep graph names and aliases.
# `development assistance` and `other official flows` fill terms the graph aliases miss.
# Qualifier matching uses text alone, so `aid` would bind ODA for any phrase containing it.
# `when_named=("aid",)` handles `aid` as the extracted variable when the flow slot is open and
# qualifier matching leaves the slot unmatched or ambiguous.
# "Imputed" names DAC2a's imputed multilateral line when the query holds no longer label of it
# ("imputed aid from Norway"), so DAC1's donor totals, which read nothing, do not cover it.
_CODE_SYNONYMS: Mapping[str, tuple[str, ...]] = {
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
_MEASURE_WORD_SYNONYMS_KEPT: frozenset[str] = frozenset(
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
_CONCEPT_GROUPS: tuple[tuple[tuple[str, str], ...], ...] = (
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
_CODE_SHORT_NAMES: Mapping[str, str] = {
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
# partition; `label_for_pairs` picks the most specific label, so a family-level label never
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
_MEASURE_PARTITION_LABELS: tuple[MeasurePartitionLabel, ...] = (
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
    # label; a gross or net partition matches its own, more specific label.
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
    # Only DAC1 StatVars hold `amountMobilised`: private finance that official flows mobilise.
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

_MEASURE_NOUNS: tuple[MeasurePartitionLabel, ...] = (
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


def _graph_base_url() -> str:
    base = os.environ.get("SEARCH_GRAPH_BASE", "").strip()
    if not base:
        raise SystemExit(
            "SEARCH_GRAPH_BASE must be set -- this script always talks to the live graph"
        )
    return base


def _coverage_artifact_uri() -> str:
    uri = os.environ.get("COVERAGE_ARTIFACT_URI", "").strip()
    if not uri:
        raise SystemExit(
            "COVERAGE_ARTIFACT_URI must be set -- the export joins each StatVar's provenances "
            "from the published coverage build"
        )
    return uri


def strip_system_suffix(name: str) -> str:
    """Strip a trailing `(CRS ...)` or `(DAC ...)` suffix from a name.

    Preserve other parentheses, including interior groups and `(malaria (severe))`.
    """
    stripped = _LABEL_SUFFIX_RE.sub("", name)
    return stripped if stripped else name


def _single_value(dcid: str, prop: str, values: tuple[str, ...]) -> str:
    if not values:
        raise SystemExit(f"{dcid} carries no {prop} arc -- cannot export a row missing it")
    return values[0]


def _check_single_valued(dcid: str, prop: str, values: tuple[str, ...]) -> None:
    """Reject multiple distinct values on `prop`.

    `parse_arcs` collapses repeated imports. Callers apply this to shape and constraint fields.
    """
    if len(values) > 1:
        raise SystemExit(
            f"{dcid}'s {prop!r} arc carries {len(values)} bound values, not one: {values!r} -- "
            "cannot export a row that would silently pick one of several"
        )


def _check_typed_as_statvar(dcid: str, type_of: tuple[str, ...]) -> None:
    """Require `StatisticalVariable` in `typeOf`.

    `ProvisionalNode` alongside a real type is allowed.
    """
    if "StatisticalVariable" not in type_of:
        raise SystemExit(
            f"{dcid}'s typeOf arc {type_of!r} does not include StatisticalVariable -- cannot "
            "export a row the graph does not hold as a StatisticalVariable"
        )


class ExportReader(GraphReader, Protocol):
    """The graph reads the export makes: `GraphReader`'s, and the full forward read whose
    values carry the provenance that asserted them, which `export_source_codes` needs."""

    def node_arcs_batch(self, dcids: list[str]) -> Mapping[str, dict | None]:
        """Each dcid's forward arcs as the graph sends them, `None` for an absent node.
        See `dc_graph.GraphClient.node_arcs_batch`."""
        ...


def export_reader(base_url: str, token: str | None) -> GraphClient:
    """Create the export's GraphClient with its configured chunk size.

    `base_url` is the `/core/api` root.
    """
    return GraphClient(base_url=base_url, token=token, node_arc_chunk_size=_ARC_CHUNK_SIZE)


def _first_or_none(values: tuple[str, ...]) -> str | None:
    return values[0] if values else None


def _fetch_arcs_many(
    reader: GraphReader, dcids: Sequence[str], *, properties: Sequence[str]
) -> dict[str, dict[str, tuple[str, ...]] | None]:
    """Read all dcids in one call.

    GraphClient chunks them at `_ARC_CHUNK_SIZE`.
    """
    return reader.node_arcs_many(dcids, properties=list(properties))


def _build_code_row(dcid: str, code_system: str, arcs: dict[str, tuple[str, ...]]) -> dict:
    code = _single_value(dcid, "classificationCode", arcs.get("classificationCode", ()))
    name = _single_value(dcid, "name", arcs.get("name", ()))
    label = strip_system_suffix(name)
    synonyms = sorted(
        {v for v in arcs.get("alternateName", ()) if v and v != label}
        | set(_CODE_SYNONYMS.get(dcid, ()))
    )
    parent = _first_or_none(arcs.get("specializationOf", ()))
    short_name = _first_or_none(arcs.get("shortName", ())) or _CODE_SHORT_NAMES.get(dcid)
    return {
        "dcid": dcid,
        "code_system": code_system,
        "code": code,
        "label": label,
        "synonyms": synonyms,
        "short_name": short_name,
        "parent": parent,
        "definition": _first_or_none(arcs.get("description", ())),
    }


def _apply_concept_groups(
    codes: Sequence[dict],
    groups: Sequence[Sequence[tuple[str, str]]],
    exported_systems: Collection[str],
) -> None:
    """Give every code row of each of `groups` the union of the group's labels and synonyms
    as its synonyms, its own label left out. A member of a code system the export does not
    enumerate (`exported_systems`) is skipped.

    Raises:
        SystemExit: a code is in two groups, a group joins two codes of one code system, or
            a group names a code of an exported system that the export holds no row of.
    """
    by_dcid = {row["dcid"]: row for row in codes}
    repeated = sorted(
        dcid
        for dcid, count in Counter(dcid for group in groups for _system, dcid in group).items()
        if count > 1
    )
    if repeated:
        raise SystemExit(f"concept groups name codes in more than one group: {repeated}")
    for group in groups:
        systems = [system for system, _dcid in group]
        if len(set(systems)) != len(systems):
            raise SystemExit(f"concept group {group!r} joins two codes of one code system")
        members = [(system, dcid) for system, dcid in group if system in exported_systems]
        missing = [
            dcid
            for system, dcid in members
            if dcid not in by_dcid or by_dcid[dcid]["code_system"] != system
        ]
        if missing:
            raise SystemExit(
                f"concept group {group!r} names codes the export holds no row of: {missing}"
            )
        rows = [by_dcid[dcid] for _system, dcid in members]
        texts = {text for row in rows for text in (row["label"], *row["synonyms"])}
        for row in rows:
            row["synonyms"] = sorted(text for text in texts if text != row["label"])


type _MeasureNames = frozenset[tuple[str, str]]


def _measure_texts(measure_values: Sequence[dict]) -> TokenLookup[_MeasureNames]:
    """Every matchable measure-value label and synonym, found as the matcher finds it
    (`code_index.tokens`), to the `(axis, dcid)` pairs it reads. A text names its value's
    whole reading: the canonical dcid and every alias dcid filed under it
    (`same_reading_as`), as the matcher files them."""
    reading: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in measure_values:
        reading[(row["axis"], row["same_reading_as"] or row["dcid"])].add(row["dcid"])
    texts: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in measure_values:
        if not row["matchable"]:
            continue
        canonical = (row["axis"], row["same_reading_as"] or row["dcid"])
        for text in (row["label"], *row["synonyms"]):
            texts[text.strip().casefold()].update(
                (row["axis"], dcid) for dcid in reading[canonical]
            )
    return TokenLookup((text, frozenset(names)) for text, names in texts.items())


def _measure_names(
    span: Sequence[Token] | None, measure_texts: TokenLookup[_MeasureNames]
) -> _MeasureNames:
    """The `(axis, dcid)` pairs every measure text matching `span` reads."""
    if span is None:
        return frozenset()
    return frozenset(pair for names in measure_texts.get(span) for pair in names)


def _measure_runs(
    text: str, measure_texts: TokenLookup[_MeasureNames]
) -> list[tuple[int, int, _MeasureNames]]:
    """The runs of `text`'s words a query holding `text` would lose to it, as `(start, end,
    names)` word ranges, left to right and longest first at each start.

    The matcher (`search.engine.qualifier_match`) reads a query as `code_index.tokens` words
    and separators, and a query holding `text` matches it whatever separators `text` holds
    ("ODA: Total Net"). A matching `text` claims its every word, so it takes each run of its
    words that a measure text matches, exact or less one trailing "s". A `text` that is
    itself a measure text ("PPP") shares its span with the measure value, which keeps both
    readings (`match_qualifier_readings`), and takes nothing.
    """
    tokens = tokenize(text)
    word_at = [position for position, token in enumerate(tokens) if token.kind == "word"]
    if not word_at or _measure_names(tokens, measure_texts):
        return []
    runs: list[tuple[int, int, _MeasureNames]] = []
    start = 0
    while start < len(word_at):
        for end in range(len(word_at), start, -1):
            span = tokens[word_at[start] : word_at[end - 1] + 1]
            names = _measure_names(span, measure_texts) or _measure_names(
                singular(span), measure_texts
            )
            if names:
                runs.append((start, end, names))
                start = end
                break
        else:
            start += 1
    return runs


def _strip_measure_words(
    codes: Sequence[dict],
    statvars: Sequence[dict],
    measure_values: Sequence[dict],
    measure_axes: Sequence[MeasureAxis],
) -> None:
    """Take the measure words out of every code synonym that would take them from a query.

    Measure values are read in every code system together, and the matcher claims the longest
    span first, so a code synonym holding "net" or "gross" would take the word from the netting
    basis in every vocabulary: "gross ODA" would read the ODA code and no basis
    (`_measure_runs`). Such a synonym keeps the rest of its words, so "Gross ODA Loans" becomes
    "ODA Loans": a query reads the code in the remainder and the measure in the word. A run
    stays when every series carrying any code that holds the text has the measure it names,
    since the text then implies it ("ODA grant equivalents of bilateral loans" on a code whose
    one series is a grant equivalent). The test covers every code holding the text, because
    the text takes the words in every vocabulary, and codes of a concept group must keep
    reading one set of texts. A synonym left with no words is dropped, and a remainder the code
    already reads adds nothing. A label stays, since it is the code's own name, and so does a
    text in `_MEASURE_WORD_SYNONYMS_KEPT`.
    """
    measure_texts = _measure_texts(measure_values)
    axes = {axis.axis: axis for axis in measure_axes}
    series_by_code: dict[str, list[dict]] = defaultdict(list)
    for statvar in statvars:
        for value in statvar["constraint_properties"].values():
            series_by_code[value].append(statvar)

    codes_by_text: dict[str, list[str]] = defaultdict(list)
    for row in codes:
        for text in {text.strip().casefold() for text in (row["label"], *row["synonyms"])}:
            codes_by_text[text].append(row["dcid"])

    def implied(text: str, names: _MeasureNames) -> bool:
        series = [
            statvar
            for dcid in codes_by_text[text.strip().casefold()]
            for statvar in series_by_code[dcid]
        ]
        return bool(series) and all(
            any(
                axis in axes and _measure_axis_value(statvar, axes[axis]) == value
                for axis, value in names
            )
            for statvar in series
        )

    for row in codes:
        kept: list[str] = []
        remainders: list[str] = []
        for text in row["synonyms"]:
            stripped = [
                (start, end)
                for start, end, names in _measure_runs(text, measure_texts)
                if not implied(text, names)
            ]
            if not stripped or text.strip().casefold() in _MEASURE_WORD_SYNONYMS_KEPT:
                kept.append(text)
                continue
            words = [token.text for token in tokenize(text) if token.kind == "word"]
            remainders.append(
                " ".join(
                    word
                    for position, word in enumerate(words)
                    if not any(start <= position < end for start, end in stripped)
                )
            )
        known = {text.casefold() for text in (row["label"], *kept)}
        for remainder in remainders:
            if remainder and remainder.casefold() not in known:
                kept.append(remainder)
                known.add(remainder.casefold())
        row["synonyms"] = sorted(kept)


def _reject_cross_system_parents(
    config: CodeSystemConfig, enumerated: set[str], rows: Sequence[dict]
) -> None:
    """Reject parents outside this code system's enumeration with `SystemExit`.

    `specializationOf` must stay within one vocabulary. Parent and child dcids are already in hand,
    so this check needs no graph read.
    """
    for row in rows:
        parent = row["parent"]
        if parent is not None and parent not in enumerated:
            raise SystemExit(
                f"{row['dcid']} in code system {config.code_system!r} has parent {parent!r}, "
                f"which {config.enum_dcid} does not enumerate -- a code row's parent must be a "
                "code of its own code system"
            )


# A birth in demographic statistics is a live birth, and the crude birth rate is live births
# per person. A births question that names no status reads live births.
SHAPE_SLOT_DEFAULTS: tuple[ShapeSlotDefault, ...] = (
    ShapeSlotDefault(
        population_type="BirthEvent", property_dcid="medicalStatus", value_dcid="LiveBirth"
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
# GiniIndex_EconomicActivity; its full term finds the aggregate. U5MR and IMR hit unrelated
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


def _shape_slot_default_to_dict(default: ShapeSlotDefault) -> dict:
    return {
        "population_type": default.population_type,
        "property": default.property_dcid,
        "value": default.value_dcid,
    }


def _partly_stated_value_to_dict(declared: PartlyStatedValue) -> dict:
    return {
        "provenance": declared.provenance,
        "property": declared.property_dcid,
        "value": declared.value_dcid,
    }


def _measure_class_to_dict(declaration: MeasureClassDeclaration) -> dict:
    return {
        "class": declaration.measure_class,
        "pattern": declaration.pattern.pattern,
        "fallback": declaration.fallback,
    }


def _stat_type_word_to_dict(declaration: StatTypeWordDeclaration) -> dict:
    return {"pattern": declaration.pattern.pattern, "stat_type": declaration.stat_type}


def export_code_system(
    reader: GraphReader, config: CodeSystemConfig, *, max_results: int = _DEFAULT_MAX_RESULTS
) -> tuple[dict, list[dict]]:
    """Return a declaration and rows for `config`'s code system.

    Derive `hierarchical` from parent arcs and reject disagreement with the configured value or
    parents outside the enumerated system.
    """
    page = reader.node_subjects(config.enum_dcid, property="typeOf", max_results=max_results)
    if page.truncated:
        raise SystemExit(
            f"{config.code_system} ({config.enum_dcid}) population truncated at "
            f"{max_results} dcids -- raise max_results, the export must read the full "
            "code system"
        )
    dcids = sorted(set(page.dcids))

    arcs_by_dcid = _fetch_arcs_many(reader, dcids, properties=_CODE_ARC_PROPERTIES)
    rows = []
    for dcid in dcids:
        arcs = arcs_by_dcid.get(dcid)
        if arcs is None:
            raise SystemExit(
                f"{dcid} was enumerated via <-typeOf on {config.enum_dcid} but the graph "
                "holds no node for it"
            )
        rows.append(_build_code_row(dcid, config.code_system, arcs))
    rows.sort(key=lambda row: row["dcid"])
    _reject_cross_system_parents(config, set(dcids), rows)

    derived_hierarchical = any(row["parent"] is not None for row in rows)
    if derived_hierarchical != config.expected_hierarchical:
        raise SystemExit(
            f"{config.code_system}: derived hierarchical={derived_hierarchical} disagrees "
            f"with the declared expected value {config.expected_hierarchical}"
        )

    declaration = {
        "code_system": config.code_system,
        "hierarchical": derived_hierarchical,
        "disjoint_from_children": config.disjoint_from_children,
        "summable": config.summable,
        "absent_reading": config.absent_reading,
        "plain_name": config.plain_name,
        "default": _declared_default_to_dict(config.default),
        "enum_dcid": config.enum_dcid,
        "no_vocabulary_terms": list(config.no_vocabulary_terms),
    }
    return declaration, rows


def _declared_default_to_dict(default: DeclaredDefault | None) -> dict | None:
    if default is None:
        return None
    return {
        "code": default.code,
        "when_bound_under": list(default.when_bound_under),
        "when_named": list(default.when_named),
        "when_open": default.when_open,
        "note": default.note,
    }


def _validate_declared_defaults(
    code_systems: Sequence[CodeSystemConfig], code_rows_by_system: dict[str, list[dict]]
) -> None:
    """Validate each default against exported rows, raising `SystemExit` on invalid entries.

    Its code must belong to its own system. `when_bound_under` codes must belong to other systems.
    `when_named` entries are non-empty lowercase strings. A default that fires on an open slot
    (`when_open`) carries a note.
    """
    dcids_by_system = {
        system: {row["dcid"] for row in rows} for system, rows in code_rows_by_system.items()
    }
    for config in code_systems:
        default = config.default
        if default is None:
            continue
        if default.code not in dcids_by_system.get(config.code_system, set()):
            raise SystemExit(
                f"{config.code_system}: declared default code {default.code!r} is not a row "
                "of that code system"
            )
        for scoping_dcid in default.when_bound_under:
            owning_system = next(
                (system for system, dcids in dcids_by_system.items() if scoping_dcid in dcids),
                None,
            )
            if owning_system is None:
                raise SystemExit(
                    f"{config.code_system}: declared default when_bound_under code "
                    f"{scoping_dcid!r} is not a row of any exported code system"
                )
            if owning_system == config.code_system:
                raise SystemExit(
                    f"{config.code_system}: declared default when_bound_under code "
                    f"{scoping_dcid!r} belongs to its own code system -- a default cannot "
                    "condition on the slot it defaults"
                )
        invalid = invalid_when_named_entry(default.when_named)
        if invalid is not None:
            raise SystemExit(
                f"{config.code_system}: declared default when_named entry "
                f"{default.when_named[invalid]!r} must be a non-empty lowercase string"
            )
        if default.when_open and not (default.note and default.note.strip()):
            raise SystemExit(
                f"{config.code_system}: declared default {default.code!r} fires on an open "
                "slot and carries no note"
            )


@dataclass(frozen=True)
class _PartitionAxis:
    """A filter axis and its split values.

    `None` uses the run-wide observed-value pool.
    """

    property: str
    values: tuple[str, ...] | None


def _complete_systems(
    dataset: DatasetDeclaration, code_systems: Sequence[CodeSystemConfig]
) -> list[CodeSystemConfig]:
    """Return the code systems every StatVar of `dataset` declares."""
    return [
        config
        for config in code_systems
        if dataset.provenance in config.declared_by_every_statvar_of
    ]


def _choose_spine(
    dataset: DatasetDeclaration,
    code_systems: Sequence[CodeSystemConfig],
    code_rows_by_system: dict[str, list[dict]],
) -> CodeSystemConfig:
    """Return the code system `dataset`'s StatVars are enumerated through, or raise if none
    qualifies: the largest system every one of its StatVars declares.
    """
    eligible = _complete_systems(dataset, code_systems)
    if not eligible:
        raise SystemExit(
            f"no code system in the table is declared by every StatVar of dataset "
            f"{dataset.provenance!r} -- cannot choose an enumeration spine"
        )
    return max(eligible, key=lambda config: len(code_rows_by_system[config.code_system]))


def _partition_axes(
    spine: CodeSystemConfig,
    dataset: DatasetDeclaration,
    code_systems: Sequence[CodeSystemConfig],
    code_rows_by_system: dict[str, list[dict]],
) -> tuple[_PartitionAxis, ...]:
    """Return `dataset`'s other complete code systems by descending size, then observed
    `populationType`."""
    code_axes = sorted(
        (
            _PartitionAxis(
                property=config.code_system,
                values=tuple(
                    sorted(row["dcid"] for row in code_rows_by_system[config.code_system])
                ),
            )
            for config in _complete_systems(dataset, code_systems)
            if config.code_system != spine.code_system
        ),
        key=lambda axis: len(axis.values or ()),
        reverse=True,
    )
    return (*code_axes, _PartitionAxis(property="populationType", values=None))


def _observe_axis_values(
    reader: GraphReader, dcids: Sequence[str], property: str
) -> tuple[str, ...]:
    """Return distinct `property` values from the dcids already read."""
    sample = sorted(set(dcids))
    if not sample:
        return ()
    arcs_by_dcid = _fetch_arcs_many(reader, sample, properties=(property,))
    values = {value for arcs in arcs_by_dcid.values() if arcs for value in arcs.get(property, ())}
    return tuple(sorted(values))


class _ObservedAxisPool:
    """Pool observed axis values across one StatVar enumeration.

    Truncated leaf samples are arbitrary prefixes, so a leaf can use values found by earlier
    reads. `used()` records split values for `_verify_axis_coverage` against exported rows.
    """

    def __init__(self) -> None:
        self._observed: dict[str, set[str]] = {}
        self._used: dict[str, set[str]] = {}

    def observe(self, reader: GraphReader, dcids: Sequence[str], property: str) -> None:
        self._observed.setdefault(property, set()).update(
            _observe_axis_values(reader, dcids, property)
        )

    def values_for(self, property: str) -> tuple[str, ...]:
        return tuple(sorted(self._observed.get(property, ())))

    def record_used(self, property: str, values: Sequence[str]) -> None:
        self._used.setdefault(property, set()).update(values)

    def used(self) -> dict[str, tuple[str, ...]]:
        return {property: tuple(sorted(values)) for property, values in self._used.items()}


def _format_filter_path(base_property: str, base_dcid: str, filters: Mapping[str, str]) -> str:
    parts = [f"{base_property}={base_dcid!r}"]
    parts.extend(f"{key}={value!r}" for key, value in sorted(filters.items()))
    return ", ".join(parts)


def _read_partitioned(
    reader: GraphReader,
    base_property: str,
    base_dcid: str,
    filters: dict[str, str],
    remaining_axes: tuple[_PartitionAxis, ...],
    pool: _ObservedAxisPool,
    *,
    max_results: int,
    max_depth: int,
    depth: int = 0,
) -> set[str]:
    """Read reverse subjects with filters and return only complete leaves.

    This graph repeats reverse-arc pages. Truncation and `GraphNotAdvancingError` both trigger
    partitioning. Other `GraphReadError`s propagate.

    Split along remaining axes in order, using declared code values or the run-wide observed
    `populationType` pool. Union completed leaves. `SystemExit` names the filter path when axes,
    observed values, or `max_depth` are exhausted.
    """
    try:
        page = reader.node_subjects(
            base_dcid, property=base_property, filters=filters or None, max_results=max_results
        )
    except GraphNotAdvancingError:
        page = None

    if page is not None and not page.truncated:
        for axis in remaining_axes:
            if axis.values is None:
                pool.observe(reader, page.dcids, axis.property)
        return set(page.dcids)

    filter_path = _format_filter_path(base_property, base_dcid, filters)
    if depth >= max_depth:
        raise SystemExit(
            f"partition recursion exceeded max_depth={max_depth} reading {filter_path} -- "
            "still truncated"
        )
    if not remaining_axes:
        raise SystemExit(
            f"exhausted every partition property reading {filter_path} -- still truncated"
        )

    axis, *rest = remaining_axes
    if axis.values is None:
        if page is not None:
            pool.observe(reader, page.dcids, axis.property)
        values = pool.values_for(axis.property)
        if not values:
            raise SystemExit(
                f"could not observe any {axis.property!r} value while partitioning "
                f"{filter_path} -- cannot split further"
            )
        pool.record_used(axis.property, values)
    else:
        values = axis.values

    dcids: set[str] = set()
    for value in values:
        dcids |= _read_partitioned(
            reader,
            base_property,
            base_dcid,
            {**filters, axis.property: value},
            tuple(rest),
            pool,
            max_results=max_results,
            max_depth=max_depth,
            depth=depth + 1,
        )
    return dcids


def _verify_spine_coverage(
    reader: GraphReader,
    enumerated: set[str],
    spines: Sequence[CodeSystemConfig],
    code_systems: Sequence[CodeSystemConfig],
    code_rows_by_system: dict[str, list[dict]],
    *,
    max_results: int,
) -> int:
    """Check that complete one-page reads from non-spine codes stay within `enumerated`.

    An outside subject raises `SystemExit`. Truncated or non-advancing reads are skipped, and the
    function returns their count.
    """
    spine_systems = {spine.code_system for spine in spines}
    skipped = 0
    total = 0
    for config in code_systems:
        if config.code_system in spine_systems:
            continue
        for row in code_rows_by_system[config.code_system]:
            code_dcid = row["dcid"]
            total += 1
            try:
                page = reader.node_subjects(
                    code_dcid, property=config.code_system, max_results=max_results
                )
            except GraphNotAdvancingError:
                skipped += 1
                continue
            if page.truncated:
                skipped += 1
                continue
            outside = set(page.dcids) - enumerated
            if outside:
                raise SystemExit(
                    f"{config.code_system}={code_dcid!r} carries {sorted(outside)!r} outside "
                    f"the population enumerated through {sorted(spine_systems)!r} -- the "
                    "declared datasets do not cover the whole StatVar population"
                )
    if skipped:
        print(
            f"spine coverage check skipped {skipped}/{total} non-spine code(s) too large to "
            "read in one response",
            file=sys.stderr,
        )
    return skipped


_AXIS_ROW_FIELD = {"populationType": "population_type"}
# Map each observed partition axis to its StatVar row field for coverage checks.


def _verify_axis_coverage(
    statvars: Sequence[dict], observed_axis_usage: Mapping[str, tuple[str, ...]]
) -> None:
    """Require every exported value on a used observed axis to appear in its partition filters.

    Unexpected values raise `SystemExit` with the axis and StatVar dcid.
    """
    for axis_property, used_values in observed_axis_usage.items():
        row_field = _AXIS_ROW_FIELD[axis_property]
        used = set(used_values)
        for row in statvars:
            value = row.get(row_field)
            if value is not None and value not in used:
                raise SystemExit(
                    f"{row['dcid']} carries {axis_property}={value!r}, which was never used to "
                    f"partition the {axis_property} axis (used values: {sorted(used)!r}) -- some "
                    "leaf may have split on an incomplete sample and dropped rows silently"
                )


# Row fields in the order of `_STATVAR_SHAPE_PROPERTIES`.
_ROW_SHAPE_FIELDS = (
    "population_type",
    "measured_property",
    "stat_type",
    "measurement_qualifier",
    "measurement_denominator",
)
# A refusal names at most this many missing StatVars, then the total.
_MISSING_STATVAR_NAME_CAP = 20

type _FiveTuple = tuple[str | None, ...]


def _row_five_tuple(row: Mapping) -> _FiveTuple:
    return tuple(row[field] for field in _ROW_SHAPE_FIELDS)


def _read_five_tuple_subjects(
    reader: GraphReader,
    five_tuple: _FiveTuple,
    axes: tuple[_PartitionAxis, ...],
    *,
    max_results: int,
    max_depth: int,
) -> set[str]:
    """Return every `<-measuredProperty` subject on `five_tuple` that carries a value of each axis.

    Filters on the tuple's other set fields and splits first on every value of the first axis,
    since an unsplit read of a CRS five-tuple does not finish inside a request. A read still
    truncated after the last axis raises `SystemExit` through `_read_partitioned`.
    """
    shape = dict(zip(_STATVAR_SHAPE_PROPERTIES, five_tuple, strict=True))
    measured_property = shape.pop("measuredProperty")
    if measured_property is None:
        raise SystemExit(f"five-tuple {five_tuple!r} has no measuredProperty to read from")
    filters = {prop: value for prop, value in shape.items() if value is not None}
    pool = _ObservedAxisPool()
    if not axes:
        return _read_partitioned(
            reader,
            "measuredProperty",
            measured_property,
            filters,
            (),
            pool,
            max_results=max_results,
            max_depth=max_depth,
        )
    first, *rest = axes
    subjects: set[str] = set()
    for value in first.values or ():
        subjects |= _read_partitioned(
            reader,
            "measuredProperty",
            measured_property,
            {**filters, first.property: value},
            tuple(rest),
            pool,
            max_results=max_results,
            max_depth=max_depth,
        )
    return subjects


def _minimal_system_sets(
    key_sets: set[frozenset[str]], code_rows_by_system: Mapping[str, list[dict]]
) -> list[frozenset[str]]:
    """Return the code systems each key set of one five-tuple declares, keeping only the sets
    no other set is a proper subset of.

    A StatVar covering some row's key set carries every system of that set, so a read split on
    a minimal set below it reaches the StatVar. Two datasets on one five-tuple with disjoint
    systems, such as the CRS and DAC1, each get a read of their own.
    """
    system_sets = {
        frozenset(system for system in keys if system in code_rows_by_system) for keys in key_sets
    }
    return sorted(
        (systems for systems in system_sets if not any(other < systems for other in system_sets)),
        key=sorted,
    )


def _verify_five_tuple_completeness(
    reader: GraphReader,
    statvars: Sequence[dict],
    code_rows_by_system: Mapping[str, list[dict]],
    spine_systems: frozenset[str],
    *,
    max_results: int = _DEFAULT_MAX_RESULTS,
    max_depth: int = _MAX_PARTITION_DEPTH,
) -> None:
    """Refuse an export that misses a StatVar the index would decide.

    Search takes a StatVar's family from the index when its constraint properties hold every
    constraint property of some exported row of its five-tuple (`index_covers_slot_keys` in
    `code_index.members`, which this check shares through `key_sets_cover`), so such a StatVar
    must be an exported row. One the index would not decide, such as an IDS debt series with
    no CRS keys, passes: it is a listed row (`_verify_listed_five_tuples`) or the graph walk
    resolves it.

    A decided StatVar carries every code system of the exported row it covers. Each five-tuple
    gets one reverse read per minimal set of code systems its rows declare
    (`_minimal_system_sets`), split on those systems' enum codes, smallest non-spine system first
    and spine systems last, so the reads reach StatVars the spine enumeration did not. Values
    come from the enum, since exported rows would hide a value no row shows. The forward read of
    each unexported subject confirms its five-tuple, including the denominator, and its key set.
    Truncated or exhausted reads refuse, as does a subject the forward read finds no node for.

    A decided StatVar whose codes lie outside the typed enums escapes both the spine and this
    check, and nothing else catches it. The check proves completeness at export time. The live
    drift test (`tests/code_index/test_export_drift.py`) re-exports through
    `export_graph_blocks` and catches rows changed or removed since the committed index.
    """
    key_sets: dict[_FiveTuple, set[frozenset[str]]] = {}
    for row in statvars:
        key_sets.setdefault(_row_five_tuple(row), set()).add(
            frozenset(row["constraint_properties"])
        )
    exported = {row["dcid"] for row in statvars}

    unexported: set[str] = set()
    for five_tuple, keys in key_sets.items():
        for systems in _minimal_system_sets(keys, code_rows_by_system):
            ordered = sorted(
                systems,
                key=lambda system: (
                    system in spine_systems,
                    len(code_rows_by_system[system]),
                    system,
                ),
            )
            axes = tuple(
                _PartitionAxis(
                    property=system,
                    values=tuple(sorted(row["dcid"] for row in code_rows_by_system[system])),
                )
                for system in ordered
            )
            subjects = _read_five_tuple_subjects(
                reader, five_tuple, axes, max_results=max_results, max_depth=max_depth
            )
            unexported |= subjects - exported
    if not unexported:
        return

    candidates = sorted(unexported)
    arcs_by_dcid = _fetch_arcs_many(
        reader, candidates, properties=("constraintProperties", *_STATVAR_SHAPE_PROPERTIES)
    )
    missing = []
    for dcid in candidates:
        arcs = arcs_by_dcid.get(dcid)
        if arcs is None:
            raise SystemExit(
                f"{dcid} is a <-measuredProperty subject of an exported five-tuple but the "
                "graph holds no node for it"
            )
        five_tuple = tuple(_first_or_none(arcs.get(prop, ())) for prop in _STATVAR_SHAPE_PROPERTIES)
        declared = frozenset(arcs.get("constraintProperties", ()))
        if key_sets_cover(key_sets, five_tuple, declared):
            missing.append(dcid)
    if missing:
        named = missing[:_MISSING_STATVAR_NAME_CAP]
        more = f" and {len(missing) - len(named)} more" if len(missing) > len(named) else ""
        raise SystemExit(
            f"{len(missing)} StatVar(s) on an exported five-tuple carry the constraint "
            f"properties of an exported row but are not exported rows: {named!r}{more} -- the "
            "index would decide their family without them"
        )


def listed_dcids(
    provenances: Mapping[str, Collection[str]], datasets: Collection[str]
) -> tuple[str, ...]:
    """Return the sorted StatVars `provenances` assigns to any of `datasets`.

    `provenances` maps a StatVar to the datasets publishing it: a coverage build's
    (`CoverageJoin.provenances`), or the committed index's listed rows for the drift test. A
    dataset of `datasets` with no StatVar raises `SystemExit`, since the export could not list
    its rows."""
    found = {dcid: set(datasets).intersection(sets) for dcid, sets in provenances.items()}
    empty = sorted(set(datasets).difference(*found.values()))
    if empty:
        raise SystemExit(
            f"listed dataset(s) {empty!r} hold no StatVar in the coverage build -- the export "
            "cannot list their rows"
        )
    return tuple(sorted(dcid for dcid, datasets_of in found.items() if datasets_of))


def export_source_codes(reader: ExportReader, dcids: Sequence[str]) -> list[dict]:
    """Build the `source_codes` rows from the `sourceIndicatorCode` arcs of `dcids`: one row
    per arc provenance and upper-cased code, naming its StatVars, sorted.

    The rows key on the provenance each arc value carries. The coverage build gives the same
    StatVar its provenances separately, and `parse_index` refuses a code row whose provenance
    its StatVar's row lacks. A StatVar of `dcids` with no code, and a code value with no
    provenance or no text, raise `SystemExit`. Every listed dataset gives each of its
    StatVars a series code (today World Bank IDS alone), so a listed dataset without codes
    must lift this rule before it is listed."""
    arcs = reader.node_arcs_batch(list(dcids))
    members: dict[tuple[str, str], set[str]] = defaultdict(set)
    uncoded: list[str] = []
    for dcid in dcids:
        values = ((arcs.get(dcid) or {}).get(_SOURCE_CODE) or {}).get("nodes", ())
        if not values:
            uncoded.append(dcid)
            continue
        for value in values:
            provenance, code = value.get("provenanceId"), (value.get("value") or "").strip()
            if not provenance or not code:
                raise SystemExit(
                    f"{dcid}'s {_SOURCE_CODE} arc carries {value!r}, which names no provenance "
                    "or no code -- cannot key a series code on it"
                )
            members[provenance, code.upper()].add(dcid)
    if uncoded:
        named = uncoded[:_MISSING_STATVAR_NAME_CAP]
        more = f" and {len(uncoded) - len(named)} more" if len(uncoded) > len(named) else ""
        raise SystemExit(
            f"{len(uncoded)} listed StatVar(s) carry no {_SOURCE_CODE}: {named!r}{more} -- "
            "search could not answer their series code"
        )
    return [
        {"provenance": provenance, "code": code, "dcids": sorted(found)}
        for (provenance, code), found in sorted(members.items())
    ]


def export_value_parents(reader: GraphReader, statvars: Sequence[dict]) -> list[dict]:
    """Build the `value_parents` rows: the `specializationOf` chain above each constraint
    value a row of `statvars` holds, one row per value with a parent, the parents' own
    parents included, sorted by dcid.

    A value with two parents, and a chain that returns to a value it holds, raise
    `SystemExit`, since `CodeIndex.value_ancestors` walks one finite chain."""
    parent_of: dict[str, str] = {}
    read: set[str] = set()
    frontier = sorted(
        {value for row in statvars for value in row["constraint_properties"].values()}
    )
    while frontier:
        read.update(frontier)
        arcs = _fetch_arcs_many(reader, frontier, properties=("specializationOf",))
        for dcid in frontier:
            parents = (arcs.get(dcid) or {}).get("specializationOf", ())
            _check_single_valued(dcid, "specializationOf", parents)
            if parents:
                parent_of[dcid] = parents[0]
        frontier = sorted({parent_of[d] for d in frontier if d in parent_of} - read)
    for dcid in parent_of:
        chain = [dcid]
        while (node := parent_of.get(chain[-1])) is not None:
            if node in chain:
                raise SystemExit(
                    f"the specializationOf chain of {dcid} returns to {node}: "
                    f"{' -> '.join([*chain, node])}"
                )
            chain.append(node)
    return [{"dcid": dcid, "parent": parent} for dcid, parent in sorted(parent_of.items())]


def _verify_listed_five_tuples(
    reader: GraphReader,
    statvars: Sequence[dict],
    listed: Sequence[dict],
    *,
    max_results: int = _UNSPLIT_MAX_RESULTS,
    max_depth: int = _MAX_PARTITION_DEPTH,
) -> None:
    """Refuse an export when a five-tuple a listed row has holds a StatVar that is neither an
    exported nor a listed row, or when a listed StatVar is also an exported row.

    Search and the mint elect a family that holds a listed row from the index's rows of its
    five-tuple, in place of a graph walk over it (`code_index.members.listed_group`), so those
    rows must be every StatVar of the five-tuple. Listed rows declare no code system to split
    on, so each five-tuple gets one unsplit read of up to `max_results` subjects, and a read
    still truncated refuses."""
    exported = {row["dcid"] for row in statvars}
    both = sorted(exported.intersection(row["dcid"] for row in listed))
    if both:
        raise SystemExit(
            f"{len(both)} listed StatVar(s) are also exported rows: "
            f"{both[:_MISSING_STATVAR_NAME_CAP]!r}"
        )
    known = exported.union(row["dcid"] for row in listed)
    unknown: set[str] = set()
    for five_tuple in sorted({_row_five_tuple(row) for row in listed}, key=repr):
        unknown |= (
            _read_five_tuple_subjects(
                reader, five_tuple, (), max_results=max_results, max_depth=max_depth
            )
            - known
        )
    if unknown:
        named = sorted(unknown)[:_MISSING_STATVAR_NAME_CAP]
        more = f" and {len(unknown) - len(named)} more" if len(unknown) > len(named) else ""
        raise SystemExit(
            f"{len(unknown)} StatVar(s) on a listed row's five-tuple are neither exported nor "
            f"listed rows: {named!r}{more} -- members elected from the index would miss them"
        )


def enumerate_statvar_dcids(
    reader: GraphReader,
    code_rows_by_system: dict[str, list[dict]],
    code_systems: Sequence[CodeSystemConfig],
    datasets: Sequence[DatasetDeclaration],
    *,
    max_results: int = _DEFAULT_MAX_RESULTS,
    max_partition_depth: int = _MAX_PARTITION_DEPTH,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Return sorted StatVar dcids and observed-axis values used for partitioning.

    Enumerate each dataset through its own spine (`_choose_spine`), split incomplete reads on the
    systems every StatVar of that dataset declares, and share one `_ObservedAxisPool` across all
    reads. Datasets with the same spine and axes share one pass. Check non-spine coverage of the
    union afterward.
    """
    pool = _ObservedAxisPool()
    dcids: set[str] = set()
    spines: dict[str, CodeSystemConfig] = {}
    read_plans: set[tuple[str, tuple[_PartitionAxis, ...]]] = set()
    for dataset in datasets:
        spine = _choose_spine(dataset, code_systems, code_rows_by_system)
        axes = _partition_axes(spine, dataset, code_systems, code_rows_by_system)
        if (spine.code_system, axes) in read_plans:
            continue
        read_plans.add((spine.code_system, axes))
        spines[spine.code_system] = spine
        for row in code_rows_by_system[spine.code_system]:
            dcids |= _read_partitioned(
                reader,
                spine.code_system,
                row["dcid"],
                {},
                axes,
                pool,
                max_results=max_results,
                max_depth=max_partition_depth,
            )

    _verify_spine_coverage(
        reader,
        dcids,
        tuple(spines.values()),
        code_systems,
        code_rows_by_system,
        max_results=max_results,
    )
    return tuple(sorted(dcids)), pool.used()


def _reject_provisional_constraint_values(
    reader: GraphReader, rows: Sequence[tuple[str, dict[str, str]]]
) -> None:
    """Require each constraint value to identify a resolved graph node.

    One batched read covers all distinct values. Missing nodes, missing `typeOf`, or nodes typed
    only `ProvisionalNode` raise `SystemExit`; a provisional type alongside a real type passes.
    """
    candidate_values = sorted(
        {value for _, constraint_properties in rows for value in constraint_properties.values()}
    )
    if not candidate_values:
        return
    type_arcs = _fetch_arcs_many(reader, candidate_values, properties=("typeOf",))
    for dcid, constraint_properties in rows:
        for prop, value in constraint_properties.items():
            held = type_arcs.get(value)
            if held is None:
                raise SystemExit(
                    f"{dcid} binds {prop!r} to {value!r}, which the graph holds no node for "
                    "at all -- cannot export a dangling reference as a plain dcid "
                    "indistinguishable from a resolved code"
                )
            types = set(held.get("typeOf", ()))
            if not types:
                raise SystemExit(
                    f"{dcid} binds {prop!r} to {value!r}, whose node carries no typeOf arc "
                    "at all -- cannot export a dangling reference as a plain dcid "
                    "indistinguishable from a resolved code"
                )
            if types == {"ProvisionalNode"}:
                raise SystemExit(
                    f"{dcid} binds {prop!r} to {value!r}, which the graph types exactly "
                    "ProvisionalNode -- cannot export a dangling reference as a plain dcid "
                    "indistinguishable from a resolved code"
                )


def export_statvars(reader: GraphReader, statvar_dcids: Sequence[str]) -> list[dict]:
    """Build StatVar rows with two forward reads.

    Pass 1 reads types, shape fields, and each node's declared `constraintProperties`. Pass 2 reads
    the union of those properties and fills each row's mapping.
    Constraint and observation properties remain list-valued.

    Missing declared values, unresolved dcids, non-StatisticalVariables, multi-valued single-value
    fields, and unverified targets raise `SystemExit`. The target check rejects missing nodes,
    missing `typeOf`, and nodes typed only `ProvisionalNode`.
    """
    pass1 = _fetch_arcs_many(reader, statvar_dcids, properties=_STATVAR_PASS1_PROPERTIES)

    partial: list[tuple[str, dict[str, tuple[str, ...]], tuple[str, ...]]] = []
    constraint_property_names: set[str] = set()
    for dcid in statvar_dcids:
        arcs = pass1.get(dcid)
        if arcs is None:
            raise SystemExit(
                f"the graph holds no node for {dcid} -- it is either a StatVar this "
                "export just enumerated and the graph dropped mid-read, or a row the "
                "committed index still carries and the graph no longer holds"
            )
        _check_typed_as_statvar(dcid, arcs.get("typeOf", ()))
        for prop in _STATVAR_SHAPE_PROPERTIES:
            _check_single_valued(dcid, prop, arcs.get(prop, ()))
        declared_properties = arcs.get("constraintProperties", ())
        constraint_property_names.update(declared_properties)
        partial.append((dcid, arcs, declared_properties))

    pass2_properties = sorted(constraint_property_names)
    pass2: dict[str, dict[str, tuple[str, ...]] | None] = {}
    if pass2_properties:
        pass2 = _fetch_arcs_many(reader, statvar_dcids, properties=pass2_properties)

    tentative: list[tuple[str, dict[str, tuple[str, ...]], dict[str, str]]] = []
    for dcid, arcs, declared_properties in partial:
        values_by_property = pass2.get(dcid) or {}
        constraint_properties = {}
        for prop in declared_properties:
            values = values_by_property.get(prop, ())
            if not values:
                raise SystemExit(
                    f"{dcid} declares {prop!r} in its constraintProperties but the pass-2 "
                    "forward read returned no value for it -- cannot export a row that "
                    "would silently drop a declared constraint"
                )
            _check_single_valued(dcid, prop, values)
            value = values[0]
            if value == UNRESOLVED_NODE:
                raise SystemExit(
                    f"{dcid} declares {prop!r} in its constraintProperties but the graph "
                    "could not resolve its value to a dcid -- cannot export a constraint "
                    "value the index would carry as if it were resolved"
                )
            constraint_properties[prop] = value
        tentative.append((dcid, arcs, constraint_properties))

    _reject_provisional_constraint_values(
        reader, [(dcid, constraint_properties) for dcid, _, constraint_properties in tentative]
    )

    rows = [
        {
            "dcid": dcid,
            "population_type": _first_or_none(arcs.get("populationType", ())),
            "measured_property": _single_value(
                dcid, "measuredProperty", arcs.get("measuredProperty", ())
            ),
            "stat_type": _first_or_none(arcs.get("statType", ())),
            "measurement_qualifier": _first_or_none(arcs.get("measurementQualifier", ())),
            "measurement_denominator": _first_or_none(arcs.get("measurementDenominator", ())),
            "observation_properties": sorted(arcs.get("observationProperties", ())),
            "constraint_properties": dict(sorted(constraint_properties.items())),
        }
        for dcid, arcs, constraint_properties in tentative
    ]
    rows.sort(key=lambda row: row["dcid"])
    return rows


_MEASURE_AXIS_SHAPE_FIELDS = {
    "populationType": "population_type",
    "measuredProperty": "measured_property",
    "measurementQualifier": "measurement_qualifier",
}
_MEASURE_VALUE_ARC_PROPERTIES = ("name", "alternateName", "shortName", "description")


def _measure_axis_value(row: dict, axis: MeasureAxis) -> str | None:
    if axis.kind == "shape":
        return row[_MEASURE_AXIS_SHAPE_FIELDS[axis.axis]]
    return row["constraint_properties"].get(axis.axis)


def _observed_measure_values(
    statvars: Sequence[dict],
    axes: Sequence[MeasureAxis],
    declared: Mapping[str, Sequence[str]],
) -> dict[str, set[str]]:
    """Return distinct axis values carried by exported StatVar rows, plus `declared` values.

    Filtering to observed values excludes unpopulated `GrossValue` and `NettingBasisEnum.Gross`,
    whose name collides with `GrossValue`. A declared value is exported only on an axis in `axes`.
    """
    observed: dict[str, set[str]] = {axis.axis: set(declared.get(axis.axis, ())) for axis in axes}
    for row in statvars:
        for axis in axes:
            value = _measure_axis_value(row, axis)
            if value is not None:
                observed[axis.axis].add(value)
    return observed


def export_measure_values(
    reader: GraphReader,
    statvars: Sequence[dict],
    axes: Sequence[MeasureAxis],
    declared: Mapping[str, Sequence[str]] = _DECLARED_MEASURE_VALUES,
) -> list[dict]:
    """Build rows for observed and declared `(axis, dcid)` pairs with graph labels and authored
    text overrides. A row's `definition` is the value's graph `description`.

    Authored labels replace graph names. Synonyms merge with `alternateName`, and `matchable`
    and `same_reading_as` pass through. A missing `name` raises `SystemExit` because a
    dcid fallback would enter text matching as a query term.
    """
    observed = _observed_measure_values(statvars, axes, declared)
    all_dcids = sorted({dcid for values in observed.values() for dcid in values})
    arcs_by_dcid = _fetch_arcs_many(reader, all_dcids, properties=_MEASURE_VALUE_ARC_PROPERTIES)

    rows = []
    for axis in axes:
        for dcid in sorted(observed[axis.axis]):
            arcs = arcs_by_dcid.get(dcid)
            if arcs is None:
                raise SystemExit(
                    f"{dcid} is a {axis.axis!r} measure value an exported StatVar row carries "
                    "or the export declares, but the graph holds no node for it"
                )
            name = _single_value(dcid, "name", arcs.get("name", ()))
            overrides = _MEASURE_VALUE_TEXT.get(dcid, _MeasureValueText())
            label = overrides.label if overrides.label is not None else strip_system_suffix(name)
            synonyms = sorted(
                {v for v in arcs.get("alternateName", ()) if v and v != label}
                | set(overrides.synonyms)
            )
            rows.append(
                {
                    "dcid": dcid,
                    "axis": axis.axis,
                    "label": label,
                    "synonyms": synonyms,
                    "short_name": _first_or_none(arcs.get("shortName", ())),
                    "matchable": overrides.matchable,
                    "same_reading_as": overrides.same_reading_as,
                    "definition": _first_or_none(arcs.get("description", ())),
                }
            )
    rows.sort(key=lambda row: (row["axis"], row["dcid"]))
    return rows


def _observed_measure_partitions(
    statvars: Sequence[dict], axes: Sequence[MeasureAxis]
) -> set[tuple[tuple[str, str | None], ...]]:
    """Return distinct measure-axis tuples exhibited by exported StatVar rows."""
    return {tuple((axis.axis, _measure_axis_value(row, axis)) for axis in axes) for row in statvars}


def _partition_sort_key(pairs: tuple[tuple[str, str | None], ...]) -> tuple:
    return tuple((axis, value is None, value or "") for axis, value in pairs)


def _verify_measure_partition_labels(
    statvars: Sequence[dict],
    axes: Sequence[MeasureAxis],
    labels: Sequence[MeasurePartitionLabel],
) -> None:
    """Require each observed partition to match exactly one label with `label_for_pairs`.

    This is the runtime rule. Empty axes make validation a no-op.
    """
    if not axes:
        return
    offending: list[tuple[tuple[str, str | None], ...]] = []
    for pairs in sorted(_observed_measure_partitions(statvars, axes), key=_partition_sort_key):
        try:
            match = label_for_pairs(pairs, labels)
        except ValueError:
            offending.append(pairs)
            continue
        if match is None:
            offending.append(pairs)
    if offending:
        raise SystemExit(
            f"{len(offending)} measure partition(s) carried by the exported StatVar rows "
            f"resolve to no single authored label: {offending!r} -- add or fix entries in "
            "_MEASURE_PARTITION_LABELS to cover each"
        )


def _verify_measure_nouns(statvars: Sequence[dict], nouns: Sequence[MeasurePartitionLabel]) -> None:
    """Require that no exported StatVar row matches a measure noun: every row belongs to a
    measure partition, whose label names it, and a noun beside it would put two names on
    one measure."""
    matched = sorted(
        row["dcid"]
        for row in statvars
        if label_for_pairs(
            [
                ("populationType", row["population_type"]),
                ("measuredProperty", row["measured_property"]),
                ("statType", row["stat_type"]),
                *row["constraint_properties"].items(),
            ],
            nouns,
        )
        is not None
    )
    if matched:
        raise SystemExit(
            f"{len(matched)} exported StatVar row(s) match a measure noun as well as their "
            f"partition label, first {matched[:5]!r} -- narrow the entries in _MEASURE_NOUNS"
        )


def _verify_measure_label_named_values(
    statvars: Sequence[dict],
    codes: Sequence[dict],
    axes: Sequence[MeasureAxis],
    labels: Sequence[MeasurePartitionLabel],
) -> None:
    """Require each label's `named_values` pair to be an exported code of its property's
    code system, carried by at least one StatVar row under the label's partition. A pair
    that fails names nothing and silences nothing in a heading. Empty axes make validation
    a no-op, as in `_verify_measure_partition_labels`."""
    if not axes:
        return
    known = {(row["code_system"], row["dcid"]) for row in codes}
    axes_by_name = {axis.axis: axis for axis in axes}
    offending: list[tuple[str, tuple[str, str]]] = []
    for label in labels:
        rows = [
            row
            for row in statvars
            if all(
                axis in axes_by_name and _measure_axis_value(row, axes_by_name[axis]) == value
                for axis, value in label.axis_values
            )
        ]
        offending.extend(
            (label.label, (prop, code))
            for prop, code in label.named_values
            if (prop, code) not in known
            or not any(row["constraint_properties"].get(prop) == code for row in rows)
        )
    if offending:
        raise SystemExit(
            f"{len(offending)} measure label named value(s) are no exported code carried by "
            f"a StatVar row under the label's partition: {offending!r} -- fix the entries' "
            "named_values in _MEASURE_PARTITION_LABELS"
        )


def _measure_axis_to_dict(axis: MeasureAxis) -> dict:
    return {
        "axis": axis.axis,
        "kind": axis.kind,
        "role": axis.role,
        "prefer": list(axis.prefer),
        "label": axis.label,
    }


def _measure_partition_label_to_dict(partition_label: MeasurePartitionLabel) -> dict:
    return {
        "axis_values": [list(pair) for pair in partition_label.axis_values],
        "label": partition_label.label,
        "named_values": [list(pair) for pair in partition_label.named_values],
        "definition": partition_label.definition,
    }


def _stated_word_to_dict(stated: StatedWord) -> dict:
    return {
        "word": stated.word,
        "measured_property": stated.measured_property,
        **({"where": dict(sorted(stated.where.items()))} if stated.where else {}),
    }


def _dataset_to_dict(dataset: DatasetDeclaration) -> dict:
    return {
        "provenance": dataset.provenance,
        "label": dataset.label,
        "description": dataset.description,
        "opt_in": dataset.opt_in,
        "absent_observation": dataset.absent_observation,
        "account": dataset.account,
        "rank": dataset.rank,
        "invariants": dict(sorted(dataset.invariants.items())),
        **(
            {"stated_words": [_stated_word_to_dict(w) for w in dataset.stated_words]}
            if dataset.stated_words
            else {}
        ),
    }


def _join_provenances(statvars: list[dict], coverage: CoverageJoin) -> int:
    """Set each row's sorted `provenances` from `coverage` and return how many rows it does not
    cover. An uncovered row carries `[]`."""
    uncovered = 0
    for row in statvars:
        provenances = sorted(coverage.provenances.get(row["dcid"], ()))
        row["provenances"] = provenances
        if not provenances:
            uncovered += 1
    return uncovered


def _verify_complete_systems(
    statvars: Sequence[dict], code_systems: Sequence[CodeSystemConfig]
) -> None:
    """Require every row of a dataset to declare each code system listing that dataset in
    `declared_by_every_statvar_of`. A row that does not raises `SystemExit`: the enumeration split
    that dataset's reads on the system and can have dropped rows like it."""
    for config in code_systems:
        complete_in = set(config.declared_by_every_statvar_of)
        for row in statvars:
            if (
                complete_in.intersection(row["provenances"])
                and config.code_system not in row["constraint_properties"]
            ):
                raise SystemExit(
                    f"{row['dcid']} belongs to {row['provenances']!r} and does not declare "
                    f"{config.code_system!r}, which is configured as declared by every StatVar "
                    f"of {sorted(complete_in)!r}"
                )


def _statvar_shape_counts(statvars: list[dict], code_systems: Sequence[str]) -> dict[str, int]:
    """Keyed by code system: how many StatVar rows name that code system in
    their `constraint_properties`.
    """
    return {
        code_system: sum(1 for row in statvars if code_system in row["constraint_properties"])
        for code_system in code_systems
    }


def export_graph_blocks(
    reader: ExportReader,
    *,
    code_systems: Sequence[CodeSystemConfig] = CODE_SYSTEMS,
    measure_axes: Sequence[MeasureAxis] = MEASURE_AXES,
    measure_partition_labels: Sequence[MeasurePartitionLabel] = _MEASURE_PARTITION_LABELS,
    measure_nouns: Sequence[MeasurePartitionLabel] = _MEASURE_NOUNS,
    datasets: Sequence[DatasetDeclaration] = DATASETS,
    party_type_kinds: Mapping[str, PartyKind | None] = PARTY_TYPE_KINDS,
    listed: Collection[str] = (),
    shape_slot_defaults: Sequence[ShapeSlotDefault] = SHAPE_SLOT_DEFAULTS,
    partly_stated_values: Sequence[PartlyStatedValue] = PARTLY_STATED_VALUES,
    measure_classes: Sequence[MeasureClassDeclaration] = MEASURE_CLASSES,
    variable_aliases: Mapping[str, str] = VARIABLE_ALIASES,
    stat_type_words: Sequence[StatTypeWordDeclaration] = STAT_TYPE_WORDS,
) -> dict:
    """Build every document block the graph and the authored config determine, sorted.

    `listed` names the StatVars of the datasets no code system enumerates (`listed_dcids`),
    read into `listed_statvars` and checked by `_verify_listed_five_tuples`, their series
    codes into `source_codes` (`export_source_codes`), and the parents of their constraint
    values into `value_parents` (`export_value_parents`).

    StatVar rows carry no `provenances` and the result has no stamp, so it does not load as an
    index. `export_index` completes it from a coverage build. The drift test compares it with the
    committed export, listing the same StatVars the committed index lists, so it needs graph
    credentials alone.
    """
    declarations = []
    codes = []
    code_rows_by_system: dict[str, list[dict]] = {}
    for config in code_systems:
        declaration, rows = export_code_system(reader, config)
        declarations.append(declaration)
        codes.extend(rows)
        code_rows_by_system[config.code_system] = rows
    _apply_concept_groups(codes, _CONCEPT_GROUPS, code_rows_by_system.keys())
    declarations.sort(key=lambda d: d["code_system"])
    codes.sort(key=lambda row: row["dcid"])
    _validate_declared_defaults(code_systems, code_rows_by_system)

    statvar_dcids, observed_axis_usage = enumerate_statvar_dcids(
        reader, code_rows_by_system, code_systems, datasets
    )
    statvars = export_statvars(reader, statvar_dcids)
    _verify_axis_coverage(statvars, observed_axis_usage)
    _verify_five_tuple_completeness(
        reader,
        statvars,
        code_rows_by_system,
        frozenset(
            _choose_spine(dataset, code_systems, code_rows_by_system).code_system
            for dataset in datasets
        ),
    )
    listed_statvars = export_statvars(reader, sorted(listed)) if listed else []
    _verify_listed_five_tuples(reader, statvars, listed_statvars)
    source_codes = export_source_codes(reader, sorted(listed)) if listed else []
    value_parents = export_value_parents(reader, listed_statvars)
    _verify_measure_partition_labels(statvars, measure_axes, measure_partition_labels)
    _verify_measure_nouns(statvars, measure_nouns)
    _verify_measure_label_named_values(statvars, codes, measure_axes, measure_partition_labels)
    measure_values = export_measure_values(reader, statvars, measure_axes)
    _strip_measure_words(codes, statvars, measure_values, measure_axes)

    return {
        "declarations": declarations,
        "datasets": sorted(
            (_dataset_to_dict(dataset) for dataset in datasets), key=lambda row: row["provenance"]
        ),
        "codes": codes,
        "statvars": statvars,
        "listed_statvars": listed_statvars,
        "source_codes": source_codes,
        "value_parents": value_parents,
        "measure_axes": [_measure_axis_to_dict(axis) for axis in measure_axes],
        "measure_values": measure_values,
        "measure_partition_labels": [
            _measure_partition_label_to_dict(label) for label in measure_partition_labels
        ],
        "measure_nouns": [_measure_partition_label_to_dict(noun) for noun in measure_nouns],
        "party_types": [
            {"type": type_, "kind": kind} for type_, kind in sorted(party_type_kinds.items())
        ],
        "shape_slot_defaults": [_shape_slot_default_to_dict(d) for d in shape_slot_defaults],
        "partly_stated_values": [_partly_stated_value_to_dict(v) for v in partly_stated_values],
        "measure_classes": [_measure_class_to_dict(d) for d in measure_classes],
        "variable_aliases": dict(variable_aliases),
        "stat_type_words": [_stat_type_word_to_dict(d) for d in stat_type_words],
    }


def stamp_counts(blocks: Mapping[str, list[dict]]) -> dict[str, dict[str, int]]:
    """Return the stamp's `code_counts` and `statvar_shape_counts` for `blocks`, keyed by each
    declared code system."""
    code_systems = [declaration["code_system"] for declaration in blocks["declarations"]]
    return {
        "code_counts": {
            code_system: sum(1 for row in blocks["codes"] if row["code_system"] == code_system)
            for code_system in code_systems
        },
        "statvar_shape_counts": _statvar_shape_counts(blocks["statvars"], code_systems),
    }


def content_hash(blocks: Mapping[str, list[dict]]) -> str:
    """SHA-256 of canonical JSON for the document's data blocks, which excludes `stamp`."""
    encoded = json.dumps(blocks, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def export_index(
    reader: ExportReader,
    *,
    graph_endpoint: str,
    now: datetime,
    coverage: CoverageJoin,
    code_systems: Sequence[CodeSystemConfig] = CODE_SYSTEMS,
    measure_axes: Sequence[MeasureAxis] = MEASURE_AXES,
    measure_partition_labels: Sequence[MeasurePartitionLabel] = _MEASURE_PARTITION_LABELS,
    measure_nouns: Sequence[MeasurePartitionLabel] = _MEASURE_NOUNS,
    datasets: Sequence[DatasetDeclaration] = DATASETS,
    party_type_kinds: Mapping[str, PartyKind | None] = PARTY_TYPE_KINDS,
    partly_stated_values: Sequence[PartlyStatedValue] = PARTLY_STATED_VALUES,
    party_kind_overrides: Mapping[str, KindOverride] = PARTY_KIND_OVERRIDES,
    total_groups: Mapping[str, TotalGroup] = TOTAL_GROUPS,
    listed_datasets: Collection[str] = (),
) -> dict:
    """Build the index document: the graph blocks, with the StatVars `coverage` assigns to
    `listed_datasets` as listed rows (`listed_dcids`), each row's provenances from `coverage`,
    every party of `coverage` with its kind and group (`export_parties`), and the stamp.
    `main` lists `LISTED_DATASETS`.

    Tests can pass smaller config tables for fixture graphs. Empty `measure_axes` yields an empty
    `measure_values` list. The finished document must load: a document `parse_index` refuses,
    such as one whose declared dataset has no row or whose datasets share a rank, raises
    `SystemExit`.
    """
    blocks = export_graph_blocks(
        reader,
        code_systems=code_systems,
        measure_axes=measure_axes,
        measure_partition_labels=measure_partition_labels,
        measure_nouns=measure_nouns,
        datasets=datasets,
        party_type_kinds=party_type_kinds,
        partly_stated_values=partly_stated_values,
        listed=listed_dcids(coverage.provenances, listed_datasets),
    )
    statvars_without_coverage = _join_provenances(blocks["statvars"], coverage)
    _join_provenances(blocks["listed_statvars"], coverage)
    _verify_complete_systems(blocks["statvars"], code_systems)
    blocks["parties"] = export_parties(
        reader,
        coverage.parties,
        type_kinds=party_type_kinds,
        kind_overrides=party_kind_overrides,
        total_groups=total_groups,
    )

    stamp = {
        "graph_endpoint": graph_endpoint,
        "exported_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        **stamp_counts(blocks),
        "content_hash": content_hash(blocks),
        "coverage_build": coverage.build_id,
        "statvars_without_coverage": statvars_without_coverage,
    }
    document = {"stamp": stamp, **blocks}
    try:
        parse_index(document)
        parse_measure_axes(document)
    except IndexFormatError as exc:
        raise SystemExit(f"the export does not load as a code index: {exc}") from exc
    return document


def _write_index(path: Path, document: dict) -> None:
    """Write gzipped JSON.

    The loader detects gzip by its magic number. `mtime=0` keeps identical documents byte-identical.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=str(_DEFAULT_OUT))
    args = parser.parse_args()

    base = _graph_base_url()
    coverage = load_coverage_join(
        _coverage_artifact_uri(),
        datasets={dataset.provenance for dataset in DATASETS} | set(LISTED_DATASETS),
    )
    reader = export_reader(f"{base}/core/api", os.environ.get("SEARCH_GRAPH_TOKEN"))
    try:
        document = export_index(
            reader,
            graph_endpoint=base,
            now=datetime.now(UTC),
            coverage=coverage,
            listed_datasets=LISTED_DATASETS,
        )
    finally:
        reader.close()
    _write_index(Path(args.out), document)
    stamp = document["stamp"]
    print(
        f"wrote {sum(stamp['code_counts'].values())} code rows, "
        f"{len(document['statvars'])} StatVar rows, "
        f"{len(document['listed_statvars'])} listed StatVar rows "
        f"({stamp['statvars_without_coverage']} not in coverage build "
        f"{stamp['coverage_build']}), {len(document['source_codes'])} source code rows, and "
        f"{len(document['measure_values'])} measure values to {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
