"""Regenerate `code_index/data/index.json.gz` from the live graph.

Four CRS code systems and `dac2aAidType` use `<-typeOf` enumeration. `crsPurpose` has 289 codes.
Each declared dataset's StatVars are enumerated through a system every StatVar of that dataset
declares, and the union is checked against complete non-spine reads. This graph repeats reverse-arc
pages, so truncated reads split by code axes and observed `populationType`, to depth 6. A
second read of each exported five-tuple refuses the export when the graph holds a StatVar the
index would decide that is not an exported row.

Pass 1 reads each StatVar's type, shape, and declared constraints. Pass 2 reads their values.
Export requires held, resolved targets with `typeOf` and one value per single-value field. Nodes
typed only `ProvisionalNode` and non-StatisticalVariables raise. Constraint and observation
properties are list-valued. Export validates rows before the mint door trusts them.

Authored config supplies system declarations, defaults, measure axes and labels, the IDS and WEO
measure nouns, synonyms, and dataset declarations. Graph reads supply codes, StatVars, and
measure values on those rows. Every observed measure partition needs one authored label.

The published series coverage build supplies each StatVar's provenances: `coverage.parquet` maps a
variable to its facet ids and `facets.json` maps a facet id to its provenance. The stamp records
the build id and the number of StatVars the build holds no series for. A declared dataset with no
StatVar row aborts the export.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from code_index.loader import IndexFormatError, parse_index, parse_measure_axes
from code_index.members import key_sets_cover
from code_index.query import DatasetsShareFiveTuple
from code_index.types import (
    DatasetDeclaration,
    DeclaredDefault,
    MeasureAxis,
    MeasurePartitionLabel,
    invalid_when_named_entry,
    label_for_pairs,
)
from dc_graph import GraphClient
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
# Twenty-dcid `/v2/node` requests avoid mixer paging loops. GraphClient reads chunks concurrently.
_ARC_CHUNK_SIZE = 20
# Six levels cover every partition axis and bound recursive splitting.
_MAX_PARTITION_DEPTH = 6

_LABEL_SUFFIX_RE = re.compile(r"\s*\((?:CRS|DAC)\s[^()]*\)\s*$")

_CODE_ARC_PROPERTIES = (
    "classificationCode",
    "name",
    "alternateName",
    "shortName",
    "specializationOf",
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

    `hierarchical` is derived from graph rows and checked against the expected value.

    `declared_by_every_statvar_of` lists the provenances of the datasets whose every StatVar
    declares this property. CRS StatVars all declare `crsPurpose`, `crsTypeOfFinance`, and
    `crsTypeOfFlow`; imputed multilateral StatVars all declare `crsPurpose`, `crsTypeOfFlow`, and
    `dac2aAidType`, and none declares `crsTypeOfFinance`. `crsCooperationModality` is optional for
    older CRS StatVars. A dataset is enumerated only through systems that list it, and the export
    checks every exported row against the listing.

    `plain_name` is the singular lowercase noun used after "any". A default code belongs to this
    system, scope codes belong to other systems, and `when_named` entries are lowercase and
    non-empty. Export validates these conditions before writing.
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


_CRS = "provenance/OECDCreditorReportingSystem"
_IMPUTED_MULTILATERAL_SECTORS = "provenance/ONEImputedMultilateralSectors"

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
    ),
    # Imputed multilateral StatVars carry `dac2aAidType`; CRS StatVars do not.
    CodeSystemConfig(
        code_system="dac2aAidType",
        enum_dcid="DAC2aAidTypeEnum",
        expected_hierarchical=False,
        disjoint_from_children=True,
        summable=False,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_IMPUTED_MULTILATERAL_SECTORS,),
        plain_name="aid type",
    ),
)


# One declaration per published table. CRS commitments and some disbursement purpose codes carry
# explicit zero rows, so a missing row for those cells reads as zero, and the imputed sectors table
# reads the same way. The imputed table sits alone on `attributedAmount`, and a question answers
# from it only when it names the table.
DATASETS: tuple[DatasetDeclaration, ...] = (
    DatasetDeclaration(
        provenance=_CRS,
        label="OECD Creditor Reporting System (CRS)",
        synonyms=("CRS", "creditor reporting system"),
        opt_in=False,
        absent_observation="zero",
    ),
    DatasetDeclaration(
        provenance=_IMPUTED_MULTILATERAL_SECTORS,
        label="ONE Imputed Multilateral Sectors",
        synonyms=(
            "imputed aid",
            "imputed multilateral",
            "imputed multilateral aid",
            "imputed multilateral ODA",
            "imputed ODA",
        ),
        opt_in=True,
        absent_observation="zero",
    ),
)


@dataclass(frozen=True)
class CoverageJoin:
    """What the export takes from one series coverage build: its build id and each covered
    variable's sorted provenances."""

    build_id: str
    provenances: Mapping[str, tuple[str, ...]]


def load_coverage_join(base_uri: str) -> CoverageJoin:
    """Load the published coverage build under `base_uri` and read its variable provenances.

    `series_coverage` needs the `coverage` extra, which the graph-only callers of this module
    run without, so the import stays inside this function.
    """
    from series_coverage.loader import load_coverage

    loaded = load_coverage(base_uri)
    return CoverageJoin(
        build_id=loaded.manifest.build_id, provenances=loaded.index.variable_provenances()
    )


# Axis order controls partition election.
# `measuredProperty` is the basis axis, and either basis value generalises.
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
_CODE_SYNONYMS: Mapping[str, tuple[str, ...]] = {
    "CRSTypeOfFlow10": ("development assistance",),
    "CRSTypeOfFlow21": ("other official flows",),
}

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

# Labels key full partitions. `measurementQualifier` stays outside the key, because no measure
# label states the price basis (`code_index.labels.compose_measure_label`).
_MEASURE_PARTITION_LABELS: tuple[MeasurePartitionLabel, ...] = (
    MeasurePartitionLabel(
        axis_values=(
            ("financialObligationType", "Commitment"),
            ("measuredProperty", "amount"),
            ("populationType", "FinancialObligation"),
        ),
        label="commitments",
    ),
    MeasurePartitionLabel(
        axis_values=(
            ("financialFlowType", "Disbursement"),
            ("measuredProperty", "amount"),
            ("nettingBasis", "Gross"),
            ("populationType", "FinancialFlow"),
        ),
        label="gross disbursements",
    ),
    MeasurePartitionLabel(
        axis_values=(
            ("financialFlowType", "Disbursement"),
            ("measuredProperty", "amount"),
            ("nettingBasis", "Net"),
            ("populationType", "FinancialFlow"),
        ),
        label="net disbursements",
    ),
    MeasurePartitionLabel(
        axis_values=(
            ("financialFlowType", "Disbursement"),
            ("measuredProperty", "attributedAmount"),
            ("nettingBasis", "Gross"),
            ("populationType", "FinancialFlow"),
        ),
        # Only ONE imputed multilateral StatVars hold this partition, all gross
        # disbursements of one aid type, so the aid type is the measure's name.
        label="imputed multilateral ODA",
        named_values=(
            ("crsTypeOfFlow", "CRSTypeOfFlow10"),
            ("dac2aAidType", "DAC2aAidType106"),
        ),
    ),
    MeasurePartitionLabel(
        axis_values=(
            ("financialFlowType", "Disbursement"),
            ("measuredProperty", "grantEquivalent"),
            ("populationType", "FinancialFlow"),
        ),
        label="grant equivalents",
    ),
)


# Nouns for IDS and WEO measures, which no measure partition covers. A heading would
# otherwise read the graph name, which carries the unit inside it: "Debt service on
# external debt (current US$)". Each entry is keyed on exact five-tuple and constraint
# values, read on the dev graph, and matched with `label_for_pairs`. A noun never names
# the price basis (`measurementQualifier`) or the denominator, which belong to the unit.
# No entry keys on `financialFlowType = Disbursement` alone, since CRS disbursements carry
# that value and take their partition's label (`_verify_measure_nouns`).
_POPULATION = "populationType"
_PROPERTY = "measuredProperty"
_STAT_TYPE = "statType"

_GDP_BASIS = (
    ("productTaxBasis", "IncludingProductTaxes"),
    ("fixedCapitalConsumptionBasis", "GrossOfFixedCapitalConsumption"),
)


def _noun(label: str, *pairs: tuple[str, str]) -> MeasurePartitionLabel:
    return MeasurePartitionLabel(axis_values=pairs, label=label)


def _ids(label: str, population: str, axis: str, value: str) -> MeasurePartitionLabel:
    return _noun(
        label,
        (_POPULATION, population),
        (_PROPERTY, "amount"),
        (_STAT_TYPE, "measuredValue"),
        (axis, value),
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
    _ids("external debt service", "FinancialFlow", "financialFlowType", "ServicingPayment"),
    _ids(
        "interest payments on external debt",
        "FinancialFlow",
        "financialFlowType",
        "InterestPayment",
    ),
    _ids(
        "principal repayments on external debt",
        "FinancialFlow",
        "financialFlowType",
        "PrincipalRepayment",
    ),
    # Every IDS StatVar declaring `netOf` nets principal repayments off disbursements.
    _noun(
        "net flows on external debt",
        *_IDS_FLOW,
        ("financialFlowType", "Disbursement"),
        ("netOf", "PrincipalRepayment"),
        ("nettingBasis", "Net"),
    ),
    # The four IDS StatVars declaring `includesIMFCredit = WithIMFCredit` are long-term
    # disbursements and principal repayments. The shared entry names a pair of them.
    _noun("flows on long-term external debt with IMF credit", *_IDS_FLOW, *_WITH_IMF_CREDIT),
    _noun(
        "disbursements on long-term external debt with IMF credit",
        *_IDS_FLOW,
        *_WITH_IMF_CREDIT,
        ("financialFlowType", "Disbursement"),
    ),
    _noun(
        "principal repayments on long-term external debt with IMF credit",
        *_IDS_FLOW,
        *_WITH_IMF_CREDIT,
        ("financialFlowType", "PrincipalRepayment"),
    ),
    _ids(
        "outstanding external debt",
        "FinancialPosition",
        "financialPositionType",
        "AmountOutstanding",
    ),
    _ids(
        "principal arrears on external debt",
        "FinancialPosition",
        "financialPositionType",
        "PrincipalArrears",
    ),
    _ids(
        "interest arrears on external debt",
        "FinancialPosition",
        "financialPositionType",
        "InterestArrears",
    ),
    _ids(
        "undisbursed external debt",
        "FinancialPosition",
        "financialPositionType",
        "AmountUndisbursed",
    ),
    # IMF World Economic Outlook.
    _noun(
        "GDP",
        (_POPULATION, "EconomicProduction"),
        (_PROPERTY, "valueAdded"),
        (_STAT_TYPE, "measuredValue"),
        *_GDP_BASIS,
    ),
    _noun(
        "GDP",
        (_POPULATION, "EconomicProduction"),
        (_PROPERTY, "valueAdded"),
        (_STAT_TYPE, "measuredValue"),
        *_GDP_BASIS,
        ("currencyConversionBasis", "PurchasingPowerParityRate"),
    ),
    _noun(
        "output gap",
        (_POPULATION, "EconomicProduction"),
        (_PROPERTY, "outputGap"),
        (_STAT_TYPE, "measuredValue"),
        *_GDP_BASIS,
    ),
    _noun(
        "gross national savings",
        (_POPULATION, "EconomicUnit"),
        (_PROPERTY, "saving"),
        (_STAT_TYPE, "measuredValue"),
        ("fixedCapitalConsumptionBasis", "GrossOfFixedCapitalConsumption"),
    ),
    _noun(
        "gross capital formation",
        (_POPULATION, "FinalExpenditure"),
        (_PROPERTY, "amount"),
        (_STAT_TYPE, "measuredValue"),
        ("finalExpenditureType", "CapitalFormation"),
        ("fixedCapitalConsumptionBasis", "GrossOfFixedCapitalConsumption"),
    ),
    # Base Data Commons series that WEO questions elect (World Development Indicators).
    _noun(
        "GDP",
        (_POPULATION, "EconomicActivity"),
        (_PROPERTY, "amount"),
        (_STAT_TYPE, "measuredValue"),
        ("activitySource", "GrossDomesticProduction"),
    ),
    _noun(
        "GDP growth",
        (_POPULATION, "EconomicActivity"),
        (_PROPERTY, "amount"),
        (_STAT_TYPE, "growthRate"),
        ("activitySource", "GrossDomesticProduction"),
    ),
    _noun(
        "health expenditure",
        (_POPULATION, "EconomicActivity"),
        (_PROPERTY, "amount"),
        (_STAT_TYPE, "measuredValue"),
        ("activitySource", "ExpenditureActivity"),
        ("expenditureType", "HealthcareExpenditure"),
    ),
    _noun(
        "government debt",
        (_POPULATION, "Debt"),
        (_PROPERTY, "amount"),
        (_STAT_TYPE, "measuredValue"),
        ("debtor", "Government"),
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
    }


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
    }
    return declaration, rows


def _declared_default_to_dict(default: DeclaredDefault | None) -> dict | None:
    if default is None:
        return None
    return {
        "code": default.code,
        "when_bound_under": list(default.when_bound_under),
        "when_named": list(default.when_named),
    }


def _validate_declared_defaults(
    code_systems: Sequence[CodeSystemConfig], code_rows_by_system: dict[str, list[dict]]
) -> None:
    """Validate each default against exported rows, raising `SystemExit` on invalid entries.

    Its code must belong to its own system. `when_bound_under` codes must belong to other systems.
    `when_named` entries are non-empty lowercase strings.
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
    no CRS keys, passes because the graph walk resolves it.

    A decided StatVar carries every code system each exported row of its five-tuple declares.
    Each five-tuple's reverse read splits on those systems' enum codes, smallest non-spine
    system first and spine systems last, so the read reaches StatVars the spine enumeration did
    not. Values come from the enum, since exported rows would hide a value no row shows. The forward read of each unexported subject confirms its five-tuple, including the
    denominator, and its key set. Truncated or exhausted reads refuse, as does a subject the
    forward read finds no node for.

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
        shared = frozenset.intersection(*keys)
        systems = sorted(
            (system for system in code_rows_by_system if system in shared),
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
            for system in systems
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
_MEASURE_VALUE_ARC_PROPERTIES = ("name", "alternateName", "shortName")


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
    text overrides.

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
    return {"axis": axis.axis, "kind": axis.kind, "role": axis.role, "prefer": list(axis.prefer)}


def _measure_partition_label_to_dict(partition_label: MeasurePartitionLabel) -> dict:
    return {
        "axis_values": [list(pair) for pair in partition_label.axis_values],
        "label": partition_label.label,
        "named_values": [list(pair) for pair in partition_label.named_values],
    }


def _dataset_to_dict(dataset: DatasetDeclaration) -> dict:
    return {
        "provenance": dataset.provenance,
        "label": dataset.label,
        "synonyms": sorted(dataset.synonyms),
        "opt_in": dataset.opt_in,
        "absent_observation": dataset.absent_observation,
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
    reader: GraphReader,
    *,
    code_systems: Sequence[CodeSystemConfig] = CODE_SYSTEMS,
    measure_axes: Sequence[MeasureAxis] = MEASURE_AXES,
    measure_partition_labels: Sequence[MeasurePartitionLabel] = _MEASURE_PARTITION_LABELS,
    measure_nouns: Sequence[MeasurePartitionLabel] = _MEASURE_NOUNS,
    datasets: Sequence[DatasetDeclaration] = DATASETS,
) -> dict:
    """Build every document block the graph and the authored config determine, sorted.

    StatVar rows carry no `provenances` and the result has no stamp, so it does not load as an
    index. `export_index` completes it from a coverage build. The drift test compares it with the
    committed export, which needs graph credentials alone.
    """
    declarations = []
    codes = []
    code_rows_by_system: dict[str, list[dict]] = {}
    for config in code_systems:
        declaration, rows = export_code_system(reader, config)
        declarations.append(declaration)
        codes.extend(rows)
        code_rows_by_system[config.code_system] = rows
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
    _verify_measure_partition_labels(statvars, measure_axes, measure_partition_labels)
    _verify_measure_nouns(statvars, measure_nouns)
    _verify_measure_label_named_values(statvars, codes, measure_axes, measure_partition_labels)

    return {
        "declarations": declarations,
        "datasets": sorted(
            (_dataset_to_dict(dataset) for dataset in datasets), key=lambda row: row["provenance"]
        ),
        "codes": codes,
        "statvars": statvars,
        "measure_axes": [_measure_axis_to_dict(axis) for axis in measure_axes],
        "measure_values": export_measure_values(reader, statvars, measure_axes),
        "measure_partition_labels": [
            _measure_partition_label_to_dict(label) for label in measure_partition_labels
        ],
        "measure_nouns": [_measure_partition_label_to_dict(noun) for noun in measure_nouns],
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
    reader: GraphReader,
    *,
    graph_endpoint: str,
    now: datetime,
    coverage: CoverageJoin,
    code_systems: Sequence[CodeSystemConfig] = CODE_SYSTEMS,
    measure_axes: Sequence[MeasureAxis] = MEASURE_AXES,
    measure_partition_labels: Sequence[MeasurePartitionLabel] = _MEASURE_PARTITION_LABELS,
    measure_nouns: Sequence[MeasurePartitionLabel] = _MEASURE_NOUNS,
    datasets: Sequence[DatasetDeclaration] = DATASETS,
) -> dict:
    """Build the index document: the graph blocks, each row's provenances from `coverage`, and
    the stamp.

    Tests can pass smaller config tables for fixture graphs. Empty `measure_axes` yields an empty
    `measure_values` list. The finished document must load: a document `parse_index` refuses,
    such as one whose declared dataset has no row or whose datasets share a five-tuple, raises
    `SystemExit`.
    """
    blocks = export_graph_blocks(
        reader,
        code_systems=code_systems,
        measure_axes=measure_axes,
        measure_partition_labels=measure_partition_labels,
        measure_nouns=measure_nouns,
        datasets=datasets,
    )
    statvars_without_coverage = _join_provenances(blocks["statvars"], coverage)
    _verify_complete_systems(blocks["statvars"], code_systems)

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
    except (IndexFormatError, DatasetsShareFiveTuple) as exc:
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
    coverage = load_coverage_join(_coverage_artifact_uri())
    reader = export_reader(f"{base}/core/api", os.environ.get("SEARCH_GRAPH_TOKEN"))
    try:
        document = export_index(
            reader, graph_endpoint=base, now=datetime.now(UTC), coverage=coverage
        )
    finally:
        reader.close()
    _write_index(Path(args.out), document)
    stamp = document["stamp"]
    print(
        f"wrote {sum(stamp['code_counts'].values())} code rows, "
        f"{len(document['statvars'])} StatVar rows "
        f"({stamp['statvars_without_coverage']} not in coverage build "
        f"{stamp['coverage_build']}), and "
        f"{len(document['measure_values'])} measure values to {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
