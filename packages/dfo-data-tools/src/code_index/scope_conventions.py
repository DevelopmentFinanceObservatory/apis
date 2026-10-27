"""The declared conventions a part's scope choices default to.

`SECTOR_GROUPINGS` names the sets of CRS purpose codes a sector concept word reads as.
`MEASURE_BASIS_CONVENTIONS` names the measure basis each part shape and window defaults to,
`BASIS_MEASURES` what each basis binds in an identity, and `BASIS_RESTRICTIONS` the code
systems besides the sector whose admitted code makes a part a restricted one.
`PROVIDER_CONVENTIONS` names whose spending a part counts by default,
`REDISTRIBUTED_PROVIDERS` the providers no pool takes the inclusive scope with, and
`CONTRIBUTION_TOTALS` the published donor totals that count core contributions to
multilaterals, which take neither the bilateral nor the inclusive scope. `OPTION_LABELS`
words the options the rules build from code and group labels, and `REGION_READINGS` names
the graph regions a DAC dataset counts as the OECD's region. `search.scope`
reads them all, and `graph_release.declarations` hashes the tables beside the other
declarations, so a change to one moves the declarations hash and the scope rule version.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from code_index.declarations import (
    CRS,
    CRS_FLOW_PRIVATE_SECTOR_INSTRUMENTS,
    CRS_PURPOSE_EMERGENCY_RESPONSE,
    CRS_PURPOSE_HEALTH,
    CRS_PURPOSE_RECONSTRUCTION_RELIEF,
    CRS_PURPOSE_REPRODUCTIVE_HEALTH,
    DAC1,
    DAC2A,
)

MeasureBasis = Literal["gross", "net", "grant_equivalent", "commitments", "oecd_headline"]
"""Every measure basis a part can be counted on. `oecd_headline` is the OECD's headline
basis per year: net flows to 2017 and grant equivalents from 2018, as two period
components."""

ProviderScope = Literal["bilateral", "inclusive"]
"""Whose spending a part counts: the donor's bilateral flows, or those plus its imputed share
of multilateral spending."""

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
class ProviderConvention:
    """Whose spending a part counts by default, and when.

    A part takes `scope` when it names a donor (`names_donor`), its shape is one of `shapes`,
    its dataset is `dataset` and its words ask for a total (`asks_total`), and the scope serves
    every year it asks for. `rule` names the convention on the scope choice it settles, and
    `note` says why it holds."""

    rule: str
    scope: ProviderScope
    shapes: tuple[PartShape, ...]
    dataset: str
    names_donor: bool
    asks_total: bool
    note: str


@dataclass(frozen=True)
class ContributionTotal:
    """A published donor total that counts the donor's bilateral flows and its core
    contributions to multilateral organisations, in the year it pays them.

    A part counts it when its identity is served from `dataset` and binds `code` on
    `code_system`. `note` says why the total takes neither the bilateral scope nor the
    inclusive one."""

    dataset: str
    code_system: str
    code: str
    note: str


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

# ONE's analytical groupings of CRS sectors. Health joins Health (120) and Population
# policies and programmes and reproductive health (130).
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
        codes=(CRS_PURPOSE_HEALTH, CRS_PURPOSE_REPRODUCTIVE_HEALTH),
        words=("health",),
        label="health incl. population and reproductive health (CRS 120 + 130)",
    ),
    SectorGrouping(
        concept="humanitarian",
        code_system=_CRS_PURPOSE,
        codes=(
            CRS_PURPOSE_EMERGENCY_RESPONSE,
            CRS_PURPOSE_RECONSTRUCTION_RELIEF,
            "CRSPurposeCode740",
        ),
        words=("humanitarian aid",),
        label="humanitarian aid (CRS 720 + 730 + 740)",
    ),
)


def grouping_of(code_system: str, codes: Iterable[str]) -> SectorGrouping | None:
    """The declared grouping whose codes on `code_system` are exactly `codes`, None when
    none is."""
    held = frozenset(codes)
    return next(
        (
            g
            for g in SECTOR_GROUPINGS
            if g.code_system == code_system and frozenset(g.codes) == held
        ),
        None,
    )


@dataclass(frozen=True)
class OptionLabels:
    """The wording of the scope options whose label is built from the part: each a
    standalone noun phrase naming its subject, since a reader sees the label alone ("Counted
    as: health, CRS 120 alone · Africa's countries only").

    `sector_code` names one CRS purpose code from its `label` and `code` number, and
    `sector_code_alone` one code of a grouping offered without the others.
    `region_with_residuals` and `region_members_only` name a region's two readings from its
    `region` name and its possessive `region_of`. `region_as_oecd_counts` and
    `region_place_members` name the two readings of a region the graph and the OECD count
    differently (`REGION_READINGS`): the OECD's region with its `count` of member countries
    and its residuals, and the graph region's `count` countries. `measure_basis` names a measure basis from
    its `words` (`BasisMeasure.words`), the `prices` of the identity's price basis
    ("current", "constant 2024") and its `currency` ("USD", "donor currency"), and
    `measure_basis_unpriced` from its words alone, for an identity that states no price
    basis. `oecd_headline` names the OECD's headline basis per year. `provider_bilateral`,
    `provider_inclusive` and `provider_contributions` name the three provider scopes, and
    `provider_inclusive_years` the inclusive scope over the `years` both of its tables cover
    ("2011 to 2024", "2023"), when they set the part's window."""

    sector_code: str
    sector_code_alone: str
    region_with_residuals: str
    region_members_only: str
    measure_basis: str
    measure_basis_unpriced: str
    oecd_headline: str
    provider_bilateral: str
    provider_inclusive: str
    provider_inclusive_years: str
    region_as_oecd_counts: str
    region_place_members: str
    provider_contributions: str


OPTION_LABELS = OptionLabels(
    sector_code="{label} (CRS {code})",
    sector_code_alone="{label}, CRS {code} alone",
    region_with_residuals="{region} incl. regional and unspecified amounts",
    region_members_only="{region_of} countries only",
    measure_basis="{words}, {prices} {currency}",
    measure_basis_unpriced="{words}",
    oecd_headline="OECD headline (net flows to 2017, grant equivalents from 2018)",
    provider_bilateral="bilateral flows only",
    provider_inclusive="bilateral plus imputed multilateral",
    provider_inclusive_years="bilateral plus imputed multilateral, {years}",
    region_as_oecd_counts="{region} as the OECD counts it, {count} countries incl. regional and "
    "unspecified amounts",
    region_place_members="{region_of} {count} countries (UN regions)",
    provider_contributions="bilateral plus core contributions to multilaterals",
)

CONVENTION_CURRENCY = "USD"
"""The currency the datasets the basis conventions read publish their amounts in, which a
measure-basis option names when no unit request converts the part."""

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


@dataclass(frozen=True)
class RegionReading:
    """A graph region a request names and the OECD region a DAC dataset counts in its place.

    `region` is the graph place the resolver reads the request's words as (`SubSaharanAfrica`
    for "sub-Saharan Africa"), `total` the DAC regional total whose group the OECD counts
    (`DACSouthOfSahara`, a `code_index.declarations.TOTAL_GROUPS` row summing its members and
    residuals), and `name` the reader's name of the region in option labels."""

    region: str
    total: str
    name: str


# The graph regions a summed recipient region on a DAC dataset reads as the OECD's region:
# its members as the DAC lists them, its regional total and its residual buckets
# (`search.engine.region.place_roles`). The graph's members differ from the OECD's for sub-
# Saharan Africa (Sudan), Asia (Cyprus, Japan, Türkiye), Europe (the OECD lists 13 ODA-eligible
# countries), the Middle East (the graph's Western Asia: the Caucasus, Cyprus, Türkiye against
# Iran), North and East Africa (Sudan); Middle and West Africa and South America hold the same
# members under a total keyed to the DAC's own group. A region whose DAC total is keyed to the
# graph place itself (Africa, the Americas, Oceania, the Caribbean, Central America, Southern
# Africa, the Pacific subregions) needs no row. East and South Asia differ from the DAC's Far East
# Asia and South and Central Asia, so no row reads them as those.
REGION_READINGS: tuple[RegionReading, ...] = (
    RegionReading("SubSaharanAfrica", "DACSouthOfSahara", "sub-Saharan Africa"),
    RegionReading("asia", "DACAsia", "Asia"),
    RegionReading("europe", "DACEurope", "Europe"),
    RegionReading("WesternAsia", "DACMiddleEast", "the Middle East"),
    RegionReading("NorthernAfrica", "DACNorthOfSahara", "North Africa"),
    RegionReading("EasternAfrica", "DACEasternAfrica", "East Africa"),
    RegionReading("MiddleAfrica", "DACMiddleAfrica", "Middle Africa"),
    RegionReading("WesternAfrica", "DACWesternAfrica", "West Africa"),
    RegionReading("southamerica", "DACSouthAmerica", "South America"),
)


# The measure basis each part shape and window is counted on, after the OECD's own
# reporting: the DAC1 headline for a donor's total, DAC2a receipts for a recipient's,
# gross CRS disbursements wherever a sector, channel, instrument or modality is bound.
MEASURE_BASIS_CONVENTIONS: tuple[BasisConvention, ...] = (
    BasisConvention(
        shape="donor_only",
        window="from_2018",
        basis="grant_equivalent",
        dataset=DAC1,
        note="OECD headline ODA, grant equivalents (DAC1 measure 11010)",
    ),
    BasisConvention(
        shape="donor_only",
        window="before_2018",
        basis="net",
        dataset=DAC1,
        note="OECD headline ODA before 2018, net flows",
    ),
    BasisConvention(
        shape="donor_only",
        window="crossing_2018",
        basis="oecd_headline",
        dataset=DAC1,
        note="OECD headline ODA per year: net flows to 2017, grant equivalents from 2018",
    ),
    BasisConvention(
        shape="recipient_all_sectors",
        window="any",
        basis="net",
        dataset=DAC2A,
        note="OECD receipts, net disbursements (DAC2a measure 206)",
    ),
    BasisConvention(
        shape="sector_or_restriction",
        window="any",
        basis="gross",
        dataset=CRS,
        note="CRS gross disbursements",
    ),
)

# A named donor's total of a CRS sector, or of a part the CRS serves for a recipient, counts
# its bilateral flows plus its imputed share of multilateral spending, after ONE's method,
# which attributes the multilateral organisations' own spending in the sector to the donors in
# proportion to their core contributions. Every other part counts bilateral flows. DAC1 donor
# totals already count core contributions to multilaterals, and DAC2a's imputed line is
# netted differently, so neither takes the inclusive scope.
PROVIDER_CONVENTIONS: tuple[ProviderConvention, ...] = (
    ProviderConvention(
        rule="named_donor_total",
        scope="inclusive",
        shapes=("sector_or_restriction", "recipient_all_sectors"),
        dataset=CRS,
        names_donor=True,
        asks_total=True,
        note=(
            "a named donor's total counts its bilateral flows plus its imputed share of "
            "multilateral spending (ONE method)"
        ),
    ),
)

# The providers the imputed multilateral table carries rows for that are themselves
# multilaterals whose own outflows its rows attribute to other providers. The EU institutions
# report to the CRS as a provider and hold an imputed share of the multilaterals they fund, and
# ONE's method attributes their own spending to the EU member states that fund them. A pool
# naming one of them beside another provider counts those outflows twice, as the multilateral's
# own flows and inside its funders' imputed shares, so it takes no inclusive scope. The graph
# types every multilateral an `Organization`, as it types a private foundation, and the code
# index marks none, so this table is the only record of the grouping.
REDISTRIBUTED_PROVIDERS: tuple[str, ...] = ("EuropeanUnionInstitutions",)

PROVIDER_DEFAULT_RULE = "bilateral_flows"
"""The rule of a part no provider convention settles. It counts the donor's bilateral flows."""

# DAC1's total ODA counts a donor's bilateral flows and its core contributions to the
# multilateral organisations, in the year it pays them. The imputed multilateral table instead
# attributes the multilaterals' own spending back to their funders, so the two differ in what
# they count, in timing and in amount, and the total takes no inclusive scope. DAC1 publishes
# bilateral ODA as another aid type, a total of another table whose whole need not hold a part
# the CRS counts, so the total takes no bilateral scope either.
CONTRIBUTION_TOTALS: tuple[ContributionTotal, ...] = (
    ContributionTotal(
        dataset=DAC1,
        code_system="dac1AidType",
        code="DAC1AidType1010",
        note="DAC1 total ODA counts bilateral flows plus core contributions to multilaterals",
    ),
)

CONTRIBUTIONS_RULE = "contribution_total"
"""The rule of a part that counts a `CONTRIBUTION_TOTALS` total, whose only provider scope is
the one it publishes."""

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
        oda_flows=(ODA_FLOW, CRS_FLOW_PRIVATE_SECTOR_INSTRUMENTS),
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
