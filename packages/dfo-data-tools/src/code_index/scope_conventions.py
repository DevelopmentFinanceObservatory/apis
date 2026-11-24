"""The declared conventions a part's scope choices default to.

`SECTOR_GROUPINGS` names the sets of CRS purpose codes a sector concept word reads as.
`MEASURE_BASIS_CONVENTIONS` names the measure basis each part shape and window defaults to,
`BASIS_MEASURES` what each basis binds in an identity, and `BASIS_RESTRICTIONS` the code
systems besides the sector whose admitted code makes a part a restricted one.
`PROVIDER_CONVENTIONS` names whose spending a part counts by default,
`PROVIDER_SCOPE_RULES` the rule each provider scope names where no convention settles it,
`REDISTRIBUTED_PROVIDERS` the providers no pool takes a scope counting multilateral aid with,
and the group whose members fund each, `PUBLISHED_PROVIDER_SCOPES` the provider scope a
published donor total counts, `INCLUSIVE_SOURCES` the composite the inclusive scope binds on
each table that offers it, `BILATERAL_SOURCES` the composite the bilateral scope binds on a
published total that counts core contributions as recipients and the population of
destinations its components read, and `DONOR_ORGANISATIONS` the organisations a published donor
total counts as donors. `OPTION_LABELS` words the options the rules build from code and group
labels, and `REGION_READINGS` names the graph regions a DAC dataset counts as the OECD's
region. `search.scope` reads them all, and `graph_release.declarations` hashes the tables
beside the other declarations, so a change to one moves the declarations hash and the scope
rule version.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Literal

from code_index.declarations import (
    CRS,
    CRS_FLOW_PRIVATE_SECTOR_INSTRUMENTS,
    CRS_INCLUSIVE,
    CRS_PURPOSE_EMERGENCY_RESPONSE,
    CRS_PURPOSE_HEALTH,
    CRS_PURPOSE_RECONSTRUCTION_RELIEF,
    CRS_PURPOSE_REPRODUCTIVE_HEALTH,
    DAC1,
    DAC1_BILATERAL_ODA,
    DAC1_MULTILATERAL_ODA,
    DAC1_TOTAL_ODA,
    DAC2A,
    DAC2A_BILATERAL,
    DAC2A_INCLUSIVE,
    DAC2A_TOTAL_ODA,
    DAC_RECIPIENT_LIST_DATASETS,
)
from code_index.multilaterals import ORGANIZATION_TYPE, party_facts
from code_index.query import CodeIndex
from code_index.shared import per_index
from code_index.types import ComponentDestinations, PartyRow

MeasureBasis = Literal["gross", "net", "grant_equivalent", "commitments", "oecd_headline"]
"""Every measure basis a part can be counted on. `oecd_headline` is the OECD's headline
basis per year: net flows to 2017 and grant equivalents from 2018, as two period
components."""

ProviderScope = Literal["bilateral", "inclusive", "contributions", "multilateral"]
"""Whose spending a part counts: the donor's bilateral flows, those plus its imputed share of
multilateral spending, those plus its core contributions to multilateral organisations, or
the core contributions alone, as a published donor total counts them
(`PUBLISHED_PROVIDER_SCOPES`). Each scope is a set of money streams, and two scopes can count
the same stream: bilateral flows sit in three of them, and the inclusive scope's imputed share
is the same donor money the core contributions count when it is paid in."""

PartShape = Literal["donor_only", "recipient_all_sectors", "sector_or_restriction"]
"""The part shapes the basis conventions distinguish. `donor_only` names a donor and no
recipient or sector. `recipient_all_sectors` binds a recipient and no sector, from any
donors. `sector_or_restriction` binds a sector, or restricts a channel, an instrument or a
modality."""

WindowClass = Literal["from_2018", "before_2018", "crossing_2018", "any"]
"""Where a part's window lies against 2018, the first year the OECD publishes grant
equivalents as its headline: every year from 2018, every year before it, years on both
sides, or any window."""


NETTING_BASIS_SLOT = "nettingBasis"
"""The slot whose value `BasisMeasure.netting_basis` names."""


@dataclass(frozen=True)
class BasisMeasure:
    """What an identity counted on `basis` binds.

    `population_type` and `measured_property` are its shape's, and `netting_basis` the
    value its `NETTING_BASIS_SLOT` slot binds, None for a measure that carries no netting.
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
class PublishedProviderScope:
    """The provider scope a published donor total counts.

    A part counts `scope` when its identity is served from `dataset`, binds `code` on
    `code_system`, and names donors alone: countries, `DONOR_ORGANISATIONS`, or totals of a
    group made only of countries. `rule` names the row on the scope choice it settles, and
    `note` says what the total counts."""

    dataset: str
    code_system: str
    code: str
    scope: ProviderScope
    rule: str
    note: str


@dataclass(frozen=True)
class InclusiveSource:
    """The stacked composite (`code_index.declarations.COMPOSITES`) the inclusive scope binds
    as the source of a part whose bilateral answer `dataset` serves: the donor's bilateral
    flows plus its imputed share of multilateral spending. The scope is offered on the
    measure bases `bases` alone, the ones the composite's imputed line is added on. `note`
    says what the composite adds."""

    dataset: str
    composite: str
    bases: tuple[MeasureBasis, ...]
    note: str


@dataclass(frozen=True)
class BilateralSource:
    """The stacked composite (`code_index.declarations.COMPOSITES`) the bilateral scope binds
    on a part counting the published total `code` on `code_system` of `dataset`
    (`PUBLISHED_PROVIDER_SCOPES`), which counts a donor's bilateral flows to countries and
    its core contributions to the multilateral agencies it lists as recipients. A component
    of `dataset` that binds its destinations to the bilateral side
    (`ComponentRecipe.destinations`) reads the population this row declares
    (`RolePopulation`).

    `rebound` is the dataset's published total of the bilateral side, which a component
    binds when the identity leaves its destination out, and `spanning` the total of both
    sides, which no such component reads; `rebound_words` and `spanning_words` name the two
    for a reader. `rule` names the default a fresh part counting the total settles on, and
    `note` says what the composite counts. `label` names the bilateral option in place of
    `OptionLabels.provider_bilateral`, and `total_label` the option holding the total itself
    in place of `OptionLabels.provider_contributions`, since the total counts every
    destination the table lists."""

    dataset: str
    code_system: str
    code: str
    composite: str
    spanning: str
    rebound: str
    spanning_words: str
    rebound_words: str
    rule: str
    note: str
    label: str
    total_label: str


@dataclass(frozen=True)
class ProviderScopeRule:
    """The rule a provider option of `scope` names where no convention
    (`PROVIDER_CONVENTIONS`) or published total (`PUBLISHED_PROVIDER_SCOPES`) settles it.
    `note` says what the scope counts."""

    scope: ProviderScope
    rule: str
    note: str


@dataclass(frozen=True)
class RedistributedProvider:
    """A multilateral that reports as a provider and whose own outflows a scope counting
    multilateral aid also counts inside its funders' shares. `provider` is its dcid, and
    `funders` the group (a membership policy ref) whose members fund it with core
    contributions."""

    provider: str
    funders: str


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
    basis or divides by something other than a head count. `oecd_headline` names the
    OECD's headline basis per year. `provider_bilateral`, `provider_inclusive`,
    `provider_contributions` and `provider_multilateral` name the four provider scopes,
    `provider_inclusive_net` the inclusive scope on a part counted on net
    flows, which adds the imputed line as published, and `provider_years` a scope's `label`
    followed by the `years` its option serves ("2011 to 2024", "2023") where they set the
    part's window. `provider_rebound` names a scope's `label` read at the total of the
    bilateral side in place of the total of both sides the part names
    (`BilateralSource.rebound_words`, `BilateralSource.spanning_words`)."""

    sector_code: str
    sector_code_alone: str
    region_with_residuals: str
    region_members_only: str
    measure_basis: str
    measure_basis_unpriced: str
    oecd_headline: str
    provider_bilateral: str
    provider_inclusive: str
    provider_years: str
    region_as_oecd_counts: str
    region_place_members: str
    provider_contributions: str
    provider_multilateral: str
    provider_inclusive_net: str
    provider_rebound: str


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
    provider_years="{label}, {years}",
    region_as_oecd_counts="{region} as the OECD counts it, {count} countries incl. regional and "
    "unspecified amounts",
    region_place_members="{region_of} {count} countries (UN regions)",
    provider_contributions="bilateral plus core contributions to multilaterals",
    provider_multilateral="core contributions to multilaterals",
    provider_inclusive_net="net bilateral plus imputed multilateral as published, by convention",
    provider_rebound="{label}, to {rebound} in place of {spanning}",
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
# totals already count core contributions to multilaterals, so they take no inclusive scope.
# DAC2a's total ODA counts core contributions too, wherever it lists a multilateral agency as a
# recipient, so a fresh DAC2a part reads the bilateral destinations alone (`BILATERAL_SOURCES`)
# and keeps them unless a switch or the request's words ask for the imputed share
# (`INCLUSIVE_SOURCES`) or the total itself.
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

_EU_INSTITUTIONS = "EuropeanUnionInstitutions"
_EUROPEAN_UNION = "EuropeanUnion"

# The providers the imputed multilateral table carries rows for that are themselves
# multilaterals whose own outflows its rows attribute to other providers. The EU institutions
# report to the CRS as a provider and hold an imputed share of the multilaterals they fund, and
# ONE's method attributes their own spending to the EU member states that fund them. A pool
# naming one of them beside another provider counts those outflows twice, as the multilateral's
# own flows and inside its funders' imputed shares, so it takes no inclusive scope. DAC1's
# total ODA counts the same outflows twice for a pool of the EU institutions and an EU member
# state, as the EU institutions' own aid and inside the member's core contributions to them,
# so such a pool has no provider record there. A donor outside the European Union funds the
# institutions with no core contribution. The graph types every multilateral an
# `Organization`, as it types a private foundation, and the code index marks none, so this
# table is the only record of the grouping.
REDISTRIBUTED_PROVIDERS: tuple[RedistributedProvider, ...] = (
    RedistributedProvider(provider=_EU_INSTITUTIONS, funders=_EUROPEAN_UNION),
)

# The rule each provider scope names on a part no convention or published row settles: the
# bilateral default, or a scope an entry or a posted identity chose.
PROVIDER_SCOPE_RULES: tuple[ProviderScopeRule, ...] = (
    ProviderScopeRule(
        scope="bilateral",
        rule="bilateral_flows",
        note="the donor's bilateral flows alone",
    ),
    ProviderScopeRule(
        scope="inclusive",
        rule="bilateral_plus_imputed",
        note="the donor's bilateral flows plus its imputed share of multilateral spending",
    ),
)

# The provider scope each published total counts for a bilateral donor. DAC1's total ODA
# counts the donor's bilateral flows and its core contributions to the multilateral
# organisations, in the year it pays them. The imputed multilateral table attributes the
# multilaterals' own spending back to their funders, so the two differ in what they count, in
# timing and in amount. DAC1's bilateral ODA counts the bilateral flows alone, and its
# multilateral ODA the core contributions alone. A part reading one row is offered every other
# row of its dataset and code system as an option that binds that row's code. DAC2a's total
# ODA counts the donor's bilateral flows to countries and regions and its core contributions to
# the multilateral agencies it lists as recipients, so over every destination it is DAC1's
# total ODA by recipient. Its bilateral scope binds the composite that reads the bilateral
# destinations alone (`BILATERAL_SOURCES`), and its inclusive scope DAC2a's own imputed line
# (`INCLUSIVE_SOURCES`).
PUBLISHED_PROVIDER_SCOPES: tuple[PublishedProviderScope, ...] = (
    PublishedProviderScope(
        dataset=DAC1,
        code_system="dac1AidType",
        code=DAC1_TOTAL_ODA,
        scope="contributions",
        rule="contribution_total",
        note="DAC1 total ODA counts bilateral flows plus core contributions to multilaterals",
    ),
    PublishedProviderScope(
        dataset=DAC1,
        code_system="dac1AidType",
        code=DAC1_BILATERAL_ODA,
        scope="bilateral",
        rule="bilateral_total",
        note="DAC1 bilateral ODA counts bilateral flows alone",
    ),
    PublishedProviderScope(
        dataset=DAC1,
        code_system="dac1AidType",
        code=DAC1_MULTILATERAL_ODA,
        scope="multilateral",
        rule="multilateral_total",
        note="DAC1 multilateral ODA counts core contributions to multilaterals alone",
    ),
    PublishedProviderScope(
        dataset=DAC2A,
        code_system="dac2aAidType",
        code=DAC2A_TOTAL_ODA,
        scope="contributions",
        rule="contribution_recipient_total",
        note=(
            "DAC2a total ODA counts bilateral flows to countries and regions plus core "
            "contributions to the multilateral agencies it lists as recipients"
        ),
    ),
)


@dataclass(frozen=True)
class ProviderWhole:
    """The provider option that holds the whole of a part's provider axis on `dataset`.

    `scope` is the whole of a part over the dataset's whole recipient population (no
    recipient named, or the population's own total), and `alternatives` the other scopes
    that also hold such a part's streams, which a client may ask for in its place.
    `named_scope` is the whole of a part naming recipients below the population's total (a
    country or a region), None where it is `scope`. `recipient` is the recipient total the
    whole reads beside `scope`, rebinding the recipient side, None where `scope` keeps the
    part's recipients. `note` says why."""

    dataset: str
    scope: ProviderScope
    alternatives: tuple[ProviderScope, ...]
    note: str
    named_scope: ProviderScope | None = None
    recipient: str | None = None


# The whole of each table's provider axis. DAC1's total ODA counts the bilateral flows and the
# core contributions, every stream its bilateral and multilateral ODA count; its inclusive
# option is never offered there. On the CRS the inclusive scope adds the imputed share to the
# bilateral flows, the only other option. DAC2a offers both: the inclusive scope counts the
# donor's multilateral money as its imputed share of the agencies' spending, and total ODA
# counts it as core contributions in the year paid, which go to the agencies and so to no
# country or region. Over a named country or region total ODA counts the bilateral flows
# alone, so the inclusive scope is the whole there. Over the whole population, a part listing
# every recipient, naming the developing countries' total or a multilateral agency, or holding
# total ODA, the whole is total ODA to all recipients, the OECD's denominator of a donor's
# bilateral share and the figure DAC1 serves for the same donor: the provider and the
# recipient side both move, so a listing's every recipient divides by one figure. The
# inclusive scope is the alternative.
PROVIDER_WHOLES: tuple[ProviderWhole, ...] = (
    ProviderWhole(
        dataset=DAC1,
        scope="contributions",
        alternatives=(),
        note="DAC1 total ODA counts bilateral flows plus core contributions",
    ),
    ProviderWhole(
        dataset=CRS,
        scope="inclusive",
        alternatives=(),
        note="the CRS inclusive scope counts bilateral flows plus the imputed share",
    ),
    ProviderWhole(
        dataset=DAC2A,
        scope="contributions",
        alternatives=("inclusive",),
        named_scope="inclusive",
        recipient="DACAllRecipients",
        note=(
            "DAC2a total ODA counts core contributions to the agencies and the inclusive "
            "scope an imputed share of their spending; over a named country or region total "
            "ODA holds the bilateral flows alone"
        ),
    ),
)


# The composite each table's inclusive scope binds. The CRS stacks on ONE's imputed
# multilateral sectors table, which publishes gross disbursements alone. DAC2a stacks its total
# ODA on its own imputed multilateral ODA, published gross alone and added to total ODA on
# either basis by convention, so the scope is offered on net flows too and its label says so
# (`OptionLabels.provider_inclusive_net`).
INCLUSIVE_SOURCES: tuple[InclusiveSource, ...] = (
    InclusiveSource(
        dataset=CRS,
        composite=CRS_INCLUSIVE,
        bases=("gross",),
        note="CRS bilateral flows plus ONE's imputed multilateral sectors",
    ),
    InclusiveSource(
        dataset=DAC2A,
        composite=DAC2A_INCLUSIVE,
        bases=("gross", "net"),
        note="DAC2a ODA plus DAC2a imputed multilateral ODA, gross as published",
    ),
)

# The organisations a row of `PUBLISHED_PROVIDER_SCOPES` reads as donors, beside countries
# and totals of a group made only of countries. The EU institutions are a DAC member and report
# their own aid to DAC1 as a donor does. Any other organisation or total a part names, the
# DAC's total of the multilaterals among them, counts multilateral outflows, so the part has
# no provider record.
DONOR_ORGANISATIONS: tuple[str, ...] = (_EU_INSTITUTIONS,)

# The graph types that place a DAC2a destination residual no bilateral total lists
# (`destination_side`). The DAC lists the recipients that stand for the multilaterals as a group,
# each a donor's core contributions to the agencies it does not itemise ("other United Nations
# entities"), as unspecified parties, and its lines of bilateral aid it allocates to no country
# ("Africa, unallocated by country") as unallocated by country.
DAC_MULTILATERAL_RESIDUAL_TYPE = "DACUnspecifiedParty"
DAC_BILATERAL_RESIDUAL_TYPE = "DACRecipientUnallocatedByCountry"

# The graph type of a DAC2a recipient country (`destination_side`).
COUNTRY_TYPE = "Country"

# The organisations the DAC counts as bilateral recipients: DACDevelopingCountries holds them
# in 206, though the graph types them `Organization` and `party_facts` reads them as
# multilaterals on the origin axis. Read on the destination axis only, ahead of `party_facts`.
DAC_BILATERAL_ORGANISATIONS: tuple[str, ...] = (
    # A regional union of partner states the DAC lists among developing-country recipients.
    "EastAfricanCommunity",
    # A river-basin fund between India and Pakistan that the DAC lists as a country recipient.
    "IndusBasinDevelopmentFund",
)

# The DAC2a recipient total of every multilateral recipient: its named organisations and the
# unspecified ones (`DAC_MULTILATERAL_RESIDUAL_TYPE`).
DAC_ALL_MULTILATERAL_RECIPIENTS = "DACAllMultilateralRecipients"

# The DAC2a total of all recipients, "Developing Countries, Total" plus the multilateral
# agencies, so it spans both sides.
DAC_ALL_RECIPIENTS = "DACAllRecipients"

# The DAC2a total of every developing-country recipient: the countries, regions and the DAC's
# unallocated lines, the bilateral side of `DAC_ALL_RECIPIENTS`.
DAC_DEVELOPING_COUNTRIES = "DACDevelopingCountries"

# The composite each published total counting core contributions over every destination binds
# for its bilateral scope, and the population every composite component binding the dataset's
# destinations reads (`RolePopulation`), the bilateral and the inclusive composite's alike.
# DAC2a's total ODA lists the multilateral agencies among its recipients, so an open
# destination binds the bilateral-side parties and residuals alone, and a named agency, the
# DAC's total of them or its total of all recipients, which holds both sides, is refused
# (`code_index.composites.component_identities`). An identity naming no destination binds the
# DAC's total of developing countries, the bilateral side of all recipients.
BILATERAL_SOURCES: tuple[BilateralSource, ...] = (
    BilateralSource(
        dataset=DAC2A,
        code_system="dac2aAidType",
        code=DAC2A_TOTAL_ODA,
        composite=DAC2A_BILATERAL,
        spanning=DAC_ALL_RECIPIENTS,
        rebound=DAC_DEVELOPING_COUNTRIES,
        spanning_words="all recipients",
        rebound_words="developing countries",
        rule="bilateral_destinations",
        note="DAC2a total ODA to countries, regions and the DAC's unallocated lines alone",
        label=(
            "bilateral flows to countries and regions, without core contributions to "
            "multilateral agencies"
        ),
        total_label="total ODA, including core contributions to multilateral agencies",
    ),
)

DestinationSide = Literal["bilateral", "multilateral", "spanning"]
"""Which side of a donor's aid a DAC2a destination is on: a country or a group of countries
(`bilateral`), a multilateral agency (`multilateral`), or `DACAllRecipients`, the total of both
(`spanning`)."""


def destination_side(
    row: PartyRow, *, datasets: Collection[str], index: CodeIndex
) -> DestinationSide | None:
    """The side of `row` as a recipient in `datasets`, or `None` when it is unclassified: no
    dataset publishes it as a recipient, they give it different kinds, or the classifier
    cannot place it. `index` is the code index `row` comes from, whose totals list the
    residuals they count.

    The rule reads the graph types the code index keeps on each row (`PartyRow.types`), so a
    recipient the OECD adds takes its side from its type. A total is `spanning` when
    `DAC_ALL_RECIPIENTS`, `multilateral` when `DAC_ALL_MULTILATERAL_RECIPIENTS` or an
    organisation (the World Bank Group beside its members), and `bilateral` otherwise. A
    residual is `bilateral` when a bilateral total of `datasets` counts it beside its members
    (`PartyRow.residuals`), as Asia counts the Mekong Delta Project, and otherwise
    `multilateral` when typed `DAC_MULTILATERAL_RESIDUAL_TYPE` and `bilateral` when typed
    `DAC_BILATERAL_RESIDUAL_TYPE`. A party is `bilateral` when in `DAC_BILATERAL_ORGANISATIONS`,
    `multilateral` where `party_facts` says so, `bilateral` when a country, and unclassified
    otherwise.
    """
    kind = row.kind_over("destination", datasets)
    if kind is None:
        return None
    if kind == "total":
        return _total_side(row)
    if kind == "residual":
        if any(
            total.kind_over("destination", datasets) == "total"
            for total in _bilateral_totals_listing(index).get(row.dcid, ())
        ):
            return "bilateral"
        if DAC_MULTILATERAL_RESIDUAL_TYPE in row.types:
            return "multilateral"
        return "bilateral" if DAC_BILATERAL_RESIDUAL_TYPE in row.types else None
    if row.dcid in DAC_BILATERAL_ORGANISATIONS:
        return "bilateral"
    if party_facts(row, axis="destination", datasets=datasets).multilateral:
        return "multilateral"
    return "bilateral" if COUNTRY_TYPE in row.types else None


@per_index
def _bilateral_totals_listing(index: CodeIndex) -> dict[str, tuple[PartyRow, ...]]:
    """Each residual a total `_total_side` places `bilateral` counts, mapped to the totals
    that count it, so `destination_side` reads a residual's totals without a pass over every
    party."""
    listing: dict[str, list[PartyRow]] = {}
    for total in index.parties.values():
        if total.residuals and _total_side(total) == "bilateral":
            for residual in total.residuals:
                listing.setdefault(residual, []).append(total)
    return {residual: tuple(totals) for residual, totals in listing.items()}


def _total_side(total: PartyRow) -> DestinationSide:
    """The side of `total`, a destination total (`destination_side`)."""
    if total.dcid == DAC_ALL_RECIPIENTS:
        return "spanning"
    if total.dcid == DAC_ALL_MULTILATERAL_RECIPIENTS or ORGANIZATION_TYPE in total.types:
        return "multilateral"
    return "bilateral"


def bilateral_side_total(dcid: str) -> bool:
    """Whether `dcid` is a published total of the bilateral destinations
    (`BilateralSource.rebound`, "Developing Countries, Total"): a population a request can
    name on the destination side. It holds the bilateral side alone, so it carries no whole
    marker (`code_index.types.WholeMarker`). A dataset that publishes no row of it reads it as
    every recipient it holds where it holds the DAC's recipient list
    (`reads_dac_recipient_list`), and lists it where it also reads the bilateral side of a
    total (`lists_bilateral_destinations`)."""
    return any(row.rebound == dcid for row in BILATERAL_SOURCES)


def reads_dac_recipient_list(datasets: Collection[str]) -> bool:
    """Whether every one of `datasets`, at least one, holds the DAC List of ODA Recipients
    alone as its recipients (`DAC_RECIPIENT_LIST_DATASETS`), as the CRS does."""
    return bool(datasets) and all(dataset in DAC_RECIPIENT_LIST_DATASETS for dataset in datasets)


def lists_bilateral_destinations(datasets: Collection[str], *, total_kept: bool) -> bool:
    """Whether every one of `datasets`, at least one, lists the bilateral destinations alone
    on an open recipient side: it holds the DAC's recipient list (`reads_dac_recipient_list`),
    or its bilateral composite reads the population of that side (`BILATERAL_SOURCES`), as
    DAC2a's does, where the part does not keep the dataset's total itself (`total_kept`, a
    `provider:contributions` entry), whose open side lists the multilateral agencies too."""
    bilateral = set() if total_kept else {row.dataset for row in BILATERAL_SOURCES}
    return bool(datasets) and all(
        dataset in DAC_RECIPIENT_LIST_DATASETS or dataset in bilateral for dataset in datasets
    )


@dataclass(frozen=True)
class RolePopulation:
    """The destinations a composite component reads where it binds its destinations to one
    side (`ComponentRecipe.destinations`), derived from one code index
    (`code_index.composites.destination_population`) and declared by `source`.

    `members` is what an open destination reads: the sorted dcids of the `side` parties and
    residuals the dataset publishes as destinations, its totals left out as an open read
    leaves them out. `named` is what a named destination may be: every destination the
    dataset publishes on `side`, its totals included. `total` is the dataset's published
    total of `side` (`BilateralSource.rebound`), which an identity naming no destination
    reads."""

    side: ComponentDestinations
    members: tuple[str, ...]
    named: frozenset[str]
    source: BilateralSource

    @property
    def total(self) -> str:
        """The dataset's published total of `side`."""
        return self.source.rebound


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
