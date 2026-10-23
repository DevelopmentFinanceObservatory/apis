"""`graph_api.kinds.classify` gives every node one kind from the closed list."""

from __future__ import annotations

from typing import get_args

import pytest

from code_index import CodeIndex, IndexStamp
from code_index.shared import get_index
from code_index.types import PartyRow, WholeMarker
from entity_resolution.membership import (
    COUNTRY_CLASS,
    Authority,
    Membership,
    MembershipPolicy,
    RegionClass,
    Row,
    Sovereignty,
)
from entity_resolution.types import EntityKind
from graph_api.kinds import KINDS, classify

_SOVEREIGNTY = Sovereignty(states={}, dissolved=frozenset(), other_states=frozenset())
_AUTHORITY = Authority(name="test", version="1")


def _membership(ref: str, region_class: RegionClass | None = None) -> Membership:
    """A group, or a region of `region_class`."""
    return Membership(
        ref=ref,
        entity_id=ref,
        label=ref,
        kind="group" if region_class is None else "region",
        authority=_AUTHORITY,
        dated=False,
        rows=(Row(member="country/KEN", member_class=COUNTRY_CLASS),),
        sovereignty=_SOVEREIGNTY,
        region_class=region_class,
    )


_POLICY = MembershipPolicy(
    [
        _membership("groups/G20"),
        _membership("EuropeanUnion"),
        _membership("africa", "geo.continent"),
        _membership("WesternAfrica", "geo.subregion"),
        _membership("undata-geo/G00134000", "geo.continent"),
    ]
)


def test_the_list_holds_sixteen_kinds_and_every_resolver_kind():
    assert len(KINDS) == len(set(KINDS)) == 16
    assert set(get_args(EntityKind)) <= set(KINDS)


@pytest.mark.parametrize(
    ("dcid", "types", "kind"),
    [
        # A curated group opens with no graph node, and outranks the place type its node has.
        ("groups/G20", (), "group"),
        ("EuropeanUnion", ("ContinentalUnion",), "group"),
        ("Count_Person", ("StatisticalVariable",), "stat_var"),
        ("provenance/OECDCreditorReportingSystem", ("Provenance",), "provenance"),
        ("dc/s/WorldBank", ("Source",), "source"),
        ("dc/d/WorldBank_WDI", ("Dataset",), "dataset"),
        # The graph types a code system a `Class`; the code index names it.
        ("CRSPurposeCodeEnum", ("Class",), "code_system"),
        ("CRSPurposeCode122", ("CRSPurposeCodeEnum",), "code"),
        ("africa", ("Continent",), "continent"),
        ("WesternAfrica", ("UNGeoRegion",), "subregion"),
        # A held region takes its authority's class, whatever the graph types its node: the
        # Americas is a `GeoRegion` node and a resolvekit continent.
        ("undata-geo/G00134000", ("GeoRegion",), "continent"),
        ("CurrencyUSD", ("CurrencyUnit",), "unit"),
        ("ChainedUSDollarBaseYr2017", ("UnitOfMeasure",), "unit"),
        # A total and a residual party of a declared dataset hold other parties' figures.
        ("DACAfrica", ("DACGroup",), "aggregate"),
        ("DACAfricaUnallocated", ("DACRecipientUnallocatedByCountry",), "aggregate"),
        ("country/IND", ("Country", "Place"), "country"),
        ("dc/o/Foundation", ("Organization",), "organisation"),
        ("geoId/0644000", ("City",), "place_other"),
        ("nuts/Town", ("Town",), "place_other"),
        ("nuts/DE11", ("EurostatNUTS2",), "place_other"),
        ("Earth", ("Place",), "place_other"),
        ("dc/base/Foo", ("SomeOtherType",), "other"),
        ("dc/base/Untyped", (), "other"),
    ],
)
def test_a_node_takes_the_first_kind_the_rule_gives(dcid, types, kind):
    assert classify(dcid=dcid, types=types, code_index=get_index(), membership=_POLICY) == kind


@pytest.mark.parametrize(
    ("row", "kind"),
    [
        (PartyRow(dcid="dc/o/Foundation", kind="party", party_class="private"), "organisation"),
        (PartyRow(dcid="dc/o/Foundation", kind="total"), "aggregate"),
        (
            PartyRow(
                dcid="dc/o/Foundation",
                kind="total",
                datasets=("provenance/P",),
                axes=("origin",),
                party_class="official",
                whole=(WholeMarker(dataset="provenance/P", covers=("official",)),),
            ),
            "aggregate",
        ),
        (PartyRow(dcid="dc/o/Foundation", kind="residual"), "aggregate"),
    ],
    ids=["private", "total", "whole-total", "residual"],
)
def test_a_private_party_is_an_organisation_and_every_total_an_aggregate(row, kind):
    """The index's party kind outranks the graph type, which types all of these `Organization`."""
    index = CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="https://example.test/core/api",
            exported_at="2026-09-09T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="deadbeef",
        ),
        declarations=(),
        codes=(),
        statvars=(),
        parties=(row,),
    )

    assert (
        classify(dcid=row.dcid, types=("Organization",), code_index=index, membership=None) == kind
    )


def test_with_no_policy_no_node_is_a_group_or_region():
    index = get_index()

    assert classify(dcid="groups/G20", types=(), code_index=index, membership=None) == "other"
    assert (
        classify(dcid="africa", types=("Continent",), code_index=index, membership=None)
        == "continent"
    )
