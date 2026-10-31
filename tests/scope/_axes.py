"""A small synthetic set of axis tables over one aid identity, for the scope rules' tests.

Sector offers CRS 120 + 130 or 120 alone, region the members with or without Africa's
residual bucket, and measure basis gross, net or grant equivalents. Two declared conflicts
tie the axes together: grant equivalents exclude the residual bucket, and net excludes
CRS 120 alone. The base identity holds the first option of every axis.
"""

from __future__ import annotations

from search.models import (
    BaseIdentity,
    ChoiceOrigin,
    EntitiesBinding,
    Entry,
    Option,
    OptionAvailability,
    ShapeBinding,
    SlotBinding,
    TextOrigin,
)
from search.scope.tables import AxisTable, CompileFacts, Conflict
from search.scope.vocabulary import option_axis

SECTOR_SET = "sector:120+130"
SECTOR_ONE = "sector:120"
REGION_ALL = "region:members_and_residuals"
REGION_MEMBERS = "region:members_only"
GROSS = "measure_basis:gross"
NET = "measure_basis:net"
GRANT_EQUIVALENT = "measure_basis:grant_equivalent"

RESIDUAL = "DACAfricaUnallocated"
MEMBERS = ("country/GHA", "country/KEN")

BASE = BaseIdentity.model_validate(
    {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "ODA",
            "measured_property": "amount",
        },
        "slots": [
            {
                "property": "crsPurpose",
                "binding": {"state": "set", "refs": ["CRSPurposeCode120", "CRSPurposeCode130"]},
            },
            {"property": "nettingBasis", "binding": {"state": "value", "ref": "Gross"}},
        ],
        "entities": {"origin": ["country/USA"], "destination": [*MEMBERS, RESIDUAL]},
        "requested_window": {"kind": "all"},
    }
)

_AVAILABLE = OptionAvailability(status="available")


def _option(option_id: str, binding, availability: OptionAvailability = _AVAILABLE) -> Option:
    return Option(option_id=option_id, label=option_id, binding=binding, availability=availability)


def tables(*, grant_equivalent_published: bool = True) -> tuple[AxisTable, ...]:
    """The three axis tables, grant equivalents unpublished when asked."""
    sector = AxisTable(
        axis="sector",
        options=(
            _option(
                SECTOR_SET,
                SlotBinding(
                    slot="crsPurpose",
                    state="set",
                    refs=["CRSPurposeCode120", "CRSPurposeCode130"],
                ),
            ),
            _option(
                SECTOR_ONE,
                SlotBinding(slot="crsPurpose", state="value", refs=["CRSPurposeCode120"]),
            ),
        ),
        preference=(SECTOR_SET, SECTOR_ONE),
        held=SECTOR_SET,
        rules={SECTOR_SET: "test_sector_grouping"},
        concept="health",
    )
    region = AxisTable(
        axis="region",
        options=(
            _option(REGION_ALL, EntitiesBinding(role="destination", add=[RESIDUAL])),
            _option(REGION_MEMBERS, EntitiesBinding(role="destination", add=[])),
        ),
        preference=(REGION_ALL, REGION_MEMBERS),
        held=REGION_ALL,
        rules={REGION_ALL: "test_summed_region"},
    )
    grant_equivalent = (
        _AVAILABLE
        if grant_equivalent_published
        else OptionAvailability(
            status="unavailable", code="basis_not_published", detail={"dataset": "test"}
        )
    )
    basis = AxisTable(
        axis="measure_basis",
        options=(
            _option(GROSS, ShapeBinding(measured_property="amount", netting_basis="Gross")),
            _option(NET, ShapeBinding(measured_property="amount", netting_basis="Net")),
            _option(
                GRANT_EQUIVALENT,
                ShapeBinding(measured_property="grantEquivalent", netting_basis=None),
                grant_equivalent,
            ),
        ),
        preference=(GROSS, NET, GRANT_EQUIVALENT),
        held=GROSS,
        rules={GROSS: "test_basis_convention"},
    )
    return sector, region, basis


CONFLICTS = (
    Conflict(GRANT_EQUIVALENT, REGION_ALL, "grant_equivalents_lack_residuals"),
    Conflict(NET, SECTOR_ONE, "net_lacks_purpose_120"),
)


def facts(*, grant_equivalent_published: bool = True) -> CompileFacts:
    return CompileFacts(
        axes=tables(grant_equivalent_published=grant_equivalent_published),
        conflicts=CONFLICTS,
    )


def explicit(option_id: str) -> Entry:
    """An explicit entry read from the request's words."""
    return Entry(
        axis=option_axis(option_id),
        option_id=option_id,
        provenance="explicit",
        origin=TextOrigin(text=option_id, start=0, end=len(option_id)),
    )


def switch(option_id: str) -> Entry:
    """An explicit entry a switch made on the axis's choice."""
    return Entry(
        axis=option_axis(option_id),
        option_id=option_id,
        provenance="explicit",
        origin=ChoiceOrigin(choice_id=f"scope:{option_axis(option_id)}"),
    )


def carried(option_id: str) -> Entry:
    """A carried entry from an earlier part's switch."""
    return Entry(
        axis=option_axis(option_id),
        option_id=option_id,
        provenance="carried",
        origin=ChoiceOrigin(choice_id=f"earlier:{option_axis(option_id)}"),
    )
