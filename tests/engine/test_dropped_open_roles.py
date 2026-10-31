"""The `query_qualifier_dropped` warning an open role the answer binds raises
(`region.result.with_dropped_open_roles`)."""

from __future__ import annotations

import pytest

from search.engine.region import RegionResult
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.region.result import with_dropped_open_roles
from search.models import EntityGroup
from tests.engine._assemble_fixtures import _make_spec

_DAC_MEMBERS = ["country/FRA", "country/USA"]


def _donor_region(donors: list[str], group: EntityGroup | None) -> RegionResult:
    """A definite region whose one Spec binds `donors` on the origin, read as `group`."""
    spec = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH")
    identity = spec.definition.identity.model_copy(update={"entities": {"origin": donors}})
    groups = None if group is None else {"origin": group}
    spec = spec.model_copy(
        update={
            "definition": spec.definition.model_copy(update={"identity": identity}),
            "resolved": spec.resolved.model_copy(update={"entity_groups": groups}),
        }
    )
    return RegionResult(
        variable_text="ODA",
        status="definite",
        specs=(spec,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        answered_roles=("donor",),
    )


@pytest.mark.parametrize(
    ("donors", "group", "warns"),
    [
        pytest.param(
            _DAC_MEMBERS,
            EntityGroup(ref="groups/OECD.DAC", label="DAC members"),
            False,
            id="a-group-read-as-its-members",
        ),
        pytest.param(["country/USA"], None, True, id="a-single-party"),
        pytest.param(
            ["DACCountriesTotal"],
            EntityGroup(ref="groups/OECD.DAC", label="DAC", total="DACCountriesTotal"),
            True,
            id="a-group-read-as-its-total",
        ),
    ],
)
def test_an_open_donor_the_answer_binds_warns_unless_a_group_reads_each_member(
    donors, group, warns
):
    """The DAC's members beside an open donor role answer one series per member, so nothing
    the open role asked for is dropped. A single donor, or a group read as its published
    total, answers one series, so the breakdown over every donor is dropped."""
    region = with_dropped_open_roles(_donor_region(donors, group), {"origin"})

    assert [w.code for w in region.warnings] == ([QUERY_QUALIFIER_DROPPED] if warns else [])
