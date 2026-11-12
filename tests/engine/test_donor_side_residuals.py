"""A donor-side residual, such as IDS's unidentified creditors, that a donor role binds beside
a group's members is no member of the group, as a recipient-side residual is on a recipient
role (`scope.region.is_residual` on the role's axis). Each reader of the residuals a role
binds reads them on that role's axis: the departures, the spec's named groups and a
resubmit's posted groups."""

from __future__ import annotations

from typing import Any

import pytest

from code_index.types import PartyRow
from search.engine import departures, spec_build
from search.engine.region import resubmit_request
from search.engine.region.resubmit_request import resubmit_named_groups
from search.engine.spec_build import NamedGroup
from search.engine.spec_id import build_spec_identity
from search.models import EntityGroup, ResolvedEntity, ResubmitIdentityPayload
from tests.engine._base_identity import DECLARING_SHAPE, origin_entity
from tests.engine.test_departures import _spec
from tests.fixtures import FakeGraph

_RESIDUAL = PartyRow(
    dcid="IDSUnidentifiedCreditors",
    kind="residual",
    party_class="residual",
    label="Creditors, unidentified",
    published=(("provenance/WorldBankIDS", "origin"),),
)
_MEMBERS = ("country/FRA", "country/USA")
_BOUND = frozenset({*_MEMBERS, _RESIDUAL.dcid})


class _WithResidual:
    """The served code index with `_RESIDUAL` among its party rows."""

    def __init__(self, index: Any) -> None:
        self._index = index
        self.parties = {**index.parties, _RESIDUAL.dcid: _RESIDUAL}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._index, name)


@pytest.fixture(autouse=True)
def _residual_indexed(monkeypatch: pytest.MonkeyPatch) -> None:
    index = _WithResidual(departures.get_index())
    for module in (departures, spec_build, resubmit_request):
        monkeypatch.setattr(module, "get_index", lambda: index)


def test_the_departures_count_a_donor_role_s_members_and_name_its_residual_apart() -> None:
    spec = _spec()
    entities = [
        ResolvedEntity(ref=dcid, label=dcid, role="origin") for dcid in (*_MEMBERS, _RESIDUAL.dcid)
    ]
    spec = spec.model_copy(
        update={"resolved": spec.resolved.model_copy(update={"entities": entities})}
    )

    assert departures._role_members(spec, "origin") == list(_MEMBERS)
    assert departures._residuals_bound(spec, "origin") == ("Creditors, unidentified",)


def test_a_donor_role_binding_a_group_s_members_and_a_residual_is_that_group() -> None:
    group = NamedGroup(label="Paris Club", dcid=None, member_dcids=frozenset(_MEMBERS))

    assert spec_build._role_group([group], "origin", set(_BOUND)) is group


def test_a_resubmit_keeps_a_donor_group_whose_role_binds_a_residual_beside_its_members(
    declare_regions,
) -> None:
    declare_regions({"donorRegion": _MEMBERS})
    _spec_id, identity = build_spec_identity(
        DECLARING_SHAPE,
        [],
        entities=[origin_entity(dcid) for dcid in sorted(_BOUND)],
        open_roles=(),
    )
    payload = ResubmitIdentityPayload(
        identity=identity,
        entity_groups={"origin": EntityGroup(ref="donorRegion", label="Donors")},
    )

    named = resubmit_named_groups(payload, graph=FakeGraph(), deadline=None)

    assert [(group.dcid, group.member_dcids) for group in named] == [
        ("donorRegion", frozenset(_MEMBERS))
    ]
