"""The spec door mints a definition in canonical form, whether the posted identity binds the
invariants of the dataset serving its family or leaves them unbound.

The imputed multilateral table declares `crsTypeOfFlow` and `dac2aAidType` as invariants.
A definition posted with both bound and one posted with both unbound mint and
report one `spec_id`, the canonical identity's (`code_index.members.canonical_identity`),
which binds both, when the definition names the table. One that names no table keeps the bindings it was
posted with.
"""

from __future__ import annotations

import httpx
import pytest

from code_index.shared import get_index
from datasets_api.adapter import resolve_definition_members
from spec_registry import recompute_identity
from statvar_members import MemberEnumerator
from tests.code_index.aid_identity import (
    AID_TYPE,
    IMPUTED_MEMBER_COUNT,
    IMPUTED_PROVENANCE,
    imputed_education_aid_identity,
)

from ._graph_doubles import FakeMintGraphReader
from ._member_bounds import mint_bounds
from .conftest import FakeDatasetStore
from .test_definition_door import _client, _coverage_service

_PROVIDER = "country/FRA"
_RECIPIENT = "country/KEN"
_YEARS = (2020, 2021)
_PINS = ("crsTypeOfFlow", AID_TYPE)

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,financingProvider,financingRecipient,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,"
    "facetId\r\n"
)


def _identity(*, pins_bound: bool, named: bool = True) -> dict:
    identity = imputed_education_aid_identity(source=IMPUTED_PROVENANCE if named else None)
    if not pins_bound:
        for slot in identity["slots"]:
            if slot["property"] in _PINS:
                slot["binding"] = {"state": "unbound"}
    return identity


def _populated() -> tuple[str, ...]:
    graph = FakeMintGraphReader()
    resolved = resolve_definition_members(
        _identity(pins_bound=True),
        spec_id="spec_test",
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=get_index(),
        bounds=mint_bounds(),
    )
    assert len(resolved.shapes) == IMPUTED_MEMBER_COUNT
    return tuple(shape.stat_var.removeprefix("dcid:") for shape in resolved.shapes[:2])


def _sdmx_transport(populated: tuple[str, ...]) -> httpx.MockTransport:
    def _respond(request: httpx.Request) -> httpx.Response:
        named = str(request.url.params.get("c[variableMeasured]", "")).split(",")
        body = _CSV_HEADER + "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{dcid},{_PROVIDER},{_RECIPIENT},USD,GrantEquivalent,"
            f"NotApplicable,{IMPUTED_PROVENANCE},{year},{100 + position},,facet1\r\n"
            for position, dcid in enumerate(populated)
            if dcid in named
            for year in _YEARS
        )
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(_respond)


def _input(identity: dict) -> dict:
    spec_id = recompute_identity(identity).spec_id
    return {
        "kind": "spec_id",
        "spec_id": spec_id,
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity},
    }


@pytest.mark.parametrize("named", [True, False], ids=["named", "unnamed"])
@pytest.mark.parametrize("pins_bound", [True, False], ids=["pins-bound", "pins-unbound"])
def test_a_definition_mints_and_reports_its_canonical_spec_id(pins_bound: bool, named: bool):
    canonical = _identity(pins_bound=pins_bound or named, named=named)
    canonical_spec_id = recompute_identity(canonical).spec_id
    posted = _input(_identity(pins_bound=pins_bound, named=named))
    assert (posted["spec_id"] == canonical_spec_id) is (pins_bound or not named)
    populated = _populated()
    store = FakeDatasetStore()
    client = _client(
        graph=FakeMintGraphReader(labels={_PROVIDER: "France", _RECIPIENT: "Kenya"}),
        store=store,
        sdmx_transport=_sdmx_transport(populated),
        coverage=_coverage_service(
            stat_vars=populated,
            donor_role="financingProvider",
            donor=_PROVIDER,
            recipient=_RECIPIENT,
            provenance=IMPUTED_PROVENANCE,
            years=_YEARS,
        ),
    )

    minted = client.post(
        "/api/datasets", json={"schema_version": 1, "input": posted, "options": {"bundle": False}}
    )

    assert minted.status_code == 200, minted.text
    manifest = minted.json()
    assert manifest["definition"]["spec_id"] == canonical_spec_id
    assert manifest["definition"]["identity"] == recompute_identity(canonical).identity
    assert manifest["coverage"]["rows"] == len(populated) * len(_YEARS)
    assert canonical_spec_id in store.state.spec_registry
