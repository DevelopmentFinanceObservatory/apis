"""A same-shape merge of DAC2a parts that settle on the bilateral composite mints the
composite over the merged identity (`spec_build.build_spec(composite=)`), so it reads no
population a second time and answers as a resubmit of its merged definition does.

The fixture is France's DAC2a 206 gross and net disbursements to Ethiopia, the European
Commission and `DACAllRecipients`, and its imputed 106 to Ethiopia, every destination open:
the bilateral composite reads the country alone."""

from __future__ import annotations

from typing import Any

import pytest

from search.models import (
    BaseIdentity,
    DatasetResubmitInput,
    DefinitePart,
    Resolution,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityEntry,
)
from tests.engine._harness import offline_resolve
from tests.engine.test_aid_dataset_bind_e2e import _ScriptedLLM
from tests.engine.test_dac2a_destination_populations import (
    _AGENCY,
    _BILATERAL,
    _DAC2A_ATTRIBUTED,
    _DAC2A_MEMBER,
    _ETH,
    _FRA,
    _HELD,
    _VARIABLE,
    _answer,
    _counted,
    _identity,
    _patched_coverage,
    _ref,
    _sdmx,
    _table_reads,
)
from tests.engine.test_whole_total_bind_e2e import _ALL_RECIPIENTS, _graph
from tests.fixtures import FakeGraph

_DAC2A_NET = _DAC2A_MEMBER.replace("_Gross_", "_Net_")
_NET_HELD = {
    **_HELD,
    (_DAC2A_NET, _FRA, _ETH): 90,
    (_DAC2A_NET, _FRA, _AGENCY): 190,
    (_DAC2A_NET, _FRA, _ALL_RECIPIENTS): 450,
}


def _netting(identity: BaseIdentity, basis: str) -> BaseIdentity:
    """`identity` with its netting basis bound to `basis`."""
    slots = [
        slot.model_dump(mode="json")
        if slot.property != "nettingBasis"
        else {"property": "nettingBasis", "binding": _ref(basis)}
        for slot in identity.slots
    ]
    return BaseIdentity.model_validate(
        {**identity.model_dump(mode="json", exclude_none=True), "slots": slots}
    )


def _graph_holding_net() -> FakeGraph:
    """The whole-total fixture graph finding DAC2a's gross and net members and the imputed
    one, holding the agency and the total spanning both sides, which a part's reads post."""
    graph = _graph(variable=_VARIABLE, members=(_DAC2A_MEMBER, _DAC2A_ATTRIBUTED, _DAC2A_NET))
    for dcid in (_AGENCY, _ALL_RECIPIENTS):
        graph._nodes.setdefault(dcid, {"label": dcid, "type": "Organization"})
    return graph


def _resubmit(*identities: BaseIdentity) -> Any:
    return offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(variable_text=_VARIABLE, identity=identity)
                    for identity in identities
                ]
            ),
            response_options=ResolveOptions(include_observations=False),
        ),
        graph=_graph_holding_net(),
        llm=_ScriptedLLM(_VARIABLE),
        sdmx=_sdmx(_NET_HELD),
    )


def test_a_merged_bilateral_composite_answers_as_a_resubmit_of_its_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gross and a net part the merge folds answer the bilateral composite over both
    bases, minted over the merged identity from its members' facets, with no read of the
    population beyond each member's own: its scope, entities, headline, coverage and
    warnings are a resubmit's of its merged definition."""
    _patched_coverage(monkeypatch, _NET_HELD)
    bilateral = _identity(_BILATERAL)
    reads = _table_reads(monkeypatch)
    gross, net = _parts(_resubmit(_netting(bilateral, "Gross"), _netting(bilateral, "Net")))

    assert _counted(reads) == [("resubmit", True), ("resubmit", True)]

    assert gross.spec == net.spec
    (again,) = _parts(_resubmit(gross.spec.definition.identity))
    assert again.spec.spec_id == gross.spec.spec_id
    assert _answer(gross.spec, gross) == _answer(again.spec, again)
    scope = gross.spec.resolved.scope
    assert scope is not None and [member.ref for member in scope.members] == [_ETH]


def _parts(response: Any) -> list[DefinitePart]:
    assert isinstance(response, Resolution), response
    parts = [part for part in response.parts if isinstance(part, DefinitePart)]
    assert len(parts) == len(response.parts)
    return parts
