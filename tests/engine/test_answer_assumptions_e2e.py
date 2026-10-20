"""End-to-end tests for a part's `assumptions` on the raw-text and dataset_resubmit doors.

"aid flows to Kenya" leaves the flow slot open, so the election fixes it to its declared
default, ODA, and the fixture's heading reads both flow roles open ("to all recipients
from all providers"). A resubmit posts the identity whole, so the engine records the flow
as posted, and the part keeps the default only through the entry's `assumptions` echo,
matched to the rebuilt slot by its value dcids.
"""

from __future__ import annotations

import pytest

from search.engine.extract import ExtractedEntity, Extraction
from search.models import (
    Assumption,
    BaseIdentity,
    DatasetResubmitInput,
    DefinitePart,
    IdentityBindingValue,
    IdentityWindowRange,
    RawTextInput,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityEntry,
)
from tests.engine._harness import offline_resolve
from tests.engine.test_operations_e2e import _EMPTY_SDMX, _OpenFlowAndSectorBindLLM
from tests.engine.test_qualifier_suppression_and_provenance import (
    _E2E_NODES,
    _E2E_OBS,
    _e2e_graph_for,
    _e2e_sv_arcs,
)
from tests.fixtures import FakeGraph

_ASSUMED_ODA = Assumption(
    kind="declared_default", binding_ref="how:crsTypeOfFlow", text="ODA", refs=["CRSTypeOfFlow10"]
)
_ASSUMED_ROLES = [
    Assumption(kind="open_role", binding_ref="where:", text="every recipient"),
    Assumption(kind="open_role", binding_ref="origin:", text="every provider"),
]
_ASSUMED_CLAUSE = " Assumed: ODA; every recipient; every provider."


def _first_answer() -> DefinitePart:
    response = offline_resolve(
        ResolveRequest(
            input=RawTextInput(query="aid flows to Kenya"), response_options=ResolveOptions()
        ),
        graph=_e2e_graph_for("aid flows"),
        llm=_OpenFlowAndSectorBindLLM(
            Extraction(
                kind="resolution",
                variables=["aid flows"],
                entities=[ExtractedEntity(text="Kenya")],
            )
        ),
        sdmx=_EMPTY_SDMX,
    )
    assert response.rendered_sentence is not None
    assert response.rendered_sentence.endswith(_ASSUMED_CLAUSE)
    part = response.parts[0]
    assert isinstance(part, DefinitePart)
    return part


# The grant equivalent of private sector instruments in Kenya: the series a flow edit from
# ODA to private sector instruments reads, held beside the fixture's ODA series.
_PSI_SV_DCID = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow60_CRSTypeOfFinance1101_"
    "CRSCooperationModalityC01_CRSPurposeCode24030_Nominal"
)


def _graph_with_psi() -> FakeGraph:
    """The fixture graph with the private sector instruments series beside the ODA one."""
    return FakeGraph(
        nodes={
            **_E2E_NODES,
            _PSI_SV_DCID: {
                "label": "test private sector instruments flow",
                "type": "StatisticalVariable",
                "arcs": _e2e_sv_arcs(_PSI_SV_DCID),
            },
            "CRSTypeOfFlow60": {"label": "private sector instruments"},
        },
        detect={},
        resolve={"Kenya": "country/KEN"},
        obs={
            **_E2E_OBS,
            f"{_PSI_SV_DCID}|country/KEN": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
            ],
        },
    )


def _bound(identity: BaseIdentity, prop: str, ref: str) -> BaseIdentity:
    """`identity` with slot `prop` bound to `ref`."""
    slots = [
        slot.model_copy(update={"binding": IdentityBindingValue(ref=ref)})
        if slot.property == prop
        else slot
        for slot in identity.slots
    ]
    return identity.model_copy(update={"slots": slots})


def _resubmit(
    identity: BaseIdentity, *, echo: list[Assumption], graph: FakeGraph | None = None
) -> DefinitePart:
    response = offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(
                        variable_text="aid flows", identity=identity, assumptions=echo
                    )
                ]
            ),
            response_options=ResolveOptions(),
        ),
        graph=graph or _e2e_graph_for("aid flows"),
        llm=_OpenFlowAndSectorBindLLM(Extraction(kind="resolution", variables=[], entities=[])),
        sdmx=_EMPTY_SDMX,
    )
    part = response.parts[0]
    assert isinstance(part, DefinitePart)
    return part


def test_a_first_answer_lists_its_declared_default_and_every_open_role():
    """The flow default and the two roles no request word named are the part's assumptions,
    in clause order, and the envelope sentence closes with them."""
    assert _first_answer().assumptions == [_ASSUMED_ODA, *_ASSUMED_ROLES]


@pytest.mark.parametrize(
    ("edit", "echo", "expected"),
    [
        pytest.param(None, None, [_ASSUMED_ODA, *_ASSUMED_ROLES], id="unchanged"),
        pytest.param("year", None, [_ASSUMED_ODA, *_ASSUMED_ROLES], id="year-edited"),
        # The echo's words differ from the rebuilt ones, as after a label change between
        # deploys; the refs still match, so the default holds in the rebuilt words.
        pytest.param(
            None,
            [_ASSUMED_ODA.model_copy(update={"text": "official development assistance"})],
            [_ASSUMED_ODA],
            id="echo-words-differ",
        ),
        # The echo names a value the slot does not bind.
        pytest.param(
            None,
            [_ASSUMED_ODA.model_copy(update={"refs": ["CRSTypeOfFlow60"]}), *_ASSUMED_ROLES],
            _ASSUMED_ROLES,
            id="echo-refs-differ",
        ),
        # An item the echo lacks is not stated.
        pytest.param(None, [_ASSUMED_ODA], [_ASSUMED_ODA], id="echo-without-roles"),
        # The limit of an unsigned echo: an item for a slot the engine never defaulted
        # holds while its refs match the posted value, stated in the rebuilt words.
        pytest.param(
            "finance",
            [
                Assumption(
                    kind="declared_default",
                    binding_ref="how:crsTypeOfFinance",
                    text="forged",
                    refs=["CRSTypeOfFinance110"],
                )
            ],
            [
                Assumption(
                    kind="declared_default",
                    binding_ref="how:crsTypeOfFinance",
                    text="standard grants",
                    refs=["CRSTypeOfFinance110"],
                )
            ],
            id="echo-for-a-posted-slot",
        ),
    ],
)
def test_a_resubmit_states_the_echoed_assumptions_that_still_hold(
    edit: str | None, echo: list[Assumption] | None, expected: list[Assumption]
):
    """Each echoed item holds while the rebuilt answer still yields it, matched by kind,
    binding and value dcids, and is worded from the rebuilt answer. With no edit to the
    flow slot, a repost or a new year keeps the ODA default."""
    first = _first_answer()
    identity = first.spec.definition.identity
    if edit == "year":
        identity = identity.model_copy(
            update={"requested_window": IdentityWindowRange(start_year=2020, end_year=2020)}
        )
    elif edit == "finance":
        identity = _bound(identity, "crsTypeOfFinance", "CRSTypeOfFinance110")
    part = _resubmit(identity, echo=first.assumptions if echo is None else echo)
    assert part.assumptions == expected


def test_a_resubmit_with_no_echo_states_the_roles_alone():
    """The posted identity binds the flow, so with no echo the engine records it as posted
    and assumes only what the identity leaves open."""
    first = _first_answer()
    assert _resubmit(first.spec.definition.identity, echo=[]).assumptions == _ASSUMED_ROLES


def test_an_edit_that_changes_the_defaulted_slot_drops_its_assumption():
    """The echo names ODA; the flow slot edited to private sector instruments binds another
    value, so the rebuilt answer states no flow default."""
    first = _first_answer()
    edited = _bound(first.spec.definition.identity, "crsTypeOfFlow", "CRSTypeOfFlow60")
    part = _resubmit(edited, echo=first.assumptions, graph=_graph_with_psi())
    flow = next(
        f for f in part.spec.resolved.trace.slot_filters if f.key.as_ref() == "how:crsTypeOfFlow"
    )
    assert [value.ref for value in flow.refs] == ["CRSTypeOfFlow60"]
    assert [item.kind for item in part.assumptions if item.kind == "declared_default"] == []
