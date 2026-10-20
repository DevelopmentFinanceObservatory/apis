"""End-to-end tests for a part's `assumptions` on the raw-text and dataset_resubmit doors.

"aid flows to Kenya" leaves the flow slot open, so the election fixes it to its declared
default, ODA, and the fixture's heading reads both flow roles open ("to all recipients
from all providers"). A resubmit posts the identity whole, so the engine records the flow
as posted, and the part keeps the default only through the entry's `assumptions` echo.
"""

from __future__ import annotations

import pytest

from search.engine.entry.dataset_entries import with_echoed_defaults
from search.engine.extract import ExtractedEntity, Extraction
from search.models import (
    Assumption,
    BindingValue,
    DatasetResubmitInput,
    DefinitePart,
    GraphRef,
    IdentityWindowRange,
    RawTextInput,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityEntry,
    Slot,
    SlotValue,
    Spec,
)
from search.render.assumptions import assumptions_for
from tests.engine._harness import offline_resolve
from tests.engine.test_operations_e2e import _EMPTY_SDMX, _OpenFlowAndSectorBindLLM
from tests.engine.test_qualifier_suppression_and_provenance import _e2e_graph_for

_ASSUMED_ODA = Assumption(kind="declared_default", binding_ref="how:crsTypeOfFlow", text="ODA")
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


def _resubmit(spec: Spec, *, echo: list[Assumption], window: bool = False) -> DefinitePart:
    identity = spec.definition.identity
    if window:
        identity = identity.model_copy(
            update={"requested_window": IdentityWindowRange(start_year=2020, end_year=2020)}
        )
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
        graph=_e2e_graph_for("aid flows"),
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


@pytest.mark.parametrize("window", [False, True], ids=["unchanged", "year-edited"])
def test_a_resubmit_echoing_the_default_keeps_it_while_the_slot_holds_it(window: bool):
    """An edit that leaves the flow slot alone, a repost or a new year, keeps the echoed
    default. The roles are read from the identity on this door as on the first."""
    first = _first_answer()
    again = _resubmit(first.spec, echo=first.assumptions, window=window)
    assert again.assumptions == [_ASSUMED_ODA, *_ASSUMED_ROLES]


def test_a_resubmit_with_no_echo_states_the_roles_alone():
    """The posted identity binds the flow, so with no echo the engine records it as posted
    and assumes only what the identity leaves open."""
    first = _first_answer()
    assert _resubmit(first.spec, echo=[]).assumptions == _ASSUMED_ROLES


def _as_posted(spec: Spec, *, flow: GraphRef | None = None) -> Spec:
    """`spec` as a resubmit rebuilds it: every slot recorded as posted, the flow slot bound
    to `flow` when one is given."""
    resolved = spec.resolved
    edits = flow is not None
    filters = [
        f.model_copy(
            update={
                "provenance": "binder",
                **({"refs": [flow]} if edits and f.key.as_ref() == "how:crsTypeOfFlow" else {}),
            }
        )
        for f in resolved.trace.slot_filters
    ]
    slots = [
        Slot(key=s.key, binding=BindingValue(value=SlotValue(ref=flow, value_kind="enum_value")))
        if edits and s.key.as_ref() == "how:crsTypeOfFlow"
        else s
        for s in resolved.slots
    ]
    trace = resolved.trace.model_copy(update={"slot_filters": filters})
    return spec.model_copy(
        update={"resolved": resolved.model_copy(update={"slots": slots, "trace": trace})}
    )


def test_an_edit_that_changes_the_defaulted_slot_drops_its_assumption():
    """The echo names ODA. The flow slot left on ODA is restored; one edited to other
    official flows reads other words, so the echo restores nothing."""
    first = _first_answer()
    kept = with_echoed_defaults(_as_posted(first.spec), first.assumptions)
    assert assumptions_for(kept) == [_ASSUMED_ODA, *_ASSUMED_ROLES]

    oof = GraphRef(ref="CRSTypeOfFlow11", label="other official flows")
    edited = _as_posted(first.spec, flow=oof)
    assert with_echoed_defaults(edited, first.assumptions) == edited
    assert assumptions_for(edited) == _ASSUMED_ROLES
