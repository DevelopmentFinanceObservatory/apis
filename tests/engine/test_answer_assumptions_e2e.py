"""End-to-end tests for a part's `assumptions` on the raw-text and dataset_resubmit doors.

"aid flows to Kenya" leaves the flow slot open, so the election fixes it to its declared
default, official development assistance (CRSTypeOfFlow10, short name "ODA"), and the
fixture's heading reads both flow roles open ("to all recipients from all providers").
A resubmit posts the identity whole, so the engine records the flow
as posted, and the part keeps the default only through the entry's `assumptions` echo,
matched to the rebuilt slot by its value dcids. A candidates entry posting the aid
reading beside a loan reading, whose debt roles read "every debtor" and "every creditor",
shows a part whose readings share no item, each reading's own list in
`reading_assumptions`.
"""

from __future__ import annotations

import pytest

from search.engine.bind import SlotBindingDraft
from search.engine.extract import ExtractedEntity, ExtractedOperation, Extraction
from search.models import (
    Assumption,
    BaseIdentity,
    CandidatesPart,
    DatasetResubmitInput,
    DefinitePart,
    IdentityBindingValue,
    IdentityWindowRange,
    Operation,
    RawTextInput,
    ResolveOptions,
    ResolveRequest,
    ResubmitCandidate,
    ResubmitCandidatesEntry,
    ResubmitIdentityEntry,
)
from tests.engine._harness import make_request, offline_resolve
from tests.engine.test_operations_e2e import (
    _EMPTY_SDMX,
    _QUERY,
    _extraction,
    _OpenFlowAndSectorBindLLM,
    _ScriptedLLM,
)
from tests.engine.test_qualifier_suppression_and_provenance import (
    _E2E_NODES,
    _E2E_OBS,
    _E2E_SV_DCID,
    _e2e_graph_for,
    _e2e_sv_arcs,
)
from tests.fixtures import FakeGraph

_ASSUMED_ODA = Assumption(
    kind="declared_default",
    binding_ref="how:crsTypeOfFlow",
    text="official development assistance",
    refs=["CRSTypeOfFlow10"],
)
_ASSUMED_ROLES = [
    Assumption(kind="open_role", binding_ref="where:", text="every recipient"),
    Assumption(kind="open_role", binding_ref="origin:", text="every provider"),
]
_ASSUMED_CLAUSE = " Assumed: official development assistance; every recipient; every provider."
# A parametrize row's echo that is the first answer's own list.
_FIRST = "first"


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


def _graph_with_psi(detect: dict | None = None) -> FakeGraph:
    """The fixture graph with the private sector instruments series beside the ODA one,
    with ``detect`` as its detect entries."""
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
        detect=detect or {},
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
    identity: BaseIdentity, *, echo: list[Assumption] | None, graph: FakeGraph | None = None
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


class _PsiFlowBindLLM(_OpenFlowAndSectorBindLLM):
    """Bind fixes the flow to private sector instruments for "psi flows" and leaves it open
    for every other phrase."""

    async def generate_structured(self, *, prompt, system, schema):
        result, usage = await super().generate_structured(
            prompt=prompt, system=system, schema=schema
        )
        if schema.__name__ == "_UniversalBindOutput" and "Variable phrase: psi flows" in prompt:
            result.bindings = [
                SlotBindingDraft(
                    axis="how",
                    property_dcid="crsTypeOfFlow",
                    kind="value",
                    value_dcids=["CRSTypeOfFlow60"],
                ),
                SlotBindingDraft(axis="how", property_dcid="crsPurpose", kind="unbound"),
            ]
        return result, usage


def test_a_merged_part_keeps_the_default_its_own_request_left():
    """ "aid flows" leaves the flow open and "psi flows" names private sector instruments, so
    the two merge into one Spec whose flow slot reads as recognised. The first part's own
    request still left its flow to the ODA default, which it states; the second asked for
    its flow and states none. The merged heading names no flow role, so neither part
    states one."""
    detect = {
        "aid flows": {"svs": [_E2E_SV_DCID], "cosine_scores": [0.9]},
        "psi flows": {"svs": [_PSI_SV_DCID], "cosine_scores": [0.9]},
    }
    response = offline_resolve(
        ResolveRequest(
            input=RawTextInput(query="aid flows and psi flows to Kenya"),
            response_options=ResolveOptions(),
        ),
        graph=_graph_with_psi(detect),
        llm=_PsiFlowBindLLM(
            Extraction(
                kind="resolution",
                variables=["aid flows", "psi flows"],
                entities=[ExtractedEntity(text="Kenya")],
            )
        ),
        sdmx=_EMPTY_SDMX,
    )
    first, second = response.parts
    assert isinstance(first, DefinitePart)
    assert isinstance(second, DefinitePart)
    assert first.spec.spec_id == second.spec.spec_id
    assert "slot_defaulted" in [warning.code for warning in first.warnings]
    assert [part.assumptions for part in response.parts] == [[_ASSUMED_ODA], []]
    assert response.rendered_sentence is not None
    assert response.rendered_sentence.endswith(" Assumed: official development assistance.")


@pytest.mark.parametrize(
    ("over", "expected"),
    [
        pytest.param(None, _ASSUMED_ROLES, id="no-operation"),
        # "top 5 donors" ranks over the providers, so the request asked every one of them.
        pytest.param("origin", _ASSUMED_ROLES[:1], id="rank-over-origin"),
        pytest.param("destination", _ASSUMED_ROLES[1:], id="rank-over-destination"),
        # An operation over places asks no flow role (`asked_open_axes`).
        pytest.param("place", _ASSUMED_ROLES, id="rank-over-place"),
    ],
)
def test_a_role_an_operation_runs_over_is_asked(over: str | None, expected: list[Assumption]):
    """A rank over a flow axis asks every member of it, so that role is no open-role
    assumption, in the part's list or in the sentence's clause."""
    operations = (
        []
        if over is None
        else [ExtractedOperation(verb="rank", variable=0, phrase="top 5 donors", over=over)]
    )
    response = offline_resolve(
        make_request("top 5 donors for standard grants in Kenya"),
        graph=_e2e_graph_for(_QUERY),
        llm=_ScriptedLLM(_extraction(operations=operations)),
        sdmx=_EMPTY_SDMX,
    )
    part = response.parts[0]
    assert isinstance(part, DefinitePart)
    assert part.assumptions == expected
    assert response.rendered_sentence is not None
    clause = "; ".join(item.text for item in expected)
    assert response.rendered_sentence.endswith(f" Assumed: {clause}.")


@pytest.mark.parametrize(
    ("edit", "echo", "expected"),
    [
        pytest.param(None, _FIRST, [_ASSUMED_ODA, *_ASSUMED_ROLES], id="unchanged"),
        pytest.param("year", _FIRST, [_ASSUMED_ODA, *_ASSUMED_ROLES], id="year-edited"),
        # The echo of a part that stated none states none, whatever a fresh request yields.
        pytest.param("year", [], [], id="year-edited-empty-echo"),
        # The echo's words differ from the rebuilt ones, as after a label change between
        # deploys; the refs still match, so the default holds in the rebuilt words.
        pytest.param(
            None,
            [_ASSUMED_ODA.model_copy(update={"text": "ODA"})],
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
                    text="standard grant",
                    refs=["CRSTypeOfFinance110"],
                )
            ],
            id="echo-for-a-posted-slot",
        ),
    ],
)
def test_a_resubmit_states_the_echoed_assumptions_that_still_hold(
    edit: str | None, echo: list[Assumption] | str, expected: list[Assumption]
):
    """Each echoed item holds while the rebuilt answer still yields it, matched by kind,
    binding and value dcids, and is worded from the rebuilt answer. With no edit to the
    flow slot, a repost or a new year keeps the official development assistance default."""
    first = _first_answer()
    identity = first.spec.definition.identity
    if edit == "year":
        identity = identity.model_copy(
            update={"requested_window": IdentityWindowRange(start_year=2020, end_year=2020)}
        )
    elif edit == "finance":
        identity = _bound(identity, "crsTypeOfFinance", "CRSTypeOfFinance110")
    part = _resubmit(identity, echo=first.assumptions if isinstance(echo, str) else echo)
    assert part.assumptions == expected


def test_a_resubmit_with_no_echo_states_the_roles_alone():
    """The posted identity binds the flow, so with no echo the engine records it as posted
    and assumes only what the identity leaves open."""
    first = _first_answer()
    assert _resubmit(first.spec.definition.identity, echo=None).assumptions == _ASSUMED_ROLES


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


def _graph_with_ghana() -> FakeGraph:
    """The fixture graph with the fixture's series also held at Ghana."""
    return FakeGraph(
        nodes={**_E2E_NODES, "country/GHA": {"label": "Ghana", "type": "Country"}},
        detect={},
        resolve={"Kenya": "country/KEN"},
        obs={
            **_E2E_OBS,
            **{key.replace("country/KEN", "country/GHA"): value for key, value in _E2E_OBS.items()},
        },
    )


def test_a_folded_part_states_only_its_own_assumptions():
    """Entries for Kenya and Ghana fold into one Spec. The first echoes the flow default
    and the providers, the second echoes none: the clause names both items under the
    first part alone, and no token claims them for the second."""
    first = _first_answer()
    identity = first.spec.definition.identity
    ghana = identity.model_copy(update={"entities": {"observationAbout": ["country/GHA"]}})
    echo = [_ASSUMED_ODA, _ASSUMED_ROLES[1]]
    response = offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(
                        variable_text="aid flows", identity=identity, assumptions=echo
                    ),
                    ResubmitIdentityEntry(
                        variable_text="aid flows", identity=ghana, assumptions=[]
                    ),
                ]
            ),
            response_options=ResolveOptions(),
        ),
        graph=_graph_with_ghana(),
        llm=_OpenFlowAndSectorBindLLM(Extraction(kind="resolution", variables=[], entities=[])),
        sdmx=_EMPTY_SDMX,
    )
    assert [part.assumptions for part in response.parts] == [echo, []]
    assert "binding_folded" in [warning.code for warning in response.parts[1].warnings]
    assert response.sentence_plan is not None
    items = {item.text for item in echo}
    assert [
        token.part_index
        for token in response.sentence_plan
        if token.segment == "assumption" and token.text in items
    ] == [0, 0]


# A loan disbursement series outside the code index: its `financialFlowType` puts it in the
# debt lending family, whose roles read "every debtor" and "every creditor" where the aid
# series reads "every recipient" and "every provider".
_LOAN_SV_DCID = "TESTFIXTURE/AnswerAssumptionsE2E/LoanDisbursement"
_ASSUMED_LOAN_ROLES = [
    Assumption(kind="open_role", binding_ref="where:", text="every debtor"),
    Assumption(kind="open_role", binding_ref="origin:", text="every creditor"),
]


def _graph_with_loans() -> FakeGraph:
    """The fixture graph with the loan series beside the aid one, both held at Kenya."""
    return FakeGraph(
        nodes={
            **_E2E_NODES,
            _LOAN_SV_DCID: {
                "label": "test loan disbursements",
                "type": "StatisticalVariable",
                "arcs": {
                    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                    "populationType": {"nodes": [{"dcid": "FinancialFlow"}]},
                    "measuredProperty": {"nodes": [{"dcid": "amount"}]},
                    "statType": {"nodes": [{"dcid": "measuredValue"}]},
                    "constraintProperties": {"nodes": [{"dcid": "financialFlowType"}]},
                    "financialFlowType": {"nodes": [{"dcid": "Disbursement"}]},
                },
            },
            "Disbursement": {"label": "disbursement"},
        },
        detect={"loans": {"svs": [_LOAN_SV_DCID], "cosine_scores": [0.9]}},
        resolve={"Kenya": "country/KEN"},
        obs={
            **_E2E_OBS,
            f"{_LOAN_SV_DCID}|country/KEN": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
            ],
        },
    )


def _loan_identity() -> BaseIdentity:
    """The identity a fresh "loans to Kenya" answers with."""
    response = offline_resolve(
        make_request("loans to Kenya"),
        graph=_graph_with_loans(),
        llm=_ScriptedLLM(
            Extraction(
                kind="resolution", variables=["loans"], entities=[ExtractedEntity(text="Kenya")]
            )
        ),
        sdmx=_EMPTY_SDMX,
    )
    part = response.parts[0]
    assert isinstance(part, DefinitePart)
    assert part.assumptions == _ASSUMED_LOAN_ROLES
    return part.spec.definition.identity


def _post(entry: ResubmitIdentityEntry | ResubmitCandidatesEntry):
    """The response to a dataset_resubmit of ``entry`` alone, on the loan graph."""
    return offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(entries=[entry]), response_options=ResolveOptions()
        ),
        graph=_graph_with_loans(),
        llm=_OpenFlowAndSectorBindLLM(Extraction(kind="resolution", variables=[], entities=[])),
        sdmx=_EMPTY_SDMX,
    )


def _aid_or_loans(operations: list[Operation]) -> CandidatesPart:
    """The candidates part a dataset_resubmit of the aid and loan readings answers with."""
    response = _post(
        ResubmitCandidatesEntry(
            variable_text="aid flows",
            ordering="intent_rank",
            truncated=False,
            candidates=[
                ResubmitCandidate(identity=_first_answer().spec.definition.identity),
                ResubmitCandidate(identity=_loan_identity()),
            ],
            operations=operations,
        )
    )
    part = response.parts[0]
    assert isinstance(part, CandidatesPart)
    assert response.rendered_sentence is not None
    assert "Assumed:" not in response.rendered_sentence
    return part


@pytest.mark.parametrize(
    ("operations", "aid", "loans"),
    [
        pytest.param([], _ASSUMED_ROLES, _ASSUMED_LOAN_ROLES, id="no-operation"),
        # A rank over the destination asks it on every reading, leaving the origin open.
        pytest.param(
            [Operation(verb="rank", phrase="top 5 recipients", over="destination")],
            _ASSUMED_ROLES[1:],
            _ASSUMED_LOAN_ROLES[1:],
            id="rank-over-destination",
        ),
    ],
)
def test_a_candidates_part_lists_each_reading_s_own_assumptions(
    operations: list[Operation], aid: list[Assumption], loans: list[Assumption]
):
    """The aid reading assumes every provider where the loan reading assumes every creditor,
    so the readings share no item: the part and its clause state none, and
    reading_assumptions holds what each reading assumes once picked, read with the part's
    operations."""
    part = _aid_or_loans(operations)
    assert part.assumptions == []
    aid_spec, loan_spec = part.candidates.specs
    assert part.reading_assumptions == {aid_spec.spec_id: aid, loan_spec.spec_id: loans}


@pytest.mark.parametrize(
    ("echo", "expected"),
    [
        pytest.param("reading", _ASSUMED_LOAN_ROLES, id="reading-assumptions"),
        # The part's shared list leaves out every item the readings word differently.
        pytest.param("part", [], id="part-assumptions"),
    ],
)
def test_a_picked_reading_echoing_its_own_list_states_it(echo: str, expected: list[Assumption]):
    """A client picks the loan reading and posts its identity with an echo: the reading's
    own list states what that reading assumed, the part's shared list states none."""
    part = _aid_or_loans([])
    loan_spec = part.candidates.specs[1]
    echoed = part.reading_assumptions[loan_spec.spec_id] if echo == "reading" else part.assumptions
    response = _post(
        ResubmitIdentityEntry(
            variable_text="aid flows",
            identity=loan_spec.definition.identity,
            assumptions=echoed,
        )
    )
    picked = response.parts[0]
    assert isinstance(picked, DefinitePart)
    assert picked.spec.spec_id == loan_spec.spec_id
    assert picked.assumptions == expected
    assert response.rendered_sentence is not None
    clause = "; ".join(item.text for item in expected)
    assert response.rendered_sentence.endswith(f" Assumed: {clause}." if expected else ".")
    assert ("Assumed:" in response.rendered_sentence) == bool(expected)
