"""A query qualifier bound to a slot grounding leaves unbound is unbound in the final trace.

When no member asserts the bound purpose, the elected door unbinds it and resolves again
(`unrealisable.unbind_unrealisable`), so the purpose slot is unbound. The purpose is a code
the question names on a slot the shape holds, so a definite read refuses
`qualifier_not_bound` with that read flagged, and a candidate set departs
`measure_not_applied`. The trace's qualifier, the slot and the identity then agree, on the
definite and the candidate path alike. A measure sibling answering for a data-less elected shape is judged
by its own materialisation. The `education` qualifier is the packaged code index's own.
"""

from __future__ import annotations

import pytest

from code_index.types import MeasureAxis, MeasureIndex, MeasurePartitionLabel, MeasureValueRow
from search.engine import spec_build
from search.engine.bind import _UniversalBindOutput
from search.engine.extract import ExtractedEntity, Extraction
from search.engine.quantity_read import QuantityOutput
from search.engine.region import candidate_grounding
from search.engine.retrieval.bound_values import MeasurePartition
from search.models import CandidatesPart, DefinitePart, NoDataPart, Resolution, Spec
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph

_KEN = "country/KEN"
_VARIABLE = "education widgets"


@pytest.fixture(autouse=True)
def _log_coverage(search_settings):
    """The word "widgets" beside education is one no coverage reader covers, so the log mode
    keeps these grounding cases answering."""
    search_settings(qualifier_coverage="log")


_AMOUNT = "Amount_Widget"
_COUNT = "Count_Widget"
_HEALTH = "CRSPurposeCode12110"
_EDUCATION = "CRSPurposeCode11110"


def _arcs(sv: str, measured_property: str, purpose: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": sv}]},
        "populationType": {"nodes": [{"dcid": "Widget"}]},
        "measuredProperty": {"nodes": [{"dcid": measured_property}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": "crsPurpose"}]},
        "crsPurpose": {"nodes": [{"dcid": purpose}]},
    }


class _LLM:
    """Extract the variable at Kenya and elect the first shape with nothing drafted, so the
    `education` qualifier alone binds the purpose. `also_plausible` offers the rest."""

    def __init__(self, also_plausible: list[int] | None) -> None:
        self._also_plausible = also_plausible

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        if schema is Extraction:
            return Extraction(
                kind="resolution",
                variables=[_VARIABLE],
                entities=[
                    ExtractedEntity(
                        text="Kenya", span="Kenya", flow_role="unspecified", role_span=None
                    )
                ],
            ), None
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], unplaced=[], also_plausible=self._also_plausible
        ), None


def _resolve(
    members: dict[str, tuple[str, str]], *, also_plausible=None, without_data=frozenset()
) -> Resolution:
    """Resolve the variable over `members`, each a StatVar's measured property and purpose,
    recalled in order and each holding data at Kenya unless named in `without_data`."""
    graph = FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            "crsPurpose": {"label": "purpose"},
            _HEALTH: {"label": "Health policy"},
            _EDUCATION: {"label": "Education policy"},
            **{
                sv: {"label": sv, "type": "StatisticalVariable", "arcs": _arcs(sv, *facts)}
                for sv, facts in members.items()
            },
        },
        detect={
            _VARIABLE: {
                "svs": list(members),
                "cosine_scores": [0.9 - 0.01 * rank for rank in range(len(members))],
            }
        },
        resolve={"Kenya": _KEN},
        obs={
            f"{sv}|{_KEN}": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2021",
                    "obsCount": 2,
                    "provenanceId": "dc/base/WidgetFixture",
                }
            ]
            for sv in members
            if sv not in without_data
        },
    )
    response = offline_resolve(
        make_request(f"{_VARIABLE} in Kenya"),
        graph=graph,
        llm=_LLM(also_plausible),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    assert isinstance(response, Resolution)
    return response


def _purpose(spec: Spec) -> tuple[list[tuple[str, str | None]], str, str]:
    """The trace's qualifier dispositions and slots, the purpose slot's binding kind and
    the identity's purpose binding state."""
    (slot,) = (s for s in spec.resolved.slots if s.key.property is not None)
    (identity_slot,) = spec.definition.identity.slots
    return (
        [(q.disposition, q.slot) for q in spec.resolved.trace.query_qualifiers],
        slot.binding.kind,
        identity_slot.binding.state,
    )


def test_a_definite_answer_whose_purpose_filter_met_no_member_refuses_with_the_read_flagged():
    response = _resolve({_AMOUNT: ("amount", _HEALTH)})

    (part,) = response.parts
    assert isinstance(part, NoDataPart)
    assert part.no_data.reason == "qualifier_not_bound"
    assert [(q.text, q.slot) for q in part.no_data.unbound_qualifiers] == [
        ("education", "crsPurpose")
    ]
    flagged = part.no_data.flagged_match
    assert flagged is not None
    # The flagged match's trace names the term the refusal names.
    assert _purpose(flagged) == ([("unbound", "crsPurpose")], "unbound", "unbound")
    assert part.departures == []
    assert "query_qualifier_dropped" not in [w.code for w in part.warnings]


def test_a_definite_answer_whose_purpose_filter_held_keeps_the_bound_qualifier():
    response = _resolve({_AMOUNT: ("amount", _EDUCATION)})

    (part,) = response.parts
    assert isinstance(part, DefinitePart)
    assert _purpose(part.spec) == ([("bound", "crsPurpose")], "value", "value")


def test_an_elected_candidate_whose_purpose_filter_met_no_member_reports_the_qualifier_unbound():
    response = _resolve(
        {_AMOUNT: ("amount", _HEALTH), _COUNT: ("count", _EDUCATION)}, also_plausible=[1]
    )

    (part,) = response.parts
    assert isinstance(part, CandidatesPart)
    elected, alternate = part.candidates.specs
    assert _purpose(elected) == ([("unbound", "crsPurpose")], "unbound", "unbound")
    assert _purpose(alternate) == ([], "value", "value")
    assert "query_qualifier_dropped" in [w.code for w in part.warnings]


def test_a_measure_sibling_whose_purpose_filter_held_answers_for_an_elected_shape_with_no_data(
    monkeypatch: pytest.MonkeyPatch,
):
    """The elected amount shape holds no education member and no data, so its purpose
    filter meets no member. Its count sibling, labelled by a synthetic measure index, answers,
    and the purpose filter held there, so no warning or departure says it was dropped."""
    count = MeasurePartition(values=(("measuredProperty", "count"),), members=(_COUNT,))
    measure_index = MeasureIndex(
        axes=(MeasureAxis(axis="measuredProperty", kind="shape", role="basis", prefer=("count",)),),
        values=(
            MeasureValueRow(
                dcid="count",
                axis="measuredProperty",
                label="count",
                synonyms=(),
                short_name=None,
                matchable=True,
            ),
        ),
        partition_labels=(
            MeasurePartitionLabel(axis_values=(("measuredProperty", "count"),), label="count"),
        ),
    )
    monkeypatch.setattr(spec_build, "get_measure_index", lambda: measure_index)
    monkeypatch.setattr(
        candidate_grounding,
        "observed_measure_family",
        lambda shape, *_args, **_kwargs: (
            (count, ()) if shape.meas_prop_dcid == "count" else (None, ())
        ),
    )

    response = _resolve(
        {_AMOUNT: ("amount", _HEALTH), _COUNT: ("count", _EDUCATION)},
        also_plausible=[1],
        without_data={_AMOUNT},
    )

    (part,) = response.parts
    assert isinstance(part, DefinitePart)
    assert "elected_shape_no_data" in [w.code for w in part.warnings]
    assert [sv.ref.ref for sv in part.spec.resolved.members] == [_COUNT]
    assert _purpose(part.spec)[1:] == ("value", "value")
    assert "query_qualifier_dropped" not in [w.code for w in part.warnings]
    assert "measure_not_applied" not in [d.code for d in part.departures]
    assert part.spec.resolved.rendered_sentence == (
        "Kenya's count for education policy and administrative management, 2020 to 2021."
    )
