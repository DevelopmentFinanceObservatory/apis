"""A word a table notes (`DatasetDeclaration.noted_words`) is read on a part answering from
that table, and the answer states the note as a ``slot_defaulted`` warning
(`phase_rank_bind.dataset_noted_words`, `phase_election._repair_bindings`).

The index and graph are those of `test_recall_reading_families`, with ONE's climate
finance table noting "aid", "ODA" and "official development assistance", and the CRS the
house default for a recipient, as the packaged index declares. The test index gives the
CRS no reader for "aid" (the packaged index reads it through the CRS flow type's declared
default), so a CRS part still refuses on the word: the climate table's note reads on its
own shapes alone. The binder nominates every menu shape beside its pick, as Jev pools
them, so the house policies weigh the CRS wherever the family election keeps it.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from code_index.types import CodeRow, NotedWord
from search.engine.binders import GeminiBinder
from search.engine.quantity_read import QuantityOutput
from search.engine.region.codes import SLOT_DEFAULTED
from search.engine.region.pipeline import resolve_variable
from tests.engine._binders import elected_reply
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine._quantity import NO_QUANTITY
from tests.engine.test_recall_reading_families import (
    _CLIMATE,
    _CLIMATE_ADAPTATION,
    _CLIMATE_TOTAL,
    _CODES,
    _CRS,
    _CRS_HEALTH,
    _DAC1,
    _DAC1_COMMITMENTS,
    _ROWS,
    _SYSTEMS,
    _coverage,
    _dataset,
    _flow,
    _graph,
    _index,
)
from tests.engine.test_rival_held_qualifiers import _Reply
from tests.fixtures import make_resolver
from tests.graph_release.support import serving_index

_NOTE = "The figure counts other official flows as well as ODA."
_NOTED = ("aid", "oda", "official development assistance")
_DATASETS = (
    dataclasses.replace(_dataset(_CRS, 0), house_default_for=("recipient",)),
    _dataset(_DAC1, 1),
    dataclasses.replace(
        _dataset(_CLIMATE, 2),
        noted_words=tuple(NotedWord(word=word, note=_NOTE) for word in _NOTED),
    ),
)
# A CRS ODA row and the CRS flow type it holds, which reads "ODA" and, tied with DAC1's
# aid type, "official development assistance", so the CRS family records the flow type on
# the climate table's five-tuple.
_ODA_FLOW = CodeRow(
    dcid="CRSTypeOfFlow10",
    code_system="crsTypeOfFlow",
    code="10",
    label="official development assistance",
    synonyms=("ODA",),
    parent=None,
)
_CRS_ODA_AGRICULTURE = _flow(
    "TESTFIXTURE/Noted/CrsOdaAgriculture",
    {"crsPurpose": "CRSPurposeCode311", "crsTypeOfFlow": _ODA_FLOW.dcid},
    _CRS,
)
_ROWS_WITH_ODA = (*_ROWS, _CRS_ODA_AGRICULTURE)


class _ElectNominating(GeminiBinder):
    """The Gemini binder electing the menu shape that holds StatVar `dcid` and nominating
    every other menu shape, so each joins the pool the house policies narrow."""

    def __init__(self, dcid: str) -> None:
        self._dcid = dcid

    async def bind(self, request, *, llm):
        chosen = next(
            position
            for position, shape in enumerate(request.shapes)
            if self._dcid in (shape.sv_arc_facts or ())
        )
        others = [position for position in range(len(request.shapes)) if position != chosen]
        reply = elected_reply(chosen, also_plausible=others)
        return await super().bind(request, llm=_Fixed(reply))


class _Fixed:
    """Answers the bind call with `reply`, and the quantity call with no quantity."""

    def __init__(self, reply) -> None:
        self._reply = reply

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        return self._reply, None


def _resolve(query, variable, recalled, elect, entities):
    graph = _graph(query, recalled, rows=_ROWS_WITH_ODA)
    index = _index(
        (*_CODES, _ODA_FLOW),
        datasets=_DATASETS,
        rows=_ROWS_WITH_ODA,
        systems={**_SYSTEMS, "crsTypeOfFlow": (False, "type of flow")},
    )
    with serving_index(index):
        coverage = _coverage(_ROWS_WITH_ODA)
        return asyncio.run(
            resolve_variable(
                variable,
                entities=entities,
                date_request=None,
                detect_query=query,
                role_query=query,
                entity_roles=door_roles_for_test_query(query, entities),
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                llm=_Reply(0),
                binder=_ElectNominating(elect),
                sdmx=FakeSdmxClient(observations={}),
                base_timing={},
                coverage=coverage.snapshot(),
                source_scope=None,
            )
        )


def _answered(result) -> set[str]:
    if result.specs:
        return {member.ref.ref for spec in result.specs for member in spec.resolved.members}
    return {ref.ref for ref in result.elected_stat_vars or ()}


def _notes(result) -> list[str]:
    return [warning.message for warning in result.warnings if warning.code == SLOT_DEFAULTED]


@pytest.mark.parametrize(
    ("query", "variable", "entities", "answers"),
    [
        pytest.param(
            "adaptation aid from Germany in 2022",
            "adaptation aid",
            ["Germany"],
            _CLIMATE_ADAPTATION.dcid,
            id="adaptation-aid",
        ),
        pytest.param(
            "climate aid to Kenya", "climate aid", ["Kenya"], _CLIMATE_TOTAL.dcid, id="climate-aid"
        ),
        pytest.param(
            "climate ODA from Germany to Kenya",
            "climate ODA",
            ["Germany", "Kenya"],
            _CLIMATE_TOTAL.dcid,
            id="climate-oda",
        ),
        pytest.param(
            "climate official development assistance from Germany to Kenya",
            "climate official development assistance",
            ["Germany", "Kenya"],
            _CLIMATE_TOTAL.dcid,
            id="climate-official-development-assistance",
        ),
    ],
)
def test_aid_on_the_climate_table_is_read_and_its_note_stated(query, variable, entities, answers):
    """Recall finds the CRS and the climate family beside it, as the dev graph's resolve
    recalls a climate finance family for "adaptation aid from Germany in 2022", and the
    answer reads "aid" on the climate table and states its note. "ODA" and
    "official development assistance" name a CRS flow type the climate rows lack. The
    climate family reads its noted word beside "climate", so the CRS, which reads the word
    alone, leaves at the family election and the recipient house default never reaches it
    (`phase_rank_bind._phrases_reader`), and the CRS family's record of the flow type
    refuses nothing (`phase_rank_bind._rival_held_qualifiers`)."""
    recalled = ((_CRS_HEALTH.dcid, 0.8), (_CLIMATE_TOTAL.dcid, 0.7))
    result = _resolve(query, variable, recalled, _CLIMATE_TOTAL.dcid, entities)

    assert result.status == "definite", result
    assert result.unbound_qualifiers == ()
    assert _answered(result) == {answers}
    assert _notes(result) == [_NOTE]


@pytest.mark.parametrize(
    ("query", "variable", "recalled", "entities", "status"),
    [
        pytest.param(
            "ODA from Germany in 2022",
            "ODA",
            _DAC1_COMMITMENTS.dcid,
            ["Germany"],
            "definite",
            id="oda-on-dac1",
        ),
        pytest.param(
            "health aid to Kenya",
            "health aid",
            _CRS_HEALTH.dcid,
            ["Kenya"],
            "no_data",
            id="health-aid-on-the-crs",
        ),
    ],
)
def test_a_table_noting_no_word_answers_as_before(query, variable, recalled, entities, status):
    """DAC1 answers "ODA" and the CRS keeps "health aid", and neither states the note. The
    CRS part refuses on "aid", as it does with no table noting the word."""
    result = _resolve(query, variable, ((recalled, 0.8),), recalled, entities)

    assert result.status == status, result
    assert _answered(result) == {recalled}
    assert _notes(result) == []
    if status == "no_data":
        assert [qualifier.text for qualifier in result.unbound_qualifiers] == ["aid"]
