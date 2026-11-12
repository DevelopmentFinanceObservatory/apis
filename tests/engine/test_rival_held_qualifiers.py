"""An answer refuses ``qualifier_not_bound`` on a code term its shape leaves unread that
another family of its five-tuple and role class records, naming the dataset that records it
(`phase_rank_bind._rival_held_qualifiers`).

The index and graph are those of `test_recall_reading_families`: CRS health and agriculture
rows beside ONE's climate finance rows on the FinancialFlow amount, DAC1 ODA commitments
beside ONE's climate finance commitments on the FinancialObligation amount, and series
outside the index. This index adds the CRS purpose "primary education", so a World Bank
enrolment series meets a code phrase no family of its five-tuple records.
"""

from __future__ import annotations

import asyncio

import pytest

from search.engine.binders import GeminiBinder
from search.engine.quantity_read import QuantityOutput
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.region.pipeline import resolve_variable
from tests.engine._binders import elected_reply
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine._quantity import NO_QUANTITY
from tests.engine._source_scopes import part_scope
from tests.engine.test_recall_reading_families import (
    _ADAPTATION,
    _CLIMATE,
    _CLIMATE_ADAPTATION,
    _CLIMATE_TOTAL,
    _CLIMATE_TOTAL_COMMITMENTS,
    _CODES,
    _CRS,
    _CRS_HEALTH,
    _DAC1_COMMITMENTS,
    _code,
    _coverage,
    _graph,
    _index,
)
from tests.fixtures import make_resolver
from tests.graph_release.support import serving_index

_WB_ENROLMENT = "TESTFIXTURE/Rival/PrimaryEnrolment"
_PRIMARY_EDUCATION = _code("CRSPurposeCode11220", "crsPurpose", "primary education", code="11220")


class _Reply:
    """Answers the bind call electing menu shape `chosen`, and the quantity call with no
    quantity."""

    def __init__(self, chosen: int) -> None:
        self._chosen = chosen

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(self._chosen), None


class _ElectHolding(GeminiBinder):
    """The Gemini binder electing the menu shape that holds StatVar `dcid`."""

    def __init__(self, dcid: str) -> None:
        self._dcid = dcid

    async def bind(self, request, *, llm):
        chosen = next(
            position
            for position, shape in enumerate(request.shapes)
            if self._dcid in (shape.sv_arc_facts or ())
        )
        return await super().bind(request, llm=_Reply(chosen))


def _resolve(query, variable, recalled, elect, *, entities, source=None, series=None):
    graph = _graph(query, recalled, series)
    with serving_index(_index((*_CODES, _PRIMARY_EDUCATION))):
        coverage = _coverage()
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
                binder=_ElectHolding(elect),
                sdmx=FakeSdmxClient(observations={}),
                base_timing={},
                coverage=coverage.snapshot(),
                source_scope=None if source is None else part_scope(named=source),
            )
        )


@pytest.mark.parametrize(
    ("query", "variable", "source"),
    [
        pytest.param(
            "CRS adaptation disbursements from Germany 2022",
            "adaptation disbursements",
            _CRS,
            id="named-crs",
        ),
        pytest.param(
            "adaptation disbursements for health from Germany in 2022",
            "adaptation disbursements for health",
            None,
            id="tie-on-phrases-read",
        ),
    ],
)
def test_a_term_only_a_rival_family_records_refuses_naming_its_dataset(query, variable, source):
    """The CRS answer leaves "adaptation" unread and the climate finance family of its
    five-tuple records it, so the part refuses on it and names ONE Data Climate Finance.
    Under the named CRS, election never sees the climate family; in the tie, the CRS reads
    "health" and the climate family "adaptation", so neither leaves and the binder elects
    the CRS. Refuse is the default coverage mode."""
    result = _resolve(
        query,
        variable,
        ((_CRS_HEALTH.dcid, 0.8),),
        _CRS_HEALTH.dcid,
        entities=["Germany"],
        source=source,
    )

    assert result.status == "no_data", result
    assert result.no_data_reason == "qualifier_not_bound", result
    (unbound,) = result.unbound_qualifiers
    assert unbound.text == "adaptation"
    assert [ref.ref for ref in unbound.named] == [_ADAPTATION]
    assert [(ref.ref, ref.label) for ref in unbound.held_by] == [
        (_CLIMATE, "ONE Data Climate Finance")
    ]


@pytest.mark.parametrize(
    ("query", "variable", "recalled", "elect", "entities", "series", "status", "answers"),
    [
        pytest.param(
            "health disbursements to Kenya",
            "health disbursements",
            ((_CRS_HEALTH.dcid, 0.8),),
            _CRS_HEALTH.dcid,
            ["Kenya"],
            None,
            "definite",
            _CRS_HEALTH.dcid,
            id="crs-reads-health",
        ),
        pytest.param(
            "primary education enrolment in Kenya",
            "primary education enrolment",
            ((_WB_ENROLMENT, 0.8),),
            _WB_ENROLMENT,
            ["Kenya"],
            {_WB_ENROLMENT: "School enrollment, primary (% gross)"},
            "no_data",
            _WB_ENROLMENT,
            id="series-outside-the-index",
        ),
        pytest.param(
            "climate finance commitments from Germany in 2022",
            "climate finance commitments",
            ((_DAC1_COMMITMENTS.dcid, 0.8),),
            _CLIMATE_TOTAL_COMMITMENTS.dcid,
            ["Germany"],
            None,
            "definite",
            _CLIMATE_TOTAL_COMMITMENTS.dcid,
            id="climate-reads-climate-finance",
        ),
        pytest.param(
            "germany adaptation disbursements 2022",
            "adaptation disbursements",
            ((_CRS_HEALTH.dcid, 0.8),),
            _CLIMATE_TOTAL.dcid,
            ["Germany"],
            None,
            "definite",
            _CLIMATE_ADAPTATION.dcid,
            id="climate-reads-adaptation",
        ),
    ],
)
def test_a_shape_reading_its_terms_or_meeting_no_rival_records_nothing(
    query, variable, recalled, elect, entities, series, status, answers
):
    """The CRS reads "health" itself; the World Bank series shares no five-tuple with the
    CRS rows that record "primary education", and with no observation in the fake it
    refuses ``no_observations``; the climate family reads "climate finance" and
    "adaptation" beside DAC1 and the CRS. None of them refuses on a term."""
    result = _resolve(query, variable, recalled, elect, entities=entities, series=series)

    assert result.status == status, result
    assert result.no_data_reason in (None, "no_observations"), result
    assert result.unbound_qualifiers == ()
    answered = (
        {member.ref.ref for spec in result.specs for member in spec.resolved.members}
        if result.specs
        else {ref.ref for ref in result.elected_stat_vars or ()}
    )
    assert answered == {answers}, result


@pytest.mark.parametrize(
    ("query", "variable", "source", "refuses"),
    [
        pytest.param(
            "CRS adaptation disbursements from Germany 2022",
            "adaptation disbursements",
            _CRS,
            True,
            id="named-crs-refuses",
        ),
        pytest.param(
            "adaptation disbursements for health from Germany in 2022",
            "adaptation disbursements for health",
            None,
            False,
            id="unnamed-departs",
        ),
    ],
)
def test_a_rival_held_term_follows_the_coverage_mode_in_log_mode(
    search_settings, query, variable, source, refuses
):
    """In log mode a part naming no source answers the CRS and departs "adaptation" as not
    applied, as it departs a term naming codes no shape can bind; a part naming the CRS
    refuses on it in every mode."""
    search_settings(qualifier_coverage="log")
    result = _resolve(
        query,
        variable,
        ((_CRS_HEALTH.dcid, 0.8),),
        _CRS_HEALTH.dcid,
        entities=["Germany"],
        source=source,
    )

    if refuses:
        assert result.no_data_reason == "qualifier_not_bound", result
        assert [q.text for q in result.unbound_qualifiers] == ["adaptation"]
        return
    assert result.no_data_reason != "qualifier_not_bound", result
    assert result.unbound_qualifiers == ()
    dropped = [w for w in result.warnings if w.code == QUERY_QUALIFIER_DROPPED]
    assert len(dropped) == 1 and "adaptation" in dropped[0].message, result.warnings
