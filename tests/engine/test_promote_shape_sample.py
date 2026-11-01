"""The promote samples a posted axis's values across the shape's five-tuple family only for a
shape the code index does not carry, the one shape whose slot taxonomy grounding reads
(`promote_shape._reads_slot_taxonomy`).

The recorded set is every declared composite's table identity and component identities,
derived from one composite identity per declaration (`code_index.composites`), over a fixture
graph holding every StatVar of the fixture code index. The OECD headline's identities post a
`dacTypeOfFinance` slot their members do not declare, so they are the ones that reach the
sample.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from code_index.composites import component_identities, table_identity
from code_index.declarations import COMPOSITES
from code_index.shared import get_index
from search.engine.config import settings
from search.engine.region import promote_shape, resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.engine.retrieval.bound_values import served_members
from search.models import BaseIdentity, ResubmitIdentityPayload
from series_coverage.schema import ABSENT_ENTITY, RoleNames
from tests.code_index.aid_identity import education_aid_identity
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine._axis_doubles import AxisConfirmingGraphReader, axis_confirming_reader
from tests.engine._base_identity import PERSON_COUNT_SHAPE, make_identity, subject_entity
from tests.engine._builders import unbound_slot
from tests.engine._fake_coverage import (
    FakeCoverageService,
    build_coverage_index,
    empty_coverage_snapshot,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import _walk_arcs, walk_options
from tests.fixtures import FakeGraph

_USA = "country/USA"
_KEN = "country/KEN"
_CRS_INCLUSIVE, _DAC2A_INCLUSIVE, _HEADLINE = (c.provenance for c in COMPOSITES)
_YEARS = list(range(2005, 2025))
_WINDOW = {"kind": "range", "start_year": 2015, "end_year": 2020}

# Sorts ahead of every fixture StatVar, so the family page holds it.
_SAMPLED_MEMBER = "0_sampled_family_member"


def _value(prop: str, ref: str) -> dict:
    return {"property": prop, "binding": {"state": "value", "ref": ref}}


def _shape(measured_property: str) -> dict:
    return {
        "kind": "statvar_schema",
        "population_type": "FinancialFlow",
        "measured_property": measured_property,
        "stat_type": "measuredValue",
        "measurement_qualifier": "Nominal",
    }


def _composites() -> dict[str, dict]:
    """One identity per declared composite (`COMPOSITES`)."""
    crs = education_aid_identity(source=_CRS_INCLUSIVE)
    crs["entities"] = {"origin": [_USA], "destination": [_KEN]}
    return {
        _CRS_INCLUSIVE: crs,
        _DAC2A_INCLUSIVE: {
            "identity_version": "spec1",
            "shape": _shape("amount"),
            "slots": [
                _value("dac2aAidType", "DAC2aAidType206"),
                _value("financialFlowType", "Disbursement"),
                _value("nettingBasis", "Net"),
            ],
            "entities": {"origin": [_USA], "destination": [_KEN]},
            "requested_window": _WINDOW,
            "facets": {"provenance": {"state": "value", "ref": _DAC2A_INCLUSIVE}},
        },
        # A donor's total aid as the website posts it.
        _HEADLINE: {
            "identity_version": "spec1",
            "shape": _shape("amount"),
            "slots": [
                _value("dac1AidType", "DAC1AidType1010"),
                {"property": "dacTypeOfFinance", "binding": {"state": "unbound"}},
                _value("financialFlowType", "Disbursement"),
                _value("nettingBasis", "Net"),
            ],
            "entities": {"origin": [_USA]},
            "requested_window": _WINDOW,
            "facets": {"provenance": {"state": "value", "ref": _HEADLINE}},
        },
    }


def _recorded_identities() -> dict[str, dict]:
    index = get_index()
    identities: dict[str, dict] = {}
    for ref, document in _composites().items():
        identities[f"{ref}:table"] = dict(table_identity(index, document))
        for component in component_identities(index, document) or ():
            identities[f"{ref}:{component.role}"] = dict(component.identity)
    return identities


_RECORDED = _recorded_identities()
_HEADLINE_NET = _RECORDED[f"{_HEADLINE}:headline_net"]


def _nodes(*dcids: str) -> dict:
    return {"nodes": [{"dcid": dcid} for dcid in dcids]}


def _member_node(row) -> dict:
    return {
        "label": row.dcid,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": _nodes("StatisticalVariable"),
            "name": {"nodes": [{"value": row.dcid}]},
            "populationType": _nodes(row.population_type),
            "measuredProperty": _nodes(row.measured_property),
            "statType": _nodes(row.stat_type),
            "measurementQualifier": _nodes(row.measurement_qualifier),
            "observationProperties": _nodes(*row.observation_properties),
            "constraintProperties": _nodes(*row.constraint_properties),
            **{p: _nodes(v) for p, v in row.constraint_properties.items()},
        },
    }


@pytest.fixture(scope="module")
def graph() -> FakeGraph:
    """The United States, Kenya and every StatVar of the fixture code index."""
    return FakeGraph(
        nodes={
            _USA: {"label": "United States", "type": "Country"},
            _KEN: {"label": "Kenya", "type": "Country"},
            **{row.dcid: _member_node(row) for row in get_index().statvars},
        }
    )


def _sampled_member(identity: dict) -> dict[str, tuple[str, ...]]:
    """A family member of `identity`'s five-tuple declaring each posted slot at a code the
    index holds for it, so a sample confirms every posted axis."""
    shape = identity["shape"]
    props = [slot["property"] for slot in identity["slots"]]
    codes = {
        prop: next(
            row.constraint_properties[prop]
            for row in get_index().statvars
            if prop in row.constraint_properties
        )
        for prop in props
    }
    return {
        "typeOf": ("StatisticalVariable",),
        "populationType": (shape["population_type"],),
        "measuredProperty": (shape["measured_property"],),
        "statType": (shape["stat_type"],),
        "measurementQualifier": (shape["measurement_qualifier"],),
        "constraintProperties": tuple(props),
        **{prop: (code,) for prop, code in codes.items()},
    }


class _SampleReader(FakeMintGraphReader):
    """Records every reverse read at the point of read, and refuses each unless `serve`."""

    def __init__(self, *, serve: bool, arcs) -> None:
        super().__init__(arcs=arcs)
        self.serve = serve
        self.subject_reads: list[str] = []

    def node_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        self.subject_reads.append(dcid)
        if not self.serve:
            raise AssertionError(f"reverse read of {dcid!r} requested")
        return super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results, deadline=deadline
        )


def _coverage(identity: BaseIdentity):
    """Series over 2005 to 2024 for every member a mint of `identity` serves."""
    members = sorted(served_members(identity) or ())
    assert members
    donor_only = not identity.entities.get("destination")
    roles = RoleNames(
        donor="financingProvider", recipient=None if donor_only else "financingRecipient"
    )
    recipient = ABSENT_ENTITY if donor_only else _KEN
    rows = [(dcid, "facet", _USA, recipient, _YEARS) for dcid in members]
    index = build_coverage_index(rows, role_names=dict.fromkeys(members, roles))
    return FakeCoverageService(index).snapshot()


def _promote(identity_document: dict, graph: FakeGraph, reader: FakeMintGraphReader):
    identity = BaseIdentity.model_validate(identity_document)
    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(identity=identity),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=_coverage(identity),
        options=EngineOptions(materialise_graph=reader, deadline=time.monotonic() + 60),
    )
    assert region.status == "definite", region.no_data_reason
    return region


def _sample_log(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(record.getMessage().removeprefix("promote family axis sample "))
        for record in caplog.records
        if record.name == promote_shape.__name__
    ]


def test_an_index_carried_shape_never_requests_the_family_sample(
    graph: FakeGraph, caplog: pytest.LogCaptureFixture
):
    """The headline's net component posts `dacTypeOfFinance`, which its members do not
    declare, on a five-tuple the index carries: the promote answers without the reverse
    read."""
    caplog.set_level(logging.INFO, logger=promote_shape.__name__)
    reader = _SampleReader(serve=False, arcs=_walk_arcs(graph))

    _promote(_HEADLINE_NET, graph, reader)

    assert reader.subject_reads == []
    assert _sample_log(caplog) == []


@pytest.mark.parametrize("key", sorted(_RECORDED))
def test_the_family_sample_leaves_an_index_carried_spec_unchanged(
    key: str, graph: FakeGraph, monkeypatch: pytest.MonkeyPatch
):
    """Forcing the sample open, served by a family member declaring every posted axis,
    yields the Spec the gated promote yields with every reverse read refused."""
    identity = _RECORDED[key]
    gated = _promote(identity, graph, _SampleReader(serve=False, arcs=_walk_arcs(graph)))

    additions: list[dict[str, list[str]]] = []
    sample = promote_shape._family_axis_sample

    def recorded_sample(state, axes):
        additions.append(sample(state, axes))
        return additions[-1]

    monkeypatch.setattr(promote_shape, "_reads_slot_taxonomy", lambda shape: True)
    monkeypatch.setattr(promote_shape, "_family_axis_sample", recorded_sample)
    arcs = {_SAMPLED_MEMBER: _sampled_member(identity), **_walk_arcs(graph)}
    sampled = _promote(identity, graph, _SampleReader(serve=True, arcs=arcs))

    if key.startswith(_HEADLINE):
        assert additions and all(additions), additions
    assert [spec.model_dump(mode="json") for spec in sampled.specs] == [
        spec.model_dump(mode="json") for spec in gated.specs
    ]


@dataclass
class _DeadlineRecordingReader(AxisConfirmingGraphReader):
    """Records each family sample's lead and deadline."""

    sample_reads: list[tuple[str, float | None]] = field(default_factory=list)

    def node_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        if max_results == settings().resubmit_axis_sample:
            self.sample_reads.append((dcid, deadline))
        return super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results, deadline=deadline
        )


def test_a_shape_the_index_does_not_carry_still_samples_its_family(
    caplog: pytest.LogCaptureFixture,
):
    """An unbound gender slot on Person/count, which the index does not carry, is confirmed
    by the family sample under the request deadline, and its options are the family's."""
    caplog.set_level(logging.INFO, logger=promote_shape.__name__)
    gender = unbound_slot("how", "gender", "gender")
    inp = ResubmitIdentityPayload(
        identity=make_identity(
            PERSON_COUNT_SHAPE, [gender], entities=[subject_entity("country/IND")]
        ),
    )
    family = axis_confirming_reader("Count_Person", "Count_Person_Male", "Count_Person_Female")
    reader = _DeadlineRecordingReader(
        family_arcs=family.family_arcs, inner=family.inner, graph=family.graph
    )
    deadline = time.monotonic() + 60

    region = resolve_spec_resubmit(
        inp=inp,
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(
            FakeGraph(), EngineOptions(materialise_graph=reader, deadline=deadline)
        ),
    )

    assert region.status == "definite"
    assert reader.sample_reads == [("count", deadline)]
    assert _sample_log(caplog) == [{"family_lead": "count", "unconfirmed_axes": ["gender"]}]
    (gender_options,) = (
        so
        for so in region.specs[0].resolved.slot_options
        if so.key.property is not None and so.key.property.ref == "gender"
    )
    assert {option.value.ref.ref for option in gender_options.options} == {"Male", "Female"}
