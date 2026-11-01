"""A same-shape merged answer states the download a resubmit of its merged identity states.

The merged answer elects as one answer: one facet group across every StatVar it reads, and
under `latest` one year for each StatVar across every place it names. So its rows are not
its members' rows summed. The merge rebuilds its read plan from the members' plans
(`merged_plan.merged_read_plan`) and states the download, coverage and headline from it, in
the response a buffered request answers and the one a streamed request sends first.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pytest

from search.engine.events import ResponseEvent
from search.models import (
    BaseIdentity,
    DatasetResubmitInput,
    DefinitePart,
    Resolution,
    ResolveRequest,
    ResubmitIdentityEntry,
    Spec,
)
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine import test_dataset_election_grid as grid
from tests.engine import test_dataset_resubmit as resubmit
from tests.engine._fake_coverage import patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_events, offline_resolve
from tests.engine.test_dataset_election_grid import index  # noqa: F401  (fixture)
from tests.fixtures import FakeGraph
from tests.materialise.conftest import FakeGraphReader

type Mode = Literal["buffered", "streamed"]

_DEU, _KEN, _UGA = "country/DEU", "country/KEN", "country/UGA"
_FLOW = RoleNames(donor="financingProvider", recipient="financingRecipient")


def _parts(
    identities: Sequence[BaseIdentity],
    *,
    graph: FakeGraph,
    mode: Mode,
    materialise_graph: FakeGraphReader | None = None,
    sdmx: FakeSdmxClient | None = None,
) -> list[Spec]:
    """The Specs a resubmit of `identities` answers, from the buffered response or the first
    response event of a streamed one."""
    request = ResolveRequest(
        input=DatasetResubmitInput(
            entries=[ResubmitIdentityEntry(variable_text="aid", identity=i) for i in identities]
        )
    )
    if mode == "buffered":
        response = offline_resolve(
            request, graph=graph, materialise_graph=materialise_graph, sdmx=sdmx
        )
    else:
        events = offline_events(
            request,
            graph=graph,
            materialise_graph=materialise_graph,
            sdmx=sdmx,
            stream_slot_options=True,
        )
        (response,) = [e.response for e in events if isinstance(e, ResponseEvent)]
    assert isinstance(response, Resolution)
    specs = []
    for part in response.parts:
        assert isinstance(part, DefinitePart), part
        specs.append(part.spec)
    return specs


def _merged_and_resubmitted(
    identities: Sequence[BaseIdentity],
    *,
    graph: FakeGraph,
    mode: Mode,
    materialise_graph: FakeGraphReader | None = None,
    sdmx: FakeSdmxClient | None = None,
) -> tuple[Spec, Spec]:
    """The merged answer of `identities`, and the answer a resubmit of its identity alone
    states."""
    clients = {"materialise_graph": materialise_graph, "sdmx": sdmx}
    first, *rest = _parts(identities, graph=graph, mode=mode, **clients)
    assert all(spec.spec_id == first.spec_id for spec in rest), "the parts did not merge"
    (alone,) = _parts([first.definition.identity], graph=graph, mode="buffered", **clients)
    assert alone.spec_id == first.spec_id
    return first, alone


def _flow_graph() -> FakeGraph:
    nodes: dict[str, dict] = {
        place: {"label": place, "type": "Country"} for place in (_DEU, _KEN, _UGA)
    }
    for row in grid._ROWS:
        nodes[row.dcid] = {
            "label": row.dcid,
            "type": "StatisticalVariable",
            "arcs": {
                **grid._arcs(row),
                "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                "observationProperties": {
                    "nodes": [{"dcid": prop} for prop in row.observation_properties]
                },
            },
        }
    return FakeGraph(nodes=nodes, obs={})


def _flow_reader() -> FakeGraphReader:
    return FakeGraphReader(
        arcs={
            row.dcid: {
                "populationType": (row.population_type,),
                "measuredProperty": (row.measured_property,),
                "statType": (row.stat_type,),
                "constraintProperties": tuple(row.constraint_properties),
                **{prop: (value,) for prop, value in row.constraint_properties.items()},
            }
            for row in grid._ROWS
        }
    )


def _flow_identity(sector: str, recipient: str, *, latest: bool = False) -> BaseIdentity:
    """Germany's grant amount to `recipient` for `sector` on the fixture index's `_SOLE`
    five-tuple, which `_BROAD` alone publishes."""
    identity = grid._first_resolve(grid._SOLE, None, "donor-recipient", {"sector": sector}).identity
    assert identity is not None
    return identity.model_validate(
        {
            **identity.model_dump(mode="json"),
            "entities": {"origin": [_DEU], "destination": [recipient]},
            **({"requested_window": {"kind": "latest"}} if latest else {}),
        }
    )


def _rows(spec: Spec) -> int | None:
    download = spec.resolved.download
    return download.rows if download is not None else None


@pytest.mark.usefixtures("index")
@pytest.mark.parametrize("mode", ["buffered", "streamed"])
def test_a_merge_whose_parts_elect_different_facets_counts_the_one_group_it_elects(
    monkeypatch, mode
):
    """Sector A's grants sit in a USD facet over three years, sector B's in a EUR facet over
    two. Each part serves its own facet, five rows between them. The merged answer elects
    one group, the USD facet holding more points, serves sector A's three rows alone, and
    names sector B's variable as one it serves nothing of."""
    patch_coverage(
        monkeypatch,
        [
            ("sole_a", "usd", _DEU, _KEN, [2019, 2020, 2021]),
            ("sole_b", "eur", _DEU, _KEN, [2020, 2021]),
        ],
        role_names={"sole_a": _FLOW, "sole_b": _FLOW},
        facet_metadata={
            "usd": FacetMetadata(provenance_id=grid._BROAD, unit="USD", observation_period="P1Y"),
            "eur": FacetMetadata(provenance_id=grid._BROAD, unit="EUR", observation_period="P1Y"),
        },
    )

    merged, alone = _merged_and_resubmitted(
        [_flow_identity("A", _KEN), _flow_identity("B", _KEN)],
        graph=_flow_graph(),
        mode=mode,
        materialise_graph=_flow_reader(),
    )

    assert _rows(merged) == _rows(alone) == 3
    assert merged.resolved.download == alone.resolved.download
    assert merged.resolved.coverage == alone.resolved.coverage
    # The group leaves sector B's variable out, and the merged answer names it as unserved.
    assert [ref.ref for ref in merged.resolved.unserved_members] == ["sole_b"]
    assert merged.resolved.unserved_members == alone.resolved.unserved_members
    assert merged.resolved.data_years == alone.resolved.data_years == (2019, 2021)


@pytest.mark.parametrize("mode", ["buffered", "streamed"])
def test_a_merge_of_a_code_and_its_child_counts_the_parent_alone(monkeypatch, mode):
    """Sector B sits under sector A, whose rows include B's. The merge binds A alone, so its
    download counts sector A's three rows, and none of sector B's two."""
    grid._use_index(monkeypatch, grid._index(parents={"B": "A"}, disjoint=False))
    patch_coverage(
        monkeypatch,
        [
            ("sole_a", "usd", _DEU, _KEN, [2019, 2020, 2021]),
            ("sole_b", "usd", _DEU, _KEN, [2020, 2021]),
        ],
        role_names={"sole_a": _FLOW, "sole_b": _FLOW},
        facet_metadata={
            "usd": FacetMetadata(provenance_id=grid._BROAD, unit="USD", observation_period="P1Y")
        },
    )

    merged, alone = _merged_and_resubmitted(
        [_flow_identity("A", _KEN), _flow_identity("B", _KEN)],
        graph=_flow_graph(),
        mode=mode,
        materialise_graph=_flow_reader(),
    )

    assert _rows(merged) == _rows(alone) == 3
    assert merged.resolved.download == alone.resolved.download
    assert merged.resolved.coverage == alone.resolved.coverage


@pytest.mark.usefixtures("index")
@pytest.mark.parametrize("mode", ["buffered", "streamed"])
def test_a_latest_merge_across_places_counts_the_year_each_variable_elects_over_both(
    monkeypatch, mode
):
    """Kenya's latest grant year is 2021 and Uganda's 2020, one row each. The merged answer
    elects 2021 for the variable over both places, which Uganda does not hold, so it serves
    Kenya's row alone."""
    patch_coverage(
        monkeypatch,
        [
            ("sole_a", "usd", _DEU, _KEN, [2020, 2021]),
            ("sole_a", "usd", _DEU, _UGA, [2019, 2020]),
        ],
        role_names={"sole_a": _FLOW},
        facet_metadata={
            "usd": FacetMetadata(provenance_id=grid._BROAD, unit="USD", observation_period="P1Y")
        },
    )

    merged, alone = _merged_and_resubmitted(
        [_flow_identity("A", _KEN, latest=True), _flow_identity("A", _UGA, latest=True)],
        graph=_flow_graph(),
        mode=mode,
        materialise_graph=_flow_reader(),
    )

    assert _rows(merged) == _rows(alone) == 1
    assert merged.resolved.download == alone.resolved.download
    assert merged.resolved.coverage == alone.resolved.coverage
    assert merged.resolved.headline == alone.resolved.headline


@pytest.mark.parametrize("mode", ["buffered", "streamed"])
def test_a_standard_merge_at_one_place_counts_both_variables(mode):
    """India's female and male population, a standard answer at one place, merge into one
    answer reading both variables, whose download counts both series."""
    merged, alone = _merged_and_resubmitted(
        [
            resubmit._gender_entry("Female", resubmit._FEMALE_SV, window=None).identity,
            resubmit._gender_entry("Male", resubmit._MALE_SV, window=None).identity,
        ],
        graph=resubmit._gender_fold_graph(),
        mode=mode,
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    assert _rows(merged) is not None
    assert merged.resolved.download == alone.resolved.download
    assert merged.resolved.download is not None and merged.resolved.download.series == 2
    assert merged.resolved.coverage == alone.resolved.coverage
