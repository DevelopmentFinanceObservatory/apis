"""A dataset_resubmit of a composite, and a fresh part whose read elected one, mint the Spec
as the composite (`spec_build.build_spec` with a `scope.composite.CompositeMint`), so each
states the composite's spec_id, warnings and counts with no restatement of the table answer
it read.

Each resubmit case answers a composite under the spec_id of its identity. A fresh answer
equals a resubmit of its definition.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from code_index.composites import table_identity
from code_index.declarations import DAC2A_INCLUSIVE
from code_index.shared import get_index
from search.engine import app as search_app
from search.engine import core as search_core
from search.engine.entry import dataset_resubmit
from search.engine.events import ResponseEvent
from search.engine.region import RegionResult
from search.engine.region.composite_answer import CompositeRelaunch
from search.engine.spec_id import identity_spec_id
from search.models import (
    BaseIdentity,
    DatasetResubmitInput,
    DefinitePart,
    Resolution,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityEntry,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import _walk_arcs, offline_events, offline_resolve
from tests.engine.test_aid_dataset_bind_e2e import _ScriptedLLM
from tests.engine.test_composite_mint import _headline_coverage, _recorded_builds
from tests.engine.test_dac2a_destination_populations import (
    _AT_TOTAL,
    _HELD,
    _IMPUTED_ONLY,
    _SCENARIOS,
    _VARIABLE,
    _answer,
    _graph_with_parties,
    _identity,
    _materialise,
    _patched_coverage,
    _resolve_fresh,
    _resolve_resubmit,
    _resubmit,
    _sdmx,
)
from tests.engine.test_promote_shape_sample import (
    _HEADLINE,
    _USA,
    _composites,
    _member_node,
    _SampleReader,
)
from tests.engine.test_whole_total_bind_e2e import _FRA
from tests.fixtures import FakeGraph

_DAC1 = "provenance/OECDDAC1"


def _part(response: object) -> DefinitePart:
    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    return part


def _stated(part: DefinitePart) -> dict[str, Any]:
    """The part's Spec, what its resolved part states beyond its dump, and its warnings."""
    return {
        "spec": part.spec.model_dump(mode="json"),
        "private": part.spec.resolved.__pydantic_private__,
        "warnings": [warning.model_dump(mode="json") for warning in part.warnings],
    }


def _dac2a(
    monkeypatch: pytest.MonkeyPatch,
    identity: BaseIdentity,
    *,
    held: dict[tuple[str, str, str], int] = _HELD,
) -> Callable[[], DefinitePart]:
    """Resubmits of `identity` over the DAC2a fixture's series `held`
    (`test_dac2a_destination_populations`)."""
    _patched_coverage(monkeypatch, held)

    def resubmitted() -> DefinitePart:
        return _part(
            offline_resolve(
                _resubmit(identity),
                graph=_graph_with_parties(),
                llm=_ScriptedLLM(_VARIABLE),
                sdmx=_sdmx(held),
            )
        )

    return resubmitted


def _fresh_definition(monkeypatch: pytest.MonkeyPatch, scenario: str) -> BaseIdentity:
    """The definition a fresh answer to `scenario` states over the DAC2a fixture."""
    _patched_coverage(monkeypatch, _HELD)
    return _part(
        offline_resolve(
            _SCENARIOS[scenario](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )
    ).spec.definition.identity


def _headline(
    monkeypatch: pytest.MonkeyPatch, *, origin: str = _USA, label: str = "United States"
) -> Callable[[], DefinitePart]:
    """The OECD headline from `origin` over 2015 to 2020, net flows to 2017 and grant
    equivalents from 2018."""
    document = {**_composites()[_HEADLINE], "entities": {"origin": [origin]}}
    service = _headline_coverage(document, origin=origin)
    monkeypatch.setattr(search_app, "get_coverage_service", lambda: service)
    monkeypatch.setattr(search_core, "get_coverage_service", lambda: service)
    graph = FakeGraph(
        nodes={
            origin: {"label": label, "type": "Country"},
            **{row.dcid: _member_node(row) for row in get_index().statvars},
        }
    )
    request = ResolveRequest(
        input=DatasetResubmitInput(
            entries=[
                ResubmitIdentityEntry(
                    variable_text="official development assistance",
                    identity=BaseIdentity.model_validate(document),
                )
            ]
        )
    )
    return lambda: _part(
        offline_resolve(
            request,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            materialise_graph=_SampleReader(serve=False, arcs=_walk_arcs(graph)),
        )
    )


@pytest.mark.parametrize(
    "case",
    [
        *(
            pytest.param(lambda mp, name=name: _dac2a(mp, _fresh_definition(mp, name)), id=name)
            for name in ("H1", "H2", "H6")
        ),
        pytest.param(
            lambda mp: _dac2a(mp, _identity(DAC2A_INCLUSIVE)), id="inclusive-open-destination"
        ),
        # The imputed line holds a recipient the bilateral table lacks, so the composite
        # counts two recipients where its table answer counts one.
        pytest.param(
            lambda mp: _dac2a(mp, _identity(DAC2A_INCLUSIVE), held=_IMPUTED_ONLY),
            id="inclusive-imputed-only-recipient",
        ),
        pytest.param(_headline, id="oecd-headline"),
    ],
)
def test_a_composite_resubmit_answers_as_the_composite(
    case: Callable[[pytest.MonkeyPatch], Callable[[], DefinitePart]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answered = case(monkeypatch)()

    assert answered.spec.definition.identity.facets is not None
    assert answered.spec.spec_id == identity_spec_id(answered.spec.definition.identity)


def test_a_streamed_composite_resubmit_launches_none_of_its_table_s_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The table identity a composite reads offers slot options to stream. The composite
    launches none and reads its preview per component under its own spec_id."""
    _patched_coverage(monkeypatch, _HELD)
    composite = _identity(DAC2A_INCLUSIVE)
    document = composite.model_dump(mode="json", exclude_none=True)
    table = BaseIdentity.model_validate(dict(table_identity(get_index(), document).identity))
    launched: list[RegionResult] = []
    launch = dataset_resubmit._launch_trailing_tasks

    def recording(region: RegionResult, **kwargs: bool) -> RegionResult:
        launched.append(region)
        return launch(region, **kwargs)

    monkeypatch.setattr(dataset_resubmit, "_launch_trailing_tasks", recording)
    for identity in (table, composite):
        materialise_graph, materialise_sdmx = _materialise()
        request = _resubmit(identity, observations=True)
        options = request.response_options.model_copy(update={"include_option_coverage": True})
        events = offline_events(
            request.model_copy(update={"response_options": options}),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            stream_slot_options=True,
        )

    read, minted = launched
    assert read.slot_options_relaunch is not None
    assert minted.slot_options_relaunch is None
    assert minted.place_options_relaunch is None
    assert isinstance(minted.observations_relaunch, CompositeRelaunch)
    assert [(e.kind, getattr(e, "spec_id", None)) for e in events if e.kind != "sentence"] == [
        ("response", None),
        ("observations", identity_spec_id(composite)),
        ("done", None),
    ]


@pytest.mark.parametrize(
    ("scenario", "held"),
    [
        *(pytest.param(scenario, _HELD, id=scenario) for scenario in ("H1", "H2", "H5", "H6")),
        pytest.param("D2", _AT_TOTAL, id="D2"),
        pytest.param("D2-ranking", _AT_TOTAL, id="D2-ranking"),
    ],
)
def test_a_fresh_answer_equals_a_resubmit_of_its_definition(
    scenario: str, held: dict[tuple[str, str, str], int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh part reads its composite within its population and mints its Spec as the
    composite, so it states the scope, headline, coverage, entities and warnings a resubmit
    of its definition states."""
    _patched_coverage(monkeypatch, held)
    fresh = _resolve_fresh(scenario, held=held)

    again = _resolve_resubmit(fresh.spec.definition.identity, held=held)

    assert again.spec.spec_id == fresh.spec.spec_id
    assert _answer(again.spec, again) == _answer(fresh.spec, fresh)


def test_a_streamed_fresh_composite_launches_none_of_its_table_s_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh part whose read elected the bilateral composite streams no slot or place
    option of the table it read, and reads its preview per component under the composite's
    spec_id."""
    _patched_coverage(monkeypatch, _HELD)
    materialise_graph, materialise_sdmx = _materialise()
    request = _SCENARIOS["H1"]().model_copy(
        update={
            "response_options": ResolveOptions(
                include_observations=True, include_option_coverage=True
            )
        }
    )

    events = offline_events(
        request,
        graph=_graph_with_parties(),
        llm=_ScriptedLLM(_VARIABLE),
        sdmx=_sdmx(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        stream_slot_options=True,
    )

    (response,) = [e for e in events if isinstance(e, ResponseEvent)]
    part = _part(response.response)
    assert (part.spec.definition.identity.facets or {}).get("provenance") is not None
    trailing = events[events.index(response) :]
    assert [(e.kind, getattr(e, "spec_id", None)) for e in trailing if e.kind != "sentence"] == [
        ("response", None),
        ("observations", part.spec.spec_id),
        ("done", None),
    ]


def test_the_oecd_headline_resubmitted_serves_net_flows_then_grant_equivalents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OECD headline over France from 2015 to 2020, posted at the resubmit door, is
    minted as the composite. Its net flows component serves 2015 to 2017 and its grant
    equivalent component 2018 to 2020, both from DAC1, and the Spec states both tables'
    series in the years each serves."""
    minted = _recorded_builds(monkeypatch)

    part = _headline(monkeypatch, origin=_FRA, label="France")()

    (built,) = [each for each in minted if each.kwargs.get("composite") is not None]
    mint = built.kwargs["composite"]
    assert [
        (
            component.role,
            [tuple(period) for period in component.periods],
            component.provenance,
        )
        for component in mint.components()
    ] == [
        ("headline_net", [(2015, 2017)], _DAC1),
        ("headline_ge", [(2018, 2020)], _DAC1),
    ]
    resolved = part.spec.resolved
    assert part.spec.spec_id == mint.spec_id == identity_spec_id(mint.identity)
    assert part.spec.definition.identity.entities == {"origin": [_FRA]}
    assert (resolved.stat_vars.member_count, resolved.stat_vars.served_count) == (2, 2)
    assert resolved.download is not None and resolved.download.series == 2
    assert resolved.coverage is not None and resolved.coverage.observation_count == 6
    assert resolved.headline is not None
    assert [year.year for year in resolved.headline.recent_years] == [2020, 2019, 2018]
