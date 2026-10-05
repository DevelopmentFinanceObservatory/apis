"""A same-shape merge reads the graph for its facet options only when the response is not
streamed.

Under SSE every leg's facet options arrive in trailing events, so a leg's spec carries none
and the merge has nothing to judge: the merged region's trailing task streams its options
over the merged probe after the response. A non-streaming response carries options inline,
so the merge judges them over the merged probe there, one read after the legs.
"""

from __future__ import annotations

import asyncio

import pytest

from search.engine.core import resolve_events
from search.engine.events import FacetOptionsEvent, ResponseEvent
from search.engine.region import option_coverage
from search.models import ParsedEntity, ParsedInput, ParsedPart, ResolveOptions, ResolveRequest
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM


def _request() -> ResolveRequest:
    return ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text=place)])
                for place in ("Kenya", "Ethiopia")
            ]
        ),
        response_options=ResolveOptions(include_option_coverage=True),
    )


@pytest.mark.parametrize(("stream", "merge_reads"), [(True, 0), (False, 1)])
def test_the_merge_reads_facet_options_only_for_a_non_streaming_response(
    monkeypatch, stream: bool, merge_reads: int
):
    reads: list[object] = []
    build = option_coverage._maybe_build_facet_options

    def _counted(probe, **kwargs):
        reads.append(probe)
        return build(probe, **kwargs)

    monkeypatch.setattr("search.engine.entry.multi._maybe_build_facet_options", _counted)

    async def _events():
        return [
            event
            async for event in resolve_events(
                _request(),
                graph=FakeGraph(),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(),
                stream_slot_options=stream,
            )
        ]

    events = asyncio.run(_events())

    (response,) = [event.response for event in events if isinstance(event, ResponseEvent)]
    first, second = response.parts
    assert first.spec.spec_id == second.spec.spec_id
    assert len(reads) == merge_reads
    if stream:
        # The merged spec's options stream after the response, under its own spec id.
        assert first.spec.resolved.facet_options == []
        assert {event.spec_id for event in events if isinstance(event, FacetOptionsEvent)} == {
            first.spec.spec_id
        }
    else:
        assert first.spec.resolved.facet_options
