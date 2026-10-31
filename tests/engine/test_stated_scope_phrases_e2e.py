"""A phrase that states the provider scope binds that scope and nothing else: its words leave
the text the binders, qualifier matching and the dropped-qualifier warnings read
(`stated_scope.without_scope_phrases`), on the parsed door and on raw text.

A binder that reads "including imputed multilateral" as a qualifier of the measure refuses
`qualifier_not_bound` on a part the inclusive scope serves, so no bind prompt may hold it.
The parts resolve over `tests.scope.test_provider`'s fixture: CRS health members from the
United States to Kenya, with imputed multilateral health series beside them.
"""

from __future__ import annotations

import pytest

from search.engine.extract import ExtractedEntity, Extraction
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.models import (
    DefinitePart,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    RawTextInput,
    Resolution,
    ResolveRequest,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.fixtures import FakeGraph
from tests.scope.test_basis import _member_node, _PickingLLM
from tests.scope.test_provider import _ALL_YEARS, _CRS_MEMBERS, _KEN, _USA, _cover, _provider

_PHRASE = "including imputed multilateral"
_VARIABLE = f"health aid {_PHRASE}"
_ENTITIES = [
    ParsedEntity(text="United States", role_text="from"),
    ParsedEntity(text="Kenya", role_text="to"),
]


class _ReadingLLM(_PickingLLM):
    """Picks as `_PickingLLM` does and keeps every prompt it reads past extraction. On raw
    text it extracts `variable`, the two places and `dropped`."""

    def __init__(
        self,
        variable: str = _VARIABLE,
        dropped: tuple[str, ...] = (),
        *,
        others: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.variable = variable
        self.others = others
        self.dropped = dropped
        self.prompts: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        if schema is Extraction:
            return Extraction(
                kind="resolution",
                variables=[*self.others, self.variable],
                entities=[
                    ExtractedEntity(text="United States", flow_role="origin", role_span="from"),
                    ExtractedEntity(text="Kenya", flow_role="destination", role_span="to"),
                ],
                dates=[],
                dropped_qualifiers=list(self.dropped),
            ), None
        self.prompts.append(prompt)
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


def _resolve(request: ResolveRequest, llm: _ReadingLLM) -> Resolution:
    """`request` over the CRS health members, which recall returns for "health aid" and for
    `llm`'s variable alike."""
    recall = {"svs": list(_CRS_MEMBERS), "cosine_scores": [0.9] * len(_CRS_MEMBERS)}
    graph = FakeGraph(
        nodes={
            _USA: {"label": "United States", "type": "Country"},
            _KEN: {"label": "Kenya", "type": "Country"},
            **{dcid: _member_node(dcid) for dcid in _CRS_MEMBERS},
        },
        detect=dict.fromkeys(("health aid", llm.variable, *llm.others), recall),
        resolve={"United States": _USA, "Kenya": _KEN},
    )
    response = offline_resolve(
        request, graph=graph, llm=llm, sdmx=FakeSdmxClient(observations={}, availability={})
    )
    assert isinstance(response, Resolution), response
    return response


def _inclusive_part(response: Resolution, llm: _ReadingLLM) -> DefinitePart:
    """The response's one part, which the stated inclusive scope serves and whose binders
    read no word of the scope phrase."""
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    choice = _provider(part.spec)
    assert (choice.active, choice.provenance) == ("provider:inclusive", "explicit")
    assert llm.prompts
    assert not [prompt for prompt in llm.prompts if "imputed" in prompt]
    return part


def _warning_codes(response: Resolution) -> list[str]:
    """The codes of the response's warnings and of each part's."""
    warnings = [*response.diagnostics.warnings, *(w for p in response.parts for w in p.warnings)]
    return [w.code for w in warnings]


def test_a_parsed_part_stating_the_inclusive_scope_binds_it_from_its_words_alone(monkeypatch):
    _cover(monkeypatch, _ALL_YEARS)
    llm = _ReadingLLM()

    response = _resolve(
        ResolveRequest(
            input=ParsedInput(
                parts=[ParsedPart(variable_text=_VARIABLE, entities=_ENTITIES, time_text="2015")]
            )
        ),
        llm,
    )

    part = _inclusive_part(response, llm)
    # Posting the restatement states the same scope.
    assert part.restatement is not None
    assert part.restatement.variable_text == _VARIABLE


@pytest.mark.parametrize(
    ("variable", "dropped"),
    [
        pytest.param(_VARIABLE, (), id="in-the-variable"),
        pytest.param("health aid", (_PHRASE,), id="dropped"),
    ],
)
def test_a_raw_query_stating_the_inclusive_scope_binds_it_and_warns_nothing(
    monkeypatch, variable, dropped
):
    _cover(monkeypatch, _ALL_YEARS)
    llm = _ReadingLLM(variable, dropped)

    response = _resolve(
        ResolveRequest(
            input=RawTextInput(query=f"health aid from the United States to Kenya {_PHRASE}")
        ),
        llm,
    )

    part = _inclusive_part(response, llm)
    assert QUERY_QUALIFIER_DROPPED not in _warning_codes(response)
    assert "measure_not_applied" not in [d.code for d in part.departures]


@pytest.mark.parametrize(
    ("query", "variable", "dropped", "asked"),
    [
        pytest.param(
            "health aid for fragile states from the US to Kenya",
            "health aid",
            ("for fragile states",),
            "for fragile states",
            id="names-no-scope",
        ),
        # "bilateral" and the inclusive phrase state two provider options, so the words
        # state no entry, and the phrase reaches the dropped phrases and the binders.
        pytest.param(
            f"bilateral health aid from the United States to Kenya {_PHRASE}",
            "bilateral health aid",
            (_PHRASE,),
            _PHRASE,
            id="two-provider-options-dropped",
        ),
        pytest.param(
            f"bilateral health aid from the United States to Kenya {_PHRASE}",
            f"bilateral health aid {_PHRASE}",
            (),
            "multilateral",
            id="two-provider-options-in-the-variable",
        ),
    ],
)
def test_a_phrase_stating_no_scope_entry_keeps_its_warning(
    monkeypatch, query, variable, dropped, asked
):
    _cover(monkeypatch, _ALL_YEARS)

    response = _resolve(
        ResolveRequest(input=RawTextInput(query=query)), _ReadingLLM(variable, dropped)
    )

    assert QUERY_QUALIFIER_DROPPED in _warning_codes(response)
    (part,) = response.parts
    assert ("measure_not_applied", asked) in _not_applied(part)


def test_a_dropped_phrase_another_part_states_no_entry_for_keeps_its_warning(monkeypatch):
    """On a request of two parts, a dropped phrase is in neither part's text, so the
    "bilateral" one part states does not license it: the phrase stays and warns."""
    _cover(monkeypatch, _ALL_YEARS)
    query = (
        "bilateral health aid from the United States to Kenya and health ODA from the "
        f"United States to Kenya {_PHRASE}"
    )

    response = _resolve(
        ResolveRequest(input=RawTextInput(query=query)),
        _ReadingLLM("health ODA", (_PHRASE,), others=("bilateral health aid",)),
    )

    assert len(response.parts) == 2
    assert QUERY_QUALIFIER_DROPPED in _warning_codes(response)
    assert any(("measure_not_applied", _PHRASE) in _not_applied(p) for p in response.parts)


def _not_applied(part) -> list[tuple[str, str | None]]:
    """The part's departures as (code, asked) pairs."""
    return [(d.code, getattr(d, "asked", None)) for d in getattr(part, "departures", ())]
