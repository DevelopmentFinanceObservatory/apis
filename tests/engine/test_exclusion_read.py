"""The raw-text door reads what a query asks to leave out beside extraction, blanks it from every
reader of the query's words, and says so on the answer.

Async calls use `asyncio.run` because the suite has no pytest-asyncio plugin.
"""

from __future__ import annotations

import asyncio
import time
import unicodedata
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

import search.engine.clients.llm as llm_module
from search.engine.bind import SlotBindingDraft
from search.engine.clients.llm import LLM
from search.engine.entry import raw_text
from search.engine.errors import LLMInfraError
from search.engine.exclusion_read import (
    EXCLUSION_READ_SYSTEM_PROMPT,
    Exclusion,
    ExclusionRead,
    LeftOut,
    admit_exclusions,
)
from search.engine.extract import ExtractedEntity, Extraction
from search.engine.extraction_store import MemoryExtractionStore
from search.engine.quantity_read import QuantityOutput
from search.engine.region import phase_election
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.region.requirements import judge
from search.engine.region.state import exclusion_requirements
from search.engine.shape import ShapeDraft, shape_draft_from
from tests.engine._binders import elected_reply
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve
from tests.engine._quantity import NO_QUANTITY
from tests.engine.test_qualifier_suppression_and_provenance import _E2E_QUERY, _e2e_graph

pytestmark = pytest.mark.reads_exclusions


def _read(*phrases: str, left_out: tuple[str, ...] = ()) -> ExclusionRead:
    """A read of `phrases`, each leaving out `left_out`, else its words after the first."""
    return ExclusionRead(
        exclusions=[
            Exclusion(phrase=phrase, left_out=list(left_out) or phrase.split()[1:])
            for phrase in phrases
        ]
    )


# The measured read of each exclusion row of the acceptance table: the query and its stretch.
_EXCLUSION_ROWS = [
    ("ODA to Kenya excluding humanitarian aid in 2022", "excluding humanitarian aid"),
    ("ODA to Kenya in 2022 not from the United States", "not from the United States"),
    ("ODA to Africa except Kenya in 2022", "except Kenya"),
    ("ODA from France except to Kenya and Uganda in 2022", "except to Kenya and Uganda"),
    ("AOD a Kenia sin ayuda humanitaria en 2022", "sin ayuda humanitaria"),
    ("gross ODA, not net, to Kenya in 2022", "not net"),
    ("not net ODA to Kenya in 2022", "not net"),
    ("no humanitarian aid to Kenya in 2022", "no humanitarian aid"),
    ("everything but humanitarian aid to Kenya", "but humanitarian aid"),
    ("long-term external debt of Kenya without IMF credit in 2020", "without IMF credit"),
    ("what does Kenya owe China, excluding debt relief", "excluding debt relief"),
    ("ODA to Kenya excluding grants in 2022", "excluding grants"),
    ("ODA excluding standard grants from France", "excluding standard grants"),
    ("ODA to Africa excluding North Africa in 2022", "excluding North Africa"),
    ("bilateral ODA to Kenya excluding bilateral loans in 2022", "excluding bilateral loans"),
    ("ODA to Kenya in euros, not dollars", "not dollars"),
    ("aid to Kenya, not imputed", "not imputed"),
    (
        "Norway's bilateral-only aid to Kenya, not counting what goes through multilaterals",
        "not counting what goes through multilaterals",
    ),
    ("education aid to Kenya excluding multilateral donors", "excluding multilateral donors"),
    ("ODA to Kenya excluding the CRS", "excluding the CRS"),
    ("education aid to Kenya excluding the CRS", "excluding the CRS"),
]


@pytest.mark.parametrize(("query", "phrase"), _EXCLUSION_ROWS)
def test_an_admitted_stretch_leaves_no_word_of_itself_in_the_text(query, phrase):
    """Only what the stretch alone wrote leaves the text, and the rest of the query stays in
    order around one blank."""
    reading = admit_exclusions(query, _read(phrase))

    assert reading.text == query.replace(phrase, ";")
    assert [(stretch.phrase, reading.text[stretch.at]) for stretch in reading.left_out] == [
        (phrase, ";")
    ]


@pytest.mark.parametrize(
    ("query", "phrase"),
    [
        pytest.param(
            "official development assistance excluding debt relief to Kenya in 2022",
            "excluding debt relief",
            id="inside-a-code-label",
        ),
        pytest.param(
            "ODA from non-DAC countries to Kenya in 2022",
            "non-DAC countries",
            id="a-party-label",
        ),
        pytest.param(
            "ODA to Kenya not including multilateral",
            "not including multilateral",
            id="the-provider-scope",
        ),
        pytest.param("ODA to Kenya minus aid", "excluding aid", id="not-the-query-words"),
        pytest.param("casino aid to Kenya", "no aid", id="inside-a-word"),
        pytest.param("not net", "not net", id="the-whole-query"),
        pytest.param("ODA to Kenya", "", id="empty"),
    ],
)
def test_a_stretch_failing_admission_leaves_the_query_byte_identical(query, phrase):
    """A query whose read admits nothing sends extraction the bytes it sends today."""
    assert admit_exclusions(query, _read(phrase, left_out=("aid",))).text == query
    assert admit_exclusions(query, _read(phrase, left_out=("aid",))).left_out == ()


def test_an_empty_read_keeps_a_query_outside_nfc_byte_identical():
    query = unicodedata.normalize("NFD", "AOD à Kenia")
    assert admit_exclusions(query, ExclusionRead(exclusions=[])).text == query


def test_a_stretch_is_found_in_nfc_and_every_occurrence_blanks():
    query = unicodedata.normalize("NFD", "AOD sin ayuda à Kenia, sin ayuda en 2022")
    reading = admit_exclusions(query, _read("sin ayuda"))
    assert reading.text == unicodedata.normalize("NFC", "AOD ; à Kenia, ; en 2022")


def test_a_stretch_read_twice_keeps_everything_it_leaves_out():
    reading = admit_exclusions(
        "ODA except Kenya",
        ExclusionRead(
            exclusions=[
                Exclusion(phrase="except Kenya", left_out=["Kenya"]),
                Exclusion(phrase="except Kenya", left_out=["Uganda"]),
            ]
        ),
    )
    assert [stretch.things for stretch in reading.left_out] == [("Kenya", "Uganda")]


@dataclass
class _Door:
    """The two calls the door makes, with the speculative parse of `query` under test control.

    `read` answers the reader. A parse of `hold` waits for ever. `order` records each call as
    it starts and `cancelled` each parse cancelled while it ran."""

    read: ExclusionRead | Exception
    hold: str | None = None
    parse_error: Exception | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    parsing: asyncio.Event = field(default_factory=asyncio.Event)

    async def generate_structured(self, *, prompt, system, schema):
        self.calls.append((schema.__name__, prompt))
        if schema is ExclusionRead:
            # The parse starts first in the event loop only when the door starts both.
            await self.parsing.wait()
            if isinstance(self.read, Exception):
                raise self.read
            return self.read, {"input_tokens": 5, "output_tokens": 1, "cached_tokens": 0}
        self.parsing.set()
        if prompt == self.hold:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.calls.append(("cancelled", prompt))
                self.cancelled.append(prompt)
                raise
        if self.parse_error is not None:
            raise self.parse_error
        await asyncio.sleep(0)
        return Extraction(kind="resolution", variables=["ODA"]), {
            "input_tokens": 7,
            "output_tokens": 2,
            "cached_tokens": 0,
        }


def _door_read(query: str, door: _Door) -> raw_text._DoorRead:
    async def run() -> raw_text._DoorRead:
        return await raw_text._read_and_extract(query, llm=door, deadline=_far())

    return asyncio.run(run())


def _far() -> float:
    """A deadline that ends a door which waits for the reader before it starts the parse."""
    return time.monotonic() + 30


_HIT_QUERY = "ODA to Kenya excluding grants in 2022"
_BLANKED = "ODA to Kenya ; in 2022"


def test_a_hit_cancels_the_speculative_parse_and_parses_the_blanked_text():
    door = _Door(read=_read("excluding grants"), hold=_HIT_QUERY)

    read = _door_read(_HIT_QUERY, door)

    assert door.calls[-2:] == [("cancelled", _HIT_QUERY), ("Extraction", _BLANKED)]
    assert read.reading.text == _BLANKED
    assert set(read.timing) == {"extract", "exclusion_read"}


def test_a_hit_spends_the_usage_of_the_read_and_the_parse_that_counts():
    read = _door_read(_HIT_QUERY, _Door(read=_read("excluding grants"), hold=_HIT_QUERY))
    assert read.usage == {"input_tokens": 12, "output_tokens": 3, "cached_tokens": 0}


def test_a_miss_keeps_the_speculative_parse_and_starts_no_second_one():
    door = _Door(read=ExclusionRead(exclusions=[]))

    read = _door_read(_HIT_QUERY, door)

    assert sorted(door.calls) == [("ExclusionRead", _HIT_QUERY), ("Extraction", _HIT_QUERY)]
    assert read.reading.text == _HIT_QUERY
    assert read.extract_result.extraction.variables == ["ODA"]


def test_a_failed_read_fails_the_query_and_cancels_the_speculative_parse():
    door = _Door(read=LLMInfraError("503"), hold=_HIT_QUERY)

    with pytest.raises(LLMInfraError):
        _door_read(_HIT_QUERY, door)

    assert door.cancelled == [_HIT_QUERY]


def test_a_failed_speculative_parse_is_no_failure_of_the_parse_that_counts():
    door = _Door(read=_read("excluding grants"))
    original = door.generate_structured

    async def failing_first(*, prompt, system, schema):
        if schema is Extraction and prompt == _HIT_QUERY:
            door.parsing.set()
            raise LLMInfraError("503")
        return await original(prompt=prompt, system=system, schema=schema)

    door.generate_structured = failing_first  # ty: ignore[invalid-assignment]

    assert _door_read(_HIT_QUERY, door).reading.text == _BLANKED


def test_a_failed_extraction_on_a_miss_fails_the_query():
    door = _Door(read=ExclusionRead(exclusions=[]), parse_error=LLMInfraError("503"))
    with pytest.raises(LLMInfraError):
        _door_read(_HIT_QUERY, door)


class _UsageMeta:
    prompt_token_count = 11
    candidates_token_count = 22
    cached_content_token_count = 0


class _CountingModels:
    """Answers the reader from `read` and extraction with one variable, recording each prompt
    by the call it answers."""

    def __init__(self, read: ExclusionRead) -> None:
        self.read = read
        self.reader_prompts: list[str] = []
        self.parse_prompts: list[str] = []

    async def generate_content(self, *, model, contents, config):
        if config.system_instruction == EXCLUSION_READ_SYSTEM_PROMPT:
            self.reader_prompts.append(contents)
            text = self.read.model_dump_json()
        else:
            self.parse_prompts.append(contents)
            text = Extraction(kind="resolution", variables=["ODA"]).model_dump_json()
        return SimpleNamespace(text=text, usage_metadata=_UsageMeta())


def _stored_llm(monkeypatch, read: ExclusionRead) -> tuple[LLM, _CountingModels]:
    models = _CountingModels(read)
    monkeypatch.setattr(
        llm_module, "_get_client", lambda: SimpleNamespace(aio=SimpleNamespace(models=models))
    )
    return LLM(extraction_store=MemoryExtractionStore(max_bytes=1024 * 1024)), models


@pytest.mark.parametrize(
    ("read", "parsed"),
    [
        pytest.param(ExclusionRead(exclusions=[]), _HIT_QUERY, id="miss"),
        pytest.param(_read("excluding grants"), _BLANKED, id="hit"),
    ],
)
def test_a_repeated_query_makes_no_reader_call_and_no_speculative_parse(monkeypatch, read, parsed):
    llm, models = _stored_llm(monkeypatch, read)

    async def run() -> tuple[raw_text._DoorRead, raw_text._DoorRead]:
        first = await raw_text._read_and_extract(_HIT_QUERY, llm=llm, deadline=_far())
        reader_calls, parse_calls = len(models.reader_prompts), len(models.parse_prompts)
        second = await raw_text._read_and_extract(_HIT_QUERY, llm=llm, deadline=_far())
        assert (len(models.reader_prompts), len(models.parse_prompts)) == (
            reader_calls,
            parse_calls,
        )
        return first, second

    first, second = asyncio.run(run())

    assert models.reader_prompts == [_HIT_QUERY]
    assert parsed in models.parse_prompts
    assert second.reading == first.reading
    assert second.timing["exclusion_read"] == 0


def test_a_stored_read_with_no_stored_parse_runs_no_speculative_parse(monkeypatch):
    """The stored read decides the text first, so only the parse that counts runs."""
    llm, models = _stored_llm(monkeypatch, _read("excluding grants"))

    async def run() -> None:
        await raw_text._read_and_extract(_HIT_QUERY, llm=llm, deadline=_far())
        store = llm.extraction_store
        assert isinstance(store, MemoryExtractionStore)
        for key in [key for key, value in store._entries.items() if "variables" in value]:
            del store._entries[key]
        models.parse_prompts.clear()
        await raw_text._read_and_extract(_HIT_QUERY, llm=llm, deadline=_far())

    asyncio.run(run())

    assert models.parse_prompts == [_BLANKED]


def test_a_failed_read_stores_nothing(monkeypatch):
    llm, models = _stored_llm(monkeypatch, ExclusionRead(exclusions=[]))

    async def failing(*, model, contents, config):
        if config.system_instruction == EXCLUSION_READ_SYSTEM_PROMPT:
            raise RuntimeError("503")
        return await _CountingModels.generate_content(
            models, model=model, contents=contents, config=config
        )

    models.generate_content = failing  # ty: ignore[invalid-assignment]

    async def run() -> None:
        with pytest.raises(LLMInfraError):
            await raw_text._read_and_extract(_HIT_QUERY, llm=llm, deadline=_far())

    asyncio.run(run())

    store = llm.extraction_store
    assert isinstance(store, MemoryExtractionStore)
    assert not any("exclusions" in value for value in store._entries.values())


_STRETCH = "excluding humanitarian aid"


class _ScriptedLLM:
    """Answers the reader from `read`, extraction with `_E2E_QUERY` and no other reading, and
    bind with no CRS binding, recording each prompt each call took."""

    def __init__(
        self,
        read: ExclusionRead,
        *,
        kind: str = "resolution",
        variables: tuple[str, ...] = (_E2E_QUERY,),
    ) -> None:
        self._read = read
        self._kind = kind
        self._variables = list(variables)
        self.prompts: dict[str, list[str]] = {}

    async def generate_structured(self, *, prompt, system, schema):
        self.prompts.setdefault(schema.__name__, []).append(prompt)
        if schema is ExclusionRead:
            return self._read, None
        if schema.__name__ == "Extraction":
            return Extraction(
                kind=self._kind,
                variables=self._variables,
                entities=[ExtractedEntity(text="Kenya")],
            ), None
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, also_plausible=[]), None


def _resolve(query: str, llm: _ScriptedLLM):
    return offline_resolve(
        make_request(query),
        graph=_e2e_graph(),
        llm=llm,
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )


def test_an_answer_with_a_stretch_left_out_equals_the_query_without_it_and_says_so():
    """The relation every exclusion row holds to the same query with the stretch deleted: one
    answer, the phrase quoted, nothing claiming the measure leaves it out."""
    query = f"{_E2E_QUERY} to Kenya {_STRETCH}"
    llm = _ScriptedLLM(_read(_STRETCH))

    answered = _resolve(query, llm)
    plain = _resolve(f"{_E2E_QUERY} to Kenya", _ScriptedLLM(ExclusionRead(exclusions=[])))

    assert llm.prompts["Extraction"][-1] == f"{_E2E_QUERY} to Kenya ;"
    assert not any(_STRETCH in prompt for prompt in llm.prompts["_UniversalBindOutput"])
    assert answered.query_echo.raw_query == query
    assert answered.parts[0].spec.definition.spec_id == plain.parts[0].spec.definition.spec_id
    part = answered.parts[0]
    assert [
        (d.code, d.asked, d.text) for d in part.departures if d.code == "measure_not_applied"
    ] == [
        (
            "measure_not_applied",
            _STRETCH,
            f'You asked for "{_STRETCH}". The search did not apply it.',
        )
    ]
    assert [
        w.message for w in answered.diagnostics.warnings if w.code == QUERY_QUALIFIER_DROPPED
    ] == [f'The requested qualifier "{_STRETCH}" was not applied.']
    assert "without it" not in answered.rendered_sentence


def test_a_stretch_binding_repair_bound_counts_as_met(monkeypatch):
    """The route "without IMF credit" takes: repair of the elected shape's bindings reads the
    part's stretches, and one it binds raises no warning and no departure on the answer."""
    seen = []

    def bind_it(bindings, shape, *, exclusions, pinned_properties):
        seen.extend(stretch.phrase for stretch in exclusions)
        return list(bindings), frozenset({_STRETCH})

    monkeypatch.setattr(phase_election, "apply_named_exclusions", bind_it)

    answered = _resolve(f"{_E2E_QUERY} to Kenya {_STRETCH}", _ScriptedLLM(_read(_STRETCH)))

    assert seen == [_STRETCH]
    assert QUERY_QUALIFIER_DROPPED not in [w.code for w in answered.diagnostics.warnings]
    assert "measure_not_applied" not in [d.code for d in answered.parts[0].departures]


def test_the_read_has_its_own_step_on_the_envelope_and_on_no_part():
    answered = _resolve(f"{_E2E_QUERY} to Kenya", _ScriptedLLM(ExclusionRead(exclusions=[])))

    assert {"extract", "exclusion_read"} <= set(answered.diagnostics.timing_ms.by_step)
    assert "exclusion_read" not in answered.parts[0].timing_ms.by_step


@pytest.mark.parametrize(
    ("kind", "variables"),
    [
        pytest.param("catalogue", (), id="a-catalogue-question"),
        pytest.param("resolution", (), id="no-variable"),
    ],
)
def test_a_blanked_text_left_with_no_measure_refuses_and_carries_the_warning(kind, variables):
    """ "everything but humanitarian aid to Kenya" leaves "; to Kenya", which reads as a
    catalogue question: the question asked for a measure, so it refuses and says which words
    went unapplied."""
    query = "everything but humanitarian aid to Kenya"
    phrase = "but humanitarian aid"

    response = _resolve(query, _ScriptedLLM(_read(phrase), kind=kind, variables=variables))

    (part,) = response.parts
    assert part.status == "no_data"
    assert part.no_data.reason == "variable_not_resolved"
    assert [w.message for w in part.warnings if w.code == QUERY_QUALIFIER_DROPPED] == [
        f'The requested qualifier "{phrase}" was not applied.'
    ]


def test_a_query_with_nothing_left_out_sends_extraction_its_own_bytes():
    query = f"{_E2E_QUERY} to Kenya"
    llm = _ScriptedLLM(ExclusionRead(exclusions=[]))

    _resolve(query, llm)

    assert llm.prompts["Extraction"] == [query]
    assert llm.prompts["ExclusionRead"] == [query]


_NETTING = "nettingBasis"


def _shape(*properties: str) -> ShapeDraft:
    return shape_draft_from(
        shape_id="flow",
        label="flow",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        constraint_props=list(properties),
        prop_labels=dict.fromkeys(properties, "slot"),
        prop_observed_values={prop: ["Gross", "Net"] for prop in properties},
        sv_arc_facts={},
        slot_taxonomy={f"how:{prop}": ["Gross", "Net"] for prop in properties},
        place_available=True,
        value_labels=None,
        recalled=False,
    )


def _bound(*values: str) -> list[SlotBindingDraft]:
    return [
        SlotBindingDraft(
            axis="how",
            property_dcid=_NETTING,
            kind="value" if values else "unbound",
            value_dcids=list(values),
        )
    ]


@pytest.mark.parametrize(
    ("shape", "bindings", "met"),
    [
        pytest.param(_shape(_NETTING), _bound("Gross"), True, id="binds-the-other-reading"),
        pytest.param(_shape(_NETTING), _bound("Net"), False, id="binds-what-was-left-out"),
        pytest.param(_shape(_NETTING), _bound(), False, id="leaves-the-property-open"),
        pytest.param(_shape(), [], False, id="shape-lacks-the-property"),
    ],
)
def test_a_measure_axis_exclusion_is_met_only_by_an_answer_that_leaves_it_out(shape, bindings, met):
    """ "gross ODA, not net" answers gross and says nothing more. An answer that leaves "net"
    unmet still answers: it names no trace term and no refusal, and the door discloses it."""
    exclusion = LeftOut(phrase="not net", things=("net",), at=0)
    requirements = exclusion_requirements([exclusion], variable="ODA")

    judgement = judge(requirements, [(shape, bindings)])

    assert judgement.met_exclusions() == ({"not net"} if met else frozenset())
    assert judgement.terms(0) == ()
    assert judgement.departures() == ()


@pytest.mark.parametrize(
    ("phrase", "things", "variable"),
    [
        pytest.param(
            "excluding imputed multilateral aid",
            ("imputed", "multilateral", "aid"),
            "ODA",
            id="no-partition",
        ),
        pytest.param("not net fragile", ("net", "fragile"), "ODA", id="a-word-no-code-names"),
        pytest.param("sin ayuda humanitaria", ("ayuda", "humanitaria"), "ODA", id="no-code-at-all"),
        pytest.param("not net", (), "ODA", id="names-no-thing"),
    ],
)
def test_an_exclusion_an_answer_cannot_be_shown_to_meet_gives_no_requirement(
    phrase, things, variable
):
    """Binding one DAC2a aid type says nothing of another it counts, so the phrase stays
    disclosed whatever the answer binds."""
    exclusion = LeftOut(phrase=phrase, things=things, at=0)
    assert exclusion_requirements([exclusion], variable=variable) == ()
