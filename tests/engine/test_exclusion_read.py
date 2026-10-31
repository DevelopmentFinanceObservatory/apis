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
from typing import cast

import pytest

import search.engine.clients.llm as llm_module
from entity_resolution import Admission, EntityResolver
from search.engine.bind import SlotBindingDraft
from search.engine.bind_repair import apply_named_exclusions
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import LLM, SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.entry import raw_text
from search.engine.errors import LLMInfraError
from search.engine.exclusion_read import (
    EXCLUSION_READ_SYSTEM_PROMPT,
    Exclusion,
    ExclusionRead,
    LeftOut,
    QueryReading,
    admit_exclusions,
)
from search.engine.extract import ExtractedEntity, Extraction
from search.engine.extraction_store import MemoryExtractionStore
from search.engine.quantity_read import QuantityOutput
from search.engine.region import phase_election
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.region.requirements import excluded_run, judge, met_measure_phrases
from search.engine.region.state import ResolveState, exclusion_requirements
from search.engine.shape import ShapeDraft, shape_draft_from
from search.engine.stated_scope import intake_key_document, part_scope_intakes
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
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
        pytest.param(
            "bilateral ODA to Kenya, not including multilateral",
            "not including multilateral",
            id="the-provider-scope-after-bilateral",
        ),
        pytest.param(
            "ODA to Kenya not including multilateral flows",
            "not including multilateral flows",
            id="the-provider-scope-and-its-noun",
        ),
        pytest.param("not net", "not net", id="the-whole-query"),
        pytest.param("ODA to Kenya", "", id="empty"),
    ],
)
def test_a_stretch_failing_admission_leaves_the_query_byte_identical(query, phrase):
    """A query whose read admits nothing sends extraction the bytes it sends today."""
    assert admit_exclusions(query, _read(phrase, left_out=("aid",))) == QueryReading(text=query)


@pytest.mark.parametrize(
    ("query", "phrase"),
    [
        pytest.param("ODA to Kenya minus aid", "excluding aid", id="not-the-query-words"),
        pytest.param("casino aid to Kenya", "no aid", id="inside-a-word"),
    ],
)
def test_a_stretch_the_query_does_not_write_is_disclosed_unblanked(query, phrase):
    """Layer 1 keys on the query, so an answer to it never drops a stretch the read found:
    one it cannot place stays in the text and is disclosed."""
    reading = admit_exclusions(query, _read(phrase, left_out=("aid",)))

    assert reading == QueryReading(text=query, left_out=(LeftOut(phrase, ("aid",), None, phrase),))


def test_a_stretch_the_query_does_not_write_is_judged_without_its_scope_words():
    """A stretch the query does not write splits as a written one does, so the applied scope
    it holds is neither quoted nor judged."""
    query = "ODA to Kenya not including multilateral, nor humanitarian aid"
    reading = admit_exclusions(
        query,
        _read(
            "not including multilateral or humanitarian aid",
            left_out=("multilateral", "humanitarian aid"),
        ),
    )

    assert reading == QueryReading(
        text=query,
        left_out=(LeftOut("or humanitarian aid", ("humanitarian aid",), None, "humanitarian aid"),),
    )


_SCOPE = "not including multilateral"


@pytest.mark.parametrize(
    ("query", "left_out", "text", "stretches"),
    [
        pytest.param(
            "ODA to Kenya not including multilateral or humanitarian aid",
            ("multilateral", "humanitarian aid"),
            "ODA to Kenya not including multilateral ;",
            (LeftOut("or humanitarian aid", ("humanitarian aid",), 40, "humanitarian aid"),),
            id="a-list",
        ),
        pytest.param(
            "ODA to Kenya not including multilateral or humanitarian aid",
            ("humanitarian", "aid"),
            "ODA to Kenya not including multilateral ;",
            (LeftOut("or humanitarian aid", ("humanitarian", "aid"), 40, "humanitarian aid"),),
            id="a-list-read-word-by-word",
        ),
        pytest.param(
            "ODA to Kenya not including multilateral or humanitarian aid",
            ("multilateral or humanitarian aid",),
            "ODA to Kenya not including multilateral ;",
            (LeftOut("or humanitarian aid", ("humanitarian", "aid"), 40, "humanitarian aid"),),
            id="a-list-read-as-one-item",
        ),
        pytest.param(
            "ODA to Kenya, not including multilateral flows or humanitarian aid",
            ("multilateral flows", "humanitarian aid"),
            "ODA to Kenya, not including multilateral flows ;",
            (LeftOut("or humanitarian aid", ("humanitarian aid",), 47, "humanitarian aid"),),
            id="a-list-after-the-scope-noun",
        ),
        pytest.param(
            "ODA to Kenya not including multilateral and without IMF credit",
            ("multilateral", "IMF credit"),
            "ODA to Kenya not including multilateral ;",
            (LeftOut("without IMF credit", ("IMF credit",), 40, "without IMF credit"),),
            id="an-own-cue",
        ),
        pytest.param(
            "ODA to Kenya not including multilateral ODA or net ODA",
            ("multilateral ODA", "net ODA"),
            "ODA to Kenya not including multilateral ODA ;",
            (LeftOut("or net ODA", ("net ODA",), 44, "net ODA"),),
            id="a-list-after-the-scope-measure",
        ),
        pytest.param(
            "ODA to Kenya excluding debt relief and not including multilateral or loans",
            ("debt relief", "multilateral", "loans"),
            "ODA to Kenya ; and not including multilateral ;",
            (
                LeftOut("excluding debt relief", ("debt relief",), 13, "excluding debt relief"),
                LeftOut("or loans", ("loans",), 46, "loans"),
            ),
            id="two-spans",
        ),
        pytest.param(
            "ODA to Kenya excluding debt relief and not including multilateral, loans",
            ("debt relief", "multilateral", "loans"),
            "ODA to Kenya ; and not including multilateral, ;",
            (
                LeftOut("excluding debt relief", ("debt relief",), 13, "excluding debt relief"),
                LeftOut("loans", ("loans",), 47, "loans"),
            ),
            id="a-coordinator-before-the-scope",
        ),
    ],
)
def test_a_stretch_stating_the_provider_scope_blanks_only_what_outlives_it(
    query, left_out, text, stretches
):
    """The scope phrase stays for its reader, so the part still reads the bilateral provider
    scope, and no stretch quotes it. A stretch after it is quoted from its own cue, else as
    the words its blank removed, and leaves out only the items those words hold."""
    reading = admit_exclusions(query, _read(query[13:].lstrip(", "), left_out=left_out))

    assert (reading.text, reading.left_out) == (text, stretches)
    assert all(_SCOPE not in stretch.phrase for stretch in reading.left_out)
    # The scope reads the same words, wherever a blank before them moved them.
    (before, after) = (
        [entry["option_id"] for entry in intake_key_document(intake)["entries"]]
        for (intake,) in (
            part_scope_intakes(message=text, part_texts=[text]) for text in (query, reading.text)
        )
    )
    # Blanking "net ODA" takes its netting basis with it; the provider scope stays.
    assert "provider:bilateral" in before
    assert "provider:bilateral" in after
    assert [entry for entry in before if entry in after] == after


@pytest.mark.parametrize(
    ("query", "left_out", "stretches"),
    [
        pytest.param(
            "ODA to Kenya excluding debt relief and multilateral or loans",
            ("debt relief", "multilateral", "loans"),
            [("excluding debt relief and multilateral or loans", ";")],
            id="one-cue",
        ),
        pytest.param(
            "ODA from France except to Kenya and Uganda",
            ("Kenya", "Uganda"),
            [("except to Kenya and Uganda", ";")],
            id="one-cue-two-places",
        ),
        pytest.param(
            "ODA to Kenya excluding debt relief and without IMF credit",
            ("debt relief", "IMF credit"),
            [("excluding debt relief", ";"), ("without IMF credit", ";")],
            id="two-cues",
        ),
        pytest.param(
            "ODA to Kenya excluding debt relief and all humanitarian aid",
            ("debt relief", "humanitarian aid"),
            [("excluding debt relief and all humanitarian aid", ";")],
            id="a-word-that-is-no-cue",
        ),
    ],
)
def test_items_under_one_cue_share_a_stretch(query, left_out, stretches):
    """A read phrase splits where an item has a cue of its own, a negation cue before the
    first word it leaves out; items under one cue are one stretch."""
    reading = admit_exclusions(query, _read(query[query.index(" e") + 1 :], left_out=left_out))

    assert [(s.phrase, reading.text[s.at]) for s in reading.left_out if s.at is not None] == (
        stretches
    )
    assert all(stretch.phrase == stretch.shown for stretch in reading.left_out)


@pytest.mark.parametrize(
    ("query", "left_out", "runs"),
    [
        pytest.param(
            "ODA to Kenya not including multilateral ODA or net ODA",
            ("multilateral ODA", "net ODA"),
            {("net", "nettingBasis")},
            id="net-after-the-scope",
        ),
        pytest.param(
            "ODA to Kenya not including multilateral flows or net flows",
            ("multilateral flows", "net flows"),
            set(),
            id="a-word-no-code-names-after-the-scope",
        ),
        pytest.param(
            "ODA to Kenya not including multilateral aid or humanitarian aid",
            ("multilateral aid", "humanitarian aid"),
            {("humanitarian aid", "crsPurpose")},
            id="humanitarian-after-the-scope",
        ),
        pytest.param(
            "ODA to Kenya not including multilateral grants or loans",
            ("multilateral grants", "loans"),
            set(),
            id="loans-names-no-code",
        ),
    ],
)
def test_a_stretch_is_judged_on_the_words_its_blank_removed(query, left_out, runs):
    """The applied scope a stretch is quoted beside judges nothing; its own words do."""
    reading = admit_exclusions(query, _read(query[13:], left_out=left_out))
    (requirements,) = exclusion_requirements([list(reading.left_out)], variables=["ODA"])

    assert {
        (excluded_run(r.rule), getattr(r.rule, "property_dcid", None) or r.rule.code.property_dcid)
        for r in requirements
    } == runs


@pytest.mark.parametrize(
    "left_out",
    [
        ("multilateral", "humanitarian aid"),
        ("humanitarian", "aid"),
        ("multilateral or humanitarian aid",),
    ],
)
def test_a_stretch_after_the_provider_scope_binds_nothing_it_leaves_out(left_out):
    """ "humanitarian aid" names CRS purpose 700, which the stretch leaves out, so repair
    binds nothing for it, however the read splits its items."""
    query = "ODA to Kenya not including multilateral or humanitarian aid"
    reading = admit_exclusions(
        query, _read("not including multilateral or humanitarian aid", left_out=left_out)
    )
    shape = shape_draft_from(
        shape_id="crs",
        label="ODA",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        constraint_props=["crsPurpose"],
        prop_labels={"crsPurpose": "purpose"},
        prop_observed_values={"crsPurpose": ["CRSPurposeCode700"]},
        sv_arc_facts={},
        slot_taxonomy={"how:crsPurpose": ["CRSPurposeCode700"]},
        place_available=True,
        value_labels={"CRSPurposeCode700": "Humanitarian aid"},
        recalled=False,
    )
    slot = SlotBindingDraft(axis="how", property_dcid="crsPurpose", kind="unbound")

    assert apply_named_exclusions([slot], shape, exclusions=reading.left_out) == ([slot], ())


def test_an_empty_read_keeps_a_query_outside_nfc_byte_identical():
    query = unicodedata.normalize("NFD", "AOD à Kenia")
    assert admit_exclusions(query, ExclusionRead(exclusions=[])).text == query


def test_a_stretch_is_found_as_layer_1_keys_the_query_and_blanked_in_its_own_bytes():
    """NFKC and collapsed whitespace find the stretch; the text keeps the query's own
    characters around each blank."""
    query = unicodedata.normalize("NFD", "AOD sin  ayuda à Kenia,\u00a0sin ayuda en 2022")
    reading = admit_exclusions(query, _read("sin ayuda"))
    assert reading.text == unicodedata.normalize("NFD", "AOD ; à Kenia,\u00a0; en 2022")
    assert [reading.text[stretch.at] for stretch in reading.left_out if stretch.at] == [";"]


def test_a_stretch_in_a_query_nfkc_cannot_map_back_is_blanked_in_its_nfc_text():
    """Conjoining jamo compose across starters, so no NFKC character maps back alone; the
    stretch is found and blanked in the query's NFC text."""
    query = "ODA to \u1112\u1161\u11ab\u1100\u116e\u11a8 excluding grants"
    reading = admit_exclusions(query, _read("excluding grants"))

    assert reading.text == unicodedata.normalize("NFC", "ODA to \ud55c\uad6d ;")
    assert [(stretch.phrase, stretch.at) for stretch in reading.left_out] == [
        ("excluding grants", len(reading.text) - 1)
    ]


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
    read_after_parse: bool = False
    calls: list[tuple[str, str]] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    parsing: asyncio.Event = field(default_factory=asyncio.Event)
    parsed: asyncio.Event = field(default_factory=asyncio.Event)

    async def generate_structured(self, *, prompt, system, schema):
        self.calls.append((schema.__name__, prompt))
        if schema is ExclusionRead:
            # The parse starts first in the event loop only when the door starts both.
            await (self.parsed if self.read_after_parse else self.parsing).wait()
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
        self.parsed.set()
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


def test_a_hit_parses_the_blanked_text_and_cancels_a_speculative_parse_still_running():
    door = _Door(read=_read("excluding grants"), hold=_HIT_QUERY)

    read = _door_read(_HIT_QUERY, door)

    # The speculative parse is cancelled before the parse of the blanked text starts.
    assert door.calls[-2:] == [("cancelled", _HIT_QUERY), ("Extraction", _BLANKED)]
    assert read.reading.text == _BLANKED
    assert read.extract_result.extraction.variables == ["ODA"]
    assert set(read.timing) == {"extract", "exclusion_read"}
    # The cancelled parse sent its request and no response reports its tokens.
    assert read.usage == {
        "input_tokens": 12,
        "output_tokens": 3,
        "cached_tokens": 0,
        "uncounted_calls": 1,
    }


def test_a_hit_counts_the_tokens_of_a_speculative_parse_that_finished():
    """The discarded parse cost what it cost: its usage reaches `diagnostics.llm_usage`."""
    read = _door_read(_HIT_QUERY, _Door(read=_read("excluding grants"), read_after_parse=True))

    assert read.reading.text == _BLANKED
    assert read.usage == {"input_tokens": 19, "output_tokens": 5, "cached_tokens": 0}


def test_a_read_kept_unblanked_keeps_the_speculative_parse():
    door = _Door(read=_read("excluding loans"))

    read = _door_read(_HIT_QUERY, door)

    assert sorted(door.calls) == [("ExclusionRead", _HIT_QUERY), ("Extraction", _HIT_QUERY)]
    assert read.reading.left_out == (
        LeftOut("excluding loans", ("loans",), None, "excluding loans"),
    )


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


@pytest.mark.parametrize(
    ("value", "met"),
    [
        pytest.param("CRSTypeOfFinance110", True, id="the-answer-binds-it"),
        pytest.param("CRSTypeOfFinance421", False, id="the-answer-binds-another"),
    ],
)
def test_a_stretch_binding_repair_bound_is_met_by_an_answer_binding_it(monkeypatch, value, met):
    """The route "without IMF credit" takes: repair of the elected shape's bindings reads the
    part's stretches, and the value it bound is judged on the served answer, which leaves the
    stretch out only when it still binds that value."""
    seen = []

    def bind_it(bindings, shape, *, exclusions, pinned_properties):
        seen.extend(stretch.phrase for stretch in exclusions)
        return list(bindings), ((_STRETCH, "crsTypeOfFinance", value),)

    monkeypatch.setattr(phase_election, "apply_named_exclusions", bind_it)

    answered = _resolve(f"{_E2E_QUERY} to Kenya {_STRETCH}", _ScriptedLLM(_read(_STRETCH)))

    assert seen == [_STRETCH]
    warned = QUERY_QUALIFIER_DROPPED in [w.code for w in answered.diagnostics.warnings]
    departed = "measure_not_applied" in [d.code for d in answered.parts[0].departures]
    assert (warned, departed) == (not met, not met)


def test_the_read_has_its_own_step_on_the_envelope_and_on_no_part():
    answered = _resolve(f"{_E2E_QUERY} to Kenya", _ScriptedLLM(ExclusionRead(exclusions=[])))

    assert {"extract", "exclusion_read"} <= set(answered.diagnostics.timing_ms.by_step)
    assert "exclusion_read" not in answered.parts[0].timing_ms.by_step


@pytest.mark.parametrize(
    ("kind", "answer"),
    [
        pytest.param("catalogue", "catalogue", id="catalogue"),
        pytest.param("out_of_scope", "declined", id="out-of-scope"),
        pytest.param("requires_analysis", "declined", id="requires-analysis"),
    ],
)
def test_a_disposition_keeps_its_answer_and_discloses_the_stretch(kind, answer):
    """A question the blanked text reads as a catalogue, out-of-scope or requires-analysis
    one answers so, and says which words it did not apply."""
    query = "which datasets cover aid to Kenya, excluding the humanitarian ones"
    phrase = "excluding the humanitarian ones"

    response = _resolve(query, _ScriptedLLM(_read(phrase), kind=kind, variables=()))

    assert response.kind == answer
    assert [
        w.message for w in response.diagnostics.warnings if w.code == QUERY_QUALIFIER_DROPPED
    ] == [f'The requested qualifier "{phrase}" was not applied.']


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
    exclusion = LeftOut(phrase="not net", things=("net",), at=0, shown="not net")
    (requirements,) = exclusion_requirements([[exclusion]], variables=["ODA"])

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
    exclusion = LeftOut(phrase=phrase, things=things, at=0, shown=phrase)
    assert exclusion_requirements([[exclusion]], variables=[variable]) == [()]


def _named_state(shape: ShapeDraft) -> ResolveState:
    state = ResolveState(
        variable="external debt",
        entities=[],
        date_request=None,
        detect_query="external debt",
        role_query="external debt",
        pac=True,
        graph=cast(EngineGraphClient, None),
        resolver=cast(EntityResolver, None),
        admission=cast(Admission, None),
        llm=cast(SupportsLLM, None),
        sdmx=cast(EngineSdmxClient, None),
        deadline=None,
        pre_resolved=None,
        variable_index=0,
        emit=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        unit_text=None,
        materialise_graph=None,
        materialise_sdmx=None,
        timing={},
        warnings=[],
        coverage=empty_coverage_snapshot(),
    )
    state.shape_draft = shape
    state.require_named_exclusion("without IMF credit", property_dcid=_NETTING, value="Gross")
    return state


@pytest.mark.parametrize(
    ("bindings", "met"),
    [
        pytest.param(_bound("Gross"), True, id="binds-the-repaired-value"),
        pytest.param(_bound("Net"), False, id="binds-another"),
        pytest.param(_bound(), False, id="never-bound-it"),
    ],
)
def test_a_value_repair_bound_is_judged_on_each_served_answer(bindings, met):
    """Only an answer that still binds the value binding repair bound leaves its stretch out.
    The requirement departs and refuses nothing."""
    state = _named_state(_shape(_NETTING))

    judgement = judge(state.requirements, [(_shape(_NETTING), bindings)])

    assert judgement.met_exclusions() == ({"without IMF credit"} if met else frozenset())
    assert (judgement.terms(0), judgement.departures()) == ((), ())


def test_unbinding_a_repaired_value_drops_what_it_left_out():
    state = _named_state(_shape(_NETTING))

    state.forget_bound({_NETTING})

    assert state.requirements == ()


def _gdp_shape(qualifier: str | None) -> ShapeDraft:
    return shape_draft_from(
        shape_id=f"gdp_{qualifier}",
        label="GDP",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=qualifier,
        meas_denom_dcid=None,
        constraint_props=[],
        prop_labels={},
        prop_observed_values={},
        sv_arc_facts={},
        slot_taxonomy={},
        place_available=True,
        value_labels=None,
        recalled=False,
    )


@pytest.mark.parametrize(
    ("qualifier", "served", "met"),
    [
        pytest.param("Nominal", None, True, id="current-prices"),
        pytest.param("RealValue", None, False, id="constant-prices"),
        pytest.param(None, None, False, id="states-no-price-basis"),
        pytest.param(None, "current", True, id="served-at-current-prices"),
        pytest.param(None, "constant", False, id="served-at-constant-prices"),
    ],
)
def test_a_shape_axis_exclusion_is_met_by_an_answer_that_does_not_read_it(qualifier, served, met):
    """ "GDP of Kenya not in constant prices" answered in current prices leaves the phrase out:
    the shape or the unit request it is served under states the price basis. A shape that
    states none, served under no unit request, sums over it, so it leaves nothing out and the
    part discloses the phrase."""
    exclusion = LeftOut(
        phrase="not in constant prices",
        things=("constant", "prices"),
        at=0,
        shown="not in constant prices",
    )
    (requirements,) = exclusion_requirements([[exclusion]], variables=["GDP"])

    judgement = judge(requirements, [(_gdp_shape(qualifier), [])], served=[served])

    assert judgement.met_exclusions() == ({"not in constant prices"} if met else frozenset())
    assert met_measure_phrases(judgement) == judgement.met_exclusions()
    assert (judgement.terms(0), judgement.departures()) == ((), ())


_MODALITY, _PURPOSE = "crsCooperationModality", "crsPurpose"


def _crs_shape(*properties: str) -> ShapeDraft:
    return shape_draft_from(
        shape_id="crs",
        label="CRS",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        constraint_props=list(properties),
        prop_labels=dict.fromkeys(properties, "slot"),
        prop_observed_values={},
        sv_arc_facts={},
        slot_taxonomy={},
        place_available=True,
        value_labels=None,
        recalled=False,
    )


def _crs_bound(**values: str) -> list[SlotBindingDraft]:
    return [
        SlotBindingDraft(axis="how", property_dcid=prop, kind="value", value_dcids=[value])
        for prop, value in values.items()
    ]


@pytest.mark.parametrize(
    ("shape", "bindings", "met"),
    [
        pytest.param(
            _crs_shape(_MODALITY, _PURPOSE),
            _crs_bound(
                crsCooperationModality="CRSCooperationModalityB01", crsPurpose="CRSPurposeCode110"
            ),
            True,
            id="leaves-out-both-systems",
        ),
        pytest.param(
            _crs_shape(_PURPOSE),
            _crs_bound(crsPurpose="CRSPurposeCode110"),
            False,
            id="sums-over-the-absent-system",
        ),
        pytest.param(
            _crs_shape(_MODALITY, _PURPOSE),
            _crs_bound(crsPurpose="CRSPurposeCode110"),
            False,
            id="leaves-one-system-open",
        ),
    ],
)
def test_a_run_naming_two_systems_is_left_out_only_on_every_property(shape, bindings, met):
    """ "budget support" is a modality and a purpose. An answer summing over the modality
    counts budget support there, so only one leaving it out on both leaves the phrase out."""
    exclusion = LeftOut(
        phrase="excluding budget support",
        things=("budget support",),
        at=0,
        shown="excluding budget support",
    )
    (requirements,) = exclusion_requirements([[exclusion]], variables=["ODA"])

    judgement = judge(requirements, [(shape, bindings)])

    assert judgement.met_exclusions() == ({"excluding budget support"} if met else frozenset())


@pytest.mark.parametrize(
    ("met", "part", "envelope"),
    [
        pytest.param((False, False), [[], ["w"]], ["w"], id="only-the-part-it-qualifies"),
        pytest.param((False, True), [[], []], [], id="that-part-leaves-it-out"),
    ],
)
def test_a_stretch_warns_only_on_the_parts_it_qualifies(met, part, envelope):
    """ "ODA to Kenya and GDP of Kenya excluding humanitarian aid" qualifies GDP: the ODA part
    states nothing of it, and the GDP part warns unless its answer leaves it out."""
    stretch = LeftOut(
        "excluding humanitarian aid", ("humanitarian aid",), 30, "excluding humanitarian aid"
    )
    prep = SimpleNamespace(
        exclusion_warnings=("w",), exclusions=(stretch,), exclusion_targets=((1,),)
    )
    regions = [
        SimpleNamespace(
            specs=("spec",), met_exclusions={stretch.phrase} if left_out else frozenset()
        )
        for left_out in met
    ]
    unmet = raw_text._ExtractionPrep._unmet_exclusion_warnings

    assert [unmet(prep, [region], parts=[i]) for i, region in enumerate(regions)] == part
    assert unmet(prep, regions, parts=range(2)) == envelope
