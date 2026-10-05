"""End-to-end tests for how an aid question elects its dataset.

The packaged index holds two datasets, each on its own five-tuple: the CRS on `amount`,
and the opt-in imputed multilateral sectors table on `attributedAmount`, whose every row
carries `dac2aAidType=DAC2aAidType106` and `crsTypeOfFlow=CRSTypeOfFlow10`, the slots the
five-tuple pins. The election implies both, so an imputed part's identity leaves them
unbound. The recall sample of every question here holds one member of each. The
acceptance table replays the recorded extraction, mention reading and bind responses in
`tests/fixtures/llm_responses.json`; the other tests script the model.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import date
from unittest.mock import patch

import pytest

from code_index.members import canonical_identity, elected_pins
from code_index.shared import get_index
from datasets_api.adapter import resolve_definition_members
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.core import resolve_async, resolve_events
from search.engine.dataset_mention import MentionReading, MentionReadings
from search.engine.errors import LLMInfraError
from search.engine.events import DoneEvent, ResponseEvent
from search.engine.extract import ExtractedEntity, Extraction
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.response_cache import ResponseCache
from search.engine.retrieval.bound_values import identity_dataset, identity_query
from search.engine.spec_id import mint_identity
from search.models import (
    DatasetResubmitInput,
    DefinitePart,
    NoDataPart,
    Operation,
    ParsedInput,
    ParsedPart,
    ResolveRequest,
    ResolveResponse,
    ResubmitParsedEntry,
    Spec,
)
from series_coverage.schema import FacetMetadata, RoleNames
from statvar_members import MemberEnumerator, member_digest
from tests.code_index.aid_identity import (
    AID_TYPE,
    CRS_MEMBER_COUNT,
    CRS_PROVENANCE,
    EDUCATION_AID_SPEC_ID,
    IMPUTED_MEMBER_COUNT,
    IMPUTED_MULTILATERAL,
    IMPUTED_PROVENANCE,
    education_aid_identity,
    imputed_education_aid_identity,
)
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine._fake_coverage import patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve
from tests.fixtures import PINNED_DATE, FakeGraph, FakeLLM

_KEN = "country/KEN"
_FRA = "country/FRA"
_CRS_MEMBER = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode11330_Nominal"
)
# The imputed member for the same purpose, read from the packaged index so its dcid
# follows the export.
_IMPUTED_MEMBER = next(
    row.dcid
    for row in get_index().statvars
    if IMPUTED_PROVENANCE in row.provenances
    and row.constraint_properties.get("crsPurpose") == "CRSPurposeCode11110"
)
_MEMBERS = (_CRS_MEMBER, _IMPUTED_MEMBER)
# An education commitments member: a shape the CRS publishes and the imputed dataset does not.
_COMMITMENTS_MEMBER = next(
    row.dcid
    for row in get_index().statvars
    if row.population_type == "FinancialObligation"
    and row.constraint_properties.get("crsPurpose") in get_index().descendants("CRSPurposeCode110")
    and CRS_PROVENANCE in row.provenances
)
_CRS_LABEL = "OECD Creditor Reporting System (CRS)"
_IMPUTED_LABEL = "ONE Imputed Multilateral Sectors"
# Every CRS member of the aid five-tuple under official development assistance, gross
# disbursements, with the purpose open.
_PLAIN_AID_MEMBER_COUNT = 3687
# The CRS education members under every flow, which a part with no bound flow resolves.
_PLAIN_CRS_EDUCATION_MEMBERS = 434


def _arcs(dcid: str) -> dict:
    row = next(row for row in get_index().statvars if row.dcid == dcid)
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": dcid}]},
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "measurementQualifier": {"nodes": [{"dcid": row.measurement_qualifier}]},
        "observationProperties": {"nodes": [{"dcid": p} for p in row.observation_properties]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }


def _graph(
    *variables: str,
    members: tuple[str, ...] = _MEMBERS,
    provenance_labels: tuple[str, str] = (_CRS_LABEL, _IMPUTED_LABEL),
) -> FakeGraph:
    """A graph whose recall returns `members` in order for each variable: one CRS member
    and one imputed member unless told otherwise."""
    crs_label, imputed_label = provenance_labels
    return FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            _FRA: {"label": "France", "type": "Country"},
            CRS_PROVENANCE: {"label": crs_label},
            IMPUTED_PROVENANCE: {"label": imputed_label},
            **{
                dcid: {"label": dcid, "type": "StatisticalVariable", "arcs": _arcs(dcid)}
                for dcid in members
            },
        },
        detect={
            variable: {
                "svs": list(members),
                "cosine_scores": [0.9 - 0.05 * rank for rank in range(len(members))],
            }
            for variable in variables
        },
        resolve={"Kenya": _KEN},
        obs={},
    )


@pytest.fixture(autouse=True)
def _coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every member holds series from France to Kenya under its own dataset, and the CRS
    disbursements member holds one under the imputed provenance as well."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            (_CRS_MEMBER, "crs", _FRA, _KEN, [2019, 2020, 2021]),
            (_CRS_MEMBER, "imputed", _FRA, _KEN, [2019, 2020, 2021]),
            (_IMPUTED_MEMBER, "imputed", _FRA, _KEN, [2019, 2020, 2021]),
            (_COMMITMENTS_MEMBER, "crs", _FRA, _KEN, [2019, 2020, 2021]),
        ],
        role_names=dict.fromkeys((*_MEMBERS, _COMMITMENTS_MEMBER), names),
        facet_metadata={
            "crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD"),
            "imputed": FacetMetadata(provenance_id=IMPUTED_PROVENANCE, unit="USD"),
        },
    )


class _RecordedLLM:
    """Replay the recorded responses, keep each bind prompt by its variable, and keep the
    mention readings and the number of calls per schema."""

    def __init__(self) -> None:
        self._llm = FakeLLM()
        self.bind_prompts: dict[str, str] = {}
        self.readings: list[tuple[str, int | None]] = []
        self.calls: Counter[str] = Counter()

    async def generate_structured(self, *, prompt, system, schema):
        self.calls[schema.__name__] += 1
        if schema is MentionReadings:
            readings, usage = await self._llm.generate_structured(
                prompt=prompt, system=system, schema=schema
            )
            self.readings = [(r.relation, r.variable) for r in readings.readings]
            return readings, usage
        if schema is _UniversalBindOutput:
            variable = next(
                line.removeprefix("Variable phrase: ")
                for line in prompt.splitlines()
                if line.startswith("Variable phrase: ")
            )
            self.bind_prompts[variable] = prompt
        return await self._llm.generate_structured(prompt=prompt, system=system, schema=schema)


def _resolve(query: str, *variables: str) -> tuple[ResolveResponse, _RecordedLLM]:
    llm = _RecordedLLM()
    response = offline_resolve(
        make_request(query),
        graph=_graph(*variables),
        llm=llm,
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    return response, llm


def _specs(response: ResolveResponse) -> list[Spec]:
    assert all(isinstance(part, DefinitePart) for part in response.parts), response.parts
    return [part.spec for part in response.parts]


def _bound(spec: Spec) -> dict[str, str]:
    """The spec's bound slots, by constraint property or by axis where a slot has none."""
    return {
        (
            slot.key.property.dcid if slot.key.property else slot.key.axis
        ): slot.binding.value.ref.dcid
        for slot in spec.resolved.slots
        if slot.binding.kind == "value" and slot.binding.value.ref is not None
    }


def _pinned(spec: Spec) -> dict[str, str]:
    """Each slot the spec binds to a pin of the dataset elected for its family
    (`code_index.members.elected_pins`), mapped to the pinned value."""
    bound = _bound(spec)
    pins = elected_pins(get_index(), identity_query(spec.definition.identity))
    return {prop: value for prop, value in pins.items() if bound.get(prop) == value}


def _sources(spec: Spec) -> list[str]:
    return [source.dcid for source in spec.resolved.trace.resolved_sources]


def _spec_id(identity: dict) -> str:
    spec_id, _definition = mint_identity(identity)
    return spec_id


def _mint_family(spec: Spec) -> set[str]:
    """The members the Datasets mint resolves for the spec's identity, as bare dcids."""
    graph = FakeMintGraphReader(arcs={})
    resolved = resolve_definition_members(
        spec.definition.identity.model_dump(mode="json", exclude_none=True),
        spec_id=spec.spec_id,
        graph=graph,
        enumerator=MemberEnumerator(graph=graph),
        code_index=get_index(),
    )
    return {shape.stat_var.removeprefix("dcid:") for shape in resolved.shapes}


def _assert_search_family_is_the_mints(spec: Spec, *, member_count: int) -> None:
    family = _mint_family(spec)
    assert len(family) == member_count
    assert spec.resolved.shape.member_count == member_count
    assert spec.resolved.shape.member_digest == member_digest(family)


def _plain_aid_identity() -> dict:
    """The education-aid identity with the purpose open: every sector's aid to Kenya."""
    identity = education_aid_identity()
    purpose = next(slot for slot in identity["slots"] if slot["property"] == "crsPurpose")
    purpose["binding"] = {"state": "unbound"}
    return identity


def _query_qualifier_warnings(response: ResolveResponse) -> list[str]:
    return [
        warning.message
        for warning in [
            *response.diagnostics.warnings,
            *(warning for part in response.parts for warning in part.warnings),
        ]
        if warning.code == QUERY_QUALIFIER_DROPPED
    ]


_E = "education aid"
_IM = "imputed multilateral"
_IM_E = "imputed multilateral education aid"
_CRS_E = "CRS education aid"
_ES = "ayuda a la educación a Kenia"
_FR = "aide à l'éducation au Kenya"

# What a part answers: the CRS education family, the imputed one, or the CRS one with the
# source bound.
_CRS = "crs"
_IMPUTED = "imputed"
_CRS_SOURCE = "crs-source"


def _assert_part(spec: Spec, family: str) -> None:
    if family == _IMPUTED:
        # The identity dev mints, and the canonical form of the same identity posted with
        # the table's invariants unbound.
        posted_unbound = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
        for slot in posted_unbound["slots"]:
            if slot["property"] in ("crsTypeOfFlow", AID_TYPE):
                slot["binding"] = {"state": "unbound"}
        assert spec.spec_id == _spec_id(imputed_education_aid_identity(source=IMPUTED_PROVENANCE))
        assert spec.spec_id == _spec_id(dict(canonical_identity(get_index(), posted_unbound)))
        assert _bound(spec)[AID_TYPE] == IMPUTED_MULTILATERAL
        assert _bound(spec)["source"] == IMPUTED_PROVENANCE
        assert spec.resolved.shape.member_count == IMPUTED_MEMBER_COUNT
        assert _sources(spec) == [IMPUTED_PROVENANCE]
        return
    assert AID_TYPE not in _bound(spec)
    assert spec.resolved.shape.member_count == CRS_MEMBER_COUNT
    assert _sources(spec) == [CRS_PROVENANCE]
    if family == _CRS_SOURCE:
        assert spec.spec_id == _spec_id(education_aid_identity(source=CRS_PROVENANCE))
        assert _bound(spec)["source"] == CRS_PROVENANCE
    else:
        assert spec.spec_id == EDUCATION_AID_SPEC_ID
        assert "source" not in _bound(spec)


# ---------------------------------------------------------------------------
# The acceptance table: recorded extraction, recorded reading, recorded bind
# ---------------------------------------------------------------------------

# (query, readings, parts as (variable, family), dropped-qualifier warnings)
_ACCEPTANCE = [
    (f"{_E} to Kenya including {_IM}", [("also", 0)], [(_E, _CRS), (_IM_E, _IMPUTED)], []),
    (f"{_E} to Kenya as well as {_IM} aid", [("also", 0)], [(_E, _CRS), (_IM_E, _IMPUTED)], []),
    (f"{_E} to Kenya and {_IM_E}", [("only", 1)], [(_E, _CRS), (_IM_E, _IMPUTED)], []),
    (f"{_E} to Kenya counting {_IM}", [("also", 0)], [(_E, _CRS), (_IM_E, _IMPUTED)], []),
    (f"{_E} to Kenya, not excluding {_IM}", [("also", 0)], [(_E, _CRS), (_IM_E, _IMPUTED)], []),
    (f"{_E} to Kenya with and without {_IM}", [("also", 0)], [(_E, _CRS), (_IM_E, _IMPUTED)], []),
    (
        f"{_ES} incluyendo la ayuda multilateral imputada",
        [("also", 0)],
        [(_E, _CRS), (_IM_E, _IMPUTED)],
        [],
    ),
    (
        f"{_FR}, y compris l'aide multilatérale imputée",
        [("also", 0)],
        [(_E, _CRS), (_IM_E, _IMPUTED)],
        [],
    ),
    (
        f"{_ES} incluyendo {_IM}",
        [("also", 0)],
        [(f"{_E} including {_IM}", _IMPUTED), (f"{_E} including", _CRS)],
        [],
    ),
    (
        f"{_FR}, y compris {_IM}",
        [("also", 0)],
        [(f"{_E} including {_IM}", _IMPUTED), (f"{_E} including", _CRS)],
        [],
    ),
    (f"{_IM_E} to Kenya", [("only", 0)], [(_IM_E, _IMPUTED)], []),
    (f"{_E} to Kenya from {_IM} sources", [("only", 0)], [(f"{_E} {_IM}", _IMPUTED)], []),
    (f"{_E} to Kenya, nothing except {_IM}", [("only", 0)], [(f"{_E} {_IM}", _IMPUTED)], []),
    (f"{_CRS_E} to Kenya", [("only", 0)], [(_CRS_E, _CRS_SOURCE)], []),
    (
        f"CRS and {_IM_E} to Kenya",
        [("only", 0), ("only", 1)],
        [(_CRS_E, _CRS_SOURCE), (_IM_E, _IMPUTED)],
        [],
    ),
    (f"{_E} to Kenya excluding {_IM}", [("excluded", 0)], [(_E, _CRS)], []),
    (f"{_E} to Kenya without {_IM}", [("excluded", 0)], [(_E, _CRS)], []),
    (f"{_E} to Kenya not counting {_IM}", [("excluded", 0)], [(_E, _CRS)], []),
    (f"do not include {_IM} aid in {_E} to Kenya", [("excluded", 0)], [(_E, _CRS)], []),
    (f"don't include {_IM} aid in {_E} to Kenya", [("excluded", 0)], [(_E, _CRS)], []),
    (f"{_E} to Kenya, {_IM} excluded", [("excluded", 0)], [(_E, _CRS)], []),
    (f"{_E} to Kenya, not including {_IM}", [("excluded", 0)], [(_E, _CRS)], []),
    (f"{_E} to Kenya minus {_IM}", [("excluded", 0)], [(_E, _CRS)], []),
    (f"{_E} to Kenya apart from {_IM}", [("excluded", 0)], [(_E, _CRS)], []),
    (
        f"{_E} to Kenya not from the ONE Imputed Multilateral Sectors",
        [("excluded", 0)],
        [(_E, _CRS)],
        [],
    ),
    (f"{_ES} excluyendo {_IM}", [("excluded", 0)], [(_E, _CRS)], []),
    (f"{_FR} hors {_IM}", [("excluded", 0)], [(_E, _CRS)], []),
    (
        "non-CRS education aid to Kenya",
        [("excluded", 0)],
        [("non-CRS education aid", _CRS)],
        ['The requested qualifier "CRS" was not applied.'],
    ),
]


@pytest.mark.parametrize(
    ("query", "readings", "parts", "warnings"), _ACCEPTANCE, ids=[row[0] for row in _ACCEPTANCE]
)
def test_a_question_naming_a_dataset_answers_what_its_reading_asks(
    query: str, readings: list, parts: list, warnings: list
):
    response, llm = _resolve(query, *(variable for variable, _family in parts))

    assert llm.readings == readings
    assert llm.calls["MentionReadings"] == 1
    assert [part.variable_text for part in response.parts] == [variable for variable, _ in parts]
    for spec, (_variable, family) in zip(_specs(response), parts, strict=True):
        _assert_part(spec, family)
    assert _query_qualifier_warnings(response) == warnings


# The recorded extraction gives the dataset a variable naming another measure, "imputed
# multilateral student aid", and the recorded reading asks the education measure of it.
_PLUS = f"{_E} to Kenya plus {_IM}"


def test_a_reading_that_names_another_measure_than_the_mentions_variable_is_reported():
    student = f"{_IM} student aid"

    response, llm = _resolve(_PLUS, _E, student)

    assert llm.readings == [("also", 0)]
    assert [part.variable_text for part in response.parts] == [_E, student]
    plain, unbound = response.parts
    assert isinstance(plain, DefinitePart)
    _assert_part(plain.spec, _CRS)
    if isinstance(unbound, DefinitePart):
        assert AID_TYPE not in _bound(unbound.spec)
        assert _sources(unbound.spec) == [CRS_PROVENANCE]
    assert _query_qualifier_warnings(response) == [
        f'The requested qualifier "{_IM}" was not applied.'
    ]


@pytest.mark.parametrize(
    ("query", "variable", "member_count"),
    [
        ("top donors of education aid to Kenya", _E, CRS_MEMBER_COUNT),
        (f"{_E} to Kenya", _E, CRS_MEMBER_COUNT),
        ("total aid to Kenya", "aid", _PLAIN_AID_MEMBER_COUNT),
        ("top 5 donantes de ayuda a Kenia", "aid", _PLAIN_AID_MEMBER_COUNT),
    ],
)
def test_a_question_naming_no_dataset_asks_no_reading_and_answers_the_crs(
    query: str, variable: str, member_count: int
):
    response, llm = _resolve(query, variable)

    (spec,) = _specs(response)
    assert llm.calls["MentionReadings"] == 0
    assert llm.calls["Extraction"] == 1
    assert spec.resolved.shape.member_count == member_count
    assert AID_TYPE not in _bound(spec)
    assert "source" not in _bound(spec)
    plain = education_aid_identity() if variable == _E else _plain_aid_identity()
    assert spec.spec_id == _spec_id(plain)


def test_a_ranked_education_aid_question_keeps_the_plain_crs_identity_and_its_operation():
    response, _llm = _resolve("top donors of education aid to Kenya", "education aid")

    (spec,) = _specs(response)
    assert spec.spec_id == EDUCATION_AID_SPEC_ID
    assert response.parts[0].operations == [
        Operation(verb="rank", phrase="top donors", over="origin")
    ]


def test_the_bind_prompt_of_a_part_naming_no_dataset_holds_no_opt_in_shape():
    _response, llm = _resolve("top donors of education aid to Kenya", "education aid")

    prompt = llm.bind_prompts["education aid"]
    assert _CRS_MEMBER in prompt
    assert _IMPUTED_MEMBER not in prompt
    assert AID_TYPE not in prompt


def test_each_part_names_the_one_dataset_elected_for_its_family():
    response, _llm = _resolve(f"{_E} to Kenya including {_IM}", _E, _IM_E)

    plain, imputed = _specs(response)
    assert (plain.resolved.rendered_sentence or "").endswith(f" {_CRS_LABEL}.")
    assert _IMPUTED_LABEL not in plain.resolved.rendered_sentence
    assert (imputed.resolved.rendered_sentence or "").endswith(f" {_IMPUTED_LABEL}.")
    assert _CRS_LABEL not in imputed.resolved.rendered_sentence


def test_a_question_naming_the_crs_binds_the_source_and_resolves_the_mints_family():
    response, _llm = _resolve(f"{_CRS_E} to Kenya", _CRS_E)

    (spec,) = _specs(response)
    assert spec.spec_id != EDUCATION_AID_SPEC_ID
    sentence = spec.resolved.rendered_sentence or ""
    assert "to Kenya for education, 2019 to 2021. " in sentence
    assert sentence.endswith(f" {_CRS_LABEL}.")
    _assert_search_family_is_the_mints(spec, member_count=CRS_MEMBER_COUNT)


# ---------------------------------------------------------------------------
# A scripted model: what each reading and each failure does
# ---------------------------------------------------------------------------


class _ScriptedLLM:
    """Extract the given variables to Kenya, read the mentions as told, and elect the
    first shape with every slot open but those `drafted` binds. Each bind prompt is kept in
    call order."""

    def __init__(
        self,
        *variables: str,
        dropped: tuple[str, ...] = (),
        unplaced: tuple[str, ...] = (),
        readings: tuple[tuple[str, int | None], ...] | Exception = (),
        drafted: tuple[SlotBindingDraft, ...] = (),
    ) -> None:
        self._variables = list(variables)
        self._dropped = list(dropped)
        self._unplaced = list(unplaced)
        self._readings = readings
        self._drafted = list(drafted)
        self.bind_prompts: list[str] = []
        self.calls: Counter[str] = Counter()

    async def generate_structured(self, *, prompt, system, schema):
        self.calls[schema.__name__] += 1
        if schema is Extraction:
            return Extraction(
                kind="resolution",
                variables=self._variables,
                entities=[
                    ExtractedEntity(
                        text="Kenya", span="Kenya", flow_role="destination", role_span="to"
                    )
                ],
                dropped_qualifiers=self._dropped,
            ), None
        if schema is MentionReadings:
            if isinstance(self._readings, Exception):
                raise self._readings
            return MentionReadings(
                readings=[
                    MentionReading(mention=number, relation=relation, variable=variable)
                    for number, (relation, variable) in enumerate(self._readings)
                ]
            ), None
        self.bind_prompts.append(prompt)
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=self._drafted, unplaced=self._unplaced
        ), None


def _resolve_scripted(
    query: str, llm: _ScriptedLLM, *variables: str, graph: FakeGraph | None = None
) -> ResolveResponse:
    return offline_resolve(
        make_request(query),
        graph=graph if graph is not None else _graph(*variables),
        llm=llm,
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )


@pytest.mark.parametrize(
    "readings",
    [
        (("unrelated", None),),
        (),
        (("only", 7),),
        (("also", None),),
        LLMInfraError("the model is unavailable"),
        TimeoutError("the model timed out"),
        ValueError("the output did not validate"),
    ],
    ids=["unrelated", "missing", "bad-variable", "no-variable", "infra", "timeout", "invalid"],
)
@pytest.mark.parametrize(
    ("variable", "dropped"),
    [(_E, (f"including {_IM}",)), (_IM_E, ()), (f"{_E} including {_IM}", ())],
    ids=["dropped-phrase", "leading", "inside"],
)
def test_no_reading_but_only_or_also_binds_and_each_other_is_reported(
    readings, variable: str, dropped: tuple[str, ...]
):
    llm = _ScriptedLLM(variable, dropped=dropped, readings=readings)

    response = _resolve_scripted(f"{_E} to Kenya including {_IM}", llm, variable)

    (spec,) = _specs(response)
    assert AID_TYPE not in _bound(spec)
    assert "source" not in _bound(spec)
    assert spec.resolved.shape.member_count == _PLAIN_CRS_EDUCATION_MEMBERS
    assert _sources(spec) == [CRS_PROVENANCE]
    (warning,) = _query_qualifier_warnings(response)
    assert _IM in warning


@pytest.mark.parametrize("name", [_IM, f"{_IM} ODA"])
def test_an_excluded_mention_binds_nothing_and_warns_nothing_wherever_it_sits(name: str):
    """ "imputed multilateral ODA" is also the label of the aid-type code the imputed rows
    carry. The mention owns its words, so no qualifier is recognised in them on the
    default part, which has no aid-type slot."""
    for variable, dropped in [(_E, (f"excluding {name}",)), (f"{_E} excluding {name}", ())]:
        llm = _ScriptedLLM(variable, dropped=dropped, readings=(("excluded", 0),))

        response = _resolve_scripted(f"{_E} to Kenya excluding {name}", llm, variable)

        (spec,) = _specs(response)
        assert AID_TYPE not in _bound(spec)
        assert spec.resolved.shape.member_count == _PLAIN_CRS_EDUCATION_MEMBERS
        assert _query_qualifier_warnings(response) == []


def test_a_drafted_source_binding_is_discarded():
    """No draft binds the source. A part naming no dataset answers the default."""
    drafted = SlotBindingDraft(
        axis="source", property_dcid=None, kind="value", value_dcids=[IMPUTED_PROVENANCE]
    )

    response = _resolve_scripted(f"{_E} to Kenya", _ScriptedLLM(_E, drafted=(drafted,)), _E)

    (spec,) = _specs(response)
    assert AID_TYPE not in _bound(spec)
    assert "source" not in _bound(spec)
    assert _sources(spec) == [CRS_PROVENANCE]


def test_the_default_dataset_asked_for_as_well_is_one_part_with_nothing_bound():
    llm = _ScriptedLLM(_E, dropped=("including CRS",), readings=(("also", 0),))

    response = _resolve_scripted(f"{_E} to Kenya including CRS", llm, _E)

    (spec,) = _specs(response)
    assert "source" not in _bound(spec)
    assert _sources(spec) == [CRS_PROVENANCE]
    assert _query_qualifier_warnings(response) == []


def test_the_datasets_part_carries_the_operations_on_its_measure():
    response, _llm = _resolve(f"top 5 donors of {_E} to Kenya including {_IM}", _E, _IM_E)

    rank = Operation(verb="rank", phrase="top 5 donors", over="origin")
    assert [part.variable_text for part in response.parts] == [_E, _IM_E]
    assert [part.operations for part in response.parts] == [[rank], [rank]]


# ---------------------------------------------------------------------------
# A dataset asked for where it has no such series
# ---------------------------------------------------------------------------


def _commitments_graph(*variables: str) -> FakeGraph:
    return _graph(*variables, members=(_COMMITMENTS_MEMBER,))


def test_a_dataset_that_publishes_no_recalled_shape_refuses_and_names_the_mention():
    """The named dataset publishes none of the recalled shapes, so candidacy leaves the part
    nothing to elect. It refuses before any bind call, naming the mention."""
    variable = "imputed multilateral education aid commitments"
    llm = _ScriptedLLM(variable, readings=(("only", 0),))

    response = _resolve_scripted(f"{variable} to Kenya", llm, graph=_commitments_graph(variable))

    (part,) = response.parts
    assert isinstance(part, NoDataPart)
    assert part.no_data.reason == "qualifier_not_bound"
    (unbound,) = part.no_data.unbound_qualifiers
    assert unbound.disposition == "unbound"
    assert [(ref.dcid, ref.label) for ref in unbound.named] == [
        (IMPUTED_PROVENANCE, _IMPUTED_LABEL)
    ]
    assert part.no_data.flagged_match is None
    assert llm.calls["_UniversalBindOutput"] == 0


def test_an_inclusion_of_a_dataset_with_no_such_series_answers_its_part_no_data():
    measure = "education aid commitments"
    response = _resolve_scripted(
        f"{measure} to Kenya including imputed multilateral",
        _ScriptedLLM(measure, dropped=("including imputed multilateral",), readings=(("also", 0),)),
        graph=_commitments_graph(measure, f"imputed multilateral {measure}"),
    )

    plain, imputed = response.parts
    assert isinstance(plain, DefinitePart)
    assert _sources(plain.spec) == [CRS_PROVENANCE]
    assert isinstance(imputed, NoDataPart)
    assert imputed.variable_text == f"imputed multilateral {measure}"
    assert imputed.no_data.reason == "qualifier_not_bound"
    assert [q.text for q in imputed.no_data.unbound_qualifiers] == ["imputed multilateral"]


# ---------------------------------------------------------------------------
# The bind prompt, the source label and the qualifier record
# ---------------------------------------------------------------------------


def test_an_opt_in_shape_recalled_first_never_reaches_a_part_naming_no_dataset():
    llm = _ScriptedLLM(_E)

    response = _resolve_scripted(
        f"{_E} to Kenya", llm, graph=_graph(_E, members=(_IMPUTED_MEMBER, _CRS_MEMBER))
    )

    (prompt,) = llm.bind_prompts
    assert f"  label: {_CRS_MEMBER}" in prompt
    assert _IMPUTED_MEMBER not in prompt
    assert AID_TYPE not in prompt
    assert _answers(response) == [(_E, CRS_PROVENANCE, False)]


def test_the_sentence_and_the_trace_name_each_dataset_by_its_declared_label():
    graph = _graph(
        _E, _IM_E, _CRS_E, provenance_labels=("GRAPH NAME FOR CRS", "GRAPH NAME FOR IMPUTED")
    )
    default_and_imputed = _resolve_scripted(
        f"{_E} to Kenya including {_IM}",
        _ScriptedLLM(_E, dropped=(f"including {_IM}",), readings=(("also", 0),)),
        graph=graph,
    )
    named = _resolve_scripted(
        f"{_CRS_E} to Kenya", _ScriptedLLM(_CRS_E, readings=(("only", 0),)), graph=graph
    )

    plain, imputed = _specs(default_and_imputed)
    (bound,) = _specs(named)
    for spec, label in ((plain, _CRS_LABEL), (imputed, _IMPUTED_LABEL), (bound, _CRS_LABEL)):
        assert [source.label for source in spec.resolved.trace.resolved_sources] == [label]
        assert (spec.resolved.rendered_sentence or "").endswith(f" {label}.")
        assert "GRAPH NAME" not in spec.resolved.rendered_sentence


@pytest.mark.parametrize(
    ("variable", "readings", "unplaced"),
    [
        (_CRS_E, (("only", 0),), ("CRS",)),
        (f"{_E} excluding {_IM}", (("excluded", 0),), (_IM,)),
    ],
    ids=["bound-source", "excluded"],
)
def test_a_dataset_name_the_bind_model_echoes_is_no_qualifier_without_vocabulary(
    variable: str, readings, unplaced: tuple[str, ...]
):
    response = _resolve_scripted(
        f"{variable} to Kenya",
        _ScriptedLLM(variable, readings=readings, unplaced=unplaced),
        variable,
    )

    (spec,) = _specs(response)
    assert [q.text for q in spec.resolved.trace.query_qualifiers] == ["education"]


# ---------------------------------------------------------------------------
# A named dataset binds the source slot and the slots its five-tuple pins
# ---------------------------------------------------------------------------


def test_a_named_dataset_binds_the_source_and_resolves_the_mints_family():
    response = _resolve_scripted(
        f"{_IM_E} to Kenya", _ScriptedLLM(_IM_E, readings=(("only", 0),)), _IM_E
    )

    (spec,) = _specs(response)
    assert _bound(spec)["source"] == IMPUTED_PROVENANCE
    identity = spec.definition.identity
    assert identity.facets is not None
    assert identity.facets["provenance"].ref == IMPUTED_PROVENANCE
    assert identity_dataset(identity) == IMPUTED_PROVENANCE
    _assert_search_family_is_the_mints(spec, member_count=IMPUTED_MEMBER_COUNT)


_OPEN_AID_TYPE = (
    SlotBindingDraft(axis="how", property_dcid=AID_TYPE, kind="unbound"),
    SlotBindingDraft(axis="how", property_dcid=AID_TYPE, kind="absent"),
    SlotBindingDraft(
        axis="how",
        property_dcid=AID_TYPE,
        kind="unbound",
        evidence="requested_all",
        requested_all_phrase="every aid type",
    ),
    SlotBindingDraft(
        axis="how", property_dcid=AID_TYPE, kind="value", value_dcids=[IMPUTED_MULTILATERAL]
    ),
    SlotBindingDraft(axis="how", property_dcid="crsTypeOfFlow", kind="unbound"),
)


@pytest.mark.parametrize(
    "drafted",
    [(), *((draft,) for draft in _OPEN_AID_TYPE)],
    ids=["none", "unbound", "absent", "requested-all", "value", "flow-unbound"],
)
def test_every_draft_of_a_pinned_slot_ends_as_its_pinned_value(drafted):
    """Every row of the imputed five-tuple holds one aid type and one flow, so the bind
    menu offers neither slot, and whatever the model drafts on either, the identity, which
    names the table, binds the declared value, and the part mints one identity. The
    sentence states neither, because the measure label already says imputed multilateral
    ODA, once."""
    llm = _ScriptedLLM(_IM_E, dropped=("every aid type",), readings=(("only", 0),), drafted=drafted)

    response = _resolve_scripted(f"{_IM_E} to Kenya, every aid type", llm, _IM_E)

    (prompt,) = llm.bind_prompts
    assert "property_dcid=crsPurpose" in prompt
    assert f"property_dcid={AID_TYPE}" not in prompt
    assert "property_dcid=crsTypeOfFlow" not in prompt
    (spec,) = _specs(response)
    _assert_part(spec, _IMPUTED)
    assert _pinned(spec)["crsTypeOfFlow"] == "CRSTypeOfFlow10"
    sentence = spec.resolved.rendered_sentence
    assert sentence.casefold().count("imputed multilateral oda") == 1
    assert "official development assistance" not in sentence.casefold()


def test_each_pinned_slot_offers_the_menu_around_its_pin():
    """The imputed answer binds its two pinned slots to the table's invariants, and each
    slot's menu is the one the code index scopes around the bound value
    (`level_scoped_options`), less the options that serve no row. Every imputed row holds
    the invariant's value, so it alone is offered."""
    llm = _ScriptedLLM(_IM_E, dropped=(), readings=(("only", 0),))

    response = _resolve_scripted(f"{_IM_E} to Kenya", llm, _IM_E)

    (spec,) = _specs(response)
    _assert_part(spec, _IMPUTED)
    pinned = _pinned(spec)
    assert set(pinned) == {AID_TYPE, "crsTypeOfFlow"}
    menus = {
        menu.key.property.dcid: [option.value.ref.dcid for option in menu.options]
        for menu in spec.resolved.slot_options
        if menu.key.property is not None and menu.key.property.dcid in pinned
    }
    assert menus == {prop: [value] for prop, value in pinned.items()}
    for prop, value in pinned.items():
        assert value in {option.dcid for option in get_index().level_scoped_options(prop, value)}


@pytest.mark.parametrize("variable", ["imputed aid", "imputed ODA"])
def test_a_synonym_ending_in_a_measure_word_keeps_the_variable_and_answers_the_dataset(
    variable: str,
):
    """The synonym's last word is the measure it asks for, so the variable is no dataset
    name alone and stays."""
    llm = _ScriptedLLM(variable, readings=(("only", 0),))

    response = _resolve_scripted(f"{variable} to Kenya", llm, variable)

    assert _answers(response) == [(variable, IMPUTED_PROVENANCE, True)]
    (spec,) = _specs(response)
    assert _pinned(spec)[AID_TYPE] == IMPUTED_MULTILATERAL
    assert _bound(spec)["source"] == IMPUTED_PROVENANCE
    assert _query_qualifier_warnings(response) == []


def test_a_variable_that_is_its_mention_alone_binds_on_the_mentions_measure_word():
    """The bind model reads "aid" alone, so "multilateral" cannot bind the multisector
    purpose (#1059)."""
    variable = f"{_IM} aid"
    llm = _ScriptedLLM(variable, readings=(("only", 0),))

    response = _resolve_scripted(f"{variable} to Kenya", llm, variable)

    (prompt,) = llm.bind_prompts
    assert "Variable phrase: aid\n" in prompt
    (spec,) = _specs(response)
    assert "crsPurpose" not in _bound(spec)
    assert _sources(spec) == [IMPUTED_PROVENANCE]


def test_a_code_no_shape_of_the_named_dataset_declares_refuses_before_any_bind_call():
    """ "grants" names types of finance, which no imputed row carries, so the part refuses
    the term at candidacy, before any role check or bind call (#1059)."""
    variable = f"{_IM} grants"
    llm = _ScriptedLLM(variable, readings=(("only", 0),))

    response = _resolve_scripted(f"{variable} to Kenya", llm, variable)

    (part,) = response.parts
    assert isinstance(part, NoDataPart)
    assert part.no_data.reason == "qualifier_not_bound"
    (unbound,) = part.no_data.unbound_qualifiers
    assert (unbound.text, unbound.disposition) == ("grants", "unbound")
    assert {ref.dcid for ref in unbound.named} == {"CRSTypeOfFinance100", "CRSTypeOfFinance110"}
    assert llm.calls["_UniversalBindOutput"] == 0


_BUDGET_SUPPORT_MEMBER = next(
    row.dcid
    for row in get_index().statvars
    if IMPUTED_PROVENANCE in row.provenances
    and row.constraint_properties.get("crsPurpose") == "CRSPurposeCode51010"
)


def test_a_code_the_named_dataset_declares_binds_on_its_own_five_tuple(
    monkeypatch: pytest.MonkeyPatch,
):
    """ "budget support" names a purpose the imputed rows carry, so the part answers the
    imputed family under that purpose. Its role checks pass on the dataset's own
    five-tuple (#1059)."""
    variable = f"{_IM} budget support"
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [(_BUDGET_SUPPORT_MEMBER, "imputed", _FRA, _KEN, [2019, 2020, 2021])],
        role_names={_BUDGET_SUPPORT_MEMBER: names},
        facet_metadata={"imputed": FacetMetadata(provenance_id=IMPUTED_PROVENANCE, unit="USD")},
    )
    llm = _ScriptedLLM(variable, readings=(("only", 0),))

    response = _resolve_scripted(
        f"{variable} to Kenya",
        llm,
        graph=_graph(variable, members=(_CRS_MEMBER, _BUDGET_SUPPORT_MEMBER)),
    )

    (spec,) = _specs(response)
    assert _bound(spec)["crsPurpose"] == "CRSPurposeCode510"
    assert _pinned(spec)[AID_TYPE] == IMPUTED_MULTILATERAL
    assert _sources(spec) == [IMPUTED_PROVENANCE]


# ---------------------------------------------------------------------------
# The parsed door
# ---------------------------------------------------------------------------


def _resolve_parsed(llm: _ScriptedLLM, *variables: str) -> ResolveResponse:
    return offline_resolve(
        ResolveRequest(
            input=ParsedInput(
                parts=[
                    ParsedPart(
                        variable_text=variable, entities=[{"text": "Kenya", "role_text": "to"}]
                    )
                    for variable in variables
                ]
            )
        ),
        graph=_graph(*variables),
        llm=llm,
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )


def test_a_parsed_part_binds_the_dataset_its_reading_asks_for():
    llm = _ScriptedLLM(readings=(("only", 0),))

    response = _resolve_parsed(llm, _IM_E)

    (spec,) = _specs(response)
    assert _pinned(spec)[AID_TYPE] == IMPUTED_MULTILATERAL
    assert spec.resolved.shape.member_count == IMPUTED_MEMBER_COUNT
    assert llm.calls["MentionReadings"] == 1


@pytest.mark.parametrize(
    "readings",
    [(("unrelated", None),), (("also", 0),), LLMInfraError("the model is unavailable")],
    ids=["unrelated", "also", "infra"],
)
def test_a_parsed_part_whose_mention_is_not_read_only_binds_nothing_and_warns(readings):
    response = _resolve_parsed(_ScriptedLLM(readings=readings), _IM_E)

    (part,) = response.parts
    assert isinstance(part, DefinitePart)
    assert AID_TYPE not in _bound(part.spec)
    assert [
        warning.message for warning in part.warnings if warning.code == QUERY_QUALIFIER_DROPPED
    ] == [f'The requested qualifier "{_IM}" was not applied.']


def test_each_parsed_part_binds_its_own_mention():
    llm = _ScriptedLLM(readings=(("only", 1),))

    response = _resolve_parsed(llm, _E, _IM_E)

    plain, imputed = _specs(response)
    assert AID_TYPE not in _bound(plain)
    assert _pinned(imputed)[AID_TYPE] == IMPUTED_MULTILATERAL
    assert llm.calls["MentionReadings"] == 1


def test_a_parsed_part_naming_no_dataset_asks_no_reading():
    llm = _ScriptedLLM()

    response = _resolve_parsed(llm, _E)

    (spec,) = _specs(response)
    assert AID_TYPE not in _bound(spec)
    assert llm.calls["MentionReadings"] == 0


# ---------------------------------------------------------------------------
# A reading takes effect only where a part carries it
# ---------------------------------------------------------------------------


def _answers(response: ResolveResponse) -> list[tuple[str | None, str, bool]]:
    """Each part's variable text, the dataset its family comes from, and whether the
    identity binds the aid type the imputed five-tuple pins."""
    return [
        (part.variable_text, _sources(spec)[0], AID_TYPE in _pinned(spec))
        for part, spec in zip(response.parts, _specs(response), strict=True)
    ]


def test_a_dataset_named_on_two_variables_binds_on_each_part():
    all_sectors = f"{_IM} aid"
    llm = _ScriptedLLM(all_sectors, _IM_E, readings=(("only", 0), ("only", 1)))

    response = _resolve_scripted(f"{all_sectors} and {_IM_E} to Kenya", llm, all_sectors, _IM_E)

    assert _answers(response) == [
        (all_sectors, IMPUTED_PROVENANCE, True),
        (_IM_E, IMPUTED_PROVENANCE, True),
    ]
    assert _query_qualifier_warnings(response) == []
    assert llm.calls["MentionReadings"] == 1


def test_a_dataset_named_on_two_variables_and_read_on_one_reports_the_other():
    all_sectors = f"{_IM} aid"
    llm = _ScriptedLLM(all_sectors, _IM_E, readings=(("only", 0), ("unrelated", None)))

    response = _resolve_scripted(f"{all_sectors} and {_IM_E} to Kenya", llm, all_sectors, _IM_E)

    assert _answers(response) == [
        (all_sectors, IMPUTED_PROVENANCE, True),
        (_IM_E, CRS_PROVENANCE, False),
    ]
    assert _query_qualifier_warnings(response) == [
        f'The requested qualifier "{_IM}" was not applied.'
    ]


@pytest.mark.parametrize(
    ("query", "variables", "dropped", "readings", "parts"),
    [
        (
            f"{_E} to Kenya including {_IM}",
            (_IM_E,),
            (),
            (("also", 0),),
            [(_IM_E, True), (_E, False)],
        ),
        (
            f"{_E} and {_IM_E} to Kenya",
            (_E, _IM_E),
            (),
            (("also", 1),),
            [(_E, False), (_IM_E, True)],
        ),
        (
            f"{_E} and {_IM_E} to Kenya",
            (_E, _IM_E),
            (),
            (("also", 0),),
            [(_E, False), (_IM_E, True)],
        ),
        (
            f"{_E} to Kenya including {_IM} ODA",
            (_E,),
            (f"including {_IM} ODA",),
            (("also", 0),),
            [(_E, False), (_IM_E, True)],
        ),
        (
            f"{_E} to Kenya as well as {_IM} ODA",
            (_E, f"{_IM} ODA"),
            (),
            (("also", 0),),
            [(_E, False), (_IM_E, True)],
        ),
        (
            f"education spending to Kenya including {_IM} aid",
            ("education spending",),
            (f"including {_IM} aid",),
            (("also", 0),),
            [("education spending", False), (f"{_IM} education spending", True)],
        ),
    ],
    ids=[
        "collapsed-into-one-variable",
        "read-on-the-datasets-variable",
        "read-on-the-measure",
        "dropped-phrase-other-measure-word",
        "name-alone-other-measure-word",
        "measure-lacking-the-mentions-word",
    ],
)
def test_also_answers_the_measure_from_the_default_and_from_the_dataset(
    query, variables, dropped, readings, parts
):
    llm = _ScriptedLLM(*variables, dropped=dropped, readings=readings)
    texts = [text for text, _imputed in parts]

    response = _resolve_scripted(query, llm, *texts)

    assert _answers(response) == [
        (text, IMPUTED_PROVENANCE if imputed else CRS_PROVENANCE, imputed)
        for text, imputed in parts
    ]
    assert len({spec.spec_id for spec in _specs(response)}) > 1
    assert _query_qualifier_warnings(response) == []


def test_an_excluded_dataset_extracted_as_a_variable_of_its_own_answers_no_part():
    llm = _ScriptedLLM(_E, f"{_IM} aid", readings=(("excluded", None),))

    response = _resolve_scripted(f"{_E} to Kenya but not {_IM} aid", llm, _E, f"{_IM} aid")

    assert _answers(response) == [(_E, CRS_PROVENANCE, False)]
    assert _query_qualifier_warnings(response) == []


def test_an_excluded_dataset_that_is_the_only_variable_answers_its_measure_from_the_default():
    llm = _ScriptedLLM(f"{_IM} aid", readings=(("excluded", None),))

    response = _resolve_scripted(f"aid to Kenya not counting {_IM}", llm, "aid")

    assert _answers(response) == [("aid", CRS_PROVENANCE, False)]
    assert _query_qualifier_warnings(response) == []


def test_an_exclusion_extracted_as_a_variable_and_read_on_the_measure_answers_no_part():
    llm = _ScriptedLLM(_E, f"{_IM} excluded", readings=(("excluded", 0),))

    response = _resolve_scripted(f"{_E} to Kenya, {_IM} excluded", llm, _E, f"{_IM} excluded")

    assert _answers(response) == [(_E, CRS_PROVENANCE, False)]
    assert _query_qualifier_warnings(response) == []


@pytest.mark.parametrize(
    ("variables", "readings", "parts"),
    [
        (
            (_E, f"{_IM} aid flows"),
            (("also", 0),),
            [(_E, CRS_PROVENANCE, False), (f"{_IM} aid flows", CRS_PROVENANCE, False)],
        ),
        (
            (f"{_IM} aid", _IM_E),
            (("also", 0), ("also", 0)),
            [(f"{_IM} aid", IMPUTED_PROVENANCE, True), (_IM_E, CRS_PROVENANCE, False)],
        ),
    ],
    ids=["another-measure", "target-is-the-mention"],
)
def test_also_read_away_from_a_variable_naming_another_measure_rewrites_nothing(
    variables, readings, parts
):
    llm = _ScriptedLLM(*variables, readings=readings)

    response = _resolve_scripted(f"{' and '.join(variables)} to Kenya", llm, *variables)

    assert _answers(response) == parts
    assert len(_query_qualifier_warnings(response)) == 1


@pytest.mark.parametrize(
    "readings", [(("only", 1),), (("unrelated", None),), (("excluded", None),)]
)
def test_a_dataset_name_with_no_measure_is_no_part_and_is_reported(readings):
    llm = _ScriptedLLM(_E, "CRS", readings=readings)

    response = _resolve_scripted(f"{_E} and CRS in Kenya", llm, _E, "CRS")

    assert [part.variable_text for part in response.parts] == [_E]
    assert _query_qualifier_warnings(response) == ['The requested qualifier "CRS" was not applied.']


# ---------------------------------------------------------------------------
# A failed reading and the response cache
# ---------------------------------------------------------------------------


class _FailingOnce(_ScriptedLLM):
    """Fail the first mention reading and answer the later ones."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is MentionReadings and not self.calls["MentionReadings"]:
            self.calls["MentionReadings"] += 1
            raise LLMInfraError("the model is unavailable")
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


def _resolve_cached(query: str, llm: _ScriptedLLM, cache: ResponseCache) -> ResolveResponse:
    with patch("search.engine.extract.date") as today:
        today.today.return_value = PINNED_DATE
        today.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        return asyncio.run(
            resolve_async(
                make_request(query),
                graph=_graph(_IM_E),
                llm=llm,
                sdmx=FakeSdmxClient(observations={}, availability={}),
                response_cache=cache,
            )
        )


def test_an_answer_built_on_a_failed_reading_enters_no_cache():
    query = f"{_IM_E} to Kenya"
    llm = _FailingOnce(_IM_E, readings=(("only", 0),))
    cache = ResponseCache()

    failed = _resolve_cached(query, llm, cache)

    assert _answers(failed) == [(_IM_E, CRS_PROVENANCE, False)]
    assert len(_query_qualifier_warnings(failed)) == 1
    assert cache.stats().entries == 0

    read = _resolve_cached(query, llm, cache)

    assert _answers(read) == [(_IM_E, IMPUTED_PROVENANCE, True)]
    assert _query_qualifier_warnings(read) == []
    assert llm.calls["MentionReadings"] == 2
    assert cache.stats().entries == 2

    _resolve_cached(query, llm, cache)

    assert llm.calls["MentionReadings"] == 2


# ---------------------------------------------------------------------------
# The parsed door reports what the raw-text door reports
# ---------------------------------------------------------------------------


def _part_warnings(response: ResolveResponse) -> list[list[str]]:
    return [
        [warning.message for warning in part.warnings if warning.code == QUERY_QUALIFIER_DROPPED]
        for part in response.parts
    ]


@pytest.mark.parametrize(
    ("variables", "readings", "warnings"),
    [
        (("non-CRS education aid",), (("excluded", None),), [["CRS"]]),
        ((f"{_E} excluding {_IM}",), (("excluded", None),), [[]]),
        ((_IM_E,), (("also", 0),), [[_IM]]),
        ((_IM_E,), (("unrelated", None),), [[_IM]]),
        ((_E, f"{_IM} health aid"), (("only", 0),), [[], [_IM]]),
        ((f"{_IM} aid", _IM_E), (("only", 0), ("unrelated", None)), [[], [_IM]]),
        ((f"CRS {_IM_E}",), (("only", 0), ("only", 0)), [[_IM]]),
    ],
    ids=[
        "excluded-default",
        "excluded",
        "also",
        "unrelated",
        "read-on-another-part",
        "second-occurrence",
        "second-dataset",
    ],
)
def test_a_parsed_part_reports_each_reading_that_took_no_effect(variables, readings, warnings):
    response = _resolve_parsed(_ScriptedLLM(readings=readings), *variables)

    assert _part_warnings(response) == [
        [f'The requested qualifier "{text}" was not applied.' for text in part] for part in warnings
    ]


def test_each_parsed_part_naming_the_dataset_binds_on_its_own_reading():
    llm = _ScriptedLLM(readings=(("only", 0), ("only", 1)))

    response = _resolve_parsed(llm, f"{_IM} aid", _IM_E)

    assert [AID_TYPE in _pinned(spec) for spec in _specs(response)] == [True, True]
    assert _part_warnings(response) == [[], []]
    assert llm.calls["MentionReadings"] == 1


# ---------------------------------------------------------------------------
# A dropped phrase that holds a mention and other words
# ---------------------------------------------------------------------------


def _dropped_section(prompt: str) -> list[str]:
    """The phrases a bind prompt lists under its dropped qualifiers."""
    lines = prompt.splitlines()
    start = next(
        (number for number, line in enumerate(lines) if line.startswith("DROPPED QUALIFIERS")),
        None,
    )
    if start is None:
        return []
    return [line.strip() for line in lines[start + 1 : lines.index("", start)]]


@pytest.mark.parametrize(
    ("dropped", "left"),
    [
        (f"including {_IM} aid, verified figures only", "verified figures only"),
        (f"audited, including {_IM} aid", "audited"),
        (f"audited; including {_IM}; verified figures only", "audited, verified figures only"),
        (f"audited including {_IM} aid", "audited"),
        (f"including {_IM} aid for girls", "for girls"),
    ],
    ids=["after", "before", "around", "same-clause-before", "same-clause-after"],
)
def test_the_clauses_beside_an_applied_mention_stay_dropped_and_reach_each_bind_leg(
    dropped: str, left: str
):
    llm = _ScriptedLLM(_E, dropped=(dropped,), readings=(("also", 0),))

    response = _resolve_scripted(f"{_E} to Kenya, {dropped}", llm, _E, _IM_E)

    assert _answers(response) == [(_E, CRS_PROVENANCE, False), (_IM_E, IMPUTED_PROVENANCE, True)]
    assert _query_qualifier_warnings(response) == [
        f'The requested qualifier "{left}" was not applied.'
    ]
    assert [_dropped_section(prompt) for prompt in llm.bind_prompts] == [[left], [left]]


def test_an_exclusion_read_away_from_a_variable_naming_more_keeps_it_and_warns():
    exclusion = f"{_E} excluding {_IM}"
    llm = _ScriptedLLM("health aid", exclusion, readings=(("excluded", 0),))

    response = _resolve_scripted(
        f"health aid and {exclusion} to Kenya", llm, "health aid", exclusion
    )

    assert [part.variable_text for part in response.parts] == ["health aid", exclusion]
    assert _query_qualifier_warnings(response) == [
        f'The requested qualifier "{_IM}" was not applied.'
    ]


def test_a_dropped_phrase_that_is_one_clause_around_its_applied_mention_leaves_nothing():
    llm = _ScriptedLLM(_E, dropped=(f"from {_IM} sources",), readings=(("only", 0),))

    response = _resolve_scripted(f"{_E} to Kenya from {_IM} sources", llm, f"{_E} {_IM}")

    assert _answers(response) == [(f"{_E} {_IM}", IMPUTED_PROVENANCE, True)]
    assert _query_qualifier_warnings(response) == []
    (prompt,) = llm.bind_prompts
    assert _dropped_section(prompt) == []


# ---------------------------------------------------------------------------
# Two names of one dataset
# ---------------------------------------------------------------------------


def test_a_dataset_the_question_names_one_way_and_its_variable_another_is_one_mention():
    llm = _ScriptedLLM(_IM_E, readings=(("only", 0),))

    response = _resolve_scripted(f"{_IMPUTED_LABEL} {_E} to Kenya", llm, _IM_E)

    assert _answers(response) == [(_IM_E, IMPUTED_PROVENANCE, True)]
    assert _query_qualifier_warnings(response) == []
    assert llm.calls["MentionReadings"] == 1


def test_two_names_of_one_dataset_in_one_variable_bind_once_and_report_nothing():
    variable = f"{_IM_E} from the {_IMPUTED_LABEL}"
    llm = _ScriptedLLM(variable, readings=(("only", 0), ("only", 0)))

    response = _resolve_scripted(f"{variable} to Kenya", llm, variable)

    assert _answers(response) == [(variable, IMPUTED_PROVENANCE, True)]
    assert _bound(_specs(response)[0])["source"] == IMPUTED_PROVENANCE
    assert _query_qualifier_warnings(response) == []


# ---------------------------------------------------------------------------
# The parsed door applies an inclusion its parts already spell out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variable", [0, 1], ids=["read-on-the-measure", "read-on-its-own-part"])
def test_a_parsed_inclusion_binds_the_part_naming_the_dataset_beside_its_measure(variable: int):
    response = _resolve_parsed(_ScriptedLLM(readings=(("also", variable),)), _E, _IM_E)

    assert _answers(response) == [(_E, CRS_PROVENANCE, False), (_IM_E, IMPUTED_PROVENANCE, True)]
    assert _part_warnings(response) == [[], []]


@pytest.mark.parametrize(
    ("variables", "readings", "warnings"),
    [
        ((_IM_E,), (("also", 0),), [[_IM]]),
        ((_E, f"{_IM} aid"), (("also", 0),), [[], [f"{_IM} aid"]]),
        ((_E, f"{_IM} health aid"), (("also", 0),), [[], [_IM]]),
        (("health aid", _IM_E, f"{_IM} aid"), (("only", 1), ("also", 0)), [[], [], [f"{_IM} aid"]]),
    ],
    ids=["would-add-the-measure", "would-rename", "another-measure", "would-remove"],
)
def test_a_parsed_inclusion_that_would_add_or_rewrite_a_part_binds_nothing_and_warns(
    variables, readings, warnings
):
    response = _resolve_parsed(_ScriptedLLM(readings=readings), *variables)

    assert _part_warnings(response) == [
        [f'The requested qualifier "{text}" was not applied.' for text in part] for part in warnings
    ]
    assert [part.variable_text for part in response.parts] == list(variables)


# ---------------------------------------------------------------------------
# Token usage counts the mention reading once
# ---------------------------------------------------------------------------

_CALL_USAGE = {"input_tokens": 100, "output_tokens": 10, "cached_tokens": 1}


class _MeteredLLM(_ScriptedLLM):
    """A scripted model whose every call reports `_CALL_USAGE`."""

    async def generate_structured(self, *, prompt, system, schema):
        output, _usage = await super().generate_structured(
            prompt=prompt, system=system, schema=schema
        )
        return output, dict(_CALL_USAGE)


def _response_and_done(
    request: ResolveRequest, llm: _MeteredLLM, *variables: str
) -> tuple[ResolveResponse, DoneEvent]:
    async def collect() -> list:
        return [
            event
            async for event in resolve_events(
                request,
                graph=_graph(*variables),
                llm=llm,
                sdmx=FakeSdmxClient(observations={}, availability={}),
            )
        ]

    events = asyncio.run(collect())
    (response,) = [event.response for event in events if isinstance(event, ResponseEvent)]
    (done,) = [event for event in events if isinstance(event, DoneEvent)]
    return response, done


def _usage_reported(request: ResolveRequest, llm: _MeteredLLM, *variables: str) -> list[dict]:
    """The token usage in the response's diagnostics and in the done event's telemetry."""
    response, done = _response_and_done(request, llm, *variables)
    return [response.diagnostics.llm_usage, done.telemetry.llm_usage]


def _usage_of(calls: Counter[str]) -> dict:
    return {key: value * sum(calls.values()) for key, value in _CALL_USAGE.items()}


@pytest.mark.parametrize("variables", [(_IM_E,), (_E, _IM_E)], ids=["one-part", "two-parts"])
def test_a_parsed_request_counts_its_mention_reading_once(variables: tuple[str, ...]):
    llm = _MeteredLLM(readings=(("only", len(variables) - 1),))
    request = ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(variable_text=variable, entities=[{"text": "Kenya", "role_text": "to"}])
                for variable in variables
            ]
        )
    )

    reported = _usage_reported(request, llm, *variables)

    assert llm.calls == {"MentionReadings": 1, "_UniversalBindOutput": len(variables)}
    assert reported == [_usage_of(llm.calls)] * 2


@pytest.mark.parametrize("variables", [(_IM_E,), (_E, _IM_E)], ids=["one-part", "two-parts"])
@pytest.mark.parametrize("relation", ["only", "unrelated"])
def test_a_resubmitted_parsed_part_reads_its_mention_as_the_parsed_door_does(
    relation: str, variables: tuple[str, ...]
):
    readings = ((relation, len(variables) - 1 if relation == "only" else None),)
    parts = [
        ParsedPart(variable_text=variable, entities=[{"text": "Kenya", "role_text": "to"}])
        for variable in variables
    ]
    answers = []
    for door in (
        ParsedInput(parts=parts),
        DatasetResubmitInput(entries=[ResubmitParsedEntry(part=part) for part in parts]),
    ):
        llm = _MeteredLLM(readings=readings)
        response, done = _response_and_done(ResolveRequest(input=door), llm, *variables)
        specs = _specs(response)
        if relation == "only":
            assert _pinned(specs[-1])[AID_TYPE] == IMPUTED_MULTILATERAL
            assert specs[-1].resolved.shape.member_count == IMPUTED_MEMBER_COUNT
        else:
            assert all(AID_TYPE not in _bound(spec) for spec in specs)
        assert llm.calls == {"MentionReadings": 1, "_UniversalBindOutput": len(variables)}
        answers.append(
            (
                [
                    (spec.spec_id, spec.resolved.shape.member_count, part.warnings)
                    for spec, part in zip(specs, response.parts, strict=True)
                ],
                _query_qualifier_warnings(response),
                [response.diagnostics.llm_usage, done.telemetry.llm_usage],
            )
        )

    parsed, resubmitted = answers
    assert resubmitted == parsed
    dropped = [] if relation == "only" else [f'The requested qualifier "{_IM}" was not applied.']
    assert parsed[1] == dropped
    assert parsed[2] == [_usage_of(llm.calls)] * 2


@pytest.mark.parametrize("variables", [(_IM_E,), (_E, _IM_E)], ids=["one-part", "two-parts"])
def test_a_raw_text_request_counts_its_mention_reading_once(variables: tuple[str, ...]):
    llm = _MeteredLLM(*variables, readings=(("only", len(variables) - 1),))
    query = f"{' and '.join(variables)} to Kenya"

    reported = _usage_reported(make_request(query), llm, *variables)

    assert llm.calls == {
        "Extraction": 1,
        "MentionReadings": 1,
        "_UniversalBindOutput": len(variables),
    }
    assert reported == [_usage_of(llm.calls)] * 2
