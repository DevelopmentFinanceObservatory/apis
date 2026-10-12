"""End-to-end tests for how an aid question elects its dataset.

The packaged index holds two datasets, each on its own five-tuple: the CRS on `amount`,
and the opt-in imputed multilateral sectors table on `attributedAmount`, whose every row
carries `dac2aAidType=DAC2aAidType106` and `crsTypeOfFlow=CRSTypeOfFlow10`, the slots the
five-tuple pins. The election implies both, so an imputed part's identity leaves them
unbound. The recall sample of a scripted question holds one member of each, so an unnamed
part's bind menu offers both. A question names no dataset, so a part answers the CRS
unless bind elects the imputed shape, which names its table; the tests of a named dataset
name it at the part's seam, `ResolveState.scope`. The recorded tests replay the
extraction and bind responses in `tests/fixtures/llm_responses.json`, and their recall
holds the CRS member alone, as the dev graph's recall for "education aid" and "aid" does;
the other tests script the model.
"""

from __future__ import annotations

import dataclasses
import sys
from collections import Counter
from collections.abc import Callable, Mapping

import pytest

from code_index import shared as shared_module
from code_index.members import canonical_identity, elected_pins
from code_index.query import CodeIndex, row_five_tuple
from code_index.shared import get_index
from code_index.types import DatasetDeclaration
from datasets_api.adapter import resolve_definition_members
from origin_catalogue import DFO, Catalogue, ProvenanceRow, SourceRef
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.extract import ExtractedEntity, Extraction
from search.engine.region import pipeline as pipeline_module
from search.engine.region.phase_rank_bind import _phase_prune_and_rank
from search.engine.region.state import ResolveState
from search.engine.retrieval.bound_values import identity_dataset, identity_query
from search.engine.spec_id import mint_identity
from search.models import (
    CandidatesPart,
    DefinitePart,
    NoDataPart,
    Operation,
    ParsedInput,
    ParsedPart,
    ResolveRequest,
    ResolveResponse,
    SourceMention,
    Spec,
)
from search.models.response import Part
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
from tests.datasets_api._member_bounds import mint_bounds
from tests.engine._fake_coverage import patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve
from tests.engine._source_scopes import part_scope
from tests.fixtures import FakeGraph, FakeLLM
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

_KEN = "country/KEN"
_FRA = "country/FRA"
_UGA = "country/UGA"
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
# A test-only table that the second-table tests declare beside the CRS on the aid five-tuple
# (`_with_second_table`), so the family has two publishers.
_SECOND = "provenance/TestDacTable2a"
_SECOND_LABEL = "DAC Table 2a"
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
    un_members: tuple[str, ...] = (),
    countries: Mapping[str, str] | None = None,
) -> FakeGraph:
    """A graph whose recall returns `members` in order for each variable: one CRS member
    and one imputed member unless told otherwise. The UN System Data Commons' recall returns
    `un_members`, each an aid amount holding a series at Kenya. Kenya resolves, and so does
    each of `countries`, by name."""
    countries = {"Kenya": _KEN, **(countries or {})}
    crs_label, imputed_label = provenance_labels
    return FakeGraph(
        nodes={
            _FRA: {"label": "France", "type": "Country"},
            **{dcid: {"label": name, "type": "Country"} for name, dcid in countries.items()},
            CRS_PROVENANCE: {"label": crs_label},
            IMPUTED_PROVENANCE: {"label": imputed_label},
            _SECOND: {"label": _SECOND_LABEL},
            **{
                dcid: {"label": dcid, "type": "StatisticalVariable", "arcs": _arcs(dcid)}
                for dcid in members
            },
            **{dcid: _un_node(dcid) for dcid in un_members},
        },
        detect={
            variable: {
                "svs": list(members),
                "cosine_scores": [0.9 - 0.05 * rank for rank in range(len(members))],
            }
            for variable in variables
        },
        un_detect={
            variable: {"svs": list(un_members), "cosine_scores": [0.9] * len(un_members)}
            for variable in variables
        }
        if un_members
        else None,
        resolve=dict(countries),
        obs={
            f"{dcid}|{_KEN}": [{"earliestDate": "2019", "latestDate": "2021", "obsCount": 3}]
            for dcid in un_members
        },
    )


def _un_node(dcid: str) -> dict:
    return {
        "label": dcid,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": dcid}]},
            "populationType": {"nodes": [{"dcid": "FinancialFlow"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "constraintProperties": {"nodes": []},
        },
    }


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
    """Replay the recorded responses, and keep each bind prompt by its variable and the
    number of calls per schema."""

    def __init__(self) -> None:
        self._llm = FakeLLM()
        self.bind_prompts: dict[str, str] = {}
        self.calls: Counter[str] = Counter()

    async def generate_structured(self, *, prompt, system, schema):
        self.calls[schema.__name__] += 1
        if issubclass(schema, _UniversalBindOutput):
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
        graph=_graph(*variables, members=(_CRS_MEMBER,)),
        llm=llm,
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    return response, llm


def _specs(response: ResolveResponse) -> list[Spec]:
    assert all(isinstance(part, DefinitePart) for part in response.parts), response.parts
    return [part.spec for part in response.parts]


def _held_spec(part: Part) -> Spec:
    """The Spec an answered part carries, or the flagged match a no-data part keeps."""
    if isinstance(part, DefinitePart):
        return part.spec
    assert isinstance(part, NoDataPart), part
    assert part.no_data.flagged_match is not None
    return part.no_data.flagged_match


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
        enumerator=MemberEnumerator(),
        code_index=get_index(),
        bounds=mint_bounds(),
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


_E = "education aid"


def _assert_imputed_part(spec: Spec) -> None:
    """The identity dev mints for imputed education aid, and the canonical form of the same
    identity posted with the table's invariants unbound."""
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


# ---------------------------------------------------------------------------
# A question naming no dataset: recorded extraction and bind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "variable", "member_count", "open_roles"),
    [
        ("top donors of education aid to Kenya", _E, CRS_MEMBER_COUNT, ["origin"]),
        (f"{_E} to Kenya", _E, CRS_MEMBER_COUNT, None),
        ("total aid to Kenya", "aid", _PLAIN_AID_MEMBER_COUNT, None),
        ("top 5 donantes de ayuda a Kenia", "aid", _PLAIN_AID_MEMBER_COUNT, ["origin"]),
    ],
)
def test_a_question_naming_no_dataset_answers_the_crs(
    query: str, variable: str, member_count: int, open_roles: list[str] | None
):
    response, llm = _resolve(query, variable)

    (spec,) = _specs(response)
    assert llm.calls == {"Extraction": 1, "_UniversalBindOutput": 1}
    assert spec.resolved.shape.member_count == member_count
    assert AID_TYPE not in _bound(spec)
    assert "source" not in _bound(spec)
    plain = education_aid_identity() if variable == _E else _plain_aid_identity()
    if open_roles is not None:
        plain["open_roles"] = open_roles
    assert spec.definition.identity.open_roles == open_roles
    assert spec.spec_id == _spec_id(plain)


def test_a_ranked_education_aid_question_asks_every_donor_and_keeps_its_operation():
    """The rank runs over donors, so the part leaves the donor open. The CRS declares the
    donor, and the identity states it beside the plain CRS one."""
    response, _llm = _resolve("top donors of education aid to Kenya", "education aid")

    (spec,) = _specs(response)
    assert spec.definition.identity.open_roles == ["origin"]
    assert spec.spec_id == _spec_id({**education_aid_identity(), "open_roles": ["origin"]})
    assert spec.spec_id != EDUCATION_AID_SPEC_ID
    assert response.parts[0].operations == [
        Operation(verb="rank", phrase="top donors", over="origin")
    ]


# ---------------------------------------------------------------------------
# A second table on the CRS aid five-tuple: recorded extraction and bind
# ---------------------------------------------------------------------------


def _with_second_table(*, opt_in: bool) -> CodeIndex:
    """The packaged index with `_SECOND` declared beside the CRS on the aid five-tuple."""
    packaged = get_index()
    aid = row_five_tuple(next(row for row in packaged.statvars if row.dcid == _CRS_MEMBER))
    return CodeIndex(
        stamp=packaged.stamp,
        declarations=tuple(packaged.declarations.values()),
        codes=packaged.codes,
        statvars=tuple(
            dataclasses.replace(row, provenances=(*row.provenances, _SECOND))
            if row_five_tuple(row) == aid and CRS_PROVENANCE in row.provenances
            else row
            for row in packaged.statvars
        ),
        datasets=(
            *packaged.datasets.values(),
            DatasetDeclaration(
                provenance=_SECOND,
                label=_SECOND_LABEL,
                description=(
                    "Official development assistance by donor and recipient, as the OECD's"
                    " aggregate tables publish it."
                ),
                opt_in=opt_in,
                absent_observation="zero",
                account="provider",
                rank=20,
            ),
        ),
    )


def _use_second_table(monkeypatch: pytest.MonkeyPatch, *, opt_in: bool) -> None:
    """Serve `_with_second_table` wherever a loaded module reads the packaged index, and
    give the CRS member a series under `_SECOND` as well."""
    index = _with_second_table(opt_in=opt_in)
    packaged = shared_module.get_index
    for module in list(sys.modules.values()):
        if vars(module).get("get_index") is packaged:
            monkeypatch.setattr(module, "get_index", lambda: index)
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            (_CRS_MEMBER, "crs", _FRA, _KEN, [2019, 2020, 2021]),
            (_CRS_MEMBER, "second", _FRA, _KEN, [2019, 2020, 2021]),
        ],
        role_names={_CRS_MEMBER: names},
        facet_metadata={
            "crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD"),
            "second": FacetMetadata(provenance_id=_SECOND, unit="USD"),
        },
    )


def _catalogue_row(provenance: str, name: str, publisher: str) -> ProvenanceRow:
    return ProvenanceRow(
        id=provenance,
        data_commons=DFO,
        source=SourceRef(ref=f"dc/s/{publisher}", label=publisher),
        name=name,
        url="https://example.org",
    )


@pytest.fixture
def source_catalogue(bind_catalogue: Callable[[Catalogue], None]) -> None:
    """Bind an origin catalogue listing the CRS, the second table and the imputed table
    under their declared names, so a part's source words resolve to them."""
    rows = (
        _catalogue_row(CRS_PROVENANCE, _CRS_LABEL, "OECD"),
        _catalogue_row(_SECOND, _SECOND_LABEL, "OECD"),
        _catalogue_row(IMPUTED_PROVENANCE, _IMPUTED_LABEL, "ONE"),
    )
    bind_catalogue(FakeCatalogue(provenances={row.id: row for row in rows}))


def _naming(text: str) -> SourceMention:
    return SourceMention(text=text)


def _excluding(text: str) -> SourceMention:
    return SourceMention(text=text, polarity="exclude")


def _query(variable: str, sources: tuple[SourceMention, ...]) -> str:
    """A question on `variable` to Kenya whose words carry each of `sources`."""
    words = [
        f"excluding {source.text}" if source.polarity == "exclude" else f"from {source.text}"
        for source in sources
    ]
    return " ".join([f"{variable} to Kenya", *words])


def _unapplied(response: ResolveResponse) -> list[str]:
    """The phrases the answer reports as not applied: each part's `query_qualifier_dropped`
    warning and `measure_not_applied` departure."""
    warned = [
        warning.message
        for part in response.parts
        for warning in part.warnings
        if warning.code == "query_qualifier_dropped"
    ]
    departed = [
        departure.asked or ""
        for part in response.parts
        for departure in part.departures or ()
        if departure.code == "measure_not_applied"
    ]
    return [*warned, *departed]


def test_a_second_table_on_the_family_leaves_the_bind_prompt_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
):
    """The bind menu offers no source: a second table publishing the CRS aid family leaves
    the prompt of a question naming no table as it was, and the part answers the CRS under
    its plain identity."""
    _response, plain = _resolve("total aid to Kenya", "aid")
    _use_second_table(monkeypatch, opt_in=False)

    response, shared = _resolve("total aid to Kenya", "aid")

    assert shared.bind_prompts["aid"] == plain.bind_prompts["aid"]
    assert "axis=source" not in shared.bind_prompts["aid"]
    (spec,) = _specs(response)
    assert "source" not in _bound(spec)
    assert spec.spec_id == _spec_id(_plain_aid_identity())


@pytest.mark.parametrize("opt_in", [False, True], ids=["default", "opt-in"])
def test_a_question_naming_the_second_table_answers_from_it(
    monkeypatch: pytest.MonkeyPatch, source_catalogue: None, opt_in: bool
):
    """The question's words name the second table, opt-in or not, so the part mints the
    identity of every aid to Kenya naming that table: the plain aid identity with its
    provenance facet bound to the table, in canonical form."""
    _use_second_table(monkeypatch, opt_in=opt_in)
    named = {**_plain_aid_identity(), "facets": {"provenance": {"state": "value", "ref": _SECOND}}}

    response, _llm = _resolve(f"aid to Kenya according to {_SECOND_LABEL}", "aid")

    assert _unapplied(response) == []
    (spec,) = _specs(response)
    assert _bound(spec)["source"] == _SECOND
    assert _sources(spec) == [_SECOND]
    assert identity_dataset(spec.definition.identity) == _SECOND
    assert spec.spec_id == _spec_id(dict(canonical_identity(get_index(), named)))


@pytest.mark.parametrize("opt_in", [False, True], ids=["default", "opt-in"])
def test_a_question_excluding_the_crs_answers_from_the_other_table_or_refuses(
    monkeypatch: pytest.MonkeyPatch, source_catalogue: None, opt_in: bool
):
    """Excluding the CRS leaves the second table when it serves by default, and the identity
    names it. An opt-in second table answers only a part that names it, so excluding the
    CRS leaves no table and the part refuses, naming the CRS and the opt-in table."""
    _use_second_table(monkeypatch, opt_in=opt_in)

    response, _llm = _resolve("aid to Kenya excluding the CRS", "aid")

    (part,) = response.parts
    if opt_in:
        assert isinstance(part, NoDataPart)
        assert part.no_data.reason == "qualifier_not_bound"
        assert [
            [(ref.dcid, ref.label) for ref in unbound.named]
            for unbound in part.no_data.unbound_qualifiers
        ] == [[(CRS_PROVENANCE, _CRS_LABEL)], [(_SECOND, _SECOND_LABEL)]]
    else:
        assert _unapplied(response) == []
        (spec,) = _specs(response)
        assert _bound(spec)["source"] == _SECOND
        assert _sources(spec) == [_SECOND]


_DAC2A = "provenance/OECDDAC2A"
# DAC2a's official development assistance member of the nominal aid five-tuple, a family
# the CRS does not publish.
_DAC2A_MEMBER = next(
    row.dcid
    for row in get_index().statvars
    if row.provenances == (_DAC2A,)
    and row.population_type == "FinancialFlow"
    and row.measured_property == "amount"
    and row.measurement_denominator is None
    and row.constraint_properties.get("dac2aAidType") == "DAC2aAidType206"
)


def test_a_question_excluding_the_crs_answers_from_dac2a_in_its_menu(
    monkeypatch: pytest.MonkeyPatch, bind_catalogue: Callable[[Catalogue], None]
):
    """ "ODA to Kenya excluding the CRS" recalls DAC2a beside the CRS, so the exclusion
    chooses DAC2a from the unscoped menu and the part answers from it."""
    bind_catalogue(
        FakeCatalogue(
            provenances={
                row.id: row
                for row in (
                    _catalogue_row(CRS_PROVENANCE, _CRS_LABEL, "OECD"),
                    _catalogue_row(_DAC2A, "DAC2A", "OECD"),
                )
            }
        )
    )
    patch_coverage(
        monkeypatch,
        [(_DAC2A_MEMBER, "dac2a", _FRA, _KEN, [2019, 2020, 2021])],
        role_names={
            _DAC2A_MEMBER: RoleNames(donor="financingProvider", recipient="financingRecipient")
        },
        facet_metadata={"dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD")},
    )
    sources = (_excluding("CRS"),)

    response = _resolve_scripted(
        _query("ODA", sources),
        _ScriptedLLM("ODA"),
        graph=_graph("ODA", members=(_DAC2A_MEMBER, _CRS_MEMBER)),
    )

    (spec,) = _specs(response)
    assert _sources(spec) == [_DAC2A]


def test_an_included_table_that_serves_nothing_of_the_answer_warns(source_catalogue: None):
    """A part naming the CRS and the imputed table whose recall reaches only the CRS answers
    from the CRS and warns `query_qualifier_dropped` that the imputed table served
    nothing, quoting its words; the CRS, which served, raises no warning."""
    sources = (_naming("CRS"), _naming(_IMPUTED_LABEL))

    response = _resolve_scripted(
        _query(_E, sources),
        _ScriptedLLM(_E),
        graph=_graph(_E, members=(_CRS_MEMBER,)),
    )

    (spec,) = _specs(response)
    assert _sources(spec) == [CRS_PROVENANCE]
    (warned,) = _unapplied(response)
    assert f'"{_IMPUTED_LABEL}"' in warned


# ---------------------------------------------------------------------------
# A scripted model
# ---------------------------------------------------------------------------


_OTHER_PHRASE = "in fragile states"


def test_a_source_the_query_names_is_no_dropped_phrase(
    monkeypatch: pytest.MonkeyPatch, source_catalogue: None
):
    """The query names the second table beside a phrase the extraction drops. The part
    answers from the table, and only the dropped phrase raises a warning or a departure."""
    _use_second_table(monkeypatch, opt_in=False)
    llm = _ScriptedLLM("aid", dropped=(_OTHER_PHRASE,))

    response = _resolve_scripted(
        f"aid to Kenya from {_SECOND_LABEL} {_OTHER_PHRASE}",
        llm,
        graph=_graph("aid", members=(_CRS_MEMBER,)),
    )

    (spec,) = _specs(response)
    assert _bound(spec)["source"] == _SECOND
    reported = _unapplied(response)
    assert reported
    assert all(_OTHER_PHRASE in phrase for phrase in reported)


class _ScriptedLLM:
    """Extract the given variables to Kenya, and elect the first
    shape with every slot open but those `drafted` binds, nominating the shapes
    `also_plausible` indexes beside it, or ask `ask` and elect none. Each bind prompt is
    kept in call order."""

    def __init__(
        self,
        *variables: str,
        dropped: tuple[str, ...] = (),
        drafted: tuple[SlotBindingDraft, ...] = (),
        also_plausible: list[int] | None = None,
        ask: str | None = None,
    ) -> None:
        self._variables = list(variables)
        self._ask = ask
        self._also_plausible = also_plausible
        self._dropped = list(dropped)
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
        self.bind_prompts.append(prompt)
        return schema(
            chosen_shape_index=0,
            bindings=self._drafted,
            also_plausible=self._also_plausible,
            ask=self._ask,
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


def _answers(response: ResolveResponse) -> list[tuple[str | None, str, bool]]:
    """Each part's variable text, the dataset its family comes from, and whether the
    identity binds the aid type the imputed five-tuple pins."""
    return [
        (part.variable_text, _sources(spec)[0], AID_TYPE in _pinned(spec))
        for part, spec in zip(response.parts, _specs(response), strict=True)
    ]


_PHASES = pipeline_module._PHASES


def _name_dataset(monkeypatch: pytest.MonkeyPatch, provenance: str) -> None:
    """Name `provenance` as every part's dataset, at the seam shape candidacy reads
    (`ResolveState.scope`). A later call replaces the dataset an earlier one named."""

    async def named(state: ResolveState):
        state.scope = part_scope(provenance)
        return await _phase_prune_and_rank(state)

    monkeypatch.setattr(
        pipeline_module,
        "_PHASES",
        tuple(named if phase is _phase_prune_and_rank else phase for phase in _PHASES),
    )


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


def test_electing_the_opt_in_shape_names_its_table():
    """The imputed shape, recalled first beside the CRS shape, is offered to a part naming
    no dataset, without the table's invariant slots. Electing it names the table, so the part mints the identity
    of a part that names it, and the identity binds the table's invariants."""
    llm = _ScriptedLLM(_E)

    response = _resolve_scripted(
        f"{_E} to Kenya", llm, graph=_graph(_E, members=(_IMPUTED_MEMBER, _CRS_MEMBER))
    )

    (prompt,) = llm.bind_prompts
    assert (
        "  label: FinancialFlow imputed multilateral ODA, ONE Imputed Multilateral Sectors"
        in prompt
    )
    assert "  label: FinancialFlow disbursements, OECD Creditor Reporting System (CRS)" in prompt
    assert f"property_dcid={AID_TYPE}" not in prompt
    (spec,) = _specs(response)
    _assert_imputed_part(spec)
    assert _pinned(spec)["crsTypeOfFlow"] == "CRSTypeOfFlow10"


def test_the_opt_in_table_reached_as_an_alternate_mints_the_identity_electing_it():
    """Bind elects the CRS commitments shape and nominates the imputed shape beside it. The
    alternate names the opt-in table its shape alone may answer from, so its identity is the
    one a fresh question electing the imputed shape mints under the purpose the alternate's
    representative member binds."""
    purpose = next(
        row.constraint_properties["crsPurpose"]
        for row in get_index().statvars
        if row.dcid == _IMPUTED_MEMBER
    )
    drafted = SlotBindingDraft(
        axis="how", property_dcid="crsPurpose", kind="value", value_dcids=[purpose]
    )

    offered = _resolve_scripted(
        "aid to Kenya",
        _ScriptedLLM("aid", also_plausible=[1]),
        graph=_graph("aid", members=(_COMMITMENTS_MEMBER, _IMPUTED_MEMBER)),
    )
    fresh = _resolve_scripted(
        "aid to Kenya",
        _ScriptedLLM("aid", drafted=(drafted,)),
        graph=_graph("aid", members=(_IMPUTED_MEMBER, _CRS_MEMBER)),
    )

    (part,) = offered.parts
    assert isinstance(part, CandidatesPart)
    alternate = next(
        spec for spec in part.candidates.specs if _sources(spec) == [IMPUTED_PROVENANCE]
    )
    (elected,) = _specs(fresh)
    assert _bound(alternate)["source"] == IMPUTED_PROVENANCE
    assert alternate.spec_id == elected.spec_id


def test_the_opt_in_measure_alternative_mints_the_identity_electing_it():
    """Bind elects the CRS education aid shape and nominates the imputed shape beside it, a
    measure sibling, so the answer folds it into its measure alternatives. The alternative
    is the identity a fresh question electing the imputed shape mints: it names the opt-in
    table and binds its invariants, on the imputed shape's own slots, under the purpose the
    question bound."""
    offered = _resolve_scripted(
        f"{_E} to Kenya",
        _ScriptedLLM(_E, also_plausible=[1]),
        graph=_graph(_E, members=(_CRS_MEMBER, _IMPUTED_MEMBER)),
    )
    fresh = _resolve_scripted(
        f"{_E} to Kenya", _ScriptedLLM(_E), graph=_graph(_E, members=(_IMPUTED_MEMBER, _CRS_MEMBER))
    )

    (spec,) = _specs(offered)
    (elected,) = _specs(fresh)
    _assert_imputed_part(elected)
    alternative = next(
        alternative
        for alternative in spec.resolved.alternative_measures or []
        if identity_dataset(alternative.definition.identity) == IMPUTED_PROVENANCE
    )
    assert alternative.definition.spec_id == elected.spec_id
    assert alternative.definition.identity == elected.definition.identity


def test_a_parsed_part_naming_no_dataset_answers_the_crs():
    llm = _ScriptedLLM()

    response = offline_resolve(
        ResolveRequest(
            input=ParsedInput(
                parts=[
                    ParsedPart(variable_text=_E, entities=[{"text": "Kenya", "role_text": "to"}])
                ]
            )
        ),
        graph=_graph(_E),
        llm=llm,
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    (spec,) = _specs(response)
    assert AID_TYPE not in _bound(spec)
    assert _sources(spec) == [CRS_PROVENANCE]
    assert llm.calls == {"_UniversalBindOutput": 1}


def _resolve_parsed(part: ParsedPart, llm: _ScriptedLLM, *, graph: FakeGraph) -> ResolveResponse:
    return offline_resolve(
        ResolveRequest(input=ParsedInput(parts=[part])),
        graph=graph,
        llm=llm,
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )


def test_a_plain_question_binding_codes_no_crs_row_holds_together_is_not_resolved():
    """The CRS alone publishes the aid family, and it holds purpose 100 and modality B02 on
    separate rows. A question naming and excluding no table can be served from no other
    table, so the part goes on to materialise and ends `variable_not_resolved` with nothing
    named, as a part whose codes match no member does."""
    purpose, modality = "CRSPurposeCode100", "CRSCooperationModalityB02"
    index = get_index()
    family = index.family_by_dcid[_CRS_MEMBER]
    assert family.datasets == (CRS_PROVENANCE,)
    held = {
        (
            row.constraint_properties.get("crsPurpose") in index.descendants(purpose),
            row.constraint_properties.get("crsCooperationModality") == modality,
        )
        for row in family.rows
    }
    assert held == {(True, False), (False, True), (False, False)}
    drafted = (
        SlotBindingDraft(
            axis="what", property_dcid="crsPurpose", kind="value", value_dcids=[purpose]
        ),
        SlotBindingDraft(
            axis="how", property_dcid="crsCooperationModality", kind="value", value_dcids=[modality]
        ),
    )

    response = _resolve_scripted(
        "aid to Kenya",
        _ScriptedLLM("aid", drafted=drafted),
        graph=_graph("aid", members=(_CRS_MEMBER,)),
    )

    (part,) = response.parts
    assert isinstance(part, NoDataPart)
    assert part.no_data.reason == "variable_not_resolved"
    assert part.no_data.unbound_qualifiers == []


def test_raw_text_words_that_name_no_source_scope_nothing(source_catalogue: None):
    """Raw-text source words that match no admitted name are no source mention: the part
    answers unscoped and restates no source."""
    response = _resolve_scripted(
        "aid to Kenya from XYZ statistics", _ScriptedLLM("aid"), graph=_graph("aid")
    )

    (part,) = response.parts
    assert not isinstance(part, NoDataPart) or part.no_data.reason != "source_not_resolved"
    assert part.restatement is not None and part.restatement.sources == []


def test_raw_text_source_words_qualify_only_the_variable_they_attach_to(
    source_catalogue: None,
):
    """A source the query names after one variable reaches that variable's part alone: the
    first part restates the CRS with its ref, and the second restates no source."""
    response = _resolve_scripted(
        f"aid from the CRS and {_E} to Kenya", _ScriptedLLM("aid", _E), graph=_graph("aid", _E)
    )

    first, second = response.parts
    assert first.variable_text == "aid"
    assert first.restatement is not None
    assert first.restatement.sources == [SourceMention(text="CRS", ref=CRS_PROVENANCE)]
    assert second.variable_text == _E
    assert second.restatement is not None and second.restatement.sources == []


def test_raw_text_source_words_that_leave_a_part_no_source_refuse_that_part_alone(
    source_catalogue: None,
):
    """The query asks for aid from the OECD and for a second variable from the CRS but not
    from the OECD, whose tables hold the CRS. The second part's words leave it nothing to
    answer from, so it answers `qualifier_not_bound` naming the CRS, which it named, and
    runs no bind, while the first part answers from the OECD's tables."""
    llm = _ScriptedLLM("aid", _E)

    response = _resolve_scripted(
        f"aid to Kenya from the OECD and {_E} to Kenya, not from the OECD but from the CRS",
        llm,
        graph=_graph("aid", _E),
    )

    first, second = response.parts
    assert not isinstance(first, NoDataPart)
    assert isinstance(second, NoDataPart)
    assert second.no_data.reason == "qualifier_not_bound"
    assert [
        [ref.dcid for ref in unbound.named] for unbound in second.no_data.unbound_qualifiers
    ] == [[CRS_PROVENANCE]]
    assert llm.calls["Extraction"] == 1
    assert len(llm.bind_prompts) == 1


# The UN System Data Commons' official development assistance series, derived from the CRS.
_UN_ODA = "undata/sdg/SERIES-DC_ODA_TOTL"


def test_an_exclusion_refusal_stands_though_the_un_system_has_a_candidate(
    monkeypatch: pytest.MonkeyPatch, source_catalogue: None
):
    """The UN System Data Commons recalls an aid series with data at Kenya. Excluding the
    CRS beside an opt-in second table leaves the part no table, so it refuses naming the
    CRS and the opt-in table, and the UN series, which the scope allows no provenance of,
    does not answer in its place."""
    _use_second_table(monkeypatch, opt_in=True)
    graph = _graph("aid", members=(_CRS_MEMBER,), un_members=(_UN_ODA,))
    excluding = (_excluding("CRS"),)

    response = _resolve_scripted(_query("aid", excluding), _ScriptedLLM("aid"), graph=graph)

    (part,) = response.parts
    assert isinstance(part, NoDataPart)
    assert part.no_data.reason == "qualifier_not_bound"
    assert [[ref.dcid for ref in unbound.named] for unbound in part.no_data.unbound_qualifiers] == [
        [CRS_PROVENANCE],
        [_SECOND],
    ]


# Each row: a raw-text question whose identity binds a table it did not name in a typed
# field, the variable, the recall members, the sources its words name, the second
# table's declaration (`None` leaves the packaged index), and the table the identity binds.
_TABLE_ROWS = [
    pytest.param("aid", (_CRS_MEMBER,), (_naming(_SECOND_LABEL),), False, _SECOND, id="names"),
    pytest.param("aid", (_CRS_MEMBER,), (_excluding("CRS"),), False, _SECOND, id="excludes"),
    pytest.param(_E, (_IMPUTED_MEMBER, _CRS_MEMBER), (), None, IMPUTED_PROVENANCE, id="measure"),
]


@pytest.mark.parametrize(("variable", "members", "sources", "opt_in", "table"), _TABLE_ROWS)
def test_a_parsed_part_naming_a_table_mints_the_raw_text_answer_from_it(
    monkeypatch: pytest.MonkeyPatch,
    source_catalogue: None,
    variable: str,
    members: tuple[str, ...],
    sources: tuple[SourceMention, ...],
    opt_in: bool | None,
    table: str,
):
    """A parsed part posting the table as `provenance` mints the spec_id of the raw-text
    question whose source words or elected measure chose that table: the identity binds
    the provenance that served."""
    if opt_in is not None:
        _use_second_table(monkeypatch, opt_in=opt_in)
    graph = _graph(variable, members=members)
    raw = _resolve_scripted(_query(variable, sources), _ScriptedLLM(variable), graph=graph)
    part = ParsedPart(
        variable_text=variable, entities=[{"text": "Kenya", "role_text": "to"}], provenance=table
    )

    # The parsed side recalls the members in reverse, so the scripted model's first shape is
    # the CRS one, and only the named dataset steers the part to the table.
    parsed = _resolve_parsed(
        part, _ScriptedLLM(), graph=_graph(variable, members=tuple(reversed(members)))
    )

    (raw_spec,) = _specs(raw)
    (parsed_spec,) = _specs(parsed)
    assert _bound(parsed_spec)["source"] == table
    assert parsed_spec.spec_id == raw_spec.spec_id


def _restated_table(restatement: ParsedPart) -> str | None:
    """The table a restatement names: the ref of its one included source mention, else
    `None`."""
    refs = [mention.ref for mention in restatement.sources if mention.polarity == "include"]
    assert len(refs) <= 1
    return refs[0] if refs else None


def _hold_series_only_away_from_kenya(monkeypatch: pytest.MonkeyPatch) -> None:
    """Move every member's series, under each table, to a recipient other than Kenya, so a
    part to Kenya resolves its shape and finds no observation."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    members = (_CRS_MEMBER, _IMPUTED_MEMBER)
    patch_coverage(
        monkeypatch,
        [
            (member, facet, _FRA, _UGA, [2019, 2020, 2021])
            for member in members
            for facet in ("crs", "imputed", "second")
        ],
        role_names=dict.fromkeys(members, names),
        facet_metadata={
            "crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD"),
            "imputed": FacetMetadata(provenance_id=IMPUTED_PROVENANCE, unit="USD"),
            "second": FacetMetadata(provenance_id=_SECOND, unit="USD"),
        },
    )


@pytest.mark.parametrize("has_data", [True, False], ids=["answered", "no-data"])
@pytest.mark.parametrize(("variable", "members", "sources", "opt_in", "table"), _TABLE_ROWS)
def test_a_restated_part_names_the_table_its_identity_binds(
    monkeypatch: pytest.MonkeyPatch,
    source_catalogue: None,
    variable: str,
    members: tuple[str, ...],
    sources: tuple[SourceMention, ...],
    opt_in: bool | None,
    table: str,
    has_data: bool,
):
    """The restatement of a part carries its source words with the refs they resolved to,
    and a part naming none the table its identity binds, so posting the restatement back
    answers from the same table under the same spec_id, and so does posting the parsed
    answer's own restatement. A part with no observation keeps the elected identity as its
    flagged match and restates the same way. An exclusion restates the excluded table, and
    names no table."""
    if opt_in is not None:
        _use_second_table(monkeypatch, opt_in=opt_in)
    if not has_data:
        _hold_series_only_away_from_kenya(monkeypatch)
    graph = _graph(variable, members=members)
    raw = _resolve_scripted(_query(variable, sources), _ScriptedLLM(variable), graph=graph)
    (raw_part,) = raw.parts
    assert raw_part.restatement is not None
    excludes = any(source.polarity == "exclude" for source in sources)
    assert _restated_table(raw_part.restatement) == (None if excludes else table)

    reposted = _resolve_parsed(raw_part.restatement, _ScriptedLLM(), graph=graph)

    (reposted_part,) = reposted.parts
    assert reposted_part.restatement == raw_part.restatement
    assert _held_spec(reposted_part).spec_id == _held_spec(raw_part).spec_id


@pytest.mark.parametrize(
    ("sources", "table"),
    [((), None), ((SourceMention(text=_SECOND_LABEL),), _SECOND)],
    ids=["no-source", "words-name"],
)
def test_a_parsed_part_restates_the_table_its_words_name(
    monkeypatch: pytest.MonkeyPatch,
    source_catalogue: None,
    sources: tuple[SourceMention, ...],
    table: str | None,
):
    """A parsed part naming a table by its words restates the mention with the table's
    ref, and a part naming none restates none when the plain CRS identity binds no
    source."""
    _use_second_table(monkeypatch, opt_in=False)
    part = ParsedPart(
        variable_text="aid", entities=[{"text": "Kenya", "role_text": "to"}], sources=list(sources)
    )

    response = _resolve_parsed(part, _ScriptedLLM(), graph=_graph("aid", members=(_CRS_MEMBER,)))

    (answered,) = response.parts
    assert _bound(answered.spec).get("source") == table
    assert answered.restatement is not None
    assert _restated_table(answered.restatement) == table


_DEU = "country/DEU"
_FROM_FRANCE = ({"text": "France", "role_text": "from"},)
_EXCLUDING_CRS = (SourceMention(text="CRS", polarity="exclude"),)
_EXCLUDING_SECOND = (SourceMention(text=_SECOND_LABEL, polarity="exclude"),)
_NAMING_SECOND = (SourceMention(text=_SECOND_LABEL),)


def _hold_france_series_only_under(monkeypatch: pytest.MonkeyPatch, facet: str | None) -> None:
    """Give the CRS and the imputed member a series from Germany to Uganda under its own
    table, and the CRS member one from France to Kenya under `facet` alone. The second
    table holds no other series, so a part from France finds an observation only where
    the table of `facet` serves it, and nowhere when `facet` is `None`."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    elsewhere = [
        (_CRS_MEMBER, "crs", _DEU, _UGA, [2019, 2020, 2021]),
        (_IMPUTED_MEMBER, "imputed", _DEU, _UGA, [2019, 2020, 2021]),
    ]
    france = [(_CRS_MEMBER, facet, _FRA, _KEN, [2019, 2020, 2021])] if facet else []
    patch_coverage(
        monkeypatch,
        [*elsewhere, *france],
        role_names=dict.fromkeys(_MEMBERS, names),
        facet_metadata={
            "crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD"),
            "imputed": FacetMetadata(provenance_id=IMPUTED_PROVENANCE, unit="USD"),
            "second": FacetMetadata(provenance_id=_SECOND, unit="USD"),
        },
    )


# Each row: a parsed part whose run settles on a table and finds no observation under it,
# with no flagged match to carry the identity. The variable, the recall members, the
# entities, the open roles, the part's source mentions, the second table's declaration
# (`None` leaves the packaged index), the one table holding a series from France, and the
# table the restatement names.
_NO_DATA_ROWS = [
    pytest.param(
        _E, (_IMPUTED_MEMBER, _CRS_MEMBER), _FROM_FRANCE, None, (), None, "crs",
        IMPUTED_PROVENANCE, id="measure",
    ),
    pytest.param(
        "aid", (_CRS_MEMBER,), _FROM_FRANCE, None, _NAMING_SECOND, False, "crs",
        _SECOND, id="words-name",
    ),
    pytest.param(
        "aid", (_CRS_MEMBER,), _FROM_FRANCE, None, _EXCLUDING_CRS, False, "crs",
        None, id="words-exclude",
    ),
    pytest.param(
        "aid", (_CRS_MEMBER,), None, ["recipient"], _NAMING_SECOND, False, "crs",
        _SECOND, id="open-recipient-no-place",
    ),
    pytest.param(
        "aid", (_CRS_MEMBER,), _FROM_FRANCE, None, _EXCLUDING_SECOND, True, None,
        None, id="exclusion-changes-nothing",
    ),
]  # fmt: skip


@pytest.mark.parametrize(
    ("variable", "members", "entities", "open_roles", "sources", "opt_in", "series", "table"),
    _NO_DATA_ROWS,
)
def test_a_no_data_part_without_a_flagged_match_restates_the_table_its_run_settled(
    monkeypatch: pytest.MonkeyPatch,
    source_catalogue: None,
    variable: str,
    members: tuple[str, ...],
    entities: tuple[dict, ...] | None,
    open_roles: list[str] | None,
    sources: tuple[SourceMention, ...],
    opt_in: bool | None,
    series: str | None,
    table: str | None,
):
    """A part that finds no observation and holds no flagged match restates the table its
    run settled on: the one its elected measure names, or the one its words name. A part
    whose words exclude a table restates the exclusion and names no table. Posting the
    restatement back, to a model that recalls the members in reverse, refuses the same
    way, though another table holds a series the part could answer from."""
    if opt_in is not None:
        _use_second_table(monkeypatch, opt_in=opt_in)
    _hold_france_series_only_under(monkeypatch, series)
    graph = _graph(variable, members=members)
    graph._resolve["France"] = _FRA
    part = ParsedPart(
        variable_text=variable,
        entities=list(entities) if entities is not None else None,
        open_roles=open_roles,
        sources=list(sources),
    )

    (refused,) = _resolve_parsed(part, _ScriptedLLM(), graph=graph).parts

    assert isinstance(refused, NoDataPart)
    assert refused.no_data.flagged_match is None
    assert refused.restatement is not None
    assert _restated_table(refused.restatement) == table

    reversed_graph = _graph(variable, members=tuple(reversed(members)))
    reversed_graph._resolve["France"] = _FRA
    (reposted,) = _resolve_parsed(refused.restatement, _ScriptedLLM(), graph=reversed_graph).parts
    assert isinstance(reposted, NoDataPart)
    assert reposted.no_data.reason == refused.no_data.reason
    assert reposted.restatement == refused.restatement


# ---------------------------------------------------------------------------
# A named dataset binds the source slot and the slots its five-tuple pins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provenance", "member_count"),
    [(CRS_PROVENANCE, _PLAIN_CRS_EDUCATION_MEMBERS), (IMPUTED_PROVENANCE, IMPUTED_MEMBER_COUNT)],
    ids=["crs", "imputed"],
)
def test_a_named_dataset_binds_the_source_and_resolves_the_mints_family(
    monkeypatch: pytest.MonkeyPatch, provenance: str, member_count: int
):
    _name_dataset(monkeypatch, provenance)

    response = _resolve_scripted(f"{_E} to Kenya", _ScriptedLLM(_E), _E)

    (spec,) = _specs(response)
    assert spec.spec_id != EDUCATION_AID_SPEC_ID
    assert _bound(spec)["source"] == provenance
    identity = spec.definition.identity
    assert identity.facets is not None
    assert identity.facets["provenance"].ref == provenance
    assert identity_dataset(identity) == provenance
    _assert_search_family_is_the_mints(spec, member_count=member_count)


def test_the_sentence_and_the_trace_name_each_dataset_by_its_declared_label(
    monkeypatch: pytest.MonkeyPatch,
):
    graph = _graph(_E, provenance_labels=("GRAPH NAME FOR CRS", "GRAPH NAME FOR IMPUTED"))
    answered = [(_resolve_scripted(f"{_E} to Kenya", _ScriptedLLM(_E), graph=graph), _CRS_LABEL)]
    for provenance, label in ((CRS_PROVENANCE, _CRS_LABEL), (IMPUTED_PROVENANCE, _IMPUTED_LABEL)):
        _name_dataset(monkeypatch, provenance)
        answered.append((_resolve_scripted(f"{_E} to Kenya", _ScriptedLLM(_E), graph=graph), label))

    for response, label in answered:
        (spec,) = _specs(response)
        assert [source.label for source in spec.resolved.trace.resolved_sources] == [label]
        assert (spec.resolved.rendered_sentence or "").endswith(f" {label}.")
        assert "GRAPH NAME" not in spec.resolved.rendered_sentence


def test_a_dataset_that_publishes_no_recalled_shape_refuses_and_names_the_dataset(
    monkeypatch: pytest.MonkeyPatch,
):
    """The named dataset publishes none of the recalled shapes, so candidacy leaves the part
    nothing to elect. It refuses before any bind call, naming the dataset."""
    _name_dataset(monkeypatch, IMPUTED_PROVENANCE)
    variable = "education aid commitments"
    llm = _ScriptedLLM(variable)

    response = _resolve_scripted(
        f"{variable} to Kenya", llm, graph=_graph(variable, members=(_COMMITMENTS_MEMBER,))
    )

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
def test_every_draft_of_a_pinned_slot_ends_as_its_pinned_value(
    monkeypatch: pytest.MonkeyPatch, drafted
):
    """Every row of the imputed five-tuple holds one aid type and one flow, so the bind
    menu offers neither slot, and whatever the model drafts on either, the identity, which
    names the table, binds the declared value, and the part mints one identity. The
    sentence states neither, because the measure label already says imputed multilateral
    ODA, once."""
    _name_dataset(monkeypatch, IMPUTED_PROVENANCE)
    llm = _ScriptedLLM(_E, dropped=("every aid type",), drafted=drafted)

    response = _resolve_scripted(f"{_E} to Kenya, every aid type", llm, _E)

    (prompt,) = llm.bind_prompts
    assert "property_dcid=crsPurpose" in prompt
    assert f"property_dcid={AID_TYPE}" not in prompt
    assert "property_dcid=crsTypeOfFlow" not in prompt
    (spec,) = _specs(response)
    _assert_imputed_part(spec)
    assert _pinned(spec)["crsTypeOfFlow"] == "CRSTypeOfFlow10"
    sentence = spec.resolved.rendered_sentence
    assert sentence.casefold().count("imputed multilateral oda") == 1
    assert "official development assistance" not in sentence.casefold()


def test_each_pinned_slot_offers_the_menu_around_its_pin(monkeypatch: pytest.MonkeyPatch):
    """The imputed answer binds its two pinned slots to the table's invariants, and each
    slot's menu is the one the code index scopes around the bound value
    (`level_scoped_options`), less the options that serve no row. The flow system is flat,
    so its menu holds every flow, and every imputed row holds the invariant's value, so it
    alone is offered. The aid-type system is hierarchical, and aid type 106 is a root with
    no children, so the index scopes the other aid-type roots around it. No imputed row
    holds any of them, so each serves nothing and the slot offers no menu."""
    _name_dataset(monkeypatch, IMPUTED_PROVENANCE)

    response = _resolve_scripted(f"{_E} to Kenya", _ScriptedLLM(_E), _E)

    (spec,) = _specs(response)
    _assert_imputed_part(spec)
    pinned = _pinned(spec)
    assert set(pinned) == {AID_TYPE, "crsTypeOfFlow"}
    menus = {
        menu.key.property.dcid: [option.value.ref.dcid for option in menu.options]
        for menu in spec.resolved.slot_options
        if menu.key.property is not None and menu.key.property.dcid in pinned
    }
    assert menus == {"crsTypeOfFlow": [pinned["crsTypeOfFlow"]]}
    flows = get_index().level_scoped_options("crsTypeOfFlow", pinned["crsTypeOfFlow"])
    assert pinned["crsTypeOfFlow"] in {option.dcid for option in flows}
    assert len(flows) > 1
    aid_types = get_index().level_scoped_options(AID_TYPE, pinned[AID_TYPE])
    assert [option.dcid for option in aid_types] == [
        option.dcid
        for option in get_index().level_scoped_options(AID_TYPE, None)
        if option.relation == "root" and option.dcid != pinned[AID_TYPE]
    ]
    assert "DAC2aAidType206" in {option.dcid for option in aid_types}


@pytest.mark.parametrize("mode", ["log", "refuse"])
@pytest.mark.parametrize(
    ("provenance", "variable", "text", "named", "near"),
    [
        (
            IMPUTED_PROVENANCE,
            "grants",
            "grants",
            {"CRSTypeOfFinance100", "CRSTypeOfFinance110"},
            {},
        ),
        (
            CRS_PROVENANCE,
            "population ODA",
            "population",
            {"CRSTypeOfFinance4"},
            {"crsTypeOfFlow": "CRSTypeOfFlow10"},
        ),
        (
            CRS_PROVENANCE,
            "multilateral ODA",
            "multilateral ODA",
            {"DAC1AidType2000"},
            {"crsTypeOfFlow": "CRSTypeOfFlow10"},
        ),
    ],
    ids=["grants-on-imputed", "population-on-crs", "multilateral-on-crs"],
)
def test_a_term_the_named_dataset_cannot_apply_refuses_with_the_near_match(
    monkeypatch: pytest.MonkeyPatch,
    source_catalogue: None,
    search_settings,
    mode,
    provenance,
    variable,
    text,
    named,
    near,
):
    """A term no row of the named dataset holds goes through bind. Once the part's bindings
    are final, the term is an unbound record with evidence "unavailable", and the part
    refuses `qualifier_not_bound` with the elected answer as its near match, in either
    coverage mode, naming the dataset after the term. "grants" names types of finance no
    imputed row carries. "population" names `CRSTypeOfFinance4`, which no index row
    holds. "multilateral ODA" names DAC1 aid type 2000, of which the CRS reading holds only
    "ODA". The near match is the CRS ODA answer."""
    search_settings(qualifier_coverage=mode)
    _name_dataset(monkeypatch, provenance)

    response = _resolve_scripted(f"{variable} to Kenya", _ScriptedLLM(variable), variable)

    (part,) = response.parts
    assert isinstance(part, NoDataPart)
    assert part.no_data.reason == "qualifier_not_bound"
    term, source = part.no_data.unbound_qualifiers
    assert (term.text, term.disposition, term.evidence) == (text, "unbound", "unavailable")
    assert {ref.dcid for ref in term.named} == named
    assert [ref.dcid for ref in source.named] == [provenance]
    assert _bound(_held_spec(part)).items() >= near.items()


@pytest.mark.parametrize("named", [True, False], ids=["named", "unnamed"])
def test_a_bind_call_electing_no_shape_refuses_a_term_no_offered_shape_holds(
    monkeypatch: pytest.MonkeyPatch, source_catalogue: None, named
):
    """With no elected shape there are no final bindings to read "grants" on, so a part
    naming the imputed table, none of whose rows carries a type of finance, refuses the
    term `qualifier_not_bound` and names the table after it. A part naming no source
    refuses `variable_not_resolved`."""
    if named:
        _name_dataset(monkeypatch, IMPUTED_PROVENANCE)

    response = _resolve_scripted(
        "grants to Kenya", _ScriptedLLM("grants", ask="which grants?"), "grants"
    )

    (part,) = response.parts
    assert isinstance(part, NoDataPart)
    if not named:
        assert part.no_data.reason == "variable_not_resolved"
        return
    assert part.no_data.reason == "qualifier_not_bound"
    term, source = part.no_data.unbound_qualifiers
    assert (term.text, term.evidence) == ("grants", "unavailable")
    assert {ref.dcid for ref in term.named} == {"CRSTypeOfFinance100", "CRSTypeOfFinance110"}
    assert [ref.dcid for ref in source.named] == [IMPUTED_PROVENANCE]


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
    _name_dataset(monkeypatch, IMPUTED_PROVENANCE)
    variable = "budget support"
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [(_BUDGET_SUPPORT_MEMBER, "imputed", _FRA, _KEN, [2019, 2020, 2021])],
        role_names={_BUDGET_SUPPORT_MEMBER: names},
        facet_metadata={"imputed": FacetMetadata(provenance_id=IMPUTED_PROVENANCE, unit="USD")},
    )

    response = _resolve_scripted(
        f"{variable} to Kenya",
        _ScriptedLLM(variable),
        graph=_graph(variable, members=(_CRS_MEMBER, _BUDGET_SUPPORT_MEMBER)),
    )

    (spec,) = _specs(response)
    assert _bound(spec)["crsPurpose"] == "CRSPurposeCode510"
    assert _pinned(spec)[AID_TYPE] == IMPUTED_MULTILATERAL
    assert _sources(spec) == [IMPUTED_PROVENANCE]


_DAC2B_PROVENANCE = "provenance/OECDDAC2B"
_DAC2B_MEMBERS = tuple(
    row.dcid for row in get_index().statvars if _DAC2B_PROVENANCE in row.provenances
)


@pytest.mark.parametrize("basis", [None, "Gross"])
def test_an_open_dac2b_flow_type_reads_the_net_total_at_its_one_basis(
    monkeypatch: pytest.MonkeyPatch, basis: str | None
):
    """DAC2b's open flow type defaults to code 296, which the table publishes net alone. A
    question naming no basis reads 296 net disbursements, since the default binds before the
    measure vote. A question naming gross asks for a row the table does not hold and is not
    resolved."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [(dcid, "dac2b", _FRA, _KEN, [2019, 2020, 2021]) for dcid in _DAC2B_MEMBERS],
        role_names=dict.fromkeys(_DAC2B_MEMBERS, names),
        facet_metadata={"dac2b": FacetMetadata(provenance_id=_DAC2B_PROVENANCE, unit="USD")},
    )
    drafted = [
        SlotBindingDraft(axis="how", property_dcid="dac2bAidType", kind="unbound"),
        SlotBindingDraft(
            axis="how",
            property_dcid="financialFlowType",
            kind="value",
            value_dcids=["Disbursement"],
        ),
    ]
    if basis is not None:
        drafted.append(
            SlotBindingDraft(
                axis="how", property_dcid="nettingBasis", kind="value", value_dcids=[basis]
            )
        )
    variable = "official flows"

    response = _resolve_scripted(
        f"{variable} to Kenya",
        _ScriptedLLM(variable, drafted=tuple(drafted)),
        graph=_graph(variable, members=_DAC2B_MEMBERS),
    )

    (part,) = response.parts
    if basis == "Gross":
        assert isinstance(part, NoDataPart)
        assert part.no_data.reason == "variable_not_resolved"
        return
    (spec,) = _specs(response)
    assert _bound(spec)["dac2bAidType"] == "DAC2bAidType296"
    assert _bound(spec)["nettingBasis"] == "Net"
    assert spec.resolved.shape.member_count == 1
    assert "slot_defaulted" in {warning.code for warning in part.warnings}


@pytest.mark.parametrize("provenance", [CRS_PROVENANCE, IMPUTED_PROVENANCE], ids=["crs", "imputed"])
def test_a_part_leaving_the_recipient_open_states_the_family_the_mint_resolves(
    provenance: str, source_catalogue: None
):
    """ "Aid from France" asks every recipient, and its role-class election keeps the
    members that are no breakdown of another: one of the imputed table's 255, and every
    CRS member. The shape states the family the mint resolves for its identity, as the
    same part states it with the recipient bound."""
    graph = _graph("aid")
    graph._resolve["France"] = _FRA
    donor = {"text": "France", "role_text": "from"}
    recipient = {"text": "Kenya", "role_text": "to"}

    counts = []
    for entities in ([donor], [donor, recipient]):
        part = ParsedPart(variable_text="aid", entities=entities, provenance=provenance)
        (spec,) = _specs(_resolve_parsed(part, _ScriptedLLM(), graph=graph))
        family = _mint_family(spec)
        assert len(family) > 1
        assert spec.resolved.shape.member_count == len(family)
        assert spec.resolved.shape.member_digest == member_digest(family)
        counts.append(spec.resolved.shape.member_count)

    assert counts[0] == counts[1]


_BRA = "country/BRA"
_DAC2B_OOF = "Amount_FinancialFlow_Gross_Disbursement_DAC2bAidType972_Nominal"
_CRS_OOF = next(
    row.dcid
    for row in get_index().statvars
    if row.population_type == "FinancialFlow"
    and row.constraint_properties.get("crsTypeOfFlow") == "CRSTypeOfFlow21"
    and row.constraint_properties.get("nettingBasis") == "Gross"
)
_DAC2A_ODA = "Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal"
# A donor aggregate, standing for the all-donors total DAC2b publishes for a recipient.
_DONOR_AGGREGATE = "dac/AllDonors"
_PAIRED = ({"text": "Germany", "role_text": "from"}, {"text": "Brazil", "role_text": "to"})


def _menus(monkeypatch: pytest.MonkeyPatch) -> list[list[tuple[str, ...]]]:
    """Each part's bind menu, as the datasets of every shape `_phase_prune_and_rank` caps."""
    menus: list[list[tuple[str, ...]]] = []

    async def recorded(state: ResolveState):
        result = await _phase_prune_and_rank(state)
        menus.append([shape.datasets or () for shape in state.capped_shapes])
        return result

    monkeypatch.setattr(
        pipeline_module,
        "_PHASES",
        tuple(recorded if phase is _phase_prune_and_rank else phase for phase in _PHASES),
    )
    return menus


@pytest.mark.parametrize(
    ("variable", "recalled", "entities", "series", "menu"),
    [
        # dq-crs-028: both tables read "other official flows", and DAC2b holds Germany's
        # flows against aggregate recipients alone, so the CRS answers.
        pytest.param(
            "other official flows",
            (_DAC2B_OOF, _CRS_OOF),
            _PAIRED,
            [(_DAC2B_OOF, _DEU, _KEN), (_CRS_OOF, _DEU, _BRA)],
            [(CRS_PROVENANCE,)],
            id="dac2b-recalled-first",
        ),
        pytest.param(
            "other official flows",
            (_CRS_OOF, _DAC2B_OOF),
            _PAIRED,
            [(_DAC2B_OOF, _DEU, _KEN), (_CRS_OOF, _DEU, _BRA)],
            [(CRS_PROVENANCE,)],
            id="crs-recalled-first",
        ),
        # A recipient alone: DAC2b holds Brazil as a recipient of a donor aggregate, so both
        # stay. Only the CRS holds Brazil as a donor, which this part does not ask.
        pytest.param(
            "other official flows",
            (_DAC2B_OOF, _CRS_OOF),
            ({"text": "Brazil", "role_text": "to"},),
            [(_DAC2B_OOF, _DONOR_AGGREGATE, _BRA), (_CRS_OOF, _DEU, _BRA), (_CRS_OOF, _BRA, _KEN)],
            [(_DAC2B_PROVENANCE,), (CRS_PROVENANCE,)],
            id="recipient-alone",
        ),
        # Neither table holds the pair, so the rule stands down and the part refuses later.
        pytest.param(
            "other official flows",
            (_DAC2B_OOF, _CRS_OOF),
            _PAIRED,
            [(_DAC2B_OOF, _DEU, _KEN), (_CRS_OOF, _FRA, _BRA)],
            [(_DAC2B_PROVENANCE,), (CRS_PROVENANCE,)],
            id="neither-holds-the-pair",
        ),
        # Another pair of tables, the other way round: DAC2a holds the pair, the CRS not.
        pytest.param(
            "ODA",
            (_CRS_MEMBER, _DAC2A_ODA),
            _PAIRED,
            [(_DAC2A_ODA, _DEU, _BRA), (_CRS_MEMBER, _DEU, _KEN)],
            [("provenance/OECDDAC2A",)],
            id="dac2a-holds-the-pair",
        ),
    ],
)
def test_of_tables_reading_a_question_alike_one_without_series_at_its_places_leaves(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    recalled: tuple[str, str],
    entities: tuple[dict, ...],
    series: list[tuple[str, str, str]],
    menu: list[tuple[str, ...]],
):
    """Two families of one five-tuple and role class read the question's phrases alike. One
    holding no series at the places the part names, on the roles they bind, leaves when the
    other holds one (`elect_five_tuple_family`), whichever recall ranks first. With a series
    on both sides, or on neither, both reach bind."""
    patch_coverage(
        monkeypatch,
        [(dcid, dcid, donor, recipient, [2019, 2020]) for dcid, donor, recipient in series],
        role_names=dict.fromkeys(
            recalled, RoleNames(donor="financingProvider", recipient="financingRecipient")
        ),
    )
    menus = _menus(monkeypatch)
    graph = _graph(variable, members=recalled, countries={"Germany": _DEU, "Brazil": _BRA})

    _resolve_parsed(
        ParsedPart(variable_text=variable, entities=list(entities)), _ScriptedLLM(), graph=graph
    )

    assert menus == [menu]
