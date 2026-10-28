"""End-to-end tests for how an aid question elects its dataset.

The fixture index holds two datasets, each on its own five-tuple: the CRS on `amount`,
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
import json
import logging
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any, cast

import pytest

from code_index import shared as shared_module
from code_index.members import canonical_identity, elected_pins
from code_index.query import CodeIndex, row_five_tuple
from code_index.shared import get_index
from code_index.types import DatasetDeclaration
from datasets_api.adapter import resolve_definition_members
from dc_graph.client import FacetDefaultPolicy
from entity_resolution import EntityResolutionError
from materialise.errors import UpstreamVersionChangedError
from materialise.rate_tables import RateTableInvalidError
from origin_catalogue import DFO, Catalogue, ProvenanceRow, SourceRef
from search.engine import degraded_reads as degraded_reads_module
from search.engine.bind import ShapeRating, SlotBindingDraft, _UniversalBindOutput
from search.engine.binders import Binder, GeminiBinder
from search.engine.errors import (
    DeclaredDefaultUnsupportedError,
    DefaultScopeIndexSourcedError,
    GraphInfraError,
)
from search.engine.extract import ExtractedEntity, ExtractedOperation, Extraction
from search.engine.quantity_read import QuantityOutput
from search.engine.region import phase_rank_bind as rank_bind_module
from search.engine.region import pipeline as pipeline_module
from search.engine.region.phase_rank_bind import _phase_prune_and_rank
from search.engine.region.state import ResolveState
from search.engine.retrieval.bound_values import identity_dataset, identity_query
from search.engine.spec_id import mint_identity
from search.engine.stated_words_read import StatedWordsOutput
from search.models import (
    CandidatesPart,
    DefinitePart,
    NoDataPart,
    Operation,
    ParsedInput,
    ParsedPart,
    RawTextInput,
    ResolveRequest,
    ResolveResponse,
    SourceMention,
    Spec,
)
from search.models.response import Part
from series_coverage.index import ABSENT_ENTITY
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
from tests.engine._binders import RatingBinder, ReplyBinder, nominee_ratings
from tests.engine._fake_coverage import patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_events, offline_resolve
from tests.engine._quantity import NO_QUANTITY, NO_RATING
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
# The imputed member for the same purpose, read from the fixture index so its dcid
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
# The CRS education members under official development assistance, the flow the CRS
# declares as its default.
_ODA_CRS_EDUCATION_MEMBERS = 257


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


def _un_node(dcid: str, *, name: str | None = None) -> dict:
    return {
        "label": dcid,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": name or dcid}]},
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
        (slot.key.property.ref if slot.key.property else slot.key.axis): slot.binding.value.ref.ref
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
    return [source.ref for source in spec.resolved.trace.resolved_sources]


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
    assert llm.calls == {"Extraction": 1, "_UniversalBindOutput": 1, "QuantityOutput": 1}
    assert spec.resolved.shape.member_count == member_count
    assert AID_TYPE not in _bound(spec)
    assert "source" not in _bound(spec)
    plain = education_aid_identity() if variable == _E else _plain_aid_identity()
    if open_roles is not None:
        plain["open_roles"] = open_roles
    assert spec.definition.identity.open_roles == open_roles
    assert spec.spec_id == _spec_id(plain)


def test_a_recipient_total_on_a_crs_only_shape_is_counted_on_the_crs_convention():
    """A recipient's all-sector total is counted net by DAC2a's convention, which applies
    only to a shape DAC2a publishes. The shape here is the CRS's alone, so the part takes the
    CRS's own convention, gross disbursements, and its choice names that convention as the
    rule that settled the basis."""
    response, _llm = _resolve("total aid to Kenya", "aid")

    (spec,) = _specs(response)
    assert _bound(spec)["nettingBasis"] == "Gross"
    scope = spec.resolved.scope_choices
    assert scope is not None
    (choice,) = (c for c in scope.choices if c.axis == "measure_basis")
    assert (choice.active, choice.provenance, choice.rule) == (
        "measure_basis:gross",
        "default",
        "measure_basis_convention",
    )


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
    """The fixture index with `_SECOND` declared beside the CRS on the aid five-tuple."""
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
    """Serve `_with_second_table` wherever a loaded module reads the fixture index, and
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
            [(ref.ref, ref.label) for ref in unbound.named]
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


_MULTILATERALS = "DACMultilaterals"


def _party_graph(
    variable: str, without_word: str, *, typed: bool = True, dac2a_first: bool = False
) -> FakeGraph:
    """A graph whose search finds the CRS alone for `variable`, or DAC2a before it with
    `dac2a_first`, and DAC2a before the CRS for `without_word`, the text recall searches
    once "multilateral" is taken out. It holds the multilaterals total as an organisation
    unless `typed` is false."""
    recalled = _graph(without_word, members=(_DAC2A_MEMBER, _CRS_MEMBER))
    party = {_MULTILATERALS: {"label": "multilaterals (DAC)", "type": "Organization"}}
    found = [_DAC2A_MEMBER, _CRS_MEMBER] if dac2a_first else [_CRS_MEMBER]
    return FakeGraph(
        nodes={**recalled._nodes, **(party if typed else {}), _DAC2A: {"label": "OECD DAC2a"}},
        detect={
            **recalled._detect,
            variable: {
                "svs": found,
                "cosine_scores": [0.9 - 0.05 * rank for rank in range(len(found))],
            },
        },
        resolve={"Kenya": _KEN},
        obs={},
    )


def _party_coverage(monkeypatch: pytest.MonkeyPatch, *, from_the_party: bool = True) -> None:
    """DAC2a holds series to Kenya from France and, unless told otherwise, from the
    multilaterals total, and the CRS from France."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    party = [(_DAC2A_MEMBER, "dac2a", _MULTILATERALS, _KEN, [2019, 2020, 2021])]
    patch_coverage(
        monkeypatch,
        [
            *(party if from_the_party else []),
            (_DAC2A_MEMBER, "dac2a", _FRA, _KEN, [2019, 2020, 2021]),
            (_CRS_MEMBER, "crs", _FRA, _KEN, [2019, 2020, 2021]),
        ],
        role_names=dict.fromkeys((_DAC2A_MEMBER, _CRS_MEMBER), names),
        facet_metadata={
            "dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD"),
            "crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD"),
        },
    )


@pytest.fixture
def party_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    _party_coverage(monkeypatch)


@pytest.mark.parametrize(
    ("variable", "without_word", "source", "providers"),
    [
        pytest.param("multilateral ODA", "ODA", _DAC2A, [_MULTILATERALS], id="party"),
        pytest.param("ODA", "ODA", None, None, id="no-word"),
    ],
)
def test_a_word_naming_a_party_answers_from_the_table_declaring_it(
    party_coverage: None, variable, without_word, source, providers
):
    """The graph's search finds the CRS alone for "multilateral ODA" and DAC2a for "ODA".
    DAC2a declares that "multilateral" names its provider `DACMultilaterals`, so recall
    adds its series, the DAC2a shape binds that party beside Kenya and reads all the CRS
    reads, and the part answers from DAC2a with the multilaterals total as its provider
    and departs nothing. Without the word no provider is bound."""
    response = _resolve_scripted(
        f"{variable} to Kenya", _ScriptedLLM(variable), graph=_party_graph(variable, without_word)
    )

    (spec,) = _specs(response)
    assert spec.definition.identity.entities.get("origin") == providers
    assert spec.definition.identity.entities["destination"] == [_KEN]
    if source is not None:
        assert _sources(spec) == [source]
        assert _unapplied(response) == []


def _reply(response: ResolveResponse) -> dict:
    """`response` as the wire carries it, less its timings and trace id."""

    def scrubbed(node: object) -> object:
        if isinstance(node, dict):
            return {
                key: scrubbed(value)
                for key, value in node.items()
                if key not in ("timing_ms", "trace_id")
            }
        return [scrubbed(item) for item in node] if isinstance(node, list) else node

    return cast(dict, scrubbed(response.model_dump(mode="json")))


@pytest.mark.parametrize("dac2a_first", [False, True], ids=["crs-first", "dac2a-first"])
@pytest.mark.parametrize(
    ("variable", "named", "typed", "from_the_party"),
    [
        pytest.param("multilateral education aid", (), True, True, id="sector"),
        pytest.param("multilateral aid for education", (), True, True, id="for-sector"),
        pytest.param("multilateral development bank ODA", (), True, True, id="a-kind-of-lender"),
        pytest.param("multilateral aid", ("CRS",), True, True, id="a-table-declaring-no-party"),
        pytest.param("multilateral aid", (), False, True, id="party-left-the-graph"),
        pytest.param("multilateral ODA", (), True, False, id="no-series-from-the-party"),
    ],
)
def test_a_party_word_that_binds_no_party_leaves_the_reply_as_without_the_rule(
    monkeypatch: pytest.MonkeyPatch,
    source_catalogue: None,
    variable,
    named,
    typed,
    from_the_party,
    dac2a_first,
):
    """A part stating "multilateral" either answers from the multilaterals total or replies
    as an index declaring no party word does, whichever shape the search ranks first for
    the scripted model to elect. DAC2a reads no sector and no "development bank", the CRS
    declares no such party, a party the graph gives no type has no set, and a DAC2a holding
    no series from the party answers nothing: each reply equals the one read without
    party words, and none holds the party."""
    _party_coverage(monkeypatch, from_the_party=from_the_party)
    without_word = variable.removeprefix("multilateral ")
    query = _query(variable, tuple(_naming(text) for text in named))

    def resolve() -> dict:
        graph = _party_graph(variable, without_word, typed=typed, dac2a_first=dac2a_first)
        return _reply(_resolve_scripted(query, _ScriptedLLM(variable), graph=graph))

    reply = resolve()
    with monkeypatch.context() as patched:
        patched.setattr(pipeline_module, "stated_party_words", lambda state: ())
        assert reply == resolve()
    assert _MULTILATERALS not in json.dumps(reply["parts"])


_DAC1 = "provenance/OECDDAC1"
# DAC1 aid type 2100, "multilateral contribution", from a donor to all recipients.
_DAC1_CONTRIBUTIONS = "Amount_FinancialFlow_Gross_Disbursement_DAC1AidType2100_Nominal"


class _FailingGraph(FakeGraph):
    """A graph whose read `fails` raises, as a failed transport does: the party's type read,
    or the search of the text without the party word."""

    def __init__(self, *args: Any, fails: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fails = fails

    def node_types_batch(self, dcids):
        if self._fails == "party-type" and _MULTILATERALS in dcids:
            raise GraphInfraError("party type read failed")
        return super().node_types_batch(dcids)

    def detect_svs(self, text, *args, **kwargs):
        if self._fails == "search-without-the-word" and text == "ODA":
            raise GraphInfraError("search failed")
        return super().detect_svs(text, *args, **kwargs)


# A default unit the graph declares on the DAC2a member alone.
_DAC2A_DEFAULT_UNIT = {
    (f"dcid:{_DAC2A_MEMBER}", "unit"): FacetDefaultPolicy(
        resolved_ref="USD", policy_node="p", policy_version="1", selection_reason="r"
    )
}


def _refusing_party_reading(refusal: Exception) -> Callable[..., list]:
    """`with_party_reading` raising `refusal` on a part read with a party."""
    elect = rank_bind_module.with_party_reading

    def refusing(state, shapes, phrases_read):
        if state.party_sets:
            raise refusal
        return elect(state, shapes, phrases_read)

    return refusing


@pytest.mark.parametrize(
    ("fails", "noted_as"),
    [
        pytest.param("party-type", ["party reading"], id="party-type-read"),
        pytest.param("search-without-the-word", ["party reading"], id="search-without-the-word"),
        pytest.param("party-table-declares-a-default", [], id="party-table-declares-a-default"),
        pytest.param(
            DeclaredDefaultUnsupportedError("default"), [], id="declared-default-unsupported"
        ),
        pytest.param(DefaultScopeIndexSourcedError("scope"), [], id="default-scope-index-sourced"),
        pytest.param(RateTableInvalidError("table"), [], id="rate-table-invalid"),
        pytest.param(UpstreamVersionChangedError("moved"), [], id="upstream-version-changed"),
        pytest.param(ValueError("defect"), [], id="defect"),
    ],
)
def test_a_party_reading_that_fails_or_refuses_leaves_the_base_reply(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    party_coverage: None,
    fails,
    noted_as,
):
    """The party's type read and the search of the text without the word are made only when
    the part is read with the party, and a refusal can come from the shape only that reading
    elects, as DAC2a declaring a default unit the CRS does not. Any of them withdraws the
    party reading: the reply equals the one read without party words, a failed read is
    noted as degraded, and a refusal is not. Any other exception is a defect, logged with
    its traceback."""
    noted: list[str] = []
    note = pipeline_module.note_degraded_read

    def noting(read: str) -> None:
        noted.append(read)
        note(read)

    monkeypatch.setattr(pipeline_module, "note_degraded_read", noting)
    if isinstance(fails, Exception):
        monkeypatch.setattr(rank_bind_module, "with_party_reading", _refusing_party_reading(fails))

    def resolve() -> dict:
        recalled = _party_graph("multilateral ODA", "ODA")
        graph = _FailingGraph(
            nodes=recalled._nodes,
            detect=recalled._detect,
            resolve={"Kenya": _KEN},
            obs={},
            facet_policies=_DAC2A_DEFAULT_UNIT
            if fails == "party-table-declares-a-default"
            else None,
            fails=fails if isinstance(fails, str) else "",
        )
        query = "multilateral ODA to Kenya"
        return _reply(_resolve_scripted(query, _ScriptedLLM("multilateral ODA"), graph=graph))

    with caplog.at_level(logging.WARNING, logger=pipeline_module.__name__):
        reply = resolve()
    assert noted == noted_as
    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    if isinstance(fails, ValueError):
        (error,) = errors
        assert error.exc_info is not None and error.exc_info[1] is fails
    else:
        assert errors == []
    if fails == "party-table-declares-a-default":
        assert [part["status"] for part in reply["parts"]] == ["no_data"]
    with monkeypatch.context() as patched:
        patched.setattr(pipeline_module, "stated_party_words", lambda state: ())
        assert reply == resolve()


def test_a_withdrawn_party_reading_carries_the_reads_it_swallowed(
    monkeypatch: pytest.MonkeyPatch, party_coverage: None
):
    """A read the party reading swallowed before it was withdrawn at election marks the
    reply degraded: the resolution without party words notes it again."""
    noted: list[str] = []
    note = pipeline_module.note_degraded_read

    def noting(read: str) -> None:
        noted.append(read)
        note(read)

    elect = rank_bind_module.with_party_reading

    def swallowing(state, shapes, phrases_read):
        degraded_reads_module.note_degraded_read("a swallowed read")
        return elect(state, shapes, phrases_read)

    monkeypatch.setattr(pipeline_module, "note_degraded_read", noting)
    monkeypatch.setattr(rank_bind_module, "with_party_reading", swallowing)
    variable = "multilateral education aid"

    _resolve_scripted(
        f"{variable} to Kenya",
        _ScriptedLLM(variable),
        graph=_party_graph(variable, "education aid"),
    )

    assert noted == ["a swallowed read"]


@pytest.mark.parametrize(
    ("variable", "from_the_party"),
    [
        pytest.param("multilateral ODA", True, id="stands"),
        pytest.param("multilateral ODA", False, id="withdrawn-after-the-answer"),
        pytest.param("multilateral education aid", True, id="withdrawn-at-election"),
        pytest.param("ODA", True, id="no-party-word"),
    ],
)
def test_the_reply_reports_the_model_usage_of_every_reading(
    monkeypatch: pytest.MonkeyPatch, variable, from_the_party
):
    """A withdrawn party reading spends model calls before the part is read again, and the
    reply reports them with the second reading's: one token per call made."""
    _party_coverage(monkeypatch, from_the_party=from_the_party)
    llm = _CountingLLM(variable)
    graph = _party_graph(variable, variable.removeprefix("multilateral "))

    response = _resolve_scripted(f"{variable} to Kenya", llm, graph=graph)

    usage = response.diagnostics.llm_usage
    assert usage is not None
    assert usage["input_tokens"] == usage["output_tokens"] == sum(llm.calls.values())


# The words of "contributions to multilateral organisations" the coverage rule leaves for
# the binder to rate on DAC1's multilateral contributions, whose graph name is its dcid.
_CONTRIBUTION_WORDS = ("contributions", "multilateral", "organisations")


@pytest.mark.parametrize(
    ("variable", "entities", "drafted"),
    [
        pytest.param(
            "contributions to multilateral organisations",
            (
                ExtractedEntity(
                    text="Germany", span="Germany", flow_role="origin", role_span="from"
                ),
            ),
            (
                SlotBindingDraft(
                    axis="how",
                    property_dcid="dac1AidType",
                    kind="value",
                    value_dcids=["DAC1AidType2100"],
                ),
            ),
            id="the-phrase-in-the-variable",
        ),
        pytest.param(
            "contributions",
            (
                ExtractedEntity(
                    text="multilateral organisations",
                    span="multilateral organisations",
                    flow_role="destination",
                    role_span="to",
                ),
                ExtractedEntity(
                    text="Germany", span="Germany", flow_role="origin", role_span="from"
                ),
            ),
            (),
            id="the-phrase-as-a-recipient",
        ),
    ],
)
def test_contributions_to_multilateral_organisations_keep_their_dac1_reading(
    monkeypatch: pytest.MonkeyPatch, variable, entities, drafted
):
    """ "Contributions to multilateral organisations from Germany" states no party word:
    "multilateral" follows a preposition. Kept in the variable, the phrase reads DAC1's
    multilateral contributions from Germany, and the reply drafts the aid type the phrase
    names, as a live binder does; a reply leaving the slot open binds DAC1's declared
    default aid type, which the member does not hold. Lifted as a recipient, the alias
    names the multilaterals total as the recipient. Either reply equals the one read
    without party words, so the party reading leaves the question alone. Kept in the
    variable, the bound aid type's label reads "contributions" and "multilateral", and the
    binder rates "organisations", the one word no reader reads, stating on that series."""
    patch_coverage(
        monkeypatch,
        [(_DAC1_CONTRIBUTIONS, "dac1", _DEU, ABSENT_ENTITY, [2019, 2020, 2021])],
        role_names={_DAC1_CONTRIBUTIONS: RoleNames(donor="financingProvider", recipient=None)},
        facet_metadata={"dac1": FacetMetadata(provenance_id=_DAC1, unit="USD")},
    )

    def resolve() -> ResolveResponse:
        graph = FakeGraph(
            nodes={
                _DEU: {"label": "Germany", "type": "Country"},
                _MULTILATERALS: {"label": "multilaterals (DAC)", "type": "Organization"},
                _DAC1: {"label": "OECD DAC1"},
                _DAC1_CONTRIBUTIONS: {
                    "label": _DAC1_CONTRIBUTIONS,
                    "type": "StatisticalVariable",
                    "arcs": _arcs(_DAC1_CONTRIBUTIONS),
                },
            },
            detect={variable: {"svs": [_DAC1_CONTRIBUTIONS], "cosine_scores": [0.9]}},
            resolve={"Germany": _DEU},
            obs={},
        )
        llm = _ScriptedLLM(variable, entities=entities, drafted=drafted)
        return _resolve_scripted(
            "contributions to multilateral organisations from Germany",
            llm,
            graph=graph,
            binder=binder,
        )

    binder = RatingBinder(GeminiBinder(), dict.fromkeys(_CONTRIBUTION_WORDS, 1.0))
    response = resolve()
    with monkeypatch.context() as patched:
        patched.setattr(pipeline_module, "stated_party_words", lambda state: ())
        assert _reply(response) == _reply(resolve())
    if len(entities) == 1:
        (spec,) = _specs(response)
        assert _sources(spec) == [_DAC1]
        assert spec.definition.identity.entities == {"origin": [_DEU]}
        assert {request.words for request in binder.requests} == {("organisations",)}


@pytest.mark.parametrize(
    "recipient", ["multilaterals", "multilateral institutions", "multilateral organisations"]
)
def test_an_alias_of_the_party_reads_as_its_name_does(
    monkeypatch: pytest.MonkeyPatch, party_coverage: None, recipient
):
    """ "Germany's ODA to multilateral institutions" names the multilaterals total as the
    recipient through an alias, exactly as "Germany's ODA to multilaterals" does through the
    party's own alias: the same outcome and the same identity."""

    def outcome(text: str) -> tuple:
        llm = _ScriptedLLM(
            "ODA",
            entities=(
                ExtractedEntity(text="Germany", span="Germany", flow_role="origin", role_span="'s"),
                ExtractedEntity(text=text, span=text, flow_role="destination", role_span="to"),
            ),
        )
        graph = _party_graph("ODA", "ODA")
        graph._resolve = {**graph._resolve, "Germany": _DEU}
        graph._nodes = {**graph._nodes, _DEU: {"label": "Germany", "type": "Country"}}
        (part,) = _resolve_scripted(f"Germany's ODA to {text}", llm, graph=graph).parts
        held = (
            _held_spec(part)
            if not isinstance(part, NoDataPart) or part.no_data.flagged_match
            else None
        )
        return (
            type(part).__name__,
            getattr(getattr(part, "no_data", None), "reason", None),
            None if held is None else dict(held.definition.identity.entities),
        )

    assert outcome(recipient) == outcome("multilaterals")


def test_a_party_word_keeps_its_party_under_an_operation_search_does_not_run(
    party_coverage: None,
):
    """ "Multilateral ODA as a share of total ODA to Kenya" asks a share of two amounts.
    The share runs across no axis, so the first part still answers the multilaterals total
    and departs the operation, and the second answers every provider."""
    graph = _party_graph("multilateral ODA", "ODA")
    graph._detect["total ODA"] = graph._detect["ODA"]
    llm = _ScriptedLLM(
        "multilateral ODA",
        "total ODA",
        operations=(ExtractedOperation(verb="share_of", variable=0, phrase="as a share of"),),
    )

    response = _resolve_scripted(
        "multilateral ODA as a share of total ODA to Kenya", llm, graph=graph
    )

    share, total = _specs(response)
    assert share.definition.identity.entities.get("origin") == [_MULTILATERALS]
    assert total.definition.identity.entities.get("origin") is None
    assert [[departure.code for departure in part.departures or ()] for part in response.parts] == [
        ["operation_not_applied"],
        [],
    ]
    assert share.definition.identity.shape.measurement_denominator is None


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


_TO_KENYA = ExtractedEntity(text="Kenya", span="Kenya", flow_role="destination", role_span="to")


class _ScriptedLLM:
    """Extract the given variables to Kenya, or among `entities`, and elect the first
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
        operations: tuple[ExtractedOperation, ...] = (),
        entities: tuple[ExtractedEntity, ...] = (_TO_KENYA,),
    ) -> None:
        self._variables = list(variables)
        self._entities = list(entities)
        self._operations = list(operations)
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
                entities=self._entities,
                dropped_qualifiers=self._dropped,
                operations=self._operations,
            ), None
        if schema is StatedWordsOutput:
            return NO_RATING
        if schema is QuantityOutput:
            return NO_QUANTITY
        self.bind_prompts.append(prompt)
        return schema(
            chosen_shape_index=0,
            bindings=self._drafted,
            also_plausible=self._also_plausible,
            ask=self._ask,
            measuring=[
                ShapeRating(index=0, rating=1.0),
                *nominee_ratings(self._also_plausible, rated=(0,)),
            ],
        ), None


class _TypeCountingGraph(FakeGraph):
    """A graph counting the reads of the multilaterals total's type alone, which a pass
    reading the part with the party makes once."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.party_type_reads = 0

    def node_types_batch(self, dcids):
        self.party_type_reads += list(dcids) == [_MULTILATERALS]
        return super().node_types_batch(dcids)


def test_the_un_pass_reads_without_the_party_word(monkeypatch: pytest.MonkeyPatch):
    """The UN pass binds no party, so it reads a part as one stating no party word. Here
    our pass's party reading is withdrawn and the UN series answers; the party's type is
    read once, by that reading."""
    _party_coverage(monkeypatch, from_the_party=False)
    recalled = _party_graph("multilateral ODA", "ODA")
    un = _graph("multilateral ODA", "ODA", members=(), un_members=(_UN_ODA,))
    graph = _TypeCountingGraph(
        nodes={**recalled._nodes, **un._nodes},
        detect=recalled._detect,
        un_detect=un._un_detect,
        resolve={"Kenya": _KEN},
        obs=un._obs,
    )

    _resolve_scripted("multilateral ODA to Kenya", _ScriptedLLM("multilateral ODA"), graph=graph)

    assert graph.party_type_reads == 1


def test_a_withdrawn_party_reading_reports_the_usage_of_its_un_pass(
    monkeypatch: pytest.MonkeyPatch,
):
    """A party reading withdrawn after its UN pass made model calls reports them with the
    rest, one token per call made, when a later step fails."""
    _party_coverage(monkeypatch, from_the_party=False)
    recalled = _party_graph("multilateral ODA", "ODA")
    un = _graph("multilateral ODA", "ODA", members=(), un_members=(_UN_ODA,))
    graph = FakeGraph(
        nodes={**recalled._nodes, **un._nodes},
        detect=recalled._detect,
        un_detect=un._un_detect,
        resolve={"Kenya": _KEN},
        obs=un._obs,
    )
    warn = pipeline_module.with_un_unavailable_warning
    calls = 0

    def failing_once(result, blocked):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("defect after the UN pass")
        return warn(result, blocked)

    monkeypatch.setattr(pipeline_module, "with_un_unavailable_warning", failing_once)
    llm = _CountingLLM("multilateral ODA")
    binder = RatingBinder(GeminiBinder(), {"multilateral": 0.0})

    response = _resolve_scripted("multilateral ODA to Kenya", llm, graph=graph, binder=binder)

    usage = response.diagnostics.llm_usage
    assert usage is not None
    assert usage["input_tokens"] == sum(llm.calls.values())


class _PoolRatingBinder(RatingBinder):
    """`RatingBinder` rating each word from `stated_by_pool` under the pool the request
    reads, 1.0 where the pool's ratings hold no word."""

    def __init__(self, binder: Binder, stated_by_pool: Mapping[str, Mapping[str, float]]):
        super().__init__(binder)
        self.stated_by_pool = stated_by_pool

    async def rate_words_stated(self, request, *, llm):
        self.requests.append(request)
        stated = self.stated_by_pool.get(request.pool, {})
        return {word: stated.get(word, 1.0) for word in request.words}, None


@pytest.mark.parametrize(
    ("pool", "states_the_word"),
    [
        pytest.param("un_system", False, id="unread-on-the-un-series"),
        pytest.param("un_system", True, id="stated-by-the-un-series"),
        pytest.param("ours", False, id="unread-on-our-series"),
        pytest.param("ours", True, id="stated-by-our-series"),
    ],
)
def test_a_declared_default_refuses_the_request_only_on_a_series_the_part_answers_on(
    monkeypatch: pytest.MonkeyPatch, pool, states_the_word
):
    """The series `pool` elects for "multilateral aid" declares a default unit: the UN
    total-ODA series, or the CRS series our pass elects. A UN pass reads the part without
    the party word, so the party reading's UN pass rates the word as the reading without
    party words does. Where the defaulted series leaves the word unread, as the binder
    rates it, its pass answers nothing on it and refuses `qualifier_not_bound` before the
    default can refuse the request (`pipeline._run_phases`), and the reading without party
    words refuses on the word. With the UN series defaulted, our pass refuses on the word
    too, and of the two refusals ours stands (`pipeline._fill_gap`). Where the defaulted
    series states the word, its pass answers on it and the declared default refuses the
    whole request."""
    _party_coverage(monkeypatch, from_the_party=False)
    unread = {"multilateral": 0.0}
    stated_by_pool = {"ours": unread, "un_system": unread, pool: {} if states_the_word else unread}

    def resolve(*, declares_default: bool) -> ResolveResponse:
        recalled = _party_graph("multilateral aid", "aid")
        un_members = (_UN_ODA,) if pool == "un_system" else ()
        un = _graph("multilateral aid", "aid", members=(), un_members=un_members)
        defaulted = _UN_ODA if pool == "un_system" else _CRS_MEMBER
        default_unit = FacetDefaultPolicy(
            resolved_ref="USD", policy_node="p", policy_version="1", selection_reason="r"
        )
        graph = FakeGraph(
            nodes={**recalled._nodes, **un._nodes},
            detect=recalled._detect,
            un_detect=un._un_detect,
            resolve={"Kenya": _KEN},
            obs=un._obs,
            facet_policies={(f"dcid:{defaulted}", "unit"): default_unit}
            if declares_default
            else None,
        )
        return _resolve_scripted(
            "multilateral aid to Kenya",
            _ScriptedLLM("multilateral aid"),
            graph=graph,
            binder=_PoolRatingBinder(GeminiBinder(), stated_by_pool),
        )

    if states_the_word:
        with pytest.raises(DeclaredDefaultUnsupportedError):
            resolve(declares_default=True)
        return
    (part,) = resolve(declares_default=True).parts
    assert isinstance(part, NoDataPart)
    assert part.no_data.reason == "qualifier_not_bound"
    assert [term.text for term in part.no_data.unbound_qualifiers] == ["multilateral"]
    # The refusal flags the series the part refuses on where no series declares a default.
    (undefaulted,) = resolve(declares_default=False).parts
    assert isinstance(undefaulted, NoDataPart)
    assert undefaulted.no_data.reason == "qualifier_not_bound"
    assert _held_spec(part).spec_id == _held_spec(undefaulted).spec_id


def _stream(query: str, variable: str) -> list[dict]:
    """The events `query` streams, less their timings."""

    def scrubbed(node: object) -> object:
        if isinstance(node, dict):
            return {key: scrubbed(value) for key, value in node.items() if key not in _UNTIMED}
        if isinstance(node, list):
            return [scrubbed(value) for value in node]
        return node

    events = offline_events(
        make_request(query),
        graph=_party_graph(variable, variable.removeprefix("multilateral ")),
        llm=_ScriptedLLM(variable),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    return [cast(dict, scrubbed(event.model_dump(mode="json"))) for event in events]


# Event fields that vary from run to run.
_UNTIMED = frozenset({"ms", "elapsed_ms", "timing_ms", "trace_id"})


def test_a_withdrawn_party_reading_streams_none_of_its_events(
    monkeypatch: pytest.MonkeyPatch, party_coverage: None
):
    """ "Multilateral education aid to Kenya" is read with the party first and withdrawn: the
    CRS reads the sector DAC2a does not. Its stream is the stream of the question read
    without party words, event for event."""
    query = "multilateral education aid to Kenya"
    streamed = _stream(query, "multilateral education aid")
    with monkeypatch.context() as patched:
        patched.setattr(pipeline_module, "stated_party_words", lambda state: ())
        assert streamed == _stream(query, "multilateral education aid")


def test_a_party_reading_that_stands_streams_its_events_in_order(party_coverage: None):
    """ "Multilateral ODA to Kenya" answers with the party bound, and streams the held events
    of that reading: its shape, then the sentence of the Spec the reply holds."""
    streamed = _stream("multilateral ODA to Kenya", "multilateral ODA")

    (reply,) = [event["response"] for event in streamed if event["kind"] == "response"]
    (spec,) = [part["spec"] for part in reply["parts"]]
    assert spec["definition"]["identity"]["entities"]["origin"] == [_MULTILATERALS]
    kinds = [event["kind"] for event in streamed if event.get("variable_index") == 0]
    (sentence,) = [event for event in streamed if event["kind"] == "sentence"]
    assert sentence["spec_id"] == spec["spec_id"]
    assert kinds.index("shape") < kinds.index("sentence")
    assert streamed[-2]["kind"] == "response"


class _WithholdingGraph(FakeGraph):
    """A graph whose registry withholds every series it is asked about, under one source,
    and which records each re-read."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.withheld_reads: list[tuple[list[str], list[str]]] = []

    def withheld_provenances(self, variables, entities, *, deadline=None):
        self.withheld_reads.append((list(variables), list(entities)))
        return (_WITHHELD_SOURCE,)


_WITHHELD_SOURCE = "dc/base/Excluded"
# A subject-only aid series the search finds, with no admitted data at Kenya.
_SUBJECT_ONLY = "dc/SubjectOnlyAidSeries"


def test_a_withdrawn_party_reading_leaves_the_withheld_re_read_to_the_answer(
    monkeypatch: pytest.MonkeyPatch,
):
    """The graph gives the multilaterals total no type and the search finds only a
    subject-only series whose data the registry withholds at Kenya. The party reading binds
    no party and is withdrawn, and the reply equals the one read without party words,
    `withheld_by_registry` and its departure included: the re-read the request makes once is
    the answering reading's."""
    _party_coverage(monkeypatch, from_the_party=False)

    def resolve() -> tuple[dict, _WithholdingGraph]:
        graph = _WithholdingGraph(
            nodes={
                _SUBJECT_ONLY: _un_node(_SUBJECT_ONLY, name="Multilateral ODA received"),
                _KEN: {"label": "Kenya", "type": "Country"},
                _WITHHELD_SOURCE: {"label": "Excluded Survey"},
            },
            detect={
                text: {"svs": [_SUBJECT_ONLY], "cosine_scores": [0.9]}
                for text in ("multilateral ODA", "ODA")
            },
            resolve={"Kenya": _KEN},
            obs={},
        )
        llm = _ScriptedLLM("multilateral ODA")
        return _reply(_resolve_scripted("multilateral ODA to Kenya", llm, graph=graph)), graph

    reply, graph = resolve()
    (part,) = reply["parts"]
    assert "withheld_by_registry" in [warning["code"] for warning in part["warnings"]]
    assert graph.withheld_reads == [([_SUBJECT_ONLY], [_KEN])]
    with monkeypatch.context() as patched:
        patched.setattr(pipeline_module, "stated_party_words", lambda state: ())
        assert reply == resolve()[0]


def test_a_party_reading_that_stands_makes_no_withheld_re_read(party_coverage: None):
    """A party reading that stands answers with the party bound, which the withheld re-read
    never serves, so the reading makes none."""
    recalled = _party_graph("multilateral ODA", "ODA")
    graph = _WithholdingGraph(
        nodes=recalled._nodes, detect=recalled._detect, resolve={"Kenya": _KEN}, obs={}
    )

    response = _resolve_scripted(
        "multilateral ODA to Kenya", _ScriptedLLM("multilateral ODA"), graph=graph
    )

    (spec,) = _specs(response)
    assert spec.definition.identity.entities["origin"] == [_MULTILATERALS]
    assert graph.withheld_reads == []


class _CountingLLM(_ScriptedLLM):
    """`_ScriptedLLM` reporting one input and one output token per call."""

    async def generate_structured(self, *, prompt, system, schema):
        answer, _usage = await super().generate_structured(
            prompt=prompt, system=system, schema=schema
        )
        return answer, {"input_tokens": 1, "output_tokens": 1, "cached_tokens": 0}


def _resolve_scripted(
    query: str,
    llm: _ScriptedLLM,
    *variables: str,
    graph: FakeGraph | None = None,
    binder: Binder | None = None,
) -> ResolveResponse:
    return offline_resolve(
        make_request(query),
        graph=graph if graph is not None else _graph(*variables),
        llm=llm,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        binder=binder,
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
        binder=ReplyBinder(),
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
        binder=ReplyBinder(),
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


# DAC2a's imputed multilateral ODA line, on the five-tuple of the imputed table's rows.
_DAC2A_IMPUTED_MEMBER = "AttributedAmount_FinancialFlow_Gross_Disbursement_DAC2aAidType106_Nominal"
_IMPUTED_LINES = (_IMPUTED_MEMBER, _DAC2A_IMPUTED_MEMBER)


def _imputed_lines_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """The imputed table's member and DAC2a's imputed line each hold France's series to
    Kenya under their own dataset."""
    patch_coverage(
        monkeypatch,
        [
            (_IMPUTED_MEMBER, "imputed", _FRA, _KEN, [2019, 2020, 2021]),
            (_DAC2A_IMPUTED_MEMBER, "dac2a", _FRA, _KEN, [2019, 2020, 2021]),
        ],
        role_names=dict.fromkeys(
            _IMPUTED_LINES, RoleNames(donor="financingProvider", recipient="financingRecipient")
        ),
        facet_metadata={
            "imputed": FacetMetadata(provenance_id=IMPUTED_PROVENANCE, unit="USD"),
            "dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD"),
        },
    )


@pytest.mark.parametrize(
    ("variable", "members", "source"),
    [
        pytest.param("imputed multilateral ODA", _IMPUTED_LINES, _DAC2A, id="imputed-first"),
        pytest.param("imputed multilateral ODA", _IMPUTED_LINES[::-1], _DAC2A, id="dac2a-first"),
        pytest.param(
            "imputed multilateral aid to education",
            _IMPUTED_LINES,
            IMPUTED_PROVENANCE,
            id="sector",
        ),
    ],
)
def test_an_unnamed_imputed_question_answers_from_dac2a_unless_it_asks_a_sector(
    monkeypatch: pytest.MonkeyPatch, variable: str, members: tuple[str, ...], source: str
):
    """The scripted bind elects the first shape it is offered. A question naming no table
    answers from DAC2a's own imputed line, whichever line recall finds first, so the bind
    is offered no imputed table. A question naming a sector, a breakdown only the
    imputed table holds, answers from it."""
    _imputed_lines_coverage(monkeypatch)
    llm = _ScriptedLLM(variable)

    response = _resolve_scripted(
        f"{variable} to Kenya", llm, graph=_graph(variable, members=members)
    )

    (spec,) = _specs(response)
    assert identity_dataset(spec.definition.identity) == source
    offered = any(_IMPUTED_LABEL in prompt for prompt in llm.bind_prompts)
    assert offered == (source == IMPUTED_PROVENANCE)


def test_a_question_naming_the_imputed_table_answers_from_it(monkeypatch: pytest.MonkeyPatch):
    """A part naming the imputed table answers from it with DAC2a's imputed line recalled
    first."""
    _imputed_lines_coverage(monkeypatch)
    _name_dataset(monkeypatch, IMPUTED_PROVENANCE)
    variable = "imputed multilateral ODA"

    response = _resolve_scripted(
        f"{variable} to Kenya",
        _ScriptedLLM(variable),
        graph=_graph(variable, members=_IMPUTED_LINES[::-1]),
    )

    (spec,) = _specs(response)
    assert identity_dataset(spec.definition.identity) == IMPUTED_PROVENANCE


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
    assert llm.calls == {"_UniversalBindOutput": 1, "QuantityOutput": 1}


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


def _provenance(spec: Spec) -> dict[str, str]:
    """The provenance of each slot filter in the spec's trace, by constraint property."""
    return {
        f.key.property.ref: f.provenance
        for f in spec.resolved.trace.slot_filters
        if f.key.property is not None
    }


@pytest.mark.parametrize(
    ("query", "variables", "dropped"),
    [
        pytest.param(
            "gross ODA, not net, to Kenya in 2022", ("gross ODA",), ("net",), id="one-variable"
        ),
        pytest.param(
            "ODA and aid to Kenya, gross not net",
            ("ODA", "aid"),
            ("gross", "net"),
            id="two-variables",
        ),
    ],
)
def test_a_part_naming_gross_and_net_takes_the_netting_basis_from_the_measure_election(
    query: str, variables: tuple[str, ...], dropped: tuple[str, ...]
):
    """The words "gross" and "net" both admit on `nettingBasis` and leave each other
    unresolved, so the text binds no netting basis. The measure election forces the CRS's
    preferred `Gross` on each part, and the trace credits it to the election alone."""
    response = _resolve_scripted(
        query,
        _ScriptedLLM(*variables, dropped=dropped),
        graph=_graph(*variables, members=(_CRS_MEMBER,)),
    )

    specs = _specs(response)
    assert len(specs) == len(variables)
    for spec in specs:
        assert _bound(spec)["nettingBasis"] == "Gross"
        assert _provenance(spec)["nettingBasis"] == "measure_elected"


def test_an_excluded_code_leaves_its_slot_to_the_binder():
    """The exclusion in "ODA excluding health" drops every `crsPurpose` match, so the text
    forces nothing there and the purpose slot keeps the model's binding, while ODA binds
    from the text."""
    variable = "ODA excluding health"
    response = _resolve_scripted(
        f"{variable} to Kenya",
        _ScriptedLLM(variable),
        graph=_graph(variable, members=(_CRS_MEMBER,)),
    )

    provenance = _provenance(_held_spec(response.parts[0]))
    assert provenance["crsPurpose"] == "binder"
    assert provenance["crsTypeOfFlow"] == "text_recognized"


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
        [ref.ref for ref in unbound.named] for unbound in second.no_data.unbound_qualifiers
    ] == [[CRS_PROVENANCE]]
    assert llm.calls["Extraction"] == 1
    assert len(llm.bind_prompts) == 1


@pytest.mark.parametrize("resolver_down", [False, True], ids=["resolver_up", "resolver_down"])
def test_a_raw_text_part_its_source_words_leave_no_source_calls_no_resolver(
    monkeypatch: pytest.MonkeyPatch, source_catalogue: None, resolver_down: bool
):
    """The query asks for aid from the CRS but not from the OECD, whose tables hold the CRS.
    The part's words leave it nothing to answer from, so it answers `qualifier_not_bound`
    naming the CRS and runs no step, which leaves its answer standing when the entity
    resolver fails."""
    if resolver_down:

        async def _resolver_down(self, texts, **kwargs):
            raise EntityResolutionError("resolver down")

        monkeypatch.setattr("entity_resolution.EntityResolver.resolve", _resolver_down)
    llm = _ScriptedLLM("aid")

    response = _resolve_scripted(
        "aid to Kenya, not from the OECD but from the CRS", llm, graph=_graph("aid")
    )

    (part,) = response.parts
    assert isinstance(part, NoDataPart)
    assert part.no_data.reason == "qualifier_not_bound"
    assert [[ref.ref for ref in unbound.named] for unbound in part.no_data.unbound_qualifiers] == [
        [CRS_PROVENANCE]
    ]
    assert llm.bind_prompts == []


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
    assert [[ref.ref for ref in unbound.named] for unbound in part.no_data.unbound_qualifiers] == [
        [CRS_PROVENANCE],
        [_SECOND],
    ]


# Each row: a raw-text question whose identity binds a table it did not name in a typed
# field, the variable, the recall members, the sources its words name, the second
# table's declaration (`None` leaves the fixture index), and the table the identity binds.
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
# (`None` leaves the fixture index), the one table holding a series from France, and the
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
    ("provenance", "member_count", "flow_provenance"),
    [
        (CRS_PROVENANCE, _ODA_CRS_EDUCATION_MEMBERS, "declared_default"),
        (IMPUTED_PROVENANCE, IMPUTED_MEMBER_COUNT, "binder"),
    ],
    ids=["crs", "imputed"],
)
def test_a_named_dataset_binds_the_source_and_resolves_the_mints_family(
    monkeypatch: pytest.MonkeyPatch, provenance: str, member_count: int, flow_provenance: str
):
    """The CRS declares official development assistance as its flow default, so a reply
    leaving the flow out binds it by that default and the family is the ODA members."""
    _name_dataset(monkeypatch, provenance)

    response = _resolve_scripted(f"{_E} to Kenya", _ScriptedLLM(_E), _E)

    (spec,) = _specs(response)
    assert spec.spec_id != EDUCATION_AID_SPEC_ID
    assert _bound(spec)["source"] == provenance
    identity = spec.definition.identity
    assert identity.facets is not None
    assert identity.facets["provenance"].ref == provenance
    assert identity_dataset(identity) == provenance
    (flow,) = [
        (slot.binding_kind, [ref.ref for ref in slot.refs], slot.provenance)
        for slot in spec.resolved.trace.slot_filters
        if slot.key.property is not None and slot.key.property.ref == "crsTypeOfFlow"
    ]
    assert flow == ("value", ["CRSTypeOfFlow10"], flow_provenance)
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
    assert [(ref.ref, ref.label) for ref in unbound.named] == [(IMPUTED_PROVENANCE, _IMPUTED_LABEL)]
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
        menu.key.property.ref: [option.value.ref.ref for option in menu.options]
        for menu in spec.resolved.slot_options
        if menu.key.property is not None and menu.key.property.ref in pinned
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
    assert {ref.ref for ref in term.named} == named
    assert [ref.ref for ref in source.named] == [provenance]
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
    assert {ref.ref for ref in term.named} == {"CRSTypeOfFinance100", "CRSTypeOfFinance110"}
    assert [ref.ref for ref in source.named] == [IMPUTED_PROVENANCE]


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


_DEU = "country/DEU"


def _dac2a_aid_type_siblings() -> list[str]:
    """DAC2a's gross disbursement members that differ from `_DAC2A_MEMBER` (ODA,
    `DAC2aAidType206`) in their aid type alone, so the elected shape offers that slot."""
    index = get_index()
    base = next(row for row in index.statvars if row.dcid == _DAC2A_MEMBER)

    def rest(row: Any) -> dict[str, str]:
        return {k: v for k, v in row.constraint_properties.items() if k != "dac2aAidType"}

    return [
        row.dcid
        for row in index.statvars
        if row.provenances == (_DAC2A,)
        and row_five_tuple(row) == row_five_tuple(base)
        and rest(row) == rest(base)
    ]


@pytest.mark.parametrize(
    "drafted",
    [
        pytest.param((), id="reply-leaves-the-slot-out"),
        pytest.param(
            (SlotBindingDraft(axis="how", property_dcid="dac2aAidType", kind="unbound"),),
            id="reply-leaves-the-slot-open",
        ),
    ],
)
def test_the_aid_type_default_binds_a_slot_the_reply_leaves_out(
    monkeypatch: pytest.MonkeyPatch, drafted: tuple[SlotBindingDraft, ...]
):
    """DAC2a's aid type defaults to ODA on every open slot. A bind reply that lists no
    binding for the slot leaves it as open as one that drafts it unbound, so "German aid to
    Kenya" binds ODA by the default and warns `slot_defaulted` either way."""
    members = _dac2a_aid_type_siblings()
    assert _DAC2A_MEMBER in members and len(members) > 1
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [(member, "dac2a", _DEU, _KEN, [2021]) for member in members],
        role_names=dict.fromkeys(members, names),
        facet_metadata={"dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD")},
    )
    graph = FakeGraph(
        nodes={
            _DEU: {"label": "Germany", "type": "Country"},
            _KEN: {"label": "Kenya", "type": "Country"},
            _DAC2A: {"label": "OECD DAC2a"},
            **{m: {"label": m, "type": "StatisticalVariable", "arcs": _arcs(m)} for m in members},
        },
        detect={
            "aid": {
                "svs": members,
                "cosine_scores": [0.9 - 0.01 * rank for rank in range(len(members))],
            }
        },
        resolve={"Germany": _DEU, "Kenya": _KEN},
        obs={},
    )
    germany = ExtractedEntity(text="Germany", span="German", flow_role="origin", role_span="German")

    response = offline_resolve(
        ResolveRequest(input=RawTextInput(query="German aid to Kenya")),
        graph=graph,
        llm=_ScriptedLLM("aid", entities=(germany, _TO_KENYA), drafted=drafted),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    (aid_type,) = [
        slot
        for slot in part.spec.resolved.trace.slot_filters
        if slot.key.property is not None and slot.key.property.ref == "dac2aAidType"
    ]
    assert (aid_type.binding_kind, [ref.ref for ref in aid_type.refs], aid_type.provenance) == (
        "value",
        ["DAC2aAidType206"],
        "declared_default",
    )
    assert "slot_defaulted" in [warning.code for warning in part.warnings]
