"""Search elects IATI, one dataset of many publisher provenances, from the fixture index.

The fixture index declares `dataset/InternationalAidTransparencyInitiative` with the ten
publisher provenances the dev graph holds, its StatVars on the CRS five-tuples with
`iatiTransactionType` beside the CRS codes, and the transaction types as the graph holds them.
IATI is opt-in, so a question answers from it when it names IATI or names a transaction type no
other table offers. A question naming neither answers from the CRS, and when the CRS finds no
data for years after its last year, from IATI where IATI answers on the same measure
(`DatasetDeclaration.default_after_others`, `region.pipeline._default_after_others`).
IATI reads a missing row as zero, so a sum over transaction types counts a type a publisher
does not report as nothing.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator, Sequence

import pytest

from code_index.members import absent_member_reads_zero
from code_index.shared import get_index
from code_index.types import summable_axes
from materialise.reshape import (
    ExpectedPopulation,
    RollupNotSummableError,
    Table,
    classify_columns,
    groupby_sum,
)
from origin_catalogue import BASE, DFO, bound_catalogue
from origin_catalogue.catalogue import DatasetRef
from search.engine.bind import ShapeRating, SlotBindingDraft
from search.engine.conjoin import assemble_region
from search.engine.core import ResolveRecord
from search.engine.degraded_reads import note_degraded_read
from search.engine.departures import dataset_defaulted
from search.engine.enrichment import late_departures, part_sources
from search.engine.errors import DefaultScopeIndexSourcedError, EngineInfraError
from search.engine.events import ObservationsEvent
from search.engine.extract import ExtractedDate, ExtractedEntity, Extraction
from search.engine.region import phase_rank_bind
from search.engine.region import pipeline as pipeline_module
from search.engine.region.codes import OBSERVATIONS_NO_POINTS
from search.engine.region.result import RegionResult
from search.engine.region.state import ResolveState, UnPool
from search.engine.source_scope import SourceScope
from search.models import DefinitePart, NoDataPart, omitted_observations
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._builders import conjoin_spec, shape_five_tuple
from tests.engine._fake_coverage import CoverageRow, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve
from tests.engine._source_scopes import part_scope
from tests.engine.test_aid_dataset_bind_e2e import (
    _IMPUTED_LABEL,
    _IMPUTED_LINES,
    _arcs,
    _bound,
    _imputed_lines_coverage,
    _name_dataset,
    _resolve_scripted,
    _ScriptedLLM,
    _sources,
    _specs,
)
from tests.engine.test_aid_dataset_bind_e2e import _graph as _aid_graph
from tests.fixtures import FakeGraph
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, provenance_row

_IATI = "dataset/InternationalAidTransparencyInitiative"
_CRS = "provenance/OECDCreditorReportingSystem"
_FCDO = "provenance/IATIUKForeignCommonwealthAndDevelopmentOffice"
_SIDA = "provenance/IATISwedenThroughSwedishInternationalDevelopmentCooperationAgency"
_UNFPA_PROVENANCE = "provenance/IATIUnitedNationsPopulationFund"
_UNFPA = "UnitedNationsPopulationFund"
_FCDO_PARTY = "ForeignCommonwealthAndDevelopmentOffice"
_SIDA_PARTY = "SwedishInternationalDevelopmentCooperationAgency"
_KEN = "country/KEN"
_GBR = "country/GBR"
_DISBURSEMENT = "IATITransactionType3"
_EXPENDITURE = "IATITransactionType4"
_TRANSACTION_TYPE = "iatiTransactionType"
_SHARED = "CRSTypeOfFlow10_CRSTypeOfFinance110_CRSCooperationModalityB01_CRSPurposeCode12240"
# One basic nutrition disbursement member of each table, and IATI's expenditure member of it.
_IATI_DISBURSEMENTS = f"Amount_FinancialFlow_Gross_Disbursement_{_DISBURSEMENT}_{_SHARED}_Nominal"
_IATI_EXPENDITURE = f"Amount_FinancialFlow_Gross_Disbursement_{_EXPENDITURE}_{_SHARED}_Nominal"
_CRS_DISBURSEMENTS = f"Amount_FinancialFlow_Gross_Disbursement_{_SHARED}_Nominal"
# DAC2a's net ODA to a recipient, on the five-tuple IATI's disbursements publish.
_DAC2A = "provenance/OECDDAC2A"
_DAC2A_NET_ODA = "Amount_FinancialFlow_Net_Disbursement_DAC2aAidType206_Nominal"
_YEARS = [2022, 2023, 2024]
# The coverage gives 2025 as the last year the CRS and DAC2a hold and 2026 as IATI's.
_CRS_YEARS = [*_YEARS, 2025]
_IATI_YEARS = [*_CRS_YEARS, 2026]
_UGA = "country/UGA"
_CRS_TO_KENYA: CoverageRow = (_CRS_DISBURSEMENTS, "crs", _GBR, _KEN, _CRS_YEARS)


@pytest.fixture(autouse=True)
def _coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_coverage(monkeypatch)


def _patch_coverage(
    monkeypatch: pytest.MonkeyPatch,
    crs: Sequence[CoverageRow] = (_CRS_TO_KENYA,),
    iati: Sequence[CoverageRow] = (),
) -> None:
    """The FCDO and Sida publish disbursements to Kenya, and the FCDO the rows `iati` gives
    beside them, UNFPA publishes expenditure alone, the CRS holds the rows `crs` gives, by
    default the UK's disbursements to Kenya, and DAC2a the UK's net ODA to Kenya, each under
    its own provenance."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    members = (_IATI_DISBURSEMENTS, _IATI_EXPENDITURE, _CRS_DISBURSEMENTS, _DAC2A_NET_ODA)
    patch_coverage(
        monkeypatch,
        [
            (_IATI_DISBURSEMENTS, "fcdo", _FCDO_PARTY, _KEN, _IATI_YEARS),
            (_IATI_DISBURSEMENTS, "sida", _SIDA_PARTY, _KEN, _IATI_YEARS),
            *iati,
            (_IATI_EXPENDITURE, "unfpa", _UNFPA, _KEN, _IATI_YEARS),
            *crs,
            (_DAC2A_NET_ODA, "dac2a", _GBR, _KEN, _CRS_YEARS),
        ],
        role_names=dict.fromkeys(members, names),
        facet_metadata={
            "fcdo": FacetMetadata(provenance_id=_FCDO, unit="USD"),
            "sida": FacetMetadata(provenance_id=_SIDA, unit="USD"),
            "unfpa": FacetMetadata(provenance_id=_UNFPA_PROVENANCE, unit="USD"),
            "crs": FacetMetadata(provenance_id=_CRS, unit="USD"),
            "dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD"),
        },
    )


def _graph(variable: str, *members: str) -> FakeGraph:
    """A graph whose recall returns `members` in order for `variable`, with Kenya, Uganda, the
    UK and UNFPA resolving by name."""
    return FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            _UGA: {"label": "Uganda", "type": "Country"},
            _GBR: {"label": "United Kingdom", "type": "Country"},
            _UNFPA: {"label": "United Nations Population Fund", "type": "Organization"},
            _CRS: {"label": "OECD Creditor Reporting System (CRS)"},
            _IATI: {"label": "IATI"},
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
        },
        resolve={"Kenya": _KEN, "Uganda": _UGA, "the UK": _GBR, "UNFPA": _UNFPA},
    )


def _resolve(query: str, llm: _ScriptedLLM, graph: FakeGraph):
    return offline_resolve(
        make_request(query),
        graph=graph,
        llm=llm,
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )


_TO_KENYA = ExtractedEntity(text="Kenya", span="Kenya", role="recipient", cue="to")


def _binds(code: str) -> tuple[SlotBindingDraft, ...]:
    """The bind model's draft binding the transaction type `code`, which the question's
    word reads on IATI's shape."""
    return (
        SlotBindingDraft(
            axis="how", property_dcid=_TRANSACTION_TYPE, kind="value", value_dcids=[code]
        ),
    )


def test_a_question_naming_iati_answers_from_the_dataset_across_its_publishers(
    monkeypatch: pytest.MonkeyPatch,
):
    """A part naming IATI answers from the dataset, whichever publisher holds the series: the
    answer cites the dataset and binds it as its source. "disbursements" names the gross
    disbursement measure, which IATI's disbursement, expenditure and equity purchase types
    all carry, so the transaction type stays open."""
    _name_dataset(monkeypatch, _IATI)
    llm = _ScriptedLLM("disbursements", entities=(_TO_KENYA,))

    response = _resolve(
        "IATI disbursements to Kenya in 2024",
        llm,
        _graph("disbursements", _IATI_DISBURSEMENTS, _CRS_DISBURSEMENTS),
    )

    (spec,) = _specs(response)
    assert _sources(spec) == [_IATI]
    assert _bound(spec)["source"] == _IATI
    assert _TRANSACTION_TYPE not in _bound(spec)


@pytest.mark.parametrize("variable", ["disbursements", "basic nutrition aid"])
def test_a_question_naming_no_dataset_leaves_the_opt_in_iati_out(variable: str):
    """IATI's shape, recalled first beside the CRS shape, is not offered to a part naming no
    dataset, and "disbursements" names the measure both tables publish. The CRS answers."""
    llm = _ScriptedLLM(variable, entities=(_TO_KENYA,))

    response = _resolve(
        f"{variable} to Kenya in 2023",
        llm,
        _graph(variable, _IATI_DISBURSEMENTS, _CRS_DISBURSEMENTS),
    )

    (prompt,) = llm.bind_prompts
    assert ", IATI\n" not in prompt
    (spec,) = _specs(response)
    assert _sources(spec) == [_CRS]
    assert _TRANSACTION_TYPE not in _bound(spec)


class _DatedLLM(_ScriptedLLM):
    """`_ScriptedLLM`, extracting `date` as the question's one date reference."""

    def __init__(self, *variables: str, date: ExtractedDate, **kwargs) -> None:
        super().__init__(*variables, **kwargs)
        self._date = date

    async def generate_structured(self, *, prompt, system, schema):
        reply = await super().generate_structured(prompt=prompt, system=system, schema=schema)
        if schema is Extraction:
            extraction, usage = reply
            return extraction.model_copy(update={"dates": [self._date]}), usage
        return reply


def _exclude_dataset(monkeypatch: pytest.MonkeyPatch, dataset: str) -> None:
    """Exclude `dataset` from every part, at the seam shape candidacy reads
    (`ResolveState.scope`)."""
    prune_and_rank = phase_rank_bind._phase_prune_and_rank

    async def excluding(state: ResolveState):
        state.scope = part_scope(excluded=[dataset])
        return await prune_and_rank(state)

    monkeypatch.setattr(
        pipeline_module,
        "_PHASES",
        tuple(excluding if phase is prune_and_rank else phase for phase in pipeline_module._PHASES),
    )


_IN_2026 = ExtractedDate(kind="point", start="2026")


_SEAT_LABEL = re.compile(r"^Shape \d+:\n  label: (.*)$", re.MULTILINE)
_TABLE_LABELS = {", IATI": _IATI, "(CRS)": _CRS}


def _offered(prompt: str) -> list[str]:
    """The table each seat of a bind prompt's menu offers, in seat order: IATI or the CRS by
    id, any other by its label."""
    return [
        next((ref for tail, ref in _TABLE_LABELS.items() if label.endswith(tail)), label)
        for label in _SEAT_LABEL.findall(prompt)
    ]


class _DefaultSeatLLM(_DatedLLM):
    """`_DatedLLM`, electing the first shape a table other than IATI publishes wherever the
    bind menu seats one, such as the CRS, so an answer from IATI comes from a menu that
    seats IATI alone."""

    async def generate_structured(self, *, prompt, system, schema):
        reply = await super().generate_structured(prompt=prompt, system=system, schema=schema)
        output, usage = reply
        seats = _offered(prompt) if hasattr(output, "chosen_shape_index") else []
        seat = next((i for i, table in enumerate(seats) if table != _IATI), None)
        if seat is None:
            return reply
        elected = output.model_copy(
            update={"chosen_shape_index": seat, "measuring": [ShapeRating(index=seat, rating=1.0)]}
        )
        return elected, usage


_TO_UGANDA = ExtractedEntity(text="Uganda", span="Uganda", role="recipient", cue="to")


def _no_observations(response) -> None:
    (part,) = response.parts
    assert isinstance(part, NoDataPart), part
    assert part.no_data.reason == "no_observations"


_LISTING = FakeCatalogue(
    provenances={
        _CRS: provenance_row(_CRS, BASE, name="OECD Creditor Reporting System (CRS)"),
        _DAC2A: provenance_row(_DAC2A, BASE, name="OECD DAC2a"),
        **{
            member: provenance_row(member, DFO, dataset=DatasetRef(ref=_IATI, label="IATI"))
            for member in (_FCDO, _SIDA, _UNFPA_PROVENANCE)
        },
    }
)
"""A catalogue listing the CRS, DAC2a and three of IATI's publishers under the IATI
Dataset, so a mention of the dataset's ref compiles, as the request's catalogue lists them."""


@pytest.fixture
def listing() -> Iterator[None]:
    with bound_catalogue(_LISTING):
        yield


def _recalls(monkeypatch: pytest.MonkeyPatch) -> list[SourceScope | None]:
    """The source scope of each pass that recalls the part, in order: our pass first, then
    the pass for a dataset declaring `default_after_others` when it starts."""
    scopes: list[SourceScope | None] = []
    recall = pipeline_module._phase_recall

    async def counting(state: ResolveState) -> None:
        scopes.append(state.scope)
        await recall(state)

    monkeypatch.setattr(pipeline_module, "_phase_recall", counting)
    return scopes


_DEFAULTED = "OECD Creditor Reporting System (CRS) publishes data up to 2025, so this shows IATI."
_IN_2025 = ExtractedDate(kind="point", start="2025")
_IN_2027 = ExtractedDate(kind="point", start="2027")


@pytest.mark.usefixtures("listing")
@pytest.mark.parametrize(
    ("query", "date", "entities", "source"),
    [
        ("disbursements to Kenya in 2026", _IN_2026, (_TO_KENYA,), _IATI),
        ("disbursements in 2026", _IN_2026, (), _IATI),
        ("disbursements to Kenya in 2025", _IN_2025, (_TO_KENYA,), _CRS),
        ("disbursements in 2025", _IN_2025, (), _CRS),
        (
            "disbursements to Kenya from 2025 to 2026",
            ExtractedDate(kind="range", start="2025", end="2026"),
            (_TO_KENYA,),
            _CRS,
        ),
        (
            "disbursements to Kenya in the latest year",
            ExtractedDate(kind="latest"),
            (_TO_KENYA,),
            _CRS,
        ),
        ("disbursements to Kenya in 2027", _IN_2027, (_TO_KENYA,), None),
        ("disbursements in 2027", _IN_2027, (), None),
    ],
)
def test_iati_answers_after_the_crs_years_when_the_crs_finds_no_data(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    date: ExtractedDate,
    entities: tuple[ExtractedEntity, ...],
    source: str | None,
):
    """The bind elects the CRS's gross disbursements wherever it is offered. For a part
    naming no dataset whose window starts after 2025, the CRS's last year at any place, the
    CRS finds no data and a second pass reads the part within IATI, which holds 2026 at
    Kenya: its answer stands and states `dataset_defaulted`. A window touching 2025 or a
    request for the latest year answers from the CRS, and 2027, which IATI holds at no
    place, finds the CRS's no data. Neither starts the second pass."""
    recalls = _recalls(monkeypatch)
    llm = _DefaultSeatLLM("disbursements", date=date, entities=entities)

    response = _resolve(
        query, llm, _graph("disbursements", _CRS_DISBURSEMENTS, _IATI_DISBURSEMENTS)
    )

    assert _offered(llm.bind_prompts[0]) == [_CRS]
    if source != _IATI:
        assert recalls == [None]
    if source is None:
        _no_observations(response)
        return
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    assert _sources(part.spec) == [source]
    defaulted = [d for d in part.departures if d.code == "dataset_defaulted"]
    if source == _CRS:
        assert not defaulted
        return
    (departure,) = defaulted
    assert (departure.binding_ref, departure.shown) == ("source:", "IATI")
    assert departure.text == _DEFAULTED
    (_ours, theirs) = recalls
    assert theirs is not None
    assert theirs.allowed == {_IATI}


@pytest.mark.usefixtures("listing")
def test_the_defaulted_answer_is_the_answer_to_the_question_naming_iati():
    """The second pass resolves the part as a part naming IATI does, so "disbursements to
    Kenya in 2026" answers with the Spec of "IATI disbursements to Kenya in 2026"."""
    graph = _graph("disbursements", _CRS_DISBURSEMENTS, _IATI_DISBURSEMENTS)

    def spec_id(query: str) -> str:
        llm = _DefaultSeatLLM("disbursements", date=_IN_2026, entities=(_TO_KENYA,))
        (spec,) = _specs(_resolve(query, llm, graph))
        return spec.spec_id

    assert spec_id("disbursements to Kenya in 2026") == spec_id(
        "IATI disbursements to Kenya in 2026"
    )


@pytest.mark.usefixtures("listing")
@pytest.mark.parametrize(
    "crs",
    [
        [_CRS_TO_KENYA, (_CRS_DISBURSEMENTS, "crs", _GBR, _UGA, [2026])],
        [],
    ],
    ids=["crs-reaches-2026-elsewhere", "crs-holds-no-year"],
)
def test_no_second_pass_while_the_crs_series_reach_the_window_anywhere(
    monkeypatch: pytest.MonkeyPatch, crs: list[CoverageRow]
):
    """The second pass reads the CRS shape's last year over every place. One CRS series
    reaching 2026, here the UK's disbursements to Uganda, keeps "disbursements to Kenya in
    2026" with the CRS's no data, and so does a CRS shape whose series the coverage holds
    no year of."""
    _patch_coverage(monkeypatch, crs)
    recalls = _recalls(monkeypatch)
    llm = _DefaultSeatLLM("disbursements", date=_IN_2026, entities=(_TO_KENYA,))

    response = _resolve(
        "disbursements to Kenya in 2026",
        llm,
        _graph("disbursements", _CRS_DISBURSEMENTS, _IATI_DISBURSEMENTS),
    )

    assert recalls == [None]
    _no_observations(response)


def _no_defaulted(response) -> None:
    """The CRS's no-data answer, with no `dataset_defaulted` departure."""
    _no_observations(response)
    (part,) = response.parts
    assert "dataset_defaulted" not in [departure.code for departure in part.departures]


@pytest.mark.usefixtures("listing")
def test_a_place_iati_holds_nothing_at_in_the_window_keeps_the_crs_no_data(
    monkeypatch: pytest.MonkeyPatch,
):
    """The FCDO's IATI disbursements to Uganda run from 2023 to 2025 and the CRS's from
    2022 to 2025, though IATI holds 2026 at Kenya. The second pass for "disbursements to
    Uganda in 2026" finds no data too, and the CRS's no-data answer stands, stating the
    CRS's years."""
    _patch_coverage(
        monkeypatch,
        crs=[_CRS_TO_KENYA, (_CRS_DISBURSEMENTS, "crs", _GBR, _UGA, _CRS_YEARS)],
        iati=[(_IATI_DISBURSEMENTS, "fcdo", _FCDO_PARTY, _UGA, [2023, 2024, 2025])],
    )
    recalls = _recalls(monkeypatch)
    llm = _DefaultSeatLLM("disbursements", date=_IN_2026, entities=(_TO_UGANDA,))

    response = _resolve(
        "disbursements to Uganda in 2026",
        llm,
        _graph("disbursements", _IATI_DISBURSEMENTS, _CRS_DISBURSEMENTS),
    )

    assert len(recalls) == 2
    _no_defaulted(response)
    (part,) = response.parts
    assert [departure.text for departure in part.departures] == [
        "You asked for 2026. The data covers only 2022 to 2025."
    ]


_CRS_TOBACCO_CONTROL = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110"
    "_CRSCooperationModalityB01_CRSPurposeCode12320_Nominal"
)


_NO_PLACE: tuple[ExtractedEntity, ...] = ()


@pytest.mark.usefixtures("listing")
@pytest.mark.parametrize(
    ("variable", "members", "entities"),
    [
        # Tobacco use control is a CRS purpose no IATI row holds.
        pytest.param(
            "tobacco use control disbursements",
            (_IATI_DISBURSEMENTS, _CRS_TOBACCO_CONTROL),
            (_TO_KENYA,),
            id="tobacco-to-kenya",
        ),
        pytest.param(
            "tobacco use control disbursements",
            (_IATI_DISBURSEMENTS, _CRS_TOBACCO_CONTROL),
            _NO_PLACE,
            id="tobacco-no-place",
        ),
        # DAC2a's bindings elect its net ODA line at Kenya, on the five-tuple of IATI's
        # gross disbursements.
        pytest.param(
            "disbursements",
            (_DAC2A_NET_ODA, _IATI_DISBURSEMENTS),
            (_TO_KENYA,),
            id="disbursements-beside-net-oda",
        ),
        *(
            pytest.param("net ODA", members, entities, id=f"net-oda-{first}-{place}")
            for members, first in (
                ((_DAC2A_NET_ODA, _IATI_DISBURSEMENTS), "dac2a-first"),
                ((_IATI_DISBURSEMENTS, _DAC2A_NET_ODA), "iati-first"),
            )
            for entities, place in (((_TO_KENYA,), "to-kenya"), (_NO_PLACE, "no-place"))
        ),
    ],
)
def test_a_second_pass_on_another_reading_leaves_the_default_no_data_standing(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    members: tuple[str, ...],
    entities: tuple[ExtractedEntity, ...],
):
    """The second pass runs for each of these 2026 parts, and its answer gives way: IATI
    reads no tobacco use control and no net ODA, so its pass ends without a definite
    answer, and its gross disbursements are another measure than DAC2a's net line, so they
    do not stand in for it. The default no-data answer stands."""
    recalls = _recalls(monkeypatch)
    llm = _DefaultSeatLLM(variable, date=_IN_2026, entities=entities)
    place = " to Kenya" if entities else ""

    response = _resolve(f"{variable}{place} in 2026", llm, _graph(variable, *members))

    assert len(recalls) == 2
    assert _IATI not in _offered(llm.bind_prompts[0])
    _no_defaulted(response)


@pytest.mark.usefixtures("listing")
@pytest.mark.parametrize(
    "first", [_DAC2A_NET_ODA, _IATI_DISBURSEMENTS], ids=["dac2a-first", "iati-first"]
)
def test_oda_to_kenya_after_the_default_years_keeps_dac2a(first: str):
    """DAC2a's net ODA line reads "ODA", and IATI's gross disbursements are another
    measure, so "ODA to Kenya in 2026" offers DAC2a's line, finds no data after DAC2a's
    last year, and the second pass's IATI answer does not stand in for it."""
    members = (first, *({_DAC2A_NET_ODA, _IATI_DISBURSEMENTS} - {first}))
    llm = _DefaultSeatLLM("ODA", date=_IN_2026, entities=(_TO_KENYA,))

    response = _resolve("ODA to Kenya in 2026", llm, _graph("ODA", *members))

    assert _offered(llm.bind_prompts[0]) == ["FinancialFlow disbursements, OECD DAC2a"]
    _no_defaulted(response)


@pytest.mark.usefixtures("listing")
def test_excluding_iati_after_the_crs_years_runs_no_second_pass(
    monkeypatch: pytest.MonkeyPatch,
):
    """A part excluding IATI, here at the seam shape candidacy reads, compiles the second
    pass's scope with IATI included and excluded, which allows nothing, so "disbursements
    to Kenya in 2026" keeps the CRS's no data and starts no second pass."""
    _exclude_dataset(monkeypatch, _IATI)
    recalls = _recalls(monkeypatch)
    llm = _DefaultSeatLLM("disbursements", date=_IN_2026, entities=(_TO_KENYA,))

    response = _resolve(
        "disbursements to Kenya in 2026",
        llm,
        _graph("disbursements", _CRS_DISBURSEMENTS, _IATI_DISBURSEMENTS),
    )

    assert recalls == [None]
    _no_defaulted(response)


@pytest.mark.usefixtures("listing")
def test_a_first_pass_refusing_on_another_reason_runs_no_second_pass(
    monkeypatch: pytest.MonkeyPatch,
):
    """Only a no-observations answer opens the second pass. Our pass's answer to
    "disbursements to Kenya in 2026", turned into a `qualifier_not_bound` refusal on the
    same elected shape, stands as it is."""
    within_source_code = pipeline_module._within_source_code

    def refused(result, state):
        return dataclasses.replace(
            within_source_code(result, state), no_data_reason="qualifier_not_bound"
        )

    monkeypatch.setattr(pipeline_module, "_within_source_code", refused)
    recalls = _recalls(monkeypatch)
    llm = _DefaultSeatLLM("disbursements", date=_IN_2026, entities=(_TO_KENYA,))

    response = _resolve(
        "disbursements to Kenya in 2026",
        llm,
        _graph("disbursements", _CRS_DISBURSEMENTS, _IATI_DISBURSEMENTS),
    )

    assert recalls == [None]
    (part,) = response.parts
    assert isinstance(part, NoDataPart), part
    assert part.no_data.reason == "qualifier_not_bound"


def test_a_streamed_read_keeps_the_dataset_defaulted_departure():
    """A streamed part ships before its observations are read, and `done` carries its
    complete departures once the read lands (`enrichment.late_departures`). The
    `dataset_defaulted` departure the part shipped stays in that list."""
    fact = dataset_defaulted(
        shown="IATI", defaulted_from=["OECD Creditor Reporting System (CRS)"], last_year=2025
    )
    spec = conjoin_spec(shape_five_tuple("FinancialFlow", "amount"), "disbursements")
    region = RegionResult(
        variable_text="disbursements",
        status="definite",
        specs=(spec,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        inferred_facts=(fact,),
    )
    response = assemble_region(
        region, extra_warnings=[], start_ms=0, engine_build="test", max_candidates=5
    )
    (source,) = part_sources(response, answered=[region], streaming=[region])
    assert [departure.code for departure in source.part.departures] == ["dataset_defaulted"]
    read = omitted_observations(OBSERVATIONS_NO_POINTS, "no points")

    late = late_departures(
        source, events=(ObservationsEvent(spec_id=spec.spec_id, observations=read),), warnings=()
    )

    assert late is None


def _unknown_measure(_shape, _bindings) -> None:
    return None


@pytest.mark.usefixtures("listing")
def test_an_answer_whose_measure_values_are_unknown_leaves_the_crs_no_data(
    monkeypatch: pytest.MonkeyPatch,
):
    """The measure guard fails closed: when either pass's elected members share no measure
    values (`coverage_series.elected_measure_values` gives `None`), IATI's definite answer
    to "disbursements to Kenya in 2026" gives way to the CRS's no data."""
    monkeypatch.setattr(pipeline_module, "elected_measure_values", _unknown_measure)
    llm = _DefaultSeatLLM("disbursements", date=_IN_2026, entities=(_TO_KENYA,))

    response = _resolve(
        "disbursements to Kenya in 2026",
        llm,
        _graph("disbursements", _CRS_DISBURSEMENTS, _IATI_DISBURSEMENTS),
    )

    _no_defaulted(response)


@pytest.mark.usefixtures("listing")
@pytest.mark.parametrize(
    ("failure", "degraded"),
    [
        (EngineInfraError("graph read failed"), True),
        (TimeoutError(), True),
        (DefaultScopeIndexSourcedError("index-sourced shape under a default scope"), False),
        (RuntimeError("defect"), False),
    ],
    ids=["upstream", "timeout", "request-refusal", "defect"],
)
def test_a_second_pass_that_fails_leaves_the_crs_no_data_standing(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, degraded: bool
):
    """A second pass that fails upstream, refuses the whole request or fails on a defect
    leaves the CRS's no-data answer to "disbursements to Kenya in 2026" standing, and the
    request answers. An upstream failure marks the part degraded."""
    run_phases = pipeline_module._run_phases
    lost: list[str] = []

    async def failing(state: ResolveState):
        if state.scope is not None and state.scope.named:
            raise failure
        return await run_phases(state)

    monkeypatch.setattr(pipeline_module, "_run_phases", failing)
    monkeypatch.setattr(pipeline_module, "note_degraded_read", lost.append)
    llm = _DefaultSeatLLM("disbursements", date=_IN_2026, entities=(_TO_KENYA,))

    response = _resolve(
        "disbursements to Kenya in 2026",
        llm,
        _graph("disbursements", _CRS_DISBURSEMENTS, _IATI_DISBURSEMENTS),
    )

    _no_defaulted(response)
    assert lost == (["default-after-others pass"] if degraded else [])


def _kenya_2026(**kwargs):
    """The response to "disbursements to Kenya in 2026", whose second pass answers from
    IATI."""
    llm = _DefaultSeatLLM("disbursements", date=_IN_2026, entities=(_TO_KENYA,))
    return offline_resolve(
        make_request("disbursements to Kenya in 2026"),
        graph=_graph("disbursements", _CRS_DISBURSEMENTS, _IATI_DISBURSEMENTS),
        llm=llm,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        **kwargs,
    )


def _on_the_second_pass(monkeypatch: pytest.MonkeyPatch, after) -> None:
    """Run `after(state)` once the second pass, the one reading under a named scope, has run
    its phases."""
    run_phases = pipeline_module._run_phases

    async def running(state: ResolveState):
        result = await run_phases(state)
        if state.scope is not None and state.scope.named:
            after(state)
        return result

    monkeypatch.setattr(pipeline_module, "_run_phases", running)


@pytest.mark.usefixtures("listing")
def test_the_un_pass_runs_before_iati_answers_and_its_warning_stays(
    monkeypatch: pytest.MonkeyPatch,
):
    """The UN pass runs on the CRS's no data before the second pass. A UN leg that did not
    answer warns `un_system_unavailable`, and IATI's answer to "disbursements to Kenya in
    2026" that then stands keeps the warning."""
    waits: list[float | None] = []

    async def missed(_leg, *, bound):
        waits.append(bound)
        return UnPool(shapes=[], missed="the UN System Data Commons did not answer")

    monkeypatch.setattr(pipeline_module, "un_pool_shapes", missed)

    response = _kenya_2026()

    assert len(waits) == 1
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    assert "dataset_defaulted" in [departure.code for departure in part.departures]
    assert "un_system_unavailable" in [warning.code for warning in part.warnings]


@pytest.mark.usefixtures("listing")
def test_a_doubted_iati_answer_leaves_the_crs_no_data_standing(monkeypatch: pytest.MonkeyPatch):
    """IATI's binder rating "none of the shapes" above its pick (`BindResult.none_first`)
    counts against its answer, so "disbursements to Kenya in 2026" keeps the CRS's no
    data."""

    def doubted(state: ResolveState) -> None:
        assert state.bind_decision is not None
        state.bind_decision = dataclasses.replace(state.bind_decision, none_first=True)

    _on_the_second_pass(monkeypatch, doubted)

    _no_defaulted(_kenya_2026())


@pytest.mark.usefixtures("listing")
@pytest.mark.parametrize("stands", [True, False], ids=["stands", "dropped"])
def test_a_second_pass_lost_read_degrades_the_part_only_when_its_answer_stands(
    monkeypatch: pytest.MonkeyPatch, stands: bool
):
    """A read the second pass loses marks the part degraded when IATI's answer stands, and
    leaves it undegraded when the answer is dropped, here on unknown measure values, since
    the part shows the CRS's no data, which the read did not bear on."""
    _on_the_second_pass(monkeypatch, lambda _state: note_degraded_read("a second-pass read"))
    if not stands:
        monkeypatch.setattr(pipeline_module, "elected_measure_values", _unknown_measure)
    record = ResolveRecord()

    response = _kenya_2026(record=record)

    (part,) = response.parts
    assert isinstance(part, DefinitePart) is stands, part
    assert record.done is not None
    assert record.done.degraded_leg is stands


def test_an_opt_in_table_declaring_no_late_default_stays_out_after_the_default_years(
    monkeypatch: pytest.MonkeyPatch,
):
    """On a five-tuple IATI does not publish, a window after every default table's last year
    leaves the opt-in table as it was: the imputed multilateral sectors table declares no
    `default_after_others`, so "imputed multilateral ODA" in 2026 offers the bind DAC2a's own
    line alone."""
    _imputed_lines_coverage(monkeypatch)
    variable = "imputed multilateral ODA"
    llm = _DatedLLM(variable, date=_IN_2026)

    _resolve_scripted(
        f"{variable} to Kenya in 2026", llm, graph=_aid_graph(variable, members=_IMPUTED_LINES)
    )

    assert llm.bind_prompts
    assert not any(_IMPUTED_LABEL in prompt for prompt in llm.bind_prompts)


def test_iatis_expenditure_answers_under_its_transaction_type(monkeypatch: pytest.MonkeyPatch):
    """A part naming IATI reads "expenditure" as IATI's transaction type 4, so UNFPA's
    expenditure answers from IATI under type 4 alone."""
    _name_dataset(monkeypatch, _IATI)
    llm = _ScriptedLLM(
        "expenditure",
        entities=(ExtractedEntity(text="UNFPA", span="UNFPA", role="source", cue="by"),),
        drafted=_binds(_EXPENDITURE),
    )

    response = _resolve(
        "IATI expenditure by UNFPA in 2024", llm, _graph("expenditure", _IATI_EXPENDITURE)
    )

    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    assert _sources(part.spec) == [_IATI]
    assert _bound(part.spec)[_TRANSACTION_TYPE] == _EXPENDITURE


@pytest.mark.parametrize(
    "first", [_DAC2A_NET_ODA, _IATI_DISBURSEMENTS], ids=["dac2a-first", "iati-first"]
)
def test_oda_named_in_a_default_tables_codes_leaves_the_opt_in_iati_out(
    monkeypatch: pytest.MonkeyPatch, first: str
):
    """ "ODA" names the CRS flow type on IATI's shape, a slot DAC2a's net ODA line lacks, but
    DAC2a reads "ODA" in its own aid type, so the word asks no breakdown only IATI offers.
    "ODA to Kenya in 2026" leaves IATI out of the bind, whichever shape recall puts first, and
    finds no data after DAC2a's last year in place of answering from IATI's 2026 rows."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            (_IATI_DISBURSEMENTS, "fcdo", _FCDO_PARTY, _KEN, [*_YEARS, 2025, 2026]),
            (_DAC2A_NET_ODA, "dac2a", _GBR, _KEN, [*_YEARS, 2025]),
        ],
        role_names=dict.fromkeys((_IATI_DISBURSEMENTS, _DAC2A_NET_ODA), names),
        facet_metadata={
            "fcdo": FacetMetadata(provenance_id=_FCDO, unit="USD"),
            "dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD"),
        },
    )
    members = (first, *({_DAC2A_NET_ODA, _IATI_DISBURSEMENTS} - {first}))
    llm = _DatedLLM("ODA", date=ExtractedDate(kind="point", start="2026"), entities=(_TO_KENYA,))

    response = _resolve("ODA to Kenya in 2026", llm, _graph("ODA", *members))

    assert llm.bind_prompts
    assert not any(", IATI\n" in prompt for prompt in llm.bind_prompts)
    (part,) = response.parts
    assert isinstance(part, NoDataPart), part
    assert part.no_data.reason == "no_observations"


# ---------------------------------------------------------------------------
# A sum over transaction types reads an absent type as zero.
# ---------------------------------------------------------------------------

_COLUMNS = (
    _TRANSACTION_TYPE,
    "financingProvider",
    "time_period",
    "value",
    "unit",
    "measurementMethod",
    "provenance",
    "status",
)


def _cell(transaction_type: str, provider: str, value: float) -> tuple:
    return (transaction_type, provider, "2024", value, "USD", None, _IATI, None)


def _spending(dataset: str) -> Table:
    """The disbursement and expenditure cells of UNFPA, which publishes expenditure alone,
    and of UNHCR, which publishes both, summed over the transaction type under `dataset`'s
    reading of a missing row."""
    index = get_index()
    table = Table(
        columns=_COLUMNS,
        rows=(
            _cell(_EXPENDITURE, _UNFPA, 40.0),
            _cell(_DISBURSEMENT, "UnitedNationsHighCommissionerForRefugees", 10.0),
            _cell(_EXPENDITURE, "UnitedNationsHighCommissionerForRefugees", 5.0),
        ),
    )
    return groupby_sum(
        table,
        axis=_TRANSACTION_TYPE,
        classification=classify_columns(table.columns),
        summable_axes=summable_axes(index.declarations),
        missing_declared_as_zero=absent_member_reads_zero(index, dataset),
        expected_population=ExpectedPopulation(
            columns=(_TRANSACTION_TYPE, "financingProvider"),
            members=frozenset(
                (code, provider)
                for code in (_DISBURSEMENT, _EXPENDITURE)
                for provider in (_UNFPA, "UnitedNationsHighCommissionerForRefugees")
            ),
        ),
    )


def test_a_sum_over_transaction_types_counts_a_type_a_publisher_omits_as_zero():
    """UNFPA's total is its expenditure, with no `missing_members` refusal, and UNHCR's adds
    its two types."""
    totals = {row[0]: row[2] for row in _spending(_IATI).rows}

    assert totals == {_UNFPA: 40.0, "UnitedNationsHighCommissionerForRefugees": 15.0}


def test_the_same_sum_under_a_table_reading_absence_as_unknown_refuses():
    """The control: DAC2a reads a missing row as unknown, so UNFPA's absent disbursements
    refuse the sum."""
    with pytest.raises(RollupNotSummableError) as refused:
        _spending("provenance/OECDDAC2A")

    assert refused.value.precondition == "missing_members"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The dev graph holds no spending parent of IATITransactionType3, 4, 7 and 8 yet, so"
        " iatiTransactionType is flat and no bound code widens to its children."
    ),
)
def test_a_spending_parent_widens_to_the_transaction_types_it_sums():
    """Once the graph holds the parent, a part binding it reads its children's StatVars, and
    `groupby` sums them as it sums CRS purpose codes under their parents."""
    index = get_index()
    parents = [
        row.dcid
        for row in index.codes
        if row.code_system == _TRANSACTION_TYPE
        and {_DISBURSEMENT, _EXPENDITURE} <= set(index.expansion(row.dcid))
    ]

    assert len(parents) == 1
