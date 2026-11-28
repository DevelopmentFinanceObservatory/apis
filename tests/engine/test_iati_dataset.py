"""Search elects IATI, one dataset of many publisher provenances, from the fixture index.

The fixture index declares `dataset/InternationalAidTransparencyInitiative` with the ten
publisher provenances the dev graph holds, its StatVars on the CRS five-tuples with
`iatiTransactionType` beside the CRS codes, the transaction types as the graph holds them, and
the spending parent of types 3 and 4 that DFO declares (`declarations.IATI_SPENDING`).
IATI is opt-in, so a question answers from it when it names IATI or names a transaction type no
other table offers, and a question naming neither answers from the CRS. IATI reads a missing
row as zero, so a sum over transaction types counts a type a publisher does not report as
nothing.
"""

from __future__ import annotations

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
from origin_catalogue import current_catalogue
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import DetectedSV, with_rival_families
from search.engine.extract import ExtractedDate, ExtractedEntity, Extraction
from search.models import DefinitePart, NoDataPart
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve
from tests.engine.test_aid_dataset_bind_e2e import (
    _arcs,
    _bound,
    _name_dataset,
    _ScriptedLLM,
    _sources,
    _specs,
)
from tests.fixtures import FakeGraph

_IATI = "dataset/InternationalAidTransparencyInitiative"
_CRS = "provenance/OECDCreditorReportingSystem"
_FCDO = "provenance/IATIUKForeignCommonwealthAndDevelopmentOffice"
_SIDA = "provenance/IATISwedenThroughSwedishInternationalDevelopmentCooperationAgency"
_UNFPA_PROVENANCE = "provenance/IATIUnitedNationsPopulationFund"
_UNFPA = "UnitedNationsPopulationFund"
_FCDO_PARTY = "ForeignCommonwealthAndDevelopmentOffice"
_SIDA_PARTY = "SwedishInternationalDevelopmentCooperationAgency"
_KEN = "country/KEN"
_UGA = "country/UGA"
_GBR = "country/GBR"
_DISBURSEMENT = "IATITransactionType3"
_EXPENDITURE = "IATITransactionType4"
_SPENDING = "IATITransactionTypeSpending"
_TRANSACTION_TYPE = "iatiTransactionType"
_SHARED = "CRSTypeOfFlow10_CRSTypeOfFinance110_CRSCooperationModalityB01_CRSPurposeCode12240"
# One basic nutrition disbursement member of each table, and IATI's expenditure member of it.
_IATI_DISBURSEMENTS = f"Amount_FinancialFlow_Gross_Disbursement_{_DISBURSEMENT}_{_SHARED}_Nominal"
_IATI_EXPENDITURE = f"Amount_FinancialFlow_Gross_Disbursement_{_EXPENDITURE}_{_SHARED}_Nominal"
_CRS_DISBURSEMENTS = f"Amount_FinancialFlow_Gross_Disbursement_{_SHARED}_Nominal"
_YEARS = [2022, 2023, 2024]


@pytest.fixture(autouse=True)
def _coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """The FCDO and Sida publish disbursements to Kenya, UNFPA publishes expenditure alone,
    and the CRS holds the UK's disbursements to Kenya, each under its own provenance."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    members = (_IATI_DISBURSEMENTS, _IATI_EXPENDITURE, _CRS_DISBURSEMENTS)
    patch_coverage(
        monkeypatch,
        [
            (_IATI_DISBURSEMENTS, "fcdo", _FCDO_PARTY, _KEN, _YEARS),
            (_IATI_DISBURSEMENTS, "sida", _SIDA_PARTY, _KEN, _YEARS),
            (_IATI_EXPENDITURE, "unfpa", _UNFPA, _KEN, _YEARS),
            (_CRS_DISBURSEMENTS, "crs", _GBR, _KEN, _YEARS),
        ],
        role_names=dict.fromkeys(members, names),
        facet_metadata={
            "fcdo": FacetMetadata(provenance_id=_FCDO, unit="USD"),
            "sida": FacetMetadata(provenance_id=_SIDA, unit="USD"),
            "unfpa": FacetMetadata(provenance_id=_UNFPA_PROVENANCE, unit="USD"),
            "crs": FacetMetadata(provenance_id=_CRS, unit="USD"),
        },
    )


def _graph(variable: str, *members: str, recalled: int | None = None) -> FakeGraph:
    """A graph holding `members` whose recall returns the first `recalled` of them, every
    one by default, in order for `variable`, with Kenya, Uganda, the UK and UNFPA resolving
    by name."""
    svs = list(members[:recalled])
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
                "svs": svs,
                "cosine_scores": [0.9 - 0.05 * rank for rank in range(len(svs))],
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
    disbursement measure and, on IATI's shape, binds transaction type 3 through the system's
    declared default, which states no note."""
    _name_dataset(monkeypatch, _IATI)
    llm = _ScriptedLLM("disbursements", entities=(_TO_KENYA,))

    response = _resolve(
        "IATI disbursements to Kenya in 2024",
        llm,
        _graph("disbursements", _IATI_DISBURSEMENTS, _CRS_DISBURSEMENTS),
    )

    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    assert _sources(part.spec) == [_IATI]
    assert _bound(part.spec)["source"] == _IATI
    assert _bound(part.spec)[_TRANSACTION_TYPE] == _DISBURSEMENT
    assert "slot_defaulted" not in {warning.code for warning in part.warnings}


def test_a_part_naming_iati_and_no_transaction_type_counts_spending(
    monkeypatch: pytest.MonkeyPatch,
):
    """A part naming no transaction type binds spending, the parent of disbursements and
    expenditure, through the system's open-slot default, and states the note naming both
    types."""
    _name_dataset(monkeypatch, _IATI)
    llm = _ScriptedLLM("funding", entities=(_TO_KENYA,))

    response = _resolve(
        "IATI funding to Kenya in 2024",
        llm,
        _graph("funding", _IATI_DISBURSEMENTS, _IATI_EXPENDITURE),
    )

    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    assert _bound(part.spec)[_TRANSACTION_TYPE] == _SPENDING
    note = get_index().declarations[_TRANSACTION_TYPE].defaults[-1].note
    assert [w.message for w in part.warnings if w.code == "slot_defaulted"] == [note]


_BY_UNFPA = ExtractedEntity(text="UNFPA", span="UNFPA", role="source", cue="by")


def test_unfpas_spending_answers_and_its_disbursements_find_no_data(
    monkeypatch: pytest.MonkeyPatch,
):
    """UNFPA publishes expenditure alone. "spending" reads the spending parent on IATI's
    shape, so UNFPA's spending answers from its expenditure, while "disbursements" binds type
    3, which UNFPA does not publish, and finds no data."""
    _name_dataset(monkeypatch, _IATI)
    spending, disbursements = (
        _resolve(
            f"IATI {variable} by UNFPA in 2024",
            _ScriptedLLM(variable, entities=(_BY_UNFPA,)),
            _graph(variable, _IATI_DISBURSEMENTS, _IATI_EXPENDITURE),
        )
        for variable in ("spending", "disbursements")
    )

    (part,) = spending.parts
    assert isinstance(part, DefinitePart), part
    assert _bound(part.spec)[_TRANSACTION_TYPE] == _SPENDING
    (part,) = disbursements.parts
    assert isinstance(part, NoDataPart), part
    assert part.no_data.reason == "no_observations"


@pytest.mark.parametrize(
    "variable",
    [
        "disbursements",
        "basic nutrition aid",
        "aid spending on health",
        "aid spend on health",
        "health expenditure",
    ],
)
def test_a_question_naming_no_dataset_leaves_the_opt_in_iati_out(variable: str):
    """IATI's shape, recalled first beside the CRS shape, is not offered to a part naming no
    dataset, and "disbursements" names the measure both tables publish. "spending", "spend"
    and "expenditure" read IATI's spending parent and type 4 on its shape, but as amount
    words they name no breakdown only IATI offers. The CRS answers."""
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


def test_disbursements_and_expenditure_bind_both_transaction_types(
    monkeypatch: pytest.MonkeyPatch,
):
    """ "expenditure" reads type 4 on IATI's shape, and "disbursements", joined to it by
    "and", binds type 3 beside it through the system's declared default."""
    _name_dataset(monkeypatch, _IATI)
    variable = "disbursements and expenditure"
    llm = _ScriptedLLM(variable, entities=(_BY_UNFPA,), drafted=_binds(_EXPENDITURE))

    response = _resolve(
        f"IATI {variable} by UNFPA in 2024",
        llm,
        _graph(variable, _IATI_DISBURSEMENTS, _IATI_EXPENDITURE),
    )

    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    (slot,) = [
        slot
        for slot in part.spec.resolved.slots
        if slot.key.property is not None and slot.key.property.ref == _TRANSACTION_TYPE
    ]
    assert slot.binding.kind == "set"
    assert {value.ref.ref for value in slot.binding.values if value.ref is not None} == {
        _DISBURSEMENT,
        _EXPENDITURE,
    }
    assert "slot_defaulted" not in {warning.code for warning in part.warnings}


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


_DAC2A = "provenance/OECDDAC2A"
_DAC2A_NET_ODA = "Amount_FinancialFlow_Net_Disbursement_DAC2aAidType206_Nominal"


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


_TO_UGANDA = ExtractedEntity(text="Uganda", span="Uganda", role="recipient", cue="to")
_IATI_EQUITY = (
    "Amount_FinancialFlow_Gross_Disbursement_EquityInstrument_IATITransactionType8_"
    "CRSTypeOfFlow10_CRSTypeOfFinance110_CRSCooperationModalityC01_CRSPurposeCode21010_Nominal"
)


def _recalled_alone(member: str) -> tuple[str, ...]:
    """`member`, an IATI row recalled alone, with the rows recall adds beside it
    (`with_rival_families`)."""
    hit = DetectedSV(dcid=member, score=0.9, type_of="StatisticalVariable", sentence=None)
    hits = with_rival_families([hit], catalogue=current_catalogue(), index=get_index())
    return tuple(h.dcid for h in hits)


def test_iati_rows_recalled_alone_gain_the_crs_family():
    """Recall holding IATI's gross disbursements alone gains a row of the CRS family, which
    publishes gross disbursements on the same five-tuple, and of no other opt-in table."""
    index = get_index()

    added = _recalled_alone(_IATI_DISBURSEMENTS)[1:]

    families = {index.family_by_dcid[dcid].datasets for dcid in added}
    assert (_CRS,) in families
    assert all(_IATI not in datasets for datasets in families)


@pytest.mark.parametrize(
    "date",
    [ExtractedDate(kind="point", start="2025"), ExtractedDate(kind="latest")],
    ids=["2025", "latest"],
)
def test_disbursements_to_uganda_with_iati_recalled_alone_name_no_iati_source(
    monkeypatch: pytest.MonkeyPatch, date: ExtractedDate
):
    """Recall returns IATI's disbursements alone, and IATI holds Uganda through 2026 while
    the default tables stop at 2024. A part naming no source answers from a default table:
    the CRS family joins recall, and IATI leaves the bind because the index's default
    families publish gross disbursements, whatever recall returned. "latest" answers 2024
    from a default table, and 2025 finds no data, in place of answering from IATI."""
    members = _recalled_alone(_IATI_DISBURSEMENTS)
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            (_IATI_DISBURSEMENTS, "fcdo", _FCDO_PARTY, _UGA, [*_YEARS, 2025, 2026]),
            *((dcid, "crs", _GBR, _UGA, _YEARS) for dcid in members[1:]),
        ],
        role_names=dict.fromkeys(members, names),
        facet_metadata={
            "fcdo": FacetMetadata(provenance_id=_FCDO, unit="USD"),
            "crs": FacetMetadata(provenance_id=_CRS, unit="USD"),
        },
    )
    llm = _DatedLLM("disbursements", date=date, entities=(_TO_UGANDA,))
    query = (
        "latest disbursements to Uganda"
        if date.kind == "latest"
        else ("disbursements to Uganda in 2025")
    )

    response = _resolve(query, llm, _graph("disbursements", *members, recalled=1))

    assert llm.bind_prompts
    assert not any(", IATI\n" in prompt for prompt in llm.bind_prompts)
    (part,) = response.parts
    if date.kind == "latest":
        assert isinstance(part, DefinitePart), part
        assert _sources(part.spec) == [_CRS]
    else:
        assert isinstance(part, NoDataPart), part
        assert part.no_data.reason == "no_observations"


def test_iati_leaves_an_unnamed_part_whatever_joined_the_menu(monkeypatch: pytest.MonkeyPatch):
    """The opt-in rule reads the index's default families, not the menu: with IATI recalled
    alone and the graph holding no row of a default table, so no default shape joins the
    menu, IATI still leaves a part naming no source, and the part finds no reading in place
    of answering 2025 from IATI."""
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [(_IATI_DISBURSEMENTS, "fcdo", _FCDO_PARTY, _UGA, [*_YEARS, 2025])],
        role_names={_IATI_DISBURSEMENTS: names},
        facet_metadata={"fcdo": FacetMetadata(provenance_id=_FCDO, unit="USD")},
    )
    llm = _DatedLLM(
        "disbursements", date=ExtractedDate(kind="point", start="2025"), entities=(_TO_UGANDA,)
    )

    response = _resolve(
        "disbursements to Uganda in 2025", llm, _graph("disbursements", _IATI_DISBURSEMENTS)
    )

    assert not llm.bind_prompts
    (part,) = response.parts
    assert isinstance(part, NoDataPart), part
    assert part.no_data.reason == "variable_not_resolved"


def test_purchase_of_equity_with_iati_recalled_alone_keeps_iati(
    monkeypatch: pytest.MonkeyPatch,
):
    """ "purchase of equity" names IATI's transaction type 8, which no default family on the
    five-tuple reads, so IATI stays beside the CRS family recall adds and answers a part
    naming no source."""
    members = _recalled_alone(_IATI_EQUITY)
    names = RoleNames(donor="financingProvider", recipient="financingRecipient")
    patch_coverage(
        monkeypatch,
        [
            (_IATI_EQUITY, "fcdo", _FCDO_PARTY, _KEN, _YEARS),
            *((dcid, "crs", _GBR, _KEN, _YEARS) for dcid in members[1:]),
        ],
        role_names=dict.fromkeys(members, names),
        facet_metadata={
            "fcdo": FacetMetadata(provenance_id=_FCDO, unit="USD"),
            "crs": FacetMetadata(provenance_id=_CRS, unit="USD"),
        },
    )
    llm = _ScriptedLLM(
        "purchase of equity", entities=(_TO_KENYA,), drafted=_binds("IATITransactionType8")
    )
    graph = _graph("purchase of equity", *members, recalled=1)

    response = _resolve("purchase of equity in Kenya in 2024", llm, graph)

    assert any(", IATI\n" in prompt for prompt in llm.bind_prompts)
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    assert _sources(part.spec) == [_IATI]


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


def test_a_spending_parent_widens_to_the_transaction_types_it_sums():
    """A part binding the spending parent reads its children's StatVars, and `groupby` sums
    them as it sums CRS purpose codes under their parents."""
    index = get_index()
    parents = [
        row.dcid
        for row in index.codes
        if row.code_system == _TRANSACTION_TYPE
        and {_DISBURSEMENT, _EXPENDITURE} <= set(index.expansion(row.dcid))
    ]

    assert parents == [_SPENDING]
