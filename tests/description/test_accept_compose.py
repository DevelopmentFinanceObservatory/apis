"""Acceptance cases for the description composer, `description.compose`.

`describe_version(record, vocabulary)` describes a dataset version or a computed version, as
the record's id says. `describe_card(record, selection, vocabulary, *, component_id,
revision)` describes a card revision drawn from the record's version. Both return JSON-ready
wire data, and every description served here is validated against the `spec-kit` repository's
`description.schema.json`.

The records and vocabulary are built from the design's own facts: the worked example
"Gross disbursements to Nigeria for health" (154 of 505 variables, 523 series, OECD), and
the quotient chain of gross ODA disbursements per gross national income. Reason names in
`left_out` are the kit's (`operations.json`).
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Sequence
from dataclasses import replace
from functools import cache
from types import MappingProxyType
from typing import Any

import pytest
from dfo_descriptions import check_description, operations
from dfo_frame_grammar import STAT_VAR_COLUMN

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import MeasureAxis, MeasureIndex, MeasurePartitionLabel, MeasureValueRow
from description.facts import (
    Builds,
    CaveatFact,
    CodeIndexStamp,
    ComputedFacts,
    Coordinate,
    DescriptionRecord,
    DimensionFacts,
    DimensionValue,
    DrawnSeries,
    Evidence,
    FixedValue,
    InputVersion,
    LabelEvidence,
    SelectionFacts,
    SeriesFacts,
    SourceEvidence,
    SourceUnitCount,
    Span,
    StatusCounts,
    VariableFacts,
)
from description.steps import ComposedUnit, LeftOut, SimpleUnit, Step, Unit, VersionRef
from description.vocabulary import SourceDocumentFacts, Vocabulary, build_vocabulary

from ._kit import validate_description

_SPEC_ID = "spec_91d7345eec81a029629938b3d0ffd775"
_CV = "cv_b785befb9f87471de96eb31cd7c56f5c"
_OTHER_SPEC_ID = "spec_0a1b2c3d4e5f60718293a4b5c6d7e8f9"
_OTHER_CV = "cv_f9e8d7c6b5a40312f9e8d7c6b5a40312"
_OPS_ID = "ops_3c9e1f7a2b8d4e6f"
_OCV = "ocv_8d2a6f1c9e3b7a5d"
_ODA = VersionRef(
    id="spec_4b1f0c2d9e8a7b6c5d4e3f2a1b0c9d8e",
    content_version="cv_0f1e2d3c4b5a69788796a5b4c3d2e1f0",
    label="Gross ODA disbursements",
)
_GNI = VersionRef(
    id="spec_7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d",
    content_version="cv_1a2b3c4d5e6f708192a3b4c5d6e7f809",
    label="Gross national income",
)
_COMPONENT_ID = "cmp_Ab3dE5gH7k"
_REVISION = 2

_CRS = "provenance/OECDCreditorReportingSystem"
_CRS_NAME = "OECD Creditor Reporting System (CRS)"
_OECD = "dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd"
_PREPARED = datetime.date(2026, 9, 28)
_BACKFILLED = datetime.date(2026, 10, 12)
_STAMP = CodeIndexStamp(
    content_hash="3f1a9c", exported_at="2026-09-27T06:00:00Z", graph_endpoint="dev"
)
_USD = SimpleUnit(ref="CurrencyUSD", prices="current")
_USD_WIRE = {"ref": "CurrencyUSD", "label": "US dollars", "prices": "current"}
_PER_USD = ComposedUnit(
    numerator=SimpleUnit(ref="USDollar", label="US dollars"),
    denominator=SimpleUnit(ref="USDollar", label="US dollars"),
)
_PER_USD_CELL = ComposedUnit(
    numerator=SimpleUnit(ref="USDollar"), denominator=SimpleUnit(ref="USDollar")
)
"""`_PER_USD` as a frame's unit column holds it, without words."""
_PER_USD_WIRE = {
    "numerator": {"ref": "USDollar", "label": "US dollars"},
    "denominator": {"ref": "USDollar", "label": "US dollars"},
}

# Codes, by code system: (dcid, label).

_HEALTH = "CRSPurposeCode120"
_EDUCATION = "CRSPurposeCode110"
_BASIC_HEALTH = ("CRSPurposeCode12220", "basic health care")
_HEALTH_POLICY = ("CRSPurposeCode12110", "health policy and administrative management")
_OTHER_HEALTH = [(f"CRSPurposeCode129{n:02d}", f"health activity {n:02d}") for n in range(17)]
# Label order is the reverse of dcid order, so a tie broken by dcid reads differently.
_AGRICULTURE = ("CRSPurposeCode31100", "agriculture")
_BANKING = ("CRSPurposeCode24000", "banking")
_WATER = ("CRSPurposeCode14000", "water supply")
_LOST_SECTOR = "CRSPurposeCode99999"
_PROJECT = ("CRSModalityC01", "project-type intervention")
_TECHNICAL = ("CRSModalityD02", "other technical assistance")
_OTHER_MODALITIES = [(f"CRSModalityX{n:02d}", f"modality {n:02d}") for n in range(10)]
_GRANT = ("CRSTypeOfFinance110", "standard grant")
_LOAN = ("CRSTypeOfFinance421", "standard loan")
_OTHER_INSTRUMENTS = [
    ("CRSTypeOfFinance210", "interest subsidy"),
    ("CRSTypeOfFinance510", "equity"),
]
_ODA_FLOW = ("CRSTypeOfFlow10", "official development assistance")
_PRIVATE_FLOW = ("CRSTypeOfFlow30", "private development finance")
_OTHER_FLOWS = [("CRSTypeOfFlow20", "other official flows"), ("CRSTypeOfFlow40", "export credits")]
_PROVIDERS = [(f"dc/o/provider{n:02d}", f"provider {n:02d}") for n in range(63)]

# Definitions, by dcid, as the dev graph describes the codes and measure values.
_DEFINED = {
    _BASIC_HEALTH[0]: (
        "Basic and primary health care programmes; paramedical and nursing care programmes; "
        "supply of drugs, medicines, and vaccines related to basic health care; and activities "
        "aimed at achieving universal health coverage."
    ),
    _HEALTH_POLICY[0]: (
        "Health sector policy, planning, and programmes; aid to health ministries and public "
        "health administration; institution capacity building and advice; medical insurance "
        "programmes; health system strengthening and health governance; and unspecified "
        "health activities."
    ),
    "Disbursement": "The placing of resources at the recipient's disposal.",
    "Gross": "An amount counted in full, with nothing offset.",
}

_SYSTEMS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "crsPurpose": (
        "sector",
        [
            (_HEALTH, "health"),
            (_EDUCATION, "education"),
            _BASIC_HEALTH,
            _HEALTH_POLICY,
            *_OTHER_HEALTH,
            _AGRICULTURE,
            _BANKING,
            _WATER,
        ],
    ),
    "crsCooperationModality": ("modality", [_PROJECT, _TECHNICAL, *_OTHER_MODALITIES]),
    "crsTypeOfFinance": ("instrument", [_GRANT, _LOAN, *_OTHER_INSTRUMENTS]),
    "crsTypeOfFlow": ("flow", [_ODA_FLOW, _PRIVATE_FLOW, *_OTHER_FLOWS]),
}

_INDEX = CodeIndex(
    stamp=IndexStamp(
        graph_endpoint="dev",
        exported_at="2026-10-11T00:00:00Z",
        code_counts={system: len(codes) for system, (_, codes) in _SYSTEMS.items()},
        statvar_shape_counts={},
        content_hash="index-hash",
    ),
    declarations=[
        CodeSystemDeclaration(
            code_system=system,
            hierarchical=system == "crsPurpose",
            disjoint_from_children=True,
            summable=True,
            absent_reading="aggregate",
            plain_name=noun,
        )
        for system, (noun, _) in _SYSTEMS.items()
    ],
    codes=[
        CodeRow(
            dcid=dcid,
            code_system=system,
            code=dcid,
            label=label,
            synonyms=(),
            parent=None,
            definition=_DEFINED.get(dcid),
        )
        for system, (_, codes) in _SYSTEMS.items()
        for dcid, label in codes
    ],
    statvars=(),
)

# Keyed as the packaged measure index keys its partition labels: on the shape's
# `populationType` and `measuredProperty` beside the constraint axes.
_MEASURES = MeasureIndex(
    axes=(
        MeasureAxis(axis="populationType", kind="shape", role="free", prefer=()),
        MeasureAxis(axis="measuredProperty", kind="shape", role="basis", prefer=()),
        MeasureAxis(axis="measurementQualifier", kind="shape", role="free", prefer=()),
        MeasureAxis(
            axis="financialFlowType", kind="constraint", role="free", prefer=(), label="flow type"
        ),
        MeasureAxis(
            axis="nettingBasis", kind="constraint", role="free", prefer=(), label="netting basis"
        ),
    ),
    values=(
        MeasureValueRow(
            dcid="Disbursement",
            axis="financialFlowType",
            label="disbursement",
            synonyms=(),
            short_name=None,
            matchable=True,
            definition=_DEFINED["Disbursement"],
        ),
        MeasureValueRow(
            dcid="Gross",
            axis="nettingBasis",
            label="gross",
            synonyms=(),
            short_name=None,
            matchable=True,
            definition=_DEFINED["Gross"],
        ),
    ),
    partition_labels=(
        MeasurePartitionLabel(
            axis_values=(
                ("financialFlowType", "Disbursement"),
                ("measuredProperty", "amount"),
                ("nettingBasis", "Gross"),
                ("populationType", "FinancialFlow"),
            ),
            label="gross disbursements",
        ),
    ),
)


@cache
def _words() -> Vocabulary:
    """Today's words: the source's current name and document, places, providers, and the
    inputs of the quotient chain."""
    return build_vocabulary(
        index=_INDEX,
        measure_index=_MEASURES,
        labels={
            _CRS: _CRS_NAME,
            "country/NGA": "Nigeria",
            "country/SSD": "South Sudan",
            "country/SOM": "Somalia",
            **dict(_PROVIDERS),
        },
        sources={
            _CRS: SourceDocumentFacts(
                publisher=_OECD, last_downloaded="2026-10-08", next_release="2026-09"
            )
        },
        currency_units={"CurrencyUSD": "USD", "USDollar": "USD"},
        graph_generation="gen-1",
        catalogue_build="ocb_20260927",
    )


def _checked(served: dict[str, Any]) -> dict[str, Any]:
    """`served`, once it round-trips through JSON unchanged, the kit's schema accepts it,
    and each of its steps and caveats carries its sentence."""
    assert json.loads(json.dumps(served)) == served, "a description is JSON-ready wire data"
    validate_description(served)
    _assert_worded(served)
    return served


def _describe_version(record: DescriptionRecord, words: Vocabulary | None = None) -> dict[str, Any]:
    from description.compose import describe_version

    return _checked(describe_version(record, words or _words()))


def _describe_card(
    record: DescriptionRecord,
    selection: SelectionFacts,
    words: Vocabulary | None = None,
    *,
    frame: VersionRef | None = None,
    steps: list[Step] | None = None,
) -> dict[str, Any]:
    from description.compose import describe_card

    served = _checked(
        describe_card(
            record,
            selection,
            words or _words(),
            component_id=_COMPONENT_ID,
            revision=_REVISION,
            frame=frame,
            steps=steps,
        )
    )
    check_description(served)
    return served


def _cut(actual: Any, like: Any) -> Any:
    """`actual` cut to the members `like` names, at every depth, so a comparison with `like`
    ignores members the design's example leaves out. A member `like` names and `actual`
    lacks reads as `"<absent>"`; lists keep their own length."""
    if isinstance(like, dict) and isinstance(actual, dict):
        return {key: _cut(actual.get(key, "<absent>"), like[key]) for key in like}
    if isinstance(like, list) and isinstance(actual, list) and len(like) == len(actual):
        return [_cut(item, model) for item, model in zip(actual, like, strict=True)]
    return actual


def _dimension(served: dict[str, Any], prop: str) -> dict[str, Any]:
    (found,) = [dim for dim in served["about"]["varies"] if dim["property"]["ref"] == prop]
    return found


def _step(served: dict[str, Any], step_id: str) -> dict[str, Any]:
    (found,) = [step for step in served["steps"] if step["id"] == step_id]
    return found


def _assert_worded(served: dict[str, Any]) -> None:
    """Every served step and caveat carries `text`, the composer's sentence for it."""
    for step in served["steps"] or []:
        assert isinstance(step.get("text"), str), step["id"]
        assert step["text"].strip(), step["id"]
    for n, caveat in enumerate(served["caveats"]):
        assert isinstance(caveat.get("text"), str), n
        assert caveat["text"].strip(), n


def _values_link(prop: str, *, id: str = _SPEC_ID, content_version: str = _CV) -> str:
    return f"/api/datasets/{id}/{content_version}/description/values?property={prop}"


def _values(pairs: Sequence[tuple[str, int]]) -> list[DimensionValue]:
    return [DimensionValue(value=dcid, variables=count) for dcid, count in pairs]


def _open(
    prop: str,
    pairs: Sequence[tuple[str, int]],
    *,
    within: str | None = None,
    undeclared: int = 0,
    partial: bool = False,
) -> DimensionFacts:
    return DimensionFacts(
        property=prop,
        within=within,
        values=_values(pairs),
        undeclared_variables=undeclared,
        partial=partial,
    )


def _series(**overrides: Any) -> SeriesFacts:
    fields: dict[str, Any] = {
        "series": 523,
        "rows": 9_814,
        "periods": 30,
        "span": Span(start="1995", end="2024"),
        "grain": "year",
        "latest_period": "2024",
        "series_reaching_latest": 146,
        "entities_by_role": None,
        "by_source_unit": [SourceUnitCount(source=_CRS, unit=_USD, series=523)],
        "status": StatusCounts(estimated=None, provisional=None),
    }
    return SeriesFacts.model_validate(fields | overrides)


def _evidence(**overrides: Any) -> Evidence:
    fields: dict[str, Any] = {
        "prepared": _PREPARED,
        "title": "Gross disbursements to Nigeria for health",
        "sources": [
            SourceEvidence(
                ref=_CRS,
                name=_CRS,
                licence="OECD Terms and Conditions",
                downloaded=None,
                pipeline_steps=None,
            )
        ],
        "source_release": None,
        "labels": [
            LabelEvidence(ref="country/NGA", label="Nigeria", read=_PREPARED),
            LabelEvidence(ref=_HEALTH, label="Health", read=_PREPARED),
        ],
        "membership": None,
        "builds": Builds(
            origin_catalogue="ocb_20260927", code_index=_STAMP, coverage=None, method_status=None
        ),
    }
    return Evidence.model_validate(fields | overrides)


_SHAPE = [
    FixedValue(property="populationType", value="FinancialFlow"),
    FixedValue(property="measuredProperty", value="amount"),
    FixedValue(property="statType", value="measuredValue"),
    FixedValue(property="measurementQualifier", value="Nominal"),
]

_FIXED = [
    FixedValue(property="financingRecipient", value="country/NGA"),
    FixedValue(property="financialFlowType", value="Disbursement"),
    FixedValue(property="nettingBasis", value="Gross"),
]


def _variables(
    dimensions: list[DimensionFacts], *, fixed: list[FixedValue] = _FIXED, **overrides: Any
) -> VariableFacts:
    fields: dict[str, Any] = {
        "shape": _SHAPE,
        "variable": None,
        "fixed": fixed,
        "dimensions": dimensions,
        "admitted_at_mint": 505,
        "admitted_from": "family_member_count",
        "with_data": 154,
    }
    return VariableFacts.model_validate(fields | overrides)


def _dataset_record(
    *, variables: VariableFacts, id: str = _SPEC_ID, content_version: str = _CV, **overrides: Any
) -> DescriptionRecord:
    """A `spec_` record minted on the worked example's day."""
    fields: dict[str, Any] = {
        "id": id,
        "content_version": content_version,
        "schema_version": 1,
        "basis": "mint",
        "index_stamp": _STAMP,
        "variables": variables,
        "computed": None,
        "series": _series(),
        "steps": [],
        "caveats": [],
        "evidence": _evidence(),
        "late_fields": [],
    }
    return DescriptionRecord.model_validate(fields | overrides)


# The worked example

_WORKED_DIMENSIONS = [
    # Values are listed out of display order, so the order served is the composer's.
    _open("financingProvider", [(dcid, 3 + n % 5) for n, (dcid, _) in enumerate(_PROVIDERS)]),
    _open(
        "crsPurpose",
        [
            *[(dcid, 8) for dcid, _ in _OTHER_HEALTH],
            (_HEALTH_POLICY[0], 19),
            (_BASIC_HEALTH[0], 19),
        ],
        within=_HEALTH,
    ),
    _open(
        "crsCooperationModality",
        [(_TECHNICAL[0], 24), *[(dcid, 7) for dcid, _ in _OTHER_MODALITIES], (_PROJECT[0], 47)],
        undeclared=16,
    ),
    _open(
        "crsTypeOfFinance",
        [
            (_OTHER_INSTRUMENTS[0][0], 3),
            (_LOAN[0], 16),
            (_GRANT[0], 134),
            (_OTHER_INSTRUMENTS[1][0], 1),
        ],
    ),
    _open(
        "crsTypeOfFlow",
        [
            (_OTHER_FLOWS[0][0], 8),
            (_PRIVATE_FLOW[0], 39),
            (_OTHER_FLOWS[1][0], 2),
            (_ODA_FLOW[0], 105),
        ],
    ),
]


def _worked_example() -> DescriptionRecord:
    """The record behind "Gross disbursements to Nigeria for health", prepared 28 September
    2026: 505 variables admitted, 154 with data, 523 series of which 146 reach 2024, one
    source stored under its bare ref, no status recorded, no step taken at mint."""
    return _dataset_record(variables=_variables(_WORKED_DIMENSIONS))


def _worked_example_wire(vocabulary: str) -> dict[str, Any]:
    """The design's served description of the worked example, member for member."""
    return {
        "describes": {"kind": "dataset_version", "id": _SPEC_ID, "content_version": _CV},
        "title": "Gross disbursements to Nigeria for health",
        "about": {
            "measure": {"label": "gross disbursements", "definition": None},
            "fixed": [
                {
                    "property": {"ref": "financingRecipient", "label": "recipient"},
                    "value": {"ref": "country/NGA", "label": "Nigeria"},
                },
                {
                    "property": {"ref": "financialFlowType", "label": "flow type"},
                    "value": {
                        "ref": "Disbursement",
                        "label": "disbursement",
                        "definition": _DEFINED["Disbursement"],
                    },
                },
                {
                    "property": {"ref": "nettingBasis", "label": "netting basis"},
                    "value": {"ref": "Gross", "label": "gross", "definition": _DEFINED["Gross"]},
                },
            ],
            "varies": [
                {
                    "property": {"ref": "financingProvider", "label": "provider"},
                    "count": 63,
                    "top": [
                        {"label": "provider 04", "variables": 7},
                        {"label": "provider 09", "variables": 7},
                    ],
                    "more": 61,
                    "values": _values_link("financingProvider"),
                },
                {
                    "property": {"ref": "crsPurpose", "label": "sector"},
                    "within": {"ref": _HEALTH, "label": "health"},
                    "count": 19,
                    "top": [
                        {"label": "basic health care", "variables": 19},
                        {"label": "health policy and administrative management", "variables": 19},
                    ],
                    "more": 17,
                    "values": _values_link("crsPurpose"),
                },
                {
                    "property": {"ref": "crsCooperationModality", "label": "modality"},
                    "count": 12,
                    "undeclared_variables": 16,
                    "top": [
                        {"label": "project-type intervention", "variables": 47},
                        {"label": "other technical assistance", "variables": 24},
                    ],
                    "more": 10,
                    "values": _values_link("crsCooperationModality"),
                },
                {
                    "property": {"ref": "crsTypeOfFinance", "label": "instrument"},
                    "count": 4,
                    "top": [
                        {"label": "standard grant", "variables": 134},
                        {"label": "standard loan", "variables": 16},
                    ],
                    "more": 2,
                    "values": _values_link("crsTypeOfFinance"),
                },
                {
                    "property": {"ref": "crsTypeOfFlow", "label": "flow"},
                    "count": 4,
                    "top": [
                        {"label": "official development assistance", "variables": 105},
                        {"label": "private development finance", "variables": 39},
                    ],
                    "more": 2,
                    "values": _values_link("crsTypeOfFlow"),
                },
            ],
            "variables": {"admitted_at_mint": 505, "with_data": 154},
            "series": 523,
            "period": {"start": "1995", "end": "2024", "grain": "year"},
            "unit": _USD_WIRE,
        },
        "sources": [
            {
                "ref": _CRS,
                "label": _CRS_NAME,
                "publisher": "OECD",
                "licence": "OECD Terms and Conditions",
                "series": 523,
                "documentation": f"/api/graph/documentation/{_CRS}",
            }
        ],
        "freshness": {
            "prepared": "2026-09-28",
            "latest_period": "2024",
            "series_reaching_it": 146,
            "source_release": None,
            "source_downloaded_at_mint": None,
            "source_downloaded_since": "2026-10-08",
            "next_release": "2026-09",
            "estimated_values": None,
            "provisional_values": None,
        },
        "steps": [],
        "caveats": [],
        "cite": {
            "title": "Gross disbursements to Nigeria for health",
            "text": f"{_CRS_NAME} (2026), via the Development Finance Observatory",
            "id": _SPEC_ID,
            "content_version": _CV,
            "data": f"/api/datasets/{_SPEC_ID}/{_CV}",
        },
        "gaps": [
            "about.measure.definition",
            "freshness.source_release",
            "freshness.source_downloaded_at_mint",
            "steps.source_pipeline",
            "freshness.estimated_values",
            "freshness.provisional_values",
        ],
        "built_from": {"facts_schema": 1, "vocabulary": vocabulary},
    }


_WORKED_BLOCKS = [
    "describes",
    "title",
    "about.measure",
    "about.fixed",
    "about.varies",
    "about.variables",
    "about.series",
    "about.period",
    "about.unit",
    "sources",
    "freshness",
    "steps",
    "caveats",
    "cite",
    "built_from",
]


def _at(document: dict[str, Any], path: str) -> Any:
    for member in path.split("."):
        document = document[member]
    return document


@pytest.mark.parametrize("path", _WORKED_BLOCKS)
def test_worked_example_serves_the_designs_description(path: str) -> None:
    """Each member of the design's worked example, as `describe_version` serves it. Every open
    dimension carries `count`, `top`, `more` and its `values` link, the provider included.
    Members the example leaves out, such as a `top` item's `ref`, are not compared."""
    words = _words()
    served = _describe_version(_worked_example(), words)
    expected = _worked_example_wire(words.revision)

    assert _cut(_at(served, path), _at(expected, path)) == _at(expected, path)


def test_worked_example_gaps_are_exactly_the_unfilled_fields() -> None:
    """The six fields the worked example cannot fill, and no others; the source's bare stored
    name gives way to today's catalogue name, so nothing is served as a word from the mint."""
    words = _words()
    served = _describe_version(_worked_example(), words)

    assert sorted(served["gaps"]) == sorted(_worked_example_wire(words.revision)["gaps"])
    assert '"at_mint"' not in json.dumps(served)
    assert all("partial" not in dim for dim in served["about"]["varies"]), "its counts are whole"


# Display rules

_SECTOR = "crsPurpose"


@pytest.mark.parametrize(
    ("values", "top", "more"),
    [
        pytest.param(
            [(_AGRICULTURE[0], 1), (_BANKING[0], 9), (_WATER[0], 5)],
            [("banking", 9), ("water supply", 5)],
            1,
            id="top-holds-two-by-variable-count-and-more-is-the-remainder",
        ),
        pytest.param(
            [(_WATER[0], 5), (_BANKING[0], 5), (_AGRICULTURE[0], 5)],
            [("agriculture", 5), ("banking", 5)],
            1,
            id="ties-order-by-label-not-dcid",
        ),
        pytest.param(
            [(_AGRICULTURE[0], 2), (_WATER[0], 7)],
            [("water supply", 7), ("agriculture", 2)],
            0,
            id="count-orders-before-label",
        ),
        pytest.param(
            [(_AGRICULTURE[0], 2), (_BANKING[0], 2), (_WATER[0], 3), (_BASIC_HEALTH[0], 2)],
            [("water supply", 3), ("agriculture", 2)],
            2,
            id="a-tie-below-the-leader-orders-by-label",
        ),
        pytest.param(
            [(_WATER[0], 4)],
            [("water supply", 4)],
            0,
            id="one-value-stays-under-varies",
        ),
    ],
)
def test_display_rules_for_an_open_dimension(
    values: list[tuple[str, int]], top: list[tuple[str, int]], more: int
) -> None:
    served = _describe_version(_dataset_record(variables=_variables([_open(_SECTOR, values)])))
    dimension = _dimension(served, _SECTOR)

    assert dimension["count"] == len(values)
    assert [(item["label"], item["variables"]) for item in dimension["top"]] == top
    assert dimension["more"] == more
    assert dimension["values"] == _values_link(_SECTOR)
    assert _SECTOR not in [fixed["property"]["ref"] for fixed in served["about"]["fixed"]], (
        "a dimension the definition leaves open is never served as fixed"
    )


@pytest.mark.parametrize(
    "order",
    [
        pytest.param(("crsTypeOfFlow", "crsPurpose", "crsTypeOfFinance"), id="unsorted"),
        pytest.param(("crsTypeOfFinance", "crsPurpose", "crsTypeOfFlow"), id="reversed"),
    ],
)
def test_properties_keep_the_definitions_order(order: tuple[str, str, str]) -> None:
    codes = {"crsTypeOfFlow": _ODA_FLOW[0], "crsPurpose": _WATER[0], "crsTypeOfFinance": _GRANT[0]}
    fixed = [FixedValue(property=prop, value=codes[prop]) for prop in reversed(order)]
    dimensions = [_open(prop, [(codes[prop], 1)]) for prop in order]
    served = _describe_version(_dataset_record(variables=_variables(dimensions, fixed=fixed)))

    assert [dim["property"]["ref"] for dim in served["about"]["varies"]] == list(order)
    assert [item["property"]["ref"] for item in served["about"]["fixed"]] == list(reversed(order))


# An unknown never reads as zero

_STATUS_GAPS = {"freshness.estimated_values", "freshness.provisional_values"}


def _assert_status(served: dict[str, Any], estimated: int | None, provisional: int | None) -> None:
    freshness = served["freshness"]
    assert freshness["estimated_values"] == estimated
    assert freshness["provisional_values"] == provisional
    expected_gaps = {
        gap
        for gap, count in zip(sorted(_STATUS_GAPS), (estimated, provisional), strict=True)
        if count is None
    }
    assert set(served["gaps"]) & _STATUS_GAPS == expected_gaps


@pytest.mark.parametrize(
    ("estimated", "provisional"),
    [
        pytest.param(None, None, id="unrecorded-is-null-with-gaps"),
        pytest.param(0, 0, id="control-known-zero-is-zero"),
        pytest.param(0, None, id="known-zero-beside-unknown"),
        pytest.param(12, 3, id="control-known-counts"),
    ],
)
def test_version_status_counts(estimated: int | None, provisional: int | None) -> None:
    record = _dataset_record(
        variables=_variables(_WORKED_DIMENSIONS),
        series=_series(status=StatusCounts(estimated=estimated, provisional=provisional)),
    )

    _assert_status(_describe_version(record), estimated, provisional)


_NONE_MARKED = StatusCounts(estimated=0, provisional=0)


def _drawn(
    series_id: str,
    coordinates: dict[str, str],
    *,
    status: StatusCounts = _NONE_MARKED,
) -> DrawnSeries:
    return DrawnSeries(
        id=series_id,
        coordinates=[
            Coordinate(column=column, value=value) for column, value in coordinates.items()
        ],
        source=_CRS,
        unit=_USD,
        first_period="2010",
        last_period="2020",
        values=11,
        missing=0,
        status=status,
    )


def _selection(*series: DrawnSeries) -> SelectionFacts:
    return SelectionFacts(window=Span(start="2010", end="2020"), series=list(series))


@pytest.mark.parametrize(
    ("statuses", "estimated", "provisional"),
    [
        pytest.param([(None, None)], None, None, id="unrecorded-is-null-with-gaps"),
        pytest.param([(2, 0), (None, 0)], None, 0, id="one-unknown-series-makes-the-sum-unknown"),
        pytest.param([(0, 0)], 0, 0, id="control-known-zero-is-zero"),
        pytest.param([(2, 0), (1, 4)], 3, 4, id="control-known-counts-sum"),
    ],
)
def test_card_status_counts(
    statuses: list[tuple[int | None, int | None]], estimated: int | None, provisional: int | None
) -> None:
    """The card counts the status of the series it draws, never the version's."""
    record = _dataset_record(
        variables=_variables(_WORKED_DIMENSIONS),
        series=_series(status=StatusCounts(estimated=40, provisional=7)),
    )
    selection = _selection(
        *[
            _drawn(
                f"series-{n}",
                {_SECTOR: _BASIC_HEALTH[0]},
                status=StatusCounts(estimated=e, provisional=p),
            )
            for n, (e, p) in enumerate(statuses)
        ]
    )

    _assert_status(_describe_card(record, selection), estimated, provisional)


# Pairings


def _mentions(served: dict[str, Any], code: tuple[str, str]) -> bool:
    text = json.dumps(served)
    return code[0] in text or code[1] in text


def test_same_totals_different_pairings_give_different_card_descriptions() -> None:
    """Design acceptance case 1. One version pairs health with grant and education with loan,
    the other the reverse. Their records hold the same dimension totals, so only the card's
    selection facts tell a health-only card on each apart."""
    dimensions = [
        _open(_SECTOR, [(_HEALTH, 1), (_EDUCATION, 1)]),
        _open("crsTypeOfFinance", [(_GRANT[0], 1), (_LOAN[0], 1)]),
    ]
    record_a = _dataset_record(variables=_variables(dimensions))
    record_b = _dataset_record(
        variables=_variables(dimensions), id=_OTHER_SPEC_ID, content_version=_OTHER_CV
    )
    health_grant = _selection(
        _drawn("health-grant", {_SECTOR: _HEALTH, "crsTypeOfFinance": _GRANT[0]})
    )
    health_loan = _selection(
        _drawn("health-loan", {_SECTOR: _HEALTH, "crsTypeOfFinance": _LOAN[0]})
    )
    assert record_a.variables == record_b.variables, "the totals alone cannot tell them apart"

    served_a = _describe_card(record_a, health_grant)
    served_b = _describe_card(record_b, health_loan)

    sector_drawn = {"property": {"ref": _SECTOR, "label": "sector"}, "count": 1, "of": 2}
    for served in (served_a, served_b):
        assert served["describes"] == {
            "kind": "card_revision",
            "component_id": _COMPONENT_ID,
            "revision": _REVISION,
        }
        dimensions_drawn = served["about"]["drawn"]["dimensions"]
        assert sector_drawn in [_cut(dim, sector_drawn) for dim in dimensions_drawn]
    assert _mentions(served_a, _GRANT) and not _mentions(served_a, _LOAN)
    assert _mentions(served_b, _LOAN) and not _mentions(served_b, _GRANT)


# An old version


def _backfilled() -> DescriptionRecord:
    """A version prepared before records existed and backfilled on 12 October 2026: a variable
    with data has left the code index, the body holds no observation counts and no family
    member count, and the series statistics have left the working tier."""
    return _dataset_record(
        basis="backfill",
        variables=_variables(
            [_open(_SECTOR, [(_WATER[0], 3), (_LOST_SECTOR, 5)], partial=True)],
            admitted_at_mint=312,
            admitted_from="resolved_variables",
            with_data=None,
        ),
        series=_series(
            span=None,
            grain=None,
            latest_period=None,
            series_reaching_latest=None,
            by_source_unit=None,
        ),
        evidence=_evidence(
            labels=[
                LabelEvidence(ref="country/NGA", label="Nigeria", read=_PREPARED),
                LabelEvidence(ref=_LOST_SECTOR, label="retired sector", read=_BACKFILLED),
                LabelEvidence(ref=_WATER[0], label="water supply and sanitation", read=_BACKFILLED),
            ]
        ),
    )


def test_old_version_reports_partial_and_unknown_counts_and_keeps_its_population() -> None:
    """Design acceptance case 4."""
    served = _describe_version(_backfilled())
    sector = _dimension(served, _SECTOR)

    assert sector["partial"] is True
    assert sector["count"] == 2
    assert served["about"]["variables"] == {"admitted_at_mint": 312, "with_data": None}
    assert served["about"]["series"] == 523
    assert served["about"]["period"] is None
    assert served["freshness"]["latest_period"] is None
    assert served["freshness"]["series_reaching_it"] is None
    assert {
        "about.variables.with_data",
        "about.period",
        "freshness.latest_period",
        "freshness.series_reaching_it",
    } <= set(served["gaps"])


def test_old_version_keeps_lost_words_from_the_mint_and_takes_todays_otherwise() -> None:
    """A code today's vocabulary has lost reads by its evidence label, marked `at_mint`. A
    code whose words changed reads by today's words, unmarked."""
    sector = _dimension(_describe_version(_backfilled()), _SECTOR)
    expected = [
        {"label": "retired sector", "variables": 5, "at_mint": True},
        {"label": "water supply", "variables": 3},
    ]

    assert [
        _cut(item, model) for item, model in zip(sector["top"], expected, strict=True)
    ] == expected
    assert "at_mint" not in sector["top"][1]


# Computed results


def _computed_record(
    steps: list[Step],
    caveats: list[CaveatFact],
    inputs: list[VersionRef],
    frame_units: Sequence[Unit] | None = None,
) -> DescriptionRecord:
    """A computed version's record; `frame_units` are the units its frame's series hold, as
    the frame's own unit column gives them, and `None` leaves them unknown."""
    return DescriptionRecord(
        id=_OPS_ID,
        content_version=_OCV,
        schema_version=1,
        basis="publish",
        index_stamp=None,
        variables=None,
        computed=ComputedFacts(
            inputs=[InputVersion.model_validate(version.model_dump()) for version in inputs],
            composed=None,
        ),
        series=_series(
            series=38,
            rows=532,
            periods=15,
            span=Span(start="2010", end="2024"),
            series_reaching_latest=38,
            by_source_unit=None
            if frame_units is None
            else [SourceUnitCount(source=None, unit=unit, series=38) for unit in frame_units],
            status=StatusCounts(estimated=0, provisional=0),
        ),
        steps=steps,
        caveats=caveats,
        evidence=_evidence(
            prepared=datetime.date(2026, 10, 11),
            title="Gross ODA disbursements per gross national income",
            sources=[],
            labels=[],
            builds=Builds(
                origin_catalogue=None, code_index=None, coverage=None, method_status=None
            ),
        ),
        late_fields=[],
    )


def _retrieve(step_id: str, version: VersionRef) -> Step:
    return Step(id=step_id, op="retrieve", input=version, options={})


def _growth_record() -> DescriptionRecord:
    return _computed_record(
        steps=[
            _retrieve("s1", _ODA),
            Step(
                id="s2",
                op="growth",
                input="s1",
                policy="growth_endpoint_years",
                params={"start": "2015", "end": "2024"},
                kept={"entities": 20},
                left_out=[LeftOut(reason="endpoint_missing", count=25, of="entities")],
            ),
        ],
        caveats=[
            CaveatFact(
                kind="coverage_gap", step="s2", entities=["country/SSD", "country/SOM"], count=25
            )
        ],
        inputs=[_ODA],
    )


@pytest.mark.parametrize(
    ("growth_unit", "unit"),
    [
        pytest.param(None, "Percent", id="output-unit-omitted"),
        pytest.param(
            SimpleUnit(ref="Percent", label="percent"), "Percent", id="output-unit-agrees"
        ),
        pytest.param(_USD, None, id="output-unit-disagrees"),
    ],
)
def test_a_computed_versions_unit_is_its_frames(
    growth_unit: SimpleUnit | None, unit: str | None
) -> None:
    """The growth of a series retrieved in US dollars is in percent, as the frame's unit
    column says; the retrieval's unit is its own output's. A last step that states a unit
    other than the frame's leaves the unit a gap."""
    retrieved, growth = _growth_record().steps or []
    steps = [
        retrieved.model_copy(update={"unit": _USD}),
        growth if growth_unit is None else growth.model_copy(update={"unit": growth_unit}),
    ]
    record = _computed_record(
        steps=steps, caveats=[], inputs=[_ODA], frame_units=[SimpleUnit(ref="Percent")]
    )

    served = _describe_version(record)

    assert (served["about"]["unit"] or {}).get("ref") == unit
    assert ("about.unit" in served["gaps"]) is (unit is None)


def test_growth_step_reports_what_it_kept_of_all_it_was_given() -> None:
    """Design acceptance case 7: a growth step over 45 entities, 25 of them without an end
    value, keeps 20 of 45."""
    served = _describe_version(_growth_record())
    growth = _step(served, "s2")

    assert served["describes"] == {
        "kind": "computed_version",
        "id": _OPS_ID,
        "content_version": _OCV,
    }
    assert growth["kept"] == {"entities": 20}
    assert growth["left_out"] == [{"reason": "endpoint_missing", "count": 25, "of": "entities"}]
    given = growth["kept"]["entities"] + sum(
        gap["count"] for gap in growth["left_out"] if gap["of"] == "entities"
    )
    assert given == 45
    (caveat,) = [item for item in served["caveats"] if item["kind"] == "coverage_gap"]
    assert caveat["step"] == "s2"
    assert caveat["count"] == 25
    assert caveat["entities"] == [
        {"ref": "country/SSD", "label": "South Sudan"},
        {"ref": "country/SOM", "label": "Somalia"},
    ]
    assert caveat["text"] == (
        "25 entities, among them South Sudan and Somalia, left out of the change with no value "
        "in 2015 or 2024."
    )


def _quotient_step(op: str = "quotient") -> Step:
    return Step(
        id="s3",
        op=op,
        inputs={"numerator": "s1", "denominator": "s2"},
        policy="quotient_of_intersect_periods",
        params={"numerator_by": ["recipient"]},
        kept={"entities": 38, "periods": 15, "entities_with_fewer_periods": 6},
        left_out=[
            LeftOut(reason="unmatched_key", count=9, of="entities"),
            LeftOut(reason="period_unmatched", count=5, of="entities"),
            LeftOut(reason="zero_denominator", count=2, of="entities"),
        ],
        unit=_PER_USD,
    )


def _quotient_record(op: str = "quotient") -> DescriptionRecord:
    """The design's illustrative chain: gross ODA disbursements over gross national income,
    38 of 54 countries kept, 16 left out for three reasons."""
    return _computed_record(
        steps=[_retrieve("s1", _ODA), _retrieve("s2", _GNI), _quotient_step(op)],
        caveats=[
            CaveatFact(
                kind="coverage_gap", step="s3", entities=["country/SSD", "country/SOM"], count=16
            )
        ],
        inputs=[_ODA, _GNI],
        frame_units=[_PER_USD_CELL],
    )


def _version_link(version: VersionRef) -> dict[str, Any]:
    return {
        "ref": version.id,
        "label": version.label,
        "content_version": version.content_version,
        "description": f"/api/datasets/{version.id}/{version.content_version}/description",
    }


def test_quotient_chain_links_each_input_to_its_own_description() -> None:
    served = _describe_version(_quotient_record())

    expected = [_version_link(_ODA), _version_link(_GNI)]
    assert [
        _cut(source, model) for source, model in zip(served["sources"], expected, strict=True)
    ] == expected
    inputs = [
        {"id": version.id, "content_version": version.content_version} for version in (_ODA, _GNI)
    ]
    for block in (served["freshness"]["inputs"], served["cite"]["inputs"]):
        assert [_cut(item, model) for item, model in zip(block, inputs, strict=True)] == inputs
    assert "about" not in json.dumps(served["sources"]), "no description nests another"


def test_a_computed_versions_data_link_is_its_presentation_frame() -> None:
    record = _quotient_record()

    served = _describe_version(record)

    assert (
        served["cite"]["data"] == f"/api/presentation/frames/{record.id}/{record.content_version}"
    )


def test_quotient_step_names_its_denominator_policy_kept_and_left_out() -> None:
    served = _describe_version(_quotient_record())
    quotient = _step(served, "s3")

    assert quotient["op"] == "quotient"
    assert quotient["inputs"] == {"denominator": "s2", "numerator": "s1"}
    assert _step(served, "s2")["input"]["label"] == "Gross national income"
    assert quotient["policy"] == "quotient_of_intersect_periods"
    assert quotient["params"] == {"numerator_by": ["recipient"]}
    assert quotient["kept"] == {"entities": 38, "entities_with_fewer_periods": 6, "periods": 15}
    assert quotient["left_out"] == [
        {"reason": "unmatched_key", "count": 9, "of": "entities"},
        {"reason": "period_unmatched", "count": 5, "of": "entities"},
        {"reason": "zero_denominator", "count": 2, "of": "entities"},
    ]
    assert quotient["kept"]["entities"] + sum(gap["count"] for gap in quotient["left_out"]) == 54
    assert quotient["unit"] == _PER_USD_WIRE
    assert not [gap for gap in served["gaps"] if gap.startswith("steps")], (
        "every operation in the chain is one the kit lists"
    )


def test_quotient_chain_steps_read_as_the_designs_sentences() -> None:
    """Each served step's `text` is the design's sentence for it. The design's chain reads
    "Retrieved gross ODA disbursements. Retrieved gross national income. Divided ...", one
    sentence per step; the step records name the reasons by the kit's names."""
    served = _describe_version(_quotient_record())

    assert [(step["id"], step["text"]) for step in served["steps"]] == [
        ("s1", "Retrieved gross ODA disbursements."),
        ("s2", "Retrieved gross national income."),
        (
            "s3",
            "Divided gross ODA disbursements by gross national income for each year both "
            "report: 38 of 54 recipients kept, 6 of them with fewer than 15 years; 16 left out, "
            "9 with no gross national income figure, 5 sharing no year with it and 2 with zero "
            "gross national income.",
        ),
    ]


def test_quotient_chain_takes_the_composed_unit_and_a_name_from_its_steps() -> None:
    served = _describe_version(_quotient_record())

    assert served["about"]["unit"] == _PER_USD_WIRE
    assert served["about"]["name"].casefold() == (
        "gross ODA disbursements per gross national income".casefold()
    )
    assert served["cite"]["title"] == served["title"] == served["about"]["name"]
    assert "cite.title" not in served["gaps"]
    assert served["freshness"]["computed"] == "2026-10-11"
    (caveat,) = served["caveats"]
    assert (caveat["kind"], caveat["step"], caveat["count"]) == ("coverage_gap", "s3", 16)
    assert caveat["text"] == (
        "16 recipients, among them South Sudan and Somalia, left out of the division: 9 with no "
        "gross national income figure, 5 sharing no year with it and 2 with zero gross national "
        "income."
    )


# An unknown operation


def _with_unknown_mint_step(op: str) -> DescriptionRecord:
    return _dataset_record(
        variables=_variables(_WORKED_DIMENSIONS),
        steps=[Step(id="m1", op=op, options={"weights": "population"})],
    )


@pytest.mark.parametrize(
    ("record", "step_id", "op"),
    [
        pytest.param(_quotient_record("quotient_v2"), "s3", "quotient_v2", id="computed-version"),
        pytest.param(_with_unknown_mint_step("reweight"), "s1", "reweight", id="dataset-version"),
    ],
)
def test_unknown_operation_is_a_gap_that_carries_its_name(
    record: DescriptionRecord, step_id: str, op: str
) -> None:
    assert op not in {item["op"] for item in operations()["operations"]}

    served = _describe_version(record)

    assert f"steps.{step_id}.op" in served["gaps"]
    assert _step(served, step_id)["op"] == op


# Parties, units, titles


@pytest.mark.parametrize(
    ("family", "recipient", "provider"),
    [
        pytest.param(None, "recipient", "provider", id="no-family-reads-generic"),
        pytest.param("aid", "recipient", "provider", id="aid"),
        pytest.param("debt_stock", "debtor", "creditor", id="ids-debt-stock"),
        pytest.param("debt_lending", "debtor", "creditor", id="ids-debt-lending"),
    ],
)
def test_parties_are_worded_by_the_records_flow_family(
    family: str | None, recipient: str, provider: str
) -> None:
    record = _dataset_record(variables=_variables(_WORKED_DIMENSIONS, family=family))

    served = _describe_version(record)

    (fixed_party,) = [
        item for item in served["about"]["fixed"] if item["property"]["ref"] == "financingRecipient"
    ]
    assert fixed_party["property"]["label"] == recipient
    assert _dimension(served, "financingProvider")["property"]["label"] == provider


@pytest.mark.parametrize(
    ("by_source_unit", "gap"),
    [
        pytest.param(None, True, id="unknown-is-null-with-a-gap"),
        pytest.param(
            [SourceUnitCount(source=_CRS, unit=None, series=523)],
            False,
            id="declared-none-is-null-without-a-gap",
        ),
    ],
)
def test_unknown_unit_and_no_unit_are_told_apart_by_the_gap(
    by_source_unit: list[SourceUnitCount] | None, gap: bool
) -> None:
    record = _dataset_record(
        variables=_variables(_WORKED_DIMENSIONS), series=_series(by_source_unit=by_source_unit)
    )

    served = _describe_version(record)

    assert served["about"]["unit"] is None
    assert ("about.unit" in served["gaps"]) is gap


def test_title_takes_todays_words_and_the_citation_keeps_the_mint_title() -> None:
    record = _dataset_record(
        variables=_variables(_WORKED_DIMENSIONS),
        evidence=_evidence(title="Gross disbursements, Nigeria, Health (CRS purpose 120)"),
    )

    served = _describe_version(record)

    assert served["title"] == "Gross disbursements to Nigeria for health"
    assert served["cite"]["title"] == "Gross disbursements, Nigeria, Health (CRS purpose 120)"


# Card inputs


_FRAME = VersionRef(id=_OPS_ID, content_version=_OCV)


@pytest.mark.parametrize(
    "frame",
    [pytest.param(_FRAME, id="live-render"), pytest.param(None, id="control-issued-revision")],
)
def test_live_render_names_its_frame(frame: VersionRef | None) -> None:
    selection = _selection(_drawn("health", {_SECTOR: _BASIC_HEALTH[0]}))

    served = _describe_card(
        _dataset_record(variables=_variables(_WORKED_DIMENSIONS)), selection, frame=frame
    )

    if frame is None:
        assert "rendering" not in served
        assert "frame" not in served
    else:
        assert served["rendering"] == "live"
        assert served["frame"] == {"id": _OPS_ID, "content_version": _OCV}


def test_card_steps_follow_the_cards_selection() -> None:
    """A stat card's change derivation is served after the card's selection, worded."""
    change = Step(
        id="change",
        op="growth",
        policy="growth_endpoint_years",
        params={"start": "2010", "end": "2020"},
        kept={"entities": 1},
    )
    selection = _selection(_drawn("health", {_SECTOR: _BASIC_HEALTH[0]}))

    served = _describe_card(
        _dataset_record(variables=_variables(_WORKED_DIMENSIONS)), selection, steps=[change]
    )

    ops = [step["op"] for step in served["steps"]]
    assert ops[-2:] == ["select", "growth"]
    assert served["steps"][-1]["kept"] == {"entities": 1}


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param("select", "change", id="version-ids-the-cards-own"),
        pytest.param("s1", "s2", id="control-numbered-ids"),
    ],
)
def test_card_and_version_steps_serve_under_distinct_ids(first: str, second: str) -> None:
    """A version whose steps take the ids a card's selection and change take serves every step
    under its own id, each input and caveat naming the step it named in its own list."""
    from description.compose import change_step

    growth = _growth_record()
    assert growth.steps is not None and growth.caveats is not None
    record = growth.model_copy(
        update={
            "steps": [
                _retrieve(first, _ODA),
                growth.steps[1].model_copy(update={"id": second, "input": first}),
            ],
            "caveats": [growth.caveats[0].model_copy(update={"step": second})],
        }
    )
    selection = _selection(_drawn("oda", {"recipient": "country/SSD"}))
    change = change_step(derivation="difference", basis=["2015", "2024"])

    version = _describe_version(record)
    card = _describe_card(record, selection, steps=[change])

    def named(served: dict[str, Any]) -> list[tuple[str, str, str | None]]:
        return [
            (step["id"], step["op"], step["input"] if isinstance(step["input"], str) else None)
            for step in served["steps"]
        ]

    assert named(version) == [("s1", "retrieve", None), ("s2", "growth", "s1")]
    assert named(card) == [*named(version), ("s3", "select", None), ("s4", "change", "s3")]
    assert [caveat.get("step") for caveat in version["caveats"]] == ["s2"]
    assert "s2" in [caveat.get("step") for caveat in card["caveats"]]


def test_a_computation_at_the_publish_ceiling_serves_a_card_within_the_kits_bound() -> None:
    """The longest step list the publish door accepts, drawn on by a stat card, serves a step
    list the kit's structure rules accept."""
    from datasets_api.publish_models import MAX_STEPS
    from description.compose import change_step

    steps = [_retrieve("s1", _ODA)] + [
        Step(id=f"s{n}", op="filter", input=f"s{n - 1}", policy="filter_row_value_predicates")
        for n in range(2, MAX_STEPS + 1)
    ]
    record = _computed_record(steps=steps, caveats=[], inputs=[_ODA])
    selection = _selection(_drawn("oda", {"recipient": "country/SSD"}))

    served = _describe_card(
        record,
        selection,
        steps=[change_step(derivation="difference", basis=["2015", "2024"])],
    )

    assert len(served["steps"]) == MAX_STEPS + 2
    assert len({step["id"] for step in served["steps"]}) == len(served["steps"])


# A card's measure, fixed values and drawn values


def _drawn_dimension(served: dict[str, Any], prop: str) -> dict[str, Any]:
    (found,) = [
        dim for dim in served["about"]["drawn"]["dimensions"] if dim["property"]["ref"] == prop
    ]
    return found


def test_a_card_names_its_versions_measure_fixed_values_and_the_codes_it_draws() -> None:
    """The card's measure is its version's. Its fixed values leave out the measure axes,
    which the measure names, and each carries the definition the vocabulary holds. Each
    dimension lists the codes the card draws in the order it first draws them, with theirs."""
    selection = _selection(
        _drawn("policy", {_SECTOR: _HEALTH_POLICY[0]}),
        _drawn("basic", {_SECTOR: _BASIC_HEALTH[0], "financingProvider": _PROVIDERS[1][0]}),
        _drawn("policy-again", {_SECTOR: _HEALTH_POLICY[0], "financingProvider": _PROVIDERS[0][0]}),
    )

    served = _describe_card(_worked_example(), selection)

    assert served["about"]["measure"] == {"label": "gross disbursements", "definition": None}
    assert "about.measure.definition" in served["gaps"]
    assert served["about"]["fixed"] == [
        {
            "property": {"ref": "financingRecipient", "label": "recipient"},
            "value": {"ref": "country/NGA", "label": "Nigeria"},
        }
    ]
    assert _drawn_dimension(served, _SECTOR) == {
        "property": {"ref": _SECTOR, "label": "sector"},
        "count": 2,
        "of": 19,
        "values": [
            {"ref": dcid, "label": label, "definition": _DEFINED[dcid]}
            for dcid, label in (_HEALTH_POLICY, _BASIC_HEALTH)
        ],
    }
    assert _drawn_dimension(served, "financingProvider")["values"] == [
        {"ref": dcid, "label": label} for dcid, label in (_PROVIDERS[1], _PROVIDERS[0])
    ]


def test_a_dimension_lists_each_code_it_draws_once_or_none_when_one_is_unlabelled() -> None:
    """A code drawn by several series is listed once, so `values` holds `count` distinct refs,
    as `check_description` requires. A dimension drawing a code with no label serves no
    `values`, with one gap naming the list, while the others keep theirs."""
    unlabelled, labelled = _PROVIDERS[1][0], _PROVIDERS[0]
    words = _words()
    words = replace(
        words,
        labels=MappingProxyType(
            {dcid: label for dcid, label in words.labels.items() if dcid != unlabelled}
        ),
    )
    selection = _selection(
        _drawn("first", {"financingProvider": unlabelled}),
        _drawn("again", {"financingProvider": unlabelled, _SECTOR: _BASIC_HEALTH[0]}),
        _drawn("other", {"financingProvider": labelled[0], _SECTOR: _BASIC_HEALTH[0]}),
    )

    served = _describe_card(_worked_example(), selection, words)

    provider = _drawn_dimension(served, "financingProvider")
    assert provider == {
        "property": {"ref": "financingProvider", "label": "provider"},
        "count": 2,
        "of": len(_PROVIDERS),
    }
    assert [value["ref"] for value in _drawn_dimension(served, _SECTOR)["values"]] == [
        _BASIC_HEALTH[0]
    ]
    position = served["about"]["drawn"]["dimensions"].index(provider)
    assert [gap for gap in served["gaps"] if gap.startswith("about.drawn.")] == [
        f"about.drawn.dimensions.{position}.values"
    ]


def test_a_card_drawing_many_unlabelled_codes_keeps_its_gaps_within_the_schema() -> None:
    """Five dimensions of 24 drawn codes, none labelled, add one gap per list given up, so
    the card's gaps stay under the schema's 100."""
    props = [f"oversizeAxis{n}" for n in range(5)]
    record = _dataset_record(
        variables=_variables(
            [_open(prop, [(f"dc/x/{prop}/{n:02d}", 1) for n in range(24)]) for prop in props]
        )
    )
    selection = _selection(
        *[
            _drawn(f"series-{n}", {prop: f"dc/x/{prop}/{n:02d}" for prop in props})
            for n in range(24)
        ]
    )

    served = _describe_card(record, selection)

    dimensions = served["about"]["drawn"]["dimensions"]
    assert [(dim["count"], "values" in dim) for dim in dimensions] == [(24, False)] * 5
    assert len(served["gaps"]) <= 100


@pytest.mark.parametrize(
    ("drawn", "listed"),
    [
        pytest.param(50, True, id="control-fifty-are-listed"),
        pytest.param(51, False, id="over-fifty-lists-none"),
    ],
)
def test_a_dimension_lists_its_drawn_values_only_when_it_lists_them_all(
    drawn: int, listed: bool
) -> None:
    selection = _selection(
        *[_drawn(dcid, {"financingProvider": dcid}) for dcid, _ in _PROVIDERS[:drawn]]
    )

    provider = _drawn_dimension(_describe_card(_worked_example(), selection), "financingProvider")

    assert provider["count"] == drawn
    assert ("values" in provider) is listed


def test_the_vocabularys_measure_definition_defines_a_versions_and_its_cards_measure() -> None:
    definition = "The resources placed at recipients' disposal, counted in full."
    labelled = replace(
        _MEASURES,
        partition_labels=tuple(
            replace(label, definition=definition) for label in _MEASURES.partition_labels
        ),
    )
    words = replace(_words(), measure_index=labelled)
    selection = _selection(_drawn("basic", {_SECTOR: _BASIC_HEALTH[0]}))

    for served in (
        _describe_version(_worked_example(), words),
        _describe_card(_worked_example(), selection, words),
    ):
        assert served["about"]["measure"] == {
            "label": "gross disbursements",
            "definition": definition,
        }
        assert "about.measure.definition" not in served["gaps"]


def test_a_card_drawn_from_a_computed_version_names_no_measure_or_fixed_values() -> None:
    served = _describe_card(_quotient_record(), _selection(_drawn("ratio", {})))

    assert "measure" not in served["about"]
    assert "fixed" not in served["about"]


_WDI = "worldBank/NY_GDP_MKTP_CD"
_IDS_VARIABLE = "Amount_FinancialFlow_Buyback_PublicOrPubliclyGuaranteed_ByCreditor_Nominal"


@pytest.mark.parametrize(
    ("variable", "description", "definition"),
    [
        pytest.param(
            _WDI,
            "GDP at purchaser's prices is the sum of gross value added by all resident producers "
            "in the economy. (WorldBank Series Code: NY.GDP.MKTP.CD)",
            "GDP at purchaser's prices is the sum of gross value added by all resident producers "
            "in the economy.",
            id="wdi-variable-is-its-description-before-the-series-code",
        ),
        pytest.param(
            _WDI,
            " (WorldBank Series Code: NY.GDP.MKTP.CD)",
            None,
            id="wdi-description-that-is-the-series-code-alone",
        ),
        pytest.param(
            _IDS_VARIABLE, "Debt buyback (current US$)", None, id="ids-title-defines-nothing"
        ),
    ],
)
def test_a_card_drawing_one_world_bank_variable_takes_its_definition(
    variable: str, description: str, definition: str | None
) -> None:
    """A measure no measure label or noun covers is the version's one variable. A card that
    draws that one World Development Indicators variable alone defines it by the variable's
    graph description, read into the vocabulary's definitions."""
    record = _dataset_record(
        variables=_variables(
            [],
            fixed=[],
            shape=[],
            variable=variable,
            admitted_at_mint=1,
            admitted_from="resolved_variables",
            with_data=1,
        )
    )
    words = replace(
        _words(),
        labels=MappingProxyType({**_words().labels, variable: "GDP (current US$)"}),
        definitions=MappingProxyType({variable: description}),
    )

    served = _describe_card(record, _selection(_drawn("gdp", {STAT_VAR_COLUMN: variable})), words)

    assert served["about"]["measure"] == {
        "ref": variable,
        "label": "GDP (current US$)",
        "definition": definition,
    }
    assert ("about.measure.definition" in served["gaps"]) is (definition is None)


_GCF = MeasurePartitionLabel(
    axis_values=(
        ("populationType", "FinalExpenditure"),
        ("measuredProperty", "amount"),
        ("finalExpenditureType", "CapitalFormation"),
        ("fixedCapitalConsumptionBasis", "GrossOfFixedCapitalConsumption"),
    ),
    label="gross capital formation",
    definition="Acquisitions less disposals of produced assets.",
)

_EXTERNAL_DEBT_SERVICE = MeasurePartitionLabel(
    axis_values=(
        ("populationType", "FinancialFlow"),
        ("measuredProperty", "amount"),
        ("statType", "measuredValue"),
        ("financialFlowType", "ServicingPayment"),
    ),
    label="external debt service",
    definition="Principal and interest actually paid on long-term external debt.",
)


@pytest.mark.parametrize(
    ("shape", "fixed", "measure", "kept"),
    [
        pytest.param(
            [
                FixedValue(property="populationType", value="FinalExpenditure"),
                FixedValue(property="measuredProperty", value="amount"),
            ],
            [
                FixedValue(property="financingRecipient", value="country/NGA"),
                FixedValue(property="finalExpenditureType", value="CapitalFormation"),
                FixedValue(
                    property="fixedCapitalConsumptionBasis", value="GrossOfFixedCapitalConsumption"
                ),
            ],
            {
                "label": "gross capital formation",
                "definition": "Acquisitions less disposals of produced assets.",
            },
            ["financingRecipient"],
            id="noun-states-its-constraints",
        ),
        pytest.param(
            [
                FixedValue(property="populationType", value="FinancialFlow"),
                FixedValue(property="measuredProperty", value="amount"),
                FixedValue(property="statType", value="measuredValue"),
            ],
            [
                FixedValue(property="financingRecipient", value="country/NGA"),
                FixedValue(property="financialFlowType", value="ServicingPayment"),
                FixedValue(property="nettingBasis", value="Gross"),
            ],
            {
                "label": "external debt service",
                "definition": "Principal and interest actually paid on long-term external debt.",
            },
            ["financingRecipient", "nettingBasis"],
            id="noun-keeps-the-measure-axis-it-leaves-uncovered",
        ),
        pytest.param(
            [
                FixedValue(property="populationType", value="Unnamed"),
                FixedValue(property="measuredProperty", value="amount"),
            ],
            [
                FixedValue(property="financingRecipient", value="country/NGA"),
                FixedValue(property="financialFlowType", value="Disbursement"),
                FixedValue(property="nettingBasis", value="Gross"),
            ],
            None,
            ["financingRecipient", "financialFlowType", "nettingBasis"],
            id="no-measure-named-keeps-the-measure-axes",
        ),
    ],
)
def test_a_cards_fixed_values_leave_out_those_its_measure_states(
    shape: list[FixedValue],
    fixed: list[FixedValue],
    measure: dict[str, str] | None,
    kept: list[str],
) -> None:
    """A measure noun states the constraints its key covers ("gross capital formation" states
    the expenditure type and the capital consumption basis), so the card's fixed values leave
    them out, as its title does, and keep the rest, a measure axis the noun leaves uncovered
    among them ("external debt service" states no netting basis). With no measure named,
    nothing states the measure axes, so the fixed values keep the flow type and netting
    basis."""
    words = replace(
        _words(), measure_index=replace(_MEASURES, nouns=(_GCF, _EXTERNAL_DEBT_SERVICE))
    )
    record = _dataset_record(variables=_variables([], shape=shape, fixed=fixed))

    served = _describe_card(record, _selection(_drawn("card", {})), words)

    assert served["about"]["measure"] == measure
    assert [item["property"]["ref"] for item in served["about"]["fixed"]] == kept
