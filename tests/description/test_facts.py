"""Tests for the stored description types: the record's facts and evidence, the step records,
and a card's selection facts."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest
from dfo_descriptions import (
    MAX_STEPS,
    StepsError,
    check_description,
    check_steps,
    description_schema,
    steps_schema,
)
from jsonschema import Draft202012Validator
from pydantic import BaseModel, TypeAdapter, ValidationError

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
    LateField,
    SelectionFacts,
    SeriesFacts,
    SourceEvidence,
    SourceUnitCount,
    Span,
    StatusCounts,
    VariableFacts,
    lowest_schema_version,
)
from description.steps import (
    MAX_SERVED_STEPS,
    ComposedUnit,
    LeftOut,
    SimpleUnit,
    Step,
    VersionRef,
)
from description.stored import LONG_TEXT_MAX_LENGTH, Period, Text

_STEPS = TypeAdapter(list[Step])

_SPEC_ID = "spec_91d7345eec81a029629938b3d0ffd775"
_CV = "cv_b785befb9f87471de96eb31cd7c56f5c"
_CRS = "provenance/OECDCreditorReportingSystem"
_STAMP = CodeIndexStamp(
    content_hash="3f1a9c", exported_at="2026-09-27T06:00:00Z", graph_endpoint="dev"
)


def _dimension(
    *, property: str, counts: list[int], undeclared: int = 0, within: str | None = None
) -> DimensionFacts:
    return DimensionFacts(
        property=property,
        within=within,
        values=[
            DimensionValue(value=f"{property}/{n}", variables=count)
            for n, count in enumerate(counts)
        ],
        undeclared_variables=undeclared,
        partial=False,
    )


def _worked_example(**overrides: Any) -> DescriptionRecord:
    """The record behind "Gross disbursements to Nigeria for health", prepared 28 September
    2026: 505 variables admitted, 154 with data, 523 series, no status recorded."""
    fields: dict[str, Any] = {
        "id": _SPEC_ID,
        "content_version": _CV,
        "schema_version": 2,
        "basis": "mint",
        "index_stamp": _STAMP,
        "variables": VariableFacts(
            shape=[
                FixedValue(property="populationType", value="FinancialFlow"),
                FixedValue(property="measuredProperty", value="amount"),
            ],
            variable=None,
            fixed=[
                FixedValue(property="financingRecipient", value="country/NGA"),
                FixedValue(property="financialFlowType", value="Disbursement"),
                FixedValue(property="nettingBasis", value="Gross"),
            ],
            dimensions=[
                _dimension(property="financingProvider", counts=[3] * 63),
                _dimension(
                    property="crsPurpose", counts=[19, 19, *[8] * 17], within="CRSPurposeCode120"
                ),
                _dimension(
                    property="crsCooperationModality", counts=[47, 24, *[7] * 10], undeclared=16
                ),
                _dimension(property="crsTypeOfFinance", counts=[134, 16, 3, 1]),
                _dimension(property="crsTypeOfFlow", counts=[105, 39, 8, 2]),
            ],
            admitted_at_mint=505,
            admitted_from="family_member_count",
            with_data=154,
        ),
        "computed": None,
        "series": SeriesFacts(
            series=523,
            rows=9_814,
            periods=30,
            span=Span(start="1995", end="2024"),
            grain="year",
            latest_period="2024",
            series_reaching_latest=146,
            entities_by_role=None,
            by_source_unit=[
                SourceUnitCount(
                    source=_CRS, unit=SimpleUnit(ref="CurrencyUSD", prices="current"), series=523
                )
            ],
            status=StatusCounts(estimated=None, provisional=None),
        ),
        "steps": [],
        "caveats": [],
        "evidence": Evidence(
            prepared="2026-09-28",
            title="Gross disbursements to Nigeria for health",
            sources=[
                SourceEvidence(
                    ref=_CRS,
                    name=_CRS,
                    licence="OECD Terms and Conditions",
                    downloaded=None,
                    pipeline_steps=None,
                )
            ],
            source_release=None,
            labels=[
                LabelEvidence(ref="country/NGA", label="Nigeria", read="2026-09-28"),
                LabelEvidence(ref="CRSPurposeCode120", label="Health", read="2026-09-28"),
            ],
            membership=None,
            builds=Builds(
                origin_catalogue="ocb_20260927",
                code_index=_STAMP,
                coverage=None,
                method_status=None,
            ),
        ),
        "late_fields": [],
    }
    return DescriptionRecord.model_validate(fields | overrides)


def _quotient_record() -> DescriptionRecord:
    """An `ops_` record whose step records arrived after it was written."""
    gni = VersionRef(id="spec_7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d", content_version=f"cv_{'1' * 32}")
    oda = VersionRef(id=_SPEC_ID, content_version=_CV, label="Gross ODA disbursements")
    record = _worked_example(
        id="ops_3c9e1f7a2b8d4e6f",
        content_version="ocv_8d2a6f1c9e3b7a5d",
        basis="backfill",
        index_stamp=None,
        variables=None,
        computed=ComputedFacts(
            inputs=[InputVersion.model_validate(ref.model_dump()) for ref in (oda, gni)],
            composed=None,
        ),
        late_fields=[
            LateField(field="steps", arrived="2026-10-11T09:30:00+00:00", basis="publish")
        ],
    )
    steps = _STEPS.validate_python(
        [
            {"id": "s1", "op": "retrieve", "input": oda, "options": {}},
            {"id": "s2", "op": "retrieve", "input": gni},
            {
                "id": "s3",
                "op": "quotient",
                "stat_var": "ops_9f8e7d6c5b4a",
                "inputs": {"numerator": "s1", "denominator": "s2"},
                "policy": "quotient_of_intersect_periods",
                "params": {"numerator_by": ["recipient"], "a": {"z": 1, "b": None}},
                "kept": {"entities": 38, "periods": 15},
                "left_out": [{"reason": "zero_denominator", "count": 2, "of": "entities"}],
                "unit": None,
            },
        ]
    )
    caveat = CaveatFact(kind="coverage_gap", entities=["country/SSD"], step="s3", count=2)
    return record.model_copy(update={"steps": steps, "caveats": [caveat]})


def _selection() -> SelectionFacts:
    return SelectionFacts(
        window=Span(start="2010", end="2020"),
        series=[
            DrawnSeries(
                id="s_health_grant",
                coordinates=[
                    Coordinate(column="variableMeasured", value="dcid:health_grant"),
                    Coordinate(column="measurementMethod", value=None),
                ],
                source=_CRS,
                unit=SimpleUnit(ref="USDollar", prices="constant", base_year=2023),
                first_period="2012",
                last_period="2019",
                values=8,
                missing=3,
                status=StatusCounts(estimated=None, provisional=None),
            )
        ],
    )


def _shuffled(value: Any) -> Any:
    """Reverses every object's key order, as jsonb's own key order would differ."""
    if isinstance(value, dict):
        return {key: _shuffled(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_shuffled(item) for item in value]
    return value


@pytest.mark.parametrize(
    "build", [_worked_example, _quotient_record, _selection], ids=["spec", "ops", "selection"]
)
def test_round_trips_byte_for_byte_through_json_and_jsonb_key_order(build):
    model: BaseModel = build()
    stored = model.model_dump_json()

    assert type(model).model_validate_json(stored).model_dump_json() == stored
    from_jsonb = _shuffled(json.loads(stored))
    assert type(model).model_validate(from_jsonb).model_dump_json() == stored


def test_worked_example_keeps_its_counts():
    variables = _worked_example().variables
    assert variables is not None
    assert variables.dimensions is not None
    counts = []
    for dimension in variables.dimensions:
        assert dimension.values is not None
        counts.append(len(dimension.values))
    assert counts == [63, 19, 12, 4, 4]
    assert variables.dimensions[2].undeclared_variables == 16


def _with_data_unknown() -> DescriptionRecord:
    variables = _worked_example().variables
    assert variables is not None
    return _worked_example(variables=variables.model_copy(update={"with_data": None}))


@pytest.mark.parametrize(
    ("build", "path"),
    [
        (_worked_example, ("series", "status", "estimated")),
        (_worked_example, ("series", "status", "provisional")),
        (_with_data_unknown, ("variables", "with_data")),
        (_selection, ("series", 0, "status", "estimated")),
        (_selection, ("series", 0, "status", "provisional")),
    ],
    ids=["estimated", "provisional", "with-data", "card-estimated", "card-provisional"],
)
def test_an_unrecorded_count_stays_null(build, path):
    model: BaseModel = build()
    held: Any = json.loads(
        type(model).model_validate_json(model.model_dump_json()).model_dump_json()
    )
    for key in path:
        held = held[key]

    assert held is None


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (StatusCounts, {}),
        (StatusCounts, {"estimated": None}),
        (
            VariableFacts,
            {"shape": [], "variable": None, "fixed": [], "dimensions": None, "admitted_at_mint": 1},
        ),
    ],
    ids=["status-absent", "provisional-absent", "with-data-absent"],
)
def test_an_unknown_must_be_stated(model, payload):
    """An unknown count has no default, so a writer that omits it is refused instead of
    storing a 0 or a silent null."""
    with pytest.raises(ValidationError, match="missing"):
        model.model_validate(payload)


def test_a_partial_dimension_keeps_admitted_at_mint():
    variables = _worked_example().variables
    assert variables is not None
    assert variables.dimensions is not None
    partial = variables.dimensions[1].model_copy(update={"partial": True})
    record = _worked_example(
        basis="backfill",
        variables=variables.model_copy(update={"dimensions": [partial], "with_data": None}),
    )

    stored = DescriptionRecord.model_validate_json(record.model_dump_json())

    assert stored.variables is not None
    assert stored.variables.dimensions is not None
    assert stored.variables.dimensions[0].partial is True
    assert stored.variables.admitted_at_mint == 505
    assert stored.variables.with_data is None


def _set(payload: dict[str, Any], path: tuple[str | int, ...], value: Any) -> dict[str, Any]:
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return payload


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("colour",), "red"),
        (("series", "status", "imputed"), 0),
        (("evidence", "sources", 0, "publisher"), "OECD"),
        (("schema_version",), 3),
        (("schema_version",), True),
        (("series", "series"), "523"),
        (("series", "series"), 523.0),
        (("series", "series"), -1),
        (("series", "status", "estimated"), False),
        (("variables", "dimensions", 0, "partial"), "yes"),
        (("basis",), "import"),
        (("content_version",), "ocv_8d2a6f1c9e3b7a5d"),
        (("variables",), None),
        (("index_stamp",), None),
        (
            ("late_fields",),
            [{"field": "steps", "arrived": "2026-10-11T09:30:00", "basis": "publish"}],
        ),
        (("caveats",), [{"kind": "coverage_gap", "step": "s9", "count": 4}]),
    ],
    ids=[
        "unknown-member",
        "unknown-status-member",
        "unknown-source-member",
        "schema-version-3",
        "schema-version-true",
        "count-as-string",
        "count-as-float",
        "count-negative",
        "status-as-bool",
        "partial-as-string",
        "unknown-basis",
        "mixed-version-family",
        "spec-without-variables",
        "dimensions-without-index-stamp",
        "late-field-without-timezone",
        "caveat-at-a-step-the-record-lacks",
    ],
)
def test_record_refuses(path, value):
    payload = _set(json.loads(_worked_example().model_dump_json()), path, value)

    with pytest.raises(ValidationError):
        DescriptionRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("unknown", "accepted"),
    [
        (("steps", "caveats", "evidence.labels"), True),
        (("caveats",), False),
        (("evidence.labels",), False),
    ],
    ids=["steps-and-what-derives-from-them", "caveats", "labels"],
)
def test_only_unknown_steps_leave_what_derives_from_them_unknown(unknown, accepted):
    payload = json.loads(_quotient_record().model_dump_json())
    for path in unknown:
        _set(payload, tuple(path.split(".")), None)

    if accepted:
        DescriptionRecord.model_validate(payload)
    else:
        with pytest.raises(ValidationError, match="unknown caveats or labels"):
            DescriptionRecord.model_validate(payload)


def test_a_late_field_is_noted_once():
    late = LateField(field="steps", arrived="2026-10-11T09:30:00Z", basis="publish")
    with pytest.raises(ValidationError, match="filled once"):
        _worked_example(late_fields=[late, late])


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (CaveatFact, {"kind": "estimated"}),
        (CaveatFact, {"kind": "revised", "periods": ["2024"]}),
        (
            CaveatFact,
            {"kind": "coverage_gap", "series": ["sv/a", "sv/b"], "entities": ["country/SSD"]},
        ),
        (
            DrawnSeries,
            json.loads(_selection().series[0].model_dump_json()) | {"first_period": None},
        ),
        (DrawnSeries, json.loads(_selection().series[0].model_dump_json()) | {"values": 0}),
        (Coordinate, {"column": "unit", "value": None, "label": "unit"}),
        (SimpleUnit, {"ref": "USDollar", "prices": "constant", "base_year": "2023"}),
        (SimpleUnit, {"ref": "USDollar", "prices": "constant", "base_year": 1899}),
        (SimpleUnit, {"ref": "USDollar", "prices": "real"}),
        (SimpleUnit, {"ref": "USDollar", "prices": "constant"}),
        (SimpleUnit, {"ref": "USDollar", "prices": "current", "base_year": 2023}),
        (SimpleUnit, {"ref": "USDollar", "base_year": 2023}),
        (SimpleUnit, {"label": "US dollars", "prices": "current"}),
        (SimpleUnit, {"currency": "usd", "prices": "current"}),
        (
            ComposedUnit,
            {"numerator": {"ref": "USDollar"}, "denominator": {"stat_var": "dcid:Count_Person"}},
        ),
        (
            ComposedUnit,
            {
                "numerator": {"ref": "USDollar"},
                "denominator": {"stat_var": "Count_Person", "ref": "X"},
            },
        ),
        (LeftOut, {"reason": "party_total", "count": 1, "of": "pooled_entities"}),
        (
            LeftOut,
            {"reason": "party_total", "count": 1, "of": "entities", "role": "financingProvider"},
        ),
        (
            LeftOut,
            {"reason": "party_total", "count": 1, "of": "pooled_entities", "role": "dcid:provider"},
        ),
        (Step, {"id": "s1", "op": "quotient", "stat_var": "dcid:ops_9f8e7d6c5b4a"}),
    ],
    ids=[
        "caveat-without-target",
        "caveat-kind-unknown",
        "caveat-entities-across-two-series",
        "first-period-without-last",
        "periods-without-values",
        "coordinate-unknown-member",
        "base-year-as-string",
        "base-year-out-of-range",
        "prices-unknown",
        "constant-prices-without-base-year",
        "current-prices-with-base-year",
        "base-year-without-prices",
        "unit-without-ref-or-currency",
        "currency-lowercase",
        "variable-side-prefixed",
        "variable-side-with-ref",
        "pooled-count-without-role",
        "role-on-kept-entities",
        "role-not-a-graph-role-name",
        "step-stat-var-prefixed",
    ],
)
def test_facts_refuse(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_text_loses_nul_and_mappings_sort_their_keys():
    record = _worked_example(
        evidence=_worked_example().evidence.model_copy(
            update={
                "title": None,
                "sources": [
                    SourceEvidence(
                        ref=f"{_CRS}\x00",
                        name="OECD\x00 CRS",
                        licence=None,
                        downloaded=None,
                        pipeline_steps=None,
                    )
                ],
            }
        )
    )
    step = Step.model_validate(
        {"id": "s1", "op": "filter", "params": {"z": "a\x00b", "a": [{"y": 1, "x\x00": "\x00"}]}}
    )

    assert record.evidence.sources[0].model_dump() == {
        "ref": _CRS,
        "name": "OECD CRS",
        "licence": None,
        "downloaded": None,
        "pipeline_steps": None,
    }
    assert step.model_dump_json() == (
        '{"id":"s1","op":"filter","params":{"a":[{"x":"","y":1}],"z":"ab"}}'
    )
    assert "\\u0000" not in record.model_dump_json()


def test_a_label_of_nul_alone_is_refused():
    with pytest.raises(ValidationError):
        LabelEvidence(ref="country/NGA", label="\x00", read="2026-09-28")


def test_importing_the_facts_loads_no_reader_or_extra():
    script = (
        "import sys\n"
        "import description.facts\n"
        "leaked = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m.split('.')[0] in {'materialise', 'dc_graph', 'httpx', 'search', 'dfo_descriptions'}\n"
        ")\n"
        "assert not leaked, f'description.facts import pulled in: {leaked}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr


# The kit's step schema and vectors.


def _kit_vectors() -> list[dict[str, Any]]:
    from importlib import resources

    vectors = resources.files("dfo_descriptions").joinpath("vectors")
    return [
        json.loads(entry.read_text(encoding="utf-8"))
        for entry in sorted(vectors.iterdir(), key=lambda entry: entry.name)
        if entry.name.endswith(".json") and entry.name != "descriptions-vector.schema.json"
    ]


def test_step_models_agree_with_every_kit_vector():
    """Every steps vector the kit's schema accepts validates and dumps back to the same
    JSON; every one it refuses is refused. A structure-rule rejection passes the schema, so
    the models accept it, and the kit's `check_steps` (a steps vector) or `check_description`
    (a description vector) refuses it with the vector's reason. A description vector's step
    list is checked when the description is accepted or breaks a structure rule. The run
    sees a vector of each kind, accepted, schema and structure. A value JSON cannot carry
    (`not_i_json`) breaks a structure rule, so it is not dumped back."""
    schema = Draft202012Validator(steps_schema())
    seen: set[str] = set()
    checked = 0
    for vector in _kit_vectors():
        error = vector.get("error", {})
        kind = error.get("by", "accepted")
        if vector["schema"] == "steps":
            steps = vector["input"]
            check = check_steps
        elif kind == "schema":
            continue
        else:
            steps = vector["input"].get("steps")
            check = check_description
            if isinstance(steps, list):
                # A served step is its stored record plus the composer's `text`.
                steps = [
                    {key: value for key, value in step.items() if key != "text"} for step in steps
                ]
        if isinstance(steps, list):
            assert schema.is_valid(steps) is (kind != "schema"), vector["id"]
            if kind == "schema":
                with pytest.raises(ValidationError):
                    _STEPS.validate_python(steps)
            else:
                validated = _STEPS.validate_python(steps)
                if kind == "accepted":
                    dumped = _STEPS.dump_json(validated)
                    assert json.loads(dumped) == steps, vector["id"]
                    assert schema.is_valid(json.loads(dumped)), vector["id"]
        if kind == "structure":
            with pytest.raises(StepsError) as refused:
                check(vector["input"])
            assert refused.value.reason == error["reason"], vector["id"]
        seen.add(kind)
        checked += 1
    assert seen == {"accepted", "schema", "structure"}
    assert checked >= 20


def test_the_served_step_bound_is_the_kits():
    """`description.steps` restates the kit's count bound, so it imports without the kit."""
    assert MAX_SERVED_STEPS == MAX_STEPS


def test_step_models_dump_what_the_kit_schema_accepts():
    record = _quotient_record()
    assert record.steps is not None
    dumped = json.loads(_STEPS.dump_json(record.steps))

    Draft202012Validator(steps_schema()).validate(dumped)
    check_steps(dumped)


def test_step_text_bounds_are_the_kits():
    """Stored step text and periods hold what the kit's step schema allows and no more, and
    served step text is clipped to the kit's bound on a served step's `text`."""
    step_defs = steps_schema()["$defs"]
    text = step_defs["text"]["maxLength"]
    period = step_defs["step_members"]["properties"]["period_coverage"]["items"]["properties"][
        "period"
    ]["maxLength"]
    served = description_schema()["$defs"]["served_step"]["properties"]["text"]["maxLength"]
    for adapter, bound in ((TypeAdapter(Text), text), (TypeAdapter(Period), period)):
        adapter.validate_python("2" * bound)
        with pytest.raises(ValidationError):
            adapter.validate_python("2" * (bound + 1))
    assert served == LONG_TEXT_MAX_LENGTH


@pytest.mark.parametrize(
    ("within_set", "lowest"),
    [(None, 1), (["CRSPurposeCode121", "CRSPurposeCode122"], 2)],
    ids=["no-set", "set"],
)
def test_a_record_is_of_the_lowest_version_that_describes_it_or_a_later_one(
    within_set: list[str] | None, lowest: int
):
    """Version 1 lacks `within_set`, so a record narrowing a dimension to a set is of version
    2 and refuses a stamp of 1; any other record is of version 1 and reads stamped 1 or 2."""
    variables = _worked_example().variables
    assert variables is not None
    assert variables.dimensions is not None
    dimensions = list(variables.dimensions)
    if within_set is not None:
        dimensions[1] = dimensions[1].model_copy(update={"within": None, "within_set": within_set})
    narrowed = variables.model_copy(update={"dimensions": dimensions})
    payload = json.loads(_worked_example(variables=narrowed).model_dump_json())

    assert lowest_schema_version(narrowed) == lowest
    for version in (1, 2):
        stamped = {**payload, "schema_version": version}
        if version >= lowest:
            assert DescriptionRecord.model_validate(stamped).schema_version == version
        else:
            with pytest.raises(ValidationError, match="schema version 2"):
                DescriptionRecord.model_validate(stamped)
