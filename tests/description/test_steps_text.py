"""Tests for the step sentences and computed names, `description.steps_text`."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

import pytest
from dfo_descriptions import operations, schema_dir
from pydantic import TypeAdapter

from code_index import CodeIndex, IndexStamp
from code_index.types import MeasureIndex
from description.citation import FIGURE_CHANGING_OPS, FIGURE_PRESERVING_OPS
from description.facts import ComposedVariable
from description.steps import (
    ComposedUnit,
    EntityCoverage,
    LeftOut,
    SimpleUnit,
    Step,
    StepRef,
    VersionRef,
)
from description.steps_text import (
    composed_names,
    describe_steps,
    name_computed,
    templated_operations,
)
from description.stored import LONG_TEXT_MAX_LENGTH
from description.vocabulary import Vocabulary, build_vocabulary

_STEPS = TypeAdapter(list[Step])

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
_PER_USD = ComposedUnit(
    numerator=SimpleUnit(ref="USDollar", label="US dollars"),
    denominator=SimpleUnit(ref="USDollar", label="US dollars"),
)

_INDEX = CodeIndex(
    stamp=IndexStamp(
        graph_endpoint="https://graph.test",
        exported_at="2026-10-11T00:00:00Z",
        code_counts={},
        statvar_shape_counts={},
        content_hash="index-hash",
    ),
    declarations=[],
    codes=[],
    statvars=(),
)


def _vocabulary(labels: Mapping[str, str] | None = None) -> Vocabulary:
    return build_vocabulary(
        index=_INDEX,
        measure_index=MeasureIndex(axes=(), values=(), partition_labels=()),
        labels=labels or {},
        sources={},
        currency_units={"USDollar": "USD"},
        graph_generation=None,
        catalogue_build=None,
    )


def _retrieve(step_id: str, version: VersionRef) -> Step:
    return Step(id=step_id, op="retrieve", input=version, options={})


def _quotient(op: str = "quotient", params: dict[str, Any] | None = None) -> Step:
    return Step(
        id="s3",
        op=op,
        inputs={"numerator": "s1", "denominator": "s2"},
        policy="quotient_of_intersect_periods",
        params={"numerator_by": ["recipient"]} if params is None else params,
        kept={"entities": 38, "periods": 15, "entities_with_fewer_periods": 6},
        left_out=[
            LeftOut(reason="unmatched_key", count=9, of="entities"),
            LeftOut(reason="period_unmatched", count=5, of="entities"),
            LeftOut(reason="zero_denominator", count=2, of="entities"),
        ],
        unit=_PER_USD,
    )


def _ratio(op: str, inputs: dict[str, Any] | None = None) -> Step:
    return Step(
        id="s3",
        op=op,
        inputs={"numerator": "s1", "denominator": "s2"} if inputs is None else inputs,
    )


def _chain(op: str = "quotient") -> list[Step]:
    return [_retrieve("s1", _ODA), _retrieve("s2", _GNI), _quotient(op)]


def _texts(steps: list[Step], **context: Any) -> list[str]:
    served = describe_steps(steps, _vocabulary(), **context).served
    assert served is not None
    return [str(step["text"]) for step in served]


# The quotient chain


def test_quotient_chain_reads_one_sentence_per_step_with_counts_from_the_record() -> None:
    """38 of 54 is the 38 kept plus the 16 entities left out; the parties are the recipients
    the numerator groups by, and the periods the version's grain."""
    assert _texts(_chain(), grain="year") == [
        "Retrieved gross ODA disbursements.",
        "Retrieved gross national income.",
        "Divided gross ODA disbursements by gross national income for each year both report: "
        "38 of 54 recipients kept, 6 of them with fewer than 15 years; 16 left out, 9 with no "
        "gross national income figure, 5 sharing no year with it and 2 with zero gross "
        "national income.",
    ]


def test_a_share_within_its_own_total_reads_each_party_against_the_total() -> None:
    steps = [
        _retrieve("s1", _ODA),
        Step(id="s2", op="aggregate", input="s1"),
        Step(
            id="s3",
            op="share",
            inputs={"numerator": "s1", "denominator": "s2"},
            params={"numerator_by": ["recipient"]},
        ),
    ]

    assert _texts(steps, grain="year")[-1] == (
        "Took each recipient's share of the total of gross ODA disbursements for each year."
    )


@pytest.mark.parametrize(
    ("context", "params", "parties", "periods"),
    [
        pytest.param({"grain": "year"}, None, "recipients", "years", id="aid-generic"),
        pytest.param(
            {"grain": "year", "family": "debt_stock"}, None, "debtors", "years", id="debt-family"
        ),
        pytest.param({}, None, "recipients", "periods", id="no-grain-reads-periods"),
        pytest.param({"grain": "year"}, {}, "entities", "years", id="no-role-reads-entities"),
    ],
)
def test_parties_and_periods_take_their_nouns_from_data(
    context: dict[str, Any], params: dict[str, Any] | None, parties: str, periods: str
) -> None:
    steps = [_retrieve("s1", _ODA), _retrieve("s2", _GNI), _quotient(params=params)]

    text = _texts(steps, **context)[-1]

    assert f"38 of 54 {parties} kept" in text
    assert f"fewer than 15 {periods}" in text


def test_a_step_record_is_served_whole_with_its_sentence() -> None:
    described = describe_steps(_chain(), _vocabulary())

    assert described.gaps == []
    assert described.served is not None
    quotient = described.served[-1]
    assert {key: value for key, value in quotient.items() if key != "text"} == json.loads(
        _quotient().model_dump_json()
    )


def test_growth_counts_what_it_kept_of_all_it_was_given() -> None:
    growth = Step(
        id="s2",
        op="growth",
        input="s1",
        policy="growth_endpoint_years",
        params={"start": "2015", "end": "2024"},
        kept={"entities": 20},
        left_out=[LeftOut(reason="endpoint_missing", count=25, of="entities")],
    )

    assert _texts([_retrieve("s1", _ODA), growth], grain="year")[-1] == (
        "Took the change in gross ODA disbursements from 2015 to 2024: 20 of 45 entities "
        "kept; 25 left out with no value in 2015 or 2024."
    )


@pytest.mark.parametrize(
    ("derivation", "text"),
    [
        pytest.param(
            "percent_change",
            "Took the percent change in gross ODA disbursements from 2015 to 2024.",
            id="percent-change",
        ),
        pytest.param(
            "difference",
            "Took the difference in gross ODA disbursements from 2015 to 2024.",
            id="difference",
        ),
    ],
)
def test_a_stat_cards_change_names_its_derivation_and_basis(derivation: str, text: str) -> None:
    select = Step(id="select", op="select", input=_ODA, policy="card_selection")
    change = Step(
        id="change",
        op="change",
        input="select",
        policy="change_between_basis_periods",
        params={"derivation": derivation, "basis": ["2015", "2024"]},
    )

    assert _texts([select, change])[-1] == text


def test_rank_counts_shortfall_from_entity_coverage() -> None:
    rank = Step(
        id="s2",
        op="rank",
        input="s1",
        policy="rank_what_is_present",
        params={"limit": 2, "order": "descending"},
        kept={"entities": 2},
        entity_coverage=[
            EntityCoverage(entity=StepRef(ref="country/NGA"), periods=15, of=15),
            EntityCoverage(entity=StepRef(ref="country/ETH"), periods=14, of=15),
        ],
    )

    text = _texts([_retrieve("s1", _ODA), rank], grain="year")[-1]

    assert text == (
        "Ranked entities by gross ODA disbursements, keeping the top 2: 2 entities kept, "
        "1 of them with fewer than 15 years."
    )


@pytest.mark.parametrize(
    ("op", "params", "family", "text"),
    [
        pytest.param(
            "share",
            {"numerator_by": ["financingRecipient"]},
            None,
            "Took each recipient's share of the total of gross ODA disbursements for each year: "
            "2 recipients kept; left out 1 provider that totals other providers.",
            id="share",
        ),
        pytest.param(
            "aggregate",
            {"group_by": ["financingRecipient"]},
            None,
            "Aggregated gross ODA disbursements, skipping absent values: 2 recipients kept; "
            "left out 1 provider that totals other providers.",
            id="aggregate",
        ),
        pytest.param(
            "aggregate",
            {"group_by": ["financingRecipient"]},
            "debt_stock",
            "Aggregated gross ODA disbursements, skipping absent values: 2 debtors kept; "
            "left out 1 creditor that totals other creditors.",
            id="aggregate-debt-family",
        ),
    ],
)
def test_a_pooled_party_total_is_worded_by_its_role_outside_the_kept_count(
    op: str, params: dict[str, Any], family: str | None, text: str
) -> None:
    """A party total of the pooled role is named by that role's noun and never joins the
    kept parties' "of N"."""
    inputs: dict[str, Any] = (
        {"inputs": {"numerator": "s1", "denominator": "s1"}} if op == "share" else {"input": "s1"}
    )
    step = Step(
        id="s2",
        op=op,
        **inputs,
        params=params,
        kept={"entities": 2},
        left_out=[
            LeftOut(reason="party_total", count=1, of="pooled_entities", role="financingProvider")
        ],
    )

    assert _texts([_retrieve("s1", _ODA), step], grain="year", family=family)[-1] == text


def test_a_pooled_count_of_a_role_with_no_noun_reads_parties_with_a_gap() -> None:
    step = Step(
        id="s2",
        op="aggregate",
        input="s1",
        params={"group_by": ["financingRecipient"]},
        kept={"entities": 2},
        left_out=[LeftOut(reason="party_total", count=3, of="pooled_entities", role="sector")],
    )

    described = describe_steps([_retrieve("s1", _ODA), step], _vocabulary(), grain="year")

    assert described.served is not None
    assert str(described.served[-1]["text"]).endswith(
        "2 recipients kept; left out 3 parties that total other parties."
    )
    assert described.gaps == ["steps.s2.left_out.0.role"]


# Labels


@pytest.mark.parametrize(
    ("labels", "stored", "at_mint", "label", "gap"),
    [
        pytest.param({"country/NGA": "Nigeria"}, "Old name", True, "Nigeria", False, id="today"),
        pytest.param({}, "Old name", True, "Old name", False, id="stored-keeps-at-mint"),
        pytest.param({}, None, False, None, True, id="none-is-a-gap"),
    ],
)
def test_a_ref_takes_todays_label_then_the_stored_one_else_a_gap(
    labels: dict[str, str], stored: str | None, at_mint: bool, label: str | None, gap: bool
) -> None:
    entity = (
        StepRef(ref="country/NGA", label=stored, at_mint=True)
        if stored
        else StepRef(ref="country/NGA")
    )
    rank = Step(
        id="s2",
        op="rank",
        input="s1",
        entity_coverage=[EntityCoverage(entity=entity, periods=15, of=15)],
    )

    described = describe_steps([_retrieve("s1", _ODA), rank], _vocabulary(labels))

    assert described.served is not None
    step: dict[str, Any] = described.served[-1]
    served = step["entity_coverage"][0]["entity"]
    assert served.get("label") == label
    assert ("at_mint" in served) is (label is not None and label == stored and at_mint)
    assert ("steps.s2.entity_coverage.0.entity.label" in described.gaps) is gap


def test_an_input_with_no_label_is_named_by_its_id_with_a_gap() -> None:
    bare = VersionRef(id=_ODA.id, content_version=_ODA.content_version)

    described = describe_steps([_retrieve("s1", bare)], _vocabulary())

    assert described.served is not None
    assert described.served[0]["text"] == f"Retrieved {_ODA.id}."
    assert described.gaps == ["steps.s1.input.label"]


# What cannot be worded


def test_unknown_operation_reads_neutrally_with_a_gap_and_no_name() -> None:
    described = describe_steps(_chain("quotient_v2"), _vocabulary())

    assert described.served is not None
    assert described.served[-1]["text"] == (
        "Applied quotient_v2, an operation this release does not describe."
    )
    assert described.gaps == ["steps.s3.op"]
    assert name_computed(_chain("quotient_v2"), _vocabulary()) is None


def test_unknown_reason_is_worded_by_its_name_with_a_gap() -> None:
    growth = Step(
        id="s2",
        op="growth",
        input="s1",
        kept={"entities": 3},
        left_out=[LeftOut(reason="cosmic_rays", count=1, of="entities")],
    )

    described = describe_steps([_retrieve("s1", _ODA), growth], _vocabulary())

    assert described.served is not None
    assert str(described.served[-1]["text"]).endswith("1 left out for cosmic_rays.")
    assert described.gaps == ["steps.s2.left_out.0.reason"]


@pytest.mark.parametrize(
    ("params", "text", "gaps"),
    [
        pytest.param(
            {"title": "Aggregate totals", "description": "Activities are summed."},
            "Activities are summed.",
            [],
            id="description-as-recorded",
        ),
        pytest.param(
            {"title": "Aggregate totals"},
            "Aggregate totals.",
            ["steps.p1.params.description"],
            id="title-with-a-gap",
        ),
        pytest.param(
            {"description": "A" * 30_000},
            "A" * (LONG_TEXT_MAX_LENGTH - 1) + "…",
            [],
            id="overlong-description-clipped",
        ),
    ],
)
def test_a_source_pipeline_step_reads_as_its_recorded_description(
    params: dict[str, Any], text: str, gaps: list[str]
) -> None:
    step = Step(id="p1", op="source_step", params={"source": "provenance/X", **params})

    described = describe_steps([step], _vocabulary())

    assert described.served is not None
    assert (described.served[0]["text"], described.gaps) == (text, gaps)


@pytest.mark.parametrize(
    ("periods", "grain", "text"),
    [
        pytest.param(
            ["2026", "2030"],
            "year",
            "Took the same source's estimated values for 2026 to 2030, after its reported "
            "series ends.",
            id="first-and-last-period",
        ),
        pytest.param(
            ["2030", "2030"],
            "year",
            "Took the same source's estimated values for 2030, after its reported series ends.",
            id="one-period",
        ),
        pytest.param(
            None,
            "year",
            "Took the same source's estimated values for the later years, after its reported "
            "series ends.",
            id="no-periods-recorded",
        ),
    ],
)
def test_a_continued_series_reads_as_the_same_sources_estimates(
    periods: list[str] | None, grain: str, text: str
) -> None:
    params: dict[str, Any] = {"method": "Estimated"}
    if periods is not None:
        params["periods"] = periods
    step = Step(
        id="m2", op="continue_series", policy="continue_with_source_estimates", params=params
    )

    described = describe_steps([step], _vocabulary(), grain=grain)

    assert described.served is not None
    assert (described.served[0]["text"], described.gaps) == (text, [])


def test_no_steps_is_null_with_a_gap() -> None:
    described = describe_steps(None, _vocabulary())

    assert (described.served, described.gaps) == (None, ["steps"])


# Computed names


@pytest.mark.parametrize(
    ("steps", "name"),
    [
        pytest.param(_chain(), "gross ODA disbursements per gross national income", id="quotient"),
        pytest.param(
            [_retrieve("s1", _ODA), Step(id="s2", op="growth", input="s1")],
            "change in gross ODA disbursements",
            id="growth",
        ),
        pytest.param(
            [_retrieve("s1", _ODA), _ratio("share", {"numerator": "s1", "denominator": "s1"})],
            "share of gross ODA disbursements",
            id="share-within-its-own-step",
        ),
        pytest.param(
            [_ratio("share", {"numerator": _ODA, "denominator": _ODA})],
            "share of gross ODA disbursements",
            id="share-within-its-own-version",
        ),
        pytest.param(
            [
                _retrieve("s1", _ODA),
                Step(id="s2", op="aggregate", input="s1"),
                _ratio("share", {"numerator": "s1", "denominator": "s2"}),
            ],
            "share of gross ODA disbursements",
            id="share-of-its-aggregate",
        ),
        pytest.param(
            [
                _retrieve("s1", _ODA),
                _retrieve("s2", _GNI),
                Step(id="s4", op="aggregate", input="s2"),
                _ratio("share", {"numerator": "s1", "denominator": "s4"}),
            ],
            "gross ODA disbursements as a share of gross national income",
            id="share-of-another-variables-aggregate",
        ),
        pytest.param(
            [_retrieve("s1", _ODA), _retrieve("s2", _GNI), _ratio("share")],
            "gross ODA disbursements as a share of gross national income",
            id="share-of-one-variable-in-another",
        ),
        pytest.param(
            [_retrieve("s1", _ODA), _ratio("quotient", {"numerator": "s1", "denominator": "s1"})],
            "gross ODA disbursements per gross ODA disbursements",
            id="quotient-of-one-variable-keeps-per",
        ),
        pytest.param(
            [_retrieve("s1", VersionRef(id=_ODA.id, content_version=_ODA.content_version))],
            None,
            id="unlabelled-input",
        ),
        pytest.param([Step(id="m1", op="choose_source")], None, id="mint-choice"),
        pytest.param([], None, id="no-steps"),
    ],
)
def test_name_computed_derives_the_name_from_the_steps(steps: list[Step], name: str | None) -> None:
    assert name_computed(steps, _vocabulary()) == name


# The comparison

_GBR = {"donor": "dcid:country/GBR"}
_FRA = {"donor": "dcid:country/FRA"}
_KEN = "dcid:country/KEN"
_BILATERAL = "spec_9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a49"
_COMPARED = {
    "country/GBR": "United Kingdom",
    "country/FRA": "France",
    "country/KEN": "Kenya",
    "country/ETH": "Ethiopia",
    _BILATERAL: "Bilateral ODA, gross disbursements",
}


_ODA_FROM_GBR = VersionRef(
    id="spec_" + "b" * 32,
    content_version="cv_" + "b" * 32,
    label="Gross ODA disbursements from United Kingdom",
)
_ODA_FROM_FRA = VersionRef(
    id="spec_" + "f" * 32,
    content_version="cv_" + "f" * 32,
    label="Gross ODA disbursements from France",
)


def _retrieval(step_id: str, content_version: str) -> Step:
    """A retrieval of the bilateral ODA variable, labelled by today's vocabulary alone."""
    return _retrieve(step_id, VersionRef(id=_BILATERAL, content_version=content_version))


def _compare(left: dict[str, str] | None = None, right: dict[str, str] | None = None) -> Step:
    """A comparison whose sides hold the entities `left` and `right`."""
    return Step(
        id="s3",
        op="compare",
        inputs={"left": "s1", "right": "s2"},
        sides={"left": left or {}, "right": right or {}},
        policy="compare_intersect_periods",
    )


@pytest.mark.parametrize(
    ("steps", "text", "name"),
    [
        pytest.param(
            [_retrieve("s1", _ODA), _retrieve("s2", _GNI), _compare()],
            "Calculated gross national income minus gross ODA disbursements for each year both "
            "report.",
            "gross national income minus gross ODA disbursements",
            id="two-variables",
        ),
        pytest.param(
            [
                _retrieve("s1", _ODA),
                _retrieve("s2", _GNI),
                _compare(_GBR).model_copy(update={"params": {"left_entity": _GBR}}),
            ],
            "Calculated gross national income minus gross ODA disbursements (United Kingdom) "
            "for each year both report.",
            "gross national income minus gross ODA disbursements (United Kingdom)",
            id="two-variables-one-side-narrowed",
        ),
        pytest.param(
            [
                _retrieve("s1", _ODA_FROM_GBR),
                _retrieve("s2", _ODA_FROM_FRA),
                _compare(_GBR, _FRA),
            ],
            "Calculated gross ODA disbursements from France minus gross ODA disbursements from "
            "United Kingdom for each year both report.",
            "gross ODA disbursements from France minus gross ODA disbursements from United Kingdom",
            id="separate-retrievals-for-two-donors",
        ),
        pytest.param(
            [
                _retrieval("s1", "cv_0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a"),
                _retrieval("s2", "cv_1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b"),
                _compare(_GBR, _FRA),
            ],
            "Calculated bilateral ODA, gross disbursements, France minus United Kingdom for "
            "each year both report.",
            "bilateral ODA, gross disbursements, France minus United Kingdom",
            id="one-variable-retrieved-twice-for-two-donors",
        ),
        pytest.param(
            [
                _retrieve("s1", _ODA),
                _retrieve("s2", _ODA),
                _compare({**_GBR, "recipient": _KEN}, {**_FRA, "recipient": _KEN}),
            ],
            "Calculated gross ODA disbursements, France minus United Kingdom for each year "
            "both report.",
            "gross ODA disbursements, France minus United Kingdom",
            id="one-variable-two-donors-to-one-recipient",
        ),
        pytest.param(
            [
                _retrieve("s1", _ODA),
                _retrieve("s2", _ODA),
                _compare({**_GBR, "recipient": _KEN}, {**_GBR, "recipient": "dcid:country/ETH"}),
            ],
            "Calculated gross ODA disbursements, Ethiopia minus Kenya for each year both report.",
            "gross ODA disbursements, Ethiopia minus Kenya",
            id="one-variable-one-donor-to-two-recipients",
        ),
        pytest.param(
            [
                _retrieve("s1", _ODA_FROM_FRA),
                _retrieve(
                    "s2",
                    _ODA_FROM_FRA.model_copy(
                        update={"id": "spec_" + "e" * 32, "content_version": "cv_" + "e" * 32}
                    ),
                ),
                _compare(_FRA, _FRA),
            ],
            "Calculated the difference between two series of gross ODA disbursements from "
            "France for each year both report.",
            "difference in gross ODA disbursements from France",
            id="two-versions-with-one-title",
        ),
        pytest.param(
            [
                _retrieve("s1", _ODA),
                _retrieve("s2", _ODA),
                _compare(_FRA, _FRA),
            ],
            "Calculated the difference between two series of gross ODA disbursements for each "
            "year both report.",
            "difference in gross ODA disbursements",
            id="one-variable-same-donor",
        ),
        pytest.param(
            [
                _retrieve("s1", _ODA),
                _retrieve("s2", _ODA),
                _compare({**_GBR, "recipient": _KEN}, {**_FRA, "recipient": "dcid:country/ETH"}),
            ],
            "Calculated the difference between two series of gross ODA disbursements for each "
            "year both report.",
            "difference in gross ODA disbursements",
            id="one-variable-two-roles-differ",
        ),
        pytest.param(
            [_retrieve("s1", _ODA), _retrieve("s2", _ODA), _compare()],
            "Calculated the difference between two series of gross ODA disbursements for each "
            "year both report.",
            "difference in gross ODA disbursements",
            id="one-variable-no-side-narrowed",
        ),
        pytest.param(
            [
                _retrieve("s1", _ODA),
                _retrieve("s2", _ODA),
                _compare().model_copy(
                    update={"params": {"left_entity": _GBR, "right_entity": _FRA}}
                ),
            ],
            "Calculated the difference between two series of gross ODA disbursements for each "
            "year both report.",
            "difference in gross ODA disbursements",
            id="one-variable-bindings-without-sides-name-no-entity",
        ),
        pytest.param(
            [
                _retrieve("s1", _ODA),
                _retrieve("s2", _ODA),
                _compare(_GBR, {"donor": "dcid:country/XYZ"}),
            ],
            "Calculated the difference between two series of gross ODA disbursements for each "
            "year both report.",
            "difference in gross ODA disbursements",
            id="one-variable-unlabelled-entity",
        ),
    ],
)
def test_a_comparison_reads_in_the_order_it_subtracts(
    steps: list[Step], text: str, name: str | None
) -> None:
    """Compare writes right minus left. With one variable on both sides, only the one role
    whose entity differs between the sides tells them apart."""
    vocabulary = _vocabulary(_COMPARED)
    served = describe_steps(steps, vocabulary, grain="year").served

    assert served is not None
    assert (served[-1]["text"], name_computed(steps, vocabulary)) == (text, name)


def _composed(variable: str, relation: str, first: str, second: str) -> ComposedVariable:
    return ComposedVariable(variable=variable, relation=relation, first=[first], second=[second])


_COMPOSED_LABELS = {"ODA": "Aid", "GDP": "Nominal GDP"}


def _bracketed(first: str, joiner: str, second: str) -> str:
    return f"[{first}] {joiner} [{second}]"


@pytest.mark.parametrize(
    ("entries", "names"),
    [
        pytest.param(
            [_composed("ops_a", "share", "ODA", "ODA")], {"ops_a": "Share of Aid"}, id="equal-share"
        ),
        pytest.param(
            [_composed("ops_a", "quotient", "ODA", "ODA")],
            {"ops_a": "Ratio of Aid"},
            id="equal-quotient",
        ),
        pytest.param([_composed("ops_a", "sum", "ODA", "ODA")], {"ops_a": "Aid"}, id="equal-sum"),
        pytest.param(
            [_composed("ops_a", "share", "ODA", "GDP")],
            {"ops_a": "[Aid] as a share of [Nominal GDP]"},
            id="two-sides-take-the-join",
        ),
        pytest.param(
            [
                _composed("ops_a", "sum", "ODA", "GDP"),
                _composed("ops_b", "quotient", "ops_a", "GDP"),
            ],
            {
                "ops_a": "[Aid] plus [Nominal GDP]",
                "ops_b": "[[Aid] plus [Nominal GDP]] per [Nominal GDP]",
            },
            id="a-later-entry-reads-an-earlier-joined-name",
        ),
        pytest.param(
            [_composed("ops_a", "difference", "ODA", "GDP")],
            {"ops_a": "[Aid] minus [Nominal GDP]"},
            id="difference",
        ),
        pytest.param(
            [_composed("ops_a", "difference", "ODA", "ODA")],
            {"ops_a": "Difference in Aid"},
            id="equal-difference",
        ),
        pytest.param([_composed("ops_a", "product", "ODA", "GDP")], {}, id="unworded-relation"),
    ],
)
def test_composed_names_word_each_entry_from_its_sides(
    entries: list[ComposedVariable], names: dict[str, str]
) -> None:
    """Two unequal sides are joined by `join`; equal sides take the relation's prefix."""
    assert composed_names(entries, _COMPOSED_LABELS, join=_bracketed) == names


def test_every_operation_is_classed_as_changing_or_preserving_figures() -> None:
    """A citation reads "processed by" for a step in `FIGURE_CHANGING_OPS`, so an operation
    added to the vocabulary fails here until it is placed in exactly one of the two sets."""
    assert FIGURE_CHANGING_OPS.isdisjoint(FIGURE_PRESERVING_OPS)
    assert templated_operations() == FIGURE_CHANGING_OPS | FIGURE_PRESERVING_OPS


# The kit's operation list and vectors.


@pytest.mark.parametrize("op", [item["op"] for item in operations()["operations"]])
def test_every_kit_operation_has_a_template(op: str) -> None:
    assert op in templated_operations()


def _kit_described_vectors() -> list[str]:
    """The file names of the kit's accepted description vectors."""
    vectors = schema_dir().parent / "vectors"
    return sorted(
        path.name
        for path in vectors.glob("description-*.json")
        if json.loads(path.read_text(encoding="utf-8")).get("expect") == "accept"
    )


@pytest.mark.parametrize("name", _kit_described_vectors())
def test_templates_reproduce_every_served_step_in_the_kit_vectors(name: str) -> None:
    """Every served step's `text` in an accepted kit description is the template's sentence
    for its stored step, and every step gap the templates raise is one the vector names."""
    vector = json.loads((schema_dir().parent / "vectors" / name).read_text(encoding="utf-8"))
    description = vector["input"]
    steps = description["steps"]
    stored = (
        None
        if steps is None
        else _STEPS.validate_python(
            [{key: value for key, value in step.items() if key != "text"} for step in steps]
        )
    )
    grain = (description["about"].get("period") or {}).get("grain")

    described = describe_steps(stored, _vocabulary(), grain=grain)

    if steps is None:
        assert described.served is None
    else:
        assert described.served is not None
        assert [step["text"] for step in described.served] == [step["text"] for step in steps]
    assert set(described.gaps) <= set(description["gaps"])


def test_importing_the_step_sentences_loads_no_reader_or_extra() -> None:
    script = (
        "import sys\n"
        "import description.steps_text\n"
        "leaked = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m.split('.')[0] in {'materialise', 'dc_graph', 'httpx', 'search', 'dfo_descriptions'}\n"
        ")\n"
        "assert not leaked, f'description.steps_text import pulled in: {leaked}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
