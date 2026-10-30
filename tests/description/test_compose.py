"""The composer's title wording, its caveat sentences, its source pipeline steps and its
install footprint."""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from dataclasses import replace
from typing import Any

import pytest

from code_index import CodeIndex
from description.compose import change_step
from description.facts import (
    CaveatFact,
    DescriptionRecord,
    DimensionFacts,
    DimensionValue,
    Evidence,
    FixedValue,
    PipelineStep,
    RoleCount,
    SourceEvidence,
)
from description.steps import SimpleUnit, Step, StepRef, Summed
from description.stored import TEXT_MAX_LENGTH
from description.vocabulary import Vocabulary, build_vocabulary
from tests.description.test_accept_compose import (
    _BASIC_HEALTH,
    _EDUCATION,
    _FIXED,
    _GRANT,
    _HEALTH,
    _HEALTH_POLICY,
    _INDEX,
    _MEASURES,
    _SECTOR,
    _WORKED_DIMENSIONS,
    _checked,
    _dataset_record,
    _describe_card,
    _describe_version,
    _drawn,
    _evidence,
    _growth_record,
    _quotient_record,
    _selection,
    _series,
    _variables,
    _words,
)


def _with_short_name(dcid: str, short_name: str | None) -> Vocabulary:
    """Today's words, with `dcid`'s code row carrying `short_name`."""
    words = _words()
    index = CodeIndex(
        stamp=words.index.stamp,
        declarations=words.index.declarations.values(),
        codes=[
            replace(row, short_name=short_name) if row.dcid == dcid else row
            for row in words.index.codes
        ],
        statvars=(),
    )
    return replace(words, index=index)


@pytest.mark.parametrize(
    ("prop", "code", "short_name", "title"),
    [
        pytest.param(
            "crsTypeOfFinance",
            _GRANT[0],
            "standard grants",
            "Gross disbursements to Nigeria as standard grants",
            id="instrument-reads-as-by-its-short-name",
        ),
        pytest.param(
            "crsTypeOfFinance",
            _GRANT[0],
            None,
            "Gross disbursements to Nigeria as standard grant",
            id="instrument-without-a-short-name-reads-its-label",
        ),
        pytest.param(
            _SECTOR,
            _HEALTH,
            None,
            "Gross disbursements to Nigeria for health",
            id="sector-reads-for",
        ),
    ],
)
def test_title_words_a_fixed_code_as_the_search_heading_does(
    prop: str, code: str, short_name: str | None, title: str
) -> None:
    from description.compose import describe_version

    variables = _variables(
        [
            DimensionFacts(
                property="financingProvider",
                within=None,
                values=[DimensionValue(value="dc/o/provider00", variables=1)],
                undeclared_variables=0,
                partial=False,
            )
        ],
        fixed=[*_FIXED, FixedValue(property=prop, value=code)],
    )

    served = describe_version(
        _dataset_record(variables=variables), _with_short_name(code, short_name)
    )

    assert served["title"] == title


_PROVIDER_DIMENSION = DimensionFacts(
    property="financingProvider",
    within=None,
    values=[DimensionValue(value="dc/o/provider00", variables=1)],
    undeclared_variables=0,
    partial=False,
)
_SECTOR_VALUES = [{"value": _EDUCATION, "variables": 1}, {"value": _HEALTH, "variables": 1}]


@pytest.mark.parametrize(
    ("sector", "title"),
    [
        pytest.param(
            {"within_set": [_EDUCATION, _HEALTH], "values": _SECTOR_VALUES},
            "Gross disbursements to Nigeria for education and health",
            id="set-binding",
        ),
        pytest.param(
            {"within_set": [_EDUCATION, _HEALTH], "values": []},
            "Gross disbursements to Nigeria for education and health",
            id="set-binding-summed-to-its-total",
        ),
        pytest.param(
            {"values": _SECTOR_VALUES},
            "Gross disbursements to Nigeria",
            id="stored-before-sets-were-recorded",
        ),
    ],
)
def test_title_names_each_code_of_a_set_binding(sector: dict[str, Any], title: str) -> None:
    """A dimension narrowed to a set names its codes as a list after the property's
    connective, whether it still varies or a collapse summed it to the set's total. A record
    stored before sets were recorded holds no `within_set` and keeps its title."""
    from description.compose import describe_version

    dimension = {
        "property": _SECTOR,
        "within": None,
        "undeclared_variables": 0,
        "partial": False,
        **sector,
    }
    stored = _dataset_record(
        variables=_variables([_PROVIDER_DIMENSION, DimensionFacts.model_validate(dimension)])
    ).model_dump_json()

    record = DescriptionRecord.model_validate_json(stored)

    assert ("within_set" in stored) == ("within_set" in sector)
    assert describe_version(record, _words())["title"] == title


@pytest.mark.parametrize(
    ("partial", "served"), [(False, False), (True, True)], ids=["whole", "partial"]
)
def test_a_dimension_no_variable_holds_a_value_of_is_served_only_while_partial(
    partial: bool, served: bool
) -> None:
    """The definition leaves it open but it varies over nothing, unless a variable that left
    the index may have held a value."""
    empty = DimensionFacts(
        property="dacTypeOfFinance",
        within=None,
        values=[],
        undeclared_variables=1,
        partial=partial,
    )

    varies = _describe_version(_dataset_record(variables=_variables([*_WORKED_DIMENSIONS, empty])))[
        "about"
    ]["varies"]

    assert [entry["property"]["ref"] for entry in varies] == [
        dimension.property for dimension in _WORKED_DIMENSIONS
    ] + (["dacTypeOfFinance"] if served else [])


def _kit_vector(name: str) -> dict[str, Any]:
    from importlib import resources

    entry = resources.files("dfo_descriptions").joinpath("vectors", f"{name}.json")
    return json.loads(entry.read_text(encoding="utf-8"))["input"]


@pytest.mark.parametrize("recorded", [True, False], ids=["pipeline-recorded", "control-unread"])
def test_source_pipeline_steps_lead_the_versions_steps(recorded: bool) -> None:
    """Each pipeline step the mint read is served as a `source_step` ahead of the mint's own
    steps, as the kit's vector serves it, and its gap goes. A pipeline the mint did not read
    stays a gap."""
    from description.compose import describe_version

    vector = _kit_vector("description-source-pipeline-steps")
    pipeline = [
        PipelineStep(title=step["params"]["title"], description=step["params"]["description"])
        for step in vector["steps"]
    ]
    (source,) = _evidence().sources
    evidence = _evidence(
        sources=[
            source.model_copy(
                update={
                    "downloaded": datetime.date(2026, 9, 26),
                    "pipeline_steps": pipeline if recorded else None,
                }
            )
        ]
    )
    record = _dataset_record(
        variables=_variables(_WORKED_DIMENSIONS),
        evidence=evidence,
        steps=[Step(id="m1", op="reweight", options={"weights": "population"})],
    )

    served = describe_version(record, _words())

    assert served["freshness"]["source_downloaded_at_mint"] == "2026-09-26"
    if recorded:
        assert served["steps"][:2] == vector["steps"]
        assert [step["id"] for step in served["steps"]] == ["p1", "p2", "s1"]
        assert "steps.source_pipeline" not in served["gaps"]
    else:
        assert [step["id"] for step in served["steps"]] == ["s1"]
        assert "steps.source_pipeline" in served["gaps"]


@pytest.mark.parametrize("over", [False, True], ids=["control-at-the-bound", "one-past-it"])
def test_pipeline_steps_past_a_cards_room_serve_as_unknown(over: bool) -> None:
    """Pipeline steps that would carry a card drawn from the version past the kit's step
    bound serve as the `steps.source_pipeline` gap, in the version's description and the
    card's, and the mint's and card's steps serve whole."""
    from description.steps import MAX_CARD_STEPS, MAX_SERVED_STEPS

    room = MAX_SERVED_STEPS - MAX_CARD_STEPS - 1
    pipeline = [
        PipelineStep(title=f"Step {n}", description=f"Step {n} of the pipeline.")
        for n in range(room + over)
    ]
    (source,) = _evidence().sources
    record = _dataset_record(
        variables=_variables(_WORKED_DIMENSIONS),
        evidence=_evidence(sources=[source.model_copy(update={"pipeline_steps": pipeline})]),
        steps=[Step(id="m1", op="reweight", options={"weights": "population"})],
    )
    change = Step(id="change", op="change", input="select", policy="change_between_basis_periods")

    version = _describe_version(record)
    card = _describe_card(
        record, _selection(_drawn("s_basic_health", {"stat_var": _BASIC_HEALTH[0]})), steps=[change]
    )

    pipeline_served = 0 if over else room
    assert [step["op"] for step in version["steps"]] == ["source_step"] * pipeline_served + [
        "reweight"
    ]
    assert [step["op"] for step in card["steps"]] == [step["op"] for step in version["steps"]] + [
        "select",
        "change",
    ]
    assert len(card["steps"]) <= MAX_SERVED_STEPS
    for served in (version, card):
        assert ("steps.source_pipeline" in served["gaps"]) is over


def _kit_caveat_text(name: str) -> str:
    (caveat,) = _kit_vector(name)["caveats"]
    return caveat["text"]


_YEARS = [str(year) for year in range(2010, 2016)]


@pytest.mark.parametrize(
    ("caveat", "text"),
    [
        pytest.param(
            CaveatFact(kind="estimated", entities=["country/NGA"], periods=["2022", "2023"]),
            "Values for Nigeria in 2022 and 2023 are estimated.",
            id="estimated-names-what-it-covers",
        ),
        pytest.param(
            CaveatFact(kind="provisional", series=[_BASIC_HEALTH[0]], periods=["2024"], count=7),
            "7 values of basic health care in 2024 are provisional.",
            id="provisional-names-its-count",
        ),
        pytest.param(
            CaveatFact(kind="provisional", series=[_BASIC_HEALTH[0]], periods=["2024"], count=1),
            "1 value of basic health care in 2024 is provisional.",
            id="one-value-reads-singular",
        ),
        pytest.param(
            CaveatFact(kind="break_in_series", series=[_BASIC_HEALTH[0]], periods=["2015"]),
            "A break in series of basic health care in 2015.",
            id="break-in-series",
        ),
        pytest.param(
            CaveatFact(
                kind="coverage_gap", step="s3", entities=["country/SSD", "country/SOM"], count=16
            ),
            "16 recipients, among them South Sudan and Somalia, left out of the division: 9 with "
            "no gross national income figure, 5 sharing no year with it and 2 with zero gross "
            "national income.",
            id="coverage-gap-words-what-its-step-left-out",
        ),
        pytest.param(
            CaveatFact(kind="coverage_gap", step="s3", entities=["country/SSD"], count=3),
            "3 recipients, among them South Sudan, left out of the division.",
            id="coverage-gap-whose-count-is-not-the-steps-gives-no-reasons",
        ),
        pytest.param(
            CaveatFact(kind="estimated", periods=_YEARS[:4]),
            "Values in 2010, 2011, 2012 and 2013 are estimated.",
            id="one-period-past-three-is-named",
        ),
        pytest.param(
            CaveatFact(kind="coverage_gap", entities=["country/SSD"], periods=_YEARS),
            "South Sudan has no data in 2010, 2011, 2012 and 3 more years.",
            id="coverage-gap-without-a-step-counts-periods-past-three",
        ),
    ],
)
def test_caveat_reads_as_its_kinds_sentence(caveat: CaveatFact, text: str) -> None:
    record = _quotient_record().model_copy(update={"caveats": [caveat]})

    (served,) = _describe_version(record)["caveats"]

    assert served["text"] == text


def test_coverage_gap_sentence_is_the_kit_vectors() -> None:
    (served,) = _describe_version(_quotient_record())["caveats"]

    assert served["text"] == _kit_caveat_text("description-computed-quotient")


def test_coverage_gap_reads_its_steps_single_reason() -> None:
    (served,) = _describe_version(_growth_record())["caveats"]

    assert served["text"] == (
        "25 entities, among them South Sudan and Somalia, left out of the change with no value "
        "in 2015 or 2024."
    )


def test_caveat_without_a_label_is_worded_by_its_dcid_with_a_gap() -> None:
    caveat = CaveatFact(kind="coverage_gap", entities=["country/SSD", "country/XYZ"])
    record = _quotient_record().model_copy(update={"caveats": [caveat]})

    served = _describe_version(record)

    assert served["caveats"][0]["text"] == "South Sudan and country/XYZ have no data."
    assert "caveats.0.entities.1.label" in served["gaps"]


def test_card_caveat_is_worded_over_the_series_it_draws() -> None:
    """A card keeps a caveat's drawn series and words only those."""
    caveat = CaveatFact(
        kind="provisional", series=[_BASIC_HEALTH[0], _HEALTH_POLICY[0]], periods=["2024"]
    )
    record = _dataset_record(variables=_variables(_WORKED_DIMENSIONS), caveats=[caveat])
    selection = _selection(_drawn("s_basic_health", {"stat_var": _BASIC_HEALTH[0]}))

    (served,) = _describe_card(record, selection)["caveats"]

    assert served["text"] == "Values of basic health care in 2024 are provisional."


_GAP_PLACES = ["country/ZWE", "country/ECU"]
_HELD_RECIPIENTS = ["country/FRA", "country/KEN", "country/NGA", "country/ETH"]


@pytest.mark.parametrize(
    ("caveat", "recipients", "served"),
    [
        pytest.param(
            CaveatFact(kind="coverage_gap", entities=_GAP_PLACES),
            ["country/FRA"],
            [],
            id="a-gap-on-places-the-card-does-not-draw-is-left-out",
        ),
        pytest.param(
            CaveatFact(kind="coverage_gap", entities=_GAP_PLACES),
            ["country/ZWE"],
            [(["country/ZWE"], [], None)],
            id="a-gap-narrows-to-the-drawn-place",
        ),
        pytest.param(
            CaveatFact(
                kind="coverage_gap", series=[_BASIC_HEALTH[0]], entities=_GAP_PLACES, count=60
            ),
            ["country/ZWE"],
            [(["country/ZWE"], [_BASIC_HEALTH[0]], None)],
            id="a-withholding-loses-its-version-wide-count",
        ),
        pytest.param(
            CaveatFact(kind="coverage_gap", series=[_HEALTH_POLICY[0]], entities=_GAP_PLACES),
            ["country/ZWE"],
            [],
            id="a-gap-on-a-variable-the-card-does-not-draw-is-left-out",
        ),
        pytest.param(
            CaveatFact(kind="coverage_gap", entities=_GAP_PLACES, count=5),
            _HELD_RECIPIENTS,
            [(_GAP_PLACES, [], 5)],
            id="a-card-drawing-every-recipient-keeps-the-gap-whole",
        ),
        pytest.param(
            CaveatFact(
                kind="coverage_gap", series=[_BASIC_HEALTH[0]], entities=_GAP_PLACES, count=60
            ),
            _HELD_RECIPIENTS,
            [(_GAP_PLACES, [_BASIC_HEALTH[0]], 60)],
            id="a-whole-gap-on-a-drawn-variable-keeps-its-series-and-count",
        ),
        pytest.param(
            CaveatFact(kind="coverage_gap", series=[_HEALTH_POLICY[0]], entities=_GAP_PLACES),
            _HELD_RECIPIENTS,
            [],
            id="a-whole-gap-on-a-variable-the-card-does-not-draw-is-left-out",
        ),
    ],
)
def test_card_caveat_names_only_the_places_it_draws(
    caveat: CaveatFact,
    recipients: list[str],
    served: list[tuple[list[str], list[str], int | None]],
) -> None:
    """A version holding four recipients has a gap on Zimbabwe and Ecuador, which hold no
    data. A card that draws only some of the recipients keeps the places it draws. A card
    that draws all four keeps the gap's places and count whole."""
    record = _dataset_record(
        variables=_variables(_WORKED_DIMENSIONS),
        caveats=[caveat],
        series=_series(entities_by_role=[RoleCount(role="recipient", entities=4)]),
    )
    selection = _selection(
        *(
            _drawn(f"s_{n}", {"stat_var": _BASIC_HEALTH[0], "recipient": recipient})
            for n, recipient in enumerate(recipients)
        )
    )

    caveats = _describe_card(record, selection)["caveats"]

    assert [
        (
            [entity["ref"] for entity in item.get("entities", [])],
            [series["ref"] for series in item.get("series", [])],
            item.get("count"),
        )
        for item in caveats
    ] == served


def test_source_pipeline_steps_serve_as_the_kit_vector_does() -> None:
    """The record behind the kit's `description-source-pipeline-steps` vector serves that
    vector's steps, gaps and freshness, member for member."""
    vector = _kit_vector("description-source-pipeline-steps")
    pipeline = [
        PipelineStep(title=step["params"]["title"], description=step["params"]["description"])
        for step in vector["steps"]
    ]
    (source,) = _evidence().sources
    evidence = _evidence(
        sources=[
            source.model_copy(
                update={"downloaded": datetime.date(2026, 9, 26), "pipeline_steps": pipeline}
            )
        ]
    )
    record = _dataset_record(variables=_variables(_WORKED_DIMENSIONS), evidence=evidence)

    served = _describe_version(record)

    assert served["steps"] == vector["steps"]
    assert served["gaps"] == vector["gaps"]
    assert served["freshness"] == vector["freshness"]


def test_importing_the_composer_loads_no_reader_or_extra():
    script = (
        "import sys\n"
        "import description.compose\n"
        "leaked = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m.split('.')[0] in {'materialise', 'dc_graph', 'httpx', 'search'}\n"
        ")\n"
        "assert not leaked, f'description.compose import pulled in: {leaked}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("kind", ["dataset", "computed"])
def test_an_unknown_series_count_serves_null_with_its_gap(kind: str) -> None:
    """A backfilled computed version whose frame no tier holds has no series count; the
    description serves `about.series` as `null` with its gap and nothing else changes."""
    record = (
        _dataset_record(variables=_variables(_WORKED_DIMENSIONS))
        if kind == "dataset"
        else _quotient_record()
    )
    known = _describe_version(record)
    unknown = record.model_copy(
        update={"series": record.series.model_copy(update={"series": None})}
    )

    served = _describe_version(unknown)

    assert served["about"]["series"] is None
    assert sorted(served.pop("gaps")) == sorted([*known.pop("gaps"), "about.series"])
    served["about"]["series"] = known["about"]["series"]
    assert served == known


def test_an_open_dimension_with_unknown_values_serves_a_null_count_with_its_gap() -> None:
    """A backfilled version whose series statistics are not held records its open role with
    unknown values; the description serves its count as `null` with a gap and no values."""
    record = _dataset_record(variables=_variables(_WORKED_DIMENSIONS))
    known = _describe_version(record)
    assert record.variables is not None and record.variables.dimensions is not None
    unread = record.variables.dimensions[0].model_copy(update={"values": None})
    unknown = record.model_copy(
        update={
            "variables": record.variables.model_copy(
                update={"dimensions": [unread, *record.variables.dimensions[1:]]}
            )
        }
    )

    served = _describe_version(unknown)

    assert served["about"]["varies"][0] == {
        "property": known["about"]["varies"][0]["property"],
        "count": None,
    }
    assert sorted(served.pop("gaps")) == sorted([*known.pop("gaps"), "about.varies.0.count"])
    served["about"]["varies"][0] = known["about"]["varies"][0]
    assert served == known


def _words_labelling(labels: dict[str, str]) -> Vocabulary:
    """Today's words, with `labels` read from the graph beside them."""
    words = _words()
    return build_vocabulary(
        index=_INDEX,
        measure_index=_MEASURES,
        labels={**words.labels, **labels},
        sources=words.sources,
        currency_units=words.currency_units,
        graph_generation="gen-1",
        catalogue_build="ocb_20260927",
    )


def test_a_graph_label_longer_than_a_served_label_is_clipped() -> None:
    """A label read from today's graph is served at most 512 characters long, ending in "…",
    as the kit's schema bounds it."""
    caveat = CaveatFact(kind="coverage_gap", entities=["country/NGA"])
    record = _quotient_record().model_copy(update={"caveats": [caveat]})

    served = _describe_version(record, _words_labelling({"country/NGA": "N" * 600}))

    (entity,) = served["caveats"][0]["entities"]
    assert entity["label"] == "N" * (TEXT_MAX_LENGTH - 1) + "…"


_WDI = "provenance/WorldDevelopmentIndicators"
_CONVERSION = Step(
    id="m1",
    op="convert",
    policy="convert_by_rate_and_deflator",
    converted_from=SimpleUnit(ref="CurrencyEUR"),
    unit=SimpleUnit(currency="USD", prices="constant", base_year=2023),
)

_COLLAPSE = Step(
    id="m1",
    op="sum_variables",
    policy="sum_family_members",
    summed=Summed(count=2, variables=[StepRef(ref="dc/var/A"), StepRef(ref="dc/var/B")]),
)


def _crediting(*refs: str) -> Evidence:
    return _evidence(
        sources=[
            SourceEvidence(ref=ref, name=None, licence=None, downloaded=None, pipeline_steps=None)
            for ref in refs
        ]
    )


@pytest.mark.parametrize(
    ("record", "text"),
    [
        pytest.param(
            _dataset_record(variables=_variables(_WORKED_DIMENSIONS)),
            "OECD Creditor Reporting System (CRS) (2026), via the Development Finance Observatory",
            id="one-source",
        ),
        pytest.param(
            _dataset_record(
                variables=_variables(_WORKED_DIMENSIONS),
                evidence=_crediting("provenance/OECDCreditorReportingSystem", _WDI),
            ),
            "OECD Creditor Reporting System (CRS) and World Development Indicators (2026), via"
            " the Development Finance Observatory",
            id="two-sources",
        ),
        pytest.param(
            _dataset_record(variables=_variables(_WORKED_DIMENSIONS), steps=[_CONVERSION]),
            "OECD Creditor Reporting System (CRS) (2026), processed by the Development Finance"
            " Observatory",
            id="converted",
        ),
        pytest.param(
            _dataset_record(variables=_variables(_WORKED_DIMENSIONS), steps=[_COLLAPSE]),
            "OECD Creditor Reporting System (CRS) (2026), processed by the Development Finance"
            " Observatory",
            id="collapsed",
        ),
        pytest.param(
            _dataset_record(variables=_variables(_WORKED_DIMENSIONS), evidence=_crediting()),
            "Development Finance Observatory (2026)",
            id="no-source",
        ),
        pytest.param(
            _quotient_record(),
            "Gross ODA disbursements and Gross national income (2026), processed by the"
            " Development Finance Observatory",
            id="computed",
        ),
    ],
)
def test_a_citation_line_credits_the_publishers_then_the_dfo(
    record: DescriptionRecord, text: str
) -> None:
    """The line credits the organisations that first published the data and the year, then
    names the DFO as the route: "via" where it passes their figures through, "processed by"
    where it converted, summed or computed them. A computed version credits its inputs."""
    served = _describe_version(record, _words_labelling({_WDI: "World Development Indicators"}))

    assert served["cite"]["text"] == text


@pytest.mark.parametrize(
    ("steps", "route"),
    [
        pytest.param([], "via", id="drawn-as-published"),
        pytest.param(
            [change_step(derivation="difference", basis=["2010", "2015"])],
            "processed by",
            id="change",
        ),
    ],
)
def test_a_card_cites_its_own_steps(steps: list[Step], route: str) -> None:
    """A card over a version whose figures are as published cites the DFO as the route, and
    one whose change computes a difference cites it as having processed them."""
    record = _dataset_record(variables=_variables(_WORKED_DIMENSIONS))

    card = _describe_card(
        record, _selection(_drawn("s_basic_health", {"stat_var": _BASIC_HEALTH[0]})), steps=steps
    )

    assert card["cite"]["text"] == (
        f"OECD Creditor Reporting System (CRS) (2026), {route} the Development Finance Observatory"
    )


def test_a_citation_line_past_its_bound_is_clipped() -> None:
    """Five sources whose names are each a full served label make a citation line past the
    2,048 characters the kit's schema bounds `cite.text` to, so the line is clipped."""
    refs = [f"provenance/Source{n}" for n in range(5)]
    sources = [
        SourceEvidence(ref=ref, name=None, licence=None, downloaded=None, pipeline_steps=None)
        for ref in refs
    ]
    record = _dataset_record(
        variables=_variables(_WORKED_DIMENSIONS), evidence=_evidence(sources=sources)
    )

    served = _describe_version(record, _words_labelling({ref: ref[-1] * 600 for ref in refs}))

    text = served["cite"]["text"]
    assert len(text) == 2048
    assert text.startswith("0" * (TEXT_MAX_LENGTH - 1) + "…, ")
    assert text.endswith("…")


def test_a_source_card_names_its_sources_clipped_as_the_kit_vector_does() -> None:
    """The twenty sources of the kit's `description-source-card-twenty-sources` vector join
    to a name past the 512 characters its schema bounds `title`, `about.name` and `cite.title`
    to, so the card serves the joined names clipped, each name whole in its `sources`."""
    from description.compose import describe_source_card

    vector = _kit_vector("description-source-card-twenty-sources")
    labels = {source["ref"]: source["label"] for source in vector["sources"]}
    sources = [
        SourceEvidence(ref=ref, name=label, licence=None, downloaded=None, pipeline_steps=None)
        for ref, label in labels.items()
    ]
    record = _dataset_record(
        variables=_variables(_WORKED_DIMENSIONS), evidence=_evidence(title=None, sources=sources)
    )

    served = _checked(
        describe_source_card(
            record,
            _words_labelling(labels),
            sources=list(labels),
            component_id=vector["describes"]["component_id"],
            revision=vector["describes"]["revision"],
        )
    )

    assert served["title"] == vector["title"]
    assert served["about"]["name"] == vector["about"]["name"]
    assert served["cite"]["title"] == vector["cite"]["title"]
    assert [source["label"] for source in served["sources"]] == list(labels.values())


_GAP_PLACES = [f"country/G{n:02d}" for n in range(60)]


@pytest.mark.parametrize(
    ("path", "named", "count"),
    [
        pytest.param("version", _GAP_PLACES[:50], 60, id="version"),
        pytest.param("card", [_GAP_PLACES[-1]], None, id="card"),
    ],
)
def test_a_coverage_gap_naming_more_than_fifty_places_serves_fifty_and_counts_all(
    path: str, named: list[str], count: int | None
) -> None:
    """A record keeps every place a coverage gap left without data; the description names 50
    and counts the 60. A card names only the place it draws, though the record lists it last,
    and carries no count, which counts across the whole version."""
    caveat = CaveatFact(kind="coverage_gap", entities=_GAP_PLACES)
    record = _dataset_record(variables=_variables(_WORKED_DIMENSIONS), caveats=[caveat])

    if path == "version":
        (served,) = _describe_version(record)["caveats"]
    else:
        selection = _selection(
            _drawn(_BASIC_HEALTH[0], {_SECTOR: _BASIC_HEALTH[0], "recipient": _GAP_PLACES[-1]})
        )
        (served,) = _describe_card(record, selection)["caveats"]

    assert [entity["ref"] for entity in served["entities"]] == named
    assert served.get("count") == count
