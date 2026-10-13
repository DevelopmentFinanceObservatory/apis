"""The variable card's `definition`: constraints from the code index or the graph's arcs, words
from the indexes before the graph, and a gap for every part the builder cannot fill."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import pytest

from code_index.shared import get_index, get_measure_index
from graph_api.card_definition import CardDefinition, build_definitions
from graph_api.reader import NodeArcsByDcid
from materialise.graph import GraphReadError

from .conftest import FakeGraphNodeReader

_CRS = "Amount_FinancialFlow_Net_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_CRSPurposeCode11110_Nominal"
_CRS_ARCS = {
    "name": (
        "Official development assistance net disbursements for education policy and "
        "administrative management (standard grant)",
    ),
    "description": (),
    "populationType": ("FinancialFlow",),
    "measuredProperty": ("amount",),
    "observationProperties": ("financingProvider", "financingRecipient"),
    "constraintProperties": (
        "crsPurpose",
        "crsTypeOfFinance",
        "crsTypeOfFlow",
        "financialFlowType",
        "nettingBasis",
    ),
}
_CODE_TEXT = {
    "CRSPurposeCode11110": "Sector policy, planning and programmes; education ministries.",
    "CRSTypeOfFinance110": "Transfers in cash or in kind with no legal debt.",
    "CRSTypeOfFlow10": "Flows to developing countries that are official and concessional.",
}

_COUNT_PERSON_TEXT = "The number of people."
_COUNT_PERSON_ARCS = {
    "name": ("Total population",),
    "description": (_COUNT_PERSON_TEXT,),
    "populationType": ("Person",),
    "measuredProperty": ("count",),
    "observationProperties": (),
    "constraintProperties": (),
    "sourceIndicatorCode": ("SP.POP.TOTL",),
}
_FEMALE = "Count_Person_Female"
_FEMALE_ARCS = {
    "name": ("Female population",),
    "description": (),
    "populationType": ("Person",),
    "measuredProperty": ("count",),
    "observationProperties": (),
    "constraintProperties": ("gender",),
}


class _FailingReader(FakeGraphNodeReader):
    """Fails any read that asks for the `failing` property."""

    def __init__(self, *, failing: str, nodes: Mapping[str, Mapping[str, tuple[str, ...]]]) -> None:
        super().__init__(nodes=nodes)
        self._failing = failing

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str], deadline: float | None = None
    ) -> NodeArcsByDcid:
        if self._failing in properties:
            self.calls["node_arcs_many"] += 1
            raise GraphReadError("exemplar", upstream_status=503)
        return super().node_arcs_many(dcids, properties=properties, deadline=deadline)


def _crs_graph(texts: Mapping[str, str]) -> dict[str, dict[str, tuple[str, ...]]]:
    return (
        {dcid: {"description": (text,)} for dcid, text in texts.items()}
        | {dcid: {} for dcid in _CODE_TEXT if dcid not in texts}
        | {"measuredValue": {"name": ("measured value",)}}
    )


def _build(
    variables: Mapping[str, Mapping[str, tuple[str, ...]]], graph: FakeGraphNodeReader
) -> dict[str, CardDefinition]:
    return build_definitions(
        variables,
        graph=graph,
        code_index=get_index(),
        measure_index=get_measure_index(),
        deadline=time.monotonic() + 5,
    )


def _ref(ref: str, label: str | None) -> dict[str, str | None]:
    return {"ref": ref, "label": label}


def test_a_crs_variable_reads_as_the_design_example():
    graph = FakeGraphNodeReader(nodes=_crs_graph(_CODE_TEXT))

    built = _build({_CRS: _CRS_ARCS}, graph)[_CRS]

    assert built.definition.model_dump(mode="json") == {
        "population_type": _ref("FinancialFlow", "financial flow"),
        "measured_property": _ref("amount", "amount"),
        "stat_type": _ref("measuredValue", "measured value"),
        "measure": {"label": "net disbursements"},
        "roles": ["financingProvider", "financingRecipient"],
        "source_indicator_codes": [],
        "constraints": [
            {
                "property": _ref("crsPurpose", "sector"),
                "value": _ref(
                    "CRSPurposeCode11110", "education policy and administrative management"
                ),
            },
            {
                "property": _ref("crsTypeOfFinance", "instrument"),
                "value": _ref("CRSTypeOfFinance110", "standard grant"),
            },
            {
                "property": _ref("crsTypeOfFlow", "flow"),
                "value": _ref("CRSTypeOfFlow10", "official development assistance"),
            },
        ],
        "definitions": [
            {
                "of": _ref("CRSPurposeCode11110", "education policy and administrative management"),
                "text": _CODE_TEXT["CRSPurposeCode11110"],
            },
            {
                "of": _ref("CRSTypeOfFinance110", "standard grant"),
                "text": _CODE_TEXT["CRSTypeOfFinance110"],
            },
            {
                "of": _ref("CRSTypeOfFlow10", "official development assistance"),
                "text": _CODE_TEXT["CRSTypeOfFlow10"],
            },
        ],
    }
    assert built.gaps == ("stat_var.definition.measure.definition",)
    assert built.transient is False
    # The indexes label every part but the statType, so one read gives its name and the
    # codes' descriptions.
    assert graph.calls["node_arcs_many"] == 1


def test_a_base_variable_carries_its_own_prose_first():
    graph = FakeGraphNodeReader(nodes={"Count_Person": {}})

    built = _build({"Count_Person": _COUNT_PERSON_ARCS}, graph)["Count_Person"]

    definition = built.definition.model_dump(mode="json")
    assert definition["definitions"] == [
        {"of": _ref("Count_Person", "Total population"), "text": _COUNT_PERSON_TEXT}
    ]
    assert definition["roles"] == ["observationAbout"]
    assert definition["source_indicator_codes"] == ["SP.POP.TOTL"]
    assert definition["constraints"] == []
    assert definition["measure"] is None
    assert definition["population_type"] == _ref("Person", None)
    assert definition["stat_type"] is None
    assert set(built.gaps) == {
        "stat_var.definition.measure",
        "stat_var.definition.population_type.label",
        "stat_var.definition.measured_property.label",
        "stat_var.definition.stat_type",
    }
    assert built.transient is False


_WDI_GDP = "worldBank/NY_GDP_MKTP_CD"
_GDP_TEXT = "GDP at purchaser's prices is the sum of gross value added by all resident producers."


@pytest.mark.parametrize(
    ("dcid", "description", "text"),
    [
        pytest.param(
            _WDI_GDP,
            f"{_GDP_TEXT} (WorldBank Series Code: NY.GDP.MKTP.CD)",
            _GDP_TEXT,
            id="wdi-prose-leaves-out-its-series-code",
        ),
        pytest.param(
            _WDI_GDP, " (WorldBank Series Code: NY.GDP.MKTP.CD)", None, id="wdi-series-code-alone"
        ),
        pytest.param(
            "Count_Person",
            "The number of people. (WorldBank Series Code: SP.POP.TOTL)",
            "The number of people. (WorldBank Series Code: SP.POP.TOTL)",
            id="control-other-prose-as-the-graph-holds-it",
        ),
    ],
)
def test_a_world_bank_variables_own_prose_leaves_out_its_series_code(
    dcid: str, description: str, text: str | None
):
    graph = FakeGraphNodeReader(nodes={dcid: {}})
    arcs = {**_COUNT_PERSON_ARCS, "name": ("GDP (current US$)",), "description": (description,)}

    built = _build({dcid: arcs}, graph)[dcid]

    assert built.definition.model_dump(mode="json")["definitions"] == (
        [] if text is None else [{"of": _ref(dcid, "GDP (current US$)"), "text": text}]
    )


def test_a_base_variable_takes_its_constraints_and_their_words_from_the_graph():
    graph = FakeGraphNodeReader(
        nodes={
            _FEMALE: {"gender": ("Female",), "statType": ("measuredValue",)},
            "Person": {"name": ("person",)},
            "measuredValue": {"name": ("measured value",)},
            "gender": {"name": ("Gender",)},
            "Female": {"name": ("female",), "description": ("Of the female sex.",)},
            "count": {"name": ("Count",)},
        }
    )

    built = _build({_FEMALE: _FEMALE_ARCS}, graph)[_FEMALE]

    definition = built.definition.model_dump(mode="json")
    assert definition["population_type"] == _ref("Person", "person")
    assert definition["measured_property"] == _ref("count", "Count")
    assert definition["stat_type"] == _ref("measuredValue", "measured value")
    assert definition["constraints"] == [
        {"property": _ref("gender", "Gender"), "value": _ref("Female", "female")}
    ]
    assert definition["definitions"] == [
        {"of": _ref("Female", "female"), "text": "Of the female sex."}
    ]
    assert built.gaps == ("stat_var.definition.measure",)
    assert graph.calls["node_arcs_many"] == 2


def test_a_code_with_no_graph_description_is_a_permanent_gap():
    graph = FakeGraphNodeReader(
        nodes=_crs_graph({"CRSPurposeCode11110": _CODE_TEXT["CRSPurposeCode11110"]})
    )

    built = _build({_CRS: _CRS_ARCS}, graph)[_CRS]

    assert [part.of.ref for part in built.definition.definitions] == ["CRSPurposeCode11110"]
    assert set(built.gaps) == {
        "stat_var.definition.measure.definition",
        "stat_var.definition.constraints.crsTypeOfFinance.definition",
        "stat_var.definition.constraints.crsTypeOfFlow.definition",
    }
    assert built.transient is False


def test_a_failed_word_read_is_a_transient_gap():
    graph = _FailingReader(failing="description", nodes=_crs_graph(_CODE_TEXT))

    built = _build({_CRS: _CRS_ARCS}, graph)[_CRS]

    assert built.definition.definitions == []
    assert built.definition.constraints is not None
    assert {"stat_var.definition.constraints.crsPurpose.definition"} <= set(built.gaps)
    assert built.transient is True


@pytest.mark.parametrize("failing", ["statType", "description"])
def test_a_failed_read_leaves_a_crs_variable_beside_it_untouched_but_for_its_words(failing):
    graph = _FailingReader(
        failing=failing,
        nodes={**_crs_graph(_CODE_TEXT), _FEMALE: {"gender": ("Female",)}},
    )

    built = _build({_CRS: _CRS_ARCS, _FEMALE: _FEMALE_ARCS}, graph)

    female = built[_FEMALE]
    assert female.transient is True
    if failing == "statType":
        assert female.definition.constraints is None
        assert female.definition.stat_type is None
        assert {
            "stat_var.definition.constraints",
            "stat_var.definition.measure",
            "stat_var.definition.stat_type",
        } <= set(female.gaps)
        assert built[_CRS].transient is False
        assert built[_CRS].gaps == ("stat_var.definition.measure.definition",)
    else:
        assert built[_CRS].transient is True
