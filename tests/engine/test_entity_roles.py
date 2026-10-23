"""All distinct mentions of one place retain their flow axes in the door map (#918)."""

from __future__ import annotations

import asyncio

import pytest

from search.engine.entry.dataset_entries import interpret_parts as dataset_door_interpretation
from search.engine.entry.parsed import (
    _entity_roles_for_part,
    _role_query_for_part,
    interpret_parts,
)
from search.engine.entry.raw_text import _entity_roles
from search.engine.extract import ExtractedEntity, _validate_role_spans
from search.engine.quantity_read import QuantityOutput
from search.engine.region.pipeline import resolve_variable
from search.engine.role_text import DoorRole, contradicts
from search.models import ParsedEntity, ParsedPart, RoleAxis
from series_coverage.schema import RoleNames
from series_coverage.shared import CoverageSnapshot
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import (
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.engine.test_role_bearing_pipeline_e2e import _SV_DCID, _declaring_arcs
from tests.fixtures import FakeGraph, make_resolver

_CREDITOR_DEBTOR_ROLE_NAMES = RoleNames(donor="creditor", recipient="debtor")

_KEN = "country/KEN"
_CHN = "country/CHN"
_SELF_FLOW_QUERY = "aid from Kenya to Kenya"


def _extracted(text: str, flow_role: RoleAxis, span: str | None = None) -> ExtractedEntity:
    """An extracted place whose cue is the preposition its span opens with, if any."""
    cue = span.split()[0] if span and span.split()[0] in {"from", "to"} else None
    return ExtractedEntity(text=text, span=span, flow_role=flow_role, role_span=cue)


def _occurrences(roles: dict[str, list[DoorRole]]) -> dict[str, list[tuple[RoleAxis, str]]]:
    """Each name's occurrences as ``(axis, span)``; `stated` has its own tests."""
    return {name: [(role.axis, role.span) for role in rs] for name, rs in roles.items()}


class TestRawTextDoorCarriesEveryOccurrence:
    def test_one_name_on_both_axes_carries_both_in_mention_order(self):
        roles = _entity_roles(
            [
                _extracted("Kenya", "origin", span="from Kenya"),
                _extracted("Kenya", "destination", span="to Kenya"),
            ],
            query=_SELF_FLOW_QUERY,
            measures=["aid"],
        )
        assert _occurrences(roles) == {
            "Kenya": [
                ("origin", "from Kenya"),
                ("destination", "to Kenya"),
            ]
        }

    @pytest.mark.parametrize(
        ("query", "entities", "stated"),
        [
            # A cue before the span, or a verb after it, states the direction.
            (
                "aid to Sub-Saharan Africa",
                [("Sub-Saharan Africa", "destination", "to")],
                {"Sub-Saharan Africa": True},
            ),
            (
                "net ODA from the United States to Pakistan in 2020",
                [("United States", "origin", "from")],
                {"United States": True},
            ),
            ("aid funded by Germany", [("Germany", "origin", "funded by")], {"Germany": True}),
            (
                "Norway funds Kenya",
                [("Norway", "origin", "funds"), ("Kenya", "destination", "funds")],
                {"Norway": True, "Kenya": True},
            ),
            (
                "How much does Norway give to the UN?",
                [("Norway", "origin", "give")],
                {"Norway": True},
            ),
            # A coordinated list shares the cue before it.
            (
                "aid to Kenya from Norway and Sweden",
                [
                    ("Kenya", "destination", "to"),
                    ("Norway", "origin", "from"),
                    ("Sweden", "origin", "from"),
                ],
                {"Kenya": True, "Norway": True, "Sweden": True},
            ),
            # ids-024: the "to" after Ghana directs China.
            (
                "future debt service of Ghana to China",
                [("Ghana", "destination", "to"), ("China", "origin", "to")],
                {"Ghana": False, "China": True},
            ),
            # ids-007: a cue that belongs to another phrase directs nothing.
            (
                "Zambia debt owed to bondholders",
                [("Zambia", "destination", "to")],
                {"Zambia": False},
            ),
            # ids-027 "of", weo-037 a possessive: the direction comes from the measure.
            ("top 10 debtors of China", [("China", "origin", "of")], {"China": False}),
            ("is Ghana's economy growing", [("Ghana", "destination", "'s")], {"Ghana": False}),
            (
                "is Ghana\u2019s economy growing",
                [("Ghana", "destination", "\u2019s")],
                {"Ghana": False},
            ),
        ],
    )
    def test_a_direction_is_stated_when_the_validated_cue_attaches_to_the_place(
        self, query: str, entities: list[tuple[str, RoleAxis, str]], stated: dict[str, bool]
    ):
        """Live extractions for each query; every cue passes role-span validation."""
        extracted = [
            ExtractedEntity(text=text, span=text, flow_role=axis, role_span=cue)
            for text, axis, cue in entities
        ]
        assert _validate_role_spans(query, extracted, ["aid"]) == extracted
        roles = _entity_roles(extracted, query=query, measures=["aid"])
        assert {name: [role.stated for role in rs] for name, rs in roles.items()} == {
            name: [value] for name, value in stated.items()
        }

    def test_a_national_adjective_states_no_direction(self):
        """The cue is the entity's own span ("French aid")."""
        entity = ExtractedEntity(
            text="France", span="French", flow_role="origin", role_span="French"
        )
        roles = _entity_roles([entity], query="French aid to Senegal", measures=["aid"])
        assert [role.stated for role in roles["France"]] == [False]

    def test_a_debt_axis_is_stated_by_construction(self):
        """The possessive debtor in "Kenya's debt to China" takes its axis from the debt
        wording, so it is stated although its cue is a possessive."""
        roles = _entity_roles(
            [
                ExtractedEntity(text="Kenya", span="Kenya", flow_role="origin", role_span="'s"),
                ExtractedEntity(
                    text="China", span="China", flow_role="destination", role_span="to"
                ),
            ],
            query="Kenya's debt to China",
            measures=["debt"],
        )
        assert roles == {
            "Kenya": [DoorRole(axis="destination", span="Kenya", stated=True)],
            "China": [DoorRole(axis="origin", span="China", stated=True)],
        }

    @pytest.mark.parametrize(
        ("query", "measure", "entities", "expected"),
        [
            # Stored readings whose cues are paraphrases ("owes", "to") or not cues at all
            # ("does"): validation drops each role, and the query's own wording sets it.
            (
                "how much does China owe India",
                "external debt",
                [("China", "destination", "owes"), ("India", "origin", "to")],
                {"China": "destination", "India": "origin"},
            ),
            (
                "what does Zambia owe the IMF",
                "IMF debt",
                [("Zambia", "destination", "owes"), ("IMF", "origin", "owes")],
                {"Zambia": "destination", "IMF": "origin"},
            ),
            (
                "how much does Kenya owe China",
                "external debt",
                [("Kenya", "unspecified", None), ("China", "origin", "does")],
                {"Kenya": "destination", "China": "origin"},
            ),
            (
                "net ODA received by Ethiopia in 1965",
                "net ODA received",
                [("Ethiopia", "destination", "to")],
                {"Ethiopia": "destination"},
            ),
            (
                "By what percentage did the total ODA Ghana received from all donors change "
                "between 2012 and 2021?",
                "total ODA",
                [("Ghana", "destination", "to")],
                {"Ghana": "destination"},
            ),
        ],
    )
    def test_query_wording_directs_a_place_whose_cue_failed_validation(
        self,
        query: str,
        measure: str,
        entities: list[tuple[str, RoleAxis, str | None]],
        expected: dict[str, RoleAxis],
    ):
        extracted = [
            ExtractedEntity(text=text, span=text, flow_role=axis, role_span=cue)
            for text, axis, cue in entities
        ]
        validated = _validate_role_spans(query, extracted, [measure])
        assert {e.flow_role for e in validated} == {"unspecified"}
        roles = _entity_roles(validated, query=query, measures=[measure])
        assert roles == {
            name: [DoorRole(axis=axis, span=name, stated=True)] for name, axis in expected.items()
        }

    def test_an_exact_repeat_collapses_to_one_occurrence(self):
        roles = _entity_roles(
            [
                _extracted("Kenya", "destination", span="to Kenya"),
                _extracted("Kenya", "destination", span="to Kenya"),
            ],
            query=_SELF_FLOW_QUERY,
            measures=["aid"],
        )
        assert _occurrences(roles) == {"Kenya": [("destination", "to Kenya")]}

    def test_an_unspecified_occurrence_is_skipped(self):
        roles = _entity_roles(
            [
                _extracted("Kenya", "unspecified"),
                _extracted("Kenya", "origin", span="from Kenya"),
            ],
            query=_SELF_FLOW_QUERY,
            measures=["aid"],
        )
        assert _occurrences(roles) == {"Kenya": [("origin", "from Kenya")]}

    def test_a_name_unspecified_in_every_occurrence_carries_no_entry(self):
        roles = _entity_roles(
            [
                _extracted("Kenya", "unspecified"),
                _extracted("Kenya", "unspecified"),
                _extracted("Uganda", "origin", span="from Uganda"),
            ],
            query=_SELF_FLOW_QUERY,
            measures=["aid"],
        )
        assert _occurrences(roles) == {"Uganda": [("origin", "from Uganda")]}


def _debt_roles(
    query: str, extracted: list[tuple[str, str | None, str]], measures: list[str]
) -> dict[str, list[RoleAxis]]:
    """The raw-text door's axes by name for ``(text, span, extracted axis)`` places."""
    roles = _entity_roles(
        [ExtractedEntity(text=text, span=span, flow_role=axis) for text, span, axis in extracted],
        query=query,
        measures=measures,
    )
    return {name: [role.axis for role in occurrences] for name, occurrences in roles.items()}


_AXIS_COMBINATIONS = [
    ("destination", "origin"),
    ("origin", "destination"),
    ("unspecified", "unspecified"),
    ("origin", "unspecified"),
    ("unspecified", "origin"),
    ("destination", "destination"),
]


def _extracted_axes(kenya: str, china: str) -> dict[str, list[str]]:
    return {
        name: [axis] for name, axis in (("Kenya", kenya), ("China", china)) if axis != "unspecified"
    }


class TestRawTextDoorReadsDebtWording:
    """In a two-party debt relation the wording sets both axes, whatever extraction says."""

    @pytest.mark.parametrize(
        "query",
        [
            "Kenya's debt to China",
            "Kenya's debt owed to China",
            "debt owed to China, Kenya's debt",
            "debt owed to China by Kenya",
            "debt owed by Kenya to China",
            "Kenya's debt service payments to China",
        ],
    )
    @pytest.mark.parametrize(("kenya", "china"), _AXIS_COMBINATIONS)
    def test_cues_on_both_places_set_the_creditor_and_the_debtor(
        self, query: str, kenya: str, china: str
    ):
        extracted = [("Kenya", "Kenya", kenya), ("China", "China", china)]
        assert _debt_roles(query, extracted, ["debt"]) == {
            "Kenya": ["destination"],
            "China": ["origin"],
        }

    @pytest.mark.parametrize(
        ("query", "uncued"),
        [
            ("Kenya debt owed to China", "Kenya"),
            ("Kenya debt to China", "Kenya"),
        ],
    )
    @pytest.mark.parametrize(("kenya", "china"), _AXIS_COMBINATIONS)
    def test_an_uncued_place_needs_an_extracted_axis(
        self, query: str, uncued: str, kenya: str, china: str
    ):
        """The uncued place is the debtor only when extraction put it on destination,
        opposite the cued creditor; on origin or on no axis it is no party, and both places
        keep their extracted axes."""
        extracted = [("Kenya", "Kenya", kenya), ("China", "China", china)]
        uncued_axis = kenya if uncued == "Kenya" else china
        expected = (
            {"Kenya": ["destination"], "China": ["origin"]}
            if uncued_axis == "destination"
            else _extracted_axes(kenya, china)
        )
        assert _debt_roles(query, extracted, ["debt"]) == expected

    def test_a_national_adjective_takes_the_party_the_creditor_leaves(self):
        extracted = [("Kenya", "Kenyan", "destination"), ("China", "China", "destination")]
        assert _debt_roles("Kenyan debt to China", extracted, ["debt"]) == {
            "Kenya": ["destination"],
            "China": ["origin"],
        }

    def test_a_national_adjective_on_the_creditors_axis_keeps_the_extracted_axes(self):
        """Extraction puts Kenya on origin, the axis China's cue claims, so Kenya is a
        second creditor and nothing is set."""
        extracted = [("Kenya", "Kenyan", "origin"), ("China", "China", "destination")]
        assert _debt_roles("Kenyan debt to China", extracted, ["debt"]) == {
            "Kenya": ["origin"],
            "China": ["destination"],
        }

    def test_a_translated_name_without_a_span_matches_by_its_text(self):
        extracted = [("Kenya", "Kenya", "unspecified"), ("China", None, "destination")]
        assert _debt_roles("Kenya's debt to China", extracted, ["debt"]) == {
            "Kenya": ["destination"],
            "China": ["origin"],
        }

    @pytest.mark.parametrize(
        ("query", "extracted"),
        [
            (
                "Kenya's external debt compared with China",
                [("Kenya", "Kenya", "unspecified"), ("China", "China", "unspecified")],
            ),
            (
                "China and Kenya's external debt",
                [("China", "China", "unspecified"), ("Kenya", "Kenya", "unspecified")],
            ),
            (
                "Kenya and Ghana's external debt",
                [("Kenya", "Kenya", "unspecified"), ("Ghana", "Ghana", "destination")],
            ),
            (
                "Kenya's external debt compared with Uganda",
                [("Kenya", "Kenya", "destination"), ("Uganda", "Uganda", "unspecified")],
            ),
            (
                "Kenya's external debt vs Uganda",
                [("Kenya", "Kenya", "origin"), ("Uganda", "Uganda", "unspecified")],
            ),
            (
                "Kenya's debt versus Ghana's",
                [("Kenya", "Kenya", "destination"), ("Ghana", "Ghana", "destination")],
            ),
            (
                "Kenya's share of Africa's debt",
                [("Kenya", "Kenya", "origin"), ("Africa", "Africa", "destination")],
            ),
            (
                "Kenya's external debt in East Africa",
                [("Kenya", "Kenya", "origin"), ("East Africa", "East Africa", "unspecified")],
            ),
            (
                "Kenya's debt relative to Sub-Saharan Africa",
                [
                    ("Kenya", "Kenya", "destination"),
                    ("Sub-Saharan Africa", "Sub-Saharan Africa", "unspecified"),
                ],
            ),
            (
                "Nairobi's debt in Kenya",
                [("Nairobi", "Nairobi", "origin"), ("Kenya", "Kenya", "unspecified")],
            ),
            # The second place on the debtor's own axis is a second debtor, as live
            # extraction gives "Kenya's debt relative to Sub-Saharan Africa".
            (
                "Kenya's debt relative to Sub-Saharan Africa",
                [
                    ("Kenya", "Kenya", "unspecified"),
                    ("Sub-Saharan Africa", "Sub-Saharan Africa", "destination"),
                ],
            ),
            (
                "Kenya's external debt compared with Uganda",
                [("Kenya", "Kenya", "unspecified"), ("Uganda", "Uganda", "destination")],
            ),
            (
                "Kenya's external debt vs Uganda",
                [("Kenya", "Kenya", "destination"), ("Uganda", "Uganda", "destination")],
            ),
            (
                "Kenya's external debt in East Africa",
                [("Kenya", "Kenya", "unspecified"), ("East Africa", "East Africa", "destination")],
            ),
            (
                "Nairobi's debt in Kenya",
                [("Nairobi", "Nairobi", "unspecified"), ("Kenya", "Kenya", "destination")],
            ),
        ],
    )
    def test_a_comparison_or_a_locative_keeps_the_extracted_axes(
        self, query: str, extracted: list[tuple[str, str, str]]
    ):
        expected = {name: [axis] for name, _span, axis in extracted if axis != "unspecified"}
        assert _debt_roles(query, extracted, ["external debt"]) == expected

    @pytest.mark.parametrize(
        ("query", "extracted", "measures", "expected"),
        [
            # A holder of someone else's debt is no debtor.
            (
                "China's holdings of Kenyan debt",
                [("China", "China", "origin"), ("Kenya", "Kenyan", "unspecified")],
                ["debt holdings"],
                {"China": ["origin"]},
            ),
            (
                "China's share of Zambia debt",
                [("China", "China", "unspecified"), ("Zambia", "Zambia", "unspecified")],
                ["debt"],
                {},
            ),
            (
                "China's debt claims on Africa",
                [("China", "China", "origin"), ("Africa", "Africa", "destination")],
                ["debt claims"],
                {"China": ["origin"], "Africa": ["destination"]},
            ),
            # A second measure leaves every leg on its extracted axes.
            (
                "Japan's ODA and external debt",
                [("Japan", "Japan", "origin")],
                ["ODA", "external debt"],
                {"Japan": ["origin"]},
            ),
            (
                "Japan's ODA disbursements and debt to Kenya",
                [("Japan", "Japan", "origin"), ("Kenya", "Kenya", "destination")],
                ["ODA disbursements", "debt"],
                {"Japan": ["origin"], "Kenya": ["destination"]},
            ),
            (
                "Kenya's GDP and debt",
                [("Kenya", "Kenya", "unspecified")],
                ["GDP", "debt"],
                {},
            ),
            # One place is no two-party relation.
            (
                "Kenya's debt to GDP ratio",
                [("Kenya", "Kenya", "origin")],
                ["debt"],
                {"Kenya": ["origin"]},
            ),
            # No debt measure.
            ("refugees owing to Syria", [("Syria", "Syria", "unspecified")], ["refugees"], {}),
            (
                "aid owed to Kenya",
                [("Kenya", "Kenya", "destination")],
                ["aid"],
                {"Kenya": ["destination"]},
            ),
            (
                "aid owed to Kenya by Norway",
                [("Kenya", "Kenya", "destination"), ("Norway", "Norway", "origin")],
                ["aid"],
                {"Kenya": ["destination"], "Norway": ["origin"]},
            ),
            (
                "Norway's debt relief to Kenya",
                [("Norway", "Norway", "origin"), ("Kenya", "Kenya", "destination")],
                ["debt relief"],
                {"Norway": ["origin"], "Kenya": ["destination"]},
            ),
            (
                "aid to Kenya from Norway",
                [("Norway", "Norway", "origin"), ("Kenya", "Kenya", "destination")],
                ["aid"],
                {"Norway": ["origin"], "Kenya": ["destination"]},
            ),
            (
                "ODA Kenya Norway",
                [("Kenya", "Kenya", "unspecified"), ("Norway", "Norway", "unspecified")],
                ["ODA"],
                {},
            ),
            # Three places.
            (
                "Kenya's debt to China and India",
                [
                    ("Kenya", "Kenya", "origin"),
                    ("China", "China", "destination"),
                    ("India", "India", "destination"),
                ],
                ["debt"],
                {"Kenya": ["origin"], "China": ["destination"], "India": ["destination"]},
            ),
        ],
    )
    def test_other_queries_keep_the_extracted_axes(
        self,
        query: str,
        extracted: list[tuple[str, str | None, str]],
        measures: list[str],
        expected: dict[str, list[str]],
    ):
        assert _debt_roles(query, extracted, measures) == expected


class TestParsedDoorCarriesEveryOccurrence:
    def test_one_name_on_both_axes_carries_both_in_mention_order(self):
        roles = _entity_roles_for_part(
            ParsedPart(
                variable_text="aid",
                entities=[
                    ParsedEntity(text="Kenya", role_text="from"),
                    ParsedEntity(text="Kenya", role_text="to"),
                ],
            )
        )
        assert roles == {
            "Kenya": [
                DoorRole(axis="origin", span="Kenya"),
                DoorRole(axis="destination", span="Kenya"),
            ]
        }

    def test_an_exact_repeat_collapses_to_one_occurrence(self):
        roles = _entity_roles_for_part(
            ParsedPart(
                variable_text="aid",
                entities=[
                    ParsedEntity(text="Kenya", role_text="to"),
                    ParsedEntity(text="Kenya", role_text="to"),
                ],
            )
        )
        assert roles == {"Kenya": [DoorRole(axis="destination", span="Kenya")]}

    def test_an_unrecognised_or_absent_role_text_carries_no_occurrence(self):
        roles = _entity_roles_for_part(
            ParsedPart(
                variable_text="aid",
                entities=[
                    ParsedEntity(text="Kenya", role_text="sideways"),
                    ParsedEntity(text="Kenya"),
                    ParsedEntity(text="Kenya", role_text="from"),
                ],
            )
        )
        assert roles == {"Kenya": [DoorRole(axis="origin", span="Kenya")]}


class TestDatasetEntriesDoorCarriesEveryOccurrence:
    def test_the_resubmit_door_reads_the_same_channel(self):
        """The dataset door uses the parsed door's interpretation."""
        assert dataset_door_interpretation is interpret_parts
        part = ParsedPart(
            variable_text="aid",
            entities=[
                ParsedEntity(text="Kenya", role_text="from"),
                ParsedEntity(text="Kenya", role_text="recipient"),
            ],
        )
        interpretation = dataset_door_interpretation(
            [part], pointer_prefixes=["/input/entries/0/part"]
        )
        (roles,) = interpretation.entity_roles
        assert roles == {
            "Kenya": [
                DoorRole(axis="origin", span="Kenya"),
                DoorRole(axis="destination", span="Kenya"),
            ]
        }


class _FixedBindLLM:
    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, bindings=[]), None


def _run_self_flow_occurrences(
    entities: list[str],
    sdmx: FakeSdmxClient,
    role_query: str,
    coverage: CoverageSnapshot | None = None,
):
    """Runs a role-bearing query with both Kenya mentions in the raw-text door map."""
    graph = FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            _SV_DCID: {
                "label": "Test flow",
                "arcs": _declaring_arcs(observation_properties=["creditor", "debtor"]),
            },
        },
        detect={role_query: {"svs": [_SV_DCID], "cosine_scores": [0.9]}},
        resolve={"Kenya": _KEN},
        obs={},
    )
    door_roles = _entity_roles(
        [
            _extracted("Kenya", "origin", span="from Kenya"),
            _extracted("Kenya", "destination", span="to Kenya"),
        ],
        query=_SELF_FLOW_QUERY,
        measures=["aid"],
    )
    return asyncio.run(
        resolve_variable(
            "aid",
            entities=entities,
            date_request=None,
            detect_query=role_query,
            role_query=role_query,
            entity_roles=door_roles,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=sdmx,
            base_timing={},
            coverage=coverage if coverage is not None else empty_coverage_snapshot(),
        )
    )


def test_one_name_on_both_axes_binds_the_dcid_on_both_axes(monkeypatch):
    """Kenya binds both roles and confirms from one self-flow coverage row."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _KEN, _KEN, [2018, 2019, 2020])],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    sdmx = FakeSdmxClient(observations={})  # Coverage confirms this case before SDMX reads.
    result = _run_self_flow_occurrences(
        ["Kenya"],
        sdmx,
        "aid from Kenya to Kenya",
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert result.status == "definite", result
    spec = result.specs[0]
    assert {(entity.ref, entity.role) for entity in spec.resolved.entities} == {
        (_KEN, "origin"),
        (_KEN, "destination"),
    }
    assert spec.definition.identity.entities == {"origin": [_KEN], "destination": [_KEN]}


@pytest.mark.parametrize("query", ["Kenya's debt to China", "Kenya's debt owed to China"])
def test_debt_wording_binds_the_creditor_as_origin_and_the_debtor_as_destination(
    monkeypatch, query: str
):
    """A by-creditor series answers with China on origin, though the extraction reversed them."""
    door_roles = _entity_roles(
        [
            ExtractedEntity(text="Kenya", span="Kenya", flow_role="origin", role_span="'s"),
            ExtractedEntity(text="China", span="China", flow_role="destination", role_span="to"),
        ],
        query=query,
        measures=["debt"],
    )
    _assert_binds_china_as_creditor(monkeypatch, query, door_roles)


def test_parsed_door_binds_a_declared_creditor_after_owed_to(monkeypatch):
    """A declared creditor after "owed to" in parsed variable text binds on origin, because
    that "to" closes a creditor cue."""
    part = ParsedPart(
        variable_text="external debt owed to",
        entities=[
            ParsedEntity(text="China", role_text="creditor"),
            ParsedEntity(text="Kenya", role_text="debtor"),
        ],
    )
    role_query, warnings = _role_query_for_part(part)
    assert (role_query, warnings) == ("external debt owed to China Kenya", [])
    door_roles = _entity_roles_for_part(part)
    assert door_roles == {
        "China": [DoorRole(axis="origin", span="China")],
        "Kenya": [DoorRole(axis="destination", span="Kenya")],
    }
    assert contradicts(role_query, span="China", axis="origin") is False
    _assert_binds_china_as_creditor(monkeypatch, role_query, door_roles)


def _assert_binds_china_as_creditor(
    monkeypatch, role_query: str, door_roles: dict[str, list[DoorRole]]
) -> None:
    """Bind a by-creditor series and assert China on origin and Kenya on destination."""
    index = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _CHN, _KEN, [2018, 2019, 2020])],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    graph = FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            _CHN: {"label": "China", "type": "Country"},
            _SV_DCID: {
                "label": "Test debt",
                "arcs": _declaring_arcs(observation_properties=["creditor", "debtor"]),
            },
        },
        detect={"debt": {"svs": [_SV_DCID], "cosine_scores": [0.9]}},
        resolve={"Kenya": _KEN, "China": _CHN},
        obs={},
    )
    result = asyncio.run(
        resolve_variable(
            "debt",
            entities=["Kenya", "China"],
            date_request=None,
            detect_query="debt",
            role_query=role_query,
            entity_roles=door_roles,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=FakeSdmxClient(observations={}),
            base_timing={},
            coverage=FakeCoverageService(index).snapshot(),
        )
    )
    assert result.status == "definite", result
    spec = result.specs[0]
    assert {(entity.ref, entity.role) for entity in spec.resolved.entities} == {
        (_CHN, "origin"),
        (_KEN, "destination"),
    }
    assert spec.definition.identity.entities == {"origin": [_CHN], "destination": [_KEN]}


def test_a_country_before_the_measure_takes_the_end_the_other_place_leaves(monkeypatch):
    """ "China loans to Kenya": China's own name before "loans" states no direction, so
    validation leaves it unspecified, and Kenya's cue makes Kenya the debtor. A by-creditor
    series places China on the end Kenya leaves open."""
    query = "China loans to Kenya"
    extracted = _validate_role_spans(
        query,
        [
            ExtractedEntity(text="China", span="China", flow_role="origin", role_span="China"),
            ExtractedEntity(text="Kenya", span="Kenya", flow_role="destination", role_span="to"),
        ],
        ["loans"],
    )
    assert [entity.flow_role for entity in extracted] == ["unspecified", "destination"]
    door_roles = _entity_roles(extracted, query=query, measures=["loans"])

    _assert_binds_china_as_creditor(monkeypatch, query, door_roles)
