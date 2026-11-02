"""All distinct mentions of one place retain their flow axes in the door map (#918), each
part reading the roles of its own places."""

from __future__ import annotations

import asyncio

import pytest

from search.engine.entry.dataset_entries import interpret_parts as dataset_door_interpretation
from search.engine.entry.parsed import (
    _entity_roles_for_part,
    _role_query_for_part,
    interpret_parts,
)
from search.engine.entry.raw_text import _prepare_extraction
from search.engine.extract import (
    ExtractedEntity,
    ExtractedOperation,
    Extraction,
    ExtractionResult,
    _ground_roles,
)
from search.engine.quantity_read import QuantityOutput
from search.engine.region.pipeline import resolve_variable
from search.engine.role_text import DoorRole, Party, contradicts
from search.models import DefinitePart, Operation, ParsedEntity, ParsedPart, RoleAxis
from series_coverage.schema import RoleNames
from series_coverage.shared import CoverageSnapshot
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import (
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve
from tests.engine._quantity import NO_QUANTITY
from tests.engine.test_role_bearing_pipeline_e2e import _SV_DCID, _declaring_arcs
from tests.fixtures import FakeGraph, make_resolver

_CREDITOR_DEBTOR_ROLE_NAMES = RoleNames(donor="creditor", recipient="debtor")
_PARTY_AXES: dict[Party, RoleAxis] = {
    "source": "origin",
    "recipient": "destination",
    "creditor": "origin",
    "debtor": "destination",
}

_KEN = "country/KEN"
_CHN = "country/CHN"
_SELF_FLOW_QUERY = "aid from Kenya to Kenya"


def _extracted(text: str, role: Party | None, span: str | None = None) -> ExtractedEntity:
    """An extracted place whose cue is the preposition its span opens with, if any."""
    cue = span.split()[0] if span and span.split()[0] in {"from", "to"} else None
    return ExtractedEntity(text=text, span=span, role=role, cue=cue)


def _part_roles(
    entities: list[ExtractedEntity], *, query: str, measures: list[str]
) -> list[dict[str, list[DoorRole]]]:
    """The raw-text door's roles of each part for scripted `entities`, grounded as `extract`
    grounds them (`_ground_roles`) and read as `_prepare_extraction` reads an extraction of
    `measures`."""
    grounded = _ground_roles(query, entities, measures)
    extraction = Extraction(kind="resolution", variables=measures, entities=grounded)
    result = ExtractionResult(extraction=extraction, qualifiers={})
    return _prepare_extraction(extraction, result, query=query).per_variable_entity_roles


def _prepared_roles(
    entities: list[ExtractedEntity], *, query: str, measures: list[str]
) -> dict[str, list[DoorRole]]:
    """The roles of the one part of `measures` for scripted `entities`."""
    (roles,) = _part_roles(entities, query=query, measures=measures)
    return roles


def _occurrences(roles: dict[str, list[DoorRole]]) -> dict[str, list[tuple[RoleAxis, str]]]:
    """Each name's occurrences as ``(axis, span)``; `stated` has its own tests."""
    return {name: [(role.axis, role.span) for role in rs] for name, rs in roles.items()}


def _linked(
    text: str,
    role: Party | None = None,
    cue: str | None = None,
    *,
    parts: tuple[int, ...] = (),
    span: str | None = None,
    members: bool = False,
) -> ExtractedEntity:
    """An extracted place linked to the measures at `parts` (`ExtractedEntity.variables`)."""
    return ExtractedEntity(
        text=text,
        span=span or text,
        variables=list(parts),
        cue=cue,
        role=role,
        members=members,
    )


def _part_axes(
    query: str, measures: list[str], entities: list[ExtractedEntity]
) -> list[dict[str, list[RoleAxis]]]:
    """Each part's axes by name."""
    return [
        {name: [role.axis for role in roles] for name, roles in part.items()}
        for part in _part_roles(entities, query=query, measures=measures)
    ]


class TestRawTextDoorCarriesEveryOccurrence:
    def test_one_name_on_both_axes_carries_both_in_mention_order(self):
        roles = _prepared_roles(
            [
                _extracted("Kenya", "source", span="from Kenya"),
                _extracted("Kenya", "recipient", span="to Kenya"),
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
        ("query", "measures", "mentions", "expected"),
        [
            pytest.param(
                "aid from African countries to Africa",
                ["aid"],
                [("source", "from", True, ()), ("recipient", "to", False, ())],
                [[("origin", True), ("destination", False)]],
                id="each-reading-on-its-own-end",
            ),
            pytest.param(
                "aid to Africa, especially African countries",
                ["aid"],
                [("recipient", "to", False, ()), (None, None, True, ())],
                [[("destination", False)]],
                id="review-253-an-uncued-reading-stays-undirected",
            ),
            pytest.param(
                "ODA to Africa and GDP growth of African countries",
                ["ODA", "GDP growth"],
                [("recipient", "to", False, (0,)), (None, None, True, (1,))],
                [[("destination", False)], []],
                id="two-parts-flow-first",
            ),
            pytest.param(
                "GDP growth of African countries and ODA to Africa",
                ["GDP growth", "ODA"],
                [(None, None, True, (0,)), ("recipient", "to", False, (1,))],
                [[], [("destination", False)]],
                id="two-parts-flow-second",
            ),
            pytest.param(
                "aid to African countries, Africa",
                ["aid"],
                [("recipient", "to", True, ()), ("recipient", "to", True, ())],
                [[("destination", True)]],
                id="shared-reading",
            ),
        ],
    )
    def test_each_reading_of_a_set_holds_only_the_role_its_own_cue_states(
        self,
        query: str,
        measures: list[str],
        mentions: list[tuple[Party | None, str | None, bool, tuple[int, ...]]],
        expected: list[list[tuple[RoleAxis, bool]]],
    ):
        """A mention of Africa read whole and one read as its members are two places, so
        each holds the role of its own cue alone, in the parts it scopes alone."""
        entities = [
            _linked(
                "Africa",
                role,
                cue,
                parts=parts,
                span="African countries" if members else "Africa",
                members=members,
            )
            for role, cue, members, parts in mentions
        ]

        per_part = _part_roles(entities, query=query, measures=measures)

        assert [
            [(r.axis, r.members) for r in roles.get("Africa", [])] for roles in per_part
        ] == expected

    @pytest.mark.parametrize(
        ("query", "entities", "stated"),
        [
            # A cue beyond the place's own words states the direction.
            pytest.param(
                "aid to Sub-Saharan Africa",
                [("Sub-Saharan Africa", "recipient", "to")],
                {"Sub-Saharan Africa": True},
                id="preposition",
            ),
            pytest.param(
                "aid funded by Germany",
                [("Germany", "source", "funded by")],
                {"Germany": True},
                id="passive",
            ),
            pytest.param(
                "Norway funds Kenya",
                [("Norway", "source", "funds"), ("Kenya", "recipient", "funds")],
                {"Norway": True, "Kenya": True},
                id="verb",
            ),
            pytest.param(
                "aid to Kenya from Norway and Sweden",
                [("Kenya", "recipient", "to"), ("Sweden", "source", "from")],
                {"Kenya": True, "Sweden": True},
                id="coordinated",
            ),
            pytest.param(
                "aid from Norway to Kenya",
                [("Kenya", "recipient", "to Kenya")],
                {"Kenya": True},
                id="cue-written-with-its-place",
            ),
            # A possessive, "of" or a national adjective reads the direction off the measure.
            pytest.param(
                "top 10 debtors of China",
                [("China", "creditor", "of")],
                {"China": False},
                id="of",
            ),
            pytest.param(
                "is Ghana's economy growing",
                [("Ghana", "recipient", "Ghana's")],
                {"Ghana": False},
                id="possessive-with-its-place",
            ),
            pytest.param(
                "is Ghana's economy growing",
                [("Ghana", "recipient", "'s")],
                {"Ghana": False},
                id="possessive-marker",
            ),
            pytest.param(
                "is Ghana\u2019s economy growing",
                [("Ghana", "recipient", "Ghana\u2019s")],
                {"Ghana": False},
                id="typographic-possessive",
            ),
            pytest.param(
                "Norway give to the UN",
                [("Norway", "source", "Norway")],
                {"Norway": False},
                id="own-name",
            ),
            pytest.param(
                "is the Gambia's debt sustainable",
                [("Gambia", "debtor", "the Gambia's")],
                {"Gambia": False},
                id="article-and-possessive",
            ),
            pytest.param(
                "future debt service of Ghana to China",
                [("Ghana", "debtor", "of"), ("China", "creditor", "to")],
                {"Ghana": False, "China": True},
                id="debt-of-and-to",
            ),
        ],
    )
    def test_a_direction_is_stated_when_the_cue_reaches_beyond_the_place(
        self, query: str, entities: list[tuple[str, Party, str]], stated: dict[str, bool]
    ):
        extracted = [
            ExtractedEntity(text=text, span=text, cue=cue, role=role)
            for text, role, cue in entities
        ]
        roles = _prepared_roles(extracted, query=query, measures=["aid"])
        assert {name: [role.stated for role in rs] for name, rs in roles.items()} == {
            name: [value] for name, value in stated.items()
        }

    @pytest.mark.parametrize(
        ("query", "text", "span", "cue", "stated"),
        [
            pytest.param(
                "French aid to Senegal", "France", "French", "French", False, id="adjective"
            ),
            pytest.param(
                "Norjasta Keniaan apua", "Norway", "Norja", "Norjasta", True, id="case-ending"
            ),
            pytest.param("Norjasta Keniaan apua", "Norway", "Norja", "sta", True, id="bare-ending"),
            # An inflected name written as both span and cue reads like a national adjective:
            # nothing offline tells its case ending from the name, so the role stands and is
            # read off the measure.
            pytest.param(
                "Norjasta Keniaan apua",
                "Norway",
                "Norjasta",
                "Norjasta",
                False,
                id="inflected-name-as-its-own-cue",
            ),
            pytest.param(
                "the US's aid to Kenya",
                "United States",
                "US",
                "the US's",
                False,
                id="article-and-possessive-on-a-short-name",
            ),
        ],
    )
    def test_a_place_written_apart_from_its_name_states_by_the_words_beyond_its_span(
        self, query: str, text: str, span: str, cue: str, stated: bool
    ):
        """A national adjective is its own cue and states nothing, nor does a possessive
        with its article; a case ending beyond the span states the direction."""
        entity = ExtractedEntity(text=text, span=span, cue=cue, role="source")
        roles = _prepared_roles([entity], query=query, measures=["aid"])
        assert [role.stated for role in roles[text]] == [stated]

    def test_a_debt_names_its_creditor_and_its_debtor_by_party_words(self):
        """The creditor is the origin and the debtor the destination. The possessive debtor
        reads its direction off the measure; the creditor's "to" states it."""
        roles = _prepared_roles(
            [
                ExtractedEntity(text="Kenya", span="Kenya", cue="Kenya's", role="debtor"),
                ExtractedEntity(text="China", span="China", cue="to", role="creditor"),
            ],
            query="Kenya's debt to China",
            measures=["debt"],
        )
        assert roles == {
            "Kenya": [DoorRole(axis="destination", span="Kenya", stated=False, party="debtor")],
            "China": [DoorRole(axis="origin", span="China", stated=True, party="creditor")],
        }

    @pytest.mark.parametrize(
        ("query", "cue"),
        [
            ("net ODA received by Ethiopia in 1965", "received by"),
            ("the total ODA Ethiopia received from all donors", "received"),
        ],
    )
    def test_a_receipt_names_its_recipient_by_party_word(self, query: str, cue: str):
        roles = _prepared_roles(
            [ExtractedEntity(text="Ethiopia", span="Ethiopia", cue=cue, role="recipient")],
            query=query,
            measures=["ODA"],
        )
        assert roles == {
            "Ethiopia": [DoorRole(axis="destination", span="Ethiopia", party="recipient")]
        }

    def test_an_exact_repeat_collapses_to_one_occurrence(self):
        roles = _prepared_roles(
            [
                _extracted("Kenya", "recipient", span="to Kenya"),
                _extracted("Kenya", "recipient", span="to Kenya"),
            ],
            query=_SELF_FLOW_QUERY,
            measures=["aid"],
        )
        assert _occurrences(roles) == {"Kenya": [("destination", "to Kenya")]}

    def test_an_occurrence_with_no_role_is_skipped(self):
        roles = _prepared_roles(
            [
                _extracted("Kenya", None),
                _extracted("Kenya", "source", span="from Kenya"),
            ],
            query=_SELF_FLOW_QUERY,
            measures=["aid"],
        )
        assert _occurrences(roles) == {"Kenya": [("origin", "from Kenya")]}

    def test_a_name_with_no_role_in_any_occurrence_carries_no_entry(self):
        roles = _prepared_roles(
            [
                _extracted("Kenya", None),
                _extracted("Kenya", None),
                _extracted("Uganda", "source", span="from Uganda"),
            ],
            query=_SELF_FLOW_QUERY,
            measures=["aid"],
        )
        assert _occurrences(roles) == {"Uganda": [("origin", "from Uganda")]}

    def test_a_role_with_no_cue_carries_no_entry(self):
        """World knowledge: "ODA Kenya Norway" states no role, so a role the model cites no
        words for leaves the place undirected."""
        roles = _prepared_roles(
            [_linked("Norway", "source"), _linked("Kenya", "recipient")],
            query="ODA Kenya Norway",
            measures=["ODA"],
        )
        assert roles == {}

    @pytest.mark.parametrize(
        ("query", "entity", "axis"),
        [
            pytest.param(
                "how much does Kenya owe China",
                _linked("China", "creditor", "to"),
                "origin",
                id="misquoted-creditor",
            ),
            pytest.param(
                "how much does Kenya owe to China",
                _linked("Kenya", "debtor", "owes"),
                "destination",
                id="misquoted-debtor",
            ),
            pytest.param("Togo aid", _linked("Togo", "recipient", "to"), "destination", id="togo"),
        ],
    )
    def test_a_role_whose_cue_the_query_does_not_hold_is_read_off_the_measure(
        self, query: str, entity: ExtractedEntity, axis: RoleAxis
    ):
        roles = _prepared_roles([entity], query=query, measures=["debt"])
        assert roles == {
            entity.text: [DoorRole(axis=axis, span=entity.text, stated=False, party=entity.role)]
        }


def test_an_operation_keeps_its_axis_beside_the_roles():
    """Roles come from the entities alone, so an operation's axis is the extraction's."""
    extraction = Extraction(
        kind="resolution",
        variables=["ODA"],
        entities=[_linked("Kenya", "recipient", "received by")],
        operations=[ExtractedOperation(verb="rank", variable=0, phrase="Top", over="origin")],
    )
    result = ExtractionResult(
        extraction=extraction,
        qualifiers={},
        operations={"ODA": (Operation(verb="rank", phrase="Top", over="origin"),)},
    )

    prep = _prepare_extraction(extraction, result, query="Top donors of ODA received by Kenya")

    assert prep.per_variable_entity_roles == [
        {"Kenya": [DoorRole(axis="destination", span="Kenya", party="recipient")]}
    ]
    assert [op.over for ops in prep.per_variable_operations for op in ops] == ["origin"]


@pytest.mark.parametrize(
    ("parts", "open_axes"),
    [
        pytest.param((0,), [frozenset({"origin"}), frozenset()], id="scoped-to-one-part"),
        pytest.param((), [frozenset({"origin"})] * 2, id="scoping-every-part"),
    ],
)
def test_a_breakdown_entity_opens_its_role_on_the_parts_it_scopes(
    parts: tuple[int, ...], open_axes: list[frozenset[RoleAxis]]
):
    """ "every donor" asks one value per donor of the measures it scopes alone."""
    extraction = Extraction(
        kind="resolution",
        variables=["aid", "remittances"],
        entities=[
            _linked("Kenya", "recipient", "to"),
            ExtractedEntity(text="every donor", span="every donor", variables=list(parts)),
        ],
    )
    result = ExtractionResult(extraction=extraction, qualifiers={})

    prep = _prepare_extraction(
        extraction, result, query="aid to Kenya from every donor and remittances to Kenya"
    )

    assert prep.per_variable_open_axes == open_axes
    assert prep.per_variable_entities == [["Kenya"], ["Kenya"]]


@pytest.mark.parametrize(
    ("query", "measures", "entities", "expected"),
    [
        pytest.param(
            "ODA to Kenya and Uganda economic growth",
            ["ODA", "GDP growth"],
            [_linked("Kenya", "recipient", "to", parts=(0,)), _linked("Uganda", parts=(1,))],
            [{"Kenya": ["destination"]}, {}],
            id="review-254-no-carry-into-another-part",
        ),
        pytest.param(
            "How much aid was Kenya provided with by France?",
            ["aid"],
            [_linked("Kenya", "recipient", "provided with"), _linked("France", "source", "by")],
            [{"Kenya": ["destination"], "France": ["origin"]}],
            id="review-255-passive-names-the-giver-after-by",
        ),
        pytest.param(
            "How much does Zambia give to China in debt repayments, and what is Zambia's GDP?",
            ["debt repayments", "GDP"],
            [
                _linked("Zambia", "debtor", "give", parts=(0,)),
                _linked("China", "creditor", "to", parts=(0,)),
                _linked("Zambia", parts=(1,)),
            ],
            [{"Zambia": ["destination"], "China": ["origin"]}, {}],
            id="review-255-debt-repayments-name-the-creditor",
        ),
        # A name repeated in two parts keeps a separate role in each.
        pytest.param(
            "ODA from Kenya and remittances to Kenya",
            ["ODA", "remittances"],
            [
                _linked("Kenya", "source", "from", parts=(0,)),
                _linked("Kenya", "recipient", "to", parts=(1,)),
            ],
            [{"Kenya": ["origin"]}, {"Kenya": ["destination"]}],
            id="one-name-two-parts",
        ),
        # One mention scoping two measures carries its one role to both.
        pytest.param(
            "Japan's ODA disbursements and loans to Kenya",
            ["ODA disbursements", "loans"],
            [
                _linked("Japan", "source", "Japan's"),
                _linked("Kenya", "recipient", "to"),
            ],
            [
                {"Japan": ["origin"], "Kenya": ["destination"]},
                {"Japan": ["origin"], "Kenya": ["destination"]},
            ],
            id="one-mention-two-parts",
        ),
        # An unlinked place scopes every part, and with no role it directs none.
        pytest.param(
            "aid to Kenya and GDP of Uganda",
            ["aid", "GDP"],
            [_linked("Kenya", "recipient", "to", parts=(0,)), _linked("Uganda")],
            [{"Kenya": ["destination"]}, {}],
            id="unlinked-place-with-no-role",
        ),
    ],
)
def test_each_part_reads_its_own_places(
    query: str,
    measures: list[str],
    entities: list[ExtractedEntity],
    expected: list[dict[str, list[RoleAxis]]],
):
    """Each part's roles come from the places linked to its measure alone."""
    assert _part_axes(query, measures, entities) == expected


# (query, measures, entities) of one part, and the same with an unrelated part added.
_WITH_AN_UNRELATED_PART = [
    pytest.param(
        ("aid from Norway to Kenya", ["aid"]),
        ("aid from Norway to Kenya and GDP of Uganda", ["aid", "GDP"]),
        [
            _linked("Norway", "source", "from", parts=(0,)),
            _linked("Kenya", "recipient", "to", parts=(0,)),
        ],
        [_linked("Uganda", parts=(1,))],
        id="cued",
    ),
    pytest.param(
        ("ODA to Kenya and Uganda", ["ODA"]),
        ("ODA to Kenya and Uganda and GDP growth of Rwanda", ["ODA", "GDP growth"]),
        [
            _linked("Kenya", "recipient", "to", parts=(0,)),
            _linked("Uganda", "recipient", "to", parts=(0,)),
        ],
        [_linked("Rwanda", parts=(1,))],
        id="coordinated",
    ),
    pytest.param(
        ("How much does Zambia owe China?", ["external debt"]),
        ("How much does Zambia owe China, and what is Kenya's GDP?", ["external debt", "GDP"]),
        [
            _linked("Zambia", "debtor", "owe", parts=(0,)),
            _linked("China", "creditor", "owe", parts=(0,)),
        ],
        [_linked("Kenya", parts=(1,))],
        id="debt",
    ),
]


@pytest.mark.parametrize(("one", "two", "entities", "added"), _WITH_AN_UNRELATED_PART)
def test_an_unrelated_part_leaves_the_roles_of_another_unchanged(
    one: tuple[str, list[str]],
    two: tuple[str, list[str]],
    entities: list[ExtractedEntity],
    added: list[ExtractedEntity],
):
    (alone,) = _part_axes(*one, entities)
    assert alone
    assert _part_axes(*two, [*entities, *added])[0] == alone


@pytest.mark.parametrize(
    ("active", "passive", "parties"),
    [
        (
            (
                "How much aid did France provide to Kenya?",
                [_linked("France", "source", "provide"), _linked("Kenya", "recipient", "to")],
            ),
            (
                "How much aid was Kenya provided with by France?",
                [_linked("Kenya", "recipient", "provided with"), _linked("France", "source", "by")],
            ),
            {"France": "source", "Kenya": "recipient"},
        ),
        (
            (
                "Norway gave aid to Kenya",
                [_linked("Norway", "source", "gave"), _linked("Kenya", "recipient", "to")],
            ),
            (
                "aid was given to Kenya by Norway",
                [_linked("Kenya", "recipient", "to"), _linked("Norway", "source", "by")],
            ),
            {"Norway": "source", "Kenya": "recipient"},
        ),
        (
            (
                "how much does Kenya owe China",
                [_linked("Kenya", "debtor", "owe"), _linked("China", "creditor", "owe")],
            ),
            (
                "debt owed to China by Kenya",
                [_linked("China", "creditor", "owed to"), _linked("Kenya", "debtor", "by")],
            ),
            {"China": "creditor", "Kenya": "debtor"},
        ),
    ],
)
def test_an_active_sentence_and_its_passive_keep_the_same_parties(
    active: tuple[str, list[ExtractedEntity]],
    passive: tuple[str, list[ExtractedEntity]],
    parties: dict[str, Party],
):
    """Voice moves the cue, and the party word keeps the axis: every cue here is a verb or
    preposition, so each direction is stated."""

    def read(query: str, entities: list[ExtractedEntity]) -> dict[str, list[DoorRole]]:
        (roles,) = _part_roles(entities, query=query, measures=["aid"])
        return roles

    expected = {
        name: [DoorRole(axis=_PARTY_AXES[party], span=name, party=party)]
        for name, party in parties.items()
    }
    assert read(*active) == expected
    assert read(*passive) == expected


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
                DoorRole(axis="destination", span="Kenya", party="recipient"),
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
    door_roles = _prepared_roles(
        [
            _extracted("Kenya", "source", span="from Kenya"),
            _extracted("Kenya", "recipient", span="to Kenya"),
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


@pytest.mark.parametrize(
    ("query", "debtor_cue", "creditor_cue"),
    [
        ("Kenya's debt to China", "Kenya's", "to"),
        ("Kenya's debt owed to China", "Kenya's", "owed to"),
        ("Kenya gives to China in debt repayments", "gives", "to"),
    ],
)
def test_debt_party_words_bind_the_creditor_as_origin_and_the_debtor_as_destination(
    monkeypatch, query: str, debtor_cue: str, creditor_cue: str
):
    """A by-creditor series answers with the creditor China on origin, whatever word
    governs the "to" before it."""
    door_roles = _prepared_roles(
        [
            ExtractedEntity(text="Kenya", span="Kenya", cue=debtor_cue, role="debtor"),
            ExtractedEntity(text="China", span="China", cue=creditor_cue, role="creditor"),
        ],
        query=query,
        measures=["debt"],
    )
    _assert_binds_china_as_creditor(monkeypatch, query, door_roles)


@pytest.mark.parametrize("variable_text", ["external debt owed to", "external debt arrears to"])
def test_parsed_door_binds_a_declared_creditor_after_to(monkeypatch, variable_text: str):
    """A declared creditor after "to" in parsed variable text binds on origin, because the
    party word "creditor" exempts it from the preposition veto."""
    part = ParsedPart(
        variable_text=variable_text,
        entities=[
            ParsedEntity(text="China", role_text="creditor"),
            ParsedEntity(text="Kenya", role_text="debtor"),
        ],
    )
    role_query, warnings = _role_query_for_part(part)
    assert (role_query, warnings) == (f"{variable_text} China Kenya", [])
    door_roles = _entity_roles_for_part(part)
    assert door_roles == {
        "China": [DoorRole(axis="origin", span="China", party="creditor")],
        "Kenya": [DoorRole(axis="destination", span="Kenya", party="debtor")],
    }
    assert contradicts(role_query, span="China", axis="origin", party="creditor") is False
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
    """ "China loans to Kenya": China's name before "loans" carries no role, and Kenya's
    cue makes Kenya the debtor. A by-creditor series places China on the end Kenya leaves
    open."""
    query = "China loans to Kenya"
    door_roles = _prepared_roles(
        [
            ExtractedEntity(text="China", span="China"),
            ExtractedEntity(text="Kenya", span="Kenya", cue="to", role="debtor"),
        ],
        query=query,
        measures=["loans"],
    )

    _assert_binds_china_as_creditor(monkeypatch, query, door_roles)


_ZMB = "country/ZMB"


class _ZambiaDebtToChinaLLM:
    """Extracts "external debt" with Zambia as `zambia_role` on its own name and China as
    the creditor its "to" names, and elects the first shape with no bindings."""

    def __init__(self, zambia_role: Party | None):
        self._zambia = ExtractedEntity(
            text="Zambia",
            span="Zambia",
            cue="Zambia" if zambia_role else None,
            role=zambia_role,
        )

    async def generate_structured(self, *, prompt, system, schema):
        if schema is Extraction:
            return Extraction(
                kind="resolution",
                variables=["external debt"],
                entities=[
                    self._zambia,
                    ExtractedEntity(text="China", span="China", cue="to", role="creditor"),
                ],
            ), None
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, bindings=[]), None


@pytest.mark.parametrize("zambia_role", ["debtor", None])
def test_a_raw_text_debt_to_a_creditor_binds_the_place_before_it_as_debtor(
    monkeypatch, zambia_role: Party | None
):
    """ "Zambia external debt to China" answers a series holding both directions with the
    creditor China on origin and Zambia on destination, whether the extraction names Zambia
    the debtor or leaves it undirected."""
    patch_coverage(
        monkeypatch,
        [
            (_SV_DCID, "f1", _CHN, _ZMB, [2018, 2019, 2020]),
            (_SV_DCID, "f1", _ZMB, _CHN, [2018, 2019, 2020]),
        ],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    graph = FakeGraph(
        nodes={
            _ZMB: {"label": "Zambia", "type": "Country"},
            _CHN: {"label": "China", "type": "Country"},
            _SV_DCID: {
                "label": "Test debt",
                "arcs": _declaring_arcs(observation_properties=["creditor", "debtor"]),
            },
        },
        detect={"external debt": {"svs": [_SV_DCID], "cosine_scores": [0.9]}},
        resolve={"Zambia": _ZMB, "China": _CHN},
        obs={},
    )

    response = offline_resolve(
        make_request("Zambia external debt to China"),
        graph=graph,
        llm=_ZambiaDebtToChinaLLM(zambia_role),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    assert {(entity.ref, entity.role) for entity in part.spec.resolved.entities} == {
        (_CHN, "origin"),
        (_ZMB, "destination"),
    }
    assert part.spec.definition.identity.entities == {"origin": [_CHN], "destination": [_ZMB]}
