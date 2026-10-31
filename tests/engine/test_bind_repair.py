"""Tests for declared defaults, the label matching they share with bind, and a gendered
query on a family whose gendered members have no data at the place.

Synthetic properties isolate constraint matching.
"""

from __future__ import annotations

import asyncio

import pytest

from code_index.shared import get_index
from code_index.tokens import tokenize
from search.engine.bind import (
    DraftBindingEvidence,
    DraftBindingKind,
    SlotBindingDraft,
)
from search.engine.bind_repair import (
    _label_named_in_text,
    apply_declared_defaults,
    apply_named_exclusions,
    declared_default_slots_named,
    defaulted_slot_notes,
    defaulted_slots_named,
)
from search.engine.exclusion_read import LeftOut
from search.engine.qualifier_match import QualifierMatch, QualifierTie
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.shape import ShapeDraft, shape_draft_from
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver


def _gender_arcs(gender: str | None) -> dict:
    """Arcs for a Person/count member: `gender` is a real constraint value
    when given, or the member is silent on the axis entirely (an ungendered
    aggregate) when None."""
    if gender is None:
        return {"constraintProperties": {"nodes": []}}
    return {
        "constraintProperties": {"nodes": [{"dcid": "gender"}]},
        "gender": {"nodes": [{"dcid": gender}]},
    }


class TestLabelNamedInTextCodeSystemValue:
    """Code-system labels omit suffixes so plain query wording can match (issue #866)."""

    def test_plain_code_system_label_matches_a_realistic_query(self):
        assert _label_named_in_text("standard grant", "give me odf standard grant flows") is True

    def test_suffixed_graph_label_never_matches_the_same_query(self):
        """The code-system suffix blocks a match to the plain query phrase."""
        assert (
            _label_named_in_text("standard grant (CRS Finance)", "give me standard grant flows")
            is False
        )


class TestGenderedQueryWhereOnlyTheAggregateHasData:
    """Only the ungendered aggregate has data in Togo. The bind menu still offers every
    gender value, since availability narrows no taxonomy, so a female-population query
    binds `Female` and the part refuses with the female member flagged."""

    _AGGREGATE_SV = "Count_Person"
    _FEMALE_SV = "Count_Person_Female"
    _MALE_SV = "Count_Person_Male"
    _RECIPIENT = "country/TGO"
    _DETECT_QUERY = "population"

    class _BindFemaleLLM:
        """Elects the sole shape and binds gender to the offered `Female` value; refuses a
        menu whose gender taxonomy does not offer it."""

        async def generate_structured(self, *, prompt, system, schema):
            if schema is QuantityOutput:
                return NO_QUANTITY
            assert schema.__name__ == "_UniversalBindOutput"
            assert "'Female'" in prompt, "the gender taxonomy must offer Female"
            return elected_reply(
                0,
                bindings=[
                    SlotBindingDraft(
                        axis="how",
                        property_dcid="gender",
                        kind="value",
                        value_dcids=["Female"],
                    ),
                ],
                also_plausible=None,
            ), None

    @staticmethod
    def _population_gender_arcs(gender: str | None) -> dict:
        """Build Person/count arcs with an optional gender constraint."""
        arcs = {
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        }
        arcs.update(_gender_arcs(gender))
        return arcs

    @classmethod
    def _graph(cls) -> FakeGraph:
        nodes = {
            "gender": {"label": "Gender"},
            "Female": {"label": "Female"},
            "Male": {"label": "Male"},
            cls._AGGREGATE_SV: {
                "label": "Population",
                "type": "StatisticalVariable",
                "arcs": cls._population_gender_arcs(None),
            },
            cls._FEMALE_SV: {
                "label": "Female population",
                "type": "StatisticalVariable",
                "arcs": cls._population_gender_arcs("Female"),
            },
            cls._MALE_SV: {
                "label": "Male population",
                "type": "StatisticalVariable",
                "arcs": cls._population_gender_arcs("Male"),
            },
            cls._RECIPIENT: {"label": "Togo", "type": "Country"},
        }
        obs = {f"{cls._AGGREGATE_SV}|{cls._RECIPIENT}": [{"obsCount": 137}]}
        return FakeGraph(
            nodes=nodes,
            obs=obs,
            detect={cls._DETECT_QUERY: {"svs": [cls._AGGREGATE_SV, cls._FEMALE_SV, cls._MALE_SV]}},
            resolve={"Togo": cls._RECIPIENT},
        )

    def test_female_query_flags_the_gendered_member_as_no_data(self):
        result = asyncio.run(
            resolve_variable(
                "female population of togo",
                entities=["Togo"],
                date_request=None,
                detect_query=self._DETECT_QUERY,
                role_query="female population of togo",
                pac=True,
                graph=self._graph(),
                resolver=make_resolver(self._graph()),
                llm=self._BindFemaleLLM(),
                base_timing={},
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        )

        assert result.status == "no_data"
        assert result.no_data_reason == "no_observations"
        assert result.gate == "no_member_has_data_at_place"
        assert result.flagged_match is not None
        assert len(result.flagged_match) == 1
        flagged = result.flagged_match[0]
        assert [sv.ref.ref for sv in flagged.resolved.members] == [self._FEMALE_SV], (
            "the flagged match must name the gendered member the binding points at, "
            "never the ungendered aggregate"
        )


# ---------------------------------------------------------------------------
# apply_declared_defaults
# ---------------------------------------------------------------------------


def _grants_shape(
    *,
    crs_type_of_flow_taxonomy: list[str] | None = ("CRSTypeOfFlow10", "CRSTypeOfFlow21"),
) -> ShapeDraft:
    """Build a CRS-flow shape with explicit slot taxonomies for default tests."""
    taxonomy: dict[str, list[str]] = {
        "how:crsTypeOfFinance": ["CRSTypeOfFinance110", "CRSTypeOfFinance421"],
    }
    if crs_type_of_flow_taxonomy is not None:
        taxonomy["how:crsTypeOfFlow"] = list(crs_type_of_flow_taxonomy)
    return shape_draft_from(
        shape_id="crs_flow_amount",
        label="CRS flow",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        constraint_props=["crsTypeOfFinance", "crsTypeOfFlow"],
        prop_labels={"crsTypeOfFinance": "instrument", "crsTypeOfFlow": "flow"},
        prop_observed_values={
            "crsTypeOfFinance": ["CRSTypeOfFinance110", "CRSTypeOfFinance421"],
            "crsTypeOfFlow": list(crs_type_of_flow_taxonomy or ()),
        },
        sv_arc_facts={},
        slot_taxonomy=taxonomy,
        place_available=True,
        value_labels=None,
        recalled=False,
    )


def _flow_unbound(evidence: DraftBindingEvidence = "mentioned") -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how", property_dcid="crsTypeOfFlow", kind="unbound", evidence=evidence
    )


def _finance(
    value: str = "CRSTypeOfFinance110",
    *,
    kind: DraftBindingKind = "value",
    evidence: DraftBindingEvidence = "mentioned",
) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how",
        property_dcid="crsTypeOfFinance",
        kind=kind,
        value_dcids=[value] if kind == "value" else [],
        evidence=evidence,
    )


def _imf_shape(*, labelled: bool = True) -> ShapeDraft:
    """An IDS shape whose `includesIMFCredit` slot holds one value, labelled in the graph."""
    return shape_draft_from(
        shape_id="ids_debt",
        label="External debt",
        pop_type_dcid="FinancialPosition",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        constraint_props=["includesIMFCredit"],
        prop_labels={"includesIMFCredit": "includes IMF credit"},
        prop_observed_values={"includesIMFCredit": ["WithoutIMFCredit"]},
        sv_arc_facts={},
        slot_taxonomy={"how:includesIMFCredit": ["WithoutIMFCredit"]},
        place_available=True,
        value_labels={"WithoutIMFCredit": "without IMF credit"} if labelled else None,
        recalled=False,
    )


def _imf_slot(**update: object) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how", property_dcid="includesIMFCredit", kind="unbound", evidence="mentioned"
    ).model_copy(update=update)


_WITHOUT_IMF = LeftOut(phrase="without IMF credit", things=("IMF credit",), at=0)


class TestApplyNamedExclusions:
    def test_a_stretch_that_is_a_values_own_label_binds_it_and_counts_as_met(self):
        repaired, met = apply_named_exclusions(
            [_imf_slot()], _imf_shape(), exclusions=[_WITHOUT_IMF]
        )

        assert [(b.kind, b.value_dcids) for b in repaired] == [("value", ["WithoutIMFCredit"])]
        assert met == {"without IMF credit"}

    @pytest.mark.parametrize(
        ("binding", "shape", "exclusion", "pinned"),
        [
            pytest.param(
                _imf_slot(evidence="requested_all"),
                _imf_shape(),
                _WITHOUT_IMF,
                frozenset(),
                id="slot-asked-in-full",
            ),
            pytest.param(
                _imf_slot(), _imf_shape(), _WITHOUT_IMF, {"includesIMFCredit"}, id="pinned"
            ),
            pytest.param(_imf_slot(), _imf_shape(labelled=False), _WITHOUT_IMF, (), id="no-label"),
            pytest.param(
                _imf_slot(),
                _imf_shape(),
                LeftOut(phrase="excluding IMF credit", things=("IMF credit",), at=0),
                (),
                id="another-wording",
            ),
        ],
    )
    def test_anything_but_an_open_slot_and_an_exact_label_binds_nothing(
        self, binding, shape, exclusion, pinned
    ):
        repaired, met = apply_named_exclusions(
            [binding], shape, exclusions=[exclusion], pinned_properties=frozenset(pinned)
        )

        assert repaired == [binding]
        assert met == frozenset()


class TestApplyDeclaredDefaultsFires:
    def test_grants_with_no_flow_type_named_defaults_to_oda(self):
        shape = _grants_shape()
        bindings = [_finance(), _flow_unbound()]

        repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert len(defaulted) == 1
        assert defaulted[0].property_dcid == "crsTypeOfFlow"
        assert defaulted[0].kind == "unbound"
        flow = next(b for b in repaired if b.property_dcid == "crsTypeOfFlow")
        assert flow.kind == "value"
        assert flow.value_dcids == ["CRSTypeOfFlow10"]
        # crsTypeOfFinance is untouched.
        finance = next(b for b in repaired if b.property_dcid == "crsTypeOfFinance")
        assert finance.value_dcids == ["CRSTypeOfFinance110"]

    def test_fires_through_a_descendant_of_the_grant_row_not_only_the_row_itself(self):
        """The real descendant CRSTypeOfFinance210 triggers the grant default through the index."""
        shape = _grants_shape()
        bindings = [_finance("CRSTypeOfFinance210"), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert len(defaulted) == 1


class TestApplyDeclaredDefaultsDoesNotFire:
    def test_shape_declares_no_crs_type_of_flow_slot(self):
        # The shape has no crsTypeOfFlow slot.
        shape = shape_draft_from(
            shape_id="crs_finance_only",
            label="CRS finance",
            pop_type_dcid="FinancialFlow",
            meas_prop_dcid="amount",
            stat_type_dcid="measuredValue",
            meas_qual_dcid=None,
            meas_denom_dcid=None,
            constraint_props=["crsTypeOfFinance"],
            prop_labels={"crsTypeOfFinance": "instrument"},
            prop_observed_values={"crsTypeOfFinance": ["CRSTypeOfFinance110"]},
            sv_arc_facts={},
            slot_taxonomy={"how:crsTypeOfFinance": ["CRSTypeOfFinance110"]},
            place_available=True,
            value_labels=None,
            recalled=False,
        )
        bindings = [_finance()]

        repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert defaulted == []
        assert repaired == bindings

    def test_crs_type_of_flow_already_bound(self):
        shape = _grants_shape()
        bindings = [
            _finance(),
            SlotBindingDraft(
                axis="how",
                property_dcid="crsTypeOfFlow",
                kind="value",
                value_dcids=["CRSTypeOfFlow21"],
            ),
        ]

        _repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert defaulted == []

    def test_evidence_requested_all_suppresses_the_default(self):
        """A requested-all flow slot stays open even when a grant instrument is bound."""
        shape = _grants_shape()
        bindings = [_finance(), _flow_unbound(evidence="requested_all")]

        _repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert defaulted == []

    def test_a_flow_type_match_recognised_in_text_suppresses_the_default(self):
        """A recognised flow-type match suppresses the default even when its binding is unbound."""
        shape = _grants_shape()
        bindings = [_finance(), _flow_unbound()]
        match = QualifierMatch(
            code_system="crsTypeOfFlow",
            code_dcid="CRSTypeOfFlow21",
            label="non-export credit official flow",
            source="synonym",
            matched_text="other official flows",
        )

        _repaired, defaulted = apply_declared_defaults(
            bindings, shape, index=get_index(), query_qualifier_matches=(match,)
        )

        assert defaulted == []

    def test_a_flow_type_tie_recognised_in_text_suppresses_the_default(self):
        shape = _grants_shape()
        bindings = [_finance(), _flow_unbound()]
        tie = QualifierTie(
            matched_text="flows",
            rows=(get_index().codes_by_tokens(tokenize("official development assistance"))[0],),
        )

        _repaired, defaulted = apply_declared_defaults(
            bindings, shape, index=get_index(), query_qualifier_ties=(tie,)
        )

        assert defaulted == []

    def test_finance_bound_but_not_mentioned_does_not_fire(self):
        """The grants default requires a text-derived finance binding."""
        shape = _grants_shape()
        bindings = [_finance(evidence="sole_option"), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert defaulted == []

    def test_finance_bound_to_a_loan_value_does_not_fire(self):
        """The grants default applies to finance values in the grant subtree."""
        shape = _grants_shape()
        bindings = [_finance("CRSTypeOfFinance421"), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert defaulted == []

    def test_finance_itself_unbound_does_not_fire(self):
        shape = _grants_shape()
        bindings = [
            SlotBindingDraft(axis="how", property_dcid="crsTypeOfFinance", kind="unbound"),
            _flow_unbound(),
        ]

        _repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert defaulted == []

    def test_shape_never_carries_the_oda_flow_type(self):
        """The default is skipped when CRSTypeOfFlow10 is absent from the shape taxonomy."""
        shape = _grants_shape(crs_type_of_flow_taxonomy=["CRSTypeOfFlow21", "CRSTypeOfFlow60"])
        bindings = [_finance(), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert defaulted == []

    def test_finance_bound_as_a_set_does_not_fire(self):
        """A mixed set leaves the grants default inactive."""
        shape = _grants_shape()
        bindings = [
            SlotBindingDraft(
                axis="how",
                property_dcid="crsTypeOfFinance",
                kind="set",
                value_dcids=["CRSTypeOfFinance110", "CRSTypeOfFinance421"],
                evidence="mentioned",
            ),
            _flow_unbound(),
        ]

        _repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert defaulted == []

    def test_finance_trigger_engine_authored_does_not_fire(self):
        """Query-derived defaults require text-authored finance bindings."""
        shape = _grants_shape()
        bindings = [_finance(), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(
            bindings,
            shape,
            index=get_index(),
            engine_authored_properties=frozenset({"crsTypeOfFinance"}),
        )

        assert defaulted == []


class TestApplyDeclaredDefaultsWhenNamed:
    """The whole-word term "aid" can trigger the default in the variable or raw query."""

    def test_aid_as_the_whole_variable_fires(self):
        shape = _grants_shape()
        bindings = [_finance(kind="unbound", evidence="mentioned"), _flow_unbound()]

        repaired, defaulted = apply_declared_defaults(
            bindings, shape, index=get_index(), variable="aid to Kenya"
        )

        assert len(defaulted) == 1
        assert defaulted[0].property_dcid == "crsTypeOfFlow"
        flow = next(b for b in repaired if b.property_dcid == "crsTypeOfFlow")
        assert flow.kind == "value"
        assert flow.value_dcids == ["CRSTypeOfFlow10"]

    def test_aid_in_query_text_alone_fires(self):
        """The raw query triggers the default when extraction omits "aid"."""
        shape = _grants_shape()
        bindings = [_finance(kind="unbound", evidence="mentioned"), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(
            bindings,
            shape,
            index=get_index(),
            variable="funding to Kenya",
            query_text="How much aid did Kenya get?",
        )

        assert [b.property_dcid for b in defaulted] == ["crsTypeOfFlow"]

    def test_negated_aid_in_query_text_does_not_fire(self):
        """The raw query uses the same negation check as the extracted variable."""
        shape = _grants_shape()
        bindings = [_finance(kind="unbound", evidence="mentioned"), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(
            bindings,
            shape,
            index=get_index(),
            variable="funding to Kenya",
            query_text="non-aid funding to Kenya",
        )

        assert defaulted == []

    def test_aid_inside_another_word_does_not_fire(self):
        """Whole-word matching distinguishes "aid" from "paid" and "aided"."""
        shape = _grants_shape()
        for variable in ("who paid Kenya in 2020", "France aided Kenya in 2020"):
            bindings = [_finance(kind="unbound", evidence="mentioned"), _flow_unbound()]

            _repaired, defaulted = apply_declared_defaults(
                bindings, shape, index=get_index(), variable=variable
            )

            assert defaulted == [], variable

    def test_a_named_flow_type_blocks_the_named_trigger(self):
        """A flow-type match suppresses the default even when the variable names "aid"."""
        shape = _grants_shape()
        bindings = [_finance(kind="unbound", evidence="mentioned"), _flow_unbound()]
        match = QualifierMatch(
            code_system="crsTypeOfFlow",
            code_dcid="CRSTypeOfFlow21",
            label="non-export credit official flow",
            source="synonym",
            matched_text="other official flows",
        )

        _repaired, defaulted = apply_declared_defaults(
            bindings,
            shape,
            index=get_index(),
            variable="aid",
            query_qualifier_matches=(match,),
        )

        assert defaulted == []

    def test_a_requested_all_flow_slot_blocks_the_named_trigger(self):
        """A requested-all flow slot stays open despite the "aid" trigger."""
        shape = _grants_shape()
        bindings = [
            _finance(kind="unbound", evidence="mentioned"),
            _flow_unbound(evidence="requested_all"),
        ]

        _repaired, defaulted = apply_declared_defaults(
            bindings, shape, index=get_index(), variable="aid to Kenya across all flows"
        )

        assert defaulted == []

    @pytest.mark.parametrize(
        "variable",
        [
            pytest.param("not aid financing to kenya", id="leading-negator"),
            pytest.param("financing to kenya excluding aid", id="exclusion-framing"),
            pytest.param("non-aid financing to kenya", id="fused-non-prefix"),
            pytest.param("non aid financing to kenya", id="spaced-non-prefix"),
            pytest.param("financing to kenya other than aid", id="other-than"),
            pytest.param("financing to kenya, exclude aid", id="exclude-verb"),
            pytest.param("financing to kenya, excludes aid", id="excludes-verb"),
            pytest.param("financing to kenya rather than aid", id="rather-than"),
        ],
    )
    def test_negated_or_excluded_aid_does_not_fire(self, variable: str):
        """Negated or excluded "aid" must not trigger the ODA default."""
        shape = _grants_shape()
        bindings = [_finance(kind="unbound", evidence="mentioned"), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(
            bindings, shape, index=get_index(), variable=variable
        )

        assert defaulted == [], variable

    def test_a_bound_flow_slot_blocks_the_named_trigger(self):
        """An existing CRSTypeOfFlow30 binding takes precedence over the "aid" trigger."""
        shape = _grants_shape()
        bindings = [
            _finance(kind="unbound", evidence="mentioned"),
            SlotBindingDraft(
                axis="how",
                property_dcid="crsTypeOfFlow",
                kind="value",
                value_dcids=["CRSTypeOfFlow30"],
            ),
        ]

        _repaired, defaulted = apply_declared_defaults(
            bindings, shape, index=get_index(), variable="private aid to Kenya"
        )

        assert defaulted == []

    @pytest.mark.parametrize(
        "variable",
        [
            pytest.param("aid-for-trade flows to Kenya", id="hyphenated-compound"),
            pytest.param("aid for trade to Kenya", id="spaced-compound"),
        ],
    )
    def test_aid_for_trade_fires_the_oda_default(self, variable: str):
        """Aid for Trade is an OECD/DAC category within ODA, with or without a hyphen."""
        shape = _grants_shape()
        bindings = [_finance(kind="unbound", evidence="mentioned"), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(
            bindings, shape, index=get_index(), variable=variable
        )

        assert len(defaulted) == 1
        assert defaulted[0].property_dcid == "crsTypeOfFlow"


class TestDefaultedSlotsNamed:
    """A fired default is reported only where a text names its slot's plain name."""

    @pytest.mark.parametrize(
        ("texts", "expected"),
        [
            pytest.param(
                ["aid flows"],
                [("flow", "official development assistance")],
                id="plural-in-variable",
            ),
            pytest.param(
                ["aid", "top 5 flow"], [("flow", "official development assistance")], id="singular"
            ),
            pytest.param(["aid", "Which sectors got the most aid"], [], id="no-flow-noun"),
            pytest.param(["aid", "cashflow"], [], id="not-a-whole-word"),
        ],
    )
    def test_reports_the_slot_and_its_default_label(self, texts, expected):
        _repaired, defaulted = apply_declared_defaults(
            [_finance(kind="unbound"), _flow_unbound()],
            _grants_shape(),
            index=get_index(),
            variable="aid",
        )

        assert defaulted_slots_named(defaulted, index=get_index(), texts=texts) == expected
        # The CRS default declares no note, so it speaks only where the query names its slot.
        assert defaulted_slot_notes(defaulted, index=get_index()) == []

    def test_declared_default_slots_named_reads_only_slots_with_a_default(self):
        """Every declared plain name appears, but only the flow slot has a default."""
        text = "aid flows by sector, instrument and modality"

        assert declared_default_slots_named(text, index=get_index()) == ("crsTypeOfFlow",)


def _dac1_shape() -> ShapeDraft:
    """A DAC1 gross-disbursement shape whose aid-type slot offers total ODA and bilateral ODA."""
    return shape_draft_from(
        shape_id="dac1_gross",
        label="DAC1 gross disbursements",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        constraint_props=["dac1AidType"],
        prop_labels={"dac1AidType": "aid type"},
        prop_observed_values={"dac1AidType": ["DAC1AidType1010", "DAC1AidType1015"]},
        sv_arc_facts={},
        slot_taxonomy={"how:dac1AidType": ["DAC1AidType1010", "DAC1AidType1015"]},
        place_available=True,
        value_labels=None,
        recalled=False,
    )


def _aid_type(kind: DraftBindingKind = "unbound", value: str | None = None) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how",
        property_dcid="dac1AidType",
        kind=kind,
        value_dcids=[value] if value else [],
        evidence="mentioned",
    )


class TestOpenSlotDefault:
    """`dac1AidType` declares total ODA as a default that fires on every open slot
    (`DeclaredDefault.when_open`) and states its note."""

    def test_fires_on_a_question_naming_no_aid_type(self):
        repaired, defaulted = apply_declared_defaults(
            [_aid_type()], _dac1_shape(), index=get_index(), variable="aid from Japan"
        )

        assert [b.value_dcids for b in repaired] == [["DAC1AidType1010"]]
        notes = defaulted_slot_notes(defaulted, index=get_index())
        assert notes == [get_index().declarations["dac1AidType"].default.note]
        assert defaulted_slots_named(defaulted, index=get_index(), texts=["aid type"]) == []

    @pytest.mark.parametrize(
        ("binding", "matches"),
        [
            pytest.param(
                _aid_type(),
                (
                    QualifierMatch(
                        code_system="dac1AidType",
                        code_dcid="DAC1AidType1015",
                        label="bilateral official development assistance",
                        source="label",
                        matched_text="bilateral ODA",
                    ),
                ),
                id="named-in-text",
            ),
            pytest.param(_aid_type("value", "DAC1AidType1015"), (), id="bound"),
            pytest.param(
                SlotBindingDraft(
                    axis="how",
                    property_dcid="dac1AidType",
                    kind="unbound",
                    evidence="requested_all",
                ),
                (),
                id="requested-all",
            ),
        ],
    )
    def test_is_silent_where_the_query_names_an_aid_type(self, binding, matches):
        _repaired, defaulted = apply_declared_defaults(
            [binding],
            _dac1_shape(),
            index=get_index(),
            variable="bilateral ODA from Japan",
            query_qualifier_matches=matches,
        )

        assert defaulted == []

    def test_the_crs_flow_default_still_needs_its_trigger(self):
        """An open CRS flow slot with no grant bound and no "aid" in the text stays open."""
        _repaired, defaulted = apply_declared_defaults(
            [_finance(kind="unbound"), _flow_unbound()],
            _grants_shape(),
            index=get_index(),
            variable="official flows to Kenya",
        )

        assert defaulted == []
