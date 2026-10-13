"""Pins the resolve response union: three kinds discriminated on `kind`, and the
operations every part carries."""

from __future__ import annotations

import jsonschema
import pytest
from pydantic import ValidationError

from search.models import (
    RESOLVE_RESPONSE_ADAPTER,
    CandidatesPart,
    CatalogueAnswer,
    DeclinedAnswer,
    DefinitePart,
    GraphRef,
    NoData,
    NoDataDenominator,
    NoDataPart,
    Operation,
    ParsedInput,
    ParsedPart,
    Resolution,
    Scope,
    Spec,
    UnsupportedPart,
)
from tests.conftest import base_response, minimal_spec

_KENYA = GraphRef(ref="country/KEN", label="Kenya")


def _resolution(worked_example: dict) -> Resolution:
    return Resolution.model_validate(worked_example)


def _catalogue() -> CatalogueAnswer:
    return CatalogueAnswer.model_validate(base_response(kind="catalogue", scope={"place": None}))


def _declined() -> DeclinedAnswer:
    return DeclinedAnswer.model_validate(
        base_response(
            kind="declined",
            reason="requires_analysis",
            scope=Scope(place=_KENYA).model_dump(),
        )
    )


@pytest.fixture
def answers(worked_example_response: dict) -> list[Resolution | CatalogueAnswer | DeclinedAnswer]:
    return [_resolution(worked_example_response), _catalogue(), _declined()]


class TestUnionRoundTrip:
    def test_each_kind_round_trips_through_python(self, answers):
        for answer in answers:
            dumped = RESOLVE_RESPONSE_ADAPTER.dump_python(answer, mode="json")
            restored = RESOLVE_RESPONSE_ADAPTER.validate_python(dumped)
            assert type(restored) is type(answer)
            assert restored == answer

    def test_each_kind_round_trips_through_json(self, answers):
        for answer in answers:
            restored = RESOLVE_RESPONSE_ADAPTER.validate_json(
                RESOLVE_RESPONSE_ADAPTER.dump_json(answer)
            )
            assert type(restored) is type(answer)
            assert restored == answer

    def test_every_dump_names_its_kind(self, answers):
        kinds = [answer.model_dump(mode="json")["kind"] for answer in answers]
        assert kinds == ["resolution", "catalogue", "declined"]

    def test_only_a_resolution_carries_parts_and_a_status(self, answers):
        resolution, catalogue, declined = (a.model_dump(mode="json") for a in answers)
        assert resolution["status"] == "definite"
        assert len(resolution["parts"]) == 1
        for body in (catalogue, declined):
            assert "parts" not in body
            assert "status" not in body

    def test_a_declined_answer_keeps_its_reason_and_confirmed_place(self):
        restored = RESOLVE_RESPONSE_ADAPTER.validate_python(_declined().model_dump(mode="json"))
        assert isinstance(restored, DeclinedAnswer)
        assert restored.reason == "requires_analysis"
        assert restored.scope.place == _KENYA

    def test_each_kind_validates_against_the_published_schema(self, answers):
        schema = RESOLVE_RESPONSE_ADAPTER.json_schema(mode="serialization")
        for answer in answers:
            jsonschema.validate(instance=answer.model_dump(mode="json"), schema=schema)


class TestUnionRefusals:
    def test_a_body_without_kind_refuses(self, worked_example_response: dict):
        body = dict(worked_example_response)
        del body["kind"]
        with pytest.raises(ValidationError):
            RESOLVE_RESPONSE_ADAPTER.validate_python(body)

    def test_an_unknown_kind_refuses(self):
        with pytest.raises(ValidationError):
            RESOLVE_RESPONSE_ADAPTER.validate_python(base_response(kind="refusal"))

    def test_a_catalogue_answer_refuses_parts(self, worked_example_response: dict):
        body = base_response(kind="catalogue", scope={}, parts=worked_example_response["parts"])
        with pytest.raises(ValidationError):
            RESOLVE_RESPONSE_ADAPTER.validate_python(body)

    def test_a_declined_answer_refuses_an_unknown_reason(self):
        with pytest.raises(ValidationError):
            RESOLVE_RESPONSE_ADAPTER.validate_python(
                base_response(kind="declined", reason="off_topic", scope={})
            )

    def test_a_resolution_still_needs_a_part(self):
        with pytest.raises(ValidationError):
            RESOLVE_RESPONSE_ADAPTER.validate_python(base_response(parts=[]))


class TestPartOperations:
    @pytest.mark.parametrize(
        "part_model", [DefinitePart, CandidatesPart, NoDataPart, UnsupportedPart]
    )
    def test_every_part_status_defaults_to_no_operations(self, part_model):
        assert part_model.model_fields["operations"].get_default(call_default_factory=True) == []

    def test_a_validated_part_without_operations_carries_none(self, worked_example_response):
        resolution = _resolution(worked_example_response)
        assert resolution.parts[0].operations == []
        no_data = NoDataPart.model_validate(
            {"status": "no_data", "no_data": {"reason": "no_observations"}}
        )
        assert no_data.operations == []

    def test_a_part_keeps_its_operations_through_a_round_trip(self, worked_example_response):
        body = dict(worked_example_response)
        body["parts"] = [
            {
                **body["parts"][0],
                "operations": [{"verb": "rank", "phrase": "top 5 donors", "over": "origin"}],
            }
        ]
        restored = RESOLVE_RESPONSE_ADAPTER.validate_python(
            RESOLVE_RESPONSE_ADAPTER.dump_python(Resolution.model_validate(body), mode="json")
        )
        assert isinstance(restored, Resolution)
        assert restored.parts[0].operations == [
            Operation(verb="rank", phrase="top 5 donors", over="origin")
        ]

    def test_an_operation_without_an_axis_is_valid(self):
        assert Operation(verb="explain", phrase="Why did fall").over is None

    @pytest.mark.parametrize(
        "payload",
        [
            {"verb": "sort", "phrase": "top 5"},
            {"verb": "rank", "phrase": ""},
            {"verb": "rank", "phrase": "top 5", "over": "donor"},
            {"verb": "rank", "phrase": "top 5", "note": "extra"},
        ],
    )
    def test_a_malformed_operation_refuses(self, payload):
        with pytest.raises(ValidationError):
            Operation.model_validate(payload)


class TestNoDataQuery:
    def test_data_query_defaults_to_none(self):
        assert NoData(reason="no_observations").data_query is None

    def test_data_query_carries_a_parsed_input(self):
        query = ParsedInput(
            parts=[
                ParsedPart(
                    variable_text="ODA",
                    operations=[Operation(verb="rank", phrase="top 5", over="origin")],
                )
            ]
        )
        no_data = NoData.model_validate(
            {"reason": "entity_ambiguous", "data_query": query.model_dump(mode="json")}
        )
        assert no_data.data_query == query


class TestNoDataNearMatch:
    @pytest.mark.parametrize(
        ("reason", "accepted"),
        [
            ("role_incompatible", True),
            ("qualifier_not_bound", True),
            ("denominator_not_available", True),
            ("no_observations", False),
        ],
    )
    def test_a_flagged_match_without_a_gate(self, reason, accepted):
        """Refusals that judge the elected shape itself flag it without a gate."""
        flagged = Spec.model_validate(minimal_spec("s1"))
        if accepted:
            assert NoData(reason=reason, flagged_match=flagged).gate is None
        else:
            with pytest.raises(ValidationError):
                NoData(reason=reason, flagged_match=flagged)

    @pytest.mark.parametrize(
        ("reason", "accepted"),
        [("denominator_not_available", True), ("no_observations", False)],
    )
    def test_denominator_belongs_to_denominator_not_available(self, reason, accepted):
        denominator = NoDataDenominator(found=_KENYA)
        if accepted:
            assert NoData(reason=reason, denominator=denominator).denominator == denominator
        else:
            with pytest.raises(ValidationError):
                NoData(reason=reason, denominator=denominator)
