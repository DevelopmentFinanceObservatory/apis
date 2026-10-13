"""Tests validation for `DatasetResubmitInput` and its four entry variants."""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from search.engine.codes import WARNING_CODES
from search.models import (
    _RESUBMIT_MAX_CANDIDATES,
    DatasetResubmitInput,
    EntityGroup,
    EntryPath,
    ResolveRequest,
    ResubmitCandidate,
    ResubmitCandidatesEntry,
    ResubmitIdentityEntry,
    ResubmitParsedEntry,
    ResubmitPlaceholderEntry,
)


def _identity(population_type: str = "df-flow") -> dict:
    return {
        "shape": {"kind": "statvar_schema", "population_type": population_type},
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
    }


def _identity_entry(**overrides) -> dict:
    entry = {"entry_kind": "identity", "variable_text": "ODA", "identity": _identity()}
    entry.update(overrides)
    return entry


def _candidates_entry(*, n_candidates: int = 2, **overrides) -> dict:
    entry = {
        "entry_kind": "candidates",
        "variable_text": "ODA",
        "ordering": "broadest_first",
        "truncated": False,
        "candidates": [{"identity": _identity()} for _ in range(n_candidates)],
    }
    entry.update(overrides)
    return entry


def _placeholder_entry(**overrides) -> dict:
    entry = {"entry_kind": "placeholder", "variable_text": "ODA", "status": "no_data"}
    entry.update(overrides)
    return entry


def _parsed_entry(**overrides) -> dict:
    entry = {"entry_kind": "parsed", "part": {"variable_text": "GDP"}}
    entry.update(overrides)
    return entry


class TestDatasetResubmitDiscrimination:
    def test_dataset_resubmit_arm(self):
        req = ResolveRequest.model_validate(
            {"input": {"kind": "dataset_resubmit", "entries": [_identity_entry()]}}
        )
        assert isinstance(req.input, DatasetResubmitInput)
        assert req.input.entries[0].identity.shape.population_type == "df-flow"

    def test_single_entry_allowed(self):
        model = DatasetResubmitInput.model_validate({"entries": [_identity_entry()]})
        assert len(model.entries) == 1

    def test_raw_query_echo_only(self):
        model = DatasetResubmitInput.model_validate(
            {"entries": [_identity_entry()], "raw_query": "ODA and FDI"}
        )
        assert model.raw_query == "ODA and FDI"

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            DatasetResubmitInput.model_validate({"entries": [_identity_entry()], "bogus": True})

    def test_zero_entries_rejected(self):
        with pytest.raises(ValidationError):
            DatasetResubmitInput.model_validate({"entries": []})

    def test_entries_beyond_max_parts_rejected(self):
        from search.models import _PARSED_MAX_PARTS

        with pytest.raises(ValidationError):
            DatasetResubmitInput.model_validate(
                {"entries": [_identity_entry() for _ in range(_PARSED_MAX_PARTS + 1)]}
            )

    def test_unknown_entry_kind_rejected(self):
        with pytest.raises(ValidationError):
            DatasetResubmitInput.model_validate(
                {"entries": [{"entry_kind": "bogus", "variable_text": "ODA"}]}
            )

    def test_parsed_entry_allowed(self):
        model = DatasetResubmitInput.model_validate({"entries": [_parsed_entry()]})
        assert isinstance(model.entries[0], ResubmitParsedEntry)
        assert model.entries[0].part.variable_text == "GDP"

    def test_entry_path_is_the_three_served_doors(self):
        """The entry-path vocabulary matches the three doors dispatched by POST `/resolve`."""
        assert set(get_args(EntryPath)) == {
            "raw_text",
            "parsed",
            "dataset_resubmit",
        }


_AFRICA = {"ref": "africa", "label": "Africa", "entity_type": "Continent"}


class TestEntityGroup:
    def test_a_node_s_type_round_trips_and_a_typeless_group_serves_no_key(self):
        typed = EntityGroup.model_validate(_AFRICA)
        typeless = EntityGroup(label="Baltic states")

        assert typed.model_dump(mode="json", exclude_none=True) == _AFRICA
        assert typeless.entity_type is None
        assert typeless.model_dump(mode="json", exclude_none=True) == {"label": "Baltic states"}

    def test_the_schema_bounds_the_type_as_the_dcid_and_requires_neither(self):
        schema = EntityGroup.model_json_schema()

        properties = schema["properties"]
        assert properties["entity_type"]["anyOf"] == properties["ref"]["anyOf"]
        assert properties["entity_type"]["default"] is None
        assert schema["required"] == ["label"]
        with pytest.raises(ValidationError):
            EntityGroup(label="Africa", entity_type="C" * 1025)

    @pytest.mark.parametrize(
        ("model", "document"),
        [
            (ResubmitIdentityEntry, _identity_entry()),
            (ResubmitCandidate, {"identity": _identity()}),
        ],
    )
    def test_a_resubmit_accepts_a_posted_group_that_carries_its_type(self, model, document):
        posted = model.model_validate({**document, "entity_groups": {"origin": _AFRICA}})

        assert posted.entity_groups == {"origin": EntityGroup.model_validate(_AFRICA)}


class TestResubmitIdentityEntry:
    @pytest.mark.parametrize(
        "model, document",
        [
            (ResubmitIdentityEntry, _identity_entry()),
            (ResubmitCandidatesEntry, _candidates_entry()),
            (
                ResubmitPlaceholderEntry,
                _placeholder_entry(reason="no_observations"),
            ),
        ],
    )
    def test_comparison_marker_is_false_elided_and_round_trips_when_true(self, model, document):
        entry = model.model_validate(document)

        assert entry.separate is False
        assert "separate" not in entry.model_dump(mode="json")
        assert (
            model.model_validate({**document, "separate": True}).model_dump(mode="json")["separate"]
            is True
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("base_spec_id", "abc123"),
            ("base_identity", _identity()),
            ("removed", {"facets": ["provenance"]}),
            ("resolved_stat_var_dcids", ["dc/statvar/1"]),
        ],
    )
    def test_a_removed_field_is_refused(self, field, value):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ResubmitIdentityEntry.model_validate(_identity_entry(**{field: value}))

    def test_requires_variable_text(self):
        entry = _identity_entry()
        del entry["variable_text"]
        with pytest.raises(ValidationError):
            ResubmitIdentityEntry.model_validate(entry)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ResubmitIdentityEntry.model_validate(_identity_entry(bogus=True))


class TestResubmitCandidatesEntry:
    def test_validates(self):
        entry = ResubmitCandidatesEntry.model_validate(_candidates_entry())
        assert len(entry.candidates) == 2
        assert entry.ordering == "broadest_first"

    def test_a_candidate_refuses_posted_members(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ResubmitCandidatesEntry.model_validate(
                _candidates_entry(
                    candidates=[
                        {"identity": _identity(), "resolved_stat_var_dcids": ["dc/statvar/1"]},
                        {"identity": _identity()},
                    ]
                )
            )

    def test_one_candidate_rejected(self):
        with pytest.raises(ValidationError):
            ResubmitCandidatesEntry.model_validate(_candidates_entry(n_candidates=1))

    def test_candidates_beyond_ceiling_rejected(self):
        with pytest.raises(ValidationError):
            ResubmitCandidatesEntry.model_validate(
                _candidates_entry(n_candidates=_RESUBMIT_MAX_CANDIDATES + 1)
            )

    def test_ordering_rejects_unknown_value(self):
        with pytest.raises(ValidationError):
            ResubmitCandidatesEntry.model_validate(_candidates_entry(ordering="alphabetical"))

    def test_ordering_does_not_gain_a_value(self):
        """The request accepts exactly the two ordering values."""
        from search.models import Ordering

        assert set(get_args(Ordering)) == {"broadest_first", "intent_rank"}

    def test_no_per_candidate_base_field(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ResubmitCandidate.model_validate({"identity": _identity(), "base_spec_id": "abc123"})

    def test_no_max_candidates_field(self):
        """The model rejects `max_candidates`; the engine applies its deployment-specific ceiling."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ResubmitCandidatesEntry.model_validate(_candidates_entry(max_candidates=6))


class TestResubmitPlaceholderEntry:
    def test_no_data_requires_reason(self):
        with pytest.raises(ValidationError, match="reason is required"):
            ResubmitPlaceholderEntry.model_validate(_placeholder_entry(status="no_data"))

    def test_no_data_with_reason_validates(self):
        entry = ResubmitPlaceholderEntry.model_validate(
            _placeholder_entry(status="no_data", reason="no_observations")
        )
        assert entry.reason == "no_observations"
        assert entry.code is None

    def test_no_data_forbids_code(self):
        with pytest.raises(ValidationError, match="code must not be posted"):
            ResubmitPlaceholderEntry.model_validate(
                _placeholder_entry(
                    status="no_data", reason="no_observations", code="facet_binding_unsupported"
                )
            )

    def test_unsupported_requires_code(self):
        with pytest.raises(ValidationError, match="code is required"):
            ResubmitPlaceholderEntry.model_validate(_placeholder_entry(status="unsupported"))

    def test_unsupported_with_code_validates(self):
        entry = ResubmitPlaceholderEntry.model_validate(
            _placeholder_entry(status="unsupported", code="facet_binding_unsupported")
        )
        assert entry.code == "facet_binding_unsupported"
        assert entry.reason is None

    def test_unsupported_forbids_reason(self):
        with pytest.raises(ValidationError, match="reason must not be posted"):
            ResubmitPlaceholderEntry.model_validate(
                _placeholder_entry(status="unsupported", code="x", reason="no_observations")
            )


class TestResubmitParsedEntry:
    def test_validates_bare_part(self):
        entry = ResubmitParsedEntry.model_validate(_parsed_entry())
        assert entry.part.variable_text == "GDP"
        assert entry.part.entities is None

    def test_carries_full_parsed_part(self):
        entry = ResubmitParsedEntry.model_validate(
            _parsed_entry(
                part={
                    "variable_text": "GDP",
                    "entities": [{"text": "Brazil"}],
                    "time_text": "2020",
                    "unit_text": "current US dollars",
                    "source_text": "IMF",
                }
            )
        )
        assert entry.part.entities[0].text == "Brazil"
        assert entry.part.time_text == "2020"
        assert entry.part.unit_text == "current US dollars"
        assert entry.part.source_text == "IMF"

    def test_no_top_level_variable_text_field(self):
        """`variable_text` belongs to the nested parsed part."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ResubmitParsedEntry.model_validate(
                {"entry_kind": "parsed", "variable_text": "GDP", "part": {"variable_text": "GDP"}}
            )

    def test_requires_part(self):
        with pytest.raises(ValidationError):
            ResubmitParsedEntry.model_validate({"entry_kind": "parsed"})

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ResubmitParsedEntry.model_validate(_parsed_entry(bogus=True))

    def test_part_own_validation_applies(self):
        """Nested `ParsedPart` validation applies to the entry."""
        with pytest.raises(ValidationError):
            ResubmitParsedEntry.model_validate(_parsed_entry(part={"variable_text": ""}))

    def test_dataset_resubmit_accepts_parsed_entry_alongside_others(self):
        model = DatasetResubmitInput.model_validate(
            {"entries": [_identity_entry(), _parsed_entry()]}
        )
        assert isinstance(model.entries[0], ResubmitIdentityEntry)
        assert isinstance(model.entries[1], ResubmitParsedEntry)


class TestPartRestatedWarningCode:
    def test_registered(self):
        assert "part_restated" in WARNING_CODES
