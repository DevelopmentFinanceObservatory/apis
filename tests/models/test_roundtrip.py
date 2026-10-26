"""Round-trip and discrimination tests for the contract response and request models."""

import pytest
from pydantic import ValidationError

from search import (
    RESOLVE_RESPONSE_ADAPTER,
    CandidatesPart,
    DatasetResubmitInput,
    DefinitePart,
    NoDataPart,
    ParsedInput,
    RawTextInput,
    Resolution,
    ResolveRequest,
    ResubmitIdentityPayload,
    Spec,
)
from search.models import BaseIdentity, IdentityBindingExact, IdentityBindingValue
from search.scope.build import served_scope
from search.scope.compile import Compiled, compile
from tests.conftest import base_response, minimal_spec
from tests.scope import _axes


class TestSlotExpansionRoundtrip:
    """`expansion` is an optional field that round-trips when present."""

    def test_slot_expansion_roundtrip(self):
        payload = minimal_spec()
        payload["resolved"]["slots"] = [
            {
                "key": {
                    "axis": "how",
                    "property": {"ref": "crsPurpose", "label": "purpose"},
                    "label": "purpose",
                },
                "binding": {
                    "kind": "value",
                    "value": {
                        "ref": {"ref": "CRSPurposeCode120", "label": "health"},
                        "value_kind": "enum_value",
                    },
                },
                "expansion": {
                    "added": [
                        {"ref": "CRSPurposeCode12262", "label": "malaria control"},
                    ]
                },
            }
        ]

        spec = Spec.model_validate(payload)
        slot = spec.resolved.slots[0]
        assert slot.expansion is not None
        assert [ref.ref for ref in slot.expansion.added] == ["CRSPurposeCode12262"]

        dumped = spec.model_dump(mode="json")
        assert dumped["resolved"]["slots"][0]["expansion"]["added"][0]["ref"] == (
            "CRSPurposeCode12262"
        )
        assert Spec.model_validate(dumped).model_dump(mode="json") == dumped

    def test_expansion_forbids_extra_fields(self):
        payload = minimal_spec()
        payload["resolved"]["slots"] = [
            {
                "key": {"axis": "how", "property": None, "label": "purpose"},
                "binding": {"kind": "unbound"},
                "expansion": {
                    "added": [{"ref": "x", "label": "x"}],
                    "unexpected": True,
                },
            }
        ]
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            Spec.model_validate(payload)

    def test_expansion_rejects_an_empty_added_list(self):
        payload = minimal_spec()
        payload["resolved"]["slots"] = [
            {
                "key": {"axis": "how", "property": None, "label": "purpose"},
                "binding": {"kind": "unbound"},
                "expansion": {"added": []},
            }
        ]
        with pytest.raises(ValidationError):
            Spec.model_validate(payload)


def _definite_part(spec_id: str = "s1") -> dict:
    return {"status": "definite", "spec": minimal_spec(spec_id)}


class TestDefinitePartRoundtrip:
    def test_definite_part_validates(self):
        payload = base_response(parts=[_definite_part()])
        resp = Resolution.model_validate(payload)
        assert resp.status == "definite"
        assert len(resp.parts) >= 1
        part = resp.parts[0]
        assert isinstance(part, DefinitePart)
        assert isinstance(part.spec, Spec)

    def test_roundtrip_stable(self):
        payload = base_response(parts=[_definite_part()])
        resp = Resolution.model_validate(payload)
        dumped = resp.model_dump(mode="json")
        resp2 = Resolution.model_validate(dumped)
        assert resp2.model_dump(mode="json") == dumped


_YEAR_DEPARTURE = {
    "code": "year_unavailable",
    "binding_ref": "when:",
    "asked": "2030",
    "shown": "1973 to 2024",
    "text": "You asked for 2030. The data ends in 2024, so this shows 1973 to 2024.",
}
_CLOSEST_DEPARTURE = {"code": "closest_match", "text": "You asked for cost of living."}


class TestDepartures:
    """`departures` rides on every part in DepartureCode precedence order."""

    def test_ordered_departures_roundtrip(self):
        part = {
            **_definite_part(),
            "departures": [_CLOSEST_DEPARTURE, _YEAR_DEPARTURE, _YEAR_DEPARTURE],
        }
        resp = Resolution.model_validate(base_response(parts=[part]))
        dumped = resp.model_dump(mode="json")
        assert [d["code"] for d in dumped["parts"][0]["departures"]] == [
            "closest_match",
            "year_unavailable",
            "year_unavailable",
        ]

    @pytest.mark.parametrize(
        "departures",
        [
            pytest.param([_YEAR_DEPARTURE, _CLOSEST_DEPARTURE], id="out-of-precedence-order"),
            pytest.param([{**_YEAR_DEPARTURE, "text": ""}], id="empty-text"),
            pytest.param([{**_YEAR_DEPARTURE, "code": "retrieval_score_weak"}], id="warning-code"),
        ],
    )
    def test_malformed_departures_refuse(self, departures):
        part = {**_definite_part(), "departures": departures}
        with pytest.raises(ValidationError):
            Resolution.model_validate(base_response(parts=[part]))


class TestPartVariants:
    def test_definite_discriminates(self):
        payload = base_response(parts=[_definite_part()])
        resp = Resolution.model_validate(payload)
        assert isinstance(resp.parts[0], DefinitePart)

    def test_candidates_discriminates(self):
        payload = base_response(
            parts=[
                {
                    "status": "candidates",
                    "candidates": {
                        "ordering": "broadest_first",
                        "max_candidates": 2,
                        "specs": [minimal_spec("s1"), minimal_spec("s2")],
                        "truncated": False,
                    },
                }
            ]
        )
        resp = Resolution.model_validate(payload)
        assert isinstance(resp.parts[0], CandidatesPart)

    def test_no_data_discriminates(self):
        payload = base_response(
            parts=[{"status": "no_data", "no_data": {"reason": "no_observations"}}]
        )
        resp = Resolution.model_validate(payload)
        assert isinstance(resp.parts[0], NoDataPart)

    def test_wrong_body_raises(self):
        # status says candidates but the body has a definite shape
        payload = base_response(
            parts=[
                {
                    "status": "candidates",
                    "spec": minimal_spec(),  # wrong field for candidates
                }
            ]
        )
        with pytest.raises(ValidationError):
            Resolution.model_validate(payload)


class TestRequestInputDiscrimination:
    def test_raw_text_arm(self):
        req = ResolveRequest.model_validate(
            {"input": {"kind": "raw_text", "query": "ODA to Ethiopia"}}
        )
        assert isinstance(req.input, RawTextInput)

    def test_parsed_arm(self):
        req = ResolveRequest.model_validate(
            {"input": {"kind": "parsed", "parts": [{"variable_text": "ODA"}]}}
        )
        assert isinstance(req.input, ParsedInput)

    def test_dataset_resubmit_identity_arm(self):
        shape = {"kind": "statvar_schema", "population_type": "df-flow"}
        req = ResolveRequest.model_validate(
            {
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": {
                                "shape": shape,
                                "slots": [],
                                "entities": {},
                                "requested_window": {"kind": "all"},
                            },
                        }
                    ],
                }
            }
        )
        assert isinstance(req.input, DatasetResubmitInput)
        assert req.input.entries[0].identity.shape.population_type == "df-flow"

    @pytest.mark.parametrize("legacy_field", ["shape", "slots", "facets", "origin_dcids"])
    def test_resubmit_identity_payload_rejects_legacy_flat_definition_fields(
        self, legacy_field: str
    ):
        value = [] if legacy_field == "slots" else {"kind": "statvar_schema"}
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ResubmitIdentityPayload.model_validate(
                {
                    "identity": {
                        "shape": {"kind": "statvar_schema", "population_type": "df-flow"},
                        "slots": [],
                        "entities": {},
                        "requested_window": {"kind": "all"},
                    },
                    legacy_field: value,
                }
            )

    def test_unknown_kind_raises(self):
        with pytest.raises(ValidationError):
            ResolveRequest.model_validate({"input": {"kind": "nonexistent", "query": "test"}})


class TestIdentityBindingExactRoundtrip:
    """Checks wire round trips for exact bindings, a distinct fourth `IdentityBinding` state."""

    def _identity_with_binding(self, binding: dict) -> dict:
        return {
            "shape": {"kind": "statvar_schema", "population_type": "df-flow"},
            "slots": [{"property": "crsPurpose", "binding": binding}],
            "entities": {},
            "requested_window": {"kind": "all"},
        }

    def test_exact_slot_binding_round_trips(self):
        identity = BaseIdentity.model_validate(
            self._identity_with_binding({"state": "exact", "ref": "CRSPurposeCode100"})
        )

        binding = identity.slots[0].binding
        assert isinstance(binding, IdentityBindingExact)
        assert binding.ref == "CRSPurposeCode100"
        assert identity.model_dump(exclude_none=True)["slots"][0]["binding"] == {
            "state": "exact",
            "ref": "CRSPurposeCode100",
        }

    def test_exact_and_value_on_the_same_ref_are_distinct_binding_types(self):
        """`exact` and `value` remain distinct states for the same ref."""
        exact = (
            BaseIdentity.model_validate(
                self._identity_with_binding({"state": "exact", "ref": "CRSPurposeCode110"})
            )
            .slots[0]
            .binding
        )
        value = (
            BaseIdentity.model_validate(
                self._identity_with_binding({"state": "value", "ref": "CRSPurposeCode110"})
            )
            .slots[0]
            .binding
        )

        assert isinstance(exact, IdentityBindingExact)
        assert isinstance(value, IdentityBindingValue)
        assert exact != value

    def test_exact_binding_with_refs_instead_of_ref_rejected(self):
        with pytest.raises(ValidationError):
            BaseIdentity.model_validate(
                self._identity_with_binding({"state": "exact", "refs": ["CRSPurposeCode100"]})
            )

    def test_exact_binding_missing_ref_rejected(self):
        with pytest.raises(ValidationError):
            BaseIdentity.model_validate(self._identity_with_binding({"state": "exact"}))


def _scoped_spec(scope: dict | None) -> dict:
    payload = minimal_spec()
    if scope is not None:
        payload["resolved"]["scope_choices"] = scope
    return payload


def _part_scope() -> dict:
    """A scope over the synthetic tables of `tests.scope._axes`, one axis asked for and a carried entry dropped."""
    compiled = compile(
        _axes.BASE,
        (_axes.explicit(_axes.GRANT_EQUIVALENT), _axes.carried(_axes.REGION_ALL)),
        _axes.facts(),
    )
    assert isinstance(compiled, Compiled)
    return {
        "id": compiled.base_spec_id,
        "base": {
            "spec_id": compiled.base_spec_id,
            "identity": _axes.BASE.model_dump(mode="json", exclude_none=True),
        },
        "rule_version": "scope1:test",
        "entries": [e.model_dump(mode="json") for e in compiled.entries],
        "dropped_entries": [d.model_dump(mode="json") for d in compiled.dropped],
        "choices": [c.model_dump(mode="json") for c in compiled.choices],
        "compiled": served_scope(compiled.spec_id, compiled.identity).model_dump(mode="json"),
    }


class TestScopeChoicesRoundtrip:
    """`resolved.scope_choices` round-trips through the wire form and the response cache's
    stored JSON, and is absent from the wire when the part settled no scope."""

    def test_a_spec_with_scope_choices_round_trips(self):
        scope = _part_scope()
        resp = Resolution.model_validate(
            base_response(parts=[{"status": "definite", "spec": _scoped_spec(scope)}])
        )

        stored = RESOLVE_RESPONSE_ADAPTER.validate_json(resp.model_dump_json())
        wire = resp.model_dump(mode="json", exclude_none=True)

        assert stored == resp
        assert Resolution.model_validate(wire) == resp
        restored = stored.parts[0].spec.resolved.scope_choices
        assert restored is not None
        assert [c.axis for c in restored.choices] == ["sector", "region", "measure_basis"]
        assert [d.cause for d in restored.dropped_entries] == ["conflicts_with_entry"]

    def test_a_spec_without_scope_choices_leaves_the_member_off_the_wire(self):
        resp = Resolution.model_validate(
            base_response(parts=[{"status": "definite", "spec": _scoped_spec(None)}])
        )

        wire = resp.model_dump(mode="json", exclude_none=True)

        assert "scope_choices" not in wire["parts"][0]["spec"]["resolved"]
        assert RESOLVE_RESPONSE_ADAPTER.validate_json(resp.model_dump_json()) == resp
