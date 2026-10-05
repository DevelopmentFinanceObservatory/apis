"""Tests observed basis periods and `difference` and `percent_change` derivations on stat cards."""

from __future__ import annotations

import jsonschema
import pytest

from assembler.errors import BindingInvalid, ConstraintViolation
from assembler.mint.validate import component_registry
from assembler.pipeline import write_component
from assembler.ports import FrameRef
from assembler.testing import build_resolved_frame

from ._pipeline_support import _stat_card_intent


def _zero_from_value_frame():
    """A zero `from` value makes `percent_change` undefined."""
    return build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "6" * 32, content_version="cv_" + "7" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": "country/ETH",
                "time_period": period,
                "value": value,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            }
            for period, value in {"2020": 0.0, "2021": 10.0}.items()
        ],
    )


def _rising_negative_from_value_frame():
    """A negative `from` value would reverse the sign of this rise, so `percent_change` refuses."""
    return build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "2" * 32, content_version="cv_" + "3" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": "country/ETH",
                "time_period": period,
                "value": value,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            }
            for period, value in {"2020": -100.0, "2021": -50.0}.items()
        ],
    )


def _positive_from_value_frame():
    """A positive `from` value permits both positive and negative `to` values."""
    return build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "4" * 32, content_version="cv_" + "5" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": "country/ETH",
                "time_period": period,
                "value": value,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            }
            for period, value in {"2020": 100.0, "2021": 50.0, "2022": -50.0}.items()
        ],
    )


def _write(intent, *, frame_store, revision_store, release_registry):
    return write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )


def test_change_over_an_unobserved_basis_period_refuses_at_change_basis(
    frame_store, revision_store, release_registry
):
    frame, intent = _stat_card_intent(change={"basis": ["2018", "2024"]})
    frame_store.put(resolved_frame=frame)

    with pytest.raises(ConstraintViolation) as excinfo:
        _write(
            intent,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.constraint_id == "stat_change_basis_observed"
    assert excinfo.value.pointer == "/change/basis"


def test_change_over_two_observed_periods_resolves_under_difference(
    frame_store, revision_store, release_registry
):
    """An omitted derivation defaults to `difference` over the observed 2023 and 2024 points."""
    frame, intent = _stat_card_intent(change={"basis": ["2023", "2024"]})
    frame_store.put(resolved_frame=frame)

    created = _write(
        intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    record = revision_store.read_revision(component_id=created.component_id, revision=1)
    change = record.resolved_payload["component"]["change"]
    assert change["derivation"] == "difference"
    assert change["basis"] == ["2023", "2024"]
    assert change["value"] == pytest.approx(441.9 - 433.7)


def test_change_over_two_observed_periods_resolves_under_percent_change(
    frame_store, revision_store, release_registry
):
    """The literal `1.8907078625778162` catches swapping the `percent_change` divisor."""
    frame, intent = _stat_card_intent(
        change={"basis": ["2023", "2024"], "derivation": "percent_change"}
    )
    frame_store.put(resolved_frame=frame)

    created = _write(
        intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    record = revision_store.read_revision(component_id=created.component_id, revision=1)
    change = record.resolved_payload["component"]["change"]
    assert change["derivation"] == "percent_change"
    assert change["basis"] == ["2023", "2024"]
    assert change["value"] == pytest.approx(1.8907078625778162)


def _negative_from_value_frame():
    """A single-series frame whose earlier basis period holds a negative
    value, distinct from `malaria_ethiopia_resolved_frame`'s all-positive
    observation set, so `difference` is exercised against a negative
    basis value it does not refuse."""
    return build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "8" * 32, content_version="cv_" + "9" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": "country/ETH",
                "time_period": period,
                "value": value,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            }
            for period, value in {"2020": -50.0, "2021": 30.0}.items()
        ],
    )


def test_difference_over_a_negative_from_value_still_computes(
    frame_store, revision_store, release_registry
):
    """`difference` returns 80.0 for -50.0 to 30.0."""
    frame = _negative_from_value_frame()
    frame_store.put(resolved_frame=frame)
    intent = {
        "type": "stat_card",
        "frame_ref": {
            "spec_id": frame.frame_ref.spec_id,
            "content_version": frame.frame_ref.content_version,
        },
        "select": {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/ETH"},
        "change": {"basis": ["2020", "2021"], "derivation": "difference"},
    }

    created = _write(
        intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    record = revision_store.read_revision(component_id=created.component_id, revision=1)
    change = record.resolved_payload["component"]["change"]
    assert change["derivation"] == "difference"
    assert change["basis"] == ["2020", "2021"]
    assert change["value"] == pytest.approx(80.0)


def test_percent_change_over_a_zero_from_value_refuses_binding_invalid(
    frame_store, revision_store, release_registry
):
    frame = _zero_from_value_frame()
    frame_store.put(resolved_frame=frame)
    intent = {
        "type": "stat_card",
        "frame_ref": {
            "spec_id": frame.frame_ref.spec_id,
            "content_version": frame.frame_ref.content_version,
        },
        "select": {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/ETH"},
        "change": {"basis": ["2020", "2021"], "derivation": "percent_change"},
    }

    with pytest.raises(BindingInvalid) as excinfo:
        _write(
            intent,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/change"


def test_percent_change_over_a_negative_from_value_refuses_binding_invalid(
    frame_store, revision_store, release_registry
):
    """A negative `from` value reverses this rise's sign, so `percent_change` refuses."""
    frame = _rising_negative_from_value_frame()
    frame_store.put(resolved_frame=frame)
    intent = {
        "type": "stat_card",
        "frame_ref": {
            "spec_id": frame.frame_ref.spec_id,
            "content_version": frame.frame_ref.content_version,
        },
        "select": {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/ETH"},
        "change": {"basis": ["2020", "2021"], "derivation": "percent_change"},
    }

    with pytest.raises(BindingInvalid) as excinfo:
        _write(
            intent,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/change"


@pytest.mark.parametrize(
    ("basis", "expected_value"),
    [
        pytest.param(["2020", "2021"], -50.0, id="smaller_to_value"),
        pytest.param(["2020", "2022"], -150.0, id="negative_to_value"),
    ],
)
def test_percent_change_over_a_positive_from_value_computes(
    basis, expected_value, frame_store, revision_store, release_registry
):
    """A positive `from` value permits both smaller-positive and negative `to` values."""
    frame = _positive_from_value_frame()
    frame_store.put(resolved_frame=frame)
    intent = {
        "type": "stat_card",
        "frame_ref": {
            "spec_id": frame.frame_ref.spec_id,
            "content_version": frame.frame_ref.content_version,
        },
        "select": {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/ETH"},
        "change": {"basis": basis, "derivation": "percent_change"},
    }

    created = _write(
        intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    record = revision_store.read_revision(component_id=created.component_id, revision=1)
    change = record.resolved_payload["component"]["change"]
    assert change["derivation"] == "percent_change"
    assert change["basis"] == basis
    assert change["value"] == pytest.approx(expected_value)


def test_change_basis_outside_the_window_refuses_at_change_basis(
    frame_store, revision_store, release_registry
):
    """`malaria_ethiopia_rows` observes 2016 (455.1); a `window` of
    2020..2024 excludes it from the bound primary series' own points, so
    a basis naming 2016 has no observed point to validate against."""
    frame, intent = _stat_card_intent(
        window={"start": "2020", "end": "2024"}, change={"basis": ["2016", "2024"]}
    )
    frame_store.put(resolved_frame=frame)

    with pytest.raises(ConstraintViolation) as excinfo:
        _write(
            intent,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.constraint_id == "stat_change_basis_observed"
    assert excinfo.value.pointer == "/change/basis"


def test_change_basis_inside_the_window_computes_from_the_windowed_points(
    frame_store, revision_store, release_registry
):
    """A `window` of 2020..2024 binds the primary series to its own 2020
    and 2024 points; a basis naming both mints a change computed from
    them."""
    frame, intent = _stat_card_intent(
        window={"start": "2020", "end": "2024"}, change={"basis": ["2020", "2024"]}
    )
    frame_store.put(resolved_frame=frame)

    created = _write(
        intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    record = revision_store.read_revision(component_id=created.component_id, revision=1)
    change = record.resolved_payload["component"]["change"]
    assert change["basis"] == ["2020", "2024"]
    assert change["value"] == pytest.approx(441.9 - 398.2)


def test_intent_without_change_mints_without_a_change_key(
    frame_store, revision_store, release_registry
):
    frame, intent = _stat_card_intent()
    frame_store.put(resolved_frame=frame)

    created = _write(
        intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    record = revision_store.read_revision(component_id=created.component_id, revision=1)
    assert "change" not in record.resolved_payload["component"]


def test_minted_stat_card_change_validates_against_the_pinned_schema(
    frame_store, revision_store, release_registry
):
    frame, intent = _stat_card_intent(
        change={"basis": ["2023", "2024"], "derivation": "percent_change"}
    )
    frame_store.put(resolved_frame=frame)

    created = _write(
        intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    record = revision_store.read_revision(component_id=created.component_id, revision=1)
    release = release_registry.pinned()
    registry = component_registry(release)
    jsonschema.Draft202012Validator(release.envelope_schema, registry=registry).validate(
        record.resolved_payload
    )
    jsonschema.Draft202012Validator(release.component_schemas["stat_card"]).validate(
        record.resolved_payload["component"]
    )
