"""Check that `write_component` refusal pointers name fields in the submitted request.

A pointer must resolve, or its parent must resolve and its final segment must name an addable
field. This catches pointers into normalized selectors such as `/select/0/bindings/0/...`.
`describe_frame`'s `/dimension` pointer is outside scope because it has no submitted intent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from assembler.errors import BindingInvalid, ConstraintViolation, ResourceExhausted, SchemaInvalid
from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore
from assembler.pipeline import write_component
from assembler.pipeline.candidate import _translate_pointer
from assembler.ports import FrameRef
from assembler.testing import build_resolved_frame, malaria_ethiopia_resolved_frame


def _resolve_pointer(document: object, pointer: str) -> object:
    if pointer == "":
        return document
    node = document
    for raw_token in pointer.lstrip("/").split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, Mapping):
            node = node[token]
        elif isinstance(node, Sequence) and not isinstance(node, str):
            node = node[int(token)]
        else:
            raise KeyError(token)
    return node


def _pointer_names_a_real_field(document: Mapping[str, object], pointer: str) -> bool:
    """Accept a resolving pointer or an addable field under a resolving parent."""
    try:
        _resolve_pointer(document, pointer)
    except (KeyError, IndexError, ValueError):
        pass
    else:
        return True
    parent_pointer, _, terminal = pointer.rpartition("/")
    try:
        parent = _resolve_pointer(document, parent_pointer)
    except (KeyError, IndexError, ValueError):
        return False
    return bool(terminal) and not terminal.isdigit() and isinstance(parent, Mapping)


@pytest.fixture
def frame_store():
    return FakeFrameStore()


@pytest.fixture
def revision_store():
    return FakeRevisionStore()


@pytest.fixture
def release_registry(release):
    return FakeReleaseRegistry(release=release)


def test_unknown_dimension_pointer_resolves(frame_store, revision_store, release_registry):
    """Unknown dimensions use a pointer to their submitted select key."""
    rf = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "a" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2020",
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            }
        ],
    )
    frame_store.put(resolved_frame=rf)
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": rf.frame_ref.spec_id,
            "content_version": rf.frame_ref.content_version,
        },
        "select": {"bogus": "x"},
    }

    with pytest.raises(BindingInvalid) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/select/bogus"
    assert _pointer_names_a_real_field(intent, excinfo.value.pointer)


def test_unknown_value_pointer_resolves(frame_store, revision_store, release_registry):
    """Unknown values point to their submitted select key."""
    rf = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "b" * 32, content_version="cv_" + "b" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2020",
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            }
        ],
    )
    frame_store.put(resolved_frame=rf)
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": rf.frame_ref.spec_id,
            "content_version": rf.frame_ref.content_version,
        },
        "select": {"recipient": "country/ZZZ"},
    }

    with pytest.raises(BindingInvalid) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/select/recipient"
    assert _pointer_names_a_real_field(intent, excinfo.value.pointer)


def test_facet_split_pointer_resolves(frame_store, revision_store, release_registry):
    """`provenance` splits one series and is absent from `select`, so its pointer names an addable
    key under `/select`."""
    rf = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "c" * 32, content_version="cv_" + "c" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2020",
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            },
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2021",
                "value": 2.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_b",
            },
        ],
    )
    frame_store.put(resolved_frame=rf)
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": rf.frame_ref.spec_id,
            "content_version": rf.frame_ref.content_version,
        },
        "select": {"recipient": "country/ETH"},
    }

    with pytest.raises(BindingInvalid) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/select/provenance"
    assert "provenance" not in intent["select"]
    assert _pointer_names_a_real_field(intent, excinfo.value.pointer)


def test_empty_cross_product_pointer_resolves(frame_store, revision_store, release_registry):
    """An empty cross-product points to the whole `select` map."""
    rf = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "d" * 32, content_version="cv_" + "d" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2020",
                "value": 1.0,
                "unit": None,
                "measurementMethod": "m1",
                "provenance": "src_a",
            },
            {
                "stat_var": "dcid:X",
                "recipient": "country/GHA",
                "time_period": "2020",
                "value": 2.0,
                "unit": None,
                "measurementMethod": "m2",
                "provenance": "src_a",
            },
        ],
    )
    frame_store.put(resolved_frame=rf)
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": rf.frame_ref.spec_id,
            "content_version": rf.frame_ref.content_version,
        },
        "select": {"recipient": "country/ETH", "measurementMethod": "m2"},
    }

    with pytest.raises(BindingInvalid) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/select"
    assert _pointer_names_a_real_field(intent, excinfo.value.pointer)


def test_dimension_name_with_pointer_metacharacters_escapes_into_a_resolvable_pointer(
    frame_store, revision_store, release_registry
):
    """Role keys may contain `/` and `~`; RFC 6901 escaping must preserve the intended path."""
    role_key = "foo/bar~baz"
    rf = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "0" * 32, content_version="cv_" + "0" * 32),
        role_keys=(role_key,),
        rows=[
            {
                "stat_var": "dcid:X",
                role_key: "country/ETH",
                "time_period": "2020",
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            }
        ],
    )
    frame_store.put(resolved_frame=rf)
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": rf.frame_ref.spec_id,
            "content_version": rf.frame_ref.content_version,
        },
        "select": {role_key: "country/ZZZ"},
    }

    with pytest.raises(BindingInvalid) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/select/foo~1bar~0baz"
    assert _pointer_names_a_real_field(intent, excinfo.value.pointer)


def test_schema_invalid_catch_all_pointer_escapes_pointer_metacharacters(
    frame_store, revision_store, release_registry
):
    """`stat_card.select` requires strings, so this array fails schema validation before frame load.

    The role key contains `/` and `~`, which the refusal pointer must escape.
    """
    role_key = "foo/bar~baz"
    intent = {
        "type": "stat_card",
        "frame_ref": {"spec_id": "spec_x", "content_version": "cv_x"},
        "select": {role_key: ["not", "a", "string"]},
    }

    with pytest.raises(SchemaInvalid) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/select/foo~1bar~0baz"
    assert _pointer_names_a_real_field(intent, excinfo.value.pointer)


def test_series_estimate_resource_exhausted_pointer_resolves(
    frame_store, revision_store, release_registry
):
    """The series-count limit raises `ResourceExhausted` at `/select`."""
    rows = [
        {
            "stat_var": "dcid:X",
            "recipient": f"country/{i:03d}",
            "time_period": "2020",
            "value": 1.0,
            "unit": None,
            "measurementMethod": None,
            "provenance": "src_a",
        }
        for i in range(26)
    ]
    rf = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "e" * 32, content_version="cv_" + "e" * 32),
        role_keys=("recipient",),
        rows=rows,
    )
    frame_store.put(resolved_frame=rf)
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": rf.frame_ref.spec_id,
            "content_version": rf.frame_ref.content_version,
        },
        "select": {"stat_var": "dcid:X"},
    }

    with pytest.raises(ResourceExhausted) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/select"
    assert _pointer_names_a_real_field(intent, excinfo.value.pointer)


def test_missing_select_schema_invalid_pointer_resolves(
    frame_store, revision_store, release_registry
):
    """A missing `select` points to the addable `/select` field."""
    frame = malaria_ethiopia_resolved_frame()
    frame_store.put(resolved_frame=frame)
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": frame.frame_ref.spec_id,
            "content_version": frame.frame_ref.content_version,
        },
    }

    with pytest.raises(SchemaInvalid) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/select"
    assert "select" not in intent
    assert _pointer_names_a_real_field(intent, excinfo.value.pointer)


def test_composition_id_binding_invalid_pointer_resolves(
    frame_store, revision_store, release_registry
):
    """`composition_id` is a separate argument, but the refusal pointer resolves against the full
    request containing both it and the revise intent."""
    frame = malaria_ethiopia_resolved_frame()
    frame_store.put(resolved_frame=frame)
    create_intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": frame.frame_ref.spec_id,
            "content_version": frame.frame_ref.content_version,
        },
        "select": {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/ETH"},
        "window": {"start": "2015", "end": "2024"},
    }
    created = write_component(
        intent=create_intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    revise_intent = {"title": "New title"}
    request = {**revise_intent, "composition_id": "cps_x"}
    with pytest.raises(BindingInvalid) as excinfo:
        write_component(
            intent=revise_intent,
            component_id=created.component_id,
            write_token=created.write_token,
            composition_id="cps_x",
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/composition_id"
    assert _pointer_names_a_real_field(request, excinfo.value.pointer)


def test_comparable_units_required_constraint_violation_pointer_resolves(
    frame_store, revision_store, release_registry
):
    """Constraint pointers resolve against the submitted intent after translation."""
    rf = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "f" * 32, content_version="cv_" + "f" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2020",
                "value": 1.0,
                "unit": "unit/USD",
                "measurementMethod": None,
                "provenance": "src_a",
            },
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2020",
                "value": 2.0,
                "unit": "unit/EUR",
                "measurementMethod": None,
                "provenance": "src_a",
            },
        ],
    )
    frame_store.put(resolved_frame=rf)
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": rf.frame_ref.spec_id,
            "content_version": rf.frame_ref.content_version,
        },
        "select": {"stat_var": "dcid:X", "unit": ["unit/USD", "unit/EUR"]},
    }

    with pytest.raises(ConstraintViolation) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.constraint_id == "comparable_units_required"
    assert _pointer_names_a_real_field(intent, excinfo.value.pointer)


def test_translate_pointer_maps_bar_chart_value_axis_like_y_axis():
    """Bar-chart `value_axis` pointers map to intent fields."""
    assert _translate_pointer(component_pointer="/component/value_axis/scale") == "/scale"
    assert _translate_pointer(component_pointer="/component/value_axis/range/min") == ("/range/min")


def test_translate_pointer_maps_bar_chart_category_axis():
    """Bar-chart `category_axis` pointers map to intent fields."""
    assert _translate_pointer(component_pointer="/component/category_axis/window") == "/window"


def test_translate_pointer_keeps_stat_card_change_basis_its_own_name():
    """`change.basis` maps directly to its intent field."""
    assert _translate_pointer(component_pointer="/component/change/basis") == "/change/basis"
