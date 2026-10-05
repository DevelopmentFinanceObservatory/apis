"""End-to-end tests for the slot option taxonomies returned with a resolved spec."""

from __future__ import annotations

from search.engine.region.grounding import _ground_answer
from search.engine.shape import ShapeDraft
from search.models import DefinitePart, RawTextInput, ResolveRequest
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.fixtures import FakeGraph


def _resolve(query: str) -> DefinitePart:
    r = offline_resolve(
        ResolveRequest(input=RawTextInput(query=query)), sdmx=FakeSdmxClient()
    ).parts[0]
    assert isinstance(r, DefinitePart)
    return r


def test_slot_options_key_matches_a_bound_slot():
    """Every SlotOptions.key corresponds to a real Slot in spec.resolved.slots
    (axis + property.dcid), so a client can always resolve the binding a set
    of options belongs to."""
    r = _resolve("fertility rate in Kenya")
    spec = r.spec
    slot_keys = {
        (s.key.axis, s.key.property.dcid if s.key.property else None) for s in spec.resolved.slots
    }
    for so in spec.resolved.slot_options:
        key = (so.key.axis, so.key.property.dcid if so.key.property else None)
        assert key in slot_keys


def test_no_taxonomy_yields_no_slot_options():
    """A shape with no constraint slots returns an empty options list.

    The direct _ground_answer call avoids live taxonomy discovery.
    """
    shape_draft = ShapeDraft(
        shape_id="person_count_measuredvalue",
        label="Total population",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
    )
    # slot_options is tuple index 7. inapplicable_slots and warnings follow at 8 and 9.
    result = _ground_answer(
        shape_draft,
        bindings=[],
        sv_dcids=["Count_Person"],
        roles={},
        graph=FakeGraph(),
    )
    slot_options = result[7]
    assert slot_options == []


def test_total_population_india_now_has_residence_classification_taxonomy():
    """Count_Person exposes its Rural and Urban residence options."""
    r = _resolve("total population India")
    spec = r.spec
    assert [sv.ref for sv in spec.resolved.stat_vars.items] == ["Count_Person"]
    by_prop = {so.key.property.dcid: so for so in spec.resolved.slot_options if so.key.property}
    residence_opts = by_prop["placeOfResidenceClassification"]
    assert residence_opts.key.axis == "how"
    assert residence_opts.source == "observed"
    assert residence_opts.truncated is True
    observed = {o.value.ref.dcid for o in residence_opts.options}
    assert observed and observed <= {"Urban", "Rural"}, (
        f"expected residence siblings drawn from Urban/Rural, got {observed}"
    )


def test_spec_id_stable_regardless_of_slot_options():
    """Recomputing slot options leaves the spec id unchanged."""
    r1 = _resolve("fertility rate in Kenya")
    r2 = _resolve("fertility rate in Kenya")
    assert r1.spec.spec_id == r2.spec.spec_id
    assert r1.spec.resolved.slot_options  # sanity: options were actually computed
