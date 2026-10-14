"""Test that candidate specs bind their place in the where slot.

The Nigeria remittances golden (remittances-pct-gdp-nigeria), which offers two shares of
GDP, checks alternate sentences, identity hashes, and StatVar slot values with offline
fixtures.
"""

from __future__ import annotations

from search.engine.spec_build import entity_from_resolved
from search.engine.spec_id import compute_spec_id
from search.models import (
    BindingUnbound,
    BindingValue,
    CandidatesPart,
    RawTextInput,
    ResolveOptions,
    ResolveRequest,
    Slot,
)
from tests.engine._harness import offline_resolve

_QUERY = "remittances as percentage of GDP in Nigeria"  # remittances-pct-gdp-nigeria
_ENTITY_DCID = "country/NGA"


def _where_slot(slots: list[Slot]) -> Slot | None:
    return next((s for s in slots if s.key.axis == "where"), None)


def _candidates() -> CandidatesPart:
    result = offline_resolve(ResolveRequest(input=RawTextInput(query=_QUERY)))
    part = result.parts[0]
    assert isinstance(part, CandidatesPart), (
        f"expected candidates for {_QUERY!r}, got status={part.status!r}"
    )
    return part


# Every candidate spec carries its place


class TestAlternateSpecsCarryWhereSlot:
    def test_every_spec_has_a_bound_where_slot(self):
        resp = _candidates()
        specs = resp.candidates.specs
        assert len(specs) >= 2, "expected at least an elected + one alternate spec"
        for spec in specs:
            where = _where_slot(spec.resolved.slots)
            assert where is not None, f"spec {spec.spec_id!r} has no where slot at all"
            assert where.key.property is None
            assert isinstance(where.binding, BindingValue), (
                f"spec {spec.spec_id!r} where-slot is {where.binding!r}, "
                "expected a bound entity so the spec self-describes its place"
            )
            assert where.binding.value.ref is not None
            assert where.binding.value.ref.ref == _ENTITY_DCID

    def test_every_spec_entities_list_also_carries_nigeria(self):
        """The where slot and resolved entity list both identify Nigeria."""
        resp = _candidates()
        for spec in resp.candidates.specs:
            assert [e.ref for e in spec.resolved.entities] == [_ENTITY_DCID]


class TestAlternateSpecSentencesIncludePlace:
    def test_every_candidate_sentence_mentions_nigeria(self):
        req = ResolveRequest(
            input=RawTextInput(query=_QUERY),
            response_options=ResolveOptions(),
        )
        part = offline_resolve(req).parts[0]
        assert isinstance(part, CandidatesPart)
        for spec in part.candidates.specs:
            assert spec.resolved.rendered_sentence is not None
            assert "Nigeria" in spec.resolved.rendered_sentence, (
                f"spec {spec.spec_id!r} sentence dropped the place: "
                f"{spec.resolved.rendered_sentence!r}"
            )


class TestPrimarySpecUnaffected:
    """The elected shape retains its own bound where slot."""

    def test_definite_single_entity_query_unaffected(self):
        result = offline_resolve(ResolveRequest(input=RawTextInput(query="total population India")))
        assert result.parts[0].status == "definite"
        spec = result.parts[0].spec
        where = _where_slot(spec.resolved.slots)
        assert where is not None
        assert isinstance(where.binding, BindingValue)
        assert where.binding.value.ref.ref == "country/IND"


# Where-slot binding leaves identity and StatVar values unchanged


class TestFixDoesNotPerturbSpecIdentity:
    def test_alt_spec_id_matches_pre_fix_unbound_where_slot(self):
        """Restoring an unbound where slot leaves each spec id unchanged."""
        resp = _candidates()
        for spec in resp.candidates.specs:
            pre_fix_slots = [
                s
                if not (s.key.axis == "where" and s.key.property is None)
                else s.model_copy(update={"binding": BindingUnbound()})
                for s in spec.resolved.slots
            ]
            pre_fix_id = compute_spec_id(
                spec.definition.identity.shape.model_dump(exclude_none=True),
                pre_fix_slots,
                entities=[entity_from_resolved(e) for e in spec.resolved.entities],
                requested_window=spec.resolved.trace.requested_window,
                open_roles=(),
            )
            assert pre_fix_id == spec.spec_id, (
                f"spec {spec.spec_id!r}: binding the where-slot perturbed spec_id "
                f"(got {pre_fix_id!r} for the unbound-where variant)"
            )


class TestStatVarSlotValuesExcludeWhereSlot:
    def test_no_statvar_gains_the_where_slot(self):
        """The where slot identifies the observation subject.

        StatVar slot values carry constraints.
        """
        resp = _candidates()
        for spec in resp.candidates.specs:
            for sv in spec.resolved.members:
                assert not any(
                    slot_value.key.axis == "where" and slot_value.key.property is None
                    for slot_value in sv.slot_values
                ), f"spec {spec.spec_id!r} StatVar {sv.ref!r} gained the where slot"
