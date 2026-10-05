"""Tests rehydrating pinned definitions, including graph drift and role-alias handling."""

from __future__ import annotations

import pytest

from code_index.shared import get_index
from datasets_api.adapter import build_identity_document, rehydrate_definition
from datasets_api.codes_materialise import (
    DefinitionNotReproducible,
    MemberNotResolvable,
    SpecNotExportable,
)
from datasets_api.mint_models import Scope
from datasets_api.shape import resolve_shape
from materialise.errors import OverCeilingError
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader, StoredDefinition
from statvar_members import MemberEnumerator

from ._graph_doubles import FakeMintGraphReader
from .conftest import FakeDatasetStore

_STAT_VAR_REF = "dcid:MalariaGrantDisbursements"


def _graph(**arcs: tuple[str, ...]) -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={"MalariaGrantDisbursements": {"typeOf": ("StatisticalVariable",), **arcs}}
    )


def _enumerator(graph: FakeMintGraphReader) -> MemberEnumerator:
    return MemberEnumerator(graph=graph)


def _stored(*, identity: dict[str, object], spec_id: str = "spec_test") -> StoredDefinition:
    """Builds a stored definition for slot and shape tests without a registry round trip.

    Pass a recomputed `spec_id` for hash validation. Other tests can use the placeholder.
    """
    return StoredDefinition(
        spec_id=spec_id,
        schema_version=1,
        identity=identity,
        identity_bytes=b"",
        identity_digest="",
    )


def test_round_trip_through_a_real_mint_agrees_with_the_original_shape():
    graph = _graph(observationProperties=("recipient",), measuredProperty=("amount",))
    shape = resolve_shape(_STAT_VAR_REF, graph=graph)
    document = build_identity_document(shape, scope=None)
    spec_id = recompute_identity(document).spec_id

    store = FakeDatasetStore()
    registry = LiveSpecRegistry(store=store)
    registry.register_definition(
        identity_definition=document, schema_version=1, supplied_spec_id=spec_id
    )
    reader = LiveSpecReader(store=store)
    stored = reader.read_definition(spec_id)
    assert stored is not None

    (rehydrated_shape,), (request,) = rehydrate_definition(
        stored, graph=graph, enumerator=_enumerator(graph), code_index=get_index()
    )

    assert rehydrated_shape == shape
    assert request.stat_var == _STAT_VAR_REF
    assert request.declared_roles == ("recipient",)


def test_an_unanchored_described_arm_identity_refuses_spec_not_exportable():
    stored = _stored(
        identity={
            "identity_version": "spec1",
            "shape": {"kind": "statvar_schema", "population_type": "Person"},
            "slots": [],
            "entities": {},
            "requested_window": {"kind": "all"},
        }
    )
    graph = _graph()

    with pytest.raises(SpecNotExportable) as exc_info:
        rehydrate_definition(
            stored, graph=graph, enumerator=_enumerator(graph), code_index=get_index()
        )

    assert exc_info.value.spec_id == "spec_test"
    assert "measured_property" in str(exc_info.value)


def test_a_pin_id_the_graph_does_not_hold_refuses_member_not_resolvable():
    stored = _stored(
        identity={
            "identity_version": "spec1",
            "shape": {"kind": "stat_var", "id": "NotAKnownStatVar"},
            "slots": [],
            "entities": {},
            "requested_window": {"kind": "all"},
        }
    )
    graph = _graph()

    with pytest.raises(MemberNotResolvable) as exc_info:
        rehydrate_definition(
            stored, graph=graph, enumerator=_enumerator(graph), code_index=get_index()
        )

    assert exc_info.value.members == ("spec_test",)


def test_resolve_shape_refusal_passes_through_as_member_not_resolvable():
    stored = _stored(
        identity={
            "identity_version": "spec1",
            "shape": {"kind": "stat_var", "id": "MalariaGrantDisbursements"},
            "slots": [],
            "entities": {},
            "requested_window": {"kind": "all"},
        }
    )
    # The graph has no `typeOf` arc for this StatVar, so `resolve_shape` refuses it.
    graph = FakeMintGraphReader()

    with pytest.raises(MemberNotResolvable) as exc_info:
        rehydrate_definition(
            stored, graph=graph, enumerator=_enumerator(graph), code_index=get_index()
        )

    assert exc_info.value.members == ("spec_test",)
    assert isinstance(exc_info.value.__cause__, MemberNotResolvable)


def test_declared_role_drift_since_minting_refuses_definition_not_reproducible():
    minted_shape = resolve_shape(
        _STAT_VAR_REF,
        graph=_graph(observationProperties=("recipient",), measuredProperty=("amount",)),
    )
    document = build_identity_document(
        minted_shape, scope=Scope(entities={"recipient": ["country/KEN"]})
    )
    spec_id = recompute_identity(document).spec_id
    stored = _stored(identity=document, spec_id=spec_id)

    # The identity hashes the caller's bare dcid, so a role change leaves its spec id fixed.
    # `entities` binds `recipient`, allowing the declared-role check to detect drift.
    drifted_graph = _graph(observationProperties=("provider",), measuredProperty=("amount",))

    with pytest.raises(DefinitionNotReproducible) as exc_info:
        rehydrate_definition(
            stored,
            graph=drifted_graph,
            enumerator=_enumerator(drifted_graph),
            code_index=get_index(),
        )

    assert exc_info.value.spec_ids == (spec_id,)


def test_a_stored_entity_role_the_statvar_never_declares_refuses_definition_not_reproducible():
    # The normal builder validates entity roles, but miss-path sidecars can store an identity
    # whose `entities` block binds `provider` when the resolved shape declares `recipient`.
    graph = _graph(observationProperties=("recipient",), measuredProperty=("amount",))
    minted_shape = resolve_shape(_STAT_VAR_REF, graph=graph)
    document = {
        **build_identity_document(minted_shape, scope=None),
        "entities": {"provider": ["country/KEN"]},
    }
    spec_id = recompute_identity(document).spec_id
    stored = _stored(identity=document, spec_id=spec_id)

    with pytest.raises(DefinitionNotReproducible) as exc_info:
        rehydrate_definition(
            stored, graph=graph, enumerator=_enumerator(graph), code_index=get_index()
        )

    assert exc_info.value.spec_ids == (spec_id,)
    assert "provider" in str(exc_info.value)


def test_a_flow_axis_alias_entity_role_rehydrates_against_the_matching_raw_declared_role():
    # The stored entity key `destination` is a flow-axis alias for the declared role
    # `financingRecipient`.
    graph = _graph(observationProperties=("financingRecipient",), measuredProperty=("amount",))
    minted_shape = resolve_shape(_STAT_VAR_REF, graph=graph)
    document = {
        **build_identity_document(minted_shape, scope=None),
        "entities": {"destination": ["country/KEN"]},
    }
    spec_id = recompute_identity(document).spec_id
    stored = _stored(identity=document, spec_id=spec_id)

    (shape,), (request,) = rehydrate_definition(
        stored, graph=graph, enumerator=_enumerator(graph), code_index=get_index()
    )

    assert shape == minted_shape
    assert request.stat_var == _STAT_VAR_REF
    # The identity hashes `destination`; the materialisation request maps it to the raw
    # `financingRecipient` property the mixer filters on.
    assert request.role_bindings == {"financingRecipient": ["country/KEN"]}


def test_a_flow_axis_alias_naming_no_declared_role_still_refuses_definition_not_reproducible():
    # The graph declares `financingProvider` (`origin`), while `destination` maps to
    # `financingRecipient`; the alias must match its own axis role.
    graph = _graph(observationProperties=("financingProvider",), measuredProperty=("amount",))
    minted_shape = resolve_shape(_STAT_VAR_REF, graph=graph)
    document = {
        **build_identity_document(minted_shape, scope=None),
        "entities": {"destination": ["country/KEN"]},
    }
    spec_id = recompute_identity(document).spec_id
    stored = _stored(identity=document, spec_id=spec_id)

    with pytest.raises(DefinitionNotReproducible) as exc_info:
        rehydrate_definition(
            stored, graph=graph, enumerator=_enumerator(graph), code_index=get_index()
        )

    assert exc_info.value.spec_ids == (spec_id,)
    assert "destination" in str(exc_info.value)


def test_no_drift_since_minting_rehydrates_cleanly():
    graph = _graph(observationProperties=("recipient",), measuredProperty=("amount",))
    minted_shape = resolve_shape(_STAT_VAR_REF, graph=graph)
    document = build_identity_document(minted_shape, scope=None)
    spec_id = recompute_identity(document).spec_id
    stored = _stored(identity=document, spec_id=spec_id)

    (shape,), (request,) = rehydrate_definition(
        stored, graph=graph, enumerator=_enumerator(graph), code_index=get_index()
    )

    assert shape == minted_shape
    assert request.stat_var == _STAT_VAR_REF


class _WithheldFromWalkGraphReader(FakeMintGraphReader):
    """A `FakeMintGraphReader` whose `node_subjects` walk always reports
    `truncated=True` and never returns `_ROOT`, simulating a member sitting
    past a truncated page's own window -- reachable only as `seed`, not
    something the walk itself would ever read. Mirrors
    `test_resolve_definition_members_guards._WithheldFromWalkGraphReader`.
    """

    def node_subjects(self, dcid, *, property, filters=None, max_results):
        page = super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )
        return type(page)(dcids=tuple(d for d in page.dcids if d != _ROOT), truncated=True)


_ROOT = "Count_Person"
_GENDERED_ARCS = {
    _ROOT: {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
    },
    "Count_Person_Male": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "constraintProperties": ("gender",),
        "gender": ("Male",),
    },
    "Count_Person_Female": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "constraintProperties": ("gender",),
        "gender": ("Female",),
    },
}


def _unbound_gender_identity() -> dict:
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "Person",
            "measured_property": "count",
        },
        "slots": [{"property": "gender", "binding": {"state": "unbound"}}],
        "entities": {},
        "requested_window": {"kind": "all"},
    }


def test_rehydrate_definition_threads_seed_through_to_resolve_definition_members():
    """Forwarding the root seed lets a truncated page prove completeness; without it,
    rehydration raises `OverCeilingError`.
    """
    stored = _stored(identity=_unbound_gender_identity())
    graph = _WithheldFromWalkGraphReader(arcs=_GENDERED_ARCS)

    with pytest.raises(OverCeilingError):
        rehydrate_definition(
            stored, graph=graph, enumerator=_enumerator(graph), code_index=get_index()
        )

    (shape,), (request,) = rehydrate_definition(
        stored, graph=graph, enumerator=_enumerator(graph), code_index=get_index(), seed=(_ROOT,)
    )

    assert shape.stat_var == f"dcid:{_ROOT}"
    assert request.stat_var == f"dcid:{_ROOT}"
