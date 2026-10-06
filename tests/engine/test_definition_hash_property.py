"""Every emitted definition hashes to its ``spec_id`` on definite, candidate, and resubmit paths.

Unchanged definitions verify and mint the same id when reposted."""

from __future__ import annotations

from dfo_spec_kit import identity

from search.engine.identity_edit import validate_edited_identity
from search.engine.region import resolve_spec_resubmit
from search.models import (
    CandidatesPart,
    IdentityEntityGroup,
    ResubmitIdentityPayload,
)
from tests.engine._base_identity import (
    make_base,
    make_identity,
    subject_entity,
)
from tests.engine._builders import value_slot as _value_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve, walk_options
from tests.fixtures import FakeGraph

_GDP_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
# Resubmit derives this five-tuple from the fixture StatVar's graph node.
_GDP_SHAPE = {
    "kind": "statvar_schema",
    "population_type": "EconomicActivity",
    "measured_property": "amount",
    "stat_type": "measuredValue",
    "measurement_qualifier": "Nominal",
}
_GDP_ENTITY_DCID = "country/IND"

# The base and posted identities both declare this activity-source slot.
_ACTIVITY_SOURCE_SLOT = _value_slot(
    "how",
    "activitySource",
    "activity source",
    "GrossDomesticProduction",
    "Gross Domestic Production",
)


def _assert_definition_hashes_to_spec_id(spec) -> None:
    """The wire identity hashes to ``spec_id``.

    Both copies of ``spec_id`` and schema version 1 agree."""
    dumped = spec.definition.identity.model_dump(exclude_none=True)
    assert identity.spec_id_from_value(dumped) == spec.spec_id
    assert spec.definition.spec_id == spec.spec_id
    assert spec.definition.schema_version == 1


# ---------------------------------------------------------------------------
# Primary spec: raw_text, place_as_constraint ON (the common case)
# ---------------------------------------------------------------------------


def test_primary_definite_spec_definition_hashes_to_spec_id():
    result = offline_resolve(
        make_request("fertility rate in Kenya"), pac=True, sdmx=FakeSdmxClient()
    ).parts[0]
    assert result.status == "definite"
    _assert_definition_hashes_to_spec_id(result.spec)


# ---------------------------------------------------------------------------
# place_as_constraint=False mode
# ---------------------------------------------------------------------------


def test_place_constraint_off_spec_definition_hashes_to_spec_id():
    """With ``pac=False``, the emitted definition hashes to ``spec_id``."""
    result = offline_resolve(
        make_request("fertility rate in Kenya"), pac=False, sdmx=FakeSdmxClient()
    ).parts[0]
    assert result.status == "definite"
    spec = result.spec
    assert all(e.role == "observationAbout" for e in spec.resolved.entities)
    _assert_definition_hashes_to_spec_id(spec)


# ---------------------------------------------------------------------------
# Candidates response: no single answer resolved
# ---------------------------------------------------------------------------


def test_candidates_response_specs_definitions_hash_to_their_spec_ids():
    """Every candidate definition hashes to its own ``spec_id``."""
    result = offline_resolve(make_request("under-5 child mortality rate")).parts[0]
    assert isinstance(result, CandidatesPart)
    assert len(result.candidates.specs) >= 2
    for spec in result.candidates.specs:
        _assert_definition_hashes_to_spec_id(spec)


# ---------------------------------------------------------------------------
# the resubmit door: standard promote
# ---------------------------------------------------------------------------


def test_standard_resubmit_definition_hashes_to_spec_id():
    entities = [subject_entity(_GDP_ENTITY_DCID)]
    inp = ResubmitIdentityPayload(
        identity=make_identity(_GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], entities=entities),
    )
    region = resolve_spec_resubmit(
        inp=inp,
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=walk_options(FakeGraph()),
    )
    assert region.status == "definite"
    _assert_definition_hashes_to_spec_id(region.specs[0])


# ---------------------------------------------------------------------------
# Round-trip: an emitted definition, reposted unchanged, verifies and mints
# the same spec_id
# ---------------------------------------------------------------------------


def test_emitted_definition_hashes_to_its_own_spec_id():
    """An emitted candidate definition hashes to its own ``spec_id``."""
    result = offline_resolve(make_request("under-5 child mortality rate")).parts[0]
    assert isinstance(result, CandidatesPart)
    _assert_definition_hashes_to_spec_id(result.candidates.specs[0])


def test_identity_carrying_entity_groups_hashes_to_the_id_minted_without_them():
    """`entity_groups` stays out of the identity core and the hash on a resubmit."""
    spec_id, plain = make_base(_GDP_SHAPE, entities=[subject_entity(_GDP_ENTITY_DCID)])
    grouped = plain.model_copy(
        update={"entity_groups": {"observationAbout": IdentityEntityGroup(label="India")}}
    )
    validate_edited_identity(grouped)
    assert identity.spec_id_from_value(grouped.core_document()) == spec_id


def test_reposting_emitted_definition_unchanged_mints_the_same_spec_id():
    """Reposting an emitted definition mints the same ``spec_id``."""
    entities = [subject_entity(_GDP_ENTITY_DCID)]
    identity = make_identity(_GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], entities=entities)
    inp = ResubmitIdentityPayload(
        identity=identity,
    )
    first = resolve_spec_resubmit(
        inp=inp,
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=walk_options(FakeGraph()),
    )
    assert first.status == "definite"
    first_spec = first.specs[0]

    # Re-post the FIRST response's own emitted definition, completely
    # unedited, as the base for a second resubmit of the identical slots.
    second_base_identity = first_spec.definition.identity
    inp2 = ResubmitIdentityPayload(
        identity=second_base_identity,
    )
    second = resolve_spec_resubmit(
        inp=inp2,
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=walk_options(FakeGraph()),
    )
    assert second.status == "definite"
    assert second.specs[0].spec_id == first_spec.spec_id
