"""A resubmit that names a dataset resolves the members the Datasets mint resolves.

The imputed education-aid identity sits on the imputed multilateral table's
`attributedAmount` five-tuple and binds `facets.provenance` to that table. Without posted
`resolved_stat_var_dcids` the resubmit elects its members from the fixture index. The
GDP identity sits outside the index, so its resubmit serves the posted seed.
"""

from __future__ import annotations

import pytest

from code_index import FamilyQuery
from code_index.members import canonical_identity, elected_pins
from code_index.shared import get_index
from datasets_api.adapter import resolve_definition_members
from materialise.errors import FacetNotAvailableError
from search.engine.errors import EngineInputError
from search.engine.region import promote_ground, resolve_spec_resubmit
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.engine.retrieval.bound_values import identity_query, observed_measure_family
from search.engine.spec_id import mint_identity
from search.models import (
    BaseIdentity,
    BindingValue,
    IdentityBindingValue,
    ResubmitIdentityPayload,
    Spec,
)
from search.render.plan import render_sentence_plan
from search.render.tokens import sentence_from_plan
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry.identity import recompute_identity
from statvar_members import MemberEnumerator, member_digest
from tests.code_index.aid_identity import (
    AID_TYPE,
    CRS_MEMBER_COUNT,
    CRS_PROVENANCE,
    IMPUTED_MEMBER_COUNT,
    IMPUTED_MULTILATERAL,
    IMPUTED_PROVENANCE,
    education_aid_identity,
    imputed_education_aid_identity,
)
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.datasets_api._member_bounds import mint_bounds
from tests.engine._axis_doubles import axis_confirming_reader
from tests.engine._base_identity import GDP_SHAPE, subject_entity, where_slots_for
from tests.engine._fake_coverage import (
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.engine.test_candidates_measure_alternatives import _ROLES
from tests.engine.test_resubmit_alternative_measure_round_trip import (
    _KEN,
    _family_graph,
    _sibling_sdmx,
)
from tests.engine.test_resubmit_edit_accounting import (
    _EXTRA_PROVENANCE_DCID,
    _GDP_ENTITY_DCID,
    _GDP_SV_DCID,
    _MEASUREMENT_METHOD_A,
    _MEASUREMENT_METHOD_B,
    _PROVENANCE_DCID,
    _edited_identity,
    _method_facet,
    _unbound_slot,
)
from tests.fixtures import FakeGraph

_ROLE_NAMES = RoleNames(donor=_ROLES[0], recipient=_ROLES[1])


def _mint_family(identity: dict, spec_id: str = "spec_test") -> frozenset[str]:
    graph = FakeMintGraphReader(arcs={})
    return resolve_definition_members(
        identity,
        spec_id=spec_id,
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=get_index(),
        bounds=mint_bounds(),
    ).family


def _resubmit_education(
    monkeypatch, identity: BaseIdentity, spec_id: str, *, seed: list[str] | None = None
) -> RegionResult:
    """Resubmit `identity` against a graph and a coverage index that hold every member of the
    CRS and the imputed education families, so the index election alone decides which the
    resubmit serves."""
    served = _mint_family(imputed_education_aid_identity()) | _mint_family(education_aid_identity())
    rows = [row for row in get_index().statvars if row.dcid in served]
    graph = _family_graph(rows)
    # Each member's facet carries the provenance of the dataset that publishes it, as the
    # coverage build records it.
    coverage_index = patch_coverage(
        monkeypatch,
        [
            (row.dcid, f"f_{provenance}", "country/USA", _KEN, [2020])
            for row in rows
            for provenance in row.provenances
        ],
        role_names={row.dcid: _ROLE_NAMES for row in rows},
        facet_metadata={
            f"f_{provenance}": FacetMetadata(provenance_id=provenance)
            for row in rows
            for provenance in row.provenances
        },
    )
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=identity,
        ),
        graph=graph,
        sdmx=_sibling_sdmx(min(served)),
        coverage=FakeCoverageService(coverage_index).snapshot(),
        options=walk_options(
            graph,
            EngineOptions(
                materialise_graph=axis_confirming_reader(*(row.dcid for row in rows), graph=graph)
            ),
        ),
    )


def _canonical_spec_id(document: dict) -> str:
    """The spec_id of `document`'s canonical form (`code_index.members.canonical_identity`),
    which binds the imputed table's invariants."""
    return mint_identity(canonical_identity(get_index(), document))[0]


def _where_dcid(spec: Spec) -> str | None:
    """The dcid the spec's where slot binds, or None when it binds no value."""
    where = next(slot for slot in spec.resolved.slots if slot.key.axis == "where")
    return where.binding.value.ref.ref if isinstance(where.binding, BindingValue) else None


def test_a_resubmit_naming_the_imputed_dataset_resolves_the_mints_family(monkeypatch):
    document = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    spec_id, identity = mint_identity(document)
    family = _mint_family(document, spec_id)
    assert len(family) == IMPUTED_MEMBER_COUNT

    measure_queries: list[FamilyQuery] = []

    def recording_measure_family(shape, index, axes, query, members, **kwargs):
        measure_queries.append(query)
        return observed_measure_family(shape, index, axes, query, members, **kwargs)

    monkeypatch.setattr(promote_ground, "observed_measure_family", recording_measure_family)

    region = _resubmit_education(monkeypatch, identity, spec_id)

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]
    assert spec.spec_id == _canonical_spec_id(document) == spec_id
    assert spec.definition.identity.facets == {
        "provenance": IdentityBindingValue(ref=IMPUTED_PROVENANCE)
    }
    assert {stat_var.ref.ref for stat_var in spec.resolved.members} == family
    assert spec.resolved.shape.member_digest == member_digest(family)
    # The served members' measure family is read under the dataset the identity names.
    assert [query.source for query in measure_queries] == [IMPUTED_PROVENANCE]
    # The named dataset binds the source slot and leaves the where slot its place.
    assert _where_dcid(spec) == _KEN


def _aid_type_binding(spec: Spec) -> object:
    """The binding of the spec's aid-type slot."""
    return next(
        slot.binding
        for slot in spec.resolved.slots
        if slot.key.property is not None and slot.key.property.ref == AID_TYPE
    )


@pytest.mark.parametrize("seeded", [False, True], ids=["unseeded", "seeded"])
def test_a_resubmit_leaving_a_pinned_slot_unbound_mints_the_canonical_spec_id(monkeypatch, seeded):
    """The imputed table declares its one aid type as an invariant. The resubmit of an
    identity naming the table that leaves the aid type unbound and of one binding it mint
    one canonical spec_id, which binds the aid type as a first answer does, and serve the
    family the mint resolves for the identity binding it."""
    pinned = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    pinned_spec_id, _ = mint_identity(pinned)
    family = _mint_family(pinned)
    document = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    next(slot for slot in document["slots"] if slot["property"] == AID_TYPE)["binding"] = {
        "state": "unbound"
    }
    spec_id, identity = mint_identity(document)
    assert spec_id != pinned_spec_id

    region = _resubmit_education(
        monkeypatch, identity, spec_id, seed=sorted(family) if seeded else None
    )

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]
    assert spec.spec_id == _canonical_spec_id(pinned) == _canonical_spec_id(document)
    assert spec.spec_id == pinned_spec_id
    binding = _aid_type_binding(spec)
    assert isinstance(binding, BindingValue)
    assert binding.value.ref.ref == IMPUTED_MULTILATERAL
    pins = elected_pins(get_index(), identity_query(spec.definition.identity))
    assert pins[AID_TYPE] == IMPUTED_MULTILATERAL
    assert {stat_var.ref.ref for stat_var in spec.resolved.members} == family


def test_a_provenance_set_of_one_names_the_dataset_its_value_spelling_names(monkeypatch):
    """The spec kit hashes a one-ref `set` and a `value` to one `spec_id`, so Search and the
    mint resolve the imputed family for both spellings of the facet."""
    value_document = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    set_document = imputed_education_aid_identity()
    set_document["facets"] = {"provenance": {"state": "set", "refs": [IMPUTED_PROVENANCE]}}
    spec_id = recompute_identity(value_document).spec_id
    assert recompute_identity(set_document).spec_id == spec_id
    family = _mint_family(value_document)
    assert _mint_family(set_document) == family
    assert len(family) == IMPUTED_MEMBER_COUNT

    # The identity as a client posts it: validated, with the facet left a set.
    region = _resubmit_education(monkeypatch, BaseIdentity.model_validate(set_document), spec_id)

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]
    assert spec.spec_id == _canonical_spec_id(value_document)
    assert {stat_var.ref.ref for stat_var in spec.resolved.members} == family
    assert spec.resolved.shape.member_digest == member_digest(family)


def test_a_provenance_set_of_two_names_no_dataset(monkeypatch):
    """Two refs name no single dataset, so the mint resolves the family the default
    election picks. That family publishes no facet of the imputed dataset, and a set binding
    needs a facet of every ref it names, so the resubmit refuses the binding."""
    document = education_aid_identity()
    document["facets"] = {
        "provenance": {"state": "set", "refs": [CRS_PROVENANCE, IMPUTED_PROVENANCE]}
    }
    spec_id, identity = mint_identity(document)
    family = _mint_family(document)
    assert family == _mint_family(education_aid_identity())
    assert len(family) == CRS_MEMBER_COUNT

    with pytest.raises(EngineInputError) as exc_info:
        _resubmit_education(monkeypatch, identity, spec_id)

    assert exc_info.value.code == "facet_not_available"
    assert IMPUTED_PROVENANCE in str(exc_info.value)


@pytest.mark.parametrize("draft", ["aid-type-bound", "aid-type-unbound", "purpose-no-row-holds"])
def test_a_posted_seed_does_not_stand_in_for_a_family_the_election_refuses(monkeypatch, draft):
    """The identity names the CRS dataset on the five-tuple only the imputed dataset
    publishes. The mint refuses it `facet_not_available`, and the resubmit refuses the same
    way and names CRS, with the imputed members posted as a seed and with none. CRS pins
    nothing on that five-tuple, so an aid-type slot left unbound refuses the same way, and
    the refusal reads the five-tuple alone, so a purpose no row holds refuses the same way."""
    document = imputed_education_aid_identity(source=CRS_PROVENANCE)
    if draft == "aid-type-unbound":
        next(slot for slot in document["slots"] if slot["property"] == AID_TYPE)["binding"] = {
            "state": "unbound"
        }
    if draft == "purpose-no-row-holds":
        next(slot for slot in document["slots"] if slot["property"] == "crsPurpose")["binding"] = {
            "state": "value",
            "ref": "CRSPurposeCodeNoRowHolds",
        }
    spec_id, identity = mint_identity(document)
    with pytest.raises(FacetNotAvailableError):
        _mint_family(document)
    imputed = sorted(_mint_family(imputed_education_aid_identity()))

    with pytest.raises(EngineInputError) as seeded:
        _resubmit_education(monkeypatch, identity, spec_id, seed=imputed[:5])
    with pytest.raises(EngineInputError) as unseeded:
        _resubmit_education(monkeypatch, identity, spec_id)

    for refusal in (seeded.value, unseeded.value):
        assert refusal.code == "facet_not_available"
        assert CRS_PROVENANCE in str(refusal)


def _resubmit_gdp_of_india(facets: dict[str, IdentityBindingValue]) -> Spec:
    """Resubmit nominal GDP of India, an identity the index does not carry, with one posted
    StatVar and `facets`. The graph holds one facet of each of two provenances."""
    entities = [subject_entity(_GDP_ENTITY_DCID)]
    graph = FakeGraph(
        obs={
            f"{_GDP_SV_DCID}|{_GDP_ENTITY_DCID}": [
                _method_facet(_PROVENANCE_DCID, _MEASUREMENT_METHOD_A, dates=["2020"]),
                _method_facet(
                    _EXTRA_PROVENANCE_DCID, _MEASUREMENT_METHOD_B, dates=["2015", "2016"]
                ),
            ]
        }
    )
    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=_edited_identity(
                GDP_SHAPE,
                [
                    _unbound_slot("how", "activitySource", "activity source"),
                    *where_slots_for(entities),
                ],
                entities=entities,
            ).model_copy(update={"facets": facets}),
        ),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(graph, EngineOptions(include_observations=True)),
    )
    assert region.status == "definite", region.no_data_reason
    return region.specs[0]


def test_a_resubmit_naming_a_source_keeps_its_where_slot_and_its_sentence():
    spec = _resubmit_gdp_of_india({"provenance": IdentityBindingValue(ref=_PROVENANCE_DCID)})

    assert _where_dcid(spec) == _GDP_ENTITY_DCID
    source = next(slot for slot in spec.resolved.slots if slot.key.axis == "source")
    assert isinstance(source.binding, BindingValue)
    assert source.binding.value.ref.ref == _PROVENANCE_DCID
    sentence = sentence_from_plan(render_sentence_plan(spec))
    assert sentence.startswith("India's ")
    assert "WorldDevelopmentIndicators" not in sentence


def test_an_identity_the_mint_walks_states_the_walked_family():
    """The index does not resolve the GDP identity, so the mint walks the graph for its
    family, and the resubmit runs the same walk. Search serves its one member and states
    the mint's served count and family digest."""
    spec = _resubmit_gdp_of_india(
        {"measurementMethod": IdentityBindingValue(ref=_MEASUREMENT_METHOD_A)}
    )

    assert [stat_var.ref.ref for stat_var in spec.resolved.members] == [_GDP_SV_DCID]
    assert spec.resolved.shape.member_count == 1
    assert spec.resolved.stat_vars.served_count == 1
    assert spec.resolved.shape.member_digest == member_digest([_GDP_SV_DCID])
