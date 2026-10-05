"""Search and the Datasets mint resolve the same members for the same identity.

A seeded fuzz draws identities from the packaged code index: ones that bind no value, sit on
the opt-in dataset's five-tuple, name a dataset, bind values no row holds together, and declare fewer
slots than any row carries. A slot binds a value, a set, or a set of one ref. The provenance
facet binds a value, a set of one ref, a set of two, or a provenance the index does not hold.
Some identities pin one StatVar. For each, every Search reader of a family agrees with
`datasets_api.adapter.resolve_definition_members`, or both refuse. An identity that leaves
unbound an invariant its canonical form binds (`code_index.members.invariant_bindings`)
resolves the family of its canonical form at the mint, and Search states that family for
both, and the resubmit answers with a canonical identity. A sample goes through a resubmit as
a client posts it.

A second fuzz draws five-tuples whose members declare several role sets, with identities
that bind entities under no role, the subject, the recipient, the provider, or both. Search's
recall election and its options walk agree with the mint's walk on the role class. A recall
sample that misses members answers only from the mint's family once the mint's walk narrows
it, and a resubmit posting that sample names every member the mint refuses. An identity that
binds no entity on a five-tuple of subject-only and role-bearing members is answered from
the subject-only shape's sample without a walk, and the mint elects that class too.
"""

from __future__ import annotations

import dataclasses
import functools
import random
from collections import defaultdict

from code_index.members import (
    elected_pins,
    index_covers_slot_keys,
    invariant_bindings,
    named_dataset,
)
from code_index.shared import get_index, get_measure_index
from code_index.types import StatVarRow
from datasets_api.adapter import resolve_definition_members
from datasets_api.codes_materialise import RoleSignatureIncompatible, SpecNotExportable
from materialise.errors import DefaultFacetUndefinedError, FacetNotAvailableError
from search.engine.bind import SlotBindingDraft
from search.engine.errors import EngineInputError, RoleSignatureError
from search.engine.options.population import options_population
from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.engine.retrieval.bound_values import (
    RoleClassWalks,
    canonicalise_identity,
    elected_role_class_shape,
    identity_dataset,
    identity_family_digest,
    identity_query,
    index_decides_membership,
    measure_family_partitions,
    posted_members_outside_role_class,
    shape_member_facts,
    shape_query,
    source_binding,
    synthetic_shape_for_identity,
)
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.models import (
    BaseIdentity,
    BindingValue,
    IdentityShapeSchema,
    ResubmitIdentityPayload,
)
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry.identity import recompute_identity
from statvar_members import FiveTuple, MemberEnumerator, member_digest
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine._axis_doubles import axis_confirming_reader
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine.test_candidates_measure_alternatives import _ROLES
from tests.engine.test_resubmit_alternative_measure_round_trip import (
    _KEN,
    _family_graph,
    _sibling_sdmx,
)
from tests.fixtures import FakeGraph

_SEED = 20261002
_IDENTITIES = 240
_RECALL_SAMPLE = 6
# One drawn identity in `_PIN_EVERY` pins a StatVar.
_PIN_EVERY = 16
# The resubmit sample: identities whose family the mint resolves, of at most this many members.
_RESUBMITS = 24
# The resubmit sample of identities that leave unbound an invariant their canonical form binds.
_RELEASING_RESUBMITS = 10
_RESUBMIT_FAMILY_CAP = 300
_ROLE_NAMES = RoleNames(donor=_ROLES[0], recipient=_ROLES[1])

# The refusals of an index-resolved identity that leave no family: no matching row, a named
# dataset that publishes none, and several datasets with no default.
_NO_FAMILY = (SpecNotExportable, FacetNotAvailableError, DefaultFacetUndefinedError)


def _five_tuple(row: StatVarRow) -> FiveTuple:
    return FiveTuple(
        population_type=row.population_type,
        measured_property=row.measured_property,
        stat_type=row.stat_type,
        measurement_qualifier=row.measurement_qualifier,
        measurement_denominator=row.measurement_denominator,
    )


@functools.cache
def _provenances() -> dict[str, tuple[str, ...]]:
    return {row.dcid: row.provenances for row in get_index().statvars}


def _arcs(row: StatVarRow) -> dict:
    """The recall arcs of an index row."""
    arcs = {
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in row.constraint_properties]},
        **{prop: {"nodes": [{"dcid": v}]} for prop, v in row.constraint_properties.items()},
    }
    if row.measurement_qualifier is not None:
        arcs["measurementQualifier"] = {"nodes": [{"dcid": row.measurement_qualifier}]}
    return arcs


def _mint_arcs(row: StatVarRow) -> dict[str, tuple[str, ...]]:
    """The arcs the mint reads for a pinned index row."""
    arcs = {
        "typeOf": ("StatisticalVariable",),
        "populationType": (row.population_type,),
        "measuredProperty": (row.measured_property,),
        "statType": (row.stat_type,),
        "observationProperties": tuple(_ROLES),
        "constraintProperties": tuple(row.constraint_properties),
        **{prop: (value,) for prop, value in row.constraint_properties.items()},
    }
    if row.measurement_qualifier is not None:
        arcs["measurementQualifier"] = (row.measurement_qualifier,)
    return arcs


@dataclasses.dataclass(frozen=True)
class _Drawn:
    identity: dict
    all_keys: bool
    """Whether the identity declares a slot for every property its five-tuple's rows carry,
    as an identity Search mints does."""
    rows: tuple[StatVarRow, ...]
    """Every index row of the identity's five-tuple."""
    pinned: StatVarRow | None = None
    """The row a pinned identity names."""


def _slot_binding(rng: random.Random, prop: str, value: str, rows: list[StatVarRow]) -> dict:
    """A binding of `prop` to `value`: a value, a set of that one ref, or a set of it and
    another value a row of the five-tuple carries."""
    form = rng.random()
    if form < 0.15:
        return {"state": "set", "refs": [value]}
    if form < 0.35:
        others = sorted(
            {row.constraint_properties[prop] for row in rows if prop in row.constraint_properties}
            - {value}
        )
        if others:
            return {"state": "set", "refs": sorted({value, rng.choice(others)})}
    return {"state": "value", "ref": value}


def _provenance_facet(rng: random.Random, datasets: list[str]) -> dict | None:
    """No facet, or one that binds a dataset as a value or as a set of one ref, two datasets
    as a set, or a provenance the index does not hold."""
    form = rng.random()
    if form < 0.35:
        return None
    dataset = rng.choice(datasets)
    if form < 0.6:
        return {"state": "value", "ref": dataset}
    if form < 0.75:
        return {"state": "set", "refs": [dataset]}
    if form < 0.88:
        return {"state": "set", "refs": sorted(rng.sample(datasets, 2))}
    return {"state": "value", "ref": f"{dataset}NotInTheIndex"}


def _draw(rng: random.Random) -> list[_Drawn]:
    index = get_index()
    rows_by_five_tuple: dict[FiveTuple, list[StatVarRow]] = defaultdict(list)
    for row in index.statvars:
        rows_by_five_tuple[_five_tuple(row)].append(row)
    opt_in = [
        five_tuple
        for five_tuple in rows_by_five_tuple
        if any(index.datasets[owner].opt_in for owner in index.datasets_by_five_tuple[five_tuple])
    ]
    assert opt_in, "the packaged index holds no five-tuple of an opt-in dataset"
    datasets = sorted(index.datasets)

    drawn = []
    for position in range(_IDENTITIES):
        # The opt-in dataset's few rows sit on one five-tuple, so a share of the draws lands there.
        five_tuple = rng.choice(opt_in if rng.random() < 0.3 else list(rows_by_five_tuple))
        rows = rows_by_five_tuple[five_tuple]
        base = rng.choice(rows)
        if position % _PIN_EVERY == _PIN_EVERY - 1:
            pinned = {
                "identity_version": "spec1",
                "shape": {"kind": "stat_var", "id": base.dcid},
                "slots": [],
                "entities": {"destination": ["country/KEN"]},
                "requested_window": {"kind": "all"},
            }
            drawn.append(_Drawn(identity=pinned, all_keys=False, rows=tuple(rows), pinned=base))
            continue
        bind_share = rng.choice((0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8))
        bound = {
            prop: value
            for prop, value in base.constraint_properties.items()
            if rng.random() < bind_share
        }
        if rng.random() < 0.3:
            # Values of another row, which no row may hold beside the first's.
            other = rng.choice(rows)
            bound.update(
                (prop, value)
                for prop, value in other.constraint_properties.items()
                if rng.random() < 0.4
            )
        for prop, value in list(bound.items()):
            code = index.code_by_dcid(value)
            if code is not None and code.parent is not None and rng.random() < 0.2:
                bound[prop] = code.parent
        keys = sorted({prop for row in rows for prop in row.constraint_properties})
        all_keys = rng.random() < 0.85
        if not all_keys:
            keys = [prop for prop in keys if prop in bound or rng.random() < 0.6]
        identity: dict = {
            "identity_version": "spec1",
            "shape": {
                "kind": "statvar_schema",
                **{
                    field: value
                    for field, value in (
                        ("population_type", five_tuple.population_type),
                        ("measured_property", five_tuple.measured_property),
                        ("stat_type", five_tuple.stat_type),
                        ("measurement_qualifier", five_tuple.measurement_qualifier),
                    )
                    if value is not None
                },
            },
            "slots": [
                {
                    "property": prop,
                    "binding": _slot_binding(rng, prop, bound[prop], rows)
                    if prop in bound
                    else {"state": "unbound"},
                }
                for prop in keys
            ],
            "entities": {"destination": ["country/KEN"]},
            "requested_window": {"kind": "all"},
        }
        facet = _provenance_facet(rng, datasets)
        if facet is not None:
            identity["facets"] = {"provenance": facet}
        drawn.append(_Drawn(identity=identity, all_keys=all_keys, rows=tuple(rows)))
    return drawn


def _mint(identity: dict, *, pinned: StatVarRow | None = None) -> tuple[frozenset[str], str | None]:
    """The mint's family and elected dataset, or an empty family and no dataset when the
    mint refuses the identity for holding no family. A pinned identity elects no dataset."""
    graph = FakeMintGraphReader(arcs={pinned.dcid: _mint_arcs(pinned)} if pinned else {})
    try:
        resolved = resolve_definition_members(
            identity,
            spec_id="spec_fuzz",
            graph=graph,
            enumerator=MemberEnumerator(graph=graph),
            code_index=get_index(),
        )
    except _NO_FAMILY:
        return frozenset(), None
    datasets = {shape.dataset.provenance for shape in resolved.shapes if shape.dataset}
    if pinned is not None:
        assert not datasets, datasets
        return resolved.family, None
    assert len(datasets) == 1, datasets
    return resolved.family, datasets.pop()


def _first_pass_shape(drawn: _Drawn, rng: random.Random) -> ShapeDraft:
    """The shape `derive_shapes` builds for the identity's five-tuple: a slot key for every
    property its index rows carry and a recall sample drawn across datasets."""
    model = BaseIdentity.model_validate(drawn.identity)
    assert isinstance(model.shape, IdentityShapeSchema)
    sample = rng.sample(drawn.rows, min(_RECALL_SAMPLE, len(drawn.rows)))
    sample += [row for row in drawn.rows if len(row.provenances) == 1][-2:]
    return dataclasses.replace(
        synthetic_shape_for_identity(model.shape),
        slot_keys=tuple(
            SlotKeyDraft(axis="how", property_dcid=slot.property, label=slot.property)
            for slot in model.slots
        ),
        sv_arc_facts={row.dcid: _arcs(row) for row in sample},
        sv_arc_facts_complete=False,
    )


def _first_pass_bindings(identity: dict) -> list[SlotBindingDraft]:
    """The drafts bind writes for the identity's slots: a value for one ref, a set for more."""
    bindings = []
    for slot in identity["slots"]:
        binding = slot["binding"]
        if binding["state"] == "unbound":
            continue
        refs = [binding["ref"]] if binding["state"] == "value" else binding["refs"]
        bindings.append(
            SlotBindingDraft(
                axis="how",
                property_dcid=slot["property"],
                kind="value" if len(refs) == 1 else "set",
                value_dcids=list(refs),
            )
        )
    return bindings


def _check_pinned(drawn: _Drawn, kinds: dict[str, int]) -> list[str]:
    """A pinned identity names its one StatVar, which the mint admits. Search reads no index
    family for it and states no dataset and no digest."""
    assert drawn.pinned is not None
    kinds["pins a StatVar"] += 1
    model = BaseIdentity.model_validate(drawn.identity)
    family, _dataset = _mint(drawn.identity, pinned=drawn.pinned)
    found: list[str] = []
    if family != {drawn.pinned.dcid}:
        found.append(f"pinned: mint family {sorted(family)!r}")
    if index_decides_membership(model):
        found.append("pinned: Search reads the index for a pinned identity")
    if identity_dataset(model) is not None or identity_family_digest(model) is not None:
        found.append("pinned: Search states an index family for a pinned identity")
    return found


def _check(drawn: _Drawn, rng: random.Random, kinds: dict[str, int], *, wide: bool) -> list[str]:
    """The readers of `drawn`'s family that disagree with the mint, counting in `kinds` what
    kind of identity it is. `wide` adds the options population and the measure family."""
    if drawn.pinned is not None:
        return _check_pinned(drawn, kinds)
    index = get_index()
    identity = drawn.identity
    model = BaseIdentity.model_validate(identity)
    assert isinstance(model.shape, IdentityShapeSchema)
    query = identity_query(model)
    facet = identity.get("facets", {}).get("provenance", {})
    states = [slot["binding"]["state"] for slot in identity["slots"]]
    refs = [len(slot["binding"].get("refs", ())) for slot in identity["slots"]]
    kinds["binds a slot to a set of one ref"] += 1 in refs
    kinds["binds a slot to a set of several refs"] += any(count > 1 for count in refs)
    kinds["binds a slot to a value"] += "value" in states
    kinds["names a dataset by a set of one ref"] += len(facet.get("refs", ())) == 1
    kinds["binds the provenance to a set of two"] += len(facet.get("refs", ())) == 2
    kinds["binds a provenance outside the index"] += (
        query.source is not None and query.source not in index.datasets
    )
    decided = index_covers_slot_keys(index, query.five_tuple, query.slots.keys())
    synthetic = dataclasses.replace(
        synthetic_shape_for_identity(model.shape), sv_arc_facts_complete=False
    )
    owners = index.datasets_by_five_tuple.get(query.five_tuple, ())
    kinds["names a dataset"] += query.source is not None
    kinds["names a dataset another five-tuple belongs to"] += query.source not in (None, *owners)
    kinds["sits on an opt-in dataset's five-tuple"] += any(
        index.datasets[owner].opt_in for owner in owners
    )
    kinds["binds nothing and names no dataset"] += not query.bound and query.source is None
    kinds["index does not decide"] += not decided
    pins = elected_pins(index, query)
    kinds["leaves a pinned slot unbound"] += any(
        query.slots.get(prop) == ("unbound",) for prop in pins
    )
    found: list[str] = []
    if invariant_bindings(index, query):
        kinds["leaves unbound an invariant its canonical form binds"] += 1
        found += _canonical_disagreements(model)

    facts, index_sourced = shape_member_facts(synthetic, query)
    # A dataset named outside the five-tuple's publishers is refused whatever slots the
    # identity declares, by the mint and by Search alike, so it is checked as an index
    # election.
    refused = query.source is not None and bool(owners) and query.source not in owners
    if not decided and not refused:
        # The mint walks the graph for this identity, and Search reads its recall pool.
        if index_sourced:
            found.append("Search read the index for an identity the index does not decide")
        return found

    family, dataset = _mint(identity)
    kinds["mint resolves"] += bool(family)
    kinds["mint refuses"] += not family
    if not index_sourced or set(facts) != family:
        found.append(f"identity path: Search {len(facts)} members, mint {len(family)}")

    if identity_dataset(model) != dataset:
        found.append(f"dataset: Search {identity_dataset(model)!r}, mint {dataset!r}")

    # The digest Search may state for the identity is the digest of the mint's family.
    stated = identity_family_digest(model)
    if stated != (member_digest(family) if family else None):
        found.append(f"digest: Search states {stated!r} for a mint family of {len(family)}")

    if not wide:
        return found + _first_pass_disagreements(drawn, rng, kinds, family)

    try:
        population = set(
            options_population(model, graph=FakeMintGraphReader(arcs={})).stat_var_dcids
        )
    except EngineInputError as exc:
        assert exc.code == "population_empty", exc
        population = set()
    if population != family:
        found.append(f"options: Search {len(population)} members, mint {len(family)}")

    if family:
        partitions = measure_family_partitions(synthetic, index, get_measure_index().axes, query)
        strays = {
            dcid
            for partition in partitions
            for dcid in partition.members
            if dataset not in _provenances()[dcid]
        }
        if strays:
            found.append(f"measure family: {len(strays)} members outside {dataset!r}")

    return found + _first_pass_disagreements(drawn, rng, kinds, family)


def _canonical_disagreements(model: BaseIdentity) -> list[str]:
    """Whether an identity that leaves unbound an invariant its canonical form binds
    resolves, at the mint, another family than its canonical form, or Search states another
    dataset or digest."""
    canonical = canonicalise_identity(model)
    found: list[str] = []
    if canonical == model:
        found.append("canonical: the canonical form binds nothing")
    if _mint(model.model_dump(mode="json", exclude_none=True)) != _mint(
        canonical.model_dump(mode="json", exclude_none=True)
    ):
        found.append("canonical: the mint resolves another family for the canonical form")
    if (identity_dataset(model), identity_family_digest(model)) != (
        identity_dataset(canonical),
        identity_family_digest(canonical),
    ):
        found.append("canonical: Search states another family for the canonical form")
    return found


def _first_pass_disagreements(
    drawn: _Drawn, rng: random.Random, kinds: dict[str, int], family: frozenset[str]
) -> list[str]:
    """Whether the reader a resolver calls before an identity exists, over a recall sample,
    disagrees with the mint. A shape Search derives declares every slot key, and a named
    dataset rides in the bindings."""
    if not drawn.all_keys:
        return []
    kinds["before an identity"] += 1
    kinds["before an identity, and the mint refuses"] += not family
    shape = _first_pass_shape(drawn, rng)
    bindings = _first_pass_bindings(drawn.identity)
    # The engine carries the dataset an identity names as one source binding.
    source = named_dataset(drawn.identity.get("facets", {}))
    if source is not None:
        bindings.append(source_binding(source))
    query = shape_query(shape, bindings, drawn.identity.get("entities", {}))
    found = []
    if query != identity_query(BaseIdentity.model_validate(drawn.identity)):
        found.append("before an identity: the bindings and the identity ask different queries")
    facts, index_sourced = shape_member_facts(shape, query)
    if not index_sourced or set(facts) != family:
        found.append(f"before an identity: Search {len(facts)} members, mint {len(family)}")
    return found


def _resubmit_disagreements(drawn: _Drawn, monkeypatch, kinds: dict[str, int], *, seed: bool):
    """Post `drawn`'s identity to the resubmit door as a client does, validated and with its
    bindings as spelt, and compare each Spec it returns with the mint of that Spec's identity.

    A resubmit elects one measure partition, so the identity it returns can bind more than the
    posted one. Whatever it returns: a stated `member_digest` is the digest of the family the
    mint resolves for the returned identity, the where slot binds the posted place, and the
    returned identity names the dataset the posted one names. `seed` posts three family
    members as `resolved_stat_var_dcids`.
    """
    identity = BaseIdentity.model_validate(drawn.identity)
    canonical = canonicalise_identity(identity)
    family, dataset = _mint(canonical.model_dump(mode="json", exclude_none=True))
    rows = [row for row in drawn.rows if row.dcid in family]
    graph = _family_graph(rows)
    # The family's facet carries its dataset's provenance, as the coverage build records it.
    coverage_index = patch_coverage(
        monkeypatch,
        [(row.dcid, "f1", "country/USA", _KEN, [2020]) for row in rows],
        role_names={row.dcid: _ROLE_NAMES for row in rows},
        facet_metadata={"f1": FacetMetadata(provenance_id=dataset)},
    )
    kinds["resubmits"] += 1
    kinds["resubmits with a posted seed"] += seed
    kinds["resubmits leaving unbound an invariant the canonical form binds"] += (
        canonical is not identity
    )
    try:
        region = resolve_spec_resubmit(
            inp=ResubmitIdentityPayload(
                identity=identity,
                base_spec_id=recompute_identity(drawn.identity).spec_id,
                base_identity=identity,
                resolved_stat_var_dcids=sorted(family)[:3] if seed else None,
            ),
            graph=graph,
            sdmx=_sibling_sdmx(min(family)),
            coverage=FakeCoverageService(coverage_index).snapshot(),
            options=EngineOptions(
                materialise_graph=axis_confirming_reader(*(row.dcid for row in rows), graph=graph)
            ),
        )
    except EngineInputError as exc:
        # A provenance set needs a facet of every member on the family, and the family's
        # facets carry its one dataset, so a set naming another dataset refuses.
        bound = drawn.identity.get("facets", {}).get("provenance", {}).get("refs", ())
        if exc.code == "facet_not_available" and set(bound) - {dataset}:
            return []
        raise
    found: list[str] = []
    for spec in region.specs:
        returned = spec.definition.identity.model_dump(mode="json", exclude_none=True)
        returned_family, _returned_dataset = _mint(returned)
        digest = spec.resolved.shape.member_digest
        kinds["resubmits that state a digest"] += digest is not None
        if digest is not None and digest != member_digest(returned_family):
            found.append(
                f"resubmit: Search states a digest that is not the digest of the "
                f"{len(returned_family)} members the mint resolves"
            )
        served = {ref.dcid for ref in spec.resolved.trace.resolved_stat_vars}
        if not served <= returned_family:
            found.append(f"resubmit: {len(served - returned_family)} members outside the mint's")
        where = next(slot for slot in spec.resolved.slots if slot.key.axis == "where")
        if not isinstance(where.binding, BindingValue) or where.binding.value.ref.dcid != _KEN:
            found.append(f"resubmit: the where slot binds {where.binding!r}")
        if named_dataset(returned.get("facets", {})) != named_dataset(
            drawn.identity.get("facets", {})
        ):
            found.append("resubmit: the returned identity names another dataset")
        if invariant_bindings(get_index(), identity_query(spec.definition.identity)):
            found.append("resubmit: the returned identity is not canonical")
    return found


_KINDS = (
    "pins a StatVar",
    "binds a slot to a value",
    "binds a slot to a set of one ref",
    "binds a slot to a set of several refs",
    "names a dataset by a set of one ref",
    "binds the provenance to a set of two",
    "binds a provenance outside the index",
    "resubmits",
    "resubmits with a posted seed",
    "resubmits that state a digest",
    "resubmits leaving unbound an invariant the canonical form binds",
    "leaves a pinned slot unbound",
    "leaves unbound an invariant its canonical form binds",
    "names a dataset",
    "names a dataset another five-tuple belongs to",
    "sits on an opt-in dataset's five-tuple",
    "binds nothing and names no dataset",
    "index does not decide",
    "mint resolves",
    "mint refuses",
    "before an identity",
    "before an identity, and the mint refuses",
)


def _resubmit_sample(drawn: list[_Drawn]) -> list[int]:
    """The first `_RESUBMITS` canonical described identities, and the first
    `_RELEASING_RESUBMITS` that leave unbound an invariant their canonical form binds, each
    declaring every slot key and with a canonical form whose family the mint resolves within
    `_RESUBMIT_FAMILY_CAP` members."""
    canonical_sample: list[int] = []
    releasing_sample: list[int] = []
    for position, item in enumerate(drawn):
        if item.pinned is not None or not item.all_keys:
            continue
        identity = BaseIdentity.model_validate(item.identity)
        releasing = bool(invariant_bindings(get_index(), identity_query(identity)))
        sample = releasing_sample if releasing else canonical_sample
        if len(sample) == (_RELEASING_RESUBMITS if releasing else _RESUBMITS):
            continue
        family, _dataset = _mint(
            canonicalise_identity(identity).model_dump(mode="json", exclude_none=True)
        )
        if 0 < len(family) <= _RESUBMIT_FAMILY_CAP:
            sample.append(position)
    return sorted(canonical_sample + releasing_sample)


def test_search_and_the_mint_resolve_the_same_family_for_every_drawn_identity(monkeypatch):
    rng = random.Random(_SEED)
    drawn = _draw(rng)
    kinds = dict.fromkeys(_KINDS, 0)

    disagreements = {
        index: found
        for index, item in enumerate(drawn)
        if (found := _check(item, rng, kinds, wide=index % 2 == 0))
    }
    for order, position in enumerate(_resubmit_sample(drawn)):
        found = _resubmit_disagreements(drawn[position], monkeypatch, kinds, seed=order % 2 == 1)
        if found:
            disagreements.setdefault(position, []).extend(found)

    assert not disagreements, (
        f"{len(disagreements)} of {len(drawn)} identities disagree; first: "
        f"{next(iter(disagreements.values()))} for "
        f"{drawn[next(iter(disagreements))].identity!r}"
    )
    # The agreement is vacuous for a kind of identity the draw never produces.
    assert all(count >= 10 for count in kinds.values()), kinds


# ---------------------------------------------------------------------------
# Five-tuples whose members declare several role sets
# ---------------------------------------------------------------------------

_ROLE_IDENTITIES = 400
_ROLE_SETS: tuple[tuple[str, ...], ...] = (
    (),
    ("financingRecipient",),
    ("financingProvider",),
    ("financingProvider", "financingRecipient"),
)
_ROLE_KEYS: tuple[tuple[str, ...], ...] = (
    (),
    ("observationAbout",),
    ("destination",),
    ("origin",),
    ("origin", "destination"),
)
_ROLE_PROPERTIES = {"alpha": ("A1", "A2"), "beta": ("B1", "B2"), "gamma": ("C1",)}
_ROLE_POPULATION = "FuzzPosition"


@dataclasses.dataclass(frozen=True)
class _RoleMember:
    dcid: str
    constraints: tuple[tuple[str, str], ...]
    roles: tuple[str, ...]

    def walk_arcs(self) -> dict[str, tuple[str, ...]]:
        return {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_ROLE_POPULATION,),
            "measuredProperty": ("amount",),
            "statType": ("measuredValue",),
            "observationProperties": self.roles,
            "constraintProperties": tuple(prop for prop, _ in self.constraints),
            **{prop: (value,) for prop, value in self.constraints},
        }

    def recall_arcs(self) -> dict:
        return {
            prop: {"nodes": [{"dcid": value} for value in values]}
            for prop, values in self.walk_arcs().items()
            if values
        }


def _draw_role_case(rng: random.Random) -> tuple[list[_RoleMember], dict]:
    """A five-tuple population of mixed role sets and an identity over it."""
    role_sets = rng.sample(_ROLE_SETS, rng.choice((1, 2, 2, 3)))
    members: dict[tuple, _RoleMember] = {}
    for position in range(rng.randint(2, 9)):
        constraints = tuple(
            (prop, rng.choice(values))
            for prop, values in _ROLE_PROPERTIES.items()
            if rng.random() < 0.45
        )
        roles = rng.choice(role_sets)
        members.setdefault(
            (constraints, roles), _RoleMember(f"Fuzz_{position}", constraints, roles)
        )
    bound = {
        prop: rng.choice(values) for prop, values in _ROLE_PROPERTIES.items() if rng.random() < 0.3
    }
    identity = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": _ROLE_POPULATION,
            "measured_property": "amount",
            "stat_type": "measuredValue",
        },
        "slots": [
            {
                "property": prop,
                "binding": {"state": "value", "ref": bound[prop]}
                if prop in bound
                else {"state": "unbound"},
            }
            for prop in _ROLE_PROPERTIES
        ],
        "entities": {key: ["country/KEN"] for key in rng.choice(_ROLE_KEYS)},
        "requested_window": {"kind": "all"},
    }
    return list(members.values()), identity


def _mint_role_case(
    members: list[_RoleMember], identity: dict
) -> tuple[str, frozenset[str], frozenset[str]]:
    """The mint's outcome: its refusal or "resolved", its family and its served members."""
    graph = FakeMintGraphReader(arcs={member.dcid: member.walk_arcs() for member in members})
    try:
        resolved = resolve_definition_members(
            identity,
            spec_id="spec_fuzz",
            graph=graph,
            enumerator=MemberEnumerator(graph=graph),
            code_index=get_index(),
        )
    except RoleSignatureIncompatible:
        return "tie", frozenset(), frozenset()
    except SpecNotExportable:
        return "empty", frozenset(), frozenset()
    served = frozenset(shape.stat_var.removeprefix("dcid:") for shape in resolved.shapes)
    return "resolved", resolved.family, served


def _matches(member: _RoleMember, identity: dict) -> bool:
    constraints = dict(member.constraints)
    return all(
        constraints.get(slot["property"]) == slot["binding"]["ref"]
        for slot in identity["slots"]
        if slot["binding"]["state"] == "value"
    )


def _check_role_case(
    members: list[_RoleMember], identity: dict, kinds: dict[str, int], sample_rng: random.Random
) -> list[str]:
    model = BaseIdentity.model_validate(identity)
    assert isinstance(model.shape, IdentityShapeSchema)
    outcome, family, served = _mint_role_case(members, identity)
    role_sets = {frozenset(member.roles) for member in members}
    kinds[f"mint: {outcome}"] += 1
    kinds["several role sets"] += len(role_sets) > 1
    kinds["a role key bound"] += bool(identity["entities"])
    by_dcid = {member.dcid: member for member in members}
    family_roles = {frozenset(by_dcid[dcid].roles) for dcid in family}
    kinds["a role class left out of the family"] += outcome == "resolved" and bool(
        role_sets - family_roles
    )
    found: list[str] = []

    # The shape `derive_shapes` builds holds every member as its recall sample.
    shape = dataclasses.replace(
        synthetic_shape_for_identity(model.shape),
        slot_keys=tuple(
            SlotKeyDraft(axis="how", property_dcid=prop, label=prop) for prop in _ROLE_PROPERTIES
        ),
        sv_arc_facts={member.dcid: member.recall_arcs() for member in members},
        sv_arc_facts_complete=False,
    )
    query = shape_query(shape, _first_pass_bindings(identity), identity["entities"])
    if query != identity_query(model):
        found.append("the shape's query differs from the identity's")
    facts, index_sourced = shape_member_facts(shape, query)
    search_family = {dcid for dcid in facts if _matches(by_dcid[dcid], identity)}
    search_roles = {frozenset(by_dcid[dcid].roles) for dcid in search_family}
    if index_sourced:
        found.append("Search read the index for a five-tuple it does not carry")
    if outcome == "resolved" and search_family != family:
        found.append(f"recall: Search {sorted(search_family)}, mint {sorted(family)}")
    if outcome == "tie" and len(search_roles) < 2:
        found.append("recall: Search holds one role set where the mint refuses a tie")
    if outcome == "empty" and search_family:
        found.append(f"recall: Search {sorted(search_family)}, mint none")

    graph = FakeMintGraphReader(arcs={member.dcid: member.walk_arcs() for member in members})
    try:
        population = frozenset(options_population(model, graph=graph).stat_var_dcids)
        options_outcome = "resolved"
    except EngineInputError as exc:
        population = frozenset()
        options_outcome = {"population_empty": "empty", "role_signature_incompatible": "tie"}[
            exc.code
        ]
    if (options_outcome, population) != (outcome, served):
        found.append(
            f"options: Search {options_outcome} {sorted(population)}, "
            f"mint {outcome} {sorted(served)}"
        )
    return (
        found
        + _incomplete_sample_disagreements(members, identity, outcome, family, kinds, sample_rng)
        + _place_less_subject_disagreements(members, identity, outcome, family, kinds, sample_rng)
    )


def _place_less_subject_disagreements(
    members: list[_RoleMember],
    identity: dict,
    outcome: str,
    family: frozenset[str],
    kinds: dict[str, int],
    sample_rng: random.Random,
) -> list[str]:
    """An identity that binds no entity, on a five-tuple whose members are subject-only
    and role-bearing, answered from the subject-only shape `derive_shapes` builds. That
    shape never walks, so its recall sample, complete or not, answers only from the
    mint's family."""
    subject = [member for member in members if not member.roles]
    if identity["entities"] or not subject or len(subject) == len(members):
        return []
    sample = sample_rng.sample(subject, sample_rng.randint(1, len(subject)))
    kinds["a place-less identity on a mixed five-tuple"] += 1
    kinds["a place-less subject-only sample that misses a member"] += len(sample) < len(subject)
    model = BaseIdentity.model_validate(identity)
    assert isinstance(model.shape, IdentityShapeSchema)
    shape = dataclasses.replace(
        synthetic_shape_for_identity(model.shape),
        slot_keys=tuple(
            SlotKeyDraft(axis="how", property_dcid=prop, label=prop) for prop in _ROLE_PROPERTIES
        ),
        sv_arc_facts={member.dcid: member.recall_arcs() for member in sample},
        sv_arc_facts_complete=False,
        declared_roles=("observationAbout",),
    )
    query = shape_query(shape, _first_pass_bindings(identity), {})
    walk = FakeMintGraphReader(arcs={member.dcid: member.walk_arcs() for member in members})
    narrowed = elected_role_class_shape(shape, query, walks=RoleClassWalks(walk)).value
    facts, _index_sourced = shape_member_facts(narrowed, query)
    by_dcid = {member.dcid: member for member in members}
    answered = {dcid for dcid in facts if _matches(by_dcid[dcid], identity)}
    if outcome == "tie" and answered:
        return [f"place-less: Search {sorted(answered)} where the mint ties"]
    if outcome != "tie" and not answered <= family:
        return [f"place-less: Search {sorted(answered)}, mint {outcome} {sorted(family)}"]
    return []


def _incomplete_sample_disagreements(
    members: list[_RoleMember],
    identity: dict,
    outcome: str,
    family: frozenset[str],
    kinds: dict[str, int],
    sample_rng: random.Random,
) -> list[str]:
    """A recall sample drawn from the role-bearing members, which `derive_shapes` groups
    into one shape, may miss any of them. The shape narrowed through the mint's walk
    answers only members of the mint's family, and a resubmit posting the sample names
    every posted member the mint refuses."""
    role_bearing = [member for member in members if member.roles]
    # The sample leaves out at least one member of the five-tuple.
    largest = len(role_bearing) if len(role_bearing) < len(members) else len(role_bearing) - 1
    if largest < 1:
        return []
    sample = sample_rng.sample(role_bearing, sample_rng.randint(1, largest))
    by_dcid = {member.dcid: member for member in members}
    family_roles = {frozenset(by_dcid[dcid].roles) for dcid in family}
    kinds["an incomplete sample"] += 1
    kinds["an incomplete sample that misses the elected class"] += outcome == "resolved" and all(
        frozenset(member.roles) not in family_roles for member in sample
    )
    model = BaseIdentity.model_validate(identity)
    assert isinstance(model.shape, IdentityShapeSchema)
    shape = dataclasses.replace(
        synthetic_shape_for_identity(model.shape),
        slot_keys=tuple(
            SlotKeyDraft(axis="how", property_dcid=prop, label=prop) for prop in _ROLE_PROPERTIES
        ),
        sv_arc_facts={member.dcid: member.recall_arcs() for member in sample},
        sv_arc_facts_complete=False,
        declared_roles=tuple(dict.fromkeys(role for member in sample for role in member.roles)),
    )
    query = shape_query(shape, _first_pass_bindings(identity), identity["entities"])
    walk = FakeMintGraphReader(arcs={member.dcid: member.walk_arcs() for member in members})
    arcs_graph = FakeGraph(
        nodes={
            member.dcid: {
                "label": member.dcid,
                "type": "StatisticalVariable",
                "arcs": member.recall_arcs(),
            }
            for member in members
        },
        obs={},
        detect={},
        resolve={},
    )
    found: list[str] = []
    try:
        narrowed = elected_role_class_shape(
            shape, query, walks=RoleClassWalks(walk), arcs_graph=arcs_graph
        ).value
    except RoleSignatureError:
        if outcome != "tie":
            found.append(f"incomplete sample: Search refuses a tie, mint {outcome}")
    else:
        facts, _index_sourced = shape_member_facts(narrowed, query)
        answered = {dcid for dcid in facts if _matches(by_dcid[dcid], identity)}
        if outcome == "tie" and answered:
            found.append(f"incomplete sample: Search {sorted(answered)} where the mint ties")
        if outcome != "tie" and not answered <= family:
            found.append(
                f"incomplete sample: Search {sorted(answered)}, mint {outcome} {sorted(family)}"
            )

    if outcome == "empty":
        return found
    posted = {member.dcid: member.recall_arcs() for member in sample}
    try:
        outside = posted_members_outside_role_class(
            posted, identity_query(model), walks=RoleClassWalks(walk)
        ).value
    except RoleSignatureError:
        if outcome != "tie":
            found.append(f"posted sample: Search refuses a tie, mint {outcome}")
        return found
    if outcome == "tie":
        found.append("posted sample: Search answers a posted set the mint refuses as a tie")
    elif outside != tuple(dcid for dcid in posted if dcid not in family):
        found.append(f"posted sample: Search refuses {outside}, mint keeps {sorted(family)}")
    return found


_ROLE_KINDS = (
    "mint: resolved",
    "mint: tie",
    "mint: empty",
    "several role sets",
    "a role key bound",
    "a role class left out of the family",
    "an incomplete sample",
    "an incomplete sample that misses the elected class",
    "a place-less identity on a mixed five-tuple",
    "a place-less subject-only sample that misses a member",
)


def test_search_and_the_mint_elect_the_same_role_class_for_every_drawn_identity():
    rng = random.Random(_SEED)
    # The incomplete samples draw from their own stream, so the drawn identities stay fixed.
    sample_rng = random.Random(_SEED + 1)
    kinds = dict.fromkeys(_ROLE_KINDS, 0)
    disagreements = {}
    for index in range(_ROLE_IDENTITIES):
        members, identity = _draw_role_case(rng)
        if found := _check_role_case(members, identity, kinds, sample_rng):
            disagreements[index] = (found, identity, members)

    assert not disagreements, (
        f"{len(disagreements)} of {_ROLE_IDENTITIES} identities disagree; first: "
        f"{next(iter(disagreements.values()))}"
    )
    # The agreement is vacuous for a kind of identity the draw never produces.
    assert all(count >= 10 for count in kinds.values()), kinds
