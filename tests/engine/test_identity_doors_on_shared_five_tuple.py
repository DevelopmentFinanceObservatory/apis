"""The identity doors elect one dataset for a posted identity on a five-tuple several
datasets publish, and read that dataset's pins as implied.

The Datasets mint (`resolve_definition_members`), the options door (`options_population`)
and the Search resubmit (`canonicalise_identity`, `refuse_source_outside_family`) each read
the identity through `code_index.members`. On a synthetic index where two default datasets
share a five-tuple and one row, and an opt-in dataset alone publishes another, every door
serves the same dataset and the same members whether the identity binds that dataset's pins
or leaves them unbound, and both forms canonicalise to one identity.
"""

from __future__ import annotations

from typing import cast

import pytest

from code_index.query import CodeIndex
from code_index.types import CodeSystemDeclaration, DatasetDeclaration, IndexStamp, StatVarRow
from datasets_api.adapter import resolve_definition_members
from materialise.errors import FacetNotAvailableError
from materialise.facets import ElectedDataset
from search.engine.clients.graph import EngineGraphClient
from search.engine.errors import EngineInputError
from search.engine.options import population as population_module
from search.engine.options.population import options_population
from search.engine.options.resolve import resolve_options_target
from search.engine.retrieval import bound_values as bound_values_module
from search.engine.retrieval.bound_values import (
    canonicalise_identity,
    refuse_source_outside_family,
)
from search.engine.spec_id import mint_identity
from search.models import IdentityBindingUnbound, OptionsFor, OptionsRequest
from statvar_members import MemberEnumerator
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.statvar_members.conftest import FakeMemberArcReader

_FIRST = "provenance/First"
_SECOND = "provenance/Second"
_OPT_IN = "provenance/OnRequest"

_SHARED = ("TestFlow", "amount")
_OPT_IN_ONLY = ("TestFlow", "lineAmount")

_PROPERTIES = ("basis", "flow", "line", "sector")


def _row(
    dcid: str, constraints: dict[str, str], *provenances: str, five_tuple=_SHARED
) -> StatVarRow:
    population_type, measured_property = five_tuple
    return StatVarRow(
        dcid=dcid,
        population_type=population_type,
        measured_property=measured_property,
        stat_type="measuredValue",
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=(),
        constraint_properties=constraints,
        provenances=provenances,
    )


def _dataset(
    provenance: str, *, rank: int, invariants: dict[str, str], opt_in: bool = False
) -> DatasetDeclaration:
    return DatasetDeclaration(
        provenance=provenance,
        label=provenance,
        synonyms=(),
        opt_in=opt_in,
        absent_observation="zero",
        rank=rank,
        invariants=invariants,
    )


def _code_system(code_system: str) -> CodeSystemDeclaration:
    return CodeSystemDeclaration(
        code_system=code_system,
        hierarchical=False,
        disjoint_from_children=False,
        summable=True,
        absent_reading="not_declared",
        plain_name=code_system,
    )


def _index() -> CodeIndex:
    """`_FIRST` (rank 0) and `_SECOND` (rank 1) publish the shared five-tuple and both
    record `shared_a`. `_FIRST`'s rows all hold `basis` G and `flow` F1, `_SECOND`'s
    `basis` G and `line` L1, and each declares its own as invariants. The opt-in `_OPT_IN`
    alone publishes its five-tuple, and declares `basis` G, which its rows all hold."""
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="test",
        ),
        declarations=tuple(_code_system(prop) for prop in _PROPERTIES),
        codes=(),
        statvars=(
            _row(
                "shared_a",
                {"basis": "G", "flow": "F1", "line": "L1", "sector": "A"},
                _FIRST,
                _SECOND,
            ),
            _row("first_b", {"basis": "G", "flow": "F1", "sector": "B"}, _FIRST),
            _row("second_c", {"basis": "G", "flow": "F2", "line": "L1", "sector": "C"}, _SECOND),
            _row("opt_a", {"basis": "G", "sector": "A"}, _OPT_IN, five_tuple=_OPT_IN_ONLY),
            _row("opt_b", {"basis": "G", "sector": "B"}, _OPT_IN, five_tuple=_OPT_IN_ONLY),
        ),
        datasets=(
            _dataset(_FIRST, rank=0, invariants={"basis": "G", "flow": "F1"}),
            _dataset(_SECOND, rank=1, invariants={"basis": "G", "line": "L1"}),
            _dataset(_OPT_IN, rank=2, invariants={"basis": "G"}, opt_in=True),
        ),
    )


# The resubmit door's path to the posted identity's provenance facet.
_RESUBMIT_POINTER = "/input/identity/facets/provenance"


@pytest.fixture
def index(monkeypatch) -> CodeIndex:
    """`_index` wherever the options door and the resubmit read the packaged index."""
    index = _index()
    for module in (bound_values_module, population_module):
        monkeypatch.setattr(module, "get_index", lambda: index)
    return index


def _document(five_tuple: tuple[str, str], source: str | None = None, **bound: str | None) -> dict:
    """An identity on `five_tuple` declaring each property in `bound`, bound to its value or
    unbound for `None`, with the provenance facet bound to `source` when given."""
    population_type, measured_property = five_tuple
    document: dict = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": population_type,
            "measured_property": measured_property,
            "stat_type": "measuredValue",
        },
        "slots": [
            {
                "property": prop,
                "binding": {"state": "unbound"}
                if value is None
                else {"state": "value", "ref": value},
            }
            for prop, value in sorted(bound.items())
        ],
        "entities": {},
        "requested_window": {"kind": "all"},
    }
    if source is not None:
        document["facets"] = {"provenance": {"state": "value", "ref": source}}
    return document


def _mint(document: dict, index: CodeIndex) -> dict[str, ElectedDataset | None]:
    """Each member the mint resolves for `document`, mapped to the dataset it carries."""
    graph = FakeMintGraphReader(arcs={})
    resolved = resolve_definition_members(
        document,
        spec_id="spec_test",
        graph=graph,
        enumerator=MemberEnumerator(graph=graph),
        code_index=index,
    )
    assert sum(graph.calls.values()) == 0
    return {shape.stat_var.removeprefix("dcid:"): shape.dataset for shape in resolved.shapes}


def _options(document: dict) -> dict[str, ElectedDataset]:
    """Each member of the options door's population for `document`, mapped to its dataset."""
    _, identity = mint_identity(document)
    population = options_population(identity, FakeMemberArcReader(arcs={}))
    assert population.source == "index"
    assert set(population.datasets) == set(population.stat_var_dcids)
    return dict(population.datasets)


def _bound_by_canonical(document: dict) -> set[str]:
    """The slots `document` leaves unbound that its canonical form binds."""
    _, posted = mint_identity(document)
    canonical = canonicalise_identity(posted)
    before = {slot.property: slot.binding for slot in posted.slots}
    return {
        slot.property
        for slot in canonical.slots
        if isinstance(before[slot.property], IdentityBindingUnbound)
        and not isinstance(slot.binding, IdentityBindingUnbound)
    }


def _canonical_spec_id(document: dict) -> str:
    _, posted = mint_identity(document)
    return mint_identity(canonicalise_identity(posted).model_dump(mode="json", exclude_none=True))[
        0
    ]


_BOTH = (_FIRST, _SECOND)


@pytest.mark.parametrize(
    ("document", "members", "bound"),
    [
        (
            _document(_SHARED, basis="G", flow="F1", line=None, sector=None),
            {
                "shared_a": ElectedDataset(provenance=_FIRST, member_provenances=_BOTH),
                "first_b": ElectedDataset(provenance=_FIRST, member_provenances=(_FIRST,)),
            },
            set(),
        ),
        (
            _document(_SHARED, _FIRST, basis=None, flow=None, line=None, sector=None),
            {
                "shared_a": ElectedDataset(provenance=_FIRST, member_provenances=_BOTH),
                "first_b": ElectedDataset(provenance=_FIRST, member_provenances=(_FIRST,)),
            },
            {"basis", "flow"},
        ),
        (
            _document(_SHARED, _SECOND, basis=None, flow=None, line=None, sector=None),
            {
                "shared_a": ElectedDataset(provenance=_SECOND, member_provenances=_BOTH),
                "second_c": ElectedDataset(provenance=_SECOND, member_provenances=(_SECOND,)),
            },
            {"basis", "line"},
        ),
        (
            _document(_OPT_IN_ONLY, basis="G", sector=None),
            {
                "opt_a": ElectedDataset(provenance=_OPT_IN, member_provenances=(_OPT_IN,)),
                "opt_b": ElectedDataset(provenance=_OPT_IN, member_provenances=(_OPT_IN,)),
            },
            set(),
        ),
    ],
    ids=["unnamed-elects-first", "named-first", "named-second", "opt-in-alone-unnamed"],
)
def test_the_mint_and_the_options_door_serve_one_elected_dataset(index, document, members, bound):
    """A row both datasets record joins the family of the one serving the identity and
    carries both. An unnamed identity on a five-tuple only an opt-in dataset publishes
    elects that dataset. A named identity's canonical form binds each invariant of that
    dataset it leaves unbound; an unnamed identity keeps the bindings it was posted with.
    Either form serves the same members."""
    _, identity = mint_identity(document)
    canonical = canonicalise_identity(identity).model_dump(mode="json", exclude_none=True)

    assert _mint(document, index) == members
    assert _options(document) == members
    assert _bound_by_canonical(document) == bound
    assert _mint(canonical, index) == members
    assert _options(canonical) == members
    refuse_source_outside_family(identity, pointer=_RESUBMIT_POINTER)


@pytest.mark.parametrize(
    ("unbound", "bound", "members"),
    [
        (
            _document(_SHARED, _FIRST, basis="G", flow=None, line=None, sector=None),
            _document(_SHARED, _FIRST, basis="G", flow="F1", line=None, sector=None),
            {"shared_a", "first_b"},
        ),
        (
            _document(_SHARED, _SECOND, basis="G", flow=None, line=None, sector=None),
            _document(_SHARED, _SECOND, basis="G", flow=None, line="L1", sector=None),
            {"shared_a", "second_c"},
        ),
        (
            _document(_OPT_IN_ONLY, _OPT_IN, basis=None, sector=None),
            _document(_OPT_IN_ONLY, _OPT_IN, basis="G", sector=None),
            {"opt_a", "opt_b"},
        ),
    ],
    ids=["named-first-pins-flow", "named-second-pins-line", "named-opt-in-pins-basis"],
)
def test_an_identity_posted_with_or_without_a_pin_reaches_one_family_and_spec_id(
    index, unbound, bound, members
):
    """The mint and the options door serve an identity naming its dataset that leaves an
    invariant of it unbound the family they serve with it bound, and the two canonicalise
    to one spec_id. `_SECOND` leaves `flow` open, so naming it implies only `line` and
    `basis`."""
    assert set(_mint(unbound, index)) == set(_mint(bound, index)) == members
    assert _options(unbound) == _options(bound)
    assert _canonical_spec_id(unbound) == _canonical_spec_id(bound)


@pytest.mark.parametrize(
    "slots",
    [{"basis": "G", "sector": None}, {"basis": "G"}],
    ids=["slot-keys-cover-a-row", "slot-keys-cover-no-row"],
)
def test_a_named_dataset_that_does_not_publish_the_five_tuple_refuses_at_every_door(index, slots):
    """`_FIRST` does not publish the opt-in five-tuple. The mint, the resubmit and the
    options door refuse it `facet_not_available`, naming `_OPT_IN`, whether or not the
    identity's slot keys cover an index row. An identity that omits `sector` is refused
    before the graph walk. Nothing is elected, so the `basis` it binds to `_OPT_IN`'s pin
    stays bound in its canonical form."""
    document = _document(_OPT_IN_ONLY, _FIRST, **slots)
    _, identity = mint_identity(document)

    with pytest.raises(FacetNotAvailableError) as minted:
        _mint(document, index)
    with pytest.raises(EngineInputError) as resubmitted:
        refuse_source_outside_family(identity, pointer=_RESUBMIT_POINTER)
    with pytest.raises(EngineInputError) as offered:
        resolve_options_target(
            OptionsRequest(identity=identity, **{"for": OptionsFor(role="observationAbout")}),
            graph=cast(EngineGraphClient, None),
            materialise_graph=FakeMintGraphReader(arcs={}),
        )

    assert _OPT_IN in str(minted.value)
    assert resubmitted.value.code == offered.value.code == "facet_not_available"
    assert _OPT_IN in str(resubmitted.value)
    assert _OPT_IN in str(offered.value)
    assert offered.value.pointer == "/identity/facets/provenance"
    assert canonicalise_identity(identity) is identity
