"""The mint resolves one role class for an identity whose five-tuple holds several.

A member lacking a role its sibling carries is the aggregate over that role, so a
recipient-only debt total answers a recipient and its by-creditor variant answers a named
creditor. `RoleSignatureIncompatible` is left for role sets the election cannot order.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.types import DatasetDeclaration, IndexStamp, StatVarRow
from datasets_api.adapter import member_resolution_key, resolve_definition_members
from datasets_api.codes_materialise import DefinitionNotReproducible, RoleSignatureIncompatible
from datasets_api.kinds.spec_id import _check_own_roles, _resolve_members
from datasets_api.mint_models import Scope
from statvar_members import MemberEnumerator

from ._graph_doubles import FakeMintGraphReader
from .test_resolve_definition_members_guards import (
    _root_and_gendered_arcs,
    _unbound_gender_identity,
    _WithheldFromWalkGraphReader,
)

_SPEC_ID = "spec_test"
_RECIPIENT = ("financingRecipient",)
_PAIR = ("financingProvider", "financingRecipient")
_PROVIDER = ("financingProvider",)


def _arcs(roles: tuple[str, ...], *, guarantee: str | None = None) -> dict[str, tuple[str, ...]]:
    arcs: dict[str, tuple[str, ...]] = {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("FinancialPosition",),
        "measuredProperty": ("amount",),
        "statType": ("measuredValue",),
        "measurementQualifier": ("Nominal",),
        "observationProperties": roles,
        "constraintProperties": ("positionType",),
        "positionType": ("AmountOutstanding",),
    }
    if guarantee is not None:
        arcs["constraintProperties"] = ("positionType", "guarantee")
        arcs["guarantee"] = (guarantee,)
    return arcs


_DEBT = {
    "Debt": _arcs(_RECIPIENT),
    "Debt_Guaranteed": _arcs(_RECIPIENT, guarantee="Public"),
    "Debt_ByCreditor": _arcs(_PAIR),
    "Debt_Guaranteed_ByCreditor": _arcs(_PAIR, guarantee="Public"),
}


def _identity(entities: Mapping[str, list[str]]) -> dict:
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "FinancialPosition",
            "measured_property": "amount",
            "stat_type": "measuredValue",
            "measurement_qualifier": "Nominal",
        },
        "slots": [
            {"property": "guarantee", "binding": {"state": "unbound"}},
            {"property": "positionType", "binding": {"state": "unbound"}},
        ],
        "entities": dict(entities),
        "requested_window": {"kind": "all"},
    }


def _resolve(identity: Mapping[str, object], arcs=_DEBT, **kwargs: object):
    graph = FakeMintGraphReader(arcs=arcs)
    return resolve_definition_members(
        identity,
        spec_id=_SPEC_ID,
        graph=graph,
        enumerator=MemberEnumerator(graph=graph),
        code_index=get_index(),
        **kwargs,  # ty: ignore[invalid-argument-type]
    )


def _members(resolved) -> list[tuple[str, tuple[str, ...]]]:
    return [(shape.stat_var, shape.declared_roles) for shape in resolved.shapes]


def test_a_recipient_alone_mints_the_recipient_only_total():
    resolved = _resolve(_identity({"destination": ["country/KEN"]}))

    assert _members(resolved) == [("dcid:Debt", _RECIPIENT)]
    assert resolved.family == {"Debt", "Debt_Guaranteed"}


def test_a_named_creditor_mints_the_by_creditor_rows():
    resolved = _resolve(_identity({"destination": ["country/KEN"], "origin": ["country/CHN"]}))

    assert _members(resolved) == [("dcid:Debt_ByCreditor", _PAIR)]
    assert resolved.family == {"Debt_ByCreditor", "Debt_Guaranteed_ByCreditor"}


def test_no_entity_mints_the_recipient_only_total():
    assert _members(_resolve(_identity({}))) == [("dcid:Debt", _RECIPIENT)]


def test_role_sets_the_election_cannot_order_refuse_role_signature_incompatible():
    arcs = {"Debt": _arcs(_RECIPIENT), "Debt_Line": _arcs(_PROVIDER)}

    with pytest.raises(RoleSignatureIncompatible) as exc_info:
        _resolve(_identity({}), arcs=arcs)

    assert exc_info.value.roles_by_stat_var == {"Debt": _RECIPIENT, "Debt_Line": _PROVIDER}
    assert _members(_resolve(_identity({"origin": ["country/FRA"]}), arcs=arcs)) == [
        ("dcid:Debt_Line", _PROVIDER)
    ]


def test_the_role_keys_argument_replaces_the_identitys_own():
    resolved = _resolve(
        _identity({"destination": ["country/KEN"]}), role_keys={"origin", "destination"}
    )

    assert _members(resolved) == [("dcid:Debt_ByCreditor", _PAIR)]


def test_the_role_keys_enter_the_member_resolution_key():
    index = get_index()
    recipient = member_resolution_key(_identity({"destination": ["country/KEN"]}), code_index=index)

    assert recipient == member_resolution_key(
        _identity({"destination": ["country/GHA"]}), code_index=index
    )
    assert recipient != member_resolution_key(
        _identity({"destination": ["country/KEN"], "origin": ["country/CHN"]}), code_index=index
    )


def test_a_scope_that_names_a_creditor_resolves_the_by_creditor_rows():
    """Scope entities supply the role keys, so the request serves the members the surgered
    identity resolves."""
    graph = FakeMintGraphReader(arcs=_DEBT)
    scope = Scope(entities={"origin": ["country/CHN"], "destination": ["country/KEN"]})

    shapes = _resolve_members(
        _identity({"destination": ["country/KEN"]}),
        spec_id=_SPEC_ID,
        graph=graph,
        enumerator=MemberEnumerator(graph=graph),
        code_index=get_index(),
        scope=scope,
    )

    assert [shape.stat_var for shape in shapes] == ["dcid:Debt_ByCreditor"]


class _Deps:
    def __init__(self, arcs) -> None:
        self.graph = FakeMintGraphReader(arcs=arcs)
        self.enumerator = MemberEnumerator(graph=self.graph)
        self.code_index = get_index()


def test_an_identitys_own_roles_are_checked_against_its_own_role_class():
    """A scope that drops the creditor serves the recipient-only total. The stored identity
    binds a creditor, which its own by-creditor members declare, so it stays reproducible."""
    deps = _Deps(_DEBT)
    identity = _identity({"destination": ["country/KEN"], "origin": ["country/CHN"]})
    scope = Scope(entities={"destination": ["country/KEN"]})
    shapes = _resolve_members(
        identity,
        spec_id=_SPEC_ID,
        graph=deps.graph,
        enumerator=deps.enumerator,
        code_index=deps.code_index,
        scope=scope,
    )
    assert [shape.stat_var for shape in shapes] == ["dcid:Debt"]

    _check_own_roles(identity, spec_id=_SPEC_ID, scope=scope, shapes=shapes, deps=deps)  # ty: ignore[invalid-argument-type]

    with pytest.raises(DefinitionNotReproducible):
        _check_own_roles(
            _identity({"observationAbout": ["country/KEN"]}),
            spec_id=_SPEC_ID,
            scope=scope,
            shapes=shapes,
            deps=deps,  # ty: ignore[invalid-argument-type]
        )


def _truncated_walk_deps(arcs) -> _Deps:
    deps = _Deps({})
    deps.graph = _WithheldFromWalkGraphReader(arcs=arcs, withheld=frozenset({"Count_Person"}))
    deps.enumerator = MemberEnumerator(graph=deps.graph)
    return deps


def test_a_place_less_identity_a_posted_root_proved_complete_mints_under_a_scope_that_binds_a_place():
    """The scope binds a subject and the identity binds nothing, so the identity's roles hold
    against the scope's own members and nothing resolves a second time."""
    deps = _truncated_walk_deps(_root_and_gendered_arcs())
    identity = _unbound_gender_identity()
    scope = Scope(entities={"observationAbout": ["country/KEN"]})
    shapes = _resolve_members(
        identity,
        spec_id=_SPEC_ID,
        graph=deps.graph,
        enumerator=deps.enumerator,
        code_index=deps.code_index,
        selection=("Count_Person",),
        scope=scope,
    )
    walks = deps.graph.calls["node_subjects"]

    _check_own_roles(
        identity,
        spec_id=_SPEC_ID,
        scope=scope,
        shapes=shapes,
        deps=deps,  # ty: ignore[invalid-argument-type]
        selection=("Count_Person",),
    )

    assert [shape.stat_var for shape in shapes] == ["dcid:Count_Person"]
    assert deps.graph.calls["node_subjects"] == walks


# ---------------------------------------------------------------------------
# The index path
# ---------------------------------------------------------------------------

_PROVENANCE = "provenance/Debt"


def _row(dcid: str, roles: tuple[str, ...], **constraints: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="FinancialPosition",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier="Nominal",
        measurement_denominator=None,
        observation_properties=roles,
        constraint_properties={"positionType": "AmountOutstanding", **constraints},
        provenances=(_PROVENANCE,),
    )


def _debt_index(*rows: StatVarRow) -> CodeIndex:
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="test",
        ),
        declarations=(),
        codes=(),
        statvars=rows,
        datasets=(
            DatasetDeclaration(
                provenance=_PROVENANCE,
                label="Debt",
                synonyms=(),
                opt_in=False,
                absent_observation="zero",
                rank=0,
            ),
        ),
    )


def _resolve_index(identity: Mapping[str, object], index: CodeIndex):
    graph = FakeMintGraphReader(arcs={})
    return resolve_definition_members(
        identity,
        spec_id=_SPEC_ID,
        graph=graph,
        enumerator=MemberEnumerator(graph=graph),
        code_index=index,
    )


_DEBT_ROWS = (
    _row("Debt", _RECIPIENT),
    _row("Debt_Guaranteed", _RECIPIENT, guarantee="Public"),
    _row("Debt_ByCreditor", _PAIR),
    _row("Debt_Guaranteed_ByCreditor", _PAIR, guarantee="Public"),
)


def test_the_index_path_elects_the_same_role_classes_as_the_walk():
    index = _debt_index(*_DEBT_ROWS)

    recipient = _resolve_index(_identity({"destination": ["country/KEN"]}), index)
    creditor = _resolve_index(
        _identity({"destination": ["country/KEN"], "origin": ["country/CHN"]}), index
    )

    assert _members(recipient) == [("dcid:Debt", _RECIPIENT)]
    assert recipient.family == {"Debt", "Debt_Guaranteed"}
    assert _members(creditor) == [("dcid:Debt_ByCreditor", _PAIR)]
    assert creditor.family == {"Debt_ByCreditor", "Debt_Guaranteed_ByCreditor"}


def test_the_index_path_refuses_role_sets_the_election_cannot_order():
    index = _debt_index(_row("Debt", _RECIPIENT), _row("Debt_Line", _PROVIDER))

    with pytest.raises(RoleSignatureIncompatible):
        _resolve_index(_identity({}), index)
