"""Tests described-arm member enumeration and how its errors map to the datasets API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
from pydantic import BaseModel, ValidationError

from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.types import IndexStamp, StatVarRow
from datasets_api.adapter import check_reproducible, resolve_definition_members
from datasets_api.codes_materialise import (
    DefinitionNotReproducible,
    MemberNotResolvable,
    RoleSignatureIncompatible,
    SpecNotExportable,
)
from datasets_api.kinds import collection_id as collection_id_kind
from datasets_api.mint_models import SpecIdInput, StatVarSetInput
from datasets_api.models import MAX_COLLECTION_MEMBERS
from materialise import graph as materialise_graph
from materialise.errors import OverCeilingError
from statvar_members import MemberEnumerator
from statvar_members.enumerate import CANDIDATE_CAP
from statvar_members.types import UNRESOLVED_NODE

from ._graph_doubles import FakeMintGraphReader

_SPEC_ID = "spec_test"
_MEASURED_PROPERTY = "amount"
_POPULATION_TYPE = "DevelopmentFinance"
_SCHEME_PROPERTY = "DevelopmentFinanceScheme"
_ODA_GRANTS = "ODAGrantsAmount"
_ODA_LOANS = "ODALoansAmount"


def _member_arcs(
    *, scheme: str, roles: tuple[str, ...] = ("recipient",)
) -> dict[str, tuple[str, ...]]:
    return {
        "typeOf": ("StatisticalVariable",),
        "populationType": (_POPULATION_TYPE,),
        "measuredProperty": (_MEASURED_PROPERTY,),
        "observationProperties": roles,
        "constraintProperties": (_SCHEME_PROPERTY,),
        _SCHEME_PROPERTY: (scheme,),
    }


def _two_member_identity(**shape_overrides: object) -> dict:
    shape: dict[str, object] = {
        "kind": "statvar_schema",
        "population_type": _POPULATION_TYPE,
        "measured_property": _MEASURED_PROPERTY,
    }
    shape.update(shape_overrides)
    return {
        "identity_version": "spec1",
        "shape": shape,
        "slots": [
            {
                "property": _SCHEME_PROPERTY,
                "binding": {"state": "set", "refs": ["ODAGrants", "ODALoans"]},
            }
        ],
        "entities": {},
        "requested_window": {"kind": "all"},
    }


def _two_member_graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _ODA_GRANTS: _member_arcs(scheme="ODAGrants"),
            _ODA_LOANS: _member_arcs(scheme="ODALoans"),
        }
    )


def _resolve(
    identity: Mapping[str, object],
    *,
    graph: FakeMintGraphReader,
    seed: Sequence[str] | None = None,
):
    return resolve_definition_members(
        identity,
        spec_id=_SPEC_ID,
        graph=graph,
        enumerator=MemberEnumerator(graph=graph),
        code_index=get_index(),
        seed=seed,
    ).shapes


def test_two_members_sharing_no_dominating_series_both_survive():
    """Both members survive because neither size-1 constraint set dominates the other."""
    shapes = _resolve(_two_member_identity(), graph=_two_member_graph())

    assert sorted(shape.stat_var for shape in shapes) == [
        "dcid:ODAGrantsAmount",
        "dcid:ODALoansAmount",
    ]


def test_empty_group_refuses_spec_not_exportable():
    # The graph has no member with this measured property, so the reverse query is empty.
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "statvar_schema", "measured_property": "nobody_asserts_this"},
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
    }
    graph = _two_member_graph()

    with pytest.raises(SpecNotExportable) as exc_info:
        _resolve(identity, graph=graph)

    assert exc_info.value.spec_id == _SPEC_ID
    assert "enumerates no member" in str(exc_info.value)


def test_member_enumeration_too_large_refuses_over_ceiling():
    # 701 candidates exceed the 700-member cap before antichain reduction, but stay under
    # the 4,000-candidate page cap.
    schemes = [f"Scheme{i}" for i in range(701)]
    graph = FakeMintGraphReader(
        arcs={f"Amount{i}": _member_arcs(scheme=scheme) for i, scheme in enumerate(schemes)}
    )
    identity = _two_member_identity()
    identity["slots"] = [
        {"property": _SCHEME_PROPERTY, "binding": {"state": "set", "refs": schemes}}
    ]

    with pytest.raises(OverCeilingError) as exc_info:
        _resolve(identity, graph=graph)

    detail = str(exc_info.value)
    assert "701" in detail
    assert "member_cap=700" in detail
    assert "slot binding" in detail
    assert "GET /api/datasets/{spec_id}" in detail
    assert "definition.identity" in detail
    assert "statvar_set" not in detail
    assert "unknown" not in detail


def test_truncated_reverse_query_refuses_over_ceiling_not_zero_members():
    # The 4,001 raw candidates exceed the 4,000 page cap. Each scheme also has 2,001
    # `vintage` variants outside the identity, leaving 2 confirmed members under the 700 cap.
    # The set-bound slot prevents the truncated page from proving completeness.
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        "ODAGrantsAmount": _member_arcs(scheme="ODAGrants"),
        "ODALoansAmount": _member_arcs(scheme="ODALoans"),
    }
    for scheme in ("ODAGrants", "ODALoans"):
        for i in range(2_001):
            arcs[f"{scheme}AmountVintage{i:04d}"] = {
                "typeOf": ("StatisticalVariable",),
                "populationType": (_POPULATION_TYPE,),
                "measuredProperty": (_MEASURED_PROPERTY,),
                "observationProperties": ("recipient",),
                "constraintProperties": (_SCHEME_PROPERTY, "vintage"),
                _SCHEME_PROPERTY: (scheme,),
                "vintage": (f"Vintage{i:04d}",),
            }
    graph = FakeMintGraphReader(arcs=arcs)
    identity = _two_member_identity()

    with pytest.raises(OverCeilingError) as exc_info:
        _resolve(identity, graph=graph)

    assert exc_info.value.ceiling_kind == "candidate_window"
    assert exc_info.value.count == CANDIDATE_CAP
    assert exc_info.value.cap == CANDIDATE_CAP
    detail = str(exc_info.value)
    # No posted root proves a set-bound family complete, so the remedy names `statvar_set`
    # alone, in the `dcid:` form `statvars` requires.
    assert "resolved_stat_var_dcids" not in detail
    assert 'kind "statvar_set"' in detail
    assert "written dcid:<ref>" in detail
    assert "a different, pinned spec" in detail
    assert str(MAX_COLLECTION_MEMBERS) in detail
    assert "member_cap=" not in detail
    assert str(CANDIDATE_CAP) in detail


def _value_bound_broad_family() -> tuple[dict, dict[str, dict[str, tuple[str, ...]]]]:
    """An identity binding `gender` to one code, over a family whose root sits past the
    candidate window behind more `vintage` variants than the window holds."""
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        f"Count_Female_Vintage{i:04d}": {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POPULATION_TYPE,),
            "measuredProperty": (_MEASURED_PROPERTY,),
            "observationProperties": ("recipient",),
            "constraintProperties": ("gender", "vintage"),
            "gender": ("Female",),
            "vintage": (f"Vintage{i:04d}",),
        }
        for i in range(CANDIDATE_CAP + 1)
    }
    # Sorts after every variant, so the truncated page never reaches it.
    arcs["Count_Female_Zroot"] = {
        "typeOf": ("StatisticalVariable",),
        "populationType": (_POPULATION_TYPE,),
        "measuredProperty": (_MEASURED_PROPERTY,),
        "observationProperties": ("recipient",),
        "constraintProperties": ("gender",),
        "gender": ("Female",),
    }
    identity = _two_member_identity()
    identity["slots"] = [{"property": "gender", "binding": {"state": "value", "ref": "Female"}}]
    identity["entities"] = {"recipient": ["country/KEN"]}
    return identity, arcs


def test_candidate_window_remedy_leads_with_the_seed_when_a_root_can_prove_the_family():
    identity, arcs = _value_bound_broad_family()

    with pytest.raises(OverCeilingError) as exc_info:
        _resolve(identity, graph=FakeMintGraphReader(arcs=arcs))

    assert exc_info.value.ceiling_kind == "candidate_window"
    detail = str(exc_info.value)
    # The seed remedy comes first and names the bare form `resolved_stat_var_dcids` takes.
    # `statvar_set` follows in the `dcid:` form `statvars` requires.
    seed_remedy = detail.index("this same spec_id again with resolved_stat_var_dcids")
    statvar_set_remedy = detail.index('kind "statvar_set"')
    assert seed_remedy < statvar_set_remedy
    assert "(no dcid: prefix)" in detail[seed_remedy:statvar_set_remedy]
    assert "written dcid:<ref>" in detail[statvar_set_remedy:]
    assert "a different, pinned spec" in detail[statvar_set_remedy:]


def test_candidate_window_remedy_names_no_seed_no_posted_root_could_prove():
    """With no entity bound, only a subject-only root elects the role class, and this
    family's members all declare a recipient. A posted root is refused again, so the
    remedy names `statvar_set` alone."""
    identity, arcs = _value_bound_broad_family()
    identity["entities"] = {}

    with pytest.raises(OverCeilingError) as unseeded:
        _resolve(identity, graph=FakeMintGraphReader(arcs=arcs))
    with pytest.raises(OverCeilingError):
        _resolve(identity, graph=FakeMintGraphReader(arcs=arcs), seed=["Count_Female_Zroot"])

    assert "resolved_stat_var_dcids" not in str(unseeded.value)
    assert 'kind "statvar_set"' in str(unseeded.value)


def test_candidate_window_remedy_leaves_the_seed_out_where_the_door_takes_none():
    identity, arcs = _value_bound_broad_family()
    graph = FakeMintGraphReader(arcs=arcs)

    with pytest.raises(OverCeilingError) as exc_info:
        resolve_definition_members(
            identity,
            spec_id=_SPEC_ID,
            graph=graph,
            enumerator=MemberEnumerator(graph=graph),
            code_index=get_index(),
            accepts_seed=False,
        )

    detail = str(exc_info.value)
    assert "resolved_stat_var_dcids" not in detail
    assert 'kind "statvar_set"' in detail


def test_a_collection_member_past_the_candidate_window_is_not_advised_to_post_a_seed():
    identity, arcs = _value_bound_broad_family()
    graph = FakeMintGraphReader(arcs=arcs)

    with pytest.raises(OverCeilingError) as exc_info:
        collection_id_kind._resolve_member(
            identity,
            spec_id=_SPEC_ID,
            graph=graph,
            enumerator=MemberEnumerator(graph=graph),
            code_index=get_index(),
        )

    assert exc_info.value.ceiling_kind == "candidate_window"
    assert "resolved_stat_var_dcids" not in str(exc_info.value)


def test_the_seed_the_candidate_window_remedy_names_mints_the_refused_family():
    identity, arcs = _value_bound_broad_family()

    shapes = _resolve(identity, graph=FakeMintGraphReader(arcs=arcs), seed=["Count_Female_Zroot"])

    assert [shape.stat_var for shape in shapes] == ["dcid:Count_Female_Zroot"]
    # The root proves the served member complete, and the candidates past the window stay
    # unread, so the walk does not state the family.
    assert shapes[0].family is None


@pytest.mark.parametrize(
    ("model", "field", "ref", "accepted"),
    [
        (SpecIdInput, "resolved_stat_var_dcids", "Count_Person_Female", True),
        (SpecIdInput, "resolved_stat_var_dcids", "dcid:Count_Person_Female", False),
        (StatVarSetInput, "statvars", "dcid:Count_Person_Female", True),
        (StatVarSetInput, "statvars", "Count_Person_Female", False),
    ],
)
def test_each_input_accepts_only_the_dcid_form_the_candidate_window_remedy_names(
    model: type[BaseModel], field: str, ref: str, accepted: bool
):
    body = {field: [ref]} | ({"spec_id": "spec_a"} if model is SpecIdInput else {})

    if accepted:
        assert getattr(model.model_validate(body), field) == [ref]
    else:
        with pytest.raises(ValidationError):
            model.model_validate(body)


def test_truncated_unproven_page_with_one_confirmed_member_still_refuses_over_ceiling():
    """A truncated one-member result can cover only part of the identity population, so it
    reports `candidate_window`.

    The `vintage` siblings leave one confirmed member under the cap while the window exceeds
    4,000 candidates.
    """
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        "AmountODAGrants": _member_arcs(scheme="ODAGrants")
    }
    for i in range(CANDIDATE_CAP + 1):
        arcs[f"AmountVintage{i:04d}"] = {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POPULATION_TYPE,),
            "measuredProperty": (_MEASURED_PROPERTY,),
            "observationProperties": ("recipient",),
            "constraintProperties": ("vintage",),
            "vintage": (f"Vintage{i:04d}",),
        }
    graph = FakeMintGraphReader(arcs=arcs)
    identity = _two_member_identity()
    identity["slots"] = [
        {
            "property": _SCHEME_PROPERTY,
            "binding": {"state": "set", "refs": ["ODAGrants", "ODALoans"]},
        }
    ]

    with pytest.raises(OverCeilingError) as exc_info:
        _resolve(identity, graph=graph)

    assert exc_info.value.ceiling_kind == "candidate_window"
    assert exc_info.value.count == CANDIDATE_CAP
    assert exc_info.value.cap == CANDIDATE_CAP
    # The refusal detail reports the candidate window only.
    assert "AmountODAGrants" not in str(exc_info.value)


class _FewCandidateTruncatedWalkGraphReader(FakeMintGraphReader):
    """Returns all held dcids while marking `node_subjects` truncated, modelling a short page
    with a continuation token. `node_arcs_many` reads from the same arc map.
    """

    def node_subjects(self, dcid, *, property, filters=None, max_results):
        page = super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )
        return type(page)(dcids=page.dcids, truncated=True)


def test_truncated_page_holding_fewer_candidates_than_the_cap_reports_the_count_it_read():
    """A truncated 37-candidate page reports the read count separately from the 4,000-candidate cap.

    The `gender` set binding prevents the page from proving completeness.
    """
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        f"Count_Person_Detail{i:02d}": {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POPULATION_TYPE,),
            "measuredProperty": (_MEASURED_PROPERTY,),
            "observationProperties": ("recipient",),
            "constraintProperties": ("gender",),
            "gender": ("Male",),
        }
        for i in range(37)
    }
    graph = _FewCandidateTruncatedWalkGraphReader(arcs=arcs)
    identity = _two_member_identity()
    identity["slots"] = [
        {"property": "gender", "binding": {"state": "set", "refs": ["Male", "Female"]}}
    ]

    with pytest.raises(OverCeilingError) as exc_info:
        _resolve(identity, graph=graph)

    assert exc_info.value.ceiling_kind == "candidate_window"
    assert exc_info.value.count == 37
    assert exc_info.value.cap == CANDIDATE_CAP
    # The refusal reports the 37 candidates read and the 4,000-candidate cap.
    detail = str(exc_info.value)
    assert "37" in detail
    assert str(CANDIDATE_CAP) in detail


def test_hierarchical_parent_value_binding_on_a_broad_family_refuses_over_ceiling_not_spec_not_exportable():
    """`CRSTypeOfFinance100` is a hollow parent with 13,620 descendant rows.

    Widening its value binding to a set leaves the query broad, so a truncated page returns
    `over_ceiling`.
    """
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        "OdaGrantsToKenya": {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POPULATION_TYPE,),
            "measuredProperty": (_MEASURED_PROPERTY,),
            "observationProperties": ("recipient",),
            "constraintProperties": ("crsTypeOfFinance",),
            "crsTypeOfFinance": ("CRSTypeOfFinance110",),
        },
        "OdaOtherGrantsToKenya": {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POPULATION_TYPE,),
            "measuredProperty": (_MEASURED_PROPERTY,),
            "observationProperties": ("recipient",),
            "constraintProperties": ("crsTypeOfFinance",),
            "crsTypeOfFinance": ("CRSTypeOfFinance210",),
        },
    }
    for i in range(4_001):
        arcs[f"AmountVintage{i:04d}"] = {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POPULATION_TYPE,),
            "measuredProperty": (_MEASURED_PROPERTY,),
            "observationProperties": ("recipient",),
            "constraintProperties": ("vintage",),
            "vintage": (f"Vintage{i:04d}",),
        }
    graph = FakeMintGraphReader(arcs=arcs)
    identity = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": _POPULATION_TYPE,
            "measured_property": _MEASURED_PROPERTY,
        },
        "slots": [
            {
                "property": "crsTypeOfFinance",
                "binding": {"state": "value", "ref": "CRSTypeOfFinance100"},
            }
        ],
        "entities": {},
        "requested_window": {"kind": "all"},
    }

    with pytest.raises(OverCeilingError) as exc_info:
        _resolve(identity, graph=graph)

    assert exc_info.value.ceiling_kind == "candidate_window"
    assert exc_info.value.count == CANDIDATE_CAP
    assert exc_info.value.cap == CANDIDATE_CAP
    assert "statvar_set" in str(exc_info.value)


def test_hierarchical_parent_set_binding_on_a_broad_family_refuses_over_ceiling_not_spec_not_exportable():
    """A parent in a set binding must widen to the same descendants as a value binding.

    This covers identities posted directly through `SpecIdInput`.
    """
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        "OdaGrantsToKenya": {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POPULATION_TYPE,),
            "measuredProperty": (_MEASURED_PROPERTY,),
            "observationProperties": ("recipient",),
            "constraintProperties": ("crsTypeOfFinance",),
            "crsTypeOfFinance": ("CRSTypeOfFinance110",),
        },
        "OdaOtherGrantsToKenya": {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POPULATION_TYPE,),
            "measuredProperty": (_MEASURED_PROPERTY,),
            "observationProperties": ("recipient",),
            "constraintProperties": ("crsTypeOfFinance",),
            "crsTypeOfFinance": ("CRSTypeOfFinance210",),
        },
    }
    for i in range(4_001):
        arcs[f"AmountVintage{i:04d}"] = {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POPULATION_TYPE,),
            "measuredProperty": (_MEASURED_PROPERTY,),
            "observationProperties": ("recipient",),
            "constraintProperties": ("vintage",),
            "vintage": (f"Vintage{i:04d}",),
        }
    graph = FakeMintGraphReader(arcs=arcs)
    identity = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": _POPULATION_TYPE,
            "measured_property": _MEASURED_PROPERTY,
        },
        "slots": [
            {
                "property": "crsTypeOfFinance",
                "binding": {"state": "set", "refs": ["CRSTypeOfFinance100"]},
            }
        ],
        "entities": {},
        "requested_window": {"kind": "all"},
    }

    with pytest.raises(OverCeilingError) as exc_info:
        _resolve(identity, graph=graph)

    assert exc_info.value.ceiling_kind == "candidate_window"
    assert exc_info.value.count == CANDIDATE_CAP
    assert exc_info.value.cap == CANDIDATE_CAP
    assert "statvar_set" in str(exc_info.value)


def test_hollow_parent_widening_is_not_safe_by_itself_only_a_second_bound_slot_makes_it_so():
    """The finance parent has 13,620 descendants, above the 4,000-candidate cap.

    Binding `recipientCountry` lets `_safe_slot_filters` narrow the query while the widened
    finance set is checked client-side. Three Kenya members resolve. An out-of-set finance
    value and the Uganda member are excluded.
    """
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        "OdaGrantsToKenya": {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POPULATION_TYPE,),
            "measuredProperty": (_MEASURED_PROPERTY,),
            "observationProperties": ("recipient",),
            "constraintProperties": ("crsTypeOfFinance", "recipientCountry"),
            "crsTypeOfFinance": ("CRSTypeOfFinance110",),
            "recipientCountry": ("Kenya",),
        },
        "OdaOtherGrantsToKenya": {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POPULATION_TYPE,),
            "measuredProperty": (_MEASURED_PROPERTY,),
            "observationProperties": ("recipient",),
            "constraintProperties": ("crsTypeOfFinance", "recipientCountry"),
            "crsTypeOfFinance": ("CRSTypeOfFinance210",),
            "recipientCountry": ("Kenya",),
        },
        "OdaLoansToKenya": {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POPULATION_TYPE,),
            "measuredProperty": (_MEASURED_PROPERTY,),
            "observationProperties": ("recipient",),
            "constraintProperties": ("crsTypeOfFinance", "recipientCountry"),
            "crsTypeOfFinance": ("CRSTypeOfFinance310",),
            "recipientCountry": ("Kenya",),
        },
        # This Kenya member has a finance value outside the widened set, so client-side
        # membership filtering excludes it.
        "OtherAmountToKenya": {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POPULATION_TYPE,),
            "measuredProperty": (_MEASURED_PROPERTY,),
            "observationProperties": ("recipient",),
            "constraintProperties": ("crsTypeOfFinance", "recipientCountry"),
            "crsTypeOfFinance": ("CRSTypeOfFinance421",),
            "recipientCountry": ("Kenya",),
        },
        # The finance value matches, but the Uganda recipient fails the server-side filter.
        "OdaGrantsToUganda": {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POPULATION_TYPE,),
            "measuredProperty": (_MEASURED_PROPERTY,),
            "observationProperties": ("recipient",),
            "constraintProperties": ("crsTypeOfFinance", "recipientCountry"),
            "crsTypeOfFinance": ("CRSTypeOfFinance110",),
            "recipientCountry": ("Uganda",),
        },
    }
    graph = FakeMintGraphReader(arcs=arcs)
    identity = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": _POPULATION_TYPE,
            "measured_property": _MEASURED_PROPERTY,
        },
        "slots": [
            {
                "property": "crsTypeOfFinance",
                "binding": {"state": "value", "ref": "CRSTypeOfFinance100"},
            },
            {
                "property": "recipientCountry",
                "binding": {"state": "value", "ref": "Kenya"},
            },
        ],
        "entities": {},
        "requested_window": {"kind": "all"},
    }

    shapes = _resolve(identity, graph=graph)

    assert sorted(shape.stat_var for shape in shapes) == [
        "dcid:OdaGrantsToKenya",
        "dcid:OdaLoansToKenya",
        "dcid:OdaOtherGrantsToKenya",
    ]


def test_truncated_reverse_query_that_proves_itself_complete_resolves_members():
    # The 4,001 `vintage` variants exceed the 4,000-candidate cap. The unconstrained `Amount`
    # row sorts inside the page and proves completeness because `vintage` is outside the identity
    # and `Amount` declares only the recipient the identity binds.
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        "Amount": {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POPULATION_TYPE,),
            "measuredProperty": (_MEASURED_PROPERTY,),
            "observationProperties": ("recipient",),
            "constraintProperties": (),
        }
    }
    for i in range(4_001):
        arcs[f"AmountVintage{i:04d}"] = {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POPULATION_TYPE,),
            "measuredProperty": (_MEASURED_PROPERTY,),
            "observationProperties": ("recipient",),
            "constraintProperties": ("vintage",),
            "vintage": (f"Vintage{i:04d}",),
        }
    graph = FakeMintGraphReader(arcs=arcs)
    identity = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": _POPULATION_TYPE,
            "measured_property": _MEASURED_PROPERTY,
        },
        "slots": [],
        "entities": {"recipient": ["country/KEN"]},
        "requested_window": {"kind": "all"},
    }

    shapes = _resolve(identity, graph=graph)

    assert [shape.stat_var for shape in shapes] == ["dcid:Amount"]


class _InconsistentReadGraphReader:
    """Returns contradictory `populationType` arcs for selected dcids to model inconsistent
    reverse-index and forward reads. Other graph methods delegate to the fake reader.
    """

    def __init__(self, delegate: FakeMintGraphReader, *, stale_dcids: Sequence[str]) -> None:
        self._delegate = delegate
        self._stale_dcids = frozenset(stale_dcids)

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        result = self._delegate.node_arcs_many(
            dcids, properties=properties, individual_probe_budget=individual_probe_budget
        )
        for dcid in self._stale_dcids:
            if dcid in result and result[dcid] is not None:
                result[dcid] = {**result[dcid], "populationType": ("Other",)}
        return result


def test_inconsistent_candidate_refuses_member_not_resolvable():
    graph = _InconsistentReadGraphReader(_two_member_graph(), stale_dcids=(_ODA_GRANTS,))

    with pytest.raises(MemberNotResolvable) as exc_info:
        _resolve(_two_member_identity(), graph=graph)

    assert exc_info.value.members == (f"dcid:{_ODA_GRANTS}",)
    assert exc_info.value.truncated is False


def test_inconsistent_candidates_over_the_naming_cap_refuse_truncated():
    # Twelve corrupted candidates exceed the naming cap of 10.
    # The refusal marks the remainder as truncated.
    schemes = [f"Scheme{i}" for i in range(12)]
    stat_vars = [f"Amount{i}" for i in range(12)]
    pairs = zip(stat_vars, schemes, strict=True)
    arcs = {stat_var: _member_arcs(scheme=scheme) for stat_var, scheme in pairs}
    graph = _InconsistentReadGraphReader(FakeMintGraphReader(arcs=arcs), stale_dcids=stat_vars)
    identity = _two_member_identity()
    identity["slots"] = [
        {"property": _SCHEME_PROPERTY, "binding": {"state": "set", "refs": schemes}}
    ]

    with pytest.raises(MemberNotResolvable) as exc_info:
        _resolve(identity, graph=graph)

    assert len(exc_info.value.members) == 10
    assert exc_info.value.truncated is True


class _BatchDropsTypeOfGraphReader(FakeMintGraphReader):
    """Drops `typeOf` from every multi-dcid read, so each antichain winner needs a
    single-dcid fallback and the re-verify budget runs out."""

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        result = super().node_arcs_many(
            dcids, properties=properties, individual_probe_budget=individual_probe_budget
        )
        if len(dcids) == 1:
            return result
        return {
            dcid: None if arcs is None else {**arcs, "typeOf": ()} for dcid, arcs in result.items()
        }


class _EmptyReadGraphReader(FakeMintGraphReader):
    """Returns no arcs for the dcids in `empty`, which the reverse-arc walk still lists."""

    def __init__(self, *, arcs, empty: Sequence[str]) -> None:
        super().__init__(arcs=arcs)
        self._empty = frozenset(empty)

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        result = super().node_arcs_many(
            dcids, properties=properties, individual_probe_budget=individual_probe_budget
        )
        return {dcid: None if dcid in self._empty else arcs for dcid, arcs in result.items()}


class _DropsArcGraphReader(FakeMintGraphReader):
    """Returns an empty arc for the property `drop` names, on every read of that dcid."""

    def __init__(self, *, arcs, drop: Mapping[str, str]) -> None:
        super().__init__(arcs=arcs)
        self._drop = drop

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        result = super().node_arcs_many(
            dcids, properties=properties, individual_probe_budget=individual_probe_budget
        )
        return {
            dcid: {**arcs, self._drop[dcid]: ()}
            if arcs is not None and dcid in self._drop
            else arcs
            for dcid, arcs in result.items()
        }


_DESTINATION = {"destination": ["country/KEN"]}
"""Entities binding the destination, which `recipient` and `target` each place."""


def _root_arcs(role: str) -> dict[str, tuple[str, ...]]:
    return {
        "typeOf": ("StatisticalVariable",),
        "populationType": (_POPULATION_TYPE,),
        "measuredProperty": (_MEASURED_PROPERTY,),
        "observationProperties": (role,),
        "constraintProperties": (),
    }


def test_a_root_twin_left_unverified_by_the_budget_is_read_alone():
    # Forty unconstrained roots that each place the destination; the last in antichain order
    # declares `target`. The 32-fallback budget leaves it unverified, and the root-twin read
    # finds its roles.
    arcs = {f"Amount{i:02d}": _root_arcs("recipient") for i in range(39)}
    arcs["AmountZ"] = _root_arcs("target")
    identity = {**_two_member_identity(), "slots": [], "entities": _DESTINATION}

    with pytest.raises(RoleSignatureIncompatible) as exc_info:
        _resolve(identity, graph=_BatchDropsTypeOfGraphReader(arcs=arcs))

    assert exc_info.value.roles_by_stat_var["AmountZ"] == ("target",)


def test_a_root_twin_whose_reads_stay_empty_refuses_member_not_resolvable():
    arcs = {name: _root_arcs("recipient") for name in ("AmountA", "AmountB", "AmountZ")}
    identity = {**_two_member_identity(), "slots": []}

    with pytest.raises(MemberNotResolvable) as exc_info:
        _resolve(identity, graph=_EmptyReadGraphReader(arcs=arcs, empty=["AmountZ"]))

    assert exc_info.value.members == ("dcid:AmountZ",)


def test_a_root_twin_whose_reads_contradict_the_reverse_index_refuses_member_not_resolvable():
    # Every read of AmountZ drops `populationType`, which the reverse query filtered on.
    # A stable read of the same arcs refuses on AmountZ's `provider` role.
    arcs = {
        "AmountA": _root_arcs("recipient"),
        "AmountB": _root_arcs("recipient"),
        "AmountZ": _root_arcs("provider"),
    }
    identity = {**_two_member_identity(), "slots": []}

    with pytest.raises(RoleSignatureIncompatible):
        _resolve(identity, graph=FakeMintGraphReader(arcs=arcs))
    with pytest.raises(MemberNotResolvable) as exc_info:
        _resolve(
            identity, graph=_DropsArcGraphReader(arcs=arcs, drop={"AmountZ": "populationType"})
        )

    assert exc_info.value.members == ("dcid:AmountZ",)


def test_confirmed_roots_that_disagree_outrank_an_unreadable_twin():
    # AmountA and AmountB are confirmed and each places the destination with the one role it
    # declares, so they disagree whatever AmountZ declares.
    arcs = {
        "AmountA": _root_arcs("recipient"),
        "AmountB": _root_arcs("target"),
        "AmountZ": _root_arcs("recipient"),
    }
    identity = {**_two_member_identity(), "slots": [], "entities": _DESTINATION}

    with pytest.raises(RoleSignatureIncompatible) as exc_info:
        _resolve(identity, graph=_EmptyReadGraphReader(arcs=arcs, empty=["AmountZ"]))

    assert exc_info.value.roles_by_stat_var == {
        "AmountA": ("recipient",),
        "AmountB": ("target",),
    }


def test_multi_valued_constraint_refuses_member_not_resolvable():
    # `ODAGrantsAmount` asserts two bound values for its own `slots`-covered
    # constraint property -- `Member.constraints` can only ever carry one,
    # so `statvar_members.MultiValuedConstraint` fires inside enumeration
    # itself, before this candidate could ever become a `Member`.
    graph = FakeMintGraphReader(
        arcs={
            _ODA_GRANTS: {
                **_member_arcs(scheme="ODAGrants"),
                _SCHEME_PROPERTY: ("ODAGrants", "ODALoans"),
            },
            _ODA_LOANS: _member_arcs(scheme="ODALoans"),
        }
    )

    with pytest.raises(MemberNotResolvable) as exc_info:
        _resolve(_two_member_identity(), graph=graph)

    assert exc_info.value.members == (_SPEC_ID,)


def test_degenerate_observation_properties_refuses_member_not_resolvable():
    # `ODAGrantsAmount`'s own `observationProperties` arc non-emptily names
    # `observationAbout` -- the same shape write-time validation
    # `shape.resolve_shape` itself refuses on the pinned path.
    graph = FakeMintGraphReader(
        arcs={
            _ODA_GRANTS: _member_arcs(scheme="ODAGrants", roles=("observationAbout", "recipient")),
            _ODA_LOANS: _member_arcs(scheme="ODALoans"),
        }
    )

    with pytest.raises(MemberNotResolvable) as exc_info:
        _resolve(_two_member_identity(), graph=graph)

    assert exc_info.value.members == (_SPEC_ID,)


def test_member_role_disagreement_refuses_role_signature_incompatible():
    # `ODAGrantsAmount` declares `recipient`, `ODALoansAmount` declares
    # `provider` -- both survive the antichain (same size-1 constraint set),
    # but disagree on `declared_roles`, caught inside `enumerate_members`
    # itself (`statvar_members.MemberRoleDisagreement`) before either ever
    # becomes a `StatVarShape`.
    graph = FakeMintGraphReader(
        arcs={
            _ODA_GRANTS: _member_arcs(scheme="ODAGrants", roles=("recipient",)),
            _ODA_LOANS: _member_arcs(scheme="ODALoans", roles=("provider",)),
        }
    )

    with pytest.raises(RoleSignatureIncompatible) as exc_info:
        _resolve(_two_member_identity(), graph=graph)

    assert exc_info.value.spec_id == _SPEC_ID
    assert set(exc_info.value.roles_by_stat_var) == {_ODA_GRANTS, _ODA_LOANS}


def test_missing_type_of_on_an_enumerated_member_refuses_member_not_resolvable():
    # The candidate's `typeOf` arc omits `StatisticalVariable`, so enumeration rejects it.
    # The described shape uses the enumerator's batched read directly.
    graph = FakeMintGraphReader(
        arcs={
            _ODA_GRANTS: {
                k: v for k, v in _member_arcs(scheme="ODAGrants").items() if k != "typeOf"
            },
            _ODA_LOANS: _member_arcs(scheme="ODALoans"),
        }
    )

    with pytest.raises(MemberNotResolvable) as exc_info:
        _resolve(_two_member_identity(), graph=graph)

    assert exc_info.value.members == (_SPEC_ID,)


class _DivergentNodeArcsGraphReader:
    """Returns different `observationProperties` from `node_arcs` and `node_arcs_many` and
    records per-dcid reads. The described arm should use only the batched result.
    """

    def __init__(self, delegate: FakeMintGraphReader) -> None:
        self._delegate = delegate
        self.node_arcs_calls: list[str] = []

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)

    def node_arcs(self, dcid: str, *, properties: Sequence[str]) -> dict[str, tuple[str, ...]]:
        self.node_arcs_calls.append(dcid)
        arcs = self._delegate.node_arcs(dcid, properties=properties)
        if "observationProperties" in arcs:
            return {**arcs, "observationProperties": ("provider", "recipient", "extra_role")}
        return arcs


def test_described_arm_shapes_come_wholly_from_enumeration_never_a_second_read():
    # Described-arm shapes must use only the enumeration's batched graph read.
    graph = _DivergentNodeArcsGraphReader(_two_member_graph())

    shapes = _resolve(_two_member_identity(), graph=graph)

    assert graph.node_arcs_calls == []
    assert sorted(shape.stat_var for shape in shapes) == [
        "dcid:ODAGrantsAmount",
        "dcid:ODALoansAmount",
    ]
    assert all(shape.declared_roles == ("recipient",) for shape in shapes)


# Seeds join reverse-query candidates and use the same checks as walked members.

_FOREIGN_STAT_VAR = "Amount_EconomicActivity"


def _graph_with_foreign_stat_var() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _ODA_GRANTS: _member_arcs(scheme="ODAGrants"),
            _ODA_LOANS: _member_arcs(scheme="ODALoans"),
            _FOREIGN_STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("EconomicActivity",),
                "measuredProperty": ("amount",),
                "observationProperties": (),
                "constraintProperties": (),
            },
        }
    )


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


def _root_and_gendered_arcs() -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        "Count_Person": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "observationProperties": (),
            "constraintProperties": (),
        },
        "Count_Person_Male": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "observationProperties": (),
            "constraintProperties": ("gender",),
            "gender": ("Male",),
        },
        "Count_Person_Female": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "observationProperties": (),
            "constraintProperties": ("gender",),
            "gender": ("Female",),
        },
    }


class _WithheldFromWalkGraphReader(FakeMintGraphReader):
    """Marks the walk truncated and withholds dcids while retaining arcs for seed checks."""

    def __init__(
        self, *, arcs: dict[str, dict[str, tuple[str, ...]]], withheld: frozenset[str]
    ) -> None:
        super().__init__(arcs=arcs)
        self._withheld = withheld

    def node_subjects(self, dcid, *, property, filters=None, max_results):
        page = super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )
        return type(page)(
            dcids=tuple(d for d in page.dcids if d not in self._withheld), truncated=True
        )


def test_a_root_seed_makes_a_truncated_page_prove_complete():
    # The unseeded page contains only the two gendered members.
    graph_without_seed = _WithheldFromWalkGraphReader(
        arcs=_root_and_gendered_arcs(), withheld=frozenset({"Count_Person"})
    )
    with pytest.raises(OverCeilingError):
        _resolve(_unbound_gender_identity(), graph=graph_without_seed)

    # The root dominates both gendered members, proving the truncated page complete.
    graph_with_seed = _WithheldFromWalkGraphReader(
        arcs=_root_and_gendered_arcs(), withheld=frozenset({"Count_Person"})
    )
    shapes = _resolve(_unbound_gender_identity(), graph=graph_with_seed, seed=("Count_Person",))

    assert [shape.stat_var for shape in shapes] == ["dcid:Count_Person"]


def test_a_non_root_seed_does_not_prove_completeness():
    # A root with empty constraint keys dominates the identity.
    # This seed carries `gender`.
    arcs = {
        **_root_and_gendered_arcs(),
        "Count_Person_Nonbinary": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "observationProperties": (),
            "constraintProperties": ("gender",),
            "gender": ("Nonbinary",),
        },
    }
    graph = _WithheldFromWalkGraphReader(
        arcs=arcs, withheld=frozenset({"Count_Person", "Count_Person_Nonbinary"})
    )

    with pytest.raises(OverCeilingError):
        _resolve(_unbound_gender_identity(), graph=graph, seed=("Count_Person_Nonbinary",))


def test_a_foreign_seed_dcid_is_excluded_and_changes_nothing():
    # A seed that matches neither the five-tuple nor a slot is excluded.
    graph = _graph_with_foreign_stat_var()

    shapes = _resolve(_two_member_identity(), graph=graph, seed=(_FOREIGN_STAT_VAR,))

    assert sorted(shape.stat_var for shape in shapes) == [
        "dcid:ODAGrantsAmount",
        "dcid:ODALoansAmount",
    ]


def test_statvar_members_unresolved_node_marker_matches_materialise_graphs():
    # The two packages define this marker separately, so their reader contract must keep it equal.
    assert UNRESOLVED_NODE == materialise_graph.UNRESOLVED_NODE


_IDS_POPULATION_TYPE = "FinancialPosition"
_IDS_TOTAL = "Amount_FinancialPosition_AmountOutstanding_PublicOrPubliclyGuaranteed_Nominal"
_IDS_BY_CREDITOR = (
    "Amount_FinancialPosition_AmountOutstanding_PublicOrPubliclyGuaranteed_ByCreditor_Nominal"
)
_IDS_CONSTRAINTS = {
    "financialPositionType": "AmountOutstanding",
    "guaranteeStatus": "PublicOrPubliclyGuaranteed",
}
_IDS_ROLES = {
    _IDS_TOTAL: ("financingRecipient",),
    _IDS_BY_CREDITOR: ("financingProvider", "financingRecipient"),
}


def _ids_identity(entities: dict[str, list[str]]) -> dict:
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": _IDS_POPULATION_TYPE,
            "measured_property": _MEASURED_PROPERTY,
        },
        "slots": [
            {"property": name, "binding": {"state": "value", "ref": value}}
            for name, value in _IDS_CONSTRAINTS.items()
        ],
        "entities": entities,
        "requested_window": {"kind": "all"},
    }


def _ids_twin_graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            dcid: {
                "typeOf": ("StatisticalVariable",),
                "populationType": (_IDS_POPULATION_TYPE,),
                "measuredProperty": (_MEASURED_PROPERTY,),
                "observationProperties": roles,
                "constraintProperties": tuple(_IDS_CONSTRAINTS),
                **{name: (value,) for name, value in _IDS_CONSTRAINTS.items()},
            }
            for dcid, roles in _IDS_ROLES.items()
        }
    )


def _ids_twin_index() -> CodeIndex:
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
        statvars=tuple(
            StatVarRow(
                dcid=dcid,
                population_type=_IDS_POPULATION_TYPE,
                measured_property=_MEASURED_PROPERTY,
                stat_type=None,
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=roles,
                constraint_properties=_IDS_CONSTRAINTS,
                provenances=("provenance/Test",),
            )
            for dcid, roles in _IDS_ROLES.items()
        ),
    )


@pytest.mark.parametrize("path", ["graph", "index"])
@pytest.mark.parametrize(
    ("entities", "expected"),
    [
        ({}, _IDS_TOTAL),
        ({"destination": ["country/KEN"]}, _IDS_TOTAL),
        ({"origin": ["country/CHN"], "destination": ["country/KEN"]}, _IDS_BY_CREDITOR),
    ],
)
def test_role_twins_resolve_by_the_identitys_bound_roles(path, entities, expected):
    graph = _ids_twin_graph()
    code_index = _ids_twin_index() if path == "index" else get_index()

    shapes = resolve_definition_members(
        _ids_identity(entities),
        spec_id=_SPEC_ID,
        graph=graph,
        enumerator=MemberEnumerator(graph=graph),
        code_index=code_index,
    ).shapes

    assert [shape.stat_var for shape in shapes] == [f"dcid:{expected}"]
    assert shapes[0].declared_roles == _IDS_ROLES[expected]


def test_a_bound_role_no_member_declares_still_refuses_definition_not_reproducible():
    """Resolution keeps the members, so the reproducibility check names the role."""
    identity = dict(_two_member_identity(), entities={"origin": ["country/FRA"]})
    graph = _two_member_graph()

    shapes = _resolve(identity, graph=graph)

    with pytest.raises(DefinitionNotReproducible) as exc_info:
        check_reproducible(identity, spec_id=_SPEC_ID, shape=shapes[0])
    assert exc_info.value.undeclared_roles == ("origin",)
