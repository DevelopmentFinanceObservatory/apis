"""Search's `Slot.summable` equals the outcome of the mint's sum over the same family.

Each family is resolved as the mint resolves it, one of its members holds a row and the
rest hold none, and the mint sums the frame down to one series per year. The sum completes
exactly when Search reads the family's summable axes true.
"""

from __future__ import annotations

import dataclasses

import pytest

import search.engine.retrieval.bound_values as bound_values_module
from code_index import CodeIndex
from code_index.shared import get_index
from datasets_api.adapter import resolve_definition_members
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.render import ReshapeResult, _RenderReads, _reshape
from materialise.enrichment import EnrichmentLookup
from materialise.frame_build import Frame
from materialise.reshape import RollupNotSummableError, Table
from search.engine.retrieval.bound_values import apply_slot_summability
from search.models import BaseIdentity
from statvar_members import MemberEnumerator
from tests.code_index.aid_identity import (
    AID_TYPE,
    CRS_MEMBER_COUNT,
    CRS_PROVENANCE,
    IMPUTED_MEMBER_COUNT,
    IMPUTED_PROVENANCE,
    education_aid_identity,
    imputed_education_aid_identity,
)
from tests.engine._builders import unbound_slot, value_slot

from ._graph_doubles import FakeMintGraphReader
from ._member_bounds import mint_bounds

_SUMMABLE_AXES = ("crsCooperationModality", "crsPurpose", "crsTypeOfFinance", "crsTypeOfFlow")
# The imputed five-tuple declares two of the CRS code systems.
_IMPUTED_SUMMABLE_AXES = ("crsPurpose", "crsTypeOfFlow")
_GRAMMAR_COLUMNS = (
    "stat_var",
    "financingProvider",
    "financingRecipient",
    "time_period",
    "value",
    "unit",
    "measurementMethod",
    "provenance",
    "status",
)


def _index_where_the_imputed_dataset_declares_no_absence_reading() -> CodeIndex:
    index = get_index()
    return CodeIndex(
        stamp=index.stamp,
        declarations=tuple(index.declarations.values()),
        codes=index.codes,
        statvars=index.statvars,
        datasets=tuple(
            dataclasses.replace(dataset, absent_observation="unknown")
            if dataset.provenance == IMPUTED_PROVENANCE
            else dataset
            for dataset in index.datasets.values()
        ),
    )


def _mint_sum(
    identity: dict, *, code_index: CodeIndex, populated: int = 1
) -> tuple[str, int, ReshapeResult | None]:
    """The elected dataset, the member count, and the mint's sum of a frame in which
    `populated` members of the family hold a row, or `None` when the sum refuses for a
    missing member."""
    graph = FakeMintGraphReader()
    shapes = resolve_definition_members(
        identity,
        spec_id="spec_test",
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=code_index,
        bounds=mint_bounds(),
    ).shapes
    dataset = shapes[0].dataset
    assert dataset is not None
    assert {shape.dataset.provenance for shape in shapes if shape.dataset} == {dataset.provenance}
    roles = shapes[0].declared_roles
    table = Table(
        columns=("stat_var", *roles, "time_period", "value", "unit", "provenance"),
        rows=tuple(
            (shape.stat_var, "country/FRA", "country/KEN", "2022", 7.0, "USD", dataset.provenance)
            for shape in shapes[:populated]
        ),
    )
    try:
        summed = _reshape(
            table,
            options=MintOptions(groupby=GroupbyOptions(by=["time_period"], op="sum")),
            entity_columns=roles,
            reads=_RenderReads(
                enrichment=EnrichmentLookup(graph, parties=lambda: {}),
                known={},
                deadline=None,
            ),
            code_index=code_index,
            resolved_stat_var_refs=tuple(shape.stat_var for shape in shapes),
            provenance_refs=(dataset.provenance,),
            subject_id="spec_test",
            pivot_max_output_cells=5_000_000,
        )
    except RollupNotSummableError as exc:
        assert exc.precondition == "missing_members"
        return dataset.provenance, len(shapes), None
    return dataset.provenance, len(shapes), summed


def _search_summable(identity: dict) -> dict[str, bool]:
    slots = [
        value_slot("how", slot["property"], slot["property"], slot["binding"]["ref"])
        if slot["binding"]["state"] == "value"
        else unbound_slot("how", slot["property"], slot["property"])
        for slot in identity["slots"]
    ]
    stamped = apply_slot_summability(slots, BaseIdentity.model_validate(identity))
    return {slot.key.property.ref: slot.summable for slot in stamped if slot.key.property}


@pytest.mark.parametrize(
    ("identity", "edit_index", "provenance", "member_count", "sums"),
    [
        (education_aid_identity(), False, CRS_PROVENANCE, CRS_MEMBER_COUNT, True),
        (
            imputed_education_aid_identity(),
            False,
            IMPUTED_PROVENANCE,
            IMPUTED_MEMBER_COUNT,
            True,
        ),
        (
            imputed_education_aid_identity(),
            True,
            IMPUTED_PROVENANCE,
            IMPUTED_MEMBER_COUNT,
            False,
        ),
    ],
    ids=["crs", "imputed", "imputed-with-no-absence-reading"],
)
def test_search_summable_equals_the_mints_sum_outcome(
    monkeypatch: pytest.MonkeyPatch,
    identity: dict,
    edit_index: bool,
    provenance: str,
    member_count: int,
    sums: bool,
):
    code_index = (
        _index_where_the_imputed_dataset_declares_no_absence_reading()
        if edit_index
        else get_index()
    )
    monkeypatch.setattr(bound_values_module, "get_index", lambda: code_index)

    elected, members, summed = _mint_sum(identity, code_index=code_index)
    flags = _search_summable(identity)

    assert (elected, members) == (provenance, member_count)
    assert (summed is not None) is sums
    expected_axes = _IMPUTED_SUMMABLE_AXES if provenance == IMPUTED_PROVENANCE else _SUMMABLE_AXES
    assert {axis: flags[axis] for axis in flags if axis in _SUMMABLE_AXES} == dict.fromkeys(
        expected_axes, sums
    )
    # CRS rows declare no aid type. The imputed rows' one aid type is not summable.
    if provenance == IMPUTED_PROVENANCE:
        assert flags[AID_TYPE] is False
    else:
        assert AID_TYPE not in flags


def test_the_summed_imputed_frame_is_a_grammar_frame_without_the_classifier_column():
    """Every imputed member carries one aid type, so the decoded classifier column is
    constant across the family and the summed frame leaves it out."""
    identity = imputed_education_aid_identity()

    _elected, _members, summed = _mint_sum(identity, code_index=get_index(), populated=2)

    assert summed is not None
    assert isinstance(summed.table, Frame)
    assert summed.table.columns == _GRAMMAR_COLUMNS
    assert summed.collapse is not None
    assert summed.collapse.frame.columns == _GRAMMAR_COLUMNS
    assert summed.table.rows == (
        (
            "dcid:spec_test",
            "country/FRA",
            "country/KEN",
            "2022",
            14.0,
            "USD",
            None,
            IMPUTED_PROVENANCE,
            None,
        ),
    )
