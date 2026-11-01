"""Pins `ManifestTrace.to_dict()` fields against the trace wire contract."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from dfo_frame_grammar import FRAME_GRAMMAR_VERSION

from materialise.trace import (
    ManifestTrace,
    MemberFamily,
    RecordedFamily,
    ResolvedStatVar,
    StatedFamily,
    parse_recorded_family,
    recorded_family_json,
)

# `conversion` appears only when a recipe runs; the dedicated test covers that field set.
_CONTRACT_TRACE_FIELDS = frozenset(
    {
        "applied_window",
        "graph_revision",
        "source_revisions",
        "resolved_stat_vars",
        "family_member_digest",
        "family_member_count",
        "facets",
        "normalisations",
        "frame_grammar_version",
        "materialiser_build",
        "capability_version",
    }
)

_SOURCE_REVISIONS = ({"source": "OECD_DAC_CRS", "revision": "rev-7"},)
_RESOLVED_STAT_VARS = (
    ResolvedStatVar(
        ref="dcid:MalariaGrantDisbursements",
        roles=("provider", "recipient"),
        observation_count=12,
    ).as_trace_entry(),
)
_FACETS = (
    {
        "axis": "provenance",
        "ref": "OECD_DAC_CRS",
        "basis": "declared_default",
        "default_policy": "policy_ref",
        "default_policy_version": "policy-v1",
        "selection_reason": "graph-declared default",
        "dispatch_facet_id": "facet_123",
    },
)


def _build_trace(*, applied_window: dict[str, object] | None) -> ManifestTrace:
    return ManifestTrace(
        applied_window=applied_window,
        graph_revision="graph-rev-2024-01",
        source_revisions=_SOURCE_REVISIONS,
        resolved_stat_vars=_RESOLVED_STAT_VARS,
        facets=_FACETS,
        normalisations=(),
        conversion=None,
        frame_grammar_version=FRAME_GRAMMAR_VERSION,
        materialiser_build="build-123",
        capability_version="cap-v1",
    )


def test_to_dict_matches_contract_trace_fields_exactly():
    trace = _build_trace(
        applied_window={"kind": "range", "start_year": 2015, "end_year": 2024, "origin": "query"}
    )

    assert trace.to_dict().keys() == _CONTRACT_TRACE_FIELDS


def test_to_dict_omits_applied_window_when_none():
    trace = _build_trace(applied_window=None)

    assert "applied_window" not in trace.to_dict()


def test_to_dict_omits_conversion_when_no_recipe_ran():
    # A ref-form or unbound unit binding plans no conversion, so
    # "conversion" stays off the wire.
    trace = _build_trace(applied_window=None)

    assert "conversion" not in trace.to_dict()


def test_to_dict_omits_code_index_stamp_when_none():
    # `_build_trace` leaves `code_index_stamp` at its default -- this
    # package's own construction site (`finish.py`) never has a
    # `CodeIndex` in scope, so this is the shape every trace it builds
    # carries; `datasets_api.manifest.assemble_manifest` is the one caller
    # that sets it.
    trace = _build_trace(applied_window=None)

    assert "code_index_stamp" not in trace.to_dict()


def test_to_dict_includes_code_index_stamp_when_set():
    stamp = {
        "content_hash": "1fc4bbfa",
        "exported_at": "2026-09-17T00:00:00Z",
        "graph_endpoint": "https://example.test/graph",
    }
    trace = _build_trace(
        applied_window={"kind": "range", "start_year": 2015, "end_year": 2024, "origin": "query"}
    )
    trace = ManifestTrace(**{**trace.__dict__, "code_index_stamp": stamp})

    result = trace.to_dict()

    assert result.keys() == _CONTRACT_TRACE_FIELDS | {"code_index_stamp"}
    assert result["code_index_stamp"] == stamp


def test_to_dict_includes_conversion_when_a_recipe_ran():
    entry = {
        "recipe": "ConversionRecipeDAC",
        "rate_table_version": "2026-01-01T00:00:00Z",
        "members": [
            {
                "stat_var": "Amount_FinancialFlow_DevelopmentAssistance_Nominal",
                "native_unit": "CurrencyUSD",
            }
        ],
        "fallback": {"count": 0, "examples": [], "truncated": False},
        "projected": {"count": 0, "examples": [], "truncated": False},
    }
    conversion = (entry,)
    trace = _build_trace(
        applied_window={"kind": "range", "start_year": 2015, "end_year": 2024, "origin": "query"}
    )
    trace = ManifestTrace(**{**trace.__dict__, "conversion": conversion})

    result = trace.to_dict()

    assert result.keys() == _CONTRACT_TRACE_FIELDS | {"conversion"}
    assert result["conversion"] == [entry]


def test_to_dict_never_omits_normalisations_even_when_empty():
    trace = _build_trace(applied_window=None)

    result = trace.to_dict()

    assert "normalisations" in result
    assert result["normalisations"] == []


def test_resolved_stat_var_as_trace_entry():
    entry = ResolvedStatVar(
        ref="dcid:X", roles=("provider", "recipient"), observation_count=7
    ).as_trace_entry()

    assert entry == {
        "ref": "dcid:X",
        "roles": ["provider", "recipient"],
        "observation_count": 7,
    }


def test_to_dict_carries_the_origin_catalogue_build_only_when_set():
    trace = _build_trace(applied_window=None)

    assert "origin_catalogue_build_id" not in trace.to_dict()
    stamped = replace(trace, origin_catalogue_build_id="build-1")
    assert stamped.to_dict()["origin_catalogue_build_id"] == "build-1"


def test_to_dict_states_the_family_beside_the_members_served():
    trace = replace(
        _build_trace(applied_window=None),
        family=MemberFamily(digest="ab" * 32, count=3),
        stub_candidates=("dcid:Count_Person_Stub",),
    )

    result = trace.to_dict()

    assert result["family_member_digest"] == "ab" * 32
    assert result["family_member_count"] == 3
    assert result["stub_candidates"] == ["dcid:Count_Person_Stub"]
    assert len(result["resolved_stat_vars"]) == 1


def test_to_dict_states_a_null_family_when_the_whole_of_it_is_not_known():
    result = _build_trace(applied_window=None).to_dict()

    assert result["family_member_digest"] is None
    assert result["family_member_count"] is None


_RECORDED = RecordedFamily(
    stated=StatedFamily(
        members=MemberFamily(digest="ab" * 32, count=3), stub_candidates=("dcid:Stub_A",)
    ),
    confirmed_at=datetime(2026, 1, 1, 0, 0, 0, 250, tzinfo=UTC),
    basis="basis",
)


def test_a_recorded_family_reads_back_as_written():
    assert parse_recorded_family(recorded_family_json(_RECORDED)) == _RECORDED


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param({"confirmed_at": None}, id="no-confirmation"),
        pytest.param({"confirmed_at": "yesterday"}, id="unparsable-confirmation"),
        pytest.param({"confirmed_at": "2026-01-01T00:00:00"}, id="naive-confirmation"),
        pytest.param({"family_member_count": "3"}, id="count-as-text"),
        pytest.param({"family_member_digest": 7}, id="digest-as-number"),
        pytest.param({"stub_candidates": "dcid:Stub_A"}, id="stubs-as-text"),
        pytest.param({"basis": ["basis"]}, id="basis-as-list"),
        pytest.param({"basis": None}, id="no-basis"),
    ],
)
def test_a_stored_family_this_code_cannot_read_states_none(edit):
    stored = {**recorded_family_json(_RECORDED), **edit}

    assert parse_recorded_family(stored) is None


def test_a_null_or_non_object_stored_family_states_none():
    assert parse_recorded_family(None) is None
    assert parse_recorded_family(["family"]) is None
