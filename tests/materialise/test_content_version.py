"""Tests content-version minting and recomputation against grammar vectors."""

from __future__ import annotations

import dataclasses
import importlib.resources
import json

import pytest
from dfo_frame_grammar import ContentVersionError, canonical_frame_bytes, mint_content_version

from materialise.content_version import (
    ConversionPreimage,
    frame_trace_fragment,
    mint_version,
    mint_version_from_chunks,
    native_unit_source_revision,
    recipe_body_source_revision,
    recompute_version,
)
from materialise.frame_build import build_frame
from materialise.recipe_bodies import RECIPE_BODIES


def _load_vector(name: str) -> dict:
    path = importlib.resources.files("dfo_frame_grammar").joinpath(f"vectors/frame/{name}")
    return json.loads(path.read_text())


def test_mint_version_matches_installed_golden_vector():
    vector = _load_vector("frame-basic-single-row.json")
    frame = build_frame(role_keys=vector["role_keys"], rows=vector["rows"])

    token = mint_version(frame, **vector["content_version_input"])

    assert token == vector["expected_content_version"]


def test_mint_version_from_chunks_matches_mint_version_over_the_concatenated_frame():
    """Hashing sorted, validated chunks matches the whole frame across a chunk boundary."""
    row_a = {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/AAA",
        "time_period": "2015",
        "value": 1000.0,
    }
    row_b = {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/BBB",
        "time_period": "2015",
        "value": 2000.0,
    }
    whole_frame = build_frame(role_keys=("recipient",), rows=[row_b, row_a])
    chunk_a = build_frame(role_keys=("recipient",), rows=[row_a])
    chunk_b = build_frame(role_keys=("recipient",), rows=[row_b])
    kwargs = {
        "graph_revision": "graph-rev-2024-01",
        "source_revisions": [{"source": "OECD_DAC_CRS", "revision": "rev-7"}],
        "facets": [{"axis": "provenance", "ref": "OECD_DAC2A", "basis": "bound"}],
    }

    whole = mint_version(whole_frame, **kwargs)
    streamed = mint_version_from_chunks((chunk_a, chunk_b), **kwargs)

    assert streamed == whole


def test_mint_version_is_order_insensitive():
    frame = build_frame(
        role_keys=("provider", "recipient"),
        rows=[
            {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "provider": "country/FRA",
                "recipient": "country/TGO",
                "time_period": "2015",
                "value": 1000.0,
            }
        ],
    )
    source_revisions = [
        {"source": "OECD_DAC_CRS", "revision": "rev-7"},
        {"source": "IMF_WEO", "revision": "rev-3"},
    ]
    facets = [
        {"axis": "provenance", "ref": "OECD_DAC2A", "basis": "bound"},
        {"axis": "unit", "ref": "USD", "basis": "declared_default"},
    ]

    token_forward = mint_version(
        frame,
        graph_revision="graph-rev-2024-01",
        source_revisions=source_revisions,
        facets=facets,
    )
    token_reversed = mint_version(
        frame,
        graph_revision="graph-rev-2024-01",
        source_revisions=list(reversed(source_revisions)),
        facets=list(reversed(facets)),
    )

    assert token_forward == token_reversed


def test_round_trip_mint_and_recompute_agree():
    vector = _load_vector("frame-basic-single-row.json")
    frame = build_frame(role_keys=vector["role_keys"], rows=vector["rows"])
    content_version_input = vector["content_version_input"]

    minted = mint_version(frame, **content_version_input)
    fragment = frame_trace_fragment(**content_version_input)
    recomputed = recompute_version(canonical_frame_bytes(frame), fragment)

    assert minted == recomputed


def test_mint_version_invalid_basis_propagates_content_version_error():
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": "country/TGO",
                "time_period": "2015",
                "value": 1000.0,
            }
        ],
    )

    with pytest.raises(ContentVersionError) as excinfo:
        mint_version(
            frame,
            graph_revision="graph-rev-2024-01",
            source_revisions=[],
            facets=[{"axis": "provenance", "ref": "OECD_DAC2A", "basis": "not_a_real_basis"}],
        )
    assert excinfo.value.category == "invalid_facet_basis"


def _stub_frame():
    """Build a valid frame for conversion vectors whose digest is monkeypatched."""
    return build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": "country/TGO",
                "time_period": "2015",
                "value": 1000.0,
            }
        ],
    )


def _stub_conversion_preimage(
    *, recipe_id: str, exchange_rate_table_version: str, deflator_table_version: str | None
) -> ConversionPreimage:
    return ConversionPreimage(
        recipe_id=recipe_id,
        exchange_rate_table_version=exchange_rate_table_version,
        deflator_table_version=deflator_table_version,
    )


@pytest.mark.parametrize(
    "vector_name",
    [
        "content-version-conversion-with-deflator.json",
        "content-version-conversion-without-deflator.json",
    ],
)
def test_mint_version_with_conversion_matches_installed_golden_vector(vector_name, monkeypatch):
    vector = _load_vector(vector_name)
    monkeypatch.setattr(
        "materialise.content_version.stream_frame_sha256", lambda chunks: vector["frame_sha256"]
    )
    conversion_input = vector["content_version_input"]["conversion"]
    preimage = _stub_conversion_preimage(
        recipe_id=conversion_input["recipe"],
        exchange_rate_table_version=conversion_input["rate_table_version"],
        deflator_table_version=conversion_input.get("deflator_table_version"),
    )

    token = mint_version(
        _stub_frame(),
        graph_revision=vector["content_version_input"]["graph_revision"],
        source_revisions=vector["content_version_input"]["source_revisions"],
        facets=vector["content_version_input"]["facets"],
        conversion=preimage,
    )

    assert token == vector["expected_content_version"]


def test_mint_content_version_missing_rate_table_version_refuses():
    """The vector carries `recipe` but omits `rate_table_version`.

    It expects `schema_violation`. `ConversionPreimage` requires the field, so this test calls
    `mint_content_version` directly.
    """
    vector = _load_vector("content-version-conversion-missing-rate-table-version-rejected.json")

    with pytest.raises(ContentVersionError) as excinfo:
        mint_content_version(frame_sha256=vector["frame_sha256"], **vector["content_version_input"])

    assert excinfo.value.category == vector["failure_category"]


def test_native_unit_source_revision_shape():
    assert native_unit_source_revision(
        stat_var="MalariaGrantDisbursements", native_unit="CurrencyUSD"
    ) == {"source": "native_unit/MalariaGrantDisbursements", "revision": "CurrencyUSD"}


def test_mint_version_unconverted_content_version_matches_base_branch_token():
    """Recipe-body revisions affect converted inputs. Unconverted mints retain this token."""
    token = mint_version(
        _stub_frame(),
        graph_revision="graph-rev-2024-01",
        source_revisions=[{"source": "OECD_DAC_CRS", "revision": "rev-7"}],
        facets=[{"axis": "provenance", "ref": "OECD_DAC2A", "basis": "bound"}],
    )
    assert token == "cv_7c02241bcdbcfa29f600bb67a3b9cebc"


def test_mint_version_with_conversion_moves_when_recipe_body_changes(monkeypatch):
    monkeypatch.setattr("materialise.content_version.stream_frame_sha256", lambda chunks: "a" * 64)
    preimage = _stub_conversion_preimage(
        recipe_id="ConversionRecipeDAC",
        exchange_rate_table_version="rate-v1",
        deflator_table_version="defl-v1",
    )
    kwargs = {
        "graph_revision": "graph-rev-2024-01",
        "source_revisions": [{"source": "OECD_DAC_CRS", "revision": "rev-7"}],
        "facets": [{"axis": "provenance", "ref": "OECD_DAC2A", "basis": "bound"}],
        "conversion": preimage,
    }

    before = mint_version(_stub_frame(), **kwargs)
    unchanged_body_again = mint_version(_stub_frame(), **kwargs)
    assert before == unchanged_body_again

    original_recipe = RECIPE_BODIES["ConversionRecipeDAC"]
    monkeypatch.setitem(
        RECIPE_BODIES,
        "ConversionRecipeDAC",
        dataclasses.replace(original_recipe, forecast_policy="SomeOtherPolicy"),
    )
    after = mint_version(_stub_frame(), **kwargs)

    assert after != before


def test_mint_version_adds_the_recipe_body_revision_only_when_absent(monkeypatch):
    """A trace written by finish carries the `recipe_body/<recipe>` revision, and a trace stored
    before that lacks it. Both mint the same token, pinned here, and the present revision is not
    appended a second time (the grammar refuses a duplicate source)."""
    monkeypatch.setattr("materialise.content_version.stream_frame_sha256", lambda chunks: "a" * 64)
    preimage = _stub_conversion_preimage(
        recipe_id="ConversionRecipeDAC",
        exchange_rate_table_version="v1",
        deflator_table_version=None,
    )
    stored = ({"source": "native_unit/X", "revision": "CurrencyUSD"},)
    body_revision = recipe_body_source_revision(preimage)
    assert body_revision is not None
    facets = [{"axis": "unit", "ref": {"currency": "EUR", "prices": "current"}, "basis": "bound"}]

    without_body = mint_version(
        _stub_frame(),
        graph_revision="g",
        source_revisions=stored,
        facets=facets,
        conversion=preimage,
    )
    with_body = mint_version(
        _stub_frame(),
        graph_revision="g",
        source_revisions=(*stored, body_revision),
        facets=facets,
        conversion=preimage,
    )

    assert without_body == with_body == "cv_3788c1b6812554eaa5d2b1bcb0a6c149"
