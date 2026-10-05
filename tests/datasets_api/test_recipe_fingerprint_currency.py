"""Tests recipe fingerprint serialization, write-time signal filling, and read-time rechecks.
A changed recipe body under the same id must rebuild a converted head. The door fixture seeds
a converted signal because its StatVar has no conversion recipe.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from datasets_api.currency import _UNVERIFIABLE_RECIPE_FINGERPRINT, recheck_recipe_fingerprint
from datasets_api.serve import _recipe_dependency_fingerprint, _signal_with_rate_provenance
from datasets_api.versions import MalformedSignalError, MemberSignal, decode_signal, encode_signal
from materialise.convert import ConversionRecord
from materialise.recipe_bodies import RECIPE_BODIES, recipe_body_digest

from ._graph_doubles import FakeMintGraphReader
from .test_definition_door import _CSV_HEADER, _RECIPIENT, _STAT_VAR, _STAT_VAR_REF, _mint_one_spec
from .test_definition_door_currency import (
    _ARCS,
    _FACET,
    _PAST_INTERVAL,
    _client,
    _counting_transport,
    _FakeHeadStore,
    _row,
)

_RATE_PROV_REFRESH = "2026-01-01"


def _graph_with_rate_provenance(facets) -> FakeMintGraphReader:
    """Adds the seeded rate-provenance date so this fixture isolates recipe fingerprint checks."""
    return FakeMintGraphReader(
        arcs={_STAT_VAR: _ARCS, "RATE_PROV": {"lastDataRefreshDate": (_RATE_PROV_REFRESH,)}},
        facets=facets,
        labels={_RECIPIENT: "Kenya"},
    )


_RECIPE_ID = "ConversionRecipeDAC"
_DIGEST = recipe_body_digest(RECIPE_BODIES[_RECIPE_ID])

# --- versions.py: wire shape ---------------------------------------------


def test_encode_signal_omits_recipe_dependency_fingerprint_when_none():
    encoded = encode_signal((MemberSignal("facet1", 1, "2020", "2020"),))

    assert encoded == {
        "members": [
            {
                "dispatch_facet_id": "facet1",
                "obs_count": 1,
                "earliest_date": "2020",
                "latest_date": "2020",
                "observation_period": None,
                "provenance_last_refresh": None,
                "provenance_next_refresh": None,
            }
        ]
    }


def test_encode_decode_round_trips_recipe_dependency_fingerprint():
    signal = (
        MemberSignal(
            "facet1", 1, "2020", "2020", recipe_dependency_fingerprint=(_RECIPE_ID, _DIGEST)
        ),
    )
    decoded = decode_signal(encode_signal(signal))
    assert decoded == signal


def test_decode_signal_accepts_a_member_missing_recipe_dependency_fingerprint():
    """A stored signal without `recipe_dependency_fingerprint` decodes to `None`."""
    decoded = decode_signal(
        {
            "members": [
                {
                    "dispatch_facet_id": "facet1",
                    "obs_count": 1,
                    "earliest_date": None,
                    "latest_date": None,
                }
            ]
        }
    )
    assert decoded == (MemberSignal("facet1", 1, None, None),)


def test_decode_signal_rejects_a_malformed_recipe_dependency_fingerprint():
    with pytest.raises(MalformedSignalError):
        decode_signal(
            {
                "members": [
                    {
                        "dispatch_facet_id": "facet1",
                        "obs_count": 1,
                        "earliest_date": None,
                        "latest_date": None,
                        "recipe_dependency_fingerprint": ["only-one-element"],
                    }
                ]
            }
        )


# --- serve.py: write-time fill --------------------------------------------


def _record(*, recipe_id: str) -> ConversionRecord:
    return ConversionRecord(
        recipe_id=recipe_id,
        exchange_rate_table_version="2026-01-01",
        deflator_table_version=None,
        native_currency="LCU",
        native_unit_ref="LCU",
        table_provenance_id="RATE_PROV",
        table_refresh_date="2026-01-01",
        fallback_count=0,
        fallback_examples=(),
        projected_count=0,
        projected_examples=(),
    )


def test_recipe_dependency_fingerprint_is_none_when_every_record_is_none():
    assert _recipe_dependency_fingerprint([None]) is None
    assert _recipe_dependency_fingerprint([]) is None


def test_recipe_dependency_fingerprint_names_the_converting_recipes_current_digest():
    record = _record(recipe_id=_RECIPE_ID)
    assert _recipe_dependency_fingerprint([record]) == (_RECIPE_ID, _DIGEST)


def test_signal_with_rate_provenance_also_fills_recipe_dependency_fingerprint():
    signal = (MemberSignal("facet1", 1, "2020", "2020"),)
    record = _record(recipe_id=_RECIPE_ID)

    filled = _signal_with_rate_provenance(signal, [record])

    assert filled[0].recipe_dependency_fingerprint == (_RECIPE_ID, _DIGEST)
    assert filled[0].rate_provenance_refresh == (("RATE_PROV", "2026-01-01"),)


def test_signal_with_rate_provenance_leaves_recipe_dependency_fingerprint_none_when_unconverted():
    signal = (MemberSignal("facet1", 1, "2020", "2020"),)

    assert _signal_with_rate_provenance(signal, [None]) == signal


# --- currency.py: read-time recheck ----------------------------------------

_FRESH = (MemberSignal("facet1", 10, "2020", "2021"),)


def test_recheck_returns_signals_unchanged_when_the_stale_signal_never_converted():
    """An unconverted member has no recipe fingerprint to recheck."""
    stale = (MemberSignal("facet1", 10, "2020", "2021"),)

    result = recheck_recipe_fingerprint(_FRESH, stale)

    assert result == _FRESH


def test_recheck_matches_when_the_recipe_body_is_unchanged():
    """A matching live digest preserves the stale head's fingerprint and signals.

    `fresh` carries the same rate-refresh date so this test isolates the recipe check.
    """
    stale = (
        MemberSignal(
            "facet1",
            10,
            "2020",
            "2021",
            rate_provenance_refresh=(("RATE_PROV", "2026-01-01"),),
            recipe_dependency_fingerprint=(_RECIPE_ID, _DIGEST),
        ),
    )
    fresh = (replace(_FRESH[0], rate_provenance_refresh=stale[0].rate_provenance_refresh),)

    result = recheck_recipe_fingerprint(fresh, stale)

    assert result[0].recipe_dependency_fingerprint == (_RECIPE_ID, _DIGEST)
    assert result == stale


def test_recheck_mismatches_when_the_recipe_body_has_changed():
    """A live digest that differs from the stored fingerprint changes the signal and forces
    `_resolve_currency` to rebuild.
    """
    stale_digest = "0" * 64  # stands in for a digest minted under an older recipe body
    stale = (
        MemberSignal(
            "facet1",
            10,
            "2020",
            "2021",
            rate_provenance_refresh=(("RATE_PROV", "2026-01-01"),),
            recipe_dependency_fingerprint=(_RECIPE_ID, stale_digest),
        ),
    )

    result = recheck_recipe_fingerprint(_FRESH, stale)

    assert result[0].recipe_dependency_fingerprint == (_RECIPE_ID, _DIGEST)
    assert result != stale


def test_recheck_forces_a_mismatch_for_a_converted_head_with_no_stored_fingerprint():
    """`_UNVERIFIABLE_RECIPE_FINGERPRINT` marks a converted signal without a fingerprint,
    forcing a rebuild.
    """
    stale = (
        MemberSignal(
            "facet1", 10, "2020", "2021", rate_provenance_refresh=(("RATE_PROV", "2026-01-01"),)
        ),
    )

    result = recheck_recipe_fingerprint(_FRESH, stale)

    assert result[0].recipe_dependency_fingerprint == _UNVERIFIABLE_RECIPE_FINGERPRINT
    assert result != stale


def test_recheck_forces_a_mismatch_for_a_retired_recipe_id():
    stale = (
        MemberSignal(
            "facet1",
            10,
            "2020",
            "2021",
            rate_provenance_refresh=(("RATE_PROV", "2026-01-01"),),
            recipe_dependency_fingerprint=("recipe/Retired", "a" * 64),
        ),
    )

    result = recheck_recipe_fingerprint(_FRESH, stale)

    assert result[0].recipe_dependency_fingerprint == _UNVERIFIABLE_RECIPE_FINGERPRINT
    assert result != stale


def test_recheck_a_length_mismatch_against_the_stale_signal_returns_signals_unchanged():
    stale = (
        MemberSignal(
            "facet1",
            10,
            "2020",
            "2021",
            rate_provenance_refresh=(("RATE_PROV", "2026-01-01"),),
            recipe_dependency_fingerprint=(_RECIPE_ID, _DIGEST),
        ),
        MemberSignal("facet2", 5, "2020", "2021"),
    )

    result = recheck_recipe_fingerprint(_FRESH, stale)

    assert result == _FRESH


# --- door-level: the full currency decision --------------------------------


def _seed_converted_head(
    head_store: _FakeHeadStore, spec_id: str, *, fingerprint: tuple[str, str] | None
) -> None:
    """Seeds a converted head signal; this fixture's graph has no conversion recipe."""
    minted = head_store.head_for(spec_id)
    converted_signal = (
        replace(
            minted.signal[0],
            rate_provenance_refresh=(("RATE_PROV", _RATE_PROV_REFRESH),),
            recipe_dependency_fingerprint=fingerprint,
        ),
    )
    head_store._heads[spec_id] = replace(minted, signal=converted_signal)


def test_unchanged_recipe_body_keeps_serving_from_head():
    """An unchanged recipe digest preserves the head hit."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _graph_with_rate_provenance(facets)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    client = _client(graph=graph, sdmx_transport=transport, head_writer=head_store)
    spec_id = _mint_one_spec(client)
    _seed_converted_head(head_store, spec_id, fingerprint=(_RECIPE_ID, _DIGEST))
    sdmx_calls_after_mint = sdmx_calls.n

    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "head"
    # The facet probe reaches coverage, not SDMX.
    assert sdmx_calls.n == sdmx_calls_after_mint
    assert head_store.head_for(spec_id).signal[0].recipe_dependency_fingerprint == (
        _RECIPE_ID,
        _DIGEST,
    )


def test_changed_recipe_body_forces_a_rematerialise_despite_an_unchanged_probe():
    """A stored 64-zero digest differs from the current `_DIGEST`, forcing rematerialisation."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _graph_with_rate_provenance(facets)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, _sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    client = _client(graph=graph, sdmx_transport=transport, head_writer=head_store)
    spec_id = _mint_one_spec(client)
    _seed_converted_head(head_store, spec_id, fingerprint=(_RECIPE_ID, "0" * 64))

    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "materialise"


def test_a_pre_existing_head_with_no_stored_fingerprint_rebuilds_once():
    """A converted head without a fingerprint rebuilds as unverifiable."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _graph_with_rate_provenance(facets)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, _sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    client = _client(graph=graph, sdmx_transport=transport, head_writer=head_store)
    spec_id = _mint_one_spec(client)
    _seed_converted_head(head_store, spec_id, fingerprint=None)

    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "materialise"
