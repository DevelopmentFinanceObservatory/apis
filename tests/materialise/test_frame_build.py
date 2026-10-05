"""Tests frame building and table-to-frame parity against SDMX observations."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import pytest
from dfo_frame_grammar import Frame, FrameError, canonical_frame_bytes

from dc_sdmx import SdmxDataResult, SdmxObservation
from dc_sdmx.results import parse_data_csv
from materialise import columnar
from materialise.content_version import mint_version
from materialise.errors import MaterialisationUsageError, UpstreamResponseError
from materialise.frame_build import build_frame, build_frame_from_table, declared_roles

from ._table_support import table_from_observations

_DC_SDMX_FIXTURES_DIR = Path(__file__).parent.parent / "dc_sdmx" / "fixtures"

_STAT_VAR = "dcid:MalariaGrantDisbursements"
_ROLE_KEYS = ("recipient",)
_ROWS = [
    {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/TGO",
        "time_period": "2015",
        "value": 1000.0,
    }
]


def test_build_frame_round_trips_with_frame_build_directly():
    via_wrapper = build_frame(role_keys=_ROLE_KEYS, rows=_ROWS)
    via_grammar = Frame.build(role_keys=_ROLE_KEYS, rows=_ROWS)

    assert via_wrapper == via_grammar


def test_build_frame_status_key_propagates_frame_error():
    with pytest.raises(FrameError):
        build_frame(
            role_keys=_ROLE_KEYS,
            rows=[{**_ROWS[0], "status": "whatever"}],
        )


def test_build_frame_empty_rows_is_stable_and_deterministic():
    frame = build_frame(role_keys=_ROLE_KEYS, rows=())
    assert frame.rows == ()

    token_1 = mint_version(
        frame,
        graph_revision="graph-rev-1",
        source_revisions=[],
        facets=[],
    )
    token_2 = mint_version(
        frame,
        graph_revision="graph-rev-1",
        source_revisions=[],
        facets=[],
    )
    assert token_1 == token_2
    assert token_1.startswith("cv_")


def test_declared_roles_reads_from_sdmx_data_result_roles_not_role_bindings():
    result = SdmxDataResult(
        roles=("recipient",),
        rows=(
            SdmxObservation(
                variable="dcid:MalariaGrantDisbursements",
                role_values={"recipient": "country/TGO"},
                unit="USD",
                measurement_method="GrantEquivalent",
                observation_period=None,
                provenance="OECD_DAC2A",
                time_period="2015",
                value="1000",
                scaling_factor=None,
                facet_id="facet1",
            ),
        ),
    )

    # The result alone supplies roles. `role_bindings` is outside this API.
    assert declared_roles(result) == ("recipient",)


def _observation(
    *,
    recipient: str | None = "country/TGO",
    time_period: str | None,
    value: str,
    unit: str | None = "USD",
    measurement_method: str | None = "GrantEquivalent",
    provenance: str | None = "OECD_DAC2A",
    facet_id: str | None = "facet1",
    variable: str = "MalariaGrantDisbursements",
) -> SdmxObservation:
    return SdmxObservation(
        variable=variable,
        role_values={} if recipient is None else {"recipient": recipient},
        unit=unit,
        measurement_method=measurement_method,
        observation_period=None,
        provenance=provenance,
        time_period=time_period,
        value=value,
        scaling_factor=None,
        facet_id=facet_id,
    )


def _observed_table(rows: tuple[SdmxObservation, ...]):
    table = table_from_observations(rows, roles=_ROLE_KEYS)
    return table.filter(columnar.observed_mask(table))


def _row_dict(row: SdmxObservation, *, roles: Sequence[str] = _ROLE_KEYS) -> dict[str, object]:
    """Map an observed `SdmxObservation` to the row shape expected by `Frame.build`."""
    stat_var = row.variable if row.variable.startswith("dcid:") else f"dcid:{row.variable}"
    built: dict[str, object] = {
        "stat_var": stat_var,
        "time_period": row.time_period,
        "value": row.value_as_float(),
        "unit": row.unit,
        "measurementMethod": row.measurement_method,
        "provenance": row.provenance,
    }
    for role in roles:
        if role in row.role_values:
            built[role] = row.role_values[role]
    return built


def test_build_frame_from_table_empty_table_matches_build_frame_empty_rows():
    empty_table = table_from_observations((), roles=_ROLE_KEYS)

    via_table = build_frame_from_table(empty_table, role_keys=_ROLE_KEYS, stat_var=_STAT_VAR)
    via_rows = build_frame(role_keys=_ROLE_KEYS, rows=())

    assert via_table == via_rows


def test_build_frame_from_table_null_role_cell_matches_build_frame_category():
    rows = (_observation(recipient=None, time_period="2015", value="1000"),)

    with pytest.raises(FrameError) as row_excinfo:
        build_frame(role_keys=_ROLE_KEYS, rows=[_row_dict(row) for row in rows])

    with pytest.raises(FrameError) as table_excinfo:
        build_frame_from_table(_observed_table(rows), role_keys=_ROLE_KEYS, stat_var=_STAT_VAR)

    assert table_excinfo.value.category == row_excinfo.value.category == "schema_violation"


def test_build_frame_from_table_null_period_matches_build_frame_category():
    rows = (_observation(time_period=None, value="1000"),)

    with pytest.raises(FrameError) as row_excinfo:
        build_frame(role_keys=_ROLE_KEYS, rows=[_row_dict(row) for row in rows])

    with pytest.raises(FrameError) as table_excinfo:
        build_frame_from_table(_observed_table(rows), role_keys=_ROLE_KEYS, stat_var=_STAT_VAR)

    assert table_excinfo.value.category == row_excinfo.value.category == "schema_violation"


def test_build_frame_from_table_duplicate_grain_matches_build_frame_category():
    rows = (
        _observation(time_period="2015", value="1000"),
        _observation(time_period="2015", value="2000"),
    )

    with pytest.raises(FrameError) as row_excinfo:
        build_frame(role_keys=_ROLE_KEYS, rows=[_row_dict(row) for row in rows])

    with pytest.raises(FrameError) as table_excinfo:
        build_frame_from_table(_observed_table(rows), role_keys=_ROLE_KEYS, stat_var=_STAT_VAR)

    assert table_excinfo.value.category == row_excinfo.value.category == "duplicate_row_key"


def test_build_frame_from_table_raises_upstream_response_error_on_second_variable():
    rows = (
        _observation(time_period="2015", value="1000"),
        _observation(
            recipient="country/KEN", time_period="2016", value="500", variable="Other_Var"
        ),
    )

    with pytest.raises(UpstreamResponseError) as excinfo:
        build_frame_from_table(_observed_table(rows), role_keys=_ROLE_KEYS, stat_var=_STAT_VAR)

    assert "Other_Var" in str(excinfo.value)
    assert _STAT_VAR in str(excinfo.value)


def test_build_frame_from_table_rejects_a_bare_stat_var_as_caller_misuse():
    rows = (_observation(time_period="2015", value="1000"),)

    with pytest.raises(MaterialisationUsageError) as excinfo:
        build_frame_from_table(
            _observed_table(rows), role_keys=_ROLE_KEYS, stat_var="MalariaGrantDisbursements"
        )

    assert "MalariaGrantDisbursements" in str(excinfo.value)
    assert "dcid:" in str(excinfo.value)


def test_build_frame_from_table_rejects_an_unfiltered_table_as_caller_misuse():
    rows = (_observation(time_period="2015", value=""),)
    unfiltered = table_from_observations(rows, roles=_ROLE_KEYS)

    with pytest.raises(MaterialisationUsageError):
        build_frame_from_table(unfiltered, role_keys=_ROLE_KEYS, stat_var=_STAT_VAR)


def test_build_frame_from_table_normalizes_bare_dcid_from_real_dc_sdmx_fixture():
    # The mixer fixture uses bare `Count_Person`. Other tests use DCID-prefixed values.
    text = (_DC_SDMX_FIXTURES_DIR / "data_single_entity.csv").read_text()
    result = parse_data_csv(text)
    assert result.rows[0].variable == "Count_Person"  # bare, confirming the premise

    table = table_from_observations(result.rows, roles=result.roles)
    observed = table.filter(columnar.observed_mask(table))
    assert observed.num_rows > 0

    frame = build_frame_from_table(observed, role_keys=result.roles, stat_var="dcid:Count_Person")

    assert len(frame.rows) == observed.num_rows


_PERIODS = ("2015", "2015-12", "2016", "2016-06", "2017")
_UNITS = ("USD", "EUR")
_METHODS = ("GrantEquivalent", "Census")
_PROVENANCES = ("OECD_DAC2A", "dc/base")
_ENTITIES = ("country/TGO", "country/USA", "country/KEN")
_VALUES = ("1000", "1000.5", "-2", "1e3", "42")


def _random_observed_row(rng: random.Random) -> SdmxObservation:
    return _observation(
        recipient=rng.choice(_ENTITIES),
        time_period=rng.choice(_PERIODS),
        value=rng.choice(_VALUES),
        unit=rng.choice(_UNITS),
        measurement_method=rng.choice(_METHODS),
        provenance=rng.choice(_PROVENANCES),
        facet_id=str(rng.randint(1, 10_000)),
    )


def test_differential_against_build_frame_over_generated_rows():
    rng = random.Random(20260904)
    for _ in range(100):
        # Unique recipient and period pairs keep duplicate-row errors out of this parity check.
        seen: set[tuple[str, str | None]] = set()
        generated: list[SdmxObservation] = []
        for _ in range(rng.randint(1, 15)):
            row = _random_observed_row(rng)
            key = (next(iter(row.role_values.values())), row.time_period)
            if key in seen:
                continue
            seen.add(key)
            generated.append(row)
        rows: tuple[SdmxObservation, ...] = tuple(generated)

        via_rows = build_frame(role_keys=_ROLE_KEYS, rows=[_row_dict(row) for row in rows])
        via_table = build_frame_from_table(
            _observed_table(rows), role_keys=_ROLE_KEYS, stat_var=_STAT_VAR
        )

        assert via_table.columns == via_rows.columns
        assert via_table.rows == via_rows.rows
        assert canonical_frame_bytes(via_table) == canonical_frame_bytes(via_rows)

        token_rows = mint_version(via_rows, graph_revision="r1", source_revisions=[], facets=[])
        token_table = mint_version(via_table, graph_revision="r1", source_revisions=[], facets=[])
        assert token_rows == token_table
