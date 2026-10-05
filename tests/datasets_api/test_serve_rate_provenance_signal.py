"""Tests that head signals take rate-provenance refresh data from completed materialisations,
using hand-built `ConversionRecord` values. The signal carries each table's admitted refresh
date; the row digest stays in the content version.
"""

from __future__ import annotations

from datasets_api.serve import _rate_provenance_refresh, _signal_with_rate_provenance
from datasets_api.versions import MemberSignal
from materialise.convert import ConversionRecord

_SIGNAL = (MemberSignal("facet1", 10, "2020", "2021"),)


# A row digest, distinct from every refresh date, so a signal that stored the digest fails.
_TABLE_DIGEST = "d" * 64


def _record(*, table_provenance_id: str, refresh_date: str) -> ConversionRecord:
    return ConversionRecord(
        recipe_id="usd_current",
        exchange_rate_table_version=_TABLE_DIGEST,
        deflator_table_version=None,
        native_currency="LCU",
        native_unit_ref="LCU",
        table_provenance_id=table_provenance_id,
        table_refresh_date=refresh_date,
        fallback_count=0,
        fallback_examples=(),
        projected_count=0,
        projected_examples=(),
    )


def test_rate_provenance_refresh_is_empty_when_every_record_is_none():
    assert _rate_provenance_refresh([None]) == ()
    assert _rate_provenance_refresh([]) == ()


def test_rate_provenance_refresh_names_one_pair_for_a_single_conversion():
    record = _record(table_provenance_id="RATE_PROV", refresh_date="2026-01-01")
    assert _rate_provenance_refresh([record]) == (("RATE_PROV", "2026-01-01"),)


def test_rate_provenance_refresh_dedupes_a_groups_shared_table_provenance():
    """Two member records sharing a provenance and version produce one refresh entry."""
    grant = _record(table_provenance_id="RATE_PROV", refresh_date="2026-01-01")
    loan = _record(table_provenance_id="RATE_PROV", refresh_date="2026-01-01")
    assert _rate_provenance_refresh([grant, loan]) == (("RATE_PROV", "2026-01-01"),)


def test_rate_provenance_refresh_ignores_a_members_own_none_within_a_group():
    """Only the converted member contributes a rate-provenance refresh."""
    converted = _record(table_provenance_id="RATE_PROV", refresh_date="2026-01-01")
    assert _rate_provenance_refresh([converted, None]) == (("RATE_PROV", "2026-01-01"),)


def test_signal_with_rate_provenance_sets_every_entry_of_a_group_signal():
    grant = _record(table_provenance_id="RATE_PROV", refresh_date="2026-01-01")
    loan = _record(table_provenance_id="RATE_PROV", refresh_date="2026-01-01")

    signal = _signal_with_rate_provenance(_SIGNAL, {"GrantA": grant, "LoanB": loan}.values())

    assert signal == (
        MemberSignal(
            "facet1", 10, "2020", "2021", rate_provenance_refresh=(("RATE_PROV", "2026-01-01"),)
        ),
    )


def test_signal_with_rate_provenance_carries_a_collection_members_own_entry():
    """A collection member's signal carries only that member's rate provenance."""
    member_signal = MemberSignal("facet2", 5, "2020", "2021")
    record = _record(table_provenance_id="OTHER_RATE_PROV", refresh_date="2026-03-01")

    signal = _signal_with_rate_provenance((member_signal,), (record,))

    assert signal == (
        MemberSignal(
            "facet2",
            5,
            "2020",
            "2021",
            rate_provenance_refresh=(("OTHER_RATE_PROV", "2026-03-01"),),
        ),
    )


def test_signal_with_rate_provenance_leaves_an_unconverted_member_empty():
    signal = _signal_with_rate_provenance(_SIGNAL, [None])
    assert signal == _SIGNAL
    assert signal[0].rate_provenance_refresh == ()
