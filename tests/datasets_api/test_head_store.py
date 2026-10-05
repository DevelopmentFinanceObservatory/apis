"""Tests `LiveHeadStore` compare-and-set writes against the SQLite-shaped fake store.
`advance_head` updates both timestamps. `confirm_head` updates `confirmed_at` only.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.version_store import LiveHeadStore
from datasets_api.versions import MalformedSignalError, MemberSignal, decode_signal, encode_signal

from ._store_fault_doubles import PoolTimeoutFakeDatasetStore
from ._version_record_store import FakeVersionRecordDatasetStore


@pytest.fixture
def version_store():
    store = FakeVersionRecordDatasetStore()
    try:
        yield store
    finally:
        store.close()


def _insert_version_record(version_store, *, subject_id: str, content_version: str) -> None:
    with version_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
            "recorded_at) VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (subject_id, content_version) DO NOTHING",
            (subject_id, "spec", content_version, 1, {}, datetime(2020, 1, 1, tzinfo=UTC)),
        )


def test_read_head_returns_none_for_an_unknown_subject(version_store):
    store = LiveHeadStore(store=version_store)
    assert store.read_head("spec_a") is None


def test_read_head_translates_a_pool_timeout_into_datasets_store_unavailable():
    """A pool timeout before transaction entry becomes `DatasetsStoreUnavailable` without SQLSTATE."""
    store = LiveHeadStore(store=PoolTimeoutFakeDatasetStore())

    with pytest.raises(DatasetsStoreUnavailable) as excinfo:
        store.read_head("spec_a")

    assert excinfo.value.sqlstate is None
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_advance_head_writes_content_version_confirmed_at_and_signal(version_store):
    """An empty-row write with `expected_content_version=None` sets both timestamps equally."""
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    store = LiveHeadStore(store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))

    written = store.advance_head(
        subject_id="spec_a",
        content_version="cv1",
        signal=[MemberSignal("facet1", 10, "2010", "2011")],
        expected_content_version=None,
    )

    assert written is not None
    assert written.content_version == "cv1"
    assert written.confirmed_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert written.materialised_at == datetime(2026, 1, 1, tzinfo=UTC)
    head = store.read_head("spec_a")
    assert head is not None
    assert head.subject_id == "spec_a"
    assert head.content_version == "cv1"
    assert head.confirmed_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert head.materialised_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert head.signal == (MemberSignal("facet1", 10, "2010", "2011"),)


def test_advance_head_round_trips_a_role_bearing_members_unknown_obs_count(version_store):
    """A role-bearing LATEST probe has no per-series count, so `obs_count=None` survives storage."""
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    store = LiveHeadStore(store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    signal = (MemberSignal("facet1", None, "2020", "2020"),)

    store.advance_head(
        subject_id="spec_a",
        content_version="cv1",
        signal=signal,
        expected_content_version=None,
    )

    head = store.read_head("spec_a")
    assert head is not None
    assert head.signal == signal
    fresh_probe_signal = (MemberSignal("facet1", None, "2020", "2020"),)
    assert head.signal == fresh_probe_signal


def test_advance_head_round_trips_observation_period_and_provenance_refresh(version_store):
    """(#439) Refresh fields round-trip through jsonb.
    Absent refresh values remain `None`.
    """
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    store = LiveHeadStore(store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    signal = (
        MemberSignal(
            "facet1",
            10,
            "2010",
            "2011",
            observation_period="P1Y",
            provenance_last_refresh="2026-01-01",
            provenance_next_refresh="2026-02-01",
        ),
        MemberSignal("facet2", 5, "2010", "2011"),
    )

    store.advance_head(
        subject_id="spec_a",
        content_version="cv1",
        signal=signal,
        expected_content_version=None,
    )

    head = store.read_head("spec_a")
    assert head is not None
    assert head.signal == signal
    assert head.signal[1].observation_period is None
    assert head.signal[1].provenance_last_refresh is None
    assert head.signal[1].provenance_next_refresh is None


def test_advance_head_round_trips_rate_provenance_refresh_and_omits_it_when_empty(version_store):
    """(C5c) Converted refresh data round-trips.
    An empty value omits the jsonb key.
    """
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    store = LiveHeadStore(store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    signal = (
        MemberSignal(
            "facet1",
            10,
            "2010",
            "2011",
            rate_provenance_refresh=(("RATE_PROV", "2026-01-01"),),
        ),
        MemberSignal("facet2", 5, "2010", "2011"),
    )

    store.advance_head(
        subject_id="spec_a",
        content_version="cv1",
        signal=signal,
        expected_content_version=None,
    )

    encoded = encode_signal(signal)
    encoded_members = encoded["members"]
    assert encoded_members[0]["rate_provenance_refresh"] == [["RATE_PROV", "2026-01-01"]]
    assert "rate_provenance_refresh" not in encoded_members[1]

    head = store.read_head("spec_a")
    assert head is not None
    assert head.signal == signal
    assert head.signal[1].rate_provenance_refresh == ()


def test_advance_head_moves_both_timestamps_on_a_real_advance(version_store):
    """Changing `content_version` updates both timestamps to the writer's clock."""
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv2")
    first_write = datetime(2026, 1, 1, tzinfo=UTC)
    second_write = datetime(2026, 1, 1, 0, 15, tzinfo=UTC)
    LiveHeadStore(store=version_store, now=lambda: first_write).advance_head(
        subject_id="spec_a",
        content_version="cv1",
        signal=[MemberSignal("facet1", 1, None, None)],
        expected_content_version=None,
    )

    written = LiveHeadStore(store=version_store, now=lambda: second_write).advance_head(
        subject_id="spec_a",
        content_version="cv2",
        signal=[MemberSignal("facet2", 2, None, None)],
        expected_content_version="cv1",
    )

    assert written is not None
    assert written.content_version == "cv2"
    assert written.confirmed_at == second_write
    assert written.materialised_at == second_write


def test_advance_head_the_slower_older_decision_loses_the_race_and_returns_none(version_store):
    """Two requests read `cv0`. The later decision writes `cv2` first, so the earlier decision
    reaches the store with a stale expectation and loses its compare-and-set.
    The compare-and-set follows the stored version, independent of request start time.
    """
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv0")
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv2")
    store = LiveHeadStore(store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    store.advance_head(
        subject_id="spec_a",
        content_version="cv0",
        signal=[MemberSignal("facet0", 0, None, None)],
        expected_content_version=None,
    )

    # The later-started decision materialises fast and writes first.
    new_result = store.advance_head(
        subject_id="spec_a",
        content_version="cv2",
        signal=[MemberSignal("facet2", 2, None, None)],
        expected_content_version="cv0",
    )
    assert new_result is not None
    assert new_result.content_version == "cv2"

    # The earlier decision also read "cv0" and reaches the store after the later write.
    old_result = store.advance_head(
        subject_id="spec_a",
        content_version="cv1",
        signal=[MemberSignal("facet1", 1, None, None)],
        expected_content_version="cv0",
    )
    assert old_result is None

    head = store.read_head("spec_a")
    assert head is not None
    assert head.content_version == "cv2"
    assert head.signal == (MemberSignal("facet2", 2, None, None),)


def test_advance_head_expected_none_against_an_existing_row_returns_none(version_store):
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    store = LiveHeadStore(store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    store.advance_head(
        subject_id="spec_a",
        content_version="cv1",
        signal=[MemberSignal("facet1", 1, None, None)],
        expected_content_version=None,
    )

    result = store.advance_head(
        subject_id="spec_a",
        content_version="cv1",
        signal=[MemberSignal("facet1", 1, None, None)],
        expected_content_version=None,
    )

    assert result is None


def test_advance_head_conflict_leaves_the_stored_row_unchanged(version_store):
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv2")
    store = LiveHeadStore(store=version_store, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    store.advance_head(
        subject_id="spec_a",
        content_version="cv1",
        signal=[MemberSignal("facet1", 1, None, None)],
        expected_content_version=None,
    )
    before = store.read_head("spec_a")

    result = store.advance_head(
        subject_id="spec_a",
        content_version="cv2",
        signal=[MemberSignal("facet2", 2, None, None)],
        expected_content_version="cv-wrong",
    )

    assert result is None
    after = store.read_head("spec_a")
    assert after == before


def test_advance_head_on_the_same_content_version_is_a_rematerialisation(version_store):
    """Rematerialising the stored version still applies and advances both timestamps."""
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    first_check = datetime(2026, 1, 1, tzinfo=UTC)
    second_check = datetime(2026, 1, 1, 0, 15, tzinfo=UTC)

    LiveHeadStore(store=version_store, now=lambda: first_check).advance_head(
        subject_id="spec_a",
        content_version="cv1",
        signal=[MemberSignal("facet1", 1, None, None)],
        expected_content_version=None,
    )
    written = LiveHeadStore(store=version_store, now=lambda: second_check).advance_head(
        subject_id="spec_a",
        content_version="cv1",
        signal=[MemberSignal("facet1", 1, None, None)],
        expected_content_version="cv1",
    )

    assert written is not None
    assert written.confirmed_at == second_check
    assert written.materialised_at == second_check
    head = LiveHeadStore(store=version_store).read_head("spec_a")
    assert head is not None
    assert head.content_version == "cv1"
    assert head.confirmed_at == second_check
    assert head.materialised_at == second_check


def test_confirm_head_on_the_stored_content_version_moves_confirmed_at_only(version_store):
    """A current probe advances `confirmed_at` and preserves the version, materialisation time,
    and signal.
    """
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    materialised_at = datetime(2026, 1, 1, tzinfo=UTC)
    confirmed_at = datetime(2026, 1, 1, 0, 15, tzinfo=UTC)
    LiveHeadStore(store=version_store, now=lambda: materialised_at).advance_head(
        subject_id="spec_a",
        content_version="cv1",
        signal=[MemberSignal("facet1", 1, None, None)],
        expected_content_version=None,
    )

    confirmed = LiveHeadStore(store=version_store, now=lambda: confirmed_at).confirm_head(
        subject_id="spec_a", content_version="cv1"
    )

    assert confirmed is not None
    assert confirmed.content_version == "cv1"
    assert confirmed.confirmed_at == confirmed_at
    assert confirmed.materialised_at == materialised_at
    assert confirmed.signal == (MemberSignal("facet1", 1, None, None),)
    head = LiveHeadStore(store=version_store).read_head("spec_a")
    assert head is not None
    assert head.confirmed_at == confirmed_at
    assert head.materialised_at == materialised_at


def test_confirm_head_against_a_different_content_version_returns_none_and_leaves_the_row(
    version_store,
):
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv2")
    materialised_at = datetime(2026, 1, 1, tzinfo=UTC)
    store = LiveHeadStore(store=version_store, now=lambda: materialised_at)
    store.advance_head(
        subject_id="spec_a",
        content_version="cv1",
        signal=[MemberSignal("facet1", 1, None, None)],
        expected_content_version=None,
    )
    before = store.read_head("spec_a")

    result = LiveHeadStore(
        store=version_store, now=lambda: datetime(2026, 1, 1, 0, 15, tzinfo=UTC)
    ).confirm_head(subject_id="spec_a", content_version="cv2")

    assert result is None
    after = store.read_head("spec_a")
    assert after == before


def test_confirm_head_on_an_unknown_subject_returns_none(version_store):
    store = LiveHeadStore(store=version_store)

    result = store.confirm_head(subject_id="spec_missing", content_version="cv1")

    assert result is None


# `decode_signal` malformed-shape coverage. Direct calls to `decode_signal`
# itself, bypassing a store, so each case pins the exact field
# `MalformedSignalError` rejects.


def test_decode_signal_rejects_a_non_object_top_level():
    with pytest.raises(MalformedSignalError):
        decode_signal(["members"])  # type: ignore[arg-type]


def test_decode_signal_rejects_a_non_list_members():
    with pytest.raises(MalformedSignalError):
        decode_signal({"members": "not-a-list"})


def test_decode_signal_rejects_a_non_object_member():
    with pytest.raises(MalformedSignalError):
        decode_signal({"members": ["not-an-object"]})


def test_decode_signal_rejects_a_member_missing_obs_count():
    with pytest.raises(MalformedSignalError):
        decode_signal({"members": [{"dispatch_facet_id": "facet1"}]})


def test_decode_signal_rejects_a_member_with_a_wrong_typed_obs_count():
    with pytest.raises(MalformedSignalError):
        decode_signal(
            {
                "members": [
                    {
                        "dispatch_facet_id": "facet1",
                        "obs_count": "10",
                        "earliest_date": None,
                        "latest_date": None,
                    }
                ]
            }
        )


def test_decode_signal_rejects_a_member_with_a_wrong_typed_date():
    with pytest.raises(MalformedSignalError):
        decode_signal(
            {
                "members": [
                    {
                        "dispatch_facet_id": "facet1",
                        "obs_count": 10,
                        "earliest_date": 2010,
                        "latest_date": None,
                    }
                ]
            }
        )


def test_decode_signal_accepts_none_dates_and_none_dispatch_facet_id():
    decoded = decode_signal(
        {
            "members": [
                {
                    "dispatch_facet_id": None,
                    "obs_count": 1,
                    "earliest_date": None,
                    "latest_date": None,
                }
            ]
        }
    )
    assert decoded == (MemberSignal(None, 1, None, None),)


def test_decode_signal_accepts_a_member_missing_rate_provenance_refresh():
    """(C5c) A stored head minted before this field existed carries no
    `"rate_provenance_refresh"` key at all. It decodes to `()`, the same
    fallback an absent `observation_period`/refresh field already gets.
    """
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
    assert decoded[0].rate_provenance_refresh == ()


def test_decode_signal_round_trips_rate_provenance_refresh():
    decoded = decode_signal(
        {
            "members": [
                {
                    "dispatch_facet_id": "facet1",
                    "obs_count": 1,
                    "earliest_date": None,
                    "latest_date": None,
                    "rate_provenance_refresh": [["RATE_PROV", "2026-01-01"], ["OTHER", None]],
                }
            ]
        }
    )
    assert decoded[0].rate_provenance_refresh == (("RATE_PROV", "2026-01-01"), ("OTHER", None))


def test_decode_signal_rejects_a_non_list_rate_provenance_refresh():
    with pytest.raises(MalformedSignalError):
        decode_signal(
            {
                "members": [
                    {
                        "dispatch_facet_id": "facet1",
                        "obs_count": 1,
                        "earliest_date": None,
                        "latest_date": None,
                        "rate_provenance_refresh": "not-a-list",
                    }
                ]
            }
        )


def test_decode_signal_rejects_a_malformed_rate_provenance_refresh_entry():
    with pytest.raises(MalformedSignalError):
        decode_signal(
            {
                "members": [
                    {
                        "dispatch_facet_id": "facet1",
                        "obs_count": 1,
                        "earliest_date": None,
                        "latest_date": None,
                        "rate_provenance_refresh": [["RATE_PROV"]],
                    }
                ]
            }
        )


@pytest.mark.parametrize("clock_window", [(2016, 2026), (2028, None), (None, 2026)])
def test_a_signal_round_trips_its_clock_window(clock_window) -> None:
    signal = (MemberSignal("facet1", 1, None, None, clock_window=clock_window),)

    assert decode_signal(encode_signal(signal)) == signal


def test_a_signal_with_no_clock_window_encodes_without_one() -> None:
    encoded = encode_signal((MemberSignal("facet1", 1, None, None),))

    assert "clock_window" not in encoded["members"][0]
    assert decode_signal(encoded)[0].clock_window is None


@pytest.mark.parametrize("clock_window", ["2016", [2016], [2016, "2026"], [True, 2026]])
def test_decode_signal_rejects_a_malformed_clock_window(clock_window) -> None:
    with pytest.raises(MalformedSignalError):
        decode_signal(
            {
                "members": [
                    {
                        "dispatch_facet_id": "facet1",
                        "obs_count": 1,
                        "earliest_date": None,
                        "latest_date": None,
                        "clock_window": clock_window,
                    }
                ]
            }
        )


def _insert_raw_head_row(
    version_store, *, subject_id: str, content_version: str, raw_signal_json: str
) -> None:
    """Inserts raw JSON for malformed stored rows that `encode_signal` cannot create."""
    with version_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.subject_head "
            "(subject_id, content_version, confirmed_at, materialised_at, signal) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                subject_id,
                content_version,
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 1, tzinfo=UTC),
                raw_signal_json,
            ),
        )


def test_read_head_translates_a_malformed_signal_into_datasets_store_unavailable(version_store):
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    _insert_raw_head_row(
        version_store,
        subject_id="spec_a",
        content_version="cv1",
        raw_signal_json=json.dumps({"members": [{"dispatch_facet_id": "facet1"}]}),
    )
    store = LiveHeadStore(store=version_store)

    with pytest.raises(DatasetsStoreUnavailable) as excinfo:
        store.read_head("spec_a")

    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_read_head_translates_a_non_object_top_level_signal(version_store):
    _insert_version_record(version_store, subject_id="spec_a", content_version="cv1")
    _insert_raw_head_row(
        version_store,
        subject_id="spec_a",
        content_version="cv1",
        raw_signal_json=json.dumps(["not", "an", "object"]),
    )
    store = LiveHeadStore(store=version_store)

    with pytest.raises(DatasetsStoreUnavailable):
        store.read_head("spec_a")
