"""Tests pin `RateTableInvalidError`'s declared status and code through the conversion
classifier and the mint door's delegated wire response.
"""

from __future__ import annotations

from datasets_api.doors.conversion_errors import classify_conversion_exception
from datasets_api.doors.mint_errors import classify_mint_exception
from materialise.rate_tables import RateTableInvalidError


def test_rate_table_invalid_error_classifies_to_its_own_declared_status_and_code():
    exc = RateTableInvalidError("exchange-rate table for 'country/FRA' carries no USD anchor row")

    spec = classify_conversion_exception(exc)

    assert spec is not None
    assert spec.status == exc.status
    assert spec.code == exc.code
    assert spec.retryable == exc.retryable
    assert spec.detail == exc.detail


def test_rate_table_invalid_error_declares_capability_unavailable_501():
    # Pins the exception's own declaration, independent of the classifier,
    # so the two tests together catch either side silently drifting.
    exc = RateTableInvalidError("exemplar")

    assert exc.status == 501
    assert exc.code == "capability_unavailable"
    assert exc.retryable is False


def test_rate_table_invalid_error_reaches_the_wire_through_classify_mint_exception():
    # This route confirms that mint classification delegates to the conversion classifier.
    exc = RateTableInvalidError("provenance carries no lastDataRefreshDate")

    spec = classify_mint_exception(exc)

    assert spec is not None
    assert spec.status == exc.status
    assert spec.code == exc.code
    assert spec.retryable == exc.retryable
