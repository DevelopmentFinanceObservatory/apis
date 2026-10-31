"""`resolve_caller` names the caller an allowance counts against, by the website's word or by the
client address the trusted proxies saw, with the digest the website computes."""

from __future__ import annotations

import hashlib
import hmac
import logging

import pytest
from starlette.requests import Request

from dfo_platform.caller import (
    Caller,
    CallerHeaderInvalid,
    CallerUnresolved,
    resolve_caller,
)
from dfo_platform.config import CallerSettings

_KEY = "test-key"
_PRINCIPAL = "website-secret"
_NAMED = "ip:" + "ab" * 16


def _digest(preimage: str) -> str:
    """The website's digest (`caller.test.ts`): HMAC-SHA-256 of the preimage, hex, 32 long."""
    return "ip:" + hmac.new(_KEY.encode(), preimage.encode(), hashlib.sha256).hexdigest()[:32]


def _request(*headers: tuple[str, str]) -> Request:
    # `client` is the address ProxyHeadersMiddleware would write from the leftmost entry.
    return Request(
        {
            "type": "http",
            "headers": [(name.encode(), value.encode()) for name, value in headers],
            "client": ("9.9.9.9", 1),
        }
    )


def _settings(*, key: str = _KEY, principal: str = _PRINCIPAL, hops: int = 0) -> CallerSettings:
    return CallerSettings(caller_hmac_key=key, caller_principal=principal, trusted_proxy_hops=hops)


def _resolve(forwarded_for: str, *, hops: int = 0) -> Caller:
    return resolve_caller(_request(("x-forwarded-for", forwarded_for)), _settings(hops=hops))


@pytest.mark.parametrize(
    ("address", "preimage"),
    [
        ("3.3.3.3", "3.3.3.3"),
        ("2001:db8:1:2:aaaa::1", "2001:db8:1:2::/64"),
        ("2001:db8:1:2:bbbb:cccc:dddd:eeee", "2001:db8:1:2::/64"),
        ("2001:db8::1", "2001:db8:0:0::/64"),
        ("fe80::1%eth0", "fe80:0:0:0::/64"),
        ("::ffff:203.0.113.7", "203.0.113.7"),
        ("::ffff:cb00:7107", "203.0.113.7"),
    ],
)
def test_an_address_is_digested_as_the_website_digests_it(address: str, preimage: str):
    assert _resolve(address) == Caller(kind="address", key=_digest(preimage))


@pytest.mark.parametrize(
    ("forwarded_for", "hops", "client"),
    [
        ("1.1.1.1, 2.2.2.2, 3.3.3.3", 0, "3.3.3.3"),
        ("1.1.1.1, 2.2.2.2, 3.3.3.3", 1, "2.2.2.2"),
        ("1.1.1.1,2.2.2.2,3.3.3.3", 2, "1.1.1.1"),
        # A client-written leftmost entry changes nothing.
        ("6.6.6.6, 3.3.3.3", 0, "3.3.3.3"),
    ],
)
def test_the_client_is_the_entry_the_trusted_hops_place_from_the_right(
    forwarded_for: str, hops: int, client: str
):
    assert _resolve(forwarded_for, hops=hops).key == _digest(client)


def test_forwarded_for_header_lines_read_as_one_list():
    request = _request(("x-forwarded-for", "1.1.1.1"), ("x-forwarded-for", "2.2.2.2"))

    assert resolve_caller(request, _settings(hops=1)).key == _digest("1.1.1.1")


@pytest.mark.parametrize(
    ("headers", "hops"),
    [
        ([("x-forwarded-for", "3.3.3.3")], 1),
        # Without X-Forwarded-For, `request.client` is not read in its place.
        ([], 0),
        ([("x-forwarded-for", "")], 0),
        ([("x-forwarded-for", "1.1.1.1, ")], 0),
        ([("x-forwarded-for", "unknown")], 0),
        ([("x-forwarded-for", "1.1.1.1:443")], 0),
    ],
    ids=["too-few-entries", "no-header", "empty", "empty-entry", "not-an-address", "with-port"],
)
def test_a_missing_or_unreadable_client_address_refuses(headers: list[tuple[str, str]], hops: int):
    with pytest.raises(CallerUnresolved):
        resolve_caller(_request(*headers), _settings(hops=hops))


def test_without_a_digest_key_a_direct_caller_refuses():
    with pytest.raises(CallerUnresolved, match="DFO_CALLER_HMAC_KEY"):
        resolve_caller(_request(("x-forwarded-for", "3.3.3.3")), _settings(key=""))


def test_the_website_names_its_caller_under_the_principal():
    request = _request(
        ("x-forwarded-for", "3.3.3.3"),
        ("x-dfo-caller-principal", _PRINCIPAL),
        ("x-dfo-caller", _NAMED),
    )

    # The named caller needs no key of this service's own.
    assert resolve_caller(request, _settings(key="")) == Caller(kind="address", key=_NAMED)


@pytest.mark.parametrize(
    "callers",
    [
        [],
        ["ip:" + "AB" * 16],
        ["ip:" + "ab" * 15],
        ["ab" * 16],
        [_NAMED, _NAMED],
    ],
    ids=["missing", "uppercase", "short", "no-prefix", "repeated"],
)
def test_the_principal_without_one_well_formed_caller_refuses_400(callers: list[str]):
    request = _request(
        ("x-forwarded-for", "3.3.3.3"),
        ("x-dfo-caller-principal", _PRINCIPAL),
        *(("x-dfo-caller", caller) for caller in callers),
    )

    with pytest.raises(CallerHeaderInvalid):
        resolve_caller(request, _settings())


@pytest.mark.parametrize(
    ("configured", "presented"),
    [
        (_PRINCIPAL, ["not-the-secret"]),
        (_PRINCIPAL, [_PRINCIPAL, _PRINCIPAL]),
        ("", [""]),
        ("", [_PRINCIPAL]),
    ],
    ids=["wrong-secret", "repeated-secret", "empty-none-configured", "none-configured"],
)
def test_a_principal_that_does_not_match_refuses_503_and_logs_neither_value(
    configured: str, presented: list[str], caplog: pytest.LogCaptureFixture
):
    """Falling back to the address would count every website client as the website."""
    request = _request(
        ("x-forwarded-for", "3.3.3.3"),
        *(("x-dfo-caller-principal", value) for value in presented),
        ("x-dfo-caller", _NAMED),
    )

    with (
        caplog.at_level(logging.ERROR, logger="dfo_platform.caller"),
        pytest.raises(CallerUnresolved) as refused,
    ):
        resolve_caller(request, _settings(principal=configured))

    assert refused.value.code == "service_unavailable"
    (record,) = caplog.records
    assert record.levelno == logging.ERROR
    assert "DFO_CALLER_PRINCIPAL" in record.getMessage()
    for value in {*presented, configured} - {""}:
        assert value not in caplog.text
        assert value not in str(refused.value)


def test_a_caller_header_without_a_principal_is_ignored():
    request = _request(("x-forwarded-for", "3.3.3.3"), ("x-dfo-caller", _NAMED))

    assert resolve_caller(request, _settings()).key == _digest("3.3.3.3")
