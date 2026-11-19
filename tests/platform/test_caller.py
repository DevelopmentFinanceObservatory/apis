"""`resolve_caller` names the caller an allowance counts against, by the website's word or by the
client address the trusted proxies saw, with the digest the website computes."""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging

import pytest
from starlette.requests import Request

from dfo_platform.caller import (
    Caller,
    CallerHeaderInvalid,
    CallerUnresolved,
    DeliveryIdentity,
    DeliverySurface,
    delivery_identity,
    resolve_caller,
)
from dfo_platform.config import CallerSettings

_KEY = "test-key"
_PRINCIPAL = "website-secret"
_NAMED = "ip:" + "ab" * 16
_CONVERSATION = "conversation:" + "cd" * 16


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
def test_a_principal_that_does_not_match_refuses_500_and_logs_neither_value(
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

    assert refused.value.code == "internal_error"
    (record,) = caplog.records
    assert record.levelno == logging.ERROR
    assert "DFO_CALLER_PRINCIPAL" in record.getMessage()
    for value in {*presented, configured} - {""}:
        assert value not in caplog.text
        assert value not in str(refused.value)


def test_a_caller_header_without_a_principal_is_ignored():
    request = _request(("x-forwarded-for", "3.3.3.3"), ("x-dfo-caller", _NAMED))

    assert resolve_caller(request, _settings()).key == _digest("3.3.3.3")


def test_a_vouched_conversation_caller_is_named_by_address():
    """The Studio harness names a conversation, which holds no allowance of its own."""
    request = _request(
        ("x-forwarded-for", "3.3.3.3"),
        ("x-dfo-caller-principal", _PRINCIPAL),
        ("x-dfo-caller", _CONVERSATION),
    )

    assert resolve_caller(request, _settings()).key == _digest("3.3.3.3")


_VOUCHED = ("x-dfo-caller-principal", _PRINCIPAL)


@pytest.mark.parametrize(
    ("headers", "caller", "source", "surface"),
    [
        ([("x-dfo-surface", "studio")], _digest("3.3.3.3"), "address", DeliverySurface.direct),
        (
            [_VOUCHED, ("x-dfo-caller", _CONVERSATION), ("x-dfo-surface", "api")],
            _CONVERSATION,
            "vouched",
            DeliverySurface.studio,
        ),
        (
            [_VOUCHED, ("x-dfo-caller", _NAMED), ("x-dfo-surface", "Website")],
            _NAMED,
            "vouched",
            DeliverySurface.website,
        ),
        (
            [_VOUCHED, ("x-dfo-caller", _NAMED), ("x-dfo-surface", "direct")],
            _NAMED,
            "vouched",
            DeliverySurface.unknown,
        ),
        ([_VOUCHED, ("x-dfo-surface", "mcp")], None, "none", DeliverySurface.mcp),
        (
            [_VOUCHED, ("x-dfo-caller", "ip:ABC"), ("x-dfo-surface", "api")],
            None,
            "none",
            DeliverySurface.api,
        ),
        (
            [("x-dfo-caller-principal", "not-the-secret"), ("x-dfo-caller", _NAMED)],
            _digest("3.3.3.3"),
            "address",
            DeliverySurface.direct,
        ),
    ],
    ids=[
        "unvouched-surface",
        "conversation-overrules",
        "vouched-surface",
        "unnameable-surface",
        "vouched-without-caller",
        "vouched-malformed-caller",
        "refused-principal",
    ],
)
def test_the_delivery_identity_believes_a_caller_and_surface_only_beside_the_principal(
    headers: list[tuple[str, str]],
    caller: str | None,
    source: str,
    surface: DeliverySurface,
    caplog: pytest.LogCaptureFixture,
):
    request = _request(("x-forwarded-for", "3.3.3.3"), *headers)

    with caplog.at_level(logging.DEBUG, logger="dfo_platform.caller"):
        identity = delivery_identity(request, _settings())

    assert (identity.caller, identity.caller_source, identity.surface) == (caller, source, surface)
    assert caplog.records == []


def test_the_delivery_identity_carries_the_upstream_client_token_and_admission():
    claims = base64.urlsafe_b64encode(b'{"email": "website@example.iam"}').decode().rstrip("=")
    request = _request(
        ("authorization", f"Bearer h.{claims}."),
        ("x-dfo-client", "dfo-python/0.1.0 (CPython 3.14)\x07"),
        ("x-dfo-mint-admission", " Background "),
    )

    assert delivery_identity(request, _settings(key="")) == DeliveryIdentity(
        caller=None,
        caller_source="none",
        upstream="website@example.iam",
        surface=DeliverySurface.direct,
        client="dfo-python/0.1.0",
        user_agent=None,
        admission="background",
    )


@pytest.mark.parametrize(
    ("headers", "client"),
    [
        ([("x-dfo-client", "dfo-python/0.3.0 (CPython 3.14)")], "dfo-python/0.3.0"),
        ([("x-dfo-client", "claude-code/2.1.0")], "claude-code/2.1.0"),
        ([("x-dfo-client", "Mozilla/5.0 AppleWebKit/537.36")], None),
        ([], None),
    ],
    ids=["runtime-dropped", "bare-token", "not-a-product-token", "absent"],
)
def test_the_delivery_client_is_the_product_token(
    headers: list[tuple[str, str]], client: str | None
):
    assert delivery_identity(_request(*headers), _settings()).client == client


_END_CLIENT = "openai-mcp/1.0.0"
_OWN = "dfo-ai-harness/1.0"


@pytest.mark.parametrize(
    ("headers", "user_agent"),
    [
        (
            [_VOUCHED, ("x-dfo-caller", _NAMED), ("x-dfo-user-agent", _END_CLIENT)],
            _END_CLIENT,
        ),
        ([_VOUCHED, ("x-dfo-caller", _NAMED)], _OWN),
        (
            [_VOUCHED, ("x-dfo-caller", _CONVERSATION), ("x-dfo-user-agent", _END_CLIENT)],
            None,
        ),
        ([("x-dfo-user-agent", _END_CLIENT)], _OWN),
        (
            [_VOUCHED, ("x-dfo-caller", _NAMED), ("x-dfo-user-agent", "x" * 300 + "\x07")],
            "x" * 256,
        ),
    ],
    ids=["forwarded", "own", "conversation", "unvouched-forwarded-ignored", "clipped"],
)
def test_the_delivery_user_agent_is_the_end_clients(
    headers: list[tuple[str, str]], user_agent: str | None
):
    request = _request(("x-forwarded-for", "3.3.3.3"), ("user-agent", _OWN), *headers)

    assert delivery_identity(request, _settings()).user_agent == user_agent
