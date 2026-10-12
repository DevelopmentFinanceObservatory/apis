"""`Principals` reads a request's principal header into grants, comparing the value with every
configured secret in constant time, and refuses a wrong, repeated, or missing header."""

from __future__ import annotations

import hmac

import pytest
from fastapi import Request

import datasets_api.principals as principals_module
from datasets_api.live_refresh import LiveRefreshPrincipal
from datasets_api.principals import (
    Grant,
    Principals,
    PublishNotPermitted,
    RefreshNotPermitted,
)

_HEADER = "x-dfo-principal"
_PRINCIPAL = "presentation-mcp"
_PRINCIPALS = Principals(
    header=_HEADER,
    secrets=(
        (_PRINCIPAL, Grant.publish),
        ("second-caller", Grant.publish),
        (_PRINCIPAL, Grant.refresh),
    ),
)
_LIVE_REFRESH = LiveRefreshPrincipal(header=_HEADER, principal=_PRINCIPAL)


def _request(*principals: str) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [(_HEADER.encode(), principal.encode()) for principal in principals],
        }
    )


@pytest.mark.parametrize(
    ("check", "refusal"),
    [
        (_PRINCIPALS.require(Grant.publish), PublishNotPermitted),
        (_PRINCIPALS.require(Grant.refresh), RefreshNotPermitted),
        (_LIVE_REFRESH.admit, RefreshNotPermitted),
    ],
    ids=["publish", "refresh", "live-refresh-view"],
)
@pytest.mark.parametrize(
    "presented",
    [
        (_PRINCIPAL[:-1] + "x",),
        (_PRINCIPAL[:-1],),
        (_PRINCIPAL + "x",),
        ("",),
        (_PRINCIPAL, _PRINCIPAL),
        (),
    ],
    ids=["equal-length", "shorter", "longer", "empty", "repeated", "missing"],
)
def test_a_wrong_or_repeated_principal_is_refused(
    check, refusal: type[Exception], presented: tuple[str, ...]
) -> None:
    with pytest.raises(refusal):
        check(_request(*presented))


def test_a_secret_configured_under_two_names_holds_both_grants() -> None:
    principal = _PRINCIPALS.parse(_request(_PRINCIPAL))

    assert principal.grants == {Grant.publish, Grant.refresh}
    assert _PRINCIPALS.parse(_request("second-caller")).grants == {Grant.publish}


def test_an_empty_configured_principal_admits_nobody() -> None:
    unset = Principals(header=_HEADER, secrets=(("", Grant.publish),))

    with pytest.raises(PublishNotPermitted):
        unset.require(Grant.publish)(_request(""))
    with pytest.raises(RefreshNotPermitted):
        LiveRefreshPrincipal(header=_HEADER, principal="").admit(_request(""))


def test_parse_compares_every_configured_secret_in_constant_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compared: list[tuple[bytes, bytes]] = []
    compare_digest = hmac.compare_digest

    def recording(presented: bytes, configured: bytes) -> bool:
        compared.append((presented, configured))
        return compare_digest(presented, configured)

    monkeypatch.setattr(principals_module.hmac, "compare_digest", recording)

    _PRINCIPALS.require(Grant.publish)(_request(_PRINCIPAL))
    # A match does not end the loop: every configured secret is compared.
    assert sorted(compared) == sorted(
        (_PRINCIPAL.encode(), secret.encode()) for secret, _grant in _PRINCIPALS.secrets
    )
