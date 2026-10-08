"""The publish and live-refresh caller checks compare a principal in constant time through
`datasets_api.principals`, and refuse a wrong or repeated principal header."""

from __future__ import annotations

import hmac

import pytest
from fastapi import Request

import datasets_api.principals as principals_module
from datasets_api.doors.publish import PublishNotPermitted, PublishPrincipals
from datasets_api.live_refresh import LiveRefreshPrincipal, RefreshNotPermitted

_HEADER = "x-dfo-principal"
_PRINCIPAL = "presentation-mcp"
_PUBLISH = PublishPrincipals(header=_HEADER, allowed=frozenset({_PRINCIPAL, "second-caller"}))
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
    [(_PUBLISH, PublishNotPermitted), (_LIVE_REFRESH, RefreshNotPermitted)],
    ids=["publish", "live-refresh"],
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
    check: PublishPrincipals | LiveRefreshPrincipal,
    refusal: type[Exception],
    presented: tuple[str, ...],
) -> None:
    with pytest.raises(refusal):
        check.admit(_request(*presented))


def test_an_empty_configured_principal_admits_nobody() -> None:
    unset = PublishPrincipals(header=_HEADER, allowed=frozenset({""}))

    with pytest.raises(PublishNotPermitted):
        unset.admit(_request(""))


def test_each_check_compares_every_configured_principal_in_constant_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compared: list[tuple[bytes, bytes]] = []
    compare_digest = hmac.compare_digest

    def recording(presented: bytes, configured: bytes) -> bool:
        compared.append((presented, configured))
        return compare_digest(presented, configured)

    monkeypatch.setattr(principals_module.hmac, "compare_digest", recording)

    _PUBLISH.admit(_request(_PRINCIPAL))
    # A match does not end the loop: every allowed principal is compared.
    assert sorted(compared) == [
        (_PRINCIPAL.encode(), _PRINCIPAL.encode()),
        (_PRINCIPAL.encode(), b"second-caller"),
    ]

    compared.clear()
    _LIVE_REFRESH.admit(_request(_PRINCIPAL))
    assert compared == [(_PRINCIPAL.encode(), _PRINCIPAL.encode())]
