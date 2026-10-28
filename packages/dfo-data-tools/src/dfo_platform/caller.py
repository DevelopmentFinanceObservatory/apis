"""Who called: the Cloud Run identity token's claims, logged once per request.

Cloud Run's IAM layer checks a caller's Google ID token and strips its signature
before the container sees the header, so the payload read here is unverified. It
names the caller for logs and says nothing about authorisation.

`X-Serverless-Authorization` carries the token when a caller sends one, as Cloud
Run checks it ahead of `Authorization`. `Authorization` is the fallback. Neither
header's value, signature included, reaches a log line. Only the `email` and `aud`
claims and a presence flag do.

`log_request` writes one `dfo_platform.request` line per request, through the
`json_fields` record attribute that `dfo_platform.cloud_logging` merges into the
entry. While the service is open to every caller, filtering these lines for a
missing token or an unexpected `email` lists the callers an invoker-only service
would turn away.

The same line carries the SDK's `X-DFO-Client` header, which names the client
as `<name>/<version> (<runtime>)`, for example `dfo-python/0.1.0 (CPython 3.14)`.
The caller writes the value, so it is logged with non-printable characters removed
and clipped, as `client_header`, and split into `client_name`, `client_version` and
`client_runtime` when it has that shape. A request without the header logs none of
these fields.

`resolve_caller` names the caller an allowance counts against, as a `Caller` whose key is
`ip:` and the first 32 hex digits of HMAC-SHA-256 over the client's address, folded
(`fold_address`), under `DFO_CALLER_HMAC_KEY`. The website computes the same key for the
callers it forwards (its `src/lib/proxy/caller.ts`), so a client has one key on either path
and no raw address is stored. A request whose `x-dfo-caller-principal` matches
`DFO_CALLER_PRINCIPAL` comes from the website, which names its client in `x-dfo-caller`. A
request that carries `x-dfo-caller-principal` without matching it, or while `DFO_CALLER_PRINCIPAL`
is unset, refuses 503 `CallerUnresolved` and logs the mismatch at ERROR, since counting it by
address would put every client of the website on the website's own allowance. A request with no
`x-dfo-caller-principal` is named from the X-Forwarded-For entry `DFO_TRUSTED_PROXY_HOPS` places
from the right, the entry the proxies in front of this service appended. `request.client` is never read:
`ProxyHeadersMiddleware` sets it from the leftmost entry, which the client writes.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from dfo_platform.cloud_logging import JSON_FIELDS_ATTR
from dfo_platform.config import CallerSettings
from dfo_platform.problem import Refusal

if TYPE_CHECKING:
    from starlette.requests import Request

SERVERLESS_AUTHORIZATION_HEADER = "x-serverless-authorization"
AUTHORIZATION_HEADER = "authorization"
CLIENT_HEADER = "X-DFO-Client"
"""Request header naming the SDK client, which browsers may send cross-origin."""

_CLIENT_FORMAT = re.compile(
    r"(?P<name>[^/\s()]+)/(?P<version>[^\s()]+)(?: \((?P<runtime>[^()]+)\))?"
)

_MAX_FIELD_LENGTH = 256
"""Longest caller-controlled string logged. A junk token costs one short line."""

_MAX_CREDENTIAL_LENGTH = 4096
"""Longest credential decoded. A Google ID token is about 1 KB, so a longer value is junk,
and decoding it would let a crafted payload exhaust the JSON parser's recursion limit."""

_log = logging.getLogger("dfo_platform.request")
_caller_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CallerIdentity:
    """The unverified identity a request's token claims.

    `email` and `aud` are `None` when no token arrived, the token is not a JWT, or the
    claim is absent or not a string.
    """

    token_present: bool
    email: str | None = None
    aud: str | None = None


def _clip(value: str) -> str:
    return value[:_MAX_FIELD_LENGTH]


def _clean(value: str) -> str:
    """Drop non-printable characters (C0 and C1 controls, tabs, the soft hyphen) and
    surrounding space, then clip."""
    return _clip("".join(ch for ch in value if ch.isprintable()).strip())


def _claim(claims: Mapping[str, object], name: str) -> str | None:
    value = claims.get(name)
    return _clip(value) if isinstance(value, str) else None


def _unverified_claims(credential: str) -> Mapping[str, object]:
    """Decode a JWT's payload segment, or return no claims when it cannot be read."""
    if len(credential) > _MAX_CREDENTIAL_LENGTH:
        return {}
    segments = credential.split(".")
    if len(segments) < 2:
        return {}
    payload = segments[1]
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (ValueError, RecursionError):
        # ValueError covers bad base64, bad UTF-8 and bad JSON. RecursionError covers
        # a payload nested deeper than the parser allows.
        return {}
    return claims if isinstance(claims, dict) else {}


def caller_identity_from_headers(headers: Mapping[str, str]) -> CallerIdentity:
    """Read the caller's claimed identity from the request headers. It never raises.

    `headers` must look up names case-insensitively, as Starlette's `Headers` does.
    """
    value = (
        headers.get(SERVERLESS_AUTHORIZATION_HEADER, "").strip()
        or headers.get(AUTHORIZATION_HEADER, "").strip()
    )
    if not value:
        return CallerIdentity(token_present=False)
    scheme, _, credential = value.partition(" ")
    claims = _unverified_claims(credential.strip() if scheme.lower() == "bearer" else "")
    return CallerIdentity(
        token_present=True, email=_claim(claims, "email"), aud=_claim(claims, "aud")
    )


def client_fields_from_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return the `X-DFO-Client` log fields, or none when the header is absent or blank.

    `client_header` is the cleaned value. The parsed fields appear only when the value
    has the `<name>/<version> (<runtime>)` shape, `client_runtime` only when it names one.
    """
    raw = headers.get(CLIENT_HEADER)
    value = _clean(raw) if raw is not None else ""
    if not value:
        return {}
    fields = {"client_header": value}
    if match := _CLIENT_FORMAT.fullmatch(value):
        fields["client_name"] = match["name"]
        fields["client_version"] = match["version"]
        if match["runtime"] is not None:
            fields["client_runtime"] = match["runtime"]
    return fields


def log_request(*, method: str, path: str, status: int | None, headers: Mapping[str, str]) -> None:
    """Log one line for a finished request with the caller's claimed identity and the
    SDK client its `X-DFO-Client` header names.

    `status` is `None` when the request ended without a response.
    """
    caller = caller_identity_from_headers(headers)
    user_agent = headers.get("user-agent")
    _log.info(
        "request",
        extra={
            JSON_FIELDS_ATTR: {
                "http_method": method,
                "http_path": path,
                "http_status": status,
                "user_agent": _clip(user_agent) if user_agent is not None else None,
                "caller_token_present": caller.token_present,
                "caller_email": caller.email,
                "caller_aud": caller.aud,
                **client_fields_from_headers(headers),
            }
        },
    )


CALLER_HEADER = "x-dfo-caller"
"""The caller the website resolved for a request it forwards, as `ip:<32 hex digits>`."""
CALLER_PRINCIPAL_HEADER = "x-dfo-caller-principal"
"""The secret that vouches for `CALLER_HEADER`, `DFO_CALLER_PRINCIPAL`."""
FORWARDED_FOR_HEADER = "x-forwarded-for"

ADDRESS_KEY_PREFIX = "ip:"
"""The prefix of every `kind="address"` caller key. Reference rows keyed by a caller start with
it, which tells them from the `secret-sha256:` labels of principals."""

_DIGEST_LENGTH = 32
_ADDRESS_KEY = re.compile(rf"{ADDRESS_KEY_PREFIX}[0-9a-f]{{{_DIGEST_LENGTH}}}")


@dataclass(frozen=True, slots=True)
class Caller:
    """Who an allowance counts against. `kind="address"` keys a client by its network address,
    the only kind there is until callers can present keys."""

    kind: Literal["address"]
    key: str


class CallerHeaderInvalid(Refusal):
    """The website vouched for a request with `CALLER_PRINCIPAL_HEADER` but sent no single
    well-formed `CALLER_HEADER`. Counting it by address would put every client of the site on
    one allowance, so it answers 400."""

    code = "request_invalid"

    def __init__(self) -> None:
        super().__init__(
            f"the request carries a valid {CALLER_PRINCIPAL_HEADER} but no single "
            f"{CALLER_HEADER} of the form ip:<32 lowercase hex digits>"
        )


class CallerUnresolved(Refusal):
    """This service cannot name a caller: a presented `x-dfo-caller-principal` does not match
    `DFO_CALLER_PRINCIPAL`, `DFO_CALLER_HMAC_KEY` is unset, or the X-Forwarded-For entry
    `DFO_TRUSTED_PROXY_HOPS` names is missing or not an address. Each is a deployment fault, so
    it answers a 503 a retry does not clear, and is logged."""

    code = "service_unavailable"

    def __init__(self, reason: str) -> None:
        super().__init__(f"this service cannot name the caller a write counts against: {reason}")


def fold_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    """The text a caller digest hashes for `address`, the website's `limiterBucket`.

    IPv4 is its dotted form. An IPv4-mapped IPv6 address is its IPv4 form. Any other IPv6
    address is its /64, one subscriber's usual block: the first four groups in lowercase hex
    without leading zeros, then `::/64`, so `2001:db8::1` folds to `2001:db8:0:0::/64`.
    """
    if isinstance(address, ipaddress.IPv4Address):
        return str(address)
    if address.ipv4_mapped is not None:
        return str(address.ipv4_mapped)
    # The integer drops any zone id, which `exploded` cannot parse back on Python 3.12.
    value = int(address)
    return ":".join(f"{(value >> shift) & 0xFFFF:x}" for shift in (112, 96, 80, 64)) + "::/64"


def address_key(address: str, *, hmac_key: str) -> str:
    """The `kind="address"` caller key for `address`, which must parse as an IP address."""
    folded = fold_address(ipaddress.ip_address(address))
    digest = hmac.new(hmac_key.encode(), folded.encode(), hashlib.sha256).hexdigest()
    return ADDRESS_KEY_PREFIX + digest[:_DIGEST_LENGTH]


def _single(request: Request, header: str) -> str | None:
    values = request.headers.getlist(header)
    return values[0] if len(values) == 1 else None


def _vouched(request: Request, settings: CallerSettings) -> bool:
    """Whether `request` carries the configured caller principal once, compared in constant
    time."""
    secret = settings.caller_principal.get_secret_value()
    presented = _single(request, CALLER_PRINCIPAL_HEADER)
    return (
        bool(secret)
        and presented is not None
        and hmac.compare_digest(presented.encode(), secret.encode())
    )


def _principal_mismatch(settings: CallerSettings) -> CallerUnresolved:
    """Log a presented caller principal this service does not accept, naming neither value, and
    return its refusal."""
    configured = bool(settings.caller_principal.get_secret_value())
    reason = (
        f"{CALLER_PRINCIPAL_HEADER} does not match DFO_CALLER_PRINCIPAL"
        if configured
        else f"{CALLER_PRINCIPAL_HEADER} is present and DFO_CALLER_PRINCIPAL is unset"
    )
    _caller_log.error("caller principal refused: %s", reason)
    return CallerUnresolved(reason)


def resolve_caller(request: Request, settings: CallerSettings) -> Caller:
    """Name the caller of `request` for an allowance. See the module docstring.

    Raises:
        CallerHeaderInvalid: The caller principal matches, and `x-dfo-caller` is missing,
            repeated or malformed.
        CallerUnresolved: `x-dfo-caller-principal` is present and does not match a configured
            `DFO_CALLER_PRINCIPAL`, or the request is named by address, and
            `DFO_CALLER_HMAC_KEY` is unset or the trusted X-Forwarded-For entry is missing or not
            an IP address.
    """
    if request.headers.getlist(CALLER_PRINCIPAL_HEADER):
        if not _vouched(request, settings):
            raise _principal_mismatch(settings)
        named = _single(request, CALLER_HEADER)
        if named is None or not _ADDRESS_KEY.fullmatch(named):
            raise CallerHeaderInvalid()
        return Caller(kind="address", key=named)

    hmac_key = settings.caller_hmac_key.get_secret_value()
    if not hmac_key:
        raise CallerUnresolved("DFO_CALLER_HMAC_KEY is unset")
    # Each proxy appends the address it saw, to the one header or as a further header line.
    entries = [
        entry.strip()
        for value in request.headers.getlist(FORWARDED_FOR_HEADER)
        for entry in value.split(",")
    ]
    position = len(entries) - 1 - settings.trusted_proxy_hops
    address = entries[position] if position >= 0 else ""
    try:
        return Caller(kind="address", key=address_key(address, hmac_key=hmac_key))
    except ValueError:
        raise CallerUnresolved(
            f"X-Forwarded-For holds {len(entries)} entries, and the one "
            f"DFO_TRUSTED_PROXY_HOPS={settings.trusted_proxy_hops} places from the right is "
            "missing or not an IP address"
        ) from None
