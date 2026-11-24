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

`resolve_caller` names the caller an allowance counts against, as a `Caller`. An address
caller (`kind="address"`) has a key of `ip:` and the first 32 hex digits of HMAC-SHA-256 over
the client's address, folded (`fold_address`), under `DFO_CALLER_HMAC_KEY`. The website
computes the same key for the callers it forwards (its `src/lib/proxy/caller.ts`), so a client
has one key on either path and no raw address is stored. An account caller (`kind="account"`)
has a key of `user:` and 32 lowercase hex digits, a digest of the account's id that the website
computes for a request made with an API key, so its allowances count per account and not per
address. This service never derives an account key, only accepts the website's. A request whose
`x-dfo-caller-principal` matches `DFO_CALLER_PRINCIPAL` comes from an upstream that holds the
secret, such as the website or the Studio harness, which names its client in `x-dfo-caller`, as
`ip:<digest>`, `user:<digest>` or `conversation:<digest>`. A request that carries
`x-dfo-caller-principal` without matching it, or while `DFO_CALLER_PRINCIPAL` is unset, refuses
`CallerUnresolved` and logs the mismatch at ERROR, since counting it by
address would put every client of the website on the website's own allowance. A request with no
`x-dfo-caller-principal` is named from the X-Forwarded-For entry `DFO_TRUSTED_PROXY_HOPS` places
from the right, the entry the proxies in front of this service appended. `request.client` goes
unused, because `ProxyHeadersMiddleware` sets it from the leftmost entry, which the client writes.
A vouched `x-dfo-caller` of the form `conversation:<32 hex digits>`, which the Studio harness sends,
names a `Caller` of `kind="conversation"` keyed on that value, so each conversation holds its own
allowances and the vouched path never reads the request's address.

`delivery_identity` names the consumer of a request for its delivery log line
(`datasets_api.delivery`), and never raises. `dfo_platform.app` resolves it once per request and
binds it for `current_delivery_identity`. A vouched request's caller is its `x-dfo-caller` (`ip:`,
`user:` or `conversation:`), or none when that header is missing, repeated or malformed, since the
request's own address is the upstream's. An unvouched request's caller is its own address key,
else none. The surface is the `x-dfo-surface` header beside a valid principal, checked against
`DeliverySurface`, except that a `conversation:` caller is always `studio`. A request no
principal vouches for is `direct`, whatever it claims. The client is the `<name>/<version>`
product token of `X-DFO-Client`. The user agent is a vouched request's `x-dfo-user-agent`, the
end client's User-Agent an upstream forwards, else the request's own User-Agent, and none for a
`conversation:` caller. The MCP client is an `ip:` or `user:` caller's vouched
`x-dfo-mcp-client`, the name the end client gave in its MCP `initialize` request
(`Anthropic/ClaudeAI`, `openai-mcp`, `claude-code`), which data-mcp and analysis-mcp forward for
public callers. It is kept when the stripped single value fully matches `_MCP_CLIENT_NAME`, is
`invalid` when the header is present but does not match, and is none when the header is absent or
blank, on an unvouched request, and for any caller that is neither an `ip:` nor a `user:` caller:
a `conversation:` caller, or none. The turn id is the `x-dfo-turn` header (`tr_` and 16
lowercase hex digits) the Studio harness sends for the turn that caused the request, kept only on
a vouched `conversation:` caller's request when the single value fully matches, else none.
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
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
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
"""The caller an upstream names for a request it forwards: the address key the website resolved,
as `ip:<32 hex digits>`, the account the website resolved for a request made with an API key, as
`user:<32 hex digits>`, or the Studio harness's conversation, as `conversation:<32 hex digits>`."""
CALLER_PRINCIPAL_HEADER = "x-dfo-caller-principal"
"""The secret that vouches for `CALLER_HEADER`, `DFO_CALLER_PRINCIPAL`."""
FORWARDED_FOR_HEADER = "x-forwarded-for"

ADDRESS_KEY_PREFIX = "ip:"
"""The prefix of every `kind="address"` caller key. Reference rows keyed by a caller start with
it, which tells them from the `secret-sha256:` labels of principals."""
USER_KEY_PREFIX = "user:"
"""The prefix of every `kind="account"` caller key, which the website names for a request made
with an API key. Reference rows keyed by an account start with it."""
CONVERSATION_KEY_PREFIX = "conversation:"
"""The prefix of the caller the Studio harness names for one conversation."""

type CallerKind = Literal["address", "account", "conversation"]
"""The kinds of `Caller`."""
CALLER_KEY_PREFIXES: Mapping[CallerKind, str] = MappingProxyType(
    {
        "address": ADDRESS_KEY_PREFIX,
        "account": USER_KEY_PREFIX,
        "conversation": CONVERSATION_KEY_PREFIX,
    }
)
"""The key prefix of each `CallerKind`. A caller key is its prefix and `CALLER_KEY_DIGEST_LENGTH`
lowercase hex digits, and a pin reference an open caller makes is keyed by it, so
`datasets_api.maintenance` builds the principals `settle` leaves alone from this mapping."""
CALLER_KEY_DIGEST_LENGTH = 32
"""The number of lowercase hex digits after a caller key's prefix."""

_DIGEST_LENGTH = CALLER_KEY_DIGEST_LENGTH
_ADDRESS_KEY = re.compile(rf"{ADDRESS_KEY_PREFIX}[0-9a-f]{{{_DIGEST_LENGTH}}}")
_USER_KEY = re.compile(rf"{USER_KEY_PREFIX}[0-9a-f]{{{_DIGEST_LENGTH}}}")
_CONVERSATION_KEY = re.compile(rf"{CONVERSATION_KEY_PREFIX}[0-9a-f]{{{_DIGEST_LENGTH}}}")
TURN_HEADER = "x-dfo-turn"
"""The Studio turn that caused a request, as `tr_<16 hex digits>`."""
_TURN_ID = re.compile(r"tr_[0-9a-f]{16}")


@dataclass(frozen=True, slots=True)
class Caller:
    """Who an allowance counts against. `kind="address"` keys a client by its network address,
    `kind="account"` keys an account the website named for a request made with an API key, and
    `kind="conversation"` keys a Studio conversation by the vouched `conversation:` value."""

    kind: CallerKind
    key: str


class CallerHeaderInvalid(Refusal):
    """An upstream vouched for a request with `CALLER_PRINCIPAL_HEADER` but sent no single
    well-formed `CALLER_HEADER`. Counting it by address would put every client of the site on
    one allowance, so it answers 400."""

    code = "request_invalid"

    def __init__(self) -> None:
        super().__init__(
            f"the request carries a valid {CALLER_PRINCIPAL_HEADER} but no single "
            f"{CALLER_HEADER} of the form ip:<32 lowercase hex digits>, "
            "user:<32 lowercase hex digits> or conversation:<32 lowercase hex digits>"
        )


class CallerUnresolved(Refusal):
    """This service cannot name a caller, because a presented `x-dfo-caller-principal` does not
    match `DFO_CALLER_PRINCIPAL`, `DFO_CALLER_HMAC_KEY` is unset, or the X-Forwarded-For entry
    `DFO_TRUSTED_PROXY_HOPS` names is missing or not an address. Each is a deployment fault a
    retry does not clear, so it answers non-retryable `internal_error` (500) on every door, and
    is logged. `service_unavailable` would tell the client a retry is expected to succeed."""

    code = "internal_error"

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
        if named is not None and _ADDRESS_KEY.fullmatch(named):
            return Caller(kind="address", key=named)
        if named is not None and _USER_KEY.fullmatch(named):
            return Caller(kind="account", key=named)
        if named is not None and _CONVERSATION_KEY.fullmatch(named):
            return Caller(kind="conversation", key=named)
        raise CallerHeaderInvalid()
    return Caller(kind="address", key=_own_address_key(request, settings))


def _own_address_key(request: Request, settings: CallerSettings) -> str:
    """The address key of the X-Forwarded-For entry `DFO_TRUSTED_PROXY_HOPS` places from the
    right.

    Raises:
        CallerUnresolved: `DFO_CALLER_HMAC_KEY` is unset, or that entry is missing or not an IP
            address.
    """
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
        return address_key(address, hmac_key=hmac_key)
    except ValueError:
        raise CallerUnresolved(
            f"X-Forwarded-For holds {len(entries)} entries, and the one "
            f"DFO_TRUSTED_PROXY_HOPS={settings.trusted_proxy_hops} places from the right is "
            "missing or not an IP address"
        ) from None


SURFACE_HEADER = "x-dfo-surface"
"""The surface an upstream names for a request it vouches for, one of `DeliverySurface`."""
USER_AGENT_HEADER = "x-dfo-user-agent"
"""The end client's User-Agent an upstream forwards for a request it vouches for."""
MCP_CLIENT_HEADER = "x-dfo-mcp-client"
"""The MCP `initialize` client name an upstream forwards for a request it vouches for."""
_MCP_CLIENT_NAME = re.compile(r"[ !#-\[\]-~]{1,128}")
"""The MCP client names a delivery line keeps: 1 to 128 printable ASCII characters, with no
quote or backslash. The same rule as dfo-ai's `common.caller.MCP_CLIENT_NAME`."""
MCP_CLIENT_INVALID = "invalid"
"""What a delivery line names for an `x-dfo-mcp-client` that fails `_MCP_CLIENT_NAME`."""
MINT_ADMISSION_HEADER = "X-DFO-Mint-Admission"
"""`background` on a mint that warms data no caller waits on yet (`datasets_api.background_admission`)."""
MINT_ADMISSION_BACKGROUND = "background"


class DeliverySurface(StrEnum):
    """Where a delivered request came from. `direct` is a request no principal vouches for, and
    `unknown` a vouched one that names no valid surface."""

    website = "website"
    api = "api"
    studio = "studio"
    mcp = "mcp"
    direct = "direct"
    unknown = "unknown"


_NAMEABLE_SURFACES = frozenset(
    {DeliverySurface.website, DeliverySurface.api, DeliverySurface.studio, DeliverySurface.mcp}
)
"""The surfaces an upstream may name in `SURFACE_HEADER`."""

type CallerSource = Literal["vouched", "address", "none"]
type Admission = Literal["foreground", "background"]


@dataclass(frozen=True, slots=True)
class DeliveryIdentity:
    """Who a request delivers data to, as its delivery log line names them.

    `caller` is the vouched `x-dfo-caller` (`caller_source="vouched"`), else, on an unvouched
    request, this service's own address key for it (`"address"`), else `None` (`"none"`).
    `upstream` is the unverified `email` claim of the request's ID token. `client` is the
    `<name>/<version>` product token of the `X-DFO-Client` value, `None` when the header is
    absent or has another shape. `user_agent` is the end client's User-Agent, cleaned and
    clipped (see the module docstring). `admission` is `background` for a request sent with
    `X-DFO-Mint-Admission: background`, else `foreground`. `turn_id` is the `x-dfo-turn` value
    of a vouched `conversation:` caller's request when it fully matches `tr_<16 hex digits>`,
    else `None`. `mcp_client` is an `ip:` or `user:` caller's vouched `x-dfo-mcp-client`, the end
    client's MCP `initialize` name, when the stripped single value fully matches
    `_MCP_CLIENT_NAME`; `MCP_CLIENT_INVALID` when the header is present and empty of a usable value
    (it fails the pattern or repeats); `None` when the header is absent or blank, or the caller is
    neither an `ip:` nor a `user:` caller.
    """

    caller: str | None
    caller_source: CallerSource
    upstream: str | None
    surface: DeliverySurface
    client: str | None
    user_agent: str | None
    admission: Admission
    turn_id: str | None
    mcp_client: str | None


UNRESOLVED_IDENTITY = DeliveryIdentity(
    caller=None,
    caller_source="none",
    upstream=None,
    surface=DeliverySurface.unknown,
    client=None,
    user_agent=None,
    admission="foreground",
    turn_id=None,
    mcp_client=None,
)
"""The identity of a delivery made outside a request `dfo_platform.app` resolved."""


def _named_surface(request: Request) -> DeliverySurface:
    named = (_single(request, SURFACE_HEADER) or "").strip().lower()
    return DeliverySurface(named) if named in _NAMEABLE_SURFACES else DeliverySurface.unknown


def _client_token(request: Request) -> str | None:
    """The `<name>/<version>` product token of `X-DFO-Client`, its runtime comment dropped."""
    match = _CLIENT_FORMAT.fullmatch(_clean(request.headers.get(CLIENT_HEADER, "")))
    return f"{match['name']}/{match['version']}" if match else None


def _mcp_client_of(request: Request) -> str | None:
    """The `x-dfo-mcp-client` name of a vouched `ip:` or `user:` caller's request: the stripped
    value when it matches `_MCP_CLIENT_NAME`, `None` for no header or a blank one, else
    `MCP_CLIENT_INVALID`."""
    values = request.headers.getlist(MCP_CLIENT_HEADER)
    if not values:
        return None
    name = values[0].strip() if len(values) == 1 else None
    if name == "":
        return None
    return name if name is not None and _MCP_CLIENT_NAME.fullmatch(name) else MCP_CLIENT_INVALID


def _caller_of(
    request: Request, settings: CallerSettings, *, vouched: bool
) -> tuple[str | None, CallerSource]:
    """The caller and its source for `delivery_identity`. A vouched request's own address is
    the upstream's, so one without a valid `x-dfo-caller` names no caller."""
    if vouched:
        named = _single(request, CALLER_HEADER)
        if named is not None and (
            _ADDRESS_KEY.fullmatch(named)
            or _USER_KEY.fullmatch(named)
            or _CONVERSATION_KEY.fullmatch(named)
        ):
            return named, "vouched"
        return None, "none"
    try:
        return _own_address_key(request, settings), "address"
    except CallerUnresolved:
        return None, "none"


def delivery_identity(request: Request, settings: CallerSettings) -> DeliveryIdentity:
    """Name the consumer of `request` for its delivery line. See the module docstring.

    It never raises. A principal `resolve_caller` refuses and logs reads here, unlogged, as no
    principal. The request is `direct` and named by address.
    """
    try:
        admission: Admission = (
            "background"
            if request.headers.get(MINT_ADMISSION_HEADER, "").strip().lower()
            == MINT_ADMISSION_BACKGROUND
            else "foreground"
        )
        vouched = _vouched(request, settings)
        caller, source = _caller_of(request, settings, vouched=vouched)
        conversation = caller is not None and caller.startswith(CONVERSATION_KEY_PREFIX)
        if not vouched:
            surface = DeliverySurface.direct
        elif conversation:
            surface = DeliverySurface.studio
        else:
            surface = _named_surface(request)
        turn = _single(request, TURN_HEADER)
        # `conversation` is true only for a vouched caller, so it carries the vouch.
        turn_id = turn if conversation and turn and _TURN_ID.fullmatch(turn) else None
        forwarded = _clean(request.headers.get(USER_AGENT_HEADER, "")) if vouched else ""
        own = _clean(request.headers.get("user-agent", ""))
        mcp_client = (
            _mcp_client_of(request)
            if vouched
            and caller is not None
            and (_ADDRESS_KEY.fullmatch(caller) or _USER_KEY.fullmatch(caller))
            else None
        )
        return DeliveryIdentity(
            caller=caller,
            caller_source=source,
            upstream=caller_identity_from_headers(request.headers).email,
            surface=surface,
            client=_client_token(request),
            user_agent=None if conversation else (forwarded or own or None),
            admission=admission,
            turn_id=turn_id,
            mcp_client=mcp_client,
        )
    except Exception as exc:
        # A log line's identity must not fail the request it describes.
        _caller_log.warning("delivery identity unresolved: %s", type(exc).__name__)
        return UNRESOLVED_IDENTITY


_delivery_identity: ContextVar[DeliveryIdentity] = ContextVar(
    "dfo_delivery_identity", default=UNRESOLVED_IDENTITY
)


def set_delivery_identity(identity: DeliveryIdentity) -> Token[DeliveryIdentity]:
    """Bind `identity` for the current request. Reset it with `reset_delivery_identity`."""
    return _delivery_identity.set(identity)


def reset_delivery_identity(token: Token[DeliveryIdentity]) -> None:
    _delivery_identity.reset(token)


def current_delivery_identity() -> DeliveryIdentity:
    """The identity `dfo_platform.app` bound for the request in flight, else
    `UNRESOLVED_IDENTITY`."""
    return _delivery_identity.get()
