"""The opaque cursor of `GET /api/graph/node/{dcid}/members`.

A cursor holds the last item's sort key and dcid, and a digest of the walk it belongs to: the
node, the relation, the projection, the `as_of` date and the order, and for a relation that
filters its items, the filter values. The next page starts after
that key, so a page read after a rebuild repeats no item and skips none present in both
builds; nothing in the cursor names a build. A cursor sent with another binding, or one that
does not decode, raises `CursorInvalid`, which the members door refuses 400 `request_invalid`
at `/cursor`.

The wire form is unpadded base64url over compact JSON, `{"b": digest, "k": key, "d": dcid}`.
The digest is not a signature: a client that forges one reaches a position in a list it may
already read."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from datetime import date

_DIGEST_CHARS = 16


class CursorInvalid(ValueError):
    """The cursor does not decode, or belongs to another walk."""


@dataclass(frozen=True, slots=True)
class Binding:
    """What a cursor is bound to. `projection` and `as_of` are `None` where the relation takes
    neither; `order` names the sort key (`code`, `dcid` or `facet_id`). `filters` holds each
    named filter and its value, a `None` value being a filter for no value, and is `None` where
    the relation takes no filters, which leaves the digest as it was without them."""

    node: str
    relation: str
    projection: str | None
    as_of: date | None
    order: str
    filters: tuple[tuple[str, str | None], ...] | None = None

    def digest(self) -> str:
        fields = [
            self.node,
            self.relation,
            self.projection,
            self.as_of.isoformat() if self.as_of is not None else None,
            self.order,
        ]
        if self.filters is not None:
            fields.append([list(pair) for pair in self.filters])
        encoded = json.dumps(fields, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:_DIGEST_CHARS]


@dataclass(frozen=True, slots=True)
class Position:
    """The last item a page returned: its sort key and its dcid, which breaks ties."""

    key: str
    dcid: str


def encode(binding: Binding, position: Position) -> str:
    """The cursor that continues `binding`'s walk after `position`."""
    payload = {"b": binding.digest(), "k": position.key, "d": position.dcid}
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode(cursor: str, binding: Binding) -> Position:
    """The position `cursor` continues from.

    Raises:
        CursorInvalid: `cursor` does not decode, or was issued for another binding.
    """
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        payload = json.loads(raw)
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise CursorInvalid("the cursor does not decode") from None
    if not isinstance(payload, dict):
        raise CursorInvalid("the cursor does not decode")
    digest, key, dcid = payload.get("b"), payload.get("k"), payload.get("d")
    if not (isinstance(digest, str) and isinstance(key, str) and isinstance(dcid, str)):
        raise CursorInvalid("the cursor does not decode")
    if digest != binding.digest():
        raise CursorInvalid(
            "the cursor belongs to another node, relation, projection, as_of date, order or filter"
        )
    return Position(key=key, dcid=dcid)
