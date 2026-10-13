"""The base model and field types every stored description type is built from.

Description records are stored as Postgres jsonb, so each type here is safe to store:

- A model refuses members it does not define.
- Free text loses every U+0000, which jsonb refuses. A failed record insert rolls back the
  version row written in the same transaction.
- A count is a non-negative JSON integer. A string, a float or a boolean is refused rather than
  coerced, so a stored count reads back as the same bytes.
- A mapping's keys are sorted on validation. jsonb reorders object keys, so a record read back
  from the store dumps to the same bytes it was written as.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    Strict,
    StringConstraints,
)

NAME_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
SPEC_ID_PATTERN = r"^spec_[0-9a-f]{32}$"
OPS_ID_PATTERN = r"^ops_[0-9a-f]{6,64}$"
VERSION_ID_PATTERN = r"^(spec_[0-9a-f]{32}|ops_[0-9a-f]{6,64})$"
CONTENT_VERSION_PATTERN = r"^(cv_[0-9a-f]{32}|ocv_[0-9a-f]{6,64})$"
TEXT_MAX_LENGTH = 512
"""The most characters `Text` holds."""
LONG_TEXT_MAX_LENGTH = 8192
"""The most characters `LongText` holds."""
DEFINITION_MAX_LENGTH = 4096
"""The most characters a served definition holds."""


def clipped(text: str, *, limit: int = TEXT_MAX_LENGTH) -> str:
    """`text`, cut to `limit` characters ending in "…" when longer."""
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def strip_nul(value: object) -> object:
    """Returns a string without its U+0000 characters; anything else unchanged."""
    return value.replace("\x00", "") if isinstance(value, str) else value


def canonical_json(value: JsonValue) -> JsonValue:
    """Returns a JSON value with U+0000 stripped from every string and key, and every
    object's keys sorted, at any depth."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        stripped = {key.replace("\x00", ""): item for key, item in value.items()}
        return {key: canonical_json(stripped[key]) for key in sorted(stripped)}
    if isinstance(value, list):
        return [canonical_json(item) for item in value]
    return value


def sorted_keys[V](value: dict[str, V]) -> dict[str, V]:
    """Returns a mapping with its keys in sorted order."""
    return {key: value[key] for key in sorted(value)}


def same_version_family(*, id: str, content_version: str) -> bool:
    """True when a version id and its content version belong together: a `spec_` id with a
    `cv_` version, or an `ops_` id with an `ocv_` version."""
    if id.startswith("spec_"):
        return content_version.startswith("cv_")
    return content_version.startswith("ocv_")


Text = Annotated[
    str, BeforeValidator(strip_nul), StringConstraints(min_length=1, max_length=TEXT_MAX_LENGTH)
]
"""Free text up to 512 characters: a label, a name, a dcid."""

LongText = Annotated[
    str,
    BeforeValidator(strip_nul),
    StringConstraints(min_length=1, max_length=LONG_TEXT_MAX_LENGTH),
]
"""Free text up to 8,192 characters, such as a pipeline step's CommonMark description."""

Period = Annotated[str, BeforeValidator(strip_nul), StringConstraints(min_length=1, max_length=32)]
"""A period as the frame writes it: `2024`, `2024-03`, `2024-03-31`."""

PartialDate = Annotated[str, StringConstraints(pattern=r"^[0-9]{4}(-[0-9]{2}(-[0-9]{2})?)?$")]
"""A year, a year and month, or a date."""

Name = Annotated[str, StringConstraints(pattern=NAME_PATTERN)]
"""A lower-snake-case name: an operation, a policy, a role, a reason or a count."""

FieldPath = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-z][a-z0-9_]*(\.([a-z][a-z0-9_]*|0|[1-9][0-9]*))*$", max_length=200
    ),
]
"""A dotted path to a field of a stored record, such as `steps`, `series.status` or
`computed.inputs.0.id`, where a number names a list item by its position."""

VersionId = Annotated[str, StringConstraints(pattern=VERSION_ID_PATTERN)]
"""A `spec_` or `ops_` id."""

ContentVersion = Annotated[str, StringConstraints(pattern=CONTENT_VERSION_PATTERN)]
"""A `cv_` or `ocv_` content version."""

Count = Annotated[int, Strict(), Field(ge=0)]
"""A non-negative integer. A count nothing recorded is `None` where its field allows it, and
never 0."""

Canonical = Annotated[JsonValue, AfterValidator(canonical_json)]
"""Any JSON value, with U+0000 stripped and object keys sorted at every depth."""


class Stored(BaseModel):
    """A stored description type: refuses unknown members and is immutable once built."""

    model_config = ConfigDict(extra="forbid", frozen=True)
