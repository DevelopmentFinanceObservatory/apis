"""The registry of admitted upstream sources, shipped as package data.

`data/registry.toml` holds one `[[base]]` entry per base Data Commons
`Source` node and one `[[un_system]]` entry per UN agency. On the UN deployment every agency sits
under the single source `undata/s/UN`, so a UN entry names the agency
provenance, which serves as its source, and the id segment its variables
carry (`undata/<segment>/...`, `undata/topic/<segment>/...`). DFO needs no
entries: every provenance with rows in our Spanner is admitted.

A base entry may exclude named provenances, each with a reason. An entry
whose `exclusions` key is absent has not been reviewed for overlap with DFO,
and an empty list states that it has. A national source names its country. An
entry may name a `scan_type`, the place type where its series sit, inside its
country when it names one (the first subnational level, or deeper where the
data is), anywhere otherwise (Eurostat's NUTS 2 regions). The catalogue job
scans those places as well as every country, Earth and the regional
aggregates.

The runtime image ships only the wheel, so the file loads through
`importlib.resources`.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import tomllib
from collections import Counter
from importlib.resources.abc import Traversable
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from origin_catalogue.origins import UN_ID_PREFIX
from origin_catalogue.schema import ScanLevel

_PACKAGE = __package__
_DATA_DIR = "data"
_REGISTRY_FILENAME = "registry.toml"

UN_CROSS_AGENCY_TOPIC_FAMILIES: Final[frozenset[str]] = frozenset({"sdgf", "theme", "abas"})
"""UN topic families that group indicators across agencies
(`undata/topic/sdgf/...`). They carry no agency segment and are admitted
whenever the UN origin is."""

_UN_RESERVED_SEGMENTS: Final[frozenset[str]] = frozenset({"p", "g", "s", "topic"})
"""Second path segments the UN deployment uses for provenances, groups,
sources and topics, which no agency segment may shadow."""

type ExclusionReason = Literal["superseded_by_dfo"]

_SourceDcid = Annotated[str, Field(pattern=r"^dc/s/[A-Za-z0-9_.-]+$")]
_CountryDcid = Annotated[str, Field(pattern=r"^country/[A-Z]{3}$")]
_PlaceType = Annotated[str, Field(pattern=r"^[A-Z][A-Za-z0-9]*$")]
_UnProvenance = Annotated[str, Field(pattern=r"^undata/p/[A-Z][A-Z0-9_]*$")]
_UnSegment = Annotated[str, Field(pattern=r"^[a-z][a-z0-9]*$")]
_Dcid = Annotated[str, Field(pattern=r"^\S+$")]
_Label = Annotated[str, Field(min_length=1)]


class RegistryError(ValueError):
    """The registry file is not valid TOML or fails validation."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Exclusion(_Model):
    """A provenance a base source holds that the API does not admit.
    `superseded_by_dfo` covers an upstream copy of something DFO imports."""

    provenance: _Dcid
    reason: ExclusionReason

    @model_validator(mode="after")
    def _is_base_provenance(self) -> Self:
        if self.provenance.startswith(UN_ID_PREFIX):
            raise ValueError(f"exclusion {self.provenance!r} is a UN provenance, not a base one")
        return self


class BaseSource(_Model):
    """A base Data Commons `Source` node whose provenances are admitted."""

    source: _SourceDcid
    label: _Label
    exclusions: tuple[Exclusion, ...] | None = None
    country: _CountryDcid | None = None
    scan_type: _PlaceType | None = None

    @property
    def scan_level(self) -> ScanLevel | None:
        """The level the catalogue job scans for this source beyond the
        global levels, or `None` when it scans only those."""
        if self.scan_type is None:
            return None
        return ScanLevel(place_type=self.scan_type, within=self.country)


class UnAgency(_Model):
    """A UN agency on the UN System Data Commons: its provenance and id segment."""

    provenance: _UnProvenance
    segment: _UnSegment
    label: _Label

    @model_validator(mode="after")
    def _segment_is_an_agency_segment(self) -> Self:
        if self.segment in _UN_RESERVED_SEGMENTS | UN_CROSS_AGENCY_TOPIC_FAMILIES:
            raise ValueError(
                f"{self.provenance!r} names segment {self.segment!r}, which the UN deployment "
                "reserves for another id family"
            )
        return self


class Registry(_Model):
    """Every admitted upstream source. Duplicate sources, provenances,
    segments and exclusions are refused."""

    base: tuple[BaseSource, ...]
    un_system: tuple[UnAgency, ...]

    @model_validator(mode="after")
    def _no_duplicates(self) -> Self:
        _refuse_duplicates("base source", [entry.source for entry in self.base])
        _refuse_duplicates("UN provenance", [agency.provenance for agency in self.un_system])
        _refuse_duplicates("UN segment", [agency.segment for agency in self.un_system])
        _refuse_duplicates(
            "excluded provenance",
            [exclusion.provenance for entry in self.base for exclusion in entry.exclusions or ()],
        )
        return self

    def scan_levels(self) -> dict[str, tuple[ScanLevel, ...]]:
        """Each base source's scan level, keyed by source, for sources that
        name one. A catalogue build records these as its `scan_levels`."""
        return {
            entry.source: (level,) for entry in self.base if (level := entry.scan_level) is not None
        }


def _refuse_duplicates(kind: str, values: list[str]) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {kind}(s): {duplicates!r}")


def packaged_registry() -> Traversable:
    """The packaged `data/registry.toml`, readable from an installed wheel."""
    return importlib.resources.files(_PACKAGE) / _DATA_DIR / _REGISTRY_FILENAME


def parse_registry(raw: bytes) -> Registry:
    """Parses and validates registry TOML. Raises `RegistryError`."""
    try:
        payload = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise RegistryError(f"registry is not valid UTF-8 TOML: {exc}") from exc
    try:
        return Registry.model_validate(payload)
    except ValidationError as exc:
        raise RegistryError(f"registry failed validation: {exc}") from exc


def registry_hash(raw: bytes) -> str:
    """SHA-256 of the registry file's bytes, part of a catalogue build's identity."""
    return hashlib.sha256(raw).hexdigest()


def load_registry() -> Registry:
    """Parses the packaged registry. Raises `RegistryError`."""
    return parse_registry(packaged_registry().read_bytes())


__all__ = [
    "UN_CROSS_AGENCY_TOPIC_FAMILIES",
    "BaseSource",
    "Exclusion",
    "ExclusionReason",
    "Registry",
    "RegistryError",
    "UnAgency",
    "load_registry",
    "packaged_registry",
    "parse_registry",
    "registry_hash",
]
