"""The registry of admitted upstream sources, shipped as package data.

`data/registry.toml` holds one `[[base]]` entry per base Data Commons
`Source` node and one `[[un_system]]` entry per UN agency. On the UN deployment every agency sits
under the single source `undata/s/UN`, so a UN entry names the agency
provenance, which serves as its source, and the id segment its variables
carry (`undata/<segment>/...`, `undata/topic/<segment>/...`). DFO needs no
entries: every provenance with rows in our Spanner is admitted.

A base entry may exclude named provenances, each with a reason. An entry
whose `exclusions` key is absent has not been reviewed for overlap with DFO,
and an empty list states that it has. No excluded provenance is admitted. A
provenance `superseded_by_dfo` is an upstream copy of something DFO imports.
A provenance `superseded_by_un_system` names its `successor`, the registered
UN agency whose series replace it, and how one of its variable ids names
the successor's: after `variable_prefix` a variable id holds its series
code, then after `code_delimiter` its constraint pairs, separated by
`pair_delimiter`, each a dimension and a value separated by
`value_delimiter`. The successor agency's variable ids follow the same form
after the agency's own prefix.

A UN entry may exclude series, each `superseded_by_dfo` and naming the DFO
provenance that holds the data the series derives from.

A national source names its country. An
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

_UN_RESERVED_SEGMENTS: Final[frozenset[str]] = frozenset(
    {"p", "g", "s", "topic", "sdgf", "theme", "abas"}
)
"""Path segments the UN deployment uses for provenances, groups, sources and
topics, and for the topic families that group indicators across agencies
(`undata/topic/sdgf/...`). No agency segment may shadow one."""

type ExclusionReason = Literal["superseded_by_dfo", "superseded_by_un_system"]

_SourceDcid = Annotated[str, Field(pattern=r"^dc/s/[A-Za-z0-9_.-]+$")]
_CountryDcid = Annotated[str, Field(pattern=r"^country/[A-Z]{3}$")]
_PlaceType = Annotated[str, Field(pattern=r"^[A-Z][A-Za-z0-9]*$")]
_UnProvenance = Annotated[str, Field(pattern=r"^undata/p/[A-Z][A-Z0-9_]*$")]
_UnSegment = Annotated[str, Field(pattern=r"^[a-z][a-z0-9]*$")]
_UnSeries = Annotated[
    str,
    Field(pattern=rf"^{UN_ID_PREFIX}[a-z][a-z0-9]*/[A-Za-z0-9_.-]+$"),
]
_DfoProvenance = Annotated[str, Field(pattern=r"^provenance/[A-Za-z0-9_.-]+$")]
_Dcid = Annotated[str, Field(pattern=r"^\S+$")]
_VariablePrefix = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+/$")]
_Delimiter = Annotated[str, Field(pattern=r"^[^A-Za-z0-9/\s]+$")]

type ConstraintKey = tuple[str, frozenset[tuple[str, str]]]
"""A variable id read as its series code and its constraint pairs, order ignored."""
_Label = Annotated[str, Field(min_length=1)]


class RegistryError(ValueError):
    """The registry file is not valid TOML or fails validation."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _BaseExclusion(_Model):
    provenance: _Dcid

    @model_validator(mode="after")
    def _is_base_provenance(self) -> Self:
        if self.provenance.startswith(UN_ID_PREFIX):
            raise ValueError(f"exclusion {self.provenance!r} is a UN provenance, not a base one")
        return self


class DfoSupersededProvenance(_BaseExclusion):
    """A base provenance that is an upstream copy of something DFO imports.
    None of it is admitted."""

    reason: Literal["superseded_by_dfo"]


class UnSupersededProvenance(_BaseExclusion):
    """A base provenance a UN agency's series replace. None of it is
    admitted. `successor` is the agency's provenance. A variable of the
    provenance has an id `<variable_prefix><code>`, or
    `<variable_prefix><code><code_delimiter><pairs>`, where `<pairs>` joins
    `<dimension><value_delimiter><value>` pairs with `pair_delimiter`. The
    successor agency's variable ids take the same form after its own
    prefix."""

    reason: Literal["superseded_by_un_system"]
    successor: _UnProvenance
    variable_prefix: _VariablePrefix
    code_delimiter: _Delimiter
    pair_delimiter: _Delimiter
    value_delimiter: _Delimiter

    @model_validator(mode="after")
    def _delimiters_are_distinct(self) -> Self:
        delimiters = (self.code_delimiter, self.pair_delimiter, self.value_delimiter)
        if any(
            a in b for i, a in enumerate(delimiters) for j, b in enumerate(delimiters) if i != j
        ):
            raise ValueError(
                f"exclusion {self.provenance!r} declares delimiters {delimiters!r}; none may "
                "equal or hold another, or one id would read two ways"
            )
        return self

    def variable_suffix(self, variable: str) -> str | None:
        """What follows `variable_prefix` in `variable`, or `None` for a
        variable that does not start with it or holds nothing after it."""
        if not variable.startswith(self.variable_prefix):
            return None
        return variable[len(self.variable_prefix) :] or None

    def constraint_key(self, suffix: str) -> ConstraintKey:
        """A variable id after its prefix, read as its series code and the
        set of its constraint pairs. A pair holding no `value_delimiter` is
        kept whole as its dimension, with an empty value."""
        code, _, rest = suffix.partition(self.code_delimiter)
        pairs = frozenset(
            (dimension, value)
            for pair in (rest.split(self.pair_delimiter) if rest else ())
            for dimension, _, value in (pair.partition(self.value_delimiter),)
        )
        return code, pairs


type Exclusion = Annotated[
    DfoSupersededProvenance | UnSupersededProvenance, Field(discriminator="reason")
]
"""A provenance a base source holds that the API does not admit, because
another origin supersedes it."""


class SeriesExclusion(_Model):
    """A UN series the API does not admit because it derives from data DFO
    imports. `dfo_provenance` is the DFO provenance that holds that data."""

    series: _UnSeries
    reason: Literal["superseded_by_dfo"]
    dfo_provenance: _DfoProvenance


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
    exclusions: tuple[SeriesExclusion, ...] = ()

    @property
    def id_prefix(self) -> str:
        """The prefix of the agency's variable and series ids: `undata/<segment>/`."""
        return f"{UN_ID_PREFIX}{self.segment}/"

    @model_validator(mode="after")
    def _segment_is_an_agency_segment(self) -> Self:
        if self.segment in _UN_RESERVED_SEGMENTS:
            raise ValueError(
                f"{self.provenance!r} names segment {self.segment!r}, which the UN deployment "
                "reserves for another id family"
            )
        return self

    @model_validator(mode="after")
    def _excludes_its_own_series(self) -> Self:
        own = self.id_prefix
        foreign = sorted(
            exclusion.series
            for exclusion in self.exclusions
            if not exclusion.series.startswith(own)
        )
        if foreign:
            raise ValueError(
                f"{self.provenance!r} excludes {foreign!r}, outside its own series ({own}...)"
            )
        return self


class Registry(_Model):
    """Every admitted upstream source. Duplicate sources, provenances,
    segments and exclusions are refused, and a UN supersession names a
    registered agency."""

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
        _refuse_duplicates(
            "superseded variable prefix",
            [
                exclusion.variable_prefix
                for entry in self.base
                for exclusion in entry.exclusions or ()
                if isinstance(exclusion, UnSupersededProvenance)
            ],
        )
        _refuse_duplicates(
            "excluded series",
            [exclusion.series for agency in self.un_system for exclusion in agency.exclusions],
        )
        return self

    @model_validator(mode="after")
    def _successors_are_registered(self) -> Self:
        registered = {agency.provenance for agency in self.un_system}
        for entry in self.base:
            for exclusion in entry.exclusions or ():
                if (
                    isinstance(exclusion, UnSupersededProvenance)
                    and exclusion.successor not in registered
                ):
                    raise ValueError(
                        f"exclusion {exclusion.provenance!r} names successor "
                        f"{exclusion.successor!r}, which is not a registered UN provenance"
                    )
        return self

    def un_supersessions(self) -> tuple[tuple[UnSupersededProvenance, UnAgency], ...]:
        """Each base provenance a UN agency supersedes, with that agency."""
        agencies = {agency.provenance: agency for agency in self.un_system}
        return tuple(
            (exclusion, agencies[exclusion.successor])
            for entry in self.base
            for exclusion in entry.exclusions or ()
            if isinstance(exclusion, UnSupersededProvenance)
        )

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
    "BaseSource",
    "ConstraintKey",
    "DfoSupersededProvenance",
    "Exclusion",
    "ExclusionReason",
    "Registry",
    "RegistryError",
    "SeriesExclusion",
    "UnAgency",
    "UnSupersededProvenance",
    "load_registry",
    "packaged_registry",
    "parse_registry",
    "registry_hash",
]
