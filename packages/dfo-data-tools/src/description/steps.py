"""Step records: what was done to a result, one step per operation, in order.

The models mirror the `spec-kit` repository's `steps.schema.json` member for member. A step
list the schema accepts validates here, and one it refuses is refused here, with one
exception: a count written as a float with no fraction, such as `3.0`, which the schema reads as an
integer and these models refuse. `op`, `policy` and `left_out[].reason` are bounded names,
so an operation or reason the kit's list does not hold still validates.

A member the schema makes optional is absent until it is set, and a model dumps only the
members it holds. `null` is accepted only where the schema accepts it: a unit that the data
declares none of. The structure rules (unique ids, inputs that name an earlier step, the
count and size bounds) belong to `dfo_descriptions.check_steps`, since they span the list.
`MAX_COMPUTED_STEPS` is the kit's count bound less the steps a card adds, so a card drawn
from the longest computation the publish door accepts stays within the kit's bound.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, Strict, StringConstraints, model_validator
from pydantic.experimental.missing_sentinel import MISSING

from description.stored import (
    Canonical,
    ContentVersion,
    Count,
    Name,
    Period,
    Stored,
    Text,
    VersionId,
    same_version_family,
    sorted_keys,
)

StepId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
"""A step's id, unique within its list."""

CONTINUE_SERIES_OP = "continue_series"
"""The op of a dataset version's step that continues its series past their last period with
the same source's estimates. Its `params.method` names the continuing `measurementMethod`."""

MAX_SERVED_STEPS = 64
"""The `spec-kit` repository's bound on one step list (`dfo_descriptions.MAX_STEPS`), which a
description's served steps meet as a stored list does."""

MAX_CARD_STEPS = 2
"""The most steps a card serves after its version's: its selection and a stat card's change."""

MAX_COMPUTED_STEPS = MAX_SERVED_STEPS - MAX_CARD_STEPS
"""The most steps a computed version holds, so that a card drawn from it serves at most
`MAX_SERVED_STEPS`. A computed version holds no source pipeline steps."""


class StepRef(Stored):
    """A reference: the dcid or id it names, and optionally the words for it.

    `at_mint` marks a label taken from the evidence stored when the version was made.
    """

    ref: Text
    label: Text | MISSING = MISSING
    at_mint: Literal[True] | MISSING = MISSING


class VersionRef(Stored):
    """A Datasets version: a `spec_` id with its `cv_` content version, or an `ops_` id with
    its `ocv_` content version, and optionally the words for it."""

    id: VersionId
    content_version: ContentVersion
    label: Text | MISSING = MISSING
    at_mint: Literal[True] | MISSING = MISSING

    @model_validator(mode="after")
    def _one_family(self) -> Self:
        if not same_version_family(id=self.id, content_version=self.content_version):
            raise ValueError(f"{self.id} cannot carry content version {self.content_version}")
        return self


StepInput = StepId | VersionRef
"""One input: an earlier step's id, or a version."""


class SimpleUnit(Stored):
    """A unit by reference, by currency, or both, with optional words.

    `ref` is the unit's dcid. `currency` is an ISO 4217 code, which a converted unit
    `{currency, prices, base_year}` names in place of a dcid. `prices` says whether money is
    in `current` or `constant` prices, and `base_year` names the year constant prices are
    expressed in; it is present exactly when prices are constant.
    """

    ref: Text | MISSING = MISSING
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")] | MISSING = MISSING
    label: Text | MISSING = MISSING
    at_mint: Literal[True] | MISSING = MISSING
    prices: Literal["current", "constant"] | MISSING = MISSING
    base_year: Annotated[int, Strict(), Field(ge=1900, le=2100)] | MISSING = MISSING

    @model_validator(mode="after")
    def _ref_or_currency(self) -> Self:
        if self.ref is MISSING and self.currency is MISSING:
            raise ValueError("a unit names a ref, a currency, or both")
        return self

    @model_validator(mode="after")
    def _base_year_with_constant_prices(self) -> Self:
        if (self.prices == "constant") != (self.base_year is not MISSING):
            raise ValueError("base_year is present exactly when prices are constant")
        return self


def _bare_dcid(value: str) -> str:
    if value.startswith("dcid:"):
        raise ValueError("a variable is named by its bare dcid, without `dcid:`")
    return value


class VariableSide(Stored):
    """One side of a composed unit that counts a variable in place of a unit, such as the
    population in a per-capita denominator. `stat_var` is the variable's bare dcid."""

    stat_var: Annotated[Text, AfterValidator(_bare_dcid)]
    label: Text | MISSING = MISSING
    at_mint: Literal[True] | MISSING = MISSING


class ComposedUnit(Stored):
    """A quotient's unit: the numerator's unit per the denominator's. Either side may be a
    variable."""

    numerator: SimpleUnit | VariableSide
    denominator: SimpleUnit | VariableSide


Unit = SimpleUnit | ComposedUnit | None
"""A unit; `None` when the data declares none."""


Role = Annotated[str, StringConstraints(pattern=r"^[a-z][A-Za-z0-9]{0,63}$")]
"""A graph role's name, the bare lower-camel-case property a flow names its party by, such
as `financingProvider`."""


class LeftOut(Stored):
    """One data gap: why some data was left out, how many, and of what. The caller's own
    selection is a parameter and never appears here.

    A `pooled_entities` count is of the distinct parties of one pooled role, named in `role`,
    across all the step's inputs. Those parties are never among the entities the step kept.
    Only that `of` takes a `role`, and it requires one.
    """

    reason: Name
    count: Count
    of: Literal["entities", "periods", "entity_periods", "pooled_entities"]
    role: Role | MISSING = MISSING

    @model_validator(mode="after")
    def _role_with_pooled_entities(self) -> Self:
        if (self.of == "pooled_entities") != (self.role is not MISSING):
            raise ValueError("a `pooled_entities` count names its `role`, and no other count does")
        return self


class PeriodCoverage(Stored):
    """How many entities one period's total covers."""

    period: Period
    entities: Count


class EntityCoverage(Stored):
    """How many of the window's periods one ranked entity holds."""

    entity: StepRef
    periods: Count
    of: Count


class Choice(Stored):
    """A choice of source at mint: the source chosen, the others it was chosen over, and the
    ranking it applied, in order."""

    chosen: StepRef
    others: Annotated[list[StepRef], Field(max_length=32)]
    ranking: Annotated[list[Name], Field(min_length=1, max_length=8)]


class Summed(Stored):
    """A sum at mint: how many variables were summed, and up to 64 of them."""

    count: Count
    variables: Annotated[list[StepRef], Field(max_length=64)]


_Choices = Annotated[dict[Name, Canonical], Field(max_length=32), AfterValidator(sorted_keys)]

RoleColumn = Annotated[str, StringConstraints(min_length=1, max_length=64)]
"""A frame's role column, such as `donor` or `observationAbout`."""

_Side = Annotated[dict[RoleColumn, Text], Field(max_length=16), AfterValidator(sorted_keys)]
"""The entity ref, such as `dcid:country/GBR`, each of a side's role columns holds."""


class Step(Stored):
    """One operation: its inputs by role, its one fixed policy, the caller's own choices,
    what it kept, what data gaps it left out, the unit it produces, and, for an operation
    that writes a new variable (sum, quotient, share, comparison), that variable's bare dcid in
    `stat_var`.

    A step takes either `input` or `inputs`, or neither, never both. `stat_var` is the bare
    dcid of the variable a step that makes a new one (a sum, quotient, share or comparison)
    writes its rows under. `sides` holds, by input role, each role column whose slice of that
    input holds one entity, with that entity's ref. A comparison records it.
    """

    id: StepId
    op: Name
    stat_var: Annotated[Text, AfterValidator(_bare_dcid)] | MISSING = MISSING
    input: StepInput | MISSING = MISSING
    inputs: (
        Annotated[
            dict[Name, StepInput],
            Field(min_length=1, max_length=8),
            AfterValidator(sorted_keys),
        ]
        | MISSING
    ) = MISSING
    sides: (
        Annotated[dict[Name, _Side], Field(min_length=1, max_length=8), AfterValidator(sorted_keys)]
        | MISSING
    ) = MISSING
    policy: Name | MISSING = MISSING
    params: _Choices | MISSING = MISSING
    options: _Choices | MISSING = MISSING
    kept: (
        Annotated[dict[Name, Count], Field(max_length=16), AfterValidator(sorted_keys)] | MISSING
    ) = MISSING
    left_out: Annotated[list[LeftOut], Field(max_length=32)] | MISSING = MISSING
    period_coverage: Annotated[list[PeriodCoverage], Field(max_length=500)] | MISSING = MISSING
    entity_coverage: Annotated[list[EntityCoverage], Field(max_length=500)] | MISSING = MISSING
    choice: Choice | MISSING = MISSING
    summed: Summed | MISSING = MISSING
    converted_from: Unit | MISSING = MISSING
    unit: Unit | MISSING = MISSING
    stat_var: Annotated[Text, AfterValidator(_bare_dcid)] | MISSING = MISSING

    @model_validator(mode="after")
    def _input_or_inputs(self) -> Self:
        if self.input is not MISSING and self.inputs is not MISSING:
            raise ValueError("a step takes `input` or `inputs`, not both")
        return self
