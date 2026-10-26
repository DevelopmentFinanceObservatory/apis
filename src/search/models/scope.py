"""How a part was counted: its scope choices, their options and the identity they compile to.

`PartScope` is `ResolvedSpec.scope_choices`. Its `entries` are the explicit and carried
entries a client posts back, beside `base.identity`, on a dataset_resubmit identity entry to
switch an option. The server recompiles from them; the rest of the record is the server's
output and is never posted.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import StrictModel
from .definition import SpecDefinition
from .identity import BaseIdentity
from .vocab import (
    SCOPE_OPTION_ID_PATTERN,
    ChoiceProvenance,
    ComponentRole,
    EntryProvenance,
    OptionStatus,
    ProviderScope,
    RoleAxis,
    ScopeAxis,
    ScopeReasonCode,
)

OptionId = Annotated[str, Field(pattern=SCOPE_OPTION_ID_PATTERN, max_length=128)]
"""An option id (`vocab.SCOPE_OPTION_ID_PATTERN`)."""

YearSpan = tuple[int, int]
"""A closed span of years, first and last."""


class TextOrigin(StrictModel):
    """The words an explicit entry was read from: the user's message on a request of one
    part, or the part's own variable text on a request of several."""

    kind: Literal["text"] = "text"
    text: str = Field(min_length=1, max_length=256, description="the words, as written")
    start: int = Field(ge=0, description="the offset of the first character in the text read")
    end: int = Field(ge=0, description="the offset past the last character in the text read")


class ChoiceOrigin(StrictModel):
    """The scope choice an entry was switched on."""

    kind: Literal["choice"] = "choice"
    choice_id: str = Field(
        min_length=1, max_length=256, description="the ScopeChoice.id the switch was made on"
    )


EntryOrigin = Annotated[TextOrigin | ChoiceOrigin, Field(discriminator="kind")]


class Entry(StrictModel):
    """One explicit or carried scope entry: an option the user asked for on one axis."""

    axis: ScopeAxis = Field(description="the axis the entry settles")
    option_id: OptionId = Field(description="the option asked for; its prefix is `axis`")
    provenance: EntryProvenance = Field(
        description=(
            "explicit: the request's own words or a switch; carried: a choice the "
            "conversation settled on an earlier part, attached by axis and concept"
        )
    )
    concept: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "the concept the entry was made for ('health'), so a carried entry attaches "
            "only to a part asking for that concept; None when the axis has no concept"
        ),
    )
    origin: EntryOrigin = Field(description="where the entry was read or switched")

    @model_validator(mode="after")
    def _option_on_axis(self) -> Entry:
        if self.option_id.partition(":")[0] != self.axis:
            raise ValueError(f"option_id {self.option_id!r} is not an option of {self.axis!r}")
        return self


class SlotBinding(StrictModel):
    """An option that binds one what/how slot of the identity."""

    kind: Literal["slot"] = "slot"
    slot: str = Field(description="the slot's constraint property dcid")
    state: Literal["value", "set"] = Field(description="one code, or a set of codes")
    refs: list[str] = Field(min_length=1, description="the bound code dcids, sorted")


class EntitiesBinding(StrictModel):
    """An option that adds entities to one role of the identity, beside its members."""

    kind: Literal["entities"] = "entities"
    role: RoleAxis = Field(description="the role the entities join")
    add: list[str] = Field(
        description="the entity dcids the option adds, sorted; [] for the members alone"
    )


class ShapeBinding(StrictModel):
    """An option that sets the identity's measure and netting basis."""

    kind: Literal["shape"] = "shape"
    measured_property: str = Field(description="the measured property dcid")
    netting_basis: str | None = Field(
        description="the netting basis code dcid; None when the measure carries none"
    )


class ProviderBinding(StrictModel):
    """An option that sets whose spending the answer counts."""

    kind: Literal["provider"] = "provider"
    scope: ProviderScope = Field(description="bilateral flows alone, or with imputed shares")


OptionBinding = Annotated[
    SlotBinding | EntitiesBinding | ShapeBinding | ProviderBinding,
    Field(discriminator="kind"),
]


class OptionAvailability(StrictModel):
    """Whether an option can be served, and why not."""

    status: OptionStatus = Field(description="available, partial or unavailable")
    years: list[YearSpan] | None = Field(
        default=None,
        description="the years a partial option covers, as closed spans; None otherwise",
    )
    code: ScopeReasonCode | None = Field(
        default=None, description="why the option is partial or unavailable; None when available"
    )
    detail: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "the code's parameters: `why` for provider_not_offered, `dataset` for "
            "basis_not_published, `entry` and `cause` for conflicts_with"
        ),
    )


class Option(StrictModel):
    """One value a scope choice can take."""

    option_id: OptionId = Field(description="the option's id")
    label: str = Field(description="the option for a reader")
    binding: OptionBinding = Field(description="what the option binds in the identity")
    availability: OptionAvailability = Field(description="whether a switch to it can be served")


class ScopeChoice(StrictModel):
    """How one axis of the part was counted, and the options it could take."""

    id: str = Field(description="'<PartScope.id>:<axis>'")
    axis: ScopeAxis = Field(description="the axis")
    rule: str = Field(description="the declared rule that settled the axis")
    provenance: ChoiceProvenance = Field(description="what settled the active option")
    active: OptionId = Field(description="the option the compiled identity holds")
    options: list[Option] = Field(min_length=1, description="every option, in declared order")


class DroppedEntry(StrictModel):
    """An entry left out because it cannot apply to this part: a carried entry, or on a
    fresh search an entry the request's words stated."""

    entry: Entry = Field(description="the entry left out")
    code: Literal["carried_choice_not_applied", "stated_choice_not_applied"] = Field(
        default="carried_choice_not_applied",
        description=(
            "carried_choice_not_applied for a carried entry, stated_choice_not_applied for "
            "an entry the request's words stated"
        ),
    )
    cause: str = Field(
        description=(
            "axis_absent, option_unknown, option_unavailable, entry_on_axis (an explicit or "
            "earlier carried entry settles the axis) or conflicts_with_entry for a carried "
            "entry; for a stated one, why the option cannot apply: axis_absent, "
            "option_unknown, the option's unavailability reason (a ScopeReasonCode) or the "
            "conflict that blocks it"
        )
    )


class Component(StrictModel):
    """One series the compiled identity is served from."""

    role: ComponentRole = Field(description="what the component holds")
    periods: list[YearSpan] | None = Field(
        default=None, description="the years the component serves; None for every year"
    )
    provenance: str | None = Field(
        default=None, description="the component's provenance dcid; None when unbound"
    )
    content_version: str | None = Field(
        default=None,
        description="the content version of the component's frame; None until it is minted",
    )
    identity: BaseIdentity = Field(description="the component's own identity")


class CompiledScope(StrictModel):
    """The identity the active options compile to."""

    spec_id: str = Field(description="the spec_id of `identity`")
    identity: BaseIdentity = Field(description="the compiled identity")
    components: list[Component] = Field(
        min_length=1, description="one for a plain Spec, one per series of a composite"
    )


class LedgerStep(StrictModel):
    """One switch, kept so it can be undone with its prior provenance."""

    axis: ScopeAxis = Field(description="the axis switched")
    prior: Entry | None = Field(description="the entry the switch replaced; None for a default")
    applied: Entry = Field(description="the entry the switch applied")
    revision: int = Field(ge=0, description="the plan revision the switch was made on")


class PartScope(StrictModel):
    """Every scope choice of one part, the entries that settled them and the identity they
    compile to."""

    id: str = Field(description="the scope's id: the spec_id of `base`")
    base: SpecDefinition = Field(
        description="the identity the part resolved; its bindings carry the constraints"
    )
    rule_version: str = Field(description="the hash of the declaration tables compiled against")
    entries: list[Entry] = Field(
        description="the explicit and carried entries applied, the input a switch posts back"
    )
    dropped_entries: list[DroppedEntry] = Field(
        default_factory=list,
        description=(
            "the entries left out, each with its code and cause: carried entries the part "
            "cannot take, and on a fresh search the stated entries it cannot serve"
        ),
    )
    choices: list[ScopeChoice] = Field(description="one per axis the part settles")
    compiled: CompiledScope = Field(description="what the active options compile to")
    ledger: list[LedgerStep] = Field(
        default_factory=list,
        description="the switches made, newest last; the server keeps none, so [] on the wire",
    )
