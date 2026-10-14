"""The interpretations of one part: the readings the catalogue could serve, each admitted or
refused under the part's frame.

An `Interpretation` is one shape as derive groups it: its five-tuple, role class,
membership authority, family, datasets and origin all ride on `ShapeDraft`, and a menu
shape carries its slot menu (`ShapeDraft.slot_taxonomy`). Code decides whether it is
admissible (`region.admissibility`); a refused one names its reason from the refusal
vocabulary, and, when a pinned part of the question refused it, which part
(`PinnedPart`). The bind model elects among the admissible interpretations only. It also
sees, as a list it cannot elect, the interpretations refused only on a pinned part, so that
when it declines every admissible one it can name the nearest of them, which becomes the
refusal's flagged match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from search.models import elect_no_data_reason

if TYPE_CHECKING:
    from search.engine.qualifier_match import QualifierMatch
    from search.engine.shape import ShapeDraft
    from search.models import NoDataReason

type PinnedPart = Literal["form", "pinned_code"]
"""The pinned part of a question a reading fails: the normalisation form it names, or a
code it names on a property the reading's slots hold."""


@dataclass(frozen=True, slots=True)
class Interpretation:
    """One reading of a part: a shape, admissible when `reason` is `None`.

    `reason` is the refusal a part answers with when this reading is the best it has, and
    `pinned_failure` the pinned part that refused it, `None` for any other refusal.
    `pinned_codes` are the codes the question names that the reading cannot bind, set when
    `pinned_failure` is `pinned_code`.
    """

    shape: ShapeDraft
    reason: NoDataReason | None = None
    pinned_failure: PinnedPart | None = None
    pinned_codes: tuple[QualifierMatch, ...] = ()

    @property
    def admissible(self) -> bool:
        return self.reason is None


@dataclass(frozen=True, slots=True)
class InterpretationSet:
    """Every interpretation of a part, the admissible ones the bind menu shows in menu
    order, and the ones refused only on a pinned part, seated as the menu is under the
    same seat count, which the menu lists as not electable.

    `interpretations` holds every shape candidacy and the score floor leave, admitted or
    refused. Family election, the measure-family collapse and the menu cap narrow the
    admissible ones without refusing any, so `admissible` can hold fewer than every
    admissible interpretation.
    """

    interpretations: tuple[Interpretation, ...] = ()
    admissible: tuple[Interpretation, ...] = ()
    refused_on_pinned: tuple[Interpretation, ...] = ()

    def refusal_reason(self) -> NoDataReason | None:
        """The reason a part with no admissible interpretation answers: the one
        `NO_DATA_RESOLUTION_ORDER` ranks first among the refused interpretations' reasons.
        `None` when no interpretation was refused."""
        reasons = [i.reason for i in self.interpretations if i.reason is not None]
        if not reasons:
            return None
        return elect_no_data_reason(reasons)

    def nearest_refused(self, index: int | None) -> Interpretation | None:
        """The interpretation `index` names in the not-electable list, `None` when it names
        none or falls outside the list."""
        if index is None or not 0 <= index < len(self.refused_on_pinned):
            return None
        return self.refused_on_pinned[index]
