"""Read the scope a user states in words: explicit scope entries and the total intent.

Pure and deterministic: the same text always reads the same entries, which the raw-text
door's response cache key relies on (`response_cache_keys.layer2_key`).

Two axes are read:

- Provider scope. "bilateral" states `provider:bilateral`, as does "multilateral" under a
  negation or exclusion ("excluding multilateral", "not multilateral institutions",
  `bind_repair.occurrence_negated`). "including multilateral" (also "incl.", "inclusive of",
  "plus", and "bilateral plus imputed multilateral") states `provider:inclusive`. Any other
  mention of "multilateral" ("multilateral development banks", "bilateral/multilateral")
  names a kind of provider and leaves the axis unstated, even beside a "bilateral".
- Measure basis. "net", "gross", "grant equivalent(s)" and "commitments" state their
  basis. The singular "commitment" is a pledge ("has the US kept its commitment") and
  states none. A word that heads another measure ("net worth", "gross national income", "ODA net
  of debt relief") states none.

A negated or excluded occurrence ("not just gross disbursements", "non-bilateral") states
nothing. An axis whose words state two options ("both gross and net ODA", "commitments and
gross disbursements") is left unstated, since the text asks for both and neither is the
scope. "total" ("total health aid", "totalled across donors") states the total intent the
provider default reads. It is a fact about the request and sits outside the entries.

Which text a part is read from is the door's choice (`part_scope_intakes`): a request of one
part reads the user's message, and a request of several reads each part's own text, so
each part's words stay with that part.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from code_index.scope_conventions import MeasureBasis
from search.engine.bind_repair import occurrence_negated
from search.engine.response_cache_keys import normalise_query
from search.models import SCOPE_AXES, Entry, ScopeAxis, TextOrigin
from search.scope.state import ScopeIntake
from search.scope.vocabulary import measure_basis_option_id, provider_option_id

# Words that, after a basis word, make it part of another measure ("net worth", "gross
# national income", "net of debt relief") or a verb ("net the two"), so it states no basis.
_NET_HEADS = (
    "of",
    "the",
    "them",
    "it",
    "out",
    "off",
    "debt",
    "worth",
    "income",
    "national",
    "migration",
    "zero",
    "enrolment",
    "enrollment",
    "exports",
    "imports",
    "lending",
    "savings",
    "assets",
    "present",
    "primary",
)
_GROSS_HEADS = (
    "domestic",
    "national",
    "capital",
    "fixed",
    "enrolment",
    "enrollment",
    "savings",
    "income",
)


def _heads(words: Sequence[str]) -> str:
    return r"(?!\s+(?:" + "|".join(words) + r")\b)"


_BASIS_WORDS: tuple[tuple[re.Pattern[str], MeasureBasis], ...] = (
    (re.compile(r"\bgrant[\s-]+equivalents?\b", re.IGNORECASE), "grant_equivalent"),
    (re.compile(r"\bnet\b" + _heads(_NET_HEADS), re.IGNORECASE), "net"),
    (re.compile(r"\bgross\b" + _heads(_GROSS_HEADS), re.IGNORECASE), "gross"),
    (re.compile(r"\bcommitments\b", re.IGNORECASE), "commitments"),
)
_BILATERAL = re.compile(r"\bbilateral\b", re.IGNORECASE)
_MULTILATERAL = re.compile(r"\bmultilaterals?\b", re.IGNORECASE)
_INCLUSIVE = re.compile(
    r"\b(?:bilateral\s+)?(?:including|incl\b\.?|inclusive\s+of|plus)\s+"
    r"(?:the\s+)?(?:imputed\s+)?multilaterals?\b",
    re.IGNORECASE,
)
_TOTAL = re.compile(r"\btotal(?:s|led|ed|ling|ing)?\b", re.IGNORECASE)
# The words that negate or exclude what follows (`bind_repair.occurrence_negated`), found
# again only to start an exclusion's origin at its cue.
_NEGATION_CUE = re.compile(
    r"\b(?:non|not|no|never|excluding|exclude|excludes|except|without|other\s+than|"
    r"rather\s+than|excl)\b",
    re.IGNORECASE,
)
_CUE_REACH_WORDS = 3


@dataclass(frozen=True, slots=True)
class StatedScope:
    """What one text states of its scope: an explicit entry per axis it states, in axis
    order, each with the words it was read from as a text origin, and whether it asks for
    a total."""

    entries: tuple[Entry, ...] = ()
    total: bool = False


@dataclass(frozen=True, slots=True)
class _Stated:
    option_id: str
    start: int
    end: int


def read_stated_scope(text: str) -> StatedScope:
    """The explicit scope entries and total intent `text` states (see the module
    docstring)."""
    stated: dict[ScopeAxis, list[_Stated] | None] = {
        "measure_basis": _basis_stated(text),
        "provider": _provider_stated(text),
    }
    entries = []
    for axis in SCOPE_AXES:
        found = stated.get(axis)
        if not found or len({s.option_id for s in found}) != 1:
            continue
        first = min(found, key=lambda s: s.start)
        origin = TextOrigin(text=text[first.start : first.end], start=first.start, end=first.end)
        entries.append(
            Entry(axis=axis, option_id=first.option_id, provenance="explicit", origin=origin)
        )
    total = any(not occurrence_negated(text, m.start(), m.end()) for m in _TOTAL.finditer(text))
    return StatedScope(entries=tuple(entries), total=total)


def _basis_stated(text: str) -> list[_Stated]:
    return [
        _Stated(measure_basis_option_id(basis), m.start(), m.end())
        for pattern, basis in _BASIS_WORDS
        for m in pattern.finditer(text)
        if not occurrence_negated(text, m.start(), m.end())
    ]


def _provider_stated(text: str) -> list[_Stated] | None:
    """The provider options `text` states, or None when a mention of "multilateral" names a
    kind of provider, which leaves the axis unstated."""
    inclusive = list(_INCLUSIVE.finditer(text))
    found = [
        _Stated(
            provider_option_id(
                "bilateral" if occurrence_negated(text, m.start(), m.end()) else "inclusive"
            ),
            _cue_start(text, m.start()),
            m.end(),
        )
        for m in inclusive
    ]

    def within_inclusive(start: int) -> bool:
        return any(m.start() <= start < m.end() for m in inclusive)

    for m in _BILATERAL.finditer(text):
        if within_inclusive(m.start()) or occurrence_negated(text, m.start(), m.end()):
            continue
        found.append(_Stated(provider_option_id("bilateral"), m.start(), m.end()))
    for m in _MULTILATERAL.finditer(text):
        if within_inclusive(m.start()):
            continue
        if not occurrence_negated(text, m.start(), m.end()):
            return None
        found.append(_Stated(provider_option_id("bilateral"), _cue_start(text, m.start()), m.end()))
    return found


def _cue_start(text: str, start: int) -> int:
    """Where the negation or exclusion before `text[start:]` begins, within its reach;
    `start` itself when there is none, or for a fused "non"."""
    words = list(re.finditer(r"[\w']+", text[:start]))
    window_start = words[-_CUE_REACH_WORDS].start() if len(words) >= _CUE_REACH_WORDS else 0
    cues = list(_NEGATION_CUE.finditer(text, window_start, start))
    return cues[-1].start() if cues else start


def part_scope_intakes(
    *,
    message: str | None,
    part_texts: Sequence[str],
    carried: Sequence[Entry] = (),
) -> list[ScopeIntake]:
    """The scope each part is posted with, in part order.

    A request of one part reads `message`, or the part's own text when there is none; a
    request of several reads each part's own text. The text is read normalised
    (`response_cache_keys.normalise_query`), and every text origin of the part indexes it
    (`ScopeIntake.text`), so requests the response cache serves alike (layer 1 by the
    normalised query, layer 2 by a single part's whitespace-collapsed query or by each
    part's own text) point at the same words. Every part also takes every `carried` entry,
    which attaches to it by axis and concept once its binders have run
    (`ScopeState.attached`)."""
    raw = [message] if len(part_texts) == 1 and message is not None else part_texts
    intakes = []
    for text in map(normalise_query, raw):
        stated = read_stated_scope(text)
        intakes.append(
            ScopeIntake(entries=(*stated.entries, *carried), total=stated.total, text=text)
        )
    return intakes


def intake_key_document(intake: ScopeIntake) -> dict[str, object] | None:
    """`intake` as the response cache key states it, every entry whole, since its origin
    reaches the answer; None when it posts nothing."""
    if not intake.entries and not intake.total:
        return None
    return {
        "entries": [entry.model_dump(mode="json") for entry in intake.entries],
        "total": intake.total,
    }
