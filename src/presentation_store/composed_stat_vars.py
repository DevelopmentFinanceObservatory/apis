"""Labels the composed StatVars a published frame describes.

An analysis sum, quotient or share writes its rows under a StatVar of its own, which the graph
does not hold. The frame's version body lists each one under `composed_stat_vars` as
`{stat_var, relation, first, second}`. `read_composed_stat_vars` reads that list, and
`composed_stat_var_labels` words each entry from its sides.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dfo_frame_grammar.grammar import STAT_VAR_REF_PREFIX

from assembler.limits import LABEL_MAX, joined_label
from assembler.ports import FrameRef

__all__ = ["ComposedStatVar", "composed_stat_var_labels", "read_composed_stat_vars"]

logger = logging.getLogger(__name__)

COMPOSED_STAT_VAR_UNREADABLE_MARKER = "PRESENTATION_COMPOSED_STAT_VAR_UNREADABLE"

_JOINERS = {"quotient": " per ", "sum": " plus ", "share": " as a share of "}
_SIDE_JOINER = " and "
_EQUAL_SIDES_PREFIXES = {"quotient": "Ratio of ", "share": "Share of "}


@dataclass(frozen=True)
class ComposedStatVar:
    """One stored entry: a StatVar and the two sides its `relation` combines."""

    stat_var: str
    relation: str
    first: tuple[str, ...]
    second: tuple[str, ...]


def read_composed_stat_vars(stored: object, *, frame_ref: FrameRef) -> tuple[ComposedStatVar, ...]:
    """Reads a version body's `composed_stat_vars` value, keeping the entries this module can
    word: a mapping with a string `stat_var`, a `relation` of `quotient`, `sum` or `share`, and two
    non-empty lists of strings.

    A label is best-effort, so an entry of any other shape is skipped and its StatVar keeps
    its ref. A value that is no list reads as no entries. Each skip logs one warning naming
    the frame and the entry's position, and none of the entry's content. `None` is an absent
    member and logs nothing."""
    if stored is None:
        return ()
    if not isinstance(stored, list):
        logger.warning(
            "%s spec_id=%s content_version=%s reason=not_a_list",
            COMPOSED_STAT_VAR_UNREADABLE_MARKER,
            frame_ref.spec_id,
            frame_ref.content_version,
        )
        return ()
    entries: list[ComposedStatVar] = []
    for position, raw in enumerate(stored):
        entry = _read_entry(raw)
        if entry is None:
            logger.warning(
                "%s spec_id=%s content_version=%s position=%d",
                COMPOSED_STAT_VAR_UNREADABLE_MARKER,
                frame_ref.spec_id,
                frame_ref.content_version,
                position,
            )
            continue
        entries.append(entry)
    return tuple(entries)


def _read_entry(raw: object) -> ComposedStatVar | None:
    if not isinstance(raw, Mapping):
        return None
    stat_var, relation = raw.get("stat_var"), raw.get("relation")
    first, second = _read_side(raw.get("first")), _read_side(raw.get("second"))
    if not isinstance(stat_var, str) or not isinstance(relation, str):
        return None
    if relation not in _JOINERS or not first or not second:
        return None
    return ComposedStatVar(stat_var=stat_var, relation=relation, first=first, second=second)


def _read_side(raw: object) -> tuple[str, ...]:
    """The side's refs, empty for a side that is no list of strings."""
    if not isinstance(raw, list):
        return ()
    refs = tuple(ref for ref in raw if isinstance(ref, str))
    return refs if len(refs) == len(raw) else ()


def composed_stat_var_labels(
    *, entries: Sequence[ComposedStatVar], labels: Mapping[str, str]
) -> dict[str, str]:
    """Returns each entry's label, keyed by its `stat_var` ref.

    Entries are worded in list order, so an entry reads the label of an earlier entry its side
    names. A side reads as its refs' labels joined by " and "; a ref `labels` does not hold
    reads as its dcid without `dcid:`. A quotient reads `<first> per <second>`, a sum
    `<first> plus <second>` and a share `<first> as a share of <second>`. With equal sides a
    quotient reads `Ratio of <side>`, a share `Share of <side>` and a sum `<side>`.
    `joined_label` fits two sides within `LABEL_MAX` characters, so both
    sides and the joiner stay in the label."""
    known = dict(labels)
    composed: dict[str, str] = {}
    for entry in entries:
        first = _side_text(entry.first, labels=known)
        if entry.first != entry.second:
            second = _side_text(entry.second, labels=known)
            label = joined_label(first, second, joiner=_JOINERS[entry.relation])
        else:
            label = (_EQUAL_SIDES_PREFIXES.get(entry.relation, "") + first)[:LABEL_MAX]
        known[entry.stat_var] = composed[entry.stat_var] = label
    return composed


def _side_text(refs: Sequence[str], *, labels: Mapping[str, str]) -> str:
    return _SIDE_JOINER.join(labels.get(ref, ref.removeprefix(STAT_VAR_REF_PREFIX)) for ref in refs)
