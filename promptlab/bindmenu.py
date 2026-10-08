"""Parse and truncate the candidate-shape menu in bind prompts.

`parse_menu` accepts the builder's exact format and skips fixed-context, dropped-
qualifier, and taxonomy-label blocks. `truncate_menu` keeps the original text and
limits the shape count and header.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

_NONE = "(none)"


class BindMenuError(ValueError):
    """Malformed bind-menu prompt. The message names the offending line."""


@dataclass(frozen=True)
class MenuEntry:
    """One candidate shape read back from a bind prompt's menu.

    `five_tuple` uses the five identity keys. `None` represents `(none)`.
    `(present, unresolved)` marks a denominator arc with an unresolved target.
    """

    index: int
    label: str
    five_tuple: dict[str, str | None]
    sample_dcids: tuple[str, ...]
    retrieval_score: float | None


_VARIABLE_RE = re.compile(r"Variable phrase: (?P<variable>.*)")
_HEADER_RE = re.compile(r"CANDIDATE SHAPES \((?P<count>\d+) total\):")
_SHAPE_RE = re.compile(r"Shape (?P<index>\d+):")
_LABEL_RE = re.compile(r"  label: (?P<label>.*)")
_FAMILY_RE = re.compile(r"  family: .*")
_FIVE_TUPLE_RE = re.compile(
    r"  five_tuple: populationType=(?P<pt>.*), measuredProperty=(?P<mp>.*), statType=(?P<st>.*)"
)
_QUAL_DENOM_RE = re.compile(
    r"    measurementQualifier=(?P<mq>.*), measurementDenominator=(?P<md>.*)"
)
_SCORE_RE = re.compile(r"  retrieval_score: (?P<score>.*)")
_FIXED_CONTEXT_HEADER_LINE = (
    "FIXED CONTEXT (already determined by the query text -- if you choose a shape "
    "that lists fixed_context, bind each constraint_slot it names kind=value to "
    "exactly that value_dcid, not your own reading)."
)
_SHAPE_FIXED_CONTEXT_LINE = "  fixed_context:"
_FIXED_CONTEXT_ENTRY_RE = re.compile(
    r"    property_dcid=(?P<prop>[^,]*), value_dcid=(?P<val>.*) \((?P<label>.*)\)"
)
_DROPPED_QUALIFIERS_HEADER_LINE = (
    "DROPPED QUALIFIERS (the query asked for these but no earlier stage could "
    "apply them -- for the shape you choose, if one of these phrases asks for "
    'every value of one of its constraint slots (e.g. "across all flows", '
    '"regardless of type"), bind that slot kind=unbound, evidence=requested_all, '
    "and set that binding's own requested_all_phrase to the phrase verbatim; "
    "leave requested_all_phrase null when no phrase names one of the chosen "
    "shape's own constraint slots):"
)
_DROPPED_QUALIFIERS_ENTRY_RE = re.compile(r"  (?P<phrase>.*)")
_CONSTRAINT_NONE_LINE = "  constraint_slots: (none)"
_CONSTRAINT_HEADER_LINE = "  constraint_slots:"
_SLOT_AXIS_RE = re.compile(
    r"    axis=(?P<axis>[^,]*), property_dcid=(?P<prop>.*) \((?P<label>.*)\)"
)
_SLOT_TAXONOMY_RE = re.compile(r"      taxonomy: (?P<values>.*)")
_SLOT_LABELS_RE = re.compile(r"      labels: .*")
_SAMPLE_RE = re.compile(r"  sample_member_dcids: (?P<values>.*)")


def _none_or(raw: str) -> str | None:
    return None if raw == _NONE else raw


def _match(
    lines: list[str], pos: int, pattern: re.Pattern[str], what: str
) -> tuple[re.Match[str], int]:
    if pos >= len(lines):
        raise BindMenuError(f"prompt ended while expecting {what}")
    found = pattern.fullmatch(lines[pos])
    if found is None:
        raise BindMenuError(f"line {pos}: expected {what}, got {lines[pos]!r}")
    return found, pos + 1


def _literal(lines: list[str], pos: int, expected: str) -> int:
    got = lines[pos] if pos < len(lines) else "<end of prompt>"
    if got != expected:
        raise BindMenuError(f"line {pos}: expected {expected!r}, got {got!r}")
    return pos + 1


def _list_literal(raw: str, *, what: str) -> list[object]:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise BindMenuError(f"{what}: not a valid list literal: {raw!r}") from exc
    if not isinstance(value, list):
        raise BindMenuError(f"{what}: not a valid list literal: {raw!r}")
    return value


def _str_tuple_literal(raw: str, *, what: str) -> tuple[str, ...]:
    strings: list[str] = []
    for value in _list_literal(raw, what=what):
        if not isinstance(value, str):
            raise BindMenuError(f"{what}: expected a list of str, got {raw!r}")
        strings.append(value)
    return tuple(strings)


def _parse_shape(lines: list[str], pos: int, idx: int) -> tuple[MenuEntry, int]:
    match, pos = _match(lines, pos, _SHAPE_RE, f"'Shape {idx}:'")
    if int(match.group("index")) != idx:
        raise BindMenuError(f"line {pos - 1}: expected 'Shape {idx}:', got {lines[pos - 1]!r}")

    match, pos = _match(lines, pos, _LABEL_RE, "a 'label:' line")
    label = match.group("label")

    _, pos = _match(lines, pos, _FAMILY_RE, "a 'family:' line")

    match, pos = _match(lines, pos, _FIVE_TUPLE_RE, "a 'five_tuple:' line")
    pt, mp, st = match.group("pt"), match.group("mp"), match.group("st")

    mq: str | None = None
    md: str | None = None
    if pos < len(lines) and _QUAL_DENOM_RE.fullmatch(lines[pos]):
        match, pos = _match(lines, pos, _QUAL_DENOM_RE, "a 'measurementQualifier=' line")
        mq = _none_or(match.group("mq"))
        md = _none_or(match.group("md"))

    retrieval_score: float | None = None
    if pos < len(lines) and _SCORE_RE.fullmatch(lines[pos]):
        match, pos = _match(lines, pos, _SCORE_RE, "a 'retrieval_score:' line")
        retrieval_score = float(match.group("score"))

    if pos < len(lines) and lines[pos] == _CONSTRAINT_NONE_LINE:
        pos += 1
    else:
        pos = _literal(lines, pos, _CONSTRAINT_HEADER_LINE)
        saw_slot = False
        while pos < len(lines) and _SLOT_AXIS_RE.fullmatch(lines[pos]):
            pos += 1
            match, pos = _match(lines, pos, _SLOT_TAXONOMY_RE, "a 'taxonomy:' line")
            _list_literal(match.group("values"), what="taxonomy")
            if pos < len(lines) and _SLOT_LABELS_RE.fullmatch(lines[pos]):
                pos += 1
            saw_slot = True
        if not saw_slot:
            raise BindMenuError(f"line {pos}: 'constraint_slots:' header with no slot lines")

    if pos < len(lines) and lines[pos] == _SHAPE_FIXED_CONTEXT_LINE:
        pos += 1
        saw_fixed_context_entry = False
        while pos < len(lines) and _FIXED_CONTEXT_ENTRY_RE.fullmatch(lines[pos]):
            pos += 1
            saw_fixed_context_entry = True
        if not saw_fixed_context_entry:
            raise BindMenuError(f"line {pos}: 'fixed_context:' header with no property_dcid lines")

    sample_dcids: tuple[str, ...] = ()
    if pos < len(lines) and _SAMPLE_RE.fullmatch(lines[pos]):
        match, pos = _match(lines, pos, _SAMPLE_RE, "a 'sample_member_dcids:' line")
        sample_dcids = _str_tuple_literal(match.group("values"), what="sample_member_dcids")

    five_tuple: dict[str, str | None] = {
        "population_type": _none_or(pt),
        "measured_property": _none_or(mp),
        "stat_type": _none_or(st),
        "measurement_qualifier": mq,
        "measurement_denominator": md,
    }
    entry = MenuEntry(
        index=idx,
        label=label,
        five_tuple=five_tuple,
        sample_dcids=sample_dcids,
        retrieval_score=retrieval_score,
    )
    return entry, pos


def parse_menu(prompt: str) -> tuple[str, tuple[MenuEntry, ...]]:
    """Parse the variable phrase and shapes from a builder-formatted bind prompt.

    Raises `BindMenuError` on a malformed line, count mismatch, or trailing content.

    Optional `FIXED CONTEXT` and `DROPPED QUALIFIERS` blocks, and each shape's
    `fixed_context` lines, are parsed and skipped. `QUERY-NAMED ALTERNATIVES` raise
    `BindMenuError`.
    """
    lines = prompt.split("\n")
    if not lines or lines[-1] != "":
        raise BindMenuError("prompt must end with exactly one trailing newline")
    lines = lines[:-1]

    pos = 0
    match, pos = _match(lines, pos, _VARIABLE_RE, "a 'Variable phrase:' line")
    variable = match.group("variable")
    pos = _literal(lines, pos, "")

    if pos < len(lines) and lines[pos] == _FIXED_CONTEXT_HEADER_LINE:
        pos = _literal(lines, pos + 1, "")

    if pos < len(lines) and lines[pos] == _DROPPED_QUALIFIERS_HEADER_LINE:
        pos += 1
        saw_dropped_qualifier_entry = False
        while pos < len(lines) and _DROPPED_QUALIFIERS_ENTRY_RE.fullmatch(lines[pos]):
            pos += 1
            saw_dropped_qualifier_entry = True
        if not saw_dropped_qualifier_entry:
            raise BindMenuError(f"line {pos}: 'DROPPED QUALIFIERS' header with no phrase lines")
        pos = _literal(lines, pos, "")

    match, pos = _match(lines, pos, _HEADER_RE, "a 'CANDIDATE SHAPES (N total):' line")
    declared_count = int(match.group("count"))
    pos = _literal(lines, pos, "")

    entries: list[MenuEntry] = []
    for idx in range(declared_count):
        entry, pos = _parse_shape(lines, pos, idx)
        entries.append(entry)
        if idx < declared_count - 1:
            pos = _literal(lines, pos, "")

    if pos != len(lines):
        raise BindMenuError(
            f"line {pos}: unexpected content after {declared_count} declared shapes"
        )

    return variable, tuple(entries)


def truncate_menu(prompt: str, k: int) -> str:
    """Keep the first `k` shapes and update the count header. Pre-shape text stays
    unchanged, and slices preserve the builder's formatting. When `k` covers the
    full menu, return `prompt` unchanged. Raises `BindMenuError` for malformed
    prompts or `k < 1`."""
    _variable, entries = parse_menu(prompt)
    total = len(entries)
    if k < 1:
        raise BindMenuError(f"k={k} out of range: must be >= 1")
    if k >= total:
        return prompt

    lines = prompt.split("\n")[:-1]
    shape_starts = [i for i, line in enumerate(lines) if _SHAPE_RE.fullmatch(line)]
    if len(shape_starts) != total:
        raise BindMenuError("shape header count does not match the parsed entry count")

    new_lines = [
        f"CANDIDATE SHAPES ({k} total):" if _HEADER_RE.fullmatch(line) else line
        for line in lines[: shape_starts[0]]
    ]
    for i in range(k):
        end = shape_starts[i + 1] - 1 if i + 1 < total else len(lines)
        if i > 0:
            new_lines.append("")
        new_lines.extend(lines[shape_starts[i] : end])

    return "\n".join(new_lines) + "\n"
