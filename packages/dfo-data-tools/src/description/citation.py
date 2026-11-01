"""How a version is cited: its citation line and its `cite` block.

`citation_line` is the one function that writes a citation line. A citation's identity is
evidence: the title composed at mint, the ids, the version and the retrieval date. The names
in its line are today's words, so they follow the vocabulary.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence

from dfo_descriptions import MAX_CITE_TEXT_LENGTH

from description.blocks import Wire, Words, data_link, source_name
from description.facts import DescriptionRecord, SourceEvidence
from description.steps import Step
from description.stored import clipped
from dfo_platform.publisher import DFO_NAME

FIGURE_CHANGING_OPS: frozenset[str] = frozenset(
    {
        "sum",
        "quotient",
        "share",
        "growth",
        "rank",
        "aggregate",
        "compare",
        "sum_variables",
        "convert",
        "change",
    }
)
"""The step operations whose output holds figures the source did not publish: arithmetic on
its values, a ranking whose values can be rank positions, a comparison that adds the
difference, a sum of a family's variables, a conversion to another unit and a card's change."""

FIGURE_PRESERVING_OPS: frozenset[str] = frozenset(
    {"retrieve", "filter", "join", "source_step", "choose_source", "continue_series", "select"}
)
"""The step operations that pass the source's figures through: they read, pick, pair or drop
values, or record the source's own pipeline and estimates, and leave every value as published."""


def citation_line(
    *, retrieved: datetime.date, sources: Sequence[str], processed: bool = False
) -> str:
    """The citation line of a version, in the short form of the site's citation guide: the
    organisations that first published the data and the year of `retrieved`, then the DFO as
    the route.

    A version whose figures the DFO passes through unchanged reads "OECD Creditor Reporting
    System (2026), via the Development Finance Observatory". A version whose figures the DFO
    changed (`processed`), such as a computed version naming the sources its inputs credit,
    reads "OECD Creditor Reporting System and World Development Indicators (2026), processed
    by the Development Finance Observatory". With no sources the line reads "Development Finance Observatory
    (2026)". A line longer than `MAX_CITE_TEXT_LENGTH` is clipped to it, ending in "…".
    """
    if not sources:
        line = f"{DFO_NAME} ({retrieved.year})"
    else:
        route = "processed by" if processed else "via"
        line = f"{join_names(sources)} ({retrieved.year}), {route} the {DFO_NAME}"
    return clipped(line, limit=MAX_CITE_TEXT_LENGTH)


def figures_processed(*, version_id: str, steps: Sequence[Step]) -> bool:
    """True when the DFO changed the figures of a result drawn from version `version_id`: an
    `ops_` version, which the DFO computed, or a result with a step in `FIGURE_CHANGING_OPS`,
    such as a conversion, a sum of a family's variables or a card's change. `steps` are the
    steps known to have made the result, the version's and then a card's, so steps that are
    unknown record no change and leave the known ones to decide."""
    return version_id.startswith("ops_") or any(step.op in FIGURE_CHANGING_OPS for step in steps)


def join_names(names: Sequence[str]) -> str:
    """Names joined as a sentence lists them: "A", "A and B", "A, B and C"."""
    if len(names) < 2:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def cite(
    record: DescriptionRecord,
    words: Words,
    *,
    title: str,
    composed: bool,
    sources: Sequence[SourceEvidence] | None = None,
    steps: Sequence[Step] = (),
) -> Wire:
    """The `cite` block of a version: the title composed at mint; for a computed version,
    which has none, the name its steps derive (`title`, when `composed`); else `title` with a
    gap. Then the citation line, naming `sources` where given and else the version's, routed
    through the DFO as the steps known to have made the cited result decide
    (`figures_processed`): the version's, then a card's own `steps`. A computed version's line
    names `sources`, the sources its inputs credit, and its inputs where none are given. Then
    the version's identity and retrieval date; the link to its data; and a computed version's
    inputs."""
    prepared = record.evidence.prepared
    cited = record.evidence.title
    if cited is None and record.computed is not None and composed:
        cited = title
    if cited is None:
        words.gap("cite.title")
    block: Wire = {"title": cited or title}
    if record.computed is not None:
        inputs = [
            words.input_ref(version, f"cite.inputs.{n}")
            for n, version in enumerate(record.computed.inputs)
        ]
        block["text"] = citation_line(
            retrieved=prepared,
            sources=[source_name(source, words) for source in sources]
            if sources
            else [item["label"] for item in inputs],
            processed=True,
        )
    else:
        inputs = None
        block["text"] = citation_line(
            retrieved=prepared,
            sources=[
                source_name(source, words)
                for source in (record.evidence.sources if sources is None else sources)
            ],
            processed=figures_processed(version_id=record.id, steps=[*record.known_steps, *steps]),
        )
    block |= {
        "id": record.id,
        "content_version": record.content_version,
        "retrieved": prepared.isoformat(),
        "data": data_link(record),
    }
    if inputs is not None:
        block["inputs"] = inputs
    return block
