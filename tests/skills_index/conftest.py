"""Provides the `SkillsGraph` test double with configurable coverage, arcs, and batch failures."""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import StrEnum

from skills_index.graph import SubjectsCoverage, SubjectsPage


def stored(markdown: str) -> str:
    """Returns `markdown` as the publisher stores a `text` value: the inside of a
    JSON string, with quotes as `\\u0022`."""
    return json.dumps(markdown)[1:-1].replace('\\"', "\\u0022")


def skill_markdown(skill_id: str, *, description: str = "Does things.", body: str = "Body.") -> str:
    """Returns a `SKILL.md` whose front matter names `skill_id`."""
    return f"---\nname: {skill_id}\ndescription: {description}\n---\n{body}\n"


def skill_node(
    skill_id: str, *, description: str = "Does things.", body: str = "Body."
) -> dict[str, tuple[str, ...]]:
    """Returns the arcs of a skill node whose stored `text` is `skill_markdown`."""
    return {"text": (stored(skill_markdown(skill_id, description=description, body=body)),)}


class FakeSkillsGraph:
    """`subjects`: `{(dcid, property): [subject_dcid, ...]}`, fed to
    `node_subjects_many` -- a `(dcid, property)` pair absent from
    `subjects` answers `SubjectsPage(dcids=(), coverage=SubjectsCoverage
    .complete)`. `capped`: the set of `(dcid, property)` pairs whose page
    reports `coverage=SubjectsCoverage.capped` regardless of how many
    subjects it carries. `budget_exhausted`: the same, for
    `coverage=SubjectsCoverage.budget_exhausted`. A pair is never in both
    sets, matching `SubjectsCoverage`'s own mutual exclusivity.
    `failing_subject_reads`: `(dcid, property)` pairs whose presence in a
    `node_subjects_many` call makes the whole call raise. `arcs`:
    `{dcid: {property: (value, ...)}}`, fed to `node_arcs_many` -- a
    dcid absent from `arcs` answers `None`; a dcid present but missing a
    requested property answers `()` for that property. A `node_arcs_many`
    call naming any dcid in `failing_arc_dcids` raises. `calls` and
    `arc_calls` record every `node_subjects_many` / `node_arcs_many`
    call's own arguments, in order. `coverage_cls` is the `StrEnum` every
    `SubjectsPage.coverage` this fake builds is drawn from -- it defaults
    to `skills_index.graph.SubjectsCoverage`, the class `build`/`expand`
    themselves import, so most tests exercise a same-class comparison.
    Passing `search.engine.clients.graph.SubjectsCoverage` here instead
    is what a real `SkillsGraph` implementation actually hands back, and
    is the only way this suite can tell a by-value comparison in
    `build`/`expand` apart from a by-identity one.
    """

    def __init__(
        self,
        *,
        subjects: dict[tuple[str, str], list[str]] | None = None,
        capped: set[tuple[str, str]] | None = None,
        budget_exhausted: set[tuple[str, str]] | None = None,
        arcs: dict[str, dict[str, tuple[str, ...]]] | None = None,
        failing_arc_dcids: frozenset[str] | None = None,
        failing_subject_reads: set[tuple[str, str]] | None = None,
        coverage_cls: type[StrEnum] = SubjectsCoverage,
    ) -> None:
        self.subjects = subjects or {}
        self.capped = capped or set()
        self.budget_exhausted = budget_exhausted or set()
        self.arcs = arcs or {}
        self.failing_arc_dcids = failing_arc_dcids or frozenset()
        self.failing_subject_reads = failing_subject_reads or set()
        self.coverage_cls = coverage_cls
        self.calls: list[tuple[tuple[str, ...], str, int]] = []
        self.arc_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def node_subjects_many(
        self, dcids: Sequence[str], *, property: str, max_results: int
    ) -> dict[str, SubjectsPage]:
        self.calls.append((tuple(dcids), property, max_results))
        if any((dcid, property) in self.failing_subject_reads for dcid in dcids):
            raise RuntimeError(f"simulated graph infra failure for {tuple(dcids)!r} <-{property}")
        result: dict[str, SubjectsPage] = {}
        for dcid in dcids:
            page_subjects = tuple(self.subjects.get((dcid, property), ()))
            if (dcid, property) in self.budget_exhausted:
                coverage = self.coverage_cls.budget_exhausted
            elif (dcid, property) in self.capped:
                coverage = self.coverage_cls.capped
            else:
                coverage = self.coverage_cls.complete
            result[dcid] = SubjectsPage(dcids=page_subjects[:max_results], coverage=coverage)
        return result

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        self.arc_calls.append((tuple(dcids), tuple(properties)))
        if self.failing_arc_dcids.intersection(dcids):
            raise RuntimeError(f"simulated graph infra failure for {tuple(dcids)!r}")
        result: dict[str, dict[str, tuple[str, ...]] | None] = {}
        for dcid in dcids:
            if dcid not in self.arcs:
                result[dcid] = None
                continue
            node = self.arcs[dcid]
            result[dcid] = {property: node.get(property, ()) for property in properties}
        return result

    def node_names_many(
        self,
        dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, tuple[str, ...]]:
        result: dict[str, tuple[str, ...]] = {}
        for dcid in dict.fromkeys(dcids):
            node = self.arcs.get(dcid, {})
            result[dcid] = tuple(
                dict.fromkeys(
                    value for prop in ("name", "alternateName") for value in node.get(prop, ())
                )
            )
        return result
