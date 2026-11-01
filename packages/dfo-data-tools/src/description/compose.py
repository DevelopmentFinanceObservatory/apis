"""Compose a description from a record's facts and evidence and a vocabulary's words.

`describe_version` describes a dataset version or a computed version, as the record's id
says. `describe_card` describes a card revision drawn from the record's version, from the
selection facts of what the card draws. `describe_source_card` describes a source card
revision from the documents of the sources it lists. All are pure and return JSON-ready wire data in the
kit's `description.schema.json` shape: five blocks (`about`, `sources`, `freshness`, `steps`,
`cite`), then `caveats`, `gaps` and `built_from`.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dfo_frame_grammar import FACET_IDENTITY_COLUMNS, STAT_VAR_COLUMN
from pydantic.experimental.missing_sentinel import MISSING

from code_index.labels import (
    HOW_CONNECTIVES,
    connective_for_phrase,
    how_value_text,
    role_class_text,
)
from description import blocks
from description.blocks import Wire, Words
from description.citation import cite, join_names
from description.facts import (
    CaveatFact,
    DescriptionRecord,
    SelectionFacts,
    Span,
)
from description.steps import (
    MAX_CARD_STEPS,
    MAX_SERVED_STEPS,
    ComposedUnit,
    Step,
    Unit,
    VariableSide,
    VersionRef,
)
from description.steps_text import describe_steps, name_computed
from description.stored import clipped
from description.vocabulary import Vocabulary
from dfo_platform.roles import axis_for_role

SELECT_OP = "select"
"""The kit's operation for what a card draws from its version."""

SELECT_POLICY = "card_selection"
"""The one policy of `SELECT_OP` in the kit's operation list."""

_NON_ENTITY_COLUMNS = frozenset({STAT_VAR_COLUMN, *FACET_IDENTITY_COLUMNS})
"""The columns of a drawn series whose values name no entity."""

CHANGE_OP = "change"
"""The kit's operation for a stat card's change between its two basis periods."""

CHANGE_POLICY = "change_between_basis_periods"
"""The one policy of `CHANGE_OP` in the kit's operation list."""

_PLACE_PROPERTY = "observationAbout"
"""The property naming the place a variable without flow roles is about."""

SOURCE_STEP_OP = "source_step"
"""The kit's operation for one step a source's pipeline applied."""

SOURCE_STEP_POLICY = "source_pipeline_as_recorded"
"""The one policy of `SOURCE_STEP_OP` in the kit's operation list."""


def describe_version(
    record: DescriptionRecord,
    vocabulary: Vocabulary,
    *,
    inputs: Sequence[DescriptionRecord] = (),
) -> dict[str, Any]:
    """The description of a dataset version (`spec_`) or a computed version (`ops_`).

    A computed version credits the sources its inputs' records credit
    (`blocks.version_credits`). Its `sources` list those that fit beside its inputs
    (`blocks.computed_sources`) ahead of the inputs, and its citation line names them.
    Without them, its citation line names its inputs.

    Args:
        record: the version's facts and evidence.
        vocabulary: today's words.
        inputs: for a computed version, the records of the versions it was computed from
            that are held; ignored for a dataset version.

    Returns:
        The description as JSON-ready wire data.
    """
    words = Words.for_record(record, vocabulary)
    if record.computed is not None:
        describes = {
            "kind": "computed_version",
            "id": record.id,
            "content_version": record.content_version,
        }
        served = blocks.computed_sources(
            blocks.version_credits(record, inputs), record.computed.inputs, words
        )
        sources, credited = served.entries, served.cited
        about, freshness = _computed_blocks(record, words)
    else:
        credited = None
        describes = {
            "kind": "dataset_version",
            "id": record.id,
            "content_version": record.content_version,
        }
        about, sources, freshness = _dataset_blocks(record, words)
    title, composed = _version_title(record, words)
    numbered = _numbered(record)
    steps = _served_steps(record, words, numbered)
    return {
        "describes": describes,
        "title": title,
        "about": about,
        "sources": sources,
        "freshness": freshness,
        "steps": steps,
        "caveats": _caveats(record, words, numbered),
        "cite": cite(record, words, title=title, composed=composed, sources=credited),
        "gaps": words.gaps,
        "built_from": _built_from(record, vocabulary),
    }


def describe_card(
    record: DescriptionRecord,
    selection: SelectionFacts,
    vocabulary: Vocabulary,
    *,
    component_id: str,
    revision: int,
    frame: VersionRef | None = None,
    steps: Sequence[Step] | None = None,
    caveats: Sequence[CaveatFact] = (),
    inputs: Sequence[DescriptionRecord] = (),
) -> dict[str, Any]:
    """The description of a card revision drawn from `record`'s version.

    A card drawn from a computed version credits the sources of the inputs the series it
    draws were computed from (`blocks.drawn_credits`), each traced from its variable back
    through the steps that wrote it. Its `sources` list them ahead of those inputs, and its
    citation line names them.

    Args:
        record: the facts and evidence of the version the card draws from.
        selection: what the card draws.
        vocabulary: today's words.
        component_id: the card's id.
        revision: the card's revision.
        frame: the frame a live render drew on. Given, the description carries
            `rendering: "live"` and the frame.
        steps: the card's own steps, such as a stat card's change (`change_step`), served
            after its selection. At most `MAX_CARD_STEPS` less the selection.
        caveats: facts about the card's selection, such as the series a rolling card leaves
            out, worded as the version's caveats are and served ahead of them.
        inputs: for a computed version, the records of the versions it was computed from
            that are held; ignored for a dataset version.

    Returns:
        The description as JSON-ready wire data.

    Raises:
        ValueError: `steps` and the selection are more than `MAX_CARD_STEPS`.
    """
    if 1 + len(steps or ()) > MAX_CARD_STEPS:
        raise ValueError(f"a card serves at most {MAX_CARD_STEPS} steps after its version's")
    words = Words.for_record(record, vocabulary)
    version_title, composed = _version_title(record, words)
    title = clipped(f"{version_title}, {_window(selection.window)}")
    selected = _selection_step(record, selection, label=version_title)
    numbered = _numbered(record, card=[selected, *(steps or [])])
    served = _served_steps(record, words, numbered)
    about = {
        **_card_measure(record, words),
        "drawn": blocks.drawn(record, selection, words),
        "unit": blocks.drawn_unit(selection, words),
    }
    if codes := blocks.source_codes(record, selection.series, vocabulary):
        about["source_codes"] = codes
    drawn_sources = blocks.drawn_sources(record, selection, words, inputs=inputs)
    description: dict[str, Any] = {
        "describes": {"kind": "card_revision", "component_id": component_id, "revision": revision},
        "title": title,
        "about": about,
        "sources": drawn_sources.entries,
        "freshness": blocks.drawn_freshness(selection, words),
        "steps": served,
        "caveats": _caveats(record, words, numbered, card=caveats, selection=selection),
        "cite": {
            "component_id": component_id,
            "revision": revision,
            **cite(
                record,
                words,
                title=version_title,
                composed=composed,
                sources=drawn_sources.cited if record.computed is not None else None,
                steps=numbered.card,
            ),
        },
        "gaps": words.gaps,
        "built_from": _built_from(record, vocabulary),
    }
    return _rendered(description, frame)


def describe_source_card(
    record: DescriptionRecord,
    vocabulary: Vocabulary,
    *,
    sources: Sequence[str],
    component_id: str,
    revision: int,
    frame: VersionRef | None = None,
) -> dict[str, Any]:
    """The description of a source card revision, composed from its sources' documents.

    Its title and `about` name the sources, clipped to `TEXT_MAX_LENGTH`. `sources`
    holds each one's full name, document fields and the series the version holds from it,
    `freshness` when each was downloaded and its next release, `steps` and `caveats` are
    empty, and `cite` names the card and the sources.

    Args:
        record: the facts and evidence of the version the card lists the sources of.
        vocabulary: today's words.
        sources: the dcids of the sources the card lists, in its order.
        component_id: the card's id.
        revision: the card's revision.
        frame: the frame a live render drew on. Given, the description carries
            `rendering: "live"` and the frame.

    Returns:
        The description as JSON-ready wire data.
    """
    words = Words.for_record(record, vocabulary)
    evidence = blocks.recorded_sources(record, sources)
    named = clipped(join_names([blocks.source_name(source, words) for source in evidence]))
    description: dict[str, Any] = {
        "describes": {"kind": "card_revision", "component_id": component_id, "revision": revision},
        "title": named,
        "about": {"name": named},
        "sources": blocks.card_sources(record, evidence, words),
        "freshness": blocks.source_downloads(evidence, words).served(words),
        "steps": [],
        "caveats": [],
        "cite": {
            "component_id": component_id,
            "revision": revision,
            **cite(record, words, title=named, composed=False, sources=evidence),
        },
        "gaps": words.gaps,
        "built_from": _built_from(record, vocabulary),
    }
    return _rendered(description, frame)


def change_step(*, derivation: str, basis: Sequence[str]) -> Step:
    """A stat card's change as the step after its selection: its `derivation` between the two
    `basis` periods."""
    return Step(
        id=CHANGE_OP,
        op=CHANGE_OP,
        input=SELECT_OP,
        policy=CHANGE_POLICY,
        params={"derivation": derivation, "basis": list(basis)},
    )


def _rendered(description: dict[str, Any], frame: VersionRef | None) -> dict[str, Any]:
    """`description`, marked `rendering: "live"` with the frame a live render drew on."""
    if frame is not None:
        description["rendering"] = "live"
        description["frame"] = {"id": frame.id, "content_version": frame.content_version}
    return description


def _dataset_blocks(record: DescriptionRecord, words: Words) -> tuple[Wire, list[Wire], Wire]:
    variables = record.variables
    assert variables is not None, "a dataset version's record holds its variables"
    about = {
        "measure": blocks.measure(
            variables, words, named=blocks.defined_measure(variables, words.vocabulary)
        ),
        "fixed": blocks.fixed_values(variables, words),
        "varies": blocks.open_dimensions(record, variables, words),
        "variables": blocks.variable_counts(variables, words),
        "series": words.known(record.series.series, "about.series"),
        "period": blocks.period(record.series, words),
        "unit": blocks.series_unit(record.series, words),
    }
    if codes := blocks.version_source_codes(record, words.vocabulary):
        about["source_codes"] = codes
    return (
        about,
        blocks.dataset_sources(record, words),
        blocks.dataset_freshness(record, words),
    )


def _card_measure(record: DescriptionRecord, words: Words) -> Wire:
    """A card's `measure` and `fixed`: its version's measure, whatever subset the card draws,
    and the values the version's definition fixes except those the measure names
    (`blocks.named_by_measure`). A card drawn from a computed version carries neither."""
    variables = record.variables
    if variables is None:
        return {}
    named = blocks.defined_measure(variables, words.vocabulary)
    return {
        "measure": blocks.measure(variables, words, named=named),
        "fixed": blocks.fixed_values(variables, words, named=blocks.named_by_measure(named)),
    }


def _computed_blocks(record: DescriptionRecord, words: Words) -> tuple[Wire, Wire]:
    computed = record.computed
    assert computed is not None, "a computed version's record holds what it was computed from"
    about: Wire = {
        "name": words.known(_computed_name(record, words.vocabulary), "about.name"),
        "series": words.known(record.series.series, "about.series"),
        "period": blocks.period(record.series, words),
        "unit": _computed_unit(record, words),
    }
    inputs = [
        words.input_ref(version, f"freshness.inputs.{n}")
        for n, version in enumerate(computed.inputs)
    ]
    freshness = {
        "computed": record.evidence.prepared.isoformat(),
        "inputs": inputs,
        **blocks.latest(record.series, words),
        **blocks.status(record.series.status, words),
    }
    return about, freshness


def _computed_name(record: DescriptionRecord, vocabulary: Vocabulary) -> str | None:
    if not record.steps:
        return None
    name = name_computed(record.steps, vocabulary)
    return _capitalised(name) if name is not None else None


def _computed_unit(record: DescriptionRecord, words: Words) -> Wire | None:
    """The unit every series holds, read from the version's frame. The last step produces
    the frame, so a unit it states must be that unit: one that differs leaves the unit a gap.
    An earlier step's unit is its own output's, and says nothing of the frame's."""
    steps = record.steps or []
    stated = steps[-1].unit if steps else MISSING
    held = record.series.by_source_unit
    if (
        stated is not MISSING
        and held is not None
        and {_unit_identity(count.unit) for count in held} != {_unit_identity(stated)}
    ):
        words.gap("about.unit")
        return None
    return blocks.series_unit(record.series, words)


def _unit_identity(unit: Unit | VariableSide) -> object:
    """What a unit is, without the words for it: two units are one when these are equal."""
    if unit is None:
        return None
    if isinstance(unit, ComposedUnit):
        return (_unit_identity(unit.numerator), _unit_identity(unit.denominator))
    return tuple(sorted(unit.model_dump(exclude={"label", "at_mint"}).items()))


def version_title(record: DescriptionRecord, vocabulary: Vocabulary) -> str | None:
    """The title `describe_version` gives `record` in `vocabulary`'s words, else the title
    composed at mint, else `None`. A computed version's description names each input by this."""
    title, composed = _version_title(record, Words.for_record(record, vocabulary))
    return title if composed else record.evidence.title


def _version_title(record: DescriptionRecord, words: Words) -> tuple[str, bool]:
    """The version's title in today's words, and whether today's words composed it. A dataset
    version's names its measure, each party and code the definition fixes, and each code it
    narrows a dimension to; a computed version's is the name its steps derive, clipped to a
    served text's 512 characters. Without those words, the title composed at mint stands,
    with a gap."""
    composed = (
        _computed_name(record, words.vocabulary)
        if record.computed is not None
        else _dataset_title(record, words)
    )
    if composed is not None:
        return clipped(composed), True
    words.gap("title")
    return record.evidence.title or record.id, False


def _dataset_title(record: DescriptionRecord, words: Words) -> str | None:
    """The measure, then each code the definition fixes or narrows a dimension to, in its
    order, worded as Search's heading words a how value. A definition no measure label or
    noun covers that resolved one variable leads with that variable's label ("Total
    population"). A party follows its role's preposition ("to Nigeria") and the place a
    variable is about follows "in"; a code follows its property's connective, by its short
    name where the code index gives one ("for health", "as standard grants"); a code whose
    property modifies the noun directly goes before the measure. The codes of a set join as a
    list ("for health and education")."""
    variables = record.variables
    if variables is None:
        return None
    vocabulary = words.vocabulary
    if (measure := blocks.defined_measure(variables, vocabulary)) is not None:
        lead, stated = measure.label, measure.stated
    elif variables.variable is not None and (named := words.name(variables.variable)):
        lead, stated = named, frozenset[str]()
    else:
        return None
    wording = vocabulary.roles(variables.family)
    codes = [
        (fixed.property, [fixed.value]) for fixed in variables.fixed if fixed.property not in stated
    ] + [
        (dimension.property, dimension.narrowed_to)
        for dimension in variables.dimensions or []
        if dimension.narrowed_to
    ]
    modifiers: list[str] = []
    phrases = [lead]
    for prop, refs in codes:
        labels = [label for code in refs if (label := words.name(code)) is not None]
        if len(labels) < len(refs):
            return None
        named = join_names(labels)
        if (axis := axis_for_role(prop)) is not None:
            phrases.append(f"{wording.noun(axis).preposition} {named}")
        elif prop == _PLACE_PROPERTY:
            phrases.append(f"in {named}")
        elif HOW_CONNECTIVES.get(prop) == "":
            modifiers.append(named)
        else:
            text = role_class_text(
                join_names(
                    [
                        how_value_text(
                            code,
                            label,
                            prop=prop,
                            property_label=vocabulary.label(prop),
                            index=vocabulary.index,
                        )
                        for code, label in zip(refs, labels, strict=True)
                    ]
                ),
                prop,
                wording,
            )
            phrases.append(f"{connective_for_phrase(prop, text, wording)}{text}".strip())
    return _capitalised(" ".join([*modifiers, *phrases]))


def _selection_step(record: DescriptionRecord, selection: SelectionFacts, *, label: str) -> Step:
    """The card's selection as a step: the coordinates and window it draws from its version,
    and how many series it keeps."""
    coordinates: dict[str, set[str]] = {}
    for series in selection.series:
        for coordinate in series.coordinates:
            if coordinate.value is not None:
                coordinates.setdefault(coordinate.column, set()).add(coordinate.value)
    return Step(
        id=SELECT_OP,
        op=SELECT_OP,
        input=VersionRef(id=record.id, content_version=record.content_version, label=label),
        policy=SELECT_POLICY,
        params={
            "coordinates": {column: sorted(values) for column, values in coordinates.items()},
            "window": {"start": selection.window.start, "end": selection.window.end},
        },
        kept={"series": len(selection.series)},
    )


@dataclass(frozen=True, slots=True)
class _Numbered:
    """A version's own steps and a card's under the ids a description serves them by, and
    the map from each list's own ids to those."""

    version: list[Step]
    card: list[Step]
    version_ids: Mapping[str, str]
    card_ids: Mapping[str, str]


def _numbered(record: DescriptionRecord, *, card: Sequence[Step] = ()) -> _Numbered:
    """The version's own steps, then the card's, numbered `s1`, `s2` in that order. Each list
    names its steps in its own ids, so a card's `select` and a version step of that id serve
    apart, and the version's steps serve under the same ids in its description and in each
    card's. The source pipelines' steps are numbered `p1`, `p2` apart from these."""
    version, version_ids = _renumbered(record.known_steps, first=1)
    card_steps, card_ids = _renumbered(card, first=len(version) + 1)
    return _Numbered(version=version, card=card_steps, version_ids=version_ids, card_ids=card_ids)


def _renumbered(steps: Sequence[Step], *, first: int) -> tuple[list[Step], dict[str, str]]:
    """`steps` under the ids `s<first>`, `s<first + 1>` and on, each input that names a step
    renamed to match, and the map from each step's own id to its new one. Each input names
    an earlier step, as the publish door and the mint ensure."""
    ids: dict[str, str] = {}
    renumbered = []
    for n, step in enumerate(steps, start=first):
        update: dict[str, Any] = {"id": f"s{n}"}
        if isinstance(step.input, str):
            update["input"] = ids[step.input]
        if step.inputs is not MISSING:
            update["inputs"] = {
                role: ids[named] if isinstance(named, str) else named
                for role, named in step.inputs.items()
            }
        renumbered.append(step.model_copy(update=update))
        ids[step.id] = update["id"]
    return renumbered, ids


def _served_steps(
    record: DescriptionRecord, words: Words, numbered: _Numbered
) -> list[dict[str, Any]] | None:
    """Each step as served, with its sentence: the source pipelines' steps, then the
    version's own, then a card's, under the ids `_numbered` gives them. `None` when none are
    held, which `describe_steps` names as a gap.

    A source whose pipeline steps the mint did not read leaves `steps.source_pipeline` as a
    gap. So do pipeline steps that would leave a card drawn from the version, with its
    `MAX_CARD_STEPS`, more than the kit's `MAX_SERVED_STEPS`: then no pipeline step is
    served, in the version's description or any card's, and the version's and card's steps
    are served whole. A version whose own steps are unknown, beside steps that are served,
    leaves `steps.version` as a gap."""
    pipeline = _source_steps(record)
    if len(pipeline) + len(numbered.version) + MAX_CARD_STEPS > MAX_SERVED_STEPS:
        pipeline = []
        words.gap("steps.source_pipeline")
    elif any(source.pipeline_steps is None for source in record.evidence.sources):
        words.gap("steps.source_pipeline")
    steps = [*pipeline, *numbered.version, *numbered.card]
    if record.steps is None and steps:
        words.gap("steps.version")
    described = describe_steps(
        steps if steps or record.steps is not None else None,
        words.vocabulary,
        grain=record.series.grain,
        family=words.family,
    )
    for gap in described.gaps:
        words.gap(gap)
    return described.served


def _source_steps(record: DescriptionRecord) -> list[Step]:
    """One step per step a source's pipeline applied, as the mint read it from the source's
    document, numbered `p1`, `p2` across the sources in their order."""
    recorded = [
        (source.ref, step)
        for source in record.evidence.sources
        for step in source.pipeline_steps or []
    ]
    return [
        Step(
            id=f"p{n}",
            op=SOURCE_STEP_OP,
            policy=SOURCE_STEP_POLICY,
            params={"source": ref, "title": step.title, "description": step.description},
        )
        for n, (ref, step) in enumerate(recorded, start=1)
    ]


def _caveats(
    record: DescriptionRecord,
    words: Words,
    numbered: _Numbered,
    *,
    card: Sequence[CaveatFact] = (),
    selection: SelectionFacts | None = None,
) -> list[Wire]:
    """The caveats of a card's selection, then the version's, worded over the version's steps
    and the card's, each naming its step by the id `_numbered` serves it under. Given what a
    card draws, each version caveat is narrowed to it (`_drawn_caveat`). The card draws the
    version whole when, for every role the version counts entities in, it draws as many
    distinct entities as the version holds."""
    if record.caveats is None:
        words.gap("caveats")
    version = record.caveats or []
    if selection is not None:
        drawn: dict[str, set[str]] = {}
        for series in selection.series:
            for coordinate in series.coordinates:
                if coordinate.value is not None:
                    drawn.setdefault(coordinate.column, set()).add(coordinate.value)
        variables = drawn.get(STAT_VAR_COLUMN, set())
        entities = {
            value
            for column, values in drawn.items()
            if column not in _NON_ENTITY_COLUMNS
            for value in values
        }
        roles = record.series.entities_by_role or []
        whole = bool(roles) and all(
            len(drawn.get(role.role, ())) == role.entities for role in roles
        )
        version = [
            narrowed
            for fact in version
            if (
                narrowed := _drawn_caveat(fact, variables=variables, entities=entities, whole=whole)
            )
            is not None
        ]
    return blocks.caveats(
        [*_targeted(card, numbered.card_ids), *_targeted(version, numbered.version_ids)],
        words,
        steps=[*numbered.version, *numbered.card],
        grain=record.series.grain,
    )


def _targeted(facts: Sequence[CaveatFact], ids: Mapping[str, str]) -> list[CaveatFact]:
    """`facts`, each naming its step by the id `ids` maps it to. A record's caveat names one
    of its steps, and a card's caveat one of the card's."""
    return [
        fact if fact.step is None else fact.model_copy(update={"step": ids[fact.step]})
        for fact in facts
    ]


def _drawn_caveat(
    fact: CaveatFact, *, variables: Collection[str], entities: Collection[str], whole: bool
) -> CaveatFact | None:
    """A version caveat as it applies to a card that draws `variables` and `entities`: the
    series and entities it names that the card draws, and no `count`, which counts across the
    whole version. `None` when it names series or entities and the card draws none of them.

    A card that draws every entity the version holds in each of its roles (`whole`) keeps a
    coverage gap's entities and count as the version serves them. Those places hold no data,
    so no card draws them, and the gap is about the whole of a role such a card draws. Such a
    gap names at most one series, and is left out when the card does not draw it."""
    series = [ref for ref in fact.series if ref in variables]
    if whole and fact.kind == "coverage_gap" and fact.entities:
        return None if fact.series and not series else fact
    named = [ref for ref in fact.entities if ref in entities]
    if (fact.series and not series) or (fact.entities and not named):
        return None
    return fact.model_copy(update={"series": series, "entities": named, "count": None})


def _built_from(record: DescriptionRecord, vocabulary: Vocabulary) -> Wire:
    return {"facts_schema": record.schema_version, "vocabulary": vocabulary.revision}


def _window(window: Span) -> str:
    if window.start == window.end:
        return window.start
    return f"{window.start} to {window.end}"


def _capitalised(text: str) -> str:
    return text[:1].upper() + text[1:]
