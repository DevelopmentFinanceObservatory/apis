"""Checks that the renderer reproduces every sentence plan the recorded sentence panel holds.

`tests/fixtures/sentence_panel/<slice>.jsonl.gz` holds the panel's responses trimmed to what
the renderer reads (`scripts/sentence_golden.py`, whose `--rerender` rewrites the recorded
plans after a renderer change), each Spec with the sentence facts the engine holds off the
wire (`golden_parts`). Each query re-renders, with no graph and no LLM, the plan
of every definite and candidate Spec, every candidates part's plan and the envelope plan,
and compares them with the recorded ones. A no-data part's own plan needs
inputs the wire does not carry (the request's entity terms, bound roles and qualifiers), so
the envelope plan covers it.

The same plans state each edit target of a part at most once (`SentenceToken.states`).
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from scripts.sentence_golden import golden_parts, rerendered_candidates
from search.heading.words import content_words
from search.models import (
    SOURCE_REF,
    UNIT_REF,
    CandidatesPart,
    DefinitePart,
    SentenceToken,
    Spec,
    role_ref,
)
from search.models.response import Part
from search.render.candidates import candidates_sentence_plan
from search.render.envelope import compose_envelope_sentence
from search.render.heading import declared_axis_names, heading_for, sibling_how_properties
from search.render.labels import MEASURE_AXIS_REF_PREFIX
from search.render.plan import render_sentence_plan, sentence_from_plan
from search.render.tokens import plan_words

GOLDEN_DIR = Path(__file__).parent.parent / "fixtures" / "sentence_panel"

_PLAN = TypeAdapter(list[SentenceToken])


def _golden_lines() -> list[dict[str, Any]]:
    lines = [
        json.loads(text)
        for path in sorted(GOLDEN_DIR.glob("*.jsonl.gz"))
        for text in gzip.decompress(path.read_bytes()).splitlines()
    ]
    assert lines, f"no golden lines under {GOLDEN_DIR}"
    return lines


def _assert_plan(
    rendered: list[SentenceToken], recorded: list[SentenceToken] | None, where: str
) -> None:
    """The sentence first, for a readable failure, then every token."""
    recorded = recorded or []
    assert sentence_from_plan(rendered) == sentence_from_plan(recorded), where
    assert rendered == recorded, where


@pytest.mark.parametrize("line", _golden_lines(), ids=lambda line: line["id"])
def test_renderer_reproduces_the_recorded_panel(line: dict[str, Any]) -> None:
    parts = golden_parts(line["parts"])
    declarations = declared_axis_names()
    for index, part in enumerate(parts):
        if isinstance(part, DefinitePart):
            _assert_plan(
                render_sentence_plan(part.spec, axis_declarations=declarations),
                part.spec.resolved.sentence_plan,
                f"part {index} spec",
            )
        elif isinstance(part, CandidatesPart):
            specs = part.candidates.specs
            for position, (rendered, spec) in enumerate(
                zip(rerendered_candidates(specs), specs, strict=True)
            ):
                _assert_plan(
                    rendered, spec.resolved.sentence_plan, f"part {index} candidate {position}"
                )
            _assert_plan(
                candidates_sentence_plan(
                    specs, truncated=part.candidates.truncated, departures=part.departures
                ),
                part.sentence_plan,
                f"part {index} candidates",
            )

    composed = compose_envelope_sentence(parts)
    recorded = line["sentence_plan"]
    if recorded is None:
        assert composed is None
        return
    assert composed is not None
    sentence, plan = composed
    assert sentence == line["rendered_sentence"]
    _assert_plan(plan, _PLAN.validate_python(recorded), "envelope")


@pytest.mark.parametrize("line", _golden_lines(), ids=lambda line: line["id"])
def test_plan_words_are_the_recorded_sentences_content_words(line: dict[str, Any]) -> None:
    """The words a departure is checked against are those of the heading and byline the
    reader sees, connectives and possessive forms included."""
    for part in golden_parts(line["parts"]):
        if isinstance(part, DefinitePart):
            shown, keep = [part.spec], frozenset[str]()
        elif isinstance(part, CandidatesPart):
            shown = part.candidates.specs
            keep = sibling_how_properties(shown)
        else:
            continue
        for spec in shown:
            recorded = sentence_from_plan(spec.resolved.sentence_plan or [])
            assert plan_words(heading_for(spec, keep_how_properties=keep)) == content_words(
                recorded
            ), recorded


def _edit_targets(spec: Spec) -> set[str]:
    """The refs a client edits on a Spec: every bound slot, every unbound slot with
    options, the when slot, the byline's unit and source, and the flow roles. A flow
    family edits both roles, and another answer each role it names or counts. The
    destination is edited through the where slot's token, as
    `render.labels._entity_token` keys it."""
    resolved = spec.resolved
    bound = {slot.key.as_ref() for slot in resolved.slots if slot.binding.kind != "unbound"}
    unbound = {slot.key.as_ref() for slot in resolved.slots if slot.binding.kind == "unbound"}
    with_options = {
        entry.key.as_ref()
        for entry in resolved.slot_options
        if entry.options and entry.key.as_ref() in unbound
    }
    when = {slot.key.as_ref() for slot in resolved.slots if slot.key.axis == "when"}
    roles = {entity.role for entity in resolved.entities}
    roles |= {fact.role for fact in resolved.role_axis_facts or ()}
    if resolved.scope is not None:
        roles.add(resolved.scope.axis)
    if heading_for(spec).heading.family is not None:
        roles |= {"origin", "destination"}
    where = next((slot.key.as_ref() for slot in resolved.slots if slot.key.axis == "where"), None)
    role_refs = {
        where if role == "destination" and where is not None else role_ref(role)
        for role in roles & {"origin", "destination"}
    }
    return bound | with_options | when | role_refs | {UNIT_REF, SOURCE_REF}


def _folded_into(parts: list[Part]) -> dict[int, int]:
    """Each folded part's index mapped to the earlier definite part whose Spec it carries,
    as `render.envelope` reads a `binding_folded` part."""
    first: dict[str, int] = {}
    folds: dict[int, int] = {}
    for index, part in enumerate(parts):
        if not isinstance(part, DefinitePart):
            continue
        spec_id = part.spec.spec_id
        if any(warning.code == "binding_folded" for warning in part.warnings):
            folds[index] = first[spec_id]
        else:
            first.setdefault(spec_id, index)
    return folds


@pytest.mark.parametrize("line", _golden_lines(), ids=lambda line: line["id"])
def test_each_edit_target_is_stated_by_at_most_one_token(line: dict[str, Any]) -> None:
    """On a definite part's own plan and on the envelope plan, no ref is stated by two
    tokens with text, a token without text states nothing, and every stated ref is an edit
    target of the part or a measure axis no slot carries (`axis:`).

    On the envelope a token with no part index stands for every part, so it counts for
    each. A folded part is stated by the tokens of the part it folded into.
    """
    parts = golden_parts(line["parts"])
    declarations = declared_axis_names()
    composed = compose_envelope_sentence(parts)
    envelope = composed[1] if composed is not None else []
    folds = _folded_into(parts)
    for index, part in enumerate(parts):
        if not isinstance(part, DefinitePart):
            continue
        targets = _edit_targets(part.spec)
        owner = folds.get(index, index)
        plans = {
            "spec": render_sentence_plan(part.spec, axis_declarations=declarations),
            "envelope": [token for token in envelope if token.part_index in (owner, None)],
        }
        for where, plan in plans.items():
            assert not [token for token in plan if token.states and not token.text], where
            counts = Counter(ref for token in plan if token.text for ref in token.states)
            assert [ref for ref, count in counts.items() if count > 1] == [], (where, index)
            strays = [
                ref
                for ref in counts
                if ref not in targets and not ref.startswith(MEASURE_AXIS_REF_PREFIX)
            ]
            assert strays == [], (where, index)


@pytest.mark.parametrize(
    ("query_id", "measure_states", "refine"),
    [
        pytest.param(
            "crs-001",
            ["how:crsTypeOfFlow", "how:financialFlowType", "how:nettingBasis"],
            {"how:crsCooperationModality", "how:crsPurpose", "how:crsTypeOfFinance"},
            id="label-states-flow-and-measure-axes",
        ),
        pytest.param(
            "crs-007",
            ["how:crsCooperationModality", "how:crsTypeOfFinance"],
            {"how:crsPurpose"},
            id="label-states-agreed-open-axes",
        ),
        pytest.param(
            "ids-008", ["how:financialFlowType"], {"how:idsDebtorType"}, id="authored-noun"
        ),
    ],
)
def test_measure_token_states_what_its_noun_silences(
    query_id: str, measure_states: list[str], refine: set[str]
) -> None:
    """The measure token states the slots its noun leaves silent, and an open slot whose
    members vary stays in Refine."""
    line = next(line for line in _golden_lines() if line["id"] == query_id)
    (part,) = golden_parts(line["parts"])
    assert isinstance(part, DefinitePart)
    plan = render_sentence_plan(part.spec, axis_declarations=declared_axis_names())
    (noun,) = [token for token in plan if token.axis == "what" and token.text]
    assert [ref for ref in noun.states if ref in set(measure_states)] == measure_states
    stated = {ref for token in plan if token.text for ref in token.states}
    assert refine <= _edit_targets(part.spec) - stated
