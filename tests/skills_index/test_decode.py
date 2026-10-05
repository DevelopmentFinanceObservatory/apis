"""The `text` arc arrives from the graph JSON-escaped and reaches the index as Markdown.

`DecodedTextGraph` decodes it once at the read. The byte ceilings and the corpus
budget count the decoded bytes, and front matter is read from the decoded text.
An undecodable skill text leaves its skill out, and an undecodable reference is
dropped, each with a named issue.
"""

from __future__ import annotations

import pytest

import skills_index.build as build_module
import skills_index.decode as decode_module
from skills_index.build import build_index
from skills_index.config import (
    CONTENT_ARC,
    REFERENCE_TYPE,
    SKILL_BODY_MAX_BYTES,
    SKILL_REFERENCE_MAX_BYTES,
    UNDECODABLE_CONTENT_ARC,
)
from skills_index.decode import DecodedTextGraph, decode_mcf_text

from .conftest import FakeSkillsGraph, skill_markdown, skill_node, stored

# A body as the pipelines store it: front matter and paragraphs joined by the
# two characters `\n`, and quotes spelled as the six characters `\u0022`.
STORED_BODY = (
    "---\\n"
    "name: crs-purpose-codes\\n"
    "description: Look up \\u0022CRS\\u0022 purpose codes\\n"
    "---\\n"
    "\\n"
    "# CRS purpose codes\\n"
    "\\n"
    "Use the \\u0022purpose\\u0022 column."
)
DECODED_BODY = (
    "---\n"
    "name: crs-purpose-codes\n"
    'description: Look up "CRS" purpose codes\n'
    "---\n"
    "\n"
    "# CRS purpose codes\n"
    "\n"
    'Use the "purpose" column.'
)

# A raw quote cannot appear inside a JSON string, so the decoder refuses it.
UNDECODABLE = 'a raw "quote" in stored text'


def _roster(*skill_ids: str) -> dict[tuple[str, str], list[str]]:
    return {("AgentSkill", "typeOf"): [f"skill/{skill_id}" for skill_id in skill_ids]}


def _markdown_of_bytes(skill_id: str, size: int, *, filler: str = "x") -> str:
    """Returns a valid `SKILL.md` for `skill_id` of exactly `size` decoded bytes,
    padded with `filler`."""
    padding = size - len(skill_markdown(skill_id, body="").encode("utf-8"))
    return skill_markdown(skill_id, body=filler * padding)


def _fully_escaped(markdown: str) -> str:
    """Spells every character of ASCII `markdown` as the six characters `\\uXXXX`,
    the longest stored form JSON gives one byte."""
    return "".join(f"\\u{ord(character):04x}" for character in markdown)


def test_stored_body_fixture_is_the_pipeline_encoding_of_the_decoded_body():
    assert stored(DECODED_BODY) == STORED_BODY


def test_live_shaped_escaped_body_is_served_as_markdown():
    graph = FakeSkillsGraph(
        subjects=_roster("crs-purpose-codes"),
        arcs={"skill/crs-purpose-codes": {"text": (STORED_BODY,)}},
    )

    index = build_index(graph=graph)

    content = index.skills["crs-purpose-codes"].content_text
    assert content == DECODED_BODY
    assert content is not None
    assert "\\n" not in content
    assert "\\u0022" not in content
    assert not index.issues


def test_resource_content_is_served_as_markdown():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={
            "skill/skill-a": skill_node("skill-a")
            | {"agentSkillReferences": ("skill/skill-a/notes",)},
            "skill/skill-a/notes": {
                "name": ("references/notes.md",),
                "text": (stored('# Notes\n\nSay "hi".'),),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    (resource,) = index.skills["skill-a"].files[1:]
    assert resource.content == '# Notes\n\nSay "hi".'


def test_front_matter_is_read_from_the_decoded_text():
    graph = FakeSkillsGraph(
        subjects=_roster("crs-purpose-codes"),
        arcs={
            "skill/crs-purpose-codes": {
                "text": (STORED_BODY,),
                "name": ("Say \\u0022hi\\u0022",),
                "description": ("Line one\\nline two",),
            }
        },
    )

    index = build_index(graph=graph)

    entry = index.skills["crs-purpose-codes"]
    assert entry.front_matter.name == "crs-purpose-codes"
    assert entry.front_matter.description == 'Look up "CRS" purpose codes'
    assert entry.frontmatter == {
        "name": "crs-purpose-codes",
        "description": 'Look up "CRS" purpose codes',
    }


def test_undecodable_text_leaves_the_skill_out_and_its_references_unread():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={
            "skill/skill-a": {
                "text": (UNDECODABLE,),
                "agentSkillReferences": ("skill/skill-a/notes",),
            },
            "skill/skill-a/notes": {
                "name": ("references/notes.md",),
                "text": ("Notes.",),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    assert index.skills == {}
    (issue,) = index.issues
    assert issue.code == "skill_content_undecodable"
    assert issue.severity == "warning"
    assert issue.skill_id == "skill/skill-a"
    assert "not decodable" in issue.message
    assert all("skill/skill-a/notes" not in requested for requested, _ in graph.arc_calls)


def test_undecodable_resource_is_rejected_while_its_skill_and_siblings_still_serve():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={
            "skill/skill-a": skill_node("skill-a")
            | {"agentSkillReferences": ("skill/skill-a/bad", "skill/skill-a/good")},
            "skill/skill-a/bad": {
                "name": ("references/bad.md",),
                "text": (UNDECODABLE,),
                "typeOf": (REFERENCE_TYPE,),
            },
            "skill/skill-a/good": {
                "name": ("references/good.md",),
                "text": ("Good.",),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    entry = index.skills["skill-a"]
    assert entry.content_text == skill_markdown("skill-a")
    assert [resource.path for resource in entry.files[1:]] == ["references/good.md"]
    (issue,) = index.issues
    assert issue.code == "skill_resource_rejected"
    assert issue.severity == "rejection"
    assert "bad.md" in issue.message
    assert "not decodable" in issue.message


def test_body_ceiling_counts_decoded_bytes():
    # Each stored `\n` is two characters and decodes to one byte.
    at_ceiling = _markdown_of_bytes("skill-at", SKILL_BODY_MAX_BYTES, filler="\n")
    past_ceiling = _markdown_of_bytes("skill-past", SKILL_BODY_MAX_BYTES + 1, filler="\n")
    graph = FakeSkillsGraph(
        subjects=_roster("skill-at", "skill-past"),
        arcs={
            "skill/skill-at": {"text": (stored(at_ceiling),)},
            "skill/skill-past": {"text": (stored(past_ceiling),)},
        },
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-at"}
    assert index.skills["skill-at"].content_text == at_ceiling
    assert [(issue.skill_id, issue.code) for issue in index.issues] == [
        ("skill/skill-past", "skill_content_too_large")
    ]


def test_resource_ceiling_counts_decoded_bytes():
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={
            "skill/skill-a": skill_node("skill-a")
            | {
                "agentSkillReferences": ("skill/skill-a/at", "skill/skill-a/past"),
            },
            "skill/skill-a/at": {
                "name": ("references/at.md",),
                "text": ("\\n" * SKILL_REFERENCE_MAX_BYTES,),
                "typeOf": (REFERENCE_TYPE,),
            },
            "skill/skill-a/past": {
                "name": ("references/past.md",),
                "text": ("\\n" * (SKILL_REFERENCE_MAX_BYTES + 1),),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    assert [resource.path for resource in index.skills["skill-a"].files[1:]] == ["references/at.md"]
    (issue,) = index.issues
    assert issue.code == "skill_resource_rejected"
    assert "past.md" in issue.message


@pytest.fixture
def decoded_lengths(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Records the stored length of every `text` value the decoder receives."""
    lengths: list[int] = []

    def spy(value: str) -> str:
        lengths.append(len(value))
        return decode_mcf_text(value)

    monkeypatch.setattr(decode_module, "decode_mcf_text", spy)
    return lengths


def _escaped_a(decoded_bytes: int) -> str:
    """A stored value at its escaped-form ceiling: every decoded byte is the
    six characters `\\u0041`, the longest spelling JSON gives one byte."""
    return "\\u0041" * decoded_bytes


def test_body_at_the_escaped_ceiling_is_decoded_and_served(decoded_lengths: list[int]):
    markdown = _markdown_of_bytes("skill-a", SKILL_BODY_MAX_BYTES)
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={"skill/skill-a": {"text": (_fully_escaped(markdown),)}},
    )

    index = build_index(graph=graph)

    assert decoded_lengths == [6 * SKILL_BODY_MAX_BYTES]
    assert index.skills["skill-a"].content_text == markdown
    assert not index.issues


def test_body_past_the_escaped_ceiling_is_too_large_and_never_decoded(
    decoded_lengths: list[int],
):
    stored_text = _fully_escaped(_markdown_of_bytes("skill-a", SKILL_BODY_MAX_BYTES)) + "A"
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={"skill/skill-a": {"text": (stored_text,)}},
    )

    index = build_index(graph=graph)

    assert decoded_lengths == []
    assert index.skills == {}
    (issue,) = index.issues
    assert issue.code == "skill_content_too_large"
    assert issue.skill_id == "skill/skill-a"
    assert str(len(stored_text)) in issue.message


def test_resource_at_the_escaped_ceiling_is_decoded_and_kept(decoded_lengths: list[int]):
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={
            "skill/skill-a": skill_node("skill-a")
            | {"agentSkillReferences": ("skill/skill-a/at",)},
            "skill/skill-a/at": {
                "name": ("references/at.md",),
                "text": (_escaped_a(SKILL_REFERENCE_MAX_BYTES),),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    assert 6 * SKILL_REFERENCE_MAX_BYTES in decoded_lengths
    (resource,) = index.skills["skill-a"].files[1:]
    assert resource.content == "A" * SKILL_REFERENCE_MAX_BYTES
    assert not index.issues


def test_resource_past_the_escaped_ceiling_is_rejected_and_never_decoded(
    decoded_lengths: list[int],
):
    stored_text = _escaped_a(SKILL_REFERENCE_MAX_BYTES) + "A"
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a"),
        arcs={
            "skill/skill-a": skill_node("skill-a")
            | {"agentSkillReferences": ("skill/skill-a/past",)},
            "skill/skill-a/past": {
                "name": ("references/past.md",),
                "text": (stored_text,),
                "typeOf": (REFERENCE_TYPE,),
            },
        },
    )

    index = build_index(graph=graph)

    assert len(stored_text) not in decoded_lengths
    assert index.skills["skill-a"].content_text == skill_markdown("skill-a")
    assert index.skills["skill-a"].files[1:] == ()
    (issue,) = index.issues
    assert issue.code == "skill_resource_rejected"
    assert "past.md" in issue.message
    assert str(len(stored_text)) in issue.message


def test_corpus_budget_counts_decoded_bytes(monkeypatch):
    # The budget equals one skill's decoded text, which stores in more characters.
    # Stored counting would spend it on the first skill; decoded counting spends
    # it only after the second.
    monkeypatch.setattr(build_module, "_FRONT_MATTER_CHUNK_SIZE", 1)
    skill_ids = ("skill-a", "skill-b", "skill-c")
    texts = {skill_id: skill_markdown(skill_id, body="\n" * 40) for skill_id in skill_ids}
    decoded_bytes = len(texts["skill-a"].encode("utf-8"))
    assert len(stored(texts["skill-a"])) > decoded_bytes
    monkeypatch.setattr(build_module, "SKILLS_CORPUS_CONTENT_BUDGET_BYTES", decoded_bytes)
    graph = FakeSkillsGraph(
        subjects=_roster(*skill_ids),
        arcs={f"skill/{skill_id}": {"text": (stored(texts[skill_id]),)} for skill_id in skill_ids},
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-a", "skill-b"}
    assert index.skills["skill-b"].content_text == texts["skill-b"]
    assert [(issue.skill_id, issue.code) for issue in index.issues] == [
        ("skill/skill-c", "skills_corpus_budget_exhausted")
    ]


def test_undecodable_text_spends_no_corpus_budget(monkeypatch):
    monkeypatch.setattr(build_module, "_FRONT_MATTER_CHUNK_SIZE", 1)
    monkeypatch.setattr(build_module, "SKILLS_CORPUS_CONTENT_BUDGET_BYTES", 5)
    graph = FakeSkillsGraph(
        subjects=_roster("skill-bad", "skill-ok"),
        arcs={
            "skill/skill-bad": {"text": (UNDECODABLE * 10,)},
            "skill/skill-ok": skill_node("skill-ok"),
        },
    )

    index = build_index(graph=graph)

    assert set(index.skills) == {"skill-ok"}
    assert [issue.code for issue in index.issues] == ["skill_content_undecodable"]


def test_undecodable_text_is_named_for_its_own_rule_even_past_the_budget(monkeypatch):
    monkeypatch.setattr(build_module, "_FRONT_MATTER_CHUNK_SIZE", 1)
    monkeypatch.setattr(build_module, "SKILLS_CORPUS_CONTENT_BUDGET_BYTES", 2)
    graph = FakeSkillsGraph(
        subjects=_roster("skill-a", "skill-b"),
        arcs={
            "skill/skill-a": skill_node("skill-a"),
            "skill/skill-b": {"text": (UNDECODABLE,)},
        },
    )

    index = build_index(graph=graph)

    assert [(issue.skill_id, issue.code) for issue in index.issues] == [
        ("skill/skill-b", "skill_content_undecodable")
    ]


class TestDecodedTextGraph:
    def test_decodes_only_the_text_arc(self):
        inner = FakeSkillsGraph(
            arcs={"n": {"text": (STORED_BODY,), "name": ("a\\nb",), "url": ("u",)}}
        )

        arcs = DecodedTextGraph(inner, max_decoded_bytes=SKILL_BODY_MAX_BYTES).node_arcs_many(
            ("n",), properties=(CONTENT_ARC, "name", "url")
        )

        assert arcs == {"n": {"text": (DECODED_BODY,), "name": ("a\\nb",), "url": ("u",)}}

    def test_leaves_arcs_alone_when_text_is_not_requested(self):
        inner = FakeSkillsGraph(arcs={"n": {"text": (STORED_BODY,), "typeOf": ("T",)}})

        arcs = DecodedTextGraph(inner, max_decoded_bytes=SKILL_BODY_MAX_BYTES).node_arcs_many(
            ("n",), properties=("typeOf",)
        )

        assert arcs == {"n": {"typeOf": ("T",)}}

    def test_passes_a_missing_node_through_as_none(self):
        arcs = DecodedTextGraph(
            FakeSkillsGraph(), max_decoded_bytes=SKILL_BODY_MAX_BYTES
        ).node_arcs_many(("gone",), properties=(CONTENT_ARC,))

        assert arcs == {"gone": None}

    def test_marks_an_undecodable_node_and_empties_its_text(self):
        inner = FakeSkillsGraph(arcs={"n": {"text": (UNDECODABLE,)}})

        arcs = DecodedTextGraph(inner, max_decoded_bytes=SKILL_BODY_MAX_BYTES).node_arcs_many(
            ("n",), properties=(CONTENT_ARC,)
        )

        node = arcs["n"]
        assert node is not None
        assert node[CONTENT_ARC] == ()
        assert len(node[UNDECODABLE_CONTENT_ARC]) == 1

    @pytest.mark.parametrize("bad_value", [UNDECODABLE, "trailing backslash \\", "bad \\q escape"])
    def test_one_bad_value_among_several_marks_the_node(self, bad_value):
        inner = FakeSkillsGraph(arcs={"n": {"text": ("fine", bad_value)}})

        arcs = DecodedTextGraph(inner, max_decoded_bytes=SKILL_BODY_MAX_BYTES).node_arcs_many(
            ("n",), properties=(CONTENT_ARC,)
        )

        node = arcs["n"]
        assert node is not None
        assert node[CONTENT_ARC] == ()
        assert UNDECODABLE_CONTENT_ARC in node

    def test_forwards_the_subject_read(self):
        inner = FakeSkillsGraph(subjects=_roster("skill-a"))
        graph = DecodedTextGraph(inner, max_decoded_bytes=SKILL_BODY_MAX_BYTES)

        page = graph.node_subjects_many(("AgentSkill",), property="typeOf", max_results=5)

        assert page["AgentSkill"].dcids == ("skill/skill-a",)
