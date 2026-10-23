"""Test lexical candidate retrieval: the ranked, bounded menu of a phrase no code reads."""

from __future__ import annotations

import pytest

from code_index import CodeIndex
from code_index.query import CANDIDATE_HITS, CANDIDATE_LIMIT, CandidateCodes
from tests.code_index.support import INDEX_FIXTURE, load_index


@pytest.fixture(scope="module")
def committed() -> CodeIndex:
    """The committed index, whose CRS labels and synonyms these menus read."""
    return load_index(INDEX_FIXTURE)


_CRS = frozenset({"crsPurpose", "crsTypeOfFlow", "crsTypeOfFinance", "crsCooperationModality"})


def test_a_hit_sharing_more_words_ranks_first_and_brings_its_relatives(committed: CodeIndex):
    """13030 leads for "contraceptive services", whose two words its synonyms share, and its
    ancestors 130 and 100 join the menu."""
    menu = committed.candidate_codes(
        ["contraceptive", "services"], code_systems=_CRS, context=["ODA"]
    )
    assert menu.hits[0] == menu.codes[0] == "CRSPurposeCode13030"
    assert {"CRSPurposeCode130", "CRSPurposeCode100"} <= set(menu.codes)
    assert "CRSPurposeCode130" not in menu.hits


def test_context_words_rank_hits_and_seeds_follow_them(committed: CodeIndex):
    """The words of the term beside the phrase rank the hits that share them first, and the
    seed and its children join the menu after the hits."""
    menu = committed.candidate_codes(
        ["reproductive"], code_systems=_CRS, context=["health"], seeds=["CRSPurposeCode120"]
    )
    hits = [
        "CRSPurposeCode130",
        "CRSPurposeCode13010",
        "CRSPurposeCode13020",
        "CRSPurposeCode13081",
    ]
    assert list(menu.hits) == list(menu.codes[: len(hits)]) == hits
    assert menu.codes[len(hits)] == "CRSPurposeCode120"
    assert "CRSPurposeCode121" in menu.codes


@pytest.mark.parametrize(
    ("words", "systems"),
    [(["maternal"], _CRS), (["contraceptive"], frozenset({"crsTypeOfFlow"}))],
    ids=["no-shared-word", "other-code-system"],
)
def test_no_shared_word_in_the_systems_asked_gives_no_menu(committed, words, systems):
    """A context word or seed alone makes no menu: "maternal" shares no CRS text, and
    "contraceptive" shares none of a flow type's."""
    assert committed.candidate_codes(
        words, code_systems=systems, context=["health"], seeds=["CRSPurposeCode120"]
    ) == CandidateCodes(hits=(), codes=())


def test_the_menu_is_bounded(committed: CodeIndex):
    """Dozens of codes share "services" and, with their relatives, number 34. The menu holds
    `CANDIDATE_LIMIT` of them, each once."""
    menu = committed.candidate_codes(["services"], code_systems=_CRS)
    assert len(menu.codes) == CANDIDATE_LIMIT
    assert len(set(menu.codes)) == len(menu.codes)
    assert len(menu.hits) == CANDIDATE_HITS
