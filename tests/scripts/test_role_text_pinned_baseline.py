"""Pins the score of corpus/direction/role_text_pinned.py -- the frozen
pre-redesign flow-direction scanner -- over the full case file, so the
baseline corpus/direction/README.md records cannot silently drift if the
case file changes underneath it. Unlike test_eval_role_direction.py, which
deliberately never runs a real scanner, this test exists specifically to
run this one frozen scanner and check its numbers, since the whole point of
freezing it is that its output over these fixed inputs never changes.
"""

from __future__ import annotations

from corpus.direction.role_text_pinned import detect_directions
from scripts.eval_role_direction import load_cases, score


def test_pinned_scanner_scores_exactly_as_recorded_in_the_readme():
    result = score(load_cases(), detect_directions)

    assert result["entity_count"] == 88
    assert result["unspecified_count"] == 23
    correct = sum(1 for v in result["verdicts"] if v["correct"])
    assert correct == 59
    assert result["accuracy"] == correct / 88
    assert result["false_confident_rate"] == 0.0

    assert result["confidently_wrong_count"] == 5

    confidently_wrong_ids = [
        v["case_id"]
        for v in result["verdicts"]
        if v["expected"] in ("origin", "destination")
        and v["predicted"] in ("origin", "destination")
        and v["predicted"] != v["expected"]
    ]
    assert confidently_wrong_ids == [
        "dir-substring-sudan-01",
        "dir-substring-korea-01",
        "dir-substring-guinea-01",
        "dir-substring-congo-01",
        "dir-possessive-03",
    ]
