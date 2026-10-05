"""Pins raw and guarded extraction scores over 40 cases using recorded model output.
Missing Extraction fixtures fail scoring instead of lowering the result silently."""

from __future__ import annotations

from corpus.direction.recorded_extraction import guarded_extraction, raw_extraction
from scripts.eval_role_direction import load_cases, score


def test_raw_model_output_scores_exactly_as_recorded_in_the_readme():
    result = score(load_cases(), raw_extraction)

    assert result["entity_count"] == 88
    assert result["unspecified_count"] == 23
    correct = sum(1 for v in result["verdicts"] if v["correct"])
    assert correct == 85
    assert result["accuracy"] == correct / 88
    assert result["false_confident_rate"] == 0.0
    assert result["confidently_wrong_count"] == 1

    wrong = [v for v in result["verdicts"] if not v["correct"]]
    assert [(v["case_id"], v["surface"]) for v in wrong] == [
        ("dir-three-plus-list-01", "the United States"),
        ("dir-leading-unevidenced-02", "the United States"),
        ("dir-possessive-03", "Kenya"),
    ]


def test_guarded_extraction_scores_exactly_as_recorded_in_the_readme():
    """`guarded_extraction` applies `_validate_role_spans` to the same
    recorded entities `raw_extraction` scores unmodified. One of the four
    wrong labels here is the guard's own cost: it downgrades
    dir-leading-unevidenced-02 (Ethiopia) from correct to `"unspecified"`,
    because it took `role_span` credit the query does not evidence (see
    corpus/direction/README.md). The other three wrong labels were already
    wrong in the raw model output and the guard leaves them alone."""
    result = score(load_cases(), guarded_extraction)

    assert result["entity_count"] == 88
    assert result["unspecified_count"] == 23
    correct = sum(1 for v in result["verdicts"] if v["correct"])
    assert correct == 84
    assert result["accuracy"] == correct / 88
    assert result["false_confident_rate"] == 0.0
    assert result["confidently_wrong_count"] == 1

    wrong = [v for v in result["verdicts"] if not v["correct"]]
    assert [(v["case_id"], v["surface"]) for v in wrong] == [
        ("dir-three-plus-list-01", "the United States"),
        ("dir-leading-unevidenced-02", "Ethiopia"),
        ("dir-leading-unevidenced-02", "the United States"),
        ("dir-possessive-03", "Kenya"),
    ]


def test_a_missing_recorded_extraction_is_logged_as_a_harvest_miss(tmp_path, monkeypatch):
    """The harvest records the fixture from the logged miss; without one it cannot see the gap."""
    import json

    import pytest

    from corpus.direction import recorded_extraction as recorded
    from tests.fixtures import HARVEST_MISS_PATH_ENV

    harvest_path = tmp_path / "misses.jsonl"
    monkeypatch.setenv(HARVEST_MISS_PATH_ENV, str(harvest_path))
    monkeypatch.setattr(recorded, "_bank", lambda: {})

    with pytest.raises(SystemExit, match="no recorded Extraction fixture"):
        recorded.raw_extraction("aid to Kenya from Norway", [("Kenya", "country/KEN")])

    (line,) = harvest_path.read_text().splitlines()
    record = json.loads(line)
    assert record["outcome"] == "miss"
    assert record["schema_name"] == "Extraction"
    assert record["prompt"] == "aid to Kenya from Norway"
    assert record["key"] == recorded._fixture_key("aid to Kenya from Norway")
