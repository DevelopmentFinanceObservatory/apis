"""Pins the shape `examples.toml` must keep and the packaged file itself."""

from __future__ import annotations

import pytest

from search.catalogue.examples import PLACE_SLOT, ExamplesError, load_examples
from search.catalogue.job import FLOOR


def _line(**fields: str) -> str:
    values = {
        "template": "Population of {place}",
        "stat_var": "Count_Person",
        "place": "country/KEN",
        **fields,
    }
    body = "\n".join(f"{key} = {value!r}".replace("'", '"') for key, value in values.items())
    return f"[[example]]\n{body}\n"


def test_packaged_file_holds_more_lines_than_the_floor_each_with_one_slot():
    loaded = load_examples()
    # A build that drops one line still publishes.
    assert len(loaded.examples) > FLOOR
    assert [example.id for example in loaded.examples] == list(range(len(loaded.examples)))
    for example in loaded.examples:
        assert example.template.count(PLACE_SLOT) == 1
        assert example.place.startswith("country/")


def test_ids_follow_file_order_and_the_digest_follows_the_bytes():
    raw = (_line() + _line(template="Inflation in {place}", stat_var="sdg/FP_CPI_TOTL_ZG")).encode()
    loaded = load_examples(raw)
    assert [(example.id, example.template) for example in loaded.examples] == [
        (0, "Population of {place}"),
        (1, "Inflation in {place}"),
    ]
    assert loaded.digest != load_examples(raw + b"\n").digest


def test_role_is_read_and_text_fills_the_slot():
    [example] = load_examples(_line(template="Aid to {place}", role="recipient").encode()).examples
    assert example.role == "recipient"
    assert example.text("Tuvalu") == "Aid to Tuvalu"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_line(template="Population"), "exactly one {place} slot"),
        (_line(template="{place} and {place}"), "exactly one {place} slot"),
        (_line(template="{year} in {place}"), "exactly one {place} slot"),
        (_line(place="geoId/06"), "not a country/ dcid"),
        (_line(role="lender"), "role 'lender'"),
        (_line(stat_var=" "), "non-empty string 'stat_var'"),
        (_line(note="x"), "unknown keys ['note']"),
        (_line() + _line(stat_var="Count_Person_Female"), "repeats template"),
        ('title = "x"\n', "only an `[[example]]` array"),
        ("[[example]\n", "does not parse"),
    ],
)
def test_malformed_lines_are_refused(raw: str, message: str):
    with pytest.raises(ExamplesError, match=message.replace("[", r"\[").replace("{", r"\{")):
        load_examples(raw.encode())
