"""Bundle naming tests cover slugs, headings, download file names and member slugs."""

from __future__ import annotations

from datetime import date

import pytest

from materialise.bundle_naming import (
    SLUG_MAX_LENGTH,
    bare_file_name,
    card_slug,
    heading,
    member_slugs,
    revision_tag,
    slugify,
    sources_slug,
    version_tag,
    zip_file_name,
)

_CRS = "OECD Creditor Reporting System (CRS)"
_SDG = "SDG Global Database"
_GDP = "OECD GDP"


@pytest.mark.parametrize(
    ("text", "max_length", "slug"),
    [
        ("Côte d'Ivoire: São Tomé", 40, "cote-d-ivoire-sao-tome"),
        ("  --Hello,   World!--  ", 40, "hello-world"),
        ("official development assistance gross", 30, "official-development"),
        ("official development", 20, "official-development"),
        ("supercalifragilisticexpialidocious", 10, "supercalif"),
        ("北京", 40, ""),
    ],
)
def test_slugify_folds_to_ascii_and_cuts_at_a_word_boundary(text, max_length, slug):
    assert slugify(text, max_length=max_length) == slug


@pytest.mark.parametrize(
    ("names", "slug"),
    [
        ([_CRS], "oecd-creditor-reporting-system"),
        ([_SDG, _GDP], "sdg-global-database-and-oecd-gdp"),
        ([_CRS, _SDG, _GDP], "oecd-creditor-reporting-and-2-more"),
        ([_CRS, _CRS], "oecd-creditor-reporting-system"),
        ([], "dataset"),
        (["(CRS)"], "dataset"),
        (
            ["International Debt Statistics", "World Development Indicators"],
            "international-debt-and-world-development",
        ),
    ],
)
def test_sources_slug_names_one_two_or_the_first_and_a_count(names, slug):
    assert sources_slug(names) == slug
    assert len(slug) <= SLUG_MAX_LENGTH


@pytest.mark.parametrize(
    ("names", "start", "end", "expected"),
    [
        ([_CRS], 1973, 2024, "OECD Creditor Reporting System (CRS), 1973\u20132024"),
        ([_SDG, _GDP], 2020, 2020, "SDG Global Database and OECD GDP, 2020"),
        ([_CRS, _SDG, _GDP], None, 2024, "OECD Creditor Reporting System (CRS) and 2 more sources"),
        ([_CRS, _CRS], 2000, 2001, "OECD Creditor Reporting System (CRS), 2000\u20132001"),
    ],
)
def test_heading_names_sources_and_years(names, start, end, expected):
    assert heading(source_names=names, start_year=start, end_year=end) == expected


def test_heading_needs_a_source():
    with pytest.raises(ValueError, match="source"):
        heading(source_names=[], start_year=None, end_year=None)


def test_file_names_follow_the_dataset_and_card_patterns():
    prepared = date(2026, 10, 4)
    tag = version_tag("cv_5c9dfd37e657d3a3c7000cc6acf49c43")
    slug = sources_slug([_CRS])

    assert zip_file_name(slug=slug, prepared=prepared, format="csv", tag=tag) == (
        "dfo_oecd-creditor-reporting-system_2026-10-04_csv_5c9dfd.zip"
    )
    assert bare_file_name(slug=slug, prepared=prepared, tag=tag, extension="parquet") == (
        "dfo_oecd-creditor-reporting-system_2026-10-04_5c9dfd.parquet"
    )
    assert bare_file_name(
        slug=slug, prepared=prepared, tag=tag, extension="parquet", kind="series-statistics"
    ) == ("dfo_oecd-creditor-reporting-system_2026-10-04_series-statistics_5c9dfd.parquet")
    assert bare_file_name(
        slug=slug, prepared=prepared, tag=tag, extension="csv", shape="pivot-9c1fe3"
    ) == ("dfo_oecd-creditor-reporting-system_2026-10-04_pivot-9c1fe3_5c9dfd.csv")
    assert zip_file_name(
        slug=slug, prepared=prepared, format="csv", tag=tag, shape="pivot-9c1fe3"
    ) == ("dfo_oecd-creditor-reporting-system_2026-10-04_csv_pivot-9c1fe3_5c9dfd.zip")
    card = card_slug("ODA from Germany to Nigeria")
    assert zip_file_name(slug=card, prepared=prepared, format="xlsx", tag=revision_tag(3)) == (
        "dfo_oda-from-germany-to-nigeria_2026-10-04_xlsx_r3.zip"
    )
    assert card_slug("…") == "card"


def test_member_slugs_suffix_collisions_and_avoid_the_codes_table():
    assert member_slugs([[_CRS], [_SDG], [_CRS], ["Codes"], [_CRS]]) == (
        "oecd-creditor-reporting-system",
        "sdg-global-database",
        "oecd-creditor-reporting-system-2",
        "codes-2",
        "oecd-creditor-reporting-system-3",
    )
