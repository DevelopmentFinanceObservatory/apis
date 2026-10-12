"""Test artifact filenames by media type and the attachment disposition header."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from datasets_api.artifact_delivery import artifact_content_disposition, artifact_file_name

_CRS = "OECD Creditor Reporting System (CRS)"
_CRS_CV = "cv_5c9dfd37e657d3a3c7000cc6acf49c43"
_PREPARED_AT = datetime(2026, 10, 4, 9, tzinfo=UTC)


@pytest.mark.parametrize(
    ("media_type", "file_name"),
    [
        ("text/csv", "dfo_oecd-creditor-reporting-system_2026-10-04_5c9dfd.csv"),
        (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "dfo_oecd-creditor-reporting-system_2026-10-04_5c9dfd.xlsx",
        ),
        (
            "application/vnd.apache.parquet",
            "dfo_oecd-creditor-reporting-system_2026-10-04_5c9dfd.parquet",
        ),
        ("application/zip", "dfo_oecd-creditor-reporting-system_2026-10-04_csv_5c9dfd.zip"),
    ],
)
def test_artifact_file_name_derives_the_extension_from_each_known_media_type(
    media_type: str, file_name: str
) -> None:
    got = artifact_file_name(
        source_names=[_CRS],
        prepared_at=_PREPARED_AT,
        content_version=_CRS_CV,
        media_type=media_type,
        rendering={"format": "csv", "bundle": media_type == "application/zip"},
        kind=None,
    )

    assert got == file_name


def test_artifact_file_name_dates_a_version_by_its_utc_day() -> None:
    late_evening_west_of_utc = datetime(2026, 10, 3, 22, tzinfo=timezone(timedelta(hours=-4)))

    got = artifact_file_name(
        source_names=[_CRS],
        prepared_at=late_evening_west_of_utc,
        content_version=_CRS_CV,
        media_type="text/csv",
        rendering={"format": "csv", "bundle": False},
        kind=None,
    )

    assert got == "dfo_oecd-creditor-reporting-system_2026-10-04_5c9dfd.csv"


def test_artifact_file_name_refuses_an_unknown_media_type() -> None:
    with pytest.raises(ValueError, match="application/pdf"):
        artifact_file_name(
            source_names=[_CRS],
            prepared_at=_PREPARED_AT,
            content_version=_CRS_CV,
            media_type="application/pdf",
            rendering={},
            kind=None,
        )


@pytest.mark.parametrize(
    ("file_name", "disposition"),
    [
        (
            "dfo_dataset_2026-01-01_cv1.csv",
            'attachment; filename="dfo_dataset_2026-01-01_cv1.csv"',
        ),
        ('a "b" \\ c.csv', 'attachment; filename="a \\"b\\" \\\\ c.csv"'),
        (
            "côte.csv",
            "attachment; filename=\"c_te.csv\"; filename*=UTF-8''c%C3%B4te.csv",
        ),
    ],
)
def test_artifact_content_disposition_quotes_the_file_name(
    file_name: str, disposition: str
) -> None:
    assert artifact_content_disposition(file_name) == disposition
