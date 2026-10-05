"""Derive download filenames and `Content-Disposition` from artifact media types.

A Datasets artifact is named from its sources, the date its version was first prepared and a
short version tag, through `materialise.bundle_naming`. `LISTABLE_MEDIA_TYPES` contains exactly
the types `artifact_file_name` can name, so snapshot manifests can filter before deriving
filenames.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from urllib.parse import quote

from materialise.bundle_naming import bare_file_name, sources_slug, version_tag, zip_file_name

_EXTENSION_BY_MEDIA_TYPE: Mapping[str, str] = {
    "text/csv": "csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.apache.parquet": "parquet",
    "application/zip": "zip",
}

LISTABLE_MEDIA_TYPES: frozenset[str] = frozenset(_EXTENSION_BY_MEDIA_TYPE)
"""The media types a snapshot manifest lists, and `artifact_file_name` can
name -- exactly the keys of `_EXTENSION_BY_MEDIA_TYPE`.
"""

_ZIP_MEDIA_TYPE = "application/zip"
_STATISTICS_KIND = "series-statistics"


def artifact_file_name(
    *,
    source_names: Sequence[str],
    prepared_at: datetime,
    content_version: str,
    media_type: str,
    rendering: Mapping[str, object],
    statistics: bool,
) -> str:
    """The download name of one artifact of a Datasets version.

    A bundle zip is `dfo_<sources-slug>_<YYYY-MM-DD>_<format>_<cv6>.zip`, its format read from
    `rendering["format"]`. A bare file is `dfo_<sources-slug>_<YYYY-MM-DD>_<cv6>.<ext>`, and the
    `series-statistics/1` file (`statistics`) puts `series-statistics` before the tag. The slug
    is `dataset` when no source name yields one. The date is `prepared_at`'s UTC date.

    Raises `ValueError` for an unknown media type.
    """
    extension = _EXTENSION_BY_MEDIA_TYPE.get(media_type)
    if extension is None:
        raise ValueError(f"no known file extension for media_type {media_type!r}")
    slug = sources_slug(source_names)
    prepared = prepared_at.astimezone(UTC).date()
    tag = version_tag(content_version)
    if media_type == _ZIP_MEDIA_TYPE:
        return zip_file_name(slug=slug, prepared=prepared, format=str(rendering["format"]), tag=tag)
    return bare_file_name(
        slug=slug,
        prepared=prepared,
        tag=tag,
        extension=extension,
        kind=_STATISTICS_KIND if statistics else None,
    )


def artifact_content_disposition(file_name: str) -> str:
    """The `Content-Disposition` header value naming `file_name` as an attachment.

    `filename` is an RFC 6266 quoted string with `"` and `\\` escaped. A name with characters
    outside printable ASCII also gets an RFC 5987 `filename*`, and its `filename` replaces each
    such character with `_`.
    """
    if file_name.isascii() and file_name.isprintable():
        return f'attachment; filename="{_quoted_string(file_name)}"'
    fallback = "".join(c if c.isascii() and c.isprintable() else "_" for c in file_name)
    return (
        f'attachment; filename="{_quoted_string(fallback)}"; '
        f"filename*=UTF-8''{quote(file_name, safe='')}"
    )


def _quoted_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')
