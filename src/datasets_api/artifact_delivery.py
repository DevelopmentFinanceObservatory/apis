"""Derive download links, filenames and `Content-Disposition` from artifact media types.

A manifest names each artifact by its stable download link (`artifact_download_path`), which
the files door resolves at the click to a short-lived signed URL.

A Datasets artifact is named from its sources, the date its version was first prepared and a
short version tag, through `materialise.bundle_naming`. `artifact_file_name` names every media
type a version links, the stored frame1 included, since the files door serves a link to any of
them. `LISTABLE_MEDIA_TYPES` is the subset a snapshot manifest lists, every type but frame1.
`STATISTICS_FILE_KIND` and `FRAME_FILE_KIND` name the two Parquet files every version holds
apart from a requested Parquet file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from urllib.parse import quote

from materialise.bundle_naming import bare_file_name, sources_slug, version_tag, zip_file_name
from materialise.formats import FRAME1_MEDIA_TYPE

_LISTABLE_EXTENSIONS: Mapping[str, str] = {
    "text/csv": "csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.apache.parquet": "parquet",
    "application/zip": "zip",
}

_EXTENSION_BY_MEDIA_TYPE: Mapping[str, str] = {**_LISTABLE_EXTENSIONS, FRAME1_MEDIA_TYPE: "json"}

LISTABLE_MEDIA_TYPES: frozenset[str] = frozenset(_LISTABLE_EXTENSIONS)
"""The media types a snapshot manifest lists: every type `artifact_file_name` names but the
stored frame1."""

_ZIP_MEDIA_TYPE = "application/zip"

STATISTICS_FILE_KIND = "series-statistics"
"""The file-name kind of a version's `series-statistics/1` file."""

FRAME_FILE_KIND = "frame"
"""The file-name kind of a version's canonical frame."""


def artifact_download_path(*, subject_id: str, content_version: str, artifact_digest: str) -> str:
    """The stable download link of one artifact of a version,
    `/api/datasets/{subject_id}/{content_version}/files/{artifact_digest}`.

    It is root-relative, as the operation poll link and its `Location` header are, so it
    resolves against whichever host served the manifest. It never expires.
    """
    return f"/api/datasets/{subject_id}/{content_version}/files/{artifact_digest}"


def artifact_file_name(
    *,
    source_names: Sequence[str],
    prepared_at: datetime,
    content_version: str,
    media_type: str,
    rendering: Mapping[str, object],
    kind: str | None,
    shape: str | None = None,
) -> str:
    """The download name of one artifact of a Datasets version.

    A bundle zip is `dfo_<sources-slug>_<YYYY-MM-DD>_<format>_<cv6>.zip`, its format read from
    `rendering["format"]`. A bare file is `dfo_<sources-slug>_<YYYY-MM-DD>_<cv6>.<ext>`, and a
    file that is not the requested data puts its `kind` (`STATISTICS_FILE_KIND`,
    `FRAME_FILE_KIND`) before the tag. A rendering that reshapes the default table puts its
    `shape` token before the tag, after the format of a zip. The slug
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
        return zip_file_name(
            slug=slug, prepared=prepared, format=str(rendering["format"]), tag=tag, shape=shape
        )
    return bare_file_name(
        slug=slug,
        prepared=prepared,
        tag=tag,
        extension=extension,
        kind=kind,
        shape=shape,
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
